// SPDX-License-Identifier: Apache-2.0

//! A small, hand-rolled JSON-RPC 2.0 server over stdio implementing the subset
//! of the Model Context Protocol (MCP) this server needs: `initialize`,
//! `tools/list`, and `tools/call`.
//!
//! Framing: newline-delimited JSON. MCP's stdio transport sends each JSON-RPC
//! message as a single line on stdin and expects each response on its own line
//! on stdout. We read line by line and write one compact JSON object per line.
//! (The `Content-Length` HTTP-style framing is for the SSE/HTTP transport, not
//! stdio.)
//!
//! Only request/response methods are served. Streaming CLI verbs (`events`,
//! `logs`) are intentionally NOT exposed as tools, so this server never streams.

use std::io::{BufRead, Write};

use serde_json::{json, Map, Value};

use crate::exec::{run_tool, ToolOutcome};
use crate::registry::{find_tool, tools_list};

/// MCP protocol version this server speaks. Clients send their preferred
/// version in `initialize`; we echo a version we support.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

pub const SERVER_NAME: &str = "waymux-mcp";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Standard JSON-RPC error codes used here.
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// Hard cap on the size of a single newline-delimited JSON-RPC message.
/// `BufRead::read_line` is unbounded: a peer that sends a giant line with no
/// newline would buffer the whole thing in memory. 8 MiB is far larger than
/// any legitimate request this server handles (the biggest is a `spawn` argv
/// or an `inject` ops array), while still bounding a hostile or buggy peer.
const MAX_LINE_SIZE: usize = 8 * 1024 * 1024;

/// Build a JSON-RPC success response.
fn rpc_result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// Build a JSON-RPC error response.
fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Serialize a `ToolOutcome` into an MCP `tools/call` result object.
fn outcome_to_result(outcome: ToolOutcome) -> Value {
    let mut result = Map::new();
    result.insert("content".into(), Value::Array(outcome.content));
    if let Some(structured) = outcome.structured {
        result.insert("structuredContent".into(), structured);
    }
    if outcome.is_error {
        result.insert("isError".into(), json!(true));
    }
    Value::Object(result)
}

/// The `initialize` result: protocol version, server info, and the tools
/// capability (we advertise `tools` with `listChanged: false`).
fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
        "capabilities": { "tools": { "listChanged": false } },
    })
}

/// Outcome of handling one parsed JSON-RPC message. `None` means "no response"
/// (a notification, which has no id).
enum Handled {
    Respond(Value),
    Silent,
}

/// Dispatch a single parsed JSON-RPC request/notification object. Pure (modulo
/// the side effect of executing the CLI inside `tools/call`); returns the
/// response value to write, or `Silent` for notifications.
fn handle_message(msg: &Value) -> Handled {
    // Notifications have no `id`. The only one we expect is
    // `notifications/initialized`; ignore any notification silently.
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str);

    let method = match method {
        Some(m) => m,
        None => {
            // No method: if it has an id it's malformed; respond with an error.
            return match id {
                Some(id) => Handled::Respond(rpc_error(id, INVALID_REQUEST, "missing method")),
                None => Handled::Silent,
            };
        }
    };

    // Notification (no id): never produce a response.
    if id.is_none() {
        return Handled::Silent;
    }
    let id = id.expect("id present in request branch");

    match method {
        "initialize" => Handled::Respond(rpc_result(id, initialize_result())),
        "tools/list" => Handled::Respond(rpc_result(id, json!({ "tools": tools_list() }))),
        "tools/call" => Handled::Respond(handle_tools_call(id, msg)),
        // `ping` is part of MCP; answer with an empty result.
        "ping" => Handled::Respond(rpc_result(id, json!({}))),
        _ => Handled::Respond(rpc_error(
            id,
            METHOD_NOT_FOUND,
            &format!("method not found: {method}"),
        )),
    }
}

/// Handle a `tools/call` request: look up the tool, extract its arguments, run
/// it, and shape the result. Unknown tools and bad params yield JSON-RPC errors;
/// CLI-reported failures yield a successful JSON-RPC response whose result has
/// `isError: true` (per the MCP tool-error convention).
fn handle_tools_call(id: Value, msg: &Value) -> Value {
    let params = match msg.get("params") {
        Some(p) => p,
        None => return rpc_error(id, INVALID_PARAMS, "missing params"),
    };
    let name = match params.get("name").and_then(Value::as_str) {
        Some(n) => n,
        None => return rpc_error(id, INVALID_PARAMS, "missing tool name"),
    };
    let spec = match find_tool(name) {
        Some(s) => s,
        None => {
            return rpc_error(id, INVALID_PARAMS, &format!("unknown tool: {name}"));
        }
    };
    // `arguments` is optional; default to an empty object.
    let args = match params.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => {
            return rpc_error(id, INVALID_PARAMS, "`arguments` must be an object");
        }
    };

    let outcome = run_tool(spec, args);
    rpc_result(id, outcome_to_result(outcome))
}

/// Outcome of `read_line_bounded`.
enum LineRead {
    /// A complete line (the trailing newline, if any, is stripped). May be
    /// empty if the peer sent a bare newline.
    Line(Vec<u8>),
    /// The line exceeded `MAX_LINE_SIZE` before a newline arrived. The
    /// remainder of the over-long line has been drained up to its terminating
    /// newline (or EOF) so the next call resynchronizes on a real boundary.
    TooLong,
    /// Clean EOF with no pending bytes.
    Eof,
}

/// Read a single newline-delimited line, rejecting any line longer than
/// `MAX_LINE_SIZE` instead of buffering it without bound. On an over-long line
/// we consume (and discard) the rest of it up to its newline so the loop can
/// resynchronize and keep serving subsequent well-formed messages.
fn read_line_bounded<R: BufRead>(input: &mut R) -> std::io::Result<LineRead> {
    let mut buf: Vec<u8> = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = input.read(&mut byte)?;
        if n == 0 {
            // EOF. If we accumulated a partial (newline-less) line, surface it.
            return Ok(if buf.is_empty() {
                LineRead::Eof
            } else {
                LineRead::Line(buf)
            });
        }
        if byte[0] == b'\n' {
            return Ok(LineRead::Line(buf));
        }
        if buf.len() >= MAX_LINE_SIZE {
            // Over-long. Drain to the next newline (or EOF) without growing
            // `buf`, then report the rejection.
            loop {
                let n = input.read(&mut byte)?;
                if n == 0 || byte[0] == b'\n' {
                    break;
                }
            }
            return Ok(LineRead::TooLong);
        }
        buf.push(byte[0]);
    }
}

/// Run the stdio server loop: read newline-delimited JSON-RPC from `input`,
/// write newline-delimited responses to `output`. Returns on EOF.
pub fn serve<R: BufRead, W: Write>(mut input: R, mut output: W) -> std::io::Result<()> {
    loop {
        let line = match read_line_bounded(&mut input)? {
            LineRead::Eof => break,
            LineRead::TooLong => {
                // A pathologically long line: reject it with a JSON-RPC parse
                // error (null id, per spec) without panicking or OOMing, then
                // keep serving.
                let resp = rpc_error(
                    Value::Null,
                    PARSE_ERROR,
                    &format!("line exceeds {MAX_LINE_SIZE} bytes; rejected"),
                );
                write_line(&mut output, &resp)?;
                continue;
            }
            LineRead::Line(bytes) => bytes,
        };
        let trimmed = String::from_utf8_lossy(&line);
        let trimmed = trimmed.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                // Parse error: respond with a null-id JSON-RPC error per spec.
                let resp = rpc_error(Value::Null, PARSE_ERROR, &format!("parse error: {e}"));
                write_line(&mut output, &resp)?;
                continue;
            }
        };

        match handle_message(&msg) {
            Handled::Respond(resp) => write_line(&mut output, &resp)?,
            Handled::Silent => {}
        }
    }
    Ok(())
}

/// Write one compact JSON object followed by a newline, and flush.
fn write_line<W: Write>(output: &mut W, v: &Value) -> std::io::Result<()> {
    let s = serde_json::to_string(v)?;
    output.write_all(s.as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn respond(msg: &Value) -> Option<Value> {
        match handle_message(msg) {
            Handled::Respond(v) => Some(v),
            Handled::Silent => None,
        }
    }

    #[test]
    fn initialize_returns_protocol_and_serverinfo() {
        let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} });
        let resp = respond(&req).unwrap();
        assert_eq!(resp["jsonrpc"], json!("2.0"));
        assert_eq!(resp["id"], json!(1));
        assert_eq!(resp["result"]["protocolVersion"], json!(PROTOCOL_VERSION));
        assert_eq!(resp["result"]["serverInfo"]["name"], json!(SERVER_NAME));
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_returns_valid_schema_per_tool() {
        let req = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
        let resp = respond(&req).unwrap();
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert!(!tools.is_empty());
        for t in tools {
            assert!(t["name"].as_str().unwrap().starts_with("waymux_"));
            let schema = &t["inputSchema"];
            assert_eq!(schema["type"], json!("object"));
            assert!(schema["properties"].is_object());
        }
    }

    #[test]
    fn notification_produces_no_response() {
        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(respond(&note).is_none());
    }

    #[test]
    fn unknown_method_is_method_not_found() {
        let req = json!({ "jsonrpc": "2.0", "id": 3, "method": "no/such" });
        let resp = respond(&req).unwrap();
        assert_eq!(resp["error"]["code"], json!(METHOD_NOT_FOUND));
    }

    #[test]
    fn tools_call_unknown_tool_is_invalid_params() {
        let req = json!({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": { "name": "waymux_nope", "arguments": {} }
        });
        let resp = respond(&req).unwrap();
        assert_eq!(resp["error"]["code"], json!(INVALID_PARAMS));
    }

    #[test]
    fn ping_returns_empty_result() {
        let req = json!({ "jsonrpc": "2.0", "id": 5, "method": "ping" });
        let resp = respond(&req).unwrap();
        assert!(resp["result"].is_object());
    }

    /// A pathologically long single line must be rejected gracefully: the
    /// server returns a JSON-RPC parse error (null id) without panicking or
    /// buffering the whole line, and remains able to serve a well-formed
    /// message that follows on the next line.
    #[test]
    fn over_long_line_is_rejected_without_panic() {
        use std::io::Cursor;

        // One line of `MAX_LINE_SIZE + 1024` non-newline bytes, then a
        // newline, then a valid `ping` request on its own line.
        let mut input: Vec<u8> = vec![b'x'; MAX_LINE_SIZE + 1024];
        input.push(b'\n');
        input.extend_from_slice(br#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#);
        input.push(b'\n');

        let mut output: Vec<u8> = Vec::new();
        serve(Cursor::new(input), &mut output).expect("serve should not error");

        let text = String::from_utf8(output).unwrap();
        let mut lines = text.lines();

        // First response: the parse-error rejection for the over-long line.
        let first: Value = serde_json::from_str(lines.next().expect("a rejection line")).unwrap();
        assert_eq!(first["error"]["code"], json!(PARSE_ERROR));
        assert_eq!(first["id"], Value::Null);

        // Second response: the ping was still served after the rejection,
        // proving the loop resynchronized on the newline boundary.
        let second: Value = serde_json::from_str(lines.next().expect("a ping line")).unwrap();
        assert_eq!(second["id"], json!(7));
        assert!(second["result"].is_object());
    }

    /// A single bounded line with no trailing newline (peer closed the pipe
    /// mid-message) is still parsed, then EOF ends the loop cleanly.
    #[test]
    fn line_without_trailing_newline_is_served() {
        use std::io::Cursor;
        let input = br#"{"jsonrpc":"2.0","id":8,"method":"ping"}"#.to_vec();
        let mut output: Vec<u8> = Vec::new();
        serve(Cursor::new(input), &mut output).expect("serve");
        let text = String::from_utf8(output).unwrap();
        let resp: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(resp["id"], json!(8));
    }
}
