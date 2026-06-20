// SPDX-License-Identifier: Apache-2.0

//! Execute the `waymux` CLI for a tool call and map its `--json` envelope onto
//! an MCP tool result.
//!
//! SECURITY: this module NEVER builds a shell command string and NEVER passes
//! arguments through a shell. Every argument is appended to
//! `std::process::Command` as a discrete argv element, so there is no command-
//! injection surface no matter what a client supplies as an argument value.

use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Map, Value};

use crate::registry::{Param, ParamKind, ParamType, ToolSpec};

/// The outcome of a tool call, ready to be serialized into an MCP
/// `tools/call` result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutcome {
    /// MCP `content` blocks (text and/or image).
    pub content: Vec<Value>,
    /// Optional structured content (the envelope `data` on success).
    pub structured: Option<Value>,
    /// Whether this is an error result (`isError: true`).
    pub is_error: bool,
}

impl ToolOutcome {
    fn ok(content: Vec<Value>, structured: Option<Value>) -> Self {
        Self {
            content,
            structured,
            is_error: false,
        }
    }

    fn error(message: String) -> Self {
        Self {
            content: vec![text_block(message)],
            structured: None,
            is_error: true,
        }
    }
}

/// Build an MCP text content block.
fn text_block(s: String) -> Value {
    json!({ "type": "text", "text": s })
}

/// Build an MCP image content block from base64 PNG bytes.
fn image_block(b64: &str) -> Value {
    json!({ "type": "image", "data": b64, "mimeType": "image/png" })
}

/// Resolve the `waymux` binary path. Resolution order:
///   1. `WAYMUX_BIN` environment variable, if set (absolute or relative path).
///   2. A `waymux` binary sibling to the running `waymux-mcp` executable
///      (covers both an installed layout and a cargo `target/<profile>` dir).
///   3. Bare `"waymux"`, resolved against `$PATH` by the OS.
pub fn resolve_waymux_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("WAYMUX_BIN") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("waymux");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("waymux")
}

/// Stringify a JSON argument value for the CLI argv according to its declared
/// type. Returns an error string if the value's JSON type does not match.
fn stringify_scalar(p: &Param, v: &Value) -> Result<String, String> {
    match p.ty {
        ParamType::StringT | ParamType::JsonStringT => match v {
            Value::String(s) => Ok(s.clone()),
            // Accept a non-string only for JsonStringT by re-serializing, so a
            // client may pass a real JSON array for `ops` instead of a string.
            other if p.ty == ParamType::JsonStringT => Ok(other.to_string()),
            _ => Err(format!("argument `{}` must be a string", p.name)),
        },
        ParamType::IntegerT => match v {
            Value::Number(n) if n.is_i64() || n.is_u64() => Ok(n.to_string()),
            _ => Err(format!("argument `{}` must be an integer", p.name)),
        },
        ParamType::NumberT => match v {
            Value::Number(n) => Ok(n.to_string()),
            _ => Err(format!("argument `{}` must be a number", p.name)),
        },
        ParamType::BooleanT => match v {
            Value::Bool(b) => Ok(b.to_string()),
            _ => Err(format!("argument `{}` must be a boolean", p.name)),
        },
        ParamType::StringArrayT => Err(format!(
            "argument `{}` is an array and must be handled as a vector",
            p.name
        )),
    }
}

/// Build the per-call argument vector (everything AFTER the verb argv) from the
/// tool spec and the client-supplied arguments object.
///
/// Order: positionals (in declaration order), then flags. Array-valued
/// trailing params (`tags`, `argv`) are emitted last as repeated elements;
/// `spawn`'s `argv` is preceded by a `--` separator so clap reads it as the
/// trailing command vector.
pub fn build_call_args(spec: &ToolSpec, args: &Map<String, Value>) -> Result<Vec<String>, String> {
    let mut positionals: Vec<String> = Vec::new();
    let mut flags: Vec<String> = Vec::new();
    let mut trailing_array: Option<(&Param, Vec<String>)> = None;

    for p in spec.params {
        let value = args.get(p.name);

        // Array-valued params (kind = Positional, ty = StringArrayT) are
        // collected by the dedicated block below and skip the scalar handling.
        if p.ty == ParamType::StringArrayT {
            match value {
                Some(Value::Array(items)) => {
                    let mut out = Vec::with_capacity(items.len());
                    for (i, it) in items.iter().enumerate() {
                        match it {
                            Value::String(s) => out.push(s.clone()),
                            _ => {
                                return Err(format!(
                                    "argument `{}`[{}] must be a string",
                                    p.name, i
                                ))
                            }
                        }
                    }
                    if p.required && out.is_empty() {
                        return Err(format!(
                            "argument `{}` must have at least one element",
                            p.name
                        ));
                    }
                    trailing_array = Some((p, out));
                }
                Some(_) => return Err(format!("argument `{}` must be an array", p.name)),
                None if p.required => {
                    return Err(format!("missing required argument `{}`", p.name))
                }
                None => {}
            }
            continue;
        }

        match p.kind {
            ParamKind::Positional => {
                match value {
                    Some(v) => positionals.push(stringify_scalar(p, v)?),
                    None if p.required => {
                        return Err(format!("missing required argument `{}`", p.name))
                    }
                    None => { /* optional positional omitted */ }
                }
            }
            ParamKind::Flag => match value {
                Some(v) => {
                    flags.push(format!("--{}", p.cli_name));
                    flags.push(stringify_scalar(p, v)?);
                }
                None if p.required => {
                    return Err(format!("missing required argument `{}`", p.name))
                }
                None => {}
            },
            ParamKind::BoolFlag => {
                if let Some(v) = value {
                    match v {
                        Value::Bool(true) => flags.push(format!("--{}", p.cli_name)),
                        Value::Bool(false) => {}
                        _ => return Err(format!("argument `{}` must be a boolean", p.name)),
                    }
                }
            }
        }
    }

    let mut argv: Vec<String> = Vec::new();
    argv.append(&mut positionals);
    argv.append(&mut flags);

    if let Some((p, mut items)) = trailing_array {
        // `spawn`'s argv must be separated from the verb flags with `--` so clap
        // captures it as the trailing command vector. `tag`'s tags are a plain
        // trailing positional vararg and need no separator.
        if p.name == "argv" {
            argv.push("--".to_string());
        }
        argv.append(&mut items);
    }

    Ok(argv)
}

/// Some verbs require an argument the MCP surface hides. `screenshot` and
/// `screenshot-desktop` require `-o/--output` from clap, but under `--json` the
/// bytes are returned in the envelope and the file path is ignored. If the
/// client did not provide `output`, inject a harmless placeholder so clap is
/// satisfied. The CLI never writes the file under `--json`.
fn inject_required_placeholders(spec: &ToolSpec, args: &mut Map<String, Value>) {
    if matches!(spec.verb, "screenshot" | "screenshot-desktop") && !args.contains_key("output") {
        args.insert("output".into(), json!("/dev/null"));
    }
}

/// Parse the CLI's stdout envelope and map it to an MCP outcome.
///
///   `{ "ok": true,  "data": <data> }`  -> success outcome (structured + text)
///   `{ "ok": false, "error": {code,message} }` -> error outcome carrying the
///                                                  code + message
///
/// For `screenshot`/`screenshot-desktop` success, `data.png_b64` is surfaced as
/// an MCP image content block (the structured data is still attached, with the
/// large b64 string elided from the text summary).
pub fn map_envelope(spec: &ToolSpec, stdout: &str) -> ToolOutcome {
    let env: Value = match serde_json::from_str(stdout.trim()) {
        Ok(v) => v,
        Err(e) => {
            return ToolOutcome::error(format!(
                "waymux returned unparseable output for `{}`: {e}\n--- stdout ---\n{}",
                spec.verb, stdout
            ))
        }
    };

    let ok = env.get("ok").and_then(Value::as_bool);
    match ok {
        Some(true) => {
            let data = env.get("data").cloned().unwrap_or(Value::Null);
            map_success(spec, data)
        }
        Some(false) => {
            let err = env.get("error");
            let code = err
                .and_then(|e| e.get("code"))
                .and_then(Value::as_str)
                .unwrap_or("E_INTERNAL");
            let message = err
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            ToolOutcome::error(format!("{code}: {message}"))
        }
        None => ToolOutcome::error(format!(
            "waymux envelope for `{}` missing boolean `ok`: {stdout}",
            spec.verb
        )),
    }
}

/// Map a success envelope's `data` to content blocks + structured content.
fn map_success(spec: &ToolSpec, data: Value) -> ToolOutcome {
    let is_screenshot = matches!(spec.verb, "screenshot" | "screenshot-desktop");
    if is_screenshot {
        if let Some(b64) = data.get("png_b64").and_then(Value::as_str) {
            // Image block + a small structured summary (without the huge b64
            // blob in the text). The full data (including png_b64) is kept as
            // structured content for clients that want it.
            let w = data.get("width").cloned().unwrap_or(Value::Null);
            let h = data.get("height").cloned().unwrap_or(Value::Null);
            let summary = format!("screenshot {w}x{h} PNG ({} base64 bytes)", b64.len());
            return ToolOutcome::ok(vec![image_block(b64), text_block(summary)], Some(data));
        }
    }

    // Generic success: a compact JSON text summary plus the structured data.
    let summary = format!("{}: {}", spec.verb, compact_summary(&data));
    ToolOutcome::ok(vec![text_block(summary)], Some(data))
}

/// A compact one-line JSON rendering of `data`, eliding any `png_b64` field so
/// the text summary never carries a megabyte of base64.
fn compact_summary(data: &Value) -> String {
    if let Value::Object(map) = data {
        let mut redacted = map.clone();
        if redacted.contains_key("png_b64") {
            redacted.insert("png_b64".into(), json!("<elided>"));
        }
        Value::Object(redacted).to_string()
    } else {
        data.to_string()
    }
}

/// Run the `waymux` CLI for a tool call end-to-end: resolve the binary, build
/// the argv, execute (argv-only; no shell), and map the result.
pub fn run_tool(spec: &ToolSpec, mut args: Map<String, Value>) -> ToolOutcome {
    inject_required_placeholders(spec, &mut args);

    let call_args = match build_call_args(spec, &args) {
        Ok(a) => a,
        Err(e) => return ToolOutcome::error(e),
    };

    let bin = resolve_waymux_bin();

    // Full argv: --json <verb...> <call args...>. Each is a discrete element.
    let mut cmd = Command::new(&bin);
    cmd.arg("--json");
    for v in spec.argv {
        cmd.arg(v);
    }
    for a in &call_args {
        cmd.arg(a);
    }

    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            return ToolOutcome::error(format!(
                "failed to execute waymux binary `{}`: {e}",
                bin.display()
            ))
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // The CLI emits an envelope on stdout for BOTH success and error and sets
    // the exit code accordingly. Prefer parsing the envelope; fall back to a
    // generic error including stderr when stdout is empty or unparseable on a
    // non-zero exit.
    if stdout.trim().is_empty() {
        if output.status.success() {
            // Every verb the MCP layer invokes runs with `--json`, and under
            // `--json` they all emit a success envelope on stdout, so this
            // empty-stdout-plus-success branch is dead in practice. It is kept
            // as a harmless belt-and-suspenders fallback: if some future verb
            // ever stayed silent on success, we still produce an ok outcome
            // rather than an error.
            return ToolOutcome::ok(
                vec![text_block(format!("{}: ok", spec.verb))],
                Some(Value::Null),
            );
        }
        return ToolOutcome::error(format!(
            "waymux `{}` exited with {} and no stdout\n--- stderr ---\n{}",
            spec.verb,
            output.status,
            stderr.trim()
        ));
    }

    map_envelope(spec, &stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::find_tool;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn build_args_positionals_then_flags() {
        let spec = find_tool("waymux_new").unwrap();
        let args = obj(json!({ "name": "eagle", "size": "800x600", "scale": 2 }));
        let argv = build_call_args(spec, &args).unwrap();
        // name is positional first; size/scale are flags.
        assert_eq!(argv[0], "eagle");
        assert!(argv.contains(&"--size".to_string()));
        assert!(argv.contains(&"800x600".to_string()));
        assert!(argv.contains(&"--scale".to_string()));
        assert!(argv.contains(&"2".to_string()));
    }

    #[test]
    fn build_args_boolflag_only_when_true() {
        let spec = find_tool("waymux_new").unwrap();
        let with =
            build_call_args(spec, &obj(json!({ "name": "x", "share_audio": true }))).unwrap();
        assert!(with.contains(&"--share-audio".to_string()));
        let without =
            build_call_args(spec, &obj(json!({ "name": "x", "share_audio": false }))).unwrap();
        assert!(!without.contains(&"--share-audio".to_string()));
    }

    #[test]
    fn build_args_missing_required_errs() {
        let spec = find_tool("waymux_info").unwrap();
        let err = build_call_args(spec, &obj(json!({}))).unwrap_err();
        assert!(err.contains("missing required argument `name`"), "{err}");
    }

    #[test]
    fn build_args_tag_trailing_vararg() {
        let spec = find_tool("waymux_tag").unwrap();
        let argv = build_call_args(
            spec,
            &obj(json!({ "name": "s", "window_id": 5, "tags": ["a", "b"] })),
        )
        .unwrap();
        // name, window_id positionals then the tags appended (no `--`).
        assert_eq!(argv[0], "s");
        assert_eq!(argv[1], "5");
        assert_eq!(&argv[2..], &["a".to_string(), "b".to_string()]);
        assert!(!argv.contains(&"--".to_string()));
    }

    #[test]
    fn build_args_spawn_uses_double_dash_separator() {
        let spec = find_tool("waymux_spawn").unwrap();
        let argv = build_call_args(
            spec,
            &obj(json!({ "name": "s", "argv": ["chromium", "--app=https://x"] })),
        )
        .unwrap();
        assert_eq!(argv[0], "s");
        let dd = argv
            .iter()
            .position(|a| a == "--")
            .expect("has -- separator");
        assert_eq!(
            &argv[dd + 1..],
            &["chromium".to_string(), "--app=https://x".to_string()]
        );
    }

    #[test]
    fn build_args_tag_requires_nonempty() {
        let spec = find_tool("waymux_tag").unwrap();
        let err = build_call_args(
            spec,
            &obj(json!({ "name": "s", "window_id": 1, "tags": [] })),
        )
        .unwrap_err();
        assert!(err.contains("at least one"), "{err}");
    }

    #[test]
    fn map_success_envelope_to_structured() {
        let spec = find_tool("waymux_ls").unwrap();
        let env = r#"{"ok":true,"verb":"ls","data":{"sessions":[]}}"#;
        let out = map_envelope(spec, env);
        assert!(!out.is_error);
        assert_eq!(out.structured, Some(json!({ "sessions": [] })));
        // text content present.
        assert!(out.content.iter().any(|c| c["type"] == "text"));
    }

    #[test]
    fn map_error_envelope_to_tool_error() {
        let spec = find_tool("waymux_info").unwrap();
        let env = r#"{"ok":false,"verb":"info","error":{"code":"E_NOT_FOUND","message":"no such session: eagle","detail":null}}"#;
        let out = map_envelope(spec, env);
        assert!(out.is_error);
        let text = out.content[0]["text"].as_str().unwrap();
        assert!(text.contains("E_NOT_FOUND"), "{text}");
        assert!(text.contains("no such session: eagle"), "{text}");
    }

    #[test]
    fn map_screenshot_returns_image_block() {
        let spec = find_tool("waymux_screenshot").unwrap();
        // "PNG" base64 stand-in.
        let env =
            r#"{"ok":true,"verb":"screenshot","data":{"width":4,"height":2,"png_b64":"AAAA"}}"#;
        let out = map_envelope(spec, env);
        assert!(!out.is_error);
        assert!(out.content.iter().any(|c| c["type"] == "image"
            && c["data"] == json!("AAAA")
            && c["mimeType"] == json!("image/png")));
        // structured retains the full data including png_b64.
        assert_eq!(out.structured.unwrap()["png_b64"], json!("AAAA"));
    }

    #[test]
    fn map_unparseable_stdout_is_error() {
        let spec = find_tool("waymux_ls").unwrap();
        let out = map_envelope(spec, "this is not json");
        assert!(out.is_error);
        assert!(out.content[0]["text"]
            .as_str()
            .unwrap()
            .contains("unparseable"));
    }

    #[test]
    fn compact_summary_elides_png_b64() {
        let s = compact_summary(&json!({ "width": 4, "png_b64": "AAAAAAAA" }));
        assert!(s.contains("<elided>"), "{s}");
        assert!(!s.contains("AAAAAAAA"), "{s}");
    }

    #[test]
    fn screenshot_placeholder_injected() {
        let spec = find_tool("waymux_screenshot").unwrap();
        let mut args = obj(json!({ "name": "s", "window_id": 1 }));
        inject_required_placeholders(spec, &mut args);
        assert_eq!(args.get("output"), Some(&json!("/dev/null")));
    }
}
