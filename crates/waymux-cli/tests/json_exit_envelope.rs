// SPDX-License-Identifier: Apache-2.0

//! Integration test: the `idle` and `wait` verbs call `std::process::exit(1)`
//! on a non-zero outcome (busy / timeout). Under `--json` they MUST still emit
//! their success envelope on stdout BEFORE exiting, so a machine consumer sees
//! the structured result and not just a bare non-zero exit code.
//!
//! We can't observe a `process::exit` from inside the crate, so we drive the
//! REAL built `waymux` binary (`CARGO_BIN_EXE_waymux`) as a subprocess against
//! a tiny in-process fake daemon on a Unix socket, then capture stdout + the
//! exit status.

use std::process::Stdio;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use waymux_protocol::{
    decode_frame, encode_frame, Request, RequestMethod, Response, WindowInfo,
    CURRENT_PROTOCOL_VERSION,
};

/// Read one length-prefixed frame; `None` on clean EOF.
async fn read_frame(stream: &mut UnixStream) -> Option<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.ok()?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut payload = vec![0u8; 4 + len];
    payload[..4].copy_from_slice(&len_buf);
    stream.read_exact(&mut payload[4..]).await.ok()?;
    Some(payload)
}

async fn write_response(stream: &mut UnixStream, resp: &Response) {
    let mut buf = Vec::new();
    encode_frame(resp, &mut buf).unwrap();
    let _ = stream.write_all(&buf).await;
}

#[derive(serde::Serialize)]
struct HelloOk {
    server_protocol: u32,
    capabilities: Vec<String>,
}

#[derive(serde::Serialize)]
struct IdleResult {
    idle: bool,
}

/// Serve a single connection: complete the hello handshake, then answer
/// whatever request the verb sends. `idle` sends `WaitForIdle`; `wait` sends
/// `Subscribe` then `ListWindows` (and then waits, which we let time out by
/// never sending window events).
async fn serve_one(listener: UnixListener) {
    let (mut stream, _) = match listener.accept().await {
        Ok(v) => v,
        Err(_) => return,
    };
    loop {
        let frame = match read_frame(&mut stream).await {
            Some(f) => f,
            None => return,
        };
        let req: Request = match decode_frame(&frame) {
            Ok(r) => r,
            Err(_) => return,
        };
        match req.method {
            RequestMethod::Hello { .. } => {
                let resp = Response::success(
                    req.id,
                    &HelloOk {
                        server_protocol: CURRENT_PROTOCOL_VERSION,
                        capabilities: vec!["subscribe".into()],
                    },
                )
                .unwrap();
                write_response(&mut stream, &resp).await;
            }
            RequestMethod::WaitForIdle { .. } => {
                // Always busy -> the CLI prints the idle envelope then exits 1.
                let resp = Response::success(req.id, &IdleResult { idle: false }).unwrap();
                write_response(&mut stream, &resp).await;
            }
            RequestMethod::Subscribe { .. } => {
                #[derive(serde::Serialize)]
                struct Unit {}
                let resp = Response::success(req.id, &Unit {}).unwrap();
                write_response(&mut stream, &resp).await;
            }
            RequestMethod::ListWindows { .. } => {
                // No windows ever appear; `wait` will time out and exit 1.
                let empty: Vec<WindowInfo> = vec![];
                let resp = Response::success(req.id, &empty).unwrap();
                write_response(&mut stream, &resp).await;
            }
            _ => {
                // Unexpected verb: fail closed.
                return;
            }
        }
    }
}

/// Run the built `waymux` binary against the fake daemon and return
/// (exit_code, stdout).
async fn run_cli(socket: &std::path::Path, extra_args: &[&str]) -> (i32, String) {
    let bin = env!("CARGO_BIN_EXE_waymux");
    let mut args: Vec<&str> = vec!["--json", "--socket"];
    let sock_str = socket.to_str().unwrap();
    args.push(sock_str);
    args.extend_from_slice(extra_args);

    let output = tokio::process::Command::new(bin)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn waymux");
    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    (code, stdout)
}

#[tokio::test]
async fn idle_emits_json_envelope_before_nonzero_exit() {
    let tmp = tempfile::tempdir().unwrap();
    let socket = tmp.path().join("waymux.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = tokio::spawn(serve_one(listener));

    let (code, stdout) = run_cli(&socket, &["idle", "eagle"]).await;
    let _ = server.await;

    // Non-zero exit (busy), AND the envelope is on stdout.
    assert_eq!(code, 1, "idle on a busy session must exit 1");
    let env: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout not JSON: {e}\n{stdout}"));
    assert_eq!(env["ok"], serde_json::json!(true));
    assert_eq!(env["verb"], serde_json::json!("idle"));
    assert_eq!(env["data"]["idle"], serde_json::json!(false));
}

#[tokio::test]
async fn wait_emits_json_envelope_before_nonzero_exit() {
    let tmp = tempfile::tempdir().unwrap();
    let socket = tmp.path().join("waymux.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = tokio::spawn(serve_one(listener));

    // Short timeout so the test is fast; no window ever appears -> timeout.
    let (code, stdout) = run_cli(
        &socket,
        &["wait", "eagle", "--app-id", "ghost", "--timeout-ms", "200"],
    )
    .await;
    let _ = server.await;

    assert_eq!(code, 1, "wait timeout must exit 1");
    let env: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout not JSON: {e}\n{stdout}"));
    assert_eq!(env["ok"], serde_json::json!(true));
    assert_eq!(env["verb"], serde_json::json!("wait"));
    // On timeout the data is null (envelope present, value null).
    assert_eq!(env["data"], serde_json::Value::Null);
}
