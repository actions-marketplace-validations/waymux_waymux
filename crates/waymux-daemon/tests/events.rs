// SPDX-License-Identifier: Apache-2.0

//! Events + spawn integration tests.
//!
//! Shares plumbing with smoke.rs but adds an event-aware client that can
//! distinguish server-pushed events from request/response traffic on the
//! same connection.

use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::time::{timeout, Instant};
use waymux_protocol::{
    encode_frame, CreateSessionParams, Event, EventBody, HelloResult, Request, RequestMethod,
    Response, CURRENT_PROTOCOL_VERSION,
};

struct Daemon {
    child: Child,
    socket: PathBuf,
    _state: TempDir,
}

impl Daemon {
    async fn spawn() -> Self {
        let state = TempDir::new().expect("tempdir");
        let socket = state.path().join("waymux.sock");
        let daemon_bin = env!("CARGO_BIN_EXE_waymuxd");
        let session_bin = std::path::Path::new(daemon_bin)
            .parent()
            .unwrap()
            .join("waymux-session");
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        assert!(std::process::Command::new(cargo)
            .args(["build", "-p", "waymux-session"])
            .status()
            .expect("cargo build")
            .success());

        let child = Command::new(daemon_bin)
            .arg("--socket")
            .arg(&socket)
            .arg("--state-dir")
            .arg(state.path().join("state"))
            .arg("--session-bin")
            .arg(&session_bin)
            .env("RUST_LOG", "warn")
            .kill_on_drop(true)
            .spawn()
            .expect("spawn daemon");

        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if socket.exists() && UnixStream::connect(&socket).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Self {
            child,
            socket,
            _state: state,
        }
    }

    async fn connect(&self) -> Client {
        let stream = timeout(Duration::from_secs(2), UnixStream::connect(&self.socket))
            .await
            .unwrap()
            .unwrap();
        Client { stream, next_id: 1 }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

enum Incoming {
    Response(Response),
    Event(Event),
}

struct Client {
    stream: UnixStream,
    next_id: u32,
}

impl Client {
    async fn send_request(&mut self, method: RequestMethod) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let req = Request { id, method };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        self.stream.write_all(&buf).await.unwrap();
        id
    }

    async fn read_one(&mut self) -> Incoming {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload).await.unwrap();

        if let Ok(resp) = rmp_serde::from_slice::<Response>(&payload) {
            Incoming::Response(resp)
        } else if let Ok(ev) = rmp_serde::from_slice::<Event>(&payload) {
            Incoming::Event(ev)
        } else {
            panic!("frame is neither response nor event: {:?}", payload);
        }
    }

    /// Send a request and return the response, queuing any events that arrive
    /// before the matching response.
    async fn request(&mut self, method: RequestMethod) -> (Response, Vec<Event>) {
        let id = self.send_request(method).await;
        let mut events = Vec::new();
        loop {
            match self.read_one().await {
                Incoming::Response(r) if r.id == id => return (r, events),
                Incoming::Response(r) => {
                    panic!("unexpected response id {} while waiting for {}", r.id, id)
                }
                Incoming::Event(e) => events.push(e),
            }
        }
    }

    async fn hello(&mut self) {
        let (resp, _) = self
            .request(RequestMethod::Hello {
                client_protocol: CURRENT_PROTOCOL_VERSION,
            })
            .await;
        assert!(resp.ok, "hello failed: {:?}", resp.error);
        let r: HelloResult = resp.decode_result().unwrap();
        assert!(r.capabilities.iter().any(|c| c == "subscribe"));
    }

    async fn next_event(&mut self, dur: Duration) -> Event {
        timeout(dur, async move {
            loop {
                if let Incoming::Event(e) = self.read_one().await {
                    return e;
                }
            }
        })
        .await
        .expect("event within timeout")
    }
}

#[tokio::test]
async fn session_events_fire_on_create_and_destroy() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    let (resp, events) = c
        .request(RequestMethod::Subscribe {
            topics: vec!["sessions".into()],
        })
        .await;
    assert!(resp.ok, "subscribe failed: {:?}", resp.error);
    assert!(events.is_empty());

    let (resp, events) = c
        .request(RequestMethod::CreateSession(CreateSessionParams {
            name: "ev".into(),
            width: 640,
            height: 480,
            scale: 1,
            env: Default::default(),
            share_clipboard: false,
            share_audio: false,
            mem_cap_mb: None,
            cpu_cap_pct: None,
            disk_quota_mb: None,
            fd_limit: None,
            api_key_id: None,
            codec: None,
            gpu_type: None,
        }))
        .await;
    assert!(resp.ok, "create failed: {:?}", resp.error);
    // session_created might arrive before or after the response; fetch it.
    let ev = if let Some(e) = events.into_iter().next() {
        e
    } else {
        c.next_event(Duration::from_secs(2)).await
    };
    match ev.body {
        EventBody::SessionCreated { name } => assert_eq!(name, "ev"),
        other => panic!("expected SessionCreated, got {:?}", other),
    }

    let (resp, _) = c
        .request(RequestMethod::DestroySession { name: "ev".into() })
        .await;
    assert!(resp.ok);
    // The supervisor broadcasts session_destroyed after the child exits.
    let ev = c.next_event(Duration::from_secs(5)).await;
    match ev.body {
        EventBody::SessionDestroyed { name, .. } => assert_eq!(name, "ev"),
        other => panic!("expected SessionDestroyed, got {:?}", other),
    }
}

#[tokio::test]
async fn spawn_child_fires_child_exited_event() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["sessions".into()],
    })
    .await;

    let (_resp, ev_during_create) = c
        .request(RequestMethod::CreateSession(CreateSessionParams {
            name: "sp".into(),
            width: 640,
            height: 480,
            scale: 1,
            env: Default::default(),
            share_clipboard: false,
            share_audio: false,
            mem_cap_mb: None,
            cpu_cap_pct: None,
            disk_quota_mb: None,
            fd_limit: None,
            api_key_id: None,
            codec: None,
            gpu_type: None,
        }))
        .await;
    eprintln!("events during create: {:?}", ev_during_create);

    // Spawn a process that exits immediately.
    let (resp, _) = c
        .request(RequestMethod::Spawn {
            name: "sp".into(),
            argv: vec!["/bin/true".into()],
            env: Default::default(),
            compositor: false,
        })
        .await;
    assert!(resp.ok, "spawn failed: {:?}", resp.error);
    #[derive(serde::Deserialize)]
    struct SpawnOk {
        pid: i32,
    }
    let r: SpawnOk = resp.decode_result().unwrap();
    assert!(r.pid > 0);

    // Drain events looking for child_exited; log anything else we see.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let remaining = match deadline.checked_duration_since(Instant::now()) {
            Some(r) if r > Duration::ZERO => r,
            _ => panic!("timed out waiting for child_exited"),
        };
        match timeout(remaining, async {
            loop {
                if let Incoming::Event(e) = c.read_one().await {
                    return e;
                }
            }
        })
        .await
        {
            Err(_) => panic!("timed out waiting for child_exited"),
            Ok(ev) => match ev.body {
                EventBody::ChildExited {
                    name,
                    pid,
                    exit_code,
                } => {
                    assert_eq!(name, "sp");
                    assert_eq!(pid, r.pid);
                    assert_eq!(exit_code, 0);
                    return;
                }
                other => {
                    eprintln!("ignoring event: {:?}", other);
                }
            },
        }
    }
}

#[tokio::test]
async fn compositor_child_exit_fires_session_crashed_event() {
    // A child spawned with `compositor: true` is the inner compositor. When
    // it exits, the daemon emits BOTH ChildExited (for legacy consumers) and
    // SessionCrashed (the SDK-facing signal that the session is now dark).
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["sessions".into()],
    })
    .await;

    let (_resp, _ev_during_create) = c
        .request(RequestMethod::CreateSession(CreateSessionParams {
            name: "crash".into(),
            width: 640,
            height: 480,
            scale: 1,
            env: Default::default(),
            share_clipboard: false,
            share_audio: false,
            mem_cap_mb: None,
            cpu_cap_pct: None,
            disk_quota_mb: None,
            fd_limit: None,
            api_key_id: None,
            codec: None,
            gpu_type: None,
        }))
        .await;

    let (resp, _) = c
        .request(RequestMethod::Spawn {
            name: "crash".into(),
            argv: vec!["/bin/sh".into(), "-c".into(), "exit 7".into()],
            env: Default::default(),
            compositor: true,
        })
        .await;
    assert!(resp.ok, "spawn failed: {:?}", resp.error);
    #[derive(serde::Deserialize)]
    struct SpawnOk {
        pid: i32,
    }
    let r: SpawnOk = resp.decode_result().unwrap();

    // Drain until SessionCrashed appears (with the expected exit code).
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut saw_child_exited = false;
    loop {
        let remaining = match deadline.checked_duration_since(Instant::now()) {
            Some(r) if r > Duration::ZERO => r,
            _ => panic!(
                "timed out waiting for SessionCrashed (saw_child_exited={})",
                saw_child_exited
            ),
        };
        match timeout(remaining, async {
            loop {
                if let Incoming::Event(e) = c.read_one().await {
                    return e;
                }
            }
        })
        .await
        {
            Err(_) => panic!(
                "timed out waiting for SessionCrashed (saw_child_exited={})",
                saw_child_exited
            ),
            Ok(ev) => match ev.body {
                EventBody::ChildExited {
                    name,
                    pid,
                    exit_code,
                } => {
                    assert_eq!(name, "crash");
                    assert_eq!(pid, r.pid);
                    assert_eq!(exit_code, 7);
                    saw_child_exited = true;
                }
                EventBody::SessionCrashed {
                    name,
                    pid,
                    exit_code,
                } => {
                    assert_eq!(name, "crash");
                    assert_eq!(pid, r.pid);
                    assert_eq!(exit_code, 7);
                    assert!(
                        saw_child_exited,
                        "ChildExited should arrive before SessionCrashed"
                    );
                    return;
                }
                other => {
                    eprintln!("ignoring event: {:?}", other);
                }
            },
        }
    }
}

#[tokio::test]
async fn log_events_capture_session_stderr() {
    // The session binary writes a tracing `INFO` line to stderr on startup
    // ("session starting ..."). Subscribing to `logs:<name>` before the
    // create should surface that line as a Log event.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["logs:loggy".into()],
    })
    .await;

    let (_resp, ev_during_create) = c
        .request(RequestMethod::CreateSession(CreateSessionParams {
            name: "loggy".into(),
            width: 320,
            height: 240,
            scale: 1,
            env: {
                // Force the session to emit at least one line at INFO.
                let mut e: std::collections::BTreeMap<String, String> = Default::default();
                e.insert("RUST_LOG".into(), "info".into());
                e
            },
            share_clipboard: false,
            share_audio: false,
            mem_cap_mb: None,
            cpu_cap_pct: None,
            disk_quota_mb: None,
            fd_limit: None,
            api_key_id: None,
            codec: None,
            gpu_type: None,
        }))
        .await;

    // Collect events that arrived during create.
    let mut saw_log = false;
    for ev in ev_during_create {
        if let EventBody::Log { name, text, .. } = &ev.body {
            if name == "loggy" && text.contains("session starting") {
                saw_log = true;
                break;
            }
        }
    }

    // If not in the create-response burst, wait for more.
    if !saw_log {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline && !saw_log {
            let remaining = deadline - Instant::now();
            let ev = match timeout(remaining, async {
                loop {
                    if let Incoming::Event(e) = c.read_one().await {
                        return e;
                    }
                }
            })
            .await
            {
                Ok(e) => e,
                Err(_) => break,
            };
            if let EventBody::Log { name, text, .. } = &ev.body {
                if name == "loggy" && text.contains("session starting") {
                    saw_log = true;
                }
            }
        }
    }
    assert!(saw_log, "never received the session's startup log line");
}

#[tokio::test]
async fn spawn_rejects_unknown_session() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    let (resp, _) = c
        .request(RequestMethod::Spawn {
            name: "nope".into(),
            argv: vec!["/bin/true".into()],
            env: Default::default(),
            compositor: false,
        })
        .await;
    assert!(!resp.ok);
    assert_eq!(
        resp.error.unwrap().code,
        waymux_protocol::ErrorCode::NotFound
    );
}
