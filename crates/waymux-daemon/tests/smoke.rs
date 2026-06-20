// SPDX-License-Identifier: Apache-2.0

//! End-to-end smoke test: spawn the real daemon binary against an ephemeral
//! socket + state dir, then talk to it with the protocol crate over a
//! regular unix socket.
//!
//! Covers: hello handshake, first-request-must-be-hello gate, session
//! create/list/destroy lifecycle, not-found on an unknown session.

use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::time::timeout;
use waymux_protocol::{
    decode_frame, encode_frame, CreateSessionParams, CreateSessionResult, ErrorCode, HelloResult,
    Request, RequestMethod, Response, SessionInfo, CURRENT_PROTOCOL_VERSION,
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
        // `waymux-session` is a sibling workspace binary; Cargo does NOT set
        // CARGO_BIN_EXE for it when testing this crate. It does land in the
        // same target profile dir as the daemon. If it's missing (user ran
        // `cargo test -p waymux-daemon` without building the world), build it.
        let daemon_path = std::path::Path::new(daemon_bin);
        let session_bin = daemon_path
            .parent()
            .expect("daemon has a parent directory")
            .join("waymux-session");
        // Always rebuild: Cargo does NOT track the sibling binary as a test
        // dependency, so a stale `waymux-session` persists across code
        // changes. The build is a no-op when nothing's changed.
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let status = std::process::Command::new(cargo)
            .args(["build", "-p", "waymux-session"])
            .status()
            .expect("spawn cargo build -p waymux-session");
        assert!(status.success(), "cargo build -p waymux-session failed");
        assert!(
            session_bin.exists(),
            "waymux-session not found at {} after build",
            session_bin.display()
        );

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

        // Wait for the socket to appear.
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while std::time::Instant::now() < deadline {
            if socket.exists() && UnixStream::connect(&socket).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(socket.exists(), "daemon did not create socket");

        Self {
            child,
            socket,
            _state: state,
        }
    }

    async fn connect(&self) -> Client {
        let stream = timeout(Duration::from_secs(2), UnixStream::connect(&self.socket))
            .await
            .expect("connect not to time out")
            .expect("connect");
        Client { stream, next_id: 1 }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

struct Client {
    stream: UnixStream,
    next_id: u32,
}

impl Client {
    async fn request(&mut self, method: RequestMethod) -> Response {
        let id = self.next_id;
        self.next_id += 1;
        let req = Request { id, method };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        self.stream.write_all(&buf).await.unwrap();

        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut payload = vec![0u8; 4 + len];
        payload[..4].copy_from_slice(&len_buf);
        self.stream.read_exact(&mut payload[4..]).await.unwrap();
        let resp: Response = decode_frame(&payload).unwrap();
        assert_eq!(resp.id, id, "response id mismatch");
        resp
    }

    async fn hello(&mut self) {
        let resp = self
            .request(RequestMethod::Hello {
                client_protocol: CURRENT_PROTOCOL_VERSION,
            })
            .await;
        assert!(resp.ok, "hello failed: {:?}", resp.error);
        let r: HelloResult = resp.decode_result().unwrap();
        assert_eq!(r.server_protocol, CURRENT_PROTOCOL_VERSION);
    }
}

#[tokio::test]
async fn lifecycle() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    // Empty list initially.
    let resp = c.request(RequestMethod::ListSessions).await;
    assert!(resp.ok);
    let empty: Vec<SessionInfo> = resp.decode_result().unwrap();
    assert!(empty.is_empty());

    // Create a session.
    let resp = c
        .request(RequestMethod::CreateSession(CreateSessionParams {
            name: "test".into(),
            width: 800,
            height: 600,
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
    let cr: CreateSessionResult = resp.decode_result().unwrap();
    assert_eq!(cr.name, "test");
    assert!(!cr.inner_socket_path.is_empty());

    // Creating again fails with E_ALREADY_EXISTS.
    let resp = c
        .request(RequestMethod::CreateSession(CreateSessionParams {
            name: "test".into(),
            width: 800,
            height: 600,
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
    assert!(!resp.ok);
    assert_eq!(resp.error.unwrap().code, ErrorCode::AlreadyExists);

    // List now has the session.
    let resp = c.request(RequestMethod::ListSessions).await;
    let list: Vec<SessionInfo> = resp.decode_result().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "test");
    assert_eq!(list[0].width, 800);
    assert_eq!(list[0].height, 600);
    assert!(list[0].pid > 0);
    assert!(!list[0].attached);

    // Destroy → listed as gone.
    let resp = c
        .request(RequestMethod::DestroySession {
            name: "test".into(),
        })
        .await;
    assert!(resp.ok, "destroy failed: {:?}", resp.error);

    let resp = c.request(RequestMethod::ListSessions).await;
    let list: Vec<SessionInfo> = resp.decode_result().unwrap();
    assert!(list.is_empty());

    // Destroy unknown → E_NOT_FOUND.
    let resp = c
        .request(RequestMethod::DestroySession {
            name: "nope".into(),
        })
        .await;
    assert!(!resp.ok);
    assert_eq!(resp.error.unwrap().code, ErrorCode::NotFound);
}

#[tokio::test]
async fn hello_required_first() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    // Skip hello; first request should be rejected.
    let resp = c.request(RequestMethod::ListSessions).await;
    assert!(!resp.ok);
    assert_eq!(resp.error.unwrap().code, ErrorCode::ProtoVersion);
}

#[tokio::test]
async fn rejects_unknown_protocol_version() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    let resp = c
        .request(RequestMethod::Hello {
            client_protocol: 999,
        })
        .await;
    assert!(!resp.ok);
    assert_eq!(resp.error.unwrap().code, ErrorCode::ProtoVersion);
}

/// End-to-end check for pre-launch hardening tasks #4 + #5: a session
/// created with cpu/disk/fd caps must succeed (the wire fields are
/// accepted, the daemon doesn't reject the request) and — on a host with
/// cgroup-v2 + delegated cpu controller — `cpu.max` should reflect the
/// requested cap. The disk + fd caps degrade silently in unprivileged
/// environments per the surrounding "best effort" contract; this test
/// asserts only that none of the three turns into a hard error.
#[tokio::test]
async fn hardening_session_with_cpu_disk_fd_caps_succeeds() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    let resp = c
        .request(RequestMethod::CreateSession(CreateSessionParams {
            name: "hardened".into(),
            width: 320,
            height: 240,
            scale: 1,
            env: Default::default(),
            share_clipboard: false,
            share_audio: false,
            mem_cap_mb: Some(64),
            cpu_cap_pct: Some(150),
            disk_quota_mb: Some(8),
            fd_limit: Some(1024),
            api_key_id: None,
            codec: None,
            gpu_type: None,
        }))
        .await;
    assert!(resp.ok, "create with caps failed: {:?}", resp.error);

    // If we have cgroup-v2 with the cpu controller delegated to us, the
    // daemon's leaf for this session has cpu.max set. Skip the assertion
    // gracefully when cgroup-v2 isn't usable from inside the test sandbox.
    if let Some(parent) = daemon_cgroup_dir() {
        let leaf = parent.join("waymux-hardened");
        match std::fs::read_to_string(leaf.join("cpu.max")) {
            Ok(contents) => {
                assert_eq!(
                    contents.trim(),
                    "150000 100000",
                    "cpu.max should be quota=150000us / period=100000us"
                );
            }
            Err(_) => {
                eprintln!(
                    "skipping cpu.max readback: cgroup leaf at {} unreadable",
                    leaf.display()
                );
            }
        }
    } else {
        eprintln!("skipping cpu.max readback: cgroup-v2 not detected");
    }

    let _ = c
        .request(RequestMethod::DestroySession {
            name: "hardened".into(),
        })
        .await;
}

/// Mirror the daemon's cgroup-v2 discovery so the integration test can
/// look up the leaf without coupling to private internals.
fn daemon_cgroup_dir() -> Option<PathBuf> {
    if !std::path::Path::new("/sys/fs/cgroup/cgroup.controllers").is_file() {
        return None;
    }
    let raw = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let v2_path = raw
        .lines()
        .find_map(|l| l.strip_prefix("0::"))?
        .trim_start_matches('/');
    let mut p = PathBuf::from("/sys/fs/cgroup");
    p.push(v2_path);
    Some(p)
}
