// SPDX-License-Identifier: Apache-2.0

//! Integration tests for daemon ↔ session-control-socket RPCs:
//! list_windows, resize, tag_window.

use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::time::{timeout, Instant};
use waymux_protocol::{
    encode_frame, CreateSessionParams, ErrorCode, HelloResult, Request, RequestMethod, Response,
    SessionInfo, WindowInfo, CURRENT_PROTOCOL_VERSION,
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
        let resp: Response = rmp_serde::from_slice(&payload[4..]).unwrap();
        assert_eq!(resp.id, id, "response id mismatch");
        resp
    }

    async fn hello(&mut self) {
        let resp = self
            .request(RequestMethod::Hello {
                client_protocol: CURRENT_PROTOCOL_VERSION,
            })
            .await;
        let _: HelloResult = resp.decode_result().unwrap();
    }
}

#[tokio::test]
async fn list_windows_empty_for_stub_session() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "win".into(),
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

    let resp = c
        .request(RequestMethod::ListWindows { name: "win".into() })
        .await;
    assert!(resp.ok, "list_windows failed: {:?}", resp.error);
    let windows: Vec<WindowInfo> = resp.decode_result().unwrap();
    assert!(
        windows.is_empty(),
        "stub session should report zero windows"
    );
}

#[tokio::test]
async fn list_windows_rejects_unknown_session() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    let resp = c
        .request(RequestMethod::ListWindows {
            name: "nope".into(),
        })
        .await;
    assert!(!resp.ok);
    assert_eq!(resp.error.unwrap().code, ErrorCode::NotFound);
}

#[tokio::test]
async fn resize_updates_session_and_daemon_metadata() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "rz".into(),
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

    let resp = c
        .request(RequestMethod::Resize {
            name: "rz".into(),
            width: 1024,
            height: 768,
        })
        .await;
    assert!(resp.ok, "resize failed: {:?}", resp.error);

    let resp = c.request(RequestMethod::ListSessions).await;
    let sessions: Vec<SessionInfo> = resp.decode_result().unwrap();
    let s = sessions.iter().find(|s| s.name == "rz").unwrap();
    assert_eq!(s.width, 1024);
    assert_eq!(s.height, 768);
}

#[tokio::test]
async fn attach_returns_connectable_socket_advertising_waymux_attach_v1() {
    use wayland_client::{globals::registry_queue_init, Connection, Dispatch, QueueHandle};

    struct DummyState;
    impl
        Dispatch<
            wayland_client::protocol::wl_registry::WlRegistry,
            wayland_client::globals::GlobalListContents,
        > for DummyState
    {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_registry::WlRegistry,
            _: wayland_client::protocol::wl_registry::Event,
            _: &wayland_client::globals::GlobalListContents,
            _: &Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "atx".into(),
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

    let resp = c
        .request(RequestMethod::Attach { name: "atx".into() })
        .await;
    assert!(resp.ok, "attach failed: {:?}", resp.error);
    #[derive(serde::Deserialize)]
    struct Ar {
        attach_socket_path: String,
    }
    let r: Ar = resp.decode_result().unwrap();
    let attach_path = PathBuf::from(&r.attach_socket_path);
    assert!(
        attach_path.exists(),
        "attach socket does not exist: {:?}",
        attach_path
    );

    // Connect a real wayland-client and verify the global is advertised.
    // Wait briefly for the server thread to finish bind+accept.
    let deadline = Instant::now() + Duration::from_secs(2);
    let saw_global = loop {
        let saw = tokio::task::spawn_blocking({
            let p = attach_path.clone();
            move || -> anyhow::Result<bool> {
                let stream = std::os::unix::net::UnixStream::connect(&p)?;
                let conn = Connection::from_socket(stream)?;
                let (globals, _queue) = registry_queue_init::<DummyState>(&conn)?;
                let list = globals.contents().clone_list();
                Ok(list.iter().any(|g| g.interface == "waymux_attach_v1"))
            }
        })
        .await
        .unwrap();
        match saw {
            Ok(true) => break true,
            Ok(false) => {}
            Err(_) => {}
        }
        if Instant::now() >= deadline {
            break false;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(
        saw_global,
        "waymux_attach_v1 not advertised on attach socket"
    );

    // detach flips the bit and doesn't error on an unattached-then-detached session.
    let resp = c
        .request(RequestMethod::Detach { name: "atx".into() })
        .await;
    assert!(resp.ok, "detach failed: {:?}", resp.error);
}

#[tokio::test]
async fn attach_unknown_session_returns_not_found() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    let resp = c
        .request(RequestMethod::Attach {
            name: "nope".into(),
        })
        .await;
    assert!(!resp.ok);
    assert_eq!(resp.error.unwrap().code, ErrorCode::NotFound);
}

#[tokio::test]
async fn tag_window_rejects_unknown_id() {
    // With a stub session reporting no windows, any tag_window call should
    // fail with E_NOT_FOUND. When the real compositor lands this test will
    // need a spawned client to succeed.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "tg".into(),
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

    let resp = c
        .request(RequestMethod::TagWindow {
            name: "tg".into(),
            window_id: 1,
            tags: vec!["main".into()],
        })
        .await;
    assert!(!resp.ok);
    assert_eq!(resp.error.unwrap().code, ErrorCode::NotFound);
}
