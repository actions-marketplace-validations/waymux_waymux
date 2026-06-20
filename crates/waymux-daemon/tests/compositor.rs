// SPDX-License-Identifier: Apache-2.0

//! End-to-end compositor test: connect a real wayland-client to the
//! session's inner socket, create an xdg_toplevel, and verify the daemon's
//! `list_windows` reports it.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::time::{timeout, Instant};
use waymux_protocol::{
    encode_frame, CreateSessionParams, Event, EventBody, HelloResult, Rect, Request, RequestMethod,
    Response, WindowInfo, CURRENT_PROTOCOL_VERSION,
};

use wayland_client::{
    globals::{registry_queue_init, GlobalListContents},
    protocol::{
        wl_buffer::WlBuffer,
        wl_compositor::WlCompositor,
        wl_keyboard::{self, WlKeyboard},
        wl_pointer::{self, WlPointer},
        wl_registry::WlRegistry,
        wl_seat::{self, WlSeat},
        wl_shm::{self, WlShm},
        wl_shm_pool::WlShmPool,
        wl_surface::WlSurface,
        wl_touch::{self, WlTouch},
    },
    Connection, Dispatch, QueueHandle,
};
use wayland_protocols::xdg::shell::client::{
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};

struct Daemon {
    child: Child,
    socket: PathBuf,
    state_dir: PathBuf,
    _state: TempDir,
}

impl Daemon {
    async fn spawn() -> Self {
        let state = TempDir::new().expect("tempdir");
        let socket = state.path().join("waymux.sock");
        let state_dir = state.path().join("state");
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
            .arg(&state_dir)
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
            state_dir,
            _state: state,
        }
    }

    async fn connect(&self) -> DaemonClient {
        let stream = timeout(Duration::from_secs(2), UnixStream::connect(&self.socket))
            .await
            .unwrap()
            .unwrap();
        DaemonClient { stream, next_id: 1 }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

struct DaemonClient {
    stream: UnixStream,
    next_id: u32,
}

enum Incoming {
    Response(Response),
    Event(Event),
}

impl DaemonClient {
    async fn read_one(&mut self) -> Incoming {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload).await.unwrap();
        if let Ok(r) = rmp_serde::from_slice::<Response>(&payload) {
            Incoming::Response(r)
        } else if let Ok(e) = rmp_serde::from_slice::<Event>(&payload) {
            Incoming::Event(e)
        } else {
            panic!("frame is neither response nor event");
        }
    }

    async fn request(&mut self, method: RequestMethod) -> Response {
        self.request_collect(method).await.0
    }

    async fn request_collect(&mut self, method: RequestMethod) -> (Response, Vec<Event>) {
        let id = self.next_id;
        self.next_id += 1;
        let req = Request { id, method };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        self.stream.write_all(&buf).await.unwrap();
        let mut events = Vec::new();
        loop {
            match self.read_one().await {
                Incoming::Response(r) if r.id == id => return (r, events),
                Incoming::Response(r) => panic!("unexpected response id {}", r.id),
                Incoming::Event(e) => events.push(e),
            }
        }
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

    async fn hello(&mut self) {
        let r = self
            .request(RequestMethod::Hello {
                client_protocol: CURRENT_PROTOCOL_VERSION,
            })
            .await;
        let _: HelloResult = r.decode_result().unwrap();
    }
}

// ─── Wayland-client minimal dispatch ────────────────────────────────────

#[derive(Default)]
struct ClientState {
    compositor: Option<WlCompositor>,
    wm_base: Option<XdgWmBase>,
    configured: Arc<Mutex<bool>>,
    /// Most recent non-zero `xdg_toplevel.configure(width, height)` the client
    /// received. The resize test reads this to confirm the engine re-sent a
    /// configure carrying the new session size.
    last_toplevel_size: Arc<Mutex<Option<(i32, i32)>>>,
}

impl Dispatch<WlRegistry, GlobalListContents> for ClientState {
    fn event(
        _state: &mut Self,
        _proxy: &WlRegistry,
        _event: wayland_client::protocol::wl_registry::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlCompositor, ()> for ClientState {
    fn event(
        _state: &mut Self,
        _proxy: &WlCompositor,
        _event: wayland_client::protocol::wl_compositor::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlSurface, ()> for ClientState {
    fn event(
        _state: &mut Self,
        _proxy: &WlSurface,
        _event: wayland_client::protocol::wl_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<XdgWmBase, ()> for ClientState {
    fn event(
        _state: &mut Self,
        proxy: &XdgWmBase,
        event: xdg_wm_base::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
}

impl Dispatch<XdgSurface, ()> for ClientState {
    fn event(
        state: &mut Self,
        proxy: &XdgSurface,
        event: xdg_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            proxy.ack_configure(serial);
            *state.configured.lock().unwrap() = true;
        }
    }
}

impl Dispatch<XdgToplevel, ()> for ClientState {
    fn event(
        state: &mut Self,
        _proxy: &XdgToplevel,
        event: xdg_toplevel::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Configure { width, height, .. } = event {
            // width/height of 0 means "client picks its own size"; record only
            // the explicit non-zero sizes the engine sends.
            if width > 0 && height > 0 {
                *state.last_toplevel_size.lock().unwrap() = Some((width, height));
            }
        }
    }
}

impl Dispatch<WlShm, ()> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlShm,
        _e: wayland_client::protocol::wl_shm::Event,
        _d: &(),
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlShmPool, ()> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlShmPool,
        _e: wayland_client::protocol::wl_shm_pool::Event,
        _d: &(),
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlBuffer, ()> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlBuffer,
        _e: wayland_client::protocol::wl_buffer::Event,
        _d: &(),
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlSeat, ()> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlSeat,
        _e: wayland_client::protocol::wl_seat::Event,
        _d: &(),
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
    }
}

#[derive(Debug, Clone, PartialEq)]
enum KbdEvent {
    Keymap,
    Enter,
    Leave,
    Modifiers,
    Key { keycode: u32, pressed: bool },
}

/// Records keyboard events in arrival order so tests can verify
/// sequences like "keymap → enter → key".
#[derive(Default, Clone)]
struct KeyLog(Arc<Mutex<Vec<KbdEvent>>>);

impl KeyLog {
    fn snapshot(&self) -> Vec<KbdEvent> {
        self.0.lock().unwrap().clone()
    }
}

/// Records pointer events for assertion.
#[derive(Default, Clone)]
struct PointerLog(Arc<Mutex<Vec<PointerEvent>>>);

#[derive(Debug, Clone, PartialEq)]
enum PointerEvent {
    Enter { x: f64, y: f64 },
    Leave,
    Motion { x: f64, y: f64 },
    Button { button: u32, pressed: bool },
    Frame,
}

impl Dispatch<WlPointer, PointerLog> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlPointer,
        event: wl_pointer::Event,
        log: &PointerLog,
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter {
                surface_x,
                surface_y,
                ..
            } => {
                log.0.lock().unwrap().push(PointerEvent::Enter {
                    x: surface_x,
                    y: surface_y,
                });
            }
            wl_pointer::Event::Leave { .. } => {
                log.0.lock().unwrap().push(PointerEvent::Leave);
            }
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                log.0.lock().unwrap().push(PointerEvent::Motion {
                    x: surface_x,
                    y: surface_y,
                });
            }
            wl_pointer::Event::Button { button, state, .. } => {
                let pressed = matches!(
                    state,
                    wayland_client::WEnum::Value(wl_pointer::ButtonState::Pressed)
                );
                log.0
                    .lock()
                    .unwrap()
                    .push(PointerEvent::Button { button, pressed });
            }
            wl_pointer::Event::Frame => {
                log.0.lock().unwrap().push(PointerEvent::Frame);
            }
            _ => {}
        }
    }
}

impl Dispatch<WlKeyboard, KeyLog> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlKeyboard,
        event: wl_keyboard::Event,
        log: &KeyLog,
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
        let e = match event {
            wl_keyboard::Event::Keymap { .. } => KbdEvent::Keymap,
            wl_keyboard::Event::Enter { .. } => KbdEvent::Enter,
            wl_keyboard::Event::Leave { .. } => KbdEvent::Leave,
            wl_keyboard::Event::Modifiers { .. } => KbdEvent::Modifiers,
            wl_keyboard::Event::Key { key, state, .. } => {
                let pressed = matches!(
                    state,
                    wayland_client::WEnum::Value(wl_keyboard::KeyState::Pressed)
                );
                KbdEvent::Key {
                    keycode: key,
                    pressed,
                }
            }
            _ => return,
        };
        log.0.lock().unwrap().push(e);
    }
}

// ─── The test ────────────────────────────────────────────────────────────

#[tokio::test]
async fn toplevel_created_by_client_shows_up_in_list_windows() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "comp".into(),
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

    let wayland_sock = d.state_dir.join("comp").join("wayland.sock");

    // Connect a real Wayland client. Do this on a blocking thread — the
    // wayland-client Connection API is sync.
    let connected = tokio::task::spawn_blocking({
        let wayland_sock = wayland_sock.clone();
        move || -> anyhow::Result<Arc<Mutex<bool>>> {
            let stream = std::os::unix::net::UnixStream::connect(&wayland_sock)?;
            let conn = Connection::from_socket(stream)?;
            let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
            let qh = queue.handle();

            let mut state = ClientState {
                compositor: Some(globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?),
                wm_base: Some(globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?),
                ..Default::default()
            };

            let compositor = state.compositor.clone().unwrap();
            let wm_base = state.wm_base.clone().unwrap();

            let surface = compositor.create_surface(&qh, ());
            let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
            let toplevel = xdg_surface.get_toplevel(&qh, ());
            toplevel.set_app_id("waymux.test".into());
            toplevel.set_title("Test Window".into());
            surface.commit();

            // Drive the queue until we get the configure event.
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            let configured = state.configured.clone();
            while !*configured.lock().unwrap() {
                if std::time::Instant::now() >= deadline {
                    anyhow::bail!("client never received configure");
                }
                queue.blocking_dispatch(&mut state)?;
            }

            // Keep the connection alive for the test to observe; leak the
            // owned objects so they don't drop.
            std::mem::forget(toplevel);
            std::mem::forget(xdg_surface);
            std::mem::forget(surface);
            std::mem::forget(conn);
            std::mem::forget(queue);
            std::mem::forget(state);
            Ok(configured)
        }
    });
    let _keep = connected
        .await
        .expect("client thread")
        .expect("client setup");

    // Poll list_windows until the toplevel shows up (or we time out).
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let resp = c
            .request(RequestMethod::ListWindows {
                name: "comp".into(),
            })
            .await;
        let windows: Vec<WindowInfo> = resp.decode_result().unwrap();
        if let Some(w) = windows.iter().find(|w| w.app_id == "waymux.test") {
            assert_eq!(w.title, "Test Window");
            assert!(w.pid > 0);
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "toplevel never surfaced into list_windows; got: {:?}",
                windows
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn resize_reconfigures_mapped_toplevel_and_output() {
    // Fix 1: `Resize` must propagate to inner clients. A real wayland-client
    // creates an xdg_toplevel (configured at the initial 800x600). We then
    // resize the session to 1024x768 and assert the client receives a fresh
    // xdg_toplevel.configure(1024, 768). This proves the control thread drove
    // the compositor thread to re-send the configure via the same wake_fd
    // mechanism inject_* uses.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "rzc".into(),
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

    let wayland_sock = d.state_dir.join("rzc").join("wayland.sock");

    // Long-lived client thread that keeps pumping its event queue so it can
    // observe the post-resize configure. Exposes the last toplevel size it saw.
    let last_size = Arc::new(Mutex::new(None::<(i32, i32)>));
    let last_size_for_client = last_size.clone();
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();
    let client_sock = wayland_sock.clone();
    let client = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let connect_deadline = std::time::Instant::now() + Duration::from_secs(3);
        let stream = loop {
            match std::os::unix::net::UnixStream::connect(&client_sock) {
                Ok(s) => break s,
                Err(_) if std::time::Instant::now() < connect_deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(anyhow::Error::from(e)),
            }
        };
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState {
            last_toplevel_size: last_size_for_client,
            ..Default::default()
        };
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id("waymux.rzc".into());
        surface.commit();
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_for_thread.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    });

    // Wait for the window to appear (proves the toplevel is mapped + registered).
    let _wid = wait_for_next_window(&mut c, "rzc").await;

    // Wait until the client has seen the initial 800x600 configure.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if *last_size.lock().unwrap() == Some((800, 600)) {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "client never received initial 800x600 configure; got {:?}",
                last_size.lock().unwrap()
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Resize the session.
    let resp = c
        .request(RequestMethod::Resize {
            name: "rzc".into(),
            width: 1024,
            height: 768,
        })
        .await;
    assert!(resp.ok, "resize failed: {:?}", resp.error);

    // The client must receive a fresh configure carrying the new size.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if *last_size.lock().unwrap() == Some((1024, 768)) {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "client never received post-resize configure(1024, 768); \
                 last size seen: {:?}",
                last_size.lock().unwrap()
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The daemon-visible session metadata must also reflect the new size.
    let resp = c.request(RequestMethod::ListSessions).await;
    let sessions: Vec<waymux_protocol::SessionInfo> = resp.decode_result().unwrap();
    let s = sessions.iter().find(|s| s.name == "rzc").unwrap();
    assert_eq!(s.width, 1024, "session width not updated after resize");
    assert_eq!(s.height, 768, "session height not updated after resize");

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client.await;
}

#[tokio::test]
async fn window_events_fire_for_create_and_destroy() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "ev".into(),
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

    let wayland_sock = d.state_dir.join("ev").join("wayland.sock");

    // Spawn a wayland-client that creates a toplevel then exits. The
    // exit destroys its resources, which should fire window_destroyed.
    let client_sock = wayland_sock.clone();
    let client_task = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let stream = std::os::unix::net::UnixStream::connect(&client_sock)?;
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();

        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        state.compositor = Some(compositor.clone());
        state.wm_base = Some(wm_base.clone());

        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id("waymux.ev".into());
        toplevel.set_title("Event Test".into());
        surface.commit();

        // Drive the queue briefly to let the server see our requests.
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        while std::time::Instant::now() < deadline {
            let _ = queue.dispatch_pending(&mut state);
            let _ = queue.flush();
            std::thread::sleep(Duration::from_millis(20));
        }
        // Explicit destroy to surface a window_destroyed event.
        toplevel.destroy();
        xdg_surface.destroy();
        surface.destroy();
        let _ = queue.flush();
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    });

    // `window_created` fires when `get_toplevel` is received — at that point
    // the client has not yet called `set_app_id`/`set_title`, so the created
    // event carries empty strings. Those arrive in subsequent `window_changed`
    // events. Aggregate the final state to verify the whole sequence.
    let mut created_pid = None;
    let mut final_app_id = String::new();
    let mut final_title = String::new();
    let mut saw_destroyed = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !saw_destroyed {
        let remaining = deadline - Instant::now();
        match timeout(remaining, async {
            loop {
                if let Incoming::Event(e) = c.read_one().await {
                    return e;
                }
            }
        })
        .await
        {
            Err(_) => break,
            Ok(ev) => match ev.body {
                EventBody::WindowCreated { name, pid, .. } if name == "ev" => {
                    created_pid = Some(pid);
                }
                EventBody::WindowChanged { name, fields, .. } if name == "ev" => {
                    if let Some(a) = fields.app_id {
                        final_app_id = a;
                    }
                    if let Some(t) = fields.title {
                        final_title = t;
                    }
                }
                EventBody::WindowDestroyed { name, .. } if name == "ev" => {
                    saw_destroyed = true;
                }
                _ => {}
            },
        }
    }
    let _ = client_task.await;
    assert!(
        created_pid.is_some() && created_pid.unwrap() > 0,
        "never received window_created with non-zero pid"
    );
    assert_eq!(final_app_id, "waymux.ev");
    assert_eq!(final_title, "Event Test");
    assert!(saw_destroyed, "never received window_destroyed");
}

#[tokio::test]
async fn wait_for_idle_on_truly_idle_session_returns_true() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "idle".into(),
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

    // No commits — wait_for_idle should resolve True on its very first poll
    // because (now - 0) > quiet_ms.
    let started = Instant::now();
    let resp = c
        .request(RequestMethod::WaitForIdle {
            name: "idle".into(),
            quiet_ms: 100,
            timeout_ms: 2_000,
        })
        .await;
    assert!(resp.ok, "wait_for_idle failed: {:?}", resp.error);
    #[derive(serde::Deserialize)]
    struct Idle {
        idle: bool,
    }
    let r: Idle = resp.decode_result().unwrap();
    assert!(r.idle, "expected idle=true on a session with no commits");
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "should return quickly — took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn wait_for_idle_busy_then_idle() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    // Subscribe to windows so we can await the first commit-producing event.
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "busy".into(),
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

    let wayland_sock = d.state_dir.join("busy").join("wayland.sock");

    // Spawn a thread that commits continuously until told to stop.
    let client_sock = wayland_sock.clone();
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();
    let client_task = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let stream = std::os::unix::net::UnixStream::connect(&client_sock)?;
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id("waymux.busy".into());
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        // Commit continuously until told to stop.
        while !stop_for_thread.load(std::sync::atomic::Ordering::SeqCst) {
            surface.commit();
            let _ = queue.flush();
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(20));
        }
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    });

    // Wait until we see a window_created for this session — proves the client
    // has reached get_toplevel, which means commits are being received by the
    // compositor (or about to be).
    let window_deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < window_deadline {
        match timeout(Duration::from_millis(200), async {
            loop {
                if let Incoming::Event(e) = c.read_one().await {
                    return e;
                }
            }
        })
        .await
        {
            Ok(ev) => {
                if matches!(
                    ev.body,
                    EventBody::WindowCreated { ref name, .. } if name == "busy"
                ) {
                    break;
                }
            }
            Err(_) => continue,
        }
    }
    // Allow a moment for the committing loop to actually tick.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // While the client is committing, wait_for_idle(quiet_ms=300) should time
    // out within timeout_ms=300.
    #[derive(serde::Deserialize)]
    struct Idle {
        idle: bool,
    }
    let resp = c
        .request(RequestMethod::WaitForIdle {
            name: "busy".into(),
            quiet_ms: 300,
            timeout_ms: 300,
        })
        .await;
    let r: Idle = resp.decode_result().unwrap();
    assert!(
        !r.idle,
        "expected idle=false while client is still committing"
    );

    // Stop the busy thread; then wait_for_idle should succeed.
    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_task.await;
    let resp = c
        .request(RequestMethod::WaitForIdle {
            name: "busy".into(),
            quiet_ms: 100,
            timeout_ms: 3_000,
        })
        .await;
    let r: Idle = resp.decode_result().unwrap();
    assert!(r.idle, "expected idle=true after client stops committing");
}

#[tokio::test]
async fn screenshot_returns_committed_buffer_pixels() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "shot".into(),
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

    let wayland_sock = d.state_dir.join("shot").join("wayland.sock");

    // Client commits a 32×16 ARGB buffer: left half fully red, right half
    // fully blue. Keeps the connection alive so the buffer stays referenced.
    let client_sock = wayland_sock.clone();
    let client = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let stream = std::os::unix::net::UnixStream::connect(&client_sock)?;
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let shm = globals.bind::<WlShm, _, _>(&qh, 1..=1, ())?;
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id("waymux.shot".into());
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        // Build a memfd-backed shm pool with our pattern.
        let width: i32 = 32;
        let height: i32 = 16;
        let stride = width * 4;
        let size = (stride * height) as usize;
        let mut pixels = vec![0u8; size];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let p = y * stride as usize + x * 4;
                let (b, g, r, a) = if x < (width as usize) / 2 {
                    (0u8, 0, 255, 255) // red
                } else {
                    (255u8, 0, 0, 255) // blue
                };
                pixels[p] = b;
                pixels[p + 1] = g;
                pixels[p + 2] = r;
                pixels[p + 3] = a;
            }
        }
        let tmp = tempfile::tempfile()?;
        {
            let mut f = &tmp;
            f.write_all(&pixels)?;
            f.seek(SeekFrom::Start(0))?;
        }
        use std::os::fd::{AsFd, OwnedFd};
        let fd: OwnedFd = tmp.try_clone()?.into();
        let pool = shm.create_pool(fd.as_fd(), size as i32, &qh, ());
        let buffer =
            pool.create_buffer(0, width, height, stride, wl_shm::Format::Argb8888, &qh, ());
        surface.attach(Some(&buffer), 0, 0);
        surface.damage(0, 0, width, height);
        surface.commit();
        // Let the server process the commit.
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        while std::time::Instant::now() < deadline {
            let _ = queue.flush();
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(20));
        }
        std::mem::forget(buffer);
        std::mem::forget(pool);
        std::mem::forget(shm);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        // Keep the fd alive for the duration of the test.
        std::mem::forget(tmp);
        Ok(())
    });

    // Wait for window_created to get the assigned id.
    let window_id = loop {
        let ev = c.next_event(Duration::from_secs(3)).await;
        if let EventBody::WindowCreated {
            name, window_id, ..
        } = &ev.body
        {
            if name == "shot" {
                break *window_id;
            }
        }
    };
    // Let the first commit settle.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let resp = c
        .request(RequestMethod::Screenshot {
            name: "shot".into(),
            window_id: Some(window_id),
            format: waymux_protocol::ScreenshotFormat::Png,
        })
        .await;
    assert!(resp.ok, "screenshot failed: {:?}", resp.error);
    let shot: waymux_protocol::SessionCtlScreenshot = resp.decode_result().unwrap();
    assert_eq!(shot.width, 32);
    assert_eq!(shot.height, 16);
    assert!(!shot.png.is_empty());

    // Decode PNG and verify the red/blue split.
    let decoder = png::Decoder::new(shot.png.as_slice());
    let mut reader = decoder.read_info().expect("read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("read PNG frame");
    assert_eq!(info.width, 32);
    assert_eq!(info.height, 16);
    // Output is RGBA, 4 bytes per pixel.
    let row_bytes = 32 * 4;
    let left = &buf[..4]; // first pixel of row 0
    let right = &buf[row_bytes - 4..row_bytes]; // last pixel of row 0
    assert_eq!(left, &[255, 0, 0, 255], "left half should be red");
    assert_eq!(right, &[0, 0, 255, 255], "right half should be blue");

    drop(client); // let it be cancelled; the test has what it needs
}

#[tokio::test]
async fn inject_key_delivers_to_focused_client() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "kbd".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("kbd").join("wayland.sock");

    // Client binds wl_seat + wl_keyboard, creates a toplevel (to grab focus),
    // and records every key event it receives. Runs until `stop_flag`.
    let key_log = KeyLog::default();
    let log_for_client = key_log.clone();
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();
    let client_sock = wayland_sock.clone();
    let client = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let stream = std::os::unix::net::UnixStream::connect(&client_sock)?;
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ())?;
        let keyboard = seat.get_keyboard(&qh, log_for_client);
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id("waymux.kbd".into());
        surface.commit();
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_for_thread.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            // Nonblocking read: actually pull bytes off the socket so
            // dispatch_pending has something to process.
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(keyboard);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    });

    // Wait for the window to appear (→ focus assigned) so inject_key will route.
    let _window_id = loop {
        let ev = c.next_event(Duration::from_secs(3)).await;
        if let EventBody::WindowCreated {
            name, window_id, ..
        } = &ev.body
        {
            if name == "kbd" {
                break *window_id;
            }
        }
    };

    // Inject: press 'A' (evdev 30), then release.
    let resp = c
        .request(RequestMethod::InjectKey {
            name: "kbd".into(),
            keycode: 30,
            state: waymux_protocol::KeyState::Pressed,
            modifiers: 0,
        })
        .await;
    assert!(resp.ok, "inject_key (press) failed: {:?}", resp.error);

    let resp = c
        .request(RequestMethod::InjectKey {
            name: "kbd".into(),
            keycode: 30,
            state: waymux_protocol::KeyState::Released,
            modifiers: 0,
        })
        .await;
    assert!(resp.ok, "inject_key (release) failed: {:?}", resp.error);

    // Give the client thread up to 1s to observe both key events.
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let recorded = key_log.snapshot();
        let keys: Vec<&KbdEvent> = recorded
            .iter()
            .filter(|e| matches!(e, KbdEvent::Key { .. }))
            .collect();
        if keys.len() >= 2 {
            assert_eq!(
                keys[0],
                &KbdEvent::Key {
                    keycode: 30,
                    pressed: true
                }
            );
            assert_eq!(
                keys[1],
                &KbdEvent::Key {
                    keycode: 30,
                    pressed: false
                }
            );
            // The protocol spec: keymap must arrive before any key event; enter
            // must precede the first key. Verify ordering.
            let keymap_idx = recorded.iter().position(|e| e == &KbdEvent::Keymap);
            let enter_idx = recorded.iter().position(|e| e == &KbdEvent::Enter);
            let first_key_idx = recorded
                .iter()
                .position(|e| matches!(e, KbdEvent::Key { .. }));
            assert!(keymap_idx.is_some(), "no keymap event received");
            assert!(enter_idx.is_some(), "no enter event received");
            assert!(
                keymap_idx < first_key_idx,
                "keymap must precede first key; got {:?}",
                recorded
            );
            assert!(
                enter_idx < first_key_idx,
                "enter must precede first key; got {:?}",
                recorded
            );
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "client never received both key events; got {:?}",
                key_log.snapshot()
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client.await;
}

#[tokio::test]
async fn inject_key_latency_is_sub_10ms() {
    // Regression guard for the eventfd wakeup. Before that fix, inject_key
    // events sat in the wayland-server buffer until the compositor thread's
    // next 100ms poll cycle. With the wakeup path, the client should
    // observe the key event within the Linux scheduling floor (usually
    // well under 1ms, but we allow 10ms to be kind to CI).
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "lat".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("lat").join("wayland.sock");

    let key_log = KeyLog::default();
    let log_for_client = key_log.clone();
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();
    let client_sock = wayland_sock.clone();
    let client = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let stream = std::os::unix::net::UnixStream::connect(&client_sock)?;
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ())?;
        let keyboard = seat.get_keyboard(&qh, log_for_client);
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id("waymux.lat".into());
        surface.commit();
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_for_thread.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(1));
        }
        std::mem::forget(keyboard);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    });

    // Wait for focus.
    let _ = loop {
        let ev = c.next_event(Duration::from_secs(3)).await;
        if let EventBody::WindowCreated {
            name, window_id, ..
        } = &ev.body
        {
            if name == "lat" {
                break *window_id;
            }
        }
    };

    // Give the client one more pump cycle to process the enter/modifiers
    // events so we're only measuring the inject_key round-trip.
    tokio::time::sleep(Duration::from_millis(30)).await;
    key_log.0.lock().unwrap().clear();

    // Measure one press.
    let sent_at = Instant::now();
    let resp = c
        .request(RequestMethod::InjectKey {
            name: "lat".into(),
            keycode: 30,
            state: waymux_protocol::KeyState::Pressed,
            modifiers: 0,
        })
        .await;
    assert!(resp.ok);

    // Wait until the client records the key; measure wall-clock.
    let deadline = Instant::now() + Duration::from_millis(100);
    let observed_at = loop {
        if key_log.snapshot().iter().any(|e| {
            matches!(
                e,
                KbdEvent::Key {
                    keycode: 30,
                    pressed: true
                }
            )
        }) {
            break Instant::now();
        }
        if Instant::now() >= deadline {
            panic!(
                "never received key within 100ms; log={:?}",
                key_log.snapshot()
            );
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    };

    let latency = observed_at - sent_at;
    assert!(
        latency < Duration::from_millis(10),
        "inject_key round-trip latency {:?} > 10ms (eventfd wakeup regressed?)",
        latency,
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client.await;
}

#[tokio::test]
async fn inject_pointer_delivers_motion_and_button() {
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;

    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;

    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "ptr".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("ptr").join("wayland.sock");

    let pointer_log = PointerLog::default();
    let log_for_client = pointer_log.clone();
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();
    let client_sock = wayland_sock.clone();
    let client = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let stream = std::os::unix::net::UnixStream::connect(&client_sock)?;
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ())?;
        let pointer = seat.get_pointer(&qh, log_for_client);
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id("waymux.ptr".into());
        surface.commit();
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_for_thread.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(pointer);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    });

    let _window_id = loop {
        let ev = c.next_event(Duration::from_secs(3)).await;
        if let EventBody::WindowCreated {
            name, window_id, ..
        } = &ev.body
        {
            if name == "ptr" {
                break *window_id;
            }
        }
    };

    // Move + left-click-down, then left-click-up at the same position.
    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "ptr".into(),
            x: 123.5,
            y: 45.75,
            button: 0x110, // BTN_LEFT
            state: waymux_protocol::KeyState::Pressed,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: None,
            content: false,
        })
        .await;
    assert!(resp.ok, "inject_pointer (press) failed: {:?}", resp.error);

    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "ptr".into(),
            x: 123.5,
            y: 45.75,
            button: 0x110,
            state: waymux_protocol::KeyState::Released,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: None,
            content: false,
        })
        .await;
    assert!(resp.ok, "inject_pointer (release) failed: {:?}", resp.error);

    // Client should see 2× Motion + 2× Button (one per RPC).
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut motion_count;
    let mut saw_press = false;
    let mut saw_release = false;
    loop {
        {
            let events = pointer_log.0.lock().unwrap().clone();
            motion_count = events
                .iter()
                .filter(|e| matches!(e, PointerEvent::Motion { .. }))
                .count();
            for e in &events {
                if let PointerEvent::Button { button, pressed } = e {
                    if *button == 0x110 && *pressed {
                        saw_press = true;
                    }
                    if *button == 0x110 && !*pressed {
                        saw_release = true;
                    }
                }
            }
        }
        if motion_count >= 2 && saw_press && saw_release {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "pointer events missing: motion={}, press={}, release={}; log={:?}",
                motion_count,
                saw_press,
                saw_release,
                pointer_log.0.lock().unwrap()
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Verify one motion carries the coordinates we sent.
    let events = pointer_log.0.lock().unwrap().clone();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, PointerEvent::Motion { x, y } if (*x - 123.5).abs() < 0.01 && (*y - 45.75).abs() < 0.01)),
        "no motion event with expected coords; got {:?}",
        events
    );

    // Every inject_pointer must be preceded by an `enter` event so
    // X-on-Wayland clients (Xwayland in rootful mode) and KWin treat the
    // motion+button as "pointer is on this surface". Without `enter`, motion
    // + button events arrive on a wl_pointer that has no surface focus, and
    // the inner X11 server (and many native clients) silently drop them.
    let enter_idx = events
        .iter()
        .position(|e| matches!(e, PointerEvent::Enter { .. }));
    assert!(
        enter_idx.is_some(),
        "no Enter event delivered before motion; got {:?}",
        events
    );
    let first_motion_idx = events
        .iter()
        .position(|e| matches!(e, PointerEvent::Motion { .. }))
        .expect("Motion should be present");
    assert!(
        enter_idx.unwrap() < first_motion_idx,
        "Enter must precede first Motion; got {:?}",
        events
    );

    // Motion-then-button must be split into two frame groups.
    // X-on-Wayland clients (Xwayland) process button events at the X11 pointer
    // position established BEFORE the same-frame motion, so SDK
    // `inject_pointer(x, y, BTN_LEFT, pressed=true)` must emit
    // `[motion, frame, button, frame]` instead of `[motion, button, frame]`.
    // Verify there is a `Frame` between the first Motion and the first Button.
    let first_btn_idx = events
        .iter()
        .position(|e| matches!(e, PointerEvent::Button { .. }))
        .expect("Button should be present");
    let frame_between = events[first_motion_idx..first_btn_idx]
        .iter()
        .any(|e| matches!(e, PointerEvent::Frame));
    assert!(
        frame_between,
        "expected a Frame between first Motion and first Button (motion + button \
         must be in separate frame groups so Xwayland updates the X11 pointer \
         position before the button event lands); got {:?}",
        events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client.await;
}

// ─── window_id routing + content offset ─────────────────────────────────────
//
// Helpers + tests for the new `window_id: Option<u32>` and `content: bool`
// arguments on `State::inject_pointer`. Each test spawns *two distinct
// Wayland clients* against the same session so routing actually
// distinguishes them — `inject_pointer` dispatches by ClientId, so a
// single-client/two-surface test would not exercise the routing branch.

/// Spawn a Wayland client thread that connects to the given inner socket,
/// creates one xdg_toplevel (named via `app_id`), and logs all pointer
/// events into `pointer_log` until `stop_flag` is set.
///
/// Connects with a 3-second retry loop because the session's inner socket
/// is created asynchronously by the spawned waymux-session process — the
/// daemon's CreateSession response may return before the inner socket
/// exists on disk. A single bare connect() races and flakes intermittently.
fn spawn_routing_client(
    wayland_sock: PathBuf,
    app_id: &'static str,
    pointer_log: PointerLog,
    stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let connect_deadline = std::time::Instant::now() + Duration::from_secs(3);
        let stream = loop {
            match std::os::unix::net::UnixStream::connect(&wayland_sock) {
                Ok(s) => break s,
                Err(_) if std::time::Instant::now() < connect_deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(anyhow::Error::from(e)),
            }
        };
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ())?;
        let pointer = seat.get_pointer(&qh, pointer_log);
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id(app_id.into());
        surface.commit();
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(pointer);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    })
}

/// Variant of `spawn_routing_client` that additionally emits
/// `xdg_surface.set_window_geometry(x, y, w, h)` after creating the
/// toplevel. Used to exercise the CSD-inset path on
/// `inject_pointer(content=true)`.
///
/// Returns a JoinHandle plus a "geometry-applied" flag that flips to true
/// once the client has flushed the set_window_geometry request to the
/// server. Callers poll this flag with a deadline before calling
/// `inject_pointer` so the test isn't racing against the wire.
fn spawn_routing_client_with_geometry(
    wayland_sock: PathBuf,
    app_id: &'static str,
    pointer_log: PointerLog,
    stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    geometry: (i32, i32, i32, i32),
    geometry_applied: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let connect_deadline = std::time::Instant::now() + Duration::from_secs(3);
        let stream = loop {
            match std::os::unix::net::UnixStream::connect(&wayland_sock) {
                Ok(s) => break s,
                Err(_) if std::time::Instant::now() < connect_deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(anyhow::Error::from(e)),
            }
        };
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ())?;
        let pointer = seat.get_pointer(&qh, pointer_log);
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id(app_id.into());
        let (gx, gy, gw, gh) = geometry;
        xdg_surface.set_window_geometry(gx, gy, gw, gh);
        surface.commit();
        let _ = queue.flush();
        geometry_applied.store(true, std::sync::atomic::Ordering::SeqCst);
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(pointer);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    })
}

/// Wait for the next `WindowCreated` event matching `session_name` and
/// return its window_id. Filters by session name only — `WindowCreated`
/// is emitted by `State::add_window` with `app_id=""` because the client's
/// `set_app_id` arrives in a later request (which fires `WindowChanged`).
/// Tests therefore can't gate on app_id at creation time; instead they
/// register clients in a known order and pair them with WindowCreated
/// events 1-for-1.
async fn wait_for_next_window(c: &mut DaemonClient, session_name: &str) -> u32 {
    loop {
        let ev = c.next_event(Duration::from_secs(5)).await;
        if let EventBody::WindowCreated {
            name, window_id, ..
        } = &ev.body
        {
            if name == session_name {
                return *window_id;
            }
        }
    }
}

#[tokio::test]
async fn inject_pointer_with_window_id_overrides_focus() {
    // Two clients A and B connect; B registers second so the session's
    // "focus follows most-recent" rule makes B focused. Routing to A via
    // window_id=Some(A) must reach pointer A *despite* B being focused.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "route".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("route").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let log_b = PointerLog::default();
    // Spawn A, wait for its WindowCreated (so registration order is
    // deterministic), then spawn B.
    let client_a = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.A",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "route").await;
    let client_b = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.B",
        log_b.clone(),
        stop_flag.clone(),
    );
    let wid_b = wait_for_next_window(&mut c, "route").await;
    assert_ne!(wid_a, wid_b);

    // Target A explicitly; B is focused (registered second).
    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "route".into(),
            x: 50.0,
            y: 60.0,
            button: 0x110,
            state: waymux_protocol::KeyState::Pressed,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: false,
        })
        .await;
    assert!(
        resp.ok,
        "inject_pointer routed to A failed: {:?}",
        resp.error
    );

    // A must receive Motion + Button; B must receive *nothing*.
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let saw_motion = a_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Motion { .. }));
        let saw_button = a_events.iter().any(|e| {
            matches!(
                e,
                PointerEvent::Button {
                    button: 0x110,
                    pressed: true
                }
            )
        });
        if saw_motion && saw_button {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "client A did not receive routed pointer events; A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let b_events = log_b.0.lock().unwrap().clone();
    assert!(
        !b_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Motion { .. })),
        "client B (focused) must NOT receive motion targeted at A; B log: {:?}",
        b_events
    );
    assert!(
        !b_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Button { .. })),
        "client B (focused) must NOT receive button targeted at A; B log: {:?}",
        b_events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
    let _ = client_b.await;
}

#[tokio::test]
async fn inject_pointer_with_no_window_id_uses_focused() {
    // Same setup; window_id=None → focused (B, registered second) gets the
    // events.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "focus".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("focus").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let log_b = PointerLog::default();
    let client_a = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.fA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let _wid_a = wait_for_next_window(&mut c, "focus").await;
    let client_b = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.fB",
        log_b.clone(),
        stop_flag.clone(),
    );
    let _wid_b = wait_for_next_window(&mut c, "focus").await;

    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "focus".into(),
            x: 77.0,
            y: 88.0,
            button: 0x110,
            state: waymux_protocol::KeyState::Pressed,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: None,
            content: false,
        })
        .await;
    assert!(resp.ok, "inject_pointer (None) failed: {:?}", resp.error);

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let b_events = log_b.0.lock().unwrap().clone();
        let saw_motion = b_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Motion { .. }));
        let saw_button = b_events.iter().any(|e| {
            matches!(
                e,
                PointerEvent::Button {
                    button: 0x110,
                    pressed: true
                }
            )
        });
        if saw_motion && saw_button {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "focused client B did not receive pointer events; B log: {:?}",
                b_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let a_events = log_a.0.lock().unwrap().clone();
    assert!(
        !a_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Motion { .. })),
        "non-focused client A must NOT receive motion; A log: {:?}",
        a_events
    );
    assert!(
        !a_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Button { .. })),
        "non-focused client A must NOT receive button; A log: {:?}",
        a_events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
    let _ = client_b.await;
}

#[tokio::test]
async fn inject_pointer_with_unknown_window_id_returns_false() {
    // Unknown id must drop (no fallback to focused). The wire-level
    // session-ctl response surfaces this as `resp.ok == false`.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "unk".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("unk").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let client_a = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.uA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let _wid_a = wait_for_next_window(&mut c, "unk").await;

    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "unk".into(),
            x: 10.0,
            y: 20.0,
            button: 0x110,
            state: waymux_protocol::KeyState::Pressed,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(99999),
            content: false,
        })
        .await;
    // inject_pointer returns false for unknown window_id; control.rs maps
    // that to a `failure` response. resp.ok must be false.
    assert!(
        !resp.ok,
        "inject_pointer with unknown window_id must NOT report ok"
    );

    // Wait a beat, then confirm the only registered client received nothing
    // (verifying "no fallback to focused").
    tokio::time::sleep(Duration::from_millis(150)).await;
    let a_events = log_a.0.lock().unwrap().clone();
    assert!(
        !a_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Motion { .. })),
        "client A must NOT receive events from an unknown-window_id call \
         (no silent fallback to focused); A log: {:?}",
        a_events
    );
    assert!(
        !a_events
            .iter()
            .any(|e| matches!(e, PointerEvent::Button { .. })),
        "client A must NOT receive button events from unknown-window_id; A log: {:?}",
        a_events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn inject_pointer_content_true_with_no_geometry_falls_back() {
    // When content=true but the client never emitted
    // xdg_surface.set_window_geometry, window_content_inset falls back to
    // (0, 0) and coords pass through unchanged. A warn-once log is emitted on
    // the session process side; this test only verifies the routing behavior.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "cont".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("cont").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let client_a = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.cA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "cont").await;

    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "cont".into(),
            x: 5.0,
            y: 7.0,
            button: 0,
            state: waymux_protocol::KeyState::Released,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: true,
        })
        .await;
    assert!(
        resp.ok,
        "inject_pointer content=true failed: {:?}",
        resp.error
    );

    // With the no-geometry fallback applying a (0, 0) inset, motion arrives
    // unchanged at (5, 7).
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let hit = a_events.iter().any(|e| {
            matches!(e, PointerEvent::Motion { x, y }
                if (*x - 5.0).abs() < 0.01 && (*y - 7.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "content=true motion at (5,7) not delivered (no-geometry \
                 fallback should pass through unchanged); A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn inject_pointer_content_true_subtracts_inset() {
    // When the client has called xdg_surface.set_window_geometry
    // (x=16, y=21, w, h), modelling Chromium's CSD shadow margin, the
    // engine treats inject_pointer(content=true, x=5, y=10) as content-space
    // coords and adds the (16, 21) inset to produce buffer-space (21, 31)
    // before delivery. The (16, 21) inset matches Chromium's CSD shadow
    // margin.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "geom".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("geom").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let geometry_applied = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Client emits xdg_surface.set_window_geometry(16, 21, 288, 198) —
    // 288 = 320 - 32 (16 px shadow each side), 198 = 240 - 42 (21 px each).
    let client_a = spawn_routing_client_with_geometry(
        wayland_sock.clone(),
        "waymux.cA",
        log_a.clone(),
        stop_flag.clone(),
        (16, 21, 288, 198),
        geometry_applied.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "geom").await;

    // Wait for the client to have flushed set_window_geometry through to the
    // session compositor. The flag flips after the client's queue.flush();
    // give the session a slice to dispatch the request before injecting.
    let deadline = Instant::now() + Duration::from_secs(2);
    while !geometry_applied.load(std::sync::atomic::Ordering::SeqCst) {
        if Instant::now() >= deadline {
            panic!("client never reported set_window_geometry applied");
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Extra slice for the session to actually receive + apply the request.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "geom".into(),
            x: 5.0,
            y: 10.0,
            button: 0,
            state: waymux_protocol::KeyState::Released,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: true,
        })
        .await;
    assert!(
        resp.ok,
        "inject_pointer content=true (with geometry) failed: {:?}",
        resp.error
    );

    // Motion should arrive at (5 + 16, 10 + 21) = (21, 31) in buffer space.
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let hit = a_events.iter().any(|e| {
            matches!(e, PointerEvent::Motion { x, y }
                if (*x - 21.0).abs() < 0.01 && (*y - 31.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "content=true motion at (21, 31) (inset (16, 21) + (5, 10)) \
                 not delivered; A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Sanity: content=false at the same SDK coords must NOT receive the
    // inset — motion should arrive at (5, 10) untouched.
    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "geom".into(),
            x: 5.0,
            y: 10.0,
            button: 0,
            state: waymux_protocol::KeyState::Released,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: false,
        })
        .await;
    assert!(
        resp.ok,
        "content=false sanity inject failed: {:?}",
        resp.error
    );

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let hit = a_events.iter().any(|e| {
            matches!(e, PointerEvent::Motion { x, y }
                if (*x - 5.0).abs() < 0.01 && (*y - 10.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "content=false motion at (5, 10) (no inset) not delivered; \
                 A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn inject_pointer_passes_logical_coords_through_at_session_scale() {
    // Scale > 1 coordinate handling. The session is constructed with scale=2;
    // the SDK passes coordinates in the session's `width x height` space, which
    // IS the compositor's logical space (advertised via
    // zxdg_output_v1.logical_size and every xdg_toplevel.configure). The
    // session scale governs only how a client renders its buffer
    // (wl_surface.set_buffer_scale); per the Wayland protocol it does NOT
    // change the coordinate space of wl_pointer.motion, which is always
    // surface-local logical pixels. So the SDK's logical (10, 20) must arrive
    // unchanged as motion (10.0, 20.0), NOT (20.0, 40.0). content=false so the
    // content_rect inset is not exercised; this isolates the scale path.
    //
    // History: an earlier attempt multiplied by `scale` here on the false
    // premise that wl_pointer.motion expects buffer pixels. That pushed every
    // click `scale`× off-target. This test pins the corrected pass-through.
    //
    // Frame-split semantics (Motion → Frame → Button → Frame, see
    // compositor.rs:1467-1486) are also asserted to confirm the scale path
    // didn't disturb event ordering.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "scale2".into(),
        width: 320,
        height: 240,
        scale: 2,
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

    let wayland_sock = d.state_dir.join("scale2").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let client_a = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.cA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "scale2").await;

    // SDK sends logical (10, 20) with BTN_LEFT press. The engine must deliver
    // motion at exactly (10, 20), unchanged, then emit motion + frame +
    // button + frame per the frame-split contract.
    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "scale2".into(),
            x: 10.0,
            y: 20.0,
            button: 0x110, // BTN_LEFT
            state: waymux_protocol::KeyState::Pressed,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: false,
        })
        .await;
    assert!(resp.ok, "inject_pointer (scale=2) failed: {:?}", resp.error);

    // Motion should arrive at logical (10, 20), unchanged by scale.
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let hit = a_events.iter().any(|e| {
            matches!(e, PointerEvent::Motion { x, y }
                if (*x - 10.0).abs() < 0.01 && (*y - 20.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "scale=2 motion at logical (10, 20) not delivered unchanged; \
                 A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Negative-control: confirm the old buggy (20, 40) = (10,20)×2 motion was
    // NOT delivered (would indicate the scale multiply regressed back in).
    let a_events = log_a.0.lock().unwrap().clone();
    let scaled = a_events.iter().any(|e| {
        matches!(e, PointerEvent::Motion { x, y }
            if (*x - 20.0).abs() < 0.01 && (*y - 40.0).abs() < 0.01)
    });
    assert!(
        !scaled,
        "scale=2 session delivered motion at (20, 40) = logical (10,20) × 2; \
         the buggy scale multiply is back. A log: {:?}",
        a_events
    );

    // Frame-split contract still holds: between the first Motion and the
    // first Button there must be a Frame event.
    let first_motion_idx = a_events
        .iter()
        .position(|e| matches!(e, PointerEvent::Motion { .. }))
        .expect("Motion should be present");
    let first_btn_idx = a_events
        .iter()
        .position(|e| matches!(e, PointerEvent::Button { .. }))
        .expect("Button should be present");
    let frame_between = a_events[first_motion_idx..first_btn_idx]
        .iter()
        .any(|e| matches!(e, PointerEvent::Frame));
    assert!(
        frame_between,
        "scale=2 path must preserve frame-split (Motion, Frame, Button) per \
         compositor.rs:1467-1486; got {:?}",
        a_events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn inject_pointer_content_true_with_no_geometry_warns_once() {
    // Behavioral verification of the warn-once fallback: when
    // content=true is called twice for a window that never emitted
    // set_window_geometry, BOTH calls must deliver motion at the
    // SDK-supplied coords (zero-inset fallback) — i.e. the second call
    // is not corrupted by the first call having populated some cache.
    //
    // The warn-once dedup logic (HashSet::insert returns true exactly once
    // per window_id) is unit-tested in waymux-session::state::tests.
    // Counting tracing::warn! events emitted inside the spawned session
    // subprocess would require a stderr-capture harness; the unit test
    // covers that semantic directly.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "warn".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("warn").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let client_a = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.cA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "warn").await;

    // First call.
    let resp1 = c
        .request(RequestMethod::InjectPointer {
            name: "warn".into(),
            x: 11.0,
            y: 22.0,
            button: 0,
            state: waymux_protocol::KeyState::Released,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: true,
        })
        .await;
    assert!(resp1.ok, "first inject failed: {:?}", resp1.error);

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let hit = a_events.iter().any(|e| {
            matches!(e, PointerEvent::Motion { x, y }
                if (*x - 11.0).abs() < 0.01 && (*y - 22.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "first call motion at (11, 22) missing; A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Second call to the same window — also no geometry, also passes through.
    let resp2 = c
        .request(RequestMethod::InjectPointer {
            name: "warn".into(),
            x: 33.0,
            y: 44.0,
            button: 0,
            state: waymux_protocol::KeyState::Released,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: true,
        })
        .await;
    assert!(resp2.ok, "second inject failed: {:?}", resp2.error);

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let hit = a_events.iter().any(|e| {
            matches!(e, PointerEvent::Motion { x, y }
                if (*x - 33.0).abs() < 0.01 && (*y - 44.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "second call motion at (33, 44) missing; A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn list_windows_returns_content_rect_when_geometry_set() {
    // When a client emits xdg_surface.set_window_geometry,
    // ListWindows must surface that rect in WindowInfo.content_rect.
    // Uses the same spawn_routing_client_with_geometry helper as the
    // inject_pointer_content_true_subtracts_inset test so we exercise the
    // real wire path (compositor.rs SetWindowGeometry → SurfaceData →
    // state.windows() overlay → control.rs ListWindows handler).
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "lwgeom".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("lwgeom").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let geometry_applied = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let client_a = spawn_routing_client_with_geometry(
        wayland_sock.clone(),
        "waymux.cA",
        log_a.clone(),
        stop_flag.clone(),
        (16, 21, 800, 600),
        geometry_applied.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "lwgeom").await;

    // Wait for the client to flush set_window_geometry through to the
    // session compositor before polling ListWindows.
    let deadline = Instant::now() + Duration::from_secs(2);
    while !geometry_applied.load(std::sync::atomic::Ordering::SeqCst) {
        if Instant::now() >= deadline {
            panic!("client never reported set_window_geometry applied");
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Extra slice for the session to dispatch the request.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Poll ListWindows until our window's content_rect arrives populated.
    // We poll because set_window_geometry crosses the wire asynchronously
    // and the session may not have applied it the first time we ask.
    let expected = Rect {
        x: 16,
        y: 21,
        width: 800,
        height: 600,
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut last_seen: Option<Option<Rect>> = None;
    loop {
        let resp = c
            .request(RequestMethod::ListWindows {
                name: "lwgeom".into(),
            })
            .await;
        let windows: Vec<WindowInfo> = resp.decode_result().unwrap();
        if let Some(w) = windows.iter().find(|w| w.id == wid_a) {
            last_seen = Some(w.content_rect);
            if w.content_rect == Some(expected) {
                break;
            }
        }
        if Instant::now() >= deadline {
            panic!(
                "ListWindows never surfaced content_rect={:?} for window {}; \
                 last seen content_rect={:?}",
                expected, wid_a, last_seen
            );
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn list_windows_returns_none_content_rect_when_geometry_unset() {
    // Negative case: when a client never emits set_window_geometry,
    // ListWindows must report content_rect=None. Uses spawn_routing_client
    // (no geometry variant) — the toplevel is created and committed but
    // xdg_surface.set_window_geometry is never sent.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "lwnogeom".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("lwnogeom").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = PointerLog::default();
    let client_a = spawn_routing_client(
        wayland_sock.clone(),
        "waymux.cA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "lwnogeom").await;

    // The window exists; verify ListWindows reports content_rect=None.
    // We poll briefly: the window may show up in ListWindows slightly
    // before/after we get the WindowCreated event, but content_rect must
    // never become Some.
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut found = false;
    loop {
        let resp = c
            .request(RequestMethod::ListWindows {
                name: "lwnogeom".into(),
            })
            .await;
        let windows: Vec<WindowInfo> = resp.decode_result().unwrap();
        if let Some(w) = windows.iter().find(|w| w.id == wid_a) {
            assert_eq!(
                w.content_rect, None,
                "window with no set_window_geometry must report content_rect=None"
            );
            found = true;
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(
        found,
        "window {} never appeared in ListWindows for negative test",
        wid_a
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

// ─── wl_touch capability + State::inject_touch routing ─────────────────────
//
// Touch tests mirror the inject_pointer tests in structure but bind
// a wl_touch resource via `wl_seat.get_touch` instead of `get_pointer`. The
// `spawn_touch_client*` helpers record `wl_touch.{down|motion|up|frame}`
// into a `TouchLog`; tests assert event order and coordinate transforms.

#[derive(Debug, Clone, PartialEq)]
enum TouchEvent {
    Down { id: i32, x: f64, y: f64 },
    Motion { id: i32, x: f64, y: f64 },
    Up { id: i32 },
    Frame,
}

/// Records touch events for assertion.
#[derive(Default, Clone)]
struct TouchLog(Arc<Mutex<Vec<TouchEvent>>>);

impl Dispatch<WlTouch, TouchLog> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlTouch,
        event: wl_touch::Event,
        log: &TouchLog,
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
        match event {
            wl_touch::Event::Down { id, x, y, .. } => {
                log.0.lock().unwrap().push(TouchEvent::Down { id, x, y });
            }
            wl_touch::Event::Motion { id, x, y, .. } => {
                log.0.lock().unwrap().push(TouchEvent::Motion { id, x, y });
            }
            wl_touch::Event::Up { id, .. } => {
                log.0.lock().unwrap().push(TouchEvent::Up { id });
            }
            wl_touch::Event::Frame => {
                log.0.lock().unwrap().push(TouchEvent::Frame);
            }
            _ => {}
        }
    }
}

/// Records the most recent `wl_seat.capabilities` event so the
/// `session_advertises_wl_touch_capability` test can verify the bitfield.
/// We need a `Dispatch<WlSeat, SeatCapLog>` so the test client can observe
/// capabilities events on its bound seat — the default `Dispatch<WlSeat, ()>`
/// in this file discards them.
#[derive(Default, Clone)]
struct SeatCapLog(Arc<Mutex<Option<u32>>>);

impl Dispatch<WlSeat, SeatCapLog> for ClientState {
    fn event(
        _s: &mut Self,
        _p: &WlSeat,
        event: wl_seat::Event,
        log: &SeatCapLog,
        _c: &Connection,
        _q: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities { capabilities } = event {
            // `capabilities` is a `WEnum<Capability>` — its inner value is the
            // bitfield as u32. The Capability enum has variants Pointer = 1,
            // Keyboard = 2, Touch = 4 (per the wl_seat XML).
            if let wayland_client::WEnum::Value(caps) = capabilities {
                *log.0.lock().unwrap() = Some(caps.into());
            } else if let wayland_client::WEnum::Unknown(raw) = capabilities {
                *log.0.lock().unwrap() = Some(raw);
            }
        }
    }
}

/// Spawn a Wayland client that binds wl_seat + wl_touch, records touch
/// events into `touch_log`, and registers an xdg_toplevel so the session
/// has a window to route to. Sibling of `spawn_routing_client`. The wl_seat
/// is bound with the default `()` data so the test doesn't have to
/// duplicate the SeatCapLog impl plumbing — `session_advertises_wl_touch_capability`
/// uses a separate inlined helper for the capabilities-observation case.
fn spawn_touch_client(
    wayland_sock: PathBuf,
    app_id: &'static str,
    touch_log: TouchLog,
    stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let connect_deadline = std::time::Instant::now() + Duration::from_secs(3);
        let stream = loop {
            match std::os::unix::net::UnixStream::connect(&wayland_sock) {
                Ok(s) => break s,
                Err(_) if std::time::Instant::now() < connect_deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(anyhow::Error::from(e)),
            }
        };
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ())?;
        let touch = seat.get_touch(&qh, touch_log);
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id(app_id.into());
        surface.commit();
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(touch);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    })
}

/// Variant of `spawn_touch_client` that ALSO records a pointer log via a
/// shared `WlSeat` bind. Used by the shared content_fallback_warned test
/// which calls `inject_pointer` and `inject_touch` against the same window
/// and asserts the warn-once HashSet survives across both.
fn spawn_touch_and_pointer_client(
    wayland_sock: PathBuf,
    app_id: &'static str,
    touch_log: TouchLog,
    pointer_log: PointerLog,
    stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let connect_deadline = std::time::Instant::now() + Duration::from_secs(3);
        let stream = loop {
            match std::os::unix::net::UnixStream::connect(&wayland_sock) {
                Ok(s) => break s,
                Err(_) if std::time::Instant::now() < connect_deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(anyhow::Error::from(e)),
            }
        };
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        let compositor = globals.bind::<WlCompositor, _, _>(&qh, 1..=6, ())?;
        let wm_base = globals.bind::<XdgWmBase, _, _>(&qh, 1..=5, ())?;
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ())?;
        let pointer = seat.get_pointer(&qh, pointer_log);
        let touch = seat.get_touch(&qh, touch_log);
        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_app_id(app_id.into());
        surface.commit();
        state.compositor = Some(compositor);
        state.wm_base = Some(wm_base);

        while !stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(pointer);
        std::mem::forget(touch);
        std::mem::forget(toplevel);
        std::mem::forget(xdg_surface);
        std::mem::forget(surface);
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    })
}

#[tokio::test]
async fn inject_touch_with_window_id_overrides_focus() {
    // Two touch clients A and B; B registers second so it's focused. Routing
    // to A via window_id=Some(A) must reach A's wl_touch despite B's focus.
    // Mirrors `inject_pointer_with_window_id_overrides_focus`.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "trte".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("trte").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = TouchLog::default();
    let log_b = TouchLog::default();
    let client_a = spawn_touch_client(
        wayland_sock.clone(),
        "waymux.tA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "trte").await;
    let client_b = spawn_touch_client(
        wayland_sock.clone(),
        "waymux.tB",
        log_b.clone(),
        stop_flag.clone(),
    );
    let wid_b = wait_for_next_window(&mut c, "trte").await;
    assert_ne!(wid_a, wid_b);

    // Target A explicitly; B is focused (registered second).
    let resp = c
        .request(RequestMethod::InjectTouch {
            name: "trte".into(),
            id: 0,
            x: 50.0,
            y: 100.0,
            phase: waymux_protocol::TouchPhase::Down,
            window_id: Some(wid_a),
            content: false,
        })
        .await;
    assert!(resp.ok, "inject_touch routed to A failed: {:?}", resp.error);

    // A must receive Down + Frame; B must receive nothing.
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let saw_down = a_events
            .iter()
            .any(|e| matches!(e, TouchEvent::Down { id: 0, .. }));
        let saw_frame = a_events.iter().any(|e| matches!(e, TouchEvent::Frame));
        if saw_down && saw_frame {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "client A did not receive routed touch events; A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let b_events = log_b.0.lock().unwrap().clone();
    assert!(
        !b_events
            .iter()
            .any(|e| matches!(e, TouchEvent::Down { .. })),
        "client B (focused) must NOT receive touch events targeted at A; B log: {:?}",
        b_events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
    let _ = client_b.await;
}

#[tokio::test]
async fn inject_touch_emits_frame_after_each_event() {
    // Down → Motion → Up across three RPCs. Each event must be followed by
    // a Frame in arrival order so the resulting log is
    // [..Down, Frame, ..Motion, Frame, ..Up, Frame].
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "tfrm".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("tfrm").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = TouchLog::default();
    let client_a = spawn_touch_client(
        wayland_sock.clone(),
        "waymux.tFA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "tfrm").await;

    for phase in [
        waymux_protocol::TouchPhase::Down,
        waymux_protocol::TouchPhase::Motion,
        waymux_protocol::TouchPhase::Up,
    ] {
        let resp = c
            .request(RequestMethod::InjectTouch {
                name: "tfrm".into(),
                id: 0,
                x: 10.0,
                y: 20.0,
                phase,
                window_id: Some(wid_a),
                content: false,
            })
            .await;
        assert!(resp.ok, "inject_touch {:?} failed: {:?}", phase, resp.error);
    }

    // Wait for the full sequence to land on the client.
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let a_events = log_a.0.lock().unwrap().clone();
        let has_down = a_events
            .iter()
            .any(|e| matches!(e, TouchEvent::Down { .. }));
        let has_motion = a_events
            .iter()
            .any(|e| matches!(e, TouchEvent::Motion { .. }));
        let has_up = a_events.iter().any(|e| matches!(e, TouchEvent::Up { .. }));
        if has_down && has_motion && has_up {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "touch sequence (Down, Motion, Up) incomplete; A log: {:?}",
                a_events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Each touch event must be followed by a Frame marker before the next
    // logical event. Walk the log filtering out wl_touch.cancel (`_ => {}`
    // in the dispatch impl). Expected order: Down, Frame, Motion, Frame,
    // Up, Frame. Any additional inter-event frames are tolerated.
    let events = log_a.0.lock().unwrap().clone();
    let down_idx = events
        .iter()
        .position(|e| matches!(e, TouchEvent::Down { .. }))
        .expect("Down must be present");
    let motion_idx = events
        .iter()
        .position(|e| matches!(e, TouchEvent::Motion { .. }))
        .expect("Motion must be present");
    let up_idx = events
        .iter()
        .position(|e| matches!(e, TouchEvent::Up { .. }))
        .expect("Up must be present");

    assert!(
        down_idx < motion_idx && motion_idx < up_idx,
        "touch events must arrive in Down → Motion → Up order; got {:?}",
        events
    );

    let frame_between_down_motion = events[down_idx..motion_idx]
        .iter()
        .any(|e| matches!(e, TouchEvent::Frame));
    assert!(
        frame_between_down_motion,
        "expected Frame between Down and Motion; got {:?}",
        events
    );
    let frame_between_motion_up = events[motion_idx..up_idx]
        .iter()
        .any(|e| matches!(e, TouchEvent::Frame));
    assert!(
        frame_between_motion_up,
        "expected Frame between Motion and Up; got {:?}",
        events
    );
    let frame_after_up = events[up_idx..]
        .iter()
        .any(|e| matches!(e, TouchEvent::Frame));
    assert!(frame_after_up, "expected Frame after Up; got {:?}", events);

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn inject_touch_falls_back_to_buffer_coords_when_geometry_unset() {
    // content=true on a window that never emitted set_window_geometry must
    // fall back to (0, 0) inset (Q4 2026-05-18 decision) and deliver the
    // touch at the unmodified SDK-supplied coords. Then a follow-up
    // `inject_pointer(content=true)` on the SAME window must NOT fire a
    // second warning — the shared `content_fallback_warned` HashSet
    // already inserted the window_id during the touch call.
    //
    // We can't read the session subprocess's tracing output from the test
    // process, so the shared-dedup invariant is asserted indirectly:
    // - both calls return ok (the fallback path doesn't error)
    // - both calls deliver at the unmodified SDK coords
    // The unit-test `content_fallback_warned_dedup_inserts_once_per_window_id`
    // in state.rs covers the HashSet semantics directly; this test verifies
    // the integrated path stays well-behaved when both inject_touch and
    // inject_pointer probe the same window's content_rect.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "tcnt".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("tcnt").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let touch_log = TouchLog::default();
    let pointer_log = PointerLog::default();
    let client_a = spawn_touch_and_pointer_client(
        wayland_sock.clone(),
        "waymux.tcA",
        touch_log.clone(),
        pointer_log.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "tcnt").await;

    // inject_touch(content=true) at SDK (5, 7) on a window with no
    // set_window_geometry. Fallback inset is (0, 0), scale=1 ⇒ buffer (5, 7).
    let resp = c
        .request(RequestMethod::InjectTouch {
            name: "tcnt".into(),
            id: 0,
            x: 5.0,
            y: 7.0,
            phase: waymux_protocol::TouchPhase::Down,
            window_id: Some(wid_a),
            content: true,
        })
        .await;
    assert!(
        resp.ok,
        "inject_touch content=true (no geometry) failed: {:?}",
        resp.error
    );

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let events = touch_log.0.lock().unwrap().clone();
        let hit = events.iter().any(|e| {
            matches!(e, TouchEvent::Down { x, y, .. }
                if (*x - 5.0).abs() < 0.01 && (*y - 7.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "content=true touch Down at (5, 7) not delivered; events: {:?}",
                events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Shared warn-once dedup: a follow-up inject_pointer(content=true) on
    // the same window must still deliver (we don't have a way to count
    // warnings from this process, but the path must remain functional —
    // the dedup HashSet returning false from `insert` is silent).
    let resp = c
        .request(RequestMethod::InjectPointer {
            name: "tcnt".into(),
            x: 11.0,
            y: 13.0,
            button: 0,
            state: waymux_protocol::KeyState::Released,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: Some(wid_a),
            content: true,
        })
        .await;
    assert!(
        resp.ok,
        "follow-up inject_pointer content=true on the same window failed: {:?}",
        resp.error
    );

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let events = pointer_log.0.lock().unwrap().clone();
        let hit = events.iter().any(|e| {
            matches!(e, PointerEvent::Motion { x, y }
                if (*x - 11.0).abs() < 0.01 && (*y - 13.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "follow-up content=true pointer motion at (11, 13) not delivered; events: {:?}",
                events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn inject_touch_with_unknown_window_id_returns_false() {
    // Unknown window_id is a hard drop — control.rs surfaces this as
    // `resp.ok == false`. Mirrors `inject_pointer_with_unknown_window_id_returns_false`.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "tunk".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("tunk").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = TouchLog::default();
    let client_a = spawn_touch_client(
        wayland_sock.clone(),
        "waymux.tuA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let _wid_a = wait_for_next_window(&mut c, "tunk").await;

    let resp = c
        .request(RequestMethod::InjectTouch {
            name: "tunk".into(),
            id: 0,
            x: 10.0,
            y: 20.0,
            phase: waymux_protocol::TouchPhase::Down,
            window_id: Some(99999),
            content: false,
        })
        .await;
    assert!(
        !resp.ok,
        "inject_touch with unknown window_id must NOT report ok"
    );

    // Wait a beat, then confirm the only registered client got nothing.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let events = log_a.0.lock().unwrap().clone();
    assert!(
        !events.iter().any(|e| matches!(e, TouchEvent::Down { .. })),
        "client A must NOT receive Down from an unknown-window_id touch call; \
         events: {:?}",
        events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn inject_touch_passes_logical_coords_through_at_session_scale() {
    // Scale > 1 coordinate handling, touch edition. Mirrors
    // `inject_pointer_passes_logical_coords_through_at_session_scale`: at
    // scale=2 the SDK's logical (10, 20) must arrive UNCHANGED as a
    // wl_touch.down at (10, 20). wl_touch coordinates are surface-local logical
    // pixels by protocol; session scale governs buffer rendering only.
    // content=false isolates the scale path from the content_rect inset.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "tscl".into(),
        width: 320,
        height: 240,
        scale: 2,
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

    let wayland_sock = d.state_dir.join("tscl").join("wayland.sock");
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let log_a = TouchLog::default();
    let client_a = spawn_touch_client(
        wayland_sock.clone(),
        "waymux.tsA",
        log_a.clone(),
        stop_flag.clone(),
    );
    let wid_a = wait_for_next_window(&mut c, "tscl").await;

    let resp = c
        .request(RequestMethod::InjectTouch {
            name: "tscl".into(),
            id: 0,
            x: 10.0,
            y: 20.0,
            phase: waymux_protocol::TouchPhase::Down,
            window_id: Some(wid_a),
            content: false,
        })
        .await;
    assert!(resp.ok, "inject_touch (scale=2) failed: {:?}", resp.error);

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let events = log_a.0.lock().unwrap().clone();
        let hit = events.iter().any(|e| {
            matches!(e, TouchEvent::Down { id: 0, x, y }
                if (*x - 10.0).abs() < 0.01 && (*y - 20.0).abs() < 0.01)
        });
        if hit {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "scale=2 touch Down at logical (10, 20) not delivered unchanged; \
                 events: {:?}",
                events
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Negative-control: confirm the old buggy (20, 40) = (10,20)×2 Down was
    // NOT delivered (would indicate the scale multiply regressed back in).
    let events = log_a.0.lock().unwrap().clone();
    let scaled = events.iter().any(|e| {
        matches!(e, TouchEvent::Down { x, y, .. }
            if (*x - 20.0).abs() < 0.01 && (*y - 40.0).abs() < 0.01)
    });
    assert!(
        !scaled,
        "scale=2 session delivered touch Down at (20, 40) = logical (10,20) × 2; \
         the buggy scale multiply is back. events: {:?}",
        events
    );

    stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = client_a.await;
}

#[tokio::test]
async fn session_advertises_wl_touch_capability() {
    // wl_seat.capabilities bitfield must include the Touch bit (= 4).
    // Connect a minimal Wayland client, bind wl_seat with a SeatCapLog data
    // type, and read the Capabilities event.
    let d = Daemon::spawn().await;
    let mut c = d.connect().await;
    c.hello().await;
    c.request(RequestMethod::CreateSession(CreateSessionParams {
        name: "tcap".into(),
        width: 320,
        height: 240,
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

    let wayland_sock = d.state_dir.join("tcap").join("wayland.sock");
    let seat_log = SeatCapLog::default();
    let seat_log_for_thread = seat_log.clone();
    let client = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let connect_deadline = std::time::Instant::now() + Duration::from_secs(3);
        let stream = loop {
            match std::os::unix::net::UnixStream::connect(&wayland_sock) {
                Ok(s) => break s,
                Err(_) if std::time::Instant::now() < connect_deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(anyhow::Error::from(e)),
            }
        };
        let conn = Connection::from_socket(stream)?;
        let (globals, mut queue) = registry_queue_init::<ClientState>(&conn)?;
        let qh = queue.handle();
        let mut state = ClientState::default();
        // Bind wl_seat with the SeatCapLog data; capabilities event is
        // delivered ~immediately after bind by the wl_seat dispatch impl in
        // compositor.rs (`GlobalDispatch<WlSeat, ()>`).
        let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, seat_log_for_thread.clone())?;

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while seat_log_for_thread.0.lock().unwrap().is_none() {
            if std::time::Instant::now() >= deadline {
                anyhow::bail!("never received wl_seat.capabilities event");
            }
            let _ = queue.flush();
            if let Some(guard) = conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = queue.dispatch_pending(&mut state);
            std::thread::sleep(Duration::from_millis(10));
        }
        std::mem::forget(seat);
        std::mem::forget(conn);
        std::mem::forget(queue);
        std::mem::forget(state);
        Ok(())
    });
    client
        .await
        .expect("touch-cap client thread")
        .expect("touch-cap client setup");

    let caps = seat_log
        .0
        .lock()
        .unwrap()
        .expect("capabilities must have been recorded");
    // Per the wl_seat XML, Touch = 4. Verify the bit is set and that
    // Pointer (1) and Keyboard (2) are also present (regression guard
    // against accidentally dropping a bit when adding Touch).
    const POINTER: u32 = 1;
    const KEYBOARD: u32 = 2;
    const TOUCH: u32 = 4;
    assert_eq!(
        caps & TOUCH,
        TOUCH,
        "wl_seat.capabilities (0x{:x}) must include Touch (0x4)",
        caps
    );
    assert_eq!(
        caps & POINTER,
        POINTER,
        "wl_seat.capabilities (0x{:x}) must still include Pointer (0x1)",
        caps
    );
    assert_eq!(
        caps & KEYBOARD,
        KEYBOARD,
        "wl_seat.capabilities (0x{:x}) must still include Keyboard (0x2)",
        caps
    );
}
