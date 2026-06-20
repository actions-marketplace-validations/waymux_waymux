// SPDX-License-Identifier: Apache-2.0

//! Per-session attach server.
//!
//! A second Wayland server (distinct from the inner compositor) that
//! advertises the `waymux_attach_v1` global. The attach client connects
//! here after calling `attach(name)` on the daemon; the first stage of
//! the protocol — client → `get_surface(outer_fd)` — is handled by
//! receiving the outer compositor's display fd via Wayland's built-in
//! fd-passing (SCM_RIGHTS under the hood).
//!
//! Today this slice only stands up the server: bind the socket, accept
//! connections, announce the global, log receipt of `get_surface`. The
//! actual outer-compositor handoff (wrapping the fd → creating a
//! `wl_surface` there → ferrying frames) is the next slice.

use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use wayland_server::{
    backend::{ClientData, ClientId, DisconnectReason},
    protocol::wl_surface::{self, WlSurface},
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};

use crate::attach_proto::waymux_attach_v1;
use crate::outer_view;
use crate::state::State;

/// Tracks the currently-running outer-view thread: its stop flag and join
/// handle, kept together so that on respawn we can both signal the old
/// thread to stop *and* await its full teardown before spawning a new one
/// (audit H16). Without joining, the old thread's late teardown can race
/// the new thread's setup — e.g. clearing each other's `outer_wake_fd`.
struct ActiveOuterView {
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
}

/// Dispatch state for the attach server. Kept separate from the inner
/// compositor's state because the two Wayland servers live on different
/// fds, run on different threads, and have entirely disjoint global sets.
pub struct AttachServer {
    pub state: Arc<State>,
    /// The most recently spawned outer-view thread. Per the protocol spec non-goal
    /// "shared-cursor multi-attach" we only expect one attach at a time, so
    /// a single slot suffices. On respawn the previous thread is signalled
    /// (`stop = true`) and joined before the new one is launched. The join
    /// is bounded — `outer_view::run` checks `stop` at minimum once per
    /// Wayland event dispatch and returns promptly when set.
    active_view: std::sync::Mutex<Option<ActiveOuterView>>,
    /// The current attach client's waymux_attach_v1 resource. Held so we
    /// can post a protocol error to disconnect `waymux-attach` when the
    /// outer window is closed by the compositor (e.g. Alt+Shift+Q in Niri).
    current_attach: std::sync::Mutex<Option<waymux_attach_v1::WaymuxAttachV1>>,
}

impl AttachServer {
    pub fn new(state: Arc<State>) -> Self {
        Self {
            state,
            active_view: std::sync::Mutex::new(None),
            current_attach: std::sync::Mutex::new(None),
        }
    }
}

/// Read the peer's uid from a `std::os::unix::net::UnixStream` using
/// SO_PEERCRED via libc directly (the stdlib `peer_cred()` is unstable).
/// Returns the connecting process's uid, used by C6's same-uid auth check.
fn peer_uid(stream: &std::os::unix::net::UnixStream) -> std::io::Result<u32> {
    use std::os::fd::AsRawFd;
    let fd = stream.as_raw_fd();
    let mut cred = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut cred as *mut _ as *mut libc::c_void,
            &mut len,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(cred.uid)
}

struct AttachClientData;
impl ClientData for AttachClientData {
    fn initialized(&self, id: ClientId) {
        debug!(?id, "attach client connected");
    }
    fn disconnected(&self, id: ClientId, reason: DisconnectReason) {
        debug!(?id, ?reason, "attach client disconnected");
    }
}

pub fn run(socket_path: &Path, state: Arc<State>) -> Result<()> {
    let listener = std::os::unix::net::UnixListener::bind(socket_path)
        .with_context(|| format!("bind attach socket {}", socket_path.display()))?;
    // Audit C6: force 0600 — same-uid only. Default umask leaves the socket
    // world-readable; any local process could connect, hijack the session's
    // video output, or force-detach the legitimate attach client. Mirror
    // the daemon main socket pattern.
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod attach socket {}", socket_path.display()))?;
    }
    listener.set_nonblocking(true)?;
    info!(path = %socket_path.display(), "attach server listening");

    let mut display: wayland_server::Display<AttachServer> =
        wayland_server::Display::new().context("create attach Display")?;
    let mut dh = display.handle();

    let _ = dh.create_global::<AttachServer, waymux_attach_v1::WaymuxAttachV1, ()>(1, ());

    let mut server = AttachServer::new(state);

    let listener_fd = listener.as_raw_fd();
    let display_fd = display.backend().poll_fd().as_raw_fd();
    loop {
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    // Audit C6: SO_PEERCRED uid check. Even with 0600 perms
                    // belt-and-braces — refuse any connection whose peer uid
                    // doesn't match ours. We call getsockopt directly because
                    // std::os::unix::net::UnixStream::peer_cred is still
                    // unstable (peer_credentials_unix_socket).
                    let my_uid = unsafe { libc::getuid() };
                    match peer_uid(&stream) {
                        Ok(uid) if uid == my_uid => {
                            match dh.insert_client(stream, Arc::new(AttachClientData)) {
                                Ok(_) => {}
                                Err(e) => warn!(error = %e, "attach insert_client failed"),
                            }
                        }
                        Ok(uid) => {
                            warn!(uid, "attach socket: rejected non-matching uid");
                        }
                        Err(e) => {
                            warn!(error = %e, "attach socket: peer_cred failed");
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => {
                    warn!(error = %e, "attach accept failed");
                    break;
                }
            }
        }

        if let Err(e) = display.dispatch_clients(&mut server) {
            warn!(error = %e, "attach dispatch_clients failed");
        }

        // If the outer view stopped because the compositor closed the window
        // (not because the attach client called detach), disconnect the attach
        // client so `waymux-attach` exits cleanly.
        {
            let view_stopped = server
                .active_view
                .lock()
                .unwrap()
                .as_ref()
                .map(|v| v.stop.load(std::sync::atomic::Ordering::Acquire))
                .unwrap_or(false);
            if view_stopped {
                // The thread set `stop = true` itself (compositor closed the
                // outer window). Take the slot out under the lock, drop the
                // guard, then join — `join()` should return promptly since
                // the thread already signalled it's exiting. Joining here
                // (rather than letting the slot linger) ensures any late
                // teardown work (clearing `outer_wake_fd`, dropping the
                // surface) completes before we touch shared state below.
                let view = server.active_view.lock().unwrap().take();
                if let Some(view) = view {
                    if let Err(e) = view.handle.join() {
                        warn!(?e, "outer view thread panicked during cleanup join");
                    }
                }
                server.state.set_outer_wake_fd(None);
                if let Some(res) = server.current_attach.lock().unwrap().take() {
                    if res.is_alive() {
                        debug!("attach: outer view closed — disconnecting attach client");
                        res.post_error(0u32, "outer window closed by compositor");
                    }
                }
            }
        }

        if let Err(e) = display.flush_clients() {
            warn!(error = %e, "attach flush_clients failed");
        }

        // Unlike the inner compositor, this server has no cross-thread
        // event wakeups (yet) — its dispatch is entirely driven by client
        // requests. A 1s idle timeout is plenty; when we start emitting
        // `frame` / `resized` / `occluded` from elsewhere we'll add an
        // eventfd like the inner compositor has.
        let mut fds = [
            libc::pollfd {
                fd: listener_fd,
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: display_fd,
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        unsafe {
            libc::poll(
                fds.as_mut_ptr(),
                fds.len() as _,
                Duration::from_secs(1).as_millis() as i32,
            );
        }
    }
}

// ─── Dispatch impls ─────────────────────────────────────────────────────

// The `wl_surface` the attach client creates via `get_surface` lives on
// THIS display (the attach server) as a no-op proxy handle. We don't
// implement any of the wl_surface protocol's semantics on the attach side
// beyond keeping the object alive so the client can destroy it cleanly.
impl Dispatch<WlSurface, ()> for AttachServer {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlSurface,
        _request: wl_surface::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // attach, commit, damage, etc. on the proxy surface are ignored —
        // the real surface lives on the outer compositor (outer_view).
    }
}

impl GlobalDispatch<waymux_attach_v1::WaymuxAttachV1, ()> for AttachServer {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<waymux_attach_v1::WaymuxAttachV1>,
        _global_data: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        debug!("attach client bound waymux_attach_v1");
        di.init(resource, ());
    }
}

impl Dispatch<waymux_attach_v1::WaymuxAttachV1, ()> for AttachServer {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &waymux_attach_v1::WaymuxAttachV1,
        request: waymux_attach_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        use waymux_attach_v1::Request;
        match request {
            Request::GetSurface {
                id,
                outer_display_fd,
            } => {
                // `id` is a `new_id` argument; we MUST init it or the
                // Wayland framework fails the connection with a protocol
                // error. Store the resource so we can disconnect the client
                // if the outer window is closed by the compositor.
                let _surface = di.init(id, ());
                info!(
                    outer_display_fd = outer_display_fd.as_raw_fd(),
                    "attach: get_surface — spawning outer view"
                );
                // Store the waymux_attach_v1 resource for later disconnect.
                *state.current_attach.lock().unwrap() = Some(resource.clone());

                // Tear down any previous view BEFORE starting a new one.
                // We must both signal `stop` *and* `join()` the old thread —
                // otherwise its late teardown (clearing `outer_wake_fd`,
                // dropping the surface) races the new thread's setup. The
                // Arc-pointer-match in `clear_outer_wake_fd_if_mine` only
                // protects one direction; this join closes the other.
                //
                // We take the slot out of the mutex first and drop the guard
                // before joining, so `join()` cannot deadlock other waiters.
                // The join is bounded: `outer_view::run` checks `stop` at
                // minimum once per Wayland event dispatch (audit H16).
                let prev = state.active_view.lock().unwrap().take();
                if let Some(prev) = prev {
                    prev.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    if let Err(e) = prev.handle.join() {
                        warn!(?e, "previous outer view thread panicked during join");
                    }
                }
                let stop = Arc::new(AtomicBool::new(false));

                let view_state = state.state.clone();
                let thread_stop = stop.clone();
                let handle = std::thread::Builder::new()
                    .name("waymux-outer-view".into())
                    .spawn(move || {
                        if let Err(e) = outer_view::run(outer_display_fd, view_state, thread_stop) {
                            warn!(error = %e, "outer view thread exited with error");
                        }
                    })
                    .expect("spawn outer view thread");
                *state.active_view.lock().unwrap() = Some(ActiveOuterView { stop, handle });
            }
            Request::Detach => {
                debug!("attach: detach");
                // Take the slot, signal stop, and join before returning. We
                // join inside the dispatch handler (rather than deferring to
                // the next `get_surface`) so that a Detach-then-reattach
                // pattern from a fresh `waymux-attach` cannot race the old
                // thread's late teardown (audit H16). The join is bounded —
                // `outer_view::run` checks `stop` at minimum once per
                // Wayland event dispatch.
                let prev = state.active_view.lock().unwrap().take();
                if let Some(prev) = prev {
                    prev.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    if let Err(e) = prev.handle.join() {
                        warn!(?e, "outer view thread panicked during detach join");
                    }
                }
                // Disconnect the attach client so `waymux-attach` exits.
                // We post a protocol error which causes wl_display.error on
                // the client side — it exits on the next dispatch.
                if let Some(res) = state.current_attach.lock().unwrap().take() {
                    if res.is_alive() {
                        res.post_error(0u32, "detached");
                    }
                }
                // outer_view::run clears outer_wake_fd on its own before
                // returning; clear it here defensively in case the thread
                // is slow to observe `stop`.
                state.state.set_outer_wake_fd(None);
            }
        }
    }
}
