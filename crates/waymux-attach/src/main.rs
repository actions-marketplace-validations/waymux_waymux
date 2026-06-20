// SPDX-License-Identifier: Apache-2.0

//! `waymux-attach` — the attach client binary.
//!
//! Usage: `waymux-attach <attach-socket-path>`
//!
//! The path is what `waymux attach <session>` prints. The client opens
//! two Wayland connections:
//!
//! 1. **Outer** — `$WAYLAND_DISPLAY` (the user's real compositor). The
//!    fd is passed to the session via SCM_RIGHTS in the `get_surface`
//!    request; the session then becomes a client of the outer compositor
//!    itself and drives the shared `wl_surface`.
//! 2. **Attach** — the session's `waymux_attach_v1` socket. This client
//!    binds the `waymux_attach_v1` global, calls `get_surface` once,
//!    then waits for `frame` / `resized` / `occluded` events.
//!
//! Input forwarding (the protocol spec `waymux_input_v1`) is not yet wired — that
//! lands in the next slice.

#![allow(clippy::all)]

use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use wayland_client::{
    globals::{registry_queue_init, GlobalListContents},
    protocol::{wl_registry::WlRegistry, wl_surface::WlSurface},
    Connection, Dispatch, QueueHandle,
};

// Client-side scanner bindings for waymux_attach_v1.
#[allow(
    dead_code,
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    unused_imports,
    unused_unsafe,
    unused_variables,
    missing_docs,
    clippy::all
)]
mod attach_proto {
    use wayland_client;
    use wayland_client::protocol::*;

    pub mod __interfaces {
        use wayland_client::protocol::__interfaces::*;
        wayland_scanner::generate_interfaces!("protocols/waymux-attach-v1.xml");
    }
    use self::__interfaces::*;

    wayland_scanner::generate_client_code!("protocols/waymux-attach-v1.xml");
}

use attach_proto::waymux_attach_v1;

#[derive(Parser, Debug)]
#[command(name = "waymux-attach", version)]
struct Args {
    /// Filesystem path of the session's attach socket, as printed by
    /// `waymux attach <session>`.
    attach_socket: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // ── 1. Open the outer compositor display ─────────────────────────────
    let outer_path = outer_wayland_path()?;
    eprintln!("attach: outer display = {}", outer_path.display());
    let outer_stream = UnixStream::connect(&outer_path)
        .with_context(|| format!("connect outer {}", outer_path.display()))?;
    let outer_fd = outer_stream.as_fd().try_clone_to_owned()?;
    // Keep `outer_stream` alive for our own use, and pass a duped fd to
    // the session via SCM_RIGHTS. (Wayland's `fd` argument type does
    // the SCM_RIGHTS handling under the hood.)

    // ── 2. Connect to the session's attach socket ────────────────────────
    eprintln!("attach: attach socket = {}", args.attach_socket.display());
    let attach_stream = UnixStream::connect(&args.attach_socket)
        .with_context(|| format!("connect attach {}", args.attach_socket.display()))?;
    let conn = Connection::from_socket(attach_stream).context("wrap attach socket")?;
    let (globals, mut queue) =
        registry_queue_init::<ClientState>(&conn).context("attach registry init")?;
    let qh = queue.handle();

    // Bind the waymux_attach_v1 global.
    let attach: waymux_attach_v1::WaymuxAttachV1 = globals
        .bind(&qh, 1..=1, ())
        .context("attach socket does not advertise waymux_attach_v1")?;
    eprintln!("attach: bound waymux_attach_v1");

    // ── 3. Call get_surface, passing the outer fd ────────────────────────
    let _surface = attach.get_surface(outer_fd.as_fd(), &qh, ());
    eprintln!("attach: sent get_surface; waiting for events");

    // Wait for events until Ctrl-C or session disconnect.
    // `prepare_read()+dispatch_pending()` is the standard polling pattern.
    // Audit H18: detect protocol error / EOF and exit instead of spinning
    // forever at 50 ms when the session has died.
    let mut state = ClientState;
    loop {
        queue.flush()?;
        if let Some(guard) = conn.prepare_read() {
            match guard.read() {
                Ok(_) => {}
                Err(wayland_client::backend::WaylandError::Io(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof
                        || e.kind() == std::io::ErrorKind::BrokenPipe
                        || e.kind() == std::io::ErrorKind::ConnectionReset =>
                {
                    eprintln!("attach: session disconnected — exiting");
                    return Ok(());
                }
                Err(wayland_client::backend::WaylandError::Protocol(err)) => {
                    eprintln!("attach: wayland protocol error: {err:?} — exiting");
                    return Ok(());
                }
                Err(_) => {
                    // Other transient errors (e.g. EAGAIN) — let dispatch
                    // surface them.
                }
            }
        }
        queue.dispatch_pending(&mut state)?;
        if let Some(err) = conn.protocol_error() {
            eprintln!("attach: wayland protocol error: {err:?} — exiting");
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn outer_wayland_path() -> Result<PathBuf> {
    let display = std::env::var("WAYLAND_DISPLAY")
        .context("WAYLAND_DISPLAY not set — are you running inside a Wayland session?")?;
    if display.starts_with('/') {
        return Ok(PathBuf::from(display));
    }
    let runtime = std::env::var("XDG_RUNTIME_DIR").context("XDG_RUNTIME_DIR not set")?;
    Ok(PathBuf::from(runtime).join(display))
}

// ─── Dispatch ───────────────────────────────────────────────────────────

struct ClientState;

impl Dispatch<WlRegistry, GlobalListContents> for ClientState {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: wayland_client::protocol::wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<waymux_attach_v1::WaymuxAttachV1, ()> for ClientState {
    fn event(
        _: &mut Self,
        _: &waymux_attach_v1::WaymuxAttachV1,
        event: waymux_attach_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            waymux_attach_v1::Event::Frame {
                serial,
                timestamp_ns,
            } => {
                eprintln!("attach event: frame serial={serial} t={timestamp_ns}ns");
            }
            waymux_attach_v1::Event::Resized { width, height } => {
                eprintln!("attach event: resized {width}×{height}");
            }
            waymux_attach_v1::Event::Occluded { is_occluded } => {
                eprintln!("attach event: occluded={is_occluded}");
            }
        }
    }
}

impl Dispatch<WlSurface, ()> for ClientState {
    fn event(
        _: &mut Self,
        _: &WlSurface,
        _: wayland_client::protocol::wl_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
