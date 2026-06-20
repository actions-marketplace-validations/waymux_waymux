// SPDX-License-Identifier: Apache-2.0

//! Per-session compositor process.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::Notify;
use tracing::{info, warn};

mod attach_proto;
mod attach_server;
mod audio;
mod buffer;
mod composite;
mod compositor;
mod control;
mod cuda_nvenc_record;
mod dmabuf;
mod events;
mod ffv1_vk_record;
mod hevc_vk_record;
mod input_bridge;
mod keymap;
mod outer_view;
mod recording;
mod shm;
mod state;
mod syncobj;
mod vaapi_h264_record;
mod viewer;
mod vulkan_record;
mod wl_drm_proto;

#[derive(Parser, Debug)]
#[command(name = "waymux-session", version)]
struct Args {
    #[arg(long)]
    name: String,
    #[arg(long)]
    width: u32,
    #[arg(long)]
    height: u32,
    #[arg(long, default_value_t = 1)]
    scale: u32,
    /// Filesystem path for the inner Wayland display socket.
    #[arg(long)]
    inner_socket: PathBuf,
    /// Filesystem path for the daemon↔session control socket.
    #[arg(long)]
    control_socket: PathBuf,
    /// Filesystem path for the session→daemon events socket. If absent,
    /// events are dropped on the floor (useful for manual testing).
    #[arg(long)]
    events_socket: Option<PathBuf>,
    /// Filesystem path where the session serves the `waymux_attach_v1`
    /// Wayland protocol (the protocol spec). Absent = no attach server; useful
    /// for manual testing without the daemon.
    #[arg(long)]
    attach_socket: Option<PathBuf>,
    /// Optional readiness socket.
    #[arg(long)]
    ready_socket: Option<PathBuf>,
    /// Bridge clipboard selections between inner session and outer compositor.
    #[arg(long, default_value_t = false)]
    share_clipboard: bool,
    /// Symlink the host's PulseAudio/PipeWire sockets into this session's
    /// runtime dir so spawned apps can auto-discover audio.
    #[arg(long, default_value_t = false)]
    share_audio: bool,
    /// multi-workspace: when set, auto-issue ViewerStart on this
    /// session's own control socket once the compositor is ready. The
    /// existing ViewerStart handler in control.rs spawns
    /// `waymux-neko-bridge` as a child bound to `0.0.0.0:<port>`. Used
    /// by the multi-workspace supervisor: each workspace's systemd unit gets a
    /// dedicated bridge_port (8082..=8096), the supervisor passes it
    /// here, and the customer browser reaches the per-workspace viewer
    /// at `http://<floating_ip>:<port>/?token=<jwt>`.
    ///
    /// None = no autostart (legacy single-tenant flow where the daemon's
    /// `waymux viewer start` RPC drives the spawn separately).
    #[arg(long)]
    viewer_port: Option<u16>,
    /// Bind address for the autostart viewer when `--viewer-port` is set.
    /// Defaults to `0.0.0.0` so the customer browser can reach it via the
    /// VM's floating IP. Override for local-loopback testing.
    #[arg(long, default_value = "0.0.0.0")]
    viewer_bind: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    // Rich panic hook so a panic in any thread (compositor, attach server,
    // outer view, control socket) logs via tracing BEFORE unwinding past
    // an FFI boundary aborts the whole process with SIGABRT. Without this,
    // a wayland-rs callback panic surfaces only as "exit_code=134" in the
    // daemon's supervisor log.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        tracing::error!(thread = %name, "session panic: {info}");
        default_hook(info);
    }));

    // SIGSEGV handler: Rust's panic machinery doesn't fire for signals, so
    // without this a memory fault from wayland-sys / libc surfaces only as
    // a bare "exit_code=139" with no hint where. The handler prints a
    // best-effort backtrace then re-raises so the kernel can still core
    // dump for gdb post-mortem. Calling Backtrace::force_capture in a
    // signal handler is not strictly async-signal-safe (it allocates +
    // symbolizes), but for development debugging the tradeoff is worth it.
    unsafe {
        let handler = sigsegv_handler as extern "C" fn(libc::c_int);
        libc::signal(libc::SIGSEGV, handler as libc::sighandler_t);
        libc::signal(libc::SIGBUS, handler as libc::sighandler_t);
    }

    let args = Args::parse();
    info!(
        name = %args.name,
        size = format!("{}x{}@{}", args.width, args.height, args.scale),
        inner = %args.inner_socket.display(),
        control = %args.control_socket.display(),
        "session starting"
    );

    cleanup_stale(&args.inner_socket);
    cleanup_stale(&args.control_socket);
    if let Some(p) = &args.attach_socket {
        cleanup_stale(p);
    }

    let session_dir = args
        .inner_socket
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let audio_links = if args.share_audio {
        audio::setup(&session_dir)
    } else {
        audio::AudioLinks::empty()
    };

    // Set up the event pipe if the daemon gave us one.
    let (event_sink, event_task, daemon_gone) = match &args.events_socket {
        Some(path) => {
            let (tx, rx) = events::channel();
            let sink = events::EventSink::new(args.name.clone(), tx);
            let (gone_tx, gone_rx) = tokio::sync::oneshot::channel();
            let path = path.clone();
            let task = tokio::spawn(async move {
                if let Err(e) = events::run(&path, rx, gone_tx).await {
                    warn!(error = %e, "events task exited with error");
                }
            });
            (Some(sink), Some(task), Some(gone_rx))
        }
        None => (None, None, None),
    };

    let state = Arc::new(state::State::new(
        args.name.clone(),
        args.width,
        args.height,
        args.scale,
        event_sink,
        args.share_clipboard,
    ));
    let shutdown = Arc::new(Notify::new());

    // Compositor on a dedicated OS thread (blocking event loop per the protocol spec).
    let compositor_state = state.clone();
    let compositor_socket = args.inner_socket.clone();
    std::thread::Builder::new()
        .name("waymux-compositor".into())
        .spawn(move || {
            if let Err(e) = compositor::run(&compositor_socket, compositor_state) {
                warn!(error = %e, "compositor thread exited with error");
            }
        })
        .context("spawn compositor thread")?;

    // Attach server (separate thread, separate Wayland Display) runs only
    // if the daemon gave us a socket path to bind.
    if let Some(attach_path) = args.attach_socket.clone() {
        let attach_state = state.clone();
        std::thread::Builder::new()
            .name("waymux-attach".into())
            .spawn(move || {
                if let Err(e) = attach_server::run(&attach_path, attach_state) {
                    warn!(error = %e, "attach server thread exited with error");
                }
            })
            .context("spawn attach thread")?;
    }

    let control_listener =
        bind_tokio_listener(&args.control_socket).context("bind control socket")?;
    let control_task = {
        let state = state.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            control::run(control_listener, state, shutdown).await;
        })
    };

    // Headless tick driver: when no attach client is active, periodically poke
    // the inner compositor to drain frame_callbacks at a fixed rate (~60 Hz).
    // Without this, KWin and its clients (DXVK, games) throttle to ~9 FPS
    // when detached. With this timer, they maintain full FPS. (task #98)
    let headless_task = {
        let state = state.clone();
        tokio::spawn(async move {
            run_headless_tick_driver(state).await;
        })
    };

    // viewer autostart for the multi-workspace supervisor.
    // When `--viewer-port` is set we dispatch a synthetic ViewerStart RPC
    // through the same handler the daemon would have invoked via the
    // control socket. The handler spawns `waymux-neko-bridge` as a child
    // bound to `<viewer_bind>:<viewer_port>`; the customer browser hits
    // it through the workspace's bridge_port allocation on the VM.
    if let Some(port) = args.viewer_port {
        let req = waymux_protocol::SessionCtlRequest {
            id: 0,
            method: waymux_protocol::SessionCtlMethod::ViewerStart {
                bind: Some(args.viewer_bind.clone()),
                port: Some(port),
            },
        };
        let resp = control::dispatch(req, &state, &shutdown);
        match resp.error.as_ref() {
            None => {
                info!(viewer_port = port, "viewer autostart succeeded");
            }
            Some(err) => {
                // Don't abort the session — the customer can still use it
                // via attach + the daemon-mediated viewer flow. The portal
                // Connect button will just be unreachable for this workspace.
                warn!(viewer_port = port, error = %err, "viewer autostart failed");
            }
        }
    }

    if let Some(ready) = &args.ready_socket {
        notify_ready(ready).await.context("notify ready")?;
    }

    let mut sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let mut sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;

    if let Some(mut daemon_gone) = daemon_gone {
        tokio::select! {
            _ = sigterm.recv() => info!("SIGTERM → exiting"),
            _ = sigint.recv()  => info!("SIGINT → exiting"),
            _ = shutdown.notified() => info!("control-socket shutdown → exiting"),
            _ = &mut daemon_gone => info!("events socket closed → daemon gone → exiting"),
        }
    } else {
        tokio::select! {
            _ = sigterm.recv() => info!("SIGTERM → exiting"),
            _ = sigint.recv()  => info!("SIGINT → exiting"),
            _ = shutdown.notified() => info!("control-socket shutdown → exiting"),
        }
    }

    control_task.abort();
    headless_task.abort();
    if let Some(t) = event_task {
        t.abort();
    }
    audio::cleanup(&audio_links);
    Ok(())
}

extern "C" fn sigsegv_handler(signum: libc::c_int) {
    let name = match signum {
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGBUS => "SIGBUS",
        _ => "signal",
    };
    let thread = std::thread::current();
    let tname = thread.name().unwrap_or("<unnamed>");
    eprintln!("=== {name} in thread '{tname}' ({}) ===", unsafe {
        libc::gettid()
    });
    let bt = std::backtrace::Backtrace::force_capture();
    eprintln!("{bt}");
    // Reset to default and re-raise so the kernel produces a core dump.
    unsafe {
        libc::signal(signum, libc::SIG_DFL);
        libc::raise(signum);
    }
}

/// Headless tick driver: when no attach client is connected, periodically poke
/// the inner compositor's wake fd to drive frame_callbacks at ~120 Hz. This
/// ensures inner clients (KWin, DXVK, games) maintain full frame rate even when
/// no one is viewing the output. With the timer, detached sessions run at full
/// FPS instead of throttling down to ~9 FPS due to frame_callback starvation.
///
/// 120 Hz (was 60 Hz) supports recording at `--min-fps 120` on hardware fast
/// enough to keep up. The extra 60 wakes/sec are negligible on every host we
/// run on; KWin and chromium both adapt to whatever cadence the frame callbacks
/// arrive at. Override via `WAYMUX_TICK_HZ` if you need to pin lower for power
/// measurements or higher for an experimental >120 fps run.
async fn run_headless_tick_driver(state: Arc<state::State>) {
    let hz: u64 = std::env::var("WAYMUX_TICK_HZ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let interval_us = 1_000_000_u64.checked_div(hz).unwrap_or(16_667);
    let tick_interval = std::time::Duration::from_micros(interval_us);
    let mut interval = tokio::time::interval(tick_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        interval.tick().await;
        // Only poke if there's no attach client active. If attach is active,
        // the outer_view's event loop already drives frame_callbacks via the
        // outer compositor's frame callbacks. Poking from here too would just
        // waste CPU (double-driving is harmless but pointless).
        if !state.is_attached() {
            state.poke_compositor_wake();
        }
    }
}

fn cleanup_stale(path: &std::path::Path) {
    if path.exists() {
        warn!(path = %path.display(), "removing stale socket");
        let _ = std::fs::remove_file(path);
    }
}

fn bind_tokio_listener(path: &std::path::Path) -> Result<tokio::net::UnixListener> {
    tokio::net::UnixListener::bind(path).with_context(|| format!("bind {}", path.display()))
}

async fn notify_ready(path: &std::path::Path) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut stream = tokio::net::UnixStream::connect(path)
        .await
        .with_context(|| format!("connect ready socket {}", path.display()))?;
    stream.write_all(b"ready\n").await?;
    stream.shutdown().await.ok();
    Ok(())
}
