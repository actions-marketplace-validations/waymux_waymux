// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use tokio::net::UnixListener;
use tokio::signal::unix::{signal, SignalKind};
use tracing::{info, warn};

mod backend;
mod cgroup;
mod quota;
mod registry;
mod server;
#[cfg(feature = "metering")]
mod usage_events;

use registry::Registry;
#[cfg(feature = "metering")]
use usage_events::UsageEventSink;

#[derive(Parser, Debug)]
#[command(name = "waymuxd", version)]
struct Args {
    /// Path to the control socket.
    /// Defaults to `$XDG_RUNTIME_DIR/waymux.sock`.
    #[arg(long, env = "WAYMUX_SOCKET")]
    socket: Option<PathBuf>,

    /// Path to the `waymux-session` binary to spawn for new sessions.
    /// Defaults to `<dir-of-waymuxd>/waymux-session`.
    #[arg(long, env = "WAYMUX_SESSION_BIN")]
    session_bin: Option<PathBuf>,

    /// Directory for per-session runtime state (inner sockets, logs).
    /// Defaults to `$XDG_RUNTIME_DIR/waymux/`.
    #[arg(long, env = "WAYMUX_STATE_DIR")]
    state_dir: Option<PathBuf>,

    /// Append-only JSONL stream of session-lifecycle usage events. When set,
    /// the daemon emits one line per `session_start` / `session_stop` and a
    /// `session_heartbeat` for each live session every 60s. Unset = no
    /// usage-event output. Only available in the `metering` build.
    #[cfg(feature = "metering")]
    #[arg(long, env = "WAYMUX_USAGE_EVENTS_SINK")]
    usage_events_sink: Option<PathBuf>,

    /// Which session backend to use. `local` (the only option today) spawns
    /// `waymux-session` subprocesses inside this daemon's host.
    #[arg(long, env = "WAYMUX_BACKEND", default_value = "local")]
    backend: backend::BackendChoice,
}

fn default_socket_path() -> Result<PathBuf> {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .context("XDG_RUNTIME_DIR not set — pass --socket or set the env var")?;
    Ok(PathBuf::from(runtime).join("waymux.sock"))
}

fn default_state_dir() -> Result<PathBuf> {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .context("XDG_RUNTIME_DIR not set — pass --state-dir or set the env var")?;
    Ok(PathBuf::from(runtime).join("waymux"))
}

fn default_session_bin() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("current_exe")?;
    let dir = exe
        .parent()
        .context("waymuxd has no parent directory (impossible)")?;
    Ok(dir.join("waymux-session"))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    let socket_path = match args.socket {
        Some(p) => p,
        None => default_socket_path()?,
    };
    let state_dir = match args.state_dir {
        Some(p) => p,
        None => default_state_dir()?,
    };
    let session_bin = match args.session_bin {
        Some(p) => p,
        None => default_session_bin()?,
    };

    std::fs::create_dir_all(&state_dir)
        .with_context(|| format!("create state dir {}", state_dir.display()))?;

    if socket_path.exists() {
        match std::os::unix::net::UnixStream::connect(&socket_path) {
            Ok(_) => anyhow::bail!(
                "socket {} is in use by another daemon",
                socket_path.display()
            ),
            Err(_) => {
                warn!(path = %socket_path.display(), "removing stale socket");
                std::fs::remove_file(&socket_path).ok();
            }
        }
    }

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("bind control socket {}", socket_path.display()))?;
    // Control socket is mode 0600 with an SO_PEERCRED same-uid check enforced
    // in the server; the permissions are belt-and-braces.
    std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("chmod {}", socket_path.display()))?;

    info!(
        socket = %socket_path.display(),
        state_dir = %state_dir.display(),
        session_bin = %session_bin.display(),
        "waymuxd listening"
    );

    #[cfg(feature = "metering")]
    let registry = {
        let usage_sink = match args.usage_events_sink {
            Some(path) => match UsageEventSink::open(&path) {
                Ok(sink) => {
                    info!(path = %path.display(), "usage events sink active");
                    Some(sink)
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "open usage events sink failed; usage events disabled");
                    None
                }
            },
            None => None,
        };

        // Audit C8: stamp every emitted usage event with a per-process UUID
        // so the consumer can disambiguate sessions that share a name across
        // daemon restarts. Without this, `session_name` is not a stable join
        // key and consecutive lifetimes silently merge usage windows.
        let run_id = uuid::Uuid::new_v4().to_string();
        info!(%run_id, "daemon run_id");

        Registry::new(state_dir, session_bin, usage_sink, run_id)
    };

    #[cfg(not(feature = "metering"))]
    let registry = Registry::new(state_dir, session_bin);

    // Build the session-lifecycle backend and HAND IT TO `server::run`. For
    // the `local` backend this wraps the same `registry` every other op reads,
    // so `create`/`destroy` route through the abstraction while staying
    // byte-identical to the old direct-registry path.
    let session_backend =
        backend::build(args.backend, registry.clone()).context("build session backend")?;
    info!(
        backend = ?args.backend,
        "session backend constructed"
    );

    // Run the accept loop with a SIGTERM/SIGINT shutdown.
    let mut sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let mut sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;

    let result = tokio::select! {
        res = server::run(listener, registry.clone(), session_backend) => res,
        _ = sigterm.recv() => {
            info!("SIGTERM → shutting down");
            Ok(())
        }
        _ = sigint.recv() => {
            info!("SIGINT → shutting down");
            Ok(())
        }
    };

    // Best-effort cleanup: kill live sessions and remove the socket.
    registry.shutdown_all().await;
    let _ = std::fs::remove_file(&socket_path);
    result
}
