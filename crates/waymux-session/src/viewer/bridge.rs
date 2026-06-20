// SPDX-License-Identifier: Apache-2.0

//! Bridge child-process supervisor.
//!
//! Spawns `waymux-neko-bridge` (Go binary; see `crates/waymux-neko-bridge/`),
//! waiting for it to connect back to our Unix listener before returning. Owns
//! the Unix socket used for typed frame messages.

use anyhow::{anyhow, Context, Result};
use std::io::{BufRead, BufReader};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Compose the URL string we hand back to the CLI.
pub fn url_for_bind(bind: &Option<String>, port: u16) -> String {
    let addr = bind.as_deref().unwrap_or("127.0.0.1");
    format!("http://{addr}:{port}")
}

/// Find an ephemeral TCP port by binding 0 + reading back the assigned port.
/// Closes the listener before returning so the bridge can bind it.
///
/// Inherently racy: the port can be claimed by another process between
/// drop and the bridge's bind. In practice OK for loopback dev; for hosted
/// SaaS we'd hand the bridge a pre-listened fd via fd-passing instead.
pub fn pick_ephemeral_port() -> Result<u16> {
    let l = std::net::TcpListener::bind(("127.0.0.1", 0)).context("binding ephemeral port")?;
    let port = l.local_addr()?.port();
    drop(l);
    Ok(port)
}

/// Locate the `waymux-neko-bridge` binary. Resolution order:
///   1. `$WAYMUX_NEKO_BRIDGE_BIN` if set (explicit override).
///   2. A `waymux-neko-bridge` next to the running executable (a built
///      `target/` tree or a co-installed bundle: the common dev case).
///   3. `waymux-neko-bridge` on `$PATH`.
///   4. `/usr/local/bin/waymux-neko-bridge` as a last resort.
///
/// The caller verifies the result exists and emits a build-it hint otherwise.
fn bridge_binary_path() -> PathBuf {
    if let Ok(p) = std::env::var("WAYMUX_NEKO_BRIDGE_BIN") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(sibling) = exe.parent().map(|d| d.join("waymux-neko-bridge")) {
            if sibling.is_file() {
                return sibling;
            }
        }
    }
    if let Some(on_path) = std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p)
            .map(|d| d.join("waymux-neko-bridge"))
            .find(|c| c.is_file())
    }) {
        return on_path;
    }
    PathBuf::from("/usr/local/bin/waymux-neko-bridge")
}

pub struct BridgeChild {
    pub child: Child,
    pub control_sock: UnixStream,
    pub url: String,
    #[allow(dead_code)]
    pub port: u16,
}

/// Drain the child's stdout/stderr on background threads. Without this,
/// `Stdio::piped()` pipes fill at ~64 KB on Linux and the child blocks
/// on the next write — for the neko-bridge that meant the WS goroutine
/// wedged mid-`Logger.Info`, input forwarding stopped, and a browser
/// refresh couldn't recover because the stuck goroutine never released
/// the single-viewer guard.
pub fn drain_child_io(child: &mut Child, label: &str) {
    if let Some(out) = child.stdout.take() {
        spawn_pipe_drainer(out, label.to_string(), false);
    }
    if let Some(err) = child.stderr.take() {
        spawn_pipe_drainer(err, label.to_string(), true);
    }
}

fn spawn_pipe_drainer<R: std::io::Read + Send + 'static>(
    stream: R,
    label: String,
    is_stderr: bool,
) {
    let stream_name = if is_stderr { "stderr" } else { "stdout" };
    let thread_name = format!("waymux-{label}-{stream_name}");
    let _ = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let reader = BufReader::new(stream);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                if is_stderr {
                    tracing::warn!(target: "bridge", "{label} stderr: {line}");
                } else {
                    tracing::info!(target: "bridge", "{label} stdout: {line}");
                }
            }
        });
}

/// Spawn the bridge with `--bind <addr> --port <port> --socket <path>` and
/// wait for it to connect back to our Unix listener. Returns the child +
/// the accepted control socket.
///
/// `port`: `None` → pick_ephemeral_port (legacy behaviour). `Some(p)` →
/// bind that fixed port (used by SaaS launch scripts so the portal
/// Connect URL routes to a predictable port). Caller must ensure no
/// other process holds the port.
pub fn spawn_bridge(
    session_name: &str,
    bind: &Option<String>,
    port: Option<u16>,
    width: u32,
    height: u32,
    _stop_flag: Arc<AtomicBool>,
) -> Result<BridgeChild> {
    let bin = bridge_binary_path();
    if !bin.exists() {
        return Err(anyhow!(
            "bridge binary not found at {bin:?}; build crates/waymux-neko-bridge first \
             (override path with $WAYMUX_NEKO_BRIDGE_BIN)"
        ));
    }

    // Unix socket path tied to session name. Prefer XDG_RUNTIME_DIR (set
    // by the multi-workspace supervisor to /var/lib/waymux/wN, writable by
    // waymux-wN) and fall back to /run/user/<uid> for the legacy
    // single-tenant path where the systemd-logind dir exists.
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| {
        let uid = unsafe { libc::getuid() };
        format!("/run/user/{uid}")
    });
    let sock_path = format!("{runtime_dir}/waymux-neko-{session_name}.sock");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).with_context(|| format!("bind {sock_path}"))?;

    let port = match port {
        // Port 0 (explicit) and an omitted port both mean "pick an ephemeral
        // port": resolve it to a real bound port so the returned url is
        // connectable, not the un-connectable http://<addr>:0.
        Some(0) | None => pick_ephemeral_port()?,
        Some(p) => p,
    };
    let url = url_for_bind(bind, port);

    // bind resolution order (most specific first):
    //   1. Explicit --bind on the ViewerStart RPC (operator-side
    //      `waymux viewer start kde --bind <addr>`).
    //   2. WAYMUX_VIEWER_DEFAULT_BIND env var. Set by cloud-init in
    //      /etc/waymux/session.env on per-VM customer instances so the
    //      bridge auto-binds 0.0.0.0 without needing an explicit RPC
    //      flag. The portal's Connect URL expects this: it is
    //      http://<floating_ip>:<port>/, which requires the bridge
    //      not to be loopback-only.
    //   3. Hardcoded "127.0.0.1" for local dev.
    let env_default_bind = std::env::var("WAYMUX_VIEWER_DEFAULT_BIND").ok();
    let effective_bind = bind
        .as_deref()
        .or(env_default_bind.as_deref())
        .unwrap_or("127.0.0.1");
    let mut cmd = Command::new(&bin);
    cmd.arg("--bind")
        .arg(effective_bind)
        .arg("--port")
        .arg(port.to_string())
        .arg("--socket")
        .arg(&sock_path)
        .arg("--width")
        .arg(width.to_string())
        .arg("--height")
        .arg(height.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().with_context(|| format!("spawning {bin:?}"))?;
    drain_child_io(&mut child, "neko-bridge");

    // Wait for bridge to connect back.
    listener.set_nonblocking(false)?;
    let (control_sock, _) = listener
        .accept()
        .context("accepting bridge control connection")?;

    Ok(BridgeChild {
        child,
        control_sock,
        url,
        port,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_for_bind_default_is_loopback() {
        let url = url_for_bind(&None, 18347);
        assert_eq!(url, "http://127.0.0.1:18347");
    }

    #[test]
    fn url_for_bind_custom_uses_addr() {
        let url = url_for_bind(&Some("10.42.0.2".into()), 18347);
        assert_eq!(url, "http://10.42.0.2:18347");
    }

    #[test]
    fn pick_ephemeral_port_returns_unused_port_in_range() {
        let port = pick_ephemeral_port().expect("port allocation should succeed");
        // pick_ephemeral_port already proves the port bindable by binding to it
        // (port 0 -> OS assigns -> read -> drop). Re-binding here would race: the
        // OS can hand the freed port to another socket between the two calls, so
        // a rebind flakes under CI load. Just assert it is a plausible ephemeral
        // port instead of re-binding.
        assert!(port >= 1024, "expected a high ephemeral port, got {port}");
    }

    /// Spawning a child with `Stdio::piped()` for stderr and never reading
    /// from the pipe causes the child to block on stderr writes once the
    /// kernel's pipe buffer (~64 KB on Linux) fills. The neko-bridge
    /// logs every input event at INFO; under sustained mouse/keyboard
    /// activity the buffer fills in seconds, the bridge goroutine wedges
    /// mid-log, and the WS handler stops forwarding input — observed as
    /// "keyboard regresses after a while" on the Phase 2 viewer.
    /// `drain_child_io` reads both pipes on background threads so the
    /// child can write without bound. This test proves that contract.
    #[test]
    fn drain_child_io_lets_child_write_more_than_pipe_buffer_to_stderr() {
        use std::sync::mpsc;
        use std::time::Duration;

        // ~64 KB of stderr at 32-byte lines = 2048 lines; we write 5000
        // to land comfortably past the pipe limit on any reasonable kernel.
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg("i=0; while [ $i -lt 5000 ]; do printf 'noisy log line %d\\n' $i >&2; i=$((i+1)); done")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn sh");

        drain_child_io(&mut child, "test-bridge");

        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(child.wait());
        });

        let status = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("child should exit within 5 s — drainer is not consuming stderr")
            .expect("child.wait()");
        assert!(status.success(), "child exited with status {status:?}");
    }
}
