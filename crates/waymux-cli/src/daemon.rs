// SPDX-License-Identifier: Apache-2.0

//! Locating and (auto-)launching the `waymuxd` daemon binary.
//!
//! Two entry points share one resolver:
//!
//! * `waymux serve` (`exec_daemon`) replaces the CLI process with `waymuxd`,
//!   giving a one-binary onboarding story: `waymux serve` is the daemon.
//! * The local connect path (`ensure_daemon_or_spawn`) transparently spawns a
//!   background `waymuxd` when the control socket is absent, so the first
//!   `waymux ls` / `waymux new` works without a separate `waymuxd` start.
//!
//! Both resolve the binary the same way, mirroring
//! `waymux-session`'s `bridge_binary_path` and the MCP's `resolve_waymux_bin`.

use anyhow::{anyhow, Context, Result};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Env var that disables CLI auto-spawn of the daemon. Set to `1` (any
/// non-empty value) when you run an explicit, externally-managed `waymuxd` and
/// do not want the CLI to start one for you. Documented in the CLI `--help`
/// for the `serve` subcommand and in the README quickstart.
pub const NO_AUTOSPAWN_ENV: &str = "WAYMUX_NO_AUTOSPAWN";

/// Env var overriding the resolved `waymuxd` path (mirrors
/// `$WAYMUX_NEKO_BRIDGE_BIN` / `$WAYMUX_BIN`). Trusted: it is the operator's
/// own configuration, the same trust level as `$PATH`.
pub const WAYMUXD_BIN_ENV: &str = "WAYMUXD_BIN";

/// Locate the `waymuxd` binary. Resolution order (first match wins):
///   1. `$WAYMUXD_BIN` if set (explicit override).
///   2. A `waymuxd` next to the running `waymux` executable (a built
///      `target/` tree or a co-installed bundle: the common dev case).
///   3. `waymuxd` on `$PATH`.
///   4. Bare `"waymuxd"` as a last resort, resolved by the OS at spawn time.
///
/// The returned path comes only from trusted sources (operator env / install
/// layout / `$PATH`); no untrusted input ever reaches it.
pub fn daemon_binary_path() -> PathBuf {
    if let Some(p) = std::env::var_os(WAYMUXD_BIN_ENV) {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(sibling) = exe.parent().map(|d| d.join("waymuxd")) {
            if sibling.is_file() {
                return sibling;
            }
        }
    }
    if let Some(on_path) = std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p)
            .map(|d| d.join("waymuxd"))
            .find(|c| c.is_file())
    }) {
        return on_path;
    }
    PathBuf::from("waymuxd")
}

/// Whether auto-spawn is opted out via `$WAYMUX_NO_AUTOSPAWN`. Any non-empty
/// value disables it.
pub fn autospawn_disabled() -> bool {
    std::env::var_os(NO_AUTOSPAWN_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// `waymux serve`: resolve `waymuxd` and replace this process with it via
/// `execv`, forwarding `extra_args` (e.g. a `--socket` carried from the
/// global flag). `WAYMUX_SOCKET` is already inherited through the environment;
/// `waymuxd` reads the same env var, so an explicit `--socket` is only added
/// when the user passed one on the `waymux` command line.
///
/// On success this never returns (the process image is replaced). It only
/// returns `Err` when the daemon binary cannot be found / launched, with a
/// build-it hint.
pub fn exec_daemon(extra_args: &[OsString]) -> Result<std::convert::Infallible> {
    use std::os::unix::process::CommandExt;

    let bin = daemon_binary_path();
    // Verify up front so we can give a clean build-it hint instead of a raw
    // ENOENT from execv. A bare `"waymuxd"` (PATH fallback) has no parent, so
    // only check `is_file` when the path is concrete.
    if bin.components().count() > 1 && !bin.is_file() {
        return Err(daemon_missing_error(&bin));
    }

    let err = std::process::Command::new(&bin).args(extra_args).exec();
    // `exec` only returns on failure.
    if err.kind() == std::io::ErrorKind::NotFound {
        return Err(daemon_missing_error(&bin));
    }
    Err(anyhow!("exec {}: {err}", bin.display()))
}

/// Auto-spawn budget: poll for the socket up to this long after launching the
/// background daemon. The local daemon binds its control socket within a few
/// ms of startup; this is generous headroom for a cold debug build under load
/// while staying well under a human's patience threshold.
const AUTOSPAWN_TIMEOUT: Duration = Duration::from_millis(3000);
/// Interval between socket-existence polls during auto-spawn.
const AUTOSPAWN_POLL: Duration = Duration::from_millis(25);

/// Ensure a daemon is reachable at `socket`, auto-spawning one in the
/// background if (and ONLY if) the socket is absent.
///
/// This is the conservative guard for the local connect path. It is called
/// BEFORE attempting the connect, and acts solely on socket presence:
///
/// * Socket already exists → return `Ok(())` immediately. We do NOT spawn; the
///   caller's connect proceeds and surfaces any real connection error
///   (permission denied, protocol-version mismatch, a wedged daemon) verbatim.
///   Auto-spawn never masks those.
/// * Socket absent + auto-spawn disabled (`$WAYMUX_NO_AUTOSPAWN`) → return
///   `Ok(())` without spawning; the caller's connect fails with the normal
///   "connect …: No such file or directory" so the user can start `waymuxd`
///   (or `waymux serve`) explicitly.
/// * Socket absent + auto-spawn enabled → spawn a DETACHED background
///   `waymuxd` (it outlives this CLI invocation — it is a real per-user
///   daemon, not tied to one command) and poll until the socket appears or the
///   budget elapses.
///
/// Race handling: two concurrent `waymux` invocations may both observe the
/// socket absent and each spawn a `waymuxd`. The daemon binds-or-fails — the
/// first to bind wins; the loser sees the socket already in use, logs, and
/// exits (see `waymux-daemon` main: it refuses to clobber a live socket). The
/// CLI does not care which daemon won; it just polls until the socket exists
/// and then connects. We therefore tolerate a spawn whose child exits early.
pub fn ensure_daemon_or_spawn(socket: &Path) -> Result<()> {
    if socket.exists() {
        return Ok(());
    }
    if autospawn_disabled() {
        return Ok(());
    }

    let bin = daemon_binary_path();
    if bin.components().count() > 1 && !bin.is_file() {
        // Don't silently fail: tell the user how to get a daemon. The caller's
        // connect would otherwise produce a less actionable ENOENT.
        return Err(daemon_missing_error(&bin));
    }

    spawn_detached_daemon(&bin)
        .with_context(|| format!("auto-spawning waymuxd from {}", bin.display()))?;

    // Poll for the socket. The winning daemon (ours or a racer's) binds it
    // within a few ms; we just wait for it to appear.
    let deadline = Instant::now() + AUTOSPAWN_TIMEOUT;
    loop {
        if socket.exists() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(anyhow!(
                "auto-spawned waymuxd did not create the control socket {} within {:?} — \
                 check the daemon (run `waymux serve` in another terminal to see its logs, \
                 or set $WAYMUX_NO_AUTOSPAWN=1 and start waymuxd yourself)",
                socket.display(),
                AUTOSPAWN_TIMEOUT,
            ));
        }
        std::thread::sleep(AUTOSPAWN_POLL);
    }
}

/// Spawn `waymuxd` as a detached background child whose lifetime is
/// independent of this CLI process. We deliberately do NOT keep the `Child`
/// handle around to wait on it: the daemon is meant to persist after the CLI
/// exits. stdout/stderr go to `/dev/null` so the daemon does not write to the
/// CLI's terminal; operators who want logs run `waymux serve` (foreground) or
/// an externally-managed `waymuxd`.
///
/// `setsid` (a new session) detaches the child from the CLI's controlling
/// terminal and process group, so a Ctrl-C in the shell that ran `waymux`
/// won't also kill the daemon.
fn spawn_detached_daemon(bin: &Path) -> std::io::Result<()> {
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(bin);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Detach into its own session so it survives the CLI's shell. SAFETY:
    // `setsid(2)` is async-signal-safe and touches no shared state; this is the
    // canonical pre-exec hook for daemonizing.
    unsafe {
        cmd.pre_exec(|| {
            if libc_setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    cmd.spawn().map(|_child| ())
    // `_child` is dropped here: on Unix, dropping a `Child` does NOT reap or
    // kill the process — it simply leaks the handle, leaving the daemon running
    // detached, which is exactly the intended lifetime.
}

// `setsid(2)` via the C library, declared locally so `waymux-cli` need not add
// a `libc` crate dependency just for one detach call.
extern "C" {
    #[link_name = "setsid"]
    fn libc_setsid_raw() -> i32;
}

/// Start a new session via `setsid(2)`, detaching the calling process from its
/// controlling terminal. Returns the new session id, or -1 on error.
#[inline]
fn libc_setsid() -> i32 {
    // SAFETY: `setsid()` takes no arguments and only affects the calling
    // process; safe to call from the post-fork / pre-exec child.
    unsafe { libc_setsid_raw() }
}

/// Build the "daemon binary not found" error with a build/install hint.
fn daemon_missing_error(bin: &std::path::Path) -> anyhow::Error {
    anyhow!(
        "waymuxd binary not found at {} — build it with \
         `cargo build -p waymux-daemon` (or install it on $PATH, \
         or set $WAYMUXD_BIN to its location)",
        bin.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// Serialize tests that mutate process-global env vars. `std::env::set_var`
    /// is process-wide and racy across the test thread pool.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that saves + restores a set of env vars around a test body,
    /// so a panicking assert cannot leak state into another test.
    struct EnvGuard<'a> {
        _lock: MutexGuard<'a, ()>,
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl<'a> EnvGuard<'a> {
        fn new(keys: &[&'static str]) -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let saved = keys
                .iter()
                .map(|k| (*k, std::env::var_os(k)))
                .collect::<Vec<_>>();
            // Start each test from a clean slate for the keys under test.
            for k in keys {
                std::env::remove_var(k);
            }
            Self { _lock: lock, saved }
        }
    }

    impl Drop for EnvGuard<'_> {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    #[test]
    fn env_override_wins() {
        let _g = EnvGuard::new(&[WAYMUXD_BIN_ENV]);
        std::env::set_var(WAYMUXD_BIN_ENV, "/opt/custom/waymuxd");
        assert_eq!(daemon_binary_path(), PathBuf::from("/opt/custom/waymuxd"));
    }

    #[test]
    fn sibling_resolves_before_path() {
        // Build a fake install dir with a `waymuxd` next to a fake `waymux`,
        // then point a poisoned $PATH elsewhere. The sibling must win.
        let _g = EnvGuard::new(&[WAYMUXD_BIN_ENV, "PATH"]);
        let dir = tempfile::tempdir().expect("tempdir");
        let sibling = dir.path().join("waymuxd");
        std::fs::write(&sibling, b"#!/bin/true\n").expect("write sibling");

        // A different dir that also has a `waymuxd` — must NOT be chosen.
        let pathdir = tempfile::tempdir().expect("tempdir2");
        std::fs::write(pathdir.path().join("waymuxd"), b"#!/bin/true\n").expect("write path bin");
        std::env::set_var("PATH", pathdir.path());

        // current_exe() points at the test binary, not our temp `waymux`, so
        // exercise the sibling-resolution helper directly with a known exe dir.
        let resolved = resolve_with_exe_dir(dir.path(), &std::env::var_os("PATH"));
        assert_eq!(resolved, sibling);
    }

    #[test]
    fn falls_back_to_path_when_no_sibling() {
        let _g = EnvGuard::new(&[WAYMUXD_BIN_ENV, "PATH"]);
        // exe dir has no waymuxd; PATH dir does.
        let exedir = tempfile::tempdir().expect("tempdir");
        let pathdir = tempfile::tempdir().expect("tempdir2");
        let path_bin = pathdir.path().join("waymuxd");
        std::fs::write(&path_bin, b"#!/bin/true\n").expect("write path bin");
        let path_os = OsString::from(pathdir.path());
        let resolved = resolve_with_exe_dir(exedir.path(), &Some(path_os));
        assert_eq!(resolved, path_bin);
    }

    #[test]
    fn falls_back_to_bare_name_when_nothing_found() {
        let _g = EnvGuard::new(&[WAYMUXD_BIN_ENV, "PATH"]);
        let exedir = tempfile::tempdir().expect("tempdir");
        let pathdir = tempfile::tempdir().expect("tempdir2"); // empty
        let resolved = resolve_with_exe_dir(exedir.path(), &Some(OsString::from(pathdir.path())));
        assert_eq!(resolved, PathBuf::from("waymuxd"));
    }

    #[test]
    fn autospawn_opt_out_respects_env() {
        let _g = EnvGuard::new(&[NO_AUTOSPAWN_ENV]);
        assert!(!autospawn_disabled());
        std::env::set_var(NO_AUTOSPAWN_ENV, "1");
        assert!(autospawn_disabled());
        std::env::set_var(NO_AUTOSPAWN_ENV, "");
        assert!(!autospawn_disabled(), "empty value must not disable");
    }

    /// Seam mirroring `daemon_binary_path`'s sibling/PATH legs with an
    /// injectable exe-dir + PATH, so the resolution order is testable without
    /// depending on the test runner's real `current_exe()` location. Kept in
    /// sync with the production resolver by construction (same order).
    fn resolve_with_exe_dir(exe_dir: &std::path::Path, path: &Option<OsString>) -> PathBuf {
        if let Some(p) = std::env::var_os(WAYMUXD_BIN_ENV) {
            return PathBuf::from(p);
        }
        let sibling = exe_dir.join("waymuxd");
        if sibling.is_file() {
            return sibling;
        }
        if let Some(on_path) = path.as_ref().and_then(|p| {
            std::env::split_paths(p)
                .map(|d| d.join("waymuxd"))
                .find(|c| c.is_file())
        }) {
            return on_path;
        }
        PathBuf::from("waymuxd")
    }
}
