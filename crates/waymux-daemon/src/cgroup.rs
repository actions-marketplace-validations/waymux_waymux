// SPDX-License-Identifier: Apache-2.0

//! Best-effort per-session cgroup-v2 memory caps.
//!
//! When a CreateSession request carries `mem_cap_mb`, the daemon tries to
//! place the new session (and every PID it later spawns via `spawn_child`)
//! into a fresh leaf cgroup with `memory.max` set. The cgroup is created
//! as a sibling-leaf under the daemon's own cgroup so we don't have to
//! reshuffle the daemon's PID or wrestle with `cgroup.subtree_control` —
//! whoever set up the daemon's slice (usually systemd) already arranged
//! that.
//!
//! Failure modes are intentionally soft: if cgroup-v2 isn't mounted, if
//! the daemon's cgroup isn't writable, or if the memory controller isn't
//! delegated, we log a warning and the session runs without a cap. The
//! production deployment story (running the daemon as a systemd service
//! under a slice with `Delegate=yes`) makes this Just Work; in `cargo run`
//! development the flag becomes a soft no-op.

use std::path::{Path, PathBuf};

/// A live per-session cgroup. Drop semantics are deliberately empty —
/// `Registry::destroy` calls `cleanup` explicitly after killing children.
#[derive(Debug)]
pub struct SessionCgroup {
    pub path: PathBuf,
}

impl SessionCgroup {
    /// Try to create `<daemon-cgroup>/waymux-<session>/`, set `memory.max`,
    /// and return the handle. Returns `None` (after logging a warning) on
    /// any error so the caller can fall through to "session runs uncapped".
    /// Retained as a thin convenience over `try_create_empty + set_memory_max`
    /// for the original mem-only entry point and unit tests.
    #[allow(dead_code)]
    pub fn try_create(session_name: &str, mem_cap_mb: u32) -> Option<Self> {
        if mem_cap_mb == 0 {
            return None;
        }
        let cg = Self::try_create_empty(session_name)?;
        if !cg.set_memory_max(mem_cap_mb) {
            // Best effort cleanup of the empty leaf we just made.
            let _ = std::fs::remove_dir(&cg.path);
            return None;
        }
        tracing::info!(
            session = session_name,
            path = %cg.path.display(),
            mem_cap_mb,
            "session cgroup created"
        );
        Some(cg)
    }

    /// Create `<daemon-cgroup>/waymux-<session>/` with no caps applied.
    /// Caller is expected to configure controllers via `set_memory_max`,
    /// `set_cpu_max`, etc. Used when the session asks for cpu/disk/fd caps
    /// without `mem_cap_mb` so we still get a leaf to scope subprocesses.
    pub fn try_create_empty(session_name: &str) -> Option<Self> {
        let parent = match daemon_cgroup_dir() {
            Some(p) => p,
            None => {
                tracing::warn!(
                    session = session_name,
                    "session cap requested but cgroup-v2 not detected; running uncapped"
                );
                return None;
            }
        };
        let path = parent.join(format!("waymux-{}", session_name));
        if let Err(e) = std::fs::create_dir_all(&path) {
            tracing::warn!(
                session = session_name,
                path = %path.display(),
                error = %e,
                "cgroup mkdir failed; running uncapped (is the daemon's cgroup writable?)"
            );
            return None;
        }
        Some(Self { path })
    }

    /// Write `memory.max` for this leaf. Returns `true` on success.
    pub fn set_memory_max(&self, mem_cap_mb: u32) -> bool {
        let bytes: u64 = u64::from(mem_cap_mb) * 1024 * 1024;
        match std::fs::write(self.path.join("memory.max"), bytes.to_string()) {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!(
                    path = %self.path.display(),
                    error = %e,
                    "cgroup memory.max write failed; running uncapped (is `memory` in the parent's cgroup.subtree_control?)"
                );
                false
            }
        }
    }

    /// Write a cgroup-v2 `cpu.max` cap. `cpu_cap_pct` is expressed as a
    /// percentage of one CPU core (200 = 2 full cores). The kernel's
    /// `cpu.max` format is `<quota_us> <period_us>`; we keep period at the
    /// 100ms default and scale the quota.
    pub fn set_cpu_max(&self, cpu_cap_pct: u32) {
        if cpu_cap_pct == 0 {
            return;
        }
        let period_us: u64 = 100_000;
        let quota_us = u64::from(cpu_cap_pct) * 1_000;
        let value = format!("{} {}", quota_us, period_us);
        let path = self.path.join("cpu.max");
        if let Err(e) = std::fs::write(&path, &value) {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                cpu_cap_pct,
                "cgroup cpu.max write failed; running uncapped (is `cpu` in the parent's cgroup.subtree_control?)"
            );
        } else {
            tracing::info!(
                path = %path.display(),
                cpu_cap_pct,
                quota_us,
                period_us,
                "cgroup cpu.max set"
            );
        }
    }

    /// Move `pid` into this cgroup. Logged-and-swallowed on error — the
    /// session still functions, just without the cap on this particular
    /// child.
    pub fn add_pid(&self, pid: u32) {
        let procs = self.path.join("cgroup.procs");
        if let Err(e) = std::fs::write(&procs, pid.to_string()) {
            tracing::warn!(
                pid,
                path = %procs.display(),
                error = %e,
                "cgroup add_pid failed"
            );
        }
    }

    /// Try once to rmdir the leaf cgroup. Returns `true` on success or when
    /// the directory is already gone, `false` if the kernel rejected rmdir
    /// (typically EBUSY because PIDs are still mid-exit). Caller is expected
    /// to retry on false; see `Registry::destroy`'s 10× 200ms retry loop.
    pub fn try_remove(&self) -> bool {
        match std::fs::remove_dir(&self.path) {
            Ok(_) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(e) => {
                tracing::debug!(
                    path = %self.path.display(),
                    error = %e,
                    "cgroup try_remove: rmdir failed (PIDs still inside?)"
                );
                false
            }
        }
    }

    /// Atomically SIGKILL every PID in the leaf via the cgroup-v2 `cgroup.kill`
    /// mechanism. Standard since Linux 5.14. Used by `Registry::destroy` to
    /// clean up forked-but-not-tracked subprocesses (e.g. Chromium's GPU /
    /// utility / zygote helpers that survive parent SIGTERM). Fails silently
    /// on older kernels or when the leaf doesn't exist.
    pub fn kill_all(&self) {
        let p = self.path.join("cgroup.kill");
        match std::fs::write(&p, "1") {
            Ok(_) => {
                tracing::debug!(path = %p.display(), "cgroup.kill: SIGKILLed all PIDs in leaf")
            }
            Err(e) => tracing::debug!(
                path = %p.display(), error = %e,
                "cgroup.kill: write failed (older kernel or missing leaf?); falling back to SIGTERM-only"
            ),
        }
    }
}

/// Resolve the daemon's own cgroup-v2 directory by reading `/proc/self/cgroup`.
/// Cgroup-v2 lines are formatted `0::<path>`; v1 lines start with a numeric
/// hierarchy id. If we don't see a v2 line, or `/sys/fs/cgroup` doesn't
/// look like a v2 mount, returns `None`.
fn daemon_cgroup_dir() -> Option<PathBuf> {
    if !is_cgroup_v2_mount(Path::new("/sys/fs/cgroup")) {
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

/// `/sys/fs/cgroup/cgroup.controllers` only exists on cgroup-v2 mounts.
fn is_cgroup_v2_mount(root: &Path) -> bool {
    root.join("cgroup.controllers").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_create_with_zero_cap_returns_none() {
        assert!(SessionCgroup::try_create("zero", 0).is_none());
    }

    #[test]
    fn is_cgroup_v2_mount_false_for_tempdir() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(!is_cgroup_v2_mount(dir.path()));
    }

    #[test]
    fn set_cpu_max_writes_quota_period_format() {
        // Point the cgroup at a tempdir we control so we can read back the
        // file the kernel would normally consume. Format must be
        // `<quota_us> <period_us>` per cgroup-v2 docs; `cpu_cap_pct=200`
        // → quota 200_000us, period 100_000us (= 2 full cores).
        let dir = tempfile::tempdir().expect("tempdir");
        let cg = SessionCgroup {
            path: dir.path().to_path_buf(),
        };
        cg.set_cpu_max(200);
        let written = std::fs::read_to_string(dir.path().join("cpu.max"))
            .expect("cpu.max should have been written");
        assert_eq!(written, "200000 100000");
    }

    #[test]
    fn set_cpu_max_zero_is_noop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cg = SessionCgroup {
            path: dir.path().to_path_buf(),
        };
        cg.set_cpu_max(0);
        assert!(
            !dir.path().join("cpu.max").exists(),
            "zero cap must not touch the file"
        );
    }

    #[test]
    fn set_cpu_max_swallows_write_errors() {
        // Path doesn't exist; the write must fail-soft (warn only).
        let cg = SessionCgroup {
            path: PathBuf::from("/nonexistent/waymux-test-cpu-max"),
        };
        cg.set_cpu_max(100);
    }

    #[test]
    fn kill_all_is_silent_when_path_missing() {
        // Audit task #75: `kill_all` must fail gracefully when the cgroup
        // leaf doesn't exist (older kernel, or cgroup setup silently
        // failed). The write to `<missing>/cgroup.kill` returns ENOENT and
        // we should swallow it without panicking so destroy() keeps moving.
        let cg = SessionCgroup {
            path: PathBuf::from("/nonexistent/waymux-test-kill-all-missing"),
        };
        cg.kill_all();
    }
}
