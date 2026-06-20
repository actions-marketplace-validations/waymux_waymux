// SPDX-License-Identifier: Apache-2.0

//! Best-effort per-session disk + fd quotas.
//!
//! Two independent controls layered on top of the cgroup leaf:
//!
//! 1. **Disk quota**: a tmpfs mount with `size=Nm` over the session's
//!    runtime directory. tmpfs is the simplest way to bound disk usage on
//!    a path the kernel already owns; the alternatives (fs-level quotas,
//!    `setrlimit(RLIMIT_FSIZE)`) either need filesystem support or only
//!    cap a single file's size, not the dir aggregate.
//!
//! 2. **fd cap**: `RLIMIT_NOFILE` set on the session subprocess. Inherited
//!    across `fork`/`exec` so it propagates into anything the session
//!    spawns. Capped at the kernel's hard limit, otherwise the call is
//!    rejected.
//!
//! Both controls match `cgroup.rs`'s warn-and-fall-through philosophy: if
//! the daemon doesn't have the privilege the kernel needs (`CAP_SYS_ADMIN`
//! for `mount`, or asking for an `fd_limit` higher than the hard ceiling),
//! the session still starts uncapped and the daemon logs.

use std::ffi::CString;
use std::path::{Path, PathBuf};

/// Handle for a per-session tmpfs mount. `Drop`-cleanup is intentionally
/// absent — destroy paths in `Registry` already do their work in async
/// task scope where panics from a sync `umount2` would be hostile. The
/// caller is expected to invoke `unmount` explicitly during teardown.
#[derive(Debug)]
pub struct SessionTmpfs {
    pub path: PathBuf,
}

impl SessionTmpfs {
    /// Mount a `tmpfs` of `size_mb` MiB over `path`. Returns `None` after
    /// logging a warning if the mount fails (typically EPERM when the
    /// daemon runs unprivileged in `cargo run`). The directory is left in
    /// place either way so the session still gets a runtime dir.
    pub fn try_mount(path: &Path, size_mb: u32) -> Option<Self> {
        if size_mb == 0 {
            return None;
        }
        let target = match CString::new(path.as_os_str().as_encoded_bytes()) {
            Ok(c) => c,
            Err(_) => {
                tracing::warn!(
                    path = %path.display(),
                    "tmpfs mount: path contains NUL byte; skipping"
                );
                return None;
            }
        };
        let source = CString::new("waymux-tmpfs").expect("static no-NUL string");
        let fstype = CString::new("tmpfs").expect("static no-NUL string");
        let options = CString::new(format!("size={}m", size_mb)).expect("size format no-NUL");

        // SAFETY: target/source/fstype/options are valid NUL-terminated
        // CStrings owned for the duration of this call. Flags=0 is the
        // standard "no special semantics" mount; tmpfs is in-kernel and
        // doesn't touch any block device so we can't corrupt anything.
        let rc = unsafe {
            libc::mount(
                source.as_ptr(),
                target.as_ptr(),
                fstype.as_ptr(),
                0,
                options.as_ptr() as *const libc::c_void,
            )
        };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            tracing::warn!(
                path = %path.display(),
                size_mb,
                error = %err,
                "tmpfs mount failed; running uncapped (CAP_SYS_ADMIN required)"
            );
            return None;
        }
        tracing::info!(
            path = %path.display(),
            size_mb,
            "session tmpfs mounted"
        );
        Some(Self {
            path: path.to_path_buf(),
        })
    }

    /// Best-effort lazy unmount. `MNT_DETACH` so the call returns even if
    /// the session's children still hold open descriptors against files
    /// inside; the kernel finishes the unmount once they close. Errors
    /// are logged at debug level — the cleanup path can't recover.
    pub fn unmount(&self) {
        let target = match CString::new(self.path.as_os_str().as_encoded_bytes()) {
            Ok(c) => c,
            Err(_) => return,
        };
        // SAFETY: target is a valid CString. MNT_DETACH won't fail on a
        // mount we own; failure means the kernel never had this mounted
        // (e.g. mount syscall was a no-op due to EPERM upstream).
        let rc = unsafe { libc::umount2(target.as_ptr(), libc::MNT_DETACH) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            tracing::debug!(
                path = %self.path.display(),
                error = %err,
                "tmpfs umount failed (mount may not have been active)"
            );
        }
    }
}

/// Set `RLIMIT_NOFILE` on the current process to `fd_limit`. Used inside
/// the `pre_exec` hook of the session subprocess `Command` so the cap is
/// applied between `fork` and `exec`. Returns an `io::Result` so the
/// `pre_exec` integration can propagate failures upward — but the registry
/// downgrades the failure to a warning per the surrounding "best effort"
/// contract.
pub fn apply_fd_limit(fd_limit: u32) -> std::io::Result<()> {
    if fd_limit == 0 {
        return Ok(());
    }
    let want = u64::from(fd_limit);

    let mut current = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit takes a writable pointer to a rlimit; we own
    // `current` for the duration. RLIMIT_NOFILE is always defined on Linux.
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut current) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }

    // Don't raise the soft cap above the hard cap — the kernel will EPERM.
    // An unprivileged process can lower the hard cap but not raise it.
    let hard = current.rlim_max;
    let new_soft = want.min(hard);
    let new_hard = hard;

    let new_lim = libc::rlimit {
        rlim_cur: new_soft as libc::rlim_t,
        rlim_max: new_hard,
    };
    // SAFETY: setrlimit takes a read-only pointer to a rlimit we own.
    let rc = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &new_lim) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_mount_zero_is_noop() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(SessionTmpfs::try_mount(dir.path(), 0).is_none());
    }

    #[test]
    fn try_mount_unprivileged_warns_and_returns_none() {
        // In `cargo test` the daemon never has CAP_SYS_ADMIN, so mount(2)
        // returns EPERM. The contract is that the function logs and
        // returns None — never panics. (If the test is somehow run as
        // root, the mount succeeds, and the test still passes by also
        // accepting Some — we then unmount to keep the suite re-runnable.)
        let dir = tempfile::tempdir().expect("tempdir");
        if let Some(handle) = SessionTmpfs::try_mount(dir.path(), 4) {
            handle.unmount()
        }
    }

    #[test]
    fn apply_fd_limit_caps_to_hard_ceiling() {
        // Asking for u32::MAX must not error: we clamp at the hard limit
        // visible to the test process. The post-state must show
        // rlim_cur <= rlim_max.
        apply_fd_limit(u32::MAX).expect("should clamp, not error");
        let mut after = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut after) };
        assert_eq!(rc, 0);
        assert!(after.rlim_cur <= after.rlim_max);
    }

    #[test]
    fn apply_fd_limit_zero_is_noop() {
        // Capture before/after: zero must leave the limit untouched.
        let mut before = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut before) };
        apply_fd_limit(0).unwrap();
        let mut after = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut after) };
        assert_eq!(before.rlim_cur, after.rlim_cur);
        assert_eq!(before.rlim_max, after.rlim_max);
    }

    #[test]
    fn apply_fd_limit_lowers_soft_then_restores_via_hard() {
        // Lower the soft cap to a small value; verify it took effect, then
        // try to raise back to the original hard limit. An unprivileged
        // process can do both as long as we never exceed `rlim_max`.
        let mut original = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut original) };

        apply_fd_limit(256).expect("256 should be well under any sane hard cap");
        let mut lowered = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lowered) };
        assert_eq!(lowered.rlim_cur, 256);

        // Restore so other tests don't trip over a tiny fd budget.
        let restore = libc::rlimit {
            rlim_cur: original.rlim_cur,
            rlim_max: original.rlim_max,
        };
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &restore) };
    }
}
