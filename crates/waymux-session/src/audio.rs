// SPDX-License-Identifier: Apache-2.0

//! Audio socket passthrough.
//!
//! When `--share-audio` is set, the session symlinks the host's PulseAudio
//! and PipeWire sockets into its own runtime dir so apps spawned with
//! `XDG_RUNTIME_DIR=<session_dir>` (set by the daemon in `spawn_child`)
//! auto-discover audio without per-app env hacks.

use std::path::{Path, PathBuf};
use tracing::{info, warn};

pub struct AudioLinks {
    /// Symlinks created at the runtime-dir root (cleaned up with remove_file).
    pub links: Vec<PathBuf>,
    /// Real directories created (e.g. `<session_dir>/pulse`); cleaned up
    /// with remove_dir_all after their inner links are removed.
    pub dirs: Vec<PathBuf>,
}

impl AudioLinks {
    pub fn empty() -> Self {
        Self {
            links: Vec::new(),
            dirs: Vec::new(),
        }
    }
}

pub fn setup(session_dir: &Path) -> AudioLinks {
    let host = host_runtime_dir();
    let mut out = AudioLinks::empty();
    let mut any_present = false;

    // PulseAudio: cannot symlink the whole `pulse/` directory because
    // libpulse's "secure directory" check refuses to follow a symlink
    // there. Instead create a real `<session_dir>/pulse` (mode 0700,
    // matching libpulse's expectation) and symlink only `native` —
    // the only file pulse clients actually need.
    let pulse_src = host.join("pulse");
    let native_src = pulse_src.join("native");
    if native_src.exists() {
        any_present = true;
        let pulse_dir = session_dir.join("pulse");
        match ensure_pulse_dir(&pulse_dir) {
            Ok(created_dir) => {
                if created_dir {
                    out.dirs.push(pulse_dir.clone());
                }
                let dst = pulse_dir.join("native");
                match link_force(&native_src, &dst) {
                    Ok(()) => out.links.push(dst),
                    Err(e) => warn!(
                        error = %e,
                        src = %native_src.display(),
                        "share_audio: pulse native symlink failed",
                    ),
                }
            }
            Err(e) => warn!(
                error = %e,
                path = %pulse_dir.display(),
                "share_audio: could not prepare pulse directory",
            ),
        }
    }

    for name in ["pipewire-0", "pipewire-0-manager"] {
        let src = host.join(name);
        if src.exists() {
            any_present = true;
            let dst = session_dir.join(name);
            match link_force(&src, &dst) {
                Ok(()) => out.links.push(dst),
                Err(e) => warn!(
                    error = %e,
                    src = %src.display(),
                    "share_audio: pipewire symlink failed",
                ),
            }
        }
    }

    if !any_present {
        warn!(
            host = %host.display(),
            "share_audio set but no pulse/pipewire socket on host; \
             apps in this session will have no audio",
        );
    } else {
        info!(
            links = out.links.len(),
            dirs = out.dirs.len(),
            "share_audio: host audio passthrough enabled",
        );
    }

    out
}

pub fn cleanup(audio: &AudioLinks) {
    for p in &audio.links {
        match std::fs::remove_file(p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!(
                error = %e,
                path = %p.display(),
                "share_audio: cleanup remove_file failed",
            ),
        }
    }
    for d in &audio.dirs {
        // remove_dir (not remove_dir_all) — refuse to recurse if anything
        // unexpected is left behind. Stale entries dangling here would
        // be cleaned by the next setup()'s link_force anyway.
        match std::fs::remove_dir(d) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!(
                error = %e,
                path = %d.display(),
                "share_audio: cleanup remove_dir failed",
            ),
        }
    }
}

/// Make sure `<session_dir>/pulse` exists as a real directory with
/// mode 0700. Returns `Ok(true)` if we created it, `Ok(false)` if it
/// already existed in an acceptable shape. Returns an error if the
/// path exists but is something else (e.g. a stray symlink); the
/// caller should warn but not fail the session.
fn ensure_pulse_dir(path: &Path) -> std::io::Result<bool> {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                std::fs::remove_file(path)?;
            } else if meta.is_dir() {
                return Ok(false);
            } else {
                std::fs::remove_file(path)?;
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    std::fs::create_dir(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(true)
}

// Idempotent: removes any pre-existing entry at `dst` so a stale link from
// a SIGKILLed previous run cannot shadow the host socket. Uses
// symlink_metadata so we never follow the existing link.
fn link_force(src: &Path, dst: &Path) -> std::io::Result<()> {
    if let Ok(meta) = std::fs::symlink_metadata(dst) {
        if meta.file_type().is_symlink() || meta.is_file() {
            std::fs::remove_file(dst)?;
        } else {
            std::fs::remove_dir_all(dst)?;
        }
    }
    std::os::unix::fs::symlink(src, dst)
}

fn host_runtime_dir() -> PathBuf {
    // SAFETY: getuid is signal-safe and infallible.
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/run/user/{uid}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_force_is_idempotent() {
        let tmp =
            std::env::temp_dir().join(format!("waymux-audio-link-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("src");
        std::fs::write(&src, b"x").unwrap();
        let dst = tmp.join("dst");
        link_force(&src, &dst).unwrap();
        link_force(&src, &dst).unwrap();
        assert!(std::fs::symlink_metadata(&dst)
            .unwrap()
            .file_type()
            .is_symlink());
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
