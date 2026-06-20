// SPDX-License-Identifier: Apache-2.0

//! Keymap the inner compositor sends to every `wl_keyboard`.
//!
//! We previously shipped a hand-written XKB keymap string. Problem: even
//! though it parsed, libxkbcommon's keycode→keysym resolution was
//! unreliable for non-letter keys (Return, BackSpace, digits…) — clients
//! compiled it but got empty symbol lists, so key events were silently
//! dropped.
//!
//! Instead, let libxkbcommon itself compile a keymap from the standard
//! `pc105+us` RMLVO. That produces whatever bytes xkbcommon guarantees
//! round-trip through its own compiler — clients that link against the
//! same library will parse it cleanly. At compositor startup we render
//! the keymap string once and write it to a memfd; every `wl_keyboard`
//! bind sends that memfd.

use std::io::Write;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::{Arc, OnceLock};

use xkbcommon::xkb;

static KEYMAP_FD: OnceLock<(Arc<OwnedFd>, u32)> = OnceLock::new();

/// Return a shared handle to the keymap fd + size (including the NUL
/// terminator libxkbcommon's text-v1 format expects). The same memfd is
/// reused for every client's `wl_keyboard.keymap` event.
pub fn keymap_fd() -> std::io::Result<(Arc<OwnedFd>, u32)> {
    if let Some((fd, size)) = KEYMAP_FD.get() {
        return Ok((fd.clone(), *size));
    }

    let ctx = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
    let keymap = xkb::Keymap::new_from_names(
        &ctx,
        "",      // rules (use default)
        "pc105", // model
        "us",    // layout
        "",      // variant
        None,    // options
        xkb::KEYMAP_COMPILE_NO_FLAGS,
    )
    .ok_or_else(|| std::io::Error::other("libxkbcommon: failed to compile pc105+us keymap"))?;
    let text = keymap.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1);

    let name = c"waymux-keymap";
    let raw = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let owned = unsafe { OwnedFd::from_raw_fd(raw) };
    let mut file: std::fs::File = owned.into();
    file.write_all(text.as_bytes())?;
    file.write_all(&[0])?;
    let size = text.len() as u32 + 1;
    let fd: OwnedFd = file.into();
    let arc = Arc::new(fd);
    let _ = KEYMAP_FD.set((arc.clone(), size));
    let (fd, size) = KEYMAP_FD.get().unwrap();
    Ok((fd.clone(), *size))
}
