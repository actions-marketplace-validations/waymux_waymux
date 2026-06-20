// SPDX-License-Identifier: Apache-2.0

//! Generated server bindings for the legacy `wl_drm` protocol (Mesa's
//! original NVIDIA-compatible interface for advertising a DRM device to
//! Wayland clients).
//!
//! Why we ship this: `libnvidia-egl-wayland` v1.1.9 (the version on
//! Ubuntu 22.04 LTS) doesn't yet implement the
//! modern `zwp_linux_dmabuf_v1` feedback path. It looks specifically for
//! `wl_drm` to learn which DRM device the compositor is using. Without
//! `wl_drm` advertised, NVIDIA EGL returns EGL_NO_DISPLAY when a client
//! calls `eglGetDisplay(wayland_display)` — so clients fall back to Mesa
//! llvmpipe (software). v1.1.13+ uses the dmabuf-feedback path which we
//! already implement; v1.1.9 needs `wl_drm`.
//!
//! On bind we send:
//!   - `device(/dev/dri/renderD*)` — the render node path
//!   - `format(...)` for each fourcc we support
//!   - `capabilities(PRIME)` so clients use the modern `create_prime_buffer`
//!     path (fd-based, dmabuf-equivalent) rather than legacy `flink`-name
//!     based buffers we couldn't handle.
//!
//! Requests are mostly no-ops:
//!   - `authenticate` → reply `authenticated` (DRM master auth is irrelevant
//!     for render-node clients; NVIDIA accepts and proceeds).
//!   - `create_buffer` / `create_planar_buffer` → unimplemented (legacy
//!     flink-name path; modern clients don't use these).
//!   - `create_prime_buffer` → could route to dmabuf import; for now,
//!     return an error so clients fall back to `zwp_linux_dmabuf_v1` for
//!     actual buffer creation.

#![allow(
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

use wayland_server;
use wayland_server::protocol::*;

pub mod __interfaces {
    use wayland_server::protocol::__interfaces::*;
    wayland_scanner::generate_interfaces!("protocols/wayland-drm.xml");
}
use self::__interfaces::*;

wayland_scanner::generate_server_code!("protocols/wayland-drm.xml");
