// SPDX-License-Identifier: Apache-2.0

//! Generated server bindings for `waymux_attach_v1`.
//!
//! Mirrors the `wayland-protocols` crate's layered scanner pattern:
//! `__interfaces` holds the low-level `Interface` constants,
//! `generate_server_code!` emits the strongly-typed Resource bindings on
//! top of them, and bringing `wayland_server::protocol::*` into scope
//! resolves built-in references (like `wl_surface`) in the generated code.

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
    wayland_scanner::generate_interfaces!("protocols/waymux-attach-v1.xml");
}
use self::__interfaces::*;

wayland_scanner::generate_server_code!("protocols/waymux-attach-v1.xml");
