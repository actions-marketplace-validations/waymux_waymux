// SPDX-License-Identifier: Apache-2.0

//! `waymux-cli` library surface — exposes pieces the binary glues together so
//! integration tests can exercise them in isolation.
//!
//! End users should depend on the `waymux` binary, not this lib.

pub mod connection;
pub mod credentials;
pub mod daemon;
pub mod transport;
pub mod viewer_token;

pub use connection::Connection;
