// SPDX-License-Identifier: Apache-2.0

//! Wire types and framing for the waymux control protocol.
//!
//! Length-prefixed msgpack frames over a unix socket:
//!
//! ```text
//! [u32 big-endian frame length][msgpack bytes]
//! ```
//!
//! This crate is pure data + framing; it performs no I/O. Callers wrap it
//! with their own transports (tokio for the daemon, a sync socket for tests,
//! a pure-Python implementation for the SDK).

mod codec;
mod messages;

pub use codec::{decode_frame, encode_frame, DecodeError, EncodeError, MAX_FRAME_SIZE};
pub use messages::*;
