// SPDX-License-Identifier: Apache-2.0

//! EBML (Extensible Binary Meta Language) primitives for the MKV muxer.
//!
//! Matroska is a thin layer on top of EBML — every element is `(id,
//! size, payload)`. Both `id` and `size` are encoded as VINTs (variable
//! length integers). This module implements just enough of the EBML
//! spec to emit a streaming H.264 MKV: VINTs, element ID writes,
//! fixed-width integer/float helpers, and a few convenience element
//! writers.
//!
//! Spec references:
//! - <https://datatracker.ietf.org/doc/html/rfc8794> (EBML)
//! - <https://www.matroska.org/technical/elements.html> (Matroska elements)

use std::io::{self, Write};

/// Sentinel size meaning "unknown length — read until the parent
/// element ends or until EOF". Matroska allows this on Segment and
/// Cluster, which is what we use for streaming output.
pub const UNKNOWN_SIZE: u64 = 0x01FF_FFFF_FFFF_FFFF;

/// Write a VINT (variable-length unsigned integer) per the EBML spec.
/// Length marker is the position of the first set bit (MSB-first):
/// 1xxxxxxx          (1 byte,  7 data bits)
/// 01xxxxxx xxxxxxxx (2 bytes, 14 data bits)
/// ... up to 8 bytes / 56 data bits
///
/// VINTs are used both for element IDs (where the marker is part of
/// the wire value — see `write_element_id`) and for element sizes
/// (where the marker is stripped — see `write_element_size`).
pub fn write_vint(w: &mut impl Write, val: u64) -> io::Result<()> {
    let bytes = vint_len(val);
    let mut buf = [0u8; 8];
    let marker = 1u64 << (bytes * 7);
    let combined = val | marker;
    let shifted = combined.to_be_bytes();
    let offset = 8 - bytes as usize;
    buf[..bytes as usize].copy_from_slice(&shifted[offset..]);
    w.write_all(&buf[..bytes as usize])
}

/// Minimum byte count needed to represent `val` as a VINT data payload.
fn vint_len(val: u64) -> u32 {
    if val == UNKNOWN_SIZE {
        return 8;
    }
    for bytes in 1..=8 {
        let max = (1u64 << (bytes * 7)) - 1;
        if val < max {
            return bytes;
        }
    }
    8
}

/// Write a Matroska element ID. The IDs in the spec already include
/// their VINT length marker (e.g. `0x1A45DFA3` is the EBML header — the
/// `0x1` at the top is the 4-byte VINT marker), so we just emit them
/// big-endian using the minimum number of bytes that fit.
pub fn write_element_id(w: &mut impl Write, id: u32) -> io::Result<()> {
    let bytes = if id >= 0x1000_0000 {
        4
    } else if id >= 0x10_0000 {
        3
    } else if id >= 0x4000 {
        2
    } else {
        1
    };
    let raw = id.to_be_bytes();
    let offset = 4 - bytes as usize;
    w.write_all(&raw[offset..])
}

/// Write an EBML element size field. `UNKNOWN_SIZE` emits the all-ones
/// pattern that means "read until end-of-stream or end-of-parent".
pub fn write_element_size(w: &mut impl Write, size: u64) -> io::Result<()> {
    if size == UNKNOWN_SIZE {
        // 0xFF means "1-byte VINT with all data bits set", i.e. 127 —
        // but the spec uses the all-ones pattern at the *chosen length*
        // as the unknown-size sentinel. We use the canonical 8-byte
        // form: 0x01 FF FF FF FF FF FF FF.
        return w.write_all(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    }
    write_vint(w, size)
}

/// Number of bytes a value of `val` will occupy as a Matroska
/// fixed-length unsigned integer (minimum representation).
pub fn uint_byte_len(val: u64) -> u32 {
    if val == 0 {
        return 1;
    }
    let mut n = 0u32;
    let mut v = val;
    while v > 0 {
        n += 1;
        v >>= 8;
    }
    n
}

/// `(id) (size) (val as N-byte big-endian uint)`.
pub fn write_element_u64(w: &mut impl Write, id: u32, val: u64) -> io::Result<()> {
    let len = uint_byte_len(val);
    write_element_id(w, id)?;
    write_element_size(w, len as u64)?;
    let raw = val.to_be_bytes();
    let offset = 8 - len as usize;
    w.write_all(&raw[offset..])
}

/// `(id) (size=8) (val as IEEE-754 double, big-endian)`.
pub fn write_element_f64(w: &mut impl Write, id: u32, val: f64) -> io::Result<()> {
    write_element_id(w, id)?;
    write_element_size(w, 8)?;
    w.write_all(&val.to_be_bytes())
}

/// `(id) (size) (val as UTF-8 string, no NUL terminator)`.
pub fn write_element_str(w: &mut impl Write, id: u32, val: &str) -> io::Result<()> {
    let bytes = val.as_bytes();
    write_element_id(w, id)?;
    write_element_size(w, bytes.len() as u64)?;
    w.write_all(bytes)
}

/// `(id) (size) (val bytes)`. Used for binary fields like CodecPrivate.
pub fn write_element_bytes(w: &mut impl Write, id: u32, data: &[u8]) -> io::Result<()> {
    write_element_id(w, id)?;
    write_element_size(w, data.len() as u64)?;
    w.write_all(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture<F: FnOnce(&mut Vec<u8>) -> io::Result<()>>(f: F) -> Vec<u8> {
        let mut buf = Vec::new();
        f(&mut buf).unwrap();
        buf
    }

    #[test]
    fn vint_one_byte() {
        assert_eq!(capture(|w| write_vint(w, 0)), vec![0x80]);
        assert_eq!(capture(|w| write_vint(w, 1)), vec![0x81]);
        assert_eq!(capture(|w| write_vint(w, 126)), vec![0xFE]);
    }

    #[test]
    fn vint_two_bytes() {
        // 127 needs 2 bytes (1-byte max is 126; 127 would be the unknown sentinel)
        let out = capture(|w| write_vint(w, 127));
        assert_eq!(out, vec![0x40, 0x7F]);
        let out = capture(|w| write_vint(w, 16_382));
        assert_eq!(out, vec![0x7F, 0xFE]);
    }

    #[test]
    fn element_id_ebml_header() {
        // 0x1A45DFA3 is the EBML header element ID (4-byte VINT).
        assert_eq!(
            capture(|w| write_element_id(w, 0x1A45_DFA3)),
            vec![0x1A, 0x45, 0xDF, 0xA3]
        );
    }

    #[test]
    fn element_id_simpleblock() {
        // 0xA3 is SimpleBlock (1-byte VINT).
        assert_eq!(capture(|w| write_element_id(w, 0xA3)), vec![0xA3]);
    }

    #[test]
    fn element_u64_compact() {
        // PixelWidth = 1920 fits in 2 bytes (0x0780). Element ID 0xB0,
        // size VINT 2 -> 0x82, payload 07 80.
        let out = capture(|w| write_element_u64(w, 0xB0, 1920));
        assert_eq!(out, vec![0xB0, 0x82, 0x07, 0x80]);
    }

    #[test]
    fn element_u64_value_zero() {
        // Zero still needs 1 byte.
        let out = capture(|w| write_element_u64(w, 0xB0, 0));
        assert_eq!(out, vec![0xB0, 0x81, 0x00]);
    }

    #[test]
    fn element_str_ebml_docttype() {
        // DocType "matroska" — ID 0x4282, size 8, payload "matroska".
        let out = capture(|w| write_element_str(w, 0x4282, "matroska"));
        assert_eq!(out[0..2], [0x42, 0x82]);
        assert_eq!(out[2], 0x88);
        assert_eq!(&out[3..], b"matroska");
    }

    #[test]
    fn unknown_size_sentinel() {
        let out = capture(|w| write_element_size(w, UNKNOWN_SIZE));
        assert_eq!(out, vec![0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    }
}
