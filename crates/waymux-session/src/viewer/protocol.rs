// SPDX-License-Identifier: Apache-2.0

//! Wire format between waymux-session (Rust) and waymux-neko-bridge (Go).
//!
//! Layout (little-endian):
//!     ┌──────────┬──────────┬──────────────────┐
//!     │ tag: u8  │ len: u32 │ payload: [u8;len]│
//!     └──────────┴──────────┴──────────────────┘
//!
//! Tags:
//!     0x01  Nalu(bytes)        session → bridge   raw Annex-B NALU
//!     0x02  PtsHint(u64)       session → bridge   frame PTS ms (optional)
//!     0x03  CursorImage        session → bridge   cursor bitmap (RGBA) + hotspot
//!     0x04  CursorPos          session → bridge   cursor position + sequence
//!     0x10  ForceKeyframe      bridge → session   peer dropped pkts, send IDR
//!     0x11  InjectOp(json)     bridge → session   browser input event
//!     0xF0  Shutdown(reason)   bridge → session   bridge exiting; cleanup
//!
//! All multi-byte integers are little-endian. Payload length is capped at
//! 1 MiB on the read side to bound per-frame allocation.

// protocol enum has reserved variants for the session ↔ bridge channel.
#![allow(dead_code)]

use std::io::{self, Read, Write};

pub const TAG_NALU: u8 = 0x01;
pub const TAG_PTS: u8 = 0x02;
pub const TAG_FORCE_KEYFRAME: u8 = 0x10;
pub const TAG_INJECT_OP: u8 = 0x11;
pub const TAG_SET_BITRATE: u8 = 0x12;
pub const TAG_CURSOR_IMAGE: u8 = 0x03;
pub const TAG_CURSOR_POS: u8 = 0x04;
pub const TAG_SHUTDOWN: u8 = 0xF0;

pub const MAX_PAYLOAD_BYTES: usize = 1024 * 1024; // 1 MiB

#[derive(Debug, PartialEq)]
pub enum Frame<'a> {
    Nalu(&'a [u8]),
    PtsHint(u64),
    ForceKeyframe,
    InjectOp(&'a [u8]),
    SetBitrate(u32),
    Shutdown(u8),
    CursorImage {
        w: u16,
        h: u16,
        hot_x: u16,
        hot_y: u16,
        rgba: &'a [u8],
    },
    CursorPos {
        x: f32,
        y: f32,
        seq: u32,
    },
}

#[derive(Debug, PartialEq)]
pub enum OwnedFrame {
    Nalu(Vec<u8>),
    PtsHint(u64),
    ForceKeyframe,
    InjectOp(Vec<u8>),
    SetBitrate(u32),
    Shutdown(u8),
    CursorImage {
        w: u16,
        h: u16,
        hot_x: u16,
        hot_y: u16,
        rgba: Vec<u8>,
    },
    CursorPos {
        x: f32,
        y: f32,
        seq: u32,
    },
}

pub fn write_frame<W: Write>(w: &mut W, frame: Frame<'_>) -> io::Result<()> {
    match frame {
        Frame::Nalu(b) => write_payload(w, TAG_NALU, b),
        Frame::PtsHint(ms) => {
            let bytes = ms.to_le_bytes();
            write_payload(w, TAG_PTS, &bytes)
        }
        Frame::ForceKeyframe => write_payload(w, TAG_FORCE_KEYFRAME, &[]),
        Frame::InjectOp(j) => write_payload(w, TAG_INJECT_OP, j),
        Frame::SetBitrate(bps) => {
            let bytes = bps.to_le_bytes();
            write_payload(w, TAG_SET_BITRATE, &bytes)
        }
        Frame::Shutdown(reason) => write_payload(w, TAG_SHUTDOWN, &[reason]),
        Frame::CursorImage {
            w: cw,
            h: ch,
            hot_x,
            hot_y,
            rgba,
        } => {
            let mut payload = Vec::with_capacity(8 + rgba.len());
            payload.extend_from_slice(&cw.to_le_bytes());
            payload.extend_from_slice(&ch.to_le_bytes());
            payload.extend_from_slice(&hot_x.to_le_bytes());
            payload.extend_from_slice(&hot_y.to_le_bytes());
            payload.extend_from_slice(rgba);
            write_payload(w, TAG_CURSOR_IMAGE, &payload)
        }
        Frame::CursorPos { x, y, seq } => {
            let mut payload = [0u8; 12];
            payload[0..4].copy_from_slice(&x.to_le_bytes());
            payload[4..8].copy_from_slice(&y.to_le_bytes());
            payload[8..12].copy_from_slice(&seq.to_le_bytes());
            write_payload(w, TAG_CURSOR_POS, &payload)
        }
    }
}

fn write_payload<W: Write>(w: &mut W, tag: u8, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "payload exceeds u32"))?;
    w.write_all(&[tag])?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(payload)?;
    Ok(())
}

pub fn read_frame<R: Read>(r: &mut R) -> io::Result<OwnedFrame> {
    let mut header = [0u8; 5];
    r.read_exact(&mut header)?;
    let tag = header[0];
    let len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    if len > MAX_PAYLOAD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload {len} exceeds {MAX_PAYLOAD_BYTES}"),
        ));
    }
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    match tag {
        TAG_NALU => Ok(OwnedFrame::Nalu(payload)),
        TAG_PTS => {
            if len != 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "PtsHint must be 8 bytes",
                ));
            }
            let mut b = [0u8; 8];
            b.copy_from_slice(&payload);
            Ok(OwnedFrame::PtsHint(u64::from_le_bytes(b)))
        }
        TAG_FORCE_KEYFRAME => Ok(OwnedFrame::ForceKeyframe),
        TAG_INJECT_OP => Ok(OwnedFrame::InjectOp(payload)),
        TAG_SET_BITRATE => {
            if len != 4 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "SetBitrate must be 4 bytes",
                ));
            }
            let mut b = [0u8; 4];
            b.copy_from_slice(&payload);
            Ok(OwnedFrame::SetBitrate(u32::from_le_bytes(b)))
        }
        TAG_SHUTDOWN => {
            if len != 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Shutdown must be 1 byte",
                ));
            }
            Ok(OwnedFrame::Shutdown(payload[0]))
        }
        TAG_CURSOR_IMAGE => {
            if len < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "CursorImage header too short",
                ));
            }
            let w = u16::from_le_bytes([payload[0], payload[1]]);
            let h = u16::from_le_bytes([payload[2], payload[3]]);
            let hot_x = u16::from_le_bytes([payload[4], payload[5]]);
            let hot_y = u16::from_le_bytes([payload[6], payload[7]]);
            let expected = 8 + (w as usize) * (h as usize) * 4;
            if len != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("CursorImage len {len} != expected {expected} for {w}x{h}"),
                ));
            }
            Ok(OwnedFrame::CursorImage {
                w,
                h,
                hot_x,
                hot_y,
                rgba: payload[8..].to_vec(),
            })
        }
        TAG_CURSOR_POS => {
            if len != 12 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "CursorPos must be 12 bytes",
                ));
            }
            let x = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
            let y = f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
            let seq = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
            Ok(OwnedFrame::CursorPos { x, y, seq })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown tag 0x{tag:02X}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nalu_frame_roundtrips() {
        let nalu = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xC0, 0x1F];
        let mut buf = Vec::new();
        write_frame(&mut buf, Frame::Nalu(&nalu)).unwrap();
        let frame = read_frame(&mut &buf[..]).unwrap();
        match frame {
            OwnedFrame::Nalu(b) => assert_eq!(b, nalu),
            _ => panic!("expected Nalu"),
        }
    }

    #[test]
    fn force_keyframe_frame_has_empty_payload() {
        let mut buf = Vec::new();
        write_frame(&mut buf, Frame::ForceKeyframe).unwrap();
        // tag=0x10 + len=0
        assert_eq!(buf, vec![0x10, 0, 0, 0, 0]);
        let frame = read_frame(&mut &buf[..]).unwrap();
        assert!(matches!(frame, OwnedFrame::ForceKeyframe));
    }

    #[test]
    fn inject_op_frame_carries_json() {
        let json = br#"{"kind":"key","params":{"keycode":30,"state":"pressed","modifiers":0}}"#;
        let mut buf = Vec::new();
        write_frame(&mut buf, Frame::InjectOp(json)).unwrap();
        let frame = read_frame(&mut &buf[..]).unwrap();
        match frame {
            OwnedFrame::InjectOp(j) => assert_eq!(j, json),
            _ => panic!("expected InjectOp"),
        }
    }

    #[test]
    fn unknown_tag_returns_error() {
        let buf = [0xFF, 0, 0, 0, 0];
        assert!(read_frame(&mut &buf[..]).is_err());
    }

    #[test]
    fn truncated_header_returns_error() {
        let buf = [0x01]; // only the tag, no length
        assert!(read_frame(&mut &buf[..]).is_err());
    }

    #[test]
    fn pts_hint_frame_roundtrips() {
        let mut buf = Vec::new();
        write_frame(&mut buf, Frame::PtsHint(123_456_789)).unwrap();
        let frame = read_frame(&mut &buf[..]).unwrap();
        assert!(matches!(frame, OwnedFrame::PtsHint(123_456_789)));
    }

    #[test]
    fn set_bitrate_frame_roundtrips() {
        let mut buf = Vec::new();
        write_frame(&mut buf, Frame::SetBitrate(7_000_000)).unwrap();
        // tag=0x12 + len=4 + bps LE (7_000_000 = 0x006ACFC0)
        assert_eq!(buf, vec![0x12, 4, 0, 0, 0, 0xC0, 0xCF, 0x6A, 0x00]);
        let frame = read_frame(&mut &buf[..]).unwrap();
        assert!(matches!(frame, OwnedFrame::SetBitrate(7_000_000)));
    }

    #[test]
    fn shutdown_frame_roundtrips() {
        let mut buf = Vec::new();
        write_frame(&mut buf, Frame::Shutdown(7)).unwrap();
        let frame = read_frame(&mut &buf[..]).unwrap();
        assert!(matches!(frame, OwnedFrame::Shutdown(7)));
    }

    #[test]
    fn cursor_image_frame_roundtrips() {
        // 2x1 RGBA, hotspot (1,0)
        let rgba = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut buf = Vec::new();
        write_frame(
            &mut buf,
            Frame::CursorImage {
                w: 2,
                h: 1,
                hot_x: 1,
                hot_y: 0,
                rgba: &rgba,
            },
        )
        .unwrap();
        let frame = read_frame(&mut &buf[..]).unwrap();
        match frame {
            OwnedFrame::CursorImage {
                w,
                h,
                hot_x,
                hot_y,
                rgba: got,
            } => {
                assert_eq!((w, h, hot_x, hot_y), (2, 1, 1, 0));
                assert_eq!(got, rgba);
            }
            _ => panic!("expected CursorImage"),
        }
    }

    #[test]
    fn cursor_image_hide_is_zero_dims() {
        let mut buf = Vec::new();
        write_frame(
            &mut buf,
            Frame::CursorImage {
                w: 0,
                h: 0,
                hot_x: 0,
                hot_y: 0,
                rgba: &[],
            },
        )
        .unwrap();
        let frame = read_frame(&mut &buf[..]).unwrap();
        assert!(matches!(frame, OwnedFrame::CursorImage { w: 0, h: 0, .. }));
    }

    #[test]
    fn cursor_image_rejects_mismatched_len() {
        // header claims 2x2 (16 bytes rgba) but payload is short
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&[0u8; 4]); // only 4 rgba bytes, need 16
        let mut buf = vec![TAG_CURSOR_IMAGE];
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(&payload);
        assert!(read_frame(&mut &buf[..]).is_err());
    }

    #[test]
    fn cursor_pos_frame_roundtrips() {
        let mut buf = Vec::new();
        write_frame(
            &mut buf,
            Frame::CursorPos {
                x: 960.5,
                y: 540.0,
                seq: 42,
            },
        )
        .unwrap();
        let frame = read_frame(&mut &buf[..]).unwrap();
        match frame {
            OwnedFrame::CursorPos { x, y, seq } => {
                assert_eq!((x, y, seq), (960.5, 540.0, 42));
            }
            _ => panic!("expected CursorPos"),
        }
    }
}
