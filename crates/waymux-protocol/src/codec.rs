// SPDX-License-Identifier: Apache-2.0

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

/// Hard cap on one frame. Large enough for a window list of 1024 entries,
/// small enough to keep a misbehaving peer from exhausting memory.
pub const MAX_FRAME_SIZE: usize = 20 << 20; // 20 MiB — large enough for full-desktop PNG screenshots

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("msgpack encode: {0}")]
    Msgpack(#[from] rmp_serde::encode::Error),
    #[error("frame exceeds {MAX_FRAME_SIZE} bytes ({actual})")]
    TooLarge { actual: usize },
}

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("msgpack decode: {0}")]
    Msgpack(#[from] rmp_serde::decode::Error),
    #[error("frame length {0} exceeds {MAX_FRAME_SIZE}")]
    TooLarge(usize),
    #[error("frame truncated: need {need} bytes, got {have}")]
    Truncated { need: usize, have: usize },
}

/// Append a length-prefixed msgpack frame to `out`.
pub fn encode_frame<T: Serialize>(value: &T, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    let frame_start = out.len();
    out.extend_from_slice(&[0u8; 4]);
    let payload_start = out.len();
    rmp_serde::encode::write_named(out, value).inspect_err(|_| {
        out.truncate(frame_start);
    })?;
    let len = out.len() - payload_start;
    if len > MAX_FRAME_SIZE {
        out.truncate(frame_start);
        return Err(EncodeError::TooLarge { actual: len });
    }
    out[frame_start..frame_start + 4].copy_from_slice(&(len as u32).to_be_bytes());
    Ok(())
}

/// Decode exactly one framed msgpack value from `bytes` (prefix + payload).
pub fn decode_frame<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, DecodeError> {
    if bytes.len() < 4 {
        return Err(DecodeError::Truncated {
            need: 4,
            have: bytes.len(),
        });
    }
    let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if len > MAX_FRAME_SIZE {
        return Err(DecodeError::TooLarge(len));
    }
    if bytes.len() != 4 + len {
        return Err(DecodeError::Truncated {
            need: 4 + len,
            have: bytes.len(),
        });
    }
    Ok(rmp_serde::from_slice(&bytes[4..])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Msg {
        id: u32,
        name: String,
    }

    #[test]
    fn roundtrip() {
        let m = Msg {
            id: 42,
            name: "hello".into(),
        };
        let mut buf = Vec::new();
        encode_frame(&m, &mut buf).unwrap();
        assert_eq!(
            buf.len(),
            4 + u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize
        );
        let decoded: Msg = decode_frame(&buf).unwrap();
        assert_eq!(m, decoded);
    }

    #[test]
    fn truncated_header() {
        let err: Result<Msg, _> = decode_frame(&[0u8; 3]);
        assert!(matches!(err, Err(DecodeError::Truncated { .. })));
    }

    #[test]
    fn length_mismatch() {
        let m = Msg {
            id: 1,
            name: "x".into(),
        };
        let mut buf = Vec::new();
        encode_frame(&m, &mut buf).unwrap();
        buf.pop();
        let err: Result<Msg, _> = decode_frame(&buf);
        assert!(matches!(err, Err(DecodeError::Truncated { .. })));
    }
}
