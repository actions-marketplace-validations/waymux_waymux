// SPDX-License-Identifier: Apache-2.0

//! Unified `wl_buffer` user-data. Before dmabuf support we stored
//! `ShmBufferData` directly in the buffer's user data; dmabuf adds a
//! second backing shape that has to travel through the same
//! `Dispatch<WlBuffer, _>` slot — hence this enum.
//!
//! The capture path only needs four things from any buffer: width,
//! height, stride, format, and a byte view. `BufferKind` exposes those
//! uniformly so `composite.rs` doesn't have to care whether the buffer
//! came from `wl_shm` or `zwp_linux_dmabuf_v1`.

use std::sync::Arc;

use wayland_server::protocol::wl_shm;
use wayland_server::WEnum;

use crate::dmabuf::DmabufBufferData;
use crate::shm::ShmBufferData;

pub enum BufferKind {
    Shm(ShmBufferData),
    Dmabuf(Arc<DmabufBufferData>),
    /// 1×1 pixel buffer from wp_single_pixel_buffer_manager_v1.
    /// Bytes are [B, G, R, A] (Wayland ARGB8888 byte order).
    SinglePixel([u8; 4]),
    /// Placeholder used when `zwp_linux_buffer_params_v1.create_immed`
    /// fails validation. The protocol requires us to initialise the
    /// `new_id wl_buffer` regardless; an Invalid variant lets us post a
    /// protocol error while still satisfying wayland-backend's
    /// "every new_id must get data" invariant.
    Invalid,
}

impl BufferKind {
    /// Borrow the buffer bytes without an intermediate allocation. The
    /// closure receives `(bytes, width, height, stride, format)` where
    /// `bytes` is a view directly into the backing mmap. Returns None for
    /// Invalid or unmappable buffers.
    pub fn with_bytes<R>(
        &self,
        f: impl FnOnce(&[u8], i32, i32, i32, WEnum<wl_shm::Format>) -> R,
    ) -> Option<R> {
        match self {
            Self::Shm(s) => {
                let len = (s.height as usize).checked_mul(s.stride as usize)?;
                s.pool.with_bytes(s.offset, len, |bytes| {
                    f(bytes, s.width, s.height, s.stride, s.format)
                })
            }
            Self::Dmabuf(d) => {
                let (w, h, stride, fmt) = (d.width, d.height, d.stride as i32, d.as_shm_format());
                d.with_bytes(|bytes| f(bytes, w, h, stride, fmt))
            }
            Self::SinglePixel(argb) => Some(f(
                argb.as_slice(),
                1,
                1,
                4,
                WEnum::Value(wl_shm::Format::Argb8888),
            )),
            Self::Invalid => None,
        }
    }
}
