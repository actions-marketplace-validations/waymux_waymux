// SPDX-License-Identifier: Apache-2.0

//! SHM buffer tracking for screenshot capture.
//!
//! We don't render anything ourselves; instead we remember enough about
//! each client's shm pools and buffers that we can mmap the backing fd and
//! pull out the bytes when asked for a screenshot. The protocol spec's `screenshot`
//! RPC preferentially returns a dmabuf fd, but falls back to a PNG over
//! shm; that is what we do here. Full dmabuf compositing lands with the GPU path.

use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex};

use memmap2::{Mmap, MmapMut, MmapOptions};
use wayland_server::protocol::wl_shm::Format;
use wayland_server::WEnum;

/// User-data for a `wl_shm_pool`. Holds the mmap of the client's fd so we
/// can read the bytes underlying any buffer carved out of this pool.
pub struct ShmPoolData {
    /// The client-provided file descriptor (we dup into our own OwnedFd).
    /// Kept alive for the lifetime of the pool so the mmap stays valid.
    #[allow(dead_code)]
    fd: OwnedFd,
    mmap: Mutex<Option<Mmap>>,
    size: Mutex<usize>,
}

impl ShmPoolData {
    pub fn new(fd: OwnedFd, size: i32) -> Arc<Self> {
        let size_usize = size.max(0) as usize;
        let mmap = if size_usize > 0 {
            unsafe { MmapOptions::new().len(size_usize).map(&fd) }.ok()
        } else {
            None
        };
        Arc::new(Self {
            fd,
            mmap: Mutex::new(mmap),
            size: Mutex::new(size_usize),
        })
    }

    pub fn resize(&self, new_size: i32) {
        let new_size = new_size.max(0) as usize;
        let mut size = self.size.lock().unwrap();
        if new_size > *size {
            *size = new_size;
            let new_mmap = unsafe { MmapOptions::new().len(new_size).map(&self.fd) }.ok();
            *self.mmap.lock().unwrap() = new_mmap;
        }
    }

    /// Borrow the bytes at [offset, offset+len) without copying them.
    /// The closure receives a `&[u8]` slice valid for its duration.
    /// Returns None if the range is out of bounds or the mapping failed.
    pub fn with_bytes<R>(&self, offset: i32, len: usize, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let offset = usize::try_from(offset).ok()?;
        let mmap_guard = self.mmap.lock().unwrap();
        let m = mmap_guard.as_ref()?;
        let end = offset.checked_add(len)?;
        if end > m.len() {
            return None;
        }
        Some(f(&m[offset..end]))
    }

    /// Copy `src` into the pool starting at `offset`. Used by screencopy
    /// to write the composited desktop into a client-supplied wl_shm
    /// buffer. Re-mmaps the fd as MmapMut for the write so we don't have
    /// to retain a writable mapping in the steady-state read path.
    pub fn write_bytes(&self, offset: i32, src: &[u8]) -> Option<()> {
        let offset = usize::try_from(offset).ok()?;
        let size = *self.size.lock().unwrap();
        let end = offset.checked_add(src.len())?;
        if end > size {
            return None;
        }
        let mut m: MmapMut = unsafe { MmapOptions::new().len(size).map_mut(&self.fd) }.ok()?;
        m[offset..end].copy_from_slice(src);
        Some(())
    }
}

/// User-data for a `wl_buffer` backed by an shm pool.
#[derive(Clone)]
pub struct ShmBufferData {
    pub pool: Arc<ShmPoolData>,
    pub offset: i32,
    pub width: i32,
    pub height: i32,
    pub stride: i32,
    pub format: WEnum<Format>,
}
