// SPDX-License-Identifier: Apache-2.0

//! `zwp_linux_dmabuf_v1` server-side state.
//!
//! Scope: the `zwp_linux_dmabuf_v1` global is advertised at version 5
//! (v4 feedback + v3 format/modifier events; see `compositor.rs`). We
//! advertise the EGL-importable modifier set (`egl_importable_bgra_modifiers`)
//! — always LINEAR, plus the tiled modifiers KWin renders to and
//! `egl_readback` can consume. The Vulkan importer (`vulkan_record`) is a
//! separate AMD/Mesa zero-copy path gated on `vulkan_importable_bgra_modifiers`
//! (a subset of the advertised EGL set). A LINEAR dmabuf is read at capture
//! time by mmapping the fd (like shm); tiled buffers are imported zero-copy
//! via Vulkan for encode, and the non-LINEAR `with_bytes` path uses EGL
//! readback as a CPU fallback.
//!
//! Why bother: Zed, Chromium/Electron with `--enable-gpu`, and basically
//! every modern GPU-first Wayland client allocates its window as a dmabuf.
//! Refusing to advertise `zwp_linux_dmabuf_v1` makes those apps fail to
//! start on the inner compositor with no clear error.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::Mutex;

use memmap2::MmapOptions;
use tracing::debug;

// DMA_BUF_SYNC ioctl — synchronises CPU ↔ GPU cache for a dmabuf fd.
// Without this, mmapping a GPU-rendered buffer may return stale zeros on
// cache-incoherent architectures or when the GPU hasn't flushed yet.
const DMA_BUF_SYNC_READ: u64 = 1;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 4;
// ioctl number: _IOW('b', 0, struct dma_buf_sync) where struct is u64
// _IOW(type, nr, size) = ((1 << 30) | (size << 16) | (type << 8) | nr)
// type='b'=0x62, nr=0, size=8 → 0x40086200
const DMA_BUF_IOCTL_SYNC: u64 = 0x4008_6200;

fn dmabuf_sync(fd: RawFd, flags: u64) {
    unsafe {
        libc::ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags as *const u64);
    }
}

/// Audit H15: hard cap on dmabuf-derived buffer sizes. 256 MiB covers
/// 8K × 8K × 4 bpp; anything above this is an overflow attempt or a
/// degenerate buffer we shouldn't try to handle.
const MAX_DMABUF_LEN: usize = 256 * 1024 * 1024;

fn dmabuf_sync_start(fd: RawFd) {
    dmabuf_sync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
}

fn dmabuf_sync_end(fd: RawFd) {
    dmabuf_sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
}

/// RAII guard that calls `dmabuf_sync_start` on construction and
/// `dmabuf_sync_end` unconditionally on drop, even through early returns.
/// Audit C12: the pre-fix code in `with_bytes` paired these manually, but
/// several early-return paths skipped the END, leaving DMA cache fences
/// open. On some drivers/architectures this causes stale reads or GPU
/// lockups on subsequent buffer access.
struct DmabufSyncGuard {
    fd: RawFd,
}

impl DmabufSyncGuard {
    fn new(fd: RawFd) -> Self {
        dmabuf_sync_start(fd);
        Self { fd }
    }
}

impl Drop for DmabufSyncGuard {
    fn drop(&mut self) {
        dmabuf_sync_end(self.fd);
    }
}

/// Wait for the dmabuf's implicit GPU fence by exporting it as a sync_file
/// and polling until signalled (rendering complete). Without this, reads
/// race with in-flight GPU operations and return stale zeros.
///
/// DMA_BUF_IOCTL_EXPORT_SYNC_FILE = _IOWR('b', 1, {u32 flags, s32 fd})
/// = 0xC008_6201
pub(crate) fn wait_for_dmabuf_fence(prime_fd: RawFd) {
    let _ = wait_for_dmabuf_fence_status(prime_fd);
}

/// Like `wait_for_dmabuf_fence` but reports whether a fence existed and
/// whether it was already signaled when first checked. Used by callers
/// that want to log or count fence behaviour.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FenceStatus {
    /// No fence attached on the buffer's read-access slot. Either the
    /// driver doesn't use implicit sync, or the buffer has never been
    /// written to with an implicit-fence-bearing submission.
    NoFence,
    /// Fence existed and was already signaled (no wait happened).
    AlreadySignaled,
    /// Fence existed; we waited for it to signal.
    Waited,
    /// Fence existed but timed out (or poll error).
    TimedOut,
}

/// Non-blocking variant of `wait_for_dmabuf_fence_status` for the
/// commit-handler recording tap. Returns true if the buffer's
/// implicit read fence is already signaled (or there is no fence at
/// all — common on drivers that don't use implicit sync), false
/// otherwise. Never blocks. Caller is expected to skip the read
/// when this returns false rather than wait, because waiting on the
/// compositor thread stalls every inner client by however long the
/// GPU takes to drain.
pub(crate) fn dmabuf_fence_ready_now(prime_fd: RawFd) -> bool {
    const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: libc::c_ulong = 0xC008_6201;
    const DMA_BUF_SYNC_READ_FENCE: u32 = 1;

    #[repr(C)]
    struct DmaBufExportSyncFile {
        flags: u32,
        fd: i32,
    }

    let mut req = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ_FENCE,
        fd: -1,
    };
    let ret = unsafe { libc::ioctl(prime_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &mut req) };
    if ret != 0 || req.fd < 0 {
        // No fence attached — driver doesn't use implicit sync, OR
        // the buffer has never had a write submission. Either way,
        // safe to read now.
        return true;
    }
    let mut pfd = libc::pollfd {
        fd: req.fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let probe = unsafe { libc::poll(&mut pfd, 1, 0) };
    let ready = probe > 0 && (pfd.revents & libc::POLLIN) != 0;
    unsafe { libc::close(req.fd) };
    ready
}

pub(crate) fn wait_for_dmabuf_fence_status(prime_fd: RawFd) -> FenceStatus {
    const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: libc::c_ulong = 0xC008_6201;
    const DMA_BUF_SYNC_READ_FENCE: u32 = 1; // DMA_BUF_SYNC_READ

    #[repr(C)]
    struct DmaBufExportSyncFile {
        flags: u32,
        fd: i32,
    }

    let mut req = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ_FENCE,
        fd: -1,
    };
    let ret = unsafe { libc::ioctl(prime_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &mut req) };
    if ret != 0 || req.fd < 0 {
        return FenceStatus::NoFence;
    }
    // Non-blocking probe first so we can tell "already done" from "had to wait".
    let mut pfd = libc::pollfd {
        fd: req.fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let probe = unsafe { libc::poll(&mut pfd, 1, 0) };
    let status = if probe > 0 && (pfd.revents & libc::POLLIN) != 0 {
        FenceStatus::AlreadySignaled
    } else {
        let waited = unsafe { libc::poll(&mut pfd, 1, 2000) };
        if waited > 0 && (pfd.revents & libc::POLLIN) != 0 {
            FenceStatus::Waited
        } else {
            FenceStatus::TimedOut
        }
    };
    unsafe { libc::close(req.fd) };
    status
}
use wayland_server::protocol::wl_shm;
use wayland_server::WEnum;

/// The LINEAR DRM modifier (`DRM_FORMAT_MOD_LINEAR` = 0). Always included
/// in the advertised/importable set; tiled modifiers are additionally
/// included when Vulkan reports them importable.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// DRM fourcc `'AR24'` (little-endian, byte order B, G, R, A).
pub const DRM_FORMAT_ARGB8888: u32 = 0x34325241;
/// DRM fourcc `'XR24'`.
pub const DRM_FORMAT_XRGB8888: u32 = 0x34325258;

/// Raw FFI bindings to libgbm for GPU-backed dmabuf readback.
///
/// Used on AMD amdgpu (and similar drivers) where plain mmap or the DRM
/// mode_map_dumb path fails. GBM provides a driver-aware CPU mapping path.
pub(crate) mod gbm_ffi {
    use std::os::fd::RawFd;

    #[repr(C)]
    pub struct GbmDevice {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct GbmBo {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct GbmImportFdData {
        pub fd: RawFd,
        pub width: u32,
        pub height: u32,
        pub stride: u32,
        pub format: u32,
    }

    /// Modifier-aware import descriptor (`GBM_BO_IMPORT_FD_MODIFIER = 0x5504`).
    #[repr(C)]
    pub struct GbmImportFdModifierData {
        pub width: u32,
        pub height: u32,
        pub format: u32,
        pub num_fds: u32,
        pub fds: [RawFd; 4],
        pub strides: [i32; 4],
        pub offsets: [i32; 4],
        pub modifier: u64,
    }

    pub const GBM_BO_IMPORT_FD: u32 = 0x5503;
    pub const GBM_BO_IMPORT_FD_MODIFIER: u32 = 0x5504;
    /// Use a linearly-tiled BO — grants CPU-mappable memory on most drivers.
    pub const GBM_BO_USE_LINEAR: u32 = 1 << 4;
    pub const GBM_BO_TRANSFER_READ: u32 = 1 << 0;

    extern "C" {
        pub fn gbm_create_device(fd: RawFd) -> *mut GbmDevice;
        pub fn gbm_device_destroy(gbm: *mut GbmDevice);
        pub fn gbm_bo_import(
            gbm: *mut GbmDevice,
            type_: u32,
            buffer: *mut libc::c_void,
            usage: u32,
        ) -> *mut GbmBo;
        pub fn gbm_bo_map(
            bo: *mut GbmBo,
            x: u32,
            y: u32,
            width: u32,
            height: u32,
            flags: u32,
            stride: *mut u32,
            map_data: *mut *mut libc::c_void,
        ) -> *mut libc::c_void;
        pub fn gbm_bo_unmap(bo: *mut GbmBo, map_data: *mut libc::c_void);
        pub fn gbm_bo_destroy(bo: *mut GbmBo);
    }
}

/// Buffer plane as provided via `zwp_linux_buffer_params_v1.add`. We only
/// use plane 0 for the packed RGBA formats we support; additional planes
/// (YUV) are accepted during parameter collection but dropped at
/// `create_immed` time — they'd fail format validation anyway.
pub struct Plane {
    pub fd: OwnedFd,
    pub offset: u32,
    pub stride: u32,
    pub modifier: u64,
}

/// Mutable staging area for a `zwp_linux_buffer_params_v1` in flight.
/// Planes are added one at a time via `params.add`; `create` / `create_immed`
/// consumes the accumulated state.
#[derive(Default)]
pub struct ParamsState {
    pub planes: Vec<(u32, Plane)>, // (plane_idx, plane)
    pub used: bool,                // create[_immed] may only be called once
}

pub type ParamsData = Mutex<ParamsState>;

/// A single DMA-BUF plane's file descriptor and layout parameters.
pub struct PlaneData {
    pub fd: OwnedFd,
    pub offset: u32,
    pub stride: u32,
}

/// User data attached to a `wl_buffer` backed by a dmabuf.
pub struct DmabufBufferData {
    /// Plane 0 fd/offset/stride (kept for backward compat with callers that
    /// access .fd/.offset/.stride directly).
    pub fd: OwnedFd,
    pub offset: u32,
    pub stride: u32,
    /// Additional planes (plane 1+). Empty for single-plane LINEAR buffers.
    pub extra_planes: Vec<PlaneData>,
    pub width: i32,
    pub height: i32,
    /// DRM fourcc code (ARGB8888 or XRGB8888 in our implementation).
    pub drm_format: u32,
    pub modifier: u64,
}

/// AMD-specific readback via `DRM_IOCTL_AMDGPU_GEM_MMAP`.
///
/// `DRM_IOCTL_MODE_MAP_DUMB` only works for dumb buffers. For prime-imported
/// GEM handles on amdgpu, the correct ioctl is `DRM_IOCTL_AMDGPU_GEM_MMAP`
/// (command 0x46 = DRM_COMMAND_BASE(0x40) + AMDGPU_GEM_MMAP(0x06)):
///   _IOWR('d', 0x46, struct { u32 handle; u32 pad; u64 addr_ptr; }) = 0xC010_6446
fn amdgpu_gem_mmap(prime_fd: RawFd, len: usize, offset: usize) -> Option<Vec<u8>> {
    #[repr(C)]
    struct DrmPrimeHandle {
        handle: u32,
        flags: u32,
        fd: i32,
        pad: u32,
    }
    #[repr(C)]
    struct AmdgpuGemMmap {
        handle: u32,
        pad: u32,
        addr_ptr: u64,
    }

    const DRM_IOCTL_PRIME_FD_TO_HANDLE: libc::c_ulong = 0xC00C_642E;
    const DRM_IOCTL_AMDGPU_GEM_MMAP: libc::c_ulong = 0xC010_6446;

    // Try both render node and primary card node — PRIME_FD_TO_HANDLE may
    // require the primary node on some driver versions.
    for path in &[
        b"/dev/dri/renderD128\0" as &[u8],
        b"/dev/dri/card1\0",
        b"/dev/dri/card0\0",
    ] {
        let drm_fd = unsafe {
            libc::open(
                path.as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CLOEXEC,
            )
        };
        if drm_fd < 0 {
            continue;
        }

        let mut prime = DrmPrimeHandle {
            handle: 0,
            flags: 0,
            fd: prime_fd,
            pad: 0,
        };
        let ret1 = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut prime) };
        if ret1 != 0 {
            let e = unsafe { *libc::__errno_location() };
            tracing::warn!(path = %std::str::from_utf8(path).unwrap_or("?").trim_end_matches('\0'),
                errno = e, "amdgpu_gem_mmap: PRIME_FD_TO_HANDLE failed");
            unsafe { libc::close(drm_fd) };
            continue;
        }

        let mut gem_mmap = AmdgpuGemMmap {
            handle: prime.handle,
            pad: 0,
            addr_ptr: 0,
        };
        let ret2 = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &mut gem_mmap) };
        if ret2 != 0 {
            let e = unsafe { *libc::__errno_location() };
            tracing::warn!(
                handle = prime.handle,
                errno = e,
                "amdgpu_gem_mmap: GEM_MMAP ioctl failed"
            );
            unsafe { libc::close(drm_fd) };
            continue;
        }

        let total = offset + len;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ,
                libc::MAP_SHARED,
                drm_fd,
                gem_mmap.addr_ptr as libc::off_t,
            )
        };

        if ptr == libc::MAP_FAILED {
            let e = unsafe { *libc::__errno_location() };
            tracing::warn!(
                addr_ptr = gem_mmap.addr_ptr,
                errno = e,
                "amdgpu_gem_mmap: mmap failed"
            );
            unsafe { libc::close(drm_fd) };
            continue;
        }

        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
        let result = bytes[offset..offset + len].to_vec();
        let nonzero = result.iter().filter(|&&b| b != 0).count();
        tracing::warn!(
            nonzero,
            addr_ptr = gem_mmap.addr_ptr,
            "amdgpu_gem_mmap: success"
        );
        unsafe {
            libc::munmap(ptr, total);
            libc::close(drm_fd);
        }
        return Some(result);
    }
    None
}

/// Try to read a dmabuf's pixels via the DRM render node.
///
/// This is a fallback for AMD amdgpu (and similar drivers) where a plain
/// `mmap` of the prime fd succeeds but returns all-zero bytes because the
/// GPU-allocated buffer requires DRM-backed CPU access:
///
/// 1. Open `/dev/dri/renderD128` (or `card0`)
/// 2. `DRM_IOCTL_PRIME_FD_TO_HANDLE` — import the prime fd as a GEM handle
/// 3. `DRM_IOCTL_MODE_MAP_DUMB` — obtain the mmap offset for the handle
///    (amdgpu allows this for LINEAR prime-imported BOs in practice)
/// 4. `mmap` via the DRM fd at that offset to read the pixels
fn drm_readback(prime_fd: RawFd, len: usize, offset: usize) -> Option<Vec<u8>> {
    #[repr(C)]
    struct DrmPrimeHandle {
        handle: u32,
        flags: u32,
        fd: i32,
        pad: u32,
    }

    #[repr(C)]
    struct DrmModeMapDumb {
        handle: u32,
        pad: u32,
        offset: u64,
    }

    // ioctl numbers for x86_64 Linux:
    // DRM_IOCTL_PRIME_FD_TO_HANDLE: _IOWR('d', 0x2e, struct drm_prime_handle)
    //   drm_prime_handle = { u32 handle, u32 flags, s32 fd } = 12 bytes
    //   _IOWR(0x64, 0x2e, 12) = (3<<30)|(12<<16)|(0x64<<8)|0x2e = 0xC00C_642E
    const DRM_IOCTL_PRIME_FD_TO_HANDLE: libc::c_ulong = 0xC00C_642E;

    // DRM_IOCTL_MODE_MAP_DUMB: _IOWR('d', 0xB3, struct drm_mode_map_dumb)
    //   drm_mode_map_dumb = { u32 handle, u32 pad, u64 offset } = 16 bytes
    //   _IOWR(0x64, 0xB3, 16) = (3<<30)|(16<<16)|(0x64<<8)|0xB3 = 0xC010_64B3
    const DRM_IOCTL_MODE_MAP_DUMB: libc::c_ulong = 0xC010_64B3;

    let drm_paths: &[&[u8]] = &[b"/dev/dri/renderD128\0", b"/dev/dri/card0\0"];

    for path in drm_paths {
        let drm_fd = unsafe {
            libc::open(
                path.as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CLOEXEC,
            )
        };
        if drm_fd < 0 {
            continue;
        }

        let mut prime = DrmPrimeHandle {
            handle: 0,
            flags: 0,
            fd: prime_fd,
            pad: 0,
        };
        if unsafe { libc::ioctl(drm_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut prime) } != 0 {
            unsafe { libc::close(drm_fd) };
            continue;
        }

        let mut map_dumb = DrmModeMapDumb {
            handle: prime.handle,
            pad: 0,
            offset: 0,
        };
        if unsafe { libc::ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &mut map_dumb) } != 0 {
            unsafe { libc::close(drm_fd) };
            continue;
        }

        let total = offset + len;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ,
                libc::MAP_SHARED,
                drm_fd,
                map_dumb.offset as libc::off_t,
            )
        };

        if ptr != libc::MAP_FAILED {
            let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
            let result = bytes[offset..offset + len].to_vec();
            unsafe {
                libc::munmap(ptr, total);
                libc::close(drm_fd);
            }
            debug!(prime_fd, "DRM readback succeeded");
            return Some(result);
        }

        unsafe { libc::close(drm_fd) };
    }

    debug!(prime_fd, "DRM readback failed on all nodes");
    None
}

/// EGL/GL extension constants and FFI types we use directly because
/// `khronos-egl`'s safe wrappers don't cover the dmabuf import path,
/// and the `gl` crate doesn't include the `OES_EGL_image` extension.
pub(crate) mod egl_ext {
    use std::os::raw::c_void;

    pub type EglDisplay = *mut c_void;
    pub type EglContext = *mut c_void;
    pub type EglImage = *mut c_void;
    pub type EglClientBuffer = *mut c_void;
    pub type EglInt = i32;
    pub type EglEnum = u32;
    pub type EglBoolean = u32;

    pub const EGL_LINUX_DMA_BUF_EXT: EglEnum = 0x3270;
    pub const EGL_LINUX_DRM_FOURCC_EXT: EglInt = 0x3271;
    pub const EGL_DMA_BUF_PLANE0_FD_EXT: EglInt = 0x3272;
    pub const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EglInt = 0x3273;
    pub const EGL_DMA_BUF_PLANE0_PITCH_EXT: EglInt = 0x3274;
    pub const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EglInt = 0x3443;
    pub const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EglInt = 0x3444;
    pub const EGL_WIDTH: EglInt = 0x3057;
    pub const EGL_HEIGHT: EglInt = 0x3056;
    pub const EGL_NONE: EglInt = 0x3038;
    pub const EGL_PLATFORM_GBM_KHR: EglEnum = 0x31D7;

    pub type EglCreateImageKHR = unsafe extern "C" fn(
        EglDisplay,
        EglContext,
        EglEnum,
        EglClientBuffer,
        *const EglInt,
    ) -> EglImage;
    pub type EglDestroyImageKHR = unsafe extern "C" fn(EglDisplay, EglImage) -> EglBoolean;
    pub type GlEGLImageTargetTexture2DOES = unsafe extern "C" fn(target: u32, image: EglImage);
}

/// EGL extension fn signature:
/// EGLBoolean eglQueryDmaBufModifiersEXT(EGLDisplay dpy, EGLint format,
///   EGLint max_modifiers, EGLuint64KHR *modifiers, EGLBoolean *external_only,
///   EGLint *num_modifiers)
type EglQueryDmaBufModifiersExt = unsafe extern "system" fn(
    dpy: *mut std::ffi::c_void,
    format: i32,
    max_modifiers: i32,
    modifiers: *mut u64,
    external_only: *mut u32,
    num_modifiers: *mut i32,
) -> u32;

/// Query the DRM modifiers EGL can import for ARGB8888/XRGB8888 on the
/// render node, LINEAR-inclusive. Returns just [LINEAR] if EGL init or the
/// extension is unavailable (software/CI). This is the set we ADVERTISE and
/// can consume via egl_readback (works on NVIDIA and Mesa).
pub fn egl_importable_bgra_modifiers() -> &'static [u64] {
    static CACHE: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let mut set = query_egl_modifiers().unwrap_or_default();
        if !set.contains(&DRM_FORMAT_MOD_LINEAR) {
            set.push(DRM_FORMAT_MOD_LINEAR);
        }
        set.sort_unstable();
        set.dedup();
        tracing::info!(?set, "dmabuf: EGL-importable BGRA modifiers");
        set
    })
}

fn query_egl_modifiers() -> Option<Vec<u64>> {
    use khronos_egl as egl;

    let drm_fd = unsafe {
        libc::open(
            c"/dev/dri/renderD128".as_ptr() as *const libc::c_char,
            libc::O_RDWR | libc::O_CLOEXEC,
        )
    };
    if drm_fd < 0 {
        return None;
    }
    let gbm = unsafe { gbm_ffi::gbm_create_device(drm_fd) };
    if gbm.is_null() {
        unsafe { libc::close(drm_fd) };
        return None;
    }
    let lib = unsafe { libloading::Library::new("libEGL.so.1") }.ok()?;
    let egl_inst = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required_from(lib) }.ok()?;

    let display = unsafe {
        egl_inst.get_platform_display(
            egl_ext::EGL_PLATFORM_GBM_KHR,
            gbm as *mut std::ffi::c_void,
            &[egl::ATTRIB_NONE],
        )
    }
    .ok()?;
    egl_inst.initialize(display).ok()?;

    let raw = egl_inst.get_proc_address("eglQueryDmaBufModifiersEXT")?;
    let query: EglQueryDmaBufModifiersExt = unsafe { std::mem::transmute(raw) };
    let dpy_ptr = display.as_ptr();

    let mut out = Vec::new();
    for &fourcc in &[DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888] {
        let mut count: i32 = 0;
        let ok = unsafe {
            query(
                dpy_ptr,
                fourcc as i32,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
            )
        };
        if ok == 0 || count <= 0 {
            continue;
        }
        let mut mods = vec![0u64; count as usize];
        let mut ext_only = vec![0u32; count as usize];
        let mut got: i32 = 0;
        let ok2 = unsafe {
            query(
                dpy_ptr,
                fourcc as i32,
                count,
                mods.as_mut_ptr(),
                ext_only.as_mut_ptr(),
                &mut got,
            )
        };
        if ok2 != 0 {
            // external_only modifiers can't be sampled as a normal GL_TEXTURE_2D,
            // which egl_readback binds — skip them so consumption works.
            for i in 0..(got as usize) {
                if ext_only[i] == 0 {
                    out.push(mods[i]);
                }
            }
        }
    }
    // drm_fd and gbm device intentionally leak for process lifetime —
    // closing them would invalidate the EGL display.
    Some(out)
}

/// Read a dmabuf's pixels via EGL: import the prime fd as an EGLImage,
/// attach it to a GL texture, blit to a framebuffer, then `glReadPixels`.
///
/// This is the final fallback for AMD amdgpu when GBM's `gbm_bo_map`
/// returns all-zero bytes for GPU-composited framebuffers (the kernel
/// driver only exposes CPU-mappable memory for plain dumb buffers, not
/// for prime-imported render BOs). EGL goes through the GPU instead, so
/// the read sees actual rendered pixels.
///
/// Output is in the same byte order as `wl_shm::Format::Argb8888` —
/// little-endian B, G, R, A — regardless of `drm_format`, so it slots
/// straight into the existing SHM blit path.
// encoder/cursor setup takes many tightly-related params by design
#[allow(clippy::too_many_arguments)]
fn egl_readback(
    prime_fd: RawFd,
    extra_planes: &[PlaneData],
    width: i32,
    height: i32,
    stride: i32,
    offset: usize,
    drm_format: u32,
    modifier: u64,
) -> Option<Vec<u8>> {
    use khronos_egl as egl;
    use std::os::raw::c_void;

    let drm_fd = unsafe {
        libc::open(
            c"/dev/dri/renderD128".as_ptr() as *const libc::c_char,
            libc::O_RDWR | libc::O_CLOEXEC,
        )
    };
    if drm_fd < 0 {
        tracing::warn!("EGL: open renderD128 failed");
        return None;
    }

    let gbm = unsafe { gbm_ffi::gbm_create_device(drm_fd) };
    if gbm.is_null() {
        unsafe { libc::close(drm_fd) };
        tracing::warn!("EGL: gbm_create_device failed");
        return None;
    }

    let lib = match unsafe { libloading::Library::new("libEGL.so.1") } {
        Ok(l) => l,
        Err(e) => {
            unsafe {
                gbm_ffi::gbm_device_destroy(gbm);
                libc::close(drm_fd);
            }
            tracing::warn!(err = %e, "EGL: failed to load libEGL.so.1");
            return None;
        }
    };
    let egl_inst = match unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required_from(lib) } {
        Ok(i) => i,
        Err(e) => {
            unsafe {
                gbm_ffi::gbm_device_destroy(gbm);
                libc::close(drm_fd);
            }
            tracing::warn!(err = ?e, "EGL: dynamic load failed");
            return None;
        }
    };

    // Body wrapped in an immediately-invoked closure so we can use `?`
    // and still run unconditional cleanup at the end.
    let mut display_to_terminate: Option<egl::Display> = None;
    let mut ctx_to_destroy: Option<(egl::Display, egl::Context)> = None;

    let result = (|| -> Option<Vec<u8>> {
        let display = unsafe {
            egl_inst.get_platform_display(
                egl_ext::EGL_PLATFORM_GBM_KHR,
                gbm as *mut c_void,
                &[egl::ATTRIB_NONE],
            )
        }
        .map_err(|e| tracing::warn!(err = ?e, "EGL: get_platform_display failed"))
        .ok()?;

        egl_inst
            .initialize(display)
            .map_err(|e| tracing::warn!(err = ?e, "EGL: initialize failed"))
            .ok()?;
        display_to_terminate = Some(display);

        egl_inst
            .bind_api(egl::OPENGL_ES_API)
            .map_err(|e| tracing::warn!(err = ?e, "EGL: bind_api OPENGL_ES failed"))
            .ok()?;

        // Render-node EGL on Mesa typically advertises GLES configs but not
        // desktop-GL ones, and may not advertise PBUFFER surfaces at all
        // (we run surfaceless). Keep the filter minimal — just the colour
        // sizes — and let GLES2_BIT come from the context attribs.
        let config_attrs = [
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::ALPHA_SIZE,
            8,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES2_BIT,
            egl::NONE,
        ];
        let config = match egl_inst.choose_first_config(display, &config_attrs) {
            Ok(Some(c)) => c,
            Ok(None) => {
                tracing::warn!("EGL: choose_first_config returned no config");
                return None;
            }
            Err(e) => {
                tracing::warn!(err = ?e, "EGL: choose_first_config errored");
                return None;
            }
        };

        let ctx_attrs = [egl::CONTEXT_MAJOR_VERSION, 2, egl::NONE];
        let ctx = egl_inst
            .create_context(display, config, None, &ctx_attrs)
            .map_err(|e| tracing::warn!(err = ?e, "EGL: create_context failed"))
            .ok()?;
        ctx_to_destroy = Some((display, ctx));

        egl_inst
            .make_current(display, None, None, Some(ctx))
            .map_err(|e| tracing::warn!(err = ?e, "EGL: make_current failed"))
            .ok()?;

        // Load GL function pointers via eglGetProcAddress.
        gl::load_with(|s| {
            egl_inst
                .get_proc_address(s)
                .map(|p| p as *const c_void)
                .unwrap_or(std::ptr::null())
        });

        let create_image_ptr = egl_inst.get_proc_address("eglCreateImageKHR").or_else(|| {
            tracing::warn!("EGL: eglCreateImageKHR not exposed");
            None
        })?;
        let destroy_image_ptr = egl_inst.get_proc_address("eglDestroyImageKHR")?;
        let image_target_ptr = egl_inst
            .get_proc_address("glEGLImageTargetTexture2DOES")
            .or_else(|| {
                tracing::warn!("EGL: glEGLImageTargetTexture2DOES not exposed");
                None
            })?;
        let create_image: egl_ext::EglCreateImageKHR =
            unsafe { std::mem::transmute(create_image_ptr) };
        let destroy_image: egl_ext::EglDestroyImageKHR =
            unsafe { std::mem::transmute(destroy_image_ptr) };
        let image_target_tex2d: egl_ext::GlEGLImageTargetTexture2DOES =
            unsafe { std::mem::transmute(image_target_ptr) };

        let mut attribs: Vec<egl_ext::EglInt> = vec![
            egl_ext::EGL_WIDTH,
            width,
            egl_ext::EGL_HEIGHT,
            height,
            egl_ext::EGL_LINUX_DRM_FOURCC_EXT,
            drm_format as i32,
            egl_ext::EGL_DMA_BUF_PLANE0_FD_EXT,
            prime_fd,
            egl_ext::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            offset as i32,
            egl_ext::EGL_DMA_BUF_PLANE0_PITCH_EXT,
            stride,
        ];
        if modifier != DRM_FORMAT_MOD_LINEAR {
            attribs.extend_from_slice(&[
                egl_ext::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                (modifier & 0xFFFF_FFFF) as i32,
                egl_ext::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                (modifier >> 32) as i32,
            ]);
        }

        // EGL attribute constants for planes 1 and 2.
        const EGL_DMA_BUF_PLANE1_FD_EXT: egl_ext::EglInt = 0x3278;
        const EGL_DMA_BUF_PLANE1_OFFSET_EXT: egl_ext::EglInt = 0x3279;
        const EGL_DMA_BUF_PLANE1_PITCH_EXT: egl_ext::EglInt = 0x327A;
        const EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT: egl_ext::EglInt = 0x3445;
        const EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT: egl_ext::EglInt = 0x3446;
        const EGL_DMA_BUF_PLANE2_FD_EXT: egl_ext::EglInt = 0x3280;
        const EGL_DMA_BUF_PLANE2_OFFSET_EXT: egl_ext::EglInt = 0x3281;
        const EGL_DMA_BUF_PLANE2_PITCH_EXT: egl_ext::EglInt = 0x3282;
        const EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT: egl_ext::EglInt = 0x3447;
        const EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT: egl_ext::EglInt = 0x3448;

        // Append extra planes (1+) so EGL can reconstruct multi-plane
        // compressed formats like AMD DCC (2 planes).
        use std::os::fd::AsRawFd as _;
        for (plane_idx, plane) in extra_planes.iter().enumerate() {
            let (fd_attr, offset_attr, pitch_attr, mod_lo, mod_hi) = match plane_idx {
                0 => (
                    EGL_DMA_BUF_PLANE1_FD_EXT,
                    EGL_DMA_BUF_PLANE1_OFFSET_EXT,
                    EGL_DMA_BUF_PLANE1_PITCH_EXT,
                    EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
                    EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
                ),
                1 => (
                    EGL_DMA_BUF_PLANE2_FD_EXT,
                    EGL_DMA_BUF_PLANE2_OFFSET_EXT,
                    EGL_DMA_BUF_PLANE2_PITCH_EXT,
                    EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT,
                    EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT,
                ),
                _ => break,
            };
            attribs.extend_from_slice(&[
                fd_attr,
                plane.fd.as_raw_fd(),
                offset_attr,
                plane.offset as i32,
                pitch_attr,
                plane.stride as i32,
                mod_lo,
                (modifier & 0xFFFF_FFFF) as i32,
                mod_hi,
                (modifier >> 32) as i32,
            ]);
        }

        attribs.push(egl_ext::EGL_NONE);

        let image = unsafe {
            create_image(
                display.as_ptr(),
                std::ptr::null_mut(),
                egl_ext::EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if image.is_null() {
            tracing::warn!("EGL: eglCreateImageKHR returned NULL");
            return None;
        }
        tracing::debug!("EGL: eglCreateImageKHR succeeded");

        let mut texture: u32 = 0;
        let mut fbo: u32 = 0;
        let pixels = unsafe {
            while gl::GetError() != gl::NO_ERROR {}

            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            image_target_tex2d(gl::TEXTURE_2D, image);
            let img_err = gl::GetError();
            if img_err != gl::NO_ERROR {
                tracing::warn!(
                    err = format!("0x{:X}", img_err),
                    "EGL: glEGLImageTargetTexture2DOES error"
                );
            }

            gl::GenFramebuffers(1, &mut fbo);
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                texture,
                0,
            );
            let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
            if status != gl::FRAMEBUFFER_COMPLETE {
                tracing::warn!(status = format!("0x{:X}", status), "EGL: FBO incomplete");
                gl::DeleteFramebuffers(1, &fbo);
                gl::DeleteTextures(1, &texture);
                destroy_image(display.as_ptr(), image);
                return None;
            }

            let buf_len = match (height as usize).checked_mul(stride as usize) {
                Some(n) if n <= MAX_DMABUF_LEN => n,
                _ => {
                    tracing::warn!(height, stride, "egl_readback: oversized buffer; aborting");
                    gl::DeleteFramebuffers(1, &fbo);
                    gl::DeleteTextures(1, &texture);
                    destroy_image(display.as_ptr(), image);
                    return None;
                }
            };
            let mut buf = vec![0u8; buf_len];
            gl::PixelStorei(gl::PACK_ALIGNMENT, 4);
            gl::ReadPixels(
                0,
                0,
                width,
                height,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                buf.as_mut_ptr() as *mut c_void,
            );
            gl::Finish();
            let read_err = gl::GetError();
            if read_err != gl::NO_ERROR {
                tracing::warn!(err = format!("0x{:X}", read_err), "EGL: glReadPixels error");
            }
            // Swizzle R↔B in place (RGBA → BGRA = Wayland ARGB8888 byte order).
            for px in buf.chunks_exact_mut(4) {
                px.swap(0, 2);
            }

            // NO row flip: we glReadPixels directly from a texture-backed FBO
            // (no quad render), so the texture's first memory row (the dmabuf's
            // top row, Wayland top-left origin) maps to FBO y=0 and is returned
            // first — i.e. rows already come out top-to-bottom. The previous
            // bottom-left→top-left flip was a double-flip that turned the image
            // upside down (proven against the Vulkan path: vflip-correlation
            // 0.92). Return the buffer as-read.
            gl::DeleteFramebuffers(1, &fbo);
            gl::DeleteTextures(1, &texture);
            destroy_image(display.as_ptr(), image);

            let nonzero = buf.iter().filter(|&&b| b != 0).count();
            tracing::warn!(nonzero_bytes = nonzero, "EGL readback complete");
            buf
        };

        Some(pixels)
    })();

    unsafe {
        if let Some((display, ctx)) = ctx_to_destroy {
            egl_inst.make_current(display, None, None, None).ok();
            egl_inst.destroy_context(display, ctx).ok();
        }
        if let Some(display) = display_to_terminate {
            egl_inst.terminate(display).ok();
        }
        gbm_ffi::gbm_device_destroy(gbm);
        libc::close(drm_fd);
    }

    result
}

/// Read a dmabuf's pixels via the GBM (Generic Buffer Management) API.
///
/// This is the preferred fallback for AMD amdgpu where plain mmap returns
/// zeros and `DRM_IOCTL_MODE_MAP_DUMB` is rejected for prime-imported BOs.
/// GBM delegates to the driver's own CPU-mapping implementation.
fn gbm_readback(
    prime_fd: RawFd,
    width: i32,
    height: i32,
    stride: i32,
    offset: usize,
    modifier: u64,
) -> Option<Vec<u8>> {
    use gbm_ffi::*;

    // Audit H15: bound the multiplication; reject oversized or overflowed
    // dimensions instead of allocating a multi-GB buffer.
    let len = match (height as usize).checked_mul(stride as usize) {
        Some(n) if n <= MAX_DMABUF_LEN => n,
        _ => {
            tracing::warn!(height, stride, "gbm_readback: oversized buffer; aborting");
            return None;
        }
    };

    // Open the DRM render node to create the GBM device.
    let drm_fd = unsafe {
        libc::open(
            c"/dev/dri/renderD128".as_ptr() as *const libc::c_char,
            libc::O_RDWR | libc::O_CLOEXEC,
        )
    };
    if drm_fd < 0 {
        debug!(prime_fd, "GBM: could not open renderD128");
        return None;
    }

    let gbm = unsafe { gbm_create_device(drm_fd) };
    if gbm.is_null() {
        unsafe { libc::close(drm_fd) };
        debug!(prime_fd, "GBM: gbm_create_device failed");
        return None;
    }

    // Try modifier-aware import first (handles tiled/compressed formats).
    let mut import_mod = GbmImportFdModifierData {
        width: width as u32,
        height: height as u32,
        format: DRM_FORMAT_ARGB8888,
        num_fds: 1,
        fds: [prime_fd, -1, -1, -1],
        strides: [stride, 0, 0, 0],
        offsets: [offset as i32, 0, 0, 0],
        modifier,
    };

    let bo = unsafe {
        gbm_bo_import(
            gbm,
            GBM_BO_IMPORT_FD_MODIFIER,
            &mut import_mod as *mut _ as *mut libc::c_void,
            GBM_BO_USE_LINEAR,
        )
    };

    if !bo.is_null() {
        return gbm_map_and_read(bo, gbm, drm_fd, width, height, stride, len);
    }

    debug!(
        prime_fd,
        "GBM: FD_MODIFIER import failed, trying plain FD import"
    );

    // Fall back to simple (no-modifier) import.
    let mut import_simple = GbmImportFdData {
        fd: prime_fd,
        width: width as u32,
        height: height as u32,
        stride: stride as u32,
        format: DRM_FORMAT_ARGB8888,
    };

    let bo2 = unsafe {
        gbm_bo_import(
            gbm,
            GBM_BO_IMPORT_FD,
            &mut import_simple as *mut _ as *mut libc::c_void,
            GBM_BO_USE_LINEAR,
        )
    };

    if bo2.is_null() {
        unsafe {
            gbm_device_destroy(gbm);
            libc::close(drm_fd);
        }
        debug!(prime_fd, "GBM: plain FD import also failed");
        return None;
    }

    gbm_map_and_read(bo2, gbm, drm_fd, width, height, stride, len)
}

fn gbm_map_and_read(
    bo: *mut gbm_ffi::GbmBo,
    gbm: *mut gbm_ffi::GbmDevice,
    drm_fd: RawFd,
    width: i32,
    height: i32,
    stride: i32,
    len: usize,
) -> Option<Vec<u8>> {
    use gbm_ffi::*;

    let mut map_stride: u32 = 0;
    let mut map_data: *mut libc::c_void = std::ptr::null_mut();

    let ptr = unsafe {
        gbm_bo_map(
            bo,
            0,
            0,
            width as u32,
            height as u32,
            GBM_BO_TRANSFER_READ,
            &mut map_stride,
            &mut map_data,
        )
    };

    let result = if !ptr.is_null() && !std::ptr::eq(ptr, libc::MAP_FAILED) {
        // Audit H15: bound map_len before slicing; gbm_bo_map can return a
        // larger map_stride than requested, so an attacker-influenced height
        // could combine with that stride to overflow.
        let map_len = match (height as usize).checked_mul(map_stride as usize) {
            Some(n) if n <= MAX_DMABUF_LEN => n,
            _ => {
                tracing::warn!(height, map_stride, "gbm_map_and_read: oversized; aborting");
                unsafe { gbm_bo_unmap(bo, map_data) };
                return None;
            }
        };
        let src = unsafe { std::slice::from_raw_parts(ptr as *const u8, map_len) };

        let mut out = vec![0u8; len];
        if map_stride as i32 == stride {
            out.copy_from_slice(&src[..len]);
        } else {
            // Strides differ — copy row by row.
            for row in 0..height as usize {
                let src_row = row * map_stride as usize;
                let dst_row = row * stride as usize;
                let row_bytes = stride as usize;
                out[dst_row..dst_row + row_bytes]
                    .copy_from_slice(&src[src_row..src_row + row_bytes]);
            }
        }
        unsafe { gbm_bo_unmap(bo, map_data) };
        let nonzero = out.iter().filter(|&&b| b != 0).count();
        tracing::warn!(nonzero_bytes = nonzero, map_stride, "GBM map succeeded");
        Some(out)
    } else {
        tracing::warn!(ptr = format!("{:?}", ptr), "GBM map returned null/failed");
        None
    };

    unsafe {
        gbm_bo_destroy(bo);
        gbm_device_destroy(gbm);
        libc::close(drm_fd);
    }
    result
}

impl DmabufBufferData {
    /// Construct from a sorted (by plane index) list of planes.
    /// `sorted_planes[0]` is plane 0; any remaining planes are stored in
    /// `extra_planes` and forwarded to EGL when performing readback.
    pub fn new(mut sorted_planes: Vec<Plane>, width: i32, height: i32, drm_format: u32) -> Self {
        let plane0 = sorted_planes.remove(0);
        let extra_planes = sorted_planes
            .into_iter()
            .map(|p| PlaneData {
                fd: p.fd,
                offset: p.offset,
                stride: p.stride,
            })
            .collect();
        Self {
            fd: plane0.fd,
            offset: plane0.offset,
            stride: plane0.stride,
            extra_planes,
            width,
            height,
            drm_format,
            modifier: plane0.modifier,
        }
    }

    /// Borrow the buffer bytes without copying. The closure receives a
    /// `&[u8]` slice valid for its duration while the mmap lock is held.
    ///
    /// For non-LINEAR modifiers (GPU-tiled formats) we skip straight to EGL
    /// readback — CPU mmap is undefined for tiled layouts, but EGL can import
    /// any modifier via EGL_EXT_image_dma_buf_import_modifiers and read back
    /// with glReadPixels. This enables screenshots of KWin/GPU compositor
    /// output regardless of the modifier negotiated with the GPU driver.
    pub fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let len = (self.height as usize).checked_mul(self.stride as usize)?;
        let fd = self.fd.as_raw_fd();
        let start = self.offset as usize;

        if self.modifier != DRM_FORMAT_MOD_LINEAR {
            // Wait for GPU rendering to complete before reading. Without this,
            // we race the compositor's in-flight render and may read stale zeros.
            wait_for_dmabuf_fence(fd);
            for plane in &self.extra_planes {
                wait_for_dmabuf_fence(plane.fd.as_raw_fd());
            }
            return egl_readback(
                fd,
                &self.extra_planes,
                self.width,
                self.height,
                self.stride as i32,
                start,
                self.drm_format,
                self.modifier,
            )
            .map(|v| f(v.as_slice()));
        }

        let total = (self.offset as usize).checked_add(len)?;
        let end = start + len;

        // Wait for any in-flight GPU writes to the buffer to complete before
        // we access it from the CPU. dmabuf_sync_start alone (below) issues
        // a cache-coherency fence on the prime fd, but on AMD amdgpu the
        // gem_mmap fallback path bypasses that fence — visible as a diagonal
        // bottom-right cutoff in recordings (top rows complete, bottom rows
        // missing data). Explicitly waiting on the buffer's fence sync_file
        // here closes that race and matches the pre-existing non-LINEAR path.
        wait_for_dmabuf_fence(fd);

        // Audit C12: RAII guard ensures dmabuf_sync_end is called even
        // through early returns. The previous manual pair was vulnerable
        // to skips, leaving DMA cache fences open (stale reads, GPU
        // lockups on some drivers).
        let _sync_guard = DmabufSyncGuard::new(fd);

        // Re-mmap each access rather than caching: the buffer may have been
        // re-rendered by the GPU since the last read, and a cached mmap on
        // write-combining memory returns stale zeros.
        let result = match unsafe { MmapOptions::new().len(total).map(&self.fd) } {
            Ok(m) => {
                if end <= m.len() {
                    let slice = &m[start..end];
                    // On AMD amdgpu (and some other drivers), a plain mmap of the
                    // prime fd may succeed but return all-zero bytes because the
                    // GPU-allocated buffer requires a DRM-backed mmap path.
                    // Detect this and fall back to the DRM readback path.
                    if slice.iter().all(|&b| b == 0) {
                        tracing::warn!(
                            fd,
                            w = self.width,
                            h = self.height,
                            "plain mmap returned all-zeros; trying AMD gem_mmap then GBM"
                        );
                        wait_for_dmabuf_fence(fd);
                        if let Some(v) = amdgpu_gem_mmap(fd, len, start) {
                            if v.iter().any(|&b| b != 0) {
                                return Some(f(v.as_slice()));
                            }
                        }
                        if let Some(v) = drm_readback(fd, len, start) {
                            tracing::warn!(fd, "DRM readback succeeded");
                            Some(f(v.as_slice()))
                        } else {
                            tracing::warn!(
                                fd,
                                w = self.width,
                                h = self.height,
                                "DRM readback failed; trying GBM readback"
                            );
                            let gbm_result = gbm_readback(
                                fd,
                                self.width,
                                self.height,
                                self.stride as i32,
                                start,
                                self.modifier,
                            );
                            // GBM may return Some(_) but with all-zero bytes for
                            // GPU-composited render BOs on amdgpu. Fall through to
                            // EGL in that case.
                            let usable = gbm_result
                                .as_ref()
                                .map(|v| v.iter().any(|&b| b != 0))
                                .unwrap_or(false);
                            if usable {
                                tracing::warn!(fd, "GBM readback succeeded");
                                gbm_result.map(|v| f(v.as_slice()))
                            } else {
                                tracing::warn!(
                                    fd,
                                    w = self.width,
                                    h = self.height,
                                    "GBM readback empty/zero; trying EGL readback"
                                );
                                egl_readback(
                                    fd,
                                    &self.extra_planes,
                                    self.width,
                                    self.height,
                                    self.stride as i32,
                                    start,
                                    self.drm_format,
                                    self.modifier,
                                )
                                .map(|v| f(v.as_slice()))
                            }
                        }
                    } else {
                        // Non-zero mmap data — check if it looks like valid content
                        // (some drivers return opaque black/zeros-RGB which is still valid)
                        Some(f(slice))
                    }
                } else {
                    None
                }
            }
            Err(e) => {
                tracing::warn!(
                    fd, w = self.width, h = self.height,
                    modifier = self.modifier, err = %e,
                    "dmabuf mmap failed — trying DRM readback"
                );
                if let Some(v) = drm_readback(fd, len, start) {
                    Some(f(v.as_slice()))
                } else {
                    debug!(
                        fd,
                        w = self.width,
                        h = self.height,
                        "DRM readback failed; trying GBM readback"
                    );
                    let gbm_result = gbm_readback(
                        fd,
                        self.width,
                        self.height,
                        self.stride as i32,
                        start,
                        self.modifier,
                    );
                    let usable = gbm_result
                        .as_ref()
                        .map(|v| v.iter().any(|&b| b != 0))
                        .unwrap_or(false);
                    if usable {
                        gbm_result.map(|v| f(v.as_slice()))
                    } else {
                        debug!(
                            fd,
                            w = self.width,
                            h = self.height,
                            "GBM readback empty/zero; trying EGL readback"
                        );
                        egl_readback(
                            fd,
                            &self.extra_planes,
                            self.width,
                            self.height,
                            self.stride as i32,
                            start,
                            self.drm_format,
                            self.modifier,
                        )
                        .map(|v| f(v.as_slice()))
                    }
                }
            }
        };

        // dmabuf_sync_end is now called by the DmabufSyncGuard's Drop, not here.
        result
    }

    /// Project the DRM format code onto the wl_shm::Format enum our
    /// existing blit code understands. Both protocols use the same byte
    /// layout for these two packed formats, so no pixel reshuffling.
    pub fn as_shm_format(&self) -> WEnum<wl_shm::Format> {
        match self.drm_format {
            DRM_FORMAT_ARGB8888 => WEnum::Value(wl_shm::Format::Argb8888),
            DRM_FORMAT_XRGB8888 => WEnum::Value(wl_shm::Format::Xrgb8888),
            // Anything else never reaches here — we reject at create_immed.
            other => WEnum::Unknown(other),
        }
    }
}

#[cfg(test)]
mod nv0_tests {
    #[test]
    fn egl_modifier_query_is_linear_inclusive() {
        // On any host (even no-GPU CI where EGL init fails) the result must
        // contain LINEAR so software/shm + LINEAR clients keep working.
        let mods = super::egl_importable_bgra_modifiers();
        assert!(
            mods.contains(&super::DRM_FORMAT_MOD_LINEAR),
            "EGL importable set must include LINEAR, got {mods:?}"
        );
    }
}
