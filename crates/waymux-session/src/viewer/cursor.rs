// SPDX-License-Identifier: Apache-2.0

//! Remote-cursor capture for the viewer overlay. Records the client's
//! `wl_pointer.set_cursor` surface + hotspot, reads the cursor buffer back to
//! RGBA on shape change, and queues CursorImage/CursorPos updates for the
//! viewer socket writer to forward to the bridge.

use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
pub struct CursorImage {
    pub w: u16,
    pub h: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CursorPos {
    pub x: f32,
    pub y: f32,
    pub seq: u32,
}

/// One queued cursor update.
pub enum CursorUpdate {
    Image(CursorImage),
    Pos(CursorPos),
}

/// Thread-safe queue: both image and position collapse to latest-wins (only
/// the newest value matters; the Vec-per-image design was an unbounded leak
/// when no CudaNvenc viewer was draining, e.g. on other encoder codecs or when
/// no viewer is connected).
pub struct CursorChannel {
    latest_image: Mutex<Option<CursorImage>>,
    latest_pos: Mutex<Option<CursorPos>>,
}

impl CursorChannel {
    pub fn new() -> Self {
        Self {
            latest_image: Mutex::new(None),
            latest_pos: Mutex::new(None),
        }
    }
    pub fn push_image(&self, img: CursorImage) {
        *self.latest_image.lock().unwrap() = Some(img);
    }
    pub fn push_pos(&self, pos: CursorPos) {
        *self.latest_pos.lock().unwrap() = Some(pos);
    }
    /// Drain queued updates: the latest image (if any) first so the browser
    /// sets the cursor shape before updating the position.
    pub fn drain(&self) -> Vec<CursorUpdate> {
        let mut out: Vec<CursorUpdate> = Vec::new();
        if let Some(img) = self.latest_image.lock().unwrap().take() {
            out.push(CursorUpdate::Image(img));
        }
        if let Some(p) = self.latest_pos.lock().unwrap().take() {
            out.push(CursorUpdate::Pos(p));
        }
        out
    }
}

impl Default for CursorChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Max cursor edge we will forward; larger cursors are dropped with a warning.
pub const MAX_CURSOR_DIM: u16 = 256;

/// Convert ARGB8888 (wl_shm / DRM little-endian; stored as B,G,R,A bytes) to
/// packed RGBA8888, honoring `stride` (bytes per row, may exceed w*4).
///
/// # Panics
/// Does not panic: if `src` is shorter than `(h-1)*stride + w*4`, the missing
/// pixels are left zeroed (defensive against malformed client buffers).
pub fn argb8888_bytes_to_rgba(src: &[u8], w: u32, h: u32, stride: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let si = (y * stride + x * 4) as usize;
            let di = ((y * w + x) * 4) as usize;
            if si + 3 >= src.len() {
                continue;
            }
            out[di] = src[si + 2]; // R
            out[di + 1] = src[si + 1]; // G
            out[di + 2] = src[si]; // B
            out[di + 3] = src[si + 3]; // A
        }
    }
    out
}

/// glReadPixels returns rows bottom-up relative to the FBO; the source dmabuf
/// is top-down. Reverse row order so the cursor isn't upside down.
pub fn flip_rows_rgba(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let row = (w * 4) as usize;
    let mut out = vec![0u8; src.len()];
    for y in 0..h as usize {
        let sy = (h as usize - 1 - y) * row;
        let dy = y * row;
        if sy + row <= src.len() && dy + row <= out.len() {
            out[dy..dy + row].copy_from_slice(&src[sy..sy + row]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// DmabufCursorReader — dedicated EGL+GLES2 readback for dmabuf-backed cursors.
// ---------------------------------------------------------------------------

/// Lazily-initialized EGL+GLES2 context dedicated to reading small cursor
/// dmabufs back to RGBA. Independent of the encoder's CUDA context. Serializes
/// on an internal mutex (cursor reads are rare). Every EGL/GL failure returns
/// None and logs — must never panic the compositor thread.
///
/// This mirrors the proven `dmabuf::egl_readback` import path (same EGL GBM
/// platform display, same GLES2 context, same `eglCreateImageKHR(..,
/// EGL_LINUX_DMA_BUF_EXT, ..)` attribute array shape, same
/// glEGLImageTargetTexture2DOES → FBO → glReadPixels), but keeps a persistent
/// display/context/FBO so repeated cursor reads don't re-init EGL each time.
pub struct DmabufCursorReader {
    inner: std::sync::Mutex<Option<EglGl>>,
}

/// Holds the persistent EGL display + GLES2 context + the loaded EGL/GL
/// extension fn pointers and a reusable FBO. The `egl` `DynamicInstance` must
/// outlive any proc-address-derived fn pointers, so it is kept here alongside
/// the GBM device / DRM fd that back the display.
struct EglGl {
    egl: khronos_egl::DynamicInstance<khronos_egl::EGL1_5>,
    display: khronos_egl::Display,
    context: khronos_egl::Context,
    gbm: *mut std::os::raw::c_void,
    drm_fd: std::os::fd::RawFd,
    create_image: crate::dmabuf::egl_ext::EglCreateImageKHR,
    destroy_image: crate::dmabuf::egl_ext::EglDestroyImageKHR,
    image_target_tex2d: crate::dmabuf::egl_ext::GlEGLImageTargetTexture2DOES,
    /// Reusable framebuffer object (created once, reused across reads).
    fbo: u32,
}

// SAFETY: the EGL display/context and the raw fn pointers / gbm handle are only
// ever touched while the owning `DmabufCursorReader::inner` Mutex is held, on
// whatever single thread calls `read_rgba` at that moment (cursor reads are
// rare and serialized). The encoder uses a *separate* EGL display + a CUDA
// context; this reader never shares either, so there is no cross-context
// state.
unsafe impl Send for EglGl {}

impl DmabufCursorReader {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(None),
        }
    }

    /// Import `(fd, modifier, drm_format, offset, w, h, stride)`, render it to
    /// an FBO-backed texture, `glReadPixels` to RGBA, then vertically flip.
    /// Waits on the producer's implicit GPU fence first (for non-LINEAR
    /// buffers) so we never read a torn/stale in-flight render; if the fence
    /// times out we skip this read (the next commit retries). Returns `None`
    /// on ANY EGL/GL/fence failure (logged) — never panics.
    // encoder/cursor setup takes many tightly-related params by design
    #[allow(clippy::too_many_arguments)]
    pub fn read_rgba(
        &self,
        fd: std::os::fd::RawFd,
        modifier: u64,
        drm_format: u32,
        offset: u32,
        width: u32,
        height: u32,
        stride: u32,
    ) -> Option<Vec<u8>> {
        // Mirror `dmabuf::egl_readback`: wait on the buffer's implicit read
        // fence before importing, but only for non-LINEAR (GPU-tiled) buffers
        // — LINEAR cursors are CPU-written and carry no GPU render fence.
        // egl_readback uses the void `wait_for_dmabuf_fence`; we use the
        // status-returning variant so we can skip on TimedOut rather than
        // read stale pixels.
        if modifier != crate::dmabuf::DRM_FORMAT_MOD_LINEAR
            && crate::dmabuf::wait_for_dmabuf_fence_status(fd)
                == crate::dmabuf::FenceStatus::TimedOut
        {
            tracing::warn!(fd, "cursor: dmabuf fence wait timed out, skipping read");
            return None;
        }
        let mut g = self.inner.lock().unwrap();
        if g.is_none() {
            match EglGl::init() {
                Some(e) => *g = Some(e),
                None => {
                    tracing::warn!("cursor: EGL/GL init failed");
                    return None;
                }
            }
        }
        g.as_mut()
            .unwrap()
            .import_and_read(fd, modifier, drm_format, offset, width, height, stride)
    }
}

impl Default for DmabufCursorReader {
    fn default() -> Self {
        Self::new()
    }
}

// GL constants we need (the `gl` crate also defines these; we use its symbols
// directly via the dynamically-loaded entry points, so no local redefinition
// is required — see import_and_read).

impl EglGl {
    /// Open renderD128 → GBM device → EGL GBM-platform display → initialize →
    /// bind GLES API → choose a GLES2 config → create a GLES2 context → make it
    /// current (surfaceless) → load the EGL-image + GL extension fns → create a
    /// reusable FBO. Mirrors `dmabuf::egl_readback`'s bring-up exactly. Returns
    /// `None` (after cleaning up) on any failure.
    fn init() -> Option<Self> {
        use crate::dmabuf::{egl_ext, gbm_ffi};
        use khronos_egl as egl;
        use std::os::raw::c_void;

        let drm_fd = unsafe {
            libc::open(
                c"/dev/dri/renderD128".as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CLOEXEC,
            )
        };
        if drm_fd < 0 {
            tracing::warn!("cursor: open renderD128 failed");
            return None;
        }
        let gbm = unsafe { gbm_ffi::gbm_create_device(drm_fd) } as *mut c_void;
        if gbm.is_null() {
            unsafe { libc::close(drm_fd) };
            tracing::warn!("cursor: gbm_create_device failed");
            return None;
        }

        let lib = match unsafe { libloading::Library::new("libEGL.so.1") } {
            Ok(l) => l,
            Err(e) => {
                unsafe {
                    gbm_ffi::gbm_device_destroy(gbm as *mut _);
                    libc::close(drm_fd);
                }
                tracing::warn!(err = %e, "cursor: failed to load libEGL.so.1");
                return None;
            }
        };
        let egl_inst = match unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required_from(lib) }
        {
            Ok(i) => i,
            Err(e) => {
                unsafe {
                    gbm_ffi::gbm_device_destroy(gbm as *mut _);
                    libc::close(drm_fd);
                }
                tracing::warn!(err = ?e, "cursor: EGL dynamic load failed");
                return None;
            }
        };

        // Helper to tear down gbm + drm_fd on the early-return paths below.
        let cleanup = |gbm: *mut c_void, drm_fd: i32| unsafe {
            gbm_ffi::gbm_device_destroy(gbm as *mut _);
            libc::close(drm_fd);
        };

        let display = match unsafe {
            egl_inst.get_platform_display(egl_ext::EGL_PLATFORM_GBM_KHR, gbm, &[egl::ATTRIB_NONE])
        } {
            Ok(d) => d,
            Err(e) => {
                cleanup(gbm, drm_fd);
                tracing::warn!(err = ?e, "cursor: get_platform_display failed");
                return None;
            }
        };
        if let Err(e) = egl_inst.initialize(display) {
            cleanup(gbm, drm_fd);
            tracing::warn!(err = ?e, "cursor: eglInitialize failed");
            return None;
        }
        if let Err(e) = egl_inst.bind_api(egl::OPENGL_ES_API) {
            egl_inst.terminate(display).ok();
            cleanup(gbm, drm_fd);
            tracing::warn!(err = ?e, "cursor: bind_api OPENGL_ES failed");
            return None;
        }

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
                egl_inst.terminate(display).ok();
                cleanup(gbm, drm_fd);
                tracing::warn!("cursor: choose_first_config returned no config");
                return None;
            }
            Err(e) => {
                egl_inst.terminate(display).ok();
                cleanup(gbm, drm_fd);
                tracing::warn!(err = ?e, "cursor: choose_first_config errored");
                return None;
            }
        };

        let ctx_attrs = [egl::CONTEXT_MAJOR_VERSION, 2, egl::NONE];
        let context = match egl_inst.create_context(display, config, None, &ctx_attrs) {
            Ok(c) => c,
            Err(e) => {
                egl_inst.terminate(display).ok();
                cleanup(gbm, drm_fd);
                tracing::warn!(err = ?e, "cursor: create_context failed");
                return None;
            }
        };
        if let Err(e) = egl_inst.make_current(display, None, None, Some(context)) {
            egl_inst.destroy_context(display, context).ok();
            egl_inst.terminate(display).ok();
            cleanup(gbm, drm_fd);
            tracing::warn!(err = ?e, "cursor: make_current failed");
            return None;
        }

        // Load GL function pointers via eglGetProcAddress (same as egl_readback).
        gl::load_with(|s| {
            egl_inst
                .get_proc_address(s)
                .map(|p| p as *const c_void)
                .unwrap_or(std::ptr::null())
        });

        let teardown_ctx = |egl_inst: &egl::DynamicInstance<egl::EGL1_5>,
                            display: egl::Display,
                            context: egl::Context| {
            egl_inst.make_current(display, None, None, None).ok();
            egl_inst.destroy_context(display, context).ok();
            egl_inst.terminate(display).ok();
        };

        let create_image_ptr = match egl_inst.get_proc_address("eglCreateImageKHR") {
            Some(p) => p,
            None => {
                teardown_ctx(&egl_inst, display, context);
                cleanup(gbm, drm_fd);
                tracing::warn!("cursor: eglCreateImageKHR not exposed");
                return None;
            }
        };
        let destroy_image_ptr = match egl_inst.get_proc_address("eglDestroyImageKHR") {
            Some(p) => p,
            None => {
                teardown_ctx(&egl_inst, display, context);
                cleanup(gbm, drm_fd);
                tracing::warn!("cursor: eglDestroyImageKHR not exposed");
                return None;
            }
        };
        let image_target_ptr = match egl_inst.get_proc_address("glEGLImageTargetTexture2DOES") {
            Some(p) => p,
            None => {
                teardown_ctx(&egl_inst, display, context);
                cleanup(gbm, drm_fd);
                tracing::warn!("cursor: glEGLImageTargetTexture2DOES not exposed");
                return None;
            }
        };
        let create_image: egl_ext::EglCreateImageKHR =
            unsafe { std::mem::transmute(create_image_ptr) };
        let destroy_image: egl_ext::EglDestroyImageKHR =
            unsafe { std::mem::transmute(destroy_image_ptr) };
        let image_target_tex2d: egl_ext::GlEGLImageTargetTexture2DOES =
            unsafe { std::mem::transmute(image_target_ptr) };

        // Create the reusable FBO once.
        let mut fbo: u32 = 0;
        unsafe {
            while gl::GetError() != gl::NO_ERROR {}
            gl::GenFramebuffers(1, &mut fbo);
        }
        if fbo == 0 {
            teardown_ctx(&egl_inst, display, context);
            cleanup(gbm, drm_fd);
            tracing::warn!("cursor: glGenFramebuffers produced 0");
            return None;
        }

        Some(EglGl {
            egl: egl_inst,
            display,
            context,
            gbm,
            drm_fd,
            create_image,
            destroy_image,
            image_target_tex2d,
            fbo,
        })
    }

    /// Import a single-plane ARGB8888/XRGB8888 dmabuf as an EGLImage, sample it into a
    /// texture, attach to the reusable FBO, glReadPixels to RGBA, then flip.
    /// Tears down the per-call EGLImage + texture on EVERY path. Returns `None`
    /// on any failure (logged) — never panics.
    // encoder/cursor setup takes many tightly-related params by design
    #[allow(clippy::too_many_arguments)]
    fn import_and_read(
        &mut self,
        fd: std::os::fd::RawFd,
        modifier: u64,
        drm_format: u32,
        offset: u32,
        width: u32,
        height: u32,
        stride: u32,
    ) -> Option<Vec<u8>> {
        use crate::dmabuf::egl_ext;
        use std::os::raw::c_void;

        const DRM_FORMAT_MOD_LINEAR: u64 = 0;

        // Ensure our context is current (cheap if already current). If another
        // EGL context on this thread became current since init, this restores
        // ours; on failure we bail rather than scribble on the wrong context.
        if let Err(e) = self
            .egl
            .make_current(self.display, None, None, Some(self.context))
        {
            tracing::warn!(err = ?e, "cursor: make_current (read) failed");
            return None;
        }

        // Attribute array shape mirrors `dmabuf::egl_readback` / encode_dmabuf:
        // WIDTH, HEIGHT, FOURCC, PLANE0 FD/OFFSET/PITCH, then (only for non-
        // LINEAR) PLANE0 MODIFIER LO/HI, terminated by EGL_NONE.
        let mut attribs: Vec<egl_ext::EglInt> = vec![
            egl_ext::EGL_WIDTH,
            width as i32,
            egl_ext::EGL_HEIGHT,
            height as i32,
            egl_ext::EGL_LINUX_DRM_FOURCC_EXT,
            drm_format as i32,
            egl_ext::EGL_DMA_BUF_PLANE0_FD_EXT,
            fd,
            egl_ext::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            offset as i32,
            egl_ext::EGL_DMA_BUF_PLANE0_PITCH_EXT,
            stride as i32,
        ];
        if modifier != DRM_FORMAT_MOD_LINEAR {
            attribs.extend_from_slice(&[
                egl_ext::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                (modifier & 0xFFFF_FFFF) as i32,
                egl_ext::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                (modifier >> 32) as i32,
            ]);
        }
        attribs.push(egl_ext::EGL_NONE);

        let image = unsafe {
            (self.create_image)(
                self.display.as_ptr(),
                std::ptr::null_mut(),
                egl_ext::EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if image.is_null() {
            tracing::warn!(
                fd,
                modifier = format!("0x{:016x}", modifier),
                stride,
                "cursor: eglCreateImageKHR returned NULL"
            );
            return None;
        }

        let mut texture: u32 = 0;
        let result: Option<Vec<u8>> = unsafe {
            while gl::GetError() != gl::NO_ERROR {}

            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            (self.image_target_tex2d)(gl::TEXTURE_2D, image);
            let img_err = gl::GetError();
            if img_err != gl::NO_ERROR {
                // Spec: return None on ANY EGL/GL failure so we never forward
                // corrupt/black pixels from a failed image bind. Yield None here;
                // the shared per-call teardown below deletes the EGLImage and the
                // texture already created, matching the other early-return paths.
                tracing::warn!(
                    err = format!("0x{:X}", img_err),
                    "cursor: glEGLImageTargetTexture2DOES error, returning None"
                );
                None
            } else {
                gl::BindFramebuffer(gl::FRAMEBUFFER, self.fbo);
                gl::FramebufferTexture2D(
                    gl::FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    texture,
                    0,
                );
                let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
                if status != gl::FRAMEBUFFER_COMPLETE {
                    tracing::warn!(status = format!("0x{:X}", status), "cursor: FBO incomplete");
                    None
                } else {
                    let buf_len = (width as usize)
                        .checked_mul(height as usize)
                        .and_then(|n| n.checked_mul(4));
                    match buf_len {
                        None => {
                            tracing::warn!(width, height, "cursor: dimension overflow");
                            None
                        }
                        Some(buf_len) => {
                            let mut buf = vec![0u8; buf_len];
                            gl::PixelStorei(gl::PACK_ALIGNMENT, 4);
                            gl::ReadPixels(
                                0,
                                0,
                                width as i32,
                                height as i32,
                                gl::RGBA,
                                gl::UNSIGNED_BYTE,
                                buf.as_mut_ptr() as *mut c_void,
                            );
                            gl::Finish();
                            let read_err = gl::GetError();
                            if read_err != gl::NO_ERROR {
                                tracing::warn!(
                                    err = format!("0x{:X}", read_err),
                                    "cursor: glReadPixels error"
                                );
                                None
                            } else {
                                // XRGB8888 dmabufs: the EGL_LINUX_DMA_BUF X-format
                                // spec guarantees the GL sampler returns alpha=1.0
                                // for opaque formats, so no explicit 0xFF fixup is
                                // needed here (unlike the SHM path where the X byte
                                // is undefined and must be forced to 0xFF by the
                                // caller).
                                Some(flip_rows_rgba(&buf, width, height))
                            }
                        }
                    }
                }
            }
        };

        // Per-call teardown on EVERY path. The FBO is reused (kept). Unbind the
        // texture from the FBO color attachment before deleting it.
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, self.fbo);
            gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, 0, 0);
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            if texture != 0 {
                gl::DeleteTextures(1, &texture);
            }
            (self.destroy_image)(self.display.as_ptr(), image);
        }

        result
    }
}

impl Drop for EglGl {
    fn drop(&mut self) {
        use crate::dmabuf::gbm_ffi;
        unsafe {
            // Make our context current to delete the FBO, then release it.
            if self
                .egl
                .make_current(self.display, None, None, Some(self.context))
                .is_ok()
                && self.fbo != 0
            {
                gl::DeleteFramebuffers(1, &self.fbo);
            }
            self.egl.make_current(self.display, None, None, None).ok();
            self.egl.destroy_context(self.display, self.context).ok();
            self.egl.terminate(self.display).ok();
            if !self.gbm.is_null() {
                gbm_ffi::gbm_device_destroy(self.gbm as *mut _);
            }
            if self.drm_fd >= 0 {
                libc::close(self.drm_fd);
            }
        }
    }
}

/// Tracks the last-read cursor buffer identity so we only read back on a real
/// shape change (cursor commits arrive on every cursor move otherwise).
#[derive(Default)]
pub struct CursorShapeTracker {
    last: Option<u64>,
}

impl CursorShapeTracker {
    /// Returns true if `buffer_id` differs from the last one seen.
    pub fn changed(&mut self, buffer_id: u64) -> bool {
        if self.last == Some(buffer_id) {
            false
        } else {
            self.last = Some(buffer_id);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_latest_image_and_pos() {
        let ch = CursorChannel::new();
        ch.push_image(CursorImage {
            w: 0,
            h: 0,
            hot_x: 0,
            hot_y: 0,
            rgba: vec![],
        });
        ch.push_pos(CursorPos {
            x: 1.0,
            y: 2.0,
            seq: 3,
        });
        let drained = ch.drain();
        assert_eq!(drained.len(), 2);
    }

    #[test]
    fn channel_image_collapses_to_latest() {
        let ch = CursorChannel::new();
        ch.push_image(CursorImage {
            w: 1,
            h: 1,
            hot_x: 0,
            hot_y: 0,
            rgba: vec![1, 2, 3, 4],
        });
        ch.push_image(CursorImage {
            w: 2,
            h: 2,
            hot_x: 0,
            hot_y: 0,
            rgba: vec![0; 16],
        });
        let imgs: Vec<_> = ch
            .drain()
            .into_iter()
            .filter_map(|u| match u {
                CursorUpdate::Image(i) => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].w, 2); // newest wins
    }

    #[test]
    fn channel_pos_collapses_to_latest() {
        let ch = CursorChannel::new();
        ch.push_pos(CursorPos {
            x: 1.0,
            y: 1.0,
            seq: 1,
        });
        ch.push_pos(CursorPos {
            x: 2.0,
            y: 2.0,
            seq: 2,
        });
        let drained = ch.drain();
        let positions: Vec<_> = drained
            .iter()
            .filter_map(|u| match u {
                CursorUpdate::Pos(p) => Some(*p),
                _ => None,
            })
            .collect();
        assert_eq!(
            positions,
            vec![CursorPos {
                x: 2.0,
                y: 2.0,
                seq: 2
            }]
        );
    }

    #[test]
    fn shm_argb_to_rgba_swaps_channels() {
        // One pixel ARGB8888 little-endian stored as B,G,R,A bytes = 10,20,30,40
        let src = [10u8, 20, 30, 40];
        let out = argb8888_bytes_to_rgba(&src, 1, 1, 4);
        // RGBA = R,G,B,A = 30,20,10,40
        assert_eq!(out, vec![30, 20, 10, 40]);
    }

    #[test]
    fn argb_to_rgba_short_src_returns_zeroes() {
        // Request 1×1 but src is too short — should return zeroed output, not panic.
        let src = [10u8, 20, 30]; // only 3 bytes, need 4
        let out = argb8888_bytes_to_rgba(&src, 1, 1, 4);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    /// XRGB8888 SHM cursors: the X byte is undefined and must NOT be used as
    /// alpha (a zero X → fully-transparent cursor → invisible). After calling
    /// `argb8888_bytes_to_rgba`, the caller forces every alpha byte to 0xFF
    /// when `opaque == true` (returned by `cursor_shm_argb_bytes` for XRGB).
    /// This test asserts that post-fixup no pixel has alpha == 0.
    #[test]
    fn xrgb_cursor_alpha_forced_opaque() {
        // Two pixels: XRGB with X=0 (would be transparent if used as alpha).
        // Stored little-endian as B, G, R, X = [10, 20, 30, 0, 50, 60, 70, 0].
        let src = [10u8, 20, 30, 0, 50, 60, 70, 0];
        let mut rgba = argb8888_bytes_to_rgba(&src, 2, 1, 8);
        // Simulate the caller-side XRGB opaque fixup.
        let opaque = true;
        if opaque {
            for a in rgba.chunks_exact_mut(4) {
                a[3] = 0xFF;
            }
        }
        // Channels: pixel 0 → R=30, G=20, B=10, A=0xFF; pixel 1 → R=70, G=60, B=50, A=0xFF.
        assert_eq!(rgba, vec![30, 20, 10, 0xFF, 70, 60, 50, 0xFF]);
    }

    #[test]
    fn shape_dedup_skips_identical_buffers() {
        let mut tracker = CursorShapeTracker::default();
        assert!(tracker.changed(7));
        assert!(!tracker.changed(7));
        assert!(tracker.changed(9));
    }

    #[test]
    fn vertical_flip_rgba_reverses_rows() {
        // 1x2 image, rows [A,B] -> [B,A]
        let src = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let out = flip_rows_rgba(&src, 1, 2);
        assert_eq!(out, vec![2, 2, 2, 2, 1, 1, 1, 1]);
    }

    #[test]
    fn dmabuf_reader_new_is_lazy_and_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DmabufCursorReader>();
        let _r = DmabufCursorReader::new(); // must NOT touch EGL/GL
    }
}
