// SPDX-License-Identifier: Apache-2.0

//! In-process zero-copy-capable VA-API H.264 encoder (AMD VCN via Mesa).
//!
//! This is the in-process counterpart to the ffmpeg-subprocess VA-API path in
//! `viewer/encoder.rs::run_nvenc_encoder`. Instead of `spawn`ing `ffmpeg
//! -c:v h264_vaapi …` and shuttling NV12 over a pipe, we drive libavcodec's
//! `h264_vaapi` encoder directly from this process. Two wins over the
//! subprocess:
//!
//! 1. **Working adaptive rate control.** The subprocess path's bitrate is
//!    fixed at spawn and the GCC `target_bitrate` it receives is a no-op
//!    (ffmpeg can't be reconfigured over the pipe). Here we hold the
//!    `AVCodecContext` and can retune `bit_rate`/`rc_max_rate` per frame to
//!    track the live send-side bandwidth estimate so the stream never floods
//!    a cellular link.
//! 2. **No subprocess** — no pipe copy, no process management, on-demand IDR
//!    (PLI) honored per frame via `pict_type`.
//!
//! Color path (this file, stage A): the caller hands us tightly-packed NV12
//! (produced by `viewer::encoder::task_to_nv12`, which de-strides the BGRA
//! dmabuf and converts on CPU). We upload it to a VA-API surface via
//! `av_hwframe_transfer_data` and encode. A later stage imports the BGRA
//! dmabuf directly (`AV_PIX_FMT_DRM_PRIME` → VA-API map → on-GPU NV12) to
//! drop the readback entirely; see `encode_dmabuf`.
//!
//! Mirrors `ffv1_vk_record.rs` for the libav hwcontext lifecycle.

use std::ffi::{c_void, CStr, CString};
use std::ptr;

use ffmpeg_sys_next as ff;

/// One encoded access unit (Annex-B, in-band SPS/PPS on IDRs).
pub struct EncodedNal {
    pub data: Vec<u8>,
    #[allow(dead_code)]
    pub is_keyframe: bool,
}

/// libavcodec `h264_vaapi` encoder bound to a VA-API device on a DRM render
/// node. Created once per viewer; width/height are fixed for its lifetime.
pub struct VaapiH264Encoder {
    width: u32,
    height: u32,
    render_node: String,
    hw_device_ref: *mut ff::AVBufferRef,
    hw_frames_ref: *mut ff::AVBufferRef,
    codec_ctx: *mut ff::AVCodecContext,
    /// Monotonic pts source is the caller's pts_us; this is just a counter for
    /// logging / first-frame detection.
    frames_sent: u64,
    /// Last NV12 frame (tightly packed) so a tick with no new content can
    /// re-emit it as a tiny P-frame and hold the wire cadence.
    last_nv12: Option<Vec<u8>>,
    /// Current peak cap (bps), so reconfigure can skip no-op churn.
    cur_peak_bps: u32,
    /// GOP length, kept so a reconfigure-driven reopen rebuilds an identical
    /// context save for the bitrate.
    gop: u32,
    /// Target frame rate (for rate-control timing); kept for reopen.
    fps: u32,
    /// When the codec ctx was last reopened — VA-API bakes rate control at
    /// `avcodec_open2`, so a ceiling change requires a reopen (brief stall +
    /// fresh IDR). Debounced against this to avoid thrashing on GCC jitter.
    last_reconfig: std::time::Instant,
    /// Lazily-built zero-copy dmabuf→NV12 filter graph (DRM_PRIME import →
    /// hwmap onto our VA-API device → on-GPU scale_vaapi to NV12). `None`
    /// until the first dmabuf frame; `Some(None)` once import proved
    /// unsupported on this host (→ permanent readback fallback).
    importer: Option<Option<DmabufFilter>>,
}

// The raw libav pointers are owned solely by this struct and only touched from
// the single encoder thread that owns it.
unsafe impl Send for VaapiH264Encoder {}

impl VaapiH264Encoder {
    /// Open the encoder on `render_node` (e.g. `/dev/dri/renderD128`).
    ///
    /// `peak_bps` is the VBR ceiling; the target average is set to half so a
    /// near-static desktop collapses toward nothing while motion bursts to the
    /// cap (the shape validated for cellular in the subprocess path).
    pub fn open(
        render_node: &str,
        width: u32,
        height: u32,
        peak_bps: u32,
        gop: u32,
        fps: u32,
    ) -> Option<Self> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            tracing::warn!(width, height, "vaapi-inproc: needs even dims");
            return None;
        }
        let peak_bps = peak_bps.max(500_000);

        unsafe {
            // ── 1. VA-API device from the DRM render node ──
            let mut hw_device_ref: *mut ff::AVBufferRef = ptr::null_mut();
            let node = CString::new(render_node).ok()?;
            let r = ff::av_hwdevice_ctx_create(
                &mut hw_device_ref,
                ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                node.as_ptr(),
                ptr::null_mut(),
                0,
            );
            if r < 0 || hw_device_ref.is_null() {
                tracing::warn!(err = %averror(r), node = render_node, "vaapi-inproc: av_hwdevice_ctx_create(VAAPI) failed");
                return None;
            }

            // ── 2. Frame pool: VA-API surfaces backing NV12 ──
            let hw_frames_ref = ff::av_hwframe_ctx_alloc(hw_device_ref);
            if hw_frames_ref.is_null() {
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
                tracing::warn!("vaapi-inproc: av_hwframe_ctx_alloc failed");
                return None;
            }
            {
                let fc = (*hw_frames_ref).data as *mut ff::AVHWFramesContext;
                (*fc).format = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
                (*fc).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV12;
                (*fc).width = width as i32;
                (*fc).height = height as i32;
                (*fc).initial_pool_size = 8;
            }
            let r = ff::av_hwframe_ctx_init(hw_frames_ref);
            if r < 0 {
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
                tracing::warn!(err = %averror(r), "vaapi-inproc: av_hwframe_ctx_init failed");
                return None;
            }

            // ── 3. h264_vaapi codec context ──
            let codec_ctx =
                match build_h264_vaapi_ctx(hw_frames_ref, width, height, peak_bps, gop, fps) {
                    Some(c) => c,
                    None => {
                        ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                        ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
                        return None;
                    }
                };

            tracing::info!(
                width,
                height,
                peak_bps,
                gop,
                "vaapi-inproc: h264_vaapi encoder opened"
            );
            Some(Self {
                width,
                height,
                render_node: render_node.to_string(),
                hw_device_ref,
                hw_frames_ref,
                codec_ctx,
                frames_sent: 0,
                last_nv12: None,
                cur_peak_bps: peak_bps,
                gop: gop.max(1),
                fps: fps.clamp(1, 120),
                last_reconfig: std::time::Instant::now(),
                importer: None,
            })
        }
    }

    /// Encode one tightly-packed NV12 frame (Y plane w*h, then interleaved UV
    /// w*h/2). `is_idr` forces a keyframe (PLI / first frame). Returns the
    /// Annex-B access unit, or None on a dropped/failed frame.
    pub fn encode_nv12(&mut self, nv12: &[u8], pts_us: i64, is_idr: bool) -> Option<EncodedNal> {
        let need = (self.width as usize) * (self.height as usize) * 3 / 2;
        if nv12.len() != need {
            tracing::warn!(
                got = nv12.len(),
                need,
                "vaapi-inproc: bad NV12 size; dropping"
            );
            return None;
        }
        // Cache for reencode_last (idle keepalive / fence-not-ready hold).
        if self.last_nv12.as_deref() != Some(nv12) {
            self.last_nv12 = Some(nv12.to_vec());
        }
        self.encode_nv12_inner(nv12, pts_us, is_idr)
    }

    /// Re-encode the last frame to hold the wire cadence when no new content
    /// arrived this tick (a tiny P-frame for static content).
    pub fn reencode_last(&mut self, pts_us: i64, is_idr: bool) -> Option<EncodedNal> {
        let last = self.last_nv12.take()?;
        let out = self.encode_nv12_inner(&last, pts_us, is_idr);
        self.last_nv12 = Some(last);
        out
    }

    /// True once the zero-copy dmabuf path has been confirmed unusable on this
    /// host (so the caller stops handing us dmabufs and reads back instead).
    pub fn zero_copy_unavailable(&self) -> bool {
        matches!(self.importer, Some(None))
    }

    /// Zero-copy encode: import the BGRA dmabuf as a VA-API surface (no GPU→CPU
    /// readback), convert BGRA→NV12 on the GPU via `scale_vaapi`, and encode.
    /// Returns None if the import/convert path is unsupported on this host —
    /// the caller should then fall back to `encode_nv12` (readback).
    ///
    /// `drm_format` is the DRM FourCC (e.g. 0x34325241 = ARGB8888), `modifier`
    /// the format modifier, `stride`/`offset` the plane-0 layout.
    // encoder/cursor setup takes many tightly-related params by design
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dmabuf(
        &mut self,
        fd: i32,
        drm_format: u32,
        modifier: u64,
        width: u32,
        height: u32,
        stride: u32,
        offset: u32,
        pts_us: i64,
        is_idr: bool,
    ) -> Option<EncodedNal> {
        if width != self.width || height != self.height {
            return None;
        }
        // Zero-copy VA-API import currently lacks a producer-sync barrier for
        // explicit-sync compositors (KWin via wp_linux_drm_syncobj): the
        // libavfilter import reads the teed dmabuf after KWin may have recycled
        // it, yielding mostly-black frames. Vulkan avoids this with a
        // QUEUE_FAMILY_FOREIGN_EXT barrier; we have no equivalent yet. So the
        // zero-copy path is OPT-IN (WAYMUX_VAAPI_ZEROCOPY=1) until that's fixed;
        // by default we disable it and the caller falls back to the EGL
        // readback (task_to_nv12), which synchronizes correctly.
        if self.importer.is_none() && std::env::var("WAYMUX_VAAPI_ZEROCOPY").as_deref() != Ok("1") {
            self.importer = Some(None);
        }
        // Lazily build the import+convert filter graph on first dmabuf.
        if self.importer.is_none() {
            let built = unsafe {
                DmabufFilter::new(
                    &self.render_node,
                    self.hw_device_ref,
                    self.width,
                    self.height,
                    drm_format,
                )
            };
            if built.is_none() {
                tracing::warn!(
                    "vaapi-inproc: zero-copy dmabuf import unavailable; using readback fallback"
                );
            } else {
                tracing::info!("vaapi-inproc: zero-copy dmabuf import ENABLED (no readback)");
            }
            self.importer = Some(built);
        }
        let filt = self.importer.as_mut().unwrap().as_mut()?;

        unsafe {
            // Map the dmabuf to a VA-API NV12 frame on the GPU.
            let nv12_vaapi = match filt.map_to_nv12(fd, drm_format, modifier, stride, offset) {
                Some(f) => f,
                None => {
                    // One failure → disable zero-copy permanently and fall back.
                    tracing::warn!(
                        "vaapi-inproc: dmabuf map failed; disabling zero-copy, using readback"
                    );
                    self.importer = Some(None);
                    return None;
                }
            };
            let _g = FrameGuard(nv12_vaapi);
            (*nv12_vaapi).pts = pts_us;
            if is_idr {
                (*nv12_vaapi).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
            }
            self.send_and_drain(nv12_vaapi)
        }
    }

    fn encode_nv12_inner(&mut self, nv12: &[u8], pts_us: i64, is_idr: bool) -> Option<EncodedNal> {
        unsafe {
            // ── SW NV12 frame ──
            let sw = ff::av_frame_alloc();
            if sw.is_null() {
                return None;
            }
            let _sw_guard = FrameGuard(sw);
            (*sw).format = ff::AVPixelFormat::AV_PIX_FMT_NV12 as i32;
            (*sw).width = self.width as i32;
            (*sw).height = self.height as i32;
            if ff::av_frame_get_buffer(sw, 0) < 0 {
                tracing::warn!("vaapi-inproc: av_frame_get_buffer(NV12) failed");
                return None;
            }
            // Copy Y then UV row-by-row, honoring libav's (aligned) linesize.
            let w = self.width as usize;
            let h = self.height as usize;
            let y_src = &nv12[..w * h];
            let uv_src = &nv12[w * h..];
            let ls0 = (*sw).linesize[0] as usize;
            let ls1 = (*sw).linesize[1] as usize;
            let d0 = (*sw).data[0];
            let d1 = (*sw).data[1];
            for row in 0..h {
                ptr::copy_nonoverlapping(y_src.as_ptr().add(row * w), d0.add(row * ls0), w);
            }
            for row in 0..(h / 2) {
                ptr::copy_nonoverlapping(uv_src.as_ptr().add(row * w), d1.add(row * ls1), w);
            }

            // ── HW VA-API surface + upload ──
            let hw = ff::av_frame_alloc();
            if hw.is_null() {
                return None;
            }
            let _hw_guard = FrameGuard(hw);
            if ff::av_hwframe_get_buffer(self.hw_frames_ref, hw, 0) < 0 {
                tracing::warn!("vaapi-inproc: av_hwframe_get_buffer failed");
                return None;
            }
            if ff::av_hwframe_transfer_data(hw, sw, 0) < 0 {
                tracing::warn!("vaapi-inproc: av_hwframe_transfer_data failed");
                return None;
            }
            (*hw).pts = pts_us;
            if is_idr {
                (*hw).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
            }

            self.send_and_drain(hw)
        }
    }

    /// send_frame + receive_packet drain, concatenating all NALUs of this AU.
    unsafe fn send_and_drain(&mut self, frame: *mut ff::AVFrame) -> Option<EncodedNal> {
        let r = ff::avcodec_send_frame(self.codec_ctx, frame);
        if r < 0 {
            tracing::warn!(err = %averror(r), "vaapi-inproc: avcodec_send_frame failed");
            return None;
        }
        self.frames_sent += 1;

        let mut out: Vec<u8> = Vec::new();
        let mut is_key = false;
        loop {
            let pkt = ff::av_packet_alloc();
            if pkt.is_null() {
                break;
            }
            let r = ff::avcodec_receive_packet(self.codec_ctx, pkt);
            if r == ff::AVERROR(ff::EAGAIN) || r == ff::AVERROR_EOF {
                ff::av_packet_free(&mut (pkt as *mut _));
                break;
            }
            if r < 0 {
                tracing::warn!(err = %averror(r), "vaapi-inproc: avcodec_receive_packet failed");
                ff::av_packet_free(&mut (pkt as *mut _));
                break;
            }
            let p = (*pkt).data;
            let n = (*pkt).size as usize;
            if !p.is_null() && n > 0 {
                out.extend_from_slice(std::slice::from_raw_parts(p, n));
            }
            if (*pkt).flags & ff::AV_PKT_FLAG_KEY != 0 {
                is_key = true;
            }
            ff::av_packet_free(&mut (pkt as *mut _));
        }
        if out.is_empty() {
            // Encoder buffered this frame (async pipeline) — no AU yet.
            return None;
        }
        Some(EncodedNal {
            data: out,
            is_keyframe: is_key,
        })
    }

    /// Retune the VBR ceiling to the live GCC estimate. VA-API bakes its rate
    /// control at `avcodec_open2`, so this reopens the codec context (reusing
    /// the existing device + frame pool). The reopened encoder's first frame
    /// is a fresh IDR (in-band SPS/PPS), which the browser re-syncs on.
    ///
    /// Debounced: ignores <20% changes and reopens at most once per second so
    /// GCC jitter can't thrash the encoder with stalls. Returns true if it
    /// actually reopened.
    pub fn reconfigure_bitrate(&mut self, peak_bps: u32) -> bool {
        let peak_bps = peak_bps.max(500_000);
        let delta = peak_bps.abs_diff(self.cur_peak_bps);
        if delta * 100 < self.cur_peak_bps * 20 {
            return false;
        }
        if self.last_reconfig.elapsed() < std::time::Duration::from_millis(1000) {
            return false;
        }
        unsafe {
            let new_ctx = match build_h264_vaapi_ctx(
                self.hw_frames_ref,
                self.width,
                self.height,
                peak_bps,
                self.gop,
                self.fps,
            ) {
                Some(c) => c,
                None => {
                    tracing::warn!(
                        peak_bps,
                        "vaapi-inproc: reopen for reconfigure failed; keeping old ceiling"
                    );
                    return false;
                }
            };
            ff::avcodec_free_context(&mut (self.codec_ctx as *mut _));
            self.codec_ctx = new_ctx;
        }
        self.cur_peak_bps = peak_bps;
        self.last_reconfig = std::time::Instant::now();
        tracing::info!(
            peak_bps,
            "vaapi-inproc: reopened encoder at new VBR ceiling"
        );
        true
    }
}

/// Build + open an `h264_vaapi` AVCodecContext bound to `hw_frames_ref`,
/// configured for capped VBR (avg = peak/2), no B-frames, in-band SPS/PPS.
/// Returns an opened context, or None on any failure.
unsafe fn build_h264_vaapi_ctx(
    hw_frames_ref: *mut ff::AVBufferRef,
    width: u32,
    height: u32,
    peak_bps: u32,
    gop: u32,
    fps: u32,
) -> Option<*mut ff::AVCodecContext> {
    let name = CString::new("h264_vaapi").unwrap();
    let codec = ff::avcodec_find_encoder_by_name(name.as_ptr());
    if codec.is_null() {
        tracing::warn!("vaapi-inproc: h264_vaapi encoder not found");
        return None;
    }
    let codec_ctx = ff::avcodec_alloc_context3(codec);
    if codec_ctx.is_null() {
        tracing::warn!("vaapi-inproc: avcodec_alloc_context3 failed");
        return None;
    }
    (*codec_ctx).width = width as i32;
    (*codec_ctx).height = height as i32;
    (*codec_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
    (*codec_ctx).time_base = ff::AVRational {
        num: 1,
        den: 1_000_000,
    };
    (*codec_ctx).framerate = ff::AVRational {
        num: fps.clamp(1, 120) as i32,
        den: 1,
    };
    (*codec_ctx).gop_size = gop.max(1) as i32;
    (*codec_ctx).max_b_frames = 0;
    // Capped VBR: average at half the cap, peak at the cap.
    (*codec_ctx).bit_rate = (peak_bps / 2) as i64;
    (*codec_ctx).rc_max_rate = peak_bps as i64;
    (*codec_ctx).rc_buffer_size = (peak_bps / 2) as i32; // ~0.5s VBV
    (*codec_ctx).hw_frames_ctx = ff::av_buffer_ref(hw_frames_ref);
    // In-band SPS/PPS (do NOT set GLOBAL_HEADER) so every IDR is
    // self-contained for the browser, like the subprocess path.

    let mut opts: *mut ff::AVDictionary = ptr::null_mut();
    let set = |opts: &mut *mut ff::AVDictionary, k: &str, v: &str| {
        let k = CString::new(k).unwrap();
        let v = CString::new(v).unwrap();
        ff::av_dict_set(opts, k.as_ptr(), v.as_ptr(), 0);
    };
    set(&mut opts, "rc_mode", "VBR");
    set(&mut opts, "async_depth", "1"); // minimize encode latency

    let r = ff::avcodec_open2(codec_ctx, codec, &mut opts);
    ff::av_dict_free(&mut opts);
    if r < 0 {
        let msg = averror(r);
        ff::avcodec_free_context(&mut (codec_ctx as *mut _));
        tracing::warn!(err = %msg, "vaapi-inproc: avcodec_open2(h264_vaapi) failed");
        return None;
    }
    Some(codec_ctx)
}

impl Drop for VaapiH264Encoder {
    fn drop(&mut self) {
        unsafe {
            if !self.codec_ctx.is_null() {
                ff::avcodec_free_context(&mut (self.codec_ctx as *mut _));
            }
            if !self.hw_frames_ref.is_null() {
                ff::av_buffer_unref(&mut (self.hw_frames_ref as *mut _));
            }
            if !self.hw_device_ref.is_null() {
                ff::av_buffer_unref(&mut (self.hw_device_ref as *mut _));
            }
        }
    }
}

/// Zero-copy dmabuf → NV12 importer: a libavfilter graph that imports a
/// DRM_PRIME (BGRA) dmabuf onto our VA-API device and color-converts to NV12
/// entirely on the GPU (`scale_vaapi`). No GPU→CPU readback.
struct DmabufFilter {
    drm_device_ref: *mut ff::AVBufferRef,
    drm_frames_ref: *mut ff::AVBufferRef,
    graph: *mut ff::AVFilterGraph,
    src: *mut ff::AVFilterContext,
    sink: *mut ff::AVFilterContext,
    width: u32,
    height: u32,
}
unsafe impl Send for DmabufFilter {}

/// Free callback for the heap `AVDRMFrameDescriptor` carried in a DRM_PRIME
/// frame's buf[0]: closes the dup'd dmabuf fd and frees the box.
unsafe extern "C" fn drm_desc_free(_opaque: *mut c_void, data: *mut u8) {
    if data.is_null() {
        return;
    }
    let desc = data as *mut ff::AVDRMFrameDescriptor;
    let n = (*desc).nb_objects.max(0) as usize;
    for i in 0..n.min(4) {
        let fd = (*desc).objects[i].fd;
        if fd >= 0 {
            libc::close(fd);
        }
    }
    drop(Box::from_raw(desc));
}

impl DmabufFilter {
    /// Build the import+convert graph: buffer(DRM_PRIME) → hwmap(onto our
    /// VA-API device) → scale_vaapi(format=nv12) → buffersink. Returns None if
    /// any stage is unavailable on this host (caller falls back to readback).
    unsafe fn new(
        render_node: &str,
        vaapi_device_ref: *mut ff::AVBufferRef,
        width: u32,
        height: u32,
        _drm_format: u32,
    ) -> Option<Self> {
        // ── DRM source device + DRM_PRIME frames ctx ──
        let mut drm_device_ref: *mut ff::AVBufferRef = ptr::null_mut();
        let node = CString::new(render_node).ok()?;
        if ff::av_hwdevice_ctx_create(
            &mut drm_device_ref,
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM,
            node.as_ptr(),
            ptr::null_mut(),
            0,
        ) < 0
            || drm_device_ref.is_null()
        {
            tracing::warn!("vaapi-inproc: av_hwdevice_ctx_create(DRM) failed");
            return None;
        }
        let drm_frames_ref = ff::av_hwframe_ctx_alloc(drm_device_ref);
        if drm_frames_ref.is_null() {
            ff::av_buffer_unref(&mut (drm_device_ref as *mut _));
            return None;
        }
        {
            let fc = (*drm_frames_ref).data as *mut ff::AVHWFramesContext;
            (*fc).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME;
            (*fc).sw_format = ff::AVPixelFormat::AV_PIX_FMT_BGR0; // ARGB8888 LE, alpha ignored
            (*fc).width = width as i32;
            (*fc).height = height as i32;
            (*fc).initial_pool_size = 0; // we supply external buffers
        }
        if ff::av_hwframe_ctx_init(drm_frames_ref) < 0 {
            ff::av_buffer_unref(&mut (drm_frames_ref as *mut _));
            ff::av_buffer_unref(&mut (drm_device_ref as *mut _));
            tracing::warn!("vaapi-inproc: DRM frames ctx init failed");
            return None;
        }

        // ── filter graph ──
        let graph = ff::avfilter_graph_alloc();
        if graph.is_null() {
            ff::av_buffer_unref(&mut (drm_frames_ref as *mut _));
            ff::av_buffer_unref(&mut (drm_device_ref as *mut _));
            return None;
        }
        let cleanup = |graph: *mut ff::AVFilterGraph,
                       drm_frames_ref: *mut ff::AVBufferRef,
                       drm_device_ref: *mut ff::AVBufferRef| {
            let mut g = graph;
            ff::avfilter_graph_free(&mut g);
            let mut f = drm_frames_ref;
            ff::av_buffer_unref(&mut f);
            let mut d = drm_device_ref;
            ff::av_buffer_unref(&mut d);
        };

        let f_buffer = ff::avfilter_get_by_name(c"buffer".as_ptr());
        let f_sink = ff::avfilter_get_by_name(c"buffersink".as_ptr());
        let f_hwmap = ff::avfilter_get_by_name(c"hwmap".as_ptr());
        let f_scale = ff::avfilter_get_by_name(c"scale_vaapi".as_ptr());
        if f_buffer.is_null() || f_sink.is_null() || f_hwmap.is_null() || f_scale.is_null() {
            tracing::warn!(
                "vaapi-inproc: a required filter (buffer/buffersink/hwmap/scale_vaapi) is missing"
            );
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }

        // buffersrc: DRM_PRIME input. For a hwaccel source the hw_frames_ctx
        // must be set (via parameters) BEFORE the filter is initialized — so
        // alloc + parameters_set + init_str rather than create_filter (which
        // inits immediately, before params can be attached).
        let src = ff::avfilter_graph_alloc_filter(graph, f_buffer, c"in".as_ptr());
        if src.is_null() {
            tracing::warn!("vaapi-inproc: alloc buffersrc failed");
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }
        let par = ff::av_buffersrc_parameters_alloc();
        if par.is_null() {
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }
        (*par).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
        (*par).width = width as i32;
        (*par).height = height as i32;
        (*par).time_base = ff::AVRational {
            num: 1,
            den: 1_000_000,
        };
        // parameters_set makes its own ref to hw_frames_ctx; borrow (no
        // av_buffer_ref) so freeing `par` doesn't leak a reference.
        (*par).hw_frames_ctx = drm_frames_ref;
        let pr = ff::av_buffersrc_parameters_set(src, par);
        ff::av_free(par as *mut c_void);
        if pr < 0 {
            tracing::warn!("vaapi-inproc: av_buffersrc_parameters_set failed");
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }
        if ff::avfilter_init_str(src, ptr::null()) < 0 {
            tracing::warn!("vaapi-inproc: buffersrc init failed");
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }

        // hwmap onto OUR VA-API device (so scale_vaapi + the encoder share it).
        let mut hwmap: *mut ff::AVFilterContext = ptr::null_mut();
        if ff::avfilter_graph_create_filter(
            &mut hwmap,
            f_hwmap,
            c"hwmap".as_ptr(),
            c"mode=read".as_ptr(),
            ptr::null_mut(),
            graph,
        ) < 0
        {
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }
        (*hwmap).hw_device_ctx = ff::av_buffer_ref(vaapi_device_ref);

        // scale_vaapi → NV12 (BGRA→NV12 CSC on the GPU).
        let mut scale: *mut ff::AVFilterContext = ptr::null_mut();
        if ff::avfilter_graph_create_filter(
            &mut scale,
            f_scale,
            c"scale".as_ptr(),
            c"format=nv12".as_ptr(),
            ptr::null_mut(),
            graph,
        ) < 0
        {
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }

        let mut sink: *mut ff::AVFilterContext = ptr::null_mut();
        if ff::avfilter_graph_create_filter(
            &mut sink,
            f_sink,
            c"out".as_ptr(),
            ptr::null(),
            ptr::null_mut(),
            graph,
        ) < 0
        {
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }

        if ff::avfilter_link(src, 0, hwmap, 0) < 0
            || ff::avfilter_link(hwmap, 0, scale, 0) < 0
            || ff::avfilter_link(scale, 0, sink, 0) < 0
        {
            tracing::warn!("vaapi-inproc: filter link failed");
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }
        if ff::avfilter_graph_config(graph, ptr::null_mut()) < 0 {
            tracing::warn!("vaapi-inproc: avfilter_graph_config failed (tiled modifier likely unsupported by VA-API)");
            cleanup(graph, drm_frames_ref, drm_device_ref);
            return None;
        }

        Some(Self {
            drm_device_ref,
            drm_frames_ref,
            graph,
            src,
            sink,
            width,
            height,
        })
    }

    /// Push one dmabuf through the graph; pull the resulting VA-API NV12 frame.
    /// The returned frame is owned by the caller (must av_frame_free it).
    unsafe fn map_to_nv12(
        &mut self,
        fd: i32,
        drm_format: u32,
        modifier: u64,
        stride: u32,
        offset: u32,
    ) -> Option<*mut ff::AVFrame> {
        let frame = ff::av_frame_alloc();
        if frame.is_null() {
            return None;
        }
        (*frame).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
        (*frame).width = self.width as i32;
        (*frame).height = self.height as i32;

        // Heap descriptor owned by buf[0] (freed via drm_desc_free).
        let mut desc: Box<ff::AVDRMFrameDescriptor> = Box::new(std::mem::zeroed());
        let dupfd = libc::dup(fd);
        if dupfd < 0 {
            let mut f = frame;
            ff::av_frame_free(&mut f);
            return None;
        }
        desc.nb_objects = 1;
        desc.objects[0].fd = dupfd;
        // Real allocation size: Mesa's DRM_PRIME importer needs a non-zero
        // object size or it maps a zero-extent buffer (→ all-black surface).
        // lseek(SEEK_END) on a dma-buf fd returns its true size (incl. tiling
        // padding); fall back to a tight plane estimate if that fails.
        let sz = libc::lseek(dupfd, 0, libc::SEEK_END);
        desc.objects[0].size = if sz > 0 {
            sz as usize
        } else {
            (self.height as usize) * (stride as usize)
        };
        libc::lseek(dupfd, 0, libc::SEEK_SET);
        desc.objects[0].format_modifier = modifier;
        desc.nb_layers = 1;
        desc.layers[0].format = drm_format;
        desc.layers[0].nb_planes = 1;
        desc.layers[0].planes[0].object_index = 0;
        desc.layers[0].planes[0].offset = offset as isize;
        desc.layers[0].planes[0].pitch = stride as isize;
        let desc_ptr = Box::into_raw(desc);

        (*frame).data[0] = desc_ptr as *mut u8;
        let buf = ff::av_buffer_create(
            desc_ptr as *mut u8,
            std::mem::size_of::<ff::AVDRMFrameDescriptor>(),
            Some(drm_desc_free),
            ptr::null_mut(),
            0,
        );
        if buf.is_null() {
            // Reclaim manually (buffer didn't take ownership).
            drm_desc_free(ptr::null_mut(), desc_ptr as *mut u8);
            let mut f = frame;
            ff::av_frame_free(&mut f);
            return None;
        }
        (*frame).buf[0] = buf;
        (*frame).hw_frames_ctx = ff::av_buffer_ref(self.drm_frames_ref);

        // Push (transfers our frame ref into the graph).
        let r = ff::av_buffersrc_add_frame_flags(self.src, frame, 0);
        let mut f = frame;
        ff::av_frame_free(&mut f);
        if r < 0 {
            tracing::warn!(err = %averror(r), "vaapi-inproc: av_buffersrc_add_frame failed");
            return None;
        }

        let out = ff::av_frame_alloc();
        if out.is_null() {
            return None;
        }
        let r = ff::av_buffersink_get_frame(self.sink, out);
        if r < 0 {
            let mut o = out;
            ff::av_frame_free(&mut o);
            return None;
        }
        Some(out)
    }
}

impl Drop for DmabufFilter {
    fn drop(&mut self) {
        unsafe {
            if !self.graph.is_null() {
                ff::avfilter_graph_free(&mut self.graph);
            }
            if !self.drm_frames_ref.is_null() {
                ff::av_buffer_unref(&mut (self.drm_frames_ref as *mut _));
            }
            if !self.drm_device_ref.is_null() {
                ff::av_buffer_unref(&mut (self.drm_device_ref as *mut _));
            }
        }
    }
}

/// RAII guard that frees an AVFrame on scope exit.
struct FrameGuard(*mut ff::AVFrame);
impl Drop for FrameGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ff::av_frame_free(&mut self.0) };
        }
    }
}

fn averror(code: i32) -> String {
    let mut buf = [0i8; 256];
    unsafe { ff::av_strerror(code, buf.as_mut_ptr(), buf.len()) };
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
    cstr.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Realistic, compressible test content (gradients + moving elements +
    // detail) so the rate controller has QP headroom to respond to a ceiling
    // change — unlike pure noise, which pins QP at max and defeats RC. Built
    // once via ffmpeg `testsrc2`; returns a ring of NV12 frames.
    fn testsrc2_nv12(w: u32, h: u32, frames: u32) -> Option<Vec<Vec<u8>>> {
        let size = format!("{w}x{h}");
        let dur = format!("{:.3}", frames as f64 / 60.0);
        let out = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={size}:rate=60:duration={dur}"),
                "-pix_fmt",
                "nv12",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let fsz = (w as usize) * (h as usize) * 3 / 2;
        let v: Vec<Vec<u8>> = out.stdout.chunks_exact(fsz).map(|c| c.to_vec()).collect();
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    }

    /// Verifies mid-stream VBR-ceiling reconfiguration actually changes the
    /// VA-API encoder's output rate. Needs real VA-API hardware (/dev/dri/
    /// renderD128) + ffmpeg, so it's `#[ignore]` by default; run with:
    ///   cargo test -p waymux-session --bin waymux-session vaapi -- --ignored --nocapture
    #[test]
    #[ignore]
    fn reconfigure_bitrate_changes_output_rate() {
        let (w, h) = (1280u32, 720u32);
        let content = match testsrc2_nv12(w, h, 160) {
            Some(c) => c,
            None => {
                eprintln!("SKIP: ffmpeg testsrc2 unavailable");
                return;
            }
        };
        let frame = |i: u32| &content[(i as usize) % content.len()];

        let mut enc = match VaapiH264Encoder::open("/dev/dri/renderD128", w, h, 10_000_000, 120, 60)
        {
            Some(e) => e,
            None => {
                eprintln!("SKIP: no VA-API device");
                return;
            }
        };
        // Warm up + measure at the high ceiling (10 Mbps).
        let mut hi_bytes = 0usize;
        for i in 0..150 {
            if let Some(n) = enc.encode_nv12(frame(i), (i as i64) * 16_666, i == 0) {
                if i >= 30 {
                    hi_bytes += n.data.len();
                }
            }
        }
        // Respect the 1s reopen debounce (warmup encodes faster than that).
        std::thread::sleep(std::time::Duration::from_millis(1100));
        // Drop the ceiling to 1 Mbps and measure again over the same content.
        assert!(
            enc.reconfigure_bitrate(1_000_000),
            "reconfigure should report a change"
        );
        let mut lo_bytes = 0usize;
        for i in 150..300 {
            if let Some(n) = enc.encode_nv12(frame(i), (i as i64) * 16_666, false) {
                if i >= 180 {
                    lo_bytes += n.data.len();
                }
            }
        }
        eprintln!(
            "hi(10Mbps)={hi_bytes}B  lo(1Mbps)={lo_bytes}B  ratio={:.2}",
            hi_bytes as f64 / lo_bytes.max(1) as f64
        );
        // If the reconfigure is honored mid-stream, the low-ceiling batch must
        // be materially smaller. Require at least a 2x drop.
        assert!(
            lo_bytes * 2 < hi_bytes,
            "mid-stream ceiling change not honored: hi={hi_bytes} lo={lo_bytes}"
        );
    }
}
