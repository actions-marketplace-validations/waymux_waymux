// SPDX-License-Identifier: Apache-2.0

//! GPU zero-copy lossless recording via ffmpeg's `hevc_vulkan` encoder (HEVC RangeExt 4:4:4 lossless, QP=0).
//!
//! Architecture:
//!
//! ```text
//!   chromium dmabuf (BGRA)
//!     │
//!     │ VK_KHR_external_memory_fd
//!     ▼
//!   waymux VkImage  ──vkCmdCopyImage──>  libav's VkImage (AVVkFrame)
//!                                              │
//!                                              ▼
//!                                        hevc_vulkan encoder
//!                                              │
//!                                              ▼
//!                                       AVPacket (HEVC NAL bytes)
//!                                              │
//!                                              ▼
//!                                       MKV muxer (libav)
//! ```
//!
//! Pixels never leave GPU memory between the dmabuf import and the
//! encoder. The hop between waymux's image and libav's pool image is a
//! single `vkCmdCopyImage`, recorded on a compute-capable queue we share
//! with libav via `AVVulkanDeviceContext`.
//!
//! ffmpeg-sys-next 8.1 does not bind `libavutil/hwcontext_vulkan.h`, so the
//! Vulkan-specific hwcontext fields are poked through a small C shim
//! (`src/ffv1_vk_shim.c` (shared with the FFV1 path)).

// experimental HEVC RangeExt encoder; helpers retained for upcoming wiring
#![allow(dead_code)]

use std::ffi::{c_int, c_void, CStr, CString};
use std::ptr;

use ash::vk;
use ash::vk::Handle;
use ffmpeg_sys_next as ff;

use crate::vulkan_record::VkDeviceCtx;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct WaymuxVkFrameView {
    img: *mut c_void,
    mem: *mut c_void,
    sem: *mut c_void,
    sem_value: u64,
    layout: i32,
    access: u32,
    queue_family: u32,
    flags: u32,
    tiling: i32,
}

unsafe extern "C" {
    fn waymux_avvk_set_device(
        hw_device_ref: *mut ff::AVBufferRef,
        instance: *mut c_void,
        phys_dev: *mut c_void,
        act_dev: *mut c_void,
        get_proc_addr: *mut c_void,
        compute_qf: u32,
        encode_qf: u32,
        enabled_dev_extensions: *const *const std::os::raw::c_char,
        nb_enabled_dev_extensions: c_int,
    ) -> c_int;

    fn waymux_avvk_set_frames(
        hw_frames_ref: *mut ff::AVBufferRef,
        tiling: u32,
        extra_usage: u32,
    ) -> c_int;

    fn waymux_avvk_frame_view(avvkframe: *const c_void, out: *mut WaymuxVkFrameView, plane: c_int);

    fn waymux_avvk_frame_update(
        avvkframe: *mut c_void,
        plane: c_int,
        sem_value: u64,
        layout: i32,
        access: u32,
        queue_family: u32,
    );
}

/// One encoded packet from `hevc_vulkan`.
pub struct EncodedPacket {
    pub data: Vec<u8>,
    pub pts_us: i64,
    pub is_keyframe: bool,
}

/// Libav-side encoder bound to a shared Vulkan device.
///
/// Created once per recording. Holds:
/// - an `AVHWDeviceContext` wrapping waymux's existing `VkDeviceCtx`
///   (no duplicate Vulkan instance/device — libav uses ours)
/// - an `AVHWFramesContext` describing the frame pool format (BGRA
///   Vulkan images at `width × height`)
/// - an `AVCodecContext` opened for the `hevc_vulkan` encoder (HEVC RangeExt 4:4:4 lossless, QP=0)
pub struct HevcVkEncoder {
    /// Display (visible) width — what the caller asked for.
    width: u32,
    height: u32,
    /// Encoder picture dimensions — `width`/`height` rounded UP to
    /// the next multiple of 64. NVIDIA's `hevc_vulkan` rext encoder
    /// produces chroma artifacts when frame dimensions are not
    /// 64-aligned; pad up + decoder/consumer crops downstream.
    coded_width: u32,
    coded_height: u32,
    hw_device_ref: *mut ff::AVBufferRef,
    hw_frames_ref: *mut ff::AVBufferRef,
    codec_ctx: *mut ff::AVCodecContext,
    extradata: Vec<u8>,
    /// Compute-family command pool owned by us — we record vkCmdCopyImage
    /// here on the same queue family libav uses, so no QF ownership
    /// transfer is required between our copy and the encoder's compute
    /// shaders. The encoder picks a queue from this family internally.
    cmd_pool: vk::CommandPool,
    cmd_buffer: vk::CommandBuffer,
    /// VkDevice handle (raw) we record commands against. We don't hold the
    /// ash::Device by reference (lifetime gymnastics in a long-lived
    /// struct); the caller of `encode_one` passes a `&VkDeviceCtx` each
    /// frame, and we sanity-check it points at the same device.
    device_handle_raw: u64,
    compute_queue_family: u32,
    /// Monotonically-increasing frame index, used as the encoder's pts.
    /// First frame is 0.
    next_frame_index: i64,
}

/// Input view for `encode_one`. Caller has already ensured the image's
/// current layout/access/queue-family matches these values. After this
/// returns, the image will have been transitioned to `TRANSFER_SRC_OPTIMAL`
/// and is safe to reuse on the same queue family next frame.
pub struct FrameInputView {
    pub image: vk::Image,
    pub current_layout: vk::ImageLayout,
    pub current_access: vk::AccessFlags2,
    pub current_stage: vk::PipelineStageFlags2,
    pub current_queue_family: u32,
}

impl HevcVkEncoder {
    /// Open the encoder. Width/height must match the recording's
    /// resolution and stay constant for the lifetime of this encoder.
    ///
    /// `vk_ctx` must outlive the encoder — libav holds raw pointers
    /// into the VkInstance/VkDevice handles.
    pub fn open(vk_ctx: &VkDeviceCtx, width: u32, height: u32) -> Result<Self, String> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(format!("hevc_vulkan needs even dims, got {width}x{height}"));
        }
        // Round display dims UP to the next multiple of 64 — HEVC's
        // largest CTU size. NVIDIA's hevc_vulkan rext encoder mishandles
        // non-64-aligned widths (visible blue-triangle artifact in the
        // upper-left + green bottom stripe). Pad-up + downstream crop
        // is the workaround.
        let coded_width = crate::vulkan_record::align_to_ctu(width);
        let coded_height = crate::vulkan_record::align_to_ctu(height);

        // ── 1. Wrap our existing VkDeviceCtx as an AVHWDeviceContext ──
        let hw_device_ref =
            unsafe { ff::av_hwdevice_ctx_alloc(ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN) };
        if hw_device_ref.is_null() {
            return Err("av_hwdevice_ctx_alloc(VULKAN) returned null".to_string());
        }

        // libav's PFN_vkGetInstanceProcAddr signature matches ash's verbatim
        // (it's a Vulkan-defined ABI). We pass the function pointer via void*.
        let pfn_get_instance_proc_addr =
            vk_ctx.entry.static_fn().get_instance_proc_addr as *mut c_void;

        let ext_ptrs = vk_ctx.enabled_dev_extensions();
        let r = unsafe {
            waymux_avvk_set_device(
                hw_device_ref,
                vk_ctx.instance.handle().as_raw() as *mut c_void,
                vk_ctx.physical_device.as_raw() as *mut c_void,
                vk_ctx.device.handle().as_raw() as *mut c_void,
                pfn_get_instance_proc_addr,
                vk_ctx.compute_queue_family,
                vk_ctx.video_encode_queue_family,
                ext_ptrs.as_ptr(),
                ext_ptrs.len() as c_int,
            )
        };
        std::mem::forget(ext_ptrs);
        if r != 0 {
            unsafe { ff::av_buffer_unref(&mut (hw_device_ref as *mut _)) };
            return Err(format!("waymux_avvk_set_device returned {r}"));
        }

        let r = unsafe { ff::av_hwdevice_ctx_init(hw_device_ref) };
        if r < 0 {
            unsafe { ff::av_buffer_unref(&mut (hw_device_ref as *mut _)) };
            return Err(format!(
                "av_hwdevice_ctx_init(VULKAN) returned {}",
                averror(r)
            ));
        }

        // ── 2. Allocate the frames context (frame pool) ──
        let hw_frames_ref = unsafe { ff::av_hwframe_ctx_alloc(hw_device_ref) };
        if hw_frames_ref.is_null() {
            unsafe { ff::av_buffer_unref(&mut (hw_device_ref as *mut _)) };
            return Err("av_hwframe_ctx_alloc returned null".to_string());
        }
        unsafe {
            let frames_ctx = (*hw_frames_ref).data as *mut ff::AVHWFramesContext;
            (*frames_ctx).format = ff::AVPixelFormat::AV_PIX_FMT_VULKAN;
            (*frames_ctx).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV24;
            // CODED (64-aligned) dims — libav allocates frame-pool
            // images at this size, the encoder reads from this size,
            // and the bitstream carries this size. Consumer crops to
            // display dims via a downstream filter or scale step.
            (*frames_ctx).width = coded_width as i32;
            (*frames_ctx).height = coded_height as i32;
            // Small pool — we only need one in-flight encode at a time
            // for first-pass. Increase later if pipelining helps.
            (*frames_ctx).initial_pool_size = 4;
        }
        // tiling=0 → OPTIMAL. extra_usage=0 → just the libav defaults
        // (sampled, storage, transfer src/dst).
        // extra_usage = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR (0x4000_0000)
        // so libav allocates the frame pool images with encode-source
        // usage — otherwise the encoder's command buffer fails at
        // record-time with VK_ERROR_INITIALIZATION_FAILED because the
        // frame's VkImage doesn't have the bit set.
        let extra_usage = vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR.as_raw();
        let r = unsafe { waymux_avvk_set_frames(hw_frames_ref, 0, extra_usage) };
        if r != 0 {
            unsafe {
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
            }
            return Err(format!("waymux_avvk_set_frames returned {r}"));
        }
        let r = unsafe { ff::av_hwframe_ctx_init(hw_frames_ref) };
        if r < 0 {
            unsafe {
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
            }
            return Err(format!("av_hwframe_ctx_init returned {}", averror(r)));
        }

        // ── 3. Open the hevc_vulkan encoder ──
        let encoder_name_cstr = CString::new("hevc_vulkan").unwrap();
        let codec = unsafe { ff::avcodec_find_encoder_by_name(encoder_name_cstr.as_ptr()) };
        if codec.is_null() {
            unsafe {
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
            }
            return Err("hevc_vulkan encoder not found in this ffmpeg build".to_string());
        }
        let codec_ctx = unsafe { ff::avcodec_alloc_context3(codec) };
        if codec_ctx.is_null() {
            unsafe {
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
            }
            return Err("avcodec_alloc_context3 returned null".to_string());
        }
        unsafe {
            // CODED dims so the encoder writes a 64-aligned bitstream
            // without auto-cropping. SPS conformance-window cropping
            // is what triggers NVIDIA's hevc_vulkan rext bug; bypass
            // it by encoding at the padded size and cropping in the
            // consumer (e.g., the marketing transcode step).
            (*codec_ctx).width = coded_width as i32;
            (*codec_ctx).height = coded_height as i32;
            (*codec_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VULKAN;
            (*codec_ctx).time_base = ff::AVRational {
                num: 1,
                den: 1_000_000,
            };
            (*codec_ctx).framerate = ff::AVRational { num: 60, den: 1 };
            (*codec_ctx).hw_device_ctx = ff::av_buffer_ref(hw_device_ref);
            (*codec_ctx).hw_frames_ctx = ff::av_buffer_ref(hw_frames_ref);
            // Encoder-flag GLOBAL_HEADER → emit SPS/PPS/VPS in extradata
            // (codec_private) instead of in-band. Required for MKV.
            (*codec_ctx).flags |= ff::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
            // Tag the bitstream BT.709 limited-range. The compute shader
            // (`vulkan_compute_yuv444_2plane.glsl`) writes BT.709
            // limited-range YUV values; without these tags decoders
            // guess and pick BT.601, producing magenta/cyan chroma
            // fringing especially visible on saturated edges.
            (*codec_ctx).colorspace = ff::AVColorSpace::AVCOL_SPC_BT709;
            (*codec_ctx).color_primaries = ff::AVColorPrimaries::AVCOL_PRI_BT709;
            (*codec_ctx).color_trc = ff::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
            (*codec_ctx).color_range = ff::AVColorRange::AVCOL_RANGE_MPEG;
            // Intra-only encoding (every frame is a keyframe). Avoids
            // inter-prediction artifacts that show up as wedge-shaped
            // chroma misalignment in fast-motion 60 fps drone footage.
            // Larger output but bit-exact frame-by-frame.
            (*codec_ctx).gop_size = 1;
            (*codec_ctx).max_b_frames = 0;
        }

        // HEVC RangeExt 4:4:4 + lossless tune + QP=0. Mirror the ffmpeg
        // CLI invocation that validated this path on RTX A6000 + driver
        // 580: `-c:v hevc_vulkan -profile:v rext -tune lossless -qp 0`.
        unsafe {
            let priv_data = (*codec_ctx).priv_data;
            for (key, val) in [
                ("profile", "rext"), // RangeExt (4:4:4)
                ("tune", "lossless"),
                ("qp", "0"),
                ("rc_mode", "cqp"), // constant QP
            ] {
                let k = CString::new(key).unwrap();
                let v = CString::new(val).unwrap();
                let r = ff::av_opt_set(priv_data, k.as_ptr(), v.as_ptr(), 0);
                if r < 0 {
                    tracing::warn!(
                        "av_opt_set({key}={val}) returned {} — encoder may not honor lossless mode",
                        averror(r)
                    );
                }
            }
        }

        let r = unsafe { ff::avcodec_open2(codec_ctx, codec, ptr::null_mut()) };
        if r < 0 {
            let msg = averror(r);
            unsafe {
                ff::avcodec_free_context(&mut (codec_ctx as *mut _));
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
            }
            return Err(format!("avcodec_open2(hevc_vulkan): {msg}"));
        }

        let extradata = unsafe {
            let p = (*codec_ctx).extradata;
            let n = (*codec_ctx).extradata_size as usize;
            if p.is_null() || n == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(p, n).to_vec()
            }
        };

        // ── 4. Allocate a compute-family command pool + one CB for our
        //       per-frame copy. ──
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(vk_ctx.compute_queue_family);
        let cmd_pool = match unsafe { vk_ctx.device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                unsafe {
                    ff::avcodec_free_context(&mut (codec_ctx as *mut _));
                    ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                    ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
                }
                return Err(format!("create_command_pool: {e:?}"));
            }
        };
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffer = match unsafe { vk_ctx.device.allocate_command_buffers(&alloc) } {
            Ok(mut v) => v.remove(0),
            Err(e) => {
                unsafe {
                    vk_ctx.device.destroy_command_pool(cmd_pool, None);
                    ff::avcodec_free_context(&mut (codec_ctx as *mut _));
                    ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                    ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
                }
                return Err(format!("allocate_command_buffers: {e:?}"));
            }
        };

        tracing::info!(
            width,
            height,
            extradata_bytes = extradata.len(),
            "hevc_vulkan encoder opened"
        );

        Ok(Self {
            width,
            height,
            coded_width,
            coded_height,
            hw_device_ref,
            hw_frames_ref,
            codec_ctx,
            extradata,
            cmd_pool,
            cmd_buffer,
            device_handle_raw: vk_ctx.device.handle().as_raw(),
            compute_queue_family: vk_ctx.compute_queue_family,
            next_frame_index: 0,
        })
    }

    /// Push one frame: copy `input.image` into a libav-pool AVVkFrame,
    /// submit it to the encoder, and drain any output packets.
    ///
    /// The copy is performed on the shared compute queue using a timeline
    /// semaphore owned by the AVVkFrame — the encoder's own command
    /// submission waits on that semaphore at the next sem_value, so there
    /// is no CPU stall here.
    ///
    /// PTS is in microseconds (time_base = 1/1_000_000).
    pub fn encode_one(
        &mut self,
        vk_ctx: &VkDeviceCtx,
        input: &FrameInputView,
        pts_us: i64,
    ) -> Result<Vec<EncodedPacket>, String> {
        if vk_ctx.device.handle().as_raw() != self.device_handle_raw {
            return Err("VkDeviceCtx changed between HevcVkEncoder::open and encode_one".into());
        }
        if input.current_queue_family != self.compute_queue_family {
            return Err(format!(
                "input image on QF {} but encoder runs on QF {}",
                input.current_queue_family, self.compute_queue_family
            ));
        }

        let frames_ctx = unsafe { (*self.hw_frames_ref).data as *mut ff::AVHWFramesContext };
        let frame = unsafe { ff::av_frame_alloc() };
        if frame.is_null() {
            return Err("av_frame_alloc returned null".into());
        }
        // RAII drop-guard wrapping the frame so we always free it.
        struct FrameGuard(*mut ff::AVFrame);
        impl Drop for FrameGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { ff::av_frame_free(&mut self.0) };
                }
            }
        }
        let _frame_guard = FrameGuard(frame);

        unsafe {
            (*frame).format = ff::AVPixelFormat::AV_PIX_FMT_VULKAN as i32;
            (*frame).width = self.coded_width as i32;
            (*frame).height = self.coded_height as i32;
        }
        let r = unsafe { ff::av_hwframe_get_buffer(self.hw_frames_ref, frame, 0) };
        if r < 0 {
            return Err(format!("av_hwframe_get_buffer: {}", averror(r)));
        }
        let avvkframe = unsafe { (*frame).data[0] as *mut c_void };
        if avvkframe.is_null() {
            return Err("av_hwframe_get_buffer succeeded but data[0] is null".into());
        }
        let _ = frames_ctx; // suppress unused warning; kept for future use

        // libav's AVVkFrame for multi-planar formats allocates SEPARATE
        // VkImages per plane: img[0] = Y plane, img[1] = UV plane (for
        // NV24). For tightly-packed formats (BGRA) only img[0] is used.
        // Probe img[1]; if non-null we write the chroma plane separately.
        let mut view0 = WaymuxVkFrameView::default();
        let mut view1 = WaymuxVkFrameView::default();
        unsafe {
            waymux_avvk_frame_view(avvkframe, &mut view0, 0);
            waymux_avvk_frame_view(avvkframe, &mut view1, 1);
        }
        let dst_image = vk::Image::from_raw(view0.img as u64);
        let dst_image_uv = if view1.img.is_null() {
            // Single-plane / packed format — UV lives inside img[0].
            vk::Image::null()
        } else {
            vk::Image::from_raw(view1.img as u64)
        };
        let dst_sem = vk::Semaphore::from_raw(view0.sem as u64);
        let dst_current_layout = vk::ImageLayout::from_raw(view0.layout);
        let dst_current_qf = view0.queue_family;
        let dst_sem_wait = view0.sem_value;
        let dst_sem_signal = view0.sem_value + 1;
        // If we have a separate UV image, we also need its sem + layout.
        let dst_uv_layout = if view1.img.is_null() {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::from_raw(view1.layout)
        };

        // ── Record our copy. ──
        let device = &vk_ctx.device;
        unsafe {
            device
                .reset_command_buffer(self.cmd_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset_command_buffer: {e:?}"))?;
            device
                .begin_command_buffer(
                    self.cmd_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("begin_command_buffer: {e:?}"))?;

            // Pre-copy barriers: src → TRANSFER_SRC, dst → TRANSFER_DST.
            let subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let pre_barriers_base = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(input.current_stage)
                    .src_access_mask(input.current_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(input.current_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(input.image)
                    .subresource_range(subresource),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::from_raw(view0.access as u64))
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(dst_current_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(dst_image)
                    .subresource_range(subresource),
            ];
            // If libav allocated a separate UV image, barrier it too.
            let pre_barrier_uv = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::from_raw(view1.access as u64))
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(dst_uv_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst_image_uv)
                .subresource_range(subresource);
            if dst_image_uv == vk::Image::null() {
                device.cmd_pipeline_barrier2(
                    self.cmd_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&pre_barriers_base),
                );
            } else {
                let pre_barriers = [pre_barriers_base[0], pre_barriers_base[1], pre_barrier_uv];
                device.cmd_pipeline_barrier2(
                    self.cmd_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&pre_barriers),
                );
            }

            // NV24 source = G8_B8R8_2PLANE_444_UNORM (PLANE_0 = Y, PLANE_1 = UV).
            // libav's AVVkFrame may allocate either:
            //   (a) a single 2-plane VkImage in img[0] (same format) → write
            //       both planes via aspect_mask PLANE_0 + PLANE_1 into dst_image
            //   (b) SEPARATE per-plane VkImages: img[0] = Y plane (R8 full
            //       res, aspect COLOR), img[1] = UV plane (R8G8 full res,
            //       aspect COLOR) → write PLANE_0 source into img[0] with
            //       aspect=COLOR, write PLANE_1 source into img[1] with
            //       aspect=COLOR.
            // We detected which by checking if img[1] was non-null above.
            // CODED dims so the copy covers the full encoder picture
            // (display content + neutral-chroma padding pre-cleared
            // in `run_compute_yuv444_into_picture`).
            let plane_extent = vk::Extent3D {
                width: self.coded_width,
                height: self.coded_height,
                depth: 1,
            };
            if dst_image_uv == vk::Image::null() {
                // Case (a): single multi-planar image in img[0].
                let make_copy = |aspect: vk::ImageAspectFlags| {
                    vk::ImageCopy::default()
                        .src_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(aspect)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .dst_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(aspect)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .extent(plane_extent)
                };
                let copies = [
                    make_copy(vk::ImageAspectFlags::PLANE_0),
                    make_copy(vk::ImageAspectFlags::PLANE_1),
                ];
                device.cmd_copy_image(
                    self.cmd_buffer,
                    input.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copies,
                );
            } else {
                // Case (b): separate VkImage per plane. Y → img[0] (COLOR
                // aspect on dest, PLANE_0 aspect on source), UV → img[1]
                // (COLOR aspect on dest, PLANE_1 aspect on source).
                let copy_to_img0 = vk::ImageCopy::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .extent(plane_extent);
                let copy_to_img1 = vk::ImageCopy::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .extent(plane_extent);
                device.cmd_copy_image(
                    self.cmd_buffer,
                    input.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_to_img0],
                );
                device.cmd_copy_image(
                    self.cmd_buffer,
                    input.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image_uv,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_to_img1],
                );
            }

            // Post-copy barriers: each written dst → GENERAL (libav's
            // expected layout for hevc_vulkan's compute shaders).
            let make_post = |img: vk::Image| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(subresource)
            };
            if dst_image_uv == vk::Image::null() {
                let post_barriers = [make_post(dst_image)];
                device.cmd_pipeline_barrier2(
                    self.cmd_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&post_barriers),
                );
            } else {
                let post_barriers = [make_post(dst_image), make_post(dst_image_uv)];
                device.cmd_pipeline_barrier2(
                    self.cmd_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&post_barriers),
                );
            }

            device
                .end_command_buffer(self.cmd_buffer)
                .map_err(|e| format!("end_command_buffer: {e:?}"))?;

            // Submit on our compute queue. Wait on the dst image's timeline
            // semaphore at its current value, signal at +1. The encoder
            // will wait on +1 before its shaders run.
            let wait_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(dst_sem)
                .value(dst_sem_wait)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
            let signal_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(dst_sem)
                .value(dst_sem_signal)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
            let cb_info = vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd_buffer);
            let submit = vk::SubmitInfo2::default()
                .wait_semaphore_infos(std::slice::from_ref(&wait_info))
                .command_buffer_infos(std::slice::from_ref(&cb_info))
                .signal_semaphore_infos(std::slice::from_ref(&signal_info));
            device
                .queue_submit2(vk_ctx.compute_queue, &[submit], vk::Fence::null())
                .map_err(|e| format!("queue_submit2: {e:?}"))?;
        }

        // ── Tell libav the new state of the AVVkFrame. ──
        // For multi-plane separate-image AVVkFrames, update each plane
        // libav considers "live" so the encoder's wait semaphore picks
        // up our copy completion for chroma too. Plane 0 always exists;
        // plane 1 only when libav allocated separate per-plane images.
        unsafe {
            waymux_avvk_frame_update(
                avvkframe,
                0,
                dst_sem_signal,
                vk::ImageLayout::GENERAL.as_raw(),
                vk::AccessFlags::SHADER_READ.as_raw(),
                dst_current_qf,
            );
            if dst_image_uv != vk::Image::null() {
                waymux_avvk_frame_update(
                    avvkframe,
                    1,
                    dst_sem_signal,
                    vk::ImageLayout::GENERAL.as_raw(),
                    vk::AccessFlags::SHADER_READ.as_raw(),
                    dst_current_qf,
                );
            }
        }

        // ── Drive the encoder. ──
        unsafe {
            (*frame).pts = pts_us;
        }
        let r = unsafe { ff::avcodec_send_frame(self.codec_ctx, frame) };
        if r < 0 {
            return Err(format!("avcodec_send_frame: {}", averror(r)));
        }

        let mut packets = Vec::new();
        loop {
            let pkt = unsafe { ff::av_packet_alloc() };
            if pkt.is_null() {
                return Err("av_packet_alloc returned null".into());
            }
            let r = unsafe { ff::avcodec_receive_packet(self.codec_ctx, pkt) };
            if r == ff::AVERROR(ff::EAGAIN) || r == ff::AVERROR_EOF {
                unsafe { ff::av_packet_free(&mut (pkt as *mut _)) };
                break;
            }
            if r < 0 {
                unsafe { ff::av_packet_free(&mut (pkt as *mut _)) };
                return Err(format!("avcodec_receive_packet: {}", averror(r)));
            }
            let data = unsafe {
                let p = (*pkt).data;
                let n = (*pkt).size as usize;
                if p.is_null() || n == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(p, n).to_vec()
                }
            };
            let is_keyframe = unsafe { (*pkt).flags & ff::AV_PKT_FLAG_KEY != 0 };
            let pts = unsafe { (*pkt).pts };
            packets.push(EncodedPacket {
                data,
                pts_us: pts,
                is_keyframe,
            });
            unsafe { ff::av_packet_free(&mut (pkt as *mut _)) };
        }

        self.next_frame_index += 1;
        Ok(packets)
    }

    /// Flush the encoder. Call once at end of recording. Returns any
    /// remaining packets.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>, String> {
        let r = unsafe { ff::avcodec_send_frame(self.codec_ctx, std::ptr::null_mut()) };
        if r < 0 {
            return Err(format!("flush avcodec_send_frame(NULL): {}", averror(r)));
        }
        let mut packets = Vec::new();
        loop {
            let pkt = unsafe { ff::av_packet_alloc() };
            if pkt.is_null() {
                return Err("av_packet_alloc returned null".into());
            }
            let r = unsafe { ff::avcodec_receive_packet(self.codec_ctx, pkt) };
            if r == ff::AVERROR(ff::EAGAIN) || r == ff::AVERROR_EOF {
                unsafe { ff::av_packet_free(&mut (pkt as *mut _)) };
                break;
            }
            if r < 0 {
                unsafe { ff::av_packet_free(&mut (pkt as *mut _)) };
                return Err(format!("flush avcodec_receive_packet: {}", averror(r)));
            }
            let data = unsafe {
                let p = (*pkt).data;
                let n = (*pkt).size as usize;
                std::slice::from_raw_parts(p, n).to_vec()
            };
            let is_keyframe = unsafe { (*pkt).flags & ff::AV_PKT_FLAG_KEY != 0 };
            let pts = unsafe { (*pkt).pts };
            packets.push(EncodedPacket {
                data,
                pts_us: pts,
                is_keyframe,
            });
            unsafe { ff::av_packet_free(&mut (pkt as *mut _)) };
        }
        Ok(packets)
    }

    /// HEVC extradata (hvcC bytes) for the MKV track's CodecPrivate.
    pub fn extradata(&self) -> &[u8] {
        &self.extradata
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn coded_width(&self) -> u32 {
        self.coded_width
    }
    pub fn coded_height(&self) -> u32 {
        self.coded_height
    }
}

impl Drop for HevcVkEncoder {
    fn drop(&mut self) {
        // NOTE: we intentionally do NOT destroy `cmd_pool` here. Doing so
        // would require holding a `&ash::Device` for the pool's lifetime,
        // which conflicts with libav's hwdevice context tearing the device
        // down first when `hw_device_ref` is unref'd. In practice the
        // encoder lives for the duration of a recording and is dropped
        // when the VkDeviceCtx is too; the OS reclaims the pool when the
        // device is destroyed. If lifetime decouples later, plumb a
        // `&VkDeviceCtx` into Drop or expose an explicit `close()`.
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

fn averror(code: i32) -> String {
    let mut buf = [0i8; 256];
    unsafe { ff::av_strerror(code, buf.as_mut_ptr(), buf.len()) };
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
    cstr.to_string_lossy().into_owned()
}

/// Convenience: open an MkvWriter pre-configured for the HEVC codec
/// using this encoder's extradata. Caller drives `write_frame` per
/// packet from `encode_one` and `flush`.
pub fn open_hevc_mkv<W: std::io::Write + std::io::Seek>(
    enc: &HevcVkEncoder,
    inner: W,
) -> std::io::Result<waymux_mux_mkv::MkvWriter<W>> {
    // MKV pixel_width/pixel_height advertise the CODED dimensions —
    // matches the bitstream's SPS, so decoders/players use the same
    // size as the encoder wrote. A consumer scaling the result back
    // to display dims can use `enc.display_width()`/`display_height()`.
    waymux_mux_mkv::MkvWriter::new_for_codec(
        inner,
        enc.coded_width(),
        enc.coded_height(),
        enc.extradata(),
        "V_MPEGH/ISO/HEVC",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First sanity check: opening the encoder with our VkDeviceCtx works.
    /// No frames pushed yet.
    #[test]
    fn hevc_vk_encoder_opens() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let enc = HevcVkEncoder::open(&ctx, 1920, 1080).expect("HevcVkEncoder::open");
        eprintln!(
            "hevc_vk_encoder_opens: extradata={} bytes",
            enc.extradata().len()
        );
    }

    /// End-to-end test: BGRA → compute shader → 2-plane NV24 image →
    /// hevc_vulkan encoder → HEVC NAL packets → MKV. Verifies the full
    /// recording-thread path without needing a Wayland client.
    #[test]
    fn hevc_vk_encode_blank_frames() {
        use crate::vulkan_record::{
            run_compute_yuv444_into_picture, BgraToYuv444Pipeline, FrameResources444,
        };

        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        if !ctx.h265_encode_supported {
            eprintln!("VK_KHR_video_encode_h265 not on this device — skipping");
            return;
        }
        if !ctx.hi444_supported {
            eprintln!("Hi444 caps not on this device — skipping");
            return;
        }
        let w: u32 = std::env::var("WAYMUX_TEST_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(320);
        let h: u32 = std::env::var("WAYMUX_TEST_H")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(240);
        let fr = FrameResources444::new(&ctx, w, h).expect("FrameResources444::new");
        let pipe = BgraToYuv444Pipeline::new(&ctx, 1).expect("BgraToYuv444Pipeline::new");
        let mut enc = HevcVkEncoder::open(&ctx, w, h).expect("HevcVkEncoder::open");

        // Deterministic BGRA test pattern.
        let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
        for y in 0..(h as usize) {
            for x in 0..(w as usize) {
                let i = (y * w as usize + x) * 4;
                bgra[i] = ((x * 255) / w as usize) as u8;
                bgra[i + 1] = ((y * 255) / h as usize) as u8;
                bgra[i + 2] = 128;
                bgra[i + 3] = 0xff;
            }
        }

        let mkv_path = std::path::PathBuf::from("/tmp/hevc_vk_blank.mkv");
        let file =
            std::io::BufWriter::new(std::fs::File::create(&mkv_path).expect("create mkv file"));
        let mut mkv = open_hevc_mkv(&enc, file).expect("open_hevc_mkv");

        let mut total_packets = 0usize;
        let mut total_bytes = 0usize;
        for i in 0..5 {
            // BGRA → NV24 via our compute shader.
            run_compute_yuv444_into_picture(&ctx, &fr, &pipe, &bgra)
                .expect("run_compute_yuv444_into_picture");
            let view = FrameInputView {
                image: fr.yuv_image,
                current_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                current_access: vk::AccessFlags2::empty(),
                current_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                current_queue_family: ctx.compute_queue_family,
            };
            let pts_us = (i as i64) * 16_667;
            let packets = enc.encode_one(&ctx, &view, pts_us).expect("encode_one");
            for p in &packets {
                eprintln!(
                    "frame {i}: packet pts={} key={} bytes={}",
                    p.pts_us,
                    p.is_keyframe,
                    p.data.len()
                );
                total_bytes += p.data.len();
                mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                    .expect("mkv write_frame");
            }
            total_packets += packets.len();
        }
        // Flush.
        let tail = enc.flush().expect("flush");
        for p in &tail {
            eprintln!(
                "flush: packet pts={} key={} bytes={}",
                p.pts_us,
                p.is_keyframe,
                p.data.len()
            );
            total_bytes += p.data.len();
            mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                .expect("mkv write_frame (flush)");
        }
        total_packets += tail.len();
        let mut buf = mkv.finish().expect("mkv finish");
        std::io::Write::flush(&mut buf).ok();

        unsafe {
            ctx.device.device_wait_idle().ok();
        }
        // fr drops here, tearing down the per-recording resources.

        assert!(total_packets > 0, "encoder produced no packets");
        assert!(total_bytes > 0, "encoder produced empty packets");
        eprintln!(
            "hevc_vk_encode_blank_frames: {total_packets} packets, {total_bytes} bytes → {}",
            mkv_path.display()
        );
        // Quick sanity on the file: non-zero bytes.
        let meta = std::fs::metadata(&mkv_path).expect("metadata");
        assert!(
            meta.len() > 100,
            "mkv file looks too small ({} bytes)",
            meta.len()
        );
    }

    /// Diagnostic for task #87: render a known BGRA gradient at the
    /// production resolution, run the compute + plane-copy path, then
    /// read back BOTH:
    ///   1. The storage images (`y_storage_image`, `uv_storage_image`)
    ///      — the direct output of the compute shader, before any
    ///      `vkCmdCopyImage` into the multi-planar picture.
    ///   2. `yuv_image` plane 0 + plane 1 — after the R8 → PLANE_0 and
    ///      R8G8 → PLANE_1 copies the kernel does at the tail of
    ///      `run_compute_yuv444_into_picture`.
    ///
    /// We write each plane to disk and print a summary of how many
    /// pixels are at the "neutral chroma" value (Cb=Cr=128). A healthy
    /// frame on a saturated gradient should have <5% neutral-chroma
    /// pixels. The bug reported in tasks #84/#85/#87 produces ~34%
    /// neutral pixels arranged in a stair-step diagonal.
    ///
    /// Set `WAYMUX_TEST_VULKAN=1` to enable. Override resolution with
    /// `WAYMUX_TEST_W=…` / `WAYMUX_TEST_H=…` (default 1952×1122 to match
    /// the actual eagle hero recording).
    #[test]
    fn hevc_vk_chroma_readback_diagnostic() {
        use crate::vulkan_record::{
            dump_yuv444_picture_planes, dump_yuv444_storage_images,
            run_compute_yuv444_into_picture, BgraToYuv444Pipeline, FrameResources444,
        };

        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        if !ctx.hi444_supported {
            eprintln!("Hi444 caps not on this device — skipping");
            return;
        }
        let w: u32 = std::env::var("WAYMUX_TEST_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1952);
        let h: u32 = std::env::var("WAYMUX_TEST_H")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1122);
        eprintln!("chroma_readback: {w}x{h}");

        let fr = FrameResources444::new(&ctx, w, h).expect("FrameResources444::new");
        let pipe = BgraToYuv444Pipeline::new(&ctx, 1).expect("BgraToYuv444Pipeline::new");

        // BGRA gradient: deeply saturated colors all over so neutral
        // chroma is everywhere unexpected. Red column on the left,
        // green column in the middle, blue column on the right.
        let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
        for yp in 0..(h as usize) {
            for xp in 0..(w as usize) {
                let i = (yp * w as usize + xp) * 4;
                let band = (xp * 3) / w as usize; // 0, 1, 2
                match band {
                    0 => {
                        bgra[i] = 0;
                        bgra[i + 1] = 0;
                        bgra[i + 2] = 255;
                    } // red
                    1 => {
                        bgra[i] = 0;
                        bgra[i + 1] = 255;
                        bgra[i + 2] = 0;
                    } // green
                    _ => {
                        bgra[i] = 255;
                        bgra[i + 1] = 0;
                        bgra[i + 2] = 0;
                    } // blue
                }
                bgra[i + 3] = 0xff;
            }
        }

        run_compute_yuv444_into_picture(&ctx, &fr, &pipe, &bgra)
            .expect("run_compute_yuv444_into_picture");

        let (stor_y, stor_uv) =
            dump_yuv444_storage_images(&ctx, &fr).expect("dump_yuv444_storage_images");
        let (pic_p0, pic_p1) =
            dump_yuv444_picture_planes(&ctx, &fr).expect("dump_yuv444_picture_planes");

        // Write to /tmp for offline inspection.
        let stor_path = std::path::PathBuf::from("/tmp/yuv444-storage-uv.bin");
        let pic_path = std::path::PathBuf::from("/tmp/yuv444-picture-uv.bin");
        let stor_y_path = std::path::PathBuf::from("/tmp/yuv444-storage-y.bin");
        let pic_p0_path = std::path::PathBuf::from("/tmp/yuv444-picture-p0.bin");
        std::fs::write(&stor_y_path, &stor_y).expect("write stor_y");
        std::fs::write(&stor_path, &stor_uv).expect("write stor_uv");
        std::fs::write(&pic_p0_path, &pic_p0).expect("write pic_p0");
        std::fs::write(&pic_path, &pic_p1).expect("write pic_p1");

        let count_neutral = |uv: &[u8]| -> (usize, usize) {
            let mut neutral = 0usize;
            let mut zero = 0usize;
            for i in 0..(uv.len() / 2) {
                let cb = uv[i * 2];
                let cr = uv[i * 2 + 1];
                if cb.abs_diff(128) < 3 && cr.abs_diff(128) < 3 {
                    neutral += 1;
                }
                if cb == 0 && cr == 0 {
                    zero += 1;
                }
            }
            (neutral, zero)
        };
        let pixels = (w as usize) * (h as usize);
        let (stor_neutral, stor_zero) = count_neutral(&stor_uv);
        let (pic_neutral, pic_zero) = count_neutral(&pic_p1);
        eprintln!(
            "STORAGE_UV   neutral_chroma={:.1}% zero_chroma={:.1}%  ({}/{} px)",
            100.0 * stor_neutral as f64 / pixels as f64,
            100.0 * stor_zero as f64 / pixels as f64,
            stor_neutral,
            pixels
        );
        eprintln!(
            "PICTURE_UV   neutral_chroma={:.1}% zero_chroma={:.1}%  ({}/{} px)",
            100.0 * pic_neutral as f64 / pixels as f64,
            100.0 * pic_zero as f64 / pixels as f64,
            pic_neutral,
            pixels
        );

        // Verdict.
        let stor_clean = stor_neutral < pixels / 20; // <5%
        let pic_clean = pic_neutral < pixels / 20;
        eprintln!(
            "DIAGNOSIS: storage_clean={} picture_clean={}",
            stor_clean, pic_clean
        );
        if stor_clean && !pic_clean {
            eprintln!(
                "→ Bug is in run_compute_yuv444_into_picture's R8G8 → PLANE_1 vkCmdCopyImage"
            );
        } else if !stor_clean {
            eprintln!("→ Bug is in the compute shader output");
        } else {
            eprintln!("→ Neither stage shows the artifact at this resolution");
        }

        unsafe {
            ctx.device.device_wait_idle().ok();
        }
    }

    /// Push a known real BGRA frame (loaded from `/tmp/eagle-frame.bgra`)
    /// through compute + encode_one to see whether the encoder bug
    /// reproduces on real photographic content. `WAYMUX_TEST_VULKAN=1`.
    /// Width/height default to 1952×1122 (matching the eagle hero).
    #[test]
    fn hevc_vk_real_frame_encode_repro() {
        use crate::vulkan_record::{
            dump_yuv444_picture_planes, run_compute_yuv444_into_picture, BgraToYuv444Pipeline,
            FrameResources444,
        };

        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let bgra_path = std::env::var("WAYMUX_TEST_BGRA")
            .unwrap_or_else(|_| "/tmp/eagle-frame.bgra".to_string());
        if !std::path::Path::new(&bgra_path).exists() {
            eprintln!("BGRA fixture not found at {bgra_path} — skipping");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        if !ctx.hi444_supported {
            eprintln!("Hi444 caps not on this device — skipping");
            return;
        }
        let w: u32 = std::env::var("WAYMUX_TEST_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1952);
        let h: u32 = std::env::var("WAYMUX_TEST_H")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1122);
        let bgra = std::fs::read(&bgra_path).expect("read bgra fixture");
        let expected = (w as usize) * (h as usize) * 4;
        if bgra.len() != expected {
            panic!(
                "bgra fixture size {} != expected {} ({}×{}×4)",
                bgra.len(),
                expected,
                w,
                h
            );
        }
        eprintln!("real_frame_encode_repro: {w}x{h} fixture={bgra_path}");

        let fr = FrameResources444::new(&ctx, w, h).expect("FrameResources444::new");
        let pipe = BgraToYuv444Pipeline::new(&ctx, 1).expect("BgraToYuv444Pipeline::new");
        let mut enc = HevcVkEncoder::open(&ctx, w, h).expect("HevcVkEncoder::open");

        run_compute_yuv444_into_picture(&ctx, &fr, &pipe, &bgra)
            .expect("run_compute_yuv444_into_picture");

        let (pic_p0, pic_p1) =
            dump_yuv444_picture_planes(&ctx, &fr).expect("dump_yuv444_picture_planes");
        let pixels = (w as usize) * (h as usize);
        let mut pic_neutral = 0usize;
        for i in 0..pixels {
            let cb = pic_p1[i * 2];
            let cr = pic_p1[i * 2 + 1];
            if cb.abs_diff(128) < 3 && cr.abs_diff(128) < 3 {
                pic_neutral += 1;
            }
        }
        eprintln!(
            "PICTURE_UV (pre-encode): neutral={:.1}%",
            100.0 * pic_neutral as f64 / pixels as f64
        );

        // Save the pre-encode YUV so we can compare against the
        // decoded MKV byte-for-byte.
        std::fs::write("/tmp/eagle-pre-encode.yuv-p0", &pic_p0).expect("write p0");
        std::fs::write("/tmp/eagle-pre-encode.yuv-p1", &pic_p1).expect("write p1");

        let view = FrameInputView {
            image: fr.yuv_image,
            current_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            current_access: vk::AccessFlags2::empty(),
            current_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            current_queue_family: ctx.compute_queue_family,
        };
        let mkv_path = std::path::PathBuf::from("/tmp/eagle-vk-encoded.mkv");
        let file =
            std::io::BufWriter::new(std::fs::File::create(&mkv_path).expect("create mkv file"));
        let mut mkv = open_hevc_mkv(&enc, file).expect("open_hevc_mkv");
        for i in 0..3 {
            let packets = enc
                .encode_one(&ctx, &view, (i as i64) * 16_667)
                .expect("encode_one");
            for p in &packets {
                mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                    .expect("mkv write_frame");
            }
        }
        let tail = enc.flush().expect("flush");
        for p in &tail {
            mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                .expect("mkv write_frame (flush)");
        }
        mkv.finish().expect("mkv finish");
        eprintln!("wrote {} for offline decode", mkv_path.display());

        unsafe {
            ctx.device.device_wait_idle().ok();
        }
    }

    /// Higher-fidelity reproduction of the chroma artifact. Use a
    /// PHOTO-like pseudo-random BGRA pattern (fine-grained chroma
    /// transitions in every direction). Run through compute + plane
    /// copies + encode_one, decode the output, and check whether the
    /// chroma plane retained per-pixel detail.
    ///
    /// `WAYMUX_TEST_VULKAN=1` to enable. `WAYMUX_TEST_W=…` / `…_H=…`
    /// override dims (default 1952×1122).
    #[test]
    fn hevc_vk_high_freq_chroma_repro() {
        use crate::vulkan_record::{
            dump_yuv444_picture_planes, run_compute_yuv444_into_picture, BgraToYuv444Pipeline,
            FrameResources444,
        };

        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        if !ctx.hi444_supported {
            eprintln!("Hi444 caps not on this device — skipping");
            return;
        }
        let w: u32 = std::env::var("WAYMUX_TEST_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1952);
        let h: u32 = std::env::var("WAYMUX_TEST_H")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1122);
        eprintln!("high_freq_chroma_repro: {w}x{h}");

        let fr = FrameResources444::new(&ctx, w, h).expect("FrameResources444::new");
        let pipe = BgraToYuv444Pipeline::new(&ctx, 1).expect("BgraToYuv444Pipeline::new");
        let mut enc = HevcVkEncoder::open(&ctx, w, h).expect("HevcVkEncoder::open");

        // Pseudo-random BGRA pattern. Per-pixel R/G/B drawn from a
        // simple LCG so every adjacent pixel differs — high chroma
        // frequency everywhere.
        let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
        let mut seed: u32 = 0xc0ffee;
        let lcg = |s: &mut u32| -> u32 {
            *s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            *s
        };
        for i in 0..((w as usize) * (h as usize)) {
            bgra[i * 4] = (lcg(&mut seed) & 0xff) as u8; // B
            bgra[i * 4 + 1] = (lcg(&mut seed) & 0xff) as u8; // G
            bgra[i * 4 + 2] = (lcg(&mut seed) & 0xff) as u8; // R
            bgra[i * 4 + 3] = 0xff;
        }

        run_compute_yuv444_into_picture(&ctx, &fr, &pipe, &bgra)
            .expect("run_compute_yuv444_into_picture");

        // Read back picture plane 1 BEFORE encode_one consumes it.
        let (pic_p0, pic_p1) =
            dump_yuv444_picture_planes(&ctx, &fr).expect("dump_yuv444_picture_planes");
        let pixels = (w as usize) * (h as usize);
        let mut pic_neutral = 0usize;
        for i in 0..pixels {
            let cb = pic_p1[i * 2];
            let cr = pic_p1[i * 2 + 1];
            if cb.abs_diff(128) < 3 && cr.abs_diff(128) < 3 {
                pic_neutral += 1;
            }
        }
        eprintln!(
            "PICTURE_UV (pre-encode): neutral={:.1}%  ({}/{})",
            100.0 * pic_neutral as f64 / pixels as f64,
            pic_neutral,
            pixels
        );

        // Now encode this same frame and inspect the encoded MKV.
        let view = FrameInputView {
            image: fr.yuv_image,
            current_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            current_access: vk::AccessFlags2::empty(),
            current_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            current_queue_family: ctx.compute_queue_family,
        };
        let mkv_path = std::path::PathBuf::from("/tmp/hevc_vk_highfreq.mkv");
        let file =
            std::io::BufWriter::new(std::fs::File::create(&mkv_path).expect("create mkv file"));
        let mut mkv = open_hevc_mkv(&enc, file).expect("open_hevc_mkv");
        for i in 0..3 {
            let packets = enc
                .encode_one(&ctx, &view, (i as i64) * 16_667)
                .expect("encode_one");
            for p in &packets {
                mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                    .expect("mkv write_frame");
            }
        }
        let tail = enc.flush().expect("flush");
        for p in &tail {
            mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                .expect("mkv write_frame (flush)");
        }
        mkv.finish().expect("mkv finish");
        eprintln!("wrote {} for offline inspection", mkv_path.display());
        let _ = pic_p0;

        unsafe {
            ctx.device.device_wait_idle().ok();
        }
    }
}
