// SPDX-License-Identifier: Apache-2.0

//! GPU zero-copy lossless recording via ffmpeg's `ffv1_vulkan` encoder.
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
//!                                        ffv1_vulkan encoder
//!                                              │
//!                                              ▼
//!                                       AVPacket (FFV1 bytes)
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
//! (`src/ffv1_vk_shim.c`).

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

/// One encoded packet from `ffv1_vulkan`.
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
/// - an `AVCodecContext` opened for the `ffv1_vulkan` encoder
pub struct Ffv1VkEncoder {
    width: u32,
    height: u32,
    hw_device_ref: *mut ff::AVBufferRef,
    hw_frames_ref: *mut ff::AVBufferRef,
    codec_ctx: *mut ff::AVCodecContext,
    extradata: Vec<u8>,
    /// Compute-family command pool owned by us — we record vkCmdCopyImage
    /// here on the same queue family libav uses, so no QF ownership
    /// transfer is required between our copy and the encoder's compute
    /// shaders. The encoder picks a queue from this family internally.
    #[allow(dead_code)]
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

impl Ffv1VkEncoder {
    /// Open the encoder. Width/height must match the recording's
    /// resolution and stay constant for the lifetime of this encoder.
    ///
    /// `vk_ctx` must outlive the encoder — libav holds raw pointers
    /// into the VkInstance/VkDevice handles.
    pub fn open(vk_ctx: &VkDeviceCtx, width: u32, height: u32) -> Result<Self, String> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(format!("ffv1_vulkan needs even dims, got {width}x{height}"));
        }

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
        // libav stored the pointers but the leaked_cstr pool keeps the
        // strings alive forever — see VkDeviceCtx::enabled_dev_extensions.
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
            (*frames_ctx).sw_format = ff::AVPixelFormat::AV_PIX_FMT_BGRA;
            (*frames_ctx).width = width as i32;
            (*frames_ctx).height = height as i32;
            // Small pool — we only need one in-flight encode at a time
            // for first-pass. Increase later if pipelining helps.
            (*frames_ctx).initial_pool_size = 4;
        }
        // tiling=0 → OPTIMAL. extra_usage=0 → just the libav defaults
        // (sampled, storage, transfer src/dst).
        let r = unsafe { waymux_avvk_set_frames(hw_frames_ref, 0, 0) };
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

        // ── 3. Open the ffv1_vulkan encoder ──
        let encoder_name_cstr = CString::new("ffv1_vulkan").unwrap();
        let codec = unsafe { ff::avcodec_find_encoder_by_name(encoder_name_cstr.as_ptr()) };
        if codec.is_null() {
            unsafe {
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
            }
            return Err("ffv1_vulkan encoder not found in this ffmpeg build".to_string());
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
            (*codec_ctx).width = width as i32;
            (*codec_ctx).height = height as i32;
            (*codec_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VULKAN;
            (*codec_ctx).time_base = ff::AVRational {
                num: 1,
                den: 1_000_000,
            };
            (*codec_ctx).framerate = ff::AVRational { num: 60, den: 1 };
            (*codec_ctx).hw_device_ctx = ff::av_buffer_ref(hw_device_ref);
            (*codec_ctx).hw_frames_ctx = ff::av_buffer_ref(hw_frames_ref);
        }

        let r = unsafe { ff::avcodec_open2(codec_ctx, codec, ptr::null_mut()) };
        if r < 0 {
            let msg = averror(r);
            unsafe {
                ff::avcodec_free_context(&mut (codec_ctx as *mut _));
                ff::av_buffer_unref(&mut (hw_frames_ref as *mut _));
                ff::av_buffer_unref(&mut (hw_device_ref as *mut _));
            }
            return Err(format!("avcodec_open2(ffv1_vulkan): {msg}"));
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
            "ffv1_vulkan encoder opened"
        );

        Ok(Self {
            width,
            height,
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
            return Err("VkDeviceCtx changed between Ffv1VkEncoder::open and encode_one".into());
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
            (*frame).width = self.width as i32;
            (*frame).height = self.height as i32;
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

        let mut view = WaymuxVkFrameView::default();
        unsafe { waymux_avvk_frame_view(avvkframe, &mut view, 0) };
        let dst_image = vk::Image::from_raw(view.img as u64);
        let dst_sem = vk::Semaphore::from_raw(view.sem as u64);
        let dst_current_layout = vk::ImageLayout::from_raw(view.layout);
        let dst_current_qf = view.queue_family;
        let dst_sem_wait = view.sem_value;
        let dst_sem_signal = view.sem_value + 1;

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
            let pre_barriers = [
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
                    .src_access_mask(vk::AccessFlags2::from_raw(view.access as u64))
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(dst_current_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(dst_image)
                    .subresource_range(subresource),
            ];
            device.cmd_pipeline_barrier2(
                self.cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&pre_barriers),
            );

            let copy = vk::ImageCopy::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            device.cmd_copy_image(
                self.cmd_buffer,
                input.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );

            // Post-copy barrier: dst → GENERAL (libav's expected layout for
            // ffv1_vulkan's compute shaders).
            let post_barriers = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst_image)
                .subresource_range(subresource)];
            device.cmd_pipeline_barrier2(
                self.cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&post_barriers),
            );

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
        unsafe {
            waymux_avvk_frame_update(
                avvkframe,
                0,
                dst_sem_signal,
                vk::ImageLayout::GENERAL.as_raw(),
                vk::AccessFlags::SHADER_READ.as_raw(),
                dst_current_qf,
            );
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

    /// FFV1 extradata for the MKV track's CodecPrivate.
    pub fn extradata(&self) -> &[u8] {
        &self.extradata
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Drop for Ffv1VkEncoder {
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

/// Convenience: open an MkvWriter pre-configured for the FFV1 codec
/// using this encoder's extradata. Caller drives `write_frame` per
/// packet from `encode_one` and `flush`.
pub fn open_ffv1_mkv<W: std::io::Write + std::io::Seek>(
    enc: &Ffv1VkEncoder,
    inner: W,
) -> std::io::Result<waymux_mux_mkv::MkvWriter<W>> {
    waymux_mux_mkv::MkvWriter::new_for_codec(
        inner,
        enc.width(),
        enc.height(),
        enc.extradata(),
        "V_FFV1",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First sanity check: opening the encoder with our VkDeviceCtx works.
    /// No frames pushed yet.
    #[test]
    fn ffv1_vk_encoder_opens() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let enc = Ffv1VkEncoder::open(&ctx, 1920, 1080).expect("Ffv1VkEncoder::open");
        eprintln!(
            "ffv1_vk_encoder_opens: extradata={} bytes",
            enc.extradata().len()
        );
    }

    /// SHM/CPU-BGRA path: upload packed BGRA bytes into a TRANSFER_SRC
    /// VkImage via `upload_bgra_to_transfer_src` (the path the `ffv1_vulkan`
    /// recording thread uses for SHM-only clients like `foot`), encode it,
    /// and verify the encoder produces packets that mux into a valid MKV.
    /// This is the unit-level guard for the "ffv1-vulkan writes no file from
    /// SHM input" bug.
    #[test]
    fn ffv1_vk_encode_bgra_upload() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        use crate::vulkan_record::upload_bgra_to_transfer_src;
        let (w, h) = (320u32, 240u32);
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let mut enc = Ffv1VkEncoder::open(&ctx, w, h).expect("Ffv1VkEncoder::open");

        // A simple non-black BGRA gradient so the encoder has real content.
        let mut bgra = vec![0u8; (w * h * 4) as usize];
        for (i, px) in bgra.chunks_exact_mut(4).enumerate() {
            px[0] = (i % 256) as u8; // B
            px[1] = ((i / 256) % 256) as u8; // G
            px[2] = 0x80; // R
            px[3] = 0xff; // A
        }

        let mkv_path = std::path::PathBuf::from("/tmp/ffv1_vk_bgra_upload.mkv");
        let file =
            std::io::BufWriter::new(std::fs::File::create(&mkv_path).expect("create mkv file"));
        let mut mkv = open_ffv1_mkv(&enc, file).expect("open_ffv1_mkv");

        let mut total_packets = 0usize;
        for i in 0..5 {
            let imported = upload_bgra_to_transfer_src(&ctx, &bgra, w, h)
                .expect("upload_bgra_to_transfer_src");
            let view = FrameInputView {
                image: imported.image,
                current_layout: vk::ImageLayout::PREINITIALIZED,
                current_access: vk::AccessFlags2::empty(),
                current_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                current_queue_family: ctx.compute_queue_family,
            };
            let pts_us = (i as i64) * 16_667;
            let packets = enc.encode_one(&ctx, &view, pts_us).expect("encode_one");
            unsafe { ctx.device.device_wait_idle().ok() };
            for p in &packets {
                mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                    .expect("mkv write_frame");
            }
            total_packets += packets.len();
            imported.destroy(&ctx);
        }
        let tail = enc.flush().expect("flush");
        for p in &tail {
            mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                .expect("mkv write_frame (flush)");
        }
        total_packets += tail.len();
        let mut buf = mkv.finish().expect("mkv finish");
        std::io::Write::flush(&mut buf).ok();

        assert!(total_packets > 0, "encoder produced no packets from BGRA");
        let meta = std::fs::metadata(&mkv_path).expect("metadata");
        assert!(meta.len() > 100, "mkv too small ({} bytes)", meta.len());
        eprintln!(
            "ffv1_vk_encode_bgra_upload: {total_packets} packets → {}",
            mkv_path.display()
        );
    }

    /// Encode a handful of frames to verify the per-frame copy + submit
    /// path actually produces packets and they mux into a valid MKV. The
    /// "input" image is a black BGRA VkImage we allocate ourselves and
    /// never write to — we're not testing pixel correctness, just the
    /// plumbing.
    #[test]
    fn ffv1_vk_encode_blank_frames() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let mut enc = Ffv1VkEncoder::open(&ctx, 320, 240).expect("Ffv1VkEncoder::open");

        // Allocate a BGRA VkImage we can use as the input.
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width: 320,
                height: 240,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { ctx.device.create_image(&image_info, None) }.expect("create_image");

        let req = unsafe { ctx.device.get_image_memory_requirements(image) };
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };
        let mut type_idx = u32::MAX;
        for i in 0..mem_props.memory_type_count {
            if req.memory_type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            {
                type_idx = i;
                break;
            }
        }
        assert_ne!(type_idx, u32::MAX, "no DEVICE_LOCAL memory type for image");
        let mem = unsafe {
            ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(type_idx),
                None,
            )
        }
        .expect("allocate_memory");
        unsafe { ctx.device.bind_image_memory(image, mem, 0) }.expect("bind_image_memory");

        let input = FrameInputView {
            image,
            current_layout: vk::ImageLayout::UNDEFINED,
            current_access: vk::AccessFlags2::empty(),
            current_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            current_queue_family: ctx.compute_queue_family,
        };

        // MKV output path (under /tmp so we can ffprobe it).
        let mkv_path = std::path::PathBuf::from("/tmp/ffv1_vk_blank.mkv");
        let file =
            std::io::BufWriter::new(std::fs::File::create(&mkv_path).expect("create mkv file"));
        let mut mkv = open_ffv1_mkv(&enc, file).expect("open_ffv1_mkv");

        let mut total_packets = 0usize;
        let mut total_bytes = 0usize;
        for i in 0..5 {
            // After the first frame, the input image's layout is now
            // TRANSFER_SRC_OPTIMAL (we transitioned it in pre_barriers).
            let view = if i == 0 {
                FrameInputView {
                    image,
                    current_layout: vk::ImageLayout::UNDEFINED,
                    current_access: vk::AccessFlags2::empty(),
                    current_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                    current_queue_family: ctx.compute_queue_family,
                }
            } else {
                FrameInputView {
                    image,
                    current_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    current_access: vk::AccessFlags2::TRANSFER_READ,
                    current_stage: vk::PipelineStageFlags2::COPY,
                    current_queue_family: ctx.compute_queue_family,
                }
            };
            let pts_us = (i as i64) * 16_667; // ~60 fps
            let packets = enc.encode_one(&ctx, &view, pts_us).expect("encode_one");
            for p in &packets {
                eprintln!(
                    "frame {i}: packet pts={} key={} bytes={}",
                    p.pts_us,
                    p.is_keyframe,
                    p.data.len()
                );
                total_bytes += p.data.len();
                // pts in μs → MKV wants ms.
                mkv.write_frame(&p.data, p.pts_us / 1_000, p.is_keyframe)
                    .expect("mkv write_frame");
            }
            total_packets += packets.len();
            let _ = input.image; // keep `input` alive for clippy
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
            ctx.device.destroy_image(image, None);
            ctx.device.free_memory(mem, None);
        }

        assert!(total_packets > 0, "encoder produced no packets");
        assert!(total_bytes > 0, "encoder produced empty packets");
        eprintln!(
            "ffv1_vk_encode_blank_frames: {total_packets} packets, {total_bytes} bytes → {}",
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
}
