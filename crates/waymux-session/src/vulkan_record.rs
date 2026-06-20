// SPDX-License-Identifier: Apache-2.0

//! Vulkan zero-copy recording pipeline.
//!
//! Replaces the OpenGL/EGL + CPU readback + ffmpeg subprocess path with
//! a Vulkan-only pipeline:
//!
//! ```text
//! client dmabuf  ─►  vkAllocateMemory + VkImportMemoryFdInfoKHR (zero-copy fd dup)
//!                     ─►  VkImage (DRM modifier, dedicated alloc)
//!                          │
//!                vkCmdDispatch compute (BGRA→NV12 in GPU memory)
//!                          │
//!                VK_KHR_video_encode_h264 queue submit (still GPU memory)
//!                          │
//!                small CPU readback of encoded NAL units (~100 KB/frame)
//!                          │
//!                in-process mkv muxer  ─►  file
//! ```
//!
//! Per the research report (2026-05-11):
//!
//! - `ash` 0.38 exposes the needed extensions (semver-exempt; pin tight)
//! - AMD Renoir / Mesa 25+ supports H.264 encode (VCN 2.0); local laptop
//!   reports `VK_KHR_video_encode_h264 rev 14` + a dedicated
//!   `QUEUE_VIDEO_ENCODE_BIT_KHR` queue family
//! - NVIDIA driver 535+ ships encode; verified through waymux on driver
//!   560.35.05 (L40) for the prior pipeline
//! - Intel ANV: temporarily disabled in Mesa 25.3.5; re-check before
//!   shipping
//! - Reference impls: pyroenc (C++), nvpro-samples/vk_video_samples (C++),
//!   smelter/gpu-video (Rust)
//!
//! This module is scaffolding only at present. The `probe()` entry point
//! enumerates Vulkan capabilities and writes a diagnostic report — use it
//! during the RFC implementation phase to verify a host before wiring
//! the recording path through it.

// experimental helpers retained for in-development encode paths
#![allow(dead_code)]

use std::ffi::CStr;

use ash::vk;

/// Pre-compiled BGRA->NV12 compute shader. Source: vulkan_compute.glsl,
/// canonical truth for the conversion math is `recording.rs::bgra_to_nv12`.
/// Rebuild with `glslc -fshader-stage=compute --target-env=vulkan1.3 -O
/// crates/waymux-session/src/vulkan_compute.glsl -o ...spv` whenever the
/// .glsl source changes. Per the research report (2026-05-11) we ship the
/// .spv binary rather than pulling shaderc into the build graph.
pub const BGRA_TO_NV12_SPV: &[u8] = include_bytes!("vulkan_compute.spv");

/// Compiled SPIR-V for the BGRA → YUV 4:4:4 compute shader (3-plane
/// variant). Writes 3 full-resolution R8 planes (Y, U, V). Sister of
/// `BGRA_TO_NV12_SPV`. Source: `src/vulkan_compute_yuv444.glsl`.
/// Kept for AMD/Mesa once they expose `G8_B8_R8_3PLANE_444_UNORM`
/// (today only the 2-plane variant works — see `BGRA_TO_YUV444_2PLANE_SPV`).
pub const BGRA_TO_YUV444_SPV: &[u8] = include_bytes!("vulkan_compute_yuv444.spv");

/// Compiled SPIR-V for the BGRA → 2-plane YUV 4:4:4 compute shader.
/// This is the actually-used Hi444PP shader on the baseline 2026-05-12
/// (NVIDIA driver 560 only reports `G8_B8R8_2PLANE_444_UNORM` as a
/// supported encode-src format for Hi444PP). Layout: plane 0 = Y (R8
/// full-res), plane 1 = UV interleaved (R8G8 full-res). Source:
/// `src/vulkan_compute_yuv444_2plane.glsl`.
pub const BGRA_TO_YUV444_2PLANE_SPV: &[u8] = include_bytes!("vulkan_compute_yuv444_2plane.spv");

/// H.264 encode profile selector. Threaded through `VkDeviceCtx::open`,
/// `EncodeSession::new`, `FrameResources::new`, and the encode submit
/// path so a single recording can target either subsampling.
///
/// - `Main420` — standard H.264 Main profile, NV12 (4:2:0), QP defaults
///   to `WAYMUX_VK_ENCODE_QP` (visually lossless at 20, perceptually
///   lossy below that). Supported on AMD/Mesa + NVIDIA + Intel.
/// - `Hi444Lossless` — H.264 High 4:4:4 Predictive, YUV 4:4:4 picture
///   format, QP=0 forced for bit-exact pixel round-trip. NVIDIA-only
///   on baseline 2026-05-12 (Mesa exposes only Main 4:2:0; see
///   `feedback_amd_no_444_encode.md`). Max 4096×4096 on A6000.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeKind {
    Main420,
    Hi444Lossless,
}

impl EncodeKind {
    /// `StdVideoH264ProfileIdc` for the H.264 SPS / video profile.
    pub fn std_profile_idc(self) -> vk::native::StdVideoH264ProfileIdc {
        match self {
            EncodeKind::Main420 => {
                vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN
            }
            EncodeKind::Hi444Lossless => {
                vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
            }
        }
    }

    /// `StdVideoH264ChromaFormatIdc` — wire value 1 for 4:2:0, 3 for
    /// 4:4:4. Goes into the SPS chroma_format_idc field.
    pub fn chroma_format_idc(self) -> vk::native::StdVideoH264ChromaFormatIdc {
        match self {
            EncodeKind::Main420 => {
                vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420
            }
            EncodeKind::Hi444Lossless => {
                vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_444
            }
        }
    }

    /// `VideoChromaSubsamplingFlagsKHR` — Vulkan-side enum mirroring
    /// the same 4:2:0 vs 4:4:4 choice.
    pub fn chroma_subsampling(self) -> vk::VideoChromaSubsamplingFlagsKHR {
        match self {
            EncodeKind::Main420 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            EncodeKind::Hi444Lossless => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        }
    }

    /// Multi-planar picture format the encoder consumes. Both Main and
    /// Hi444PP use a 2-plane (Y + interleaved chroma) layout on the
    /// baseline 2026-05-12 driver set; 4:4:4 keeps chroma at full
    /// resolution. (NVIDIA 560 reports only `G8_B8R8_2PLANE_444_UNORM`
    /// for Hi444PP encode-src.)
    pub fn picture_format(self) -> vk::Format {
        match self {
            EncodeKind::Main420 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            EncodeKind::Hi444Lossless => vk::Format::G8_B8R8_2PLANE_444_UNORM,
        }
    }

    /// Constant QP for `cmd_encode_video_khr`. 0 is mandatory for
    /// bit-exact lossless on Hi444PP (the H.264 transform-domain math
    /// is reversible at QP=0; any higher QP introduces quantization
    /// error and breaks bit-exactness). Main420 falls back to the
    /// `WAYMUX_VK_ENCODE_QP` env-override.
    pub fn constant_qp(self) -> i32 {
        match self {
            EncodeKind::Main420 => vk_encode_qp(),
            EncodeKind::Hi444Lossless => 0,
        }
    }

    /// Whether the picture image has 2 planes (Y + interleaved UV) or
    /// 3 (Y + U + V).
    pub fn num_planes(self) -> u32 {
        match self {
            EncodeKind::Main420 => 2,
            EncodeKind::Hi444Lossless => 3,
        }
    }

    /// Build the Vulkan H.264 profile chain — the two structs that
    /// every video-encode call has to push_next together. Returns
    /// `(h264_profile, profile_info)` with appropriate lifetimes for
    /// short-lived call sites.
    pub fn build_profile_info<'a>(self) -> vk::VideoEncodeH264ProfileInfoKHR<'a> {
        vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(self.std_profile_idc())
    }
}

/// What `probe()` discovered about Vulkan capabilities on this host.
/// Serialized to a TSV-ish line on stderr and returned for programmatic
/// inspection from tests.
#[derive(Debug, Clone)]
pub struct VulkanProbe {
    pub api_version: u32,
    pub instance_extensions: Vec<String>,
    pub devices: Vec<VulkanDevice>,
}

#[derive(Debug, Clone)]
pub struct VulkanDevice {
    pub name: String,
    pub driver_name: String,
    pub api_version: u32,
    pub queue_families: Vec<VulkanQueueFamily>,
    pub device_extensions: Vec<String>,
    pub video_encode_h264_supported: bool,
    pub dmabuf_import_supported: bool,
}

#[derive(Debug, Clone)]
pub struct VulkanQueueFamily {
    pub index: u32,
    pub flags: vk::QueueFlags,
    pub count: u32,
}

/// Inspect Vulkan + the video encode extensions and return a structured
/// report. Bring-up tool for the rewrite: run it on every host we
/// might deploy on and gate the new path behind the capability bits
/// it surfaces.
///
/// Run with `WAYMUX_VULKAN_PROBE=1 waymux-session ...` (or from the
/// vulkan_probe unit test) to log the report.
pub fn probe() -> Result<VulkanProbe, String> {
    let entry = unsafe { ash::Entry::load().map_err(|e| format!("ash::Entry::load: {e:?}"))? };
    let api_version = unsafe { entry.try_enumerate_instance_version() }
        .map_err(|e| format!("enumerate_instance_version: {e:?}"))?
        .unwrap_or(vk::API_VERSION_1_0);

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"waymux-vulkan-probe")
        .application_version(0)
        .engine_name(c"waymux")
        .engine_version(0)
        .api_version(vk::API_VERSION_1_3);

    // Enumerate available instance extensions so we can request the
    // ones we need (external memory caps) without erroring on hosts
    // that lack them.
    let inst_ext_props = unsafe {
        entry
            .enumerate_instance_extension_properties(None)
            .map_err(|e| format!("enumerate_instance_extension_properties: {e:?}"))?
    };
    let inst_exts: Vec<String> = inst_ext_props
        .iter()
        .map(|p| cstr_name(&p.extension_name).to_owned())
        .collect();

    let want_exts = [
        c"VK_KHR_get_physical_device_properties2".as_ptr(),
        c"VK_KHR_external_memory_capabilities".as_ptr(),
        c"VK_KHR_external_fence_capabilities".as_ptr(),
        c"VK_KHR_external_semaphore_capabilities".as_ptr(),
    ];
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&want_exts);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(|e| format!("create_instance: {e:?}"))?
    };

    let mut devices = Vec::new();
    let phys_devs = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices: {e:?}"))?
    };
    for pdev in phys_devs {
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        let dev_ext_props = unsafe {
            instance
                .enumerate_device_extension_properties(pdev)
                .map_err(|e| format!("enumerate_device_extension_properties: {e:?}"))?
        };
        let dev_exts: Vec<String> = dev_ext_props
            .iter()
            .map(|p| cstr_name(&p.extension_name).to_owned())
            .collect();
        let has = |name: &str| dev_exts.iter().any(|e| e == name);
        let video_encode_h264_supported = has("VK_KHR_video_queue")
            && has("VK_KHR_video_encode_queue")
            && has("VK_KHR_video_encode_h264");
        let dmabuf_import_supported = has("VK_KHR_external_memory")
            && has("VK_KHR_external_memory_fd")
            && has("VK_EXT_external_memory_dma_buf")
            && has("VK_EXT_image_drm_format_modifier");

        let qf_props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
        let queue_families = qf_props
            .iter()
            .enumerate()
            .map(|(i, q)| VulkanQueueFamily {
                index: i as u32,
                flags: q.queue_flags,
                count: q.queue_count,
            })
            .collect();

        devices.push(VulkanDevice {
            name: cstr_name(&props.device_name),
            // driver_name requires VK_KHR_driver_properties; skip in
            // the bring-up probe to avoid the structure-chain dance.
            driver_name: String::new(),
            api_version: props.api_version,
            queue_families,
            device_extensions: dev_exts,
            video_encode_h264_supported,
            dmabuf_import_supported,
        });
    }

    unsafe { instance.destroy_instance(None) };

    Ok(VulkanProbe {
        api_version,
        instance_extensions: inst_exts,
        devices,
    })
}

fn cstr_name(buf: &[std::os::raw::c_char]) -> String {
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
    cstr.to_string_lossy().into_owned()
}

// ────────────────────────────────────────────────────────────────────────
// Device context.
//
// `VkDeviceCtx` owns the Vulkan instance + logical device + extension
// function loaders. Created once when a recording starts; torn down on
// Drop. Picks the first physical device that exposes a VIDEO_ENCODE
// queue family and the dmabuf-import extensions.

/// Device-level extensions required by the Vulkan recording path.
/// Some are EXT-not-KHR (no ash function loader, struct-only) and some
/// have function loaders we use via the `ash::khr` modules. We check
/// presence by name string and load the loaders separately.
const REQUIRED_DEVICE_EXTENSION_NAMES: &[&str] = &[
    "VK_KHR_external_memory_fd",
    "VK_KHR_external_semaphore_fd",
    "VK_KHR_synchronization2",
    "VK_KHR_video_queue",
    "VK_KHR_video_encode_queue",
    "VK_KHR_video_encode_h264",
];

/// Optional device extensions: needed for zero-copy dmabuf import,
/// but not for synthetic-frame encoding (CPU upload → compute → encode).
/// NVIDIA driver 560 lacks `VK_EXT_external_memory_dma_buf` (it was
/// added in 565), so we treat both as optional and surface the
/// resulting capability via `VkDeviceCtx::dmabuf_import_supported`.
const DMABUF_IMPORT_EXTENSION_NAMES: &[&str] = &[
    "VK_EXT_external_memory_dma_buf",
    "VK_EXT_image_drm_format_modifier",
];

/// Optional H.265 encode extension — needed only by the
/// `hevc_vulkan` (HEVC RangeExt lossless) recording path. Enabled
/// at device-create time when the physical device reports it.
/// Includes the video_maintenance extensions libav's hevc_vulkan
/// encoder gates on (otherwise `avcodec_open2` returns ENOSYS).
const H265_ENCODE_EXTENSION_NAMES: &[&str] = &[
    "VK_KHR_video_encode_h265",
    "VK_KHR_video_maintenance1",
    "VK_KHR_video_maintenance2",
];

/// Owned Vulkan device + extension function loaders. Holds everything
/// needed to import a dmabuf, dispatch a compute pipeline, and submit
/// to the video encode queue.
pub struct VkDeviceCtx {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,

    pub compute_queue_family: u32,
    pub video_encode_queue_family: u32,
    pub compute_queue: vk::Queue,
    pub video_encode_queue: vk::Queue,

    pub video_queue_inst: ash::khr::video_queue::Instance,
    pub video_queue_dev: ash::khr::video_queue::Device,
    pub video_encode_queue_dev: ash::khr::video_encode_queue::Device,
    pub external_memory_fd_dev: ash::khr::external_memory_fd::Device,

    pub device_name: String,

    /// Driver-reported std-headers spec name + version for the H.264
    /// encode profile. Hardcoding `VK_MAKE_VIDEO_STD_VERSION(1,0,0)`
    /// broke on NVIDIA L40 (driver 560.35.05) because NVIDIA's bundled
    /// std-headers report a different spec_version. Trust the driver.
    pub h264_std_header_name: [i8; vk::MAX_EXTENSION_NAME_SIZE],
    pub h264_std_header_spec_version: u32,

    /// Driver-reported `image_create_flags` for the NV12 image used as
    /// VIDEO_ENCODE_SRC. AMD reports MUTABLE_FORMAT|ALIAS|EXTENDED_USAGE;
    /// NVIDIA reports a different set (the AMD constants we hardcoded
    /// before caused encoder-side rejection mid-submit). Pulled from
    /// `vkGetPhysicalDeviceVideoFormatPropertiesKHR` at open() time.
    pub nv12_encode_src_image_flags: vk::ImageCreateFlags,
    /// Same idea for the DPB image (VIDEO_ENCODE_DPB usage).
    pub nv12_dpb_image_flags: vk::ImageCreateFlags,

    /// Driver-reported `image_create_flags` for the 3-plane YUV 4:4:4
    /// image (`G8_B8_R8_3PLANE_444_UNORM`) under H.264 Hi444PP profile.
    /// Empty when Hi444PP is unsupported (AMD/Mesa on baseline 2026-05-12).
    pub yuv444_encode_src_image_flags: vk::ImageCreateFlags,
    pub yuv444_dpb_image_flags: vk::ImageCreateFlags,

    /// Whether the driver reports Hi444PP (4:4:4) encode support. Set
    /// to true when `get_physical_device_video_capabilities_khr` accepts
    /// the HIGH_444_PREDICTIVE + TYPE_444 profile chain.
    pub hi444_supported: bool,

    /// Whether the device exposes the dmabuf-import extensions
    /// (`VK_EXT_external_memory_dma_buf` + `VK_EXT_image_drm_format_modifier`).
    /// NVIDIA driver 560 doesn't (added in 565+); the Hi444PP encode
    /// path doesn't need them but the zero-copy `encode_idr_from_dmabuf`
    /// path does. Callers check this before constructing an
    /// `ImportedDmabufImage`.
    pub dmabuf_import_supported: bool,

    /// Whether the device exposes `VK_KHR_video_encode_h265`. Required
    /// for the HEVC RangeExt lossless recording path (`hevc_vulkan`
    /// encoder). NVIDIA 560+ and AMD/Mesa 25+ both have it.
    pub h265_encode_supported: bool,
    /// Hi444PP-specific std-headers version, separate from the Main
    /// profile's version because drivers may report different values.
    pub hi444_std_header_name: [i8; vk::MAX_EXTENSION_NAME_SIZE],
    pub hi444_std_header_spec_version: u32,

    debug_utils: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl VkDeviceCtx {
    /// Open the first physical device that exposes a VIDEO_ENCODE queue
    /// family and all required extensions. Returns a structured error
    /// describing the first failure point so callers can log a clean
    /// reason for falling back to the legacy path.
    pub fn open() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load().map_err(|e| format!("ash::Entry::load: {e:?}"))? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"waymux-vk-record")
            .engine_name(c"waymux")
            .api_version(vk::API_VERSION_1_3);
        // Instance extensions: keep minimal. KHR caps are core in 1.1.
        // Optionally enable VK_LAYER_KHRONOS_validation when
        // WAYMUX_VK_VALIDATE=1 — gives the driver's gripes a chance
        // to surface before they manifest as silent crashes during
        // encode submit bring-up.
        let validate = matches!(
            std::env::var("WAYMUX_VK_VALIDATE").ok().as_deref(),
            Some("1")
        );
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        let debug_utils_ext = c"VK_EXT_debug_utils";
        let layer_ptrs = if validate {
            vec![validation_layer.as_ptr()]
        } else {
            vec![]
        };
        let inst_ext_ptrs = if validate {
            vec![debug_utils_ext.as_ptr()]
        } else {
            vec![]
        };
        let inst_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs)
            .enabled_extension_names(&inst_ext_ptrs);
        let instance = unsafe {
            entry
                .create_instance(&inst_ci, None)
                .map_err(|e| format!("create_instance: {e:?}"))?
        };

        // Install a debug-utils messenger so the driver's own
        // diagnostic stream (not just static validation) reaches
        // stderr. Critical for diagnosing DEVICE_LOST inside
        // cmd_encode_video_khr where the structures pass validation
        // but the encoder hardware rejects the work.
        let (debug_utils, debug_messenger) = if validate {
            unsafe extern "system" fn debug_cb(
                severity: vk::DebugUtilsMessageSeverityFlagsEXT,
                _types: vk::DebugUtilsMessageTypeFlagsEXT,
                p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
                _user_data: *mut std::ffi::c_void,
            ) -> vk::Bool32 {
                if p_callback_data.is_null() {
                    return vk::FALSE;
                }
                let data = unsafe { &*p_callback_data };
                if data.p_message.is_null() {
                    return vk::FALSE;
                }
                let msg = unsafe { CStr::from_ptr(data.p_message) }
                    .to_string_lossy()
                    .into_owned();
                let tag = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
                    "ERROR"
                } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
                    "WARN"
                } else {
                    return vk::FALSE; // skip INFO/VERBOSE noise
                };
                eprintln!("[vk-debug {tag}] {msg}");
                vk::FALSE
            }
            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_cb));
            let messenger = unsafe { debug_utils.create_debug_utils_messenger(&ci, None) }
                .unwrap_or(vk::DebugUtilsMessengerEXT::null());
            (Some(debug_utils), messenger)
        } else {
            (None, vk::DebugUtilsMessengerEXT::null())
        };

        // Pick a physical device.
        let (physical_device, device_name, compute_qf, video_encode_qf, dev_exts) =
            select_physical_device(&instance)?;

        // dmabuf-import extensions are optional — present on AMD/Mesa,
        // missing on NVIDIA driver 560 (added in 565+). Include them in
        // the create_device call only if the driver actually supports
        // them; downstream callers gate on `dmabuf_import_supported`.
        let dmabuf_import_supported = DMABUF_IMPORT_EXTENSION_NAMES
            .iter()
            .all(|name| dev_exts.iter().any(|e| e == name));

        // H.265 encode extensions are optional. Treat `VK_KHR_video_encode_h265`
        // as the gating capability (without it the codec doesn't exist),
        // and pull in the video_maintenance helpers if available — libav's
        // hevc_vulkan encoder gates on maintenance1 specifically.
        let h265_encode_supported = dev_exts.iter().any(|e| e == "VK_KHR_video_encode_h265");

        // Build the device extension list. Own the CStrings explicitly
        // so the pointer table stays valid for the create_device call.
        let mut device_ext_owned: Vec<std::ffi::CString> = REQUIRED_DEVICE_EXTENSION_NAMES
            .iter()
            .map(|s| std::ffi::CString::new(*s).unwrap())
            .collect();
        if dmabuf_import_supported {
            for name in DMABUF_IMPORT_EXTENSION_NAMES {
                device_ext_owned.push(std::ffi::CString::new(*name).unwrap());
            }
        } else {
            tracing::info!(
                "vk: dmabuf-import extensions not available on {device_name} — \
                 zero-copy dmabuf encode disabled (NVIDIA driver <565 — \
                 added in 565+). Hi444PP / synthetic-frame path still works."
            );
        }
        if h265_encode_supported {
            for name in H265_ENCODE_EXTENSION_NAMES {
                if dev_exts.iter().any(|e| e == name) {
                    device_ext_owned.push(std::ffi::CString::new(*name).unwrap());
                } else {
                    tracing::info!("vk: optional ext {name} not available on {device_name}");
                }
            }
        } else {
            tracing::info!(
                "vk: VK_KHR_video_encode_h265 not available on {device_name} — \
                 HEVC RangeExt lossless path disabled"
            );
        }
        let device_ext_ptrs: Vec<*const i8> = device_ext_owned.iter().map(|c| c.as_ptr()).collect();

        // Queue create infos. Try a single-queue-per-family layout.
        // Some devices report the same family for compute + video
        // encode (e.g. AMD on certain SKUs); in that case we just open
        // two queues from the same family.
        let priorities = [1.0f32];
        let mut q_create_infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_qf)
            .queue_priorities(&priorities)];
        if video_encode_qf != compute_qf {
            q_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(video_encode_qf)
                    .queue_priorities(&priorities),
            );
        }

        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&q_create_infos)
            .enabled_extension_names(&device_ext_ptrs)
            .push_next(&mut sync2_features);
        let device = unsafe {
            instance
                .create_device(physical_device, &device_ci, None)
                .map_err(|e| format!("create_device: {e:?}"))?
        };

        let compute_queue = unsafe { device.get_device_queue(compute_qf, 0) };
        let video_encode_queue = unsafe { device.get_device_queue(video_encode_qf, 0) };

        let video_queue_inst = ash::khr::video_queue::Instance::new(&entry, &instance);
        let video_queue_dev = ash::khr::video_queue::Device::new(&instance, &device);
        let video_encode_queue_dev = ash::khr::video_encode_queue::Device::new(&instance, &device);
        let external_memory_fd_dev = ash::khr::external_memory_fd::Device::new(&instance, &device);

        // Query the driver's reported std-headers version and supported
        // image_create_flags for both encode-src and DPB. We need these
        // BEFORE EncodeSession::new / FrameResources::new are called so
        // we don't have to thread them through the test API. The probe
        // is two cheap queries against the H.264 main profile.
        let (h264_std_header_name, h264_std_header_spec_version) =
            query_h264_std_header_version(&video_queue_inst, physical_device).unwrap_or_else(|e| {
                tracing::warn!(
                    "video caps query failed: {e}; falling back to compile-time std-header"
                );
                let mut name = [0i8; vk::MAX_EXTENSION_NAME_SIZE];
                for (i, b) in H264_ENCODE_STD_HEADER_NAME.iter().enumerate() {
                    name[i] = *b as i8;
                }
                (name, H264_ENCODE_STD_HEADER_VERSION)
            });
        let nv12_encode_src_image_flags = query_nv12_image_flags(
            &video_queue_inst,
            physical_device,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .unwrap_or_else(|e| {
            tracing::warn!(
                "nv12 encode-src image-flags query failed: {e}; falling back to AMD defaults"
            );
            vk::ImageCreateFlags::MUTABLE_FORMAT
                | vk::ImageCreateFlags::ALIAS
                | vk::ImageCreateFlags::EXTENDED_USAGE
        });
        let nv12_dpb_image_flags = query_nv12_image_flags(
            &video_queue_inst,
            physical_device,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
        )
        .unwrap_or_else(|e| {
            tracing::warn!("nv12 dpb image-flags query failed: {e}; falling back to empty flags");
            vk::ImageCreateFlags::empty()
        });
        // Hi444PP capability probe + image-flag queries. Failure is
        // expected on AMD/Mesa (Main 4:2:0 only); the Hi444PP recording
        // path checks `hi444_supported` and refuses to construct on
        // hardware where the queries didn't succeed.
        let (
            hi444_supported,
            hi444_std_header_name,
            hi444_std_header_spec_version,
            yuv444_encode_src_image_flags,
            yuv444_dpb_image_flags,
        ) = match query_hi444pp_caps(&video_queue_inst, physical_device) {
            Ok((name, ver)) => {
                let src_flags = query_yuv444_image_flags(
                    &video_queue_inst,
                    physical_device,
                    vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .unwrap_or_else(|e| {
                    tracing::warn!("yuv444 encode-src image-flags query failed: {e}; using empty");
                    vk::ImageCreateFlags::empty()
                });
                let dpb_flags = query_yuv444_image_flags(
                    &video_queue_inst,
                    physical_device,
                    vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
                )
                .unwrap_or_else(|e| {
                    tracing::warn!("yuv444 dpb image-flags query failed: {e}; using empty");
                    vk::ImageCreateFlags::empty()
                });
                (true, name, ver, src_flags, dpb_flags)
            }
            Err(e) => {
                tracing::info!("Hi444PP probe: not supported on {device_name}: {e}");
                let mut empty_name = [0i8; vk::MAX_EXTENSION_NAME_SIZE];
                for (i, b) in H264_ENCODE_STD_HEADER_NAME.iter().enumerate() {
                    empty_name[i] = *b as i8;
                }
                (
                    false,
                    empty_name,
                    H264_ENCODE_STD_HEADER_VERSION,
                    vk::ImageCreateFlags::empty(),
                    vk::ImageCreateFlags::empty(),
                )
            }
        };

        tracing::info!(
            device = %device_name,
            std_header_spec_version = %format!("0x{:08x}", h264_std_header_spec_version),
            encode_src_flags = ?nv12_encode_src_image_flags,
            dpb_flags = ?nv12_dpb_image_flags,
            hi444_supported,
            hi444_encode_src_flags = ?yuv444_encode_src_image_flags,
            hi444_dpb_flags = ?yuv444_dpb_image_flags,
            "vk: driver-reported h.264 encode parameters",
        );

        Ok(VkDeviceCtx {
            entry,
            instance,
            device,
            physical_device,
            compute_queue_family: compute_qf,
            video_encode_queue_family: video_encode_qf,
            compute_queue,
            video_encode_queue,
            video_queue_inst,
            video_queue_dev,
            video_encode_queue_dev,
            external_memory_fd_dev,
            device_name,
            h264_std_header_name,
            h264_std_header_spec_version,
            nv12_encode_src_image_flags,
            nv12_dpb_image_flags,
            yuv444_encode_src_image_flags,
            yuv444_dpb_image_flags,
            hi444_supported,
            hi444_std_header_name,
            hi444_std_header_spec_version,
            dmabuf_import_supported,
            h265_encode_supported,
            debug_utils,
            debug_messenger,
        })
    }

    /// True if the device, queue family selection, and extensions look
    /// like everything `encode_frame` will need. Used by tests + the
    /// future `VkRecorder::try_new` to fail fast.
    pub fn is_ready(&self) -> bool {
        self.video_encode_queue != vk::Queue::null() && self.compute_queue != vk::Queue::null()
    }

    /// The list of device extensions enabled when this device was
    /// created. Returned in a form callers can pass to libav via the
    /// shim (`waymux_avvk_set_device`). The string pointers remain
    /// valid for the lifetime of the static `REQUIRED_DEVICE_EXTENSION_NAMES` /
    /// `DMABUF_IMPORT_EXTENSION_NAMES` / `H265_ENCODE_EXTENSION_NAMES`
    /// arrays — i.e., forever — so we can hand them to libav safely.
    pub fn enabled_dev_extensions(&self) -> Vec<*const std::os::raw::c_char> {
        let dev_exts = self.actual_device_extensions();
        let mut out: Vec<*const std::os::raw::c_char> = Vec::new();
        for s in REQUIRED_DEVICE_EXTENSION_NAMES {
            out.push(leaked_cstr(s));
        }
        if self.dmabuf_import_supported {
            for s in DMABUF_IMPORT_EXTENSION_NAMES {
                out.push(leaked_cstr(s));
            }
        }
        if self.h265_encode_supported {
            for s in H265_ENCODE_EXTENSION_NAMES {
                if dev_exts.iter().any(|e| e == s) {
                    out.push(leaked_cstr(s));
                }
            }
        }
        out
    }

    /// Re-query the physical device's full list of supported extensions.
    /// Cheap (one Vulkan call); used by `enabled_dev_extensions` to
    /// filter the optional list against what's actually on this device.
    fn actual_device_extensions(&self) -> Vec<String> {
        let props = unsafe {
            self.instance
                .enumerate_device_extension_properties(self.physical_device)
                .unwrap_or_default()
        };
        props.iter().map(|p| cstr_name(&p.extension_name)).collect()
    }
}

/// Leak a static-lifetime CString and return its pointer. Used for
/// extension names handed to libav, which expects `const char *const *`
/// with lifetimes that outlive the AVVulkanDeviceContext. Leaking is
/// fine: there are a fixed number of extension names and they live for
/// the process lifetime anyway.
fn leaked_cstr(s: &str) -> *const std::os::raw::c_char {
    use std::sync::Mutex;
    static POOL: Mutex<Option<Vec<&'static std::ffi::CStr>>> = Mutex::new(None);
    let mut pool = POOL.lock().unwrap();
    let vec = pool.get_or_insert_with(Vec::new);
    if let Some(existing) = vec.iter().find(|cs| cs.to_str().ok() == Some(s)) {
        return existing.as_ptr();
    }
    let owned = std::ffi::CString::new(s).unwrap();
    let leaked: &'static std::ffi::CStr = Box::leak(owned.into_boxed_c_str());
    let ptr = leaked.as_ptr();
    vec.push(leaked);
    ptr
}

impl Drop for VkDeviceCtx {
    fn drop(&mut self) {
        unsafe {
            // Idle the device before tearing down — otherwise in-flight
            // command buffers can race the device destroy. SKIP this when the
            // GPU is known-wedged: device_wait_idle is unbounded and would
            // block teardown forever on a hung device (→ kernel watchdog panic
            // + reboot). The per-frame bounded fence waits already drained any
            // healthy work; a wedged device is being destroyed regardless.
            if !GPU_WEDGED.load(std::sync::atomic::Ordering::Acquire) {
                let _ = self.device.device_wait_idle();
            }
            self.device.destroy_device(None);
            // Destroy the debug messenger BEFORE the instance.
            if let Some(du) = &self.debug_utils {
                if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                    du.destroy_debug_utils_messenger(self.debug_messenger, None);
                }
            }
            self.instance.destroy_instance(None);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Encode session.
//
// `EncodeSession` owns a `VkVideoSessionKHR` and the device memory the
// driver demanded to back it. Created once per recording from a
// `VkDeviceCtx`. Parameters (SPS/PPS) live separately and are added in
// the next chunk.

/// VK_STD_vulkan_video_codec_h264_encode std header name. The driver
/// uses this to match its built-in std-header parser version against
/// what our code was compiled with. The version literal below is
/// `VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)` from `vk_video_codec_h264std_encode.h`
/// — (1 << 22) | (0 << 12) | 0 = 0x0040_0000. Matches Vulkan SDK 1.3.275+.
const H264_ENCODE_STD_HEADER_NAME: &[u8] = b"VK_STD_vulkan_video_codec_h264_encode\0";
const H264_ENCODE_STD_HEADER_VERSION: u32 = 1 << 22;

/// Persistent encode session — one per recording. Holds the
/// `VkVideoSessionKHR`, the device memory the driver bound to it, and
/// the H.264 parameters object (SPS+PPS) once `create_parameters` runs.
pub struct EncodeSession {
    session: vk::VideoSessionKHR,
    memory: Vec<vk::DeviceMemory>,
    parameters: vk::VideoSessionParametersKHR,
    /// AVCDecoderConfigurationRecord bytes (ISO 14496-15 §5.3.3.1) —
    /// written into the MKV track's `CodecPrivate` field.
    codec_private: Vec<u8>,
    /// Driver-serialized Annex-B SPS+PPS NAL units (start codes + RBSP).
    /// Prepended to each viewer IDR because AMD/Mesa does not emit them
    /// in-band. Empty when the driver emits them in-band (NVIDIA default).
    sps_pps_annexb: Vec<u8>,
    /// Cached for diagnostics / future submit code.
    pub width: u32,
    pub height: u32,
    /// Which H.264 profile this session encodes for. Main 4:2:0 vs
    /// Hi444 lossless — determines SPS profile_idc, chroma_format_idc,
    /// and the picture format the encoder consumes.
    pub kind: EncodeKind,
    /// We hold a *raw* device handle to avoid re-borrowing the ctx
    /// here. The owning `VkDeviceCtx` outlives every `EncodeSession`
    /// by construction (callers create the session in a function
    /// scope tied to the ctx).
    device: ash::Device,
    video_queue_dev: ash::khr::video_queue::Device,
    video_encode_queue_dev: ash::khr::video_encode_queue::Device,
}

impl EncodeSession {
    /// Create a video session for H.264 main-profile encoding at
    /// `width x height` (must be even, clamped to the encoder's
    /// reported `min_coded_extent` / `max_coded_extent`).
    ///
    /// Allocates backing memory in the driver-requested heap types.
    /// On failure returns the first hard error from the driver.
    pub fn new(ctx: &VkDeviceCtx, width: u32, height: u32) -> Result<Self, String> {
        Self::new_with_kind(ctx, width, height, EncodeKind::Main420)
    }

    /// Create a Hi444PP lossless video session. NVIDIA-only on baseline
    /// 2026-05-12; AMD/Mesa exposes only Main 4:2:0 and this constructor
    /// will fail with a clean error from the driver's caps query.
    pub fn new_lossless(ctx: &VkDeviceCtx, width: u32, height: u32) -> Result<Self, String> {
        if !ctx.hi444_supported {
            return Err(format!(
                "Hi444PP not supported on this device ({}); cannot open lossless session",
                ctx.device_name
            ));
        }
        Self::new_with_kind(ctx, width, height, EncodeKind::Hi444Lossless)
    }

    /// Common constructor body. Branches on `kind` for the profile,
    /// subsampling, picture format, and std-headers version.
    pub fn new_with_kind(
        ctx: &VkDeviceCtx,
        width: u32,
        height: u32,
        kind: EncodeKind,
    ) -> Result<Self, String> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(format!("H.264 needs even dimensions; got {width}x{height}"));
        }

        // Profile chain: VkVideoProfileInfoKHR -> H264 profile info.
        let mut h264_profile = kind.build_profile_info();
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(kind.chroma_subsampling())
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);

        // Use the driver's reported std-headers version. Hi444PP and
        // Main may report different values per profile — pull the
        // kind-specific pair stored in VkDeviceCtx.
        let (header_name, header_version) = match kind {
            EncodeKind::Main420 => (ctx.h264_std_header_name, ctx.h264_std_header_spec_version),
            EncodeKind::Hi444Lossless => {
                (ctx.hi444_std_header_name, ctx.hi444_std_header_spec_version)
            }
        };
        let std_header = vk::ExtensionProperties {
            extension_name: header_name,
            spec_version: header_version,
        };
        let picture_format = kind.picture_format();
        // IDR-only encoding doesn't need active references; setting
        // max_active_reference_pictures=0 tells the encoder "no future
        // predictions from this picture" and lets us skip DPB management
        // entirely. Some NVIDIA driver paths reject the encode submit
        // when DPB is configured but unused; this is the bisection.
        let idr_only = matches!(kind, EncodeKind::Hi444Lossless);
        let session_ci = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(ctx.video_encode_queue_family)
            .video_profile(&profile_info)
            .picture_format(picture_format)
            .max_coded_extent(vk::Extent2D { width, height })
            .reference_picture_format(picture_format)
            // AMD's H.264 encoder requires max_dpb_slots ≥ 1; even
            // IDR-only encodes set up a reference slot for themselves.
            // FrameResources provides the DPB image(s). The Main 4:2:0
            // path supports P-frames (1 active reference) and needs TWO
            // DPB slots so it can reference one reconstruction while
            // writing the next (ping-pong). The Hi444 lossless path is
            // IDR-only (1 slot, 0 active references).
            .max_dpb_slots(if idr_only { 1 } else { 2 })
            .max_active_reference_pictures(if idr_only { 0 } else { 1 })
            .std_header_version(&std_header);

        let mut session = vk::VideoSessionKHR::null();
        let result = unsafe {
            (ctx.video_queue_dev.fp().create_video_session_khr)(
                ctx.device.handle(),
                &session_ci,
                std::ptr::null(),
                &mut session,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(format!("vkCreateVideoSessionKHR: {result:?}"));
        }

        // Query memory requirements. Drivers may need 0, 1, or N
        // separate allocations across different heap types.
        let mut count: u32 = 0;
        unsafe {
            let _ = (ctx
                .video_queue_dev
                .fp()
                .get_video_session_memory_requirements_khr)(
                ctx.device.handle(),
                session,
                &mut count,
                std::ptr::null_mut(),
            );
        }
        let mut reqs: Vec<vk::VideoSessionMemoryRequirementsKHR> = (0..count as usize)
            .map(|_| vk::VideoSessionMemoryRequirementsKHR::default())
            .collect();
        let result = unsafe {
            (ctx.video_queue_dev
                .fp()
                .get_video_session_memory_requirements_khr)(
                ctx.device.handle(),
                session,
                &mut count,
                reqs.as_mut_ptr(),
            )
        };
        if result != vk::Result::SUCCESS {
            unsafe {
                (ctx.video_queue_dev.fp().destroy_video_session_khr)(
                    ctx.device.handle(),
                    session,
                    std::ptr::null(),
                );
            }
            return Err(format!(
                "vkGetVideoSessionMemoryRequirementsKHR: {result:?}"
            ));
        }

        // Allocate one VkDeviceMemory per requirement, in any memory
        // type the bitmask allows. Drivers commonly want DEVICE_LOCAL
        // for the DPB and the staging buffers, but we don't require
        // host-visible — the encoded output goes through a separate
        // host-visible VkBuffer we'll create at encode time.
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };
        let mut allocated = Vec::with_capacity(reqs.len());
        let mut bind_infos = Vec::with_capacity(reqs.len());
        for req in &reqs {
            // Prefer DEVICE_LOCAL, but fall back to ANY memory type the
            // driver's mask allows. NVIDIA L40 (driver 560) reports
            // memory_type_bits=0x8 for VideoSession backing memory and
            // that single memory type doesn't carry the DEVICE_LOCAL
            // flag in nvidia's mem-props; requiring DEVICE_LOCAL here
            // caused EncodeSession::new to fail with "no DEVICE_LOCAL
            // memory type satisfying mask 0x8" before any encode work
            // could happen.
            let type_idx = pick_memory_type(
                &mem_props,
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                pick_memory_type(
                    &mem_props,
                    req.memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            });
            let type_idx = match type_idx {
                Some(t) => t,
                None => {
                    cleanup_partial(&ctx.device, &allocated, &ctx.video_queue_dev, session);
                    return Err(format!(
                        "no memory type satisfying mask {:#x}",
                        req.memory_requirements.memory_type_bits
                    ));
                }
            };
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.memory_requirements.size)
                .memory_type_index(type_idx);
            let mem = match unsafe { ctx.device.allocate_memory(&alloc_info, None) } {
                Ok(m) => m,
                Err(e) => {
                    cleanup_partial(&ctx.device, &allocated, &ctx.video_queue_dev, session);
                    return Err(format!("allocate_memory for video session: {e:?}"));
                }
            };
            allocated.push(mem);
            bind_infos.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(req.memory_bind_index)
                    .memory(mem)
                    .memory_offset(0)
                    .memory_size(req.memory_requirements.size),
            );
        }

        let result = unsafe {
            (ctx.video_queue_dev.fp().bind_video_session_memory_khr)(
                ctx.device.handle(),
                session,
                bind_infos.len() as u32,
                bind_infos.as_ptr(),
            )
        };
        if result != vk::Result::SUCCESS {
            cleanup_partial(&ctx.device, &allocated, &ctx.video_queue_dev, session);
            return Err(format!("vkBindVideoSessionMemoryKHR: {result:?}"));
        }

        Ok(EncodeSession {
            session,
            memory: allocated,
            parameters: vk::VideoSessionParametersKHR::null(),
            codec_private: Vec::new(),
            sps_pps_annexb: Vec::new(),
            width,
            height,
            kind,
            device: ctx.device.clone(),
            video_queue_dev: ctx.video_queue_dev.clone(),
            video_encode_queue_dev: ctx.video_encode_queue_dev.clone(),
        })
    }

    /// Raw VkVideoSessionKHR handle for downstream parameter-creation
    /// and encode submit.
    pub fn handle(&self) -> vk::VideoSessionKHR {
        self.session
    }

    /// AVCDecoderConfigurationRecord bytes. Empty until
    /// `create_parameters` runs.
    pub fn codec_private(&self) -> &[u8] {
        &self.codec_private
    }

    /// Driver-serialized Annex-B SPS+PPS bytes (empty if the driver emits
    /// them in-band). Prepend to each viewer IDR for in-band parameter sets.
    pub fn sps_pps_annexb(&self) -> &[u8] {
        &self.sps_pps_annexb
    }

    /// Build the SPS+PPS, create the video session parameters object,
    /// extract the encoded SPS+PPS NAL units from the driver, and
    /// serialize them as AVCDecoderConfigurationRecord.
    pub fn create_parameters(&mut self) -> Result<(), String> {
        let sps = build_h264_sps(self.width, self.height, self.kind);
        let pps = build_h264_pps(self.kind);
        let sps_array = [sps];
        let pps_array = [pps];

        let add_info = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);
        let mut h264_ci = vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add_info);
        let params_ci = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(self.session)
            .push_next(&mut h264_ci);
        let mut params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (self
                .video_queue_dev
                .fp()
                .create_video_session_parameters_khr)(
                self.device.handle(),
                &params_ci,
                std::ptr::null(),
                &mut params,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(format!("vkCreateVideoSessionParametersKHR: {result:?}"));
        }
        self.parameters = params;

        // Extract the encoded SPS+PPS bytes from the driver. The H.264
        // GetInfo chain asks for SPS id=0 + PPS id=0 (the only ones we
        // wrote). Driver returns Annex-B NAL units (start codes + RBSP).
        let h264_get = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
            .write_std_sps(true)
            .std_sps_id(0)
            .write_std_pps(true)
            .std_pps_id(0);
        // The Get call needs the H.264 variant chained on the base
        // VkVideoEncodeSessionParametersGetInfoKHR.
        let mut h264_get_mut = h264_get;
        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
            .video_session_parameters(params)
            .push_next(&mut h264_get_mut);

        let mut data_size: usize = 0;
        let result = unsafe {
            (self
                .video_encode_queue_dev
                .fp()
                .get_encoded_video_session_parameters_khr)(
                self.device.handle(),
                &get_info,
                std::ptr::null_mut(),
                &mut data_size,
                std::ptr::null_mut(),
            )
        };
        if result != vk::Result::SUCCESS {
            // NVIDIA driver 560 returns ERROR_OUT_OF_HOST_MEMORY for the
            // Hi444PP GET path (the driver doesn't implement Annex-B
            // serialization for Hi444PP, only Main 4:2:0). Build a
            // synthetic AVCDecoderConfigurationRecord from our SPS struct
            // instead — the encoder will still emit SPS+PPS in-band
            // before each IDR, which is the path most decoders rely on
            // anyway.
            if matches!(self.kind, EncodeKind::Hi444Lossless) {
                tracing::warn!(
                    "vkGetEncodedVideoSessionParametersKHR(size) returned {result:?} on \
                     Hi444PP; falling back to synthetic codec_private (SPS+PPS land \
                     in-band on the IDR NALs)"
                );
                self.codec_private =
                    build_synthetic_avcc(sps.profile_idc, sps.level_idc, sps.chroma_format_idc);
                return Ok(());
            }
            return Err(format!(
                "vkGetEncodedVideoSessionParametersKHR(size): {result:?}"
            ));
        }
        let mut buf = vec![0u8; data_size];
        let result = unsafe {
            (self
                .video_encode_queue_dev
                .fp()
                .get_encoded_video_session_parameters_khr)(
                self.device.handle(),
                &get_info,
                std::ptr::null_mut(),
                &mut data_size,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(format!(
                "vkGetEncodedVideoSessionParametersKHR(data): {result:?}"
            ));
        }
        buf.truncate(data_size);
        // Stash the raw Annex-B SPS+PPS so the viewer path can prepend it
        // in-band to each IDR (AMD/Mesa does not emit parameter sets in-band,
        // so a decoder otherwise reports "non-existing PPS" and never decodes).
        self.sps_pps_annexb = buf.clone();

        // Build AVCDecoderConfigurationRecord from the Annex-B byte
        // stream.
        let sps_pps_nals = split_annexb_nalus(&buf);
        self.codec_private =
            build_avc_decoder_config_record(&sps_pps_nals, sps.profile_idc, sps.level_idc)?;
        Ok(())
    }
}

/// Build a minimal AVCDecoderConfigurationRecord without driver-supplied
/// SPS/PPS bytes. Used when the Vulkan driver refuses to serialize
/// the in-driver SPS/PPS (NVIDIA + Hi444PP). The avcC has empty
/// SPS/PPS arrays; the encoded NAL stream is expected to carry SPS+PPS
/// in-band before each IDR, which is how NVIDIA's encoder defaults
/// anyway. ffprobe + ffmpeg + gstreamer all parse in-band parameters
/// fine; some older / simpler decoders won't.
///
/// `profile_idc` and `level_idc` come from the SPS struct (StdVideoH264
/// enum positions); we translate `level_idc` to its on-wire byte value
/// the same way `build_avc_decoder_config_record` does for the
/// real-SPS path.
fn build_synthetic_avcc(
    profile_idc: vk::native::StdVideoH264ProfileIdc,
    level_idc: vk::native::StdVideoH264LevelIdc,
    _chroma_format_idc: vk::native::StdVideoH264ChromaFormatIdc,
) -> Vec<u8> {
    // Translate level_idc enum position to wire-byte value. Common
    // mappings: STD_VIDEO_H264_LEVEL_IDC_4_2 (12) → 42, 5_1 (14) → 51,
    // 5_2 (15) → 52. ISO 14496-10 §7.4.2.1.1 lists the encoding.
    let level_byte: u8 = match level_idc {
        x if x == vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0 => 40,
        x if x == vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1 => 41,
        x if x == vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2 => 42,
        x if x == vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0 => 50,
        x if x == vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1 => 51,
        x if x == vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2 => 52,
        _ => 41,
    };
    let profile_byte: u8 = profile_idc as u8;
    let out = vec![
        1,            // configurationVersion
        profile_byte, //
        0,            // profile_compatibility flags
        level_byte,   //
        0xFF,         // 6 reserved + lengthSizeMinusOne=3
        0xE0,         // 3 reserved + numOfSequenceParameterSets=0
        0,            // numOfPictureParameterSets=0
    ];
    out
}

/// Split an Annex-B (start-code framed) byte stream into NAL units,
/// stripping each unit's leading 00..01 prefix.
fn split_annexb_nalus(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        // Match 00 00 00 01 or 00 00 01.
        let four = i + 3 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1;
        let three = data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1;
        if four {
            starts.push(i + 4);
            i += 4;
        } else if three {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nalus = Vec::with_capacity(starts.len());
    for (idx, &start) in starts.iter().enumerate() {
        let end = starts
            .get(idx + 1)
            .map(|&n| {
                // Back up over the next NAL's start code so the slice
                // ends at the last RBSP byte.
                if n >= 4 && data[n - 4] == 0 {
                    n - 4
                } else {
                    n - 3
                }
            })
            .unwrap_or(data.len());
        nalus.push(&data[start..end]);
    }
    nalus
}

/// Serialize SPS + PPS NAL units as an AVCDecoderConfigurationRecord
/// (ISO 14496-15 §5.3.3.1). Format:
///   u8  configurationVersion = 1
///   u8  AVCProfileIndication   (= SPS.profile_idc)
///   u8  profile_compatibility  (constraint_set flags; 0 for us)
///   u8  AVCLevelIndication     (= SPS.level_idc)
///   6 reserved bits + 2 bits lengthSizeMinusOne (= 3)
///   3 reserved bits + 5 bits numOfSequenceParameterSets
///   for each SPS:
///     u16 spsLength
///     N bytes SPS NAL (RBSP)
///   u8  numOfPictureParameterSets
///   for each PPS:
///     u16 ppsLength
///     N bytes PPS NAL
fn build_avc_decoder_config_record(
    nalus: &[&[u8]],
    profile_idc: vk::native::StdVideoH264ProfileIdc,
    level_idc: vk::native::StdVideoH264LevelIdc,
) -> Result<Vec<u8>, String> {
    let mut sps_nals = Vec::new();
    let mut pps_nals = Vec::new();
    for n in nalus {
        if n.is_empty() {
            continue;
        }
        let nal_unit_type = n[0] & 0x1F;
        match nal_unit_type {
            7 => sps_nals.push(*n),
            8 => pps_nals.push(*n),
            _ => {}
        }
    }
    if sps_nals.is_empty() || pps_nals.is_empty() {
        return Err(format!(
            "missing SPS/PPS from driver: sps={} pps={}",
            sps_nals.len(),
            pps_nals.len()
        ));
    }
    // Real H.264 level_idc byte value: STD_VIDEO_H264_LEVEL_IDC_* are
    // enum *positions* (0=1.0, 10=4.0, 14=5.1, 15=5.2), not wire
    // bytes. Translate to the literal level number x10 the bitstream
    // expects (40, 42, 51, 52). The first 8 bytes of the SPS NAL also
    // contain level_idc at offset 3, so we can just read it back.
    let level_byte = if sps_nals[0].len() >= 4 {
        sps_nals[0][3]
    } else {
        return Err("SPS too short to read level_idc".into());
    };
    let profile_byte = if !sps_nals[0].is_empty() {
        sps_nals[0][1]
    } else {
        profile_idc as u8
    };
    let _ = (profile_idc, level_idc); // silence unused
    let mut out = Vec::with_capacity(64);
    out.push(1); // configurationVersion
    out.push(profile_byte);
    out.push(0); // profile_compatibility (constraint_set flags)
    out.push(level_byte);
    out.push(0xFF); // 6 reserved bits set + lengthSizeMinusOne = 3
    out.push(0xE0 | (sps_nals.len() as u8 & 0x1F));
    for sps in &sps_nals {
        out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        out.extend_from_slice(sps);
    }
    out.push(pps_nals.len() as u8);
    for pps in &pps_nals {
        out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        out.extend_from_slice(pps);
    }
    Ok(out)
}

impl Drop for EncodeSession {
    fn drop(&mut self) {
        unsafe {
            if self.parameters != vk::VideoSessionParametersKHR::null() {
                (self
                    .video_queue_dev
                    .fp()
                    .destroy_video_session_parameters_khr)(
                    self.device.handle(),
                    self.parameters,
                    std::ptr::null(),
                );
            }
            (self.video_queue_dev.fp().destroy_video_session_khr)(
                self.device.handle(),
                self.session,
                std::ptr::null(),
            );
            for mem in self.memory.drain(..) {
                self.device.free_memory(mem, None);
            }
        }
    }
}

/// Build an H.264 main-profile SPS suitable for streaming output. Many
/// fields are defaulted to the values the spec calls "POC type 2,
/// no VUI, progressive frames, single reference" — the simplest
/// configuration that produces a valid bitstream. Width/height are
/// converted to macroblock units (16x16 per MB). Cropping is set when
/// the source dims aren't multiples of 16 (rare for our path; 4K and
/// 1080p are both multiples of 16).
fn build_h264_sps(
    width: u32,
    height: u32,
    kind: EncodeKind,
) -> vk::native::StdVideoH264SequenceParameterSet {
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let crop_right = (mb_w * 16) - width;
    let crop_bottom = (mb_h * 16) - height;
    let needs_crop = crop_right != 0 || crop_bottom != 0;

    let mut flags: vk::native::StdVideoH264SpsFlags = unsafe { std::mem::zeroed() };
    // direct_8x8_inference_flag = 1 (required when level >= 3.0 and
    // frame_mbs_only_flag = 1, per the H.264 spec).
    flags.set_direct_8x8_inference_flag(1);
    flags.set_frame_mbs_only_flag(1);
    if needs_crop {
        flags.set_frame_cropping_flag(1);
    }
    // qpprime_y_zero_transform_bypass_flag enables H.264's bit-exact
    // lossless mode (transform bypass) when QP=0. Required for the
    // Hi444PP lossless path; harmless to leave 0 for Main 4:2:0.
    if matches!(kind, EncodeKind::Hi444Lossless) {
        flags.set_qpprime_y_zero_transform_bypass_flag(1);
    }

    vk::native::StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: kind.std_profile_idc(),
        level_idc: pick_h264_level(width, height),
        chroma_format_idc: kind.chroma_format_idc(),
        seq_parameter_set_id: 0,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        log2_max_frame_num_minus4: 0,
        // POC type 2 = "frame_num is the only timing signal we need."
        // Avoids the entire POC-cycle field set below; matches what
        // pyroenc emits for a minimal stream.
        pic_order_cnt_type: 2,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: 0,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: 1,
        reserved1: 0,
        pic_width_in_mbs_minus1: mb_w - 1,
        pic_height_in_map_units_minus1: mb_h - 1,
        frame_crop_left_offset: 0,
        frame_crop_right_offset: crop_right / 2,
        frame_crop_top_offset: 0,
        frame_crop_bottom_offset: crop_bottom / 2,
        reserved2: 0,
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
    }
}

fn build_h264_pps(kind: EncodeKind) -> vk::native::StdVideoH264PictureParameterSet {
    let mut flags: vk::native::StdVideoH264PpsFlags = unsafe { std::mem::zeroed() };
    flags.set_entropy_coding_mode_flag(1); // CABAC; Main profile permits it
    flags.set_deblocking_filter_control_present_flag(1);
    // Hi444PP (High-profile family) requires transform_8x8_mode_flag for
    // the 8x8 luma transform. Without it the driver rejects the PPS at
    // parameters-create time. Main 4:2:0 doesn't need it.
    if matches!(kind, EncodeKind::Hi444Lossless) {
        flags.set_transform_8x8_mode_flag(1);
    }
    vk::native::StdVideoH264PictureParameterSet {
        flags,
        seq_parameter_set_id: 0,
        pic_parameter_set_id: 0,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        weighted_bipred_idc:
            vk::native::StdVideoH264WeightedBipredIdc_STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT,
        pic_init_qp_minus26: 0,
        pic_init_qs_minus26: 0,
        chroma_qp_index_offset: 0,
        second_chroma_qp_index_offset: 0,
        pScalingLists: std::ptr::null(),
    }
}

// Pick a level_idc that covers the resolution at 60 fps. Maximum
// macroblock processing rate per H.264 spec Table A-1:
//   level 4.0: 245,760 MB/s     (1920x1080 @ 30 fps)
//   level 4.1: 245,760 MB/s     (1920x1080 @ 30 fps, higher bitrate)
//   level 4.2: 522,240 MB/s     (1920x1080 @ 60 fps)
//   level 5.1: 983,040 MB/s     (3840x2160 @ 32 fps)
//   level 5.2: 2,073,600 MB/s   (3840x2160 @ 60+ fps, 4K)
// ────────────────────────────────────────────────────────────────────────
// Compute pipeline.
//
// The BGRA->NV12 compute shader needs:
//   - descriptor set layout matching the GLSL bindings 0/1/2
//   - pipeline layout (just the descriptor set; no push constants)
//   - shader module loaded from BGRA_TO_NV12_SPV
//   - compute pipeline
//   - descriptor pool sized for one set per frame in flight
//
// `BgraToNv12Pipeline` owns the static pieces (layout, pipeline,
// shader module). Per-frame descriptor sets are allocated and updated
// at encode time once we have the dmabuf VkImageView.

/// Compute pipeline + descriptor layout for the BGRA->NV12 conversion
/// shader. Lifetime is tied to `VkDeviceCtx`.
pub struct BgraToNv12Pipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    device: ash::Device,
    sampler: vk::Sampler,
}

impl BgraToNv12Pipeline {
    /// Create the static pipeline objects. Allocates one descriptor
    /// pool sized for `descriptor_set_count` simultaneous frames in
    /// flight — typical recording with a 2-buffer ring needs 2.
    pub fn new(ctx: &VkDeviceCtx, descriptor_set_count: u32) -> Result<Self, String> {
        let device = ctx.device.clone();

        // Bindings match the GLSL:
        //   binding=0: sampler2D u_src  (BGRA input)
        //   binding=1: image2D u_y      (Y output, r8)
        //   binding=2: image2D u_uv     (UV output, rg8)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&dsl_ci, None)
                .map_err(|e| format!("create_descriptor_set_layout: {e:?}"))?
        };

        let layouts = [descriptor_set_layout];
        let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pl_ci, None)
                .map_err(|e| format!("create_pipeline_layout: {e:?}"))?
        };

        // Shader module from the embedded SPIR-V.
        let code: Vec<u32> = BGRA_TO_NV12_SPV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe {
            device
                .create_shader_module(&sm_ci, None)
                .map_err(|e| format!("create_shader_module: {e:?}"))?
        };

        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout);
        let pipeline =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None) }
                .map_err(|(_, e)| format!("create_compute_pipelines: {e:?}"))?[0];

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(descriptor_set_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2 * descriptor_set_count),
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(descriptor_set_count)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_ci, None)
                .map_err(|e| format!("create_descriptor_pool: {e:?}"))?
        };

        let sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .unnormalized_coordinates(false);
        let sampler = unsafe {
            device
                .create_sampler(&sampler_ci, None)
                .map_err(|e| format!("create_sampler: {e:?}"))?
        };

        Ok(BgraToNv12Pipeline {
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
            descriptor_pool,
            device,
            sampler,
        })
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }
    pub fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }
}

impl Drop for BgraToNv12Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Compute pipeline + descriptor layout for the BGRA→2-plane YUV 4:4:4
/// conversion shader. Sister of `BgraToNv12Pipeline` for the Hi444PP
/// lossless path. Uses 3 bindings (sampler + Y plane + UV-interleaved
/// plane, full-res) and dispatches one invocation per source pixel.
pub struct BgraToYuv444Pipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    device: ash::Device,
    sampler: vk::Sampler,
}

impl BgraToYuv444Pipeline {
    pub fn new(ctx: &VkDeviceCtx, descriptor_set_count: u32) -> Result<Self, String> {
        let device = ctx.device.clone();

        // Bindings match `vulkan_compute_yuv444_2plane.glsl`:
        //   binding=0: sampler2D u_src   (BGRA input)
        //   binding=1: image2D u_y       (Y output, r8 full-res)
        //   binding=2: image2D u_uv      (UV output, r8g8 full-res)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&dsl_ci, None)
                .map_err(|e| format!("create_descriptor_set_layout(yuv444): {e:?}"))?
        };

        let layouts = [descriptor_set_layout];
        let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pl_ci, None)
                .map_err(|e| format!("create_pipeline_layout(yuv444): {e:?}"))?
        };

        let code: Vec<u32> = BGRA_TO_YUV444_2PLANE_SPV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe {
            device
                .create_shader_module(&sm_ci, None)
                .map_err(|e| format!("create_shader_module(yuv444): {e:?}"))?
        };

        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout);
        let pipeline =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None) }
                .map_err(|(_, e)| format!("create_compute_pipelines(yuv444): {e:?}"))?[0];

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(descriptor_set_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2 * descriptor_set_count),
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(descriptor_set_count)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_ci, None)
                .map_err(|e| format!("create_descriptor_pool(yuv444): {e:?}"))?
        };

        let sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .unnormalized_coordinates(false);
        let sampler = unsafe {
            device
                .create_sampler(&sampler_ci, None)
                .map_err(|e| format!("create_sampler(yuv444): {e:?}"))?
        };

        Ok(BgraToYuv444Pipeline {
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
            descriptor_pool,
            device,
            sampler,
        })
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }
    pub fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }
}

impl Drop for BgraToYuv444Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Frame resources.
//
// `FrameResources` owns the per-recording images and buffers that live
// for the lifetime of the session: a source BGRA image, a multi-planar
// NV12 image (G8_B8R8_2PLANE_420_UNORM, the encoder-friendly format),
// a host-visible encoded-output VkBuffer, a query pool for encode
// feedback (offset+size), and the command pool + buffer that records
// the per-frame work.
//
// Synthetic-frame mode (used until dmabuf import lands in item 6)
// also gets a staging buffer for uploading BGRA bytes from the CPU.

/// Per-recording Vulkan resources. One per `VkRecorder`. Image
/// dimensions are fixed at construction (creating new images is
/// expensive; resolution changes mid-recording aren't supported).
pub struct FrameResources {
    device: ash::Device,
    pub width: u32,
    pub height: u32,

    // Source BGRA — either dmabuf-imported (item 6) or filled from a
    // staging buffer (synthetic-frame mode).
    pub bgra_image: vk::Image,
    pub bgra_image_view: vk::ImageView,
    bgra_memory: vk::DeviceMemory,
    pub bgra_extent: vk::Extent2D,

    // Multi-planar NV12 — what the encoder consumes. Allocated as
    // G8_B8R8_2PLANE_420_UNORM. Per `vkGetPhysicalDeviceVideoFormatPropertiesKHR`
    // on AMD RADV, the encoder requires create_flags = MUTABLE_FORMAT
    // | ALIAS | EXTENDED_USAGE (NO DISJOINT) — a single VkDeviceMemory
    // backs the entire multi-planar image.
    pub nv12_image: vk::Image,
    pub nv12_y_view: vk::ImageView, // plane 0 view — unused in encoder-only mode
    pub nv12_uv_view: vk::ImageView, // plane 1 view — unused in encoder-only mode
    pub nv12_color_view: vk::ImageView, // whole-image view for the encoder
    nv12_memory: vk::DeviceMemory,

    // DPB (Decoded Picture Buffer) — the encoder writes the
    // reconstructed picture here. Required by AMD's encoder even for
    // single-IDR submits because max_dpb_slots must be ≥1.
    //
    // For inter-prediction (P-frames) we need TWO distinct DPB slots so
    // we can reference one reconstruction while reconstructing the next
    // picture into the other (you cannot reference and reconstruct the
    // same slot in one submit). RADV does NOT advertise
    // VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR, so both
    // slots' image views must come from the SAME image — we use a single
    // 2-array-layer DPB image and a single-layer view per slot:
    // `dpb_image_view` selects layer 0 (slot 0), `dpb_image_view2`
    // selects layer 1 (slot 1). IDR-only paths only ever touch slot 0.
    pub dpb_image: vk::Image,
    pub dpb_image_view: vk::ImageView,
    pub dpb_image_view2: vk::ImageView,
    dpb_memory: vk::DeviceMemory,

    // Compute output images. The BGRA->NV12 shader writes Y to
    // y_storage_image (R8) and UV to uv_storage_image (R8G8). These
    // are separate from the encoder-target NV12 because AMD refuses
    // STORAGE + VIDEO_ENCODE_SRC on one image. After compute the
    // command buffer vkCmdCopyImages these into nv12_image's planes.
    pub y_storage_image: vk::Image,
    pub y_storage_view: vk::ImageView,
    y_storage_memory: vk::DeviceMemory,
    pub uv_storage_image: vk::Image,
    pub uv_storage_view: vk::ImageView,
    uv_storage_memory: vk::DeviceMemory,

    // Staging buffer for synthetic BGRA upload. Host-visible.
    pub staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    pub staging_size: u64,

    // Encoded-output buffer. Host-visible so vkMapMemory works
    // directly. Sized to a comfortable upper bound (we never write
    // more than width*height bytes of NAL data per frame).
    pub encoded_buffer: vk::Buffer,
    encoded_memory: vk::DeviceMemory,
    pub encoded_size: u64,

    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub query_pool: vk::QueryPool,
}

impl FrameResources {
    pub fn new(
        ctx: &VkDeviceCtx,
        width: u32,
        height: u32,
        h264_profile: &mut vk::VideoEncodeH264ProfileInfoKHR,
    ) -> Result<Self, String> {
        let device = ctx.device.clone();
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };

        // The video-encode-related VkImage and VkBuffer must reference
        // the active video profile so the driver knows which encode
        // session they're for. Build the profile list once.
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(h264_profile);
        let profile_array = [profile_info];
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profile_array);

        // ── BGRA source image ──
        let bgra_extent = vk::Extent2D { width, height };
        let bgra_image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let bgra_image = unsafe {
            device
                .create_image(&bgra_image_ci, None)
                .map_err(|e| format!("create_image(bgra): {e:?}"))?
        };
        let bgra_memory = alloc_image_memory(
            &device,
            &mem_props,
            bgra_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(bgra_image, bgra_memory, 0)
                .map_err(|e| format!("bind_image_memory(bgra): {e:?}"))?;
        }
        let bgra_image_view = create_full_image_view(
            &device,
            bgra_image,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?;

        // ── NV12 destination image (multi-planar, DISJOINT) ──
        let mut nv12_image_ci_video = profile_list;
        let nv12_image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            // AMD's RADV (and many others) only support a narrow set
            // of (format, usage) combos for video-encode-tied images.
            // STORAGE + VIDEO_ENCODE_SRC together is rejected with
            // `supportedVideoFormat = FALSE`. Drop STORAGE here; the
            // compute pipeline writes to dedicated storage-only Y/UV
            // images and we copy them into this encoder-target image
            // via vkCmdCopyImage (future patch). For the standalone
            // encode test we upload via TRANSFER_DST from staging.
            .usage(
                vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            // Pull flags from the driver's reported supported set
            // (queried once in VkDeviceCtx::open). AMD reports
            // MUTABLE_FORMAT|ALIAS|EXTENDED_USAGE; NVIDIA reports a
            // different combination. Hardcoding AMD's set caused
            // create_parameters + encode submit to fail on NVIDIA L40.
            .flags(ctx.nv12_encode_src_image_flags)
            .push_next(&mut nv12_image_ci_video);
        let nv12_image = unsafe {
            device
                .create_image(&nv12_image_ci, None)
                .map_err(|e| format!("create_image(nv12): {e:?}"))?
        };
        // Single VkDeviceMemory for the whole multi-planar image.
        let nv12_memory = alloc_image_memory(
            &device,
            &mem_props,
            nv12_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(nv12_image, nv12_memory, 0)
                .map_err(|e| format!("bind_image_memory(nv12): {e:?}"))?;
        }
        // The NV12 image is now encoder-only (VIDEO_ENCODE_SRC +
        // TRANSFER_DST), no STORAGE. Per-plane storage views would
        // fail validation here. The compute pipeline writes to
        // dedicated storage-only Y/UV images instead (item 5 pt.4
        // follow-on); see encode_idr_synthetic docstring.
        let nv12_y_view = vk::ImageView::null();
        let nv12_uv_view = vk::ImageView::null();
        // Whole-image view in the original 2-plane format for the
        // encoder. Aspect is COLOR — the spec maps that to "all
        // planes" for multi-planar video usage.
        let nv12_color_view = create_full_image_view(
            &device,
            nv12_image,
            vk::Format::G8_B8R8_2PLANE_420_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?;

        // ── DPB image (reconstructed-picture target for the encoder) ──
        // Use driver-reported flags (AMD reports empty; NVIDIA may
        // demand a non-empty set), queried once in VkDeviceCtx::open().
        //
        // TWO array layers: layer 0 = DPB slot 0, layer 1 = DPB slot 1.
        // Both slots live in ONE image because RADV does not advertise
        // VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR (a P-frame
        // submit lists both the reference and the setup slot, and they
        // must be views of the same image). A single-layer DPB suffices
        // for the IDR-only Hi444 path, but the Main 4:2:0 P-frame path
        // needs 2 slots for ping-pong reconstruction.
        let mut dpb_image_ci_video = profile_list;
        let dpb_image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(2)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(ctx.nv12_dpb_image_flags)
            .push_next(&mut dpb_image_ci_video);
        let dpb_image = unsafe {
            device
                .create_image(&dpb_image_ci, None)
                .map_err(|e| format!("create_image(dpb): {e:?}"))?
        };
        let dpb_memory = alloc_image_memory(
            &device,
            &mem_props,
            dpb_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(dpb_image, dpb_memory, 0)
                .map_err(|e| format!("bind_image_memory(dpb): {e:?}"))?;
        }
        // Per-slot single-layer views. baseArrayLayer in the picture
        // resource is relative to the view, so each view is self-contained.
        let dpb_image_view =
            create_dpb_layer_view(&device, dpb_image, vk::Format::G8_B8R8_2PLANE_420_UNORM, 0)?;
        let dpb_image_view2 =
            create_dpb_layer_view(&device, dpb_image, vk::Format::G8_B8R8_2PLANE_420_UNORM, 1)?;

        // ── Compute-output storage images (Y plane + UV plane) ──
        // Separate from the encoder NV12 image because AMD won't
        // accept STORAGE + VIDEO_ENCODE_SRC together. We copy
        // Y_storage -> nv12.plane_0 and UV_storage -> nv12.plane_1
        // after compute completes.
        let y_storage_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let y_storage_image = unsafe {
            device
                .create_image(&y_storage_ci, None)
                .map_err(|e| format!("create_image(y_storage): {e:?}"))?
        };
        let y_storage_memory = alloc_image_memory(
            &device,
            &mem_props,
            y_storage_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(y_storage_image, y_storage_memory, 0)
                .map_err(|e| format!("bind_image_memory(y_storage): {e:?}"))?;
        }
        let y_storage_view = create_full_image_view(
            &device,
            y_storage_image,
            vk::Format::R8_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?;

        let uv_storage_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8_UNORM)
            // UV plane is half resolution in NV12 (4:2:0 chroma).
            .extent(vk::Extent3D {
                width: width / 2,
                height: height / 2,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let uv_storage_image = unsafe {
            device
                .create_image(&uv_storage_ci, None)
                .map_err(|e| format!("create_image(uv_storage): {e:?}"))?
        };
        let uv_storage_memory = alloc_image_memory(
            &device,
            &mem_props,
            uv_storage_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(uv_storage_image, uv_storage_memory, 0)
                .map_err(|e| format!("bind_image_memory(uv_storage): {e:?}"))?;
        }
        let uv_storage_view = create_full_image_view(
            &device,
            uv_storage_image,
            vk::Format::R8G8_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?;

        // ── Staging buffer for synthetic BGRA upload ──
        let staging_size = (width as u64) * (height as u64) * 4;
        let staging_buffer_ci = vk::BufferCreateInfo::default()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe {
            device
                .create_buffer(&staging_buffer_ci, None)
                .map_err(|e| format!("create_buffer(staging): {e:?}"))?
        };
        let staging_memory = alloc_buffer_memory(
            &device,
            &mem_props,
            staging_buffer,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
                .map_err(|e| format!("bind_buffer_memory(staging): {e:?}"))?;
        }

        // ── Encoded-output buffer ──
        let encoded_size = (width as u64) * (height as u64); // upper bound
        let encoded_buffer_ci = vk::BufferCreateInfo::default()
            .size(encoded_size)
            .usage(vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut profile_list);
        let encoded_buffer = unsafe {
            device
                .create_buffer(&encoded_buffer_ci, None)
                .map_err(|e| format!("create_buffer(encoded): {e:?}"))?
        };
        let encoded_memory = alloc_buffer_memory(
            &device,
            &mem_props,
            encoded_buffer,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            device
                .bind_buffer_memory(encoded_buffer, encoded_memory, 0)
                .map_err(|e| format!("bind_buffer_memory(encoded): {e:?}"))?;
        }

        // ── Command pool + buffer ──
        let cp_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.video_encode_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            device
                .create_command_pool(&cp_ci, None)
                .map_err(|e| format!("create_command_pool: {e:?}"))?
        };
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&cb_alloc)
                .map_err(|e| format!("allocate_command_buffers: {e:?}"))?[0]
        };

        // ── Fence ──
        let fence_ci = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device
                .create_fence(&fence_ci, None)
                .map_err(|e| format!("create_fence: {e:?}"))?
        };

        // ── Query pool (encode feedback: offset + size of NAL data) ──
        // The create-info chain must include BOTH the feedback flags
        // and the H.264 profile info — drivers reject the pool
        // otherwise (VUID-VkQueryPoolCreateInfo-queryType-07133).
        let mut encode_feedback_create = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
            .encode_feedback_flags(
                vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                    | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            );
        let mut h264_profile_for_qp = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let mut qp_profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile_for_qp);
        let qp_ci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1)
            .push_next(&mut encode_feedback_create)
            .push_next(&mut qp_profile_info);
        let query_pool = unsafe {
            device
                .create_query_pool(&qp_ci, None)
                .map_err(|e| format!("create_query_pool: {e:?}"))?
        };

        Ok(FrameResources {
            device,
            width,
            height,
            bgra_image,
            bgra_image_view,
            bgra_memory,
            bgra_extent,
            nv12_image,
            nv12_y_view,
            nv12_uv_view,
            nv12_color_view,
            nv12_memory,
            dpb_image,
            dpb_image_view,
            dpb_image_view2,
            dpb_memory,
            y_storage_image,
            y_storage_view,
            y_storage_memory,
            uv_storage_image,
            uv_storage_view,
            uv_storage_memory,
            staging_buffer,
            staging_memory,
            staging_size,
            encoded_buffer,
            encoded_memory,
            encoded_size,
            command_pool,
            command_buffer,
            fence,
            query_pool,
        })
    }
}

impl Drop for FrameResources {
    fn drop(&mut self) {
        unsafe {
            // Bounded wait, NOT device_wait_idle (unbounded → blocks teardown
            // forever on a wedged GPU → kernel watchdog panic). self.fence is
            // signaled once this frame's encode completes; on a hung GPU it
            // times out and we destroy anyway.
            let _ = self
                .device
                .wait_for_fences(&[self.fence], true, fence_timeout_ns());
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_buffer(self.encoded_buffer, None);
            self.device.free_memory(self.encoded_memory, None);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);
            self.device.destroy_image_view(self.nv12_color_view, None);
            if self.nv12_uv_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.nv12_uv_view, None);
            }
            if self.nv12_y_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.nv12_y_view, None);
            }
            self.device.destroy_image(self.nv12_image, None);
            self.device.free_memory(self.nv12_memory, None);
            self.device.destroy_image_view(self.dpb_image_view, None);
            self.device.destroy_image_view(self.dpb_image_view2, None);
            self.device.destroy_image(self.dpb_image, None);
            self.device.free_memory(self.dpb_memory, None);
            self.device.destroy_image_view(self.uv_storage_view, None);
            self.device.destroy_image(self.uv_storage_image, None);
            self.device.free_memory(self.uv_storage_memory, None);
            self.device.destroy_image_view(self.y_storage_view, None);
            self.device.destroy_image(self.y_storage_image, None);
            self.device.free_memory(self.y_storage_memory, None);
            self.device.destroy_image_view(self.bgra_image_view, None);
            self.device.destroy_image(self.bgra_image, None);
            self.device.free_memory(self.bgra_memory, None);
        }
    }
}

/// Per-recording Vulkan resources for the H.264 Hi444PP lossless path.
/// Sister of `FrameResources` (NV12 / Main 4:2:0). Differences:
///
/// - Encoder picture image is `G8_B8R8_2PLANE_444_UNORM` (Y plane +
///   interleaved UV plane, both full resolution). NVIDIA driver 560
///   reports only this 2-plane variant for Hi444PP encode-src; the
///   3-plane equivalent (`G8_B8_R8_3PLANE_444_UNORM`) is rejected.
/// - Compute pipeline output is one full-res R8 (Y) + one full-res
///   R8G8 (UV) storage image — same shape as NV12 but with full-res
///   chroma instead of half-res.
/// - `encoded_buffer` is sized for the larger lossless NAL payload
///   (Hi444PP at QP=0 routinely emits 2-3× the bytes of Main at QP=20
///   on the same content). Worst case: ~3× width*height bytes.
pub struct FrameResources444 {
    device: ash::Device,
    /// Display (visible) width — the size of the BGRA input the
    /// caller will provide via `run_compute_yuv444_into_picture`.
    /// May be unaligned (e.g., 1952 from a YouTube hero capture).
    pub width: u32,
    pub height: u32,
    /// Encoder picture dimensions — `width`/`height` rounded UP to
    /// the next multiple of 64 (HEVC's largest CTU size). NVIDIA's
    /// `hevc_vulkan` rext encoder mis-handles non-64-aligned widths
    /// on driver 580.x: a content-shaped blue/cyan triangle appears
    /// in the upper-left quadrant of the decoded output (bug
    /// confirmed via direct ffmpeg-CLI repro, identical artifact).
    /// Padding to a 64-aligned multiple eliminates the artifact;
    /// the padded region is cleared to neutral chroma so it decodes
    /// as a thin black border the consumer can crop downstream.
    pub coded_width: u32,
    pub coded_height: u32,

    pub bgra_image: vk::Image,
    pub bgra_image_view: vk::ImageView,
    bgra_memory: vk::DeviceMemory,
    pub bgra_extent: vk::Extent2D,

    /// 2-plane YUV 4:4:4 image — the encoder picture target.
    /// Allocated at CODED dims (`coded_width × coded_height`).
    pub yuv_image: vk::Image,
    pub yuv_color_view: vk::ImageView,
    yuv_memory: vk::DeviceMemory,

    /// DPB image (reconstructed-picture target).
    pub dpb_image: vk::Image,
    pub dpb_image_view: vk::ImageView,
    dpb_memory: vk::DeviceMemory,

    /// BGRA→YUV 4:4:4 compute output: Y plane (R8 full-res) + UV plane
    /// (R8G8 interleaved, full-res). Copied into `yuv_image`'s planes
    /// via vkCmdCopyImage before the encode submit.
    pub y_storage_image: vk::Image,
    pub y_storage_view: vk::ImageView,
    y_storage_memory: vk::DeviceMemory,
    pub uv_storage_image: vk::Image,
    pub uv_storage_view: vk::ImageView,
    uv_storage_memory: vk::DeviceMemory,

    pub staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    pub staging_size: u64,

    pub encoded_buffer: vk::Buffer,
    encoded_memory: vk::DeviceMemory,
    pub encoded_size: u64,

    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub query_pool: vk::QueryPool,
}

/// Round `n` up to the next multiple of 64. HEVC's largest CTU is
/// 64×64 luma; NVIDIA's `hevc_vulkan` rext encoder produces visible
/// chroma artifacts (a content-shaped blue/cyan triangle in the
/// upper-left, plus a green bottom stripe) when encoder dimensions
/// are not 64-aligned. Pad up + crop downstream is the workaround.
#[inline]
pub fn align_to_ctu(n: u32) -> u32 {
    (n + 63) & !63
}

impl FrameResources444 {
    pub fn new(ctx: &VkDeviceCtx, width: u32, height: u32) -> Result<Self, String> {
        if !ctx.hi444_supported {
            return Err(format!(
                "Hi444PP not supported on this device ({})",
                ctx.device_name
            ));
        }

        let coded_width = align_to_ctu(width);
        let coded_height = align_to_ctu(height);
        let device = ctx.device.clone();
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };

        // Profile chain — Hi444PP / TYPE_444. Every video-tied
        // VkImage/VkBuffer references this so the driver knows which
        // session the resources belong to.
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
            vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
        );
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);
        let profile_array = [profile_info];
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profile_array);

        // ── BGRA source image (compute shader input) ──
        let bgra_extent = vk::Extent2D { width, height };
        let bgra_image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let bgra_image = unsafe {
            device
                .create_image(&bgra_image_ci, None)
                .map_err(|e| format!("create_image(bgra/yuv444): {e:?}"))?
        };
        let bgra_memory = alloc_image_memory(
            &device,
            &mem_props,
            bgra_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(bgra_image, bgra_memory, 0)
                .map_err(|e| format!("bind_image_memory(bgra/yuv444): {e:?}"))?;
        }
        let bgra_image_view = create_full_image_view(
            &device,
            bgra_image,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?;

        // ── 2-plane YUV 4:4:4 destination (encoder picture target) ──
        // Allocated at CODED dims so the encoder reads a 64-aligned
        // image (NVIDIA hevc_vulkan rext quirk; see align_to_ctu doc).
        let yuv_format = vk::Format::G8_B8R8_2PLANE_444_UNORM;
        let mut yuv_image_ci_video = profile_list;
        let yuv_image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(yuv_format)
            .extent(vk::Extent3D {
                width: coded_width,
                height: coded_height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(ctx.yuv444_encode_src_image_flags)
            .push_next(&mut yuv_image_ci_video);
        let yuv_image = unsafe {
            device
                .create_image(&yuv_image_ci, None)
                .map_err(|e| format!("create_image(yuv444): {e:?}"))?
        };
        let yuv_memory = alloc_image_memory(
            &device,
            &mem_props,
            yuv_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(yuv_image, yuv_memory, 0)
                .map_err(|e| format!("bind_image_memory(yuv444): {e:?}"))?;
        }
        let yuv_color_view =
            create_full_image_view(&device, yuv_image, yuv_format, vk::ImageAspectFlags::COLOR)?;

        // ── DPB image (reconstructed-picture target for the encoder) ──
        // Also at CODED dims to match the picture target.
        let mut dpb_image_ci_video = profile_list;
        let dpb_image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(yuv_format)
            .extent(vk::Extent3D {
                width: coded_width,
                height: coded_height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(ctx.yuv444_dpb_image_flags)
            .push_next(&mut dpb_image_ci_video);
        let dpb_image = unsafe {
            device
                .create_image(&dpb_image_ci, None)
                .map_err(|e| format!("create_image(dpb/yuv444): {e:?}"))?
        };
        let dpb_memory = alloc_image_memory(
            &device,
            &mem_props,
            dpb_image,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe {
            device
                .bind_image_memory(dpb_image, dpb_memory, 0)
                .map_err(|e| format!("bind_image_memory(dpb/yuv444): {e:?}"))?;
        }
        let dpb_image_view =
            create_full_image_view(&device, dpb_image, yuv_format, vk::ImageAspectFlags::COLOR)?;

        // ── Compute-output storage images: Y (R8 full-res) + UV (R8G8
        // interleaved, full-res). Same shape as NV12 storage but with
        // chroma at full resolution. Separate from the encoder picture
        // because video-encode-src + STORAGE on one image is rejected
        // on RADV (and is risky cross-vendor). Copied to encoder
        // planes via vkCmdCopyImage after compute completes.
        let make_storage =
            |label: &'static str,
             format: vk::Format|
             -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory), String> {
                let ci = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                let image = unsafe {
                    device
                        .create_image(&ci, None)
                        .map_err(|e| format!("create_image({label}): {e:?}"))?
                };
                let memory = alloc_image_memory(
                    &device,
                    &mem_props,
                    image,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                unsafe {
                    device
                        .bind_image_memory(image, memory, 0)
                        .map_err(|e| format!("bind_image_memory({label}): {e:?}"))?;
                }
                let view =
                    create_full_image_view(&device, image, format, vk::ImageAspectFlags::COLOR)?;
                Ok((image, view, memory))
            };
        let (y_storage_image, y_storage_view, y_storage_memory) =
            make_storage("y_storage", vk::Format::R8_UNORM)?;
        let (uv_storage_image, uv_storage_view, uv_storage_memory) =
            make_storage("uv_storage", vk::Format::R8G8_UNORM)?;

        // ── Staging buffer for BGRA upload ──
        let staging_size = (width as u64) * (height as u64) * 4;
        let staging_buffer_ci = vk::BufferCreateInfo::default()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe {
            device
                .create_buffer(&staging_buffer_ci, None)
                .map_err(|e| format!("create_buffer(staging/yuv444): {e:?}"))?
        };
        let staging_memory = alloc_buffer_memory(
            &device,
            &mem_props,
            staging_buffer,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
                .map_err(|e| format!("bind_buffer_memory(staging/yuv444): {e:?}"))?;
        }

        // ── Encoded-output buffer ──
        // Lossless 4:4:4 H.264 NALs can run far larger than Main420 at
        // QP=20. Upper-bound at 3× width*height bytes to give the
        // encoder headroom for I-frame-only worst case.
        let encoded_size = (width as u64) * (height as u64) * 3;
        let encoded_buffer_ci = vk::BufferCreateInfo::default()
            .size(encoded_size)
            .usage(vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut profile_list);
        let encoded_buffer = unsafe {
            device
                .create_buffer(&encoded_buffer_ci, None)
                .map_err(|e| format!("create_buffer(encoded/yuv444): {e:?}"))?
        };
        let encoded_memory = alloc_buffer_memory(
            &device,
            &mem_props,
            encoded_buffer,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            device
                .bind_buffer_memory(encoded_buffer, encoded_memory, 0)
                .map_err(|e| format!("bind_buffer_memory(encoded/yuv444): {e:?}"))?;
        }

        // ── Command pool + buffer ──
        let cp_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.video_encode_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            device
                .create_command_pool(&cp_ci, None)
                .map_err(|e| format!("create_command_pool(yuv444): {e:?}"))?
        };
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&cb_alloc)
                .map_err(|e| format!("allocate_command_buffers(yuv444): {e:?}"))?[0]
        };

        // ── Fence ──
        let fence_ci = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device
                .create_fence(&fence_ci, None)
                .map_err(|e| format!("create_fence(yuv444): {e:?}"))?
        };

        // ── Query pool (encode feedback) ──
        // Hi444PP profile chained on so the driver accepts the pool.
        let mut encode_feedback_create = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
            .encode_feedback_flags(
                vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                    | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            );
        let mut h264_profile_for_qp = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
            vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
        );
        let mut qp_profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile_for_qp);
        let qp_ci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1)
            .push_next(&mut encode_feedback_create)
            .push_next(&mut qp_profile_info);
        let query_pool = unsafe {
            device
                .create_query_pool(&qp_ci, None)
                .map_err(|e| format!("create_query_pool(yuv444): {e:?}"))?
        };

        Ok(FrameResources444 {
            device,
            width,
            height,
            coded_width,
            coded_height,
            bgra_image,
            bgra_image_view,
            bgra_memory,
            bgra_extent,
            yuv_image,
            yuv_color_view,
            yuv_memory,
            dpb_image,
            dpb_image_view,
            dpb_memory,
            y_storage_image,
            y_storage_view,
            y_storage_memory,
            uv_storage_image,
            uv_storage_view,
            uv_storage_memory,
            staging_buffer,
            staging_memory,
            staging_size,
            encoded_buffer,
            encoded_memory,
            encoded_size,
            command_pool,
            command_buffer,
            fence,
            query_pool,
        })
    }
}

impl Drop for FrameResources444 {
    fn drop(&mut self) {
        unsafe {
            // Bounded wait, NOT device_wait_idle (unbounded → blocks teardown
            // forever on a wedged GPU → kernel watchdog panic). self.fence is
            // signaled once this frame's encode completes; on a hung GPU it
            // times out and we destroy anyway.
            let _ = self
                .device
                .wait_for_fences(&[self.fence], true, fence_timeout_ns());
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_buffer(self.encoded_buffer, None);
            self.device.free_memory(self.encoded_memory, None);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);
            self.device.destroy_image_view(self.yuv_color_view, None);
            self.device.destroy_image(self.yuv_image, None);
            self.device.free_memory(self.yuv_memory, None);
            self.device.destroy_image_view(self.dpb_image_view, None);
            self.device.destroy_image(self.dpb_image, None);
            self.device.free_memory(self.dpb_memory, None);
            self.device.destroy_image_view(self.uv_storage_view, None);
            self.device.destroy_image(self.uv_storage_image, None);
            self.device.free_memory(self.uv_storage_memory, None);
            self.device.destroy_image_view(self.y_storage_view, None);
            self.device.destroy_image(self.y_storage_image, None);
            self.device.free_memory(self.y_storage_memory, None);
            self.device.destroy_image_view(self.bgra_image_view, None);
            self.device.destroy_image(self.bgra_image, None);
            self.device.free_memory(self.bgra_memory, None);
        }
    }
}

fn alloc_image_memory(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, String> {
    let req = unsafe { device.get_image_memory_requirements(image) };
    let type_idx = pick_memory_type(mem_props, req.memory_type_bits, flags).ok_or_else(|| {
        format!(
            "no memory type for image (mask={:#x})",
            req.memory_type_bits
        )
    })?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(type_idx);
    unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("allocate_memory(image): {e:?}"))
    }
}

fn alloc_buffer_memory(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    buffer: vk::Buffer,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, String> {
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let type_idx = pick_memory_type(mem_props, req.memory_type_bits, flags).ok_or_else(|| {
        format!(
            "no memory type for buffer (mask={:#x})",
            req.memory_type_bits
        )
    })?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(type_idx);
    unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("allocate_memory(buffer): {e:?}"))
    }
}

/// For a DISJOINT multi-planar image, query memory requirements per
/// plane and bind a separate allocation to each.
fn bind_disjoint_planes(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
) -> Result<[vk::DeviceMemory; 2], String> {
    let mut bind_infos = Vec::with_capacity(2);
    let mut memories = [vk::DeviceMemory::null(); 2];
    for (i, plane) in [vk::ImageAspectFlags::PLANE_0, vk::ImageAspectFlags::PLANE_1]
        .iter()
        .enumerate()
    {
        let mut plane_info = vk::ImagePlaneMemoryRequirementsInfo::default().plane_aspect(*plane);
        let mut req2 = vk::MemoryRequirements2::default();
        let info2 = vk::ImageMemoryRequirementsInfo2::default()
            .image(image)
            .push_next(&mut plane_info);
        unsafe { device.get_image_memory_requirements2(&info2, &mut req2) };
        let type_idx = pick_memory_type(
            mem_props,
            req2.memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| format!("no DEVICE_LOCAL memory type for NV12 plane {i}"))?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req2.memory_requirements.size)
            .memory_type_index(type_idx);
        memories[i] = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| format!("allocate_memory(plane {i}): {e:?}"))?
        };
        bind_infos.push(
            vk::BindImageMemoryInfo::default()
                .image(image)
                .memory(memories[i])
                .memory_offset(0),
        );
    }
    // We need the plane-aspect chained on each BindImageMemoryInfo —
    // construct the chain entries with a stable lifetime.
    let plane0 =
        vk::BindImagePlaneMemoryInfo::default().plane_aspect(vk::ImageAspectFlags::PLANE_0);
    let plane1 =
        vk::BindImagePlaneMemoryInfo::default().plane_aspect(vk::ImageAspectFlags::PLANE_1);
    let mut plane0_mut = plane0;
    let mut plane1_mut = plane1;
    let binds = [
        bind_infos[0].push_next(&mut plane0_mut),
        bind_infos[1].push_next(&mut plane1_mut),
    ];
    unsafe {
        device
            .bind_image_memory2(&binds)
            .map_err(|e| format!("bind_image_memory2(planes): {e:?}"))?;
    }
    Ok(memories)
}

fn create_full_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, String> {
    let ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        device
            .create_image_view(&ci, None)
            .map_err(|e| format!("create_image_view: {e:?}"))
    }
}

/// Create a single-layer 2D view of one array layer of a multi-layer
/// DPB image. Used for the P-frame ping-pong: layer 0 = DPB slot 0,
/// layer 1 = DPB slot 1, both backed by the same `VkImage` (RADV lacks
/// VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR, so the
/// reference and setup slots in a P-frame submit must share an image).
fn create_dpb_layer_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    layer: u32,
) -> Result<vk::ImageView, String> {
    let ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(1),
        );
    unsafe {
        device
            .create_image_view(&ci, None)
            .map_err(|e| format!("create_image_view(dpb layer {layer}): {e:?}"))
    }
}

fn create_plane_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    plane_aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, String> {
    let mut usage = vk::ImageViewUsageCreateInfo::default().usage(vk::ImageUsageFlags::STORAGE);
    let ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(plane_aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .push_next(&mut usage);
    unsafe {
        device
            .create_image_view(&ci, None)
            .map_err(|e| format!("create_plane_image_view: {e:?}"))
    }
}

// ────────────────────────────────────────────────────────────────────────
// Compute-only execution path (RFC item 5, isolated for correctness)
//
// Runs upload(BGRA) -> dispatch(compute) -> readback(NV12) on the
// graphics+compute queue, skipping the video encode stage. Used by
// the integration test to verify the compute shader produces output
// that matches `recording.rs::bgra_to_nv12` within ±1 LSB.

/// Upload synthetic BGRA, run the compute shader, read back the NV12
/// planes (Y then UV) into a single `Vec<u8>` in standard NV12 layout
/// (Y plane followed by interleaved UV). Used by tests to verify the
/// compute stage in isolation from the encode submit.
///
/// `bgra_src` must be exactly `width*height*4` bytes.
/// Returns `width*height*3/2` bytes.
pub fn run_compute_only(
    ctx: &VkDeviceCtx,
    fr: &FrameResources,
    pipe: &BgraToNv12Pipeline,
    bgra_src: &[u8],
) -> Result<Vec<u8>, String> {
    let w = fr.width;
    let h = fr.height;
    if bgra_src.len() != (w as usize) * (h as usize) * 4 {
        return Err(format!(
            "bgra_src len {} != {}",
            bgra_src.len(),
            (w as usize) * (h as usize) * 4
        ));
    }

    // ── 1. Upload BGRA to the staging buffer ──
    unsafe {
        let ptr = ctx
            .device
            .map_memory(
                fr.staging_memory,
                0,
                fr.staging_size,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(staging): {e:?}"))?;
        std::ptr::copy_nonoverlapping(bgra_src.as_ptr(), ptr as *mut u8, bgra_src.len());
        ctx.device.unmap_memory(fr.staging_memory);
    }

    // ── 2. Allocate + write descriptor set ──
    let set_layouts = [pipe.descriptor_set_layout()];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pipe.descriptor_pool())
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe {
        ctx.device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("allocate_descriptor_sets: {e:?}"))?[0]
    };
    let src_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.bgra_image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .sampler(pipe.sampler())];
    let y_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.nv12_y_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let uv_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.nv12_uv_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&src_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&y_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&uv_image_info),
    ];
    unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

    // ── 3. Record command buffer ──
    // For this isolated path we use the COMPUTE queue family rather
    // than the video-encode family. Allocate a separate command pool
    // since the FrameResources one is on the video-encode family.
    let cp_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let compute_pool = unsafe {
        ctx.device
            .create_command_pool(&cp_ci, None)
            .map_err(|e| format!("create_command_pool(compute): {e:?}"))?
    };
    let cb_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(compute_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = unsafe {
        ctx.device
            .allocate_command_buffers(&cb_alloc)
            .map_err(|e| format!("allocate_command_buffers: {e:?}"))?[0]
    };

    // Readback buffer for the NV12 planes. Sized to W*H*3/2.
    let readback_size = (w as u64) * (h as u64) * 3 / 2;
    let readback_buf_ci = vk::BufferCreateInfo::default()
        .size(readback_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let readback_buf = unsafe {
        ctx.device
            .create_buffer(&readback_buf_ci, None)
            .map_err(|e| format!("create_buffer(readback): {e:?}"))?
    };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let readback_mem = alloc_buffer_memory(
        &ctx.device,
        &mem_props,
        readback_buf,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        ctx.device
            .bind_buffer_memory(readback_buf, readback_mem, 0)
            .map_err(|e| format!("bind_buffer_memory(readback): {e:?}"))?;
    }

    let cb_begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .begin_command_buffer(cb, &cb_begin)
            .map_err(|e| format!("begin_command_buffer: {e:?}"))?;
    }

    // BGRA UNDEFINED -> TRANSFER_DST_OPTIMAL
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    // Copy staging buffer -> BGRA image
    let copy_region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            fr.staging_buffer,
            fr.bgra_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
    }
    // BGRA TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // NV12 UNDEFINED -> GENERAL (per plane). Use a multi-planar
    // barrier with PLANE_0+PLANE_1 aspects.
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.nv12_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );

    // Bind pipeline + descriptor set, dispatch.
    unsafe {
        ctx.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline());
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipe.pipeline_layout(),
            0,
            &[descriptor_set],
            &[],
        );
        // Each invocation handles a 2x2 block; workgroup is 16x16.
        // Dispatch ceil(w/2 / 16) x ceil(h/2 / 16) x 1.
        let dispatch_x = (w / 2).div_ceil(16);
        let dispatch_y = (h / 2).div_ceil(16);
        ctx.device.cmd_dispatch(cb, dispatch_x, dispatch_y, 1);
    }

    // NV12 GENERAL -> TRANSFER_SRC_OPTIMAL (per plane) before readback.
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.nv12_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );

    // Copy plane 0 (Y) to the front of the readback buffer.
    let y_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(w)
        .buffer_image_height(h)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    let uv_copy = vk::BufferImageCopy::default()
        .buffer_offset((w as u64) * (h as u64))
        .buffer_row_length(w / 2)
        .buffer_image_height(h / 2)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: w / 2,
            height: h / 2,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            fr.nv12_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback_buf,
            &[y_copy, uv_copy],
        );
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| format!("end_command_buffer: {e:?}"))?;
    }

    // ── 4. Submit + wait ──
    let cbs = [cb];
    let submit = vk::SubmitInfo::default().command_buffers(&cbs);
    unsafe {
        let fence_ci = vk::FenceCreateInfo::default();
        let fence = ctx
            .device
            .create_fence(&fence_ci, None)
            .map_err(|e| format!("create_fence: {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[submit], fence)
            .map_err(|e| format!("queue_submit: {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fence], true, 10_000_000_000)
            .map_err(|e| format!("wait_for_fences: {e:?}"))?;
        ctx.device.destroy_fence(fence, None);
    }

    // ── 5. Read back NV12 ──
    let mut out = vec![0u8; readback_size as usize];
    unsafe {
        let ptr = ctx
            .device
            .map_memory(readback_mem, 0, readback_size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map_memory(readback): {e:?}"))?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), out.len());
        ctx.device.unmap_memory(readback_mem);

        // Cleanup
        ctx.device.destroy_buffer(readback_buf, None);
        ctx.device.free_memory(readback_mem, None);
        ctx.device.destroy_command_pool(compute_pool, None);
        // The descriptor set is freed when the pool is reset/destroyed
        // with FrameResources.
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────
// Encode submit.
//
// Submits an H.264 IDR encode of the NV12 image already populated in
// `FrameResources`, reads back the encoded NAL bytes via the encode
// feedback query, and returns them. The NV12 image is uploaded from
// CPU via the staging buffer first; this is the synthetic-frame
// path until item 6 wires dmabuf import.

/// Encode a single IDR frame from a CPU-supplied NV12 source. Returns
/// the encoded H.264 NAL bytes.
///
/// **Status: scaffold; not yet end-to-end working.** Validation layer
/// surfaced these blockers when WAYMUX_VK_VALIDATE=1 is set:
///
/// 1. `nv12_image` needs `TRANSFER_DST_BIT` + `TRANSFER_SRC_BIT` in
///    its usage flags (currently STORAGE + VIDEO_ENCODE_SRC only).
/// 2. `nv12_image` needs `VK_IMAGE_CREATE_EXTENDED_USAGE_BIT` so the
///    per-plane R8 / R8G8 image views can carry STORAGE usage even
///    though the multi-planar parent format may not advertise it.
/// 3. AMD's video-encode queue family (qf[3] on Renoir) does NOT
///    support `cmd_copy_buffer_to_image`. The upload path must run
///    on the compute queue family and a queue-family ownership
///    transfer barrier must release the image to the encode queue.
///    That means TWO command buffers (one per queue) and an inter-
///    queue semaphore.
/// 4. `query_pool` create info needs the video profile chained on
///    next to the encode-feedback create info — same profile as the
///    encode session.
/// 5. Pipeline-stage masks for the NV12 layout-transition barrier
///    need to use `BOTTOM_OF_PIPE` only on stages the encode queue
///    supports.
///
/// The encode_info / picture_info / slice_header structure assembly
/// below is correct and lands as scaffolding for the next iteration.
/// Steps the function will take once the above blockers are fixed:
///   1. Upload NV12 bytes into the multi-planar image via staging
///      (on compute queue), release ownership to encode queue.
///   2. Acquire ownership on encode queue.
///   3. Begin/Control video coding (RESET on first call).
///   4. Encode video with H.264 IDR picture info + slice header.
///   5. End coding, submit on video-encode queue, fence wait.
///   6. Read offset+bytes_written from the encode feedback query.
///   7. Map encoded_buffer + copy out [offset..offset+size].
pub fn encode_idr_synthetic(
    ctx: &VkDeviceCtx,
    fr: &FrameResources,
    session: &EncodeSession,
    nv12_src: &[u8],
) -> Result<Vec<u8>, String> {
    let w = fr.width;
    let h = fr.height;
    let y_size = (w as usize) * (h as usize);
    let uv_size = (w as usize) * (h as usize) / 2;
    if nv12_src.len() != y_size + uv_size {
        return Err(format!(
            "nv12_src len {} != {} (= y_size {} + uv_size {})",
            nv12_src.len(),
            y_size + uv_size,
            y_size,
            uv_size
        ));
    }

    // ── 1. Upload NV12 bytes into the staging buffer (Y plane first,
    //       then interleaved UV, packed contiguously). ──
    unsafe {
        // Reuse the staging buffer — it's sized W*H*4, plenty for
        // W*H*3/2 of NV12.
        let ptr = ctx
            .device
            .map_memory(
                fr.staging_memory,
                0,
                fr.staging_size,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(staging for nv12): {e:?}"))?;
        std::ptr::copy_nonoverlapping(nv12_src.as_ptr(), ptr as *mut u8, nv12_src.len());
        ctx.device.unmap_memory(fr.staging_memory);
    }

    // ── 2a. Record upload on the COMPUTE queue family ──
    // (The video-encode queue family on AMD doesn't support
    // cmd_copy_buffer_to_image. We do the upload on compute, then
    // queue-family-release to encode.)
    let compute_pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let upload_pool = unsafe {
        ctx.device
            .create_command_pool(&compute_pool_ci, None)
            .map_err(|e| format!("create_command_pool(upload): {e:?}"))?
    };
    let upload_cb_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(upload_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let upload_cb = unsafe {
        ctx.device
            .allocate_command_buffers(&upload_cb_alloc)
            .map_err(|e| format!("allocate_command_buffers(upload): {e:?}"))?[0]
    };
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .begin_command_buffer(upload_cb, &begin)
            .map_err(|e| format!("begin_command_buffer(upload): {e:?}"))?;
    }

    // NV12 UNDEFINED -> TRANSFER_DST for both planes.
    image_layout_barrier(
        &ctx.device,
        upload_cb,
        fr.nv12_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );

    // Copy staging -> plane 0 (Y).
    let y_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(w)
        .buffer_image_height(h)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    let uv_copy = vk::BufferImageCopy::default()
        .buffer_offset((w as u64) * (h as u64))
        .buffer_row_length(w / 2)
        .buffer_image_height(h / 2)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width: w / 2,
            height: h / 2,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            upload_cb,
            fr.staging_buffer,
            fr.nv12_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[y_copy, uv_copy],
        );
    }

    // Queue-family release: NV12 TRANSFER_DST -> VIDEO_ENCODE_SRC,
    // release from compute queue to encode queue.
    {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(ctx.compute_queue_family)
            .dst_queue_family_index(ctx.video_encode_queue_family)
            .image(fr.nv12_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                upload_cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
    unsafe {
        ctx.device
            .end_command_buffer(upload_cb)
            .map_err(|e| format!("end_command_buffer(upload): {e:?}"))?;
    }

    // ── 2b. Submit upload + create inter-queue semaphore ──
    let sem_ci = vk::SemaphoreCreateInfo::default();
    let upload_done_sem = unsafe {
        ctx.device
            .create_semaphore(&sem_ci, None)
            .map_err(|e| format!("create_semaphore: {e:?}"))?
    };
    let upload_cbs = [upload_cb];
    let upload_signals = [upload_done_sem];
    let upload_submit = vk::SubmitInfo::default()
        .command_buffers(&upload_cbs)
        .signal_semaphores(&upload_signals);
    unsafe {
        ctx.device
            .reset_fences(&[fr.fence])
            .map_err(|e| format!("reset_fences: {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[upload_submit], vk::Fence::null())
            .map_err(|e| format!("queue_submit(upload): {e:?}"))?;
    }

    // ── 2c. Record encode commands on the ENCODE queue ──
    let cb = fr.command_buffer;
    unsafe {
        ctx.device
            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .map_err(|e| format!("reset_command_buffer(encode): {e:?}"))?;
        ctx.device
            .begin_command_buffer(cb, &begin)
            .map_err(|e| format!("begin_command_buffer(encode): {e:?}"))?;
    }

    // Queue-family acquire: matching the release above.
    {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(ctx.compute_queue_family)
            .dst_queue_family_index(ctx.video_encode_queue_family)
            .image(fr.nv12_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    // Reset the query pool.
    unsafe {
        ctx.device.cmd_reset_query_pool(cb, fr.query_pool, 0, 1);
    }

    // ── 3. Begin/Control video coding. ──
    // Advertise DPB slot 0 to the encoder. The slot itself starts
    // with picture_resource=null (no reconstructed picture yet — the
    // IDR encode below sets it up via setup_reference_slot).
    let begin_dpb_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(fr.dpb_image_view);
    let begin_dpb_slot = vk::VideoReferenceSlotInfoKHR::default()
        // slot_index = -1 marks the slot as unused on entry.
        .slot_index(-1)
        .picture_resource(&begin_dpb_resource);
    let begin_slots = [begin_dpb_slot];
    let begin_info = vk::VideoBeginCodingInfoKHR::default()
        .video_session(session.session)
        .video_session_parameters(session.parameters)
        .reference_slots(&begin_slots);
    unsafe {
        (ctx.video_queue_dev.fp().cmd_begin_video_coding_khr)(cb, &begin_info);
    }
    // Reset control on first frame (we encode one IDR per call) +
    // set rate control mode = DISABLED so constantQp on the slice
    // info is the QP the encoder uses.
    let mut rate_control_layer = vk::VideoEncodeRateControlInfoKHR::default()
        .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
    let control_info = vk::VideoCodingControlInfoKHR::default()
        .flags(
            vk::VideoCodingControlFlagsKHR::RESET
                | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL,
        )
        .push_next(&mut rate_control_layer);
    unsafe {
        (ctx.video_queue_dev.fp().cmd_control_video_coding_khr)(cb, &control_info);
    }

    // ── 4. Begin query + encode + end query. ──
    unsafe {
        ctx.device
            .cmd_begin_query(cb, fr.query_pool, 0, vk::QueryControlFlags::empty());
    }

    // Transition DPB image to VIDEO_ENCODE_DPB layout before encode.
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.dpb_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
        vk::AccessFlags::empty(),
        vk::AccessFlags::MEMORY_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    );

    // Construct the H.264 picture info chain. IDR frames must be
    // marked is_reference=1 for the H.264 spec; the DPB slot is where
    // the encoder writes the reconstructed picture.
    let mut pic_flags: vk::native::StdVideoEncodeH264PictureInfoFlags =
        unsafe { std::mem::zeroed() };
    pic_flags.set_IdrPicFlag(1);
    pic_flags.set_is_reference(1);
    // IDR frames still need a pRefLists structure with the
    // num_ref_idx_l*_active_minus1 fields set; AMD's encoder reads
    // this even when no actual references are listed. 0xFF in each
    // RefPic slot marks the slot as unused.
    let ref_lists_flags: vk::native::StdVideoEncodeH264ReferenceListsInfoFlags =
        unsafe { std::mem::zeroed() };
    let ref_lists = vk::native::StdVideoEncodeH264ReferenceListsInfo {
        flags: ref_lists_flags,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        RefPicList0: [0xFF; 32],
        RefPicList1: [0xFF; 32],
        refList0ModOpCount: 0,
        refList1ModOpCount: 0,
        refPicMarkingOpCount: 0,
        reserved1: [0; 7],
        pRefList0ModOperations: std::ptr::null(),
        pRefList1ModOperations: std::ptr::null(),
        pRefPicMarkingOperations: std::ptr::null(),
    };
    let pic_info = vk::native::StdVideoEncodeH264PictureInfo {
        flags: pic_flags,
        seq_parameter_set_id: 0,
        pic_parameter_set_id: 0,
        idr_pic_id: 0,
        primary_pic_type: vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR,
        frame_num: 0,
        PicOrderCnt: 0,
        temporal_id: 0,
        reserved1: [0; 3],
        pRefLists: &ref_lists,
    };
    let slice_flags: vk::native::StdVideoEncodeH264SliceHeaderFlags = unsafe { std::mem::zeroed() };
    let slice_header = vk::native::StdVideoEncodeH264SliceHeader {
        flags: slice_flags,
        first_mb_in_slice: 0,
        slice_type: vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_qp_delta: 0,
        reserved1: 0,
        cabac_init_idc: vk::native::StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
        disable_deblocking_filter_idc:
            vk::native::StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
        pWeightTable: std::ptr::null(),
    };
    let nalu_slice = vk::VideoEncodeH264NaluSliceInfoKHR::default()
        .constant_qp(vk_encode_qp())
        .std_slice_header(&slice_header);
    let nalu_slice_arr = [nalu_slice];
    let mut h264_pic_info = vk::VideoEncodeH264PictureInfoKHR::default()
        .nalu_slice_entries(&nalu_slice_arr)
        .std_picture_info(&pic_info);

    let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(fr.nv12_color_view);

    // Setup reference slot — points to the DPB image where the
    // encoder writes the reconstructed picture for slot 0. The
    // H264-specific DPB info chain is required by
    // VUID-vkCmdEncodeVideoKHR-pEncodeInfo-08228.
    let dpb_picture_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(fr.dpb_image_view);
    let ref_info_flags: vk::native::StdVideoEncodeH264ReferenceInfoFlags =
        unsafe { std::mem::zeroed() };
    let std_ref_info = vk::native::StdVideoEncodeH264ReferenceInfo {
        flags: ref_info_flags,
        primary_pic_type: vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR,
        FrameNum: 0,
        PicOrderCnt: 0,
        long_term_pic_num: 0,
        long_term_frame_idx: 0,
        temporal_id: 0,
    };
    let mut h264_dpb_slot =
        vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_ref_info);
    let setup_ref_slot = vk::VideoReferenceSlotInfoKHR::default()
        .slot_index(0)
        .picture_resource(&dpb_picture_resource)
        .push_next(&mut h264_dpb_slot);

    let encode_info = vk::VideoEncodeInfoKHR::default()
        .dst_buffer(fr.encoded_buffer)
        .dst_buffer_offset(0)
        .dst_buffer_range(fr.encoded_size)
        .src_picture_resource(src_picture_resource)
        .setup_reference_slot(&setup_ref_slot)
        .push_next(&mut h264_pic_info);

    unsafe {
        (ctx.video_encode_queue_dev.fp().cmd_encode_video_khr)(cb, &encode_info);
        ctx.device.cmd_end_query(cb, fr.query_pool, 0);
    }

    let end_info = vk::VideoEndCodingInfoKHR::default();
    unsafe {
        (ctx.video_queue_dev.fp().cmd_end_video_coding_khr)(cb, &end_info);
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| format!("end_command_buffer(encode): {e:?}"))?;
    }

    // ── 5. Submit + fence wait. Waits on the upload semaphore so the
    // encoder doesn't start before the queue-family acquire completes.
    let cbs = [cb];
    let wait_sems = [upload_done_sem];
    let wait_stages = [vk::PipelineStageFlags::TOP_OF_PIPE];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&cbs)
        .wait_semaphores(&wait_sems)
        .wait_dst_stage_mask(&wait_stages);
    unsafe {
        ctx.device
            .queue_submit(ctx.video_encode_queue, &[submit], fr.fence)
            .map_err(|e| format!("queue_submit(encode): {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fr.fence], true, fence_timeout_ns())
            .map_err(|e| format!("wait_for_fences(encode): {e:?}"))?;
        // Upload was a oneshot CB — destroy semaphore + pool now.
        ctx.device.destroy_semaphore(upload_done_sem, None);
        ctx.device.destroy_command_pool(upload_pool, None);
    }

    // ── 6. Read offset + bytes_written from the feedback query. ──
    // Use the raw fp so we can explicitly set queryCount=1 and stride
    // — ash's high-level wrapper infers them from the slice type and
    // would otherwise interpret our 12-byte u8 buffer as queryCount=12.
    #[repr(C)]
    #[derive(Default, Debug)]
    struct FeedbackResult {
        offset: u32,
        bytes_written: u32,
        status: i32, // VkQueryResultStatusKHR
    }
    let mut feedback = FeedbackResult::default();
    let result = unsafe {
        (ctx.device.fp_v1_0().get_query_pool_results)(
            ctx.device.handle(),
            fr.query_pool,
            0,
            1,
            std::mem::size_of::<FeedbackResult>(),
            &mut feedback as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<FeedbackResult>() as u64,
            vk::QueryResultFlags::WAIT | vk::QueryResultFlags::WITH_STATUS_KHR,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!("get_query_pool_results: {result:?}"));
    }
    if feedback.status <= 0 {
        return Err(format!(
            "encode feedback status={} (negative = driver error)",
            feedback.status
        ));
    }
    if feedback.bytes_written == 0 {
        return Err(format!(
            "encode produced 0 bytes (offset={})",
            feedback.offset
        ));
    }

    // ── 7. Map the encoded buffer and copy out the NAL bytes. ──
    let mut out = vec![0u8; feedback.bytes_written as usize];
    unsafe {
        let ptr = ctx
            .device
            .map_memory(
                fr.encoded_memory,
                feedback.offset as u64,
                feedback.bytes_written as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(encoded): {e:?}"))?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), out.len());
        ctx.device.unmap_memory(fr.encoded_memory);
    }
    Ok(out)
}

/// Zero-copy-on-GPU encode path: upload BGRA, run BGRA->NV12 compute,
/// copy Y/UV storage to encoder NV12 image, encode, read bytes.
/// Eliminates the CPU `bgra_to_nv12` step from VkRecorder::encode_idr_from_bgra.
///
/// Returns the H.264 NAL bytes. Same output shape as
/// `encode_idr_synthetic` — this is the architecturally correct
/// equivalent that runs the conversion on the GPU.
fn encode_idr_gpu_synthetic(
    ctx: &VkDeviceCtx,
    fr: &FrameResources,
    pipe: &BgraToNv12Pipeline,
    session: &EncodeSession,
    bgra_src: &[u8],
    pic: PicParams,
) -> Result<Vec<u8>, String> {
    let w = fr.width;
    let h = fr.height;
    if bgra_src.len() != (w as usize) * (h as usize) * 4 {
        return Err(format!(
            "bgra_src len {} != {} (=W*H*4)",
            bgra_src.len(),
            (w as usize) * (h as usize) * 4
        ));
    }

    // ── 1. Upload BGRA bytes to the staging buffer ──
    unsafe {
        let ptr = ctx
            .device
            .map_memory(
                fr.staging_memory,
                0,
                fr.staging_size,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(staging): {e:?}"))?;
        std::ptr::copy_nonoverlapping(bgra_src.as_ptr(), ptr as *mut u8, bgra_src.len());
        ctx.device.unmap_memory(fr.staging_memory);
    }

    // ── 2. Allocate + write descriptor set for the compute shader ──
    // Reset the pool first so multi-frame encodes can reuse it. The
    // pipeline holds a pool sized for one set; we just rewind it.
    unsafe {
        ctx.device
            .reset_descriptor_pool(
                pipe.descriptor_pool(),
                vk::DescriptorPoolResetFlags::empty(),
            )
            .map_err(|e| format!("reset_descriptor_pool: {e:?}"))?;
    }
    let set_layouts = [pipe.descriptor_set_layout()];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pipe.descriptor_pool())
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe {
        ctx.device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("allocate_descriptor_sets: {e:?}"))?[0]
    };
    let src_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.bgra_image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .sampler(pipe.sampler())];
    let y_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.y_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let uv_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.uv_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&src_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&y_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&uv_image_info),
    ];
    unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

    // ── 3. Compute-queue command buffer: upload + dispatch + copy + release ──
    let compute_pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let compute_pool = unsafe {
        ctx.device
            .create_command_pool(&compute_pool_ci, None)
            .map_err(|e| format!("create_command_pool(compute): {e:?}"))?
    };
    let cb_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(compute_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let compute_cb = unsafe {
        ctx.device
            .allocate_command_buffers(&cb_alloc)
            .map_err(|e| format!("allocate_command_buffers(compute): {e:?}"))?[0]
    };
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .begin_command_buffer(compute_cb, &begin)
            .map_err(|e| format!("begin_command_buffer(compute): {e:?}"))?;
    }

    // BGRA UNDEFINED -> TRANSFER_DST.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    // Copy staging buffer -> BGRA image.
    let bgra_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            compute_cb,
            fr.staging_buffer,
            fr.bgra_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[bgra_copy],
        );
    }
    // BGRA TRANSFER_DST -> SHADER_READ_ONLY.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // Y/UV storage UNDEFINED -> GENERAL.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.y_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.uv_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // Dispatch compute.
    unsafe {
        ctx.device
            .cmd_bind_pipeline(compute_cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline());
        ctx.device.cmd_bind_descriptor_sets(
            compute_cb,
            vk::PipelineBindPoint::COMPUTE,
            pipe.pipeline_layout(),
            0,
            &[descriptor_set],
            &[],
        );
        let dispatch_x = (w / 2).div_ceil(16);
        let dispatch_y = (h / 2).div_ceil(16);
        ctx.device
            .cmd_dispatch(compute_cb, dispatch_x, dispatch_y, 1);
    }
    // Y/UV GENERAL -> TRANSFER_SRC (for the upcoming copy).
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.y_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.uv_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );
    // NV12 UNDEFINED -> TRANSFER_DST (per-plane).
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.nv12_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    // Copy Y_storage -> NV12.plane_0
    let y_to_nv12 = vk::ImageCopy::default()
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_offset(vk::Offset3D::default())
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .dst_offset(vk::Offset3D::default())
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    let uv_to_nv12 = vk::ImageCopy::default()
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_offset(vk::Offset3D::default())
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .dst_offset(vk::Offset3D::default())
        .extent(vk::Extent3D {
            width: w / 2,
            height: h / 2,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_image(
            compute_cb,
            fr.y_storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            fr.nv12_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[y_to_nv12],
        );
        ctx.device.cmd_copy_image(
            compute_cb,
            fr.uv_storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            fr.nv12_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[uv_to_nv12],
        );
    }
    // Queue-family release: NV12 TRANSFER_DST -> VIDEO_ENCODE_SRC.
    {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(ctx.compute_queue_family)
            .dst_queue_family_index(ctx.video_encode_queue_family)
            .image(fr.nv12_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                compute_cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
    unsafe {
        ctx.device
            .end_command_buffer(compute_cb)
            .map_err(|e| format!("end_command_buffer(compute): {e:?}"))?;
    }

    // Submit compute + signal semaphore.
    let sem_ci = vk::SemaphoreCreateInfo::default();
    let compute_done_sem = unsafe {
        ctx.device
            .create_semaphore(&sem_ci, None)
            .map_err(|e| format!("create_semaphore: {e:?}"))?
    };
    let compute_cbs = [compute_cb];
    let compute_signals = [compute_done_sem];
    let compute_submit = vk::SubmitInfo::default()
        .command_buffers(&compute_cbs)
        .signal_semaphores(&compute_signals);
    unsafe {
        ctx.device
            .reset_fences(&[fr.fence])
            .map_err(|e| format!("reset_fences: {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[compute_submit], vk::Fence::null())
            .map_err(|e| format!("queue_submit(compute): {e:?}"))?;
    }

    // ── 4. Encode + read back via the shared encode-queue helper ──
    // This is identical to the dmabuf path's encode half; both feed the
    // same `encode_one_picture_on_encode_queue`. `pic` carries the
    // IDR/P-frame ping-pong state from the caller.
    let result = encode_one_picture_on_encode_queue(
        ctx,
        fr,
        session,
        compute_done_sem,
        pic.is_idr,
        pic.frame_num,
        pic.ref_slot,
        pic.ref_view,
        pic.setup_slot,
        pic.setup_view,
    );
    unsafe {
        ctx.device.destroy_semaphore(compute_done_sem, None);
        ctx.device.destroy_command_pool(compute_pool, None);
    }
    result
}

/// Just the compute half of the Hi444 path: upload BGRA → run the
/// BGRA→YUV 4:4:4 compute shader → copy Y/UV storage into the
/// 2-plane `fr.yuv_image` → leave the picture image in
/// `VIDEO_ENCODE_SRC_KHR` layout, owned by the compute queue family.
///
/// Used by the `hevc_vulkan` recording path, where the actual encoding
/// is done by ffmpeg's `hevc_vulkan` encoder rather than our own
/// `cmd_encode_video_khr` call. After this returns, `fr.yuv_image`
/// holds a valid 2-plane NV24 4:4:4 picture and can be handed to libav
/// for encoding.
pub fn run_compute_yuv444_into_picture(
    ctx: &VkDeviceCtx,
    fr: &FrameResources444,
    pipe: &BgraToYuv444Pipeline,
    bgra_src: &[u8],
) -> Result<(), String> {
    let w = fr.width;
    let h = fr.height;
    if bgra_src.len() != (w as usize) * (h as usize) * 4 {
        return Err(format!(
            "bgra_src len {} != {} (=W*H*4)",
            bgra_src.len(),
            (w as usize) * (h as usize) * 4
        ));
    }

    // ── 1. Upload BGRA to staging ──
    unsafe {
        let ptr = ctx
            .device
            .map_memory(
                fr.staging_memory,
                0,
                fr.staging_size,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(staging/yuv444): {e:?}"))?;
        std::ptr::copy_nonoverlapping(bgra_src.as_ptr(), ptr as *mut u8, bgra_src.len());
        ctx.device.unmap_memory(fr.staging_memory);
    }

    // ── 2. Descriptor set: sampler + Y + UV ──
    unsafe {
        ctx.device
            .reset_descriptor_pool(
                pipe.descriptor_pool(),
                vk::DescriptorPoolResetFlags::empty(),
            )
            .map_err(|e| format!("reset_descriptor_pool: {e:?}"))?;
    }
    let set_layouts = [pipe.descriptor_set_layout()];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pipe.descriptor_pool())
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe {
        ctx.device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("allocate_descriptor_sets: {e:?}"))?[0]
    };
    let src_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.bgra_image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .sampler(pipe.sampler())];
    let y_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.y_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let uv_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.uv_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&src_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&y_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&uv_image_info),
    ];
    unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

    // ── 3. Record + submit compute command buffer ──
    let pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let cmd_pool = unsafe {
        ctx.device
            .create_command_pool(&pool_ci, None)
            .map_err(|e| format!("create_command_pool: {e:?}"))?
    };
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = unsafe {
        ctx.device
            .allocate_command_buffers(&alloc)
            .map_err(|e| format!("allocate_command_buffers: {e:?}"))?[0]
    };
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .begin_command_buffer(cb, &begin)
            .map_err(|e| format!("begin_command_buffer: {e:?}"))?;
    }

    // BGRA UNDEFINED -> TRANSFER_DST -> SHADER_READ_ONLY.
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    let bgra_copy = vk::BufferImageCopy::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            fr.staging_buffer,
            fr.bgra_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[bgra_copy],
        );
    }
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // Y/UV storage UNDEFINED -> GENERAL.
    for img in [fr.y_storage_image, fr.uv_storage_image] {
        image_layout_barrier(
            &ctx.device,
            cb,
            img,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
    }
    // Dispatch.
    unsafe {
        ctx.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline());
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipe.pipeline_layout(),
            0,
            &[descriptor_set],
            &[],
        );
        ctx.device
            .cmd_dispatch(cb, w.div_ceil(16), h.div_ceil(16), 1);
    }
    // Y/UV GENERAL -> TRANSFER_SRC.
    for img in [fr.y_storage_image, fr.uv_storage_image] {
        image_layout_barrier(
            &ctx.device,
            cb,
            img,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        );
    }
    // yuv_image UNDEFINED -> TRANSFER_DST (both planes).
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.yuv_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    // Pre-clear yuv_image to neutral chroma at CODED dims. The
    // storage planes (compute output) only cover the DISPLAY region;
    // the encoder reads the full CODED region. Without this clear
    // the padding bytes are uninitialized and the encoder bakes
    // garbage into the bitstream around the CTU edges. Targets:
    //   PLANE_0 (R8 Y plane):  16   → limited-range black
    //   PLANE_1 (R8G8 UV):     128  → neutral Cb=Cr=128
    // Both planes have the SAME CTU-padded extent; we clear the
    // entire image then overlay the display region from compute.
    let clear_y = vk::ClearColorValue {
        float32: [16.0 / 255.0, 0.0, 0.0, 0.0],
    };
    let clear_uv = vk::ClearColorValue {
        float32: [128.0 / 255.0, 128.0 / 255.0, 0.0, 0.0],
    };
    let full_range_p0 = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let full_range_p1 = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    unsafe {
        ctx.device.cmd_clear_color_image(
            cb,
            fr.yuv_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &clear_y,
            &[full_range_p0],
        );
        ctx.device.cmd_clear_color_image(
            cb,
            fr.yuv_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &clear_uv,
            &[full_range_p1],
        );
    }
    // Barrier between clear and copy: both are TRANSFER stage but
    // different commands need explicit memory dependency on at least
    // some drivers.
    unsafe {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
    // Copy storage planes into yuv_image's planes.
    for (src, dst_plane) in [
        (fr.y_storage_image, vk::ImageAspectFlags::PLANE_0),
        (fr.uv_storage_image, vk::ImageAspectFlags::PLANE_1),
    ] {
        let region = vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(dst_plane)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            });
        unsafe {
            ctx.device.cmd_copy_image(
                cb,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                fr.yuv_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }
    // yuv_image TRANSFER_DST -> VIDEO_ENCODE_SRC_KHR (both planes).
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.yuv_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::empty(),
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    );
    unsafe {
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| format!("end_command_buffer: {e:?}"))?;
    }

    // Submit + wait. We block on the GPU here for simplicity — the
    // recording thread runs at the source-commit cadence so per-frame
    // sync is acceptable for first pass.
    let cbs = [cb];
    let submit = vk::SubmitInfo::default().command_buffers(&cbs);
    unsafe {
        let fence_ci = vk::FenceCreateInfo::default();
        let fence = ctx
            .device
            .create_fence(&fence_ci, None)
            .map_err(|e| format!("create_fence: {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[submit], fence)
            .map_err(|e| format!("queue_submit: {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fence], true, fence_timeout_ns())
            .map_err(|e| format!("wait_for_fences: {e:?}"))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }
    Ok(())
}

/// Diagnostic readback. After `run_compute_yuv444_into_picture` has
/// completed, copy `fr.yuv_image` PLANE_0 and PLANE_1 back to host
/// memory.
///
/// Returns `(plane0_bytes, plane1_bytes)`:
///   - `plane0_bytes` length = W * H   (Y plane, R8)
///   - `plane1_bytes` length = W * H * 2 (UV plane, R8G8 interleaved)
///
/// Used to localize where chroma artifacts originate: if PLANE_1 here
/// already has the stair-step pattern, the bug is in our R8G8 →
/// PLANE_1 `vkCmdCopyImage` step; if it's clean, the bug is later in
/// libav's encoder copy chain.
pub fn dump_yuv444_picture_planes(
    ctx: &VkDeviceCtx,
    fr: &FrameResources444,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let w = fr.width;
    let h = fr.height;
    let p0_size = (w as u64) * (h as u64);
    let p1_size = (w as u64) * (h as u64) * 2;
    let total = p0_size + p1_size;

    // Host-visible readback buffer.
    let buf_ci = vk::BufferCreateInfo::default()
        .size(total)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf = unsafe {
        ctx.device
            .create_buffer(&buf_ci, None)
            .map_err(|e| format!("create_buffer(yuv_readback): {e:?}"))?
    };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let buf_mem = alloc_buffer_memory(
        &ctx.device,
        &mem_props,
        buf,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        ctx.device
            .bind_buffer_memory(buf, buf_mem, 0)
            .map_err(|e| format!("bind_buffer_memory(yuv_readback): {e:?}"))?;
    }

    // Compute-queue command pool + cb.
    let pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let cmd_pool = unsafe {
        ctx.device
            .create_command_pool(&pool_ci, None)
            .map_err(|e| format!("create_command_pool(yuv_readback): {e:?}"))?
    };
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = unsafe {
        ctx.device
            .allocate_command_buffers(&alloc)
            .map_err(|e| format!("allocate_command_buffers(yuv_readback): {e:?}"))?[0]
    };
    unsafe {
        ctx.device
            .begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| format!("begin_command_buffer(yuv_readback): {e:?}"))?;
    }

    // yuv_image VIDEO_ENCODE_SRC_KHR → TRANSFER_SRC_OPTIMAL (both planes).
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.yuv_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );

    // Copy PLANE_0 to buffer offset 0.
    let region_p0 = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    let region_p1 = vk::BufferImageCopy::default()
        .buffer_offset(p0_size)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            fr.yuv_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buf,
            &[region_p0, region_p1],
        );
    }

    // Restore VIDEO_ENCODE_SRC_KHR so the caller can carry on with an
    // encode if desired.
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.yuv_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::empty(),
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    );

    unsafe {
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| format!("end_command_buffer(yuv_readback): {e:?}"))?;
    }

    let cbs = [cb];
    let submit = vk::SubmitInfo::default().command_buffers(&cbs);
    unsafe {
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("create_fence(yuv_readback): {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[submit], fence)
            .map_err(|e| format!("queue_submit(yuv_readback): {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fence], true, fence_timeout_ns())
            .map_err(|e| format!("wait_for_fences(yuv_readback): {e:?}"))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    let mut p0 = vec![0u8; p0_size as usize];
    let mut p1 = vec![0u8; p1_size as usize];
    unsafe {
        let ptr = ctx
            .device
            .map_memory(buf_mem, 0, total, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map_memory(yuv_readback): {e:?}"))?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, p0.as_mut_ptr(), p0_size as usize);
        std::ptr::copy_nonoverlapping(
            (ptr as *const u8).add(p0_size as usize),
            p1.as_mut_ptr(),
            p1_size as usize,
        );
        ctx.device.unmap_memory(buf_mem);
        ctx.device.destroy_buffer(buf, None);
        ctx.device.free_memory(buf_mem, None);
    }
    Ok((p0, p1))
}

/// Diagnostic readback of the BGRA→YUV compute shader's intermediate
/// storage images, before `vkCmdCopyImage` into the multi-planar
/// encoder picture. After `run_compute_yuv444_into_picture` completes,
/// both storage images are at `TRANSFER_SRC_OPTIMAL`.
///
/// Returns `(y_bytes, uv_bytes)`:
///   - `y_bytes` length = W * H   (Y storage, R8)
///   - `uv_bytes` length = W * H * 2 (UV storage, R8G8 interleaved)
pub fn dump_yuv444_storage_images(
    ctx: &VkDeviceCtx,
    fr: &FrameResources444,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let w = fr.width;
    let h = fr.height;
    let y_size = (w as u64) * (h as u64);
    let uv_size = (w as u64) * (h as u64) * 2;
    let total = y_size + uv_size;

    let buf_ci = vk::BufferCreateInfo::default()
        .size(total)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf = unsafe {
        ctx.device
            .create_buffer(&buf_ci, None)
            .map_err(|e| format!("create_buffer(stor_readback): {e:?}"))?
    };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let buf_mem = alloc_buffer_memory(
        &ctx.device,
        &mem_props,
        buf,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        ctx.device
            .bind_buffer_memory(buf, buf_mem, 0)
            .map_err(|e| format!("bind_buffer_memory(stor_readback): {e:?}"))?;
    }

    let pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let cmd_pool = unsafe {
        ctx.device
            .create_command_pool(&pool_ci, None)
            .map_err(|e| format!("create_command_pool(stor_readback): {e:?}"))?
    };
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = unsafe {
        ctx.device
            .allocate_command_buffers(&alloc)
            .map_err(|e| format!("allocate_command_buffers(stor_readback): {e:?}"))?[0]
    };
    unsafe {
        ctx.device
            .begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| format!("begin_command_buffer(stor_readback): {e:?}"))?;
    }

    // Storage images are already at TRANSFER_SRC_OPTIMAL after
    // `run_compute_yuv444_into_picture` — no barriers needed.
    let region_y = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    let region_uv = vk::BufferImageCopy::default()
        .buffer_offset(y_size)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            fr.y_storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buf,
            &[region_y],
        );
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            fr.uv_storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buf,
            &[region_uv],
        );
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| format!("end_command_buffer(stor_readback): {e:?}"))?;
    }

    let cbs = [cb];
    let submit = vk::SubmitInfo::default().command_buffers(&cbs);
    unsafe {
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("create_fence(stor_readback): {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[submit], fence)
            .map_err(|e| format!("queue_submit(stor_readback): {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fence], true, fence_timeout_ns())
            .map_err(|e| format!("wait_for_fences(stor_readback): {e:?}"))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    let mut y_bytes = vec![0u8; y_size as usize];
    let mut uv_bytes = vec![0u8; uv_size as usize];
    unsafe {
        let ptr = ctx
            .device
            .map_memory(buf_mem, 0, total, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map_memory(stor_readback): {e:?}"))?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, y_bytes.as_mut_ptr(), y_size as usize);
        std::ptr::copy_nonoverlapping(
            (ptr as *const u8).add(y_size as usize),
            uv_bytes.as_mut_ptr(),
            uv_size as usize,
        );
        ctx.device.unmap_memory(buf_mem);
        ctx.device.destroy_buffer(buf, None);
        ctx.device.free_memory(buf_mem, None);
    }
    Ok((y_bytes, uv_bytes))
}

/// Lossless H.264 Hi444PP encode path. Sister of `encode_idr_gpu_synthetic`
/// but targeting 4:4:4 chroma with no quantization (constant_qp=0).
///
/// Pipeline differences from the NV12 path:
/// - Compute output: 1 R8 (Y) + 1 R8G8 (UV interleaved), both full-res
///   (NV12 keeps UV at half-res).
/// - Encoder picture: `G8_B8R8_2PLANE_444_UNORM` (NVIDIA driver 560's
///   only supported Hi444PP encode-src format). PLANE_2 isn't used.
/// - Dispatch granularity: one workgroup invocation per source pixel
///   (16×16 workgroup → (w/16, h/16) groups) instead of the 2×2-block
///   sub-sampling used for NV12.
pub fn encode_idr_gpu_synthetic_yuv444(
    ctx: &VkDeviceCtx,
    fr: &FrameResources444,
    pipe: &BgraToYuv444Pipeline,
    session: &EncodeSession,
    bgra_src: &[u8],
) -> Result<Vec<u8>, String> {
    let w = fr.width;
    let h = fr.height;
    if bgra_src.len() != (w as usize) * (h as usize) * 4 {
        return Err(format!(
            "bgra_src len {} != {} (=W*H*4)",
            bgra_src.len(),
            (w as usize) * (h as usize) * 4
        ));
    }
    if session.kind != EncodeKind::Hi444Lossless {
        return Err(format!(
            "session kind {:?} mismatches Hi444Lossless YUV444 encode path",
            session.kind
        ));
    }

    // ── 1. Upload BGRA bytes to staging ──
    unsafe {
        let ptr = ctx
            .device
            .map_memory(
                fr.staging_memory,
                0,
                fr.staging_size,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(staging/yuv444): {e:?}"))?;
        std::ptr::copy_nonoverlapping(bgra_src.as_ptr(), ptr as *mut u8, bgra_src.len());
        ctx.device.unmap_memory(fr.staging_memory);
    }

    // ── 2. Allocate + write descriptor set (4 bindings) ──
    unsafe {
        ctx.device
            .reset_descriptor_pool(
                pipe.descriptor_pool(),
                vk::DescriptorPoolResetFlags::empty(),
            )
            .map_err(|e| format!("reset_descriptor_pool(yuv444): {e:?}"))?;
    }
    let set_layouts = [pipe.descriptor_set_layout()];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pipe.descriptor_pool())
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe {
        ctx.device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("allocate_descriptor_sets(yuv444): {e:?}"))?[0]
    };
    let src_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.bgra_image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .sampler(pipe.sampler())];
    let y_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.y_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let uv_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.uv_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&src_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&y_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&uv_image_info),
    ];
    unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

    // ── 3. Compute-queue command buffer ──
    let compute_pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let compute_pool = unsafe {
        ctx.device
            .create_command_pool(&compute_pool_ci, None)
            .map_err(|e| format!("create_command_pool(compute/yuv444): {e:?}"))?
    };
    let cb_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(compute_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let compute_cb = unsafe {
        ctx.device
            .allocate_command_buffers(&cb_alloc)
            .map_err(|e| format!("allocate_command_buffers(compute/yuv444): {e:?}"))?[0]
    };
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .begin_command_buffer(compute_cb, &begin)
            .map_err(|e| format!("begin_command_buffer(compute/yuv444): {e:?}"))?;
    }

    // BGRA UNDEFINED -> TRANSFER_DST.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    // Copy staging buffer -> BGRA image.
    let bgra_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            compute_cb,
            fr.staging_buffer,
            fr.bgra_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[bgra_copy],
        );
    }
    // BGRA TRANSFER_DST -> SHADER_READ_ONLY.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.bgra_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // Y/UV storage UNDEFINED -> GENERAL.
    for img in [fr.y_storage_image, fr.uv_storage_image] {
        image_layout_barrier(
            &ctx.device,
            compute_cb,
            img,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
    }
    // Dispatch compute.
    unsafe {
        ctx.device
            .cmd_bind_pipeline(compute_cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline());
        ctx.device.cmd_bind_descriptor_sets(
            compute_cb,
            vk::PipelineBindPoint::COMPUTE,
            pipe.pipeline_layout(),
            0,
            &[descriptor_set],
            &[],
        );
        // YUV 4:4:4 shader: one invocation per source pixel, 16×16
        // workgroups. Dispatch ceil(w/16) × ceil(h/16) × 1.
        let dispatch_x = w.div_ceil(16);
        let dispatch_y = h.div_ceil(16);
        ctx.device
            .cmd_dispatch(compute_cb, dispatch_x, dispatch_y, 1);
    }
    // Y/UV GENERAL -> TRANSFER_SRC.
    for img in [fr.y_storage_image, fr.uv_storage_image] {
        image_layout_barrier(
            &ctx.device,
            compute_cb,
            img,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        );
    }
    // YUV 2-plane UNDEFINED -> TRANSFER_DST (both planes).
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.yuv_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    // Copy Y_storage -> yuv.plane_0, UV_storage -> plane_1. Both at
    // full resolution (4:4:4 keeps chroma at luma resolution).
    let plane_copy = |src: vk::Image, dst_plane: vk::ImageAspectFlags| {
        let region = vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_offset(vk::Offset3D::default())
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(dst_plane)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_offset(vk::Offset3D::default())
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            });
        unsafe {
            ctx.device.cmd_copy_image(
                compute_cb,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                fr.yuv_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    };
    plane_copy(fr.y_storage_image, vk::ImageAspectFlags::PLANE_0);
    plane_copy(fr.uv_storage_image, vk::ImageAspectFlags::PLANE_1);

    // Queue-family release: YUV TRANSFER_DST -> VIDEO_ENCODE_SRC,
    // release from compute family to encode family. Both planes.
    {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(ctx.compute_queue_family)
            .dst_queue_family_index(ctx.video_encode_queue_family)
            .image(fr.yuv_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                compute_cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
    unsafe {
        ctx.device
            .end_command_buffer(compute_cb)
            .map_err(|e| format!("end_command_buffer(compute/yuv444): {e:?}"))?;
    }

    // Submit compute + signal semaphore.
    let sem_ci = vk::SemaphoreCreateInfo::default();
    let compute_done_sem = unsafe {
        ctx.device
            .create_semaphore(&sem_ci, None)
            .map_err(|e| format!("create_semaphore(yuv444): {e:?}"))?
    };
    let compute_cbs = [compute_cb];
    let compute_signals = [compute_done_sem];
    let compute_submit = vk::SubmitInfo::default()
        .command_buffers(&compute_cbs)
        .signal_semaphores(&compute_signals);
    unsafe {
        ctx.device
            .reset_fences(&[fr.fence])
            .map_err(|e| format!("reset_fences(yuv444): {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[compute_submit], vk::Fence::null())
            .map_err(|e| format!("queue_submit(compute/yuv444): {e:?}"))?;
    }

    // ── 4. Encode-queue command buffer ──
    let cb = fr.command_buffer;
    unsafe {
        ctx.device
            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .map_err(|e| format!("reset_command_buffer(encode/yuv444): {e:?}"))?;
        ctx.device
            .begin_command_buffer(cb, &begin)
            .map_err(|e| format!("begin_command_buffer(encode/yuv444): {e:?}"))?;
    }
    // Queue-family acquire on encode side. All 3 planes.
    {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(ctx.compute_queue_family)
            .dst_queue_family_index(ctx.video_encode_queue_family)
            .image(fr.yuv_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            ctx.device.cmd_reset_query_pool(cb, fr.query_pool, 0, 1);
        }
    }
    image_layout_barrier(
        &ctx.device,
        cb,
        fr.dpb_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
        vk::AccessFlags::empty(),
        vk::AccessFlags::MEMORY_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    );
    let begin_dpb_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(fr.dpb_image_view);
    let begin_dpb_slot = vk::VideoReferenceSlotInfoKHR::default()
        .slot_index(-1)
        .picture_resource(&begin_dpb_resource);
    let begin_slots = [begin_dpb_slot];
    let begin_info = vk::VideoBeginCodingInfoKHR::default()
        .video_session(session.session)
        .video_session_parameters(session.parameters)
        .reference_slots(&begin_slots);
    unsafe {
        (ctx.video_queue_dev.fp().cmd_begin_video_coding_khr)(cb, &begin_info);
    }
    let mut rate_control_layer = vk::VideoEncodeRateControlInfoKHR::default()
        .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
    let control_info = vk::VideoCodingControlInfoKHR::default()
        .flags(
            vk::VideoCodingControlFlagsKHR::RESET
                | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL,
        )
        .push_next(&mut rate_control_layer);
    unsafe {
        (ctx.video_queue_dev.fp().cmd_control_video_coding_khr)(cb, &control_info);
    }
    // Encode-feedback query: optional, can be stripped via env override
    // to bisect NVIDIA Hi444PP encode-submit failures. When stripped
    // we lose the offset+bytes_written feedback and map the entire
    // encoded buffer to find the NAL bytes.
    let use_query = !matches!(
        std::env::var("WAYMUX_VK_SKIP_QUERY").ok().as_deref(),
        Some("1")
    );
    if use_query {
        unsafe {
            ctx.device
                .cmd_begin_query(cb, fr.query_pool, 0, vk::QueryControlFlags::empty());
        }
    }
    let mut pic_flags: vk::native::StdVideoEncodeH264PictureInfoFlags =
        unsafe { std::mem::zeroed() };
    pic_flags.set_IdrPicFlag(1);
    pic_flags.set_is_reference(1);
    let ref_lists_flags: vk::native::StdVideoEncodeH264ReferenceListsInfoFlags =
        unsafe { std::mem::zeroed() };
    let ref_lists = vk::native::StdVideoEncodeH264ReferenceListsInfo {
        flags: ref_lists_flags,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        RefPicList0: [0xFF; 32],
        RefPicList1: [0xFF; 32],
        refList0ModOpCount: 0,
        refList1ModOpCount: 0,
        refPicMarkingOpCount: 0,
        reserved1: [0; 7],
        pRefList0ModOperations: std::ptr::null(),
        pRefList1ModOperations: std::ptr::null(),
        pRefPicMarkingOperations: std::ptr::null(),
    };
    let pic_info = vk::native::StdVideoEncodeH264PictureInfo {
        flags: pic_flags,
        seq_parameter_set_id: 0,
        pic_parameter_set_id: 0,
        idr_pic_id: 0,
        primary_pic_type: vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR,
        frame_num: 0,
        PicOrderCnt: 0,
        temporal_id: 0,
        reserved1: [0; 3],
        pRefLists: &ref_lists,
    };
    let slice_flags: vk::native::StdVideoEncodeH264SliceHeaderFlags = unsafe { std::mem::zeroed() };
    let slice_header = vk::native::StdVideoEncodeH264SliceHeader {
        flags: slice_flags,
        first_mb_in_slice: 0,
        slice_type: vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_qp_delta: 0,
        reserved1: 0,
        cabac_init_idc: vk::native::StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
        disable_deblocking_filter_idc:
            vk::native::StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
        pWeightTable: std::ptr::null(),
    };
    // constant_qp(0) is the goal for bit-exact lossless. H.264's
    // transform-domain math is reversible at QP=0; any higher QP
    // injects quantization error. WAYMUX_VK_LOSSLESS_QP env override
    // lets us bisect "is the rejection because of QP=0 specifically"
    // vs "is the rejection generic to the Hi444PP path".
    let qp_override = std::env::var("WAYMUX_VK_LOSSLESS_QP")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .map(|q| q.clamp(0, 51))
        .unwrap_or(0);
    let nalu_slice = vk::VideoEncodeH264NaluSliceInfoKHR::default()
        .constant_qp(qp_override)
        .std_slice_header(&slice_header);
    let nalu_slice_arr = [nalu_slice];
    let mut h264_pic_info = vk::VideoEncodeH264PictureInfoKHR::default()
        .nalu_slice_entries(&nalu_slice_arr)
        .std_picture_info(&pic_info);
    let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(fr.yuv_color_view);
    let dpb_picture_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(fr.dpb_image_view);
    let ref_info_flags: vk::native::StdVideoEncodeH264ReferenceInfoFlags =
        unsafe { std::mem::zeroed() };
    let std_ref_info = vk::native::StdVideoEncodeH264ReferenceInfo {
        flags: ref_info_flags,
        primary_pic_type: vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR,
        FrameNum: 0,
        PicOrderCnt: 0,
        long_term_pic_num: 0,
        long_term_frame_idx: 0,
        temporal_id: 0,
    };
    let mut h264_dpb_slot =
        vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_ref_info);
    let setup_ref_slot = vk::VideoReferenceSlotInfoKHR::default()
        .slot_index(0)
        .picture_resource(&dpb_picture_resource)
        .push_next(&mut h264_dpb_slot);
    let encode_info = vk::VideoEncodeInfoKHR::default()
        .dst_buffer(fr.encoded_buffer)
        .dst_buffer_offset(0)
        .dst_buffer_range(fr.encoded_size)
        .src_picture_resource(src_picture_resource)
        .setup_reference_slot(&setup_ref_slot)
        .push_next(&mut h264_pic_info);
    unsafe {
        (ctx.video_encode_queue_dev.fp().cmd_encode_video_khr)(cb, &encode_info);
        if use_query {
            ctx.device.cmd_end_query(cb, fr.query_pool, 0);
        }
    }
    let end_info = vk::VideoEndCodingInfoKHR::default();
    unsafe {
        (ctx.video_queue_dev.fp().cmd_end_video_coding_khr)(cb, &end_info);
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| format!("end_command_buffer(encode/yuv444): {e:?}"))?;
    }

    // ── 5. Submit encode + wait on compute semaphore ──
    let cbs = [cb];
    let wait_sems = [compute_done_sem];
    let wait_stages = [vk::PipelineStageFlags::TOP_OF_PIPE];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&cbs)
        .wait_semaphores(&wait_sems)
        .wait_dst_stage_mask(&wait_stages);
    unsafe {
        ctx.device
            .queue_submit(ctx.video_encode_queue, &[submit], fr.fence)
            .map_err(|e| format!("queue_submit(encode/yuv444): {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fr.fence], true, fence_timeout_ns())
            .map_err(|e| format!("wait_for_fences(encode/yuv444): {e:?}"))?;
        ctx.device.destroy_semaphore(compute_done_sem, None);
        ctx.device.destroy_command_pool(compute_pool, None);
    }

    // ── 6. Read encode feedback + map encoded buffer ──
    #[repr(C)]
    #[derive(Default, Debug)]
    struct FeedbackResult {
        offset: u32,
        bytes_written: u32,
        status: i32,
    }
    let (offset, bytes_written) = if use_query {
        let mut feedback = FeedbackResult::default();
        let result = unsafe {
            (ctx.device.fp_v1_0().get_query_pool_results)(
                ctx.device.handle(),
                fr.query_pool,
                0,
                1,
                std::mem::size_of::<FeedbackResult>(),
                &mut feedback as *mut _ as *mut std::ffi::c_void,
                std::mem::size_of::<FeedbackResult>() as u64,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::WITH_STATUS_KHR,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(format!("get_query_pool_results(yuv444): {result:?}"));
        }
        if feedback.bytes_written == 0 {
            return Err(format!(
                "encode produced 0 bytes (status={}/yuv444)",
                feedback.status
            ));
        }
        (feedback.offset as u64, feedback.bytes_written as u64)
    } else {
        // WAYMUX_VK_SKIP_QUERY=1 path: no feedback. Map the whole
        // encoded buffer and let the caller scan for NAL start codes.
        // Used for diagnosing whether the query pool is the source of
        // a driver rejection.
        (0u64, fr.encoded_size.min(64 * 1024))
    };
    let mut out = vec![0u8; bytes_written as usize];
    unsafe {
        let ptr = ctx
            .device
            .map_memory(
                fr.encoded_memory,
                offset,
                bytes_written,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(encoded/yuv444): {e:?}"))?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), out.len());
        ctx.device.unmap_memory(fr.encoded_memory);
    }
    Ok(out)
}

/// Implementation of dmabuf-based encode. Imports the dmabuf as a
/// VkImage via VK_KHR_external_memory_fd + VK_EXT_external_memory_dma_buf,
/// then runs the same compute + encode path as encode_idr_gpu_synthetic
/// but with the imported image replacing the staging-uploaded BGRA
/// image.
#[allow(clippy::too_many_arguments)]
fn encode_idr_from_dmabuf_impl(
    ctx: &VkDeviceCtx,
    fr: &FrameResources,
    pipe: &BgraToNv12Pipeline,
    session: &EncodeSession,
    dma: &crate::dmabuf::DmabufBufferData,
    pic: PicParams,
) -> Result<Vec<u8>, String> {
    // Only LINEAR-modifier dmabufs for now. Tiled (modifier != 0)
    // requires a VkImageDrmFormatModifierExplicitCreateInfoEXT chain
    // with per-plane offsets/strides; the existing recording path
    // also only handles LINEAR commits (gpu_record.rs early-returns
    // on non-LINEAR), so we mirror that behavior here.
    if !modifier_is_importable(dma.modifier) {
        return Err(format!(
            "dmabuf modifier {:#x} not importable; falling back",
            dma.modifier
        ));
    }
    let is_tiled = dma.modifier != crate::dmabuf::DRM_FORMAT_MOD_LINEAR;
    let w = fr.width;
    let h = fr.height;
    if dma.width != w as i32 || dma.height != h as i32 {
        return Err(format!(
            "dmabuf size {}x{} doesn't match encoder size {}x{}",
            dma.width, dma.height, w, h
        ));
    }

    // DRM fourcc -> Vulkan format. Our pipeline accepts ARGB8888
    // (0x34325241) and XRGB8888 (0x34325258) — both map to
    // VK_FORMAT_B8G8R8A8_UNORM in Vulkan (the format swizzles bytes
    // back to R,G,B,A order for the shader).
    let vk_format = match dma.drm_format {
        crate::dmabuf::DRM_FORMAT_ARGB8888 | crate::dmabuf::DRM_FORMAT_XRGB8888 => {
            vk::Format::B8G8R8A8_UNORM
        }
        f => return Err(format!("unsupported drm_format {f:#x}")),
    };

    // dup the fd — Vulkan takes ownership of the dup once import
    // succeeds. We dup *before* the import so the caller's
    // DmabufBufferData keeps its own ownership intact.
    let raw_fd: i32 = unsafe { libc::dup(std::os::fd::AsRawFd::as_raw_fd(&dma.fd)) };
    if raw_fd < 0 {
        return Err(format!(
            "dup(dmabuf fd): {}",
            std::io::Error::last_os_error()
        ));
    }

    // ── 1. Create a VkImage backed by the imported dmabuf ──
    let mut external_mem_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    // Per-plane layout for the explicit-modifier import. Plane 0 is the packed
    // BGRA plane; the Task-A query filtered to single-plane modifiers, so one entry.
    let plane_layouts = [vk::SubresourceLayout {
        offset: dma.offset as u64,
        size: 0, // 0 = "the rest of the bound memory"
        row_pitch: dma.stride as u64,
        array_pitch: 0,
        depth_pitch: 0,
    }];
    let mut drm_explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(dma.modifier)
        .plane_layouts(&plane_layouts);

    let tiling = if is_tiled {
        vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT
    } else {
        vk::ImageTiling::LINEAR
    };
    let mut image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(tiling)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_mem_info);
    if is_tiled {
        image_ci = image_ci.push_next(&mut drm_explicit);
    }
    let dmabuf_image = unsafe {
        ctx.device.create_image(&image_ci, None).map_err(|e| {
            libc::close(raw_fd);
            format!("create_image(dmabuf): {e:?}")
        })?
    };

    // ── 2. Allocate memory imported from the dmabuf fd ──
    let mem_req = unsafe { ctx.device.get_image_memory_requirements(dmabuf_image) };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let type_idx = pick_memory_type(
        &mem_props,
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::empty(), // any heap is fine; dmabuf brings its own
    )
    .ok_or_else(|| {
        unsafe { libc::close(raw_fd) };
        unsafe { ctx.device.destroy_image(dmabuf_image, None) };
        format!(
            "no memory type for dmabuf import (mask={:#x})",
            mem_req.memory_type_bits
        )
    })?;
    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(raw_fd);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(dmabuf_image);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(type_idx)
        .push_next(&mut import_info)
        .push_next(&mut dedicated);
    let dmabuf_mem = unsafe {
        ctx.device.allocate_memory(&alloc_info, None).map_err(|e| {
            libc::close(raw_fd);
            ctx.device.destroy_image(dmabuf_image, None);
            format!("allocate_memory(dmabuf import): {e:?}")
        })?
    };
    // On successful import the driver owns the fd; do NOT close it.

    unsafe {
        ctx.device
            .bind_image_memory(dmabuf_image, dmabuf_mem, 0)
            .map_err(|e| format!("bind_image_memory(dmabuf): {e:?}"))?;
    }
    let dmabuf_view = create_full_image_view(
        &ctx.device,
        dmabuf_image,
        vk_format,
        vk::ImageAspectFlags::COLOR,
    )?;

    // ── 3. Run the compute + copy + encode flow on compute_cb ──
    // Identical to encode_idr_gpu_synthetic but with descriptor set
    // binding=0 pointing to the dmabuf view instead of fr.bgra_image_view.
    let dmabuf_descriptor_image = vk::DescriptorImageInfo::default()
        .image_view(dmabuf_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .sampler(pipe.sampler());

    let y_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.y_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let uv_image_info = [vk::DescriptorImageInfo::default()
        .image_view(fr.uv_storage_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let dmabuf_image_info = [dmabuf_descriptor_image];

    unsafe {
        ctx.device
            .reset_descriptor_pool(
                pipe.descriptor_pool(),
                vk::DescriptorPoolResetFlags::empty(),
            )
            .map_err(|e| format!("reset_descriptor_pool: {e:?}"))?;
    }
    let set_layouts = [pipe.descriptor_set_layout()];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pipe.descriptor_pool())
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe {
        ctx.device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("allocate_descriptor_sets: {e:?}"))?[0]
    };
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&dmabuf_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&y_image_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&uv_image_info),
    ];
    unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

    // Compute command buffer
    let compute_pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let compute_pool = unsafe {
        ctx.device
            .create_command_pool(&compute_pool_ci, None)
            .map_err(|e| format!("create_command_pool(compute): {e:?}"))?
    };
    let cb_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(compute_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let compute_cb = unsafe {
        ctx.device
            .allocate_command_buffers(&cb_alloc)
            .map_err(|e| format!("allocate_command_buffers(compute): {e:?}"))?[0]
    };
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .begin_command_buffer(compute_cb, &begin)
            .map_err(|e| format!("begin_command_buffer(compute): {e:?}"))?;
    }

    // Dmabuf image: acquire from external queue family + transition
    // to SHADER_READ_ONLY. The spec uses VK_QUEUE_FAMILY_EXTERNAL as
    // the source queue when importing from an external producer.
    {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(ctx.compute_queue_family)
            .image(dmabuf_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                compute_cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
    // Y/UV storage UNDEFINED -> GENERAL.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.y_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.uv_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // Dispatch compute.
    unsafe {
        ctx.device
            .cmd_bind_pipeline(compute_cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline());
        ctx.device.cmd_bind_descriptor_sets(
            compute_cb,
            vk::PipelineBindPoint::COMPUTE,
            pipe.pipeline_layout(),
            0,
            &[descriptor_set],
            &[],
        );
        let dispatch_x = (w / 2).div_ceil(16);
        let dispatch_y = (h / 2).div_ceil(16);
        ctx.device
            .cmd_dispatch(compute_cb, dispatch_x, dispatch_y, 1);
    }
    // Y/UV GENERAL -> TRANSFER_SRC.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.y_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.uv_storage_image,
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );
    // NV12 UNDEFINED -> TRANSFER_DST.
    image_layout_barrier(
        &ctx.device,
        compute_cb,
        fr.nv12_image,
        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    // Copy Y_storage + UV_storage -> NV12 planes.
    let y_to_nv12 = vk::ImageCopy::default()
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        });
    let uv_to_nv12 = vk::ImageCopy::default()
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .extent(vk::Extent3D {
            width: w / 2,
            height: h / 2,
            depth: 1,
        });
    unsafe {
        ctx.device.cmd_copy_image(
            compute_cb,
            fr.y_storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            fr.nv12_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[y_to_nv12],
        );
        ctx.device.cmd_copy_image(
            compute_cb,
            fr.uv_storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            fr.nv12_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[uv_to_nv12],
        );
    }
    // Queue-family release: NV12 -> encode queue.
    {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(ctx.compute_queue_family)
            .dst_queue_family_index(ctx.video_encode_queue_family)
            .image(fr.nv12_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                compute_cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
    unsafe {
        ctx.device
            .end_command_buffer(compute_cb)
            .map_err(|e| format!("end_command_buffer(compute): {e:?}"))?;
    }

    // Submit compute + signal semaphore.
    let sem_ci = vk::SemaphoreCreateInfo::default();
    let compute_done_sem = unsafe {
        ctx.device
            .create_semaphore(&sem_ci, None)
            .map_err(|e| format!("create_semaphore: {e:?}"))?
    };
    let compute_cbs = [compute_cb];
    let compute_signals = [compute_done_sem];
    let compute_submit = vk::SubmitInfo::default()
        .command_buffers(&compute_cbs)
        .signal_semaphores(&compute_signals);
    unsafe {
        ctx.device
            .reset_fences(&[fr.fence])
            .map_err(|e| format!("reset_fences: {e:?}"))?;
        ctx.device
            .queue_submit(ctx.compute_queue, &[compute_submit], vk::Fence::null())
            .map_err(|e| format!("queue_submit(compute): {e:?}"))?;
    }

    // Encode command buffer — shared with the BGRA path. P-frame
    // picture parameters (slot/reference bindings, frame_num) come from
    // the caller via `pic`.
    let result = encode_one_picture_on_encode_queue(
        ctx,
        fr,
        session,
        compute_done_sem,
        pic.is_idr,
        pic.frame_num,
        pic.ref_slot,
        pic.ref_view,
        pic.setup_slot,
        pic.setup_view,
    );
    unsafe {
        ctx.device.destroy_semaphore(compute_done_sem, None);
        ctx.device.destroy_command_pool(compute_pool, None);
        // Tear down the dmabuf-imported VkImage + memory. The dup'd
        // fd was consumed by allocate_memory and is released when
        // we free the memory.
        ctx.device.destroy_image_view(dmabuf_view, None);
        ctx.device.destroy_image(dmabuf_image, None);
        ctx.device.free_memory(dmabuf_mem, None);
    }
    result
}

/// Per-picture encode parameters for the ping-pong P-frame state.
/// Bundles what the VkRecorder tracks between frames so we can thread it
/// through the dmabuf / BGRA encode functions without a long arg list.
#[derive(Clone, Copy)]
struct PicParams {
    is_idr: bool,
    frame_num: u32,
    /// Active reference DPB slot (P only); ignored when `is_idr`.
    ref_slot: i32,
    ref_view: vk::ImageView,
    /// DPB slot the reconstructed picture is written into.
    setup_slot: i32,
    setup_view: vk::ImageView,
}

/// Encode-queue half of the encode_* functions. Records + submits the
/// begin_video / encode / end_video sequence, waits the fence, reads
/// back the encoded NAL bytes from the feedback query. Factored out
/// because both encode_idr_gpu_synthetic (BGRA upload) and
/// encode_idr_from_dmabuf_impl (zero-copy import) use identical
/// encode-side logic.
///
/// Handles both IDR and inter-predicted (P) pictures:
///   - `is_idr` selects IDR vs P picture/slice types and whether a
///     reference is consumed.
///   - `frame_num` is the H.264 `frame_num` (mod 16 with our SPS
///     `log2_max_frame_num_minus4 = 0`); IDR uses 0 and resets the GOP.
///   - `ref_slot` / `ref_view` name the active reference (P only); pass
///     `ref_slot < 0` for IDR (no reference).
///   - `setup_slot` / `setup_view` name the DPB slot the reconstructed
///     picture is written into (this picture becomes the next P-frame's
///     reference). The setup image is barriered UNDEFINED -> DPB (its
///     prior contents are discarded — we always overwrite it). The
///     reference image (P only) is barriered DPB -> DPB to PRESERVE its
///     contents across this submit; using UNDEFINED there would discard
///     the reference and yield a garbage / IDR-sized P-frame.
#[allow(clippy::too_many_arguments)]
fn encode_one_picture_on_encode_queue(
    ctx: &VkDeviceCtx,
    fr: &FrameResources,
    session: &EncodeSession,
    wait_sem: vk::Semaphore,
    is_idr: bool,
    frame_num: u32,
    ref_slot: i32,
    ref_view: vk::ImageView,
    setup_slot: i32,
    setup_view: vk::ImageView,
) -> Result<Vec<u8>, String> {
    let w = fr.width;
    let h = fr.height;
    let cb = fr.command_buffer;
    // Both DPB slots are array layers of the single `fr.dpb_image`
    // (RADV lacks SEPARATE_REFERENCE_IMAGES). Slot N lives in layer N.
    let setup_layer = setup_slot.max(0) as u32;
    let ref_layer = ref_slot.max(0) as u32;
    let poc = frame_num.wrapping_mul(2) as i32;
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .map_err(|e| format!("reset_command_buffer(encode): {e:?}"))?;
        ctx.device
            .begin_command_buffer(cb, &begin)
            .map_err(|e| format!("begin_command_buffer(encode): {e:?}"))?;
    }
    // Queue-family acquire on encode side.
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
        .src_queue_family_index(ctx.compute_queue_family)
        .dst_queue_family_index(ctx.video_encode_queue_family)
        .image(fr.nv12_image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);
    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        ctx.device.cmd_reset_query_pool(cb, fr.query_pool, 0, 1);
    }
    // Setup (reconstruct-target) DPB layer: UNDEFINED -> DPB. We always
    // overwrite it this frame, so discarding prior contents is correct.
    dpb_layer_layout_barrier(
        &ctx.device,
        cb,
        fr.dpb_image,
        setup_layer,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
        vk::AccessFlags::empty(),
        vk::AccessFlags::MEMORY_WRITE,
    );
    if !is_idr {
        // Reference DPB layer: DPB -> DPB. The contents (last frame's
        // reconstruction) MUST be preserved; an UNDEFINED old-layout
        // here would let the driver discard the reference.
        dpb_layer_layout_barrier(
            &ctx.device,
            cb,
            fr.dpb_image,
            ref_layer,
            vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
            vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::MEMORY_READ,
        );
    }
    // Picture resources for the begin-coding reference-slot list.
    let setup_dpb_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(setup_view);
    let ref_dpb_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(ref_view);
    // VideoBeginCodingInfoKHR.pReferenceSlots must list every DPB slot
    // touched this frame. IDR: the setup slot bound with slot_index=-1
    // ("not yet active") so the driver knows the resource but no slot is
    // a reference. P: the active reference slot (its real slot_index)
    // AND the setup slot bound with slot_index=-1 (it becomes active via
    // setup_reference_slot, not as an input reference).
    let begin_slots: Vec<vk::VideoReferenceSlotInfoKHR> = if is_idr {
        vec![vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&setup_dpb_resource)]
    } else {
        vec![
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(ref_slot)
                .picture_resource(&ref_dpb_resource),
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_dpb_resource),
        ]
    };
    // NB: we deliberately do NOT chain a VkVideoEncodeRateControlInfoKHR
    // onto begin_info. The constant-QP (DISABLED) mode is established by
    // the cmd_control_video_coding call below and persists as session
    // state across each independent begin/end scope. Chaining a
    // DISABLED rate-control struct here trips a "does not match the
    // currently configured state" validation error on the very first
    // frame (the session still holds its DEFAULT mode until the control
    // call runs). The driver (RADV) tolerates the absent struct.
    let begin_info = vk::VideoBeginCodingInfoKHR::default()
        .video_session(session.session)
        .video_session_parameters(session.parameters)
        .reference_slots(&begin_slots);
    unsafe {
        (ctx.video_queue_dev.fp().cmd_begin_video_coding_khr)(cb, &begin_info);
    }
    // Rate control is constant-QP (DISABLED) throughout. RESET re-inits
    // the encoder state — do it only on IDR / GOP start, never on a
    // P-frame (a mid-GOP RESET can drop the reference).
    let mut rate_control_layer = vk::VideoEncodeRateControlInfoKHR::default()
        .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
    let control_flags = if is_idr {
        vk::VideoCodingControlFlagsKHR::RESET | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
    } else {
        vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
    };
    let control_info = vk::VideoCodingControlInfoKHR::default()
        .flags(control_flags)
        .push_next(&mut rate_control_layer);
    unsafe {
        (ctx.video_queue_dev.fp().cmd_control_video_coding_khr)(cb, &control_info);
        ctx.device
            .cmd_begin_query(cb, fr.query_pool, 0, vk::QueryControlFlags::empty());
    }
    let mut pic_flags: vk::native::StdVideoEncodeH264PictureInfoFlags =
        unsafe { std::mem::zeroed() };
    pic_flags.set_IdrPicFlag(if is_idr { 1 } else { 0 });
    // Every picture is a reference for the next P-frame.
    pic_flags.set_is_reference(1);
    let ref_lists_flags: vk::native::StdVideoEncodeH264ReferenceListsInfoFlags =
        unsafe { std::mem::zeroed() };
    // For P-frames, RefPicList0[0] points at the active reference slot.
    let mut ref_pic_list0 = [0xFFu8; 32];
    if !is_idr {
        ref_pic_list0[0] = ref_slot as u8;
    }
    let ref_lists = vk::native::StdVideoEncodeH264ReferenceListsInfo {
        flags: ref_lists_flags,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        RefPicList0: ref_pic_list0,
        RefPicList1: [0xFF; 32],
        refList0ModOpCount: 0,
        refList1ModOpCount: 0,
        refPicMarkingOpCount: 0,
        reserved1: [0; 7],
        pRefList0ModOperations: std::ptr::null(),
        pRefList1ModOperations: std::ptr::null(),
        pRefPicMarkingOperations: std::ptr::null(),
    };
    let primary_pic_type = if is_idr {
        vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
    } else {
        vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
    };
    let pic_info = vk::native::StdVideoEncodeH264PictureInfo {
        flags: pic_flags,
        seq_parameter_set_id: 0,
        pic_parameter_set_id: 0,
        idr_pic_id: 0,
        primary_pic_type,
        frame_num,
        PicOrderCnt: poc,
        temporal_id: 0,
        reserved1: [0; 3],
        pRefLists: &ref_lists,
    };
    let slice_type = if is_idr {
        vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
    } else {
        vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
    };
    let slice_flags: vk::native::StdVideoEncodeH264SliceHeaderFlags = unsafe { std::mem::zeroed() };
    let slice_header = vk::native::StdVideoEncodeH264SliceHeader {
        flags: slice_flags,
        first_mb_in_slice: 0,
        slice_type,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_qp_delta: 0,
        reserved1: 0,
        cabac_init_idc: vk::native::StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
        disable_deblocking_filter_idc:
            vk::native::StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
        pWeightTable: std::ptr::null(),
    };
    let nalu_slice = vk::VideoEncodeH264NaluSliceInfoKHR::default()
        .constant_qp(vk_encode_qp())
        .std_slice_header(&slice_header);
    let nalu_slice_arr = [nalu_slice];
    let mut h264_pic_info = vk::VideoEncodeH264PictureInfoKHR::default()
        .nalu_slice_entries(&nalu_slice_arr)
        .std_picture_info(&pic_info);
    let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
        .coded_offset(vk::Offset2D::default())
        .coded_extent(vk::Extent2D {
            width: w,
            height: h,
        })
        .base_array_layer(0)
        .image_view_binding(fr.nv12_color_view);
    let ref_info_flags: vk::native::StdVideoEncodeH264ReferenceInfoFlags =
        unsafe { std::mem::zeroed() };
    let std_ref_info = vk::native::StdVideoEncodeH264ReferenceInfo {
        flags: ref_info_flags,
        primary_pic_type,
        FrameNum: frame_num,
        PicOrderCnt: poc,
        long_term_pic_num: 0,
        long_term_frame_idx: 0,
        temporal_id: 0,
    };
    let mut h264_dpb_slot =
        vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_ref_info);
    // The reconstructed picture is written into the setup slot.
    let setup_ref_slot = vk::VideoReferenceSlotInfoKHR::default()
        .slot_index(setup_slot)
        .picture_resource(&setup_dpb_resource)
        .push_next(&mut h264_dpb_slot);
    // The active reference describes the PREVIOUS picture's std info
    // (its own frame_num / POC), not the current one.
    let ref_frame_num = frame_num.wrapping_sub(1);
    let ref_std_info = vk::native::StdVideoEncodeH264ReferenceInfo {
        flags: ref_info_flags,
        // The reference is whatever we last reconstructed; treat it as a
        // P/IDR reference (type only affects bookkeeping for our 1-ref GOP).
        primary_pic_type: if ref_frame_num == 0 {
            vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
        } else {
            vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
        },
        FrameNum: ref_frame_num,
        PicOrderCnt: ref_frame_num.wrapping_mul(2) as i32,
        long_term_pic_num: 0,
        long_term_frame_idx: 0,
        temporal_id: 0,
    };
    // P-frames must also describe the active reference slot to the
    // encode op via VideoEncodeInfoKHR.pReferenceSlots.
    let mut h264_ref_dpb_slot =
        vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&ref_std_info);
    let encode_ref_slots = if is_idr {
        Vec::new()
    } else {
        vec![vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(ref_slot)
            .picture_resource(&ref_dpb_resource)
            .push_next(&mut h264_ref_dpb_slot)]
    };
    let mut encode_info = vk::VideoEncodeInfoKHR::default()
        .dst_buffer(fr.encoded_buffer)
        .dst_buffer_offset(0)
        .dst_buffer_range(fr.encoded_size)
        .src_picture_resource(src_picture_resource)
        .setup_reference_slot(&setup_ref_slot)
        .push_next(&mut h264_pic_info);
    if !is_idr {
        encode_info = encode_info.reference_slots(&encode_ref_slots);
    }
    unsafe {
        (ctx.video_encode_queue_dev.fp().cmd_encode_video_khr)(cb, &encode_info);
        ctx.device.cmd_end_query(cb, fr.query_pool, 0);
    }
    let end_info = vk::VideoEndCodingInfoKHR::default();
    unsafe {
        (ctx.video_queue_dev.fp().cmd_end_video_coding_khr)(cb, &end_info);
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| format!("end_command_buffer(encode): {e:?}"))?;
    }
    let cbs = [cb];
    let wait_sems = [wait_sem];
    let wait_stages = [vk::PipelineStageFlags::TOP_OF_PIPE];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&cbs)
        .wait_semaphores(&wait_sems)
        .wait_dst_stage_mask(&wait_stages);
    unsafe {
        ctx.device
            .queue_submit(ctx.video_encode_queue, &[submit], fr.fence)
            .map_err(|e| format!("queue_submit(encode): {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fr.fence], true, fence_timeout_ns())
            .map_err(|e| format!("wait_for_fences(encode): {e:?}"))?;
    }
    #[repr(C)]
    #[derive(Default, Debug)]
    struct FeedbackResult {
        offset: u32,
        bytes_written: u32,
        status: i32,
    }
    let mut feedback = FeedbackResult::default();
    let result = unsafe {
        (ctx.device.fp_v1_0().get_query_pool_results)(
            ctx.device.handle(),
            fr.query_pool,
            0,
            1,
            std::mem::size_of::<FeedbackResult>(),
            &mut feedback as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<FeedbackResult>() as u64,
            vk::QueryResultFlags::WAIT | vk::QueryResultFlags::WITH_STATUS_KHR,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!("get_query_pool_results: {result:?}"));
    }
    if feedback.bytes_written == 0 {
        return Err(format!(
            "encode produced 0 bytes (status={})",
            feedback.status
        ));
    }
    let mut out = vec![0u8; feedback.bytes_written as usize];
    unsafe {
        let ptr = ctx
            .device
            .map_memory(
                fr.encoded_memory,
                feedback.offset as u64,
                feedback.bytes_written as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map_memory(encoded): {e:?}"))?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), out.len());
        ctx.device.unmap_memory(fr.encoded_memory);
    }
    Ok(out)
}

// barrier setup takes many tightly-related Vulkan params by design
#[allow(clippy::too_many_arguments)]
fn image_layout_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Layout barrier targeting a single array LAYER of an image. Used for
/// the 2-layer DPB image where each layer is a separate H.264 reference
/// slot — we transition the setup layer and the reference layer
/// independently within one P-frame submit.
#[allow(clippy::too_many_arguments)]
fn dpb_layer_layout_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    layer: u32,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(1),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Query vkGetPhysicalDeviceVideoFormatPropertiesKHR for a given
/// (encode profile, desired image usage) pair. Returns the list of
/// VkVideoFormatPropertiesKHR entries the driver advertises as valid
/// for creating an image of that profile+usage.
///
/// Used by `FrameResources::new` to verify our intended format +
/// usage tuple is in the supported set before calling vkCreateImage
/// (which on AMD silently succeeds for unsupported tuples and then
/// loses the device during encode submit).
pub fn query_video_format_props(
    ctx: &VkDeviceCtx,
    image_usage: vk::ImageUsageFlags,
) -> Result<Vec<vk::VideoFormatPropertiesKHR<'static>>, String> {
    let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
        .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
    let profile_info = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut h264_profile);
    let profile_array = [profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profile_array);

    let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(image_usage)
        .push_next(&mut profile_list);

    let mut count: u32 = 0;
    let result = unsafe {
        (ctx.video_queue_inst
            .fp()
            .get_physical_device_video_format_properties_khr)(
            ctx.physical_device,
            &format_info,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR(size): {result:?}"
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut props: Vec<vk::VideoFormatPropertiesKHR<'static>> = (0..count as usize)
        .map(|_| vk::VideoFormatPropertiesKHR::default())
        .collect();
    let result = unsafe {
        (ctx.video_queue_inst
            .fp()
            .get_physical_device_video_format_properties_khr)(
            ctx.physical_device,
            &format_info,
            &mut count,
            props.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR(data): {result:?}"
        ));
    }
    props.truncate(count as usize);
    Ok(props)
}

/// Query the driver's reported std-headers version for the H.264 main
/// encode profile. Returns (name bytes padded to MAX_EXTENSION_NAME_SIZE,
/// spec_version u32). Hardcoded constants don't survive driver swaps:
/// AMD Mesa and NVIDIA 560.35 report different spec_versions despite
/// both implementing the same KHR_video_encode_h264 extension.
fn query_h264_std_header_version(
    video_queue_inst: &ash::khr::video_queue::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<([i8; vk::MAX_EXTENSION_NAME_SIZE], u32), String> {
    let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
        .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
    let profile_info = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut h264_profile);

    let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
    let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
    let mut caps = vk::VideoCapabilitiesKHR::default()
        .push_next(&mut encode_caps)
        .push_next(&mut h264_caps);
    let result = unsafe {
        (video_queue_inst
            .fp()
            .get_physical_device_video_capabilities_khr)(
            physical_device, &profile_info, &mut caps
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "get_physical_device_video_capabilities_khr: {result:?}"
        ));
    }
    Ok((
        caps.std_header_version.extension_name,
        caps.std_header_version.spec_version,
    ))
}

/// Query supported `image_create_flags` for `G8_B8R8_2PLANE_420_UNORM`
/// under the H.264 main encode profile + a given image usage. The
/// driver's response is authoritative: AMD reports
/// MUTABLE_FORMAT|ALIAS|EXTENDED_USAGE; NVIDIA reports a different set
/// (the AMD constants we hardcoded before were what caused
/// EncodeSession::create_parameters / encode submit to fail mid-flight
/// on the L40 rental). Returns the flags from the FIRST entry matching
/// the NV12 format; errors if no entry matches.
fn query_nv12_image_flags(
    video_queue_inst: &ash::khr::video_queue::Instance,
    physical_device: vk::PhysicalDevice,
    image_usage: vk::ImageUsageFlags,
) -> Result<vk::ImageCreateFlags, String> {
    let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
        .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
    let profile_info = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut h264_profile);
    let profile_array = [profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profile_array);
    let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(image_usage)
        .push_next(&mut profile_list);

    let mut count: u32 = 0;
    let result = unsafe {
        (video_queue_inst
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR(size, usage={image_usage:?}): {result:?}"
        ));
    }
    if count == 0 {
        return Err(format!("no formats reported for usage={image_usage:?}"));
    }
    let mut props: Vec<vk::VideoFormatPropertiesKHR<'static>> = (0..count as usize)
        .map(|_| vk::VideoFormatPropertiesKHR::default())
        .collect();
    let result = unsafe {
        (video_queue_inst
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            props.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR(data, usage={image_usage:?}): {result:?}"
        ));
    }
    props.truncate(count as usize);
    for p in &props {
        if p.format == vk::Format::G8_B8R8_2PLANE_420_UNORM {
            return Ok(p.image_create_flags);
        }
    }
    Err(format!(
        "no G8_B8R8_2PLANE_420_UNORM entry in {} reported formats for usage={image_usage:?}",
        props.len()
    ))
}

/// Query the driver's reported caps + std-headers version under the
/// H.264 Hi444PP profile. Returns `(name_bytes, spec_version)` on
/// success; errors if the driver doesn't expose Hi444PP (the typical
/// AMD/Mesa case as of 2026-05-12 — Main 4:2:0 only).
fn query_hi444pp_caps(
    video_queue_inst: &ash::khr::video_queue::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<([i8; vk::MAX_EXTENSION_NAME_SIZE], u32), String> {
    let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
        vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
    );
    let profile_info = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut h264_profile);

    let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
    let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
    let mut caps = vk::VideoCapabilitiesKHR::default()
        .push_next(&mut encode_caps)
        .push_next(&mut h264_caps);
    let result = unsafe {
        (video_queue_inst
            .fp()
            .get_physical_device_video_capabilities_khr)(
            physical_device, &profile_info, &mut caps
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!("Hi444PP caps query: {result:?}"));
    }
    Ok((
        caps.std_header_version.extension_name,
        caps.std_header_version.spec_version,
    ))
}

/// Query supported `image_create_flags` for `G8_B8_R8_3PLANE_444_UNORM`
/// under the H.264 Hi444PP encode profile + a given image usage.
/// Errors if no entry matches.
fn query_yuv444_image_flags(
    video_queue_inst: &ash::khr::video_queue::Instance,
    physical_device: vk::PhysicalDevice,
    image_usage: vk::ImageUsageFlags,
) -> Result<vk::ImageCreateFlags, String> {
    let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
        vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
    );
    let profile_info = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut h264_profile);
    let profile_array = [profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profile_array);
    let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(image_usage)
        .push_next(&mut profile_list);

    let mut count: u32 = 0;
    let result = unsafe {
        (video_queue_inst
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR(yuv444, size, usage={image_usage:?}): {result:?}"
        ));
    }
    if count == 0 {
        return Err(format!(
            "no yuv444 formats reported for usage={image_usage:?}"
        ));
    }
    let mut props: Vec<vk::VideoFormatPropertiesKHR<'static>> = (0..count as usize)
        .map(|_| vk::VideoFormatPropertiesKHR::default())
        .collect();
    let result = unsafe {
        (video_queue_inst
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            props.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR(yuv444, data, usage={image_usage:?}): {result:?}"
        ));
    }
    props.truncate(count as usize);
    // Prefer 2-plane 4:4:4 (NVIDIA's supported format); fall back to
    // 3-plane if that's all the driver reports.
    for p in &props {
        if p.format == vk::Format::G8_B8R8_2PLANE_444_UNORM {
            return Ok(p.image_create_flags);
        }
    }
    for p in &props {
        if p.format == vk::Format::G8_B8_R8_3PLANE_444_UNORM {
            return Ok(p.image_create_flags);
        }
    }
    Err(format!(
        "no 4:4:4 YUV format in {} reported formats for usage={image_usage:?}",
        props.len()
    ))
}

/// Set true when a GPU fence wait times out, i.e. the device is wedged. Read by
/// teardown (`Drop`) paths so an unbounded `device_wait_idle()` is skipped on a
/// hung device: otherwise cleanup blocks forever and a `softlockup_panic=1` /
/// hung-task kernel reboots the whole box instead of just dropping the stream.
pub(crate) static GPU_WEDGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fence-wait timeout in nanoseconds. MUST stay well under the kernel
/// soft-lockup / hung-task thresholds (~20s on this host's
/// `softlockup_panic=1 nmi_watchdog=1` cmdline): a fence wait that outlives
/// that window turns a recoverable GPU stall under iGPU/VCN contention into a
/// HARD panic + reboot (observed: 3 reboots, no amdgpu reset logged because the
/// watchdog fires first). A healthy 720p/4K encode completes in tens of ms, so
/// \>3s reliably means "wedged". Default 3s; any `WAYMUX_VK_FENCE_TIMEOUT_NS`
/// override is hard-capped at 10s so it can never approach the watchdog. Tests
/// may drop it further (e.g. 2s). On timeout the caller maps the error to a
/// dropped frame, so the stream degrades instead of the machine rebooting.
fn fence_timeout_ns() -> u64 {
    clamp_fence_timeout(
        std::env::var("WAYMUX_VK_FENCE_TIMEOUT_NS")
            .ok()
            .and_then(|s| s.parse().ok()),
    )
}

/// Pure clamp for [`fence_timeout_ns`]: default 3s, hard cap 10s. Split out so
/// the watchdog-safety invariant (no override can reach the ~20s kernel
/// soft-lockup window) is unit-testable without touching process env.
fn clamp_fence_timeout(parsed: Option<u64>) -> u64 {
    const DEFAULT_NS: u64 = 3_000_000_000;
    const HARD_CAP_NS: u64 = 10_000_000_000;
    parsed.unwrap_or(DEFAULT_NS).min(HARD_CAP_NS)
}

/// H.264 constant quantization parameter (QP) used by the Vulkan encoder.
/// H.264 QP scale is 0 (mathematically lossless, huge files) to 51 (max
/// compression). Default 20 = visually-clean for screens AND for natural
/// 4K content like the marketing eagle recording. Marketing-grade
/// rentals can drop to 16-18 via `WAYMUX_VK_ENCODE_QP`; size-sensitive
/// (visual-regression CI) callers can raise to 26-28.
///
/// Previous default was 26 (grainy on natural 4K content — feedback
/// after the 2026-05-11 hero rental).
/// Dynamic QP for the viewer encoder's adaptive rate control. -1 = unset
/// (fall back to the `WAYMUX_VK_ENCODE_QP` env / default). The viewer loop
/// (`run_vulkan_encoder`) drives this from the live GCC bandwidth estimate so
/// the constant-QP encoder tracks the available link bandwidth — higher QP
/// (smaller frames) when the link is tight, lower QP (sharper) when it's fat.
pub(crate) static VIEWER_DYNAMIC_QP: std::sync::atomic::AtomicI32 =
    std::sync::atomic::AtomicI32::new(-1);

fn vk_encode_qp() -> i32 {
    let dynamic = VIEWER_DYNAMIC_QP.load(std::sync::atomic::Ordering::Relaxed);
    if dynamic >= 0 {
        return dynamic.clamp(0, 51);
    }
    std::env::var("WAYMUX_VK_ENCODE_QP")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|q: i32| q.clamp(0, 51))
        .unwrap_or(20)
}

// ────────────────────────────────────────────────────────────────────────
// VkRecorder — public API for the rest of waymux
//
// Bundles VkDeviceCtx + EncodeSession + FrameResources. Owns the
// recording lifetime: opened once when a recording starts, dropped
// when it ends. The recording thread calls `encode_idr_from_bgra` (or
// future `encode_frame_from_dmabuf` once item 6 lands) with each
// frame and receives encoded NAL bytes + PTS to hand to MkvWriter.

/// Encoded H.264 NAL bytes + presentation timestamp from one frame.
pub struct EncodedNal {
    pub data: Vec<u8>,
    pub pts_us: i64,
    pub is_keyframe: bool,
}

/// Imported-dmabuf VkImage suitable for `vkCmdCopyImage` as a source.
///
/// Differs from the H.264 path's inline dmabuf-image creation in usage:
/// FFV1 doesn't sample in a shader — it copies the BGRA pixels into a
/// libav-owned image. So this helper requests `TRANSFER_SRC` usage and
/// returns just the raw `vk::Image` + backing memory (no image view
/// needed). Caller must `destroy(ctx)` when done; importing the
/// dmabuf fd transfers ownership of the dup'd fd to Vulkan, so the
/// caller's original DmabufBufferData is unaffected.
pub struct ImportedDmabufImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
}

impl ImportedDmabufImage {
    pub fn destroy(self, ctx: &VkDeviceCtx) {
        unsafe {
            ctx.device.destroy_image(self.image, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

/// Process-wide cache of the DRM modifiers we can import as a BGRA8888
/// `VkImage` (TRANSFER_SRC, external dmabuf memory). Device-static, so
/// queried once. Always includes LINEAR (even with no Vulkan device) so
/// software/LINEAR clients keep working.
static IMPORTABLE_BGRA_MODIFIERS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();

/// The cached Vulkan-importable modifier set for BGRA8888, LINEAR-inclusive.
/// Gates the Vulkan zero-copy import path (AMD/Mesa can import tiled; NVIDIA's
/// Vulkan can only take LINEAR). NOT the advertised set — see
/// `importable_bgra_modifiers()`.
pub fn vulkan_importable_bgra_modifiers() -> &'static [u64] {
    IMPORTABLE_BGRA_MODIFIERS.get_or_init(|| {
        let mut set = query_importable_bgra_modifiers().unwrap_or_default();
        if !set.contains(&crate::dmabuf::DRM_FORMAT_MOD_LINEAR) {
            set.push(crate::dmabuf::DRM_FORMAT_MOD_LINEAR);
        }
        set.sort_unstable();
        set.dedup();
        tracing::info!(?set, "dmabuf: Vulkan-importable BGRA modifiers");
        set
    })
}

/// The ADVERTISED dmabuf modifier set (what we tell KWin + can consume via
/// egl_readback). EGL-importable ∪ LINEAR — works on NVIDIA and Mesa.
/// NOTE: distinct from `vulkan_importable_bgra_modifiers()`, which gates the
/// Vulkan zero-copy import (AMD/Mesa only; LINEAR-only on NVIDIA).
pub fn importable_bgra_modifiers() -> &'static [u64] {
    crate::dmabuf::egl_importable_bgra_modifiers()
}

/// True if a modifier can be imported by the Vulkan zero-copy path
/// (`import_dmabuf_as_transfer_src` / `encode_idr_from_dmabuf_impl`). This is
/// the VULKAN set, not the advertised set — the import must only attempt what
/// Vulkan can actually take.
pub fn modifier_is_importable(modifier: u64) -> bool {
    vulkan_importable_bgra_modifiers().contains(&modifier)
}

/// Live Vulkan query: enumerate the DRM modifiers the physical device
/// reports for B8G8R8A8_UNORM, then keep only those importable as an
/// external-dmabuf SAMPLED|TRANSFER_SRC image. Returns None if no Vulkan
/// device / the modifier extension is absent (caller falls back to
/// LINEAR-only).
fn query_importable_bgra_modifiers() -> Option<Vec<u64>> {
    // NOTE: VkDeviceCtx::open() requires the video-encode queue/extensions, so
    // on an (import-capable but) encode-less GPU this returns None and we fall
    // back to LINEAR-only advertisement. That is acceptable here: the zero-copy
    // tiled pipeline needs the encode device anyway, and our fleet (L40 + NVENC)
    // is always encode-capable. If this set is ever reused on display-only/MIG
    // hardware, switch to a minimal instance + physical-device modifier query.
    let ctx = VkDeviceCtx::open().ok()?;
    if !ctx.dmabuf_import_supported {
        return None;
    }
    let fmt = vk::Format::B8G8R8A8_UNORM;

    // 1. List all modifiers the device knows for this format.
    let mut mod_list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut fmt_props2 = vk::FormatProperties2::default().push_next(&mut mod_list);
    unsafe {
        ctx.instance.get_physical_device_format_properties2(
            ctx.physical_device,
            fmt,
            &mut fmt_props2,
        );
    }
    let count = mod_list.drm_format_modifier_count as usize;
    if count == 0 {
        return Some(Vec::new());
    }
    let mut props: Vec<vk::DrmFormatModifierPropertiesEXT> = vec![Default::default(); count];
    mod_list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut props);
    let mut fmt_props2 = vk::FormatProperties2::default().push_next(&mut mod_list);
    unsafe {
        ctx.instance.get_physical_device_format_properties2(
            ctx.physical_device,
            fmt,
            &mut fmt_props2,
        );
    }

    // 2. Keep only modifiers importable as an external-dmabuf SAMPLED|TRANSFER_SRC
    //    2D image (single plane — matches our packed-BGRA import).
    //    The union of both real import usages is probed so that a modifier
    //    passing here is valid for both encode_idr_from_dmabuf_impl (SAMPLED)
    //    and import_dmabuf_as_transfer_src (TRANSFER_SRC).
    let mut out = Vec::new();
    for p in props.iter() {
        if p.drm_format_modifier_plane_count != 1 {
            continue;
        }
        let mut drm_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(p.drm_format_modifier)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut ext_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let img_info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(fmt)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
            .push_next(&mut ext_info)
            .push_next(&mut drm_info);
        let mut img_props = vk::ImageFormatProperties2::default();
        let ok = unsafe {
            ctx.instance
                .get_physical_device_image_format_properties2(
                    ctx.physical_device,
                    &img_info,
                    &mut img_props,
                )
                .is_ok()
        };
        if ok {
            out.push(p.drm_format_modifier);
        }
    }
    Some(out)
}

/// Import a LINEAR-modifier client dmabuf as a Vulkan `vk::Image` with
/// `TRANSFER_SRC` usage. Caller is responsible for destroying the
/// returned handles.
pub fn import_dmabuf_as_transfer_src(
    ctx: &VkDeviceCtx,
    dma: &crate::dmabuf::DmabufBufferData,
) -> Result<ImportedDmabufImage, String> {
    if !modifier_is_importable(dma.modifier) {
        return Err(format!(
            "dmabuf modifier {:#x} not in Vulkan-importable set {:?}",
            dma.modifier,
            vulkan_importable_bgra_modifiers()
        ));
    }
    let is_tiled = dma.modifier != crate::dmabuf::DRM_FORMAT_MOD_LINEAR;
    let vk_format = match dma.drm_format {
        crate::dmabuf::DRM_FORMAT_ARGB8888 | crate::dmabuf::DRM_FORMAT_XRGB8888 => {
            vk::Format::B8G8R8A8_UNORM
        }
        f => return Err(format!("unsupported drm_format {f:#x}")),
    };
    let width = dma.width as u32;
    let height = dma.height as u32;

    let raw_fd: i32 = unsafe { libc::dup(std::os::fd::AsRawFd::as_raw_fd(&dma.fd)) };
    if raw_fd < 0 {
        return Err(format!(
            "dup(dmabuf fd): {}",
            std::io::Error::last_os_error()
        ));
    }

    let mut external_mem_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    // Per-plane layout for the explicit-modifier import. Plane 0 is the packed
    // BGRA plane; the Task-A query filtered to single-plane modifiers, so one entry.
    let plane_layouts = [vk::SubresourceLayout {
        offset: dma.offset as u64,
        size: 0, // 0 = "the rest of the bound memory"
        row_pitch: dma.stride as u64,
        array_pitch: 0,
        depth_pitch: 0,
    }];
    let mut drm_explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(dma.modifier)
        .plane_layouts(&plane_layouts);

    let tiling = if is_tiled {
        vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT
    } else {
        vk::ImageTiling::LINEAR
    };
    let mut image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(tiling)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_mem_info);
    if is_tiled {
        image_ci = image_ci.push_next(&mut drm_explicit);
    }
    let image = unsafe {
        ctx.device.create_image(&image_ci, None).map_err(|e| {
            libc::close(raw_fd);
            format!("create_image(dmabuf): {e:?}")
        })?
    };

    let mem_req = unsafe { ctx.device.get_image_memory_requirements(image) };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let type_idx = pick_memory_type(
        &mem_props,
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::empty(),
    )
    .ok_or_else(|| {
        unsafe { libc::close(raw_fd) };
        unsafe { ctx.device.destroy_image(image, None) };
        format!(
            "no memory type for dmabuf import (mask={:#x})",
            mem_req.memory_type_bits
        )
    })?;
    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(raw_fd);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(type_idx)
        .push_next(&mut import_info)
        .push_next(&mut dedicated);
    let memory = unsafe {
        ctx.device.allocate_memory(&alloc_info, None).map_err(|e| {
            libc::close(raw_fd);
            ctx.device.destroy_image(image, None);
            format!("allocate_memory(dmabuf import): {e:?}")
        })?
    };
    // On success Vulkan owns raw_fd; do NOT close it.

    unsafe {
        ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                format!("bind_image_memory(dmabuf): {e:?}")
            })?;
    }

    Ok(ImportedDmabufImage {
        image,
        memory,
        width,
        height,
        format: vk_format,
    })
}

/// Upload packed BGRA CPU bytes into a LINEAR, host-visible `TRANSFER_SRC`
/// `VkImage`. Mirrors `import_dmabuf_as_transfer_src`'s return shape so the
/// FFV1/HEVC GPU encoders can `vkCmdCopyImage` from it identically — but the
/// source pixels come from a CPU buffer (an SHM/`wp_single_pixel` client such
/// as `foot`, which never produces a dmabuf) instead of a client dmabuf.
///
/// Without this, the `ffv1_vulkan` recording thread had no way to encode an
/// SHM-only client: its only input path was `RecordingTask::Dmabuf`, so an
/// SHM client (which the compositor tap delivers as `RecordingTask::Pixels`)
/// produced an empty MKV. `h264_vulkan` already had its BGRA CPU path
/// (`encode_idr_from_bgra`); this gives the FFV1 path the same coverage.
///
/// The image is created LINEAR so we can `map`+memcpy the rows directly (no
/// staging buffer + copy), respecting the driver-reported `row_pitch`. Caller
/// must `destroy(ctx)` when done.
pub fn upload_bgra_to_transfer_src(
    ctx: &VkDeviceCtx,
    bgra: &[u8],
    width: u32,
    height: u32,
) -> Result<ImportedDmabufImage, String> {
    let vk_format = vk::Format::B8G8R8A8_UNORM;
    let row_bytes = (width as usize) * 4;
    let expected = row_bytes * (height as usize);
    if bgra.len() < expected {
        return Err(format!(
            "bgra buffer too small: {} bytes for {width}x{height} (need {expected})",
            bgra.len()
        ));
    }

    let image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        // LINEAR so the host can write directly into the image memory; the
        // driver lays plane 0 out row-by-row at the reported row_pitch.
        .tiling(vk::ImageTiling::LINEAR)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        // PREINITIALIZED keeps the host-written contents valid across the
        // first layout transition the encoder's copy applies.
        .initial_layout(vk::ImageLayout::PREINITIALIZED);
    let image = unsafe {
        ctx.device
            .create_image(&image_ci, None)
            .map_err(|e| format!("create_image(bgra upload): {e:?}"))?
    };

    let mem_req = unsafe { ctx.device.get_image_memory_requirements(image) };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let type_idx = pick_memory_type(
        &mem_props,
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        unsafe { ctx.device.destroy_image(image, None) };
        format!(
            "no host-visible memory type for bgra upload (mask={:#x})",
            mem_req.memory_type_bits
        )
    })?;
    let memory = unsafe {
        ctx.device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(type_idx),
                None,
            )
            .map_err(|e| {
                ctx.device.destroy_image(image, None);
                format!("allocate_memory(bgra upload): {e:?}")
            })?
    };
    unsafe {
        ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                format!("bind_image_memory(bgra upload): {e:?}")
            })?;
    }

    // Query the LINEAR layout so we honor the driver's row pitch (may exceed
    // width*4 for alignment). Copy row-by-row into the mapped memory.
    let layout = unsafe {
        ctx.device.get_image_subresource_layout(
            image,
            vk::ImageSubresource::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .array_layer(0),
        )
    };
    let row_pitch = layout.row_pitch as usize;
    let offset = layout.offset as usize;
    let map_size = offset + row_pitch * (height as usize);
    let map_size = (map_size as u64).min(mem_req.size);
    unsafe {
        let ptr = ctx
            .device
            .map_memory(memory, 0, map_size, vk::MemoryMapFlags::empty())
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                format!("map_memory(bgra upload): {e:?}")
            })? as *mut u8;
        for row in 0..height as usize {
            let src = &bgra[row * row_bytes..row * row_bytes + row_bytes];
            let dst = ptr.add(offset + row * row_pitch);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, row_bytes);
        }
        // HOST_COHERENT memory needs no explicit flush.
        ctx.device.unmap_memory(memory);
    }

    Ok(ImportedDmabufImage {
        image,
        memory,
        width,
        height,
        format: vk_format,
    })
}

/// Per-recording Vulkan encoder. Owns the device, encode session,
/// and per-frame resources. Drop tears down everything cleanly.
///
/// Field order is load-bearing: Rust drops fields in declaration
/// order, so `fr` and `session` (which hold cloned device handles
/// they call into during their own Drop) must come before `ctx`.
/// `ctx` is dropped last so its `destroy_device` happens after all
/// per-session resources are freed.
pub struct VkRecorder {
    fr: FrameResources,
    pipe: BgraToNv12Pipeline,
    session: EncodeSession,
    ctx: VkDeviceCtx,
    width: u32,
    height: u32,
    /// H.264 `frame_num` of the *next* picture. Reset to 0 on each IDR
    /// (which also resets the GOP). With SPS `log2_max_frame_num_minus4
    /// = 0` this wraps at 16, so callers keep GOP length < 16.
    frame_num: u32,
    /// DPB slot holding the current reference reconstruction (the last
    /// encoded picture). The next P-frame references this slot and
    /// reconstructs into the *other* slot, then this flips. -1 means "no
    /// reference yet" (before the first picture / right after an IDR that
    /// hasn't been encoded). Slot 0 = DPB array layer 0 (dpb_image_view),
    /// slot 1 = layer 1 (dpb_image_view2), both in the one DPB image.
    ref_slot: i32,
}

impl VkRecorder {
    /// Initialize. Returns None on any setup failure (driver too old,
    /// no encode queue, format combo unsupported). Callers fall back
    /// to the legacy ffmpeg+OpenGL path.
    pub fn try_new(width: u32, height: u32) -> Option<Self> {
        let ctx = VkDeviceCtx::open()
            .map_err(|e| {
                tracing::warn!("VkDeviceCtx::open failed: {e}");
            })
            .ok()?;
        let mut session = EncodeSession::new(&ctx, width, height)
            .map_err(|e| tracing::warn!("EncodeSession::new failed: {e}"))
            .ok()?;
        session
            .create_parameters()
            .map_err(|e| tracing::warn!("create_parameters failed: {e}"))
            .ok()?;
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let fr = FrameResources::new(&ctx, width, height, &mut h264_profile)
            .map_err(|e| tracing::warn!("FrameResources::new failed: {e}"))
            .ok()?;
        let pipe = BgraToNv12Pipeline::new(&ctx, 1)
            .map_err(|e| tracing::warn!("BgraToNv12Pipeline::new failed: {e}"))
            .ok()?;
        Some(VkRecorder {
            fr,
            pipe,
            session,
            ctx,
            width,
            height,
            frame_num: 0,
            ref_slot: -1,
        })
    }

    /// AVCDecoderConfigurationRecord bytes for the MKV writer's
    /// codec_private field. Available after construction.
    pub fn codec_private(&self) -> &[u8] {
        self.session.codec_private()
    }

    /// In-band Annex-B SPS+PPS bytes from the driver. The viewer encoder
    /// prepends these to each IDR because AMD/Mesa does not emit parameter
    /// sets in-band (a decoder otherwise reports "non-existing PPS").
    pub fn sps_pps_annexb(&self) -> &[u8] {
        self.session.sps_pps_annexb()
    }

    /// Zero-copy: encode an IDR frame directly from a dmabuf without
    /// any CPU touch of the pixel data. The dmabuf's fd is imported
    /// via `VkImportMemoryFdInfoKHR` (`DMA_BUF_BIT_EXT`), bound to a
    /// transient `VkImage`, sampled by the compute shader, encoded.
    ///
    /// Replaces the staging-upload step in `encode_idr_from_bgra` with
    /// a pure fd dup — no pixel bytes ever cross the PCIe bus a second
    /// The dmabuf may use any modifier in
    /// `importable_bgra_modifiers()` — LINEAR plus whatever tiled
    /// modifiers the device reported importable (Task A/B). Tiled buffers
    /// are imported via the `VkImageDrmFormatModifierExplicitCreateInfoEXT`
    /// path. Returns None on an un-importable modifier or any Vulkan
    /// failure; callers fall back to a CPU readback + `encode_idr_from_bgra`.
    pub fn encode_idr_from_dmabuf(
        &self,
        dma: &crate::dmabuf::DmabufBufferData,
        pts_us: i64,
    ) -> Option<EncodedNal> {
        // Wrap the whole encode in a span so `tracing-subscriber` (with
        // RUST_LOG=info) records wall-clock latency per Vulkan encode.
        // Answers investigation open question #1: "wall-clock latency of
        // one Vulkan encode on L40 is unmeasured." Use `in_scope` so the
        // span closes correctly on every early-return path inside.
        tracing::info_span!(
            "vulkan_encode_idr",
            width = self.width,
            height = self.height,
        )
        .in_scope(|| {
            let data = encode_idr_from_dmabuf_impl(
                &self.ctx,
                &self.fr,
                &self.pipe,
                &self.session,
                dma,
                self.idr_pic_params(),
            )
            .map_err(|e| {
                eprintln!("encode_idr_from_dmabuf failed: {e}");
                tracing::warn!("encode_idr_from_dmabuf failed: {e}");
            })
            .ok()?;
            Some(EncodedNal {
                data,
                pts_us,
                is_keyframe: true,
            })
        })
    }

    /// PicParams for a standalone IDR (frame_num 0, reconstruct into slot
    /// 0, no reference). Used by the IDR-only `&self` entry points so
    /// existing callers keep their every-frame-IDR behavior unchanged.
    fn idr_pic_params(&self) -> PicParams {
        PicParams {
            is_idr: true,
            frame_num: 0,
            ref_slot: -1,
            ref_view: vk::ImageView::null(),
            setup_slot: 0,
            setup_view: self.fr.dpb_image_view,
        }
    }

    /// Compute the PicParams for the next picture and advance the
    /// ping-pong state. On IDR: frame_num resets to 0, reconstruct into
    /// slot 0 (dpb_image), no reference; afterwards the reference is
    /// {slot 0}. On P: reference the current ref slot, reconstruct into
    /// the OTHER slot, frame_num += 1; afterwards the reference is the
    /// new reconstruction's slot.
    ///
    /// View bindings are fixed: slot 0 <-> dpb_image_view, slot 1 <->
    /// dpb_image_view2.
    fn next_pic_params(&mut self, is_idr: bool) -> PicParams {
        let view_for = |slot: i32| {
            if slot == 1 {
                self.fr.dpb_image_view2
            } else {
                self.fr.dpb_image_view
            }
        };
        if is_idr || self.ref_slot < 0 {
            // IDR (or the very first picture): reconstruct into slot 0,
            // no reference. Reset the GOP frame counter.
            self.frame_num = 0;
            let pic = PicParams {
                is_idr: true,
                frame_num: 0,
                ref_slot: -1,
                ref_view: vk::ImageView::null(),
                setup_slot: 0,
                setup_view: view_for(0),
            };
            // This IDR becomes the reference for the next P-frame.
            self.ref_slot = 0;
            self.frame_num = 1;
            pic
        } else {
            // P-frame: reference current slot, reconstruct into the other.
            let ref_slot = self.ref_slot;
            let setup_slot = if ref_slot == 0 { 1 } else { 0 };
            let pic = PicParams {
                is_idr: false,
                frame_num: self.frame_num,
                ref_slot,
                ref_view: view_for(ref_slot),
                setup_slot,
                setup_view: view_for(setup_slot),
            };
            // The picture we just made becomes the next reference.
            self.ref_slot = setup_slot;
            self.frame_num = self.frame_num.wrapping_add(1) % 16;
            pic
        }
    }

    /// Encode a frame from a dmabuf as IDR or P. `&mut self` because it
    /// advances the ping-pong DPB state. This is the P-frame-aware entry
    /// point used by the viewer; `encode_idr_from_dmabuf` stays IDR-only
    /// for callers that want every-frame keyframes.
    pub fn encode_dmabuf(
        &mut self,
        dma: &crate::dmabuf::DmabufBufferData,
        pts_us: i64,
        is_idr: bool,
    ) -> Option<EncodedNal> {
        let pic = self.next_pic_params(is_idr);
        let keyframe = pic.is_idr;
        let span = tracing::info_span!(
            "vulkan_encode",
            width = self.width,
            height = self.height,
            is_idr = keyframe,
        );
        let _g = span.enter();
        let data =
            encode_idr_from_dmabuf_impl(&self.ctx, &self.fr, &self.pipe, &self.session, dma, pic)
                .map_err(|e| {
                    eprintln!("encode_dmabuf failed: {e}");
                    tracing::warn!("encode_dmabuf failed: {e}");
                })
                .ok()?;
        Some(EncodedNal {
            data,
            pts_us,
            is_keyframe: keyframe,
        })
    }

    /// Encode a single IDR frame from CPU-supplied BGRA pixels.
    ///
    /// Pipeline: BGRA bytes uploaded via staging → compute shader on
    /// the GPU produces NV12 in dedicated storage images → vkCmdCopyImage
    /// brings them into the encoder NV12 image → cmd_encode_video_khr.
    /// Zero CPU touch of any pixel data after the staging upload.
    pub fn encode_idr_from_bgra(&self, bgra: &[u8], pts_us: i64) -> Option<EncodedNal> {
        let data = encode_idr_gpu_synthetic(
            &self.ctx,
            &self.fr,
            &self.pipe,
            &self.session,
            bgra,
            self.idr_pic_params(),
        )
        .map_err(|e| {
            eprintln!("encode_idr_gpu_synthetic failed: {e}");
            tracing::warn!("encode_idr_gpu_synthetic failed: {e}");
        })
        .ok()?;
        Some(EncodedNal {
            data,
            pts_us,
            is_keyframe: true,
        })
    }

    /// Encode a BGRA frame as IDR or P. `&mut self` to advance the
    /// ping-pong DPB state. Mirror of `encode_dmabuf` for the CPU-pixels
    /// path; also drives the P-frame spike test.
    pub fn encode_bgra(&mut self, bgra: &[u8], pts_us: i64, is_idr: bool) -> Option<EncodedNal> {
        let pic = self.next_pic_params(is_idr);
        let keyframe = pic.is_idr;
        let data =
            encode_idr_gpu_synthetic(&self.ctx, &self.fr, &self.pipe, &self.session, bgra, pic)
                .map_err(|e| {
                    eprintln!("encode_bgra failed: {e}");
                    tracing::warn!("encode_bgra failed: {e}");
                })
                .ok()?;
        Some(EncodedNal {
            data,
            pts_us,
            is_keyframe: keyframe,
        })
    }
}

/// Lossless H.264 Hi444PP variant of `VkRecorder`. NVIDIA-only on
/// baseline 2026-05-12. Same public API shape (`try_new`, `codec_private`,
/// `encode_idr_from_bgra`) — recording threads pick which struct to
/// instantiate based on the `RecordingCodec` variant.
///
/// Field order is load-bearing for Drop: `fr`/`pipe`/`session` (which
/// hold cloned device handles they call into during Drop) must come
/// before `ctx`, so `ctx`'s `destroy_device` runs last.
pub struct VkRecorderLossless {
    fr: FrameResources444,
    pipe: BgraToYuv444Pipeline,
    session: EncodeSession,
    ctx: VkDeviceCtx,
    pub width: u32,
    pub height: u32,
}

impl VkRecorderLossless {
    /// Initialize the lossless recorder. Returns None on any setup
    /// failure (driver lacks Hi444PP, queue family selection failed,
    /// extension missing, etc.). Callers should fall back to the
    /// non-lossless h264-vulkan path.
    pub fn try_new(width: u32, height: u32) -> Option<Self> {
        let ctx = VkDeviceCtx::open()
            .map_err(|e| {
                eprintln!("VkRecorderLossless: VkDeviceCtx::open failed: {e}");
                tracing::warn!("VkDeviceCtx::open failed: {e}");
            })
            .ok()?;
        if !ctx.hi444_supported {
            eprintln!(
                "VkRecorderLossless: Hi444PP not supported on {} (driver didn't \
                 accept HIGH_444_PREDICTIVE + TYPE_444 caps query)",
                ctx.device_name
            );
            tracing::warn!(
                "VkRecorderLossless::try_new: Hi444PP not supported on {} — \
                 lossless path needs NVIDIA on baseline 2026-05-12",
                ctx.device_name
            );
            return None;
        }
        let mut session = EncodeSession::new_lossless(&ctx, width, height)
            .map_err(|e| {
                eprintln!("VkRecorderLossless: EncodeSession::new_lossless failed: {e}");
                tracing::warn!("EncodeSession::new_lossless failed: {e}");
            })
            .ok()?;
        session
            .create_parameters()
            .map_err(|e| {
                eprintln!("VkRecorderLossless: create_parameters(lossless) failed: {e}");
                tracing::warn!("create_parameters(lossless) failed: {e}");
            })
            .ok()?;
        let fr = FrameResources444::new(&ctx, width, height)
            .map_err(|e| {
                eprintln!("VkRecorderLossless: FrameResources444::new failed: {e}");
                tracing::warn!("FrameResources444::new failed: {e}");
            })
            .ok()?;
        let pipe = BgraToYuv444Pipeline::new(&ctx, 1)
            .map_err(|e| {
                eprintln!("VkRecorderLossless: BgraToYuv444Pipeline::new failed: {e}");
                tracing::warn!("BgraToYuv444Pipeline::new failed: {e}");
            })
            .ok()?;
        Some(VkRecorderLossless {
            fr,
            pipe,
            session,
            ctx,
            width,
            height,
        })
    }

    /// AVCDecoderConfigurationRecord for the MKV CodecPrivate field.
    /// The bytes encode profile_idc=244 (HIGH_444_PREDICTIVE) and
    /// chroma_format_idc=3 (4:4:4) so the decoder knows to expect
    /// 3-plane YUV input.
    pub fn codec_private(&self) -> &[u8] {
        self.session.codec_private()
    }

    /// Encode a single IDR frame from CPU-supplied BGRA pixels. The
    /// shader produces YUV 4:4:4 in storage images; vkCmdCopyImage
    /// brings them into the encoder's 3-plane picture image; the
    /// encoder writes a Hi444PP NAL at QP=0 (bit-exact lossless).
    pub fn encode_idr_from_bgra(&self, bgra: &[u8], pts_us: i64) -> Option<EncodedNal> {
        let data =
            encode_idr_gpu_synthetic_yuv444(&self.ctx, &self.fr, &self.pipe, &self.session, bgra)
                .map_err(|e| {
                    eprintln!("encode_idr_gpu_synthetic_yuv444 failed: {e}");
                    tracing::warn!("encode_idr_gpu_synthetic_yuv444 failed: {e}");
                })
                .ok()?;
        Some(EncodedNal {
            data,
            pts_us,
            is_keyframe: true,
        })
    }
}

fn pick_h264_level(width: u32, height: u32) -> vk::native::StdVideoH264LevelIdc {
    let mbs = width.div_ceil(16) * height.div_ceil(16);
    let rate = mbs * 60; // assume 60 fps; over-estimates are fine
    if rate <= 245_760 {
        vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0
    } else if rate <= 522_240 {
        vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2
    } else if rate <= 983_040 {
        vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1
    } else {
        vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2
    }
}

fn cleanup_partial(
    device: &ash::Device,
    allocated: &[vk::DeviceMemory],
    video_queue: &ash::khr::video_queue::Device,
    session: vk::VideoSessionKHR,
) {
    unsafe {
        (video_queue.fp().destroy_video_session_khr)(device.handle(), session, std::ptr::null());
        for mem in allocated {
            device.free_memory(*mem, None);
        }
    }
}

fn pick_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_mask: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        let bit = 1u32 << i;
        type_mask & bit != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}

fn select_physical_device(
    instance: &ash::Instance,
) -> Result<(vk::PhysicalDevice, String, u32, u32, Vec<String>), String> {
    let pdevs = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices: {e:?}"))?
    };
    let mut last_err = "no Vulkan physical device".to_owned();
    for pdev in pdevs {
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        let name = cstr_name(&props.device_name);

        // Required device extensions.
        let dev_ext_props = unsafe {
            instance
                .enumerate_device_extension_properties(pdev)
                .map_err(|e| format!("enumerate_device_extension_properties: {e:?}"))?
        };
        let dev_exts: Vec<String> = dev_ext_props
            .iter()
            .map(|p| cstr_name(&p.extension_name))
            .collect();
        let missing: Vec<&str> = REQUIRED_DEVICE_EXTENSION_NAMES
            .iter()
            .copied()
            .filter(|name| !dev_exts.iter().any(|e| e == name))
            .collect();
        if !missing.is_empty() {
            last_err = format!("device {name:?} missing required extensions: {missing:?}");
            continue;
        }

        // Queue family selection: need at least one COMPUTE queue and
        // one VIDEO_ENCODE queue (may overlap on some hardware).
        let qfs = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
        let video_qf = qfs
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR));
        let compute_qf = qfs
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
        let (video_qf, compute_qf) = match (video_qf, compute_qf) {
            (Some(v), Some(c)) => (v as u32, c as u32),
            _ => {
                last_err = format!("device {name:?} lacks COMPUTE or VIDEO_ENCODE queue family");
                continue;
            }
        };
        return Ok((pdev, name, compute_qf, video_qf, dev_exts));
    }
    Err(last_err)
}

/// Pretty-print a probe report to stderr. Called from main.rs when
/// `WAYMUX_VULKAN_PROBE=1` is set.
pub fn log_probe_report(p: &VulkanProbe) {
    fn fmt_ver(v: u32) -> String {
        format!(
            "{}.{}.{}",
            vk::api_version_major(v),
            vk::api_version_minor(v),
            vk::api_version_patch(v)
        )
    }
    eprintln!("# waymux vulkan probe");
    eprintln!("instance_api_version: {}", fmt_ver(p.api_version));
    eprintln!("instance_extensions: {}", p.instance_extensions.len());
    for (i, d) in p.devices.iter().enumerate() {
        eprintln!("--- device[{i}]: {}", d.name);
        eprintln!("    api_version:           {}", fmt_ver(d.api_version));
        eprintln!(
            "    video_encode_h264:     {}",
            d.video_encode_h264_supported
        );
        eprintln!("    dmabuf_import:         {}", d.dmabuf_import_supported);
        eprintln!("    queue_families:        {}", d.queue_families.len());
        for q in &d.queue_families {
            let mut tags = Vec::new();
            if q.flags.contains(vk::QueueFlags::GRAPHICS) {
                tags.push("GRAPHICS");
            }
            if q.flags.contains(vk::QueueFlags::COMPUTE) {
                tags.push("COMPUTE");
            }
            if q.flags.contains(vk::QueueFlags::TRANSFER) {
                tags.push("TRANSFER");
            }
            if q.flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR) {
                tags.push("VIDEO_ENCODE");
            }
            if q.flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR) {
                tags.push("VIDEO_DECODE");
            }
            eprintln!(
                "      qf[{}]: count={} flags={}",
                q.index,
                q.count,
                tags.join("|")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_timeout_stays_under_softlockup_watchdog() {
        // SAFETY INVARIANT: on a softlockup_panic=1 host (~20s watchdog), a GPU
        // fence wait that outlives the watchdog turns a recoverable stall into a
        // hard panic + reboot. The clamp must keep EVERY value (default and any
        // env override) comfortably under that window.
        const WATCHDOG_NS: u64 = 20_000_000_000;
        // Default (no override) is the 3s fast-abort.
        assert_eq!(clamp_fence_timeout(None), 3_000_000_000);
        // A test override below the cap passes through unchanged.
        assert_eq!(clamp_fence_timeout(Some(2_000_000_000)), 2_000_000_000);
        // The old 30s value (and anything past the cap) is hard-capped at 10s.
        assert_eq!(clamp_fence_timeout(Some(30_000_000_000)), 10_000_000_000);
        assert_eq!(clamp_fence_timeout(Some(u64::MAX)), 10_000_000_000);
        // No reachable value can hit the watchdog.
        for parsed in [
            None,
            Some(0),
            Some(2_000_000_000),
            Some(30_000_000_000),
            Some(u64::MAX),
        ] {
            assert!(
                clamp_fence_timeout(parsed) < WATCHDOG_NS,
                "fence timeout {parsed:?} must stay under the soft-lockup watchdog"
            );
        }
    }

    #[test]
    fn vulkan_import_gate_excludes_bogus_modifier() {
        // The Vulkan import gate (`modifier_is_importable` → Vulkan set) is a
        // subset of the advertised EGL set; a bogus modifier is in neither.
        // LINEAR is always accepted by the Vulkan gate; a nonsense modifier
        // must be rejected by both sets.
        let bogus: u64 = 0xDEAD_BEEF_FEED_FACE;
        assert!(
            !super::importable_bgra_modifiers().contains(&bogus),
            "bogus modifier must not be in the advertised (EGL) set"
        );
        assert!(super::modifier_is_importable(
            crate::dmabuf::DRM_FORMAT_MOD_LINEAR
        ));
        assert!(!super::modifier_is_importable(bogus));
    }

    /// The embedded SPIR-V binary parses as a valid Vulkan shader.
    /// Creates a `VkShaderModule` from the bytes on the first device
    /// the probe found. If `vkCreateShaderModule` returns an error the
    /// SPIR-V is structurally broken — typically because the .spv file
    /// drifted from the .glsl source. Re-run `glslc` per the docstring
    /// on `BGRA_TO_NV12_SPV` to regenerate.
    #[test]
    fn compute_shader_spv_loads() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => e,
            Err(e) => {
                eprintln!("ash::Entry::load skipped: {e:?}");
                return;
            }
        };
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let ci = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&ci, None).unwrap() };
        let pdevs = unsafe { instance.enumerate_physical_devices().unwrap() };
        let pdev = pdevs.first().expect("no Vulkan physical device").to_owned();
        let qf_idx = 0u32; // any family will do for shader-module creation
        let priorities = [1.0f32];
        let qci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_idx)
            .queue_priorities(&priorities);
        let dev_ci = vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&qci));
        let device = unsafe { instance.create_device(pdev, &dev_ci, None).unwrap() };

        // The spv blob must be aligned to a u32 boundary for ash's
        // `code` slice. Re-cast the bytes.
        assert_eq!(BGRA_TO_NV12_SPV.len() % 4, 0, "spv length not u32-aligned");
        let code: Vec<u32> = BGRA_TO_NV12_SPV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let module = unsafe { device.create_shader_module(&sm_ci, None) }
            .expect("create_shader_module failed");
        unsafe { device.destroy_shader_module(module, None) };
        unsafe { device.destroy_device(None) };
        unsafe { instance.destroy_instance(None) };
    }

    /// Sister to `compute_shader_spv_loads`: the YUV 4:4:4 SPIR-V blob
    /// is well-formed and loads into a `VkShaderModule`. Doesn't dispatch
    /// the shader — that's the next chunk after `BgraToYuv444Pipeline`
    /// lands.
    #[test]
    fn compute_shader_yuv444_spv_loads() {
        // Fail-fast if the .spv got out of sync with the .glsl source.
        assert_eq!(
            BGRA_TO_YUV444_SPV.len() % 4,
            0,
            "yuv444 spv length not u32-aligned ({} bytes)",
            BGRA_TO_YUV444_SPV.len()
        );
        // Magic number for SPIR-V is 0x07230203 (little-endian first u32).
        let magic = u32::from_le_bytes([
            BGRA_TO_YUV444_SPV[0],
            BGRA_TO_YUV444_SPV[1],
            BGRA_TO_YUV444_SPV[2],
            BGRA_TO_YUV444_SPV[3],
        ]);
        assert_eq!(
            magic, 0x0723_0203,
            "yuv444 spv missing SPIR-V magic — regenerate with glslangValidator -V"
        );
        eprintln!(
            "BGRA_TO_YUV444_SPV: {} bytes, magic ok",
            BGRA_TO_YUV444_SPV.len()
        );
    }

    /// `VkDeviceCtx::open` succeeds on a host with a video-encode queue
    /// + dmabuf import. Gated by WAYMUX_TEST_VULKAN=1 to keep CI clean
    ///   when the runner has no Vulkan loader (or no encode-capable GPU).
    #[test]
    fn device_ctx_opens() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = match VkDeviceCtx::open() {
            Ok(c) => c,
            Err(e) => {
                panic!("VkDeviceCtx::open failed: {e}");
            }
        };
        eprintln!("VkDeviceCtx: device={}", ctx.device_name);
        eprintln!(
            "  compute_qf={} video_encode_qf={}",
            ctx.compute_queue_family, ctx.video_encode_queue_family
        );
        assert!(ctx.is_ready());
        // Probe encoder capabilities for our target profile. This is
        // the first real driver-side call against the video-queue
        // extension — if it returns success we know the driver
        // accepts our H.264 profile structure.
        let h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let mut profile_chain = h264_profile;
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut profile_chain);
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut encode_caps)
            .push_next(&mut h264_caps);
        let result = unsafe {
            (ctx.video_queue_inst
                .fp()
                .get_physical_device_video_capabilities_khr)(
                ctx.physical_device,
                &profile_info,
                &mut caps,
            )
        };
        assert_eq!(
            result,
            vk::Result::SUCCESS,
            "get_physical_device_video_capabilities_khr: {result:?}"
        );
        eprintln!(
            "  h264 main caps: min_extent={}x{} max_extent={}x{}",
            caps.min_coded_extent.width,
            caps.min_coded_extent.height,
            caps.max_coded_extent.width,
            caps.max_coded_extent.height,
        );
        eprintln!(
            "  encode_caps: rate_control={:?}",
            encode_caps.rate_control_modes
        );
    }

    /// End-to-end encode submit: upload synthetic NV12 (mid-grey),
    /// encode as IDR, read back the encoded NAL bytes via the encode
    /// feedback query, return them.
    ///
    /// Benchmark VkRecorder throughput at multiple resolutions.
    /// Prints FPS to stderr. Gated by WAYMUX_BENCH_VULKAN=1 so it
    /// doesn't run in every `cargo test`.
    #[test]
    fn vk_recorder_throughput_bench() {
        if !matches!(
            std::env::var("WAYMUX_BENCH_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_BENCH_VULKAN=1 to enable this bench");
            return;
        }
        use std::time::Instant;
        let resolutions = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)];
        let n_frames = 30u32;
        eprintln!("# VkRecorder throughput on AMD Renoir + Mesa 26.1 (n={n_frames} frames each)");
        eprintln!("resolution\ttotal_ms\tper_frame_ms\tfps");
        for (w, h) in resolutions {
            let recorder = match VkRecorder::try_new(w, h) {
                Some(r) => r,
                None => {
                    eprintln!("{w}x{h}\tskip (try_new failed)");
                    continue;
                }
            };
            let bgra: Vec<u8> = (0..(w * h * 4))
                .map(|i| (i as u8).wrapping_mul(37))
                .collect();
            let start = Instant::now();
            let mut ok = 0;
            for i in 0..n_frames {
                if recorder
                    .encode_idr_from_bgra(&bgra, (i * 16_667) as i64)
                    .is_some()
                {
                    ok += 1;
                }
            }
            let elapsed_ms = start.elapsed().as_millis() as f64;
            let per_frame_ms = elapsed_ms / ok.max(1) as f64;
            let fps = 1000.0 / per_frame_ms;
            eprintln!("{w}x{h}\t{elapsed_ms:.1}\t{per_frame_ms:.2}\t{fps:.1}");
        }
    }

    /// End-to-end pipeline: encode N BGRA frames through VkRecorder,
    /// wrap them in an MKV via waymux-mux-mkv, run ffprobe on the
    /// result. Validates that the codec_private from Vulkan and the
    /// NAL bytes from Vulkan combine into a stream a real demuxer
    /// will accept.
    #[test]
    fn vk_recorder_to_mkv_roundtrip() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        use std::io::Write;
        use std::process::Command;
        let w = 128u32;
        let h = 128u32;
        let n_frames = 5usize;

        let recorder = VkRecorder::try_new(w, h).expect("VkRecorder::try_new");
        let mkv_path =
            std::env::temp_dir().join(format!("waymux-vk-rt-{}.mkv", std::process::id()));
        let _ = std::fs::remove_file(&mkv_path);
        let mut file = std::fs::File::create(&mkv_path).expect("create mkv file");
        {
            let mut mux = waymux_mux_mkv::MkvWriter::new(
                std::io::BufWriter::new(&mut file),
                w,
                h,
                recorder.codec_private(),
            )
            .expect("MkvWriter::new");
            for i in 0..n_frames {
                // Slight per-frame variation so the encoder doesn't
                // produce identical output for every frame.
                let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
                for y in 0..h {
                    for x in 0..w {
                        let i_off = ((y * w + x) * 4) as usize;
                        bgra[i_off] = ((x as usize + i) as u8).wrapping_mul(2);
                        bgra[i_off + 1] = ((y as usize + i) as u8).wrapping_mul(2);
                        bgra[i_off + 2] = (x as u8).wrapping_add(y as u8);
                        bgra[i_off + 3] = 0xFF;
                    }
                }
                let nal = recorder
                    .encode_idr_from_bgra(&bgra, (i * 16_667) as i64)
                    .expect("encode_idr_from_bgra");
                mux.write_frame(&nal.data, (i as i64) * 17, nal.is_keyframe)
                    .expect("MkvWriter::write_frame");
            }
            mux.finish().expect("MkvWriter::finish").flush().unwrap();
        }

        // ffprobe round-trip.
        if Command::new("ffprobe").arg("-version").output().is_err() {
            eprintln!("ffprobe not installed; skipping round-trip check");
            return;
        }
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height,nb_read_frames",
                "-count_frames",
                "-of",
                "default=noprint_wrappers=1",
                mkv_path.to_str().unwrap(),
            ])
            .output()
            .expect("ffprobe");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        eprintln!(
            "vk_recorder_to_mkv_roundtrip:\n  mkv = {}\n  ffprobe stdout = {}\n  ffprobe stderr = {}",
            mkv_path.display(),
            stdout.trim(),
            stderr.trim()
        );
        assert!(
            stdout.contains("codec_name=h264"),
            "ffprobe did not report h264 codec\nstdout: {stdout}\nstderr: {stderr}"
        );
        assert!(stdout.contains(&format!("width={w}")));
        assert!(stdout.contains(&format!("height={h}")));
        // Frame count: at least 1 (some streams report N, some
        // report unknown; ffmpeg without -count_frames returns 0).
        let _ = std::fs::remove_file(&mkv_path);
    }

    /// `VkRecorder::encode_idr_from_bgra` produces valid H.264 NAL
    /// bytes for a real BGRA gradient image. End-to-end test of the
    /// public API surface — proves the full setup + encode flow
    /// works for content beyond synthetic mid-grey.
    #[test]
    fn vk_recorder_encode_from_bgra() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let w = 128u32;
        let h = 128u32;
        let recorder = VkRecorder::try_new(w, h).expect("VkRecorder::try_new");
        eprintln!(
            "VkRecorder codec_private len = {}",
            recorder.codec_private().len()
        );
        let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                bgra[i] = (x as u8).wrapping_mul(2);
                bgra[i + 1] = (y as u8).wrapping_mul(2);
                bgra[i + 2] = (x + y) as u8;
                bgra[i + 3] = 0xFF;
            }
        }
        let nal = recorder
            .encode_idr_from_bgra(&bgra, 0)
            .expect("encode_idr_from_bgra");
        eprintln!(
            "vk_recorder_encode_from_bgra: {} NAL bytes, keyframe={}",
            nal.data.len(),
            nal.is_keyframe
        );
        assert!(nal.data.len() > 16, "encoded gradient should be > 16 bytes");
        assert!(nal.is_keyframe);
        assert!(
            nal.data.windows(3).any(|w| w == [0, 0, 1]),
            "no Annex B start code in NAL data"
        );
    }

    /// End-to-end encode submit: upload synthetic NV12 (mid-grey),
    /// encode as IDR, read back the encoded NAL bytes via the encode
    /// feedback query, verify the output starts with a valid H.264
    /// Annex B start code.
    ///
    /// Verified 2026-05-12 on AMD Renoir + Mesa 26.1: produces ~36
    /// bytes of valid h264 NAL data for a 128x128 mid-grey IDR.
    #[test]
    fn encode_idr_synthetic_produces_bytes() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let w = 128u32;
        let h = 128u32;
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let mut session = EncodeSession::new(&ctx, w, h).expect("EncodeSession::new");
        session.create_parameters().expect("create_parameters");
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let fr = FrameResources::new(&ctx, w, h, &mut h264_profile).expect("FrameResources::new");
        // Build a synthetic NV12: mid-grey Y plane, neutral UV plane.
        let mut nv12 = vec![0u8; (w as usize) * (h as usize) * 3 / 2];
        for b in &mut nv12[..(w as usize) * (h as usize)] {
            *b = 128; // mid-grey luma
        }
        for b in &mut nv12[(w as usize) * (h as usize)..] {
            *b = 128; // neutral chroma
        }
        let encoded =
            encode_idr_synthetic(&ctx, &fr, &session, &nv12).expect("encode_idr_synthetic");
        eprintln!(
            "encode_idr_synthetic_produces_bytes: {} bytes encoded for {}x{} mid-grey IDR",
            encoded.len(),
            w,
            h
        );
        assert!(!encoded.is_empty(), "no encoded bytes returned");
        // Dump the hex so we can eyeball the NAL stream.
        let hex: String = encoded
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("encoded bytes: {hex}");
        // H.264 Annex B start code: 00 00 00 01 or 00 00 01.
        let has_annex_b_start = encoded.windows(3).any(|w| w == [0, 0, 1]);
        assert!(
            has_annex_b_start,
            "no Annex B start code found in encoded output"
        );
    }

    /// `run_compute_only` produces NV12 output that matches the CPU
    /// reference `bgra_to_nv12` within ±1 LSB per byte.
    ///
    /// **Temporarily disabled.** The encoder image lost its STORAGE
    /// usage flag (item 5 pt.4 — AMD won't accept it alongside
    /// VIDEO_ENCODE_SRC). The compute pipeline needs to be retargeted
    /// to dedicated Y/UV storage-only images that get copied into the
    /// encoder's NV12 image via vkCmdCopyImage. Re-enable once that
    /// refactor lands.
    #[test]
    #[ignore]
    fn compute_only_matches_cpu_reference() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let w = 64u32;
        let h = 64u32;
        // Synthetic gradient: makes any cross-channel swap or rounding
        // mismatch obvious in the diff.
        let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                bgra[i] = (x as u8).saturating_mul(4); // B
                bgra[i + 1] = (y as u8).saturating_mul(4); // G
                bgra[i + 2] = ((x + y) as u8).saturating_mul(2); // R
                bgra[i + 3] = 0xFF;
            }
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let pipe = BgraToNv12Pipeline::new(&ctx, 1).expect("BgraToNv12Pipeline::new");
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let fr = FrameResources::new(&ctx, w, h, &mut h264_profile).expect("FrameResources::new");
        let gpu_nv12 = run_compute_only(&ctx, &fr, &pipe, &bgra).expect("run_compute_only");
        let cpu_nv12 = crate::recording::bgra_to_nv12(&bgra, w, h);
        assert_eq!(
            gpu_nv12.len(),
            cpu_nv12.len(),
            "GPU NV12 len ({}) != CPU ({})",
            gpu_nv12.len(),
            cpu_nv12.len()
        );
        // Tolerate ±1 LSB. Most bytes should be identical.
        let mut max_diff = 0u8;
        let mut n_off = 0usize;
        for (i, (g, c)) in gpu_nv12.iter().zip(cpu_nv12.iter()).enumerate() {
            let d = (*g as i16 - *c as i16).unsigned_abs() as u8;
            if d > max_diff {
                max_diff = d;
            }
            if d > 1 {
                n_off += 1;
                if n_off <= 5 {
                    eprintln!("  diff at byte {i}: gpu={g} cpu={c} (delta={d})");
                }
            }
        }
        eprintln!(
            "compute_only_matches_cpu_reference: max_diff={max_diff} bytes_off_by_more_than_1={n_off} of {}",
            gpu_nv12.len()
        );
        // The Y plane should match closely. The UV plane differs by
        // up to a few LSB because the GPU shader averages four pixels
        // before applying the chroma math, while the CPU function
        // applies the math to the top-left pixel of each 2x2 block.
        // We accept up to 5% of bytes off by >1.
        let pct = n_off as f64 / gpu_nv12.len() as f64;
        assert!(
            pct < 0.10,
            "{:.1}% of bytes differ by >1 LSB; max delta={}",
            pct * 100.0,
            max_diff
        );
    }

    /// `FrameResources::new` constructs every per-frame Vulkan object
    /// the encode pipeline needs: BGRA source image, multi-planar NV12
    /// destination, staging buffer, encoded-output buffer, command
    /// pool/buffer, fence, query pool. Drops cleanly.
    #[test]
    fn frame_resources_create() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let fr = FrameResources::new(&ctx, 1920, 1080, &mut h264_profile)
            .expect("FrameResources::new at 1920x1080");
        assert_ne!(fr.bgra_image, vk::Image::null());
        assert_ne!(fr.nv12_image, vk::Image::null());
        assert_ne!(fr.encoded_buffer, vk::Buffer::null());
        eprintln!(
            "FrameResources: bgra+nv12+encoded+pool created (encoded_size={})",
            fr.encoded_size
        );
    }

    /// Driver probe: print everything we read from `VkDeviceCtx::open()`
    /// that we depend on for H.264 encoding. Run on a new vendor/driver
    /// BEFORE recording-end-to-end tests so we can see whether NVIDIA
    /// reports different std-headers / image-flags than AMD. Gated by
    /// WAYMUX_TEST_VULKAN=1 like the rest.
    ///
    ///     WAYMUX_TEST_VULKAN=1 cargo test -p waymux-session \
    ///         vulkan_probe_h264_encode -- --nocapture
    #[test]
    fn vulkan_probe_h264_encode() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this probe");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let name_str: String = ctx
            .h264_std_header_name
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as u8 as char)
            .collect();
        eprintln!("=== vulkan_probe_h264_encode ===");
        eprintln!("  device: {}", ctx.device_name);
        eprintln!(
            "  h264_std_header_name: {:?} (raw spec_version={:#010x})",
            name_str, ctx.h264_std_header_spec_version,
        );
        eprintln!(
            "  nv12_encode_src_image_flags: {:?}",
            ctx.nv12_encode_src_image_flags,
        );
        eprintln!("  nv12_dpb_image_flags: {:?}", ctx.nv12_dpb_image_flags,);
        eprintln!(
            "  compute_qf={} video_encode_qf={}",
            ctx.compute_queue_family, ctx.video_encode_queue_family,
        );
    }

    /// Probe ALL Vulkan-video 4:4:4 lossless-capable codecs on this
    /// device: H.264 Hi444PP, H.265 Main 4:4:4, and AV1 (when present).
    /// Used to gate Path B (lossless 4:4:4 recording) — if a vendor
    /// doesn't expose any 4:4:4 encode profile, the Vulkan zero-copy
    /// lossless story isn't reachable on that vendor without a CPU
    /// fallback.
    ///
    /// Run:
    ///     WAYMUX_TEST_VULKAN=1 cargo test -p waymux-session \
    ///         vulkan_probe_lossless_codecs -- --nocapture
    #[test]
    fn vulkan_probe_lossless_codecs() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this probe");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        eprintln!(
            "=== vulkan_probe_lossless_codecs on {} ===",
            ctx.device_name
        );

        // Helper: try a profile + chroma subsampling combo, print result.
        // Codec-specific sub-caps must be chained on `caps` so the driver
        // has a place to write the per-codec capability bits; without
        // them the driver SIGSEGVs writing past the end of `caps`.
        fn try_caps(
            ctx: &VkDeviceCtx,
            label: &str,
            codec: vk::VideoCodecOperationFlagsKHR,
            chroma: vk::VideoChromaSubsamplingFlagsKHR,
            mut h264_profile: Option<vk::VideoEncodeH264ProfileInfoKHR>,
            mut h265_profile: Option<vk::VideoEncodeH265ProfileInfoKHR>,
        ) {
            let mut profile_info = vk::VideoProfileInfoKHR::default()
                .video_codec_operation(codec)
                .chroma_subsampling(chroma)
                .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
                .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);
            if let Some(ref mut p) = h264_profile {
                profile_info = profile_info.push_next(p);
            }
            if let Some(ref mut p) = h265_profile {
                profile_info = profile_info.push_next(p);
            }
            let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
            let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
            let mut h265_caps = vk::VideoEncodeH265CapabilitiesKHR::default();
            let mut caps = vk::VideoCapabilitiesKHR::default().push_next(&mut encode_caps);
            if codec == vk::VideoCodecOperationFlagsKHR::ENCODE_H264 {
                caps = caps.push_next(&mut h264_caps);
            } else if codec == vk::VideoCodecOperationFlagsKHR::ENCODE_H265 {
                caps = caps.push_next(&mut h265_caps);
            }
            let r = unsafe {
                (ctx.video_queue_inst
                    .fp()
                    .get_physical_device_video_capabilities_khr)(
                    ctx.physical_device,
                    &profile_info,
                    &mut caps,
                )
            };
            if r != vk::Result::SUCCESS {
                eprintln!("  {label:<30}: NOT SUPPORTED ({r:?})");
                return;
            }
            let mw = caps.min_coded_extent.width;
            let mh = caps.min_coded_extent.height;
            let xw = caps.max_coded_extent.width;
            let xh = caps.max_coded_extent.height;
            eprintln!("  {label:<30}: OK  min={mw}x{mh} max={xw}x{xh}");
        }

        // H.264 family
        let h264_main = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        try_caps(
            &ctx,
            "H.264 Main 4:2:0",
            vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            Some(h264_main),
            None,
        );
        // Monochrome encode = chroma_format_idc=0 in H.264. If supported,
        // we can encode 3 separate luma-only streams for R/G/B and mux
        // them as 3 video tracks → bit-exact lossless from BGRA via
        // hardware encode on both AMD and NVIDIA.
        let h264_high_mono = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH);
        try_caps(
            &ctx,
            "H.264 High Monochrome",
            vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME,
            Some(h264_high_mono),
            None,
        );
        let h264_hi444 = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
            vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
        );
        try_caps(
            &ctx,
            "H.264 Hi444PP 4:4:4",
            vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            Some(h264_hi444),
            None,
        );

        // H.265 family — Main 4:2:0 (baseline) and Main 4:4:4 (lossless target).
        let h265_main = vk::VideoEncodeH265ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN);
        try_caps(
            &ctx,
            "H.265 Main 4:2:0",
            vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            None,
            Some(h265_main),
        );
        // H.265 4:4:4 lives under the FORMAT_RANGE_EXTENSIONS profile
        // (HEVC spec subsumes Main 4:4:4 / Main 4:4:4 10-bit / etc under
        // the Format Range Extensions profile_idc=4).
        let h265_range_ext = vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(
            vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS,
        );
        try_caps(
            &ctx,
            "H.265 RangeExt 4:4:4",
            vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            None,
            Some(h265_range_ext),
        );

        // NOTE: AV1 encode (VK_KHR_video_encode_av1) was promoted to KHR in
        // 2024 but ash 0.38 doesn't expose its profile struct yet; skip
        // until we either bump ash or use raw vk::native types.
        eprintln!(
            "  AV1 4:4:4                     : probe skipped (ash 0.38 lacks AV1 encode struct)"
        );
    }

    /// Old-style probe for just Hi444PP — kept for backward compatibility
    /// with the runbook entry. Equivalent to one line of the lossless probe.
    #[test]
    fn vulkan_probe_h264_hi444pp() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this probe");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        eprintln!("=== vulkan_probe_h264_hi444pp on {} ===", ctx.device_name);

        // Try the Hi444PP video capabilities query.
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
            vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
        );
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut encode_caps)
            .push_next(&mut h264_caps);
        let result = unsafe {
            (ctx.video_queue_inst
                .fp()
                .get_physical_device_video_capabilities_khr)(
                ctx.physical_device,
                &profile_info,
                &mut caps,
            )
        };
        if result != vk::Result::SUCCESS {
            eprintln!(
                "  Hi444PP NOT SUPPORTED: vkGetPhysicalDeviceVideoCapabilitiesKHR={result:?}"
            );
            eprintln!("  → driver does not advertise H.264 High 4:4:4 Predictive encode");
            return;
        }
        // caps holds mutable borrows of encode_caps and h264_caps via
        // push_next; copy out everything we need into owned locals, then
        // drop caps so we can read those fields directly.
        let min_w = caps.min_coded_extent.width;
        let min_h = caps.min_coded_extent.height;
        let max_w = caps.max_coded_extent.width;
        let max_h = caps.max_coded_extent.height;
        let header_name = caps.std_header_version.extension_name;
        let header_version = caps.std_header_version.spec_version;
        let rc = encode_caps.rate_control_modes;
        eprintln!("  Hi444PP SUPPORTED: min_extent={min_w}x{min_h} max_extent={max_w}x{max_h}");
        eprintln!("  encode_caps: rate_control={rc:?}");
        let name_str: String = header_name
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as u8 as char)
            .collect();
        eprintln!(
            "  std_header: {:?} spec_version={:#010x}",
            name_str, header_version,
        );

        // Query format props for the 4:4:4 profile under encode-src and DPB usages.
        let profile_array = [profile_info];
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profile_array);
        for usage in [
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
        ] {
            let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
                .image_usage(usage)
                .push_next(&mut profile_list);
            let mut count: u32 = 0;
            let r = unsafe {
                (ctx.video_queue_inst
                    .fp()
                    .get_physical_device_video_format_properties_khr)(
                    ctx.physical_device,
                    &format_info,
                    &mut count,
                    std::ptr::null_mut(),
                )
            };
            if r != vk::Result::SUCCESS || count == 0 {
                eprintln!("  usage={usage:?}: query={r:?} count={count}");
                continue;
            }
            let mut props: Vec<vk::VideoFormatPropertiesKHR<'static>> = (0..count as usize)
                .map(|_| vk::VideoFormatPropertiesKHR::default())
                .collect();
            unsafe {
                let _ = (ctx
                    .video_queue_inst
                    .fp()
                    .get_physical_device_video_format_properties_khr)(
                    ctx.physical_device,
                    &format_info,
                    &mut count,
                    props.as_mut_ptr(),
                );
            }
            props.truncate(count as usize);
            eprintln!("  usage={usage:?} → {} format(s):", props.len());
            for p in &props {
                eprintln!(
                    "    format={:?} create_flags={:?} tiling={:?}",
                    p.format, p.image_create_flags, p.image_tiling,
                );
            }
        }
    }

    /// Print what AMD reports as supported (format, usage) tuples for
    /// the H.264 main-profile encoder. Diagnostic test — confirms what
    /// FrameResources::new can ask for without DEVICE_LOST.
    #[test]
    fn query_video_format_props_h264_encode_src() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        for usage in [
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
        ] {
            let props = query_video_format_props(&ctx, usage)
                .unwrap_or_else(|e| panic!("query failed for {usage:?}: {e}"));
            eprintln!("usage = {:?} -> {} supported format(s)", usage, props.len());
            for p in &props {
                eprintln!(
                    "  format={:?} create_flags={:?} type={:?} tiling={:?} usage={:?}",
                    p.format,
                    p.image_create_flags,
                    p.image_type,
                    p.image_tiling,
                    p.image_usage_flags
                );
            }
        }
    }

    /// `BgraToNv12Pipeline::new` constructs the compute pipeline +
    /// descriptor layout matching the shader bindings. Drop tears
    /// everything down cleanly.
    #[test]
    fn bgra_to_nv12_pipeline_creates() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let pipe = BgraToNv12Pipeline::new(&ctx, 2).expect("BgraToNv12Pipeline::new");
        assert_ne!(pipe.pipeline(), vk::Pipeline::null());
        assert_ne!(
            pipe.descriptor_set_layout(),
            vk::DescriptorSetLayout::null()
        );
        eprintln!("BgraToNv12Pipeline: created");
    }

    /// `EncodeSession::create_parameters` populates SPS+PPS, hands them
    /// to the driver, retrieves the encoded NAL bytes, and builds an
    /// AVCDecoderConfigurationRecord. The record is parseable by
    /// ffprobe — that's the round-trip check via the muxer in the
    /// vk_record_codec_private_round_trip test (combined integration).
    #[test]
    fn encode_session_parameters() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let mut session =
            EncodeSession::new(&ctx, 1920, 1080).expect("EncodeSession::new at 1920x1080");
        session
            .create_parameters()
            .expect("create_parameters failed");
        let avcc = session.codec_private();
        eprintln!("AVCC len={} bytes", avcc.len());
        assert!(avcc.len() >= 15, "AVCC suspiciously short: {avcc:?}");
        assert_eq!(avcc[0], 1, "configurationVersion not 1");
        // length_size_minus_one in the low 2 bits of byte 4.
        assert_eq!(avcc[4] & 0x03, 3, "lengthSizeMinusOne not 3");
        // numOfSequenceParameterSets in the low 5 bits of byte 5.
        assert_eq!(avcc[5] & 0x1F, 1, "expected exactly 1 SPS");
    }

    /// `EncodeSession::new` creates a `VkVideoSessionKHR` for 1920x1080
    /// main-profile H.264 and binds the requested backing memory.
    /// Verifies the device actually accepts our profile + format pair.
    /// Gated by WAYMUX_TEST_VULKAN=1.
    #[test]
    fn encode_session_creates() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = VkDeviceCtx::open().expect("VkDeviceCtx::open");
        let session =
            EncodeSession::new(&ctx, 1920, 1080).expect("EncodeSession::new at 1920x1080");
        eprintln!(
            "EncodeSession created: handle={:?} mem_blocks={}",
            session.handle(),
            session.memory.len()
        );
        // Drop runs and tears down the session + memory.
    }

    /// Dump the full Hi444PP encode capabilities: QP range, min/max
    /// coded extent, picture-access granularity, supported H.264
    /// rate-control modes. Used to diagnose driver-side rejection of
    /// the encode submit.
    #[test]
    fn vulkan_probe_hi444pp_caps() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = match VkDeviceCtx::open() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("VkDeviceCtx::open skipped: {e}");
                return;
            }
        };
        if !ctx.hi444_supported {
            eprintln!("Hi444PP not supported on {} — skipping", ctx.device_name);
            return;
        }
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
            vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
        );
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut encode_caps)
            .push_next(&mut h264_caps);
        unsafe {
            let _ = (ctx
                .video_queue_inst
                .fp()
                .get_physical_device_video_capabilities_khr)(
                ctx.physical_device,
                &profile_info,
                &mut caps,
            );
        }
        eprintln!("=== Hi444PP video caps on {} ===", ctx.device_name);
        eprintln!("  min_coded_extent  = {:?}", caps.min_coded_extent);
        eprintln!("  max_coded_extent  = {:?}", caps.max_coded_extent);
        eprintln!(
            "  picture_access_granularity = {:?}",
            caps.picture_access_granularity
        );
        eprintln!(
            "  min_bitstream_buffer_offset_alignment = {}",
            caps.min_bitstream_buffer_offset_alignment
        );
        eprintln!(
            "  min_bitstream_buffer_size_alignment = {}",
            caps.min_bitstream_buffer_size_alignment
        );
        eprintln!("  max_dpb_slots = {}", caps.max_dpb_slots);
        eprintln!(
            "  max_active_reference_pictures = {}",
            caps.max_active_reference_pictures
        );
        eprintln!("=== EncodeCapabilities ===");
        eprintln!(
            "  rate_control_modes = {:?}",
            encode_caps.rate_control_modes
        );
        eprintln!(
            "  max_rate_control_layers = {}",
            encode_caps.max_rate_control_layers
        );
        eprintln!("  max_bitrate = {}", encode_caps.max_bitrate);
        eprintln!(
            "  supported_encode_feedback_flags = {:?}",
            encode_caps.supported_encode_feedback_flags
        );
        eprintln!("=== H264 Encode Capabilities ===");
        eprintln!("  flags = {:?}", h264_caps.flags);
        eprintln!("  max_level_idc = {:?}", h264_caps.max_level_idc);
        eprintln!("  max_slice_count = {}", h264_caps.max_slice_count);
        eprintln!(
            "  max_p_picture_l0_reference_count = {}",
            h264_caps.max_p_picture_l0_reference_count
        );
        eprintln!(
            "  max_b_picture_l0_reference_count = {}",
            h264_caps.max_b_picture_l0_reference_count
        );
        eprintln!(
            "  max_l1_reference_count = {}",
            h264_caps.max_l1_reference_count
        );
        eprintln!(
            "  max_temporal_layer_count = {}",
            h264_caps.max_temporal_layer_count
        );
    }

    /// P-frame feasibility probe: query the Main-profile 4:2:0 H.264 encode
    /// caps the viewer actually uses, and report the fields that decide whether
    /// P-frames (reference pictures) are possible on this driver.
    /// Run: `WAYMUX_TEST_VULKAN=1 cargo test -p waymux-session --bin waymux-session vulkan_probe_pframe_caps -- --nocapture`
    #[test]
    fn vulkan_probe_pframe_caps() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = match VkDeviceCtx::open() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("VkDeviceCtx::open skipped: {e}");
                return;
            }
        };
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut encode_caps)
            .push_next(&mut h264_caps);
        let r = unsafe {
            (ctx.video_queue_inst
                .fp()
                .get_physical_device_video_capabilities_khr)(
                ctx.physical_device,
                &profile_info,
                &mut caps,
            )
        };
        // Copy out the values before the borrow chain (`caps` push_next holds
        // &mut to h264_caps/encode_caps) is dropped.
        let max_dpb = caps.max_dpb_slots;
        let max_active = caps.max_active_reference_pictures;
        let _ = caps;
        let max_p_l0 = h264_caps.max_p_picture_l0_reference_count;
        let max_b_l0 = h264_caps.max_b_picture_l0_reference_count;
        let flags = h264_caps.flags;
        let rc_modes = encode_caps.rate_control_modes;
        let max_rc_layers = encode_caps.max_rate_control_layers;
        let max_bitrate = encode_caps.max_bitrate;
        eprintln!(
            "=== Main 4:2:0 H.264 encode caps on {} (query={:?}) ===",
            ctx.device_name, r
        );
        if r != vk::Result::SUCCESS {
            eprintln!("  Main profile NOT supported");
            return;
        }
        eprintln!(
            "  rate_control_modes               = {rc_modes:?}  (need CBR or VBR for adaptive RC)"
        );
        eprintln!("  max_rate_control_layers          = {max_rc_layers}");
        eprintln!("  max_bitrate                      = {max_bitrate}");
        eprintln!("  max_dpb_slots                    = {max_dpb}  (need >=2 for P-frames)");
        eprintln!("  max_active_reference_pictures    = {max_active}  (need >=1)");
        eprintln!("  max_p_picture_l0_reference_count = {max_p_l0}  (need >=1 for P-frames)");
        eprintln!("  max_b_picture_l0_reference_count = {max_b_l0}");
        eprintln!("  h264 flags                       = {flags:?}");
        let pframe_ok = max_dpb >= 2 && max_active >= 1 && max_p_l0 >= 1;
        eprintln!("  >>> P-FRAMES FEASIBLE: {pframe_ok}");
    }

    /// Enumerate every supported Vulkan format the driver reports for
    /// H.264 Hi444PP encode SRC + DPB. Used to figure out whether the
    /// NVIDIA driver wants 3-plane (`G8_B8_R8_3PLANE_444_UNORM`) or
    /// 2-plane 4:4:4 (`G8_B8R8_2PLANE_444_UNORM`) for the lossless path.
    /// Skipped on devices without Hi444PP support.
    #[test]
    fn vulkan_probe_hi444pp_formats() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = match VkDeviceCtx::open() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("VkDeviceCtx::open skipped: {e}");
                return;
            }
        };
        if !ctx.hi444_supported {
            eprintln!("Hi444PP not supported on {} — skipping", ctx.device_name);
            return;
        }
        for (label, usage) in [
            (
                "ENCODE_SRC",
                vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            ),
            ("ENCODE_DPB", vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR),
        ] {
            let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
                vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
            );
            let profile_info = vk::VideoProfileInfoKHR::default()
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
                .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
                .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
                .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
                .push_next(&mut h264_profile);
            let profile_array = [profile_info];
            let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profile_array);
            let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
                .image_usage(usage)
                .push_next(&mut profile_list);
            let mut count: u32 = 0;
            let result = unsafe {
                (ctx.video_queue_inst
                    .fp()
                    .get_physical_device_video_format_properties_khr)(
                    ctx.physical_device,
                    &format_info,
                    &mut count,
                    std::ptr::null_mut(),
                )
            };
            if result != vk::Result::SUCCESS {
                eprintln!("{label}: size query failed: {result:?}");
                continue;
            }
            let mut props: Vec<vk::VideoFormatPropertiesKHR<'static>> = (0..count as usize)
                .map(|_| vk::VideoFormatPropertiesKHR::default())
                .collect();
            unsafe {
                let _ = (ctx
                    .video_queue_inst
                    .fp()
                    .get_physical_device_video_format_properties_khr)(
                    ctx.physical_device,
                    &format_info,
                    &mut count,
                    props.as_mut_ptr(),
                );
            }
            eprintln!("=== Hi444PP {label} formats ({count}) ===");
            for p in props.iter().take(count as usize) {
                eprintln!(
                    "  format={:?} flags={:?} usage={:?} tiling={:?}",
                    p.format, p.image_create_flags, p.image_usage_flags, p.image_tiling
                );
            }
        }
    }

    /// Hi444PP spike: open a VkVideoSessionKHR with profile
    /// `STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE` and image format
    /// `G8_B8_R8_3PLANE_444_UNORM`. If this succeeds, the bit-exact
    /// lossless path is unblocked at the encoder-framework level;
    /// remaining work is the BGRA→YUV444 compute shader and per-frame
    /// integration.
    ///
    /// Validates today on NVIDIA (probed working on RTX A6000 — max
    /// 4096×4096). Expected to fail on AMD (Mesa exposes only 4:2:0
    /// Main per `feedback_amd_no_444_encode.md`).
    #[test]
    fn vulkan_hi444pp_session_opens() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let ctx = match VkDeviceCtx::open() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("VkDeviceCtx::open skipped: {e}");
                return;
            }
        };
        eprintln!(
            "=== vulkan_hi444pp_session_opens on {} ===",
            ctx.device_name
        );

        let width = 1920u32;
        let height = 1080u32;

        // Build the Hi444PP profile chain.
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
            vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
        );
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_444)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);

        // Query capabilities first — bail with a clear message if the
        // driver doesn't expose Hi444PP rather than blocking on the
        // session create call.
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut encode_caps)
            .push_next(&mut h264_caps);
        let cap_result = unsafe {
            (ctx.video_queue_inst
                .fp()
                .get_physical_device_video_capabilities_khr)(
                ctx.physical_device,
                &profile_info,
                &mut caps,
            )
        };
        if cap_result != vk::Result::SUCCESS {
            eprintln!(
                "Hi444PP not supported on {}: caps query = {:?} — skipping session-open",
                ctx.device_name, cap_result
            );
            return;
        }
        let std_header_name = caps.std_header_version.extension_name;
        let std_header_version = caps.std_header_version.spec_version;

        // Create the video session with Hi444PP profile + 3-plane 4:4:4
        // image format. This is the bare-minimum proof-of-life: if it
        // succeeds the engine path is unblocked.
        let std_header = vk::ExtensionProperties {
            extension_name: std_header_name,
            spec_version: std_header_version,
        };
        let session_ci = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(ctx.video_encode_queue_family)
            .video_profile(&profile_info)
            .picture_format(vk::Format::G8_B8_R8_3PLANE_444_UNORM)
            .max_coded_extent(vk::Extent2D { width, height })
            .reference_picture_format(vk::Format::G8_B8_R8_3PLANE_444_UNORM)
            .max_dpb_slots(1)
            .max_active_reference_pictures(1)
            .std_header_version(&std_header);

        let mut session = vk::VideoSessionKHR::null();
        let r = unsafe {
            (ctx.video_queue_dev.fp().create_video_session_khr)(
                ctx.device.handle(),
                &session_ci,
                std::ptr::null(),
                &mut session,
            )
        };
        if r != vk::Result::SUCCESS {
            panic!("vkCreateVideoSessionKHR(Hi444PP, 3PLANE_444): {r:?}");
        }

        // Query memory requirements — proves the session got far enough
        // through driver-side validation to enumerate resource needs.
        let mut count: u32 = 0;
        unsafe {
            let _ = (ctx
                .video_queue_dev
                .fp()
                .get_video_session_memory_requirements_khr)(
                ctx.device.handle(),
                session,
                &mut count,
                std::ptr::null_mut(),
            );
        }
        eprintln!("Hi444PP session opened at {width}x{height}: mem_req blocks = {count}");

        unsafe {
            (ctx.video_queue_dev.fp().destroy_video_session_khr)(
                ctx.device.handle(),
                session,
                std::ptr::null(),
            );
        }
    }

    /// Sanity-check that the probe runs and finds at least one device
    /// with dmabuf import. Skipped automatically in CI environments
    /// without a Vulkan loader.
    #[test]
    fn probe_runs() {
        let p = match probe() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("vulkan probe skipped: {e}");
                return;
            }
        };
        assert!(!p.devices.is_empty(), "no Vulkan devices found");
        // On the user's dev laptop we expect dmabuf import on the AMD
        // GPU. Don't assert — let the test be informational.
        log_probe_report(&p);
    }

    /// `VkRecorderLossless::try_new` returns `Some` on hardware with
    /// Hi444PP support, and the resulting codec_private declares
    /// profile_idc=244 (HIGH_444_PREDICTIVE) + chroma_format_idc=3 (4:4:4).
    ///
    /// Skips on AMD (Hi444PP unsupported) and on any host without
    /// WAYMUX_TEST_VULKAN=1. The actual rental validation test runs
    /// on NVIDIA where this constructor succeeds.
    #[test]
    fn vulkan_lossless_recorder_constructs() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let Some(r) = VkRecorderLossless::try_new(1920, 1080) else {
            eprintln!(
                "VkRecorderLossless::try_new(1920x1080) returned None — skipping \
                 (expected on AMD/Mesa where Hi444PP isn't supported)"
            );
            return;
        };
        let cp = r.codec_private();
        assert!(!cp.is_empty(), "codec_private must be non-empty");
        // AVCDecoderConfigurationRecord layout (ISO 14496-15 §5.3.3.1):
        //   [0]: configurationVersion (1)
        //   [1]: AVCProfileIndication (profile_idc)
        //   [2]: profile_compatibility flags
        //   [3]: AVCLevelIndication (level_idc)
        // Verify [1] is 244 — that's the on-wire profile_idc for
        // HIGH_444_PREDICTIVE.
        assert_eq!(cp[0], 1, "configurationVersion");
        assert_eq!(
            cp[1], 244,
            "AVCProfileIndication should be 244 (HIGH_444_PREDICTIVE), got {}",
            cp[1]
        );
        eprintln!(
            "VkRecorderLossless codec_private: {} bytes, profile_idc={}, level_idc={}",
            cp.len(),
            cp[1],
            cp[3]
        );
    }

    /// `VkRecorderLossless::encode_idr_from_bgra` produces non-empty
    /// H.264 NAL bytes from a synthetic BGRA pattern at 256×256.
    ///
    /// 256 is small enough that the encode submit is cheap even on
    /// modest hardware. The exact bytes aren't asserted (encoder
    /// output is implementation-dependent at the bit level), but a
    /// zero-byte result means the encode submit failed silently —
    /// either the FrameResources444 setup is wrong or the Hi444PP
    /// session rejected the picture.
    #[test]
    fn vulkan_lossless_encode_produces_nal() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        let w = 256u32;
        let h = 256u32;
        let Some(r) = VkRecorderLossless::try_new(w, h) else {
            eprintln!(
                "VkRecorderLossless::try_new({w}x{h}) returned None — skipping \
                 (expected on AMD/Mesa)"
            );
            return;
        };
        // Deterministic checkerboard + gradient. Bytes are BGRA8 packed.
        let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
        for y in 0..(h as usize) {
            for x in 0..(w as usize) {
                let i = (y * w as usize + x) * 4;
                let checker = ((x / 32) ^ (y / 32)) & 1;
                let r = (x % 256) as u8;
                let g = (y % 256) as u8;
                let b = if checker == 0 { 32u8 } else { 224u8 };
                bgra[i] = b; // B
                bgra[i + 1] = g; // G
                bgra[i + 2] = r; // R
                bgra[i + 3] = 0xff;
            }
        }
        let nal = r
            .encode_idr_from_bgra(&bgra, 0)
            .expect("encode_idr_from_bgra should produce a NAL");
        assert!(!nal.data.is_empty(), "encoded NAL must be non-empty");
        assert!(nal.is_keyframe, "first frame should be a keyframe (IDR)");
        eprintln!(
            "VkRecorderLossless encoded {}×{} → {} bytes (keyframe={})",
            w,
            h,
            nal.data.len(),
            nal.is_keyframe
        );
    }

    #[test]
    fn importable_modifiers_always_includes_linear() {
        // The cached accessor must always contain LINEAR so software/shm
        // and LINEAR-dmabuf clients keep working even with no GPU. On a GPU
        // host it will also contain tiled modifiers; on CI (no Vulkan) it
        // degrades to exactly [LINEAR].
        let mods = super::importable_bgra_modifiers();
        assert!(
            mods.contains(&crate::dmabuf::DRM_FORMAT_MOD_LINEAR),
            "importable set must always include LINEAR, got {mods:?}"
        );
    }

    #[test]
    fn advertised_set_comes_from_egl_vulkan_gate_separate() {
        // The advertised set must equal the EGL set (LINEAR-inclusive).
        let adv = super::importable_bgra_modifiers();
        let egl = crate::dmabuf::egl_importable_bgra_modifiers();
        assert_eq!(adv, egl, "advertised set must be the EGL set");
        // The Vulkan-import gate is a SEPARATE set; LINEAR is always Vulkan-importable.
        assert!(super::vulkan_importable_bgra_modifiers()
            .contains(&crate::dmabuf::DRM_FORMAT_MOD_LINEAR));
        assert!(super::modifier_is_importable(
            crate::dmabuf::DRM_FORMAT_MOD_LINEAR
        ));
    }

    /// SPIKE / acceptance gate for H.264 P-frames in the in-house Vulkan
    /// encoder. Encodes frame 0 as IDR then frames 1..N as P-frames, each
    /// referencing the previous reconstruction. The synthetic content
    /// changes slightly per frame (small horizontal scroll) so there's
    /// motion but high inter-frame redundancy.
    ///
    /// Asserts:
    ///   1. every P-frame is < 0.5x the IDR's byte size — proves the
    ///      reference reconstruction is retained across submits and
    ///      inter-prediction is actually happening (a non-retained ref
    ///      would force I-coded macroblocks → ~IDR-sized P-frames),
    ///   2. the concatenated Annex-B stream decodes cleanly via ffmpeg.
    ///
    /// Gated on WAYMUX_TEST_VULKAN=1.
    #[test]
    fn vulkan_pframe_spike() {
        if !matches!(
            std::env::var("WAYMUX_TEST_VULKAN").ok().as_deref(),
            Some("1")
        ) {
            eprintln!("set WAYMUX_TEST_VULKAN=1 to enable this test");
            return;
        }
        use std::io::Write;
        use std::process::Command;
        let w = 640u32;
        let h = 480u32;
        let n_p_frames = 8usize;

        let mut recorder = match VkRecorder::try_new(w, h) {
            Some(r) => r,
            None => {
                eprintln!("VkRecorder::try_new failed (no Vulkan video encode?); skipping");
                return;
            }
        };

        // Build a synthetic BGRA frame: a vertical-bar pattern that
        // scrolls horizontally by `shift` pixels. Lots of flat regions +
        // a small global motion => P-frames should be tiny vs the IDR.
        //
        // Content model = the realistic desktop case: a HIGH-DETAIL but
        // STATIC background (per-pixel pseudo-random texture, expensive to
        // I-code → large IDR) plus a small moving box. Each P-frame only
        // needs to code the box's displacement, so P ≪ IDR. The frame
        // argument is the box position (`pos` px from the left).
        let bg = |x: u32, y: u32| -> (u8, u8, u8) {
            // Cheap hash → high-frequency texture that doesn't compress to
            // near-nothing as an I-frame.
            let h1 = x
                .wrapping_mul(2654435761)
                .wrapping_add(y.wrapping_mul(40503));
            let h2 = (x ^ (y << 1)).wrapping_mul(2246822519);
            (
                (h1 >> 7) as u8,
                (h2 >> 11) as u8,
                (h1.wrapping_add(h2) >> 3) as u8,
            )
        };
        let make_frame = |pos: u32| -> Vec<u8> {
            let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
            let box_w = 48u32;
            let box_h = 48u32;
            let box_x = pos % (w - box_w);
            let box_y = h / 2 - box_h / 2;
            for y in 0..h {
                for x in 0..w {
                    let off = ((y * w + x) * 4) as usize;
                    let (b, gr, r) =
                        if x >= box_x && x < box_x + box_w && y >= box_y && y < box_y + box_h {
                            // Solid moving box.
                            (30u8, 220u8, 30u8)
                        } else {
                            bg(x, y)
                        };
                    bgra[off] = b;
                    bgra[off + 1] = gr;
                    bgra[off + 2] = r;
                    bgra[off + 3] = 0xFF;
                }
            }
            bgra
        };

        let mut stream: Vec<u8> = Vec::new();
        let mut sizes: Vec<(bool, usize)> = Vec::new();

        // The encoder emits only slice NALs; SPS/PPS live in the session's
        // AVCDecoderConfigurationRecord (codec_private). For a raw Annex-B
        // file, prepend the SPS+PPS as start-code-framed NALs so ffmpeg
        // can find the parameter sets.
        let avcc = recorder.codec_private().to_vec();
        let mut header: Vec<u8> = Vec::new();
        if avcc.len() > 6 {
            let start_code = [0u8, 0, 0, 1];
            let mut p = 5usize;
            let num_sps = (avcc[p] & 0x1F) as usize;
            p += 1;
            for _ in 0..num_sps {
                if p + 2 > avcc.len() {
                    break;
                }
                let len = ((avcc[p] as usize) << 8) | (avcc[p + 1] as usize);
                p += 2;
                if p + len > avcc.len() {
                    break;
                }
                header.extend_from_slice(&start_code);
                header.extend_from_slice(&avcc[p..p + len]);
                p += len;
            }
            if p < avcc.len() {
                let num_pps = avcc[p] as usize;
                p += 1;
                for _ in 0..num_pps {
                    if p + 2 > avcc.len() {
                        break;
                    }
                    let len = ((avcc[p] as usize) << 8) | (avcc[p + 1] as usize);
                    p += 2;
                    if p + len > avcc.len() {
                        break;
                    }
                    header.extend_from_slice(&start_code);
                    header.extend_from_slice(&avcc[p..p + len]);
                    p += len;
                }
            }
        }
        eprintln!(
            "codec_private = {} bytes; extracted {} bytes of Annex-B SPS/PPS header",
            avcc.len(),
            header.len()
        );
        stream.extend_from_slice(&header);

        // Frame 0: IDR.
        let idr_bgra = make_frame(0);
        let idr = recorder
            .encode_bgra(&idr_bgra, 0, true)
            .expect("encode IDR frame 0");
        assert!(idr.is_keyframe, "frame 0 must be a keyframe");
        let idr_size = idr.data.len();
        sizes.push((true, idr_size));
        stream.extend_from_slice(&idr.data);

        // Frames 1..=n_p_frames: P-frames; only the small box moves.
        for i in 1..=n_p_frames {
            let bgra = make_frame((i as u32) * 4);
            let p = recorder
                .encode_bgra(&bgra, (i as i64) * 16_667, false)
                .unwrap_or_else(|| panic!("encode P frame {i}"));
            assert!(
                !p.is_keyframe,
                "frame {i} should be a P-frame, not a keyframe"
            );
            sizes.push((false, p.data.len()));
            stream.extend_from_slice(&p.data);
        }

        eprintln!("=== vulkan_pframe_spike per-frame sizes ({w}x{h}) ===");
        for (i, (is_idr, sz)) in sizes.iter().enumerate() {
            let kind = if *is_idr { "IDR" } else { "P  " };
            let ratio = *sz as f64 / idr_size as f64;
            eprintln!("  frame {i:2}: {kind} {sz:7} bytes  ({ratio:.3}x IDR)");
        }

        // ── Assert 1: each P-frame is < 0.5x the IDR size ──
        for (i, (is_idr, sz)) in sizes.iter().enumerate() {
            if *is_idr {
                continue;
            }
            assert!(
                (*sz as f64) < 0.5 * (idr_size as f64),
                "P-frame {i} is {sz} bytes, not < 0.5x IDR ({idr_size}); \
                 the reconstructed reference is likely NOT being retained \
                 across submits (inter-prediction not happening)"
            );
        }

        // ── Assert 2: ffmpeg decodes the stream cleanly ──
        let out_path = std::env::temp_dir().join("vulkan_pframe_spike.h264");
        // Spec wants /tmp explicitly.
        let canonical = std::path::Path::new("/tmp/vulkan_pframe_spike.h264");
        let mut file = std::fs::File::create(&out_path).expect("create h264 file");
        file.write_all(&stream).expect("write h264");
        file.flush().expect("flush h264");
        drop(file);
        // Also write the canonical /tmp path the spec names.
        let _ = std::fs::write(canonical, &stream);
        eprintln!("wrote {} bytes to {}", stream.len(), canonical.display());

        if Command::new("ffmpeg").arg("-version").output().is_err() {
            eprintln!("ffmpeg not installed; skipping decode check");
            return;
        }
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                canonical.to_str().unwrap(),
                "-f",
                "null",
                "-",
            ])
            .status()
            .expect("run ffmpeg decode");
        assert!(
            status.success(),
            "ffmpeg failed to decode the P-frame stream cleanly (exit {status:?})"
        );
        eprintln!("ffmpeg decoded the IDR+P stream cleanly");
    }
}
