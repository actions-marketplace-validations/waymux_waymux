// SPDX-License-Identifier: Apache-2.0

//! NV2: in-process CUDA/NVENC H.264 recorder for KWin's GPU-tiled dmabuf.
//!
//! Parallel to `vulkan_record.rs`, but for NVIDIA: NVIDIA's Vulkan can't import
//! dmabufs, so we go EGL -> CUDA (`cuGraphicsEGLRegisterImage`) and feed the
//! resulting CUDA surface to NVENC (later tasks). This task (NV2.1) is just the
//! module scaffold + the CUDA/EGL FFI, lifted verbatim from the proven NV1
//! probe (`examples/egl_cuda_interop_spike.rs`) plus the extra CUDA driver
//! symbols the encoder will need (module load / kernel launch / mem / texobj).
//!
//! No NVENC and no encode logic yet — those land in later NV2 tasks. Everything
//! here is currently unused on purpose; the dlopen'd symbols are exercised by
//! the live run, the offline build only checks it compiles.

#![allow(dead_code)]
// `RawFd` is imported per the NV2.1 scaffold spec for the dmabuf-import path
// that later tasks add; it's unused in this scaffold-only commit.
#[allow(unused_imports)]
use std::os::fd::RawFd;
use std::os::raw::{c_char, c_void};

// ---------------------------------------------------------------------------
// EGL dmabuf-import constants (lifted from the NV1 probe / dmabuf.rs egl_ext).
// ---------------------------------------------------------------------------

/// `EGL_LINUX_DMA_BUF_EXT` — the import target for a dmabuf-backed EGLImage.
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: i32 = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: i32 = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: i32 = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: i32 = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: i32 = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: i32 = 0x3444;
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;
const EGL_NONE: i32 = 0x3038;

// ---------------------------------------------------------------------------
// EGL extension fn FFI (lifted from the NV1 probe).
// ---------------------------------------------------------------------------

/// `EGLImageKHR eglCreateImageKHR(EGLDisplay, EGLContext, EGLenum target,
///   EGLClientBuffer buffer, const EGLint *attrib_list)`
type EglCreateImageKHR = unsafe extern "C" fn(
    dpy: *mut c_void,
    ctx: *mut c_void,
    target: u32,
    buffer: *mut c_void,
    attrib_list: *const i32,
) -> *mut c_void;

/// `EGLBoolean eglDestroyImageKHR(EGLDisplay, EGLImageKHR)`
type EglDestroyImageKHR = unsafe extern "C" fn(dpy: *mut c_void, image: *mut c_void) -> u32;

// ---------------------------------------------------------------------------
// CUDA driver-API FFI + CUeglFrame (vendored from cudaEGL.h). Field ORDER is
// load-bearing for the live run — copied verbatim from the proven NV1 probe.
// ---------------------------------------------------------------------------

/// `typedef union { CUarray array[3]; CUdeviceptr pitch[3]; } frame;`
/// On 64-bit, both `CUarray` (a pointer) and `CUdeviceptr` (unsigned long long)
/// are 8 bytes, so a `[*mut c_void; 3]` per arm matches the C layout.
#[repr(C)]
union CUeglFramePtr {
    array: [*mut c_void; 3],
    pitch: [*mut c_void; 3],
}

/// Vendored from `CUeglFrame_st` in cudaEGL.h. Exact field order:
///   union frame; width; height; depth; pitch; planeCount; numChannels;
///   frameType; eglColorFormat; cuFormat;
#[repr(C)]
struct CUeglFrame {
    frame: CUeglFramePtr,
    width: u32,
    height: u32,
    depth: u32,
    pitch: u32,
    plane_count: u32,
    num_channels: u32,
    /// `CUeglFrameType`: 0 = CU_EGL_FRAME_TYPE_ARRAY, 1 = CU_EGL_FRAME_TYPE_PITCH
    frame_type: u32,
    /// `CUeglColorFormat`
    egl_color_format: u32,
    /// `CUarray_format`
    cu_format: u32,
}

// --- CUDA driver fn-pointer types lifted from the NV1 probe -----------------

type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(dev: *mut i32, ordinal: i32) -> i32;
type CuCtxCreateV2 = unsafe extern "C" fn(pctx: *mut *mut c_void, flags: u32, dev: i32) -> i32;
type CuGraphicsEglRegisterImage =
    unsafe extern "C" fn(pres: *mut *mut c_void, image: *mut c_void, flags: u32) -> i32;
type CuGraphicsResourceGetMappedEglFrame =
    unsafe extern "C" fn(frame: *mut CUeglFrame, res: *mut c_void, index: u32, mip: u32) -> i32;
type CuGraphicsUnregisterResource = unsafe extern "C" fn(res: *mut c_void) -> i32;
type CuGetErrorString = unsafe extern "C" fn(err: i32, pstr: *mut *const c_char) -> i32;

// --- NEW CUDA driver fn-pointer types the encoder will need (NV2.1) ---------

type CuModuleLoadData = unsafe extern "C" fn(module: *mut *mut c_void, image: *const c_void) -> i32;
type CuModuleGetFunction =
    unsafe extern "C" fn(func: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> i32;
type CuLaunchKernel = unsafe extern "C" fn(
    f: *mut c_void,
    gx: u32,
    gy: u32,
    gz: u32,
    bx: u32,
    by: u32,
    bz: u32,
    shmem: u32,
    stream: *mut c_void,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32;
type CuMemAllocV2 = unsafe extern "C" fn(dptr: *mut u64, bytesize: usize) -> i32;
type CuMemFreeV2 = unsafe extern "C" fn(dptr: u64) -> i32;
type CuStreamSynchronize = unsafe extern "C" fn(stream: *mut c_void) -> i32;
type CuTexObjectCreate = unsafe extern "C" fn(
    obj: *mut u64,
    res: *const CudaResourceDesc,
    tex: *const CudaTextureDesc,
    rv: *const c_void,
) -> i32;
type CuTexObjectDestroy = unsafe extern "C" fn(obj: u64) -> i32;
type CuCtxDestroyV2 = unsafe extern "C" fn(ctx: *mut c_void) -> i32;
type CuModuleUnload = unsafe extern "C" fn(module: *mut c_void) -> i32;

// --- CUDA driver fn-pointer types for the host-pixels input path (Part A) ---
//
// `encode_pixels` copies a host BGRA frame into a persistent input cuArray,
// then runs the SAME tex→CSC→NVENC tail as `encode_dmabuf`. These three driver
// symbols back that path (array create/destroy + a 2D host→array copy).
type CuArrayCreateV2 =
    unsafe extern "C" fn(handle: *mut *mut c_void, desc: *const CUDA_ARRAY_DESCRIPTOR) -> i32;
type CuMemcpy2DV2 = unsafe extern "C" fn(copy: *const CUDA_MEMCPY2D) -> i32;
type CuArrayDestroy = unsafe extern "C" fn(handle: *mut c_void) -> i32;

// ---------------------------------------------------------------------------
// Vendored CUDA descriptor structs (from cuda.h). Layout/size is load-bearing
// for the live texobj-create call; the offline build only checks they compile.
// ---------------------------------------------------------------------------

/// CUDA resource type. We only use the array path for the tiled-surface texture.
const CU_RESOURCE_TYPE_ARRAY: i32 = 0;
/// Clamp out-of-bounds texture coords.
const CU_TR_ADDRESS_MODE_CLAMP: i32 = 1;
/// Point (nearest) sampling — we read raw texels, never interpolate.
const CU_TR_FILTER_MODE_POINT: i32 = 0;
/// Read texels as raw integers (so `tex2D<uchar4>` returns 0..255, not
/// normalized floats) and keep coordinates NON-normalized.
const CU_TRSF_READ_AS_INTEGER: u32 = 1;

/// `CUDA_RESOURCE_DESC` from cuda.h.
///
/// The real C struct is `{ CUresourcetype resType; union { ... } res; uint
/// flags; }`. The union is a tagged blob whose largest arm (`linear`/`pitch2D`)
/// is several pointers + format/size fields; modelling it as `[u64; 16]` is at
/// least as large as the real union (16 * 8 = 128 bytes), so the struct is
/// large enough that the driver never reads past our allocation. For the array
/// path the CUarray handle goes in `handle[0]`.
#[repr(C)]
struct CudaResourceDesc {
    res_type: i32,
    _pad: i32,
    handle: [u64; 16],
    flags: u32,
    _pad2: u32,
}

/// `CUDA_TEXTURE_DESC` from cuda.h. Field set/order mirrors the C struct:
///   CUaddress_mode addressMode[3]; CUfilter_mode filterMode; uint flags;
///   uint maxAnisotropy; CUfilter_mode mipmapFilterMode; float mipmapLevelBias;
///   float minMipmapLevelClamp; float maxMipmapLevelClamp;
///   float borderColor[4]; int reserved[12];
#[repr(C)]
struct CudaTextureDesc {
    address_mode: [i32; 3],
    filter_mode: i32,
    flags: u32,
    max_anisotropy: u32,
    mipmap_filter_mode: i32,
    mipmap_level_bias: f32,
    min_mipmap_level_clamp: f32,
    max_mipmap_level_clamp: f32,
    border_color: [f32; 4],
    _reserved: [i32; 12],
}

// --- CUDA memcpy / array-create descriptors for the host-pixels path --------
//
// Vendored from cuda.h (CUDA 12.x). Layout/size is load-bearing for the live
// `cuMemcpy2D_v2` / `cuArrayCreate_v2` calls; the offline build only checks the
// static asserts below.

/// `CUmemorytype` values used by CUDA_MEMCPY2D.
const CU_MEMORYTYPE_HOST: u32 = 1;
const CU_MEMORYTYPE_DEVICE: u32 = 2;
const CU_MEMORYTYPE_ARRAY: u32 = 3;
/// `CU_AD_FORMAT_UNSIGNED_INT8` — 8-bit unsigned components (CUarray_format).
const CU_AD_FORMAT_UNSIGNED_INT8: u32 = 0x01;

/// `CUDA_ARRAY_DESCRIPTOR` from cuda.h:
///   `{ size_t Width; size_t Height; CUarray_format Format; unsigned int NumChannels; }`.
/// On 64-bit: 8 + 8 + 4 + 4 = 24 bytes (asserted below).
#[repr(C)]
struct CUDA_ARRAY_DESCRIPTOR {
    width: usize,
    height: usize,
    format: u32,
    num_channels: u32,
}

/// `CUDA_MEMCPY2D` from cuda.h. EXACT field order. NOTE: `CUmemorytype` is a
/// 32-bit enum but the following pointer/handle field is 8-byte-aligned on
/// 64-bit, so C inserts 4 bytes of implicit padding after each `*MemoryType`.
/// We model that padding explicitly (`_pad0`/`_pad1`) so the layout matches the
/// C ABI byte-for-byte.
///
/// Layout (64-bit, bytes):
///   srcXInBytes\@0 (8) srcY\@8 (8)
///   srcMemoryType\@16 (4) _pad0\@20 (4)
///   srcHost\@24 (8) srcDevice\@32 (8) srcArray\@40 (8) srcPitch\@48 (8)
///   dstXInBytes\@56 (8) dstY\@64 (8)
///   dstMemoryType\@72 (4) _pad1\@76 (4)
///   dstHost\@80 (8) dstDevice\@88 (8) dstArray\@96 (8) dstPitch\@104 (8)
///   WidthInBytes\@112 (8) Height\@120 (8)  → sizeof = 128.
#[repr(C)]
struct CUDA_MEMCPY2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: u32,
    _pad0: u32,
    src_host: *const c_void,
    src_device: u64,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: u32,
    _pad1: u32,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

// Static size cross-checks against cuda.h (CUDA 12.x).
const _: () = assert!(std::mem::size_of::<CUDA_ARRAY_DESCRIPTOR>() == 24);
const _: () = assert!(std::mem::size_of::<CUDA_MEMCPY2D>() == 128);

// ---------------------------------------------------------------------------
// CudaLib: dlopen libcuda.so.1 + load every symbol. The `Library` must outlive
// the fn pointers, so it is kept first in the struct.
// ---------------------------------------------------------------------------

/// Holds the dlopened libcuda symbols (NV1 probe set + the NV2.1 additions).
pub struct CudaLib {
    _lib: libloading::Library,
    init: CuInit,
    device_get: CuDeviceGet,
    ctx_create: CuCtxCreateV2,
    register_image: CuGraphicsEglRegisterImage,
    get_mapped_frame: CuGraphicsResourceGetMappedEglFrame,
    unregister: CuGraphicsUnregisterResource,
    get_error_string: CuGetErrorString,
    module_load_data: CuModuleLoadData,
    module_get_function: CuModuleGetFunction,
    module_unload: CuModuleUnload,
    launch_kernel: CuLaunchKernel,
    mem_alloc: CuMemAllocV2,
    mem_free: CuMemFreeV2,
    stream_synchronize: CuStreamSynchronize,
    tex_object_create: CuTexObjectCreate,
    tex_object_destroy: CuTexObjectDestroy,
    cu_ctx_destroy_v2: CuCtxDestroyV2,
    array_create: CuArrayCreateV2,
    memcpy_2d: CuMemcpy2DV2,
    array_destroy: CuArrayDestroy,
}

impl CudaLib {
    /// dlopen `libcuda.so.1`, resolve every driver symbol, then `cuInit(0)`.
    ///
    /// Returns `None` cleanly on any failure (no GPU / no driver / missing
    /// symbol / cuInit != CUDA_SUCCESS) so a no-GPU CI host degrades gracefully
    /// rather than panicking.
    pub fn load() -> Option<Self> {
        let lib = unsafe { libloading::Library::new("libcuda.so.1") }.ok()?;
        // SAFETY: signatures match the CUDA driver API; symbols come from libcuda.
        unsafe {
            macro_rules! sym {
                ($name:expr) => {
                    match lib.get::<_>($name) {
                        Ok(s) => *s,
                        Err(_) => return None,
                    }
                };
            }
            let init: CuInit = sym!(b"cuInit\0");
            let device_get: CuDeviceGet = sym!(b"cuDeviceGet\0");
            // The versioned symbol `cuCtxCreate_v2` is what the driver exports;
            // unversioned `cuCtxCreate` is a header macro.
            let ctx_create: CuCtxCreateV2 = sym!(b"cuCtxCreate_v2\0");
            let register_image: CuGraphicsEglRegisterImage = sym!(b"cuGraphicsEGLRegisterImage\0");
            let get_mapped_frame: CuGraphicsResourceGetMappedEglFrame =
                sym!(b"cuGraphicsResourceGetMappedEglFrame\0");
            let unregister: CuGraphicsUnregisterResource = sym!(b"cuGraphicsUnregisterResource\0");
            let get_error_string: CuGetErrorString = sym!(b"cuGetErrorString\0");
            let module_load_data: CuModuleLoadData = sym!(b"cuModuleLoadData\0");
            let module_get_function: CuModuleGetFunction = sym!(b"cuModuleGetFunction\0");
            let module_unload: CuModuleUnload = sym!(b"cuModuleUnload\0");
            let launch_kernel: CuLaunchKernel = sym!(b"cuLaunchKernel\0");
            let mem_alloc: CuMemAllocV2 = sym!(b"cuMemAlloc_v2\0");
            let mem_free: CuMemFreeV2 = sym!(b"cuMemFree_v2\0");
            let stream_synchronize: CuStreamSynchronize = sym!(b"cuStreamSynchronize\0");
            let tex_object_create: CuTexObjectCreate = sym!(b"cuTexObjectCreate\0");
            let tex_object_destroy: CuTexObjectDestroy = sym!(b"cuTexObjectDestroy\0");
            let cu_ctx_destroy_v2: CuCtxDestroyV2 = sym!(b"cuCtxDestroy_v2\0");
            let array_create: CuArrayCreateV2 = sym!(b"cuArrayCreate_v2\0");
            let memcpy_2d: CuMemcpy2DV2 = sym!(b"cuMemcpy2D_v2\0");
            let array_destroy: CuArrayDestroy = sym!(b"cuArrayDestroy\0");

            // cuInit(0): CUDA_SUCCESS == 0. Any non-zero means no usable GPU.
            if init(0) != 0 {
                return None;
            }

            Some(CudaLib {
                _lib: lib,
                init,
                device_get,
                ctx_create,
                register_image,
                get_mapped_frame,
                unregister,
                get_error_string,
                module_load_data,
                module_get_function,
                module_unload,
                launch_kernel,
                mem_alloc,
                mem_free,
                stream_synchronize,
                tex_object_create,
                tex_object_destroy,
                cu_ctx_destroy_v2,
                array_create,
                memcpy_2d,
                array_destroy,
            })
        }
    }

    /// Render a CUresult as "N (message)".
    fn err(&self, code: i32) -> String {
        if code == 0 {
            return "0 (CUDA_SUCCESS)".to_string();
        }
        let mut s: *const c_char = std::ptr::null();
        let rc = unsafe { (self.get_error_string)(code, &mut s) };
        if rc == 0 && !s.is_null() {
            let msg = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy();
            format!("{code} ({msg})")
        } else {
            format!("{code} (unknown)")
        }
    }
}

// ---------------------------------------------------------------------------
// Embedded PTX for the BT.709 NV12 CSC kernel (NV2.3a).
// The placeholder file ends with a NUL byte so include_bytes! yields a
// NUL-terminated buffer suitable for cuModuleLoadData. The real PTX is
// generated in NV2.3b with: nvcc -ptx -arch=sm_89 cuda_nv12_csc.cu
// ---------------------------------------------------------------------------
const EMBEDDED_PTX: &[u8] = include_bytes!("cuda_nv12_csc.ptx");

// ===========================================================================
// NV2.2 — nvEncodeAPI FFI (vendored from nvEncodeAPI.h, NVENC SDK 13 / r580).
//
// The header is NOT available on this dev box, so the struct layouts, version
// minor numbers, and GUID bytes below are transcribed from knowledge of the
// NVENC SDK 12/13 API and best-effort matched to the documented layout. A
// LATER live task (NV2.3b) cross-checks these against the real header on the
// GPU VM and fixes any mismatch. The offline build + struct-size asserts here
// only catch gross errors.
//
// EVERY item flagged with `// VERIFY NV2.3b:` is a value we were UNSURE about.
// ===========================================================================

// --- Version macros --------------------------------------------------------
//
//   NVENCAPI_VERSION = NVENC_MAJOR | (NVENC_MINOR << 24)
//   struct_ver(v, extra) = NVENCAPI_VERSION | (v << 16) | (0x7 << 28) | (extra << 31)

/// NVENC SDK 13.0 → major = 13.
pub const NVENCAPI_MAJOR: u32 = 13;
/// NVENC SDK 13.0 → minor = 0.
pub const NVENCAPI_MINOR: u32 = 0;
/// `NVENCAPI_VERSION = major | (minor << 24)`.
pub const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24);

/// `NVENCAPI_STRUCT_VERSION(ver)` without the capability bit.
const fn struct_ver(v: u32) -> u32 {
    NVENCAPI_VERSION | (v << 16) | (0x7 << 28)
}
/// Variant carrying the high capability bit (bit 31) — used by structs the
/// header marks with the trailing `| (1 << 31)` (notably NV_ENC_CONFIG and
/// NV_ENC_INITIALIZE_PARAMS in recent SDKs).
const fn struct_ver_cap(v: u32) -> u32 {
    struct_ver(v) | (1u32 << 31)
}

// The struct version minor numbers below are the documented NVENC SDK values.
// VERIFY NV2.3b: every minor number in this block against nvEncodeAPI.h.
pub fn nv_enc_initialize_params_ver() -> u32 {
    // NV_ENC_INITIALIZE_PARAMS_VER = struct_ver(7) | (1<<31)  [verified vs nvEncodeAPI.h SDK 13]
    struct_ver_cap(7)
}
pub fn nv_enc_config_ver() -> u32 {
    // NV_ENC_CONFIG_VER = struct_ver(9) | (1<<31)  [verified vs nvEncodeAPI.h SDK 13]
    struct_ver_cap(9)
}
fn nv_enc_preset_config_ver() -> u32 {
    // NV_ENC_PRESET_CONFIG_VER = struct_ver(5) | (1<<31)  [verified vs nvEncodeAPI.h SDK 13]
    struct_ver_cap(5)
}
fn nv_enc_rc_params_ver() -> u32 {
    // NV_ENC_RC_PARAMS_VER = struct_ver(1) (no cap bit)  [verified vs nvEncodeAPI.h SDK 13]
    struct_ver(1)
}
fn nv_enc_open_encode_session_ex_params_ver() -> u32 {
    struct_ver(1)
}
fn nv_enc_register_resource_ver() -> u32 {
    struct_ver(5) // [verified vs nvEncodeAPI.h SDK 13]
}
fn nv_enc_map_input_resource_ver() -> u32 {
    struct_ver(4)
}
fn nv_enc_create_bitstream_buffer_ver() -> u32 {
    struct_ver(1)
}
fn nv_enc_reconfigure_params_ver() -> u32 {
    // NV_ENC_RECONFIGURE_PARAMS_VER = NVENCAPI_STRUCT_VERSION(2) | (1<<31)
    // [verified vs nvEncodeAPI.h SDK 13 — NOT struct_ver(1)].
    struct_ver_cap(2)
}
fn nv_enc_pic_params_ver() -> u32 {
    // NV_ENC_PIC_PARAMS_VER = struct_ver(7) | (1<<31)
    struct_ver_cap(7)
}
fn nv_enc_lock_bitstream_ver() -> u32 {
    struct_ver_cap(2) // NV_ENC_LOCK_BITSTREAM_VER carries the cap bit [verified vs nvEncodeAPI.h SDK 13]
}
fn nv_enc_sequence_param_payload_ver() -> u32 {
    struct_ver(1)
}
fn nv_encode_api_function_list_ver() -> u32 {
    struct_ver(2)
}

// --- GUIDs -----------------------------------------------------------------

/// NVENC GUID. C layout: `{ uint32 Data1; uint16 Data2; uint16 Data3; uint8 Data4[8]; }`.
#[repr(C)]
#[derive(Clone, Copy)]
// matches the NVENC C API type name
#[allow(clippy::upper_case_acronyms)]
pub struct GUID {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

impl GUID {
    const fn zero() -> Self {
        GUID {
            data1: 0,
            data2: 0,
            data3: 0,
            data4: [0; 8],
        }
    }
}

// VERIFY NV2.3b: the GUID byte values below against nvEncodeAPI.h.
//
// NV_ENC_CODEC_H264_GUID
//   {0x6BC82762, 0x4E63, 0x4ca4, {0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf}}
const NV_ENC_CODEC_H264_GUID: GUID = GUID {
    data1: 0x6BC8_2762,
    data2: 0x4E63,
    data3: 0x4ca4,
    data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
};

// NV_ENC_PRESET_P3_GUID [verified vs nvEncodeAPI.h SDK 13]
//   {0x36850110, 0x3a07, 0x441f, {0x94, 0xd5, 0x36, 0x70, 0x63, 0x1f, 0x91, 0xf6}}
const NV_ENC_PRESET_P3_GUID: GUID = GUID {
    data1: 0x3685_0110,
    data2: 0x3a07,
    data3: 0x441f,
    data4: [0x94, 0xd5, 0x36, 0x70, 0x63, 0x1f, 0x91, 0xf6],
};

// NV_ENC_H264_PROFILE_HIGH_GUID [verified vs nvEncodeAPI.h SDK 13]
//   {0xe7cbc309, 0x4f7a, 0x4b89, {0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10}}
const NV_ENC_H264_PROFILE_HIGH_GUID: GUID = GUID {
    data1: 0xe7cb_c309,
    data2: 0x4f7a,
    data3: 0x4b89,
    data4: [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10],
};

// NV_ENC_PRESET_P5_GUID [verified vs /usr/local/include/ffnvcodec/nvEncodeAPI.h
// on the L40 build VM 2026-06-01 — the earlier transcription was WRONG bytes,
// which made nvEncGetEncodePresetConfigEx return error 4 and CudaNvenc fall back
// to the flipping CPU-readback H264Nvenc path]
//   {0x21c6e6b4, 0x297a, 0x4cba, {0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3}}
const NV_ENC_PRESET_P5_GUID: GUID = GUID {
    data1: 0x21c6_e6b4,
    data2: 0x297a,
    data3: 0x4cba,
    data4: [0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3],
};

// --- enums -----------------------------------------------------------------

/// `NV_ENC_DEVICE_TYPE_CUDA` == 0x1 [verified vs nvEncodeAPI.h SDK 13].
const NV_ENC_DEVICE_TYPE_CUDA: u32 = 1;
/// `NV_ENC_TUNING_INFO_LOW_LATENCY` == 2 [verified].
const NV_ENC_TUNING_INFO_LOW_LATENCY: u32 = 2;

/// `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR` == 0x1 [verified vs nvEncodeAPI.h SDK 13].
const NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR: u32 = 1;
/// `NV_ENC_BUFFER_FORMAT_NV12` == 0x00000001 [verified].
const NV_ENC_BUFFER_FORMAT_NV12: u32 = 0x0000_0001;
/// `NV_ENC_BUFFER_USAGE` input-image == 0 [verified: NV_ENC_INPUT_IMAGE default].
const NV_ENC_INPUT_IMAGE: u32 = 0;
/// `NV_ENC_PIC_STRUCT_FRAME` == 0x1 [verified].
const NV_ENC_PIC_STRUCT_FRAME: u32 = 1;
/// `NV_ENC_PIC_FLAG_FORCEIDR` == 0x2 [verified vs nvEncodeAPI.h SDK 13].
const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x2;
/// `NV_ENC_PIC_FLAG_OUTPUT_SPSPPS` == 0x4 [verified vs nvEncodeAPI.h SDK 13].
const NV_ENC_PIC_FLAG_OUTPUT_SPSPPS: u32 = 0x4;
/// `NV_ENC_PARAMS_RC_CBR` (constant bitrate) == 0x2.
/// VERIFIED vs nvEncodeAPI.h (NVENC SDK 13) on disk: `enum _NV_ENC_PARAMS_RC_MODE
/// { NV_ENC_PARAMS_RC_CONSTQP=0x0, NV_ENC_PARAMS_RC_VBR=0x1, NV_ENC_PARAMS_RC_CBR=0x2 }`.
/// NOTE: the older SDK-9-era `0x10` value for CBR is NOT what SDK 13 uses —
/// the header on disk is authoritative and gives 0x2.
const NV_ENC_PARAMS_RC_CBR: u32 = 0x2;

/// `NV_ENC_MULTI_PASS_DISABLED` == 0 [verified]. Multi-pass under low-latency
/// tuning spikes encode latency to ~37 frames; pin it off explicitly.
const NV_ENC_MULTI_PASS_DISABLED: u32 = 0;
/// NV_ENC_RC_PARAMS.flags bits [verified vs nvEncodeAPI.h SDK 13]: the bitfield
/// word packs enableMinQP@0, enableMaxQP@1, enableInitialRCQP@2, enableAQ@3,
/// reserved@4, enableLookahead@5, disableIadapt@6, disableBadapt@7,
/// enableTemporalAQ@8, ..., aqStrength@12-15.
const NV_ENC_RC_FLAG_ENABLE_AQ: u32 = 1 << 3; // spatial AQ
const NV_ENC_RC_FLAG_ENABLE_TEMPORAL_AQ: u32 = 1 << 8;

/// Build the OR-mask added to `NV_ENC_RC_PARAMS.flags` for the viewer encoder:
/// spatial AQ + temporal AQ + the given strength (0..=15, masked to 4 bits; 0 = driver-default strength).
/// Temporal AQ is the single best NVENC knob for static text (a constant,
/// high-spatial-detail region).
const fn viewer_rc_flags(aq_strength: u32) -> u32 {
    NV_ENC_RC_FLAG_ENABLE_AQ | NV_ENC_RC_FLAG_ENABLE_TEMPORAL_AQ | ((aq_strength & 0xF) << 12)
}

// ===========================================================================
// NV_ENCODE_API_FUNCTION_LIST
//
// The real header lays out `version`, `reserved`, then ~41 named function
// pointers, then a `void* reserved2[N]` tail. Getting every slot + order
// exactly right is what NV2.3b verifies; for NV2.2 we model the function
// pointers as a flat `[*mut c_void; FN_SLOTS]` array indexed by documented
// slot constants, so the struct is guaranteed big enough and `version` is
// first. We only ever read the slots we call.
// VERIFY NV2.3b: the slot INDICES and the total slot count against the header.
// ===========================================================================

/// Number of named function-pointer slots in NV_ENCODE_API_FUNCTION_LIST
/// (NVENC SDK 13). The header lays out, after `version`+`reserved`, exactly 43
/// pointer-sized members (one of which is a `void* reserved1` filler at index
/// 33), followed by `void* reserved2[275]`. The total pointer-table size is
/// therefore 43 + 275 = 318. We model the named region as a flat slot array
/// indexed by the slot constants below, and keep the full reserved2 tail so the
/// driver never writes past our allocation when it fills the list.
const FN_SLOTS: usize = 43;

// Slot indices (0-based) into the function-pointer array, transcribed
// field-by-field from nvEncodeAPI.h (NVENC SDK 13) NV_ENCODE_API_FUNCTION_LIST.
// nvEncOpenEncodeSession is slot 0; the order below matches the header EXACTLY.
const SLOT_OPEN_ENCODE_SESSION: usize = 0;
const SLOT_GET_ENCODE_GUID_COUNT: usize = 1;
const SLOT_GET_ENCODE_PROFILE_GUID_COUNT: usize = 2;
const SLOT_GET_ENCODE_PROFILE_GUIDS: usize = 3;
const SLOT_GET_ENCODE_GUIDS: usize = 4;
const SLOT_GET_INPUT_FORMAT_COUNT: usize = 5;
const SLOT_GET_INPUT_FORMATS: usize = 6;
const SLOT_GET_ENCODE_CAPS: usize = 7;
const SLOT_GET_ENCODE_PRESET_COUNT: usize = 8;
const SLOT_GET_ENCODE_PRESET_GUIDS: usize = 9;
const SLOT_GET_ENCODE_PRESET_CONFIG: usize = 10;
const SLOT_INITIALIZE_ENCODER: usize = 11;
const SLOT_CREATE_INPUT_BUFFER: usize = 12;
const SLOT_DESTROY_INPUT_BUFFER: usize = 13;
const SLOT_CREATE_BITSTREAM_BUFFER: usize = 14;
const SLOT_DESTROY_BITSTREAM_BUFFER: usize = 15;
const SLOT_ENCODE_PICTURE: usize = 16;
const SLOT_LOCK_BITSTREAM: usize = 17;
const SLOT_UNLOCK_BITSTREAM: usize = 18;
const SLOT_LOCK_INPUT_BUFFER: usize = 19;
const SLOT_UNLOCK_INPUT_BUFFER: usize = 20;
const SLOT_GET_ENCODE_STATS: usize = 21;
const SLOT_GET_SEQUENCE_PARAMS: usize = 22;
const SLOT_REGISTER_ASYNC_EVENT: usize = 23;
const SLOT_UNREGISTER_ASYNC_EVENT: usize = 24;
const SLOT_MAP_INPUT_RESOURCE: usize = 25;
const SLOT_UNMAP_INPUT_RESOURCE: usize = 26;
const SLOT_DESTROY_ENCODER: usize = 27;
const SLOT_INVALIDATE_REF_FRAMES: usize = 28;
const SLOT_OPEN_ENCODE_SESSION_EX: usize = 29;
const SLOT_REGISTER_RESOURCE: usize = 30;
const SLOT_UNREGISTER_RESOURCE: usize = 31;
const SLOT_RECONFIGURE_ENCODER: usize = 32;
// 33 = reserved1 (void*); 34..38 = MV/ME/lasterror/setIOCudaStreams.
const SLOT_GET_ENCODE_PRESET_CONFIG_EX: usize = 39;

/// `NV_ENCODE_API_FUNCTION_LIST` — version first, then the fn-pointer table,
/// then the `void* reserved2[275]` tail. Modeled as a flat slot array (see
/// module note). The reserved2 size MUST match the header so the driver does
/// not write past our allocation.
#[repr(C)]
pub struct NV_ENCODE_API_FUNCTION_LIST {
    pub version: u32,
    pub reserved: u32,
    pub functions: [*mut c_void; FN_SLOTS],
    /// `void* reserved2[275]` trailing padding from the header.
    pub reserved2: [*mut c_void; 275],
}

impl NV_ENCODE_API_FUNCTION_LIST {
    fn zeroed() -> Self {
        // SAFETY: every field is plain-data / nullable pointer; all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
    #[inline]
    fn slot(&self, idx: usize) -> *mut c_void {
        self.functions[idx]
    }
}

// --- nvEncodeAPI struct typedefs -------------------------------------------
//
// We define only the fields we set; the rest is reserved padding sized to
// match the header. The header uses `uint32_t reserved[N]` / `void* reserved2[N]`
// trailing blocks.

/// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS`.
#[repr(C)]
pub struct NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
    pub version: u32,
    pub device_type: u32,
    pub device: *mut c_void,
    pub reserved: *mut c_void,
    pub api_version: u32,
    /// `uint32_t reserved1[253]`.
    pub reserved1: [u32; 253],
    /// `void* reserved2[64]`.
    pub reserved2: [*mut c_void; 64],
}

/// `NV_ENC_CONFIG_H264_VUI_PARAMETERS`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NV_ENC_CONFIG_H264_VUI_PARAMETERS {
    pub overscan_info_present_flag: u32,
    pub overscan_info: u32,
    pub video_signal_type_present_flag: u32,
    pub video_format: u32,
    pub video_full_range_flag: u32,
    pub colour_description_present_flag: u32,
    pub colour_primaries: u32,
    pub transfer_characteristics: u32,
    pub colour_matrix: u32,
    pub chroma_sample_location_flag: u32,
    pub chroma_sample_location_top: u32,
    pub chroma_sample_location_bot: u32,
    pub bitstream_restriction_flag: u32,
    pub timing_info_present_flag: u32,
    pub num_unit_in_ticks: u32,
    pub time_scale: u32,
    /// `uint32_t reserved[12]`.
    pub reserved: [u32; 12],
}

/// `NV_ENC_CONFIG_H264`. Only the VUI + idrPeriod/repeatSPSPPS fields are set;
/// the rest is reserved padding sized generously.
/// VERIFY NV2.3b: exact field offsets of idrPeriod / repeatSPSPPS / VUI within
/// NV_ENC_CONFIG_H264.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NV_ENC_CONFIG_H264 {
    /// Bitfield word (enableTemporalSVC, enableStereoMVC, ..., repeatSPSPPS at
    /// bit 12, ..., reservedBitFields). Modeled as a single u32; set bits to
    /// drive repeatSPSPPS etc. [verified vs nvEncodeAPI.h SDK 13 — single word].
    pub flags: u32,
    pub level: u32,
    pub idr_period: u32,
    pub separate_colour_plane_flag: u32,
    pub disable_deblocking_filter_idc: u32,
    pub num_temporal_layers: u32,
    pub sps_id: u32,
    pub pps_id: u32,
    pub adaptive_transform_mode: u32,
    pub fmo_mode: u32,
    pub bdirect_mode: u32,
    pub entropy_coding_mode: u32,
    pub stereo_mode: u32,
    pub intra_refresh_period: u32,
    pub intra_refresh_cnt: u32,
    pub max_num_ref_frames: u32,
    pub slice_mode: u32,
    pub slice_mode_data: u32,
    pub vui_parameters: NV_ENC_CONFIG_H264_VUI_PARAMETERS,
    pub ltr_num_frames: u32,
    pub ltr_trust_mode: u32,
    pub chroma_format_idc: u32,
    pub max_temporal_layers: u32,
    pub use_bframes_as_ref: u32,
    pub num_ref_l0: u32,
    pub num_ref_l1: u32,
    pub output_bit_depth: u32,
    pub input_bit_depth: u32,
    pub tf_level: u32,
    /// `uint32_t reserved1[264]` + `void* reserved2[64]` in the header.
    pub reserved1: [u32; 264],
    pub reserved2: [*mut c_void; 64],
}

/// repeatSPSPPS is bit 12 of the H264 leading bitfield word
/// (enableTemporalSVC:0 .. repeatSPSPPS:12). [verified vs nvEncodeAPI.h SDK 13].
const NV_ENC_H264_FLAG_REPEAT_SPSPPS: u32 = 1 << 12;

/// Union arm holder for `NV_ENC_CONFIG.encodeCodecConfig`. The header union is
/// `{ NV_ENC_CONFIG_H264 h264Config; NV_ENC_CONFIG_HEVC hevcConfig;
///    NV_ENC_CONFIG_AV1 av1Config; uint32 reserved[320]; }`. The H.264 arm
/// (1760 bytes) is larger than the `uint32 reserved[320]` arm (1280 bytes), so
/// the union size IS sizeof(NV_ENC_CONFIG_H264). We model the union as exactly
/// the H.264 arm — no extra padding (the previous `_union_pad[320]` made the
/// union 1280 bytes too large, shifting NV_ENC_CONFIG's reserved tail and
/// breaking the overall layout the driver reads).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NV_ENC_CODEC_CONFIG {
    pub h264: NV_ENC_CONFIG_H264,
}

/// `NV_ENC_RC_PARAMS` — rate control. Field layout transcribed EXACTLY from
/// nvEncodeAPI.h (NVENC SDK 13). `NV_ENC_QP` is `{int32 qpInterP; int32 qpInterB;
/// int32 qpIntra;}` = `[i32; 3]`. The single bitfield word (enableMinQP:1 ..
/// reservedBitFields:15) is modeled as `flags: u32`. This struct is embedded
/// INLINE inside NV_ENC_CONFIG, so its size + every field offset must match the
/// header or the entire config the driver reads is misaligned.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NV_ENC_RC_PARAMS {
    pub version: u32,
    pub rate_control_mode: u32,
    pub const_qp: [i32; 3],
    pub average_bit_rate: u32,
    pub max_bit_rate: u32,
    pub vbv_buffer_size: u32,
    pub vbv_initial_delay: u32,
    /// Bitfield word: enableMinQP:1 .. aqStrength:4 .. reservedBitFields:15.
    pub flags: u32,
    pub min_qp: [i32; 3],
    pub max_qp: [i32; 3],
    pub initial_rc_qp: [i32; 3],
    pub temporallayer_idx_mask: u32,
    pub temporal_layer_qp: [u8; 8],
    pub target_quality: u8,
    pub target_quality_lsb: u8,
    pub lookahead_depth: u16,
    pub low_delay_key_frame_scale: u8,
    pub y_dc_qp_index_offset: i8,
    pub u_dc_qp_index_offset: i8,
    pub v_dc_qp_index_offset: i8,
    pub qp_map_mode: u32,
    pub multi_pass: u32,
    pub alpha_layer_bitrate_ratio: u32,
    pub cb_qp_index_offset: i8,
    pub cr_qp_index_offset: i8,
    pub reserved2: u16,
    pub lookahead_level: u32,
    /// `uint8_t viewBitrateRatios[MAX_NUM_VIEWS_MINUS_1]` (= 7).
    pub view_bitrate_ratios: [u8; 7],
    pub reserved3: u8,
    pub reserved1: u32,
}

/// `NV_ENC_CONFIG`.
#[repr(C)]
pub struct NV_ENC_CONFIG {
    pub version: u32,
    pub profile_guid: GUID,
    pub gop_length: u32,
    pub frame_interval_p: i32,
    pub mono_chrome_encoding: u32,
    pub frame_field_mode: u32,
    pub mv_precision: u32,
    pub rc_params: NV_ENC_RC_PARAMS,
    pub encode_codec_config: NV_ENC_CODEC_CONFIG,
    /// `uint32_t reserved[278]` + `void* reserved2[64]` in the header.
    pub reserved: [u32; 278],
    pub reserved2: [*mut c_void; 64],
}

/// `NV_ENC_PRESET_CONFIG`. The driver fills `presetCfg` for a given
/// codec/preset/tuning via nvEncGetEncodePresetConfigEx. Layout per the header:
/// `{ uint32 version; uint32 reserved; NV_ENC_CONFIG presetCfg;
///    uint32 reserved1[256]; void* reserved2[64]; }`.
#[repr(C)]
pub struct NV_ENC_PRESET_CONFIG {
    pub version: u32,
    pub reserved: u32,
    pub preset_cfg: NV_ENC_CONFIG,
    pub reserved1: [u32; 256],
    pub reserved2: [*mut c_void; 64],
}

/// `NV_ENC_INITIALIZE_PARAMS`, transcribed field-by-field from nvEncodeAPI.h
/// (NVENC SDK 13). Critical layout points (a wrong offset here makes
/// `encodeConfig` point at garbage → INVALID_PARAM at nvEncInitializeEncoder):
///   - after `enablePTD` there is a SINGLE u32 of packed bitfields
///     (reportSliceOffsets:1 .. reservedBitFields:19), modeled as `flags: u32`.
///   - then privDataSize, reserved, privData*, encodeConfig*, maxEncodeWidth,
///     maxEncodeHeight, then `maxMEHintCountsPerBlock[2]` — each element is a
///     u32 bitfield (NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE), so the [2]
///     array is exactly 8 bytes = `[u32; 2]`.
///   - tuningInfo, bufferFormat, numStateBuffers, outputStatsLevel, then the
///     `uint32_t reserved1[284]` + `void* reserved2[64]` tail.
#[repr(C)]
pub struct NV_ENC_INITIALIZE_PARAMS {
    pub version: u32,
    pub encode_guid: GUID,
    pub preset_guid: GUID,
    pub encode_width: u32,
    pub encode_height: u32,
    pub dar_width: u32,
    pub dar_height: u32,
    pub frame_rate_num: u32,
    pub frame_rate_den: u32,
    pub enable_encode_async: u32,
    pub enable_ptd: u32,
    /// Single bitfield word: reportSliceOffsets:1 .. reservedBitFields:19.
    pub flags: u32,
    pub priv_data_size: u32,
    pub reserved: u32,
    pub priv_data: *mut c_void,
    pub encode_config: *mut NV_ENC_CONFIG,
    pub max_encode_width: u32,
    pub max_encode_height: u32,
    /// `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE maxMEHintCountsPerBlock[2]`.
    /// Each element is `{ uint32 bitfield; uint32 reserved1[3]; }` = 16 bytes,
    /// so the [2] array is 32 bytes = `[u32; 8]`. (Getting this wrong shifts
    /// tuningInfo and everything after it — verified vs offsetof in the header:
    /// maxMEHintCountsPerBlock@104, tuningInfo@136.)
    pub max_me_hint_counts_per_block: [u32; 8],
    pub tuning_info: u32,
    pub buffer_format: u32,
    pub num_state_buffers: u32,
    pub output_stats_level: u32,
    /// `uint32_t reserved1[284]` + `void* reserved2[64]`.
    pub reserved1: [u32; 284],
    pub reserved2: [*mut c_void; 64],
}

/// `NV_ENC_RECONFIGURE_PARAMS`, transcribed field-by-field from nvEncodeAPI.h
/// (NVENC SDK 13). VERIFIED vs the header — note the layout differs from a
/// naive `{version, reInitEncodeParams, bitfield}`:
///   typedef struct _NV_ENC_RECONFIGURE_PARAMS {
///       uint32_t                 version;            // @0
///       uint32_t                 reserved;           // @4  (must be 0)
///       NV_ENC_INITIALIZE_PARAMS reInitEncodeParams; // @8  (1800 bytes)
///       uint32_t resetEncoder:1, forceIDR:1, reserved1:30; // @1808
///       uint32_t                 reserved2;          // @1812 (must be 0)
///   };                                               // sizeof = 1816
/// So `re_init_encode_params` is at offset 8 (NOT 4), and there is a trailing
/// `reserved2` word — total size 1816, not 1808. The version macro is
/// `NVENCAPI_STRUCT_VERSION(2) | (1<<31)` (= `struct_ver_cap(2)`).
#[repr(C)]
pub struct NV_ENC_RECONFIGURE_PARAMS {
    pub version: u32,
    /// Header `reserved` (must be 0). `re_init_encode_params` follows at @8.
    pub reserved: u32,
    pub re_init_encode_params: NV_ENC_INITIALIZE_PARAMS,
    /// Packed bitfield word: `resetEncoder:1, forceIDR:1, reserved1:30`.
    pub flags: u32,
    /// Header `reserved2` (must be 0).
    pub reserved2: u32,
}

/// `NV_ENC_CREATE_BITSTREAM_BUFFER`.
#[repr(C)]
pub struct NV_ENC_CREATE_BITSTREAM_BUFFER {
    pub version: u32,
    pub size: u32,
    pub memory_heap: u32,
    pub reserved: u32,
    pub bitstream_buffer: *mut c_void,
    pub bitstream_buffer_ptr: *mut c_void,
    /// `uint32_t reserved1[58]` + `void* reserved2[64]`.
    pub reserved1: [u32; 58],
    pub reserved2: [*mut c_void; 64],
}

/// `NV_ENC_SEQUENCE_PARAM_PAYLOAD`.
#[repr(C)]
pub struct NV_ENC_SEQUENCE_PARAM_PAYLOAD {
    pub version: u32,
    pub in_buffer_size: u32,
    pub sps_id: u32,
    pub pps_id: u32,
    pub sps_pps_buffer: *mut c_void,
    pub out_sps_pps_payload_size: *mut u32,
    /// `uint32_t reserved[250]` + `void* reserved2[64]`.
    pub reserved: [u32; 250],
    pub reserved2: [*mut c_void; 64],
}

/// `NV_ENC_REGISTER_RESOURCE` — byte-verified vs nvEncodeAPI.h (NVENC SDK 13).
/// sizeof = 1536; offsetof registeredResource=32, bufferFormat=40,
/// pInputFencePoint=48, chromaOffset=56. The header has BOTH `chromaOffset[2]`
/// (out) AND `chromaOffsetIn[2]` (in) after pInputFencePoint — the previous
/// layout omitted these 16 bytes (and over-sized reserved1 to [247]/[60]),
/// shifting nothing we read but making the struct the wrong total size.
/// reserved1[244] (976B) + reserved2[61] ptrs (488B) fills 1536 exactly.
#[repr(C)]
pub struct NV_ENC_REGISTER_RESOURCE {
    pub version: u32,
    pub resource_type: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub sub_resource_index: u32,
    pub resource_to_register: *mut c_void,
    pub registered_resource: *mut c_void,
    pub buffer_format: u32,
    pub buffer_usage: u32,
    pub input_fence_point: *mut c_void,
    /// `uint32_t chromaOffset[2]` (out, recon path).
    pub chroma_offset: [u32; 2],
    /// `uint32_t chromaOffsetIn[2]` (in, NVCUVID padded planes).
    pub chroma_offset_in: [u32; 2],
    pub reserved1: [u32; 244],
    pub reserved2: [*mut c_void; 61],
}

/// `NV_ENC_MAP_INPUT_RESOURCE` — byte-verified vs nvEncodeAPI.h (NVENC SDK 13).
/// sizeof = 1544; offsetof mappedResource=24. Layout was already correct:
/// version, subResourceIndex, inputResource*, registeredResource*,
/// mappedResource*, mappedBufferFmt, reserved1[251] (1004B), reserved2[63]*.
#[repr(C)]
pub struct NV_ENC_MAP_INPUT_RESOURCE {
    pub version: u32,
    pub sub_resource_index: u32,
    pub input_resource: *mut c_void,
    pub registered_resource: *mut c_void,
    pub mapped_resource: *mut c_void,
    pub mapped_buffer_fmt: u32,
    pub reserved1: [u32; 251],
    pub reserved2: [*mut c_void; 63],
}

/// `NV_ENC_CODEC_PIC_PARAMS` — the per-picture codec union inside
/// NV_ENC_PIC_PARAMS. The largest arm is `NV_ENC_PIC_PARAMS_H264` (NOT the
/// `uint32 reserved[256]` arm), so the union is 1544 bytes, 8-aligned (it holds
/// pointers). We never set any field for our forced-IDR encode, so a zeroed
/// blob of the exact size + alignment is sufficient and keeps the surrounding
/// NV_ENC_PIC_PARAMS layout byte-exact.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct NV_ENC_CODEC_PIC_PARAMS {
    _bytes: [u8; 1544],
}

/// `NV_ENC_PIC_PARAMS` — byte-verified vs nvEncodeAPI.h (NVENC SDK 13).
/// sizeof = 3360. Key offsets the driver reads: inputBuffer=40, bufferFmt=64,
/// pictureType=72, codecPicParams=80, meHintCountsPerBlock=1624. The previous
/// layout jumped from pictureType straight to reserved1[256]/reserved2[64],
/// OMITTING the 1544-byte codecPicParams union and the entire ME/qp/recon tail
/// — so the struct was the wrong size and `version` was the only valid field.
///
/// Tail after `picture_type` (offsets in bytes):
///   codecPicParams@80 (1544) → meHintCountsPerBlock@1624 (2×16=32)
///   → meExternalHints*@1656, reserved2[7]@1664, reserved5[2]*@1696,
///   qpDeltaMap*@1712, qpDeltaMapSize@1720, reservedBitFields@1724,
///   meHintRefPicDist[2]@1728, reserved4@1732, alphaBuffer*@1736,
///   meExternalSbHints*@1744, meSbHintsCount@1752, stateBufferIdx@1756,
///   outputReconBuffer*@1760, reserved3[284]@1768, reserved6[57]*@2904 → 3360.
#[repr(C)]
pub struct NV_ENC_PIC_PARAMS {
    pub version: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub input_pitch: u32,
    pub encode_pic_flags: u32,
    pub frame_idx: u32,
    pub input_timestamp: u64,
    pub input_duration: u64,
    pub input_buffer: *mut c_void,
    pub output_bitstream: *mut c_void,
    pub completion_event: *mut c_void,
    pub buffer_fmt: u32,
    pub picture_structure: u32,
    pub picture_type: u32,
    /// 4 bytes of tail padding here (pictureType ends @76, union is 8-aligned
    /// @80) are inserted automatically by repr(C).
    pub codec_pic_params: NV_ENC_CODEC_PIC_PARAMS,
    /// `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE meHintCountsPerBlock[2]`.
    /// Each element is one packed u32 bitfield + `uint32 reserved1[3]` = 16
    /// bytes, so the [2] array is 32 bytes = `[u32; 8]`.
    pub me_hint_counts_per_block: [u32; 8],
    pub me_external_hints: *mut c_void,
    pub reserved2: [u32; 7],
    pub reserved5: [*mut c_void; 2],
    pub qp_delta_map: *mut c_void,
    pub qp_delta_map_size: u32,
    pub reserved_bit_fields: u32,
    /// `uint16_t meHintRefPicDist[2]`.
    pub me_hint_ref_pic_dist: [u16; 2],
    pub reserved4: u32,
    pub alpha_buffer: *mut c_void,
    pub me_external_sb_hints: *mut c_void,
    pub me_sb_hints_count: u32,
    pub state_buffer_idx: u32,
    pub output_recon_buffer: *mut c_void,
    pub reserved3: [u32; 284],
    pub reserved6: [*mut c_void; 57],
}

/// `NV_ENC_LOCK_BITSTREAM` — byte-verified vs nvEncodeAPI.h (NVENC SDK 13).
/// sizeof = 1544. The previous layout was badly wrong: it put `frameIdx`
/// directly after the bitfield word and placed `outputBitstream` near the end,
/// whereas the header places `outputBitstream` SECOND (right after the bitfield
/// word) and `sliceOffsets` third. That misaligned the two fields we read —
/// `bitstreamSizeInBytes` (header offset 36) and `bitstreamBufferPtr` (header
/// offset 56) — so we were reading garbage. Exact header order below:
///   version@0, bitfield@4, outputBitstream*@8, sliceOffsets*@16, frameIdx@24,
///   hwEncodeStatus@28, numSlices@32, bitstreamSizeInBytes@36,
///   (4B pad) outputTimeStamp@48, outputDuration@56? — actually outputTimeStamp
///   is 8-aligned at 40 (40%8==0), outputDuration@48, bitstreamBufferPtr@56,
///   pictureType@64 ... outputStatsPtr*@120, frameIdxDisplay@128,
///   reserved1[219]@132, reserved2[63]*@1008, reservedInternal[8]@1512 → 1544.
#[repr(C)]
pub struct NV_ENC_LOCK_BITSTREAM {
    pub version: u32,
    /// Bitfield word: doNotWait:1, ltrFrame:1, getRCStats:1, reservedBitFields:29.
    pub flags: u32,
    /// `void* outputBitstream` — the buffer being locked (header offset 8).
    pub output_bitstream: *mut c_void,
    pub slice_offsets: *mut u32,
    pub frame_idx: u32,
    pub hw_encode_status: u32,
    pub num_slices: u32,
    /// Actual encoded byte count — read by the caller (header offset 36).
    pub bitstream_size_in_bytes: u32,
    pub output_time_stamp: u64,
    pub output_duration: u64,
    /// Pointer to the generated bitstream — read by the caller (header offset 56).
    pub bitstream_buffer_ptr: *mut c_void,
    pub picture_type: u32,
    pub picture_structure: u32,
    pub frame_avg_qp: u32,
    pub frame_satd: u32,
    pub ltr_frame_idx: u32,
    pub ltr_frame_bitmap: u32,
    pub temporal_id: u32,
    pub intra_mb_count: u32,
    pub inter_mb_count: u32,
    pub average_mvx: i32,
    pub average_mvy: i32,
    pub alpha_layer_size_in_bytes: u32,
    pub output_stats_ptr_size: u32,
    pub reserved: u32,
    pub output_stats_ptr: *mut c_void,
    pub frame_idx_display: u32,
    pub reserved1: [u32; 219],
    pub reserved2: [*mut c_void; 63],
    pub reserved_internal: [u32; 8],
}

// --- nvEncodeAPI function-pointer signatures we call -----------------------

type PfnNvEncOpenEncodeSessionEx = unsafe extern "C" fn(
    params: *mut NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
    encoder: *mut *mut c_void,
) -> i32;
type PfnNvEncInitializeEncoder =
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NV_ENC_INITIALIZE_PARAMS) -> i32;
type PfnNvEncGetSequenceParams =
    unsafe extern "C" fn(encoder: *mut c_void, payload: *mut NV_ENC_SEQUENCE_PARAM_PAYLOAD) -> i32;
type PfnNvEncDestroyEncoder = unsafe extern "C" fn(encoder: *mut c_void) -> i32;
type PfnNvEncReconfigureEncoder =
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NV_ENC_RECONFIGURE_PARAMS) -> i32;

/// `PNVENCGETENCODEPRESETCONFIGEX` — note GUIDs are passed BY VALUE (the C
/// signature is `(void* encoder, GUID encodeGUID, GUID presetGUID,
/// NV_ENC_TUNING_INFO tuningInfo, NV_ENC_PRESET_CONFIG* presetConfig)`). `GUID`
/// is `#[repr(C)]` 16 bytes, so by-value passing matches the SysV ABI the
/// driver expects.
type PfnNvEncGetEncodePresetConfigEx = unsafe extern "C" fn(
    encoder: *mut c_void,
    encode_guid: GUID,
    preset_guid: GUID,
    tuning_info: u32,
    preset_config: *mut NV_ENC_PRESET_CONFIG,
) -> i32;

type PfnNvEncRegisterResource =
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NV_ENC_REGISTER_RESOURCE) -> i32;
type PfnNvEncUnregisterResource =
    unsafe extern "C" fn(encoder: *mut c_void, registered_resource: *mut c_void) -> i32;
type PfnNvEncMapInputResource =
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NV_ENC_MAP_INPUT_RESOURCE) -> i32;
type PfnNvEncUnmapInputResource =
    unsafe extern "C" fn(encoder: *mut c_void, mapped_resource: *mut c_void) -> i32;
type PfnNvEncCreateBitstreamBuffer =
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NV_ENC_CREATE_BITSTREAM_BUFFER) -> i32;
type PfnNvEncDestroyBitstreamBuffer =
    unsafe extern "C" fn(encoder: *mut c_void, bitstream_buffer: *mut c_void) -> i32;
type PfnNvEncEncodePicture =
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NV_ENC_PIC_PARAMS) -> i32;
type PfnNvEncLockBitstream =
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NV_ENC_LOCK_BITSTREAM) -> i32;
type PfnNvEncUnlockBitstream =
    unsafe extern "C" fn(encoder: *mut c_void, bitstream_buffer: *mut c_void) -> i32;

/// Entry point exported by `libnvidia-encode.so.1`.
type NvEncodeApiCreateInstance = unsafe extern "C" fn(*mut NV_ENCODE_API_FUNCTION_LIST) -> i32;

// ===========================================================================
// EglCtx — gbm renderD128 + EGL GBM-platform display + initialize. Reuses the
// bring-up proven in examples/egl_cuda_interop_spike.rs. Returns None cleanly
// on any failure (no GPU render node / no EGL / init failure).
// ===========================================================================

const EGL_PLATFORM_GBM_KHR: khronos_egl::Enum = 0x31D7;

// Reuse the crate's existing gbm FFI (dmabuf::gbm_ffi) rather than redeclaring
// the symbols, so the typed `*mut GbmDevice` signatures don't clash. These are
// thin wrappers that cast the opaque crate type to/from `*mut c_void`.
use crate::dmabuf::gbm_ffi;

unsafe fn gbm_create_device(fd: i32) -> *mut c_void {
    gbm_ffi::gbm_create_device(fd) as *mut c_void
}
unsafe fn gbm_device_destroy(dev: *mut c_void) {
    gbm_ffi::gbm_device_destroy(dev as *mut _);
}

/// Holds the EGL display + the dmabuf-import extension fns. The dynamic EGL
/// instance (`_egl`) must outlive any proc-address-derived fn pointers, so it
/// is kept here; `_gbm`/`_drm_fd` likewise back the display.
pub struct EglCtx {
    display: *mut c_void,
    create_image: EglCreateImageKHR,
    destroy_image: Option<EglDestroyImageKHR>,
    _egl: khronos_egl::DynamicInstance<khronos_egl::EGL1_5>,
    _gbm: *mut c_void,
    _drm_fd: i32,
}

impl EglCtx {
    /// Open renderD128 → gbm device → EGL GBM-platform display → initialize.
    pub fn open() -> Option<Self> {
        let drm_fd = unsafe {
            libc::open(
                c"/dev/dri/renderD128".as_ptr() as *const c_char,
                libc::O_RDWR | libc::O_CLOEXEC,
            )
        };
        if drm_fd < 0 {
            tracing::warn!("cuda_nvenc: open /dev/dri/renderD128 failed (no GPU render node?)");
            return None;
        }
        let gbm = unsafe { gbm_create_device(drm_fd) };
        if gbm.is_null() {
            tracing::warn!("cuda_nvenc: gbm_create_device failed");
            unsafe { libc::close(drm_fd) };
            return None;
        }

        let lib = match unsafe { libloading::Library::new("libEGL.so.1") } {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("cuda_nvenc: failed to load libEGL.so.1: {e}");
                unsafe {
                    gbm_device_destroy(gbm);
                    libc::close(drm_fd);
                }
                return None;
            }
        };
        let egl = match unsafe {
            khronos_egl::DynamicInstance::<khronos_egl::EGL1_5>::load_required_from(lib)
        } {
            Ok(i) => i,
            Err(e) => {
                tracing::warn!("cuda_nvenc: EGL dynamic load failed: {e:?}");
                unsafe {
                    gbm_device_destroy(gbm);
                    libc::close(drm_fd);
                }
                return None;
            }
        };
        let display = match unsafe {
            egl.get_platform_display(EGL_PLATFORM_GBM_KHR, gbm, &[khronos_egl::ATTRIB_NONE])
        } {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("cuda_nvenc: get_platform_display(GBM) failed: {e:?}");
                unsafe {
                    gbm_device_destroy(gbm);
                    libc::close(drm_fd);
                }
                return None;
            }
        };
        if let Err(e) = egl.initialize(display) {
            tracing::warn!("cuda_nvenc: eglInitialize failed: {e:?}");
            unsafe {
                gbm_device_destroy(gbm);
                libc::close(drm_fd);
            }
            return None;
        }
        let dpy_ptr = display.as_ptr();

        let create_image: EglCreateImageKHR = match egl.get_proc_address("eglCreateImageKHR") {
            Some(p) => unsafe {
                std::mem::transmute::<
                    extern "system" fn(),
                    unsafe extern "C" fn(
                        *mut libc::c_void,
                        *mut libc::c_void,
                        u32,
                        *mut libc::c_void,
                        *const i32,
                    ) -> *mut libc::c_void,
                >(p)
            },
            None => {
                tracing::warn!("cuda_nvenc: eglCreateImageKHR not exposed");
                unsafe {
                    gbm_device_destroy(gbm);
                    libc::close(drm_fd);
                }
                return None;
            }
        };
        let destroy_image: Option<EglDestroyImageKHR> = egl
            .get_proc_address("eglDestroyImageKHR")
            .map(|p| unsafe { std::mem::transmute(p) });

        Some(EglCtx {
            display: dpy_ptr,
            create_image,
            destroy_image,
            _egl: egl,
            _gbm: gbm,
            _drm_fd: drm_fd,
        })
    }
}

impl Drop for EglCtx {
    fn drop(&mut self) {
        // Terminate the EGL display first (before GBM/fd cleanup). The
        // DynamicInstance (`_egl`) drops after this body by field order, so
        // calling through it here is valid.
        unsafe {
            if !self.display.is_null() {
                let dpy = khronos_egl::Display::from_ptr(self.display);
                let _ = self._egl.terminate(dpy);
            }
            if !self._gbm.is_null() {
                gbm_device_destroy(self._gbm);
            }
            if self._drm_fd >= 0 {
                libc::close(self._drm_fd);
            }
        }
    }
}

// ===========================================================================
// CudaCtx — CudaLib::load + cuDeviceGet(0) + cuCtxCreate_v2.
// ===========================================================================

/// Owns the CUDA driver lib + a primary context.
pub struct CudaCtx {
    lib: CudaLib,
    ctx: *mut c_void,
}

impl Drop for CudaCtx {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: ctx was created by cuCtxCreate_v2 and is non-null.
            // `lib` (and its `_lib: Library`) is a field of CudaCtx and drops
            // after this body, so the fn pointer remains valid here.
            let _ = unsafe { (self.lib.cu_ctx_destroy_v2)(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

impl CudaCtx {
    /// Load libcuda, init, get device 0, create a context.
    pub fn open() -> Option<Self> {
        let lib = CudaLib::load()?;
        let mut dev: i32 = 0;
        let rc = unsafe { (lib.device_get)(&mut dev, 0) };
        if rc != 0 {
            tracing::warn!("cuda_nvenc: cuDeviceGet(0) -> {}", lib.err(rc));
            return None;
        }
        let mut ctx: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { (lib.ctx_create)(&mut ctx, 0, dev) };
        if rc != 0 || ctx.is_null() {
            tracing::warn!("cuda_nvenc: cuCtxCreate_v2 -> {}", lib.err(rc));
            return None;
        }
        Some(CudaCtx { lib, ctx })
    }

    /// Load the BGRA→NV12 CSC module from the embedded PTX.
    ///
    /// The embedded PTX is a placeholder until NV2.3b generates the real PTX
    /// with nvcc. On a host without a GPU this is never called (CudaCtx::open
    /// returns None first), so a runtime cuModuleLoadData failure on the
    /// placeholder is irrelevant offline. On a GPU host NV2.3b replaces the
    /// PTX and this succeeds.
    pub fn load_csc_module(&self) -> Option<CuModule> {
        let mut module: *mut c_void = std::ptr::null_mut();
        // SAFETY: EMBEDDED_PTX is a NUL-terminated byte slice baked in at
        // compile time; cuModuleLoadData expects a NUL-terminated PTX/cubin.
        let rc = unsafe {
            (self.lib.module_load_data)(&mut module, EMBEDDED_PTX.as_ptr() as *const c_void)
        };
        if rc != 0 || module.is_null() {
            tracing::warn!(
                "cuda_nvenc: cuModuleLoadData (CSC PTX) -> {}",
                self.lib.err(rc)
            );
            return None;
        }
        let mut func: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.lib.module_get_function)(&mut func, module, c"argb_to_nv12_bt709".as_ptr())
        };
        if rc != 0 || func.is_null() {
            tracing::warn!(
                "cuda_nvenc: cuModuleGetFunction(argb_to_nv12_bt709) -> {}",
                self.lib.err(rc)
            );
            // Best-effort: unload the module to avoid leaking it.
            let _ = unsafe { (self.lib.module_unload)(module) };
            return None;
        }
        Some(CuModule {
            module,
            func,
            module_unload: self.lib.module_unload,
        })
    }
}

/// A loaded CUDA module + the CSC kernel function handle. NV2.3 populates this.
pub struct CuModule {
    module: *mut c_void,
    func: *mut c_void,
    /// Fn pointer copy for use in Drop (avoids borrowing CudaLib from the
    /// recorder, which may have a different lifetime). Fn pointers are Copy.
    module_unload: CuModuleUnload,
}

impl Drop for CuModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: module is a valid CUmodule handle obtained from
            // cuModuleLoadData; the CUDA context (CudaCtx) is still alive
            // because cu_module is declared before cuda in CudaNvencRecorder's
            // field list (Rust drops in declaration order).
            let _ = unsafe { (self.module_unload)(self.module) };
            self.module = std::ptr::null_mut();
        }
    }
}

// ===========================================================================
// NvencSession — NvEncodeAPICreateInstance + OpenEncodeSessionEx(CUDA) +
// InitializeEncoder(H.264 preset P3, low-latency tuning, BT.709 VUI). Drop
// calls nvEncDestroyEncoder.
// ===========================================================================

/// Owns the dlopened libnvidia-encode lib, the filled function list, and the
/// opened encoder handle. The `_lib` must outlive `funcs`/`encoder`.
pub struct NvencSession {
    _lib: libloading::Library,
    funcs: NV_ENCODE_API_FUNCTION_LIST,
    encoder: *mut c_void,
    width: u32,
    height: u32,
    /// Output bitstream buffer handle from nvEncCreateBitstreamBuffer; the
    /// per-frame encode reuses this single buffer. Destroyed in Drop.
    bitstream: *mut c_void,
}

impl NvencSession {
    /// Default viewer target bitrate the encoder opens at (1080p60). The live
    /// reconfigure path can move off this without re-opening the encoder.
    const VIEWER_BITRATE_BPS: u32 = 8_000_000; // GCC initial; BWE adapts ~1.5–12 Mbps live.
    const VIEWER_FPS: u32 = 60;
    /// Infinite GOP: emit IDRs ONLY on demand — the first frame, and whenever a
    /// viewer requests one via RTCP PLI (now forwarded to force_idr by the
    /// bridge). A fixed 2s GOP used to inject a periodic full keyframe whose
    /// bitrate spike delayed a burst of frames every 2s → the periodic stutter
    /// (and the jitter-buffer "speed-up" as it drained the backlog). With PLI
    /// recovery wired, the periodic keyframe is pure downside, so drop it.
    /// 0xFFFFFFFF == NV_ENC_INFINITE_GOPLENGTH [verified vs nvEncodeAPI.h].
    const VIEWER_GOP_LENGTH: u32 = 0xFFFF_FFFF;

    /// Apply the viewer's encode policy onto a driver-filled (preset) config:
    /// H.264 High profile, IPP (no B-frames), a finite GOP / periodic IDR,
    /// CBR at `bps`, single-frame VBV, AQ, and the BT.709 VUI. Called from BOTH
    /// `open` (with the default bitrate) and `reconfigure_bitrate` (with the new
    /// bitrate) so the only thing that changes on a live reconfigure is the
    /// bitrate — everything else stays byte-identical, which is required or the
    /// driver resets the other settings.
    ///
    /// Caller must have already set `config.version` (the driver fills the
    /// embedded `rc_params.version` via get-preset; we preserve it and only set
    /// the policy fields).
    fn apply_viewer_encode_config(config: &mut NV_ENC_CONFIG, bps: u32) {
        config.profile_guid = NV_ENC_H264_PROFILE_HIGH_GUID;
        // Infinite GOP: IDRs on demand only (first frame + PLI), no periodic spike.
        config.gop_length = Self::VIEWER_GOP_LENGTH;
        config.frame_interval_p = 1; // IPP, no B-frames (required for low latency)

        // Rate control: CBR with an explicit budget. The preset's default RC
        // mode is left to the driver (often CONSTQP, effectively unbounded at
        // motion); pin it so the bitrate can never saturate the link. Keep the
        // preset-filled `version` and override only the policy fields.
        let rc = &mut config.rc_params;
        rc.rate_control_mode = NV_ENC_PARAMS_RC_CBR;
        rc.average_bit_rate = bps;
        rc.max_bit_rate = bps;
        // Single-frame VBV: one frame's worth of bits at the target fps. The
        // tightest low-latency bound — the encoder can't burst above the
        // per-frame budget, smoothing delivery over a constrained uplink.
        rc.vbv_buffer_size = bps / Self::VIEWER_FPS;
        rc.vbv_initial_delay = bps / Self::VIEWER_FPS;
        // Never multi-pass under low-latency tuning (a high preset + 2-pass
        // spikes latency to ~37 frames); pin it off in case the P5 preset
        // default enabled it.
        rc.multi_pass = NV_ENC_MULTI_PASS_DISABLED;
        // Scale down the keyframe QP bump so on-demand IDRs don't blow the tight
        // single-frame VBV.
        rc.low_delay_key_frame_scale = 1;
        // Spatial + temporal AQ (strength 8). Temporal AQ is the best NVENC knob
        // for static text. OR into the preset-filled flags (preserve enableMinQP
        // etc. the driver set).
        rc.flags |= viewer_rc_flags(8);

        let h264 = &mut config.encode_codec_config.h264;
        // Match idrPeriod to gopLength (infinite — IDRs on demand only).
        // REPEAT_SPSPPS attaches SPS/PPS to every IDR so a PLI-driven keyframe
        // is self-contained and a late joiner can decode it immediately.
        h264.idr_period = Self::VIEWER_GOP_LENGTH;
        h264.flags |= NV_ENC_H264_FLAG_REPEAT_SPSPPS; // repeatSPSPPS bit
        h264.vui_parameters.video_signal_type_present_flag = 1;
        h264.vui_parameters.video_format = 5; // unspecified
        h264.vui_parameters.video_full_range_flag = 0; // limited range
        h264.vui_parameters.colour_description_present_flag = 1;
        h264.vui_parameters.colour_primaries = 1; // BT.709
        h264.vui_parameters.transfer_characteristics = 1; // BT.709
        h264.vui_parameters.colour_matrix = 1; // BT.709
    }

    /// Bring up an NVENC H.264 session bound to the given CUcontext.
    pub fn open(cuctx: *mut c_void, width: u32, height: u32) -> Option<Self> {
        let lib = match unsafe { libloading::Library::new("libnvidia-encode.so.1") } {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("cuda_nvenc: failed to dlopen libnvidia-encode.so.1: {e}");
                return None;
            }
        };
        // SAFETY: signature matches the NVENC API entry point.
        let create_instance: NvEncodeApiCreateInstance =
            match unsafe { lib.get::<NvEncodeApiCreateInstance>(b"NvEncodeAPICreateInstance\0") } {
                Ok(s) => *s,
                Err(e) => {
                    tracing::warn!("cuda_nvenc: NvEncodeAPICreateInstance symbol missing: {e}");
                    return None;
                }
            };

        let mut funcs = NV_ENCODE_API_FUNCTION_LIST::zeroed();
        funcs.version = nv_encode_api_function_list_ver();
        let rc = unsafe { create_instance(&mut funcs) };
        tracing::debug!("cuda_nvenc: NvEncodeAPICreateInstance -> status={rc}");
        if rc != 0 {
            tracing::warn!("cuda_nvenc: NvEncodeAPICreateInstance -> {rc}");
            return None;
        }

        // OpenEncodeSessionEx(deviceType=CUDA, device=CUcontext).
        let open_ex_ptr = funcs.slot(SLOT_OPEN_ENCODE_SESSION_EX);
        if open_ex_ptr.is_null() {
            tracing::warn!("cuda_nvenc: nvEncOpenEncodeSessionEx slot is null");
            return None;
        }
        let open_ex: PfnNvEncOpenEncodeSessionEx = unsafe { std::mem::transmute(open_ex_ptr) };

        let mut open_params: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS = unsafe { std::mem::zeroed() };
        open_params.version = nv_enc_open_encode_session_ex_params_ver();
        open_params.device_type = NV_ENC_DEVICE_TYPE_CUDA;
        open_params.device = cuctx;
        open_params.api_version = NVENCAPI_VERSION;

        let mut encoder: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { open_ex(&mut open_params, &mut encoder) };
        tracing::debug!(
            "cuda_nvenc: nvEncOpenEncodeSessionEx -> status={rc} encoder={:?}",
            encoder
        );
        if rc != 0 || encoder.is_null() {
            tracing::warn!("cuda_nvenc: nvEncOpenEncodeSessionEx -> {rc}");
            return None;
        }

        // -- Build the encode config via the PRESET-CONFIG flow ---------------
        // Hand-building NV_ENC_CONFIG is fragile (any mis-set field →
        // INVALID_PARAM). Instead ask the driver to fill a known-good config
        // for (H264, P3, low-latency), then override only the few fields we
        // care about. Both the outer PRESET_CONFIG version AND the embedded
        // presetCfg version must be set BEFORE the call (header requirement).
        let mut preset_config: NV_ENC_PRESET_CONFIG = unsafe { std::mem::zeroed() };
        preset_config.version = nv_enc_preset_config_ver();
        preset_config.preset_cfg.version = nv_enc_config_ver();

        let get_preset_ptr = funcs.slot(SLOT_GET_ENCODE_PRESET_CONFIG_EX);
        if get_preset_ptr.is_null() {
            tracing::warn!("cuda_nvenc: nvEncGetEncodePresetConfigEx slot is null");
            Self::destroy_encoder(&funcs, encoder);
            return None;
        }
        let get_preset: PfnNvEncGetEncodePresetConfigEx =
            unsafe { std::mem::transmute(get_preset_ptr) };
        let rc = unsafe {
            get_preset(
                encoder,
                NV_ENC_CODEC_H264_GUID,
                NV_ENC_PRESET_P5_GUID,
                NV_ENC_TUNING_INFO_LOW_LATENCY,
                &mut preset_config,
            )
        };
        tracing::debug!("cuda_nvenc: nvEncGetEncodePresetConfigEx -> status={rc}");
        if rc != 0 {
            tracing::warn!("cuda_nvenc: nvEncGetEncodePresetConfigEx -> {rc}");
            Self::destroy_encoder(&funcs, encoder);
            return None;
        }

        // Override on the driver-filled config: H.264 High profile, P-frames
        // (no B-frames — low-latency), periodic IDR safety net, repeat SPS/PPS,
        // BT.709 VUI, and an explicit CBR bitrate cap.
        //
        // Why a bitrate cap: the viewer previously forced EVERY frame to be an
        // IDR with SPS/PPS, which at motion produced a huge bitrate that
        // saturated the link (esp. over TURN/mobile), so the browser dropped
        // frames → perceived <10 fps. We now emit P-frames (Part 1 threads
        // `force_idr` so only the first frame / PLI / heartbeat are IDRs) and
        // pin the rate-control mode to CBR with a fixed average/max bitrate so
        // the encoder spends a bounded, link-friendly bit budget per second.
        //
        // Struct field offsets relied on (verified vs nvEncodeAPI.h SDK 13):
        //   NV_ENC_CONFIG: gopLength@8, frameIntervalP@12, rcParams@... (the
        //     rcParams sub-struct is embedded inline after mvPrecision).
        //   NV_ENC_RC_PARAMS: version@0, rateControlMode@4, constQP@8 (12B),
        //     averageBitRate@20, maxBitRate@24, vbvBufferSize@28,
        //     vbvInitialDelay@32. All match the Rust struct field order.
        //   NV_ENC_CONFIG_H264: idrPeriod@8 (after the leading bitfield word
        //     @0 and `level`@4).
        //
        // The override logic lives in `apply_viewer_encode_config` so the live
        // bitrate reconfigure path (reconfigure_bitrate) applies BYTE-IDENTICAL
        // settings — any drift there would silently reset profile/GOP/AQ/VUI.
        let config = &mut preset_config.preset_cfg;
        config.version = nv_enc_config_ver();
        Self::apply_viewer_encode_config(config, Self::VIEWER_BITRATE_BPS);

        let mut init_params: NV_ENC_INITIALIZE_PARAMS = unsafe { std::mem::zeroed() };
        init_params.version = nv_enc_initialize_params_ver();
        init_params.encode_guid = NV_ENC_CODEC_H264_GUID;
        init_params.preset_guid = NV_ENC_PRESET_P5_GUID;
        init_params.encode_width = width;
        init_params.encode_height = height;
        init_params.dar_width = width;
        init_params.dar_height = height;
        init_params.frame_rate_num = Self::VIEWER_FPS;
        init_params.frame_rate_den = 1;
        init_params.enable_encode_async = 0;
        init_params.enable_ptd = 1;
        init_params.max_encode_width = width;
        init_params.max_encode_height = height;
        init_params.tuning_info = NV_ENC_TUNING_INFO_LOW_LATENCY;
        init_params.encode_config = config as *mut NV_ENC_CONFIG;

        let init_ptr = funcs.slot(SLOT_INITIALIZE_ENCODER);
        if init_ptr.is_null() {
            tracing::warn!("cuda_nvenc: nvEncInitializeEncoder slot is null");
            Self::destroy_encoder(&funcs, encoder);
            return None;
        }
        let initialize: PfnNvEncInitializeEncoder = unsafe { std::mem::transmute(init_ptr) };
        let rc = unsafe { initialize(encoder, &mut init_params) };
        tracing::debug!("cuda_nvenc: nvEncInitializeEncoder -> status={rc}");
        if rc != 0 {
            tracing::warn!("cuda_nvenc: nvEncInitializeEncoder -> {rc}");
            Self::destroy_encoder(&funcs, encoder);
            return None;
        }

        // Create the (single, reused) output bitstream buffer.
        let create_bs_ptr = funcs.slot(SLOT_CREATE_BITSTREAM_BUFFER);
        if create_bs_ptr.is_null() {
            tracing::warn!("cuda_nvenc: nvEncCreateBitstreamBuffer slot is null");
            Self::destroy_encoder(&funcs, encoder);
            return None;
        }
        let create_bs: PfnNvEncCreateBitstreamBuffer =
            unsafe { std::mem::transmute(create_bs_ptr) };
        let mut create_bs_params: NV_ENC_CREATE_BITSTREAM_BUFFER = unsafe { std::mem::zeroed() };
        create_bs_params.version = nv_enc_create_bitstream_buffer_ver();
        let rc = unsafe { create_bs(encoder, &mut create_bs_params) };
        tracing::debug!(
            "cuda_nvenc: nvEncCreateBitstreamBuffer -> status={rc} buf={:?}",
            create_bs_params.bitstream_buffer
        );
        if rc != 0 || create_bs_params.bitstream_buffer.is_null() {
            tracing::warn!("cuda_nvenc: nvEncCreateBitstreamBuffer -> {rc}");
            Self::destroy_encoder(&funcs, encoder);
            return None;
        }
        let bitstream = create_bs_params.bitstream_buffer;

        Some(NvencSession {
            _lib: lib,
            funcs,
            encoder,
            width,
            height,
            bitstream,
        })
    }

    /// Change the live encoder's target bitrate to `bps` (CBR) via
    /// nvEncReconfigureEncoder, so GCC can adapt the bitrate at runtime without
    /// tearing down + re-opening the session (which would drop frames).
    ///
    /// Re-derives the FULL encode config the way `open` does — a fresh
    /// driver-filled preset config for (H264, P5, low-latency), then
    /// `apply_viewer_encode_config(config, bps)`. The reconfigure must carry the
    /// entire config (identical except the bitrate) or NVENC resets the other
    /// settings. We force an IDR (forceIDR bit) so the new bitrate takes effect
    /// cleanly on a keyframe, but do NOT set resetEncoder (a full RC-state reset
    /// would visibly hitch the stream).
    ///
    /// Returns true on success (status 0). All the structs below are stack
    /// locals that outlive the single synchronous FFI call, so the
    /// `encode_config` pointer stays valid for the duration.
    pub fn reconfigure_bitrate(&mut self, bps: u32) -> bool {
        // Fresh driver-filled preset config (same flow as open()).
        let mut preset_config: NV_ENC_PRESET_CONFIG = unsafe { std::mem::zeroed() };
        preset_config.version = nv_enc_preset_config_ver();
        preset_config.preset_cfg.version = nv_enc_config_ver();

        let get_preset_ptr = self.funcs.slot(SLOT_GET_ENCODE_PRESET_CONFIG_EX);
        if get_preset_ptr.is_null() {
            tracing::warn!("cuda_nvenc: reconfigure: nvEncGetEncodePresetConfigEx slot is null");
            return false;
        }
        let get_preset: PfnNvEncGetEncodePresetConfigEx =
            unsafe { std::mem::transmute(get_preset_ptr) };
        let rc = unsafe {
            get_preset(
                self.encoder,
                NV_ENC_CODEC_H264_GUID,
                NV_ENC_PRESET_P5_GUID,
                NV_ENC_TUNING_INFO_LOW_LATENCY,
                &mut preset_config,
            )
        };
        if rc != 0 {
            tracing::warn!("cuda_nvenc: reconfigure: nvEncGetEncodePresetConfigEx -> {rc}");
            return false;
        }

        let config = &mut preset_config.preset_cfg;
        config.version = nv_enc_config_ver();
        Self::apply_viewer_encode_config(config, bps);

        // Re-init params identical to open()'s, pointing at the live `config`.
        let mut init_params: NV_ENC_INITIALIZE_PARAMS = unsafe { std::mem::zeroed() };
        init_params.version = nv_enc_initialize_params_ver();
        init_params.encode_guid = NV_ENC_CODEC_H264_GUID;
        init_params.preset_guid = NV_ENC_PRESET_P5_GUID;
        init_params.encode_width = self.width;
        init_params.encode_height = self.height;
        init_params.dar_width = self.width;
        init_params.dar_height = self.height;
        init_params.frame_rate_num = Self::VIEWER_FPS;
        init_params.frame_rate_den = 1;
        init_params.enable_encode_async = 0;
        init_params.enable_ptd = 1;
        init_params.max_encode_width = self.width;
        init_params.max_encode_height = self.height;
        init_params.tuning_info = NV_ENC_TUNING_INFO_LOW_LATENCY;
        init_params.encode_config = config as *mut NV_ENC_CONFIG;

        let mut reconfigure_params: NV_ENC_RECONFIGURE_PARAMS = unsafe { std::mem::zeroed() };
        reconfigure_params.version = nv_enc_reconfigure_params_ver();
        reconfigure_params.re_init_encode_params = init_params;
        // flags = 0: neither resetEncoder (bit 0) nor forceIDR (bit 1). NVENC
        // applies the new bitrate to subsequent P-frames with no keyframe — so
        // GCC's rapid early bitrate steps don't each emit a big IDR (those
        // spikes were the "choppy at startup until GCC settles" stutter). A
        // viewer that needs a keyframe still gets one via PLI / the periodic GOP.
        reconfigure_params.flags = 0;
        // The moved `init_params.encode_config` still points at `config`, a live
        // stack local in THIS scope that outlives the FFI call below — fine.

        let reconfigure_ptr = self.funcs.slot(SLOT_RECONFIGURE_ENCODER);
        if reconfigure_ptr.is_null() {
            tracing::warn!("cuda_nvenc: reconfigure: nvEncReconfigureEncoder slot is null");
            return false;
        }
        let reconfigure: PfnNvEncReconfigureEncoder =
            unsafe { std::mem::transmute(reconfigure_ptr) };
        let rc = unsafe { reconfigure(self.encoder, &mut reconfigure_params) };
        tracing::debug!("cuda_nvenc: nvEncReconfigureEncoder(bps={bps}) -> status={rc}");
        if rc != 0 {
            tracing::warn!("cuda_nvenc: nvEncReconfigureEncoder -> {rc}");
            return false;
        }
        true
    }

    /// Fetch the SPS/PPS payload (codec_private) via nvEncGetSequenceParams.
    pub fn sequence_params(&self) -> Option<Vec<u8>> {
        let get_ptr = self.funcs.slot(SLOT_GET_SEQUENCE_PARAMS);
        if get_ptr.is_null() {
            tracing::warn!("cuda_nvenc: nvEncGetSequenceParams slot is null");
            return None;
        }
        let get_seq: PfnNvEncGetSequenceParams = unsafe { std::mem::transmute(get_ptr) };

        let mut buf = vec![0u8; 1024];
        let mut out_size: u32 = 0;
        let mut payload: NV_ENC_SEQUENCE_PARAM_PAYLOAD = unsafe { std::mem::zeroed() };
        payload.version = nv_enc_sequence_param_payload_ver();
        payload.in_buffer_size = buf.len() as u32;
        payload.sps_pps_buffer = buf.as_mut_ptr() as *mut c_void;
        payload.out_sps_pps_payload_size = &mut out_size;

        let rc = unsafe { get_seq(self.encoder, &mut payload) };
        if rc != 0 {
            tracing::warn!("cuda_nvenc: nvEncGetSequenceParams -> {rc}");
            return None;
        }
        let n = (out_size as usize).min(buf.len());
        buf.truncate(n);
        Some(buf)
    }

    /// Best-effort nvEncDestroyEncoder via the function list.
    fn destroy_encoder(funcs: &NV_ENCODE_API_FUNCTION_LIST, encoder: *mut c_void) {
        if encoder.is_null() {
            return;
        }
        let ptr = funcs.slot(SLOT_DESTROY_ENCODER);
        if ptr.is_null() {
            return;
        }
        let destroy: PfnNvEncDestroyEncoder = unsafe { std::mem::transmute(ptr) };
        let _ = unsafe { destroy(encoder) };
    }
}

impl Drop for NvencSession {
    fn drop(&mut self) {
        // Destroy the bitstream buffer before the encoder it belongs to.
        if !self.bitstream.is_null() {
            let ptr = self.funcs.slot(SLOT_DESTROY_BITSTREAM_BUFFER);
            if !ptr.is_null() {
                let destroy_bs: PfnNvEncDestroyBitstreamBuffer =
                    unsafe { std::mem::transmute(ptr) };
                let _ = unsafe { destroy_bs(self.encoder, self.bitstream) };
            }
            self.bitstream = std::ptr::null_mut();
        }
        Self::destroy_encoder(&self.funcs, self.encoder);
        self.encoder = std::ptr::null_mut();
    }
}

// ===========================================================================
// CudaNvencRecorder — the public NV2 recorder. Mirrors VkRecorder's
// try_new/codec_private shape. Per the spec, field DROP ORDER is:
//   nvenc → cu_module → cuda → egl
// so the fields are DECLARED in that order (Rust drops in declaration order).
// ===========================================================================

pub struct CudaNvencRecorder {
    nvenc: NvencSession,
    cu_module: Option<CuModule>,
    cuda: CudaCtx,
    egl: EglCtx,
    width: u32,
    height: u32,
    codec_private: Vec<u8>,
    /// Persistent CUDA devptr holding the NV12 output of the CSC kernel; the
    /// CUDA-devptr input NVENC encodes from. Allocated in `try_new`, freed in
    /// `Drop` (BEFORE `cuda`'s context is torn down — see field/Drop order).
    nv12_devptr: u64,
    /// Row pitch of `nv12_devptr` (width rounded up to NVENC's 256-byte align).
    nv12_pitch: u32,
    /// Persistent INPUT cuArray (ARGB8888 / BGRA byte order) for the host-pixels
    /// path (`encode_pixels`). 0 = unallocated; lazily created on the first
    /// `encode_pixels` call sized to (self.width × self.height). It is a CUarray
    /// handle (opaque pointer) stored as a raw pointer value; freed in `Drop`
    /// BEFORE `cuda`'s context is torn down (same discipline as `nv12_devptr`).
    input_array: *mut c_void,
    /// True once at least one real frame (dmabuf or host-pixels) has been fully
    /// CSC'd into `nv12_devptr` and NVENC-encoded. `reencode_last` (the idle
    /// heartbeat) refuses to run until this is set, since before the first real
    /// encode `nv12_devptr` holds only the uninitialized `cuMemAlloc` contents.
    has_encoded: bool,
    /// Monotonic counter for the optional NV12 debug capture harness
    /// (`WAYMUX_NV12_DUMP_DIR`). Bumped on every dump-eligible call; used both
    /// to rate-limit (dump every Nth) and to name the PNG files. Zero cost when
    /// the env var is unset.
    dump_counter: u64,
    /// Latch so the oversized-dmabuf rejection (audit M6 / #14) logs at most
    /// once per recorder. A hostile or buggy inner Wayland client could submit
    /// oversized frames every tick; without this latch the warn would spam the
    /// log. Set true on the first rejection.
    oversize_rejected_logged: bool,
}

/// Host-side, GPU-free safety guard for the CSC kernel write bounds (audit M6 /
/// finding #14). The NV12 output buffer (`nv12_devptr`) is allocated ONCE in
/// `try_new` sized to the recorder's fixed `width × height` at pitch
/// `nv12_pitch`. The CSC kernel, however, is launched with caller-supplied
/// `width`/`height`/`stride` that on the dmabuf path originate from an untrusted
/// inner Wayland client. If any of those exceed the allocation the kernel writes
/// out of bounds in the shared per-session encoder context — a device-side heap
/// overflow.
///
/// This is the pure decision used by both the dmabuf entry (`encode_dmabuf`) and
/// the universal chokepoint (`encode_from_cuarray`). It REJECTS (returns `None`)
/// any frame whose dimensions exceed the allocation rather than clamping: a
/// genuinely wrong-sized frame produces garbage output anyway, and rejecting is
/// the simplest guarantee that no kernel launch can ever write past the fixed
/// NV12 buffer. Boundary-equal dimensions (`== max`) are allowed. A request of
/// `stride == 0` is treated as "unspecified" and accepted (the dmabuf path's
/// stride is the source row pitch, not a destination bound; the destination
/// pitch is always `nv12_pitch`).
///
/// Mirrors the existing host-pixels clamp in `encode_pixels`, but as a reject so
/// it can sit at the kernel-launch chokepoint that both paths funnel through.
fn safe_encode_dims(
    req_w: u32,
    req_h: u32,
    req_stride: u32,
    max_w: u32,
    max_h: u32,
    max_pitch: u32,
) -> Option<(u32, u32, u32)> {
    if req_w > max_w || req_h > max_h {
        return None;
    }
    // stride == 0 means the caller did not specify a meaningful source pitch;
    // the destination pitch is always the fixed nv12_pitch, so it cannot drive
    // an OOB write on its own. A positive stride larger than the destination
    // pitch indicates a malformed/oversized buffer → reject.
    if req_stride != 0 && req_stride > max_pitch {
        return None;
    }
    Some((req_w, req_h, req_stride))
}

impl CudaNvencRecorder {
    /// Apply a new target bitrate (bps) to the live NVENC session without a
    /// reopen. Thin delegate to `NvencSession::reconfigure_bitrate`; used by the
    /// viewer encode loop to drive GCC dynamic-bitrate. Returns false if the
    /// driver rejects the reconfigure.
    pub fn reconfigure_bitrate(&mut self, bps: u32) -> bool {
        self.nvenc.reconfigure_bitrate(bps)
    }

    /// Bring up EGL + CUDA + an NVENC H.264 session and fetch codec_private.
    /// Returns None on any failure (notably: no GPU → EglCtx/CudaCtx None), so
    /// callers fall back to the legacy path. No per-frame encode here (NV2.4).
    pub fn try_new(width: u32, height: u32) -> Option<Self> {
        let egl = match EglCtx::open() {
            Some(e) => {
                tracing::debug!("cuda_nvenc: try_new EGL ok");
                e
            }
            None => {
                tracing::warn!("cuda_nvenc: try_new EGL FAILED");
                return None;
            }
        };
        let cuda = match CudaCtx::open() {
            Some(c) => {
                tracing::debug!("cuda_nvenc: try_new CUDA ctx ok");
                c
            }
            None => {
                tracing::warn!("cuda_nvenc: try_new CUDA ctx FAILED");
                return None;
            }
        };
        let cu_module = cuda.load_csc_module(); // None until NV2.3 supplies PTX.
        tracing::debug!(
            "cuda_nvenc: try_new csc module = {}",
            if cu_module.is_some() {
                "loaded"
            } else {
                "NONE (will fail at encode)"
            }
        );
        let nvenc = match NvencSession::open(cuda.ctx, width, height) {
            Some(n) => {
                tracing::debug!("cuda_nvenc: try_new NVENC session ok");
                n
            }
            None => {
                tracing::warn!("cuda_nvenc: try_new NVENC session FAILED");
                return None;
            }
        };
        let codec_private = nvenc.sequence_params().unwrap_or_default();
        tracing::debug!(
            "cuda_nvenc: try_new codec_private = {} bytes",
            codec_private.len()
        );

        // Allocate the persistent NV12 buffer. NVENC wants the luma row pitch
        // aligned (256 bytes); the chroma plane is interleaved UV at half
        // height (ceil(height/2) rows). Using `* 3 / 2` truncates for odd
        // heights (one chroma row short → potential heap overflow), so we
        // compute chroma_rows explicitly via ceil division.
        let nv12_pitch = width.div_ceil(256) * 256;
        let chroma_rows = (height as usize).div_ceil(2); // ceil(height/2)
        let nv12_bytes = (nv12_pitch as usize) * (height as usize + chroma_rows);
        let mut nv12_devptr: u64 = 0;
        let rc = unsafe { (cuda.lib.mem_alloc)(&mut nv12_devptr, nv12_bytes) };
        if rc != 0 || nv12_devptr == 0 {
            tracing::warn!(
                "cuda_nvenc: cuMemAlloc_v2(NV12 {nv12_bytes} bytes) -> {}",
                cuda.lib.err(rc)
            );
            return None;
        }
        tracing::debug!(
            "cuda_nvenc: try_new NV12 buffer ok ({nv12_bytes} bytes, pitch={nv12_pitch})"
        );

        Some(Self {
            nvenc,
            cu_module,
            cuda,
            egl,
            width,
            height,
            codec_private,
            nv12_devptr,
            nv12_pitch,
            input_array: std::ptr::null_mut(),
            has_encoded: false,
            dump_counter: 0,
            oversize_rejected_logged: false,
        })
    }

    /// SPS/PPS payload for the MKV writer's codec_private field.
    pub fn codec_private(&self) -> &[u8] {
        &self.codec_private
    }

    /// Per-frame zero-copy encode of a KWin GPU-tiled ARGB8888 dmabuf.
    ///
    /// Imports the dmabuf via EGL → registers it with CUDA → reads the mapped
    /// cuArray → runs the BT.709 ARGB→NV12 CSC kernel into the persistent NV12
    /// devptr → feeds that devptr to NVENC → locks out the Annex-B bitstream.
    ///
    /// Returns `None` on ANY failure (the frame is simply dropped). Every early
    /// return unwinds exactly the resources acquired so far in this call — KWin
    /// recycles the underlying buffer, so a per-frame leak would be fatal.
    ///
    /// Unreachable offline: a recorder only exists when `try_new` succeeded,
    /// which requires a GPU. The encode itself is validated live in NV2.5.
    // encoder/cursor setup takes many tightly-related params by design
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dmabuf(
        &mut self,
        fd: std::os::fd::RawFd,
        modifier: u64,
        width: u32,
        height: u32,
        stride: u32,
        pts_us: i64,
        force_idr: bool,
    ) -> Option<crate::vulkan_record::EncodedNal> {
        const DRM_FORMAT_ARGB8888: u32 = 0x3432_5241; // 'AR24' — little-endian BGRA.

        // -- Step 0: bound the client-supplied dimensions (audit M6 / #14) ----
        // `width`/`height`/`stride` here originate from the inner Wayland
        // client's dmabuf and are NOT trusted. The persistent NV12 output
        // (`nv12_devptr`) was allocated once for `self.width × self.height` at
        // `self.nv12_pitch`; an oversized frame would drive the CSC kernel to
        // write out of bounds in the shared per-session encoder context. Reject
        // such a frame up front (before doing any EGL/CUDA import work) and log
        // at most once.
        if safe_encode_dims(
            width,
            height,
            stride,
            self.width,
            self.height,
            self.nv12_pitch,
        )
        .is_none()
        {
            if !self.oversize_rejected_logged {
                self.oversize_rejected_logged = true;
                tracing::warn!(
                    "cuda_nvenc: rejecting oversized dmabuf frame {width}x{height} stride={stride} \
                     (encoder allocated for {}x{} pitch={}); dropping frame to avoid OOB device write \
                     (further such rejections suppressed)",
                    self.width,
                    self.height,
                    self.nv12_pitch
                );
            }
            return None;
        }

        // -- Step 1: import the dmabuf fd as an EGLImage ----------------------
        // Attribute array shape copied verbatim from the NV1 probe.
        let attrs: [i32; 19] = [
            EGL_WIDTH,
            width as i32,
            EGL_HEIGHT,
            height as i32,
            EGL_LINUX_DRM_FOURCC_EXT,
            DRM_FORMAT_ARGB8888 as i32,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            stride as i32,
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (modifier & 0xFFFF_FFFF) as i32,
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (modifier >> 32) as i32,
            EGL_NONE,
            // Trailing pad past EGL_NONE; ignored by EGL.
            EGL_NONE,
            EGL_NONE,
        ];

        // EGL_NO_CONTEXT == null.
        let egl_image = unsafe {
            (self.egl.create_image)(
                self.egl.display,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attrs.as_ptr(),
            )
        };
        if egl_image.is_null() {
            tracing::warn!("cuda_nvenc: eglCreateImageKHR(dmabuf fd={fd} modifier=0x{modifier:016x} stride={stride}) returned NO_IMAGE");
            return None;
        }
        tracing::trace!("cuda_nvenc: eglCreateImageKHR ok (image={egl_image:?})");
        // From here on, every early return must eglDestroyImageKHR(egl_image).

        // Helper to destroy the EGLImage (best effort). Capture only the EGL
        // display + destroy fn (both Copy) rather than `self`, so the closure
        // does not borrow `self` — otherwise it would conflict with the
        // `&mut self` call into `encode_from_cuarray` below.
        let egl_display = self.egl.display;
        let egl_destroy_image = self.egl.destroy_image;
        let destroy_egl_image = move |img: *mut c_void| {
            if let Some(d) = egl_destroy_image {
                if !img.is_null() {
                    unsafe { d(egl_display, img) };
                }
            }
        };

        // -- Step 2: register with CUDA + map the EGL frame -> cuArray --------
        let mut res: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { (self.cuda.lib.register_image)(&mut res, egl_image, 0) };
        if rc != 0 || res.is_null() {
            tracing::warn!(
                "cuda_nvenc: cuGraphicsEGLRegisterImage -> {}",
                self.cuda.lib.err(rc)
            );
            destroy_egl_image(egl_image);
            return None;
        }
        tracing::trace!("cuda_nvenc: cuGraphicsEGLRegisterImage ok (res={res:?})");
        // From here: also cuGraphicsUnregisterResource(res).

        let mut frame: CUeglFrame = unsafe { std::mem::zeroed() };
        let rc = unsafe { (self.cuda.lib.get_mapped_frame)(&mut frame, res, 0, 0) };
        if rc != 0 {
            tracing::warn!(
                "cuda_nvenc: cuGraphicsResourceGetMappedEglFrame -> {}",
                self.cuda.lib.err(rc)
            );
            unsafe { (self.cuda.lib.unregister)(res) };
            destroy_egl_image(egl_image);
            return None;
        }
        // NV1 proved frameType == ARRAY, so the surface is in array[0].
        let cu_array = unsafe { frame.frame.array[0] };
        tracing::trace!(
            "cuda_nvenc: cuGraphicsResourceGetMappedEglFrame ok (frameType={} array[0]={cu_array:?} w={} h={} pitch={} planes={})",
            frame.frame_type, frame.width, frame.height, frame.pitch, frame.plane_count
        );
        if cu_array.is_null() {
            tracing::warn!("cuda_nvenc: mapped EGL frame array[0] is null");
            unsafe { (self.cuda.lib.unregister)(res) };
            destroy_egl_image(egl_image);
            return None;
        }

        // -- Steps 3-9: tex → CSC kernel → NVENC → Annex-B (shared tail) ------
        // `encode_from_cuarray` owns the texture object + NVENC per-frame
        // resources and tears them down on every path. The EGL-import
        // resources (`res`, `egl_image`) are owned HERE and torn down below on
        // BOTH the success and failure paths. On success it marks
        // `self.has_encoded = true` so the idle heartbeat can re-emit.
        let result = self.encode_from_cuarray(cu_array, width, height, pts_us, force_idr);

        unsafe { (self.cuda.lib.unregister)(res) };
        destroy_egl_image(egl_image);

        result
    }

    /// Shared encode tail: wrap `array` (a CUarray handle) in a texture object,
    /// run the BT.709 ARGB→NV12 CSC kernel into the persistent NV12 devptr,
    /// feed that devptr to NVENC, and lock out the Annex-B bitstream.
    ///
    /// Both `encode_dmabuf` (EGL-mapped cuArray) and `encode_pixels`
    /// (host-uploaded input cuArray) call this with a compatible handle. The
    /// handle type matches `CUeglFrame.frame.array[0]` (`*mut c_void`).
    ///
    /// Tears down EVERY resource it acquires on EVERY path (tex object + NVENC
    /// register/map). Returns `None` on any failure.
    fn encode_from_cuarray(
        &mut self,
        array: *mut c_void,
        width: u32,
        height: u32,
        pts_us: i64,
        force_idr: bool,
    ) -> Option<crate::vulkan_record::EncodedNal> {
        // The CSC kernel must exist; if NV2.3 didn't supply real PTX, bail.
        let func = match self.cu_module.as_ref() {
            Some(m) => m.func,
            None => {
                tracing::warn!("cuda_nvenc: CSC module not loaded; cannot encode");
                return None;
            }
        };

        // Universal write-bounds chokepoint (audit M6 / #14). BOTH the dmabuf
        // path (`encode_dmabuf`, untrusted client dims) and the host-pixels
        // path (`encode_pixels`, already clamped) funnel through here. Re-check
        // against the fixed NV12 allocation so no kernel launch can EVER write
        // past `nv12_devptr`, independent of caller discipline. `encode_dmabuf`
        // already rejected oversized frames up front; this is the defense in
        // depth that makes the kernel launch below unconditionally safe.
        if safe_encode_dims(width, height, 0, self.width, self.height, self.nv12_pitch).is_none() {
            if !self.oversize_rejected_logged {
                self.oversize_rejected_logged = true;
                tracing::warn!(
                    "cuda_nvenc: refusing CSC launch for oversized frame {width}x{height} \
                     (encoder allocated for {}x{}); dropping to avoid OOB device write",
                    self.width,
                    self.height
                );
            }
            return None;
        }

        // -- Step 3: wrap the cuArray in a texture object ---------------------
        let mut resdesc: CudaResourceDesc = unsafe { std::mem::zeroed() };
        resdesc.res_type = CU_RESOURCE_TYPE_ARRAY;
        resdesc.handle[0] = array as u64;

        let mut texdesc: CudaTextureDesc = unsafe { std::mem::zeroed() };
        texdesc.address_mode = [
            CU_TR_ADDRESS_MODE_CLAMP,
            CU_TR_ADDRESS_MODE_CLAMP,
            CU_TR_ADDRESS_MODE_CLAMP,
        ];
        texdesc.filter_mode = CU_TR_FILTER_MODE_POINT;
        texdesc.flags = CU_TRSF_READ_AS_INTEGER;

        let mut tex: u64 = 0;
        let rc = unsafe {
            (self.cuda.lib.tex_object_create)(&mut tex, &resdesc, &texdesc, std::ptr::null())
        };
        if rc != 0 || tex == 0 {
            tracing::warn!(
                "cuda_nvenc: cuTexObjectCreate -> {} tex={tex}",
                self.cuda.lib.err(rc)
            );
            return None;
        }
        tracing::trace!("cuda_nvenc: cuTexObjectCreate ok (tex={tex})");
        // From here: also cuTexObjectDestroy(tex).

        // -- Step 4: launch the CSC kernel into the persistent NV12 devptr ----
        // The kernel processes one 2x2 chroma block per thread, so the grid
        // covers width/2 x height/2 in 16x16 thread blocks.
        let gx = (width / 2).div_ceil(16);
        let gy = (height / 2).div_ceil(16);

        // Kernel signature (NV2.3a):
        //   argb_to_nv12_bt709(cudaTextureObject_t src, uint8_t* dst_nv12,
        //                      int width, int height, int dst_pitch)
        // cuLaunchKernel takes an array of POINTERS to each argument value, so
        // these locals must outlive the launch+sync below — they all live to
        // the end of this block. Bind `&mut` of each (the FFI wants *mut).
        let mut arg_tex: u64 = tex;
        let mut arg_dst: u64 = self.nv12_devptr;
        let mut arg_w: i32 = width as i32;
        let mut arg_h: i32 = height as i32;
        let mut arg_pitch: i32 = self.nv12_pitch as i32;
        let mut params: [*mut c_void; 5] = [
            &mut arg_tex as *mut u64 as *mut c_void,
            &mut arg_dst as *mut u64 as *mut c_void,
            &mut arg_w as *mut i32 as *mut c_void,
            &mut arg_h as *mut i32 as *mut c_void,
            &mut arg_pitch as *mut i32 as *mut c_void,
        ];

        let rc = unsafe {
            (self.cuda.lib.launch_kernel)(
                func,
                gx,
                gy,
                1,
                16,
                16,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            tracing::warn!(
                "cuda_nvenc: cuLaunchKernel -> {} (grid={gx}x{gy})",
                self.cuda.lib.err(rc)
            );
            unsafe { (self.cuda.lib.tex_object_destroy)(tex) };
            return None;
        }
        tracing::trace!("cuda_nvenc: cuLaunchKernel ok (grid={gx}x{gy} block=16x16)");

        // -- Step 5: synchronize the CSC before NVENC reads the devptr --------
        let rc = unsafe { (self.cuda.lib.stream_synchronize)(std::ptr::null_mut()) };
        if rc != 0 {
            tracing::warn!(
                "cuda_nvenc: cuStreamSynchronize -> {}",
                self.cuda.lib.err(rc)
            );
            unsafe { (self.cuda.lib.tex_object_destroy)(tex) };
            return None;
        }
        tracing::trace!("cuda_nvenc: cuStreamSynchronize ok");

        // Debug capture: the CSC has just written `nv12_devptr` with this
        // frame's content — dump it (if WAYMUX_NV12_DUMP_DIR is set) so we can
        // see server-side whether real content reached the encoder input.
        self.dump_nv12_png("frame");

        // -- Steps 6-9: NVENC-encode the freshly-written NV12 devptr ----------
        // The CSC kernel has now written `self.nv12_devptr`. The NVENC steps
        // (register → map → encode → lock → copy → unmap/unregister) are shared
        // with the idle-heartbeat path, so they live in
        // `nvenc_encode_current_nv12`. The texture object created above is owned
        // HERE: tear it down on BOTH the success and failure paths.
        let result = self.nvenc_encode_current_nv12(width, height, force_idr, pts_us);
        unsafe { (self.cuda.lib.tex_object_destroy)(tex) };
        if result.is_some() {
            // A real frame (dmabuf or host-pixels) has been fully encoded, so
            // `nv12_devptr` now holds decodable content the heartbeat can
            // re-emit. Mark it for `reencode_last`.
            self.has_encoded = true;
        }
        result
    }

    /// Debug capture harness: copy the current `nv12_devptr` (the exact NV12
    /// the encoder is about to compress) device->host, convert to RGB, and write
    /// a PNG. Gated entirely on `WAYMUX_NV12_DUMP_DIR`; a no-op (one env read,
    /// no allocation, no copy) when that is unset, so it costs nothing in
    /// production. Rate-limited to every `WAYMUX_NV12_DUMP_EVERY`-th eligible
    /// call (default 30). `tag` distinguishes real content frames ("frame")
    /// from idle heartbeat re-emits ("heartbeat") — the latter exposes whether
    /// the heartbeat is holding a stale/black frame (investigation hypothesis 2).
    ///
    /// Never panics and never fails the encode: any error just logs and returns.
    /// Unreachable offline (needs a live recorder + CUDA ctx); the fallible math
    /// it relies on lives in the pure `nv12_to_rgb`, which unit-tests with no GPU.
    fn dump_nv12_png(&mut self, tag: &str) {
        let dir = match std::env::var("WAYMUX_NV12_DUMP_DIR") {
            Ok(d) if !d.is_empty() => d,
            _ => return,
        };
        let every: u64 = std::env::var("WAYMUX_NV12_DUMP_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(30);
        let seq = self.dump_counter;
        self.dump_counter += 1;
        if !seq.is_multiple_of(every) {
            return;
        }

        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.nv12_pitch as usize;
        let chroma_rows = height.div_ceil(2);
        let total_rows = height + chroma_rows;
        let mut host = vec![0u8; pitch * total_rows];

        // Single 2D device->host copy of the whole pitched NV12 buffer (luma +
        // interleaved chroma are contiguous), preserving the row pitch so
        // `nv12_to_rgb` indexes identically to the device layout.
        let copy = CUDA_MEMCPY2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: CU_MEMORYTYPE_DEVICE,
            _pad0: 0,
            src_host: std::ptr::null(),
            src_device: self.nv12_devptr,
            src_array: std::ptr::null_mut(),
            src_pitch: pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: CU_MEMORYTYPE_HOST,
            _pad1: 0,
            dst_host: host.as_mut_ptr() as *mut c_void,
            dst_device: 0,
            dst_array: std::ptr::null_mut(),
            dst_pitch: pitch,
            width_in_bytes: pitch,
            height: total_rows,
        };
        let rc = unsafe { (self.cuda.lib.memcpy_2d)(&copy) };
        if rc != 0 {
            tracing::warn!("nv12_dump: cuMemcpy2D D->H -> {}", self.cuda.lib.err(rc));
            return;
        }

        let rgb = nv12_to_rgb(&host, width, height, pitch);
        let path = format!("{dir}/nv12-{tag}-{seq:06}.png");
        match std::fs::File::create(&path) {
            Ok(file) => {
                let buf = std::io::BufWriter::new(file);
                let mut enc = png::Encoder::new(buf, width as u32, height as u32);
                enc.set_color(png::ColorType::Rgb);
                enc.set_depth(png::BitDepth::Eight);
                match enc
                    .write_header()
                    .and_then(|mut w| w.write_image_data(&rgb))
                {
                    Ok(()) => tracing::info!("nv12_dump: wrote {path} ({width}x{height})"),
                    Err(e) => tracing::warn!("nv12_dump: png encode {path} failed: {e}"),
                }
            }
            Err(e) => tracing::warn!("nv12_dump: create {path} failed: {e}"),
        }
    }

    /// Shared NVENC encode tail: register `self.nv12_devptr` as an NVENC input
    /// resource, map it, encode one forced-IDR picture, lock + copy out the
    /// Annex-B bitstream, then unmap/unregister. Does NOT touch EGL/CUDA import
    /// or the CSC kernel — the caller must have already written `nv12_devptr`
    /// (via the CSC) OR be deliberately re-emitting the last-written NV12
    /// (`reencode_last`).
    ///
    /// `width`/`height` are the picture dimensions to encode (the dmabuf path
    /// may clamp; the heartbeat passes the recorder's full dims). The NVENC
    /// session itself was initialized at `self.width × self.height`, so these
    /// must not exceed those bounds.
    ///
    /// Tears down every NVENC resource it acquires on every path (so a failure
    /// mid-sequence never leaks a mapped/registered resource). Returns `None`
    /// on any failure. The texture object (dmabuf/pixels path) is owned by the
    /// caller, not here.
    fn nvenc_encode_current_nv12(
        &mut self,
        width: u32,
        height: u32,
        force_idr: bool,
        pts_us: i64,
    ) -> Option<crate::vulkan_record::EncodedNal> {
        // -- Step 6: register the NV12 devptr as an NVENC input resource ------
        let reg_ptr = self.nvenc.funcs.slot(SLOT_REGISTER_RESOURCE);
        let map_ptr = self.nvenc.funcs.slot(SLOT_MAP_INPUT_RESOURCE);
        let unmap_ptr = self.nvenc.funcs.slot(SLOT_UNMAP_INPUT_RESOURCE);
        let unreg_ptr = self.nvenc.funcs.slot(SLOT_UNREGISTER_RESOURCE);
        let enc_ptr = self.nvenc.funcs.slot(SLOT_ENCODE_PICTURE);
        let lock_ptr = self.nvenc.funcs.slot(SLOT_LOCK_BITSTREAM);
        let unlock_ptr = self.nvenc.funcs.slot(SLOT_UNLOCK_BITSTREAM);
        if reg_ptr.is_null()
            || map_ptr.is_null()
            || unmap_ptr.is_null()
            || unreg_ptr.is_null()
            || enc_ptr.is_null()
            || lock_ptr.is_null()
            || unlock_ptr.is_null()
        {
            tracing::warn!("cuda_nvenc: an NVENC encode-path slot is null");
            return None;
        }
        let register_resource: PfnNvEncRegisterResource = unsafe { std::mem::transmute(reg_ptr) };
        let map_input: PfnNvEncMapInputResource = unsafe { std::mem::transmute(map_ptr) };
        let unmap_input: PfnNvEncUnmapInputResource = unsafe { std::mem::transmute(unmap_ptr) };
        let unregister_resource: PfnNvEncUnregisterResource =
            unsafe { std::mem::transmute(unreg_ptr) };
        let encode_picture: PfnNvEncEncodePicture = unsafe { std::mem::transmute(enc_ptr) };
        let lock_bitstream: PfnNvEncLockBitstream = unsafe { std::mem::transmute(lock_ptr) };
        let unlock_bitstream: PfnNvEncUnlockBitstream = unsafe { std::mem::transmute(unlock_ptr) };

        let mut reg: NV_ENC_REGISTER_RESOURCE = unsafe { std::mem::zeroed() };
        reg.version = nv_enc_register_resource_ver();
        reg.resource_type = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        reg.width = width;
        reg.height = height;
        reg.pitch = self.nv12_pitch;
        reg.resource_to_register = self.nv12_devptr as *mut c_void;
        reg.buffer_format = NV_ENC_BUFFER_FORMAT_NV12;
        reg.buffer_usage = NV_ENC_INPUT_IMAGE;
        let rc = unsafe { register_resource(self.nvenc.encoder, &mut reg) };
        if rc != 0 || reg.registered_resource.is_null() {
            tracing::warn!(
                "cuda_nvenc: nvEncRegisterResource -> {rc} registered={:?}",
                reg.registered_resource
            );
            return None;
        }
        tracing::trace!(
            "cuda_nvenc: nvEncRegisterResource ok status={rc} (registered={:?})",
            reg.registered_resource
        );
        let registered_resource = reg.registered_resource;
        // From here: also nvEncUnregisterResource(registered_resource).

        let mut map: NV_ENC_MAP_INPUT_RESOURCE = unsafe { std::mem::zeroed() };
        map.version = nv_enc_map_input_resource_ver();
        map.registered_resource = registered_resource;
        let rc = unsafe { map_input(self.nvenc.encoder, &mut map) };
        if rc != 0 || map.mapped_resource.is_null() {
            tracing::warn!(
                "cuda_nvenc: nvEncMapInputResource -> {rc} mapped={:?}",
                map.mapped_resource
            );
            unsafe { unregister_resource(self.nvenc.encoder, registered_resource) };
            return None;
        }
        tracing::trace!(
            "cuda_nvenc: nvEncMapInputResource ok status={rc} (mapped={:?} fmt={})",
            map.mapped_resource,
            map.mapped_buffer_fmt
        );
        let mapped_resource = map.mapped_resource;
        // From here: also nvEncUnmapInputResource(mapped_resource).

        // -- Step 7: encode the picture ---------------------------------------
        let mut pic: NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        pic.version = nv_enc_pic_params_ver();
        pic.input_width = width;
        pic.input_height = height;
        pic.input_pitch = self.nv12_pitch;
        pic.input_buffer = mapped_resource;
        pic.output_bitstream = self.nvenc.bitstream;
        pic.buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
        pic.picture_structure = NV_ENC_PIC_STRUCT_FRAME;
        // Only force an IDR (with SPS/PPS attached, so mid-stream joiners get
        // them) when the caller explicitly asks — the first frame, a viewer
        // PLI, or the idle heartbeat. Otherwise emit a normal P-frame with no
        // repeated headers: that is the whole point of the bitrate fix (see the
        // RC/GOP config in `NvencSession::open`). The periodic gopLength IDR
        // (Part 2) is the in-encoder safety net between explicit requests.
        pic.encode_pic_flags = if force_idr {
            NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS
        } else {
            0
        };
        pic.input_timestamp = pts_us as u64;
        let rc = unsafe { encode_picture(self.nvenc.encoder, &mut pic) };
        if rc != 0 {
            tracing::warn!("cuda_nvenc: nvEncEncodePicture -> {rc}");
            unsafe { unmap_input(self.nvenc.encoder, mapped_resource) };
            unsafe { unregister_resource(self.nvenc.encoder, registered_resource) };
            return None;
        }
        tracing::trace!("cuda_nvenc: nvEncEncodePicture ok status={rc}");

        // -- Step 8: lock the bitstream and copy out the Annex-B NALUs --------
        let mut lock: NV_ENC_LOCK_BITSTREAM = unsafe { std::mem::zeroed() };
        lock.version = nv_enc_lock_bitstream_ver();
        lock.output_bitstream = self.nvenc.bitstream;
        let rc = unsafe { lock_bitstream(self.nvenc.encoder, &mut lock) };
        if rc != 0 || lock.bitstream_buffer_ptr.is_null() {
            tracing::warn!(
                "cuda_nvenc: nvEncLockBitstream -> {rc} bitstreamSizeInBytes={} bufPtr={:?}",
                lock.bitstream_size_in_bytes,
                lock.bitstream_buffer_ptr
            );
            unsafe { unmap_input(self.nvenc.encoder, mapped_resource) };
            unsafe { unregister_resource(self.nvenc.encoder, registered_resource) };
            return None;
        }
        let n = lock.bitstream_size_in_bytes as usize;
        tracing::trace!(
            "cuda_nvenc: nvEncLockBitstream ok status={rc} bitstreamSizeInBytes={n} bufPtr={:?}",
            lock.bitstream_buffer_ptr
        );
        // SAFETY: NVENC guarantees `bitstream_buffer_ptr` points at `n` valid
        // bytes while the bitstream is locked; we copy them out before unlock.
        let data: Vec<u8> =
            unsafe { std::slice::from_raw_parts(lock.bitstream_buffer_ptr as *const u8, n) }
                .to_vec();
        unsafe { unlock_bitstream(self.nvenc.encoder, self.nvenc.bitstream) };

        // -- Step 9: tear down the NVENC resources acquired this frame --------
        // (The texture object, when there is one, is owned + destroyed by the
        // caller — `encode_from_cuarray`. The heartbeat path has no texture.)
        unsafe { unmap_input(self.nvenc.encoder, mapped_resource) };
        unsafe { unregister_resource(self.nvenc.encoder, registered_resource) };

        Some(crate::vulkan_record::EncodedNal {
            data,
            pts_us,
            is_keyframe: force_idr,
        })
    }

    /// Re-emit the LAST encoded frame as a fresh IDR (NVENC-encode the existing
    /// nv12_devptr again — no import/CSC). Used by the viewer's idle heartbeat so
    /// a static desktop / a refreshing peer always gets a recent keyframe instead
    /// of a black screen. None until at least one real frame has been encoded.
    pub fn reencode_last(
        &mut self,
        pts_us: i64,
        force_idr: bool,
    ) -> Option<crate::vulkan_record::EncodedNal> {
        if !self.has_encoded {
            return None;
        }
        // Debug capture: dump what the heartbeat is about to re-emit. If this
        // shows a stale/black frame while the desktop has live content, the
        // heartbeat is masking a stalled source (investigation hypothesis 2).
        self.dump_nv12_png("heartbeat");
        // Encode at the recorder's full init dimensions — `nv12_devptr` holds a
        // full-frame NV12 surface from the last real encode.
        self.nvenc_encode_current_nv12(self.width, self.height, force_idr, pts_us)
    }

    /// Per-frame encode of a HOST BGRA frame (the idle-desktop /
    /// `wp_single_pixel_buffer` / SHM path). Copies the host pixels into a
    /// persistent input cuArray (texture-sampled, ARGB8888 byte order, same as
    /// the dmabuf path) and runs the SAME tex→CSC→NVENC→Annex-B tail.
    ///
    /// `bgra` is tightly-packed (`stride` bytes per row, expected `width*4`).
    /// Returns `None` on any failure (the frame is dropped).
    ///
    /// Unreachable offline: a recorder only exists when `try_new` succeeded,
    /// which requires a GPU.
    pub fn encode_pixels(
        &mut self,
        bgra: &[u8],
        width: u32,
        height: u32,
        stride: u32,
        pts_us: i64,
        force_idr: bool,
    ) -> Option<crate::vulkan_record::EncodedNal> {
        // -- Step 1: ensure the persistent input cuArray exists ---------------
        // Sized to (self.width × self.height) — the recorder's fixed encode
        // dimensions — so it is allocated once and reused. Format
        // UNSIGNED_INT8 × 4 channels = ARGB8888 / BGRA byte order, matching what
        // the CSC kernel samples as uchar4 (.x=B .y=G .z=R), identical to the
        // dmabuf path's EGL-mapped array.
        if self.input_array.is_null() {
            let desc = CUDA_ARRAY_DESCRIPTOR {
                width: self.width as usize,
                height: self.height as usize,
                format: CU_AD_FORMAT_UNSIGNED_INT8,
                num_channels: 4,
            };
            let mut handle: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { (self.cuda.lib.array_create)(&mut handle, &desc) };
            if rc != 0 || handle.is_null() {
                tracing::warn!(
                    "cuda_nvenc: cuArrayCreate_v2({}x{}) -> {}",
                    self.width,
                    self.height,
                    self.cuda.lib.err(rc)
                );
                return None;
            }
            self.input_array = handle;
            tracing::debug!(
                "cuda_nvenc: allocated input cuArray {}x{}",
                self.width,
                self.height
            );
        }

        // -- Step 2: copy the host BGRA into the input cuArray ----------------
        // Clamp the copied region to the recorder's fixed array dimensions so
        // an oversized source can't write past the array allocation.
        let copy_w = width.min(self.width);
        let copy_h = height.min(self.height);
        let mut copy: CUDA_MEMCPY2D = unsafe { std::mem::zeroed() };
        copy.src_x_in_bytes = 0;
        copy.src_y = 0;
        copy.src_memory_type = CU_MEMORYTYPE_HOST;
        copy.src_host = bgra.as_ptr() as *const c_void;
        copy.src_pitch = stride as usize;
        copy.dst_x_in_bytes = 0;
        copy.dst_y = 0;
        copy.dst_memory_type = CU_MEMORYTYPE_ARRAY;
        copy.dst_array = self.input_array;
        copy.width_in_bytes = (copy_w as usize) * 4;
        copy.height = copy_h as usize;
        let rc = unsafe { (self.cuda.lib.memcpy_2d)(&copy) };
        if rc != 0 {
            tracing::warn!(
                "cuda_nvenc: cuMemcpy2D_v2(host->array {copy_w}x{copy_h}) -> {}",
                self.cuda.lib.err(rc)
            );
            return None;
        }

        // -- Step 3: run the shared tex → CSC → NVENC → Annex-B tail ----------
        self.encode_from_cuarray(self.input_array, copy_w, copy_h, pts_us, force_idr)
    }
}

impl Drop for CudaNvencRecorder {
    fn drop(&mut self) {
        // Free the NV12 devptr while the CUDA context is still alive. This Drop
        // body runs BEFORE the fields drop (so `self.cuda`'s context is intact),
        // which is exactly the ordering we need.
        if self.nv12_devptr != 0 {
            let _ = unsafe { (self.cuda.lib.mem_free)(self.nv12_devptr) };
            self.nv12_devptr = 0;
        }
        // Free the persistent input cuArray (host-pixels path) likewise BEFORE
        // the CUDA context is torn down.
        if !self.input_array.is_null() {
            let _ = unsafe { (self.cuda.lib.array_destroy)(self.input_array) };
            self.input_array = std::ptr::null_mut();
        }
    }
}

/// Inverse of the `argb_to_nv12_bt709` CSC kernel: convert a pitched NV12
/// buffer (BT.709 **limited range** — the kernel's output space, confirmed by
/// `cuda_nv12_csc.cu` + `video_full_range_flag = 0`) to tightly packed RGB8
/// (`width*height*3`, R,G,B order).
///
/// Layout matches `nv12_devptr`: the luma plane is `height` rows of `pitch`
/// bytes (only the first `width` per row are valid); the interleaved CbCr plane
/// follows at byte offset `pitch*height`, `ceil(height/2)` rows of `pitch`
/// bytes, one Cb,Cr pair per 2x2 luma block.
///
/// Pure + free-standing on purpose: the GPU dump glue (`dump_nv12_png`) can't
/// run offline, but THIS — the colour-space math that can actually be wrong —
/// unit-tests with no GPU. Used only by the `WAYMUX_NV12_DUMP_DIR` debug
/// capture harness for the viewer content-rendering investigation.
pub(crate) fn nv12_to_rgb(nv12: &[u8], width: usize, height: usize, pitch: usize) -> Vec<u8> {
    const KR: f32 = 0.2126;
    const KG: f32 = 0.7152;
    const KB: f32 = 0.0722;
    let uv_base = pitch * height;
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        let y_off = y * pitch;
        let uv_off = uv_base + (y / 2) * pitch;
        for x in 0..width {
            let luma = nv12[y_off + x] as f32;
            // Each 2x2 luma block shares one Cb,Cr pair, starting at the even
            // column index (`x & !1`): Cb then Cr.
            let cb = nv12[uv_off + (x & !1)] as f32;
            let cr = nv12[uv_off + (x & !1) + 1] as f32;
            // De-scale limited range, then the BT.709 inverse matrix (exact
            // inverse of the kernel's forward 219/224 scaling).
            let yp = (luma - 16.0) * (255.0 / 219.0);
            let cbp = (cb - 128.0) * (255.0 / 224.0);
            let crp = (cr - 128.0) * (255.0 / 224.0);
            let r = yp + 2.0 * (1.0 - KR) * crp;
            let b = yp + 2.0 * (1.0 - KB) * cbp;
            let g = (yp - KR * r - KB * b) / KG;
            let o = (y * width + x) * 3;
            rgb[o] = r.clamp(0.0, 255.0) as u8;
            rgb[o + 1] = g.clamp(0.0, 255.0) as u8;
            rgb[o + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    // Offline unit test for the capture harness's NV12->RGB inverse. No GPU.
    #[test]
    fn nv12_to_rgb_range_anchors() {
        // 2x2 image, luma pitch 4 (rows padded past width=2). Limited-range
        // BT.709 anchors: black = Y16/neutral-chroma, white = Y235/neutral.
        let (w, h, pitch) = (2usize, 2usize, 4usize);
        let mut buf = vec![0u8; pitch * h + pitch * h.div_ceil(2)];
        let luma = [16u8, 235, 235, 235]; // (0,0) black, rest white
        for row in 0..h {
            for col in 0..w {
                buf[row * pitch + col] = luma[row * w + col];
            }
        }
        // Neutral chroma (128,128) for the single shared 2x2 block.
        let uv = pitch * h;
        buf[uv] = 128;
        buf[uv + 1] = 128;
        let rgb = super::nv12_to_rgb(&buf, w, h, pitch);
        assert_eq!(rgb.len(), w * h * 3);
        // (0,0) is exactly black.
        assert_eq!(&rgb[0..3], &[0, 0, 0]);
        // (1,0) is white (allow 1 LSB of float rounding on the 255/219 scale).
        for (i, &c) in rgb[3..6].iter().enumerate() {
            assert!(c >= 254, "white byte {i} = {c}, expected ~255");
        }
        // (0,1) on the second row is white too (row-pitch indexing correct).
        let p01 = w * 3;
        assert!(rgb[p01] >= 254, "row-1 luma indexing wrong: {}", rgb[p01]);
    }
    // Host-side write-bounds guard for the CSC kernel (audit M6 / #14). Pure
    // logic, no GPU. The dmabuf path feeds untrusted client dimensions into a
    // kernel that writes a fixed-size NV12 allocation; `safe_encode_dims` must
    // REJECT anything that would exceed that allocation and pass in-bounds /
    // boundary-equal frames through unchanged.
    #[test]
    fn safe_encode_dims_passes_in_bounds() {
        // Strictly under all limits → accepted, returned verbatim.
        assert_eq!(
            super::safe_encode_dims(1280, 720, 5120, 1920, 1080, 7680),
            Some((1280, 720, 5120))
        );
    }

    #[test]
    fn safe_encode_dims_allows_exact_boundary() {
        // width == max_w, height == max_h, stride == max_pitch → still allowed.
        assert_eq!(
            super::safe_encode_dims(1920, 1080, 7680, 1920, 1080, 7680),
            Some((1920, 1080, 7680))
        );
    }

    #[test]
    fn safe_encode_dims_rejects_over_width() {
        assert_eq!(
            super::safe_encode_dims(1921, 1080, 7680, 1920, 1080, 7680),
            None
        );
    }

    #[test]
    fn safe_encode_dims_rejects_over_height() {
        assert_eq!(
            super::safe_encode_dims(1920, 1081, 7680, 1920, 1080, 7680),
            None
        );
    }

    #[test]
    fn safe_encode_dims_rejects_over_stride() {
        // In-bounds w/h but a stride larger than the destination pitch is the
        // classic "malformed buffer" OOB driver → reject.
        assert_eq!(
            super::safe_encode_dims(1920, 1080, 7681, 1920, 1080, 7680),
            None
        );
    }

    #[test]
    fn safe_encode_dims_zero_stride_is_unspecified_and_allowed() {
        // The chokepoint in `encode_from_cuarray` passes stride 0 (the
        // destination pitch is always nv12_pitch); 0 must not be treated as an
        // overflow.
        assert_eq!(
            super::safe_encode_dims(1920, 1080, 0, 1920, 1080, 7680),
            Some((1920, 1080, 0))
        );
    }

    #[test]
    fn safe_encode_dims_rejects_grossly_oversized_attack() {
        // A hostile client claiming a huge dmabuf to overflow the device heap.
        assert_eq!(
            super::safe_encode_dims(65535, 65535, 262140, 1920, 1080, 7680),
            None
        );
    }

    #[test]
    fn cuda_lib_loader_is_none_without_gpu() {
        // On no-GPU CI, loading libcuda.so.1 / cuInit fails cleanly → None.
        assert!(super::CudaLib::load().is_none());
    }

    #[test]
    fn nvenc_struct_versions_are_sane() {
        assert_ne!(super::nv_enc_initialize_params_ver(), 0);
        assert_ne!(super::nv_enc_config_ver(), 0);
        assert_eq!(super::NVENCAPI_VERSION & 0xff, 13);
    }

    #[test]
    fn viewer_rc_flags_packs_aq_bits() {
        // enableAQ @ bit 3 (0x8), enableTemporalAQ @ bit 8 (0x100),
        // aqStrength=8 @ bits 12-15 (0x8000) => 0x8108. Bit layout is from
        // nvEncodeAPI.h SDK 13; this test pins the packing so a future edit
        // can't silently shift a bit.
        assert_eq!(super::viewer_rc_flags(8), 0x8108);
        // strength masks to 4 bits.
        assert_eq!(super::viewer_rc_flags(0xFF) & 0xF000, 0xF000);
    }

    #[test]
    fn nvenc_tuning_constants_are_sane() {
        assert_eq!(super::NV_ENC_MULTI_PASS_DISABLED, 0);
        assert_eq!(super::NV_ENC_RC_FLAG_ENABLE_AQ, 1 << 3);
        assert_eq!(super::NV_ENC_RC_FLAG_ENABLE_TEMPORAL_AQ, 1 << 8);
        // P5 GUID, verified vs nvEncodeAPI.h (a wrong transcription broke
        // CudaNvenc live with nvEncGetEncodePresetConfigEx error 4).
        assert_eq!(super::NV_ENC_PRESET_P5_GUID.data1, 0x21c6_e6b4);
        assert_ne!(
            super::NV_ENC_PRESET_P5_GUID.data1,
            super::NV_ENC_PRESET_P3_GUID.data1
        );
    }

    #[test]
    fn nvenc_structs_have_expected_min_sizes() {
        use std::mem::size_of;
        assert!(size_of::<super::NV_ENC_INITIALIZE_PARAMS>() >= 64);
        assert!(size_of::<super::NV_ENC_CONFIG>() >= 64);
        assert!(size_of::<super::NV_ENCODE_API_FUNCTION_LIST>() >= 200);
    }

    /// EXACT struct sizes cross-checked against the real nvEncodeAPI.h (NVENC
    /// SDK 13) via `sizeof`/`offsetof`. A wrong size means a mis-modeled field
    /// (the class of bug that caused nvEncInitializeEncoder INVALID_PARAM).
    #[test]
    fn nvenc_structs_match_header_byte_sizes() {
        use std::mem::size_of;
        assert_eq!(size_of::<super::NV_ENC_CONFIG_H264_VUI_PARAMETERS>(), 112);
        assert_eq!(size_of::<super::NV_ENC_RC_PARAMS>(), 128);
        assert_eq!(size_of::<super::NV_ENC_CONFIG_H264>(), 1792);
        assert_eq!(size_of::<super::NV_ENC_CONFIG>(), 3584);
        assert_eq!(size_of::<super::NV_ENC_INITIALIZE_PARAMS>(), 1800);
        assert_eq!(size_of::<super::NV_ENC_PRESET_CONFIG>(), 5128);
        assert_eq!(size_of::<super::NV_ENCODE_API_FUNCTION_LIST>(), 2552);
    }

    /// `NV_ENC_RECONFIGURE_PARAMS` layout cross-checked against nvEncodeAPI.h
    /// (NVENC SDK 13). The header struct is `{ uint32 version; uint32 reserved;
    /// NV_ENC_INITIALIZE_PARAMS reInitEncodeParams; uint32 bitfield; uint32
    /// reserved2; }` — i.e. reInitEncodeParams is at offset 8 (a `reserved`
    /// word precedes it) and there's a trailing reserved2 word, so sizeof =
    /// 4+4+1800+4+4 = 1816 (NOT 1808). The reconfigure call hands NVENC this
    /// whole struct; a wrong size/offset shifts the embedded init params (the
    /// class of bug that breaks the encoder).
    #[test]
    fn nvenc_reconfigure_params_matches_header_size() {
        use std::mem::size_of;
        assert_eq!(
            size_of::<super::NV_ENC_INITIALIZE_PARAMS>(),
            1800,
            "init params size drifted"
        );
        assert_eq!(size_of::<super::NV_ENC_RECONFIGURE_PARAMS>(), 1816);
        assert_eq!(
            std::mem::offset_of!(super::NV_ENC_RECONFIGURE_PARAMS, re_init_encode_params),
            8
        );
        assert_eq!(
            std::mem::offset_of!(super::NV_ENC_RECONFIGURE_PARAMS, flags),
            1808
        );
        assert_eq!(
            std::mem::offset_of!(super::NV_ENC_RECONFIGURE_PARAMS, reserved2),
            1812
        );
    }

    /// EXACT per-frame struct sizes cross-checked against nvEncodeAPI.h (NVENC
    /// SDK 13) via a C `sizeof` probe. These are the structs `encode_dmabuf`
    /// passes to NVENC every frame; a wrong size means a mis-modeled field.
    #[test]
    fn nvenc_perframe_structs_match_header_byte_sizes() {
        use std::mem::size_of;
        assert_eq!(size_of::<super::NV_ENC_REGISTER_RESOURCE>(), 1536);
        assert_eq!(size_of::<super::NV_ENC_MAP_INPUT_RESOURCE>(), 1544);
        assert_eq!(size_of::<super::NV_ENC_CODEC_PIC_PARAMS>(), 1544);
        assert_eq!(size_of::<super::NV_ENC_PIC_PARAMS>(), 3360);
        assert_eq!(size_of::<super::NV_ENC_LOCK_BITSTREAM>(), 1544);
    }

    /// CUDA descriptor sizes vs cuda.h (CUDA 12.8) `sizeof`. cuTexObjectCreate
    /// reads both; a wrong size is a candidate cause of an INVALID_VALUE.
    #[test]
    fn cuda_desc_structs_match_header_byte_sizes() {
        use std::mem::size_of;
        assert_eq!(size_of::<super::CudaResourceDesc>(), 144);
        assert_eq!(size_of::<super::CudaTextureDesc>(), 104);
        // Host-pixels path descriptors (cuda.h CUDA 12.x).
        assert_eq!(size_of::<super::CUDA_ARRAY_DESCRIPTOR>(), 24);
        assert_eq!(size_of::<super::CUDA_MEMCPY2D>(), 128);
    }

    /// Field offsets of CUDA_MEMCPY2D vs cuda.h (CUDA 12.x). The implicit 4-byte
    /// padding after each `*MemoryType` u32 must place the following pointer at
    /// an 8-aligned offset, exactly as the C ABI lays it out.
    #[test]
    fn cuda_memcpy2d_field_offsets_match_header() {
        macro_rules! off {
            ($T:ty, $f:ident) => {{
                let base = std::mem::MaybeUninit::<$T>::uninit();
                let p = base.as_ptr();
                // SAFETY: offset-of only; no field reads.
                unsafe { (std::ptr::addr_of!((*p).$f) as usize) - (p as usize) }
            }};
        }
        use super::CUDA_MEMCPY2D;
        assert_eq!(off!(CUDA_MEMCPY2D, src_x_in_bytes), 0);
        assert_eq!(off!(CUDA_MEMCPY2D, src_y), 8);
        assert_eq!(off!(CUDA_MEMCPY2D, src_memory_type), 16);
        assert_eq!(off!(CUDA_MEMCPY2D, src_host), 24);
        assert_eq!(off!(CUDA_MEMCPY2D, src_device), 32);
        assert_eq!(off!(CUDA_MEMCPY2D, src_array), 40);
        assert_eq!(off!(CUDA_MEMCPY2D, src_pitch), 48);
        assert_eq!(off!(CUDA_MEMCPY2D, dst_x_in_bytes), 56);
        assert_eq!(off!(CUDA_MEMCPY2D, dst_y), 64);
        assert_eq!(off!(CUDA_MEMCPY2D, dst_memory_type), 72);
        assert_eq!(off!(CUDA_MEMCPY2D, dst_host), 80);
        assert_eq!(off!(CUDA_MEMCPY2D, dst_device), 88);
        assert_eq!(off!(CUDA_MEMCPY2D, dst_array), 96);
        assert_eq!(off!(CUDA_MEMCPY2D, dst_pitch), 104);
        assert_eq!(off!(CUDA_MEMCPY2D, width_in_bytes), 112);
        assert_eq!(off!(CUDA_MEMCPY2D, height), 120);
    }

    /// Offsets of the fields NVENC reads on the encode path must match the
    /// header exactly (offsets from the C `offsetof` probe).
    #[test]
    fn nvenc_perframe_field_offsets_match_header() {
        macro_rules! off {
            ($T:ty, $f:ident) => {{
                let base = std::mem::MaybeUninit::<$T>::uninit();
                let p = base.as_ptr();
                // SAFETY: offset-of only; no field reads.
                unsafe { (std::ptr::addr_of!((*p).$f) as usize) - (p as usize) }
            }};
        }
        // NV_ENC_REGISTER_RESOURCE
        assert_eq!(
            off!(super::NV_ENC_REGISTER_RESOURCE, registered_resource),
            32
        );
        assert_eq!(off!(super::NV_ENC_REGISTER_RESOURCE, buffer_format), 40);
        assert_eq!(off!(super::NV_ENC_REGISTER_RESOURCE, input_fence_point), 48);
        assert_eq!(off!(super::NV_ENC_REGISTER_RESOURCE, chroma_offset), 56);
        // NV_ENC_MAP_INPUT_RESOURCE
        assert_eq!(off!(super::NV_ENC_MAP_INPUT_RESOURCE, mapped_resource), 24);
        // NV_ENC_PIC_PARAMS
        assert_eq!(off!(super::NV_ENC_PIC_PARAMS, input_buffer), 40);
        assert_eq!(off!(super::NV_ENC_PIC_PARAMS, buffer_fmt), 64);
        assert_eq!(off!(super::NV_ENC_PIC_PARAMS, picture_type), 72);
        assert_eq!(off!(super::NV_ENC_PIC_PARAMS, codec_pic_params), 80);
        assert_eq!(
            off!(super::NV_ENC_PIC_PARAMS, me_hint_counts_per_block),
            1624
        );
        // NV_ENC_LOCK_BITSTREAM (the two fields the caller reads out)
        assert_eq!(off!(super::NV_ENC_LOCK_BITSTREAM, output_bitstream), 8);
        assert_eq!(
            off!(super::NV_ENC_LOCK_BITSTREAM, bitstream_size_in_bytes),
            36
        );
        assert_eq!(off!(super::NV_ENC_LOCK_BITSTREAM, bitstream_buffer_ptr), 56);
    }

    /// `encodeConfig` and `tuningInfo` offsets must match the header exactly —
    /// if `encodeConfig` is wrong the driver reads a garbage config pointer.
    #[test]
    fn nvenc_init_params_field_offsets_match_header() {
        let base = std::mem::MaybeUninit::<super::NV_ENC_INITIALIZE_PARAMS>::uninit();
        let p = base.as_ptr();
        // SAFETY: computing field offsets on an uninit value (no reads of the
        // fields themselves), then casting back to usize.
        unsafe {
            let origin = p as usize;
            assert_eq!(
                (std::ptr::addr_of!((*p).encode_config) as usize) - origin,
                88
            );
            assert_eq!(
                (std::ptr::addr_of!((*p).tuning_info) as usize) - origin,
                136
            );
        }
    }

    #[test]
    fn recorder_try_new_is_none_without_gpu() {
        assert!(super::CudaNvencRecorder::try_new(1920, 1080).is_none());
    }

    // ---------------------------------------------------------------------
    // NV2.5 — live encode smoke. We reuse the crate's typed `dmabuf::gbm_ffi`
    // (gbm_create_device / gbm_device_destroy / gbm_bo_map / gbm_bo_unmap /
    // gbm_bo_destroy) so the extern signatures don't clash, and declare ONLY
    // the three symbols `gbm_ffi` is missing (gbm_bo_create / gbm_bo_get_fd /
    // gbm_bo_get_stride). All resolve against `-l gbm` from build.rs.
    // ---------------------------------------------------------------------
    use crate::dmabuf::gbm_ffi::{self, GbmBo, GbmDevice};
    use std::os::raw::c_void;

    const GBM_FORMAT_ARGB8888: u32 = 0x3432_5241; // 'AR24'
    const GBM_BO_TRANSFER_WRITE: u32 = 1 << 1;
    /// `DRM_FORMAT_MOD_LINEAR` == 0.
    const DRM_FORMAT_MOD_LINEAR: u64 = 0;

    extern "C" {
        // NVIDIA's GBM rejects the plain `gbm_bo_create` path (returns null on an
        // L40); only `gbm_bo_create_with_modifiers` succeeds — mirror the proven
        // NV1 probe (examples/egl_cuda_interop_spike.rs).
        fn gbm_bo_create_with_modifiers(
            dev: *mut GbmDevice,
            w: u32,
            h: u32,
            format: u32,
            modifiers: *const u64,
            count: u32,
        ) -> *mut GbmBo;
        fn gbm_bo_get_fd(bo: *mut GbmBo) -> i32;
        fn gbm_bo_get_stride(bo: *mut GbmBo) -> u32;
        fn gbm_bo_get_modifier(bo: *mut GbmBo) -> u64;
    }

    #[test]
    #[ignore] // live — needs an NVIDIA GPU; run with `--ignored` on image 1432
    fn live_encode_smoke() {
        const W: u32 = 1920;
        const H: u32 = 1080;

        // 1. open renderD128 + gbm_create_device.
        let drm_fd = unsafe {
            libc::open(
                c"/dev/dri/renderD128".as_ptr() as *const std::os::raw::c_char,
                libc::O_RDWR,
            )
        };
        if drm_fd < 0 {
            eprintln!("no /dev/dri/renderD128, skipping");
            return;
        }
        let dev = unsafe { gbm_ffi::gbm_create_device(drm_fd) };
        if dev.is_null() {
            eprintln!("gbm_create_device failed, skipping");
            unsafe { libc::close(drm_fd) };
            return;
        }

        // 2. allocate an ARGB8888 BO via gbm_bo_create_with_modifiers (NVIDIA's
        //    GBM rejects the plain gbm_bo_create path). Pick a modifier from the
        //    EGL-importable set: prefer the FIRST non-LINEAR (tiled) modifier,
        //    falling back to LINEAR (0) if none is offered.
        let importable = crate::dmabuf::egl_importable_bgra_modifiers();
        let chosen_mod = importable
            .iter()
            .copied()
            .find(|&m| m != DRM_FORMAT_MOD_LINEAR)
            .unwrap_or(DRM_FORMAT_MOD_LINEAR);
        eprintln!("importable modifiers={importable:?} -> requesting 0x{chosen_mod:016x}");
        let mod_list = [chosen_mod];
        let bo = unsafe {
            gbm_bo_create_with_modifiers(
                dev,
                W,
                H,
                GBM_FORMAT_ARGB8888,
                mod_list.as_ptr(),
                mod_list.len() as u32,
            )
        };
        if bo.is_null() {
            eprintln!("gbm_bo_create_with_modifiers returned null, can't test");
            unsafe { gbm_ffi::gbm_device_destroy(dev) };
            unsafe { libc::close(drm_fd) };
            return;
        }
        // The actual modifier the driver allocated (may differ from requested).
        let bo_modifier = unsafe { gbm_bo_get_modifier(bo) };
        eprintln!("allocated BO modifier=0x{bo_modifier:016x}");

        // 3. Best-effort CPU fill: NVIDIA tiled BOs typically aren't
        //    CPU-mappable, so map may return null. If so, skip the fill and
        //    encode uninitialized content — this live test's gate is that the
        //    EGL→CUDA→kernel→NVENC chain produces a valid decodable H.264
        //    stream. (The CSC color math is already verified offline byte-for-
        //    byte vs vulkan_compute.glsl.) If the fill DID work, colors are
        //    checkable from /tmp/nvenc_out.h264.
        let mut map_stride: u32 = 0;
        let mut map_data: *mut c_void = std::ptr::null_mut();
        let ptr = unsafe {
            gbm_ffi::gbm_bo_map(
                bo,
                0,
                0,
                W,
                H,
                GBM_BO_TRANSFER_WRITE,
                &mut map_stride,
                &mut map_data,
            )
        };
        if ptr.is_null() || map_stride == 0 {
            eprintln!(
                "gbm_bo_map unavailable (tiled BO) — encoding uninitialized content; validating pipeline + stream validity only"
            );
        } else {
            // Quadrant colors as [B, G, R, A] (ARGB8888 little-endian byte order).
            let top_left = [0u8, 0, 255, 255]; // red
            let top_right = [0u8, 255, 0, 255]; // green
            let bottom_left = [255u8, 0, 0, 255]; // blue
            let bottom_right = [255u8, 255, 255, 255]; // white
            let base = ptr as *mut u8;
            for y in 0..H {
                let row = unsafe { base.add((y as usize) * (map_stride as usize)) };
                for x in 0..W {
                    let px = if y < H / 2 {
                        if x < W / 2 {
                            top_left
                        } else {
                            top_right
                        }
                    } else if x < W / 2 {
                        bottom_left
                    } else {
                        bottom_right
                    };
                    unsafe {
                        let p = row.add((x as usize) * 4);
                        p.copy_from_nonoverlapping(px.as_ptr(), 4);
                    }
                }
            }
            unsafe { gbm_ffi::gbm_bo_unmap(bo, map_data) };
        }

        // 4. fd + stride from the BO.
        let fd = unsafe { gbm_bo_get_fd(bo) };
        let stride = unsafe { gbm_bo_get_stride(bo) };
        eprintln!("gbm bo: fd={fd} stride={stride} map_stride={map_stride}");

        // 5. bring up the recorder — no-op without a GPU.
        let Some(mut rec) = super::CudaNvencRecorder::try_new(W, H) else {
            eprintln!("no GPU, skipping");
            // The BO owns the fd; destroy reclaims it. Don't close `fd`.
            unsafe { gbm_ffi::gbm_bo_destroy(bo) };
            unsafe { gbm_ffi::gbm_device_destroy(dev) };
            unsafe { libc::close(drm_fd) };
            return;
        };
        let cp_len = rec.codec_private().len();
        eprintln!("codec_private = {cp_len} bytes");

        // 6. concat codec_private + ~12 frames of Annex-B.
        let mut out = Vec::new();
        out.extend_from_slice(rec.codec_private());
        for i in 0..12u32 {
            // Pass the BO's ACTUAL modifier (tiled or LINEAR) so the EGL import
            // describes the buffer layout correctly.
            match rec.encode_dmabuf(fd, bo_modifier, W, H, stride, (i as i64) * 33_333, true) {
                Some(nal) => {
                    eprintln!(
                        "frame {i}: {} bytes idr={}",
                        nal.data.len(),
                        nal.is_keyframe
                    );
                    out.extend_from_slice(&nal.data);
                }
                None => eprintln!("frame {i}: encode returned None"),
            }
        }

        // 7. write the output.
        std::fs::write("/tmp/nvenc_out.h264", &out).unwrap();
        eprintln!("wrote /tmp/nvenc_out.h264 = {} bytes total", out.len());

        // 8. cleanup. encode_dmabuf dups the fd internally via the EGL import,
        //    so the original fd stays owned by the BO — let gbm_bo_destroy
        //    reclaim it; do NOT close `fd` manually.
        unsafe { gbm_ffi::gbm_bo_destroy(bo) };
        unsafe { gbm_ffi::gbm_device_destroy(dev) };
        unsafe { libc::close(drm_fd) };

        // At least one frame must have encoded beyond codec_private.
        assert!(
            out.len() > cp_len,
            "no frames encoded: out.len()={} cp_len={cp_len}",
            out.len()
        );
    }
}
