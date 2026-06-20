// SPDX-License-Identifier: Apache-2.0

//! NV1 spike: EGL -> CUDA interop probe.
//!
//! `waymux-session` must encode KWin 6's GPU-tiled dmabuf on NVIDIA. NVIDIA's
//! Vulkan can't import dmabufs, but EGL can, and CUDA can register an EGLImage
//! (`cuGraphicsEGLRegisterImage`). This probe answers: *does that interop yield
//! an NVENC-usable CUDA surface, and is it a pitch-linear devptr or a
//! block-linear array?*
//!
//! It allocates a tiled test BO, imports it via EGL as an EGLImage, registers
//! that with CUDA, and PRINTS the resulting `CUeglFrame` shape.
//!
//! This is a KEPT diagnostic — NOT shipped in the binary. It dlopens
//! `libcuda.so.1` at runtime, so it compiles fine without CUDA installed, but
//! it can only be *run* on a box with an L40 (or any NVIDIA) GPU.
//!
//! Build (the only verification done in CI / dev box without a GPU):
//!   cargo build -p waymux-session --example egl_cuda_interop_spike
//!
//! Run (NVIDIA GPU host only):
//!   cargo run -p waymux-session --example egl_cuda_interop_spike
//!
//! Every failure prints a message and returns early — a probe must not panic on
//! a host that lacks the GPU / drivers.

use khronos_egl as egl;
use std::os::raw::{c_char, c_void};

// ---------------------------------------------------------------------------
// DRM / GBM constants
// ---------------------------------------------------------------------------

/// `DRM_FORMAT_ARGB8888` — little-endian B,G,R,A. KWin composites BGRA.
const DRM_FORMAT_ARGB8888: u32 = 0x3432_5241; // 'AR24'
/// GBM uses the same fourcc value for its format token.
const GBM_FORMAT_ARGB8888: u32 = DRM_FORMAT_ARGB8888;
/// `DRM_FORMAT_MOD_LINEAR` == 0. Anything else is a tiled/compressed layout.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

const TEST_WIDTH: u32 = 1920;
const TEST_HEIGHT: u32 = 1080;

// ---------------------------------------------------------------------------
// EGL constants (mirrors crates/waymux-session/src/dmabuf.rs egl_ext)
// ---------------------------------------------------------------------------

const EGL_PLATFORM_GBM_KHR: egl::Enum = 0x31D7;
const EGL_LINUX_DMA_BUF_EXT: egl::Enum = 0x3270;
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
// EGL extension fn FFI (declared inline — the crate's egl_ext module is
// pub(crate) and invisible to an example crate).
// ---------------------------------------------------------------------------

/// `EGLBoolean eglQueryDmaBufModifiersEXT(EGLDisplay, EGLint format,
///   EGLint max_modifiers, EGLuint64KHR *modifiers, EGLBoolean *external_only,
///   EGLint *num_modifiers)`
type EglQueryDmaBufModifiersExt = unsafe extern "system" fn(
    dpy: *mut c_void,
    format: i32,
    max_modifiers: i32,
    modifiers: *mut u64,
    external_only: *mut u32,
    num_modifiers: *mut i32,
) -> u32;

/// `EGLImageKHR eglCreateImageKHR(EGLDisplay, EGLContext, EGLenum target,
///   EGLClientBuffer buffer, const EGLint *attrib_list)`
type EglCreateImageKHR = unsafe extern "C" fn(
    dpy: *mut c_void,
    ctx: *mut c_void,
    target: egl::Enum,
    buffer: *mut c_void,
    attrib_list: *const i32,
) -> *mut c_void;

/// `EGLBoolean eglDestroyImageKHR(EGLDisplay, EGLImageKHR)`
type EglDestroyImageKHR = unsafe extern "C" fn(dpy: *mut c_void, image: *mut c_void) -> u32;

// ---------------------------------------------------------------------------
// GBM FFI (declared inline). The crate's gbm_ffi is pub(crate).
// ---------------------------------------------------------------------------

// libgbm is linked via the crate's build.rs (`cargo:rustc-link-lib=gbm`),
// which applies to examples too.
extern "C" {
    fn gbm_create_device(fd: i32) -> *mut c_void;
    fn gbm_device_destroy(dev: *mut c_void);
    fn gbm_bo_create_with_modifiers(
        dev: *mut c_void,
        width: u32,
        height: u32,
        format: u32,
        modifiers: *const u64,
        count: u32,
    ) -> *mut c_void;
    fn gbm_bo_get_fd(bo: *mut c_void) -> i32;
    fn gbm_bo_get_stride(bo: *mut c_void) -> u32;
    fn gbm_bo_get_modifier(bo: *mut c_void) -> u64;
    fn gbm_bo_destroy(bo: *mut c_void);
    // `gbm_bo_get_offset(bo, plane)` exists on modern libgbm. If a host's
    // libgbm is too old to export it the link would fail; we only use plane 0
    // (offset 0) so we don't bind it and just pass 0 below.
}

// ---------------------------------------------------------------------------
// CUDA driver-API FFI + CUeglFrame (vendored from cudaEGL.h). Field ORDER is
// load-bearing for the live run — see report notes.
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

type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(dev: *mut i32, ordinal: i32) -> i32;
type CuCtxCreateV2 = unsafe extern "C" fn(pctx: *mut *mut c_void, flags: u32, dev: i32) -> i32;
type CuGraphicsEglRegisterImage =
    unsafe extern "C" fn(pres: *mut *mut c_void, image: *mut c_void, flags: u32) -> i32;
type CuGraphicsResourceGetMappedEglFrame =
    unsafe extern "C" fn(frame: *mut CUeglFrame, res: *mut c_void, index: u32, mip: u32) -> i32;
type CuGraphicsUnregisterResource = unsafe extern "C" fn(res: *mut c_void) -> i32;
type CuGetErrorString = unsafe extern "C" fn(err: i32, pstr: *mut *const c_char) -> i32;

/// Holds the dlopened libcuda symbols. The `Library` must outlive the fn
/// pointers, so it is kept in the struct.
struct Cuda {
    _lib: libloading::Library,
    init: CuInit,
    device_get: CuDeviceGet,
    ctx_create: CuCtxCreateV2,
    register_image: CuGraphicsEglRegisterImage,
    get_mapped_frame: CuGraphicsResourceGetMappedEglFrame,
    unregister: CuGraphicsUnregisterResource,
    get_error_string: CuGetErrorString,
}

impl Cuda {
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

fn load_cuda() -> Option<Cuda> {
    let lib = match unsafe { libloading::Library::new("libcuda.so.1") } {
        Ok(l) => l,
        Err(e) => {
            eprintln!("CUDA: failed to dlopen libcuda.so.1: {e}");
            return None;
        }
    };
    // SAFETY: signatures match the CUDA driver API; symbols come from libcuda.
    unsafe {
        macro_rules! sym {
            ($name:expr) => {
                match lib.get::<_>($name) {
                    Ok(s) => *s,
                    Err(e) => {
                        eprintln!(
                            "CUDA: symbol {} missing: {e}",
                            String::from_utf8_lossy($name)
                        );
                        return None;
                    }
                }
            };
        }
        let init: CuInit = sym!(b"cuInit\0");
        let device_get: CuDeviceGet = sym!(b"cuDeviceGet\0");
        // The versioned symbol name `cuCtxCreate_v2` is what the driver actually
        // exports; the unversioned `cuCtxCreate` is a macro in the C headers.
        let ctx_create: CuCtxCreateV2 = sym!(b"cuCtxCreate_v2\0");
        let register_image: CuGraphicsEglRegisterImage = sym!(b"cuGraphicsEGLRegisterImage\0");
        let get_mapped_frame: CuGraphicsResourceGetMappedEglFrame =
            sym!(b"cuGraphicsResourceGetMappedEglFrame\0");
        let unregister: CuGraphicsUnregisterResource = sym!(b"cuGraphicsUnregisterResource\0");
        let get_error_string: CuGetErrorString = sym!(b"cuGetErrorString\0");
        Some(Cuda {
            _lib: lib,
            init,
            device_get,
            ctx_create,
            register_image,
            get_mapped_frame,
            unregister,
            get_error_string,
        })
    }
}

fn main() {
    eprintln!("=== NV1 EGL->CUDA interop spike ===");

    // -- Step 1: open render node + create GBM device ----------------------
    let drm_fd = unsafe {
        libc::open(
            c"/dev/dri/renderD128".as_ptr() as *const c_char,
            libc::O_RDWR | libc::O_CLOEXEC,
        )
    };
    if drm_fd < 0 {
        eprintln!("[1] open /dev/dri/renderD128 failed (no GPU render node?) — exiting");
        return;
    }
    eprintln!("[1] opened /dev/dri/renderD128 (fd={drm_fd})");

    let gbm = unsafe { gbm_create_device(drm_fd) };
    if gbm.is_null() {
        eprintln!("[1] gbm_create_device failed — exiting");
        unsafe { libc::close(drm_fd) };
        return;
    }
    eprintln!("[1] gbm_create_device OK");

    // -- Step 2: EGL bootstrap (GBM platform display) ----------------------
    let lib = match unsafe { libloading::Library::new("libEGL.so.1") } {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[2] failed to load libEGL.so.1: {e} — exiting");
            unsafe { gbm_device_destroy(gbm) };
            unsafe { libc::close(drm_fd) };
            return;
        }
    };
    let egl_inst = match unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required_from(lib) } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("[2] EGL dynamic load failed: {e:?} — exiting");
            unsafe { gbm_device_destroy(gbm) };
            unsafe { libc::close(drm_fd) };
            return;
        }
    };
    let display = match unsafe {
        egl_inst.get_platform_display(EGL_PLATFORM_GBM_KHR, gbm, &[egl::ATTRIB_NONE])
    } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[2] get_platform_display(GBM) failed: {e:?} — exiting");
            return;
        }
    };
    if let Err(e) = egl_inst.initialize(display) {
        eprintln!("[2] eglInitialize failed: {e:?} — exiting");
        return;
    }
    eprintln!("[2] EGL initialized on GBM platform display");
    let dpy_ptr = display.as_ptr();

    // -- Step 3: query importable modifiers, pick a tiled one --------------
    let query_raw = match egl_inst.get_proc_address("eglQueryDmaBufModifiersEXT") {
        Some(p) => p,
        None => {
            eprintln!("[3] eglQueryDmaBufModifiersEXT not exposed — exiting");
            return;
        }
    };
    let query: EglQueryDmaBufModifiersExt = unsafe { std::mem::transmute(query_raw) };

    let mut count: i32 = 0;
    let ok = unsafe {
        query(
            dpy_ptr,
            DRM_FORMAT_ARGB8888 as i32,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut count,
        )
    };
    if ok == 0 || count <= 0 {
        eprintln!("[3] no importable modifiers for ARGB8888 (ok={ok}, count={count}) — exiting");
        return;
    }
    let mut mods = vec![0u64; count as usize];
    let mut ext_only = vec![0u32; count as usize];
    let mut got: i32 = 0;
    let ok2 = unsafe {
        query(
            dpy_ptr,
            DRM_FORMAT_ARGB8888 as i32,
            count,
            mods.as_mut_ptr(),
            ext_only.as_mut_ptr(),
            &mut got,
        )
    };
    if ok2 == 0 {
        eprintln!("[3] eglQueryDmaBufModifiersEXT (populate) failed — exiting");
        return;
    }
    mods.truncate(got as usize);
    eprintln!("[3] importable ARGB8888 modifiers ({got}):");
    for (i, m) in mods.iter().enumerate() {
        let ext = ext_only.get(i).copied().unwrap_or(0);
        eprintln!("      0x{m:016x}  external_only={ext}");
    }
    let chosen = mods.iter().copied().find(|&m| m != DRM_FORMAT_MOD_LINEAR);
    let modifier = match chosen {
        Some(m) => {
            eprintln!("[3] chosen non-LINEAR modifier: 0x{m:016x}");
            m
        }
        None => {
            eprintln!("[3] no tiled modifier available — can't test tiled on this host");
            // Clean exit 0: this host simply can't exercise the tiled path.
            unsafe { gbm_device_destroy(gbm) };
            unsafe { libc::close(drm_fd) };
            return;
        }
    };

    // -- Step 4: allocate a tiled BO --------------------------------------
    let mod_list = [modifier];
    let bo = unsafe {
        gbm_bo_create_with_modifiers(
            gbm,
            TEST_WIDTH,
            TEST_HEIGHT,
            GBM_FORMAT_ARGB8888,
            mod_list.as_ptr(),
            mod_list.len() as u32,
        )
    };
    if bo.is_null() {
        eprintln!("[4] gbm_bo_create_with_modifiers failed — exiting");
        unsafe { gbm_device_destroy(gbm) };
        unsafe { libc::close(drm_fd) };
        return;
    }
    let bo_fd = unsafe { gbm_bo_get_fd(bo) };
    let stride = unsafe { gbm_bo_get_stride(bo) };
    let bo_mod = unsafe { gbm_bo_get_modifier(bo) };
    let offset: u32 = 0; // plane 0; gbm_bo_get_offset(bo,0) would also be 0.
    eprintln!(
        "[4] tiled BO created: fd={bo_fd} stride={stride} offset={offset} modifier=0x{bo_mod:016x}"
    );
    if bo_fd < 0 {
        eprintln!("[4] gbm_bo_get_fd returned a bad fd — exiting");
        unsafe { gbm_bo_destroy(bo) };
        unsafe { gbm_device_destroy(gbm) };
        unsafe { libc::close(drm_fd) };
        return;
    }

    // -- Step 5: import the BO fd as an EGLImage --------------------------
    let create_image_raw = match egl_inst.get_proc_address("eglCreateImageKHR") {
        Some(p) => p,
        None => {
            eprintln!("[5] eglCreateImageKHR not exposed — exiting");
            cleanup_bo(bo, gbm, drm_fd);
            return;
        }
    };
    let create_image: EglCreateImageKHR = unsafe { std::mem::transmute(create_image_raw) };
    let destroy_image: Option<EglDestroyImageKHR> = egl_inst
        .get_proc_address("eglDestroyImageKHR")
        .map(|p| unsafe { std::mem::transmute(p) });

    // Mirror egl_readback's attribute usage: width/height/fourcc, plane-0
    // fd/offset/pitch, then plane-0 modifier lo/hi (since it's non-LINEAR).
    let attrs: [i32; 19] = [
        EGL_WIDTH,
        TEST_WIDTH as i32,
        EGL_HEIGHT,
        TEST_HEIGHT as i32,
        EGL_LINUX_DRM_FOURCC_EXT,
        DRM_FORMAT_ARGB8888 as i32,
        EGL_DMA_BUF_PLANE0_FD_EXT,
        bo_fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT,
        offset as i32,
        EGL_DMA_BUF_PLANE0_PITCH_EXT,
        stride as i32,
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
        (modifier & 0xFFFF_FFFF) as i32,
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
        (modifier >> 32) as i32,
        EGL_NONE,
        // pad to keep a clean even pair count; trailing entries past EGL_NONE
        // are ignored by EGL.
        EGL_NONE,
        EGL_NONE,
    ];

    // EGL_NO_CONTEXT == null.
    let egl_image = unsafe {
        create_image(
            dpy_ptr,
            std::ptr::null_mut(),
            EGL_LINUX_DMA_BUF_EXT,
            std::ptr::null_mut(),
            attrs.as_ptr(),
        )
    };
    if egl_image.is_null() {
        let egl_err = egl_inst.get_error();
        eprintln!("[5] eglCreateImageKHR returned EGL_NO_IMAGE (egl error {egl_err:?}) — exiting");
        cleanup_bo(bo, gbm, drm_fd);
        return;
    }
    eprintln!("[5] eglCreateImageKHR OK: EGLImage={egl_image:p} (non-null)");

    // -- Step 6: load CUDA, init, device, context ------------------------
    let cuda = match load_cuda() {
        Some(c) => c,
        None => {
            eprintln!("[6] CUDA unavailable — exiting (EGL import already proven above)");
            if let Some(d) = destroy_image {
                unsafe { d(dpy_ptr, egl_image) };
            }
            cleanup_bo(bo, gbm, drm_fd);
            return;
        }
    };

    let rc = unsafe { (cuda.init)(0) };
    eprintln!("[6] cuInit(0) -> {}", cuda.err(rc));
    if rc != 0 {
        cleanup_image(
            &cuda,
            None,
            egl_image,
            destroy_image,
            dpy_ptr,
            bo,
            gbm,
            drm_fd,
        );
        return;
    }

    let mut dev: i32 = 0;
    let rc = unsafe { (cuda.device_get)(&mut dev, 0) };
    eprintln!("[6] cuDeviceGet(0) -> {} (dev={dev})", cuda.err(rc));
    if rc != 0 {
        cleanup_image(
            &cuda,
            None,
            egl_image,
            destroy_image,
            dpy_ptr,
            bo,
            gbm,
            drm_fd,
        );
        return;
    }

    let mut ctx: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { (cuda.ctx_create)(&mut ctx, 0, dev) };
    eprintln!("[6] cuCtxCreate_v2 -> {} (ctx={ctx:p})", cuda.err(rc));
    if rc != 0 {
        cleanup_image(
            &cuda,
            None,
            egl_image,
            destroy_image,
            dpy_ptr,
            bo,
            gbm,
            drm_fd,
        );
        return;
    }

    // -- Step 7: register the EGLImage with CUDA and map the EGL frame ----
    let mut res: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { (cuda.register_image)(&mut res, egl_image, 0) };
    eprintln!(
        "[7] cuGraphicsEGLRegisterImage -> {} (res={res:p})",
        cuda.err(rc)
    );
    if rc != 0 {
        cleanup_image(
            &cuda,
            None,
            egl_image,
            destroy_image,
            dpy_ptr,
            bo,
            gbm,
            drm_fd,
        );
        return;
    }

    // Zeroed CUeglFrame; the union arm is irrelevant for a zero init.
    let mut frame: CUeglFrame = unsafe { std::mem::zeroed() };
    let rc = unsafe { (cuda.get_mapped_frame)(&mut frame, res, 0, 0) };
    eprintln!(
        "[7] cuGraphicsResourceGetMappedEglFrame -> {}",
        cuda.err(rc)
    );
    if rc != 0 {
        cleanup_image(
            &cuda,
            Some(res),
            egl_image,
            destroy_image,
            dpy_ptr,
            bo,
            gbm,
            drm_fd,
        );
        return;
    }

    // -- Step 8: print the CUeglFrame shape — THE DELIVERABLE -------------
    let frame_type_str = match frame.frame_type {
        0 => "ARRAY (block-linear cuArray)",
        1 => "PITCH (pitch-linear devptr)",
        other => {
            // keep `other` referenced so the arm is meaningful
            eprintln!("      (unexpected frameType value {other})");
            "UNKNOWN"
        }
    };
    let ptr0 = unsafe { frame.frame.pitch[0] };
    let array0 = unsafe { frame.frame.array[0] };
    eprintln!("=== CUeglFrame shape (the answer NV1 needs) ===");
    eprintln!(
        "    frameType      = {} ({})",
        frame.frame_type, frame_type_str
    );
    eprintln!("    cuFormat       = {}", frame.cu_format);
    eprintln!("    eglColorFormat = {}", frame.egl_color_format);
    eprintln!("    width          = {}", frame.width);
    eprintln!("    height         = {}", frame.height);
    eprintln!("    depth          = {}", frame.depth);
    eprintln!("    pitch          = {}", frame.pitch);
    eprintln!("    planeCount     = {}", frame.plane_count);
    eprintln!("    numChannels    = {}", frame.num_channels);
    eprintln!(
        "    frame.ptr[0]   = {:p} (non-null: {})",
        ptr0,
        !ptr0.is_null()
    );
    eprintln!(
        "    frame.array[0] = {:p} (non-null: {})",
        array0,
        !array0.is_null()
    );
    eprintln!(
        "    => NVENC interop verdict: {}",
        match frame.frame_type {
            0 => "ARRAY — needs a copy/blit to a pitch-linear devptr before NVENC, or use the array path",
            1 => "PITCH — directly usable as an NVENC input devptr (best case)",
            _ => "INDETERMINATE",
        }
    );

    // -- Step 9: cleanup --------------------------------------------------
    cleanup_image(
        &cuda,
        Some(res),
        egl_image,
        destroy_image,
        dpy_ptr,
        bo,
        gbm,
        drm_fd,
    );
    eprintln!("[9] cleanup done — spike complete");
}

/// Destroy the test BO + GBM device + render fd.
fn cleanup_bo(bo: *mut c_void, gbm: *mut c_void, drm_fd: i32) {
    unsafe {
        if !bo.is_null() {
            gbm_bo_destroy(bo);
        }
        if !gbm.is_null() {
            gbm_device_destroy(gbm);
        }
        if drm_fd >= 0 {
            libc::close(drm_fd);
        }
    }
}

/// Best-effort: unregister the CUDA resource, destroy the EGLImage, then BO.
#[allow(clippy::too_many_arguments)]
fn cleanup_image(
    cuda: &Cuda,
    res: Option<*mut c_void>,
    egl_image: *mut c_void,
    destroy_image: Option<EglDestroyImageKHR>,
    dpy_ptr: *mut c_void,
    bo: *mut c_void,
    gbm: *mut c_void,
    drm_fd: i32,
) {
    if let Some(res) = res {
        if !res.is_null() {
            let rc = unsafe { (cuda.unregister)(res) };
            eprintln!("    cuGraphicsUnregisterResource -> {}", cuda.err(rc));
        }
    }
    if let Some(d) = destroy_image {
        if !egl_image.is_null() {
            unsafe { d(dpy_ptr, egl_image) };
        }
    }
    cleanup_bo(bo, gbm, drm_fd);
}
