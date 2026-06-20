// SPDX-License-Identifier: Apache-2.0

//! Lossless video recording via ffmpeg FFV1/MKV.
//!
//! Architecture: outer_view forwards each KWin dmabuf to niri AND, when
//! recording is active, hands a clone of the same dmabuf (Arc) plus a set
//! of `InnerBufferHold` ref-counts to the recording thread via a bounded
//! mpsc channel. The recording thread does the GPU readback (slow on AMD —
//! 30+ms at 1080p) off the compositor's critical path, then drops its
//! holds. The buffer is released back to KWin only after BOTH niri and the
//! recording reader are done, so KWin can never overwrite a buffer mid-read.

// recording protocol has reserved variants + experimental helper functions
#![allow(dead_code, unused_assignments)]

use std::io::Write;
use std::process::Stdio;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Condvar, Mutex,
};
use tracing::{info, warn};
use wayland_server::protocol::wl_buffer::WlBuffer;

use crate::dmabuf::DmabufBufferData;

/// How long each codec's recording thread waits for its first frame
/// before aborting. KDE Plasma 5 commits its first surface ~3–8 s after
/// kwin_wayland starts (kded5 + kglobalaccel5 + kactivitymanagerd +
/// kbuildsycoca5 + plasmashell all init before any wl_surface.commit).
/// The original 5 s timeout 5s-aborted any `waymux record start <kde>`
/// issued right after `waymux-launch-kde-p5.sh`. 30 s is patient enough
/// for any reasonable session-launch + KDE-startup combination while
/// still surfacing genuinely-hung sessions.
const FIRST_FRAME_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Wraps an inner-client wl_buffer and releases it on drop.
///
/// Outer_view creates one per inner buffer, then hands cloned `Arc`s to:
///  - the OuterBufKind::Dmabuf user data (released on niri.release)
///  - the recording task (released after readback)
///
/// When the last `Arc` is dropped the inner-client buffer is released, so
/// KWin only reuses the GPU memory after both consumers are done.
pub struct InnerBufferHold {
    buf: WlBuffer,
    state: Arc<crate::state::State>,
}

impl InnerBufferHold {
    pub fn new(buf: WlBuffer, state: Arc<crate::state::State>) -> Arc<Self> {
        // Register the hold so `release_inner_buffer` defers the real
        // wl_buffer.release until this hold (and any others) drop — otherwise
        // the headless commit handler releases the buffer right after the tap,
        // before the GPU encoder has imported it (half-composited frame).
        state.register_buffer_hold(&buf);
        Arc::new(Self { buf, state })
    }
}

impl Drop for InnerBufferHold {
    fn drop(&mut self) {
        // Decrement the hold count; when the last hold drops AND a release was
        // requested while held, this fires the deferred wl_buffer.release +
        // any pending wp_linux_drm_syncobj release-point signal (KWin
        // deadlocks waiting on un-signaled release timelines).
        self.state.drop_buffer_hold(&self.buf);
        self.state.poke_compositor_wake();
    }
}

/// One recording task, dispatched to the recording thread.
pub enum RecordingTask {
    /// Pre-read packed BGRA pixels (used by the SHM software-composite path).
    Pixels {
        pixels: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Dmabuf reference; the recording thread does the readback. The
    /// `_holds` vector keeps inner buffers pinned until the readback
    /// completes (dropping releases them via `InnerBufferHold::Drop`).
    Dmabuf {
        dma: Arc<DmabufBufferData>,
        _holds: Vec<Arc<InnerBufferHold>>,
    },
    /// Pre-encoded H.264 NAL bytes. Used by the Vulkan zero-copy path
    /// where the compositor thread runs `VkRecorder::encode_idr_from_dmabuf`
    /// inline and hands the recording thread a ready-to-mux byte
    /// stream. The recording thread just writes these to MkvWriter.
    /// Width/height carried so the recording thread can construct
    /// MkvWriter from the first Nal task — there's no Pixels fallback
    /// on the H264Vulkan path.
    Nal {
        data: Vec<u8>,
        pts_us: i64,
        is_keyframe: bool,
        width: u32,
        height: u32,
        /// AVCDecoderConfigurationRecord for MkvWriter's codec_private
        /// field. Same value on every frame (cheap to clone — ~25 bytes).
        codec_private: Vec<u8>,
    },
}

/// Latest-only task slot shared between outer_view (producer) and the
/// recording thread (consumer).
///
/// Replaces a bounded mpsc channel: at most ONE pending task waits in the
/// slot, plus the one currently being processed by the recording thread.
/// This caps the number of pinned inner buffers at any given time at 2,
/// avoiding the case where a longer queue extends buffer-hold lifetimes
/// beyond the readback duration and exhausts KWin's GBM pool.
///
/// When a new task arrives while the slot is full, the OLD task is dropped
/// (releasing its `InnerBufferHold`s back to KWin immediately) and replaced
/// with the new one — recording always reflects the latest available frame
/// rather than falling further behind live.
pub struct LatestTaskSlot {
    inner: Mutex<Option<RecordingTask>>,
    cvar: Condvar,
    /// Counts frames produced by outer_view (regardless of whether they
    /// got captured or dropped). Useful for diagnostics.
    pub produced: AtomicU64,
    /// Counts frames evicted from the slot by a newer one (i.e. dropped
    /// because the recording thread couldn't keep up). Diagnostic only.
    pub dropped: AtomicU64,
}

impl LatestTaskSlot {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            cvar: Condvar::new(),
            produced: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Replace the slot's contents with `task`. If the slot already had a
    /// task, that old task is dropped here — releasing its buffer holds so
    /// KWin can recycle the GBM immediately rather than waiting for the
    /// readback that would have used it.
    pub fn put(&self, task: RecordingTask) {
        self.produced.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.inner.lock().unwrap();
        if guard.is_some() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        *guard = Some(task);
        self.cvar.notify_one();
    }

    /// Take the current task, blocking up to `timeout` for one to arrive.
    /// Returns None on timeout. Spurious wakeups are tolerated.
    pub fn take_blocking(&self, timeout: std::time::Duration) -> Option<RecordingTask> {
        let mut guard = self.inner.lock().unwrap();
        if let Some(t) = guard.take() {
            return Some(t);
        }
        let (mut guard, _wait_result) = self.cvar.wait_timeout(guard, timeout).unwrap();
        guard.take()
    }

    /// Wake the consumer so it can re-check `stop` even if no task arrives.
    pub fn wake(&self) {
        self.cvar.notify_one();
    }
}

/// What the recording captures.
///
/// `FocusedWindow` (default): the recording follows whichever window is
/// currently focused, capturing only its surface tree. Right shape for
/// "record what my agent did" workflows where the agent interacts with
/// a single browser window.
///
/// `WholeDesktop`: capture the full inner-compositor surface set —
/// every mapped toplevel + layer surface — composited together into one
/// frame per commit. Right shape for "show me everything that happened
/// on this session" recordings (multi-window flows, Plasma demos).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum CaptureMode {
    #[default]
    FocusedWindow,
    WholeDesktop,
}

/// Holds the primary recording thread's handle plus an optional secondary
/// encoder running in parallel from the same compositor frame stream.
///
/// The compositor's commit tap fans the same captured frame out to both
/// slots when `secondary` is `Some`. The two encoders drain independently
/// — the slow encoder (e.g. HEVC 4:4:4 lossless) cannot back-pressure the
/// fast encoder (e.g. H.264 low-latency for live streaming). On
/// `record stop` both threads observe their own stop flag and finalize.
///
/// See `crates/waymux-session/src/control.rs::RecordStart` for the
/// spawn-time wiring.
pub struct DualRecordingHandle {
    pub primary: RecordingHandle,
    pub secondary: Option<RecordingHandle>,
}

impl DualRecordingHandle {
    /// True iff the primary recording is still accepting frames. The
    /// secondary is allowed to be stopped independently in future
    /// (e.g. live-stream ends but archive continues) — but in Phase 1
    /// they are stopped together so the primary's flag is authoritative.
    pub fn is_active(&self) -> bool {
        !self.primary.stop.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Signal both encoders to stop and wake their slot condvars so the
    /// threads observe the flag immediately instead of waiting up to the
    /// 200 ms heartbeat.
    pub fn stop_both(&self) {
        use std::sync::atomic::Ordering;
        self.primary.stop.store(true, Ordering::Release);
        self.primary.slot.wake();
        if let Some(sec) = &self.secondary {
            sec.stop.store(true, Ordering::Release);
            sec.slot.wake();
        }
    }

    /// Test-only constructor. Primary slot is a minimal stub; secondary
    /// is None. Used by tap-counting unit tests that only need
    /// `encoder_count_for_tap()` to count, not to dereference inner fields.
    #[cfg(test)]
    pub fn test_stub_primary_only() -> Self {
        Self {
            primary: RecordingHandle::test_stub(),
            secondary: None,
        }
    }
}

/// Active-recording handle. Held by `State::recording` for the lifetime
/// of one recording. The compositor's commit handler reads `mode` and
/// pushes captured frames into `slot`; the ffmpeg-feeder thread drains
/// `slot` and writes the encoded MKV.
pub struct RecordingHandle {
    pub stop: Arc<AtomicBool>,
    /// Latest-only slot. outer_view puts tasks here; the recording thread
    /// takes them. Replaces an mpsc channel so older queued tasks don't
    /// pile up holding buffers.
    pub slot: Arc<LatestTaskSlot>,
    /// Capture mode set at start time; immutable for the recording's
    /// lifetime. The compositor commit handler reads this on each
    /// commit to decide focused-window-only vs full-desktop composite.
    /// Public so the compositor module's tap can read it without
    /// adding another accessor.
    pub mode: CaptureMode,
    /// Codec set at start time. The compositor commit handler reads this
    /// to decide whether to push BGRA (ffv1) or NV12 (h264 codecs) into
    /// the slot. Pushing NV12 from the compositor — instead of the
    /// recording thread doing BGRA→NV12 later — saves one full-frame
    /// memcpy of BGRA per frame. At 4K that's 33 MB/frame ≈ 3 ms saved
    /// per frame on PCIe-mapped dmabuf reads.
    pub codec: waymux_protocol::RecordingCodec,
    /// Absolute output path resolved at start time. Kept so `RecordStatus`
    /// can report where the recording is being written without re-deriving
    /// it. The recording thread owns its own copy for the actual file I/O;
    /// this is a cheap clone for status reporting.
    pub path: String,
}

impl RecordingHandle {
    /// Test-only stub. Bare struct with default-ish values; the tap-counting
    /// tests only need the struct to exist — they don't dereference the slot
    /// or the stop flag in ways that would require real encoder state.
    #[cfg(test)]
    pub fn test_stub() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            slot: Arc::new(LatestTaskSlot::new()),
            mode: CaptureMode::FocusedWindow,
            codec: waymux_protocol::RecordingCodec::H264Nvenc,
            path: String::new(),
        }
    }
}

/// Check that `ffmpeg` is available in PATH without starting a recording.
/// Call before acquiring the recording lock so blocking subprocess I/O
/// doesn't hold the mutex.
pub fn probe_ffmpeg() -> Result<(), String> {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|_| "ffmpeg not found in PATH — install ffmpeg to use recording".to_string())?;
    Ok(())
}

/// Check that the requested codec is actually available in this ffmpeg
/// build. `ffmpeg -h encoder=<name>` exits 0 if known, 1 otherwise — far
/// cheaper than spawning a real encode and parsing the failure that
/// would otherwise surface only after the first frame arrives.
///
/// For VAAPI, additionally checks for `/dev/dri/renderD128` so the
/// "no DRM render node" failure surfaces synchronously at start time,
/// not asynchronously inside ffmpeg.
pub fn probe_codec(codec: waymux_protocol::RecordingCodec) -> Result<(), String> {
    use waymux_protocol::RecordingCodec;
    // H264Vulkan doesn't use ffmpeg at all — the probe is a Vulkan
    // device open instead, deferred until the recording thread
    // actually constructs the VkRecorder. If the host lacks
    // VK_KHR_video_encode_h264, VkRecorder::try_new returns None and
    // we surface the failure then.
    if matches!(
        codec,
        RecordingCodec::H264Vulkan
            | RecordingCodec::H264VulkanLossless
            | RecordingCodec::HevcVulkanLossless
    ) {
        return Ok(());
    }
    let encoder = encoder_name(codec);
    let status = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-h", &format!("encoder={encoder}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("probe ffmpeg encoder={encoder}: {e}"))?;
    if !status.success() {
        return Err(format!(
            "encoder {encoder:?} not available in this ffmpeg build — \
             install an ffmpeg with the matching codec, or pick a different \
             --codec value"
        ));
    }
    if matches!(codec, RecordingCodec::H264Vaapi)
        && !std::path::Path::new("/dev/dri/renderD128").exists()
    {
        return Err(
            "h264-vaapi requires /dev/dri/renderD128 — host has no DRM render \
             node available; either grant the daemon access to it or pick a \
             different --codec"
                .to_string(),
        );
    }
    Ok(())
}

/// ffmpeg's encoder name for a given codec choice. Centralised so the
/// probe path and the spawn path can't disagree.
fn encoder_name(codec: waymux_protocol::RecordingCodec) -> &'static str {
    use waymux_protocol::RecordingCodec;
    match codec {
        RecordingCodec::Ffv1 => "ffv1",
        RecordingCodec::H264Nvenc => "h264_nvenc",
        RecordingCodec::H264Vaapi => "h264_vaapi",
        // Vulkan paths don't go through ffmpeg; placeholders for any
        // internal use that just wants a stable codec label.
        RecordingCodec::H264Vulkan => "h264-vulkan",
        RecordingCodec::Ffv1Vulkan => "ffv1-vulkan",
        RecordingCodec::H264VulkanLossless => "h264-vulkan-lossless",
        RecordingCodec::HevcVulkanLossless => "hevc-vulkan-lossless",
        // CudaNvenc uses the direct CUDA driver API, not an ffmpeg subprocess;
        // return the same label as H264Nvenc for any internal codec-name uses.
        RecordingCodec::CudaNvenc => "h264_nvenc",
    }
}

/// Resolve the home directory: try `$HOME` first, then fall back to
/// `getpwuid(getuid())->pw_dir`. The daemon spawns session processes
/// without `HOME` set, so the env var alone is insufficient.
fn home_dir() -> String {
    if let Ok(h) = std::env::var("HOME") {
        if !h.is_empty() {
            return h;
        }
    }
    let uid = unsafe { libc::getuid() };
    let pw = unsafe { libc::getpwuid(uid) };
    if !pw.is_null() {
        let cstr = unsafe { std::ffi::CStr::from_ptr((*pw).pw_dir) };
        if let Ok(s) = cstr.to_str() {
            return s.to_string();
        }
    }
    "/tmp".into()
}

/// Return the base recordings directory (`$HOME/.local/share/waymux/recordings`).
/// Creates the directory. Used by `default_recording_path` and by the H2
/// path-validation gate in `control::RecordStart`.
pub fn recordings_dir() -> Result<std::path::PathBuf, String> {
    let home = home_dir();
    let dir = std::path::Path::new(&home).join(".local/share/waymux/recordings");
    std::fs::create_dir_all(&dir).map_err(|e| format!("create recordings dir: {e}"))?;
    Ok(dir)
}

/// Resolve the default output path for a session. Creates parent dirs.
pub fn default_recording_path(session_name: &str) -> Result<String, String> {
    let dir = recordings_dir()?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = dir.join(format!("{session_name}-{ts}.mkv"));
    path.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "non-UTF8 path".into())
}

/// Spawn the ffmpeg-feeder thread and return a handle.
///
/// Caller must have already called `probe_ffmpeg()` outside any locks.
/// The returned handle owns the stop flag and slot Arc; setting `stop`
/// (e.g. on RecordStop) signals the thread to exit cleanly.
///
/// **Capture is commit-driven, not polling-based** (Fix C, 2026-05-09).
/// The previous architecture spawned a second "headless producer"
/// thread that polled `state.capture_desktop()` at 30 fps. That
/// approach raced chromium's swap chain — see
/// `docs/recording-architecture-analysis-2026-05-09.md` and the
/// commit history of this file for the postmortem. The current
/// architecture has only ONE worker (this ffmpeg feeder); frames are
/// pushed into its slot synchronously by the inner-compositor's
/// commit handler via `State::maybe_tap_for_recording`. Capture rate
/// matches the inner client's commit cadence (~60 fps for animated
/// content; 0 fps for idle pages).
///
/// outer_view *also* feeds the slot when an attach client is
/// connected (`State::try_push_recording_*`). The slot is latest-only,
/// so concurrent producers don't double-encode — newer frames evict
/// older ones.
pub fn spawn_recording_thread(
    state: Arc<crate::state::State>,
    path: String,
    codec: waymux_protocol::RecordingCodec,
    mode: CaptureMode,
    min_fps: Option<u32>,
) -> Result<RecordingHandle, String> {
    // Fail fast on a codec the GPU can't actually encode, rather than spawning a
    // recording thread that dies after the first frame (logging only a warning)
    // and writes no file — leaving `record status` reporting `recording=true`
    // with nothing on disk. hevc-vulkan-lossless needs Vulkan-video HEVC 4:4:4
    // (Hi444) caps, which integrated AMD parts (e.g. RENOIR) do not expose.
    if matches!(codec, waymux_protocol::RecordingCodec::HevcVulkanLossless) {
        match crate::vulkan_record::VkDeviceCtx::open() {
            Ok(ctx) if !ctx.hi444_supported => {
                return Err(format!(
                    "hevc-vulkan-lossless requires Vulkan-video HEVC 4:4:4 (Hi444) caps, \
                     not exposed by this device ({}); use ffv1-vulkan or h264-vulkan-lossless \
                     for lossless capture on this GPU",
                    ctx.device_name
                ));
            }
            Ok(_) => {}
            Err(e) => {
                return Err(format!(
                    "hevc-vulkan-lossless: cannot open a Vulkan device to verify caps: {e}"
                ));
            }
        }
    }
    let _ = state; // reserved for future tap parameters; not used today.
    let slot = Arc::new(LatestTaskSlot::new());
    let stop = Arc::new(AtomicBool::new(false));

    let handle = RecordingHandle {
        stop: stop.clone(),
        slot: slot.clone(),
        mode,
        codec,
        path: path.clone(),
    };

    {
        let slot = slot.clone();
        let stop = stop.clone();
        let codec_for_thread = codec;
        std::thread::Builder::new()
            .name("waymux-recording".into())
            .spawn(move || match codec_for_thread {
                waymux_protocol::RecordingCodec::H264Vulkan => {
                    vulkan_recording_thread(slot, stop, path, min_fps);
                }
                waymux_protocol::RecordingCodec::Ffv1Vulkan => {
                    ffv1_vulkan_recording_thread(slot, stop, path, min_fps);
                }
                waymux_protocol::RecordingCodec::H264VulkanLossless => {
                    vulkan_lossless_recording_thread(slot, stop, path, min_fps);
                }
                waymux_protocol::RecordingCodec::HevcVulkanLossless => {
                    hevc_vulkan_recording_thread(slot, stop, path, min_fps);
                }
                _ => {
                    recording_thread(slot, stop, path, codec_for_thread, min_fps);
                }
            })
            .map_err(|e| format!("spawn recording thread: {e}"))?;
    }

    Ok(handle)
}

/// Recording thread for the Vulkan zero-copy path.
///
/// The compositor thread does NOT touch Vulkan — it dups the dmabuf fd
/// and pushes a `RecordingTask::Dmabuf` (with InnerBufferHold pins)
/// into the slot. This thread then:
///   1. Lazily inits a `VkRecorder` sized to the first usable commit
///   2. Reads tasks from the slot and dispatches:
///      - `Dmabuf` → `VkRecorder::encode_idr_from_dmabuf` (zero-copy)
///      - `Pixels` → `VkRecorder::encode_idr_from_bgra` (CPU fallback,
///        e.g. SHM-only clients)
///      - `Nal`   → write directly (legacy; left for backward compat
///        but no longer produced by the compositor)
///   3. Writes encoded NAL bytes to MkvWriter
///
/// All the heavy work (compute dispatch + encode submit + fence wait
/// + bitstream readback) happens here, off the compositor thread.
///   Inner clients get their frame_callbacks back immediately after the
///   commit ack; the attach view doesn't choke when recording is active.
fn vulkan_recording_thread(
    slot: Arc<LatestTaskSlot>,
    stop: Arc<AtomicBool>,
    path: String,
    min_fps: Option<u32>,
) {
    // Wait for the first frame to discover dimensions. The Vulkan
    // path accepts both pre-encoded NAL (zero-copy from
    // VkRecorder::encode_idr_from_dmabuf inline on the compositor
    // thread) and BGRA bytes (CPU fallback). For dimensions we need
    // BGRA; if the first task is a NAL we can't open the muxer yet
    // because codec_private comes from VkRecorder which needs the
    // resolution to construct.
    //
    // First task may be either:
    //   Pixels — we run encode_idr_from_bgra later
    //   Nal — we need an out-of-band dimensions discovery
    //   Dmabuf — we should never see one here in the new path,
    //            but read_task will CPU-readback it as BGRA
    let mut first_pixels: Option<Vec<u8>> = None;
    let mut first_nal: Option<(Vec<u8>, i64, bool)> = None;
    let mut first_dmabuf: Option<(
        Arc<crate::dmabuf::DmabufBufferData>,
        Vec<Arc<InnerBufferHold>>,
    )> = None;
    let mut codec_private_from_nal: Vec<u8> = Vec::new();
    let (w, h) = {
        info!(
            "vulkan recording: waiting up to {}s for first frame from session",
            FIRST_FRAME_TIMEOUT.as_secs()
        );
        let deadline = std::time::Instant::now() + FIRST_FRAME_TIMEOUT;
        loop {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                warn!(
                    "vulkan recording: no frames received within {}s; aborting",
                    FIRST_FRAME_TIMEOUT.as_secs()
                );
                return;
            }
            let wait = remaining.min(std::time::Duration::from_millis(200));
            let Some(task) = slot.take_blocking(wait) else {
                continue;
            };
            // KDE/Plasma sessions often commit a 1x1 SinglePixel
            // placeholder before plasmashell's real surfaces appear.
            // Treat anything below MIN_DIM as bookkeeping and keep
            // waiting for a real frame. 32 is conservative — H.264's
            // min coded extent is 128, our encoder accepts anything
            // even and >= 16.
            const MIN_DIM: u32 = 32;
            let too_small = |w: u32, h: u32| w < MIN_DIM || h < MIN_DIM;
            match task {
                RecordingTask::Pixels {
                    pixels,
                    width,
                    height,
                } => {
                    if too_small(width, height) {
                        continue;
                    }
                    first_pixels = Some(pixels);
                    break (width, height);
                }
                RecordingTask::Nal {
                    data,
                    pts_us,
                    is_keyframe,
                    width,
                    height,
                    codec_private,
                } => {
                    if too_small(width, height) {
                        continue;
                    }
                    // Zero-copy path: the compositor pre-encoded this
                    // frame. The Nal carries dimensions + codec_private
                    // so we can construct the muxer directly without
                    // needing a fallback VkRecorder.
                    first_nal = Some((data, pts_us, is_keyframe));
                    codec_private_from_nal = codec_private;
                    break (width, height);
                }
                RecordingTask::Dmabuf { dma, _holds } => {
                    let w = dma.width as u32;
                    let h = dma.height as u32;
                    if too_small(w, h) {
                        continue;
                    }
                    // Zero-copy path: stash the dmabuf + holds so the
                    // post-discovery code can encode it via
                    // VkRecorder::encode_idr_from_dmabuf. No CPU
                    // readback — that defeats the whole point of the
                    // Vulkan path.
                    first_dmabuf = Some((dma, _holds));
                    break (w, h);
                }
            }
        }
    };
    info!(path = %path, width = w, height = h, ?min_fps, "vulkan recording starting");

    // The recording thread owns a VkRecorder for both zero-copy
    // (Dmabuf tasks) and CPU fallback (Pixels tasks). Pre-encoded Nal
    // tasks bypass the recorder entirely (legacy path; the live
    // compositor no longer produces them, but we still write them
    // out if they appear).
    let recorder: Option<crate::vulkan_record::VkRecorder> =
        if first_dmabuf.is_some() || first_pixels.is_some() {
            match crate::vulkan_record::VkRecorder::try_new(w, h) {
                Some(r) => Some(r),
                None => {
                    warn!("vulkan recording: VkRecorder::try_new failed; aborting");
                    return;
                }
            }
        } else {
            None
        };
    let codec_private: Vec<u8> = if !codec_private_from_nal.is_empty() {
        codec_private_from_nal
    } else if let Some(ref r) = recorder {
        r.codec_private().to_vec()
    } else {
        warn!("vulkan recording: no codec_private available; aborting");
        return;
    };

    let file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            warn!(error = %e, path = %path, "vulkan recording: open output failed");
            return;
        }
    };
    let mut writer =
        match waymux_mux_mkv::MkvWriter::new(std::io::BufWriter::new(file), w, h, &codec_private) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "vulkan recording: MkvWriter::new failed");
                return;
            }
        };

    let start = std::time::Instant::now();
    let mut frames: u64 = 0;
    let mut first_pts_us: Option<i64> = None;
    // When min_fps is set, cache the last encoded NAL so we can write
    // it again with a fresher pts_ms whenever no new commit arrived in
    // the last tick window — keeps the output at a uniform cadence even
    // when the inner client (chromium during YouTube playback, etc.)
    // only commits at ~46 fps.
    let mut last_nal_bytes: Option<(Vec<u8>, bool)> = None;
    let min_frame_interval = min_fps
        .filter(|n| *n > 0)
        .map(|n| std::time::Duration::from_micros(1_000_000 / n as u64));
    // Tick-driven cadence when min_fps is set: wake exactly every
    // 1/min_fps seconds, take whatever's in the slot non-blocking,
    // encode that or duplicate the last NAL. This caps the upper rate
    // at min_fps as well as guaranteeing the floor — keeps the MKV's
    // PTS cadence uniform instead of letting commit bursts spike above
    // min_fps and then sag below it.
    let mut next_tick = std::time::Instant::now();

    // Write the first frame we already pulled out.
    if let Some((data, pts_us, is_keyframe)) = first_nal {
        first_pts_us = Some(pts_us);
        let _ = writer.write_frame(&data, 0, is_keyframe);
        if min_frame_interval.is_some() {
            last_nal_bytes = Some((data, is_keyframe));
        }
        frames += 1;
    } else if let (Some((dma, _holds)), Some(r)) = (first_dmabuf.take(), recorder.as_ref()) {
        let pts_us = start.elapsed().as_micros() as i64;
        if let Some(nal) = r.encode_idr_from_dmabuf(&dma, pts_us) {
            let _ = writer.write_frame(&nal.data, 0, nal.is_keyframe);
            if min_frame_interval.is_some() {
                last_nal_bytes = Some((nal.data, nal.is_keyframe));
            }
            frames += 1;
        }
        // _holds drops here, releasing the inner buffer.
    } else if let (Some(first), Some(r)) = (first_pixels, recorder.as_ref()) {
        if let Some(nal) = r.encode_idr_from_bgra(&first, 0) {
            let _ = writer.write_frame(&nal.data, 0, nal.is_keyframe);
            if min_frame_interval.is_some() {
                last_nal_bytes = Some((nal.data, nal.is_keyframe));
            }
            frames += 1;
        }
    }

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        // In min-fps mode: sleep until the next tick boundary, then take
        // non-blocking. In commit-driven mode: block up to 200 ms for the
        // next commit. The two modes diverge on how we pace; everything
        // after the take is shared.
        let take_result = if let Some(interval) = min_frame_interval {
            let now = std::time::Instant::now();
            if next_tick > now {
                std::thread::sleep(next_tick - now);
            }
            next_tick = std::time::Instant::now() + interval;
            slot.take_blocking(std::time::Duration::ZERO)
        } else {
            slot.take_blocking(std::time::Duration::from_millis(200))
        };
        if stop.load(Ordering::Acquire) {
            break;
        }
        let task = match take_result {
            Some(t) => t,
            None => {
                if min_frame_interval.is_none() {
                    continue; // commit-driven mode — idle is fine
                }
                // Min-fps mode: rewrite the last NAL with a fresh pts so
                // the muxer's cadence stays uniform at the requested rate.
                if let Some((ref bytes, is_kf)) = last_nal_bytes {
                    let pts_ms = start.elapsed().as_millis() as i64;
                    if let Err(e) = writer.write_frame(bytes, pts_ms, is_kf) {
                        warn!(error = %e, "vulkan recording: write_frame(duplicate) failed");
                        break;
                    }
                    frames += 1;
                }
                continue;
            }
        };
        match task {
            RecordingTask::Nal {
                data,
                pts_us,
                is_keyframe,
                ..
            } => {
                let first_pts = *first_pts_us.get_or_insert(pts_us);
                let pts_ms = (pts_us - first_pts) / 1000;
                if let Err(e) = writer.write_frame(&data, pts_ms, is_keyframe) {
                    warn!(error = %e, "vulkan recording: write_frame(Nal) failed");
                    break;
                }
                if min_frame_interval.is_some() {
                    last_nal_bytes = Some((data, is_keyframe));
                }
                frames += 1;
            }
            RecordingTask::Dmabuf { dma, _holds } => {
                // Zero-copy: dmabuf in, NAL out, no CPU readback.
                let fw = dma.width as u32;
                let fh = dma.height as u32;
                if fw != w || fh != h {
                    warn!("vulkan recording: frame size {fw}x{fh} != recorder {w}x{h}; skipping");
                    continue;
                }
                let Some(ref r) = recorder else { continue };
                let pts_ms = start.elapsed().as_millis() as i64;
                let Some(nal) = r.encode_idr_from_dmabuf(&dma, pts_ms * 1000) else {
                    warn!("vulkan recording: encode_idr_from_dmabuf returned None");
                    continue;
                };
                if let Err(e) = writer.write_frame(&nal.data, pts_ms, nal.is_keyframe) {
                    warn!(error = %e, "vulkan recording: write_frame(Dmabuf) failed");
                    break;
                }
                if min_frame_interval.is_some() {
                    last_nal_bytes = Some((nal.data, nal.is_keyframe));
                }
                frames += 1;
                // _holds drops here, releasing the inner buffer back
                // to the inner client.
            }
            RecordingTask::Pixels {
                pixels,
                width: fw,
                height: fh,
            } => {
                if fw != w || fh != h {
                    warn!("vulkan recording: frame size {fw}x{fh} != recorder {w}x{h}; skipping");
                    continue;
                }
                let Some(ref r) = recorder else { continue };
                let pts_ms = start.elapsed().as_millis() as i64;
                let Some(nal) = r.encode_idr_from_bgra(&pixels, pts_ms * 1000) else {
                    warn!("vulkan recording: encode_idr_from_bgra returned None");
                    continue;
                };
                if let Err(e) = writer.write_frame(&nal.data, pts_ms, nal.is_keyframe) {
                    warn!(error = %e, "vulkan recording: write_frame(Pixels) failed");
                    break;
                }
                if min_frame_interval.is_some() {
                    last_nal_bytes = Some((nal.data, nal.is_keyframe));
                }
                frames += 1;
            }
        }
    }

    if let Err(e) = writer.finish() {
        warn!(error = %e, "vulkan recording: finish failed");
    }
    info!(path = %path, frames = frames, "vulkan recording finished");
}

/// Recording thread for the GPU zero-copy lossless ffv1_vulkan path.
///
/// Mirrors `vulkan_recording_thread`'s structure but routes Dmabuf
/// tasks through a different encode chain:
///
///   dmabuf fd
///     │ import_dmabuf_as_transfer_src
///     ▼
///   waymux VkImage  ──vkCmdCopyImage──>  libav AVVkFrame  ──>  ffv1_vulkan
///                                                                  │
///                                                                  ▼
///                                                          MkvWriter (V_FFV1)
///
/// Pixels never leave the GPU between dmabuf import and the encoder.
///
/// SHM/`Pixels` clients (e.g. `foot`, which never produces a dmabuf) are
/// ALSO supported: the BGRA bytes are uploaded into a LINEAR host-visible
/// `TRANSFER_SRC` VkImage (`upload_bgra_to_transfer_src`) and copied into the
/// libav-pool AVVkFrame exactly like the dmabuf path. The only difference is
/// the source image's origin (CPU upload vs zero-copy dmabuf import); both
/// flow through the same `encode_one` + `vkCmdCopyImage`. Pre-encoded `Nal`
/// tasks remain unsupported on this path (no compositor-side pre-encode).
fn ffv1_vulkan_recording_thread(
    slot: Arc<LatestTaskSlot>,
    stop: Arc<AtomicBool>,
    path: String,
    min_fps: Option<u32>,
) {
    use crate::ffv1_vk_record::{open_ffv1_mkv, Ffv1VkEncoder, FrameInputView};
    use crate::vulkan_record::{
        import_dmabuf_as_transfer_src, upload_bgra_to_transfer_src, ImportedDmabufImage,
        VkDeviceCtx,
    };
    use ash::vk;

    /// One frame's source, normalized so the encode path is identical
    /// regardless of whether it arrived as a client dmabuf or CPU BGRA bytes.
    enum FrameSource {
        Dmabuf {
            dma: Arc<crate::dmabuf::DmabufBufferData>,
            holds: Vec<Arc<InnerBufferHold>>,
        },
        Pixels {
            bgra: Vec<u8>,
        },
    }

    // Wait for the first frame (dmabuf OR pixels) to discover dimensions.
    let mut first_frame: Option<FrameSource> = None;
    let (w, h) = {
        info!(
            "ffv1 recording: waiting up to {}s for first frame from session",
            FIRST_FRAME_TIMEOUT.as_secs()
        );
        let deadline = std::time::Instant::now() + FIRST_FRAME_TIMEOUT;
        loop {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                warn!(
                    "ffv1 recording: no frames received within {}s; aborting",
                    FIRST_FRAME_TIMEOUT.as_secs()
                );
                return;
            }
            let wait = remaining.min(std::time::Duration::from_millis(200));
            let Some(task) = slot.take_blocking(wait) else {
                continue;
            };
            const MIN_DIM: u32 = 32;
            let too_small = |w: u32, h: u32| w < MIN_DIM || h < MIN_DIM;
            match task {
                RecordingTask::Dmabuf { dma, _holds } => {
                    let dw = dma.width as u32;
                    let dh = dma.height as u32;
                    if too_small(dw, dh) {
                        continue;
                    }
                    first_frame = Some(FrameSource::Dmabuf { dma, holds: _holds });
                    break (dw, dh);
                }
                RecordingTask::Pixels {
                    pixels,
                    width,
                    height,
                } => {
                    if too_small(width, height) {
                        continue;
                    }
                    first_frame = Some(FrameSource::Pixels { bgra: pixels });
                    break (width, height);
                }
                RecordingTask::Nal { .. } => {
                    warn!("ffv1 recording: pre-encoded NAL task unexpected; aborting");
                    return;
                }
            }
        }
    };
    info!(path = %path, width = w, height = h, "ffv1 recording starting");

    let ctx = match VkDeviceCtx::open() {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "ffv1 recording: VkDeviceCtx::open failed; aborting");
            return;
        }
    };
    let mut enc = match Ffv1VkEncoder::open(&ctx, w, h) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "ffv1 recording: Ffv1VkEncoder::open failed; aborting");
            return;
        }
    };
    let file = match std::fs::File::create(&path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            warn!(error = %e, path = %path, "ffv1 recording: open output failed");
            return;
        }
    };
    let mut mkv = match open_ffv1_mkv(&enc, file) {
        Ok(m) => m,
        Err(e) => {
            warn!(error = %e, "ffv1 recording: open_ffv1_mkv failed");
            return;
        }
    };

    let start = std::time::Instant::now();
    let mut frames: u64 = 0;
    // Last frame's BGRA bytes, cached for min-fps duplication. We keep the
    // last frame as packed BGRA regardless of its origin: for a dmabuf we
    // can't re-read the client buffer after its hold drops, so the min-fps
    // duplicate path re-uploads these cached bytes via the same Pixels path.
    let mut last_bgra: Option<Vec<u8>> = None;
    let min_frame_interval = min_fps
        .filter(|n| *n > 0)
        .map(|n| std::time::Duration::from_micros(1_000_000 / n as u64));
    let mut next_tick = std::time::Instant::now();

    // Encode one already-imported source image (dmabuf import OR BGRA upload)
    // to FFV1 packets and write them to the MKV. `initial_layout` is the
    // image's current layout: UNDEFINED for a freshly-imported dmabuf,
    // PREINITIALIZED for a host-uploaded BGRA image.
    let encode_imported = |enc: &mut Ffv1VkEncoder,
                           mkv: &mut waymux_mux_mkv::MkvWriter<_>,
                           imported: ImportedDmabufImage,
                           initial_layout: vk::ImageLayout,
                           pts_us: i64,
                           pts_ms: i64|
     -> Result<usize, String> {
        if imported.width != w || imported.height != h {
            let msg = format!(
                "frame size {}x{} != recorder {w}x{h}; skipping",
                imported.width, imported.height
            );
            imported.destroy(&ctx);
            return Err(msg);
        }
        let view = FrameInputView {
            image: imported.image,
            current_layout: initial_layout,
            current_access: vk::AccessFlags2::empty(),
            current_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            current_queue_family: ctx.compute_queue_family,
        };
        let packets = match enc.encode_one(&ctx, &view, pts_us) {
            Ok(p) => p,
            Err(e) => {
                // Wait for any in-flight work referencing the image
                // before destroying it.
                unsafe { ctx.device.device_wait_idle().ok() };
                imported.destroy(&ctx);
                return Err(e);
            }
        };
        // The encoder's command buffer references our `imported.image`
        // until the device finishes the copy. Wait before destroy.
        unsafe { ctx.device.device_wait_idle().ok() };
        let n = packets.len();
        for p in packets {
            mkv.write_frame(&p.data, pts_ms, p.is_keyframe)
                .map_err(|e| format!("mkv write_frame: {e:?}"))?;
        }
        imported.destroy(&ctx);
        Ok(n)
    };

    // Import/upload a FrameSource into a TRANSFER_SRC image, encode it, and
    // (when caching for min-fps) capture the BGRA bytes for later duplication.
    // Returns the encoded-packet count.
    let encode_source = |enc: &mut Ffv1VkEncoder,
                         mkv: &mut waymux_mux_mkv::MkvWriter<_>,
                         src: FrameSource,
                         pts_us: i64,
                         pts_ms: i64,
                         last_bgra: &mut Option<Vec<u8>>,
                         cache: bool|
     -> Result<usize, String> {
        match src {
            FrameSource::Dmabuf { dma, holds } => {
                let imported = import_dmabuf_as_transfer_src(&ctx, &dma)?;
                let n = encode_imported(
                    enc,
                    mkv,
                    imported,
                    vk::ImageLayout::UNDEFINED,
                    pts_us,
                    pts_ms,
                )?;
                // For min-fps duplication, read the dmabuf back to BGRA so a
                // later idle tick can re-upload it (the hold drops here).
                if cache {
                    if let Some((bgra, _, _)) = read_task(RecordingTask::Dmabuf {
                        dma: dma.clone(),
                        _holds: Vec::new(),
                    }) {
                        *last_bgra = Some(bgra);
                    }
                }
                drop(holds);
                Ok(n)
            }
            FrameSource::Pixels { bgra } => {
                let imported = upload_bgra_to_transfer_src(&ctx, &bgra, w, h)?;
                let n = encode_imported(
                    enc,
                    mkv,
                    imported,
                    vk::ImageLayout::PREINITIALIZED,
                    pts_us,
                    pts_ms,
                )?;
                if cache {
                    *last_bgra = Some(bgra);
                }
                Ok(n)
            }
        }
    };

    // Write the first frame.
    if let Some(src) = first_frame.take() {
        match encode_source(
            &mut enc,
            &mut mkv,
            src,
            0,
            0,
            &mut last_bgra,
            min_frame_interval.is_some(),
        ) {
            Ok(n) => frames += n as u64,
            Err(e) => warn!(error = %e, "ffv1 recording: first-frame encode failed"),
        }
    }

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let take_result = if let Some(interval) = min_frame_interval {
            let now = std::time::Instant::now();
            if next_tick > now {
                std::thread::sleep(next_tick - now);
            }
            next_tick = std::time::Instant::now() + interval;
            slot.take_blocking(std::time::Duration::ZERO)
        } else {
            slot.take_blocking(std::time::Duration::from_millis(200))
        };
        if stop.load(Ordering::Acquire) {
            break;
        }
        let task = match take_result {
            Some(t) => t,
            None => {
                if min_frame_interval.is_none() {
                    continue;
                }
                // Min-fps mode: re-encode the last frame (cached BGRA) with a
                // fresh pts by re-uploading it through the Pixels path.
                if let Some(bgra) = last_bgra.clone() {
                    let pts_ms = start.elapsed().as_millis() as i64;
                    let pts_us = pts_ms * 1000;
                    match encode_source(
                        &mut enc,
                        &mut mkv,
                        FrameSource::Pixels { bgra },
                        pts_us,
                        pts_ms,
                        &mut last_bgra,
                        false,
                    ) {
                        Ok(n) => frames += n as u64,
                        Err(e) => warn!(error = %e, "ffv1 recording: duplicate encode failed"),
                    }
                }
                continue;
            }
        };
        match task {
            RecordingTask::Dmabuf { dma, _holds } => {
                let pts_ms = start.elapsed().as_millis() as i64;
                let pts_us = pts_ms * 1000;
                match encode_source(
                    &mut enc,
                    &mut mkv,
                    FrameSource::Dmabuf { dma, holds: _holds },
                    pts_us,
                    pts_ms,
                    &mut last_bgra,
                    min_frame_interval.is_some(),
                ) {
                    Ok(n) => frames += n as u64,
                    Err(e) => warn!(error = %e, "ffv1 recording: encode_source(dmabuf) failed"),
                }
            }
            RecordingTask::Pixels {
                pixels,
                width: fw,
                height: fh,
            } => {
                if fw != w || fh != h {
                    warn!("ffv1 recording: frame size {fw}x{fh} != recorder {w}x{h}; skipping");
                    continue;
                }
                let pts_ms = start.elapsed().as_millis() as i64;
                let pts_us = pts_ms * 1000;
                match encode_source(
                    &mut enc,
                    &mut mkv,
                    FrameSource::Pixels { bgra: pixels },
                    pts_us,
                    pts_ms,
                    &mut last_bgra,
                    min_frame_interval.is_some(),
                ) {
                    Ok(n) => frames += n as u64,
                    Err(e) => warn!(error = %e, "ffv1 recording: encode_source(pixels) failed"),
                }
            }
            RecordingTask::Nal { .. } => {
                warn!("ffv1 recording: ignored NAL task mid-stream");
            }
        }
    }

    // Flush encoder + tail packets.
    match enc.flush() {
        Ok(tail) => {
            for p in tail {
                // Encoder's flush packets carry the encoder's own pts;
                // wall-clock fallback if they're junk.
                let pts_ms = if p.pts_us >= 0 {
                    p.pts_us / 1_000
                } else {
                    start.elapsed().as_millis() as i64
                };
                if let Err(e) = mkv.write_frame(&p.data, pts_ms, p.is_keyframe) {
                    warn!(error = %e, "ffv1 recording: flush write_frame failed");
                    break;
                }
                frames += 1;
            }
        }
        Err(e) => warn!(error = %e, "ffv1 recording: flush failed"),
    }
    if let Err(e) = mkv.finish() {
        warn!(error = %e, "ffv1 recording: finish failed");
    }
    info!(path = %path, frames = frames, "ffv1 recording finished");
}

/// Recording thread for the H.264 Hi444PP bit-exact lossless path.
///
/// Mirrors `vulkan_recording_thread` but uses `VkRecorderLossless`. The
/// dmabuf path goes through a CPU readback into BGRA first because the
/// lossless recorder doesn't (yet) expose a zero-copy dmabuf-to-444
/// entry. Pixels (SHM clients) and pre-encoded NAL tasks are unsupported
/// — the latter doesn't make sense for the lossless path (no
/// compositor-side pre-encode exists), the former is rare for the
/// kind of content this codec is meant for.
fn vulkan_lossless_recording_thread(
    slot: Arc<LatestTaskSlot>,
    stop: Arc<AtomicBool>,
    path: String,
    min_fps: Option<u32>,
) {
    use crate::vulkan_record::VkRecorderLossless;

    // Wait for the first frame to discover dimensions.
    let mut first_pixels: Option<Vec<u8>> = None;
    let mut first_dmabuf: Option<(
        Arc<crate::dmabuf::DmabufBufferData>,
        Vec<Arc<InnerBufferHold>>,
    )> = None;
    let (w, h) = {
        info!(
            "vulkan-lossless recording: waiting up to {}s for first frame from session",
            FIRST_FRAME_TIMEOUT.as_secs()
        );
        let deadline = std::time::Instant::now() + FIRST_FRAME_TIMEOUT;
        loop {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                warn!(
                    "vulkan-lossless recording: no frames within {}s; aborting",
                    FIRST_FRAME_TIMEOUT.as_secs()
                );
                return;
            }
            let wait = remaining.min(std::time::Duration::from_millis(200));
            let Some(task) = slot.take_blocking(wait) else {
                continue;
            };
            const MIN_DIM: u32 = 32;
            let too_small = |w: u32, h: u32| w < MIN_DIM || h < MIN_DIM;
            match task {
                RecordingTask::Pixels {
                    pixels,
                    width,
                    height,
                } => {
                    if too_small(width, height) {
                        continue;
                    }
                    first_pixels = Some(pixels);
                    break (width, height);
                }
                RecordingTask::Dmabuf { dma, _holds } => {
                    let dw = dma.width as u32;
                    let dh = dma.height as u32;
                    if too_small(dw, dh) {
                        continue;
                    }
                    first_dmabuf = Some((dma, _holds));
                    break (dw, dh);
                }
                RecordingTask::Nal { .. } => {
                    warn!(
                        "vulkan-lossless recording: pre-encoded NAL task unexpected on this codec; aborting"
                    );
                    return;
                }
            }
        }
    };
    info!(path = %path, width = w, height = h, "vulkan-lossless recording starting");

    let recorder = match VkRecorderLossless::try_new(w, h) {
        Some(r) => r,
        None => {
            warn!(
                "vulkan-lossless recording: VkRecorderLossless::try_new failed at {w}x{h}; aborting"
            );
            return;
        }
    };
    let codec_private = recorder.codec_private().to_vec();

    let file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            warn!(error = %e, path = %path, "vulkan-lossless recording: open output failed");
            return;
        }
    };
    let mut writer =
        match waymux_mux_mkv::MkvWriter::new(std::io::BufWriter::new(file), w, h, &codec_private) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "vulkan-lossless recording: MkvWriter::new failed");
                return;
            }
        };

    let start = std::time::Instant::now();
    let mut frames: u64 = 0;
    let mut last_nal_bytes: Option<(Vec<u8>, bool)> = None;
    let min_frame_interval = min_fps
        .filter(|n| *n > 0)
        .map(|n| std::time::Duration::from_micros(1_000_000 / n as u64));
    let mut next_tick = std::time::Instant::now();

    // Encode + write first frame.
    if let Some(first) = first_pixels.take() {
        if let Some(nal) = recorder.encode_idr_from_bgra(&first, 0) {
            let _ = writer.write_frame(&nal.data, 0, nal.is_keyframe);
            if min_frame_interval.is_some() {
                last_nal_bytes = Some((nal.data, nal.is_keyframe));
            }
            frames += 1;
        }
    } else if let Some((dma, holds)) = first_dmabuf.take() {
        if let Some((pixels, fw, fh)) = read_task(RecordingTask::Dmabuf {
            dma: dma.clone(),
            _holds: holds,
        }) {
            if fw == w && fh == h {
                if let Some(nal) = recorder.encode_idr_from_bgra(&pixels, 0) {
                    let _ = writer.write_frame(&nal.data, 0, nal.is_keyframe);
                    if min_frame_interval.is_some() {
                        last_nal_bytes = Some((nal.data, nal.is_keyframe));
                    }
                    frames += 1;
                }
            }
        }
    }

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let take_result = if let Some(interval) = min_frame_interval {
            let now = std::time::Instant::now();
            if next_tick > now {
                std::thread::sleep(next_tick - now);
            }
            next_tick = std::time::Instant::now() + interval;
            slot.take_blocking(std::time::Duration::ZERO)
        } else {
            slot.take_blocking(std::time::Duration::from_millis(200))
        };
        if stop.load(Ordering::Acquire) {
            break;
        }
        let task = match take_result {
            Some(t) => t,
            None => {
                if min_frame_interval.is_none() {
                    continue;
                }
                if let Some((ref bytes, is_kf)) = last_nal_bytes {
                    let pts_ms = start.elapsed().as_millis() as i64;
                    if let Err(e) = writer.write_frame(bytes, pts_ms, is_kf) {
                        warn!(error = %e, "vulkan-lossless recording: write_frame(duplicate) failed");
                        break;
                    }
                    frames += 1;
                }
                continue;
            }
        };
        // Convert any task type to BGRA for the lossless encoder.
        let Some((pixels, fw, fh)) = read_task(task) else {
            continue;
        };
        if fw != w || fh != h {
            warn!("vulkan-lossless recording: frame size {fw}x{fh} != recorder {w}x{h}; skipping");
            continue;
        }
        let pts_ms = start.elapsed().as_millis() as i64;
        let Some(nal) = recorder.encode_idr_from_bgra(&pixels, pts_ms * 1000) else {
            warn!("vulkan-lossless recording: encode_idr_from_bgra returned None");
            continue;
        };
        if let Err(e) = writer.write_frame(&nal.data, pts_ms, nal.is_keyframe) {
            warn!(error = %e, "vulkan-lossless recording: write_frame failed");
            break;
        }
        if min_frame_interval.is_some() {
            last_nal_bytes = Some((nal.data, nal.is_keyframe));
        }
        frames += 1;
    }

    if let Err(e) = writer.finish() {
        warn!(error = %e, "vulkan-lossless recording: finish failed");
    }
    info!(path = %path, frames = frames, "vulkan-lossless recording finished");
}

/// Recording thread for the HEVC RangeExt 4:4:4 lossless path.
///
/// Architecture:
///   1. Compositor commits BGRA → our compute shader writes 2-plane
///      YUV 4:4:4 (NV24 layout: Y in plane 0, UV interleaved in plane 1)
///      into a `FrameResources444.yuv_image`.
///   2. `HevcVkEncoder` copies that 2-plane image into a libav-pool
///      AVVkFrame (same NV24 format) via vkCmdCopyImage.
///   3. ffmpeg's `hevc_vulkan` encoder produces HEVC NAL packets.
///   4. Packets get muxed into MKV with `V_MPEGH/ISO/HEVC`.
///
/// Pixels never leave GPU memory between the compute shader output
/// and the encoder. The only CPU-side data is the encoded NAL bytes.
///
/// Verified working on NVIDIA RTX A6000 + driver 580.159.03, ffmpeg
/// 8.0 + hevc_vulkan encoder, 1080p @ 60 fps QP=0 lossless,
/// 2.5× real-time.
fn hevc_vulkan_recording_thread(
    slot: Arc<LatestTaskSlot>,
    stop: Arc<AtomicBool>,
    path: String,
    min_fps: Option<u32>,
) {
    use crate::hevc_vk_record::{open_hevc_mkv, FrameInputView, HevcVkEncoder};
    use crate::vulkan_record::{BgraToYuv444Pipeline, FrameResources444, VkDeviceCtx};
    use ash::vk;

    // Discover dimensions from the first frame.
    let mut first_pixels: Option<Vec<u8>> = None;
    let mut first_dmabuf: Option<(
        Arc<crate::dmabuf::DmabufBufferData>,
        Vec<Arc<InnerBufferHold>>,
    )> = None;
    let (w, h) = {
        info!(
            "hevc recording: waiting up to {}s for first frame from session",
            FIRST_FRAME_TIMEOUT.as_secs()
        );
        let deadline = std::time::Instant::now() + FIRST_FRAME_TIMEOUT;
        loop {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                warn!(
                    "hevc recording: no frames within {}s; aborting",
                    FIRST_FRAME_TIMEOUT.as_secs()
                );
                return;
            }
            let wait = remaining.min(std::time::Duration::from_millis(200));
            let Some(task) = slot.take_blocking(wait) else {
                continue;
            };
            const MIN_DIM: u32 = 32;
            let too_small = |w: u32, h: u32| w < MIN_DIM || h < MIN_DIM;
            match task {
                RecordingTask::Pixels {
                    pixels,
                    width,
                    height,
                } => {
                    if too_small(width, height) {
                        continue;
                    }
                    first_pixels = Some(pixels);
                    break (width, height);
                }
                RecordingTask::Dmabuf { dma, _holds } => {
                    let dw = dma.width as u32;
                    let dh = dma.height as u32;
                    if too_small(dw, dh) {
                        continue;
                    }
                    first_dmabuf = Some((dma, _holds));
                    break (dw, dh);
                }
                RecordingTask::Nal { .. } => {
                    warn!("hevc recording: pre-encoded NAL task unexpected; aborting");
                    return;
                }
            }
        }
    };
    info!(path = %path, width = w, height = h, "hevc recording starting");

    let ctx = match VkDeviceCtx::open() {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "hevc recording: VkDeviceCtx::open failed; aborting");
            return;
        }
    };
    if !ctx.hi444_supported {
        warn!(
            "hevc recording: device {} doesn't expose Hi444 caps — hevc_vulkan \
             RangeExt requires Vulkan-video 4:4:4 support",
            ctx.device_name
        );
        return;
    }
    let fr = match FrameResources444::new(&ctx, w, h) {
        Ok(f) => f,
        Err(e) => {
            warn!(error = %e, "hevc recording: FrameResources444::new failed; aborting");
            return;
        }
    };
    let pipe = match BgraToYuv444Pipeline::new(&ctx, 1) {
        Ok(p) => p,
        Err(e) => {
            warn!(error = %e, "hevc recording: BgraToYuv444Pipeline::new failed; aborting");
            return;
        }
    };
    let mut enc = match HevcVkEncoder::open(&ctx, w, h) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "hevc recording: HevcVkEncoder::open failed; aborting");
            return;
        }
    };
    let file = match std::fs::File::create(&path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            warn!(error = %e, path = %path, "hevc recording: open output failed");
            return;
        }
    };
    let mut mkv = match open_hevc_mkv(&enc, file) {
        Ok(m) => m,
        Err(e) => {
            warn!(error = %e, "hevc recording: open_hevc_mkv failed");
            return;
        }
    };

    let start = std::time::Instant::now();
    let mut frames: u64 = 0;
    let mut last_pixels: Option<Vec<u8>> = None;
    let min_frame_interval = min_fps
        .filter(|n| *n > 0)
        .map(|n| std::time::Duration::from_micros(1_000_000 / n as u64));
    let mut next_tick = std::time::Instant::now();

    // Encode one BGRA frame: run the compute shader to populate
    // fr.yuv_image (2-plane NV24), then hand to the hevc_vulkan encoder.
    let encode_one_bgra = |enc: &mut HevcVkEncoder,
                           mkv: &mut waymux_mux_mkv::MkvWriter<_>,
                           bgra: &[u8],
                           pts_us: i64,
                           pts_ms: i64|
     -> Result<usize, String> {
        // Run compute shader BGRA → 2-plane NV24 into fr.yuv_image.
        // The shader path is the same one we built for Hi444PP
        // (encode_idr_gpu_synthetic_yuv444), but we only need the
        // compute half — the actual encode is via hevc_vulkan.
        crate::vulkan_record::run_compute_yuv444_into_picture(&ctx, &fr, &pipe, bgra)?;
        // Hand the populated yuv_image to the encoder.
        let view = FrameInputView {
            image: fr.yuv_image,
            current_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            current_access: vk::AccessFlags2::empty(),
            current_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            current_queue_family: ctx.compute_queue_family,
        };
        let packets = enc.encode_one(&ctx, &view, pts_us)?;
        let n = packets.len();
        for p in packets {
            mkv.write_frame(&p.data, pts_ms, p.is_keyframe)
                .map_err(|e| format!("mkv write_frame: {e:?}"))?;
        }
        Ok(n)
    };

    // First frame.
    if let Some(first) = first_pixels.take() {
        match encode_one_bgra(&mut enc, &mut mkv, &first, 0, 0) {
            Ok(n) => frames += n as u64,
            Err(e) => warn!(error = %e, "hevc recording: first-frame encode failed"),
        }
        if min_frame_interval.is_some() {
            last_pixels = Some(first);
        }
    } else if let Some((dma, holds)) = first_dmabuf.take() {
        if let Some((pixels, fw, fh)) = read_task(RecordingTask::Dmabuf {
            dma: dma.clone(),
            _holds: holds,
        }) {
            if fw == w && fh == h {
                match encode_one_bgra(&mut enc, &mut mkv, &pixels, 0, 0) {
                    Ok(n) => frames += n as u64,
                    Err(e) => warn!(error = %e, "hevc recording: first-frame encode failed"),
                }
                if min_frame_interval.is_some() {
                    last_pixels = Some(pixels);
                }
            }
        }
    }

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let take_result = if let Some(interval) = min_frame_interval {
            let now = std::time::Instant::now();
            if next_tick > now {
                std::thread::sleep(next_tick - now);
            }
            next_tick = std::time::Instant::now() + interval;
            slot.take_blocking(std::time::Duration::ZERO)
        } else {
            slot.take_blocking(std::time::Duration::from_millis(200))
        };
        if stop.load(Ordering::Acquire) {
            break;
        }
        let task = match take_result {
            Some(t) => t,
            None => {
                if min_frame_interval.is_none() {
                    continue;
                }
                if let Some(ref pixels) = last_pixels {
                    let pts_ms = start.elapsed().as_millis() as i64;
                    let pts_us = pts_ms * 1000;
                    match encode_one_bgra(&mut enc, &mut mkv, pixels, pts_us, pts_ms) {
                        Ok(n) => frames += n as u64,
                        Err(e) => warn!(error = %e, "hevc recording: duplicate encode failed"),
                    }
                }
                continue;
            }
        };
        let Some((pixels, fw, fh)) = read_task(task) else {
            continue;
        };
        if fw != w || fh != h {
            warn!("hevc recording: frame size {fw}x{fh} != recorder {w}x{h}; skipping");
            continue;
        }
        let pts_ms = start.elapsed().as_millis() as i64;
        let pts_us = pts_ms * 1000;
        match encode_one_bgra(&mut enc, &mut mkv, &pixels, pts_us, pts_ms) {
            Ok(n) => frames += n as u64,
            Err(e) => warn!(error = %e, "hevc recording: encode failed"),
        }
        if min_frame_interval.is_some() {
            last_pixels = Some(pixels);
        }
    }

    // Flush encoder tail.
    match enc.flush() {
        Ok(tail) => {
            for p in tail {
                let pts_ms = if p.pts_us >= 0 {
                    p.pts_us / 1_000
                } else {
                    start.elapsed().as_millis() as i64
                };
                if let Err(e) = mkv.write_frame(&p.data, pts_ms, p.is_keyframe) {
                    warn!(error = %e, "hevc recording: flush write_frame failed");
                    break;
                }
                frames += 1;
            }
        }
        Err(e) => warn!(error = %e, "hevc recording: flush failed"),
    }
    if let Err(e) = mkv.finish() {
        warn!(error = %e, "hevc recording: finish failed");
    }
    info!(path = %path, frames = frames, "hevc recording finished");
}

/// Read pixels from a recording task into a packed BGRA Vec.
///
/// For the dmabuf path, the readback (and any wait_for_dmabuf_fence) happens
/// here on the recording thread, NOT on the compositor's outer_view thread.
/// On 1080p AMD this is ~30ms; doing it inline would cap display fps at ~30.
fn read_task(task: RecordingTask) -> Option<(Vec<u8>, u32, u32)> {
    match task {
        RecordingTask::Pixels {
            pixels,
            width,
            height,
        } => Some((pixels, width, height)),
        RecordingTask::Nal { .. } => {
            // NAL tasks are only meaningful in the Vulkan recording
            // thread; the legacy ffmpeg path can't consume encoded
            // bytes. Drop on the floor here.
            None
        }
        RecordingTask::Dmabuf { dma, _holds } => {
            // Skip when the implicit read fence isn't ready. Without this,
            // `with_bytes` blocks up to 2 s waiting on chromium's GPU when
            // it falls behind under load (heavy multi-tab, scrolling). Each
            // 2 s stall caps the frame rate at <1 fps; the first frame goes
            // through, then we fall further behind every iteration. The
            // next commit from the inner client will queue a fresher
            // buffer, whose fence is more likely to be ready.
            use std::os::fd::AsRawFd;
            if !crate::dmabuf::dmabuf_fence_ready_now(dma.fd.as_raw_fd()) {
                return None;
            }
            let pixels = dma.with_bytes(|raw| {
                destride_bgra(raw, dma.width as u32, dma.height as u32, dma.stride)
            })?;
            Some((pixels, dma.width as u32, dma.height as u32))
        }
    }
}

/// Build the ffmpeg command-line argv for the given codec choice.
///
/// All three codecs share the rawvideo BGRA stdin input + wall-clock
/// timestamping; the divergence is the encoder + per-encoder pixel
/// format conversion + (for VAAPI) hardware-device init.
///
/// Bitrate target: 5 Mbps for the lossy paths. That's roughly the
/// quality YouTube uses for 1080p; over-shoots for static pages,
/// undershoots slightly for video. Tunable later via env / RPC if a
/// customer needs a different point on the size/quality curve.
fn build_ffmpeg_argv(
    codec: waymux_protocol::RecordingCodec,
    size_str: &str,
    path: &str,
) -> Vec<String> {
    use waymux_protocol::RecordingCodec;

    // Common args every codec wants: read raw BGRA frames from stdin,
    // wall-clock timestamps, VFR output, no audio.
    let mut argv: Vec<String> = vec!["-y".into(), "-hide_banner".into()];
    if matches!(codec, RecordingCodec::H264Vaapi) {
        // VAAPI needs the device set BEFORE the input so the hwupload
        // filter can reference it.
        argv.extend(["-vaapi_device".into(), "/dev/dri/renderD128".into()]);
    }
    // For h264_nvenc and h264_vaapi we hand ffmpeg pre-converted NV12 to
    // skip ffmpeg's single-threaded swscale BGRA→NV12 stage (the encoder-
    // side ceiling at 4K). Ffv1 stays BGRA (lossless capture format).
    let push_nv12 = matches!(codec, RecordingCodec::H264Nvenc | RecordingCodec::H264Vaapi);
    let input_pix_fmt = if push_nv12 { "nv12" } else { "bgra" };
    argv.extend([
        "-f".into(),
        "rawvideo".into(),
        "-pixel_format".into(),
        input_pix_fmt.into(),
        "-video_size".into(),
        size_str.into(),
        // -framerate is taken literally by NVENC for H.264 level validation
        // (4K × 1000 fps = no valid level → "Invalid Level"). 240 covers
        // 4K@240/8K@30 and still fits cleanly in level 6.2. Real frame timing
        // comes from -use_wallclock_as_timestamps so this is just a hint.
        "-framerate".into(),
        "240".into(),
        "-use_wallclock_as_timestamps".into(),
        "1".into(),
        "-i".into(),
        "pipe:0".into(),
    ]);

    // Per-codec encoding args.
    match codec {
        RecordingCodec::Ffv1 => {
            argv.extend(["-vcodec".into(), "ffv1".into(), "-level".into(), "3".into()]);
        }
        RecordingCodec::H264Nvenc => {
            // Bitrate scales with pixel count so 4K hero clips don't look like
            // 5 Mbps mush. Base of 5 Mbps for 1080p (≈2.4 Mbits per megapixel
            // per 30 fps), capped at 50 Mbps so we don't generate disk-eaters.
            // Override via WAYMUX_NVENC_BITRATE_KBPS if a workflow needs a
            // specific point on the size/quality curve.
            let bitrate = bitrate_for_size(size_str);
            // Input is already NV12 (waymux converted it pre-pipe). No -vf
            // filter needed — ffmpeg hands the NV12 frame straight to NVENC.
            argv.extend([
                "-c:v".into(),
                "h264_nvenc".into(),
                "-preset".into(),
                "p4".into(),
                "-tune".into(),
                "ll".into(),
                "-rc".into(),
                "vbr".into(),
                "-b:v".into(),
                format!("{}k", bitrate),
                // ffmpeg's auto -level pick fails ("Invalid Level") at 4K on
                // some NVENC builds. Pin to 6.2 — covers 8K@120 / 4K@240,
                // safely larger than anything we'd ever record.
                "-level".into(),
                "6.2".into(),
            ]);
        }
        RecordingCodec::H264Vaapi => {
            // Input is already NV12. hwupload alone moves it to a VAAPI
            // surface for the H.264 encoder. qp=22 is approximately CRF 22
            // for libx264 — visually transparent for screencast content.
            argv.extend([
                "-vf".into(),
                "hwupload".into(),
                "-c:v".into(),
                "h264_vaapi".into(),
                "-qp".into(),
                "22".into(),
            ]);
        }
        RecordingCodec::H264Vulkan => {
            // Should never reach the ffmpeg argv builder — the
            // Vulkan path uses an in-process MkvWriter and bypasses
            // ffmpeg entirely. recording_thread dispatches before
            // we get here. Treat as a programming error rather than
            // a silent fallback.
            unreachable!("H264Vulkan does not use ffmpeg argv");
        }
        RecordingCodec::Ffv1Vulkan => {
            // Same as H264Vulkan: in-process Vulkan-compute encode path
            // (ffmpeg's ffv1_vulkan called via libavcodec, not via ffmpeg
            // subprocess). recording_thread dispatches before we get here.
            unreachable!("Ffv1Vulkan does not use ffmpeg argv");
        }
        RecordingCodec::H264VulkanLossless => {
            unreachable!("H264VulkanLossless does not use ffmpeg argv");
        }
        RecordingCodec::HevcVulkanLossless => {
            unreachable!("HevcVulkanLossless does not use ffmpeg argv");
        }
        RecordingCodec::CudaNvenc => {
            // CudaNvenc drives the NVENC hardware via the CUDA driver API
            // directly (cuda_nvenc_record path); it does not go through the
            // ffmpeg subprocess. recording_thread dispatches before reaching
            // this argv builder.
            unreachable!("CudaNvenc does not use ffmpeg argv");
        }
    }

    // Common output args: VFR for accurate timing, no audio, output path.
    argv.extend(["-vsync".into(), "vfr".into(), "-an".into(), path.into()]);
    argv
}

/// Pick a target bitrate (kbps) for h264_nvenc given the recording
/// dimensions. 5 Mbps baseline for 1080p, scaled linearly by pixel
/// count, clamped 2–50 Mbps so 4K recordings look good without
/// blowing up disk. Override via `WAYMUX_NVENC_BITRATE_KBPS`.
fn bitrate_for_size(size_str: &str) -> u32 {
    if let Some(kbps) = std::env::var("WAYMUX_NVENC_BITRATE_KBPS")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return kbps;
    }
    let (w, h) = match size_str.split_once('x') {
        Some((w, h)) => match (w.parse::<u32>(), h.parse::<u32>()) {
            (Ok(w), Ok(h)) => (w, h),
            _ => return 5_000,
        },
        None => return 5_000,
    };
    let pixels = (w as u64) * (h as u64);
    let pixels_1080p: u64 = 1920 * 1080;
    let scaled = 5_000u64 * pixels / pixels_1080p.max(1);
    scaled.clamp(2_000, 50_000) as u32
}

fn recording_thread(
    slot: Arc<LatestTaskSlot>,
    stop: Arc<AtomicBool>,
    path: String,
    codec: waymux_protocol::RecordingCodec,
    min_fps: Option<u32>,
) {
    // ── Wait for the first frame ──────────────────────────────────────────
    //
    // We don't know the recording dimensions until outer_view puts the first
    // task in the slot. ffmpeg requires fixed -video_size, so we can't spawn
    // it until we know. Block here patiently (FIRST_FRAME_TIMEOUT); abort on
    // RecordStop or true timeout.
    let first = {
        info!(
            "recording: waiting up to {}s for first frame from session",
            FIRST_FRAME_TIMEOUT.as_secs()
        );
        let deadline = std::time::Instant::now() + FIRST_FRAME_TIMEOUT;
        loop {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                warn!(
                    "recording: no frames received within {}s; aborting",
                    FIRST_FRAME_TIMEOUT.as_secs()
                );
                return;
            }
            let wait = remaining.min(std::time::Duration::from_millis(200));
            if let Some(task) = slot.take_blocking(wait) {
                if let Some(p) = read_task(task) {
                    break p;
                }
                // read_task failure (None): loop and try again with the next
                // arriving frame. Re-check stop on next iteration.
            }
        }
    };
    let (first_pixels, first_w, first_h) = first;

    let size_str = format!("{}x{}", first_w, first_h);
    info!(path = %path, size = %size_str, "recording starting");

    // ── Spawn ffmpeg (VFR, wall-clock timestamps) ─────────────────────────
    //
    // Common flags across codecs:
    //   -framerate 1000      → 1 ms input timebase.
    //   -use_wallclock_as_timestamps 1 → ffmpeg stamps each frame by the
    //                          wall clock at the moment it reads from stdin.
    //   -vsync vfr           → preserves wall-clock PTS in MKV output, so
    //                          playback is correctly-timed regardless of
    //                          actual capture rate.
    //
    // Codec branches build argv per encoder. ffv1 stays the lossless
    // default; h264_nvenc / h264_vaapi are the lossy hardware encoders
    // selected via --codec. All three target an MKV container — h264 in
    // MKV is widely playable (mpv/vlc/ffmpeg all open it).
    let log_path = format!("{path}.log");
    let stderr_file = std::fs::File::create(&log_path)
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null());

    let argv = build_ffmpeg_argv(codec, &size_str, &path);
    let mut child = match std::process::Command::new("ffmpeg")
        .args(&argv)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(stderr_file)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "recording: failed to spawn ffmpeg");
            return;
        }
    };

    let mut stdin = child.stdin.take().expect("stdin was piped");
    // Grow the stdin pipe buffer. Linux default is 64 KB — at 4K BGRA
    // (33 MB/frame) each write blocks immediately on ffmpeg's read rate,
    // serializing producer and encoder. 4 MB lets ~one full frame ride
    // in the buffer, decoupling the threads. F_SETPIPE_SZ clamps to
    // /proc/sys/fs/pipe-max-size (1 MB on most kernels, raise that to
    // 4 MB+ on rented Blackwell boxes for full effect).
    {
        use std::os::fd::AsRawFd;
        const F_SETPIPE_SZ: i32 = 1031;
        let _ = unsafe { libc::fcntl(stdin.as_raw_fd(), F_SETPIPE_SZ, 4 * 1024 * 1024) };
    }
    let mut frames: u64 = 0;

    // Codec-dependent CPU-side conversion. For h264_nvenc and h264_vaapi we
    // hand ffmpeg NV12 directly (build_ffmpeg_argv set `-pixel_format nv12`)
    // so it skips its single-threaded swscale BGRA→NV12 stage. ffv1 stays
    // BGRA. The closure captures the codec choice for both first-frame and
    // loop bodies.
    let needs_nv12 = matches!(
        codec,
        waymux_protocol::RecordingCodec::H264Nvenc | waymux_protocol::RecordingCodec::H264Vaapi,
    );
    // The compositor may already have done BGRA→NV12 on the GPU
    // (gpu_record::GpuConverter). Detect by byte length: NV12 is
    // w*h*3/2 bytes, BGRA is w*h*4. If it matches NV12 already, pass
    // through; otherwise run the CPU rayon converter.
    let to_wire = |buf: Vec<u8>, w: u32, h: u32| -> Vec<u8> {
        if !needs_nv12 {
            return buf;
        }
        let nv12_len = (w as usize) * (h as usize) * 3 / 2;
        let bgra_len = (w as usize) * (h as usize) * 4;
        if buf.len() == nv12_len {
            buf
        } else if buf.len() == bgra_len {
            bgra_to_nv12(&buf, w, h)
        } else {
            // Unexpected size — pass through so ffmpeg's error surfaces
            // immediately rather than silently corrupting the encode.
            buf
        }
    };

    let first_wire = to_wire(first_pixels, first_w, first_h);
    if let Err(e) = stdin.write_all(&first_wire) {
        warn!(error = %e, "recording: ffmpeg stdin write failed (first frame)");
        drop(stdin);
        let _ = child.wait();
        return;
    }
    frames += 1;

    // ── Frame loop ────────────────────────────────────────────────────────
    //
    // Two pacing modes:
    //   - `min_fps = None` (default): pure commit-driven. Block up to
    //     200 ms for the next commit, write each new frame as it arrives,
    //     write nothing when the inner client is idle. Idle pages produce
    //     0 fps and files stay small. This is the headline behavior.
    //   - `min_fps = Some(n)`: guaranteed-minimum pacing. Wait at most
    //     1/n seconds for a new commit; if one arrives, encode it; if the
    //     wait expires, encode the MOST RECENT frame again so the output
    //     has ≥ n fps throughout. Use for marketing hero clips that need
    //     a uniform frame rate.
    let min_frame_interval = min_fps
        .filter(|n| *n > 0)
        .map(|n| std::time::Duration::from_micros(1_000_000 / n as u64));
    let take_timeout = min_frame_interval.unwrap_or(std::time::Duration::from_millis(200));
    // Cache of the last successfully-encoded pixels for the duplication
    // path. Holds wire-format bytes (NV12 for h264 codecs, BGRA for ffv1)
    // so the min-fps duplication path doesn't re-run the conversion.
    let mut last_pixels: Vec<u8> = first_wire;
    let mut last_log = std::time::Instant::now();
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let take_result = slot.take_blocking(take_timeout);
        if stop.load(Ordering::Acquire) {
            break;
        }
        let (wire_pixels, w, h, read_ms) = match take_result {
            Some(task) => {
                let read_started = std::time::Instant::now();
                let Some((pixels, w, h)) = read_task(task) else {
                    continue;
                };
                let wire = to_wire(pixels, w, h);
                (wire, w, h, read_started.elapsed().as_millis())
            }
            None => {
                // Slot was empty for the whole wait window.
                if min_frame_interval.is_none() {
                    // Commit-driven mode — idle is OK, just keep waiting.
                    continue;
                }
                // Min-fps mode — duplicate the last (already wire-format)
                // frame to meet the pacing guarantee. No re-conversion.
                (last_pixels.clone(), first_w, first_h, 0)
            }
        };
        if w != first_w || h != first_h {
            warn!(
                "recording: dimensions changed {}x{} → {}x{}, stopping",
                first_w, first_h, w, h
            );
            break;
        }
        if let Err(e) = stdin.write_all(&wire_pixels) {
            warn!(error = %e, "recording: ffmpeg stdin write failed");
            break;
        }
        last_pixels = wire_pixels;
        frames += 1;

        if last_log.elapsed() >= std::time::Duration::from_secs(2) {
            let produced = slot.produced.load(Ordering::Relaxed);
            let dropped = slot.dropped.load(Ordering::Relaxed);
            info!(
                frames,
                produced,
                dropped,
                last_read_ms = read_ms as u64,
                "recording: progress"
            );
            last_log = std::time::Instant::now();
        }
    }

    // ── Shutdown: close stdin → ffmpeg finalizes MKV ──────────────────────
    drop(stdin);
    let _ = child.wait();
    info!(path = %path, frames, "recording stopped");
}

/// Copy a strided BGRA dmabuf slice into a packed Vec<u8> suitable for
/// ffmpeg rawvideo input. When stride == width*4 (no row padding) this
/// is a single fast memcpy; otherwise each row is copied individually.
pub fn destride_bgra(src: &[u8], width: u32, height: u32, stride: u32) -> Vec<u8> {
    let row_bytes = (width * 4) as usize;
    let stride = stride as usize;
    let total = row_bytes * height as usize;
    if stride == row_bytes {
        src[..total].to_vec()
    } else {
        let mut out = Vec::with_capacity(total);
        for row in 0..height as usize {
            out.extend_from_slice(&src[row * stride..row * stride + row_bytes]);
        }
        out
    }
}

/// Convert a packed BGRA frame to NV12 (planar Y + interleaved UV) using
/// BT.709 limited range, parallelized across rayon's thread pool.
///
/// NV12 layout:
///   - Y plane: width × height bytes (one byte per pixel)
///   - UV plane: width × (height/2) bytes (one U+V pair per 2×2 pixel block)
///
/// Result Vec is 1.5 × width × height bytes — 2.67× smaller than BGRA.
/// Pushing NV12 instead of BGRA to ffmpeg lets us drop the single-threaded
/// `-vf format=nv12` swscale stage, which is the encoder-side ceiling at 4K
/// (~30 ms/frame). Rayon parallelism scales the conversion across all
/// available CPU cores.
///
/// Width and height must be even (NV12 chroma subsamples 2:1 in both axes).
pub fn bgra_to_nv12(src: &[u8], width: u32, height: u32) -> Vec<u8> {
    use rayon::prelude::*;
    let w = width as usize;
    let h = height as usize;
    debug_assert!(
        w.is_multiple_of(2) && h.is_multiple_of(2),
        "NV12 needs even dimensions"
    );
    let y_size = w * h;
    let uv_size = w * h / 2;
    let mut out = vec![0u8; y_size + uv_size];
    let (y_plane, uv_plane) = out.split_at_mut(y_size);

    // BT.709 limited-range fixed-point coefficients (×256, +128 rounding).
    // Y'  = ( 47*R + 157*G +  16*B + 128) / 256 + 16
    // Cb' = (-26*R -  87*G + 112*B + 128) / 256 + 128
    // Cr' = (112*R - 102*G -  10*B + 128) / 256 + 128
    //
    // Source is BGRA: B at offset 0, G at 1, R at 2, A at 3.

    // Y plane: one byte per source pixel. Parallelize over rows.
    y_plane
        .par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(row, y_row)| {
            let src_row = &src[row * w * 4..(row + 1) * w * 4];
            for x in 0..w {
                let b = src_row[x * 4] as i32;
                let g = src_row[x * 4 + 1] as i32;
                let r = src_row[x * 4 + 2] as i32;
                let y = (47 * r + 157 * g + 16 * b + 128) >> 8;
                y_row[x] = (y + 16).clamp(0, 255) as u8;
            }
        });

    // UV plane: one (Cb, Cr) pair per 2×2 source block. Parallelize over
    // pairs of source rows; each chunk_exact_mut yields one UV row (w bytes,
    // alternating Cb, Cr) corresponding to two consecutive source rows.
    uv_plane
        .par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(uv_row, uv_dst)| {
            let r0 = uv_row * 2;
            let r1 = r0 + 1;
            let src_r0 = &src[r0 * w * 4..(r0 + 1) * w * 4];
            let src_r1 = &src[r1 * w * 4..(r1 + 1) * w * 4];
            for bx in 0..w / 2 {
                let x = bx * 2;
                // Average R/G/B over the 2×2 block.
                let avg = |off: usize| {
                    let v = src_r0[x * 4 + off] as i32
                        + src_r0[(x + 1) * 4 + off] as i32
                        + src_r1[x * 4 + off] as i32
                        + src_r1[(x + 1) * 4 + off] as i32;
                    v / 4
                };
                let b = avg(0);
                let g = avg(1);
                let r = avg(2);
                let cb = (-26 * r - 87 * g + 112 * b + 128) >> 8;
                let cr = (112 * r - 102 * g - 10 * b + 128) >> 8;
                uv_dst[bx * 2] = (cb + 128).clamp(0, 255) as u8;
                uv_dst[bx * 2 + 1] = (cr + 128).clamp(0, 255) as u8;
            }
        });

    out
}

/// Strided BGRA → NV12 single-pass converter, parallelized via rayon.
/// Combines `destride_bgra` and `bgra_to_nv12` — saves one full-frame
/// memcpy and one pass over the data versus calling them in sequence.
pub fn bgra_to_nv12_destride(src: &[u8], width: u32, height: u32, stride: u32) -> Vec<u8> {
    let row_bytes = (width * 4) as usize;
    let stride = stride as usize;
    if stride == row_bytes {
        return bgra_to_nv12(src, width, height);
    }
    // Padded stride: pack to a contiguous BGRA buffer first, then convert.
    // (Could be fused, but stride padding is rare; cost is one extra row-copy
    // pass which is negligible compared to the conversion math.)
    let packed = destride_bgra(src, width, height, stride as u32);
    bgra_to_nv12(&packed, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bgra_pixel(b: u8, g: u8, r: u8) -> [u8; 4] {
        [b, g, r, 0xff]
    }

    #[test]
    fn nv12_dimensions() {
        let w = 4u32;
        let h = 4u32;
        let src = vec![0u8; (w * h * 4) as usize];
        let nv12 = bgra_to_nv12(&src, w, h);
        assert_eq!(nv12.len(), ((w * h * 3) / 2) as usize, "Y + UV size");
    }

    #[test]
    fn nv12_black_input() {
        // All-black BGRA → Y=16, UV=128 (BT.709 limited range black).
        let (w, h) = (4u32, 4u32);
        let src = vec![0u8; (w * h * 4) as usize];
        let nv12 = bgra_to_nv12(&src, w, h);
        let y_size = (w * h) as usize;
        for &y in &nv12[..y_size] {
            assert_eq!(y, 16, "Y plane should be 16 for black");
        }
        for uv in nv12[y_size..].chunks_exact(2) {
            assert_eq!(uv, &[128, 128], "UV should be 128/128 for black");
        }
    }

    #[test]
    fn nv12_white_input() {
        // All-white BGRA → Y≈235, UV≈128 (BT.709 limited range white).
        let (w, h) = (4u32, 4u32);
        let mut src = vec![0u8; (w * h * 4) as usize];
        for px in src.chunks_exact_mut(4) {
            px.copy_from_slice(&bgra_pixel(0xff, 0xff, 0xff));
        }
        let nv12 = bgra_to_nv12(&src, w, h);
        let y_size = (w * h) as usize;
        for &y in &nv12[..y_size] {
            assert!((230..=240).contains(&y), "Y plane ≈ 235 for white, got {y}");
        }
        for uv in nv12[y_size..].chunks_exact(2) {
            for &c in uv {
                assert!((124..=132).contains(&c), "UV ≈ 128 for white, got {c}");
            }
        }
    }

    #[test]
    fn nv12_pure_red() {
        // Pure red → Y plane ~63, Cr (V) plane >> 128, Cb (U) < 128.
        let (w, h) = (4u32, 4u32);
        let mut src = vec![0u8; (w * h * 4) as usize];
        for px in src.chunks_exact_mut(4) {
            px.copy_from_slice(&bgra_pixel(0, 0, 0xff));
        }
        let nv12 = bgra_to_nv12(&src, w, h);
        let y_size = (w * h) as usize;
        for &y in &nv12[..y_size] {
            assert!((60..=70).contains(&y), "Y for pure red ≈ 63, got {y}");
        }
        let uv0 = nv12[y_size];
        let uv1 = nv12[y_size + 1];
        assert!(uv0 < 128, "Cb < 128 for pure red, got {uv0}");
        assert!(uv1 > 200, "Cr > 200 for pure red, got {uv1}");
    }
}
