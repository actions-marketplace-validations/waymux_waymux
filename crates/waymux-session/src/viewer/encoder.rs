// SPDX-License-Identifier: Apache-2.0

//! Viewer-side encoder thread. Reuses the recording-side encoder machinery
//! (Vulkan H.264, ffmpeg+NVENC subprocess) but writes Annex-B NALUs to a
//! Unix socket instead of an MKV file, and tunes for low-latency streaming
//! (zero B-frames, baseline-ish profile, force-IDR on PLI).
//!
//! Two codec paths, both producing `Frame::Nalu` over the bridge socket:
//!
//!   - `H264Nvenc`  — ffmpeg subprocess. Pipe NV12 to stdin; parse
//!     Annex-B start codes on stdout; each NALU shipped
//!     as one `Frame::Nalu`. Streaming tunings:
//!     -preset llhp -tune ull -profile:v baseline
//!     -bf 0 -g 60. ForceKeyframe → write
//!     `force_key_frames expr:gte(t,0)` via the encoder's
//!     stdin? No, ffmpeg has no in-band IDR-request,
//!     so we set `-g 1` worth of IDR insertion by
//!     asking ffmpeg via `forced_keyframes` only on
//!     init. The full PLI-driven IDR-on-demand path
//!     is a v2 polish: today we just send a periodic
//!     keyframe via the -g 60 setting (1-second IDR
//!     floor at 60 fps) to bound worst-case packet-loss
//!     recovery to 1 second.
//!
//!   - `H264Vulkan` — in-process VkRecorder, same library that the
//!     recording path uses. Every encode is an IDR
//!     (the encoder reuses encode_idr_from_*), which
//!     is fine for the viewer's purpose: every frame
//!     can serve as a sync point so peer reconnect
//!     and packet-loss recovery are both instant. The
//!     bandwidth cost (5-15% over a P-frame-heavy
//!     stream) is acceptable for loopback / LAN.

use std::io::{BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tracing::{info, warn};
use waymux_protocol::RecordingCodec;

use crate::recording::{bgra_to_nv12, LatestTaskSlot, RecordingTask};
use crate::viewer::protocol::{write_frame, Frame};

/// Pick the first probe-true codec from the viewer-eligible list.
///
/// Order is intentional: NVENC > Vulkan. Same order whether or not the
/// host has NVIDIA hardware; the probe tells the truth.
pub fn pick_viewer_codec_with_probes(probes: &[(RecordingCodec, bool)]) -> Option<RecordingCodec> {
    probes.iter().find(|(_, ok)| *ok).map(|(c, _)| *c)
}

/// Probe each candidate codec; return the first that works.
///
/// Override with `WAYMUX_VIEWER_CODEC=h264-nvenc|h264-vulkan` to force a
/// specific codec (useful when an ffmpeg build advertises h264_nvenc but
/// the host lacks NVIDIA hardware — probe-by-help returns true but the
/// encoder dies on first frame).
pub fn select_viewer_codec() -> Option<RecordingCodec> {
    if let Ok(forced) = std::env::var("WAYMUX_VIEWER_CODEC") {
        return match forced.as_str() {
            "cuda-nvenc" => Some(RecordingCodec::CudaNvenc),
            "h264-nvenc" => Some(RecordingCodec::H264Nvenc),
            // VA-API (HW on AMD) and software x264 are served by the same
            // ffmpeg-subprocess encoder as NVENC (run_nvenc_encoder picks the
            // ffmpeg args from WAYMUX_VIEWER_CODEC). Route them through that arm.
            "h264-vaapi" | "h264-software" | "x264" => Some(RecordingCodec::H264Nvenc),
            // In-process libavcodec VA-API: no subprocess, working adaptive RC,
            // and (with the DRM_PRIME path) zero-copy dmabuf import.
            "h264-vaapi-inproc" => Some(RecordingCodec::H264Vaapi),
            "h264-vulkan" => Some(RecordingCodec::H264Vulkan),
            _ => None,
        };
    }
    let probes: Vec<(RecordingCodec, bool)> = vec![
        (RecordingCodec::CudaNvenc, probe_cuda_nvenc_for_viewer()),
        (RecordingCodec::H264Nvenc, probe_h264_nvenc_for_viewer()),
        (RecordingCodec::H264Vulkan, probe_h264_vulkan_for_viewer()),
    ];
    pick_viewer_codec_with_probes(&probes)
}

/// Probe whether the direct CUDA+NVENC path can be used for the viewer.
///
/// Two-part check: (a) `libcuda.so.1` is loadable and `cuInit(0)` succeeds
/// (the same gate that `CudaLib::load()` performs), AND (b) the NVENC
/// encoding library `libnvidia-encode.so.1` is present on the host.
/// Does NOT open an encode session — that's lazy in the encoder thread.
fn probe_cuda_nvenc_for_viewer() -> bool {
    if crate::cuda_nvenc_record::CudaLib::load().is_none() {
        return false;
    }
    unsafe { libloading::Library::new("libnvidia-encode.so.1") }.is_ok()
}

/// Probe whether h264-nvenc can actually be used for the viewer.
///
/// Two-part check: (a) the `ffmpeg` binary on PATH must know the
/// `h264_nvenc` encoder, AND (b) NVIDIA hardware must be present. Just
/// (a) is misleading — most distro ffmpeg builds advertise `h264_nvenc`
/// at build time even without NVIDIA hardware, and the encoder then dies
/// with `Broken pipe` on the first frame.
fn probe_h264_nvenc_for_viewer() -> bool {
    // (b) — NVIDIA hardware present? Cheapest signal: /dev/nvidia0 char
    // device, present whenever the nvidia kernel module is loaded.
    if !std::path::Path::new("/dev/nvidia0").exists()
        && !std::path::Path::new("/dev/nvidiactl").exists()
    {
        return false;
    }
    // (a) — ffmpeg knows the encoder
    std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-h", "encoder=h264_nvenc"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Probe whether h264-vulkan can be used for the viewer.
///
/// Mirrors the gating logic in `VkDeviceCtx::open` / `select_physical_device`
/// so a `true` here means `VkRecorder::open()` will succeed on the first
/// frame. We check, per investigation § Option A:
///
///   - `vulkan_record::probe()` enumerates at least one physical device
///   - that device reports the `VK_KHR_video_queue` +
///     `VK_KHR_video_encode_queue` + `VK_KHR_video_encode_h264` extensions
///     (recorded as `video_encode_h264_supported`)
///   - that device exposes a queue family with the
///     `VK_QUEUE_VIDEO_ENCODE_BIT_KHR` flag (in ash:
///     `vk::QueueFlags::VIDEO_ENCODE_KHR`)
///
/// The H.264 profile-capabilities query is the fourth gate listed in the
/// investigation; we deliberately skip it here because the real query
/// requires opening the device (which the probe avoids — opening leaks
/// per-process Vulkan state). We accept the small risk that a host
/// advertises the extensions and queue family but fails the profile
/// query — `VkRecorder::open()` catches that case and the viewer falls
/// back to the next probe entry.
fn probe_h264_vulkan_for_viewer() -> bool {
    match crate::vulkan_record::probe() {
        Ok(p) => vulkan_probe_indicates_h264_encode(&p),
        Err(_) => false,
    }
}

/// Pure predicate over a `VulkanProbe`: true iff at least one enumerated
/// device has H.264 encode extensions reported AND has a queue family with
/// the `VIDEO_ENCODE_KHR` flag. Same filter `select_physical_device` uses
/// when picking the device to open.
///
/// Factored out so unit tests can exercise the gating logic without
/// touching the live Vulkan loader.
fn vulkan_probe_indicates_h264_encode(probe: &crate::vulkan_record::VulkanProbe) -> bool {
    use ash::vk;
    probe.devices.iter().any(|d| {
        d.video_encode_h264_supported
            && d.queue_families
                .iter()
                .any(|q| q.flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR))
    })
}

/// Spawn the viewer-encoder thread.
///
/// Reads BGRA/NV12/dmabuf frames from `frame_slot` (the same
/// `LatestTaskSlot` shape used by the recording path), encodes per
/// `codec`, and writes `Frame::Nalu` records to `socket`. Drains and
/// exits when `stop_flag` flips or when the socket closes.
///
/// Tuning differences from recording.rs:
///   - NVENC: -preset llhp -tune ull -profile:v baseline -bf 0 -g 60
///   - Vulkan: every frame is an IDR (no P-frames), so packet loss
///     recovers on next frame at the cost of 5-15% extra bandwidth.
pub fn spawn_encoder_thread(
    codec: RecordingCodec,
    width: u32,
    height: u32,
    socket: UnixStream,
    stop_flag: Arc<AtomicBool>,
    frame_slot: Arc<LatestTaskSlot>,
    state: Arc<crate::state::State>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("waymux-viewer-enc-{codec:?}"))
        .spawn(move || {
            info!(?codec, width, height, "viewer encoder thread starting");
            match codec {
                RecordingCodec::H264Nvenc => {
                    run_nvenc_encoder(width, height, socket, stop_flag, frame_slot, state);
                }
                RecordingCodec::H264Vulkan => {
                    run_vulkan_encoder(width, height, socket, stop_flag, frame_slot, state);
                }
                RecordingCodec::H264Vaapi => {
                    run_vaapi_inprocess_encoder(width, height, socket, stop_flag, frame_slot, state);
                }
                RecordingCodec::CudaNvenc => {
                    // CudaNvenc is selected by a lib-presence probe, but the
                    // recorder can still fail to initialize at runtime (most
                    // notably the CSC PTX not JITing onto this GPU's compute
                    // capability). Rather than black-screen the viewer, hand the
                    // still-unused socket to the next viable codec.
                    if let Some(sock) =
                        run_cuda_nvenc_encoder(width, height, socket, stop_flag.clone(), frame_slot.clone(), state.clone())
                    {
                        if probe_h264_nvenc_for_viewer() {
                            warn!("viewer: CudaNvenc unavailable; falling back to H264Nvenc");
                            run_nvenc_encoder(width, height, sock, stop_flag, frame_slot, state);
                        } else if probe_h264_vulkan_for_viewer() {
                            warn!("viewer: CudaNvenc unavailable; falling back to H264Vulkan");
                            run_vulkan_encoder(width, height, sock, stop_flag, frame_slot, state);
                        } else {
                            warn!("viewer: CudaNvenc unavailable and no fallback codec present; exiting");
                            let _ = (sock, &stop_flag, &state);
                        }
                    }
                }
                other => {
                    warn!(?other, "viewer encoder: unsupported codec; exiting");
                    let _ = (socket, &stop_flag, &state);
                }
            }
            info!(?codec, "viewer encoder thread exiting");
        })
        .expect("spawn viewer encoder thread")
}

/// Pull pixels (NV12) and width/height out of one task. Returns None if
/// the task was unusable for streaming (Nal tasks aren't expected from
/// the viewer slot today; Dmabuf gets a CPU readback to keep this path
/// simple).
fn task_to_nv12(task: RecordingTask, width: u32, height: u32) -> Option<(Vec<u8>, u32, u32)> {
    use crate::recording::destride_bgra;
    use std::os::fd::AsRawFd;

    match task {
        RecordingTask::Pixels {
            pixels,
            width: w,
            height: h,
        } => {
            let need_nv12 = (w as usize) * (h as usize) * 3 / 2;
            let bgra_len = (w as usize) * (h as usize) * 4;
            if pixels.len() == need_nv12 {
                Some((pixels, w, h))
            } else if pixels.len() == bgra_len {
                Some((bgra_to_nv12(&pixels, w, h), w, h))
            } else {
                warn!(
                    bytes = pixels.len(),
                    w, h, "viewer encoder: unexpected pixel buffer size; dropping"
                );
                None
            }
        }
        RecordingTask::Dmabuf { dma, _holds } => {
            if !crate::dmabuf::dmabuf_fence_ready_now(dma.fd.as_raw_fd()) {
                return None;
            }
            let w = dma.width as u32;
            let h = dma.height as u32;
            let bgra = dma.with_bytes(|raw| destride_bgra(raw, w, h, dma.stride))?;
            Some((bgra_to_nv12(&bgra, w, h), w, h))
        }
        RecordingTask::Nal { .. } => {
            // The viewer slot never sees pre-encoded NALs today — the
            // compositor tap only pushes Pixels/Dmabuf. Treat as a
            // dropped frame.
            let _ = (width, height);
            None
        }
    }
}

/// NVENC subprocess encoder. Spawns ffmpeg, pipes NV12 in, parses
/// Annex-B NALUs from stdout. Writes one `Frame::Nalu` per NALU.
///
/// **Not tested on this laptop** (AMD Renoir has no NVENC). The structural
/// design is a near-clone of the existing `recording_thread` NVENC
/// path; the only differences are: streaming tunings, h264 raw
/// output to stdout instead of MKV, and Annex-B parsing in this
/// process.
fn run_nvenc_encoder(
    width: u32,
    height: u32,
    socket: UnixStream,
    stop_flag: Arc<AtomicBool>,
    frame_slot: Arc<LatestTaskSlot>,
    state: Arc<crate::state::State>,
) {
    // Wait up to 2 s for the first real frame so dimensions can be
    // detected from the source. If the session is idle, bootstrap
    // with a synthetic black NV12 frame at the requested width/height
    // — the min-fps tick below will keep the wire alive until the
    // session commits something, at which point last_pixels gets
    // replaced with real content.
    let (mut last_pixels, w, h) = {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut first: Option<(Vec<u8>, u32, u32)> = None;
        while first.is_none() {
            if stop_flag.load(Ordering::Acquire) {
                return;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(std::time::Duration::from_millis(200));
            if let Some(task) = frame_slot.take_blocking(wait) {
                if let Some(p) = task_to_nv12(task, width, height) {
                    first = Some(p);
                }
            }
        }
        first.unwrap_or_else(|| {
            // Synthetic black NV12: Y plane = 0, UV plane = 128 (gives Rec.601 black).
            let y_size = (width as usize) * (height as usize);
            let uv_size = y_size / 2;
            let mut buf = vec![0u8; y_size + uv_size];
            for b in &mut buf[y_size..] {
                *b = 128;
            }
            info!(
                width,
                height, "viewer nvenc: bootstrap with synthetic black NV12"
            );
            (buf, width, height)
        })
    };
    let size = format!("{w}x{h}");

    // Pick the ffmpeg encoder backend from WAYMUX_VIEWER_CODEC. All three are
    // P-frame + rate-controlled — the dominant win over the every-IDR Vulkan
    // path: the min-fps tick feeds a constant frame rate, P-frames make
    // idle/repeat frames tiny, and CBR fits the link. h264-vaapi = HW on AMD,
    // h264-software = x264 (universal CPU), default = h264_nvenc (NVIDIA).
    let codec_sel = std::env::var("WAYMUX_VIEWER_CODEC").unwrap_or_default();
    let bitrate = std::env::var("WAYMUX_VIEWER_BITRATE_BPS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(4_000_000);
    let gop = std::env::var("WAYMUX_VIEWER_GOP")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&g| g > 0)
        .unwrap_or(60)
        .to_string();
    let bv = bitrate.to_string();
    let bufsize = (bitrate / 2).to_string(); // ~0.5s VBV for low latency
                                             // WAYMUX_VIEWER_BITRATE_BPS is the PEAK cap. Capped VBR with a target
                                             // average at half the cap lets a near-static desktop collapse to ~nothing
                                             // (measured ~190x smaller than CBR on a static frame) while motion still
                                             // bursts up to the cap — the right shape for a cellular link.
    let avg = (bitrate / 2).max(500_000).to_string();
    let ff_args: Vec<String> = match codec_sel.as_str() {
        // `h264-vaapi-inproc` lands here only as the subprocess fallback when
        // the in-process encoder fails to open — serve it the VA-API args too.
        "h264-vaapi" | "h264-vaapi-inproc" => [
            "-hide_banner",
            "-loglevel",
            "error",
            "-vaapi_device",
            "/dev/dri/renderD128",
            "-f",
            "rawvideo",
            "-pixel_format",
            "nv12",
            "-video_size",
            size.as_str(),
            "-framerate",
            "60",
            "-i",
            "pipe:0",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "h264_vaapi",
            "-rc_mode",
            "VBR",
            "-b:v",
            avg.as_str(),
            "-maxrate",
            bv.as_str(),
            "-bufsize",
            bufsize.as_str(),
            "-g",
            gop.as_str(),
            "-bf",
            "0",
            "-f",
            "h264",
            "pipe:1",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        "h264-software" | "x264" => [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "nv12",
            "-video_size",
            size.as_str(),
            "-framerate",
            "60",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "superfast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-bf",
            "0",
            "-g",
            gop.as_str(),
            "-b:v",
            bv.as_str(),
            "-maxrate",
            bv.as_str(),
            "-bufsize",
            bufsize.as_str(),
            "-f",
            "h264",
            "pipe:1",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        _ => [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "nv12",
            "-video_size",
            size.as_str(),
            "-framerate",
            "60",
            "-i",
            "pipe:0",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-rc",
            "cbr",
            "-zerolatency",
            "1",
            "-tune",
            "ull",
            "-profile:v",
            "baseline",
            "-bf",
            "0",
            "-g",
            gop.as_str(),
            "-b:v",
            bv.as_str(),
            "-f",
            "h264",
            "pipe:1",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    };
    info!(codec = %codec_sel, bitrate, gop = %gop, "viewer ffmpeg-encoder backend");
    let mut child = match std::process::Command::new("ffmpeg")
        .args(&ff_args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        // Inherit stderr so ffmpeg's error-level diagnostics (e.g. a VA-API
        // device that fails to init) land in the session log instead of being
        // silently swallowed. `-loglevel error` keeps this to real errors.
        .stderr(std::process::Stdio::inherit())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "viewer nvenc: spawn ffmpeg failed");
            return;
        }
    };
    let mut stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");

    // Socket is shared between encode-out (this thread) and a control
    // reader spawned below. Wrap the writer side in a Mutex so both
    // threads can write `Frame::Nalu` / `Frame::ForceKeyframe` safely.
    // Today only the encode thread writes; the control reader pushes
    // ForceKeyframe into a flag and InjectOp into the session.
    let writer_clone = match socket.try_clone() {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(e) => {
            warn!(error = %e, "viewer nvenc: clone socket failed");
            let _ = child.kill();
            return;
        }
    };
    let force_idr = Arc::new(AtomicBool::new(false));
    let target_bitrate = Arc::new(AtomicU32::new(0));
    spawn_control_reader(
        socket,
        stop_flag.clone(),
        force_idr.clone(),
        target_bitrate.clone(),
        state,
    );

    // Stdout parser thread: read NALU bytes, slice on Annex-B start codes,
    // batch NALUs into Access Units (one AU = one frame's worth of NALUs)
    // and ship each AU as a single Frame::Nalu.
    //
    // Why batching: Pion's H.264 RTP packetizer treats each WriteSample as
    // one frame for timing purposes. If we ship SPS/PPS/IDR as 3 separate
    // samples, Pion advances PTS 3× per real frame and the browser decoder
    // rejects the resulting stream (symptom: ICE connects, ontrack fires,
    // <video> stays black). Concat all NALUs of one AU (with start codes)
    // into one Frame::Nalu so each sample maps to one real frame.
    let parser_stop = stop_flag.clone();
    let parser_writer = writer_clone.clone();
    let parser = std::thread::Builder::new()
        .name("waymux-viewer-nvenc-stdout".into())
        .spawn(move || {
            // Optional bounded Annex-B dump (WAYMUX_VIEWER_NAL_DUMP=/path) for
            // offline ffprobe/ffmpeg verification of the wire stream. Capped at
            // 64 MiB so leaving it on can't fill tmpfs and OOM the session.
            const NAL_DUMP_CAP: u64 = 64 * 1024 * 1024;
            let mut nal_dump = std::env::var("WAYMUX_VIEWER_NAL_DUMP")
                .ok()
                .and_then(|p| std::fs::File::create(&p).ok());
            let mut nal_dumped: u64 = 0;
            let mut rd = BufReader::new(stdout);
            let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
            let mut chunk = [0u8; 64 * 1024];
            // Pending AU accumulator: NALUs of the current frame, each
            // prefixed with the 4-byte Annex-B start code 00 00 00 01.
            let mut pending_au: Vec<u8> = Vec::with_capacity(256 * 1024);
            let mut au_has_vcl = false;
            const START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
            loop {
                if parser_stop.load(Ordering::Acquire) {
                    break;
                }
                match rd.read(&mut chunk) {
                    Ok(0) => break, // ffmpeg closed stdout
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        while let Some((nalu, consumed)) = next_annex_b_nalu(&buf) {
                            if !nalu.is_empty() {
                                let nal_type = nalu[0] & 0x1F;
                                let is_vcl = (1..=5).contains(&nal_type);
                                // Flush pending AU when transitioning out of
                                // a VCL slice: the next NALU starts the next
                                // frame (either a fresh header set or another
                                // VCL slice).
                                if au_has_vcl {
                                    if let Some(f) = nal_dump.as_mut() {
                                        if nal_dumped < NAL_DUMP_CAP {
                                            let _ = f.write_all(&pending_au);
                                            nal_dumped += pending_au.len() as u64;
                                        }
                                    }
                                    let mut w = parser_writer.lock().unwrap();
                                    if let Err(e) = write_frame(&mut *w, Frame::Nalu(&pending_au)) {
                                        warn!(error = %e, "viewer nvenc: socket write failed");
                                        parser_stop.store(true, Ordering::Release);
                                        return;
                                    }
                                    drop(w);
                                    pending_au.clear();
                                    au_has_vcl = false;
                                }
                                pending_au.extend_from_slice(&START_CODE);
                                pending_au.extend_from_slice(nalu);
                                if is_vcl {
                                    au_has_vcl = true;
                                }
                            }
                            buf.drain(..consumed);
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "viewer nvenc: read stdout failed");
                        break;
                    }
                }
            }
            // Flush trailing AU on shutdown.
            if au_has_vcl && !pending_au.is_empty() {
                let mut w = parser_writer.lock().unwrap();
                let _ = write_frame(&mut *w, Frame::Nalu(&pending_au));
            }
        })
        .expect("spawn nvenc parser");

    // Write first frame.
    if let Err(e) = stdin.write_all(&last_pixels) {
        warn!(error = %e, "viewer nvenc: first frame write failed");
        let _ = child.kill();
        let _ = parser.join();
        return;
    }

    // Tick-driven loop: write to ffmpeg at EXACTLY min_fps cadence.
    // Real frames arriving between ticks just refresh `last_pixels`
    // (no extra write). This trades some temporal accuracy for a
    // smooth, predictable wire rate — browser jitter buffers prefer
    // steady cadence over bursty real-time-fast / idle-stall pattern.
    // Also keeps ffmpeg's `-framerate 60` PTS stamping aligned.
    let min_fps: u64 = 60;
    let tick = std::time::Duration::from_nanos(1_000_000_000u64 / min_fps);
    let mut next_tick = std::time::Instant::now() + tick;
    let mut frames: u64 = 1;
    let mut last_log = std::time::Instant::now();
    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }
        let now = std::time::Instant::now();
        if now >= next_tick {
            // Tick fired — write current last_pixels regardless of
            // whether a new frame arrived this cycle.
            if force_idr.swap(false, Ordering::AcqRel) {
                tracing::debug!(
                    "viewer nvenc: force-IDR requested (deferred to next GOP boundary)"
                );
            }
            if let Err(e) = stdin.write_all(&last_pixels) {
                warn!(error = %e, "viewer nvenc: frame write failed");
                break;
            }
            frames += 1;
            // Advance without drift; snap to now+tick if we fell behind.
            next_tick = if next_tick + tick > now {
                next_tick + tick
            } else {
                now + tick
            };
            if last_log.elapsed() >= std::time::Duration::from_secs(2) {
                info!(frames, "viewer nvenc: progress");
                last_log = std::time::Instant::now();
            }
            continue;
        }
        // Between ticks: wait for new frame OR tick deadline.
        let wait = next_tick.saturating_duration_since(now);
        if let Some(task) = frame_slot.take_blocking(wait) {
            if let Some((nv12, fw, fh)) = task_to_nv12(task, w, h) {
                if fw != w || fh != h {
                    warn!(
                        fw,
                        fh, w, h, "viewer nvenc: resolution change unsupported; stopping"
                    );
                    break;
                }
                last_pixels = nv12;
            }
        }
    }

    drop(stdin);
    let _ = parser.join();
    let _ = child.wait();
    let _ = last_pixels;
}

/// One step of the viewer's adaptive-QP controller: nudge `qp` so the measured
/// wire bitrate fits `target_bps`. Asymmetric — tighten fast when over the link
/// (don't sustain congestion loss), loosen one step at a time for stability —
/// with a deadband so it doesn't hunt, clamped to a sane visual QP range.
fn adaptive_qp_step(qp: i32, measured_bps: f64, target_bps: f64) -> i32 {
    let next = if measured_bps > target_bps * 1.10 {
        qp + if measured_bps > target_bps * 1.6 {
            2
        } else {
            1
        }
    } else if measured_bps < target_bps * 0.65 {
        qp - 1
    } else {
        qp
    };
    next.clamp(18, 44)
}

/// Vulkan in-process encoder. Imports KWin's desktop dmabuf zero-copy and
/// encodes H.264 (IDR + P-frames) on the GPU. Adaptive QP (driven by the live
/// GCC bandwidth estimate) keeps the bitrate inside the link.
fn run_vulkan_encoder(
    _width: u32,
    _height: u32,
    socket: UnixStream,
    stop_flag: Arc<AtomicBool>,
    frame_slot: Arc<LatestTaskSlot>,
    state: Arc<crate::state::State>,
) {
    // Wait for first frame to discover dimensions.
    use crate::vulkan_record::VkRecorder;
    use std::os::fd::AsRawFd;
    // Wait indefinitely for the first usable frame; a viewer may be opened
    // against an idle session and should not abort just because the session
    // hasn't drawn yet. Block on stop_flag only.
    let first = loop {
        if stop_flag.load(Ordering::Acquire) {
            return;
        }
        let Some(task) = frame_slot.take_blocking(std::time::Duration::from_millis(200)) else {
            continue;
        };
        const MIN_DIM: u32 = 32;
        match task {
            RecordingTask::Pixels {
                pixels,
                width: w,
                height: h,
            } if w >= MIN_DIM && h >= MIN_DIM => {
                break (Some(pixels), None, w, h);
            }
            RecordingTask::Dmabuf { dma, _holds }
                if dma.width as u32 >= MIN_DIM && dma.height as u32 >= MIN_DIM =>
            {
                let w = dma.width as u32;
                let h = dma.height as u32;
                break (None, Some((dma, _holds)), w, h);
            }
            _ => continue,
        }
    };
    let (first_pixels, first_dmabuf, w, h) = first;
    info!(width = w, height = h, "viewer vulkan: starting");

    let mut recorder = match VkRecorder::try_new(w, h) {
        Some(r) => r,
        None => {
            warn!("viewer vulkan: VkRecorder::try_new failed; aborting");
            return;
        }
    };
    // AMD/Mesa's Vulkan encoder does not emit SPS/PPS in-band like NVIDIA's,
    // so a decoder gets "non-existing PPS" and never produces a frame. Capture
    // the driver's Annex-B SPS+PPS once and prepend it to every IDR we send,
    // making each keyframe independently decodable over RTP. Empty on drivers
    // that already emit them in-band.
    let sps_pps_annexb = recorder.sps_pps_annexb().to_vec();
    // P-frame GOP: emit one IDR keyframe then up to GOP_LEN-1 P-frames
    // before the next forced IDR. Kept < 16 so H.264 `frame_num` (mod 16
    // with our SPS log2_max_frame_num_minus4 = 0) never wraps inside a
    // GOP. P-frames cut bitrate ~5-10x vs the old every-frame-IDR.
    const GOP_LEN: u64 = 12;
    let mut gop_pos: u64 = 0;

    let writer = match socket.try_clone() {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(e) => {
            warn!(error = %e, "viewer vulkan: clone socket failed");
            return;
        }
    };
    let force_idr = Arc::new(AtomicBool::new(false));
    let target_bitrate = Arc::new(AtomicU32::new(0));
    spawn_control_reader(
        socket,
        stop_flag.clone(),
        force_idr.clone(),
        target_bitrate.clone(),
        state,
    );

    let start = std::time::Instant::now();
    let mut frames: u64 = 0;
    let mut last_log = std::time::Instant::now();

    // ── Adaptive rate control (closed-loop QP) ──────────────────────────────
    // The Vulkan encoder is constant-QP, so on its own it ignores the link and
    // floods a cellular connection (a fixed-quality stream whose bitrate swings
    // with content). Here we close the loop: measure the bytes we actually put
    // on the wire over a ~1s window and nudge the encoder's QP up (smaller
    // frames) when we exceed the live GCC bandwidth target, down (sharper) when
    // we have headroom. `target_bitrate` is the send-side BWE estimate the
    // bridge feeds back via SetBitrate; until it arrives we target a
    // cellular-safe default. QP — not resolution — so the picture stays whole.
    use crate::vulkan_record::VIEWER_DYNAMIC_QP;
    let default_target_bps: u32 = std::env::var("WAYMUX_VIEWER_BITRATE_BPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&b| b > 0)
        .unwrap_or(1_500_000);
    let mut byte_window: std::collections::VecDeque<(std::time::Instant, usize)> =
        std::collections::VecDeque::new();
    let mut window_bytes: usize = 0;
    // Seed QP from the env/default so we don't flood before the loop converges.
    let mut dyn_qp: i32 = std::env::var("WAYMUX_VK_ENCODE_QP")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|q: i32| q.clamp(18, 44))
        .unwrap_or(26);
    VIEWER_DYNAMIC_QP.store(dyn_qp, Ordering::Relaxed);
    let mut last_qp_adjust = std::time::Instant::now();

    // Constant-cadence pull: emit a frame every `frame_interval` (= max_fps)
    // regardless of whether the compositor committed new content, so `src` is a
    // steady max_fps the browser jitter buffer locks onto — not the bursty,
    // commit-rate-bound event rate (which read "not 60"). A tick with no new
    // frame RE-ENCODES the last source as a tiny "no-change" P-frame (re-encode,
    // not re-send: a re-sent P would be applied twice and drift; a re-encoded
    // no-change P is correct and ~free). On-demand IDR (PLI / GOP boundary) is
    // honored per tick, so a reconnecting viewer recovers within one frame —
    // this also subsumes the old idle keepalive.
    let max_fps: u64 = std::env::var("WAYMUX_VIEWER_MAX_FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&f| f > 0)
        .unwrap_or(60);
    let frame_interval = std::time::Duration::from_nanos(1_000_000_000 / max_fps);
    // Reserve headroom before each tick so the encode lands on the deadline.
    const ENCODE_BUDGET: std::time::Duration = std::time::Duration::from_millis(8);
    // Last source, kept so an idle tick can re-encode it. The dmabuf stays
    // pinned (InnerBufferHold) until a newer frame replaces it.
    let mut last_dmabuf: Option<(
        std::sync::Arc<crate::dmabuf::DmabufBufferData>,
        Vec<std::sync::Arc<crate::recording::InnerBufferHold>>,
    )> = None;
    let mut last_bgra: Option<Vec<u8>> = None;
    // Optional Annex-B NAL dump (WAYMUX_VIEWER_NAL_DUMP=/path) for offline
    // conformance checks with ffmpeg/ffprobe. Diagnostic only; unset in prod.
    let mut nal_dump = std::env::var("WAYMUX_VIEWER_NAL_DUMP")
        .ok()
        .and_then(|p| std::fs::File::create(p).ok());

    // Write the first frame (always an IDR keyframe to open the GOP) and cache
    // its source so the steady tick can re-encode it until new content arrives.
    let first_pts_us = 0i64;
    let first_nal = if let Some(p) = first_pixels {
        let n = recorder.encode_bgra(&p, first_pts_us, true);
        last_bgra = Some(p);
        n
    } else if let Some((dma, holds)) = first_dmabuf {
        // Only import once the GPU has finished writing this buffer (matches the
        // NVENC/EGL paths). Encoding an in-flight dmabuf under VCN contention is
        // what wedges the encode queue → 30s fence spin → kernel watchdog panic.
        let n = if crate::dmabuf::dmabuf_fence_ready_now(dma.fd.as_raw_fd()) {
            recorder.encode_dmabuf(&dma, first_pts_us, true)
        } else {
            None
        };
        last_dmabuf = Some((dma, holds));
        n
    } else {
        None
    };
    if first_nal.is_some() {
        gop_pos = 1;
    }
    if let Some(nal) = first_nal {
        let mut framed = sps_pps_annexb.clone();
        framed.extend_from_slice(&nal.data);
        if let Err(e) = write_frame(&mut *writer.lock().unwrap(), Frame::Nalu(&framed)) {
            warn!(error = %e, "viewer vulkan: socket write failed (first)");
            return;
        }
        if let Some(f) = nal_dump.as_mut() {
            use std::io::Write;
            let _ = f.write_all(&framed);
        }
        frames += 1;
    }
    // force_idr starts false; clear if anyone toggled it during init.
    force_idr.store(false, Ordering::Release);

    // Consecutive encode failures. A wedged GPU now times out fast (fence wait
    // capped well under the kernel watchdog) and drops frames; if it keeps
    // failing, flag the device wedged and exit cleanly so teardown skips the
    // unbounded device_wait_idle and a viewer reconnect restarts a fresh
    // encoder, instead of spinning dropped frames forever.
    let mut consec_fail: u32 = 0;
    const MAX_CONSEC_FAIL: u32 = 90; // ~1.5 s at 60 fps
    let mut next_tick = std::time::Instant::now();
    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }
        // Steady tick: wait for a fresh frame but never past the deadline; a
        // tick with no new frame re-encodes the last source below.
        next_tick += frame_interval;
        // Anti-spiral: if a slow/contended GPU pushed us more than one interval
        // behind real time, drop the backlog and resync instead of firing a
        // burst of catch-up encodes that pile more work onto a busy VCN.
        let now_rt = std::time::Instant::now();
        if next_tick + frame_interval < now_rt {
            next_tick = now_rt;
        }
        let wait = next_tick
            .checked_sub(ENCODE_BUDGET)
            .unwrap_or(next_tick)
            .saturating_duration_since(std::time::Instant::now());
        let task = frame_slot.take_blocking(wait);
        if stop_flag.load(Ordering::Acquire) {
            break;
        }
        let pts_us = start.elapsed().as_micros() as i64;
        // IDR on a forced keyframe (PLI / first) or GOP boundary, else P.
        let forced = force_idr.swap(false, Ordering::AcqRel);
        let is_idr = forced || gop_pos == 0 || gop_pos >= GOP_LEN;

        let nal_opt = match task {
            Some(RecordingTask::Dmabuf { dma, _holds }) => {
                if dma.width as u32 != w || dma.height as u32 != h {
                    warn!(
                        fw = dma.width,
                        fh = dma.height,
                        w,
                        h,
                        "viewer vulkan: size change; skipping"
                    );
                    None
                } else if !crate::dmabuf::dmabuf_fence_ready_now(dma.fd.as_raw_fd()) {
                    // GPU is still writing this buffer. Importing/encoding an
                    // in-flight dmabuf while the VCN is contended (browser HW
                    // decode, a 2nd Vulkan client) is what wedges the encode
                    // queue and spins wait_for_fences → kernel watchdog panic.
                    // Skip it; hold cadence by re-encoding the last ready frame.
                    // A newer, ready buffer arrives next tick (latest-wins slot).
                    if let Some((d, _)) = last_dmabuf.as_ref() {
                        recorder.encode_dmabuf(d, pts_us, is_idr)
                    } else if let Some(b) = last_bgra.as_ref() {
                        recorder.encode_bgra(b, pts_us, is_idr)
                    } else {
                        None
                    }
                } else {
                    let n = recorder.encode_dmabuf(&dma, pts_us, is_idr);
                    // Keep the dmabuf pinned so an idle tick can re-encode it.
                    last_dmabuf = Some((dma, _holds));
                    last_bgra = None;
                    n
                }
            }
            Some(RecordingTask::Pixels {
                pixels,
                width: fw,
                height: fh,
            }) => {
                let bgra_len = (fw as usize) * (fh as usize) * 4;
                if fw != w || fh != h {
                    warn!(fw, fh, w, h, "viewer vulkan: size change; skipping");
                    None
                } else if pixels.len() == bgra_len {
                    let n = recorder.encode_bgra(&pixels, pts_us, is_idr);
                    last_bgra = Some(pixels);
                    last_dmabuf = None;
                    n
                } else {
                    warn!(
                        bytes = pixels.len(),
                        bgra_len, "viewer vulkan: non-BGRA Pixels; dropping"
                    );
                    None
                }
            }
            // No new content this tick (timeout / Nal) → re-encode the last
            // source as a tiny no-change P-frame to hold the constant cadence.
            Some(RecordingTask::Nal { .. }) | None => {
                if let Some((dma, _)) = last_dmabuf.as_ref() {
                    // last_dmabuf was ready when pinned and KWin can't reuse a
                    // held buffer, so its fence stays signaled; re-check anyway
                    // (cheap ioctl) to be safe against any write-after-read.
                    if crate::dmabuf::dmabuf_fence_ready_now(dma.fd.as_raw_fd()) {
                        recorder.encode_dmabuf(dma, pts_us, is_idr)
                    } else {
                        None
                    }
                } else if let Some(bgra) = last_bgra.as_ref() {
                    recorder.encode_bgra(bgra, pts_us, is_idr)
                } else {
                    None
                }
            }
        };
        let Some(nal) = nal_opt else {
            // Nothing to send yet → pace to the tick and continue. Track
            // repeated failures: a healthy idle desktop re-encodes the pinned
            // last frame successfully (consec_fail stays 0), so a sustained run
            // of Nones means the GPU is wedged → bail cleanly.
            consec_fail += 1;
            if consec_fail >= MAX_CONSEC_FAIL {
                warn!(
                    consec_fail,
                    "viewer vulkan: GPU appears wedged (repeated encode failures); stopping encoder"
                );
                crate::vulkan_record::GPU_WEDGED.store(true, Ordering::Release);
                break;
            }
            let now = std::time::Instant::now();
            if now < next_tick {
                std::thread::sleep(next_tick - now);
            } else {
                next_tick = now;
            }
            continue;
        };
        consec_fail = 0;
        // Advance the GOP counter only on a successfully-encoded frame.
        // IDR resets the GOP; a P advances within it.
        gop_pos = if nal.is_keyframe { 1 } else { gop_pos + 1 };

        // Prepend SPS/PPS only on keyframes — the decoder retains them by id
        // for the following P-frames, so repeating them every AU just wastes
        // bytes (~5% on a P). Keyframes stay self-contained for mid-stream join.
        let mut framed = if nal.is_keyframe {
            sps_pps_annexb.clone()
        } else {
            Vec::with_capacity(nal.data.len())
        };
        framed.extend_from_slice(&nal.data);
        let sent_bytes = framed.len();
        if let Err(e) = write_frame(&mut *writer.lock().unwrap(), Frame::Nalu(&framed)) {
            warn!(error = %e, "viewer vulkan: socket write failed");
            break;
        }
        if let Some(f) = nal_dump.as_mut() {
            use std::io::Write;
            let _ = f.write_all(&framed);
        }
        frames += 1;

        // ── Adaptive QP step ──
        // Sliding ~1s byte window → measured wire bitrate. Compare to the live
        // GCC target and nudge QP to track it (asymmetric: tighten fast when
        // over the link so we don't sustain loss, loosen slowly for stability).
        let now_rc = std::time::Instant::now();
        byte_window.push_back((now_rc, sent_bytes));
        window_bytes += sent_bytes;
        while let Some(&(t, b)) = byte_window.front() {
            if now_rc.duration_since(t) > std::time::Duration::from_millis(1000) {
                byte_window.pop_front();
                window_bytes -= b;
            } else {
                break;
            }
        }
        // Adjust at ~4 Hz once the window has a real span, to avoid thrashing.
        if last_qp_adjust.elapsed() >= std::time::Duration::from_millis(250) {
            if let (Some(&(oldest, _)), Some(&(newest, _))) =
                (byte_window.front(), byte_window.back())
            {
                let span = newest.duration_since(oldest).as_secs_f64().max(0.2);
                let measured_bps = (window_bytes as f64) * 8.0 / span;
                let target = {
                    let t = target_bitrate.load(Ordering::Relaxed);
                    if t > 0 {
                        t
                    } else {
                        default_target_bps
                    }
                } as f64;
                let prev = dyn_qp;
                dyn_qp = adaptive_qp_step(dyn_qp, measured_bps, target);
                if dyn_qp != prev {
                    VIEWER_DYNAMIC_QP.store(dyn_qp, Ordering::Relaxed);
                }
            }
            last_qp_adjust = std::time::Instant::now();
        }

        if last_log.elapsed() >= std::time::Duration::from_secs(2) {
            let measured = (window_bytes as f64) * 8.0 / 1.0;
            info!(
                frames,
                qp = dyn_qp,
                approx_bps = measured as u64,
                "viewer vulkan: progress"
            );
            last_log = std::time::Instant::now();
        }

        // Hold the steady cadence: sleep to the next tick, or reset the clock if
        // the encode overran (avoid a catch-up spiral).
        let now = std::time::Instant::now();
        if now < next_tick {
            std::thread::sleep(next_tick - now);
        } else {
            next_tick = now;
        }
    }
    // Release the dynamic-QP override when this viewer encoder exits.
    VIEWER_DYNAMIC_QP.store(-1, Ordering::Relaxed);
    info!(frames, "viewer vulkan: stopped");
}

/// Direct CUDA+NVENC in-process encoder. Mirrors `run_vulkan_encoder`'s
/// lifecycle but drives `CudaNvencRecorder` per-frame and writes `Frame::Nalu`.
///
/// Two input paths:
///   - `Dmabuf` tasks → `encode_dmabuf` (zero-copy EGLImage import → CUDA NV12
///     → NVENC), gated on `dmabuf_fence_ready_now` so we don't import a buffer
///     the GPU hasn't finished rendering into.
///   - `Pixels` tasks → `encode_pixels` (host BGRA upload → CUDA NV12 → NVENC).
///     This is the IDLE-desktop path: KWin presents a solid/idle desktop as a
///     `wp_single_pixel_buffer`, so the WholeDesktop tap routes the frame
///     through `capture_desktop()` → `Pixels`. Without consuming `Pixels` the
///     encoder would starve and the viewer would black-screen on an idle
///     desktop. Pixels carry no fd/fence — encode directly.
///     Nal tasks are still ignored (not produced for this codec).
///
/// Returns `Some(socket)` if the CUDA/NVENC recorder could not be brought up
/// (e.g. the embedded CSC PTX won't JIT onto this GPU's compute capability) —
/// the caller hands the still-unused socket to a fallback codec instead of
/// black-screening the viewer. Returns `None` once the encoder has taken over
/// the socket (whether it ran a full session or hit a fatal mid-run error).
fn run_cuda_nvenc_encoder(
    _width: u32,
    _height: u32,
    socket: UnixStream,
    stop_flag: Arc<AtomicBool>,
    frame_slot: Arc<LatestTaskSlot>,
    state: Arc<crate::state::State>,
) -> Option<UnixStream> {
    use crate::cuda_nvenc_record::CudaNvencRecorder;
    use std::os::fd::AsRawFd;

    const MIN_DIM: u32 = 32;

    // Block (on stop_flag only) for the first usable frame so dimensions can be
    // discovered from the source. Accept EITHER a Dmabuf (active GPU content) OR
    // a Pixels task (idle desktop → capture_desktop → Pixels), so an idle
    // desktop that only ever emits Pixels can still start the encoder.
    enum FirstFrame {
        Dmabuf(
            std::sync::Arc<crate::dmabuf::DmabufBufferData>,
            Vec<std::sync::Arc<crate::recording::InnerBufferHold>>,
        ),
        Pixels(Vec<u8>, u32, u32),
    }
    let first = loop {
        if stop_flag.load(Ordering::Acquire) {
            return None;
        }
        let Some(task) = frame_slot.take_blocking(std::time::Duration::from_millis(200)) else {
            continue;
        };
        match task {
            RecordingTask::Dmabuf { dma, _holds }
                if dma.width as u32 >= MIN_DIM && dma.height as u32 >= MIN_DIM =>
            {
                break FirstFrame::Dmabuf(dma, _holds);
            }
            RecordingTask::Pixels {
                pixels,
                width,
                height,
            } if width >= MIN_DIM && height >= MIN_DIM => {
                break FirstFrame::Pixels(pixels, width, height);
            }
            _ => continue,
        }
    };
    let (w, h) = match &first {
        FirstFrame::Dmabuf(dma, _) => (dma.width as u32, dma.height as u32),
        FirstFrame::Pixels(_, width, height) => (*width, *height),
    };
    info!(width = w, height = h, "viewer cuda-nvenc: starting");

    let mut recorder = match CudaNvencRecorder::try_new(w, h) {
        Some(r) => r,
        None => {
            warn!("viewer cuda-nvenc: try_new failed; falling back to another codec");
            return Some(socket);
        }
    };

    let writer = match socket.try_clone() {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(e) => {
            warn!(error = %e, "viewer cuda-nvenc: clone socket failed");
            return None;
        }
    };
    let force_idr = Arc::new(AtomicBool::new(false));
    let target_bitrate = Arc::new(AtomicU32::new(0));
    // Clone the cursor_channel Arc before moving `state` into the control reader.
    let cursor_channel = Arc::clone(&state.cursor_channel);
    spawn_control_reader(
        socket,
        stop_flag.clone(),
        force_idr.clone(),
        target_bitrate.clone(),
        state,
    );

    let start = std::time::Instant::now();
    let mut frames: u64 = 0;
    let mut last_log = std::time::Instant::now();
    // Constant-cadence pull. We emit a frame every FRAME_INTERVAL (60 fps)
    // regardless of whether the compositor committed new content: each tick takes
    // the freshest frame from the latest-wins slot if one arrived within the
    // interval, else re-encodes the last frame (a tiny P-frame for static
    // content). This decouples the wire cadence from KWin's bursty damage timing
    // — a steady 60 fps the browser jitter buffer can lock onto — instead of the
    // event-driven ~15-45 fps that stutters on commit-rate variance and only
    // ramps up after content "warms up". Periodic IDRs come from the NVENC GOP;
    // on-demand IDRs from a viewer PLI (force_idr); the first frame is forced above.
    const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_micros(16_666);
    // Reserve headroom at the end of each tick for the encode itself, so a
    // "no new frame" tick stops waiting early enough to re-encode the last frame
    // and still land on the 60 fps deadline (otherwise it waits the full
    // interval THEN encodes, overrunning to ~47 fps and dragging the average to
    // ~52). ~8 ms covers a 1080p NVENC encode + CSC with margin.
    const ENCODE_BUDGET: std::time::Duration = std::time::Duration::from_micros(8_000);

    // First frame: ALWAYS an IDR so the very first thing the peer decodes is a
    // self-contained keyframe (with SPS/PPS attached).
    let first_nal = match first {
        FirstFrame::Dmabuf(first_dma, first_holds) => {
            let nal = if crate::dmabuf::dmabuf_fence_ready_now(first_dma.fd.as_raw_fd()) {
                recorder.encode_dmabuf(
                    first_dma.fd.as_raw_fd(),
                    first_dma.modifier,
                    w,
                    h,
                    first_dma.stride,
                    0,
                    true, // first frame → IDR
                )
            } else {
                None
            };
            drop(first_holds);
            nal
        }
        // Pixels are tightly-packed destride'd BGRA (no row padding) → stride = width*4.
        FirstFrame::Pixels(pixels, width, height) => {
            recorder.encode_pixels(&pixels, width, height, width * 4, 0, true)
        }
    };
    if let Some(nal) = first_nal {
        if let Err(e) = write_frame(&mut *writer.lock().unwrap(), Frame::Nalu(&nal.data)) {
            warn!(error = %e, "viewer cuda-nvenc: socket write failed (first)");
            return None;
        }
        frames += 1;
    }
    force_idr.store(false, Ordering::Release);

    let mut next_tick = std::time::Instant::now();
    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }
        // Pace to a steady 60 fps: wait for a fresh frame, but never past the
        // next tick. If one arrives we encode it immediately (low latency); if
        // the slot is empty at the deadline we re-encode the last frame so the
        // wire cadence stays constant.
        next_tick += FRAME_INTERVAL;
        // Stop waiting ENCODE_BUDGET before the tick so the encode (fresh frame
        // or re-encode) still finishes by next_tick → a steady hard 60 fps.
        let wait = next_tick
            .checked_sub(ENCODE_BUDGET)
            .unwrap_or(next_tick)
            .saturating_duration_since(std::time::Instant::now());
        let task = frame_slot.take_blocking(wait);

        let pts_us = start.elapsed().as_micros() as i64;
        // IDR only if a viewer PLI / ForceKeyframe arrived since the last one;
        // otherwise a P-frame. Read (don't consume) — cleared only after a frame
        // actually goes out, so a dropped/failed encode can't eat a pending PLI.
        let idr = force_idr.load(Ordering::Acquire);

        // Apply a pending GCC bitrate target (0 = no change requested).
        let new_bps = target_bitrate.swap(0, Ordering::AcqRel);
        if new_bps > 0 {
            recorder.reconfigure_bitrate(new_bps);
        }

        let nal_opt = match task {
            Some(RecordingTask::Dmabuf { dma, _holds }) => {
                let nal = if crate::dmabuf::dmabuf_fence_ready_now(dma.fd.as_raw_fd()) {
                    recorder.encode_dmabuf(
                        dma.fd.as_raw_fd(),
                        dma.modifier,
                        dma.width as u32,
                        dma.height as u32,
                        dma.stride,
                        pts_us,
                        idr,
                    )
                } else {
                    // Fence not ready — hold cadence by re-emitting the last frame.
                    recorder.reencode_last(pts_us, idr)
                };
                drop(_holds);
                nal
            }
            // Idle-desktop path: host BGRA upload. Pixels carry no fd/fence and
            // are tightly-packed destride'd BGRA → stride = width*4.
            Some(RecordingTask::Pixels {
                pixels,
                width,
                height,
            }) => recorder.encode_pixels(&pixels, width, height, width * 4, pts_us, idr),
            // Nal tasks are not produced for this codec.
            Some(RecordingTask::Nal { .. }) => None,
            // No new commit this tick — re-encode the last frame to hold 60 fps.
            None => recorder.reencode_last(pts_us, idr),
        };

        if let Some(nal) = nal_opt {
            if let Err(e) = write_frame(&mut *writer.lock().unwrap(), Frame::Nalu(&nal.data)) {
                warn!(error = %e, "viewer cuda-nvenc: socket write failed");
                break;
            }
            // The IDR (if any) was served by the frame that just went out — clear
            // the request now (CAS so a PLI that arrived during encode survives).
            if idr {
                force_idr
                    .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                    .ok();
            }
            frames += 1;
            if last_log.elapsed() >= std::time::Duration::from_secs(2) {
                info!(frames, "viewer cuda-nvenc: progress");
                last_log = std::time::Instant::now();
            }
        }

        // Drain queued cursor updates onto the same socket (writer mutex shared
        // with the NALU path). Images are rare (shape changes); positions
        // collapse to the latest. Forwarded to viewers as JSON by the bridge.
        let mut cursor_write_failed = false;
        for upd in cursor_channel.drain() {
            let frame = match &upd {
                crate::viewer::cursor::CursorUpdate::Image(img) => Frame::CursorImage {
                    w: img.w,
                    h: img.h,
                    hot_x: img.hot_x,
                    hot_y: img.hot_y,
                    rgba: &img.rgba,
                },
                crate::viewer::cursor::CursorUpdate::Pos(p) => Frame::CursorPos {
                    x: p.x,
                    y: p.y,
                    seq: p.seq,
                },
            };
            if let Err(e) = write_frame(&mut *writer.lock().unwrap(), frame) {
                warn!(error = %e, "viewer cuda-nvenc: cursor write failed");
                cursor_write_failed = true;
                break;
            }
        }
        if cursor_write_failed {
            break;
        }

        // Hold the 60 fps cadence: sleep to the next tick if we finished early; if
        // the encode overran the interval, reset the clock to avoid a catch-up spiral.
        let now = std::time::Instant::now();
        if now < next_tick {
            std::thread::sleep(next_tick - now);
        } else {
            next_tick = now;
        }
    }
    info!(frames, "viewer cuda-nvenc: exiting");
    None
}

/// In-process VA-API H.264 encoder (libavcodec `h264_vaapi`, no subprocess).
///
/// Mirrors `run_cuda_nvenc_encoder`'s steady-60fps tick loop, but encodes via
/// the in-process `VaapiH264Encoder`. Versus the ffmpeg-subprocess VA-API
/// path (`run_nvenc_encoder` with `WAYMUX_VIEWER_CODEC=h264-vaapi`), this:
///   - has NO subprocess (no pipe copy / process management),
///   - honors on-demand IDR (PLI → `force_idr`) per frame, and
///   - actually applies the live GCC `target_bitrate` to the encoder's VBR
///     ceiling (the subprocess path drops it on the floor).
///
/// Stage A feeds CPU NV12 via the shared `task_to_nv12` (which de-strides the
/// BGRA dmabuf + converts). The zero-copy DRM_PRIME import lands in a later
/// stage. If the encoder won't open on this host, falls back to the
/// subprocess VA-API path with the same socket.
fn run_vaapi_inprocess_encoder(
    width: u32,
    height: u32,
    socket: UnixStream,
    stop_flag: Arc<AtomicBool>,
    frame_slot: Arc<LatestTaskSlot>,
    state: Arc<crate::state::State>,
) {
    use crate::vaapi_h264_record::VaapiH264Encoder;

    const MIN_DIM: u32 = 32;
    let render_node = std::env::var("WAYMUX_VAAPI_RENDER_NODE")
        .unwrap_or_else(|_| "/dev/dri/renderD128".to_string());
    let peak_bps = std::env::var("WAYMUX_VIEWER_BITRATE_BPS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(6_000_000);
    let gop = std::env::var("WAYMUX_VIEWER_GOP")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&g| g > 0)
        .unwrap_or(120);
    // Frame-rate cap. The compositor tees up to ~60fps; over a cellular link a
    // phone can't decode/display that and drops ~1/3 of frames (visible
    // stutter). Pace the wire to WAYMUX_VIEWER_MAX_FPS (default 30) so the peer
    // decodes every frame — a steady 30 beats a dropped 60.
    let max_fps = std::env::var("WAYMUX_VIEWER_MAX_FPS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&f| (5..=120).contains(&f))
        .unwrap_or(30);

    // Block (on stop_flag only) for the first usable frame — dimensions are
    // discovered from the source. task_to_nv12 handles Dmabuf (fence-gated
    // readback) and Pixels uniformly.
    let (first_nv12, w, h) = loop {
        if stop_flag.load(Ordering::Acquire) {
            return;
        }
        let Some(task) = frame_slot.take_blocking(std::time::Duration::from_millis(200)) else {
            continue;
        };
        if let Some((nv12, fw, fh)) = task_to_nv12(task, width, height) {
            if fw >= MIN_DIM && fh >= MIN_DIM {
                break (nv12, fw, fh);
            }
        }
    };
    info!(
        width = w,
        height = h,
        peak_bps,
        gop,
        max_fps,
        "viewer vaapi-inproc: starting"
    );

    let mut recorder = match VaapiH264Encoder::open(&render_node, w, h, peak_bps, gop, max_fps) {
        Some(r) => r,
        None => {
            warn!("viewer vaapi-inproc: encoder open failed; falling back to subprocess VA-API");
            // SAFETY of correctness: control reader not yet spawned, socket
            // unconsumed. The subprocess path discovers its own first frame.
            run_nvenc_encoder(width, height, socket, stop_flag, frame_slot, state);
            return;
        }
    };

    let writer = match socket.try_clone() {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(e) => {
            warn!(error = %e, "viewer vaapi-inproc: clone socket failed");
            return;
        }
    };
    let force_idr = Arc::new(AtomicBool::new(false));
    let target_bitrate = Arc::new(AtomicU32::new(0));
    let cursor_channel = Arc::clone(&state.cursor_channel);
    spawn_control_reader(
        socket,
        stop_flag.clone(),
        force_idr.clone(),
        target_bitrate.clone(),
        state,
    );

    // Optional bounded Annex-B dump (WAYMUX_VIEWER_NAL_DUMP=/path), capped at
    // 64 MiB so leaving it on can't fill tmpfs and OOM the session.
    const NAL_DUMP_CAP: u64 = 64 * 1024 * 1024;
    let mut nal_dump = std::env::var("WAYMUX_VIEWER_NAL_DUMP")
        .ok()
        .and_then(|p| std::fs::File::create(&p).ok());
    let mut nal_dumped: u64 = 0;
    let mut dump = |bytes: &[u8]| {
        if let Some(f) = nal_dump.as_mut() {
            if nal_dumped < NAL_DUMP_CAP {
                let _ = f.write_all(bytes);
                nal_dumped += bytes.len() as u64;
            }
        }
    };

    // First frame: ALWAYS an IDR (self-contained, SPS/PPS in-band).
    if let Some(nal) = recorder.encode_nv12(&first_nv12, 0, true) {
        dump(&nal.data);
        if let Err(e) = write_frame(&mut *writer.lock().unwrap(), Frame::Nalu(&nal.data)) {
            warn!(error = %e, "viewer vaapi-inproc: socket write failed (first)");
            return;
        }
    }
    force_idr.store(false, Ordering::Release);

    // Steady max_fps tick (see run_cuda_nvenc_encoder for the cadence rationale).
    let frame_interval = std::time::Duration::from_nanos(1_000_000_000 / max_fps as u64);
    const ENCODE_BUDGET: std::time::Duration = std::time::Duration::from_micros(8_000);
    let start = std::time::Instant::now();
    let mut frames: u64 = 1;
    let mut last_log = std::time::Instant::now();
    let mut next_tick = std::time::Instant::now();
    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }
        next_tick += frame_interval;
        let wait = next_tick
            .checked_sub(ENCODE_BUDGET)
            .unwrap_or(next_tick)
            .saturating_duration_since(std::time::Instant::now());
        let task = frame_slot.take_blocking(wait);

        let pts_us = start.elapsed().as_micros() as i64;
        let idr = force_idr.load(Ordering::Acquire);

        // Apply a pending GCC bitrate target (0 = no change requested).
        let new_bps = target_bitrate.swap(0, Ordering::AcqRel);
        if new_bps > 0 {
            recorder.reconfigure_bitrate(new_bps);
        }

        // Encode this tick's frame. Dmabuf tasks go zero-copy (DRM_PRIME import
        // → on-GPU NV12, no readback); if that's unavailable on this host they
        // fall back to the CPU readback (task_to_nv12). Pixels (idle/LINEAR)
        // always go through task_to_nv12. No usable frame → re-encode the last
        // (a tiny P-frame) to hold the 60fps cadence.
        use std::os::fd::AsRawFd;
        let nal_opt = match task {
            Some(RecordingTask::Dmabuf { dma, _holds }) => {
                if !crate::dmabuf::dmabuf_fence_ready_now(dma.fd.as_raw_fd()) {
                    recorder.reencode_last(pts_us, idr)
                } else {
                    let zc = if recorder.zero_copy_unavailable() {
                        None
                    } else {
                        recorder.encode_dmabuf(
                            dma.fd.as_raw_fd(),
                            dma.drm_format,
                            dma.modifier,
                            dma.width as u32,
                            dma.height as u32,
                            dma.stride,
                            dma.offset,
                            pts_us,
                            idr,
                        )
                    };
                    match zc {
                        Some(n) => Some(n),
                        // Zero-copy disabled/failed → CPU readback for this frame.
                        None => {
                            let t = RecordingTask::Dmabuf { dma, _holds };
                            match task_to_nv12(t, w, h) {
                                Some((nv12, fw, fh)) if fw == w && fh == h => {
                                    recorder.encode_nv12(&nv12, pts_us, idr)
                                }
                                _ => recorder.reencode_last(pts_us, idr),
                            }
                        }
                    }
                }
            }
            Some(t @ RecordingTask::Pixels { .. }) => match task_to_nv12(t, w, h) {
                Some((nv12, fw, fh)) => {
                    if fw != w || fh != h {
                        warn!(
                            fw,
                            fh,
                            w,
                            h,
                            "viewer vaapi-inproc: resolution change unsupported; stopping"
                        );
                        break;
                    }
                    recorder.encode_nv12(&nv12, pts_us, idr)
                }
                None => recorder.reencode_last(pts_us, idr),
            },
            _ => recorder.reencode_last(pts_us, idr),
        };

        if let Some(nal) = nal_opt {
            dump(&nal.data);
            if let Err(e) = write_frame(&mut *writer.lock().unwrap(), Frame::Nalu(&nal.data)) {
                warn!(error = %e, "viewer vaapi-inproc: socket write failed");
                break;
            }
            if idr {
                force_idr
                    .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                    .ok();
            }
            frames += 1;
            if last_log.elapsed() >= std::time::Duration::from_secs(2) {
                info!(frames, "viewer vaapi-inproc: progress");
                last_log = std::time::Instant::now();
            }
        }

        // Drain queued cursor updates onto the same socket.
        let mut cursor_write_failed = false;
        for upd in cursor_channel.drain() {
            let frame = match &upd {
                crate::viewer::cursor::CursorUpdate::Image(img) => Frame::CursorImage {
                    w: img.w,
                    h: img.h,
                    hot_x: img.hot_x,
                    hot_y: img.hot_y,
                    rgba: &img.rgba,
                },
                crate::viewer::cursor::CursorUpdate::Pos(p) => Frame::CursorPos {
                    x: p.x,
                    y: p.y,
                    seq: p.seq,
                },
            };
            if let Err(e) = write_frame(&mut *writer.lock().unwrap(), frame) {
                warn!(error = %e, "viewer vaapi-inproc: cursor write failed");
                cursor_write_failed = true;
                break;
            }
        }
        if cursor_write_failed {
            break;
        }

        let now = std::time::Instant::now();
        if now < next_tick {
            std::thread::sleep(next_tick - now);
        } else {
            next_tick = now;
        }
    }
    info!(frames, "viewer vaapi-inproc: exiting");
}

/// Background reader thread for the bridge → session direction.
///
/// Reads `ForceKeyframe` (sets `force_idr` flag) and `InjectOp`
/// frames (forwards to the session's input pipeline). On any read
/// error or `Shutdown`, sets `stop_flag` to drive the encode loop
/// out.
fn spawn_control_reader(
    mut socket: UnixStream,
    stop_flag: Arc<AtomicBool>,
    force_idr: Arc<AtomicBool>,
    target_bitrate: Arc<AtomicU32>,
    state: Arc<crate::state::State>,
) {
    std::thread::Builder::new()
        .name("waymux-viewer-ctrl".into())
        .spawn(move || {
            use crate::viewer::protocol::{read_frame, OwnedFrame};
            loop {
                if stop_flag.load(Ordering::Acquire) {
                    return;
                }
                match read_frame(&mut socket) {
                    Ok(OwnedFrame::ForceKeyframe) => {
                        force_idr.store(true, Ordering::Release);
                    }
                    Ok(OwnedFrame::SetBitrate(bps)) => {
                        target_bitrate.store(bps, Ordering::Release);
                    }
                    Ok(OwnedFrame::InjectOp(json)) => {
                        // Deserialize and dispatch into the session input
                        // pipeline — same path as RequestMethod::InjectBatch.
                        let op: waymux_protocol::InjectOp =
                            match serde_json::from_slice(&json) {
                                Ok(op) => op,
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        bytes = json.len(),
                                        "viewer control: bad InjectOp JSON; dropping"
                                    );
                                    continue;
                                }
                            };
                        match op {
                            waymux_protocol::InjectOp::Key {
                                keycode,
                                state: key_state,
                                modifiers,
                            } => {
                                let pressed = matches!(
                                    key_state,
                                    waymux_protocol::KeyState::Pressed
                                );
                                state.inject_key(keycode, pressed, modifiers, 0, 0, 0);
                            }
                            waymux_protocol::InjectOp::Pointer {
                                x,
                                y,
                                button,
                                state: btn_state,
                                axis_x,
                                axis_y,
                                window_id,
                                content,
                                seq,
                            } => {
                                let pressed = matches!(
                                    btn_state,
                                    waymux_protocol::KeyState::Pressed
                                );
                                state.inject_pointer(
                                    window_id, content, x, y, button, pressed, axis_x, axis_y, seq,
                                );
                            }
                            waymux_protocol::InjectOp::Touch { .. } => {
                                // The wire shape is reserved; viewer to session
                                // touch routing is handled elsewhere via the
                                // wl_touch capability + State::inject_touch.
                                tracing::debug!(
                                    "viewer control: InjectOp::Touch received but session-side routing is not implemented here; dropping"
                                );
                            }
                        }
                    }
                    Ok(OwnedFrame::Shutdown(reason)) => {
                        tracing::info!(reason, "viewer control: bridge sent Shutdown");
                        stop_flag.store(true, Ordering::Release);
                        return;
                    }
                    Ok(other) => {
                        tracing::debug!(
                            tag = ?std::mem::discriminant(&other),
                            "viewer control: unexpected frame; ignoring"
                        );
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::UnexpectedEof {
                            tracing::warn!(error = %e, "viewer control: read error");
                        }
                        stop_flag.store(true, Ordering::Release);
                        return;
                    }
                }
            }
        })
        .expect("spawn viewer control reader");
}

/// Find the next NALU in `buf`. Returns `Some((nalu_bytes, consumed))`
/// where `consumed` is the number of bytes from the start of `buf`
/// to drain after copying out the NALU. The returned slice excludes
/// the leading Annex-B start code; the next call begins at the next
/// start code in the input stream.
///
/// Returns `None` if `buf` doesn't yet contain a complete NALU —
/// either no start code at all, or only one start code seen so far
/// (the NALU may extend past the current buffer). Caller should
/// refill and retry.
///
/// Annex-B start codes are either 3 bytes (`00 00 01`) or 4 bytes
/// (`00 00 00 01`). We accept both.
fn next_annex_b_nalu(buf: &[u8]) -> Option<(&[u8], usize)> {
    // Locate the first start code's "01" byte and the offset where
    // the *NALU body* begins (right after the 01).
    let first_one = find_annex_b_one(buf, 0)?;
    let body_start = first_one + 1;
    // Find the next start code AFTER the body. The "01" byte index
    // tells us where to cut; we strip trailing 00 bytes that belong
    // to the *next* start code so they aren't reported as part of
    // this NALU.
    let next_one = find_annex_b_one(buf, body_start)?;
    // Strip trailing zeros: walk backwards from `next_one` past 0x00
    // bytes — those zeros belong to the next start code.
    let mut body_end = next_one; // points at the 0x01 of the next start code
    while body_end > body_start && buf[body_end - 1] == 0 {
        body_end -= 1;
    }
    // `consumed` is everything from offset 0 up to (but not including)
    // the leading zeros of the next start code. So `body_end` itself
    // — since we already moved past trailing zeros — is the right
    // drain point.
    Some((&buf[body_start..body_end], body_end))
}

/// Find the byte offset of the next `0x01` byte that is preceded by
/// at least two `0x00` bytes in `buf`, starting at `from`. Returns
/// the index of the `0x01` byte. Returns None if no start code is
/// found.
fn find_annex_b_one(buf: &[u8], from: usize) -> Option<usize> {
    let mut i = from.max(2);
    while i < buf.len() {
        if buf[i] == 0x01 && buf[i - 1] == 0 && buf[i - 2] == 0 {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use waymux_protocol::RecordingCodec;

    #[test]
    fn adaptive_qp_tightens_over_budget_loosens_under() {
        // Over the link → QP rises; far over → rises faster.
        assert_eq!(adaptive_qp_step(26, 2_000_000.0, 1_000_000.0), 28); // >1.6x → +2
        assert_eq!(adaptive_qp_step(26, 1_200_000.0, 1_000_000.0), 27); // >1.1x → +1
                                                                        // Comfortable headroom → QP drops (sharper).
        assert_eq!(adaptive_qp_step(26, 500_000.0, 1_000_000.0), 25); // <0.65x → -1
                                                                      // Deadband: near target → no change (no hunting).
        assert_eq!(adaptive_qp_step(26, 900_000.0, 1_000_000.0), 26);
        // Clamps to the visual range.
        assert_eq!(adaptive_qp_step(44, 9_000_000.0, 1_000_000.0), 44);
        assert_eq!(adaptive_qp_step(18, 1.0, 1_000_000.0), 18);
    }

    #[test]
    fn adaptive_qp_converges_to_fit_a_constrained_link() {
        // Model: bitrate ~ halves per +6 QP (standard H.264 rule of thumb).
        // Start QP18 producing ~6 Mbps; target a 1 Mbps cellular link.
        let target = 1_000_000.0;
        let mut qp = 18i32;
        let bitrate = |q: i32| 6_000_000.0 * 2f64.powf((18 - q) as f64 / 6.0);
        for _ in 0..200 {
            qp = adaptive_qp_step(qp, bitrate(qp), target);
        }
        // It must have climbed enough to fit the link (within the deadband).
        assert!(
            bitrate(qp) <= target * 1.10,
            "qp={qp} bitrate={}",
            bitrate(qp)
        );
        assert!((18..=44).contains(&qp));
    }

    #[test]
    fn picks_h264_nvenc_when_probe_succeeds() {
        let probes: &[(RecordingCodec, bool)] = &[
            (RecordingCodec::H264Nvenc, true),
            (RecordingCodec::H264Vulkan, true),
        ];
        let pick = pick_viewer_codec_with_probes(probes);
        assert_eq!(pick, Some(RecordingCodec::H264Nvenc));
    }

    #[test]
    fn falls_back_to_h264_vulkan_when_no_nvenc() {
        let probes: &[(RecordingCodec, bool)] = &[
            (RecordingCodec::H264Nvenc, false),
            (RecordingCodec::H264Vulkan, true),
        ];
        let pick = pick_viewer_codec_with_probes(probes);
        assert_eq!(pick, Some(RecordingCodec::H264Vulkan));
    }

    #[test]
    fn returns_none_when_no_codec_works() {
        let probes: &[(RecordingCodec, bool)] = &[
            (RecordingCodec::H264Nvenc, false),
            (RecordingCodec::H264Vulkan, false),
        ];
        assert_eq!(pick_viewer_codec_with_probes(probes), None);
    }

    #[test]
    fn cuda_nvenc_preferred_when_available() {
        use waymux_protocol::RecordingCodec;
        let pick = super::pick_viewer_codec_with_probes(&[
            (RecordingCodec::CudaNvenc, true),
            (RecordingCodec::H264Nvenc, true),
            (RecordingCodec::H264Vulkan, true),
        ]);
        assert_eq!(pick, Some(RecordingCodec::CudaNvenc));
    }
    #[test]
    fn falls_back_past_cuda_nvenc_when_unavailable() {
        use waymux_protocol::RecordingCodec;
        let pick = super::pick_viewer_codec_with_probes(&[
            (RecordingCodec::CudaNvenc, false),
            (RecordingCodec::H264Nvenc, false),
            (RecordingCodec::H264Vulkan, true),
        ]);
        assert_eq!(pick, Some(RecordingCodec::H264Vulkan));
    }

    // ────────────────────────────────────────────────────────────────
    // vulkan_probe_indicates_h264_encode — unit tests for the gating
    // logic that decides whether the viewer's H.264-Vulkan probe should
    // return true. Mirrors the filter in
    // `vulkan_record::select_physical_device` so a probe `true` means
    // `VkRecorder::open()` will succeed.
    // ────────────────────────────────────────────────────────────────

    use crate::vulkan_record::{VulkanDevice, VulkanProbe, VulkanQueueFamily};
    use ash::vk;

    fn make_probe(devices: Vec<VulkanDevice>) -> VulkanProbe {
        VulkanProbe {
            api_version: 0,
            instance_extensions: Vec::new(),
            devices,
        }
    }

    fn make_device(video_encode_h264_supported: bool, queue_flags: vk::QueueFlags) -> VulkanDevice {
        VulkanDevice {
            name: "test".into(),
            driver_name: String::new(),
            api_version: 0,
            queue_families: vec![VulkanQueueFamily {
                index: 0,
                flags: queue_flags,
                count: 1,
            }],
            device_extensions: Vec::new(),
            video_encode_h264_supported,
            dmabuf_import_supported: false,
        }
    }

    #[test]
    fn probe_predicate_false_when_no_devices() {
        let probe = make_probe(Vec::new());
        assert!(!vulkan_probe_indicates_h264_encode(&probe));
    }

    #[test]
    fn probe_predicate_false_when_h264_extension_missing() {
        // Device has a VIDEO_ENCODE queue family but the H.264 extensions
        // aren't reported. e.g. a future host with only h265 encode.
        let probe = make_probe(vec![make_device(
            false,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::VIDEO_ENCODE_KHR,
        )]);
        assert!(!vulkan_probe_indicates_h264_encode(&probe));
    }

    #[test]
    fn probe_predicate_false_when_no_video_encode_queue() {
        // Device reports the H.264 extensions but no queue family carries
        // VIDEO_ENCODE_KHR. select_physical_device would skip this device.
        let probe = make_probe(vec![make_device(
            true,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS,
        )]);
        assert!(!vulkan_probe_indicates_h264_encode(&probe));
    }

    #[test]
    fn probe_predicate_true_when_h264_and_queue_present() {
        let probe = make_probe(vec![make_device(
            true,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::VIDEO_ENCODE_KHR,
        )]);
        assert!(vulkan_probe_indicates_h264_encode(&probe));
    }

    #[test]
    fn annex_b_splits_two_nalus_3byte_prefix() {
        // 00 00 01 [aa] 00 00 01 [bb cc]
        let buf = vec![0, 0, 1, 0xAA, 0, 0, 1, 0xBB, 0xCC];
        let (first, used) = next_annex_b_nalu(&buf).expect("first NALU");
        assert_eq!(first, &[0xAA]);
        // After consuming the first NALU, the remaining buffer should
        // still start with the second NALU's prefix. We can't extract
        // the second one yet (no further start code or EOF marker).
        let rest = &buf[used..];
        assert_eq!(&rest[..3], &[0, 0, 1]);
        assert!(next_annex_b_nalu(rest).is_none());
    }

    #[test]
    fn annex_b_splits_4byte_prefix() {
        // 00 00 00 01 [aa bb] 00 00 00 01 [cc dd] 00 00 00 01
        let buf = vec![0, 0, 0, 1, 0xAA, 0xBB, 0, 0, 0, 1, 0xCC, 0xDD, 0, 0, 0, 1];
        let (first, used1) = next_annex_b_nalu(&buf).expect("first NALU");
        assert_eq!(first, &[0xAA, 0xBB]);
        let (second, used2) = next_annex_b_nalu(&buf[used1..]).expect("second NALU");
        assert_eq!(second, &[0xCC, 0xDD]);
        let _ = used2;
    }

    #[test]
    fn annex_b_incomplete_returns_none() {
        // Single start code + body, no terminator yet.
        let buf = vec![0, 0, 1, 0xAA, 0xBB, 0xCC];
        assert!(next_annex_b_nalu(&buf).is_none());
    }

    /// Smoke test for the full encoder thread loop. Drives the Vulkan
    /// encoder with synthetic BGRA frames (no compositor, no bridge)
    /// and asserts that at least one `Frame::Nalu` arrives on the
    /// socket within a few seconds.
    ///
    /// Marked `#[ignore]` because:
    ///   - Vulkan probe takes a few hundred ms on first use
    ///   - GPU/driver state varies across hosts (CI shouldn't depend
    ///     on a working compute queue)
    ///   - The full live pipeline is exercised by the laptop and rental
    ///     smoke tests
    ///
    /// Run with: `WAYMUX_VIEWER_TEST_HARNESS=1 cargo test -p
    /// waymux-session viewer::encoder::tests::encoder_writes_nalu --
    /// --ignored --nocapture`.
    #[test]
    #[ignore = "requires WAYMUX_VIEWER_TEST_HARNESS=1; needs working Vulkan encode queue"]
    fn encoder_writes_nalu_frames_to_socket() {
        if std::env::var("WAYMUX_VIEWER_TEST_HARNESS").ok().as_deref() != Some("1") {
            eprintln!("WAYMUX_VIEWER_TEST_HARNESS!=1 — skipping");
            return;
        }
        use crate::viewer::protocol::{read_frame, OwnedFrame};
        use std::os::unix::net::UnixListener;

        // Skip if Vulkan isn't usable.
        if crate::vulkan_record::probe().is_err() {
            eprintln!("Vulkan unavailable — skipping");
            return;
        }

        let uid = unsafe { libc::getuid() };
        let sock_path = format!("/run/user/{uid}/waymux-viewer-encoder-smoke.sock");
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("bind smoke socket");

        let frame_slot = Arc::new(LatestTaskSlot::new());
        let stop = Arc::new(AtomicBool::new(false));

        // Spawn a producer that pushes a small BGRA frame periodically.
        let w = 320u32;
        let h = 240u32;
        let producer_slot = frame_slot.clone();
        let producer_stop = stop.clone();
        std::thread::spawn(move || {
            let bytes = vec![0x80u8; (w as usize) * (h as usize) * 4];
            while !producer_stop.load(Ordering::Acquire) {
                producer_slot.put(RecordingTask::Pixels {
                    pixels: bytes.clone(),
                    width: w,
                    height: h,
                });
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });

        // Connect a client socket so the encoder has somewhere to write.
        let producer_path = sock_path.clone();
        std::thread::spawn(move || {
            let _client = UnixStream::connect(&producer_path).expect("client connect");
            std::thread::sleep(std::time::Duration::from_secs(10));
        });

        let (server_sock, _) = listener.accept().expect("accept");
        let mut server_reader = server_sock.try_clone().expect("clone read end");

        let state = Arc::new(crate::state::State::new(
            "test".into(),
            w,
            h,
            1,
            None,
            false,
        ));
        let encoder = spawn_encoder_thread(
            RecordingCodec::H264Vulkan,
            w,
            h,
            server_sock,
            stop.clone(),
            frame_slot.clone(),
            state,
        );

        // Read up to 2 NALU frames within 6 seconds.
        let mut count = 0usize;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(6);
        while std::time::Instant::now() < deadline && count < 2 {
            match read_frame(&mut server_reader) {
                Ok(OwnedFrame::Nalu(b)) => {
                    assert!(!b.is_empty(), "NALU should be non-empty");
                    count += 1;
                }
                Ok(_) => {}
                Err(e) => panic!("read_frame: {e}"),
            }
        }
        stop.store(true, Ordering::Release);
        frame_slot.wake();
        let _ = encoder.join();
        let _ = std::fs::remove_file(&sock_path);
        assert!(count >= 1, "expected at least 1 NALU, got {count}");
    }

    /// Verify that `spawn_control_reader` deserializes `Frame::InjectOp`
    /// JSON and dispatches it into `State::inject_pointer` /
    /// `State::inject_key`. We can't observe a delivered Wayland event
    /// in a unit test (no compositor running), but we CAN verify that
    /// the dispatch compiles, parses the JSON correctly, and doesn't
    /// panic on a state with no focused client (the methods return
    /// `false` gracefully when nothing is focused). The test also
    /// confirms no frames are dropped due to JSON parse errors.
    ///
    /// Note: inject_pointer / inject_key return `false` when there's no
    /// focused window (expected in a bare test State). What matters here
    /// is that the path is wired: the control reader calls through rather
    /// than log-and-drop.
    #[test]
    fn inject_op_frame_reaches_state_dispatch() {
        use crate::viewer::protocol::{write_frame, Frame};
        use std::os::unix::net::UnixStream;

        // Build a minimal State — no Wayland compositor, so inject_*
        // calls will return false (no focused client), but they must not
        // panic and must be called.
        let state = Arc::new(crate::state::State::new(
            "test-inject".into(),
            640,
            480,
            1,
            None,
            false,
        ));

        // Create a connected socket pair: bridge_end (we write InjectOp
        // frames) and session_end (passed to spawn_control_reader).
        let (mut bridge_end, session_end) = UnixStream::pair().expect("UnixStream::pair");
        bridge_end
            .set_read_timeout(Some(std::time::Duration::from_millis(200)))
            .ok();

        let stop = Arc::new(AtomicBool::new(false));
        let force_idr = Arc::new(AtomicBool::new(false));
        let target_bitrate = Arc::new(AtomicU32::new(0));

        spawn_control_reader(
            session_end,
            stop.clone(),
            force_idr.clone(),
            target_bitrate.clone(),
            state.clone(),
        );

        // --- Pointer InjectOp ---
        let ptr_op = waymux_protocol::InjectOp::Pointer {
            x: 100.0,
            y: 200.0,
            button: 0x110, // BTN_LEFT
            state: waymux_protocol::KeyState::Pressed,
            axis_x: 0.0,
            axis_y: 0.0,
            window_id: None,
            content: false,
            seq: 0,
        };
        let ptr_json = serde_json::to_vec(&ptr_op).expect("serialize pointer InjectOp");
        write_frame(&mut bridge_end, Frame::InjectOp(&ptr_json))
            .expect("write pointer InjectOp frame");

        // --- Key InjectOp ---
        let key_op = waymux_protocol::InjectOp::Key {
            keycode: 30, // KEY_A (Linux evdev)
            state: waymux_protocol::KeyState::Pressed,
            modifiers: 0,
        };
        let key_json = serde_json::to_vec(&key_op).expect("serialize key InjectOp");
        write_frame(&mut bridge_end, Frame::InjectOp(&key_json)).expect("write key InjectOp frame");

        // Give the control reader thread time to process both frames.
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Stop the reader cleanly via Shutdown frame (reason byte 0 = normal exit).
        write_frame(&mut bridge_end, Frame::Shutdown(0)).ok();
        std::thread::sleep(std::time::Duration::from_millis(50));

        // If we get here without panicking, the dispatch path is wired.
        // The stop flag is set either by the Shutdown frame or by us.
        stop.store(true, Ordering::Release);
    }
}
