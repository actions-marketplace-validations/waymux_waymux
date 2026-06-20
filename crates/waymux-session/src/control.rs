// SPDX-License-Identifier: Apache-2.0

//! Session-control socket server.
//!
//! Simpler wire protocol than the top-level daemon socket: a per-session
//! control socket with one request per connection, no hello handshake, and
//! no events.

use crate::state::State;
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Notify;
use tracing::{debug, warn};
use waymux_protocol::{
    decode_frame, encode_frame, SessionCtlInfo, SessionCtlMethod, SessionCtlRecordStarted,
    SessionCtlRequest, SessionCtlResponse, SessionCtlScreenshot, SessionCtlWindows,
};

use wayland_server::protocol::wl_shm::Format as ShmFormat;
use wayland_server::WEnum;

pub async fn run(listener: UnixListener, state: Arc<State>, shutdown: Arc<Notify>) {
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "control accept failed");
                continue;
            }
        };
        let state = state.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(stream, state, shutdown).await {
                debug!(error = %e, "control connection closed with error");
            }
        });
    }
}

async fn handle(mut stream: UnixStream, state: Arc<State>, shutdown: Arc<Notify>) -> Result<()> {
    // Audit C5: SO_PEERCRED uid check. The session control socket lives at
    // $XDG_RUNTIME_DIR/waymux/<name>/control.sock and the pre-fix code
    // accepted any connection. Methods include Shutdown, RecordStart (path
    // traversal!), InjectKey/Pointer, Screenshot — any local process that
    // can reach the path could drive the session. Mirror the daemon main
    // socket's same-uid gate.
    let my_uid = unsafe { libc::getuid() };
    match stream.peer_cred() {
        Ok(cred) if cred.uid() == my_uid => {}
        Ok(cred) => {
            warn!(
                uid = cred.uid(),
                "control socket: rejected non-matching uid"
            );
            return Ok(());
        }
        Err(e) => {
            warn!(error = %e, "control socket: peer_cred failed; closing");
            return Ok(());
        }
    }
    // One request, one response. Loop to allow batched clients eventually,
    // but daemon currently opens a fresh connection per RPC.
    loop {
        let frame = match read_frame(&mut stream).await? {
            Some(f) => f,
            None => return Ok(()),
        };
        let req: SessionCtlRequest = decode_frame(&frame).context("decode session-ctl request")?;
        let resp = dispatch(req, &state, &shutdown);
        let mut buf = Vec::with_capacity(128);
        encode_frame(&resp, &mut buf)?;
        stream
            .write_all(&buf)
            .await
            .context("write session-ctl response")?;
    }
}

pub fn dispatch(
    req: SessionCtlRequest,
    state: &Arc<State>,
    shutdown: &Notify,
) -> SessionCtlResponse {
    match req.method {
        SessionCtlMethod::Info => {
            let (w, h, s) = state.snapshot();
            let info = SessionCtlInfo {
                width: w,
                height: h,
                scale: s,
                window_count: state.window_count(),
                last_damage_ns: state.last_damage_ns(),
            };
            SessionCtlResponse::success(req.id, &info)
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
        SessionCtlMethod::ListWindows => {
            let windows = SessionCtlWindows {
                windows: state.windows(),
            };
            SessionCtlResponse::success(req.id, &windows)
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
        SessionCtlMethod::Resize { width, height } => {
            // `State::resize` updates the logical size and propagates it to
            // inner clients: it re-sends the wl_output `mode` (+ `done`) to
            // every bound output and an xdg_toplevel/xdg_surface `configure`
            // to every mapped toplevel, then pokes the compositor wake_fd to
            // flush. Same cross-thread mechanism as inject_* (this runs on the
            // tokio control thread, not the compositor thread).
            state.resize(width, height);
            SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
        SessionCtlMethod::InjectKey {
            keycode,
            state: key_state,
            modifiers,
        } => {
            let pressed = matches!(key_state, waymux_protocol::KeyState::Pressed);
            let delivered = state.inject_key(keycode, pressed, modifiers, 0, 0, 0);
            if delivered {
                SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                    .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
            } else {
                SessionCtlResponse::failure(
                    req.id,
                    "no focused client with a wl_keyboard to deliver the event to",
                )
            }
        }
        SessionCtlMethod::InjectPointer {
            x,
            y,
            button,
            state: btn_state,
            axis_x,
            axis_y,
            window_id,
            content,
        } => {
            let pressed = matches!(btn_state, waymux_protocol::KeyState::Pressed);
            let delivered =
                state.inject_pointer(window_id, content, x, y, button, pressed, axis_x, axis_y, 0);
            if delivered {
                SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                    .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
            } else {
                SessionCtlResponse::failure(
                    req.id,
                    "no focused client with a wl_pointer to deliver the event to",
                )
            }
        }
        SessionCtlMethod::InjectTouch {
            id,
            x,
            y,
            phase,
            window_id,
            content,
        } => {
            // Routes through `State::inject_touch`, which mirrors
            // `inject_pointer` (resolves window_id → client, applies content
            // inset + scale, emits wl_touch.{down|motion|up} + frame).
            let delivered = state.inject_touch(window_id, content, id, x, y, phase);
            if delivered {
                SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                    .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
            } else {
                SessionCtlResponse::failure(
                    req.id,
                    "no target client with a wl_touch to deliver the event to",
                )
            }
        }
        SessionCtlMethod::InjectBatch { ops } => {
            // Audit H10: dispatch a list of ops in order. We tolerate
            // individual ops that find no focused client (e.g. the focused
            // surface changes mid-batch); they're counted as not delivered
            // but don't fail the whole batch. Returns success if at least
            // one op was delivered, or success with zero ops when the batch
            // is empty (no-op call).
            let mut delivered_count = 0usize;
            for op in ops {
                match op {
                    waymux_protocol::InjectOp::Key {
                        keycode,
                        state: key_state,
                        modifiers,
                    } => {
                        let pressed = matches!(key_state, waymux_protocol::KeyState::Pressed);
                        if state.inject_key(keycode, pressed, modifiers, 0, 0, 0) {
                            delivered_count += 1;
                        }
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
                        let pressed = matches!(btn_state, waymux_protocol::KeyState::Pressed);
                        if state.inject_pointer(
                            window_id, content, x, y, button, pressed, axis_x, axis_y, seq,
                        ) {
                            delivered_count += 1;
                        }
                    }
                    waymux_protocol::InjectOp::Touch {
                        id,
                        x,
                        y,
                        phase,
                        window_id,
                        content,
                    } => {
                        // Real routing through `State::inject_touch`.
                        if state.inject_touch(window_id, content, id, x, y, phase) {
                            delivered_count += 1;
                        }
                    }
                }
            }
            tracing::debug!(delivered_count, "inject_batch dispatched");
            SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
        SessionCtlMethod::Screenshot { window_id } => match state.capture_window(window_id) {
            None => SessionCtlResponse::failure(
                req.id,
                format!("no committed buffer for window {}", window_id),
            ),
            Some((pixels, width, height, format, stride)) => {
                match encode_png(&pixels, width, height, stride, format) {
                    Ok(png) => {
                        let s = SessionCtlScreenshot {
                            width: width as u32,
                            height: height as u32,
                            png,
                        };
                        SessionCtlResponse::success(req.id, &s)
                            .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
                    }
                    Err(e) => SessionCtlResponse::failure(req.id, format!("encode png: {e}")),
                }
            }
        },
        SessionCtlMethod::ScreenshotDesktop => match state.capture_desktop() {
            None => {
                SessionCtlResponse::failure(req.id, "no windows available to composite".to_string())
            }
            Some((pixels, width, height, format, stride)) => {
                match encode_png(&pixels, width, height, stride, format) {
                    Ok(png) => {
                        let s = SessionCtlScreenshot {
                            width: width as u32,
                            height: height as u32,
                            png,
                        };
                        SessionCtlResponse::success(req.id, &s)
                            .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
                    }
                    Err(e) => SessionCtlResponse::failure(req.id, format!("encode png: {e}")),
                }
            }
        },
        SessionCtlMethod::Shutdown => {
            // Stop any active recording before shutdown so ffmpeg can finalize
            // the MKV. Setting stop + waking the slot's condvar makes the
            // recording thread exit promptly instead of blocking on a 200 ms
            // heartbeat.
            if let Some(dual) = state.recording.lock().unwrap().take() {
                dual.stop_both();
                drop(dual);
            }
            // Stop any active viewer: signal the encoder thread and reap the
            // bridge child so it doesn't outlive the session as an orphan.
            if let Some(mut h) = state.viewer.lock().unwrap().take() {
                h.stop_flag
                    .store(true, std::sync::atomic::Ordering::Release);
                drop(h.control_sock.take());
                kill_viewer_child(h.child.take());
                // Don't join the encoder thread on the Shutdown path to avoid
                // blocking the Wayland event loop; the thread will exit when
                // the stop flag and socket closure propagate.
            }
            shutdown.notify_waiters();
            SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
        SessionCtlMethod::RecordStart {
            path,
            codec,
            secondary_codec,
            mode,
            min_fps,
        } => {
            let codec = codec.unwrap_or_default();
            // Map the protocol-level CaptureMode to the recording crate's
            // local mode enum (no shared crate yet to host both).
            let mode = match mode.unwrap_or_default() {
                waymux_protocol::CaptureMode::FocusedWindow => {
                    crate::recording::CaptureMode::FocusedWindow
                }
                waymux_protocol::CaptureMode::WholeDesktop => {
                    crate::recording::CaptureMode::WholeDesktop
                }
            };
            let primary_path = match path {
                Some(p) => p,
                None => match crate::recording::default_recording_path(&state.name) {
                    Ok(p) => p,
                    Err(e) => return SessionCtlResponse::failure(req.id, e),
                },
            };
            // For dual recordings, derive the secondary output path by
            // inserting `.secondary` before the file extension. e.g.
            // `out.mkv` → `out.secondary.mkv`. Falls back to appending if
            // the path has no extension.
            let secondary_path = secondary_codec.map(|_| {
                let p = std::path::Path::new(&primary_path);
                match (p.parent(), p.file_stem(), p.extension()) {
                    (Some(parent), Some(stem), Some(ext)) => parent
                        .join(format!(
                            "{}.secondary.{}",
                            stem.to_string_lossy(),
                            ext.to_string_lossy()
                        ))
                        .to_string_lossy()
                        .into_owned(),
                    _ => format!("{primary_path}.secondary"),
                }
            });
            // Audit H2: defense-in-depth validation. C5's same-uid gate
            // already restricts callers, but defend against `..` traversal
            // and writes outside the allowed directory even from a
            // compromised same-uid process. Validate BOTH paths up front.
            let validate = |path: &str| -> Result<(), String> {
                use std::path::{Component, Path};
                let p = Path::new(path);
                if !p.is_absolute() {
                    return Err("recording path must be absolute".to_string());
                }
                if p.components().any(|c| matches!(c, Component::ParentDir)) {
                    return Err("recording path must not contain '..'".to_string());
                }
                let allowed = crate::recording::recordings_dir()?;
                if !p.starts_with(&allowed) {
                    return Err(format!(
                        "recording path must be inside {}",
                        allowed.display()
                    ));
                }
                Ok(())
            };
            if let Err(e) = validate(&primary_path) {
                return SessionCtlResponse::failure(req.id, e);
            }
            if let Some(sp) = &secondary_path {
                if let Err(e) = validate(sp) {
                    return SessionCtlResponse::failure(req.id, e);
                }
            }
            // Probe ffmpeg BEFORE acquiring the recording lock so blocking I/O
            // (subprocess fork+exec+wait) doesn't hold the mutex. Also probe
            // BOTH requested encoders so a misconfigured codec choice fails
            // fast at start time with a clear error, instead of silently
            // ffmpeg-aborting after the first frame arrives.
            if let Err(e) = crate::recording::probe_ffmpeg() {
                return SessionCtlResponse::failure(req.id, e);
            }
            if let Err(e) = crate::recording::probe_codec(codec) {
                return SessionCtlResponse::failure(req.id, e);
            }
            if let Some(sc) = secondary_codec {
                if let Err(e) = crate::recording::probe_codec(sc) {
                    return SessionCtlResponse::failure(req.id, format!("secondary codec: {e}"));
                }
            }
            // Hold the lock across check + spawn + store to prevent a TOCTOU
            // race where two concurrent RecordStart RPCs both observe no active
            // recording and each spawn their own ffmpeg process.
            let mut guard = state.recording.lock().unwrap();
            if guard.is_some() {
                return SessionCtlResponse::failure(req.id, "already recording".to_string());
            }
            let primary = match crate::recording::spawn_recording_thread(
                state.clone(),
                primary_path.clone(),
                codec,
                mode,
                min_fps,
            ) {
                Ok(h) => h,
                Err(e) => return SessionCtlResponse::failure(req.id, e),
            };
            let secondary = match (secondary_codec, secondary_path.clone()) {
                (Some(sc), Some(sp)) => match crate::recording::spawn_recording_thread(
                    state.clone(),
                    sp,
                    sc,
                    mode,
                    min_fps,
                ) {
                    Ok(h) => Some(h),
                    Err(e) => {
                        // Roll back the primary so we don't leak a thread on
                        // dual-encoder failure.
                        primary
                            .stop
                            .store(true, std::sync::atomic::Ordering::Release);
                        primary.slot.wake();
                        return SessionCtlResponse::failure(
                            req.id,
                            format!("secondary encoder: {e}"),
                        );
                    }
                },
                _ => None,
            };
            *guard = Some(crate::recording::DualRecordingHandle { primary, secondary });
            drop(guard);
            // Seed the first frame from the current desktop buffer so recording
            // an idle (static) desktop emits output immediately instead of
            // blocking on the next commit, which an unchanging Plasma desktop
            // may never make before the first-frame timeout. Live commits take
            // over the moment anything animates.
            state.prime_recording_first_frame();
            let started = SessionCtlRecordStarted {
                path: primary_path,
                secondary_path,
            };
            SessionCtlResponse::success(req.id, &started)
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
        SessionCtlMethod::RecordStatus => {
            // Read the active recording state under the same lock RecordStart
            // and RecordStop use. If a recording is present, report its
            // primary path/codec plus the secondary path when dual-encoding;
            // otherwise report recording=false with everything else None.
            let resp = match state.recording.lock().unwrap().as_ref() {
                Some(dual) => waymux_protocol::RecordStatusResponse {
                    recording: true,
                    path: Some(dual.primary.path.clone()),
                    secondary_path: dual.secondary.as_ref().map(|h| h.path.clone()),
                    codec: Some(codec_to_kebab(dual.primary.codec)),
                },
                None => waymux_protocol::RecordStatusResponse::default(),
            };
            SessionCtlResponse::success(req.id, &resp)
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
        SessionCtlMethod::RecordStop => {
            let handle = state.recording.lock().unwrap().take();
            match handle {
                None => SessionCtlResponse::failure(req.id, "not currently recording".to_string()),
                Some(dual) => {
                    // Release ordering pairs with Acquire on each recording
                    // thread's stop check. Waking each slot's condvar lets
                    // both threads exit immediately instead of waiting for
                    // their 200 ms heartbeat. Each thread then closes its
                    // ffmpeg's stdin and waits for ffmpeg to finalize the MKV.
                    dual.stop_both();
                    drop(dual);
                    SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                        .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
                }
            }
        }

        SessionCtlMethod::ViewerStart { bind, port } => {
            // Hold the lock for the entire spawn sequence so concurrent
            // ViewerStart RPCs can't both pass the is_some() check and
            // race on the Unix-socket bind (TOCTOU fix).
            let mut guard = state.viewer.lock().unwrap();
            if guard.is_some() {
                return SessionCtlResponse::failure(
                    req.id,
                    "viewer already active for this session".to_string(),
                );
            }

            // Pick codec — prefer NVENC, fall back to Vulkan H.264.
            let codec = match crate::viewer::encoder::select_viewer_codec() {
                Some(c) => c,
                None => {
                    return SessionCtlResponse::failure(
                        req.id,
                        "no viewer-capable encoder available (probed h264-nvenc, h264-vulkan)"
                            .to_string(),
                    )
                }
            };

            let (width, height, _scale) = state.snapshot();
            let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

            let bridge = match crate::viewer::bridge::spawn_bridge(
                &state.name,
                &bind,
                port,
                width,
                height,
                stop_flag.clone(),
            ) {
                Ok(b) => b,
                Err(e) => {
                    return SessionCtlResponse::failure(req.id, format!("bridge spawn failed: {e}"))
                }
            };

            let sock_clone = match bridge.control_sock.try_clone() {
                Ok(s) => s,
                Err(e) => {
                    // bridge.child is dropped here, which does NOT send a
                    // signal on Linux — the process outlives us. Best effort.
                    return SessionCtlResponse::failure(req.id, format!("clone socket: {e}"));
                }
            };

            // Allocate the frame slot now so the compositor tap can
            // start pushing frames before the encoder thread is ready.
            let frame_slot = std::sync::Arc::new(crate::recording::LatestTaskSlot::new());

            let encoder_thread = crate::viewer::encoder::spawn_encoder_thread(
                codec,
                width,
                height,
                sock_clone,
                stop_flag.clone(),
                frame_slot.clone(),
                state.clone(),
            );

            let url = bridge.url.clone();
            let handle = crate::viewer::ViewerHandle {
                url: bridge.url.clone(),
                // Take ownership of the Child so ViewerStop can send
                // SIGTERM → wait → SIGKILL without relying on a bare pid.
                child: Some(bridge.child),
                encoder_thread: Some(encoder_thread),
                control_sock: Some(bridge.control_sock),
                stop_flag,
                codec,
                frame_slot,
            };
            *guard = Some(handle);
            drop(guard);

            let resp = waymux_protocol::ViewerStarted { url };
            SessionCtlResponse::success(req.id, &resp)
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }

        SessionCtlMethod::ViewerStop => {
            let handle = state.viewer.lock().unwrap().take();
            if let Some(mut h) = handle {
                h.stop_flag
                    .store(true, std::sync::atomic::Ordering::Release);
                if let Some(t) = h.encoder_thread.take() {
                    let _ = t.join();
                }
                // Closing the socket signals the bridge to exit.
                drop(h.control_sock.take());
                // Reap the bridge child: SIGTERM → short wait → SIGKILL.
                kill_viewer_child(h.child.take());
            }
            SessionCtlResponse::success(req.id, &serde_unit::Unit {})
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }

        SessionCtlMethod::ViewerStatus => {
            let url = state.viewer.lock().unwrap().as_ref().map(|v| v.url.clone());
            let resp = waymux_protocol::ViewerStatusResponse { url };
            SessionCtlResponse::success(req.id, &resp)
                .unwrap_or_else(|e| SessionCtlResponse::failure(req.id, e.to_string()))
        }
    }
}

/// Render a `RecordingCodec` to its kebab-case wire string (`ffv1`,
/// `h264-nvenc`, …) for `RecordStatus`. Goes through the same serde path
/// the wire uses so the string can never drift from the enum's
/// `rename_all = "kebab-case"` representation.
fn codec_to_kebab(codec: waymux_protocol::RecordingCodec) -> String {
    let bytes = rmp_serde::to_vec_named(&codec).expect("codec enum serializes");
    rmp_serde::from_slice(&bytes).expect("codec serializes to a string")
}

/// Terminate a bridge child process: SIGTERM first, wait up to ~2 s, then
/// escalate to SIGKILL if still alive.
///
/// On Linux, `Child::drop` does NOT send any signal, so a bare drop of the
/// `Child` leaves an orphan. This function must be called explicitly from
/// ViewerStop and from the Shutdown path.
fn kill_viewer_child(child: Option<std::process::Child>) {
    let mut child = match child {
        Some(c) => c,
        None => return,
    };
    let pid = child.id() as i32;
    // SIGTERM first — give the Go bridge a chance to clean up.
    unsafe { libc::kill(pid, libc::SIGTERM) };
    // Poll for up to 2 s in 100 ms increments.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return, // exited cleanly
            Ok(None) => {}         // still running
            Err(_) => return,      // no longer our child or already reaped
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // Escalate to SIGKILL.
    let _ = child.kill(); // sends SIGKILL on Linux
    let _ = child.wait(); // reap to avoid a zombie
}

mod serde_unit {
    use serde::Serialize;
    #[derive(Serialize)]
    pub struct Unit {}
}

/// Encode the captured shm bytes as PNG. Supports ARGB8888 / XRGB8888
/// (wayland premultiplied BGRA / BGRX → we swizzle to RGBA).
fn encode_png(
    pixels: &[u8],
    width: i32,
    height: i32,
    stride: i32,
    format: WEnum<ShmFormat>,
) -> anyhow::Result<Vec<u8>> {
    let w = u32::try_from(width).map_err(|_| anyhow::anyhow!("negative width"))?;
    let h = u32::try_from(height).map_err(|_| anyhow::anyhow!("negative height"))?;
    let stride_u = usize::try_from(stride).map_err(|_| anyhow::anyhow!("negative stride"))?;
    let (has_alpha, color_type) = match format {
        WEnum::Value(ShmFormat::Argb8888) => (true, png::ColorType::Rgba),
        WEnum::Value(ShmFormat::Xrgb8888) => (false, png::ColorType::Rgb),
        other => anyhow::bail!("unsupported shm format: {:?}", other),
    };

    // Wayland's Argb8888/Xrgb8888 are little-endian 32-bit values interpreted
    // as (A<<24 | R<<16 | G<<8 | B). On little-endian memory that's bytes
    // [B, G, R, A] (or [B, G, R, X] for Xrgb). Swizzle to RGB(A) for PNG.
    let out_bpp = if has_alpha { 4 } else { 3 };
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * out_bpp);
    for y in 0..h as usize {
        let row_start = y * stride_u;
        for x in 0..w as usize {
            let p = row_start + x * 4;
            if p + 4 > pixels.len() {
                anyhow::bail!("shm buffer shorter than declared geometry");
            }
            let b = pixels[p];
            let g = pixels[p + 1];
            let r = pixels[p + 2];
            out.push(r);
            out.push(g);
            out.push(b);
            if has_alpha {
                out.push(pixels[p + 3]);
            }
        }
    }

    let mut encoded = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut encoded, w, h);
        enc.set_color(color_type);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header()?;
        writer.write_image_data(&out)?;
    }
    Ok(encoded)
}

async fn read_frame(stream: &mut UnixStream) -> Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e).context("read session-ctl frame length"),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > waymux_protocol::MAX_FRAME_SIZE {
        anyhow::bail!("session-ctl frame too large: {}", len);
    }
    let mut payload = vec![0u8; 4 + len];
    payload[..4].copy_from_slice(&len_buf);
    stream
        .read_exact(&mut payload[4..])
        .await
        .context("read session-ctl frame")?;
    Ok(Some(payload))
}
