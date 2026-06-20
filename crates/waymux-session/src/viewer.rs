// SPDX-License-Identifier: Apache-2.0

//! Browser WebRTC viewer ("neko-bridge"). Phase 2 of the dual-encoder line.
//!
//! Spawns an h264-nvenc encoder thread + a Go `waymux-neko-bridge` child
//! process per session. Encoder NALUs ship over a typed Unix socket;
//! browser input comes back over the same socket as JSON InjectOp values.

use std::os::unix::net::UnixStream;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::JoinHandle;

pub mod bridge;
pub mod cursor;
pub mod encoder;
pub mod protocol;

/// Active viewer state. One per session at any time (v1 caps at 1 viewer).
pub struct ViewerHandle {
    pub url: String,
    /// Owns the bridge child process. Keeping the `Child` here (rather than
    /// just the pid) ensures we can call `child.kill()` + `child.wait()` for
    /// proper SIGKILL escalation. The `Option` is `None` only in test stubs.
    pub child: Option<std::process::Child>,
    pub encoder_thread: Option<JoinHandle<()>>,
    pub control_sock: Option<UnixStream>,
    pub stop_flag: Arc<AtomicBool>,
    /// Codec in use for this viewer's encoder thread. Set at ViewerStart
    /// time; read by the compositor tap to decide how to shape each frame
    /// (Dmabuf vs Pixels, BGRA vs NV12) — same logic as recording slots.
    pub codec: waymux_protocol::RecordingCodec,
    /// Frame slot shared between the compositor tap (producer) and the
    /// viewer encoder thread (consumer). Same latest-only semantics as
    /// `RecordingHandle::slot`: newer frames evict older ones so a slow
    /// encode can never back-pressure the compositor.
    pub frame_slot: Arc<crate::recording::LatestTaskSlot>,
}

impl ViewerHandle {
    /// Test-only stub. Not exercised in production.
    #[cfg(test)]
    pub fn test_stub(url: String) -> Self {
        Self {
            url,
            child: None,
            encoder_thread: None,
            control_sock: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            codec: waymux_protocol::RecordingCodec::H264Nvenc,
            frame_slot: Arc::new(crate::recording::LatestTaskSlot::new()),
        }
    }
}
