// SPDX-License-Identifier: Apache-2.0

//! Request, response, and event types for the waymux control protocol.
//!
//! The wire is a single msgpack-encoded map per frame. We distinguish the
//! three kinds of frame by structural fields:
//!
//! - Request:  `{ "id": u32 != 0, "method": str, "params": map }`
//! - Response: `{ "id": u32 != 0, "ok": bool, "result"?: any, "error"?: map }`
//! - Event:    `{ "id": 0, "event": str, "params": map }`
//!
//! Keeping these as three distinct Rust types (rather than one giant untagged
//! enum) lets serde produce clean, unambiguous msgpack and keeps each side's
//! error paths explicit.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bumped on any wire-breaking change. The daemon accepts any client
/// protocol from 1 up to and including this current version, and rejects 0
/// or anything newer than it understands.
///
/// v2: InjectPointer + InjectBatch gain `window_id: Option<u32>`
/// and `content: bool` (both default-able via `#[serde(default)]`, so v1
/// clients keep working: they get focused-window targeting and buffer-local
/// coords, identical to the prior behaviour). Coord docstrings are framed as
/// logical pixels; scale=1 makes this a no-op.
///
/// v3: two additive changes, both protected by
/// `#[serde(default)]` for back-compat with v2 clients/daemons.
/// (1) `RequestMethod::InjectSelector` reserves a wire slot for future
/// semantic (CDP / AT-SPI) targeting. The daemon returns
/// `ErrorCode::NotImplemented` via the existing catch-all arm; the real
/// handler is not yet implemented.
/// (2) `WindowInfo` gains `content_rect: Option<Rect>` carrying the
/// `xdg_surface.set_window_geometry` inset stored on SurfaceData. v2 clients
/// reading a v3 daemon's response ignore the extra field; v3 clients reading
/// a v2 daemon's response see `None`.
///
/// v4: adds touch input support. `RequestMethod::InjectTouch` and
/// `InjectOp::Touch` carry single-finger tap/motion/lift events, plus a
/// `TouchPhase` enum (Down/Motion/Up). `window_id` and `content` carry
/// the same `#[serde(default)]` back-compat semantics as InjectPointer.
/// Touch is routed end to end: the session synthesizes real `wl_touch`
/// events via `State::inject_touch`.
pub const CURRENT_PROTOCOL_VERSION: u32 = 4;

// ─── Request ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub id: u32,
    #[serde(flatten)]
    pub method: RequestMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum RequestMethod {
    Hello {
        client_protocol: u32,
    },
    ListSessions,
    CreateSession(CreateSessionParams),
    DestroySession {
        name: String,
    },
    Attach {
        name: String,
    },
    Detach {
        name: String,
    },
    Resize {
        name: String,
        width: u32,
        height: u32,
    },
    Spawn {
        name: String,
        argv: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        /// Marks this child as the inner compositor (e.g. KWin). When this
        /// process exits, the daemon emits `SessionCrashed` instead of (or
        /// alongside) `ChildExited` so SDKs can react. Only one compositor
        /// child per session; later `compositor: true` spawns replace the
        /// tracked PID.
        #[serde(default)]
        compositor: bool,
    },
    ListWindows {
        name: String,
    },
    TagWindow {
        name: String,
        window_id: u32,
        tags: Vec<String>,
    },
    Screenshot {
        name: String,
        #[serde(default)]
        window_id: Option<u32>,
        #[serde(default)]
        format: ScreenshotFormat,
    },
    ScreenshotDesktop {
        name: String,
        #[serde(default)]
        format: ScreenshotFormat,
    },
    WaitForIdle {
        name: String,
        quiet_ms: u32,
        timeout_ms: u32,
    },
    Subscribe {
        topics: Vec<String>,
    },
    InjectKey {
        name: String,
        keycode: u32,
        state: KeyState,
        modifiers: u32,
    },
    /// Deliver a synthetic pointer event to the focused window's client.
    ///
    /// This is the single wire-level entry point for all pointer activity; the
    /// four logical event types an SDK typically distinguishes — `MouseMove`,
    /// `MouseDown`, `MouseUp`, `Scroll` — are all expressed by varying the
    /// fields below. Every call delivers a `wl_pointer.motion(x, y)` first
    /// (so the pointer always lands at the requested coordinates before any
    /// button or axis event), optionally followed by a `button` event when
    /// `button != 0` and/or `axis` events when `axis_x`/`axis_y` are non-zero.
    /// On `wl_pointer` v5+ a `frame()` event is appended so all sub-events in
    /// one call form a single logical input frame on the client.
    ///
    /// **`MouseMove`** — set `button = 0`, leave `axis_x`/`axis_y` at `0.0`.
    /// `state` is ignored. Coordinates are in **logical pixels** — i.e. the
    /// session's `width × height` as passed to `CreateSession`. This is the
    /// same coordinate space the SDK observes through screenshots and window
    /// geometry, and it is the compositor's logical space (the virtual output
    /// advertises `zxdg_output_v1.logical_size(width, height)`). The session
    /// `scale` affects only how many *buffer* pixels a client renders per
    /// logical pixel (`wl_surface.set_buffer_scale`); it does **not** change
    /// the coordinate space of `wl_pointer.motion`, which Wayland fixes as
    /// surface-local logical pixels. Coordinates are therefore passed through
    /// to the client unchanged at every scale (including `scale > 1`), handled
    /// engine-side in `crates/waymux-session/src/state.rs::inject_pointer`.
    ///
    /// **`MouseDown`** — set `button` to the evdev code (`BTN_LEFT = 0x110`,
    /// `BTN_RIGHT = 0x111`, `BTN_MIDDLE = 0x112`, `BTN_SIDE = 0x113`,
    /// `BTN_EXTRA = 0x114`) and `state = Pressed`. The pointer also moves to
    /// `(x, y)` first; pass the current cursor position if you don't want it
    /// to jump.
    ///
    /// **`MouseUp`** — same as `MouseDown` but `state = Released`. Clients
    /// that treat unmatched releases as "click cancelled" expect a paired
    /// `Pressed` to have arrived first; the SDK is responsible for that
    /// pairing — the daemon does not synthesize one half from the other.
    ///
    /// **`Scroll`** — set `button = 0` and provide non-zero `axis_x`
    /// (horizontal) and/or `axis_y` (vertical) discrete-pixel delta. Both
    /// axes can fire in the same call. Sign convention follows Wayland's
    /// `wl_pointer.axis`: positive `axis_y` scrolls down, positive `axis_x`
    /// scrolls right.
    ///
    /// Returns `failure` with a "no focused client with a wl_pointer"
    /// message when no surface is focused or the focused client has not
    /// bound `wl_pointer` yet.
    InjectPointer {
        name: String,
        /// Logical-pixel X coordinate (session `width` space). See the
        /// docstring above for full semantics; delivered to the client
        /// unchanged at every scale.
        x: f64,
        /// Logical-pixel Y coordinate (session `height` space). See the
        /// docstring above for full semantics; delivered to the client
        /// unchanged at every scale.
        y: f64,
        /// Evdev button code (BTN_LEFT=0x110, BTN_RIGHT=0x111, etc.). Use 0
        /// to emit only motion/axis (i.e. a `MouseMove` or `Scroll`).
        #[serde(default)]
        button: u32,
        /// Pressed/released — ignored when `button` is 0.
        #[serde(default = "default_released")]
        state: KeyState,
        /// Horizontal scroll delta in discrete pixels; zero = no horizontal
        /// axis event. Positive scrolls right.
        #[serde(default)]
        axis_x: f64,
        /// Vertical scroll delta in discrete pixels; zero = no vertical
        /// axis event. Positive scrolls down.
        #[serde(default)]
        axis_y: f64,
        /// If `Some(id)`, deliver the event to the window with that id
        /// regardless of current focus. If `None` (the default), fall back
        /// to the focused-window behaviour (the historical v1 default).
        ///
        /// `#[serde(default)]` so v1 clients talking to a v2 daemon
        /// continue to parse: the missing field decodes as `None`, which
        /// preserves today's focused-window semantics.
        #[serde(default)]
        window_id: Option<u32>,
        /// If `true`, treat `(x, y)` as **content-space** coordinates: the
        /// session subtracts the target window's `xdg_surface.set_window_geometry`
        /// offset before delivering, so a click at `(0, 0)` lands on the
        /// visible content origin even when the client draws a CSD shadow
        /// outside its content rect. If `false` (default), `(x, y)` is
        /// buffer-local (surface-local in Wayland terms).
        ///
        /// `#[serde(default)]` so v1 clients talking to a v2 daemon
        /// continue to parse: the missing field decodes as `false`, which
        /// preserves today's buffer-local semantics.
        #[serde(default)]
        content: bool,
    },
    /// Single-shot touch event injection. Mirrors `InjectPointer`'s shape
    /// (window_id + content) but routes through `wl_touch` instead of
    /// `wl_pointer`. The session emits the corresponding
    /// `wl_touch.down`/`motion`/`up` event followed by a `frame` marker
    /// on the target surface.
    ///
    /// Single-finger taps are the primary use case; multi-finger
    /// pinch/rotate is out of scope. `id` is the wl_touch tracking id: use
    /// `0` for single-finger flows. The same id must be reused across the
    /// Down -> Motion* -> Up lifecycle for a single finger.
    ///
    /// For multi-event sequences (down -> motion -> up across multiple
    /// timestamps), use `InjectBatch` with multiple `InjectOp::Touch`
    /// ops; the session emits each in order with a single `wl_touch.frame`
    /// at the end of the batch.
    ///
    /// Coords are **logical pixels** (same convention as `InjectPointer`);
    /// `content=true` subtracts the target window's
    /// `xdg_surface.set_window_geometry` CSD inset before delivering,
    /// mirroring `InjectPointer::content`.
    ///
    /// Touch is implemented end to end: the session advertises the
    /// `wl_touch` capability and routes these events through
    /// `State::inject_touch`.
    InjectTouch {
        /// Session name (matches the `name` field on `InjectPointer`).
        name: String,
        /// wl_touch tracking id — identifies one finger across the
        /// Down → Motion* → Up lifecycle. Use `0` for single-finger
        /// flows; multi-touch callers should allocate distinct ids per
        /// concurrent contact.
        id: u32,
        /// Logical-pixel X coordinate. Same semantics as
        /// `InjectPointer::x`; delivered to the client unchanged at every scale.
        x: f64,
        /// Logical-pixel Y coordinate. Same semantics as
        /// `InjectPointer::y`; delivered to the client unchanged at every scale.
        y: f64,
        /// Touch lifecycle phase: `Down` (finger touched), `Motion`
        /// (finger moved), or `Up` (finger lifted). `wl_touch.cancel`
        /// is internal to the session (e.g. grab broken) and not exposed
        /// on the wire.
        phase: TouchPhase,
        /// Per-event target window id; mirrors
        /// `RequestMethod::InjectPointer::window_id`. `None` (default)
        /// = focused window. `#[serde(default)]` keeps v3 clients
        /// parseable.
        #[serde(default)]
        window_id: Option<u32>,
        /// Treat `(x, y)` as content-space coordinates (CSD inset
        /// applied); mirrors `RequestMethod::InjectPointer::content`.
        /// `#[serde(default)]` keeps v3 clients parseable; missing
        /// means buffer-local coords.
        #[serde(default)]
        content: bool,
    },
    /// Audit H10: deliver a batch of input ops to a session in one RPC.
    /// Eliminates the per-event Unix-socket round-trip that made
    /// `type("hello")` cost 10 RTTs. Ops are dispatched in order with no
    /// inter-op delay; SDKs that need timing (e.g. `double_click`'s
    /// 50ms gap) should split into multiple `inject_batch` calls.
    InjectBatch {
        name: String,
        ops: Vec<InjectOp>,
    },
    /// Click/scroll/drag at a target resolved by selector: e.g. a CDP CSS
    /// selector for Chromium or an AT-SPI role+name for native apps.
    /// Reserved wire slot (introduced with wire v3); the real handler is
    /// not yet implemented, so the daemon's catch-all arm returns
    /// `ErrorCode::NotImplemented`.
    ///
    /// Field set mirrors `InjectPointer` minus coordinates (the selector
    /// supplies the target) and minus axis fields (scroll/drag variants
    /// may be added later if needed; this starts click-only).
    InjectSelector {
        /// Session name (matches the `name` field on `InjectPointer`).
        session: String,
        /// CDP CSS selector (for Chromium content) or AT-SPI role+name
        /// (for native GTK/Qt apps). Format is selector-engine specific
        /// and validated by the future handler.
        selector: String,
        /// Evdev button code (BTN_LEFT=0x110, etc.); same convention as
        /// `InjectPointer::button`.
        button: u32,
        /// `true` = press, `false` = release. A click-pair helper that
        /// emits both with the right inter-event delay may be added later;
        /// until then this is a single half-click event.
        pressed: bool,
        /// Optional content-space flag (mirrors `InjectPointer::content`).
        /// Engine looks up the selector-resolved window's `content_rect`
        /// and applies the inset before sending the synthetic click. If
        /// the selector doesn't resolve, the daemon returns
        /// `NotImplemented` regardless of this field.
        #[serde(default)]
        content: bool,
    },
    StreamLogs {
        name: String,
        follow: bool,
    },
    RecordStart {
        name: String,
        #[serde(default)]
        path: Option<String>,
        /// Video encoder. None == default (`Ffv1`) for backwards compat
        /// with older CLIs.
        #[serde(default)]
        codec: Option<RecordingCodec>,
        /// Optional second encoder running in parallel from the same frame
        /// source. Output goes to a derived path (`<output>.secondary.mkv`).
        /// Intended for a bit-exact archive plus a low-latency live stream
        /// driven by one compositor commit cadence. None == single-encoder
        /// behaviour.
        #[serde(default)]
        secondary_codec: Option<RecordingCodec>,
        /// Capture mode. None == default (`FocusedWindow`) for
        /// backwards compat with older CLIs. Set to `WholeDesktop` for
        /// multi-window / full-desktop recordings.
        #[serde(default)]
        mode: Option<CaptureMode>,
        /// Minimum frames-per-second pacing. None (default) = pure
        /// commit-driven: capture rate matches inner-client commits, so
        /// idle pages produce 0 fps. `Some(n)` guarantees ≥ n fps by
        /// re-encoding the most-recent captured frame when no new commit
        /// arrives within 1/n seconds. Use 60 to produce a steady-rate
        /// hero clip; leave None to keep small files for idle content.
        #[serde(default)]
        min_fps: Option<u32>,
    },
    RecordStop {
        name: String,
    },
    /// Return the current recording state for `session`. Mirror of
    /// `RequestMethod::ViewerStatus`: reports whether a recording is active
    /// and, if so, its output path(s) and primary codec.
    RecordStatus {
        name: String,
    },
    /// Open a WebRTC viewer URL for `session`. The waymux-session process
    /// spawns an h264-nvenc encoder thread + a neko-bridge child that serves
    /// HTML+WebSocket+WebRTC on `bind:<port>`. `bind` defaults to
    /// `127.0.0.1`; set to a WireGuard interface IP for rental testing.
    ViewerStart {
        session: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bind: Option<String>,
        /// Optional explicit TCP port for the bridge. `None` →
        /// pick_ephemeral_port (legacy behaviour, OK for local dev).
        /// `Some(p)` → bind that port (used by the SaaS launch script
        /// to pin bridge to 8080 so the portal Connect URL routes there
        /// instead of an unpredictable ephemeral). Caller is responsible
        /// for ensuring nothing else holds the port.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
    },
    /// Close the active viewer (if any) for `session`. Idempotent.
    ViewerStop {
        session: String,
    },
    /// Return the viewer URL for `session` (None if no viewer is active).
    ViewerStatus {
        session: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionParams {
    pub name: String,
    pub width: u32,
    pub height: u32,
    #[serde(default = "default_scale")]
    pub scale: u32,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub share_clipboard: bool,
    #[serde(default)]
    pub share_audio: bool,
    /// Optional cgroup-v2 `memory.max` cap (in MiB) applied to the session
    /// process and every PID spawned into it via `spawn_child`. `None` /
    /// `0` means no cap. Best-effort: if cgroup-v2 is unavailable or the
    /// daemon's cgroup isn't writable, the session still starts and the
    /// daemon logs a warning instead of failing.
    #[serde(default)]
    pub mem_cap_mb: Option<u32>,
    /// Optional cgroup-v2 `cpu.max` cap, expressed as a percentage of one
    /// CPU core. `200` = two full cores; `50` = half a core. `None` / `0`
    /// means no cap. Same best-effort semantics as `mem_cap_mb`.
    #[serde(default)]
    pub cpu_cap_pct: Option<u32>,
    /// Optional per-session disk quota (in MiB) for
    /// `$XDG_RUNTIME_DIR/waymux/<session>/`. Implemented as a tmpfs mount
    /// with `size=Nm`. `None` / `0` means no cap. Best-effort: requires
    /// `CAP_SYS_ADMIN` (i.e. the daemon runs as root or in a user
    /// namespace); otherwise the session falls back to the parent
    /// XDG_RUNTIME_DIR with a warning.
    #[serde(default)]
    pub disk_quota_mb: Option<u32>,
    /// Optional file-descriptor cap applied via `RLIMIT_NOFILE` to the
    /// session subprocess. `None` / `0` means inherit from the daemon.
    /// Subject to the kernel's hard limit; failures degrade with a warning.
    #[serde(default)]
    pub fd_limit: Option<u32>,
    /// Optional waymux API-key id (UUID, as returned by `POST /api-keys`)
    /// used to attribute usage to a billing account. The daemon embeds this
    /// verbatim in usage-event JSONL output so the Day 7 reporter can join
    /// without needing the plaintext key.
    #[serde(default)]
    pub api_key_id: Option<String>,
    /// Optional codec hint forwarded from the API for the recording path
    /// (e.g. `"h264"`, `"hevc-vulkan-lossless"`, `"ffv1-vulkan"`). Plan-gating
    /// happens at the API edge; the daemon currently treats this as advisory
    /// until the recording pipeline wires it through.
    #[serde(default)]
    pub codec: Option<String>,
    /// Optional GPU type hint forwarded from the API (e.g. `"a6000"`,
    /// `"l40"`). Same advisory semantics as `codec` above: gating lives at
    /// the API edge, daemon honoring is a separate downstream concern.
    #[serde(default)]
    pub gpu_type: Option<String>,
}

fn default_scale() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScreenshotFormat {
    #[default]
    Dmabuf,
    Png,
}

/// Video encoder used for `RecordStart`. Default is `Ffv1` (lossless,
/// CPU-encoded, exact-pixel determinism — preferred for visual-regression
/// CI artifacts). Hardware codecs (`H264Nvenc`, `H264Vaapi`) trade
/// determinism for ~10× smaller files and lower CPU usage; preferred for
/// marketing screencasts and customer "see what your agent did" videos.
///
/// Wire format is kebab-case: `ffv1`, `h264-nvenc`, `h264-vaapi`. Forward-
/// compatible: unknown codec strings deserialize as a SerdeError; the
/// daemon surfaces that as a typed `BadRequest`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RecordingCodec {
    /// Lossless FFV1 inside MKV. ~70 MB / minute at 1080p; CPU-encoded.
    /// The default — preserves exact pixels for visual-regression CI.
    #[default]
    Ffv1,
    /// Lossless H.264 (RGB) via `libx264rgb -qp 0 -preset ultrafast` inside
    /// MKV. CPU-encoded like `Ffv1` but much lighter per frame and inherently
    /// multithreaded, so it sustains a higher fps on constrained (few-core) CI
    /// runners, and files are smaller than FFV1. Bit-exact RGB (no chroma
    /// subsampling). The lean CPU-lossless option for headless CI recording.
    X264Lossless,
    /// H.264 via NVENC (NVIDIA hardware encoder). Lossy, ~5 MB / minute.
    /// Requires an NVIDIA GPU with NVENC + ffmpeg built `--enable-nvenc`.
    H264Nvenc,
    /// H.264 via VAAPI (AMD / Intel hardware encoder). Lossy, ~5 MB /
    /// minute. Requires a working `/dev/dri/renderD128` + libva +
    /// ffmpeg built `--enable-vaapi`.
    H264Vaapi,
    /// H.264 via Vulkan video encode + in-process Matroska muxer.
    /// Zero-copy: imports the inner-client dmabuf, runs a compute
    /// shader BGRA->NV12 on the GPU, submits to VK_KHR_video_encode_h264.
    /// No ffmpeg subprocess. Requires NVIDIA driver 535+, AMD Mesa 25+
    /// (Renoir+ / VCN 2+), or Intel Mesa 24.3+ (currently disabled in
    /// 25.3.5 — re-check upstream). Falls back to legacy paths on
    /// unsupported drivers.
    H264Vulkan,
    /// Lossless FFV1 encoded via Vulkan compute shaders (ffmpeg's
    /// `ffv1_vulkan` encoder). Mathematically lossless from BGRA
    /// source, GPU-accelerated, cross-vendor (AMD + NVIDIA). On
    /// integrated graphics this is far slower than real-time (~0.1×
    /// at 1080p on Renoir); on discrete cards it's expected to hit
    /// real-time for 4K 60 fps. Use for marketing-grade lossless
    /// recordings on rented GPU hardware.
    Ffv1Vulkan,
    /// Bit-exact lossless H.264 via the Vulkan video-encode pipeline
    /// at Hi444PP profile, QP=0. NVIDIA-only on baseline 2026-05-12
    /// — Mesa exposes only Main 4:2:0; see `feedback_amd_no_444_encode.md`.
    /// Max 4096×4096 on RTX A6000. Bytes the encoder writes are
    /// reversible: decoding the resulting NAL stream produces a
    /// pixel-exact copy of the BGRA input (modulo BT.709 colorspace
    /// rounding in the YUV conversion).
    ///
    /// **Status 2026-05-12:** the Vulkan video-encode submit fails on
    /// NVIDIA driver 560 + 580 (`VK_ERROR_INITIALIZATION_FAILED` at
    /// `vkEndCommandBuffer`) — the profile probe succeeds but the
    /// kernel-side encode shim is unwired. Use `HevcVulkanLossless`
    /// instead, which works on the same hardware.
    H264VulkanLossless,
    /// Lossless H.265 / HEVC via the `hevc_vulkan` encoder
    /// (ffmpeg 8.0) at RangeExt profile (4:4:4) with `-tune lossless
    /// -qp 0`. Validated end-to-end on NVIDIA RTX A6000 + driver
    /// 580.159.03 (2.5× real-time at 1080p). The actual NVIDIA path
    /// to lossless 4:4:4 through Vulkan video encode KHR — replaces
    /// the H.264 Hi444PP attempt for the production lossless codec.
    HevcVulkanLossless,
    /// H.264 via the NVIDIA CUDA-based NVENC path (direct CUDA driver API +
    /// libnvidia-encode). Zero ffmpeg subprocess: the encoder thread drives
    /// the NVENC hardware through `cuda_nvenc_record::CudaNvencEncoder`
    /// directly. NVIDIA GPU with NVENC required; degrades to `H264Nvenc`
    /// (ffmpeg) if `libcuda.so.1` or `libnvidia-encode.so.1` are absent.
    CudaNvenc,
}

/// What the recording captures.
///
/// `FocusedWindow` (default) follows whichever window is currently
/// focused, capturing only its surface tree. Right shape for "record
/// what my agent did" workflows where a single browser window is the
/// subject.
///
/// `WholeDesktop` captures the full inner-compositor surface set
/// composited together — every mapped toplevel, every layer surface.
/// Right shape for "show me everything that happened on this session"
/// recordings: multi-window debugging, Plasma desktop demos, etc.
///
/// Wire format is kebab-case: `focused-window`, `whole-desktop`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CaptureMode {
    #[default]
    FocusedWindow,
    WholeDesktop,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KeyState {
    Pressed,
    Released,
}

/// wl_touch lifecycle phase. One of the three wire events:
///
/// - `Down`: finger contacted the surface; corresponds to `wl_touch.down`.
/// - `Motion`: that finger moved while still in contact; corresponds to
///   `wl_touch.motion`.
/// - `Up`: finger lifted off the surface; corresponds to `wl_touch.up`.
///
/// `wl_touch.cancel` (touch sequence canceled, e.g. grab broken) is
/// internal to the session and not exposed as a phase; clients can't
/// synthesize a cancel.
///
/// Wire format is snake_case (`"down"` / `"motion"` / `"up"`); the
/// `touch_phase_serializes_snake_case` test guards against accidental
/// enum renaming.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TouchPhase {
    Down,
    Motion,
    Up,
}

/// One element of an `InjectBatch` request. See `RequestMethod::InjectBatch`
/// (audit H10). The variants carry the same field shapes as their per-call
/// counterparts so SDKs can build a batch by mapping their existing
/// `inject_key` / `inject_pointer` calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "params", rename_all = "snake_case")]
pub enum InjectOp {
    Key {
        keycode: u32,
        state: KeyState,
        #[serde(default)]
        modifiers: u32,
    },
    Pointer {
        /// Logical-pixel X coordinate. See `RequestMethod::InjectPointer`
        /// for full semantics; delivered to the client unchanged at every scale.
        x: f64,
        /// Logical-pixel Y coordinate. See `RequestMethod::InjectPointer`
        /// for full semantics; delivered to the client unchanged at every scale.
        y: f64,
        #[serde(default)]
        button: u32,
        #[serde(default = "default_released")]
        state: KeyState,
        #[serde(default)]
        axis_x: f64,
        #[serde(default)]
        axis_y: f64,
        /// Per-op target window id; mirrors `RequestMethod::InjectPointer`.
        /// `#[serde(default)]` keeps v1 batch payloads parseable; missing
        /// means "send to focused window".
        #[serde(default)]
        window_id: Option<u32>,
        /// Per-op content-space coord flag; mirrors
        /// `RequestMethod::InjectPointer`. `#[serde(default)]` keeps v1
        /// batch payloads parseable; missing means "buffer-local coords".
        #[serde(default)]
        content: bool,
        /// Monotonic browser input sequence number, echoed back to the viewer
        /// as the confirmed-position seq for the cursor-overlay latency display.
        /// `#[serde(default)]` keeps existing batch payloads (which omit seq)
        /// parseable; browser-not-yet-sending-seq also defaults to 0.
        #[serde(default)]
        seq: u32,
    },
    /// Single touch event inside a batch. Mirrors
    /// `RequestMethod::InjectTouch` minus the session `name` (implicit
    /// from the enclosing batch). Use multiple `Touch` ops in one batch
    /// to express a multi-step gesture (down -> motion -> up) atomically.
    Touch {
        /// wl_touch tracking id; see `RequestMethod::InjectTouch::id`.
        id: u32,
        /// Logical-pixel X coordinate. See `RequestMethod::InjectTouch`
        /// for full semantics.
        x: f64,
        /// Logical-pixel Y coordinate. See `RequestMethod::InjectTouch`
        /// for full semantics.
        y: f64,
        /// Touch lifecycle phase (Down / Motion / Up).
        phase: TouchPhase,
        /// Per-op target window id; mirrors
        /// `RequestMethod::InjectTouch::window_id`. `#[serde(default)]`
        /// keeps v3 batch payloads parseable.
        #[serde(default)]
        window_id: Option<u32>,
        /// Per-op content-space coord flag; mirrors
        /// `RequestMethod::InjectTouch::content`. `#[serde(default)]`
        /// keeps v3 batch payloads parseable.
        #[serde(default)]
        content: bool,
    },
}

// ─── Response ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub id: u32,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub result: Option<rmpv::Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<ResponseError>,
}

impl Response {
    /// Build a success response wrapping `value`. Serializes the payload with
    /// `to_vec_named` so struct fields appear as map keys on the wire (rather
    /// than as positional arrays, which `rmpv::ext::to_value` would produce).
    /// The resulting `rmpv::Value::Map` is then re-serialized as-is when the
    /// Response is framed, preserving named fields for non-Rust clients.
    pub fn success<T: Serialize>(id: u32, value: &T) -> Result<Self, rmp_serde::encode::Error> {
        let bytes = rmp_serde::to_vec_named(value)?;
        let result: rmpv::Value = rmp_serde::from_slice(&bytes)
            .expect("rmp_serde must round-trip bytes it just produced");
        Ok(Self {
            id,
            ok: true,
            result: Some(result),
            error: None,
        })
    }

    pub fn failure(id: u32, error: ResponseError) -> Self {
        Self {
            id,
            ok: false,
            result: None,
            error: Some(error),
        }
    }

    /// Decode a successful response's `result` into a typed value. Re-encodes
    /// the stored `rmpv::Value` to msgpack bytes and deserializes via
    /// rmp_serde, which accepts both map-with-string-keys and positional
    /// arrays for named structs.
    pub fn decode_result<T: serde::de::DeserializeOwned>(
        &self,
    ) -> Result<T, rmp_serde::decode::Error> {
        match &self.result {
            Some(v) => {
                let mut bytes = Vec::new();
                rmpv::encode::write_value(&mut bytes, v)
                    .map_err(|e| rmp_serde::decode::Error::Uncategorized(e.to_string()))?;
                rmp_serde::from_slice(&bytes)
            }
            None => Err(rmp_serde::decode::Error::Uncategorized(
                "response has no result".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub detail: Option<rmpv::Value>,
}

impl ResponseError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            detail: None,
        }
    }
}

/// 503 envelope for transient infrastructure errors that the SDK should
/// retry. `reason` is a short stable token (e.g. `"upstream_unavailable"`,
/// `"resume_timeout"`, `"vm_not_ready"`) suitable for client-side branching.
/// `retry_after_s` is an OPTIONAL suggested delay in seconds; `None` means
/// "use your default backoff."
///
/// This is the JSON body shape the SaaS api (`waymux-api`) emits on the
/// hosted SDK proxy path (`/v1/sessions/<id>/…`) when the upstream
/// per-customer VM is temporarily not reachable / not ready. The HTTP
/// status is always 503 and the body looks like:
///
/// ```json
/// {"error": "session_unavailable", "reason": "resume_timeout", "retry_after_s": 10}
/// ```
///
/// The literal `error: "session_unavailable"` tag distinguishes this from
/// the generic `{"error": "<message>"}` 503 the api crate emits for other
/// reasons (the generic `ApiError::Unavailable`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionUnavailable {
    /// Always the literal string `"session_unavailable"` on the wire.
    /// Serialized into the same `error` field every other api error uses
    /// so SDKs can distinguish this envelope by string equality before
    /// inspecting the rest of the body.
    pub error: String,
    /// Stable, short snake_case token. Current set:
    /// `"upstream_unavailable"`, `"resume_timeout"`, `"vm_not_ready"`,
    /// `"hibernated_no_backend"`. SDKs MAY branch on this; unknown
    /// reasons MUST be treated as retryable.
    pub reason: String,
    /// Suggested retry delay in seconds. `None` means "client decides"
    /// (use default backoff). The api also sets an HTTP `Retry-After: N`
    /// header when this is `Some`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub retry_after_s: Option<u32>,
}

impl SessionUnavailable {
    /// Construct with the canonical `error` tag pre-filled. The two-arg
    /// form is what call sites in `waymux-api` use; callers never set
    /// `error` directly.
    pub fn new(reason: impl Into<String>, retry_after_s: Option<u32>) -> Self {
        Self {
            error: "session_unavailable".into(),
            reason: reason.into(),
            retry_after_s,
        }
    }
}

/// Stable string error codes. New codes may be added, but clients must
/// treat unknown codes as `Internal`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorCode {
    #[serde(rename = "E_NOT_FOUND")]
    NotFound,
    #[serde(rename = "E_ALREADY_EXISTS")]
    AlreadyExists,
    #[serde(rename = "E_PROTO_VERSION")]
    ProtoVersion,
    #[serde(rename = "E_NO_RENDER_NODE")]
    NoRenderNode,
    #[serde(rename = "E_RESIZE_REJECTED")]
    ResizeRejected,
    #[serde(rename = "E_BACKPRESSURE")]
    Backpressure,
    #[serde(rename = "E_NOT_IMPLEMENTED")]
    NotImplemented,
    /// Caller-input validation failure (e.g. an empty/relative/oversized
    /// argv, an invalid session name). Distinct from `Internal`, which is
    /// reserved for genuine server faults. Additive and back-compatible:
    /// older clients fall back to `Internal` for unknown codes.
    #[serde(rename = "E_BAD_REQUEST")]
    BadRequest,
    #[serde(rename = "E_INTERNAL")]
    Internal,
    /// Fallback for unknown codes during deserialization. Clients treat
    /// unknown codes as `Internal` for retry decisions: surface this
    /// variant at that layer.
    #[serde(other)]
    Unknown,
}

// ─── Events ───────────────────────────────────────────────────────────────

/// Server-initiated frame. On the wire `id` is always 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: u32,
    #[serde(flatten)]
    pub body: EventBody,
}

impl Event {
    pub fn new(body: EventBody) -> Self {
        Self { id: 0, body }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "params", rename_all = "snake_case")]
pub enum EventBody {
    SessionCreated {
        name: String,
    },
    SessionDestroyed {
        name: String,
        exit_code: i32,
    },
    ChildExited {
        name: String,
        pid: i32,
        exit_code: i32,
    },
    /// The inner compositor (a child spawned with `compositor: true`)
    /// exited. The session's Wayland socket is now dark — SDKs typically
    /// surface this as a `SessionCrashed` exception.
    SessionCrashed {
        name: String,
        pid: i32,
        exit_code: i32,
    },
    /// Attach state changed. `occluded: true` means no outer-view client is
    /// currently connected (or the outer surface was unmapped), so the
    /// session is not driving any host surface. SDKs can use this as a
    /// signal that recordings/screenshots are still possible but latency
    /// targets no longer apply.
    Occluded {
        name: String,
        occluded: bool,
    },
    WindowCreated {
        name: String,
        window_id: u32,
        app_id: String,
        title: String,
        pid: i32,
    },
    WindowDestroyed {
        name: String,
        window_id: u32,
    },
    WindowChanged {
        name: String,
        window_id: u32,
        fields: WindowChange,
    },
    Damage {
        name: String,
        serial: u64,
        timestamp_ns: u64,
    },
    /// A line of output from the session process.
    /// `stream` is `"stdout"` or `"stderr"`.
    Log {
        name: String,
        stream: String,
        text: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowChange {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub app_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub geometry: Option<Rect>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub focused: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

// ─── Typed result structs ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloResult {
    pub server_protocol: u32,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub name: String,
    pub pid: i32,
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub attached: bool,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionResult {
    pub name: String,
    pub inner_socket_path: String,
}

/// Response shape for `attach(name)`. Rather than passing a file descriptor
/// directly, this returns a filesystem path (the attach client
/// `UnixStream::connect`s it) because msgpack framing can't carry fds without
/// a separate SCM_RIGHTS layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachResult {
    pub attach_socket_path: String,
}

// ─── Session-control protocol (daemon ↔ session process) ────────────────
//
// Simpler than the client protocol: no hello handshake, no events, one
// request per connection. The daemon opens a fresh connection to
// `<state_dir>/<name>/control.sock` for each RPC.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCtlRequest {
    pub id: u32,
    #[serde(flatten)]
    pub method: SessionCtlMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum SessionCtlMethod {
    /// Return the session's view of its size + window list.
    Info,
    /// List xdg-shell toplevels known to the compositor.
    ListWindows,
    /// Change the logical output size.
    Resize { width: u32, height: u32 },
    /// Capture the current shm buffer of a given window as PNG.
    Screenshot { window_id: u32 },
    /// Composite every registered window into a single session-sized PNG.
    /// Layer surfaces are rendered on top of xdg toplevels.
    ScreenshotDesktop,
    /// Synthesize a keyboard event to the focused client.
    InjectKey {
        keycode: u32,
        state: KeyState,
        modifiers: u32,
    },
    /// Synthesize a pointer event. Mirror of `RequestMethod::InjectPointer`
    /// minus `name` (implicit from the per-session control socket).
    /// `window_id`/`content` carry the same semantics — see the public
    /// `RequestMethod::InjectPointer` docstring.
    InjectPointer {
        /// Logical-pixel X coordinate.
        x: f64,
        /// Logical-pixel Y coordinate.
        y: f64,
        #[serde(default)]
        button: u32,
        #[serde(default = "default_released")]
        state: KeyState,
        #[serde(default)]
        axis_x: f64,
        #[serde(default)]
        axis_y: f64,
        /// Target window id; `None` = focused window.
        /// `#[serde(default)]` keeps v1 daemons talking to v2 sessions and
        /// vice versa working — missing means today's focused-window
        /// behaviour.
        #[serde(default)]
        window_id: Option<u32>,
        /// Treat `(x, y)` as content-space coordinates (CSD inset applied).
        /// `#[serde(default)]` keeps v1 daemons talking to v2 sessions and
        /// vice versa working — missing means today's buffer-local
        /// behaviour.
        #[serde(default)]
        content: bool,
    },
    /// Synthesize a touch event. Mirror of
    /// `RequestMethod::InjectTouch` minus `name` (implicit from the
    /// per-session control socket). `window_id`/`content` carry the
    /// same semantics: see the public `RequestMethod::InjectTouch`
    /// docstring. Routed through the session's `wl_touch` capability and
    /// `State::inject_touch`.
    InjectTouch {
        /// wl_touch tracking id.
        id: u32,
        /// Logical-pixel X coordinate.
        x: f64,
        /// Logical-pixel Y coordinate.
        y: f64,
        /// Touch lifecycle phase (Down / Motion / Up).
        phase: TouchPhase,
        /// Target window id; `None` = focused window.
        /// `#[serde(default)]` keeps v3 daemons talking to v4 sessions
        /// and vice versa working.
        #[serde(default)]
        window_id: Option<u32>,
        /// Treat `(x, y)` as content-space coordinates (CSD inset applied).
        /// `#[serde(default)]` keeps v3 daemons talking to v4 sessions
        /// and vice versa working.
        #[serde(default)]
        content: bool,
    },
    /// Audit H10: deliver a batch of input ops in a single RPC. The SDK's
    /// `type("foo")` was 10 round-trips (5 chars × press+release); a `click`
    /// was 2. Each round-trip is a fresh Unix-socket connect from the
    /// daemon to the session plus msgpack encode/decode/await. Batched,
    /// `type()` and `click()` collapse to one RTT. Ops dispatch in order.
    InjectBatch { ops: Vec<InjectOp> },
    /// Request graceful shutdown. Session exits after acknowledging.
    Shutdown,
    /// Start recording the session's composited output to MKV.
    /// `path` is the absolute output file path; if None, the session chooses
    /// a default under ~/.local/share/waymux/recordings/.
    /// `codec` selects the video encoder; None defaults to `Ffv1` (lossless).
    /// `mode` selects what's captured; None defaults to `FocusedWindow`.
    RecordStart {
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        codec: Option<RecordingCodec>,
        /// Optional second encoder; see RequestMethod::RecordStart docs.
        #[serde(default)]
        secondary_codec: Option<RecordingCodec>,
        #[serde(default)]
        mode: Option<CaptureMode>,
        #[serde(default)]
        min_fps: Option<u32>,
    },
    /// Stop an active recording and finalize the MKV container.
    RecordStop,
    /// Return the current recording state (active flag + path(s) + codec).
    RecordStatus,
    /// Open a WebRTC viewer for this session. Mirror of
    /// `RequestMethod::ViewerStart` but without `session` (already implicit
    /// from the per-session control socket connection).
    ViewerStart {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bind: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
    },
    /// Close the active viewer (if any). Idempotent.
    ViewerStop,
    /// Return the viewer URL (None if no viewer is active).
    ViewerStatus,
}

fn default_released() -> KeyState {
    KeyState::Released
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCtlResponse {
    pub id: u32,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub result: Option<rmpv::Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<String>,
}

impl SessionCtlResponse {
    pub fn success<T: Serialize>(id: u32, value: &T) -> Result<Self, rmp_serde::encode::Error> {
        let bytes = rmp_serde::to_vec_named(value)?;
        let result: rmpv::Value =
            rmp_serde::from_slice(&bytes).expect("round-trip bytes we just produced");
        Ok(Self {
            id,
            ok: true,
            result: Some(result),
            error: None,
        })
    }

    pub fn failure(id: u32, message: impl Into<String>) -> Self {
        Self {
            id,
            ok: false,
            result: None,
            error: Some(message.into()),
        }
    }

    pub fn decode_result<T: serde::de::DeserializeOwned>(
        &self,
    ) -> Result<T, rmp_serde::decode::Error> {
        match &self.result {
            Some(v) => {
                let mut bytes = Vec::new();
                rmpv::encode::write_value(&mut bytes, v)
                    .map_err(|e| rmp_serde::decode::Error::Uncategorized(e.to_string()))?;
                rmp_serde::from_slice(&bytes)
            }
            None => Err(rmp_serde::decode::Error::Uncategorized(
                "session-control response has no result".into(),
            )),
        }
    }
}

/// Result of `SessionCtlMethod::Info`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCtlInfo {
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub window_count: u32,
    /// Nanoseconds since UNIX epoch at the last compositor commit.
    /// Zero if the session has never seen a commit.
    #[serde(default)]
    pub last_damage_ns: u64,
}

/// Result of `SessionCtlMethod::ListWindows`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCtlWindows {
    pub windows: Vec<WindowInfo>,
}

/// Result of `SessionCtlMethod::RecordStart` — the resolved absolute path
/// of the primary recording, plus the secondary recording's path if a
/// secondary encoder was configured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCtlRecordStarted {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary_path: Option<String>,
}

/// Result of `RequestMethod::ViewerStart` — the browser-accessible WebRTC URL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViewerStarted {
    pub url: String,
}

/// Result of `RequestMethod::ViewerStatus` — the active viewer URL, if any.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViewerStatusResponse {
    pub url: Option<String>,
}

/// Result of `RequestMethod::RecordStatus` — the current recording state.
/// `recording` is true iff an active recording exists. When true, `path`
/// (and `secondary_path` for dual-encoder recordings) and `codec` (the
/// primary encoder, kebab-case) describe it; when false the rest are None.
///
/// All fields carry `#[serde(default)]` so an older daemon's response that
/// omits any of them deserializes cleanly into a newer client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RecordStatusResponse {
    #[serde(default)]
    pub recording: bool,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub secondary_path: Option<String>,
    #[serde(default)]
    pub codec: Option<String>,
}

/// Result of `SessionCtlMethod::Screenshot` — a PNG blob plus geometry.
///
/// Screenshots could prefer dmabuf fd-passing, but in the shm-only software
/// path we inline the PNG. msgpack's binary type is used (via `serde_bytes`)
/// to keep the envelope small.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCtlScreenshot {
    pub width: u32,
    pub height: u32,
    #[serde(with = "serde_bytes")]
    pub png: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowInfo {
    pub id: u32,
    pub app_id: String,
    pub title: String,
    pub tags: Vec<String>,
    pub geometry: Rect,
    pub focused: bool,
    pub pid: i32,
    /// The window's content rectangle as set by `xdg_surface.set_window_geometry`,
    /// or `None` if the client never emitted the request (Electron sometimes,
    /// parts of GTK skip it). Engine subtracts this inset internally when
    /// inject calls pass `content=true`; callers can also use it to do manual
    /// CSD math (alternative to `content=True`).
    ///
    /// The session stores this on `SurfaceData` and its `ListWindows` handler
    /// reads and emits it here. `#[serde(default)]` means v2 daemons (whose
    /// response omits the field) deserialize cleanly into the v3 client's
    /// struct as `None`, preserving back-compat.
    #[serde(default)]
    pub content_rect: Option<Rect>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_frame, encode_frame};

    #[test]
    fn hello_roundtrip() {
        let req = Request {
            id: 1,
            method: RequestMethod::Hello {
                client_protocol: CURRENT_PROTOCOL_VERSION,
            },
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::Hello { client_protocol } => {
                assert_eq!(client_protocol, CURRENT_PROTOCOL_VERSION);
            }
            other => panic!("expected Hello, got {:?}", other),
        }
        assert_eq!(decoded.id, 1);
    }

    #[test]
    fn response_success_with_typed_result() {
        let result = HelloResult {
            server_protocol: 1,
            capabilities: vec!["damage_events".into()],
        };
        let resp = Response::success(7, &result).unwrap();
        let mut buf = Vec::new();
        encode_frame(&resp, &mut buf).unwrap();
        let decoded: Response = decode_frame(&buf).unwrap();
        assert_eq!(decoded.id, 7);
        assert!(decoded.ok);
        let decoded_result: HelloResult = decoded.decode_result().unwrap();
        assert_eq!(decoded_result.server_protocol, 1);
        assert_eq!(
            decoded_result.capabilities,
            vec!["damage_events".to_string()]
        );
    }

    #[test]
    fn response_failure() {
        let resp = Response::failure(
            2,
            ResponseError::new(ErrorCode::NotFound, "no such session: foo"),
        );
        let mut buf = Vec::new();
        encode_frame(&resp, &mut buf).unwrap();
        let decoded: Response = decode_frame(&buf).unwrap();
        assert!(!decoded.ok);
        let err = decoded.error.unwrap();
        assert_eq!(err.code, ErrorCode::NotFound);
    }

    #[test]
    fn event_has_id_zero() {
        let ev = Event::new(EventBody::SessionCreated {
            name: "test".into(),
        });
        let mut buf = Vec::new();
        encode_frame(&ev, &mut buf).unwrap();
        let decoded: Event = decode_frame(&buf).unwrap();
        assert_eq!(decoded.id, 0);
    }

    #[test]
    fn create_session_roundtrip() {
        let req = Request {
            id: 3,
            method: RequestMethod::CreateSession(CreateSessionParams {
                name: "x".into(),
                width: 800,
                height: 600,
                scale: 1,
                env: BTreeMap::new(),
                share_clipboard: false,
                share_audio: false,
                mem_cap_mb: None,
                cpu_cap_pct: None,
                disk_quota_mb: None,
                fd_limit: None,
                api_key_id: None,
                codec: None,
                gpu_type: None,
            }),
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::CreateSession(p) => {
                assert_eq!(p.name, "x");
                assert_eq!(p.scale, 1);
                assert!(!p.share_clipboard);
                assert!(!p.share_audio);
            }
            _ => panic!("wrong method"),
        }
    }

    #[test]
    fn create_session_accepts_new_quota_fields() {
        // Wire compat: a client that knows about cpu_cap_pct / disk_quota_mb /
        // fd_limit must round-trip through the daemon, and an older client
        // that omits them must still decode (covered by
        // create_session_accepts_missing_optional_fields). This asserts the
        // forward direction.
        let req = Request {
            id: 9,
            method: RequestMethod::CreateSession(CreateSessionParams {
                name: "q".into(),
                width: 1,
                height: 1,
                scale: 1,
                env: BTreeMap::new(),
                share_clipboard: false,
                share_audio: false,
                mem_cap_mb: Some(512),
                cpu_cap_pct: Some(150),
                disk_quota_mb: Some(64),
                fd_limit: Some(2048),
                api_key_id: None,
                codec: None,
                gpu_type: None,
            }),
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::CreateSession(p) => {
                assert_eq!(p.cpu_cap_pct, Some(150));
                assert_eq!(p.disk_quota_mb, Some(64));
                assert_eq!(p.fd_limit, Some(2048));
            }
            _ => panic!("wrong method"),
        }
    }

    // ── edge case / property-style coverage ─────────────────────────────

    #[test]
    fn create_session_accepts_missing_optional_fields() {
        // Wire shape from an older client that doesn't know about `scale`,
        // `env`, `share_clipboard`, or `share_audio`. All carry #[serde(default)]
        // so forward compat must hold.
        let older_params: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("name"), rmpv::Value::from("old")),
            (rmpv::Value::from("width"), rmpv::Value::from(800u32)),
            (rmpv::Value::from("height"), rmpv::Value::from(600u32)),
        ]);
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(1u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("create_session"),
            ),
            (rmpv::Value::from("params"), older_params),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let req: Request = decode_frame(&framed).unwrap();
        match req.method {
            RequestMethod::CreateSession(p) => {
                assert_eq!(p.name, "old");
                assert_eq!(p.scale, 1, "scale default");
                assert!(p.env.is_empty(), "env default");
                assert!(!p.share_clipboard, "share_clipboard default");
                assert!(!p.share_audio, "share_audio default");
            }
            _ => panic!("wrong method"),
        }
    }

    #[test]
    fn unknown_error_code_deserialises_as_unknown() {
        // Future daemon version introduces E_SESSION_FROZEN. Current client
        // must not crash on it; it should decode to ErrorCode::Unknown
        // (clients must treat unknown codes as Internal for retry
        // decisions).
        let err: rmpv::Value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("code"),
                rmpv::Value::from("E_SESSION_FROZEN"),
            ),
            (rmpv::Value::from("message"), rmpv::Value::from("temporary")),
        ]);
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(1u32)),
            (rmpv::Value::from("ok"), rmpv::Value::from(false)),
            (rmpv::Value::from("error"), err),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let resp: Response = decode_frame(&framed).unwrap();
        assert!(!resp.ok);
        let e = resp.error.unwrap();
        assert_eq!(e.code, ErrorCode::Unknown);
        assert_eq!(e.message, "temporary");
    }

    #[test]
    fn bad_request_error_code_roundtrips() {
        // E_BAD_REQUEST is an additive, back-compatible code for caller-input
        // validation failures. Assert the rename in both directions.
        let json = serde_json::to_string(&ErrorCode::BadRequest).unwrap();
        assert_eq!(json, "\"E_BAD_REQUEST\"");
        let decoded: ErrorCode = serde_json::from_str("\"E_BAD_REQUEST\"").unwrap();
        assert_eq!(decoded, ErrorCode::BadRequest);

        // It must also survive the wire (msgpack) framing used on the socket.
        let resp = Response::failure(
            7,
            ResponseError::new(ErrorCode::BadRequest, "argv[0] must be an absolute path"),
        );
        let mut buf = Vec::new();
        encode_frame(&resp, &mut buf).unwrap();
        let decoded: Response = decode_frame(&buf).unwrap();
        assert_eq!(decoded.error.unwrap().code, ErrorCode::BadRequest);
    }

    #[test]
    fn unknown_method_fails_deserialisation_gracefully() {
        // Forward compat: a request with an unknown method name should
        // produce a clean Error::Msgpack (not a panic), letting the
        // server send an error response and keep the connection alive.
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(1u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("warp_to_mars"),
            ),
            (rmpv::Value::from("params"), rmpv::Value::Map(vec![])),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let err = decode_frame::<Request>(&framed).expect_err("unknown method must error");
        assert!(
            matches!(err, crate::DecodeError::Msgpack(_)),
            "expected Msgpack decode error, got {err:?}"
        );
    }

    #[test]
    fn event_flatten_unambiguously_distinguishes_response() {
        // A Response frame must NOT decode as an Event and vice-versa.
        // This is what the daemon's event-forwarder relies on when it
        // dispatches an Outgoing::Response vs Outgoing::Event.
        let resp = Response::success(
            5,
            &HelloResult {
                server_protocol: 1,
                capabilities: vec![],
            },
        )
        .unwrap();
        let mut rb = Vec::new();
        encode_frame(&resp, &mut rb).unwrap();
        assert!(
            rmp_serde::from_slice::<Event>(&rb[4..]).is_err(),
            "a Response frame must fail to parse as Event"
        );

        let ev = Event::new(EventBody::SessionCreated { name: "t".into() });
        let mut eb = Vec::new();
        encode_frame(&ev, &mut eb).unwrap();
        assert!(
            rmp_serde::from_slice::<Response>(&eb[4..]).is_err(),
            "an Event frame must fail to parse as Response"
        );
    }

    #[test]
    fn all_event_variants_roundtrip() {
        // Make sure we didn't break any variant's serde tagging. Iterate
        // a canonical instance of each body and assert decode == encode.
        let cases = vec![
            EventBody::SessionCreated { name: "a".into() },
            EventBody::SessionDestroyed {
                name: "a".into(),
                exit_code: 0,
            },
            EventBody::ChildExited {
                name: "a".into(),
                pid: 1,
                exit_code: 137,
            },
            EventBody::WindowCreated {
                name: "a".into(),
                window_id: 1,
                app_id: "foo".into(),
                title: "bar".into(),
                pid: 1,
            },
            EventBody::WindowDestroyed {
                name: "a".into(),
                window_id: 1,
            },
            EventBody::WindowChanged {
                name: "a".into(),
                window_id: 1,
                fields: WindowChange {
                    title: Some("t".into()),
                    ..Default::default()
                },
            },
            EventBody::Damage {
                name: "a".into(),
                serial: 7,
                timestamp_ns: 999,
            },
            EventBody::Log {
                name: "a".into(),
                stream: "stderr".into(),
                text: "boom".into(),
            },
        ];
        for body in cases {
            let ev = Event::new(body.clone());
            let mut buf = Vec::new();
            encode_frame(&ev, &mut buf).unwrap();
            let decoded: Event = decode_frame(&buf).unwrap();
            assert_eq!(
                decoded.id, 0,
                "event id must always serialise as 0 on the wire"
            );
            // The body should round-trip; use debug equality since we
            // derived Debug but not PartialEq on WindowChange.
            assert_eq!(format!("{:?}", decoded.body), format!("{:?}", body));
        }
    }

    #[test]
    fn oversized_frame_rejected_on_decode() {
        // Synthesize a header claiming MAX_FRAME_SIZE + 1 bytes.
        let mut framed = Vec::new();
        framed.extend_from_slice(&((crate::MAX_FRAME_SIZE as u32) + 1).to_be_bytes());
        framed.resize(4 + 1024, 0); // short payload; length field is what matters
        let err = decode_frame::<Request>(&framed).expect_err("oversize must be rejected");
        assert!(
            matches!(err, crate::DecodeError::TooLarge(_)),
            "expected TooLarge, got {err:?}"
        );
    }

    #[test]
    fn screenshot_format_defaults_to_dmabuf() {
        // If an older client omits `format`, the server-side params should
        // default to Dmabuf (the preferred the protocol spec path).
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(1u32)),
            (rmpv::Value::from("method"), rmpv::Value::from("screenshot")),
            (
                rmpv::Value::from("params"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("name"), rmpv::Value::from("s")),
                    (rmpv::Value::from("window_id"), rmpv::Value::from(1u32)),
                ]),
            ),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let req: Request = decode_frame(&framed).unwrap();
        match req.method {
            RequestMethod::Screenshot { format, .. } => {
                assert_eq!(format, ScreenshotFormat::Dmabuf);
            }
            _ => panic!("wrong method"),
        }
    }

    #[test]
    fn inject_pointer_accepts_motion_only_wire() {
        // Motion-only: just x/y. All button/axis/state fields are
        // defaulted — breakage here would hurt the CLI `waymux click x y`.
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(1u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("inject_pointer"),
            ),
            (
                rmpv::Value::from("params"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("name"), rmpv::Value::from("p")),
                    (rmpv::Value::from("x"), rmpv::Value::from(12.5)),
                    (rmpv::Value::from("y"), rmpv::Value::from(7.0)),
                ]),
            ),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let req: Request = decode_frame(&framed).unwrap();
        match req.method {
            RequestMethod::InjectPointer {
                x,
                y,
                button,
                state,
                axis_x,
                axis_y,
                ..
            } => {
                assert_eq!(x, 12.5);
                assert_eq!(y, 7.0);
                assert_eq!(button, 0);
                assert_eq!(state, KeyState::Released); // default
                assert_eq!(axis_x, 0.0);
                assert_eq!(axis_y, 0.0);
            }
            _ => panic!("wrong method"),
        }
    }

    #[test]
    fn record_start_roundtrip() {
        let req = Request {
            id: 42,
            method: RequestMethod::RecordStart {
                name: "xonotic".into(),
                path: Some("/tmp/test.mkv".into()),
                codec: Some(RecordingCodec::H264Nvenc),
                secondary_codec: None,
                mode: Some(CaptureMode::WholeDesktop),
                min_fps: Some(60),
            },
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::RecordStart {
                name,
                path,
                codec,
                secondary_codec,
                mode,
                min_fps,
            } => {
                assert_eq!(name, "xonotic");
                assert_eq!(path, Some("/tmp/test.mkv".into()));
                assert_eq!(codec, Some(RecordingCodec::H264Nvenc));
                assert_eq!(secondary_codec, None);
                assert_eq!(mode, Some(CaptureMode::WholeDesktop));
                assert_eq!(min_fps, Some(60));
            }
            other => panic!("expected RecordStart, got {:?}", other),
        }
    }

    #[test]
    fn record_start_dual_encoder_roundtrip() {
        // Dual-encoder: primary HEVC lossless archive + secondary H264 live.
        let req = Request {
            id: 7,
            method: RequestMethod::RecordStart {
                name: "saas".into(),
                path: Some("/tmp/saas.mkv".into()),
                codec: Some(RecordingCodec::HevcVulkanLossless),
                secondary_codec: Some(RecordingCodec::H264Vulkan),
                mode: None,
                min_fps: Some(60),
            },
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::RecordStart {
                codec,
                secondary_codec,
                ..
            } => {
                assert_eq!(codec, Some(RecordingCodec::HevcVulkanLossless));
                assert_eq!(secondary_codec, Some(RecordingCodec::H264Vulkan));
            }
            other => panic!("expected RecordStart, got {:?}", other),
        }
    }

    #[test]
    fn record_start_omits_codec_for_backcompat() {
        // Legacy CLIs that don't know about the codec / mode / min_fps /
        // secondary_codec fields send no keys; serde must default to None.
        let req = Request {
            id: 1,
            method: RequestMethod::RecordStart {
                name: "old-cli".into(),
                path: None,
                codec: None,
                secondary_codec: None,
                mode: None,
                min_fps: None,
            },
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::RecordStart {
                codec,
                secondary_codec,
                mode,
                min_fps,
                ..
            } => {
                assert_eq!(codec, None);
                assert_eq!(secondary_codec, None);
                assert_eq!(mode, None);
                assert_eq!(min_fps, None);
            }
            other => panic!("expected RecordStart, got {:?}", other),
        }
    }

    #[test]
    fn record_started_secondary_path_optional() {
        // Single-encoder response omits secondary_path (skip_serializing_if
        // None); decoded value defaults to None.
        let started = SessionCtlRecordStarted {
            path: "/tmp/p.mkv".into(),
            secondary_path: None,
        };
        let bytes = rmp_serde::to_vec_named(&started).unwrap();
        let back: SessionCtlRecordStarted = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.path, "/tmp/p.mkv");
        assert_eq!(back.secondary_path, None);

        // Dual-encoder response carries both paths.
        let dual = SessionCtlRecordStarted {
            path: "/tmp/p.mkv".into(),
            secondary_path: Some("/tmp/p.secondary.mkv".into()),
        };
        let bytes = rmp_serde::to_vec_named(&dual).unwrap();
        let back: SessionCtlRecordStarted = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.secondary_path.as_deref(), Some("/tmp/p.secondary.mkv"));
    }

    #[test]
    fn capture_mode_kebab_case_wire_format() {
        for (variant, expected) in [
            (CaptureMode::FocusedWindow, "focused-window"),
            (CaptureMode::WholeDesktop, "whole-desktop"),
        ] {
            let bytes = rmp_serde::to_vec_named(&variant).unwrap();
            let s: String = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(s, expected);
        }
    }

    #[test]
    fn recording_codec_kebab_case_wire_format() {
        // The wire format must be kebab-case strings (match what the CLI
        // takes as `--codec` argument values). Use the rmp_serde
        // round-trip — same path the live wire uses.
        for (variant, expected) in [
            (RecordingCodec::Ffv1, "ffv1"),
            (RecordingCodec::H264Nvenc, "h264-nvenc"),
            (RecordingCodec::H264Vaapi, "h264-vaapi"),
            (RecordingCodec::H264Vulkan, "h264-vulkan"),
            (RecordingCodec::Ffv1Vulkan, "ffv1-vulkan"),
            (RecordingCodec::H264VulkanLossless, "h264-vulkan-lossless"),
            (RecordingCodec::HevcVulkanLossless, "hevc-vulkan-lossless"),
        ] {
            let bytes = rmp_serde::to_vec_named(&variant).unwrap();
            let s: String = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(
                s, expected,
                "variant {variant:?} should serialize to {expected:?}"
            );
        }
    }

    #[test]
    fn record_stop_roundtrip() {
        let req = Request {
            id: 43,
            method: RequestMethod::RecordStop {
                name: "xonotic".into(),
            },
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::RecordStop { name } => assert_eq!(name, "xonotic"),
            other => panic!("expected RecordStop, got {:?}", other),
        }
    }

    #[test]
    fn record_status_request_roundtrip() {
        let req = Request {
            id: 44,
            method: RequestMethod::RecordStatus {
                name: "xonotic".into(),
            },
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: Request = decode_frame(&buf).unwrap();
        match decoded.method {
            RequestMethod::RecordStatus { name } => assert_eq!(name, "xonotic"),
            other => panic!("expected RecordStatus, got {:?}", other),
        }
    }

    #[test]
    fn record_status_response_roundtrip() {
        let resp = RecordStatusResponse {
            recording: true,
            path: Some("/run/user/1000/waymux/recordings/e2e.mkv".into()),
            secondary_path: Some("/run/user/1000/waymux/recordings/e2e.secondary.mkv".into()),
            codec: Some("ffv1".into()),
        };
        let bytes = rmp_serde::to_vec_named(&resp).unwrap();
        let decoded: RecordStatusResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded, resp);

        // The inactive shape, and the default for back-compat.
        let inactive = RecordStatusResponse::default();
        assert!(!inactive.recording);
        let bytes = rmp_serde::to_vec_named(&inactive).unwrap();
        let decoded: RecordStatusResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded, inactive);
    }

    #[test]
    fn session_ctl_record_start_roundtrip() {
        let req = SessionCtlRequest {
            id: 1,
            method: SessionCtlMethod::RecordStart {
                path: None,
                codec: None,
                secondary_codec: None,
                mode: None,
                min_fps: None,
            },
        };
        let mut buf = Vec::new();
        encode_frame(&req, &mut buf).unwrap();
        let decoded: SessionCtlRequest = decode_frame(&buf).unwrap();
        match decoded.method {
            SessionCtlMethod::RecordStart {
                path,
                codec,
                secondary_codec,
                mode,
                min_fps,
            } => {
                assert!(path.is_none());
                assert_eq!(codec, None);
                assert_eq!(secondary_codec, None);
                assert_eq!(mode, None);
                assert_eq!(min_fps, None);
            }
            other => panic!("expected RecordStart, got {:?}", other),
        }
    }

    #[test]
    fn session_ctl_record_started_result_roundtrip() {
        let started = SessionCtlRecordStarted {
            path: "/run/user/1000/waymux/recordings/xonotic-1234.mkv".into(),
            secondary_path: None,
        };
        let resp = SessionCtlResponse::success(1, &started).unwrap();
        let decoded: SessionCtlRecordStarted = resp.decode_result().unwrap();
        assert_eq!(decoded.path, started.path);
        assert_eq!(decoded.secondary_path, None);
    }

    #[test]
    fn viewer_start_request_round_trips() {
        let req = Request {
            id: 1,
            method: RequestMethod::ViewerStart {
                session: "eagle".into(),
                bind: Some("10.42.0.2".into()),
                port: Some(8080),
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            RequestMethod::ViewerStart {
                session,
                bind,
                port,
            } => {
                assert_eq!(session, "eagle");
                assert_eq!(bind, Some("10.42.0.2".into()));
                assert_eq!(port, Some(8080));
            }
            m => panic!("unexpected method: {m:?}"),
        }
    }

    #[test]
    fn viewer_start_default_bind_omits_field_on_wire() {
        let req = Request {
            id: 2,
            method: RequestMethod::ViewerStart {
                session: "eagle".into(),
                bind: None,
                port: None,
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        if let RequestMethod::ViewerStart { bind, port, .. } = back.method {
            assert_eq!(bind, None);
            assert_eq!(port, None);
        } else {
            panic!("wrong method");
        }
    }

    /// task #109: legacy clients (pre-port-flag CLI) send ViewerStart
    /// without a `port` field; the server must accept it (port → None,
    /// falls back to pick_ephemeral_port). Pins backwards compat.
    #[test]
    fn viewer_start_decodes_legacy_payload_without_port_field() {
        // Hand-encoded MessagePack request from a pre-task-#109 client:
        // method.ViewerStart with only `session` + `bind` — no `port`.
        let legacy = Request {
            id: 99,
            method: RequestMethod::ViewerStart {
                session: "legacy-client".into(),
                bind: None,
                port: None,
            },
        };
        // Serialize with port=None (skip_serializing_if drops it on wire),
        // then re-decode. This is the same shape an older client would emit.
        let bytes = rmp_serde::to_vec_named(&legacy).unwrap();
        // Smoke: the wire bytes don't mention "port" at all.
        let as_str = String::from_utf8_lossy(&bytes);
        assert!(
            !as_str.contains("port"),
            "skip_serializing_if dropped port=None; wire shouldn't carry it"
        );
        // Decode round-trips cleanly.
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            RequestMethod::ViewerStart {
                session,
                bind,
                port,
            } => {
                assert_eq!(session, "legacy-client");
                assert_eq!(bind, None);
                assert_eq!(port, None);
            }
            m => panic!("unexpected method: {m:?}"),
        }
    }

    #[test]
    fn viewer_started_response_carries_url() {
        let resp = ViewerStarted {
            url: "http://127.0.0.1:18347".into(),
        };
        let v = rmp_serde::to_vec_named(&resp).unwrap();
        let back: ViewerStarted = rmp_serde::from_slice(&v).unwrap();
        assert_eq!(back.url, "http://127.0.0.1:18347");
    }

    #[test]
    fn session_unavailable_json_roundtrip() {
        // The wire shape is HTTP/JSON (the SDK speaks JSON, not msgpack, to
        // the api proxy), so we round-trip via serde_json. Verify the
        // canonical fields are present and `retry_after_s: None` is omitted
        // from the serialization (skip_serializing_if).
        let env = SessionUnavailable::new("resume_timeout", Some(10));
        let s = serde_json::to_string(&env).unwrap();
        assert!(s.contains("\"error\":\"session_unavailable\""), "body: {s}");
        assert!(s.contains("\"reason\":\"resume_timeout\""), "body: {s}");
        assert!(s.contains("\"retry_after_s\":10"), "body: {s}");
        let back: SessionUnavailable = serde_json::from_str(&s).unwrap();
        assert_eq!(back, env);

        // None retry_after_s is omitted on the wire.
        let env2 = SessionUnavailable::new("upstream_unavailable", None);
        let s2 = serde_json::to_string(&env2).unwrap();
        assert!(
            !s2.contains("retry_after_s"),
            "expected field omitted: {s2}"
        );
        let back2: SessionUnavailable = serde_json::from_str(&s2).unwrap();
        assert_eq!(back2, env2);
        assert_eq!(back2.retry_after_s, None);

        // And explicit null is accepted on the wire too (Stripe-style
        // tolerant parsing).
        let with_null = "{\"error\":\"session_unavailable\",\"reason\":\"vm_not_ready\",\"retry_after_s\":null}";
        let parsed: SessionUnavailable = serde_json::from_str(with_null).unwrap();
        assert_eq!(parsed.reason, "vm_not_ready");
        assert_eq!(parsed.retry_after_s, None);
    }

    // ── wire v2: InjectPointer + InjectBatch new fields ────────────────────
    // (The v2 version guard `current_protocol_version_is_two` was superseded
    // by `current_protocol_version_is_three` when v3 landed.)

    #[test]
    fn inject_pointer_v2_roundtrip_preserves_window_id_and_content() {
        // The new v2 shape: both fields present and non-default. Round-trip
        // through rmp_serde::to_vec_named + from_slice (the production
        // path) and assert every field survives.
        let req = Request {
            id: 11,
            method: RequestMethod::InjectPointer {
                name: "s".into(),
                x: 10.0,
                y: 20.0,
                button: 0,
                state: KeyState::Released,
                axis_x: 0.0,
                axis_y: 0.0,
                window_id: Some(7),
                content: true,
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            RequestMethod::InjectPointer {
                name,
                x,
                y,
                button,
                state,
                axis_x,
                axis_y,
                window_id,
                content,
            } => {
                assert_eq!(name, "s");
                assert_eq!(x, 10.0);
                assert_eq!(y, 20.0);
                assert_eq!(button, 0);
                assert_eq!(state, KeyState::Released);
                assert_eq!(axis_x, 0.0);
                assert_eq!(axis_y, 0.0);
                assert_eq!(window_id, Some(7));
                assert!(content);
            }
            other => panic!("expected InjectPointer, got {:?}", other),
        }
    }

    #[test]
    fn inject_pointer_v1_fixture_defaults_to_focused_window_and_buffer_coords() {
        // Back-compat invariant: a v1 client (no window_id / no content
        // fields) talking to a v2 daemon must still parse. The two new
        // fields are #[serde(default)], so missing → None / false, which
        // preserves today's behaviour (focused-window targeting,
        // buffer-local coords).
        let v1_params: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("name"), rmpv::Value::from("s")),
            (rmpv::Value::from("x"), rmpv::Value::F64(100.0)),
            (rmpv::Value::from("y"), rmpv::Value::F64(50.0)),
            (rmpv::Value::from("button"), rmpv::Value::from(0x110u32)),
            (rmpv::Value::from("state"), rmpv::Value::from("pressed")),
            (rmpv::Value::from("axis_x"), rmpv::Value::F64(0.0)),
            (rmpv::Value::from("axis_y"), rmpv::Value::F64(0.0)),
        ]);
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(2u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("inject_pointer"),
            ),
            (rmpv::Value::from("params"), v1_params),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let req: Request = decode_frame(&framed).unwrap();
        match req.method {
            RequestMethod::InjectPointer {
                name,
                x,
                y,
                button,
                state,
                window_id,
                content,
                ..
            } => {
                assert_eq!(name, "s");
                assert_eq!(x, 100.0);
                assert_eq!(y, 50.0);
                assert_eq!(button, 0x110);
                assert_eq!(state, KeyState::Pressed);
                assert_eq!(window_id, None, "v1 fixture must default window_id to None");
                assert!(!content, "v1 fixture must default content to false");
            }
            other => panic!("expected InjectPointer, got {:?}", other),
        }
    }

    #[test]
    fn inject_batch_per_op_window_id_roundtrips() {
        // Mixed batch: one v2-shaped pointer op targeting window 3 in
        // content space, plus one bare-bones op that defaults. Both must
        // survive a serialize/deserialize cycle with their fields intact.
        let req = Request {
            id: 12,
            method: RequestMethod::InjectBatch {
                name: "s".into(),
                ops: vec![
                    InjectOp::Pointer {
                        x: 1.0,
                        y: 2.0,
                        button: 0,
                        state: KeyState::Released,
                        axis_x: 0.0,
                        axis_y: 0.0,
                        window_id: Some(3),
                        content: true,
                        seq: 0,
                    },
                    InjectOp::Pointer {
                        x: 9.0,
                        y: 9.0,
                        button: 0x110,
                        state: KeyState::Pressed,
                        axis_x: 0.0,
                        axis_y: 0.0,
                        window_id: Some(11),
                        content: false,
                        seq: 0,
                    },
                ],
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            RequestMethod::InjectBatch { name, ops } => {
                assert_eq!(name, "s");
                assert_eq!(ops.len(), 2);
                match &ops[0] {
                    InjectOp::Pointer {
                        window_id,
                        content,
                        x,
                        y,
                        ..
                    } => {
                        assert_eq!(*window_id, Some(3));
                        assert!(*content);
                        assert_eq!(*x, 1.0);
                        assert_eq!(*y, 2.0);
                    }
                    other => panic!("expected Pointer op, got {:?}", other),
                }
                match &ops[1] {
                    InjectOp::Pointer {
                        window_id,
                        content,
                        button,
                        ..
                    } => {
                        assert_eq!(*window_id, Some(11));
                        assert!(!*content);
                        assert_eq!(*button, 0x110);
                    }
                    other => panic!("expected Pointer op, got {:?}", other),
                }
            }
            other => panic!("expected InjectBatch, got {:?}", other),
        }
    }

    #[test]
    fn inject_batch_mixed_v1_and_v2_ops_deserialize_with_defaults() {
        // The strongest back-compat test: a single InjectBatch carrying one
        // v1-shaped Pointer op (no window_id / content) and one v2-shaped
        // op (both present). Both must round-trip into the same enum, with
        // the v1 op's missing fields defaulting (None / false).
        let v1_op: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("kind"), rmpv::Value::from("pointer")),
            (
                rmpv::Value::from("params"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("x"), rmpv::Value::F64(5.0)),
                    (rmpv::Value::from("y"), rmpv::Value::F64(6.0)),
                    (rmpv::Value::from("button"), rmpv::Value::from(0u32)),
                    (rmpv::Value::from("state"), rmpv::Value::from("released")),
                    (rmpv::Value::from("axis_x"), rmpv::Value::F64(0.0)),
                    (rmpv::Value::from("axis_y"), rmpv::Value::F64(0.0)),
                ]),
            ),
        ]);
        let v2_op: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("kind"), rmpv::Value::from("pointer")),
            (
                rmpv::Value::from("params"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("x"), rmpv::Value::F64(7.0)),
                    (rmpv::Value::from("y"), rmpv::Value::F64(8.0)),
                    (rmpv::Value::from("button"), rmpv::Value::from(0u32)),
                    (rmpv::Value::from("state"), rmpv::Value::from("released")),
                    (rmpv::Value::from("axis_x"), rmpv::Value::F64(0.0)),
                    (rmpv::Value::from("axis_y"), rmpv::Value::F64(0.0)),
                    (rmpv::Value::from("window_id"), rmpv::Value::from(42u32)),
                    (rmpv::Value::from("content"), rmpv::Value::Boolean(true)),
                ]),
            ),
        ]);
        let params: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("name"), rmpv::Value::from("s")),
            (
                rmpv::Value::from("ops"),
                rmpv::Value::Array(vec![v1_op, v2_op]),
            ),
        ]);
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(13u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("inject_batch"),
            ),
            (rmpv::Value::from("params"), params),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let req: Request = decode_frame(&framed).unwrap();
        match req.method {
            RequestMethod::InjectBatch { ops, .. } => {
                assert_eq!(ops.len(), 2);
                match &ops[0] {
                    InjectOp::Pointer {
                        window_id, content, ..
                    } => {
                        assert_eq!(*window_id, None, "v1 op in batch must default window_id");
                        assert!(!*content, "v1 op in batch must default content");
                    }
                    other => panic!("expected Pointer op, got {:?}", other),
                }
                match &ops[1] {
                    InjectOp::Pointer {
                        window_id, content, ..
                    } => {
                        assert_eq!(*window_id, Some(42));
                        assert!(*content);
                    }
                    other => panic!("expected Pointer op, got {:?}", other),
                }
            }
            other => panic!("expected InjectBatch, got {:?}", other),
        }
    }

    #[test]
    fn session_ctl_inject_pointer_v2_roundtrip() {
        // The session-control mirror of InjectPointer carries the same two
        // new fields. Round-trip through msgpack and assert both survive.
        let req = SessionCtlRequest {
            id: 14,
            method: SessionCtlMethod::InjectPointer {
                x: 1.5,
                y: 2.5,
                button: 0x110,
                state: KeyState::Pressed,
                axis_x: 0.0,
                axis_y: 0.0,
                window_id: Some(99),
                content: true,
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: SessionCtlRequest = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            SessionCtlMethod::InjectPointer {
                window_id, content, ..
            } => {
                assert_eq!(window_id, Some(99));
                assert!(content);
            }
            other => panic!("expected SessionCtlMethod::InjectPointer, got {:?}", other),
        }
    }

    #[test]
    fn session_ctl_inject_pointer_v1_fixture_defaults() {
        // Old-shape SessionCtl payload (no window_id / content) must still
        // decode — this is what an in-flight daemon → session call would
        // look like during a partial deploy. Defaults to None / false.
        let v1_params: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("x"), rmpv::Value::F64(1.0)),
            (rmpv::Value::from("y"), rmpv::Value::F64(2.0)),
            (rmpv::Value::from("button"), rmpv::Value::from(0u32)),
            (rmpv::Value::from("state"), rmpv::Value::from("released")),
            (rmpv::Value::from("axis_x"), rmpv::Value::F64(0.0)),
            (rmpv::Value::from("axis_y"), rmpv::Value::F64(0.0)),
        ]);
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(15u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("inject_pointer"),
            ),
            (rmpv::Value::from("params"), v1_params),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let req: SessionCtlRequest = rmp_serde::from_slice(&bytes).unwrap();
        match req.method {
            SessionCtlMethod::InjectPointer {
                window_id, content, ..
            } => {
                assert_eq!(window_id, None);
                assert!(!content);
            }
            other => panic!("expected SessionCtlMethod::InjectPointer, got {:?}", other),
        }
    }

    // ── wire v3: InjectSelector + WindowInfo.content_rect ──────────────────
    // (The v3 version guard `current_protocol_version_is_three` was superseded
    // by `current_protocol_version_is_four` when v4 landed.)

    #[test]
    fn current_protocol_version_is_four() {
        // Guards against accidental version regression; the constant should
        // only change when the wire format does.
        assert_eq!(CURRENT_PROTOCOL_VERSION, 4);
    }

    #[test]
    fn inject_selector_v3_roundtrip() {
        // Reserved wire slot. The variant exists in v3 and must survive a
        // serialize/deserialize cycle with every field intact; the daemon
        // currently returns NotImplemented but the wire-level shape is
        // locked in here.
        let req = Request {
            id: 21,
            method: RequestMethod::InjectSelector {
                session: "s".into(),
                selector: "button.signin".into(),
                button: 0x110,
                pressed: true,
                content: true,
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            RequestMethod::InjectSelector {
                session,
                selector,
                button,
                pressed,
                content,
            } => {
                assert_eq!(session, "s");
                assert_eq!(selector, "button.signin");
                assert_eq!(button, 0x110);
                assert!(pressed);
                assert!(content);
            }
            other => panic!("expected InjectSelector, got {:?}", other),
        }
        assert_eq!(back.id, 21);
    }

    #[test]
    fn inject_selector_default_content_is_false() {
        // A v3 client that omits `content` from the payload must
        // deserialize to `content: false` (the `#[serde(default)]`
        // bool default). This is the same back-compat shape that
        // InjectPointer's content field uses.
        let params: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("session"), rmpv::Value::from("s")),
            (
                rmpv::Value::from("selector"),
                rmpv::Value::from("[aria-label='Sign in']"),
            ),
            (rmpv::Value::from("button"), rmpv::Value::from(0x110u32)),
            (rmpv::Value::from("pressed"), rmpv::Value::Boolean(true)),
            // content intentionally omitted
        ]);
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(22u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("inject_selector"),
            ),
            (rmpv::Value::from("params"), params),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let req: Request = decode_frame(&framed).unwrap();
        match req.method {
            RequestMethod::InjectSelector {
                content, selector, ..
            } => {
                assert!(!content, "missing content must default to false");
                assert_eq!(selector, "[aria-label='Sign in']");
            }
            other => panic!("expected InjectSelector, got {:?}", other),
        }
    }

    #[test]
    fn window_info_with_content_rect_roundtrips() {
        // A v3 daemon populates content_rect when the inner client emitted
        // xdg_surface.set_window_geometry. Round-trip via msgpack (the
        // production wire) and assert the field survives.
        let wi = WindowInfo {
            id: 5,
            app_id: "org.gnome.Calculator".into(),
            title: "Calculator".into(),
            tags: vec![],
            geometry: Rect {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            focused: true,
            pid: 4242,
            content_rect: Some(Rect {
                x: 10,
                y: 10,
                width: 780,
                height: 580,
            }),
        };
        let bytes = rmp_serde::to_vec_named(&wi).unwrap();
        let back: WindowInfo = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.id, 5);
        assert_eq!(
            back.content_rect,
            Some(Rect {
                x: 10,
                y: 10,
                width: 780,
                height: 580,
            })
        );
    }

    // ── wire v4: InjectTouch + InjectOp::Touch + TouchPhase ────────────────

    #[test]
    fn inject_touch_v4_roundtrip() {
        // Construct an InjectTouch with every field populated and confirm
        // every field survives a msgpack round-trip. Same coverage shape
        // as inject_pointer_v2_roundtrip_preserves_window_id_and_content.
        let req = Request {
            id: 31,
            method: RequestMethod::InjectTouch {
                // Field is `name` (matches InjectPointer); the spec brief
                // showed `session` but the in-tree InjectPointer uses
                // `name` and we mirror that for cross-variant consistency.
                name: "x".into(),
                id: 0,
                x: 50.0,
                y: 100.0,
                phase: TouchPhase::Down,
                window_id: Some(7),
                content: true,
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            RequestMethod::InjectTouch {
                name,
                id,
                x,
                y,
                phase,
                window_id,
                content,
            } => {
                assert_eq!(name, "x");
                assert_eq!(id, 0);
                assert_eq!(x, 50.0);
                assert_eq!(y, 100.0);
                assert_eq!(phase, TouchPhase::Down);
                assert_eq!(window_id, Some(7));
                assert!(content);
            }
            other => panic!("expected InjectTouch, got {:?}", other),
        }
        assert_eq!(back.id, 31);
    }

    #[test]
    fn inject_touch_default_content_is_false() {
        // A v3-shaped payload (no `content` field — which is what a v3
        // client would send if it learned about InjectTouch from a
        // bleeding-edge daemon but still spoke v3 itself) must deserialize
        // with `content == false`. Same back-compat invariant the v2 → v1
        // pointer test enforces; `#[serde(default)]` on the field is what
        // makes this work.
        let params: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("name"), rmpv::Value::from("x")),
            (rmpv::Value::from("id"), rmpv::Value::from(0u32)),
            (rmpv::Value::from("x"), rmpv::Value::F64(10.0)),
            (rmpv::Value::from("y"), rmpv::Value::F64(20.0)),
            (rmpv::Value::from("phase"), rmpv::Value::from("down")),
            // window_id and content intentionally omitted
        ]);
        let wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(32u32)),
            (
                rmpv::Value::from("method"),
                rmpv::Value::from("inject_touch"),
            ),
            (rmpv::Value::from("params"), params),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &wire).unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&bytes);
        let req: Request = decode_frame(&framed).unwrap();
        match req.method {
            RequestMethod::InjectTouch {
                window_id,
                content,
                phase,
                ..
            } => {
                assert_eq!(window_id, None, "missing window_id must default to None");
                assert!(!content, "missing content must default to false");
                assert_eq!(phase, TouchPhase::Down);
            }
            other => panic!("expected InjectTouch, got {:?}", other),
        }
    }

    #[test]
    fn inject_op_touch_round_trips_in_batch() {
        // A batch carrying a Down → Up pair must preserve op order across
        // a msgpack round-trip. Multi-finger gestures aren't covered in
        // T4 (deferred), so single-finger Down/Up is the production shape.
        let req = Request {
            id: 33,
            method: RequestMethod::InjectBatch {
                name: "x".into(),
                ops: vec![
                    InjectOp::Touch {
                        id: 0,
                        x: 5.0,
                        y: 6.0,
                        phase: TouchPhase::Down,
                        window_id: Some(3),
                        content: true,
                    },
                    InjectOp::Touch {
                        id: 0,
                        x: 5.0,
                        y: 6.0,
                        phase: TouchPhase::Up,
                        window_id: Some(3),
                        content: true,
                    },
                ],
            },
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let back: Request = rmp_serde::from_slice(&bytes).unwrap();
        match back.method {
            RequestMethod::InjectBatch { ops, .. } => {
                assert_eq!(ops.len(), 2);
                match &ops[0] {
                    InjectOp::Touch {
                        phase,
                        id,
                        window_id,
                        content,
                        ..
                    } => {
                        assert_eq!(*phase, TouchPhase::Down, "first op must be Down");
                        assert_eq!(*id, 0);
                        assert_eq!(*window_id, Some(3));
                        assert!(*content);
                    }
                    other => panic!("expected Touch op, got {:?}", other),
                }
                match &ops[1] {
                    InjectOp::Touch { phase, .. } => {
                        assert_eq!(*phase, TouchPhase::Up, "second op must be Up");
                    }
                    other => panic!("expected Touch op, got {:?}", other),
                }
            }
            other => panic!("expected InjectBatch, got {:?}", other),
        }
    }

    #[test]
    fn touch_phase_serializes_snake_case() {
        // Each TouchPhase variant must emit its snake_case wire form;
        // catches accidental enum renaming (e.g. someone "improving" the
        // variant to PascalCase via #[serde(rename_all = "PascalCase")]).
        // We assert via JSON (the canonical, human-readable form of serde
        // output) — msgpack would emit the same string but in a binary
        // wrapper that's awkward to grep.
        let down = serde_json::to_string(&TouchPhase::Down).unwrap();
        let motion = serde_json::to_string(&TouchPhase::Motion).unwrap();
        let up = serde_json::to_string(&TouchPhase::Up).unwrap();
        assert_eq!(down, "\"down\"");
        assert_eq!(motion, "\"motion\"");
        assert_eq!(up, "\"up\"");
    }

    #[test]
    fn window_info_v2_fixture_defaults_content_rect_to_none() {
        // Back-compat: a v2 daemon's ListWindows response omits the
        // content_rect field. A v3 client must still parse it; the
        // `#[serde(default)]` on the field makes `None` the natural
        // default.
        let v2_wire: rmpv::Value = rmpv::Value::Map(vec![
            (rmpv::Value::from("id"), rmpv::Value::from(1u32)),
            (rmpv::Value::from("app_id"), rmpv::Value::from("firefox")),
            (rmpv::Value::from("title"), rmpv::Value::from("Mozilla")),
            (rmpv::Value::from("tags"), rmpv::Value::Array(vec![])),
            (
                rmpv::Value::from("geometry"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("x"), rmpv::Value::from(0i32)),
                    (rmpv::Value::from("y"), rmpv::Value::from(0i32)),
                    (rmpv::Value::from("width"), rmpv::Value::from(1024u32)),
                    (rmpv::Value::from("height"), rmpv::Value::from(768u32)),
                ]),
            ),
            (rmpv::Value::from("focused"), rmpv::Value::Boolean(true)),
            (rmpv::Value::from("pid"), rmpv::Value::from(99i32)),
            // content_rect intentionally omitted (v2 shape)
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &v2_wire).unwrap();
        let wi: WindowInfo = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(wi.id, 1);
        assert_eq!(wi.app_id, "firefox");
        assert_eq!(wi.title, "Mozilla");
        assert_eq!(
            wi.content_rect, None,
            "v2 fixture must default content_rect to None"
        );
    }

    #[test]
    fn pointer_op_seq_defaults_and_parses() {
        // Absent seq -> default 0
        let j = r#"{"kind":"pointer","params":{"x":1.0,"y":2.0}}"#;
        let op: InjectOp = serde_json::from_str(j).unwrap();
        match op {
            InjectOp::Pointer { seq, .. } => assert_eq!(seq, 0),
            _ => panic!("expected Pointer variant"),
        }
        // Present seq -> parsed
        let j2 = r#"{"kind":"pointer","params":{"x":1.0,"y":2.0,"seq":7}}"#;
        let op2: InjectOp = serde_json::from_str(j2).unwrap();
        match op2 {
            InjectOp::Pointer { seq, .. } => assert_eq!(seq, 7),
            _ => panic!("expected Pointer variant"),
        }
    }
}
