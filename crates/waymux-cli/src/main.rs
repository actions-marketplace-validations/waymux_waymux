// SPDX-License-Identifier: Apache-2.0

//! `waymux` — command-line interface for the daemon.
//!
//! Two transport modes:
//!
//! * **Local** (default): opens a Unix socket from `WAYMUX_SOCKET` env or
//!   `$XDG_RUNTIME_DIR/waymux.sock` and speaks msgpack-RPC.
//! * **Remote** (`--remote` or `WAYMUX_REMOTE=1`): routes session-control
//!   subcommands over HTTPS to a hosted `waymux-api` endpoint, using
//!   credentials persisted by `waymux login`.
//!
//! TRUST MODEL (`--json`): error envelopes carry the raw error message, which
//! may include absolute host paths and other local detail. The `--json`
//! surface assumes a same-uid / semi-trusted consumer: the local daemon is
//! SO_PEERCRED same-uid gated, so anyone who can drive this CLI against it
//! already shares its uid. Messages are intentionally not sanitized; scrub at
//! the call site before crossing any trust boundary.

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Value};
use std::path::PathBuf;

use waymux_cli::credentials::{self, Profile};
use waymux_cli::transport::{InjectOp, LocalTransport, RemoteTransport, Transport};
use waymux_cli::Connection;

use waymux_protocol::{
    AttachResult, Event, EventBody, KeyState, RecordStatusResponse, RecordingCodec, RequestMethod,
    SessionCtlRecordStarted, ViewerStarted, ViewerStatusResponse, WindowInfo,
};

// ─── JSON envelope ──────────────────────────────────────────────────────────
//
// When `--json` is set, every non-streaming verb emits a single uniform
// envelope on stdout instead of human-readable text:
//
//   success: { "ok": true,  "verb": "<verb>", "data": { ... } }
//   error:   { "ok": false, "verb": "<verb>",
//              "error": { "code": "E_...", "message": "...", "detail": null } }
//
// Streaming verbs (`events`, `logs`) emit newline-delimited JSON instead and
// are handled inline (they do not use these helpers).

/// A locally-detected caller-input error that must surface with an explicit
/// wire `E_...` code rather than the default `E_INTERNAL` fallback.
///
/// Most CLI errors are either daemon RPC failures (whose `"{ErrorCode:?}:
/// {msg}"` prefix `error_code_for` recognizes) or genuine local faults
/// (connection refused, decode failures) that correctly map to `E_INTERNAL`.
/// A few verbs, though, validate caller input *before* sending it to the
/// daemon (a malformed `--ops` payload, a record path outside the sandbox).
/// Those are caller errors, not server faults, so they must report
/// `E_BAD_REQUEST` — the same family as the daemon-side `spawn`/`create`
/// `BadRequest` work. This typed error carries the intended code so the
/// envelope builder can honor it via a downcast instead of pattern-matching
/// the message string.
#[derive(Debug)]
struct CliError {
    /// Stable wire code string (e.g. `E_BAD_REQUEST`).
    code: &'static str,
    message: String,
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CliError {}

impl CliError {
    /// A caller-input error that should surface as `E_BAD_REQUEST`.
    fn bad_request(message: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(CliError {
            code: "E_BAD_REQUEST",
            message: message.into(),
        })
    }

    /// A "no such session/resource" error that should surface as
    /// `E_NOT_FOUND`, consistent with the daemon's own NotFound mapping.
    fn not_found(message: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(CliError {
            code: "E_NOT_FOUND",
            message: message.into(),
        })
    }
}

/// Build the success envelope value for a verb.
fn success_envelope(verb: &str, data: Value) -> Value {
    json!({ "ok": true, "verb": verb, "data": data })
}

/// Build the error envelope value for a verb. `code` is a stable `E_...`
/// string; `message` is human-readable; `detail` is currently always null.
fn error_envelope(verb: &str, code: &str, message: &str) -> Value {
    json!({
        "ok": false,
        "verb": verb,
        "error": { "code": code, "message": message, "detail": Value::Null },
    })
}

/// Print a success envelope as a single compact line on stdout.
fn print_success(verb: &str, data: Value) {
    println!("{}", success_envelope(verb, data));
}

/// Map a single recognized daemon `ErrorCode` Debug name to its wire `E_...`
/// string, or `None` if the name is not one this CLI knows. Kept as its own
/// function so a contract test can assert it stays in sync with the protocol
/// enum's variants.
fn known_error_code(debug_name: &str) -> Option<&'static str> {
    Some(match debug_name {
        "NotFound" => "E_NOT_FOUND",
        "AlreadyExists" => "E_ALREADY_EXISTS",
        "ProtoVersion" => "E_PROTO_VERSION",
        "NoRenderNode" => "E_NO_RENDER_NODE",
        "ResizeRejected" => "E_RESIZE_REJECTED",
        "Backpressure" => "E_BACKPRESSURE",
        "NotImplemented" => "E_NOT_IMPLEMENTED",
        "BadRequest" => "E_BAD_REQUEST",
        "Internal" => "E_INTERNAL",
        // `Unknown` is the protocol's own deserialize-fallback variant; a
        // daemon that returns it has already lost the original wire code, so
        // E_INTERNAL is the faithful mapping.
        "Unknown" => "E_INTERNAL",
        _ => return None,
    })
}

/// Derive a stable `E_...` code string from an anyhow error.
///
/// RPC failures from the daemon surface through `Connection::request` as
/// `"{ErrorCode:?}: {message}"` (e.g. `"NotFound: no such session"`), where
/// the prefix is the `ErrorCode` Debug name. We recognize those prefixes and
/// map them to the wire `E_...` strings.
///
/// An *unrecognized* prefix means the daemon and this CLI have diverged on the
/// wire enum (a code added daemon-side but not here). Rather than silently
/// collapsing it to `E_INTERNAL` and hiding the drift, surface it as
/// `E_UNKNOWN:<name>` so the real code is visible to the consumer. Errors with
/// no recognizable prefix at all (local `bail!`s, connection failures, parse
/// errors) still fall back to plain `E_INTERNAL`.
fn error_code_for(err: &anyhow::Error) -> String {
    // A locally-detected caller-input error carries its intended wire code
    // explicitly; honor it before falling back to message-prefix matching.
    if let Some(cli_err) = err.downcast_ref::<CliError>() {
        return cli_err.code.to_string();
    }
    let msg = err.to_string();
    // Only treat the head as a code prefix when the message actually had a
    // `<head>: <rest>` shape; a message with no colon is a local error, not a
    // wire code, and must not be reported as an unknown daemon code.
    let had_colon = msg.contains(':');
    let head = msg.split(':').next().unwrap_or("").trim();
    if let Some(code) = known_error_code(head) {
        return code.to_string();
    }
    // A recognizable-looking but unknown prefix (a single bare CamelCase token
    // before the colon) is a wire-format divergence: preserve it.
    if had_colon
        && !head.is_empty()
        && head.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && head.chars().next().is_some_and(|c| c.is_ascii_uppercase())
    {
        return format!("E_UNKNOWN:{head}");
    }
    "E_INTERNAL".to_string()
}

/// Map a command to its stable verb string for the envelope. Matches the
/// CLI subcommand names (kebab-case where applicable).
fn verb_name(cmd: &Cmd) -> &'static str {
    match cmd {
        Cmd::Ls => "ls",
        Cmd::New { .. } => "new",
        Cmd::Rm { .. } => "rm",
        Cmd::Info { .. } => "info",
        Cmd::Inject { .. } => "inject",
        Cmd::Spawn { .. } => "spawn",
        Cmd::Windows { .. } => "windows",
        Cmd::Tag { .. } => "tag",
        Cmd::Resize { .. } => "resize",
        Cmd::Screenshot { .. } => "screenshot",
        Cmd::ScreenshotDesktop { .. } => "screenshot-desktop",
        Cmd::Idle { .. } => "idle",
        Cmd::Wait { .. } => "wait",
        Cmd::Key { .. } => "key",
        Cmd::Click { .. } => "click",
        Cmd::Events { .. } => "events",
        Cmd::Logs { .. } => "logs",
        Cmd::Attach { .. } => "attach",
        Cmd::Detach { .. } => "detach",
        Cmd::Record { .. } => "record",
        Cmd::Viewer { .. } => "viewer",
        Cmd::Login { .. } => "login",
        Cmd::Serve => "serve",
    }
}

/// Locate the `waymux-attach` binary. Prefers a sibling in the same directory
/// as the running `waymux` binary (covers both installed and cargo dev builds),
/// then falls back to `waymux-attach` on `$PATH`.
fn find_attach_binary() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("waymux-attach");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("waymux-attach")
}

#[derive(Parser, Debug)]
#[command(name = "waymux", version)]
struct Args {
    /// Control socket path. Defaults to `$XDG_RUNTIME_DIR/waymux.sock`.
    #[arg(long, env = "WAYMUX_SOCKET", global = true)]
    socket: Option<PathBuf>,

    /// Route session-control subcommands through HTTPS to the credentialed
    /// remote endpoint instead of the local Unix-socket daemon.
    #[arg(long, env = "WAYMUX_REMOTE", global = true)]
    remote: bool,

    /// Override base URL when --remote is set. Default: value from credentials.
    #[arg(long, global = true)]
    base_url: Option<String>,

    /// Emit machine-readable JSON instead of human-readable text. Every verb
    /// prints a single uniform envelope `{ ok, verb, data | error }` on stdout
    /// (streaming verbs `events`/`logs` stay newline-delimited JSON). Opt-in:
    /// without this flag the default text output is unchanged.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// List sessions.
    #[command(alias = "list")]
    Ls,
    /// Create a session.
    New {
        name: String,
        /// Size in `WxH` pixels.
        #[arg(long, default_value = "1920x1080")]
        size: String,
        /// Output scale.
        #[arg(long, default_value_t = 1)]
        scale: u32,
        /// Share host PulseAudio/PipeWire sockets with the session so apps
        /// inside it can play (and record) audio. Default off. (Local-only.)
        #[arg(long)]
        share_audio: bool,
        /// Aggregate memory cap (in MiB) applied via cgroup-v2 to the session.
        #[arg(long)]
        mem_cap_mb: Option<u32>,
        /// Aggregate CPU cap as % of one core. `200` = two full cores.
        /// Applied via cgroup-v2 `cpu.max`.
        #[arg(long)]
        cpu_cap_pct: Option<u32>,
        /// Per-session disk quota (in MiB) for the runtime dir, applied as
        /// a tmpfs mount. Requires daemon CAP_SYS_ADMIN to take effect.
        #[arg(long)]
        disk_quota_mb: Option<u32>,
        /// File-descriptor cap (RLIMIT_NOFILE) applied to the session
        /// subprocess and inherited by everything it spawns.
        #[arg(long)]
        fd_limit: Option<u32>,
        /// Waymux API-key id (UUID) to attribute usage to. Get it from the
        /// `id` field of `POST /api-keys`. The id is embedded verbatim in
        /// usage-event JSONL so the reporter can join without needing the
        /// plaintext. Pass via `--api-key-id` or `WAYMUX_API_KEY_ID`.
        #[arg(long, env = "WAYMUX_API_KEY_ID")]
        api_key_id: Option<String>,
    },
    /// Destroy a session.
    #[command(alias = "destroy")]
    Rm { name: String },
    /// Show details for a single session.
    Info { name: String },
    /// Inject one or more input ops as a JSON array.
    ///
    /// Example:
    ///   waymux inject mysession --ops '[{"type":"key","keycode":30}]'
    Inject {
        name: String,
        /// JSON array of inject ops. See `InjectOp` in the source for shape.
        #[arg(long)]
        ops: String,
    },
    /// Launch a client inside a session. (Local-only.)
    Spawn {
        name: String,
        #[arg(long)]
        compositor: bool,
        #[arg(last = true, required = true)]
        argv: Vec<String>,
    },
    /// List windows in a session.
    Windows {
        name: String,
        /// Only list windows whose tag set contains this tag. Filtering is
        /// done client-side after the daemon returns the full window list;
        /// omit the flag for the unfiltered list.
        #[arg(long)]
        tag: Option<String>,
    },
    /// Apply one or more tags to a window in a session. Tags are free-form
    /// labels you can later select on (e.g. `waymux windows <s> --tag <t>`
    /// or `waymux wait <s> --tag <t>`). Replaces the window's tag set.
    Tag {
        /// Session name.
        name: String,
        /// Window id (see `waymux windows <session>`).
        window_id: u32,
        /// One or more tags to set on the window. At least one is required.
        #[arg(required = true)]
        tags: Vec<String>,
    },
    /// Resize a session's virtual output. (Local-only.)
    Resize { name: String, size: String },
    /// Capture a window's current buffer as PNG.
    Screenshot {
        name: String,
        /// Window id (see `waymux windows <session>`).
        window_id: u32,
        /// Output PNG path, or `-` for stdout. Optional under `--json` (the
        /// PNG is returned as `data.png_b64`); required otherwise. If given
        /// under `--json` the file is also written.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Composite every window in the session into a single desktop PNG.
    ScreenshotDesktop {
        name: String,
        /// Output PNG path, or `-` for stdout. Optional under `--json` (the
        /// PNG is returned as `data.png_b64`); required otherwise. If given
        /// under `--json` the file is also written.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Wait until the session has been quiescent for `--quiet-ms`.
    Idle {
        name: String,
        #[arg(long, default_value_t = 500)]
        quiet_ms: u32,
        #[arg(long, default_value_t = 10_000)]
        timeout_ms: u32,
    },
    /// Block until a window matching a selector appears, or timeout.
    /// (Local-only — uses the events stream.)
    Wait {
        name: String,
        #[arg(long)]
        app_id: Option<String>,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        tag: Option<String>,
        #[arg(long)]
        pid: Option<i32>,
        #[arg(long)]
        nth: Option<usize>,
        #[arg(long, default_value_t = 5_000)]
        timeout_ms: u32,
    },
    /// Send a synthetic key event. (Local-only convenience; remote uses `inject`.)
    Key {
        name: String,
        keycode: u32,
        #[arg(long)]
        release: bool,
        #[arg(long, default_value_t = 0)]
        modifiers: u32,
    },
    /// Move the pointer / press a button. (Local-only convenience.)
    Click {
        name: String,
        x: f64,
        y: f64,
        #[arg(long, default_value_t = 0)]
        button: u32,
    },
    /// Stream session events as newline-delimited JSON. (Local-only.)
    Events {
        name: String,
        #[arg(long = "topic", default_values_t = ["sessions".to_string(), "windows".to_string()])]
        topics: Vec<String>,
    },
    /// Tail a session's stdout/stderr. (Local-only.)
    Logs {
        name: String,
        #[arg(short, long)]
        follow: bool,
        #[arg(long, default_value_t = 200)]
        settle_ms: u64,
    },
    /// Print the path of the attach Wayland socket. (Local-only.)
    Attach { name: String },
    /// Mark a session as detached. (Local-only.)
    Detach { name: String },
    /// Record a session's composited output to a lossless MKV file. (Local-only.)
    Record {
        #[command(subcommand)]
        action: RecordCmd,
    },
    /// Open a browser WebRTC viewer for a session. (Local-only.)
    Viewer {
        #[command(subcommand)]
        sub: ViewerSub,
    },
    /// Authenticate against a hosted `waymux-api` endpoint and persist the
    /// credentials.
    Login {
        /// API key. Required for the MVP non-browser flow.
        #[arg(long)]
        api_key: Option<String>,
        /// Base URL of the remote endpoint to authenticate against. No
        /// production host is baked into the OSS build; point this at your
        /// own deployment (defaults to a localhost placeholder).
        #[arg(long, default_value = "http://localhost:8080")]
        base_url: String,
    },
    /// Run the `waymuxd` daemon in the foreground (one-binary onboarding).
    ///
    /// `waymux serve` resolves the `waymuxd` binary ($WAYMUXD_BIN, then a
    /// sibling of this `waymux`, then $PATH) and replaces this process with it
    /// (execv). It is a supervised foreground daemon: it does NOT background.
    /// The global `--socket` flag (or `$WAYMUX_SOCKET`) is forwarded.
    ///
    /// You usually do not need to run this explicitly: when a local verb finds
    /// no daemon socket, the CLI auto-spawns a background `waymuxd` and retries.
    /// Set `$WAYMUX_NO_AUTOSPAWN=1` to disable that and manage the daemon
    /// yourself (e.g. via `waymux serve` under a process supervisor). An
    /// auto-spawned daemon outlives the CLI invocation; `waymux` has no
    /// `shutdown` verb today, so stop it with a signal (Ctrl-C in the `serve`
    /// terminal, or `kill` / `pkill waymuxd`).
    Serve,
}

/// Mirror of `waymux_protocol::RecordingCodec` for clap's `value_enum`
/// derive. Kept separate from the protocol type so the protocol crate
/// doesn't pull a clap dependency.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum CodecArg {
    /// Lossless FFV1 in MKV. Default. ~70 MB / minute at 1080p.
    Ffv1,
    /// H.264 via NVIDIA NVENC. Lossy, ~5 MB / minute. NVIDIA GPU only.
    H264Nvenc,
    /// H.264 via libva VAAPI. Lossy, ~5 MB / minute. AMD/Intel GPU.
    H264Vaapi,
    /// H.264 via Vulkan video encode + in-process MKV muxer. Zero
    /// ffmpeg subprocess, zero-copy on capable hardware. Requires
    /// NVIDIA driver 535+, AMD Mesa 25+ (VCN 2+), or Intel Mesa 24.3+.
    H264Vulkan,
    /// Lossless FFV1 via Vulkan compute (ffmpeg's `ffv1_vulkan`
    /// encoder). GPU zero-copy from inner-client dmabuf through the
    /// encoder. Far slower than real-time on integrated GPUs (~0.1× at
    /// 1080p on Renoir); expected real-time at 4K 60 fps on discrete
    /// cards. Use for marketing-grade pixel-exact recordings.
    Ffv1Vulkan,
    /// Bit-exact lossless H.264 via Vulkan video encode at Hi444PP
    /// profile, QP=0. **Currently broken on NVIDIA drivers 560+580** —
    /// the Vulkan encode submit fails (`ERROR_INITIALIZATION_FAILED`).
    /// Use `hevc-vulkan-lossless` instead for NVIDIA lossless on
    /// shipping hardware.
    H264VulkanLossless,
    /// Lossless H.265 / HEVC via the `hevc_vulkan` encoder (ffmpeg
    /// 8.0) at RangeExt profile (4:4:4) with `-tune lossless -qp 0`.
    /// Validated on NVIDIA RTX A6000 + driver 580.159.03 at 1080p
    /// 60 fps QP=0, 2.5× real-time. Pixels never leave GPU memory
    /// between the compute shader and the encoder.
    HevcVulkanLossless,
}

impl From<CodecArg> for RecordingCodec {
    fn from(c: CodecArg) -> Self {
        match c {
            CodecArg::Ffv1 => RecordingCodec::Ffv1,
            CodecArg::H264Nvenc => RecordingCodec::H264Nvenc,
            CodecArg::H264Vaapi => RecordingCodec::H264Vaapi,
            CodecArg::H264Vulkan => RecordingCodec::H264Vulkan,
            CodecArg::Ffv1Vulkan => RecordingCodec::Ffv1Vulkan,
            CodecArg::H264VulkanLossless => RecordingCodec::H264VulkanLossless,
            CodecArg::HevcVulkanLossless => RecordingCodec::HevcVulkanLossless,
        }
    }
}

/// Mirror of `waymux_protocol::CaptureMode` for clap's `value_enum`.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModeArg {
    /// Default. Capture only the focused window's surface tree.
    /// One memcpy per commit — single-surface fast path.
    FocusedWindow,
    /// Capture the full inner-compositor surface set composited
    /// together. Multi-window flows + Plasma demos. Per-commit
    /// software composite — slower but visually complete.
    WholeDesktop,
}

impl From<ModeArg> for waymux_protocol::CaptureMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::FocusedWindow => waymux_protocol::CaptureMode::FocusedWindow,
            ModeArg::WholeDesktop => waymux_protocol::CaptureMode::WholeDesktop,
        }
    }
}

#[derive(Subcommand, Debug)]
enum RecordCmd {
    /// Start recording a session's output to a lossless FFV1 MKV.
    ///
    /// Works headless — no attach client required. A dedicated
    /// recording-producer thread feeds frames into ffmpeg, decoupled
    /// from the `outer_view` render path. Two capture paths:
    ///
    /// - **Fast (dmabuf zero-copy)**: when the focused window committed
    ///   a GPU-rendered dmabuf (chromium with `--ozone-platform=wayland`
    ///   and most Wayland-native GPU clients), the producer hands the
    ///   buffer reference straight to ffmpeg without recompositing.
    ///   Hits the 30 fps target on commodity hardware.
    /// - **Slow (composite fallback)**: SHM-only clients or
    ///   multi-surface scenes that need real compositing fall back to
    ///   a software composite of the session's surface set. ~2 fps on
    ///   AMD integrated graphics; adequate for replay-debug recordings,
    ///   slow for video.
    ///
    /// Resolution: matches the focused window's surface on the fast
    /// path (chromium's actual rendered size including chrome) or the
    /// session geometry on the slow path. If the producer's chosen path
    /// switches mid-recording the dimensions can change and ffmpeg
    /// stops with a warning in the daemon log — uncommon in practice
    /// since one fullscreen client = one path for the whole recording.
    Start {
        /// Session name to record.
        name: String,
        /// Output MKV path. Must be inside `~/.local/share/waymux/recordings/`.
        /// Defaults to `~/.local/share/waymux/recordings/<session>-<ts>.mkv`.
        output: Option<PathBuf>,
        /// Video encoder. `ffv1` (default) is lossless, CPU-encoded,
        /// ~70 MB / minute at 1080p — preferred for visual-regression CI
        /// artifacts. `h264-nvenc` and `h264-vaapi` are lossy hardware
        /// encoders (~10× smaller files, lower CPU) — preferred for
        /// marketing screencasts and customer review videos. The hardware
        /// encoders require the matching GPU + ffmpeg build; the daemon
        /// surfaces a typed error at start time if unavailable.
        #[arg(long, value_enum)]
        codec: Option<CodecArg>,
        /// Optional second encoder running in parallel from the same
        /// frame source. Output goes to `<output>.secondary.mkv`. Used
        /// for a bit-exact archive + low-latency live stream from one
        /// compositor commit cadence.
        #[arg(long, value_enum)]
        secondary_codec: Option<CodecArg>,
        /// Capture mode. `focused-window` (default) captures only the
        /// focused window's surface tree — one buffer per commit, no
        /// compositing — and is the right shape for "record what my
        /// agent did" workflows. `whole-desktop` captures the full
        /// inner-compositor surface set composited together; right for
        /// multi-window flows or full Plasma desktop recordings.
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        /// Minimum frames-per-second pacing. Without this flag, recording
        /// is pure commit-driven — idle pages produce 0 fps, files stay
        /// small. With `--min-fps 60`, the recorder re-encodes the most
        /// recent captured frame at 60 Hz whenever no new commit arrives
        /// within 1/60 s, guaranteeing a steady 60 fps output. Use for
        /// marketing hero clips; leave unset for visual-regression CI.
        #[arg(long)]
        min_fps: Option<u32>,
    },
    /// Stop a session's recording.
    Stop { name: String },
    /// Report whether a session is currently recording.
    ///
    /// Prints a human-readable line (`recording: yes  path=…  codec=…` or
    /// `recording: no`). With `--json`, emits the recording state as a
    /// structured envelope.
    Status { name: String },
}

#[derive(clap::Subcommand, Debug)]
pub enum ViewerSub {
    /// Start a viewer for a session. Prints the URL on stdout.
    Start {
        name: String,
        /// Bind address for the bridge's HTTP+WS+WebRTC endpoints.
        /// Defaults to 127.0.0.1 (loopback). Set to a WireGuard IP for
        /// rental testing — SSH local-forward does NOT pass WebRTC's
        /// UDP media plane.
        #[arg(long)]
        bind: Option<String>,
        /// Explicit TCP port for the bridge. Defaults to
        /// pick_ephemeral_port (the bridge picks a random free port).
        /// Set to a fixed port (8080) on SaaS VMs so the portal Connect
        /// URL routes there instead of an unpredictable ephemeral.
        /// Caller is responsible for ensuring nothing else holds the
        /// port (e.g. stop the waymux-healthz shim first on baked
        /// images).
        #[arg(long)]
        port: Option<u16>,
    },
    /// Stop the active viewer for a session.
    Stop { name: String },
    /// Print the URL if a viewer is active, else exit 0 with no output.
    Status { name: String },
    /// Mint an ephemeral EdDSA viewer JWT + its public key for the LOCAL/dev
    /// laptop-viewer path. Prints the token, the public key the session must
    /// trust, and the expiry. (Rust port of
    /// `scripts/laptop-mint-viewer-token.py`; that helper still works.)
    ///
    /// The private key is throwaway: generated, used once to sign, then
    /// discarded. Set the session's `WAYMUX_VIEWER_TOKEN_ED25519_PK` to the
    /// printed public key and `WAYMUX_VM_SESSION_ID` to the printed session id
    /// so the neko-bridge verifies this token (fail-closed EdDSA path).
    Token {
        /// The `vm_session_id` to bind the token to (must equal the session's
        /// `WAYMUX_VM_SESSION_ID`). Pass a UUID; omit to mint a fresh random
        /// one (printed in the output so you can configure the session).
        session: Option<String>,
        /// Token lifetime in seconds. Default 8 h, matching the python helper.
        #[arg(long, alias = "ttl", default_value_t = waymux_cli::viewer_token::DEFAULT_EXP_SECS)]
        exp_secs: u64,
        /// Optional `sub` UUID (the owning user). Defaults to a fresh random
        /// UUID, matching the python helper.
        #[arg(long)]
        sub: Option<String>,
    },
}

fn default_socket_path() -> Result<PathBuf> {
    let runtime =
        std::env::var_os("XDG_RUNTIME_DIR").context("XDG_RUNTIME_DIR not set — pass --socket")?;
    Ok(PathBuf::from(runtime).join("waymux.sock"))
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once('x')
        .ok_or_else(|| anyhow!("expected WxH, got {s}"))?;
    Ok((w.parse()?, h.parse()?))
}

/// Resolve the recordings sandbox directory the same way the session does:
/// `$HOME/.local/share/waymux/recordings`. Does NOT create the directory (this
/// is a pre-send validation, not the recorder); the session creates it.
fn recordings_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .context("HOME not set — cannot resolve the recordings directory")?;
    Ok(PathBuf::from(home).join(".local/share/waymux/recordings"))
}

/// Validate a user-supplied recording output path before it is sent to the
/// daemon. Mirrors the session-side H2 gate (absolute, no `..`, inside the
/// recordings sandbox) so a bad path surfaces as a CLI-local E_BAD_REQUEST
/// instead of the E_INTERNAL the session backstop would emit. The session
/// retains its own identical check as defense-in-depth.
fn validate_recording_path(path: &std::path::Path) -> Result<()> {
    use std::path::Component;
    if !path.is_absolute() {
        return Err(CliError::bad_request(format!(
            "recording path must be absolute (got {})",
            path.display()
        )));
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(CliError::bad_request(format!(
            "recording path must not contain '..' (got {})",
            path.display()
        )));
    }
    let allowed = recordings_dir()?;
    if !path.starts_with(&allowed) {
        return Err(CliError::bad_request(format!(
            "recording path must be inside {} (got {})",
            allowed.display(),
            path.display()
        )));
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

/// Whether this subcommand can operate against the remote endpoint. Used to
/// reject e.g. `waymux --remote spawn …` cleanly.
fn supports_remote(cmd: &Cmd) -> bool {
    matches!(
        cmd,
        Cmd::Ls
            | Cmd::New { .. }
            | Cmd::Rm { .. }
            | Cmd::Info { .. }
            | Cmd::Inject { .. }
            | Cmd::Screenshot { .. }
            | Cmd::ScreenshotDesktop { .. }
            | Cmd::Windows { .. }
            | Cmd::Login { .. }
    )
}

/// Build a `RemoteTransport` from credentials + flag overrides. Errors with a
/// friendly message when nothing is configured.
fn build_remote_transport(args_base_url: &Option<String>) -> Result<RemoteTransport> {
    let path = credentials::credentials_path()?;
    let creds = credentials::load_from(&path)?
        .ok_or_else(|| anyhow!("No credentials. Run `waymux login --api-key <KEY>` first."))?;
    let profile = creds.default_profile().ok_or_else(|| {
        anyhow!(
            "Credentials at {} have no [default] profile",
            path.display()
        )
    })?;
    let base_url = args_base_url
        .clone()
        .unwrap_or_else(|| profile.base_url.clone());
    RemoteTransport::new(base_url, profile.api_key.clone())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    let json = args.json;

    // Run the requested verb. When `--json` is set and the verb returns an
    // error, convert it into an ERROR envelope on stdout (preserving the
    // non-zero exit code) instead of letting anyhow print a plain-text error
    // to stderr. The verb string is captured up front because `dispatch`
    // consumes `args.command`.
    let verb = verb_name(&args.command);
    let result = dispatch(args).await;
    if let Err(err) = result {
        if json {
            // Streaming verbs already emit NDJSON and never reach here with an
            // envelope expectation; for everything else, emit the error
            // envelope to stdout and exit non-zero.
            //
            // TRUST NOTE: the error `message` is the underlying anyhow chain
            // verbatim and MAY contain absolute filesystem paths (socket
            // locations, session runtime dirs) and other host-local detail. We
            // do NOT sanitize it: the `--json` consumer is assumed to be a
            // same-uid / semi-trusted process. The local daemon itself is
            // SO_PEERCRED same-uid gated (see waymux-daemon server.rs), so a
            // caller able to read this output already shares the daemon's uid.
            // If this surface is ever exposed across a trust boundary, scrub
            // the message here first.
            println!(
                "{}",
                error_envelope(verb, &error_code_for(&err), &err.to_string())
            );
            std::process::exit(1);
        }
        return Err(err);
    }
    Ok(())
}

/// Route the parsed command to the right handler. Split out of `main` so the
/// JSON error-envelope wrapper can intercept any `Err` uniformly.
async fn dispatch(args: Args) -> Result<()> {
    let json = args.json;

    // `login` is a special case — no daemon connection, no remote dispatch.
    if let Cmd::Login { api_key, base_url } = &args.command {
        return run_login(api_key.clone(), base_url.clone(), json);
    }

    // `viewer token` is a special case — pure local key/JWT mint, no daemon
    // connection and no remote dispatch. Handle it before the auto-spawn path
    // so minting a token never starts a `waymuxd`.
    if let Cmd::Viewer {
        sub:
            ViewerSub::Token {
                session,
                exp_secs,
                sub,
            },
    } = &args.command
    {
        return run_viewer_token(session.clone(), *exp_secs, sub.clone(), json);
    }

    // `serve` runs the daemon in the foreground by exec-replacing this process
    // with `waymuxd`. It never uses --remote and never auto-spawns (it IS the
    // daemon). Forward an explicit --socket if the user passed one; the
    // WAYMUX_SOCKET env var is already inherited by the exec'd daemon.
    if matches!(args.command, Cmd::Serve) {
        let mut extra: Vec<std::ffi::OsString> = Vec::new();
        if let Some(sock) = &args.socket {
            extra.push("--socket".into());
            extra.push(sock.clone().into_os_string());
        }
        // On success this never returns (the process image is replaced).
        waymux_cli::daemon::exec_daemon(&extra)?;
        unreachable!("exec_daemon returns Ok only by diverging");
    }

    if args.remote {
        if !supports_remote(&args.command) {
            bail!(
                "this subcommand has no remote equivalent yet — drop --remote or run on the host"
            );
        }
        let transport = build_remote_transport(&args.base_url)?;
        return run_with_transport(args.command, Box::new(transport), json).await;
    }

    // Local mode. Some subcommands (Spawn, Wait, Events, Logs, Attach, Detach,
    // Record, Resize, Tag, Key, Click) need raw access to the Connection
    // because they use streaming events or daemon-only RPCs (no remote HTTP
    // endpoint). Route them through a dedicated handler to avoid bloating the
    // Transport trait.
    let socket = match args.socket {
        Some(p) => p,
        None => default_socket_path()?,
    };

    // Quickstart: if the control socket is absent, transparently start a
    // background `waymuxd` and wait for it, so a fresh user's first local verb
    // works without a separate daemon start. No-op when the socket already
    // exists (so real connection errors are never masked), for --remote (which
    // never reaches here), and when opted out via $WAYMUX_NO_AUTOSPAWN. See
    // `waymux_cli::daemon::ensure_daemon_or_spawn`.
    waymux_cli::daemon::ensure_daemon_or_spawn(&socket)?;

    match args.command {
        // The transport-routable subcommands can use LocalTransport directly.
        Cmd::Ls
        | Cmd::New { .. }
        | Cmd::Rm { .. }
        | Cmd::Info { .. }
        | Cmd::Inject { .. }
        | Cmd::Screenshot { .. }
        | Cmd::ScreenshotDesktop { .. }
        | Cmd::Windows { .. } => {
            let local = LocalTransport::connect(&socket).await?;
            run_with_transport(args.command, Box::new(local), json).await
        }
        // Local-only subcommands.
        other => run_local_only(other, &socket, json).await,
    }
}

fn run_login(api_key: Option<String>, base_url: String, json: bool) -> Result<()> {
    let key = api_key.ok_or_else(|| {
        anyhow!(
            "browser flow not yet implemented — pass --api-key <KEY> for the MVP. \
             Get a key from <{}>",
            base_url
        )
    })?;
    if key.trim().is_empty() {
        bail!("--api-key cannot be empty");
    }
    let mut creds = credentials::load()?.unwrap_or_default();
    creds.set_default(Profile {
        base_url: base_url.clone(),
        api_key: key.clone(),
    });
    let path = credentials::save(&creds)?;
    if json {
        // NEVER echo the api key: only the profile + base_url.
        print_success(
            "login",
            json!({ "profile": "default", "base_url": base_url }),
        );
    } else {
        println!(
            "Logged in to {} as {} (credentials: {})",
            base_url,
            credentials::redact_key(&key),
            path.display()
        );
    }
    Ok(())
}

/// `waymux viewer token` — mint an ephemeral EdDSA viewer JWT + its public key
/// for the LOCAL/dev laptop-viewer path. No daemon, no network. The private key
/// is generated inside `viewer_token::mint`, used once, and dropped on return:
/// it is never persisted, printed, or written to disk.
fn run_viewer_token(
    session: Option<String>,
    exp_secs: u64,
    sub: Option<String>,
    json: bool,
) -> Result<()> {
    use waymux_cli::viewer_token;

    // The token's vm_session_id must be a UUID (the bridge runs uuid.Parse on
    // the session match). Accept the positional as that UUID; mint a fresh one
    // when omitted (matching the python helper) so the user can configure the
    // session's WAYMUX_VM_SESSION_ID to it.
    let vm_session_id = match session {
        Some(s) => uuid::Uuid::parse_str(s.trim()).with_context(|| {
            format!(
                "viewer token <session> must be a vm_session_id UUID (got {s:?}); \
                 omit it to mint a fresh random one"
            )
        })?,
        None => uuid::Uuid::new_v4(),
    };
    let sub = match sub {
        Some(s) => Some(
            uuid::Uuid::parse_str(s.trim())
                .with_context(|| format!("--sub must be a UUID (got {s:?})"))?,
        ),
        None => None,
    };

    let minted = viewer_token::mint(vm_session_id, exp_secs, sub)?;

    if json {
        print_success(
            "viewer",
            json!({
                "token": minted.token,
                "public_key": minted.public_key_b64,
                "vm_session_id": minted.vm_session_id.to_string(),
                "expires_at": minted.expires_at,
            }),
        );
    } else {
        // Human output: the three values the local viewer path needs, plus a
        // one-line hint on which env vars configure the session to trust them.
        println!("token:          {}", minted.token);
        println!("public_key:     {}", minted.public_key_b64);
        println!("vm_session_id:  {}", minted.vm_session_id);
        println!("expires_at:     {} (unix seconds)", minted.expires_at);
        println!();
        println!("Configure the session to trust this token by exporting before launch:");
        println!("  WAYMUX_VIEWER_TOKEN_ED25519_PK={}", minted.public_key_b64);
        println!("  WAYMUX_VM_SESSION_ID={}", minted.vm_session_id);
        println!("Then open the viewer URL with ?token=<token>.");
    }
    Ok(())
}

/// Drive whichever transport-routable subcommand is in `cmd`. When `json` is
/// set, each verb prints a success envelope on stdout instead of text.
async fn run_with_transport(cmd: Cmd, mut t: Box<dyn Transport>, json: bool) -> Result<()> {
    match cmd {
        Cmd::Ls => {
            let sessions = t.list_sessions().await?;
            if json {
                print_success("ls", json!({ "sessions": sessions }));
            } else if sessions.is_empty() {
                println!("(no sessions)");
            } else {
                println!(
                    "{:<20}  {:>8}  {:>12}  {:>8}",
                    "NAME", "PID", "SIZE", "ATTACHED"
                );
                for s in sessions {
                    println!(
                        "{:<20}  {:>8}  {:>12}  {:>8}",
                        s.name,
                        s.pid,
                        format!("{}x{}", s.width, s.height),
                        s.attached
                    );
                }
            }
        }
        Cmd::New {
            name,
            size,
            scale,
            share_audio,
            mem_cap_mb,
            cpu_cap_pct,
            disk_quota_mb,
            fd_limit,
            api_key_id,
        } => {
            let (width, height) = parse_size(&size)?;
            let s = t
                .create_session(
                    &name,
                    width,
                    height,
                    scale,
                    share_audio,
                    mem_cap_mb,
                    cpu_cap_pct,
                    disk_quota_mb,
                    fd_limit,
                    api_key_id,
                )
                .await?;
            if json {
                print_success(
                    "new",
                    json!({ "name": s.name, "width": s.width, "height": s.height }),
                );
            } else {
                println!("{} ({}x{})", s.name, s.width, s.height);
            }
        }
        Cmd::Rm { name } => {
            t.destroy_session(&name).await?;
            if json {
                print_success("rm", json!({ "name": name, "destroyed": true }));
            } else {
                println!("destroyed {name}");
            }
        }
        Cmd::Info { name } => {
            // The transport's `get_session` filters the session list locally,
            // so a missing session is reported as `None` rather than a daemon
            // NotFound. Surface it with an explicit E_NOT_FOUND code so `info`
            // matches `rm`'s behavior instead of falling back to E_INTERNAL.
            let s = t
                .get_session(&name)
                .await?
                .ok_or_else(|| CliError::not_found(format!("no such session: {name}")))?;
            if json {
                // `scale` is not carried on the transport's SessionSummary, so
                // it is omitted here; created_at/pid are emitted only when the
                // transport populated them (remote leaves pid=0, local leaves
                // created_at empty), matching the text output's conditionals.
                let mut data = json!({
                    "name": s.name,
                    "width": s.width,
                    "height": s.height,
                    "attached": s.attached,
                });
                let obj = data.as_object_mut().expect("info data is an object");
                if !s.created_at.is_empty() {
                    obj.insert("created_at".into(), json!(s.created_at));
                }
                if s.pid != 0 {
                    obj.insert("pid".into(), json!(s.pid));
                }
                print_success("info", data);
            } else {
                println!("name      {}", s.name);
                println!("size      {}x{}", s.width, s.height);
                if !s.created_at.is_empty() {
                    println!("created   {}", s.created_at);
                }
                if s.pid != 0 {
                    println!("pid       {}", s.pid);
                }
                println!("attached  {}", s.attached);
            }
        }
        Cmd::Inject { name, ops } => {
            // A malformed `--ops` payload is caller input, not a server fault:
            // map it to E_BAD_REQUEST and show the expected op shapes so the
            // user can fix it without reading the source.
            let parsed: Vec<InjectOp> = serde_json::from_str(&ops).map_err(|e| {
                CliError::bad_request(format!(
                    "--ops must be a JSON array of inject ops ({e}). Expected shapes:\n  \
                     {{\"type\":\"key\",\"keycode\":<u32>,\"release\":<bool=false>,\"modifiers\":<u32=0>}}\n  \
                     {{\"type\":\"pointer\",\"x\":<f64>,\"y\":<f64>,\"button\":<u32=0>,\"state\":\"press\"|\"release\"}}"
                ))
            })?;
            t.inject(&name, &parsed).await?;
            if json {
                print_success("inject", json!({ "ok": true }));
            } else {
                println!("injected {} op(s) into {name}", parsed.len());
            }
        }
        Cmd::Screenshot {
            name,
            window_id,
            output,
        } => {
            let shot = t.screenshot(&name, Some(window_id)).await?;
            emit_screenshot("screenshot", &shot, output.as_deref(), json).await?;
        }
        Cmd::ScreenshotDesktop { name, output } => {
            let shot = t.screenshot(&name, None).await?;
            emit_screenshot("screenshot-desktop", &shot, output.as_deref(), json).await?;
        }
        Cmd::Windows { name, tag } => {
            let windows = t.list_windows(&name).await?;
            let windows = filter_windows_by_tag(windows, tag.as_deref());
            if json {
                print_success("windows", json!({ "windows": windows }));
            } else {
                print_windows(&windows);
            }
        }
        // Unreachable: gated by supports_remote()/dispatch above.
        other => bail!("internal: unexpected subcommand for transport: {:?}", other),
    }
    Ok(())
}

/// Emit a screenshot success envelope: the PNG bytes are base64-encoded into
/// the envelope.
fn print_screenshot_envelope(verb: &str, png: &[u8], w: u32, h: u32) {
    let b64 = base64::engine::general_purpose::STANDARD.encode(png);
    print_success(verb, json!({ "width": w, "height": h, "png_b64": b64 }));
}

/// Output a captured screenshot per the active mode.
///
/// * `--json`: the PNG is returned as `data.png_b64` in the envelope, so `-o`
///   is OPTIONAL. If `-o` is also given, the file is still written (the
///   envelope is the machine-readable result; the file is a convenience).
/// * text mode: `-o` is REQUIRED — without it there is no b64 fallback, and
///   silently dumping binary PNG to a terminal would be surprising. A missing
///   `-o` is caller input, so it surfaces as E_BAD_REQUEST.
async fn emit_screenshot(
    verb: &str,
    shot: &waymux_cli::transport::ScreenshotResult,
    output: Option<&std::path::Path>,
    json: bool,
) -> Result<()> {
    if json {
        print_screenshot_envelope(verb, &shot.png, shot.width, shot.height);
        // Honor an explicit -o even under --json: write the file too. `-o -`
        // would interleave raw PNG with the envelope on stdout, so skip the
        // stdout sentinel here and keep stdout to the single JSON line.
        if let Some(out) = output {
            if out.to_string_lossy() != "-" {
                tokio::fs::write(out, &shot.png).await?;
            }
        }
        return Ok(());
    }
    let out = output.ok_or_else(|| {
        CliError::bad_request(
            "-o/--output is required without --json (pass `-o <file>`, `-o -` for stdout, \
             or add --json to receive the PNG as data.png_b64)",
        )
    })?;
    write_screenshot(out, &shot.png, shot.width, shot.height).await
}

async fn write_screenshot(output: &std::path::Path, png: &[u8], w: u32, h: u32) -> Result<()> {
    if output.to_string_lossy() == "-" {
        use std::io::Write;
        std::io::stdout().write_all(png)?;
    } else {
        tokio::fs::write(output, png).await?;
        eprintln!("{} ({}×{}, {} bytes)", output.display(), w, h, png.len());
    }
    Ok(())
}

/// Filter a window list to those whose `tags` set contains `tag`. With `None`
/// the list is returned unchanged (the no-filter behavior). Pure helper so the
/// same filter applies before both the text table and the `--json` output.
/// Client-side `--tag` filter. `None` means no filter (every window passes).
/// `Some(t)` keeps only windows whose tag set contains `t` by exact match. An
/// empty `t` is treated as an ordinary tag value (matches only windows carrying
/// a literal empty-string tag), NOT as "no filter". Tag length is unbounded
/// here: tags are free-form and validated/stored daemon-side, so the client
/// filter intentionally applies no length cap of its own.
fn filter_windows_by_tag(windows: Vec<WindowInfo>, tag: Option<&str>) -> Vec<WindowInfo> {
    match tag {
        None => windows,
        Some(t) => windows
            .into_iter()
            .filter(|w| w.tags.iter().any(|x| x == t))
            .collect(),
    }
}

fn print_windows(windows: &[WindowInfo]) {
    if windows.is_empty() {
        println!("(no windows)");
        return;
    }
    println!(
        "{:<6}  {:<20}  {:<30}  {:<15}  {:<6}",
        "ID", "APP_ID", "TITLE", "TAGS", "FOCUS"
    );
    for w in windows {
        println!(
            "{:<6}  {:<20}  {:<30}  {:<15}  {:<6}",
            w.id,
            truncate(&w.app_id, 20),
            truncate(&w.title, 30),
            truncate(&w.tags.join(","), 15),
            w.focused
        );
    }
}

/// Subcommands that always need a raw `Connection` (events, streaming, attach,
/// recording, raw input). Local-only. When `json` is set, each verb prints a
/// success envelope on stdout (streaming verbs stay newline-delimited JSON).
async fn run_local_only(cmd: Cmd, socket: &std::path::Path, json: bool) -> Result<()> {
    let mut conn = Connection::connect(socket).await?;
    conn.hello().await?;

    match cmd {
        Cmd::Spawn {
            name,
            compositor,
            argv,
        } => {
            #[derive(serde::Deserialize)]
            struct SpawnResult {
                pid: i32,
            }
            let result: SpawnResult = conn
                .request(RequestMethod::Spawn {
                    name,
                    argv,
                    env: Default::default(),
                    compositor,
                })
                .await?
                .decode_result()
                .map_err(|e| anyhow!("decode spawn result: {e}"))?;
            if json {
                print_success("spawn", json!({ "pid": result.pid }));
            } else {
                println!("pid {}", result.pid);
            }
        }
        Cmd::Resize { name, size } => {
            let (width, height) = parse_size(&size)?;
            conn.request(RequestMethod::Resize {
                name: name.clone(),
                width,
                height,
            })
            .await?;
            if json {
                print_success(
                    "resize",
                    json!({ "name": name, "width": width, "height": height }),
                );
            } else {
                println!("resized {name} to {width}x{height}");
            }
        }
        Cmd::Tag {
            name,
            window_id,
            tags,
        } => {
            // On an unknown window the daemon returns NotFound; the `?` bubbles
            // it up to the JSON error-envelope wrapper in main (E_NOT_FOUND).
            conn.request(RequestMethod::TagWindow {
                name: name.clone(),
                window_id,
                tags: tags.clone(),
            })
            .await?;
            if json {
                print_success(
                    "tag",
                    json!({ "name": name, "window_id": window_id, "tags": tags }),
                );
            } else {
                println!("tagged window {window_id} in {name}: {}", tags.join(", "));
            }
        }
        Cmd::Idle {
            name,
            quiet_ms,
            timeout_ms,
        } => {
            #[derive(serde::Deserialize)]
            struct IdleResult {
                idle: bool,
            }
            let r: IdleResult = conn
                .request(RequestMethod::WaitForIdle {
                    name,
                    quiet_ms,
                    timeout_ms,
                })
                .await?
                .decode_result()
                .map_err(|e| anyhow!("decode wait_for_idle: {e}"))?;
            if json {
                print_success("idle", json!({ "idle": r.idle }));
                if !r.idle {
                    std::process::exit(1);
                }
            } else if r.idle {
                println!("idle");
            } else {
                eprintln!("busy (timeout)");
                std::process::exit(1);
            }
        }
        Cmd::Wait {
            name,
            app_id,
            title,
            tag,
            pid,
            nth,
            timeout_ms,
        } => {
            let matched =
                wait_for_window(&mut conn, &name, app_id, title, tag, pid, nth, timeout_ms).await?;
            match matched {
                Some(w) => {
                    if json {
                        print_success(
                            "wait",
                            json!({ "id": w.id, "app_id": w.app_id, "title": w.title }),
                        );
                    } else {
                        println!("{}\t{}\t{}", w.id, w.app_id, w.title);
                    }
                }
                None => {
                    if json {
                        // Timeout: data is null, exit non-zero (preserved).
                        print_success("wait", Value::Null);
                    } else {
                        eprintln!("timeout");
                    }
                    std::process::exit(1);
                }
            }
        }
        Cmd::Key {
            name,
            keycode,
            release,
            modifiers,
        } => {
            conn.request(RequestMethod::InjectKey {
                name: name.clone(),
                keycode,
                state: if release {
                    KeyState::Released
                } else {
                    KeyState::Pressed
                },
                modifiers,
            })
            .await?;
            if json {
                print_success("key", json!({ "ok": true }));
            } else {
                println!(
                    "sent keycode {keycode} ({}) to {name}",
                    if release { "release" } else { "press" }
                );
            }
        }
        Cmd::Click { name, x, y, button } => {
            if button == 0 {
                conn.request(RequestMethod::InjectPointer {
                    name: name.clone(),
                    x,
                    y,
                    button: 0,
                    state: KeyState::Released,
                    axis_x: 0.0,
                    axis_y: 0.0,
                    // CLI surface doesn't yet expose window_id / content;
                    // pass v1-compatible defaults.
                    window_id: None,
                    content: false,
                })
                .await?;
            } else {
                conn.request(RequestMethod::InjectPointer {
                    name: name.clone(),
                    x,
                    y,
                    button,
                    state: KeyState::Pressed,
                    axis_x: 0.0,
                    axis_y: 0.0,
                    window_id: None,
                    content: false,
                })
                .await?;
                conn.request(RequestMethod::InjectPointer {
                    name: name.clone(),
                    x,
                    y,
                    button,
                    state: KeyState::Released,
                    axis_x: 0.0,
                    axis_y: 0.0,
                    window_id: None,
                    content: false,
                })
                .await?;
            }
            if json {
                print_success("click", json!({ "ok": true }));
            } else {
                println!("clicked button {button} at ({x}, {y}) in {name}");
            }
        }
        Cmd::Logs {
            name,
            follow,
            settle_ms,
        } => {
            conn.request(RequestMethod::Subscribe {
                topics: vec![format!("logs:{}", name)],
            })
            .await?;
            let settle = std::time::Duration::from_millis(settle_ms);
            loop {
                let read = if follow {
                    conn.read_raw_frame().await.map(Some)
                } else {
                    match tokio::time::timeout(settle, conn.read_raw_frame()).await {
                        Ok(Ok(f)) => Ok(Some(f)),
                        Ok(Err(e)) => Err(e),
                        Err(_) => break,
                    }
                };
                let frame = match read? {
                    Some(f) => f,
                    None => break,
                };
                if let Ok(ev) = rmp_serde::from_slice::<Event>(&frame[4..]) {
                    if let EventBody::Log {
                        name: n,
                        stream,
                        text,
                    } = ev.body
                    {
                        if n == name {
                            if json {
                                // NDJSON: one object per log line, not wrapped
                                // in the success envelope.
                                println!("{}", json!({ "stream": stream, "text": text }));
                            } else {
                                println!("[{}] {}", stream, text);
                            }
                        }
                    }
                }
            }
        }
        Cmd::Events { name, topics } => {
            conn.request(RequestMethod::Subscribe {
                topics: topics.clone(),
            })
            .await?;
            eprintln!(
                "subscribed to {:?}; streaming events (Ctrl-C to stop)",
                topics
            );
            loop {
                let frame = conn.read_raw_frame().await?;
                if let Ok(ev) = rmp_serde::from_slice::<Event>(&frame[4..]) {
                    let belongs = match &ev.body {
                        EventBody::SessionCreated { name: n }
                        | EventBody::SessionDestroyed { name: n, .. }
                        | EventBody::SessionCrashed { name: n, .. }
                        | EventBody::Occluded { name: n, .. }
                        | EventBody::ChildExited { name: n, .. }
                        | EventBody::WindowCreated { name: n, .. }
                        | EventBody::WindowDestroyed { name: n, .. }
                        | EventBody::WindowChanged { name: n, .. }
                        | EventBody::Damage { name: n, .. }
                        | EventBody::Log { name: n, .. } => n == &name,
                    };
                    if belongs {
                        println!("{}", serde_json::to_string(&ev)?);
                    }
                }
            }
        }
        Cmd::Attach { name } => {
            let result: AttachResult = conn
                .request(RequestMethod::Attach { name: name.clone() })
                .await?
                .decode_result()
                .map_err(|e| anyhow!("decode attach result: {e}"))?;
            if json {
                // Under --json we surface the socket path rather than spawning
                // the interactive attach client (which would never produce a
                // machine-readable result).
                print_success(
                    "attach",
                    json!({ "attach_socket_path": result.attach_socket_path }),
                );
            } else {
                let attach_bin = find_attach_binary();
                tokio::process::Command::new(&attach_bin)
                    .arg(&result.attach_socket_path)
                    .spawn()
                    .with_context(|| format!("launch {}", attach_bin.display()))?
                    .wait()
                    .await?;
            }
        }
        Cmd::Detach { name } => {
            conn.request(RequestMethod::Detach { name: name.clone() })
                .await?;
            if json {
                print_success("detach", json!({ "detached": true }));
            } else {
                println!("detached {name}");
            }
        }
        Cmd::Record { action } => match action {
            RecordCmd::Start {
                name,
                output,
                codec,
                secondary_codec,
                mode,
                min_fps,
            } => {
                // Validate any user-supplied output path CLI-side BEFORE
                // sending: a path outside the recordings sandbox is caller
                // input, so it must surface as E_BAD_REQUEST rather than the
                // E_INTERNAL the session-side backstop would otherwise produce.
                // The session keeps its own check as defense-in-depth.
                if let Some(p) = &output {
                    validate_recording_path(p)?;
                }
                let path = output.map(|p| p.display().to_string());
                let codec = codec.map(RecordingCodec::from);
                let secondary_codec = secondary_codec.map(RecordingCodec::from);
                let mode = mode.map(waymux_protocol::CaptureMode::from);
                let r: SessionCtlRecordStarted = conn
                    .request(RequestMethod::RecordStart {
                        name,
                        path,
                        codec,
                        secondary_codec,
                        mode,
                        min_fps,
                    })
                    .await?
                    .decode_result()
                    .map_err(|e| anyhow!("decode record_start: {e}"))?;
                if json {
                    let mut data = json!({ "path": r.path });
                    if let Some(secondary) = r.secondary_path {
                        data.as_object_mut()
                            .expect("record data is an object")
                            .insert("secondary_path".into(), json!(secondary));
                    }
                    print_success("record", data);
                } else {
                    println!("{}", r.path);
                    if let Some(secondary) = r.secondary_path {
                        println!("{}", secondary);
                    }
                }
            }
            RecordCmd::Stop { name } => {
                conn.request(RequestMethod::RecordStop { name }).await?;
                if json {
                    print_success("record", json!({ "stopped": true }));
                } else {
                    println!("recording stopped");
                }
            }
            RecordCmd::Status { name } => {
                let resp: RecordStatusResponse = conn
                    .request(RequestMethod::RecordStatus { name })
                    .await?
                    .decode_result()
                    .map_err(|e| anyhow!("decode record_status: {e}"))?;
                if json {
                    print_success(
                        "record",
                        json!({
                            "recording": resp.recording,
                            "path": resp.path,
                            "secondary_path": resp.secondary_path,
                            "codec": resp.codec,
                        }),
                    );
                } else if resp.recording {
                    let mut line = String::from("recording: yes");
                    if let Some(path) = &resp.path {
                        line.push_str(&format!("  path={path}"));
                    }
                    if let Some(secondary) = &resp.secondary_path {
                        line.push_str(&format!("  secondary_path={secondary}"));
                    }
                    if let Some(codec) = &resp.codec {
                        line.push_str(&format!("  codec={codec}"));
                    }
                    println!("{line}");
                } else {
                    println!("recording: no");
                }
            }
        },
        Cmd::Viewer { sub } => match sub {
            ViewerSub::Start { name, bind, port } => {
                let resp: ViewerStarted = conn
                    .request(RequestMethod::ViewerStart {
                        session: name,
                        bind,
                        port,
                    })
                    .await?
                    .decode_result()
                    .map_err(|e| anyhow!("decode viewer_start: {e}"))?;
                if json {
                    print_success("viewer", json!({ "url": resp.url }));
                } else {
                    println!("{}", resp.url);
                }
            }
            ViewerSub::Stop { name } => {
                conn.request(RequestMethod::ViewerStop {
                    session: name.clone(),
                })
                .await?;
                if json {
                    print_success("viewer", json!({ "stopped": true }));
                } else {
                    println!("viewer stopped for {name}");
                }
            }
            ViewerSub::Status { name } => {
                let resp: ViewerStatusResponse = conn
                    .request(RequestMethod::ViewerStatus { session: name })
                    .await?
                    .decode_result()
                    .map_err(|e| anyhow!("decode viewer_status: {e}"))?;
                if json {
                    // url is null when no viewer is active.
                    print_success("viewer", json!({ "url": resp.url }));
                } else if let Some(url) = resp.url {
                    println!("{}", url);
                }
            }
            // `viewer token` is a pure local mint — intercepted in `dispatch`
            // before any daemon connection, so it never reaches run_local_only.
            ViewerSub::Token { .. } => {
                unreachable!("viewer token is handled in dispatch() without a daemon")
            }
        },
        // These are routed via run_with_transport above; reaching them here
        // means the dispatcher diverged from supports_remote().
        Cmd::Ls
        | Cmd::New { .. }
        | Cmd::Rm { .. }
        | Cmd::Info { .. }
        | Cmd::Inject { .. }
        | Cmd::Screenshot { .. }
        | Cmd::ScreenshotDesktop { .. }
        | Cmd::Windows { .. }
        | Cmd::Login { .. } => {
            unreachable!("dispatched via transport path");
        }
        // `serve` is intercepted in `dispatch` and exec-replaces the process
        // before reaching any connect path, so it never arrives here.
        Cmd::Serve => unreachable!("serve is handled in dispatch via exec"),
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn wait_for_window(
    conn: &mut Connection,
    name: &str,
    app_id: Option<String>,
    title: Option<String>,
    tag: Option<String>,
    pid: Option<i32>,
    nth: Option<usize>,
    timeout_ms: u32,
) -> Result<Option<WindowInfo>> {
    fn matches(
        w: &WindowInfo,
        app_id: &Option<String>,
        title: &Option<String>,
        tag: &Option<String>,
        pid: &Option<i32>,
    ) -> bool {
        if let Some(a) = app_id {
            if &w.app_id != a {
                return false;
            }
        }
        if let Some(t) = title {
            if &w.title != t {
                return false;
            }
        }
        if let Some(t) = tag {
            if !w.tags.iter().any(|x| x == t) {
                return false;
            }
        }
        if let Some(p) = pid {
            if &w.pid != p {
                return false;
            }
        }
        true
    }

    conn.request(RequestMethod::Subscribe {
        topics: vec!["windows".into()],
    })
    .await?;

    let resolve = |windows: Vec<WindowInfo>| -> Option<WindowInfo> {
        let filtered: Vec<WindowInfo> = windows
            .into_iter()
            .filter(|w| matches(w, &app_id, &title, &tag, &pid))
            .collect();
        match nth {
            None => filtered.into_iter().next(),
            Some(n) => filtered.into_iter().nth(n),
        }
    };

    let windows: Vec<WindowInfo> = conn
        .request(RequestMethod::ListWindows {
            name: name.to_string(),
        })
        .await?
        .decode_result()
        .map_err(|e| anyhow!("decode list_windows: {e}"))?;
    if let Some(w) = resolve(windows) {
        return Ok(Some(w));
    }

    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms as u64);
    loop {
        let remaining = match deadline.checked_duration_since(tokio::time::Instant::now()) {
            Some(r) if r > std::time::Duration::ZERO => r,
            _ => return Ok(None),
        };
        let frame = match tokio::time::timeout(remaining, conn.read_raw_frame()).await {
            Ok(Ok(f)) => f,
            _ => return Ok(None),
        };
        let Ok(ev) = rmp_serde::from_slice::<Event>(&frame[4..]) else {
            continue;
        };
        let ev_session = match &ev.body {
            EventBody::WindowCreated { name: n, .. }
            | EventBody::WindowDestroyed { name: n, .. }
            | EventBody::WindowChanged { name: n, .. } => Some(n.clone()),
            _ => None,
        };
        if ev_session.as_deref() != Some(name) {
            continue;
        }
        let windows: Vec<WindowInfo> = conn
            .request(RequestMethod::ListWindows {
                name: name.to_string(),
            })
            .await?
            .decode_result()
            .map_err(|e| anyhow!("decode list_windows: {e}"))?;
        if let Some(w) = resolve(windows) {
            return Ok(Some(w));
        }
    }
}
// `Connection` lives in connection.rs after the Transport refactor (task #91).

#[cfg(test)]
mod json_envelope_tests {
    use super::{
        error_code_for, error_envelope, known_error_code, success_envelope, verb_name, Args as Cli,
    };
    use clap::Parser;
    use serde_json::{json, Value};

    #[test]
    fn success_envelope_shape() {
        let env = success_envelope("ls", json!({ "sessions": [] }));
        assert_eq!(env["ok"], json!(true));
        assert_eq!(env["verb"], json!("ls"));
        assert_eq!(env["data"]["sessions"], json!([]));
        // No `error` key on success.
        assert!(env.get("error").is_none());
    }

    #[test]
    fn success_envelope_carries_arbitrary_data() {
        let env = success_envelope(
            "new",
            json!({ "name": "eagle", "width": 1920, "height": 1080 }),
        );
        assert_eq!(env["ok"], json!(true));
        assert_eq!(env["verb"], json!("new"));
        assert_eq!(env["data"]["name"], json!("eagle"));
        assert_eq!(env["data"]["width"], json!(1920));
        assert_eq!(env["data"]["height"], json!(1080));
    }

    #[test]
    fn error_envelope_shape() {
        let env = error_envelope("info", "E_NOT_FOUND", "no such session: eagle");
        assert_eq!(env["ok"], json!(false));
        assert_eq!(env["verb"], json!("info"));
        assert_eq!(env["error"]["code"], json!("E_NOT_FOUND"));
        assert_eq!(env["error"]["message"], json!("no such session: eagle"));
        assert_eq!(env["error"]["detail"], Value::Null);
        // No `data` key on error.
        assert!(env.get("data").is_none());
    }

    #[test]
    fn error_envelope_serializes_to_single_line() {
        let env = error_envelope("ls", "E_INTERNAL", "boom");
        let s = env.to_string();
        assert!(
            !s.contains('\n'),
            "envelope must serialize to a single line"
        );
        // Round-trip back to a Value to assert content regardless of the
        // serializer's key ordering.
        let parsed: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed["ok"], json!(false));
        assert_eq!(parsed["error"]["code"], json!("E_INTERNAL"));
    }

    /// RPC errors surface as `"{ErrorCode:?}: {message}"`; verify the Debug
    /// prefixes map to the wire `E_...` codes.
    #[test]
    fn error_code_maps_rpc_debug_prefixes() {
        let cases = [
            ("NotFound: no such session", "E_NOT_FOUND"),
            ("AlreadyExists: session exists", "E_ALREADY_EXISTS"),
            ("ProtoVersion: mismatch", "E_PROTO_VERSION"),
            ("NoRenderNode: none", "E_NO_RENDER_NODE"),
            ("ResizeRejected: too big", "E_RESIZE_REJECTED"),
            ("Backpressure: slow down", "E_BACKPRESSURE"),
            ("NotImplemented: nope", "E_NOT_IMPLEMENTED"),
            ("BadRequest: argv too large", "E_BAD_REQUEST"),
            ("Internal: kaboom", "E_INTERNAL"),
        ];
        for (msg, expected) in cases {
            let err = anyhow::anyhow!("{msg}");
            assert_eq!(
                error_code_for(&err).as_str(),
                expected,
                "for message {msg:?}"
            );
        }
    }

    /// Local `bail!`s and connection failures (no recognizable prefix) fall
    /// back to E_INTERNAL.
    #[test]
    fn error_code_falls_back_to_internal() {
        let err = anyhow::anyhow!("Connection refused (os error 111)");
        assert_eq!(error_code_for(&err).as_str(), "E_INTERNAL");
        let err2 = anyhow::anyhow!("expected WxH, got 1920");
        assert_eq!(error_code_for(&err2).as_str(), "E_INTERNAL");
    }

    /// A daemon code this CLI does not recognize (a wire-format divergence)
    /// must be SURFACED as `E_UNKNOWN:<name>`, not silently collapsed to
    /// E_INTERNAL, otherwise the drift is invisible.
    #[test]
    fn unknown_daemon_code_is_surfaced() {
        // A future ErrorCode variant the CLI has not learned yet.
        let err = anyhow::anyhow!("SomeNewCode: a brand-new failure");
        assert_eq!(error_code_for(&err).as_str(), "E_UNKNOWN:SomeNewCode");
    }

    /// Contract test: every `waymux-protocol::ErrorCode` variant's Debug name
    /// is recognized by the CLI's known-code mapping, so the match arms cannot
    /// silently fall out of sync with the protocol enum. We enumerate the enum
    /// explicitly (the compiler forces this list to stay exhaustive: adding a
    /// variant to the protocol breaks this test until it is handled here).
    #[test]
    fn known_codes_stay_in_sync_with_protocol_enum() {
        use waymux_protocol::ErrorCode;
        // Exhaustive: adding a variant in waymux-protocol makes this match
        // fail to compile until the new variant is added below AND to
        // `known_error_code`.
        let all = [
            ErrorCode::NotFound,
            ErrorCode::AlreadyExists,
            ErrorCode::ProtoVersion,
            ErrorCode::NoRenderNode,
            ErrorCode::ResizeRejected,
            ErrorCode::Backpressure,
            ErrorCode::NotImplemented,
            ErrorCode::BadRequest,
            ErrorCode::Internal,
            ErrorCode::Unknown,
        ];
        // Compile-time exhaustiveness guard: if a new variant lands, this
        // match stops compiling, forcing an update here and in the mapping.
        for code in &all {
            match code {
                ErrorCode::NotFound
                | ErrorCode::AlreadyExists
                | ErrorCode::ProtoVersion
                | ErrorCode::NoRenderNode
                | ErrorCode::ResizeRejected
                | ErrorCode::Backpressure
                | ErrorCode::NotImplemented
                | ErrorCode::BadRequest
                | ErrorCode::Internal
                | ErrorCode::Unknown => {}
            }
        }
        for code in &all {
            // The Debug name is exactly what arrives on the wire prefix.
            let debug_name = format!("{code:?}");
            assert!(
                known_error_code(&debug_name).is_some(),
                "ErrorCode::{debug_name} has no known_error_code mapping; \
                 the CLI's error-code match has drifted from the protocol enum"
            );
        }
    }

    /// `--json` parses as a global flag positioned before OR after the verb,
    /// across several subcommands.
    #[test]
    fn json_is_a_global_flag() {
        // Before the verb.
        let cli = Cli::try_parse_from(["waymux", "--json", "ls"]).unwrap();
        assert!(cli.json);
        assert!(matches!(cli.command, crate::Cmd::Ls));

        // After the verb (clap global args accept either position).
        let cli = Cli::try_parse_from(["waymux", "ls", "--json"]).unwrap();
        assert!(cli.json);

        // With a verb that takes args.
        let cli =
            Cli::try_parse_from(["waymux", "--json", "new", "eagle", "--size", "800x600"]).unwrap();
        assert!(cli.json);
        assert!(matches!(cli.command, crate::Cmd::New { .. }));

        // info, rm, windows, viewer.
        assert!(
            Cli::try_parse_from(["waymux", "--json", "info", "eagle"])
                .unwrap()
                .json
        );
        assert!(
            Cli::try_parse_from(["waymux", "rm", "eagle", "--json"])
                .unwrap()
                .json
        );
        assert!(
            Cli::try_parse_from(["waymux", "--json", "windows", "eagle"])
                .unwrap()
                .json
        );
        assert!(
            Cli::try_parse_from(["waymux", "--json", "viewer", "start", "eagle"])
                .unwrap()
                .json
        );
    }

    /// Without `--json` the flag defaults to false (default text output path).
    #[test]
    fn json_defaults_off() {
        let cli = Cli::try_parse_from(["waymux", "ls"]).unwrap();
        assert!(!cli.json);
    }

    #[test]
    fn verb_names_are_stable() {
        let cli = Cli::try_parse_from(["waymux", "ls"]).unwrap();
        assert_eq!(verb_name(&cli.command), "ls");
        let cli =
            Cli::try_parse_from(["waymux", "screenshot-desktop", "eagle", "-o", "x.png"]).unwrap();
        assert_eq!(verb_name(&cli.command), "screenshot-desktop");
    }
}

/// Tests for the 2026-06-18 e2e-triage caller-error fixes (bugs 3, 4, 5, 9):
/// caller-input errors must surface with a specific wire code, not E_INTERNAL.
#[cfg(test)]
mod caller_error_tests {
    use super::{error_code_for, validate_recording_path, Args as Cli, CliError};
    use clap::Parser;

    /// Bug 3: a locally-detected NotFound (e.g. `info <missing>`, which the
    /// transport reports as `None`) carries E_NOT_FOUND, matching `rm`, rather
    /// than falling back to E_INTERNAL.
    #[test]
    fn cli_not_found_maps_to_e_not_found() {
        let err = CliError::not_found("no such session: nope");
        assert_eq!(error_code_for(&err).as_str(), "E_NOT_FOUND");
        // The message is preserved verbatim.
        assert_eq!(err.to_string(), "no such session: nope");
    }

    /// Bug 4/8: a malformed `--ops` payload is caller input and maps to
    /// E_BAD_REQUEST, with a message that shows the expected op shapes.
    #[test]
    fn bad_ops_json_maps_to_bad_request_with_shapes() {
        // Reproduce the inject path's parse-and-map error construction.
        let ops = "not json";
        let err = serde_json::from_str::<Vec<crate::InjectOp>>(ops)
            .map_err(|e| {
                CliError::bad_request(format!(
                    "--ops must be a JSON array of inject ops ({e}). Expected shapes:\n  \
                     {{\"type\":\"key\",\"keycode\":<u32>,\"release\":<bool=false>,\"modifiers\":<u32=0>}}\n  \
                     {{\"type\":\"pointer\",\"x\":<f64>,\"y\":<f64>,\"button\":<u32=0>,\"state\":\"press\"|\"release\"}}"
                ))
            })
            .unwrap_err();
        assert_eq!(error_code_for(&err).as_str(), "E_BAD_REQUEST");
        let msg = err.to_string();
        assert!(msg.contains("\"type\":\"key\""), "missing key shape: {msg}");
        assert!(
            msg.contains("\"type\":\"pointer\""),
            "missing pointer shape: {msg}"
        );
    }

    /// Bug 5: a record path outside the recordings sandbox is caller input and
    /// is rejected CLI-side as E_BAD_REQUEST (the session keeps its own
    /// backstop). A path inside the sandbox validates Ok. Reads the real HOME
    /// (no env mutation, so no cross-test races) and derives both cases from
    /// the same `recordings_dir()` the validator uses.
    #[test]
    fn record_path_outside_sandbox_is_bad_request() {
        let allowed = super::recordings_dir().expect("HOME set in test env");

        // Outside the sandbox -> E_BAD_REQUEST. `/waymux-e2e-not-the-sandbox`
        // is absolute and cannot be a prefix of the recordings dir.
        let err =
            validate_recording_path(std::path::Path::new("/waymux-e2e-not-the-sandbox/x.mkv"))
                .unwrap_err();
        assert_eq!(error_code_for(&err).as_str(), "E_BAD_REQUEST");
        assert!(
            err.to_string().contains("must be inside"),
            "expected sandbox message, got: {err}"
        );

        // A relative path -> E_BAD_REQUEST.
        let rel = validate_recording_path(std::path::Path::new("x.mkv")).unwrap_err();
        assert_eq!(error_code_for(&rel).as_str(), "E_BAD_REQUEST");

        // A `..` traversal anchored in the sandbox -> E_BAD_REQUEST.
        let dotdot = validate_recording_path(&allowed.join("../../escape.mkv")).unwrap_err();
        assert_eq!(error_code_for(&dotdot).as_str(), "E_BAD_REQUEST");

        // Inside the sandbox -> Ok.
        assert!(validate_recording_path(&allowed.join("ok.mkv")).is_ok());
    }

    /// Bug 9: `--json screenshot <s> <wid>` parses WITHOUT `-o` (the PNG comes
    /// back as data.png_b64), and `screenshot-desktop` likewise. With `-o` it
    /// still parses. Non-json without `-o` also parses at the clap layer (the
    /// requirement is enforced at runtime as E_BAD_REQUEST, not by clap).
    #[test]
    fn screenshot_output_is_optional_for_clap() {
        // Per-window, json, no -o.
        let cli = Cli::try_parse_from(["waymux", "--json", "screenshot", "s", "7"]).unwrap();
        assert!(cli.json);
        match cli.command {
            crate::Cmd::Screenshot {
                name,
                window_id,
                output,
            } => {
                assert_eq!(name, "s");
                assert_eq!(window_id, 7);
                assert_eq!(output, None);
            }
            _ => panic!("wrong subcommand parsed"),
        }

        // Desktop, json, no -o.
        let cli = Cli::try_parse_from(["waymux", "--json", "screenshot-desktop", "s"]).unwrap();
        match cli.command {
            crate::Cmd::ScreenshotDesktop { name, output } => {
                assert_eq!(name, "s");
                assert_eq!(output, None);
            }
            _ => panic!("wrong subcommand parsed"),
        }

        // Still accepts -o when provided.
        let cli = Cli::try_parse_from(["waymux", "screenshot", "s", "7", "-o", "out.png"]).unwrap();
        match cli.command {
            crate::Cmd::Screenshot { output, .. } => {
                assert_eq!(output.as_deref(), Some(std::path::Path::new("out.png")));
            }
            _ => panic!("wrong subcommand parsed"),
        }
    }
}

#[cfg(test)]
mod viewer_cli_tests {
    use super::{success_envelope, Args as Cli};
    use clap::Parser;
    use serde_json::{json, Value};
    use waymux_protocol::RecordStatusResponse;

    #[test]
    fn parses_viewer_start_default_bind() {
        let cli = Cli::try_parse_from(["waymux", "viewer", "start", "eagle"]).unwrap();
        match cli.command {
            crate::Cmd::Viewer {
                sub: crate::ViewerSub::Start { name, bind, port },
            } => {
                assert_eq!(name, "eagle");
                assert_eq!(bind, None);
                assert_eq!(port, None);
            }
            _ => panic!("wrong subcommand parsed"),
        }
    }

    #[test]
    fn parses_viewer_start_with_bind() {
        let cli =
            Cli::try_parse_from(["waymux", "viewer", "start", "eagle", "--bind", "10.42.0.2"])
                .unwrap();
        if let crate::Cmd::Viewer {
            sub: crate::ViewerSub::Start { name, bind, port },
        } = cli.command
        {
            assert_eq!(name, "eagle");
            assert_eq!(bind, Some("10.42.0.2".into()));
            assert_eq!(port, None);
        } else {
            panic!();
        }
    }

    #[test]
    fn parses_viewer_token_default() {
        let cli = Cli::try_parse_from(["waymux", "viewer", "token"]).unwrap();
        match cli.command {
            crate::Cmd::Viewer {
                sub:
                    crate::ViewerSub::Token {
                        session,
                        exp_secs,
                        sub,
                    },
            } => {
                assert_eq!(session, None);
                assert_eq!(exp_secs, waymux_cli::viewer_token::DEFAULT_EXP_SECS);
                assert_eq!(sub, None);
            }
            _ => panic!("wrong subcommand parsed"),
        }
    }

    #[test]
    fn parses_viewer_token_with_session_and_flags() {
        let session = "22222222-2222-2222-2222-222222222222";
        let subv = "11111111-1111-1111-1111-111111111111";
        let cli = Cli::try_parse_from([
            "waymux",
            "viewer",
            "token",
            session,
            "--exp-secs",
            "300",
            "--sub",
            subv,
        ])
        .unwrap();
        match cli.command {
            crate::Cmd::Viewer {
                sub:
                    crate::ViewerSub::Token {
                        session: s,
                        exp_secs,
                        sub,
                    },
            } => {
                assert_eq!(s.as_deref(), Some(session));
                assert_eq!(exp_secs, 300);
                assert_eq!(sub.as_deref(), Some(subv));
            }
            _ => panic!("wrong subcommand parsed"),
        }
    }

    /// `--ttl` is an alias for `--exp-secs`.
    #[test]
    fn parses_viewer_token_ttl_alias() {
        let cli = Cli::try_parse_from(["waymux", "viewer", "token", "--ttl", "60"]).unwrap();
        match cli.command {
            crate::Cmd::Viewer {
                sub: crate::ViewerSub::Token { exp_secs, .. },
            } => assert_eq!(exp_secs, 60),
            _ => panic!("wrong subcommand parsed"),
        }
    }

    /// task #109: --port flag wires through to the RPC.
    #[test]
    fn parses_viewer_start_with_port() {
        let cli = Cli::try_parse_from([
            "waymux", "viewer", "start", "kde", "--bind", "0.0.0.0", "--port", "8080",
        ])
        .unwrap();
        if let crate::Cmd::Viewer {
            sub: crate::ViewerSub::Start { name, bind, port },
        } = cli.command
        {
            assert_eq!(name, "kde");
            assert_eq!(bind, Some("0.0.0.0".into()));
            assert_eq!(port, Some(8080));
        } else {
            panic!();
        }
    }

    #[test]
    fn parses_record_status() {
        let cli = Cli::try_parse_from(["waymux", "record", "status", "eagle"]).unwrap();
        match cli.command {
            crate::Cmd::Record {
                action: crate::RecordCmd::Status { name },
            } => assert_eq!(name, "eagle"),
            _ => panic!("wrong subcommand parsed"),
        }
    }

    /// The `record status --json` envelope carries the recording state under
    /// the `data` object. Mirrors the shape `print_success("record", …)`
    /// emits in the RecordCmd::Status handler.
    #[test]
    fn record_status_json_envelope_shape() {
        let resp = RecordStatusResponse {
            recording: true,
            path: Some("/run/user/1000/waymux/recordings/e2e.mkv".into()),
            secondary_path: None,
            codec: Some("ffv1".into()),
        };
        let env = success_envelope(
            "record",
            json!({
                "recording": resp.recording,
                "path": resp.path,
                "secondary_path": resp.secondary_path,
                "codec": resp.codec,
            }),
        );
        assert_eq!(env["ok"], json!(true));
        assert_eq!(env["verb"], json!("record"));
        assert_eq!(env["data"]["recording"], json!(true));
        assert_eq!(
            env["data"]["path"],
            json!("/run/user/1000/waymux/recordings/e2e.mkv")
        );
        assert_eq!(env["data"]["secondary_path"], Value::Null);
        assert_eq!(env["data"]["codec"], json!("ffv1"));

        // The inactive shape: recording=false, the rest null.
        let inactive = RecordStatusResponse::default();
        let env = success_envelope(
            "record",
            json!({
                "recording": inactive.recording,
                "path": inactive.path,
                "secondary_path": inactive.secondary_path,
                "codec": inactive.codec,
            }),
        );
        assert_eq!(env["data"]["recording"], json!(false));
        assert_eq!(env["data"]["path"], Value::Null);
        assert_eq!(env["data"]["codec"], Value::Null);
    }

    /// OSS hygiene: the binary must not bake the private hosted SaaS URL
    /// as the `login` default. The default is a neutral localhost
    /// placeholder; remote use requires an explicit `--base-url`.
    #[test]
    fn login_base_url_default_is_neutral() {
        let cli = Cli::try_parse_from(["waymux", "login", "--api-key", "wmx_test"]).unwrap();
        match cli.command {
            crate::Cmd::Login { api_key, base_url } => {
                assert_eq!(api_key.as_deref(), Some("wmx_test"));
                assert_ne!(base_url, "https://waymux.cloud");
                assert!(
                    !base_url.contains("waymux.cloud"),
                    "OSS build must not default to the hosted SaaS host, got {base_url}"
                );
                assert_eq!(base_url, "http://localhost:8080");
            }
            _ => panic!("wrong subcommand parsed"),
        }
    }
}

#[cfg(test)]
mod tag_cli_tests {
    use super::{filter_windows_by_tag, Args as Cli};
    use clap::Parser;
    use waymux_protocol::{Rect, WindowInfo};

    /// `waymux tag s 5 a b c` parses to the Tag variant with the positional
    /// vararg collecting all three tags.
    #[test]
    fn parses_tag_with_multiple_tags() {
        let cli = Cli::try_parse_from(["waymux", "tag", "s", "5", "a", "b", "c"]).unwrap();
        match cli.command {
            crate::Cmd::Tag {
                name,
                window_id,
                tags,
            } => {
                assert_eq!(name, "s");
                assert_eq!(window_id, 5);
                assert_eq!(tags, vec!["a", "b", "c"]);
            }
            _ => panic!("wrong subcommand parsed"),
        }
    }

    /// `tag` requires at least one tag (the positional vararg is `required`).
    #[test]
    fn tag_requires_at_least_one_tag() {
        assert!(Cli::try_parse_from(["waymux", "tag", "s", "5"]).is_err());
    }

    /// `windows s --tag foo` parses with the optional filter populated, and
    /// `windows s` leaves it `None` (unfiltered).
    #[test]
    fn parses_windows_with_tag_filter() {
        let cli = Cli::try_parse_from(["waymux", "windows", "s", "--tag", "foo"]).unwrap();
        match cli.command {
            crate::Cmd::Windows { name, tag } => {
                assert_eq!(name, "s");
                assert_eq!(tag.as_deref(), Some("foo"));
            }
            _ => panic!("wrong subcommand parsed"),
        }

        let cli = Cli::try_parse_from(["waymux", "windows", "s"]).unwrap();
        match cli.command {
            crate::Cmd::Windows { name, tag } => {
                assert_eq!(name, "s");
                assert_eq!(tag, None);
            }
            _ => panic!("wrong subcommand parsed"),
        }
    }

    fn win(id: u32, tags: &[&str]) -> WindowInfo {
        WindowInfo {
            id,
            app_id: format!("app{id}"),
            title: format!("title{id}"),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            geometry: Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            focused: false,
            pid: 0,
            content_rect: None,
        }
    }

    /// `--tag` returns only windows whose tag set contains the tag.
    #[test]
    fn filter_returns_matching_subset() {
        let windows = vec![
            win(1, &["foo", "bar"]),
            win(2, &["baz"]),
            win(3, &["foo"]),
            win(4, &[]),
        ];
        let filtered = filter_windows_by_tag(windows, Some("foo"));
        let ids: Vec<u32> = filtered.iter().map(|w| w.id).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    /// `None` is the no-filter path: the list comes back unchanged.
    #[test]
    fn filter_none_is_identity() {
        let windows = vec![win(1, &["foo"]), win(2, &[])];
        let filtered = filter_windows_by_tag(windows, None);
        let ids: Vec<u32> = filtered.iter().map(|w| w.id).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    /// A tag with no matches yields an empty list (not the full list).
    #[test]
    fn filter_no_match_is_empty() {
        let windows = vec![win(1, &["foo"]), win(2, &["bar"])];
        let filtered = filter_windows_by_tag(windows, Some("nope"));
        assert!(filtered.is_empty());
    }

    /// `Some("")` (an empty filter tag) is a normal exact-match filter: it
    /// matches only windows that literally carry an empty-string tag. We do NOT
    /// special-case it as "no filter" (that is `None`'s job): an empty string
    /// is a distinct, if unusual, tag value. With no window carrying one, the
    /// result is empty.
    #[test]
    fn filter_empty_tag_matches_only_empty_string_tags() {
        let windows = vec![win(1, &["foo"]), win(2, &[""]), win(3, &[])];
        let filtered = filter_windows_by_tag(windows, Some(""));
        let ids: Vec<u32> = filtered.iter().map(|w| w.id).collect();
        assert_eq!(
            ids,
            vec![2],
            "only the window with an empty-string tag matches"
        );
    }

    /// Duplicate tags on a window do not break the filter or duplicate the
    /// window in the result (each window appears at most once).
    #[test]
    fn filter_handles_duplicate_tags_without_duplicating_window() {
        let windows = vec![win(1, &["foo", "foo", "foo"]), win(2, &["bar"])];
        let filtered = filter_windows_by_tag(windows, Some("foo"));
        let ids: Vec<u32> = filtered.iter().map(|w| w.id).collect();
        assert_eq!(ids, vec![1], "duplicate tags still yield a single match");
    }
}
