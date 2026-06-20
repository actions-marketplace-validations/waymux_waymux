# WayMux roadmap

WayMux is an early public release (`0.2.0-dev`). This roadmap is deliberately
honest about what works today versus what is planned. Dates are omitted on
purpose: phases ship when they are clean, not on a calendar.

## What works today

Validated and in daily use:

- **Headless Wayland sessions.** One `wayland-server`-based compositor per
  session, each in its own process, rendering to a virtual output (never your
  real display).
- **Spawning Wayland clients.** Any Wayland-native app (Chromium, foot, editors)
  via `waymux spawn <session> -- <argv>`.
- **Nested compositors.** KWin / Plasma 6 runs as an inner client (`--compositor`).
- **Capture.** Per-window and whole-desktop screenshots to PNG; recording to
  Matroska with lossless FFV1 (default) or hardware H.264/HEVC (NVENC, VA-API,
  Vulkan encode), with focused-window or whole-desktop modes and optional fps
  pacing.
- **Drive.** Synthetic keyboard/pointer input and batched `inject` ops; waiting
  on a window appearing or the frame stream going idle.
- **Live viewer.** A built-in WebRTC viewer that serves a browser page, bound to
  `127.0.0.1` by default and fail-closed on non-loopback addresses unless you
  supply a signed viewer token.
- **Attach.** A native client that embeds a session's surface into an outer
  compositor.
- **Machine interface.** A stable `--json` envelope on every verb (success
  `{ok,verb,data}`, error `{ok,verb,error}`, base64 PNG for screenshots); the
  `tag`/`windows --tag` and `record status`/`viewer status` verbs; and an MCP
  server (`waymux-mcp`) that exposes every discrete verb to agents by execing
  the CLI (argv-only, no shell injection).

Local-first by design: the CLI drives a per-user daemon over a unix socket
(SO_PEERCRED, same-uid, 0600). No telemetry, no phone-home.

## Known limitations today

- **Compositor support is still narrowing.** Direct Wayland clients, KWin /
  Plasma 6, and niri (Smithay) are validated as nested inner compositors.
  Hyprland (wlroots) is still experimental: its Aquamarine backend binds
  `xdg_wm_base` v6 and `wl_seat` v9, but the session advertises v5 / v7, so it
  fails to start. More generally, the session advertises a limited set of dmabuf
  *formats* (ARGB/XRGB8888 only, no multiplanar/YUV) with LINEAR plus
  EGL-importable tiled modifiers, a limited set of Wayland interface versions,
  and a limited set of globals, so a compositor that needs more may fall back to
  software or fail. Broader validation and a published support matrix are
  Phase 3.

## Planned phases

Each phase is a design-to-implementation unit. Phases 1 and 2 can overlap.

### Phase 1: CLI completeness - shipped

- Stable `--json` output on every verb (uniform envelope, base64 PNG, NDJSON for
  the streaming verbs), verified end to end.
- New verbs: `tag` + `windows --tag` filtering, `record status` (symmetric with
  `viewer status`), and a native `viewer token` mint (no Python helper needed).
- Batched `inject` (one round trip per op-batch) and implemented touch
  injection. Window targeting stays client-side via `windows`/`wait` + an
  explicit `window_id` (`inject_selector` is a documented reserved slot).
- Auto-located neko-bridge binary (`$WAYMUX_NEKO_BRIDGE_BIN`, next to the
  running executable, or `$PATH`, then a fallback).
- Single-command quickstart: `waymux serve` runs the daemon from the one binary,
  and local verbs auto-spawn the daemon when its socket is absent (opt out with
  `WAYMUX_NO_AUTOSPAWN`).

The `waymux-core` library crate from the original plan is intentionally not
pursued. It presupposed the CLI and MCP server both linking a shared engine
library; the shipped pi-harness model instead has the MCP server exec the CLI
(the CLI is the single surface), so a separate engine crate would have no
caller. The engine stays in the daemon, whose `SessionBackend` is a wired,
feature-gated extension point for alternate backends.

### Phase 2: MCP server (pi-harness model) - shipped

- `waymux-mcp` exposes every discrete request/response CLI verb to agents as an
  MCP tool by executing the CLI through an argument vector (no shell string, so
  no command injection). The streaming verbs (`events`, `logs`) and the
  credential-writing `login` are intentionally excluded. The CLI stays the
  single source of truth; MCP is a thin wrapper.
- Contract tests pin the MCP tool set to the CLI verbs, and an argument-injection
  regression test locks in the no-shell guarantee.

### Phase 3: Compositor validation and the support matrix

- [x] Confirm Plasma 6 + direct clients as the supported path.
- [x] Validate niri (Smithay) as a nested inner compositor (hardware-rendered;
  `AMD_DEBUG=nodcc` / `RADV_DEBUG=nodcc` on AMD).
- [ ] Unblock Hyprland (wlroots): raise the advertised Wayland interface
  version floors (`xdg_wm_base`, `wl_seat`) and the globals / dmabuf modifiers
  wlroots backends want (output management, foreign-toplevel-list, more
  modifiers).
- [ ] Ship per-compositor launcher recipes and smoke tests; publish the matrix.

### Phase 4: Docs, demo, polish

- Rewrite the design doc to the actual `wayland-server` architecture.
- A 60 fps hero demo.
- Continued documentation and example coverage.

## Non-goals

- A Python SDK. It was retired for the open-source debut; agents can wrap the
  CLI (or, soon, the MCP server) directly.
- A hosted control plane, GPU provisioning, or billing in this repository.
  WayMux follows an open-core model: this CLI, library, and MCP server are
  Apache-2.0; a separate hosted service is built on top of them.

## Contributing

Issues and pull requests are welcome. The repository is a single Cargo workspace
plus one Go module (the web viewer); see the README for build steps. CI runs
`cargo fmt`, `cargo clippy -D warnings`, tests, `cargo deny`, a `cargo publish`
dry-run for the library crates, the Go build, and a secret scan on every push.
