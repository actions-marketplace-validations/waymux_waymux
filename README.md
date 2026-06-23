# waymux

**Headless Wayland sessions you can spawn, screenshot, record, script, and watch
live in a browser, driven by a CLI and an MCP server.**

[![pipeline status](https://gitlab.com/tek.cat/waymux/badges/main/pipeline.svg)](https://gitlab.com/tek.cat/waymux/-/pipelines)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![MSRV: Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](./rust-toolchain.toml)

Project home: [gitlab.com/tek.cat/waymux](https://gitlab.com/tek.cat/waymux).

Multiplex Wayland clients (and nested compositors like KWin / Plasma 6) as
named, headless, detachable sessions: spawn an app or a whole desktop into a
virtual output, screenshot it, record it, drive it with synthetic input, and
view it live in a browser over loopback. Scriptable from the shell with a
stable `--json` contract on every verb, and an MCP server (`waymux-mcp`) so AI
agents can drive sessions through the same surface.

waymux runs entirely on your machine. There is no account, no telemetry, and no
phone-home. The only network exposure is what you configure: binding the viewer
to a LAN address (a local listener), or `login` / `--remote` against a server
you name.

License: Apache-2.0. Status: early public release (`0.2.0-dev`). See
[ROADMAP.md](./ROADMAP.md) for what works today and what is planned.

## What it does

- **Headless Wayland sessions.** Each session is a `wayland-server`-based
  compositor in its own process, rendering to a virtual output. It is not your
  real screen and never touches your physical display.
- **Run Wayland clients and nested compositors.** Spawn a single app (Chromium,
  foot, an editor), or nest a full compositor inside a session. KWin / Plasma 6
  and niri are both validated as nested inner compositors; Hyprland is still
  experimental. See the support matrix.
- **Capture.** Screenshot a window or the whole session to PNG. Record the
  session losslessly (FFV1) or with a hardware H.264/HEVC encoder to Matroska.
- **Drive.** Inject keyboard and pointer input; wait for windows; wait for the
  frame stream to go idle.
- **View live.** Start a built-in WebRTC viewer that serves an HTML page; open
  it in a browser to watch and control the session. Binds to `127.0.0.1` by
  default.
- **Test in CI without a GPU.** The compositor, capture, screenshot, input
  injection, and FFV1 recording all run on the CPU (Mesa llvmpipe), so you can
  drive and assert real GUI apps on a stock shared runner. Published images
  (`waymux-ci`, `waymux-ci-plasma`), a `waymux-run` wrapper ("xvfb-run for
  Wayland"), a GitHub Action (`waymux/waymux@v1`), and a GitLab CI `include`
  template make it a drop-in. See
  [docs/headless-ci-testing.md](docs/headless-ci-testing.md).

## Try it

A full session from one binary. The first local command auto-starts a
background `waymuxd`, so there is nothing else to run first:

```sh
# Run the daemon in the foreground in another terminal (optional; local
# commands auto-spawn it otherwise).
waymux serve

# Create a session, spawn a terminal into it, and wait for the window.
waymux new demo --size 1280x720
waymux spawn demo -- foot
waymux wait demo --app-id foot

# List windows, screenshot the whole desktop, and record a clip.
waymux windows demo
waymux screenshot-desktop demo --output /tmp/demo.png
waymux record start demo
waymux record stop demo

# Tear it down.
waymux rm demo
```

Every verb takes `--json` for a machine-readable envelope on stdout:

```sh
$ waymux --json new demo --size 1280x720
{"ok":true,"verb":"new","data":{"name":"demo","width":1280,"height":720}}

$ waymux --json windows demo
{"ok":true,"verb":"windows","data":{"windows":[{"window_id":1,"app_id":"foot","title":"foot"}]}}

$ waymux --json record start demo
{"ok":true,"verb":"record","data":{"path":"/home/you/.local/share/waymux/recordings/demo-....mkv"}}
```

The same surface is available to AI agents over MCP. Point your MCP client at
the built `waymux-mcp` binary (it execs the `waymux` CLI with `--json`, argv
only, so the CLI stays the single source of truth):

```sh
WAYMUX_BIN=/path/to/waymux  /path/to/waymux-mcp
```

## Demo

A recorded 60 fps video demo is planned (see [ROADMAP.md](./ROADMAP.md),
Phase 4). Until then, the command sequence above is the fastest way to see
waymux end to end on your own machine.

## Build

waymux is a Cargo workspace of Rust crates plus one Go module (the web viewer).

Requirements:

- Rust 1.88+ (the MSRV; `rust-toolchain.toml` pins the channel to `stable`, so
  a fresh clone fetches a current compiler).
- Go 1.26+ (only to build the WebRTC viewer bridge).
- System libraries: `wayland`, `ffmpeg` 6.1+ (the floor for building and running
  FFV1 plus the basic H.264 hardware paths, invoked via the system LGPL
  libraries; the in-process Vulkan encoder compiles against ffmpeg 6 and 7+),
  and your GPU's Vulkan / VA-API stack for hardware video encoding. The lossless
  Vulkan codecs need newer ffmpeg: `ffv1-vulkan` wants a recent build, and
  `hevc-vulkan-lossless` requires ffmpeg 8.0. See `ARCHITECTURE.md` for the
  design.

```sh
# Rust binaries: daemon, session, CLI, attach client.
cargo build --release

# The web viewer (Go). Produces ./waymux-neko-bridge in that directory.
( cd crates/waymux-neko-bridge && go build -o waymux-neko-bridge . )

# Tests.
cargo test
```

The release binaries land in `target/release/`: `waymux` (the CLI),
`waymux-daemon`, `waymux-session`, `waymux-attach`.

## Quickstart: run a session

The `waymux` CLI is all you need: when a local command finds no running daemon,
it auto-spawns one in the background and retries, so a fresh install works from
a single binary:

```sh
# Create a 1920x1080 headless session named "demo".
# (The first local command auto-starts a background `waymuxd`.)
waymux new demo --size 1920x1080

# Spawn a Wayland-native app into it. Everything after `--` is the argv.
waymux spawn demo -- foot

# Wait for the window to appear, then list windows.
waymux wait demo --app-id foot
waymux windows demo

# Screenshot a window (by id from `windows`) to a PNG.
waymux screenshot demo <window_id> -o /tmp/demo.png

# Tear it down.
waymux rm demo
```

### The daemon: auto-spawn, `waymux serve`, and lifetime

The `waymux` CLI talks to a local control daemon (`waymuxd`) over a Unix socket
at `$WAYMUX_SOCKET` (default `$XDG_RUNTIME_DIR/waymux.sock`). You have three ways
to run it:

* **Auto-spawn (default).** When a local command finds the socket absent, the
  CLI starts a detached background `waymuxd` (resolved from `$WAYMUXD_BIN`, then
  a `waymuxd` next to the `waymux` binary, then `$PATH`) and retries the connect.
  Auto-spawn fires **only** when the socket is missing: a permission error, a
  protocol-version mismatch, or any other connection failure is surfaced
  verbatim, never masked. It never fires for `--remote`.
* **`waymux serve`.** Runs `waymuxd` in the foreground from the same binary
  (it `exec`s the resolved `waymuxd`, forwarding `--socket`). Use this under a
  process supervisor, or just to watch the daemon's logs in a terminal.
* **Explicit `waymuxd`.** Run the daemon yourself
  (`cargo run --release -p waymux-daemon`, or an installed `waymuxd`).

An auto-spawned daemon is a real per-user daemon: it **outlives** the CLI
command that started it and keeps your sessions alive across invocations. There
is no `waymux shutdown` verb today, so stop it with a signal: Ctrl-C in the
`waymux serve` terminal, or `pkill waymuxd`. To opt out of auto-spawn entirely
(e.g. when you manage the daemon yourself and want an explicit error if it is
not running), set `WAYMUX_NO_AUTOSPAWN=1`.

The `scripts/waymux-launch.sh` helper wraps create + attach-viewer +
auto-teardown for quick interactive exploration of a single app:

```sh
scripts/waymux-launch.sh --size 1280x800 --name term -- foot
```

## Spawn Chromium

Chromium is a Wayland client. Wrap it in `dbus-run-session` (Chromium needs a
session bus; without one it spams D-Bus errors and may not start) and pass the
Wayland backend flags:

```sh
waymux new web --size 1280x800
waymux spawn web -- dbus-run-session -- /usr/bin/chromium --ozone-platform=wayland \
    --user-data-dir=/tmp/waymux-chromium https://example.org
waymux wait web --app-id chromium

# Capture the running page. screenshot-desktop captures the whole session (no
# window id needed); use `waymux screenshot web <window_id> -o ...` for one window.
waymux screenshot-desktop web --output /tmp/web.png

# Record it with the zero-copy hardware encoder. On AMD the daemon defaults
# AMD_DEBUG=nodcc for spawned clients so the GPU dmabuf is encoder-importable.
waymux record start web
waymux record stop web
```

Single-instance gotcha: most browsers and editors route to an existing
host-side instance over D-Bus unless you pass a fresh profile
(`--user-data-dir` / `--new-window`). Without it, your "inner" launch silently
focuses the host window instead of rendering inside the session.

## Spawn a full Plasma 6 desktop

waymux can host a nested compositor as an inner client. Launch KWin (Plasma 6)
into a session with its own D-Bus bus:

```sh
waymux new kde --size 1280x720

# Launch KWin into the session's inner socket on its own D-Bus bus.
# KWIN_WAYLAND_NO_PERMISSION_CHECKS lets KWin's nested backend start without a
# seat; on AMD, AMD_DEBUG/RADV_DEBUG=nodcc hand out a hardware-encoder-importable
# dmabuf modifier (DCC-tiled buffers are not importable by the Vulkan H.264
# encoder).
dbus-run-session -- env \
    AMD_DEBUG=nodcc RADV_DEBUG=nodcc \
    WAYLAND_DISPLAY=$XDG_RUNTIME_DIR/waymux/kde/wayland.sock \
    KWIN_WAYLAND_NO_PERMISSION_CHECKS=1 XDG_CURRENT_DESKTOP=KDE QT_QPA_PLATFORM=wayland \
    kwin_wayland --socket wayland-kwin --xwayland --no-lockscreen &
# then start plasmashell (and any apps) against WAYLAND_DISPLAY=wayland-kwin
# inside that same bus.
```

The `scripts/laptop-local-viewer.sh` script automates the full nested-KDE +
loopback-viewer path end to end (see "View a session in a browser" below).

To record the whole nested desktop (rather than a single window), use
`--mode whole-desktop`:

```sh
waymux record start kde --codec h264-vulkan --mode whole-desktop --min-fps 30
# ... drive the desktop ...
waymux record stop kde
```

This tees KWin's composited output dmabuf straight into the Vulkan H.264 encoder
with no CPU copy, and starts immediately even over an idle desktop.

## Record a session

```sh
# Start a lossless FFV1 recording (default output dir
# ~/.local/share/waymux/recordings/).
waymux record start demo

# ... drive the session ...

# Stop and finalize the Matroska file.
waymux record stop demo
```

`record start` defaults to lossless FFV1. Pass `--codec x264-lossless` for a
leaner CPU-lossless encode (RGB-lossless libx264rgb; higher fps and much smaller
files than FFV1 on a small runner, matching the Xvfb + ffmpeg x11grab approach),
or `--codec h264-nvenc` or
`--codec h264-vaapi` (and Vulkan-encode variants) for hardware H.264/HEVC, plus
`--mode focused-window|whole-desktop`, `--min-fps` for steady pacing, and
`--secondary-codec` to encode a second output from the same frames. For
GPU-lossless, `ffv1-vulkan` is the portable choice; `hevc-vulkan-lossless` needs
Vulkan-video HEVC 4:4:4 (Hi444) encode caps, which integrated GPUs may not
expose, so `record start` fails fast with a clear error there rather than
producing an empty file. (The
separate `WAYMUX_VIEWER_CODEC` env var selects the live *viewer's* codec, not
the recorder's.) ffmpeg is dynamically linked against the system LGPL libraries.
The recorder defaults to FFV1 (LGPL-only). `--codec x264-lossless` additionally
drives your system ffmpeg's libx264 (GPL) through the subprocess, so it works
only where ffmpeg was built with libx264; waymux itself never links it. The live
viewer has the same opt-in CPU x264 (GPL) via `WAYMUX_VIEWER_CODEC=h264-software`.

## Headless CI testing (no GPU)

waymux doubles as an "Xvfb for Wayland": host real GUI apps in a nested session,
inject input, and assert on what they drew, with no display and no GPU.
Screenshots are the cheap, immediate primitive; recording works too (FFV1 on the
CPU) but at roughly 10 fps, so treat a clip as a record of what happened rather
than smooth video. `tests/e2e/run-e2e-embedded.sh` drives Chromium and a KDE app
under software rendering (Mesa llvmpipe) end to end, and `tests/e2e/Dockerfile`
runs it in a GPU-free container. See
[docs/headless-ci-testing.md](docs/headless-ci-testing.md).

### GitHub Action

Build your app into an image `FROM ghcr.io/waymux/waymux-ci`, then run any
command in a fresh, software-rendered session and collect the artifacts:

```yaml
- uses: waymux/waymux@v1
  with:
    image: ghcr.io/waymux/my-app-with-waymux:latest
    run: ./run-gui-tests.sh
    record: 'true'        # also capture a clip (optional)
```

The screenshot (and recording) are uploaded as a build artifact. See
[`action.yml`](action.yml) for every input.

### GitLab CI

Include the template and extend `.waymux-test`:

```yaml
include:
  - remote: 'https://gitlab.com/tek.cat/waymux/-/raw/main/templates/waymux-test.gitlab-ci.yml'

gui-test:
  extends: .waymux-test
  variables:
    WAYMUX_IMAGE: '$CI_REGISTRY_IMAGE/my-app-with-waymux:latest'
    WAYMUX_RUN: './run-gui-tests.sh'
```

Both hosts also run the full harness plus demo and benchmark jobs
(`kde-app-demo`, `plasma-demo`, `benchmark`) that exercise KDE apps and a nested
Plasma 6 desktop.

## View a session in a browser (loopback)

A session can autostart a WebRTC web viewer on a local port. The viewer binds
to `127.0.0.1` by default; binding to a non-loopback address is fail-closed
(every WebSocket upgrade is rejected) unless you supply a viewer-token public
key.

The simplest path is `scripts/laptop-local-viewer.sh`, which builds the
binaries, starts a headless session with the viewer bound to loopback, nests a
KDE desktop into it, and prints a URL:

```sh
# Loopback only (this machine):
WAYMUX_LOCAL_PORT=8082 scripts/laptop-local-viewer.sh up
#   -> open http://127.0.0.1:8082/?token=...

scripts/laptop-local-viewer.sh status
scripts/laptop-local-viewer.sh down
```

To bind the viewer to a LAN address (so another device on your network can
connect), the bridge requires a signed viewer token. Mint an ephemeral one with
`scripts/laptop-mint-viewer-token.py` (it generates a throwaway Ed25519 keypair,
signs a short-lived token, and discards the private key); the script wires the
public key into the session's environment automatically. See
`docs/laptop-local-viewer.md` for the LAN walkthrough.

Under the hood, a session is started directly like this (what the script does):

```sh
WAYMUX_VIEWER_CODEC=h264-vulkan \
WAYMUX_NEKO_BRIDGE_BIN="$PWD/crates/waymux-neko-bridge/waymux-neko-bridge" \
target/release/waymux-session --name demo --width 1280 --height 720 \
    --inner-socket /tmp/inner.sock --control-socket /tmp/control.sock \
    --viewer-port 8082 --viewer-bind 127.0.0.1
```

## Connecting to a remote endpoint

Almost every `waymux` subcommand drives the local daemon over its unix socket
and needs no configuration. The one exception is `login`, which authenticates
against a hosted `waymux-api` deployment and saves the credentials for later
`--remote` use. No production host is baked into the OSS build, so you must
point it at your own deployment:

```sh
waymux login --base-url http://your-host:8080 --api-key <key>
```

The default `--base-url` is the localhost placeholder `http://localhost:8080`.
Running `waymux login` with no `--base-url` saves that placeholder, so always
pass your own endpoint when you intend to reach a remote service.

## Driving from an agent (MCP)

Every discrete request/response verb is also exposed through an MCP server,
`waymux-mcp`, so an AI agent can create sessions, spawn apps, screenshot,
record, tag windows, and read status over the Model Context Protocol. (The
streaming verbs `events` and `logs`, and the credential-writing `login`, are
intentionally excluded.) The server is a thin wrapper: it
runs the `waymux` CLI with `--json` through an argument vector (never a shell
string), so the CLI stays the single source of truth and there is no
shell-injection surface. Point your MCP client at the built `waymux-mcp` binary
(set `WAYMUX_BIN` if `waymux` is not on `PATH`).

## Machine-readable output

Pass `--json` to any verb for a uniform envelope on stdout:

```sh
waymux --json ls        # {"ok":true,"verb":"ls","data":{"sessions":[...]}}
waymux --json new demo --size 1920x1080
waymux --json windows demo --tag editor
```

Success is `{"ok":true,"verb":"<verb>","data":{...}}`; failure is
`{"ok":false,"verb":"<verb>","error":{"code":"E_...","message":"..."}}` with a
non-zero exit. Screenshots return the PNG base64 in `data.png_b64`. The
streaming verbs `events` and `logs` emit newline-delimited JSON instead.

## Compositor support matrix

waymux spawns any Wayland client; a *nested compositor* works when the session
advertises the globals and dmabuf modifiers it needs. Validated today:

| Inner client / compositor | Status | Notes |
|---|---|---|
| Direct Wayland clients (Chromium, foot, editors) | Validated | The core path; backs the demo. |
| KWin / Plasma 6 (nested compositor) | Validated | Use `AMD_DEBUG=nodcc` on AMD for an importable dmabuf modifier. |
| niri (Smithay) | Validated | Runs nested via `waymux spawn <s> --compositor -- niri`, hardware-rendered. On AMD set `AMD_DEBUG=nodcc` / `RADV_DEBUG=nodcc` (the same untiled-output workaround KWin needs). |
| Hyprland (wlroots) | Experimental | Blocked on Wayland interface version floors: its Aquamarine backend binds `xdg_wm_base` v6 and `wl_seat` v9, but the session advertises v5 / v7. |

Known risk areas for an experimental compositor: the session advertises only
ARGB8888/XRGB8888 formats (no multiplanar/YUV), with LINEAR plus the
EGL-importable tiled modifiers, so a compositor needing multiplanar or non-BGRA
formats may fall back to software or fail; it advertises a limited set of
Wayland interface versions (`xdg_wm_base` v5, `wl_seat` v7), so a backend that
binds a newer floor fails to start; and wlroots / Smithay compositors may want
globals (`wlr-output-management`, `ext-foreign-toplevel-list`, `drm-lease`,
gamma control) the session does not advertise.

## Layout

    crates/waymux-protocol     msgpack wire types + framing (the wire contract)
    crates/waymux-daemon       waymuxd: local control daemon, session registry
    crates/waymux-session      per-session headless wayland-server compositor
    crates/waymux-cli          the `waymux` CLI binary
    crates/waymux-mcp          waymux-mcp: MCP server (execs the CLI, argv-only)
    crates/waymux-attach       native attach client (embed a session surface)
    crates/waymux-mux-mkv      zero-dependency streaming Matroska muxer
    crates/waymux-neko-bridge  WebRTC web viewer (Go; vendored neko fork)

The CLI is a thin shell over the daemon's msgpack control protocol. Sessions
are isolated processes that own their own control socket; every local socket is
SO_PEERCRED same-uid gated, and the daemon and attach sockets are additionally
mode 0600.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md)
for build steps, how to run the full CI gate locally, and the branch / PR
conventions. All changes go through CI, which must be green before merge.

## Security

Please do not file public issues for security vulnerabilities. See
[SECURITY.md](./SECURITY.md) for the private disclosure process and the
supported versions. The [CHANGELOG.md](./CHANGELOG.md) tracks released changes.

## Attribution

The web viewer (`crates/waymux-neko-bridge`) is a vendored fork of
[neko](https://github.com/m1k1o/neko). See `crates/waymux-neko-bridge/LICENSE.neko`,
`crates/waymux-neko-bridge/NEKO-VENDORED.md`, and the root `NOTICE`.
