# Headless Wayland app testing in CI (no GPU)

waymux runs a real nested Wayland session entirely in software, so you can test
GUI apps in CI the way you already test with Xvfb on X11: launch an app, drive
it, and check what it drew, with no display and no GPU.

Think of it as "Xvfb for Wayland, with batteries included." The compositor does
no rendering of its own, frames are captured from CPU memory, screenshots are
encoded on the CPU, and the default recording codec (FFV1) is a CPU encoder. The
whole loop runs green on a stock shared CI runner.

Screenshots are the cheap, immediate primitive: encoding one is fast on the CPU.
Recording works on the CPU too. The default FFV1 codec runs around 10 frames per
second on a small runner; `--codec x264-lossless` (RGB-lossless libx264rgb) is the
leaner CPU-lossless option, roughly doubling that and matching the usual Xvfb +
ffmpeg x11grab recipe at a fraction of the file size (see the
[benchmarks](https://waymux.dev/benchmarks.html)). Either way, treat a software
recording as a record of what happened (a handy failure artifact), not smooth
video; for high-frame-rate recording, add a GPU and a hardware encoder.

```
  launch app  ->  inject input  ->  screenshot + assert  ->  record (FFV1)
```

## What works with no GPU

| Capability                          | GPU needed? |
| ----------------------------------- | ----------- |
| Nested compositor + virtual output  | no          |
| Hosting Wayland + XWayland apps      | no          |
| Screenshot (PNG, from CPU memory)    | no          |
| FFV1 lossless recording (CPU codec)  | no          |
| Keyboard / pointer / touch injection | no          |
| Live WebRTC viewer (default codecs)  | yes         |
| Hardware video encode (NVENC/VAAPI/Vulkan) | yes   |

The first five are everything you need for functional, layout, input, and
visual-regression testing, and none of the flows below use a GPU. The live
viewer's default codecs are GPU encoders (it also has an opt-in CPU encoder via
`WAYMUX_VIEWER_CODEC=h264-software`, just slower), but CI does not need the
viewer at all: it screenshots and records instead.

## Quickstart

Force software rendering and start the daemon:

```sh
export LIBGL_ALWAYS_SOFTWARE=1      # Mesa software GL (llvmpipe)
export WAYMUX_DISABLE_SYNCOBJ=1     # no /dev/dri => implicit sync, no DRM node
waymux serve &                      # or let the CLI auto-spawn the daemon
```

Create a session and launch an app as a direct Wayland client. Point the app's
`WAYLAND_DISPLAY` at the session's inner socket:

```sh
waymux new app --size 1280x800
INNER="$XDG_RUNTIME_DIR/waymux/app/wayland.sock"

# A Qt/KDE app:
WAYLAND_DISPLAY="$INNER" QT_QPA_PLATFORM=wayland kwrite notes.txt &

# Or Chromium. Use --app so the page is the toplevel surface (the cleanest
# visual-diff target) and disable the GPU so it renders via software:
WAYLAND_DISPLAY="$INNER" chromium \
  --ozone-platform=wayland --no-sandbox --disable-gpu \
  --disable-gpu-compositing --in-process-gpu --disable-dev-shm-usage \
  --app="file://$PWD/page.html" &

waymux wait app --timeout-ms 15000
```

Drive it and capture the result:

```sh
waymux key app 17                       # inject a keystroke (evdev keycode)
waymux screenshot-desktop app -o shot.png
waymux record start app --codec ffv1            # lossless, CPU-encoded (exact-pixel)
# or: --codec x264-lossless                      # leaner CPU-lossless, much smaller files
# ... exercise the app ...
waymux record stop app
```

That is the entire testing loop, and none of it touched a GPU.

## The embedded-app e2e

`tests/e2e/run-e2e-embedded.sh` is a ready-made, software-only end-to-end test.
It launches Chromium and a KDE app as direct Wayland clients under llvmpipe,
asserts the captured frames have real content (a center-contrast check, so a
blank body cannot pass on the strength of a rendered toolbar), injects
keystrokes and proves they changed the capture, and records FFV1, verifying the
recording with `ffprobe` and counting genuinely unique frames (duplicate
min-fps padding is stripped with `mpdecimate` first). Each scenario skips
cleanly when its app is not installed.

```sh
tests/e2e/run-e2e-embedded.sh           # build + run
WAYMUX_E2E_NO_BUILD=1 tests/e2e/run-e2e-embedded.sh   # binaries already built
```

## In a container

`tests/e2e/Dockerfile` builds a GPU-free image that bundles the binaries, Mesa
software rendering, ffmpeg, and the apps, then runs the embedded e2e. It needs
no `--gpus` and no `/dev/dri`:

```sh
docker build -f tests/e2e/Dockerfile -t waymux-e2e .
docker run --rm waymux-e2e
```

## In CI

Run the embedded e2e on any GPU-less shared runner. The job installs the
software-render stack and the apps, builds the three binaries it needs, and runs
the harness under a session bus (the KDE app wants one).

GitLab CI:

```yaml
embedded-e2e:
  stage: check
  image: rust:1.94-trixie
  before_script:
    - apt-get update
    - >
      apt-get install -y --no-install-recommends
      libwayland-dev libgbm-dev libavutil-dev libavformat-dev libavcodec-dev
      libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev
      libvulkan-dev libxkbcommon-dev pkg-config build-essential
      libgl1-mesa-dri libegl-mesa0 libegl1 libgles2
      ffmpeg python3 dbus procps fonts-dejavu-core
      chromium kwrite foot
  script:
    - cargo build -p waymux-cli -p waymux-daemon -p waymux-session
    - >
      WAYMUX_E2E_NO_BUILD=1 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
      dbus-run-session -- bash tests/e2e/run-e2e-embedded.sh
```

GitHub Actions (run in a Debian container so `apt install chromium` is the real
package, matching the runner-agnostic path above):

```yaml
  embedded-e2e:
    name: embedded app e2e (software / no GPU)
    runs-on: ubuntu-24.04
    container: rust:1.94-trixie
    steps:
      - uses: actions/checkout@v4
      - name: deps
        run: |
          apt-get update
          apt-get install -y --no-install-recommends \
            libwayland-dev libgbm-dev libavutil-dev libavformat-dev libavcodec-dev \
            libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev \
            libvulkan-dev libxkbcommon-dev pkg-config build-essential \
            libgl1-mesa-dri libegl-mesa0 libegl1 libgles2 \
            ffmpeg python3 dbus procps fonts-dejavu-core \
            chromium kwrite foot
      - run: cargo build -p waymux-cli -p waymux-daemon -p waymux-session
      - name: embedded e2e
        env:
          WAYMUX_E2E_NO_BUILD: "1"
          LIBGL_ALWAYS_SOFTWARE: "1"
          GALLIUM_DRIVER: llvmpipe
        run: dbus-run-session -- bash tests/e2e/run-e2e-embedded.sh
```

## CI demo jobs and benchmark

The repository ships three additional CI jobs that run on stock, GPU-less
shared runners via `tests/e2e/ci-demo-all.sh` and the purpose-built demo image
`tests/e2e/Dockerfile.demo`. The image builds release binaries, installs Mesa
llvmpipe, Plasma 6, and a set of KDE apps, strips the `cap_sys_nice` capability
from `kwin_wayland` (unnecessary under software rendering, and it causes `execve`
to fail on shared runners), and runs `ci-demo-all.sh` as its entrypoint.

**`kde-app-demo` (gating).** Runs `tests/e2e/demo-kde-app.sh`, which launches
KWrite as a direct Wayland client against a software-rendered waymux session,
asserts the screenshot has real content (not blank), injects keystrokes, records
an FFV1 clip, and verifies the codec and that at least two unique frames were
captured. Failure blocks the pipeline. Artifacts: `kde-app.png` and
`kde-app.mkv`.

**`plasma-demo` (allow-failure).** Runs `tests/e2e/demo-plasma.sh`, which
launches nested KWin plus plasmashell inside a waymux session under llvmpipe,
waits up to 60 seconds for a desktop window to appear, screenshots the full
desktop, optionally opens Dolphin (best-effort), and records a 6-second
whole-desktop FFV1 clip. A failure here does not block the pipeline (Plasma
under software rendering is slower and more variable than a single app).
Artifacts: `plasma.png`, `plasma.mkv`, and (when Dolphin appears)
`plasma-app.png`.

**`benchmark` (gating).** Runs `tests/e2e/ci-bench.sh`, which sweeps four
recording configurations: Chromium at 1280x720 and 1920x1080 (whole-desktop),
Chromium at 1920x1080 (focused-window), and KWrite at 1920x1080
(whole-desktop). The Chromium configs render an animated page (a per-frame
counter plus sweeping elements) so every captured frame is genuinely distinct.
For each configuration it records a lossless FFV1 clip and reports the
**unique** frame rate (distinct frames per second, via `ffmpeg mpdecimate`)
next to the nominal container rate (which pads toward `--min-fps` with
duplicates), plus screenshot latency. Functional gates: the codec is `ffv1` and
the unique-frame rate is above zero (capture is not frozen). No fps threshold is
enforced; this is a capability check, not a performance gate. Failure blocks the
pipeline. Artifacts: `benchmark.md` (host specs plus a row per configuration),
`benchmark.json` (the same, structured), and `bench-sample.mkv` (a saved
recording of the animated page), all downloadable straight from the build.

Reproduce it anywhere with the published image (no GPU, no checkout needed):

```sh
mkdir -p out
docker run --rm --shm-size=512m -e RECORD_SECS=10 -e ARTIFACT_DIR=/out \
  -v "$PWD/out:/out" --entrypoint dbus-run-session \
  ghcr.io/waymux/waymux-ci -- bash /app/tests/e2e/ci-bench.sh
cat out/benchmark.md          # numbers + host specs
mpv out/bench-sample.mkv      # the recorded animated clip
```

Software FFV1 throughput is dominated by whole-desktop frame readback, not CPU
cores. The same harness, `unique_fps` (FFV1 lossless, llvmpipe, no GPU):

| config | GitLab (EPYC 7B13, 2 vCPU) | GitHub (EPYC 7763, 4 vCPU) | laptop (Ryzen 5700U, 16T) |
|--------|---------------------------|----------------------------|---------------------------|
| Chromium 1280x720 whole-desktop  | 10.0 | 10.1 | 9.9 |
| Chromium 1920x1080 whole-desktop | 9.7  | 9.9  | 9.9 |
| Chromium 1920x1080 focused-window | 15.5 | 36.3 | 43.2 |
| KWrite 1920x1080 whole-desktop    | 7.6  | 7.8  | 8.0 |

Whole-desktop capture pins near 10 fps regardless of core count (2 -> 4 -> 16):
it is bounded by synchronous frame readback on the compositor thread, not by the
CPU. Focused-window capture is several times faster and *does* scale with the
machine. Practical guidance: to record a single app smoothly, use
`--mode focused-window`; whole-desktop is for capturing the broader session at a
steady ~10 fps. The exact per-run numbers and the recorded clip are in each
build's `benchmark` artifact.

## Notes

- Launch GUI apps as direct clients (set `WAYLAND_DISPLAY` to the session's
  inner socket). This gives you full control of the app's environment.
- For Chromium, `--app=<url>` puts the page on the toplevel surface, which is
  both the most reliable thing to capture in software and the cleanest target
  for a pixel diff.
- `--disable-dev-shm-usage` keeps Chromium off the small default `/dev/shm` in a
  container (or use `docker run --shm-size=512m`).
- Recording duplicate-pads to `--min-fps`. To measure real motion, drop
  duplicates first: `ffmpeg -i out.mkv -vf mpdecimate ...` then count frames.

## Use it in your own CI

Published images and thin wrappers let you test your own Wayland / Qt / KDE app
in CI with no GPU. Build your app into an image based on the lean test image,
then drive it with `waymux-run` (the "xvfb-run for Wayland": it creates a
session, points `WAYLAND_DISPLAY` at it, runs your command, and saves a
screenshot, plus a recording with `--record`, into `ARTIFACT_DIR`).

Two images are published to GHCR, Docker Hub, and the GitLab registry:

- `waymux-ci` (lean: Chromium, a Qt app, the binaries, and `waymux-run`)
- `waymux-ci-plasma` (adds a full Plasma 6 desktop)

Build your app in:

```dockerfile
FROM ghcr.io/waymux/waymux-ci:latest
RUN apt-get update && apt-get install -y --no-install-recommends my-app
```

Plain Docker:

```sh
docker run --rm -v "$PWD/out:/out" -e ARTIFACT_DIR=/out \
  --entrypoint dbus-run-session ghcr.io/waymux/waymux-ci \
  -- waymux-run --record -- my-app-test-command
# screenshot + recording land in ./out
```

GitHub Actions (the repo doubles as a composite action):

```yaml
- uses: waymux/waymux@v1
  with:
    image: ghcr.io/me/my-app-with-waymux:latest
    run: ./run-gui-tests.sh
    record: 'true'
```

GitLab CI (include the template and extend `.waymux-test`):

```yaml
include:
  - remote: 'https://gitlab.com/tek.cat/waymux/-/raw/main/templates/waymux-test.gitlab-ci.yml'

gui-test:
  extends: .waymux-test
  variables:
    WAYMUX_IMAGE: '$CI_REGISTRY_IMAGE/my-app-with-waymux:latest'
    WAYMUX_RUN: './run-gui-tests.sh'
```

The images are published by a tag-triggered job (`publish-images`) to GHCR and
the GitLab registry automatically, and to Docker Hub when `DOCKERHUB_USERNAME` /
`DOCKERHUB_TOKEN` are set.

## Security

A waymux session is a **test harness, not a security sandbox.** Run untrusted or
third-party code in a container or VM, not in a bare session.

- **Any client in a session can screen-capture the whole session.** The
  compositor advertises `wlr-screencopy` to every client, so a client connected
  to the inner `wayland.sock` can read the pixels of every other window in that
  session. Treat all apps sharing one session as mutually trusting; isolate
  untrusted apps in separate sessions, containers, or VMs. (Cross-client
  clipboard sharing is off by default.)
- **Access control depends on `XDG_RUNTIME_DIR` being `0700`.** The daemon's
  control socket and each session's `control`/`attach` sockets are same-uid
  gated (SO_PEERCRED), but the per-session sockets live in a world-traversable
  directory, so their protection comes from the parent runtime dir's `0700`
  mode. On systemd, `/run/user/<uid>` is already `0700`; if you point
  `XDG_RUNTIME_DIR` elsewhere, keep it `0700`. Do not place it on a
  world-traversable path.
- **No network listener is opened by the test path.** The live WebRTC viewer
  binds `127.0.0.1` and is never started by the embedded e2e; capture and
  recording are entirely local.
- **The container runs as root and Chromium with `--no-sandbox`.** That is
  acceptable here only because the image is single-tenant and renders trusted
  first-party content. Do not reuse it to open untrusted URLs.
