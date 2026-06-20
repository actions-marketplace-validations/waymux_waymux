# Laptop-local viewer (nested KDE → LAN browser / Fire tablet)

Run waymux **locally** on a Linux laptop and stream a **nested KDE desktop** to a
browser on the same LAN (e.g. a Fire tablet) over WebRTC. No cloud, no remote
VM. This is the self-host path, driven by `scripts/laptop-local-viewer.sh`.

> **It is NOT screen mirroring.** waymux-session runs *headless* and captures a
> fresh nested KWin 6 / Plasma 6 desktop it creates. Your real laptop screen,
> windows, and session are never captured or touched.

## Prerequisites

- KDE Plasma 6 (`kwin_wayland`, `plasmashell`) and `dbus-run-session`.
- `ffmpeg`, a working Vulkan stack, and a GPU with **H.264 encode**. Validated on
  AMD RADV (Renoir). NVIDIA also works (NVENC path); see the codec note below.
- `go` + `cargo` (to build the bridge / session the first time).
- `python3` with the `cryptography` package (token minting).
- Optional: `qrencode` (prints a scannable QR for the tablet).

## Usage

```bash
scripts/laptop-local-viewer.sh up       # build if needed, start, print URL + QR path
scripts/laptop-local-viewer.sh status   # is it up? which port? current URL + encoder fps
scripts/laptop-local-viewer.sh down      # tear everything down, free the port
```

`up` prints two URLs (loopback for this laptop, LAN IP for the tablet) plus a QR
at `<runtime-dir>/tablet-qr.png`. Open the LAN URL on the tablet (same Wi-Fi).

Env overrides: `WAYMUX_LOCAL_PORT` (default 8082), `WAYMUX_LOCAL_WIDTH` (1280),
`WAYMUX_LOCAL_HEIGHT` (720).

## How it works

```
waymux-session (headless, virtual 1280x720)         ← Rust outer compositor + WholeDesktop tap + Vulkan H.264 encode
├── waymux-neko-bridge  --bind 0.0.0.0 --port 8082   ← Go/Pion; the tablet connects here over WebRTC
└── dbus-run-session (private bus)
    └── kwin_wayland --socket wayland-kwin           ← nested KDE compositor
        └── plasmashell (+ kactivitymanagerd, kded6, kglobalacceld)
```

- **Encoder = Vulkan H.264.** It is auto-selected on non-NVIDIA hardware
  (`WAYMUX_VIEWER_CODEC=h264-vulkan`, set by the script).
- **`AMD_DEBUG=nodcc` is required on AMD** and the script sets it when launching
  KWin. Reason: with DCC enabled, KWin's GL output dmabuf uses a tiled modifier
  (e.g. `0x20000044051ba01`) that the Vulkan encoder cannot import, so the
  encoder drops every frame. `nodcc` yields modifier `0x200000000401a01`, which
  imports cleanly. (ffmpeg's own `h264_vulkan` encoder is *also* unsupported on
  RADV Renoir, with "missing encode feedback flags", but waymux's in-process
  Vulkan encoder works; only the DCC modifier had to change.)

## Auth (fail-closed EdDSA)

On a LAN (non-loopback) bind, the bridge rejects every `/ws` upgrade unless a
valid EdDSA viewer token is presented. The script mints a **local, ephemeral**
Ed25519 keypair (`scripts/laptop-mint-viewer-token.py`), hands the public key to
the bridge via environment inheritance (`spawn_bridge` inherits the session's
env, no code change), and puts the signed token (`aud=viewer`, ~8h `exp`) in the
URL. The private key is discarded at mint. No production secret is involved.

Verify: the URL with `?token=` returns HTTP 101 on `/ws`; without it, HTTP 401.

## Isolation (does not disturb other work)

- **Headless** virtual output: never takes over your real display or session.
- Private `XDG_RUNTIME_DIR` (mktemp, 0700); a private `dbus-run-session` for the
  nested desktop (so its KDE daemons can't touch your real session bus).
- A single dedicated TCP port; no `sudo`, no `/etc/hosts`, no installs.
- `down` kills **only** processes whose environ references this run's runtime
  dir (your real session uses `/run/user/<uid>`, never `/tmp/waymux-local.*`).

## Notes / gotchas

- **Idle desktop streams at low fps.** Frames are produced on damage, so a static
  Plasma desktop trickles ~sub-1fps. Touch/drag from the tablet (input flows back
  through the bridge) and it comes alive. This matches how the streaming path
  behaves with no moving content.
- **`org.kde.kdeconnect` aborting** in `kde.log` is harmless (phone-integration
  service; not needed for the desktop).
- **plasmashell needs `kactivitymanagerd`**, whose path differs by distro
  (`/usr/lib/kactivitymanagerd` on Arch). The script searches common locations;
  if plasmashell still can't start, the script falls back to a `foot` terminal so
  the stream is never blank.
- **Headless H.264 browsers (e.g. Playwright's bundled Chromium) cannot decode**
  the stream: use a real browser (the tablet's, or desktop Chrome/Firefox).
- TURN/STUN are unset (LAN host ICE candidates suffice). For traffic beyond the
  LAN, that would need to change.
