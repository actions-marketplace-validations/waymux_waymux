#!/usr/bin/env bash
# scripts/laptop-local-viewer.sh
#
# Local self-host TEST pathway: run waymux on this laptop and stream a nested
# KDE desktop to a browser on the same LAN (e.g. a Fire tablet) over WebRTC.
# NO cloud, no remote VM. See docs/laptop-local-viewer.md.
#
#   up      build (if needed) + start a headless waymux-session, a nested KDE
#           desktop, and the LAN-bound neko-bridge; print the tablet URL.
#   down    tear everything down (process tree + runtime dir + ephemeral keys).
#   status  show whether a session is up, the port, and the current URL.
#
# Design notes (validated 2026-06-14 on AMD RADV Renoir, KWin 6.6.5):
#   * Capture is the nested KWin output (WholeDesktop tap), NOT your real
#     screen. waymux-session runs HEADLESS (virtual output) so it never
#     touches your physical display/session.
#   * Encoder: Vulkan H.264 (the only HW path without NVIDIA). It requires the
#     nested KWin to hand out a Vulkan-importable dmabuf modifier, so we launch
#     KWin with AMD_DEBUG=nodcc (DCC-tiled buffers are NOT importable by the
#     encoder; nodcc yields modifier 0x...401a01 which is).
#   * Auth: the bridge is fail-closed EdDSA on a LAN (non-loopback) bind, so we
#     mint a local ephemeral Ed25519 keypair, hand the pubkey to the bridge via
#     env inheritance (spawn_bridge inherits our environment), and put the
#     signed token in the URL (?token=...). Private key is discarded at mint.
#   * Isolation: a private XDG_RUNTIME_DIR (mktemp), a dedicated port, a private
#     dbus-run-session for the nested desktop, no sudo / no /etc/hosts / no
#     installs. `down` kills only processes whose environ references THIS run's
#     runtime dir.
set -euo pipefail

PORT="${WAYMUX_LOCAL_PORT:-8082}"
WIDTH="${WAYMUX_LOCAL_WIDTH:-1280}"
HEIGHT="${WAYMUX_LOCAL_HEIGHT:-720}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/waymux-local"
STATE_FILE="$STATE_DIR/session-${PORT}.state"
MINT="$ROOT/scripts/laptop-mint-viewer-token.py"

log() { printf '[laptop-local %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

# --- binaries -----------------------------------------------------------------
# Always (re)build before launch. cargo/go are incremental, so this is ~instant
# when the binary is already current. The old "use it if it exists" logic would
# silently launch a STALE binary (e.g. a months-old artifact predating the
# current Vulkan encoder), which presents as "no video" with no obvious cause.
# Correctness beats the sub-second check; never shadow new code with old bits.
session_binary() {
  log "ensuring waymux-session is current (cargo build --release)..."
  ( cd "$ROOT" && cargo build --release -p waymux-session >&2 ) \
    || die "waymux-session build failed"
  echo "$ROOT/target/release/waymux-session"
}
bridge_binary() {
  local b="$ROOT/crates/waymux-neko-bridge/waymux-neko-bridge"
  log "ensuring waymux-neko-bridge is current (go build)..."
  ( cd "$ROOT/crates/waymux-neko-bridge" && go build -o waymux-neko-bridge . >&2 ) \
    || die "waymux-neko-bridge build failed"
  echo "$b"
}

# Find a KActivities manager daemon across distro layouts (Arch vs Ubuntu).
find_kactivitymanagerd() {
  local c
  for c in kactivitymanagerd \
           /usr/lib/kactivitymanagerd \
           /usr/lib/x86_64-linux-gnu/libexec/kactivitymanagerd \
           /usr/libexec/kactivitymanagerd; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
}

lan_ip() {
  ip -4 -o addr show scope global 2>/dev/null \
    | awk '$2 !~ /^(docker|br-|veth|wg)/ {print $4}' | cut -d/ -f1 | head -1
}

# --- up -----------------------------------------------------------------------
cmd_up() {
  command -v kwin_wayland >/dev/null || die "kwin_wayland not found (need KDE Plasma 6)"
  command -v dbus-run-session >/dev/null || die "dbus-run-session not found"
  [[ -f "$MINT" ]] || die "missing $MINT"
  if ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
    die "port $PORT already in use (another session up? try: $0 down, or set WAYMUX_LOCAL_PORT)"
  fi

  local SESSION BRIDGE
  SESSION="$(session_binary)"
  BRIDGE="$(bridge_binary)"

  local RUNTIME_DIR; RUNTIME_DIR="$(mktemp -d /tmp/waymux-local.XXXXXX)"; chmod 700 "$RUNTIME_DIR"
  mkdir -p "$STATE_DIR"

  # Mint the ephemeral viewer token; export PK + session id for the bridge.
  local VIEWER_TOKEN
  eval "$(python3 "$MINT")"   # sets WAYMUX_VIEWER_TOKEN_ED25519_PK / _VM_SESSION_ID / _TOKEN
  VIEWER_TOKEN="$WAYMUX_VIEWER_TOKEN"

  # 1. Headless session + LAN-bound viewer bridge.
  #
  # Localhost/LAN high-quality profile. The old cellular-safe 4 Mbps cap made
  # the picture soft on a loopback link that has effectively unlimited
  # bandwidth: the adaptive-QP loop raised QP toward 44 to hit 4 Mbps. We now
  # target 30 Mbps with a 40 Mbps ceiling and seed QP at the sharp floor (18).
  # WAYMUX_BWE_MIN_BPS stays low so a constrained tablet-over-WiFi link still
  # ramps down gracefully; GCC discovers the real capacity. These are ceilings,
  # not forced rates.
  log "starting headless waymux-session (${WIDTH}x${HEIGHT}) + bridge on 0.0.0.0:$PORT"
  XDG_RUNTIME_DIR="$RUNTIME_DIR" \
  WAYMUX_VIEWER_CODEC="${WAYMUX_VIEWER_CODEC:-h264-vulkan}" \
  WAYMUX_VIEWER_GOP="${WAYMUX_VIEWER_GOP:-60}" \
  WAYMUX_H264_PROFILE_LEVEL_ID="${WAYMUX_H264_PROFILE_LEVEL_ID:-4d0033}" \
  WAYMUX_VIEWER_MAX_FPS="${WAYMUX_VIEWER_MAX_FPS:-60}" \
  WAYMUX_VIEWER_BITRATE_BPS="${WAYMUX_VIEWER_BITRATE_BPS:-30000000}" \
  WAYMUX_VK_ENCODE_QP="${WAYMUX_VK_ENCODE_QP:-18}" \
  WAYMUX_BWE_MIN_BPS="${WAYMUX_BWE_MIN_BPS:-400000}" \
  WAYMUX_BWE_INITIAL_BPS="${WAYMUX_BWE_INITIAL_BPS:-20000000}" \
  WAYMUX_BWE_MAX_BPS="${WAYMUX_BWE_MAX_BPS:-40000000}" \
  WAYMUX_VIEWER_TOKEN_ED25519_PK="$WAYMUX_VIEWER_TOKEN_ED25519_PK" \
  WAYMUX_VM_SESSION_ID="$WAYMUX_VM_SESSION_ID" \
  WAYMUX_NEKO_BRIDGE_BIN="$BRIDGE" \
  RUST_LOG="${RUST_LOG:-info}" \
  setsid "$SESSION" --name localtest --width "$WIDTH" --height "$HEIGHT" \
    --inner-socket "$RUNTIME_DIR/inner.sock" \
    --control-socket "$RUNTIME_DIR/control.sock" \
    --viewer-port "$PORT" --viewer-bind 0.0.0.0 \
    >"$RUNTIME_DIR/session.log" 2>&1 < /dev/null &

  # Wait for the inner socket + bridge READY.
  local i
  for i in $(seq 1 100); do [[ -S "$RUNTIME_DIR/inner.sock" ]] && break; sleep 0.1; done
  [[ -S "$RUNTIME_DIR/inner.sock" ]] || die "session inner socket never appeared (see $RUNTIME_DIR/session.log)"
  for i in $(seq 1 100); do grep -q 'READY addr=' "$RUNTIME_DIR/session.log" && break; sleep 0.1; done

  # 2. Nested KDE desktop (private dbus bus; AMD_DEBUG=nodcc for importable dmabuf).
  log "starting nested KDE desktop (KWin 6 + Plasma, software-dbus-isolated)"
  local KACT; KACT="$(find_kactivitymanagerd || true)"
  XDG_RUNTIME_DIR="$RUNTIME_DIR" INNER="$RUNTIME_DIR/inner.sock" KACT="$KACT" \
  setsid dbus-run-session -- bash -c '
    set +e
    export AMD_DEBUG=nodcc RADV_DEBUG=nodcc
    export WAYLAND_DISPLAY="$INNER"
    KWIN_WAYLAND_NO_PERMISSION_CHECKS=1 XDG_CURRENT_DESKTOP=KDE XDG_SESSION_DESKTOP=KDE \
      QT_QPA_PLATFORM=wayland kwin_wayland --socket wayland-kwin --xwayland --no-lockscreen &
    KW=$!
    for i in $(seq 1 150); do [ -S "$XDG_RUNTIME_DIR/wayland-kwin" ] && break; sleep 0.1; done
    export WAYLAND_DISPLAY=wayland-kwin QT_QPA_PLATFORM=wayland \
           XDG_CURRENT_DESKTOP=KDE XDG_SESSION_DESKTOP=KDE XDG_MENU_PREFIX=plasma-
    [ -n "$KACT" ] && "$KACT" >/dev/null 2>&1 &
    kded6 >/dev/null 2>&1 &
    kglobalacceld >/dev/null 2>&1 &
    kbuildsycoca6 --noincremental >/dev/null 2>&1
    plasmashell >/dev/null 2>&1 &
    # Fallback content so the stream is never blank even if plasmashell bails.
    ( sleep 6
      if ! pgrep -x plasmashell >/dev/null 2>&1; then
        command -v foot >/dev/null && foot -e sh -c "echo waymux local viewer; exec bash" >/dev/null 2>&1 &
      fi ) &
    wait $KW
  ' >"$RUNTIME_DIR/kde.log" 2>&1 < /dev/null &

  # 3. Record state + emit the URL.
  local LANIP URL; LANIP="$(lan_ip)"; URL="http://${LANIP}:${PORT}/?token=${VIEWER_TOKEN}"
  {
    echo "RUNTIME_DIR=$RUNTIME_DIR"
    echo "PORT=$PORT"
    echo "URL=$URL"
    echo "TOKEN=$VIEWER_TOKEN"
  } > "$STATE_FILE"

  if command -v qrencode >/dev/null; then
    qrencode -o "$RUNTIME_DIR/tablet-qr.png" -s 6 -m 2 "$URL" 2>/dev/null \
      && log "QR: $RUNTIME_DIR/tablet-qr.png"
  fi
  cat >&2 <<EOF

============================================================
  waymux local viewer is UP on port $PORT
    this laptop:  http://127.0.0.1:${PORT}/?token=${VIEWER_TOKEN}
    the tablet:   $URL
  (same Wi-Fi/LAN; EdDSA token valid ~8h; nested KDE desktop,
   NOT your real screen.)  Tear down with:  $0 down
============================================================
EOF
}

# --- down ---------------------------------------------------------------------
cmd_down() {
  [[ -f "$STATE_FILE" ]] || { log "no state file for port $PORT ($STATE_FILE); nothing to do"; return 0; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  [[ -n "${RUNTIME_DIR:-}" ]] || die "state file has no RUNTIME_DIR"
  log "tearing down session for runtime dir $RUNTIME_DIR"
  # Kill ONLY processes whose environ references this run's runtime dir.
  # Safe: your real session uses /run/user/<uid>, never /tmp/waymux-local.*.
  local p pid n=0
  for p in /proc/[0-9]*; do
    if grep -qaF "$RUNTIME_DIR" "$p/environ" 2>/dev/null; then
      pid="${p#/proc/}"; kill -KILL "$pid" 2>/dev/null && n=$((n+1))
    fi
  done
  log "killed $n process(es)"
  rm -rf "$RUNTIME_DIR" 2>/dev/null || true
  rm -f "$STATE_FILE"
  if ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
    log "WARNING: port $PORT still held"
  else
    log "port $PORT free; teardown complete"
  fi
}

# --- status -------------------------------------------------------------------
cmd_status() {
  if [[ ! -f "$STATE_FILE" ]]; then echo "down (no session on port $PORT)"; return 0; fi
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  local alive=no
  if ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then alive=yes; fi
  echo "port $PORT: bridge-listening=$alive  runtime=$RUNTIME_DIR"
  echo "URL: ${URL:-?}"
  [[ -f "${RUNTIME_DIR:-/nonexistent}/session.log" ]] && \
    echo "encoder: $(grep 'vulkan: progress' "$RUNTIME_DIR/session.log" 2>/dev/null | tail -1 | sed -E 's/\x1b\[[0-9;]*m//g' | sed 's/^.*INFO //')"
}

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  *) echo "usage: $0 {up|down|status}   (env: WAYMUX_LOCAL_PORT=$PORT WAYMUX_LOCAL_WIDTH=$WIDTH WAYMUX_LOCAL_HEIGHT=$HEIGHT)" >&2; exit 2 ;;
esac
