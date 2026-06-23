#!/usr/bin/env bash
# tests/e2e/bench-headtohead.sh
#
# Head-to-head: record the SAME animated page (a glowing software-WebGL waymux
# logo) for the SAME duration on the SAME machine, two ways, and compare unique
# fps, file size, and CPU:
#
#   A) waymux  --mode focused-window, lossless FFV1 (zero-copy Wayland capture)
#   B) Xvfb + ffmpeg x11grab, lossless x264 (the standard X11 CI recording)
#
# The point is an apples-to-apples "lossless recording in CI" comparison: same
# source, same length, software rendering, no GPU. unique_fps is the honest
# metric (distinct frames via mpdecimate); the container's nominal fps pads with
# duplicates and is not comparable across tools.
#
# Env: RES (default 1280x720), RECORD_SECS (8), ARTIFACT_DIR (/tmp/wmx-h2h),
#      WAYMUX_BINDIR, PAGE (path to an .html; default writes the logo).
#
# Skips cleanly if a tool is missing (Xvfb / ffmpeg x11grab / chromium).

set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd))"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RES="${RES:-1280x720}"
W="${RES%x*}"; H="${RES#*x}"
RECORD_SECS="${RECORD_SECS:-8}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/wmx-h2h}"
mkdir -p "$ARTIFACT_DIR"

CHROMIUM=$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)
[ -n "$CHROMIUM" ] || { echo "SKIP: chromium not installed"; exit 0; }
command -v ffmpeg >/dev/null || { echo "SKIP: ffmpeg not installed"; exit 0; }
command -v ffprobe >/dev/null || { echo "SKIP: ffprobe not installed"; exit 0; }

export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}" WAYMUX_DISABLE_SYNCOBJ=1
WORK="$(mktemp -d /tmp/wmx-h2h-work.XXXXXX)"; OUT="$WORK"
trap 'pkill -9 -f wmx-h2h 2>/dev/null; rm -rf "$WORK"' EXIT

# Software WebGL flags (SwiftShader): WebGL with no GPU.
GL_FLAGS="--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --in-process-gpu"
CR_COMMON="--no-sandbox --no-first-run --disable-dev-shm-usage $GL_FLAGS"

# --- the animated page (or use $PAGE) ----------------------------------------
PAGE="${PAGE:-}"
if [ -z "$PAGE" ]; then
  PAGE="$WORK/logo.html"
  if [ -f site/logo.html ]; then cp site/logo.html "$PAGE"; else cp "$(dirname "${BASH_SOURCE[0]}")/logo.html" "$PAGE" 2>/dev/null || true; fi
fi
[ -f "$PAGE" ] || { echo "SKIP: no logo page found (set PAGE=...)"; exit 0; }
PAGE_URL="file://$PAGE"

# --- CPU sampling over a window (whole-system busy core-seconds) --------------
cpu_busy() { awk '/^cpu /{idle=$5+$6; tot=0; for(i=2;i<=NF;i++)tot+=$i; print tot, idle}' /proc/stat; }
NPROC=$(nproc 2>/dev/null || echo 1)

analyze() { # path -> "uniq_fps nom_fps file_mb codec"
  local f="$1" codec nb dur uniq bytes
  [ -f "$f" ] || { echo "0 0 0 missing"; return; }
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f" 2>/dev/null)
  nb=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$f" 2>/dev/null)
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  uniq=$(unique_frames "$f"); bytes=$(stat -c%s "$f" 2>/dev/null || echo 0)
  python3 -c "
d=float('${dur:-0}' or 0); print('%.2f %.2f %.2f %s' % (
  (int('${uniq:-0}')/d if d>0 else 0),(int('${nb:-0}')/d if d>0 else 0),${bytes}/1048576,'${codec}'))"
}

ROWS=""

# ============================================================================
# A) waymux focused-window, lossless FFV1
# ============================================================================
echo "=== A) waymux focused-window FFV1 ($RES, ${RECORD_SECS}s) ==="
WA_OK=0
XDG_RUNTIME_DIR="$(mktemp -d /tmp/wmx-h2h-run.XXXXXX)"; chmod 700 "$XDG_RUNTIME_DIR"; export XDG_RUNTIME_DIR
wmx_pick_bindir
waymuxd >"$WORK/daemon.log" 2>&1 & DPID=$!
for _ in $(seq 1 400); do [ -S "$XDG_RUNTIME_DIR/waymux.sock" ] && break; sleep 0.05; done

# Sweep one or more waymux lossless codecs (fresh session + page each).
WAYMUX_CODECS="${WAYMUX_CODECS:-ffv1 x264-lossless}"
idx=0
for CODEC in $WAYMUX_CODECS; do
  idx=$((idx + 1)); sess="h2h${idx}"; APG=""
  echo "  -- waymux --codec ${CODEC} --"
  if waymux --json new "$sess" --size "$RES" >"$WORK/a-${idx}-new.json" 2>&1 \
     && [ "$(jget '["ok"]' <"$WORK/a-${idx}-new.json")" = "True" ]; then
    INNER="$XDG_RUNTIME_DIR/waymux/${sess}/wayland.sock"
    WAYLAND_DISPLAY="$INNER" setsid $CHROMIUM --ozone-platform=wayland $CR_COMMON \
      --user-data-dir="$WORK/wmx-h2h-a-${idx}" --window-size="$W,$H" --app="$PAGE_URL" \
      >"$WORK/a-${idx}-cr.log" 2>&1 & APG=$!
    if [ "$(wait_window "$sess" chromium)" -ge 1 ] 2>/dev/null; then
      waymux idle "$sess" --quiet-ms 600 --timeout-ms 6000 >/dev/null 2>&1 || true
      waymux --json record start "$sess" --codec "$CODEC" --mode focused-window --min-fps 60 \
        >"$WORK/a-${idx}-rec.json" 2>&1
      RECP=$(jget '["data"]["path"]' <"$WORK/a-${idx}-rec.json" 2>/dev/null)
      read ct0 cb0 < <(cpu_busy); sleep "$RECORD_SECS"; read ct1 cb1 < <(cpu_busy)
      waymux --json record stop "$sess" >/dev/null 2>&1
      waymux idle "$sess" --quiet-ms 800 --timeout-ms 2500 >/dev/null 2>&1 || true
      A_CPU=$(python3 -c "dt=${ct1}-${ct0}; di=${cb1}-${cb0}; print('%.1f'%(((dt-di)/dt*${NPROC}) if dt>0 else 0))")
      if [ -n "$RECP" ] && [ -f "$RECP" ]; then
        cp "$RECP" "$ARTIFACT_DIR/waymux-${CODEC}.mkv"
        read A_U A_N A_MB A_C <<<"$(analyze "$RECP")"
        echo "  waymux/${CODEC}: uniq=$A_U nom=$A_N mb=$A_MB cpu=${A_CPU}/${NPROC} codec=$A_C"
        ROWS="${ROWS}| waymux focused-window (${CODEC}) | ${A_U} | ${A_N} | ${A_MB} | ${A_CPU} | ${A_C} |
"
      else
        echo "  waymux/${CODEC}: FAILED (no file; see $WORK/a-${idx}-rec.json)"
      fi
    fi
    waymux --json rm "$sess" >/dev/null 2>&1 || true
  fi
  [ -n "$APG" ] && { kill -9 -- "-$APG" 2>/dev/null || true; }
done
kill "$DPID" 2>/dev/null || true

# ============================================================================
# B) Xvfb + ffmpeg x11grab, lossless x264
# ============================================================================
echo "=== B) Xvfb + ffmpeg x11grab x264-lossless ($RES, ${RECORD_SECS}s) ==="
XV=$(command -v Xvfb || true)
if [ -z "$XV" ]; then
  echo "  SKIP: Xvfb not installed"
else
  DISP=":97"; Xvfb "$DISP" -screen 0 "${W}x${H}x24" >"$WORK/xvfb.log" 2>&1 & XPID=$!
  sleep 1
  DISPLAY="$DISP" setsid $CHROMIUM --window-position=0,0 $CR_COMMON \
    --user-data-dir="$WORK/wmx-h2h-b" --window-size="$W,$H" --start-fullscreen \
    --app="$PAGE_URL" >"$WORK/b-cr.log" 2>&1 & BPG=$!
  sleep 4
  BX="$ARTIFACT_DIR/x11grab-x264.mkv"
  read ct0 ci0 < <(cpu_busy)
  DISPLAY="$DISP" ffmpeg -y -loglevel error -f x11grab -draw_mouse 0 -framerate 60 \
    -video_size "${W}x${H}" -i "$DISP" -t "$RECORD_SECS" \
    -c:v libx264rgb -qp 0 -preset ultrafast "$BX" >"$WORK/b-ff.log" 2>&1
  read ct1 ci1 < <(cpu_busy)
  B_CPU=$(python3 -c "dt=${ct1}-${ct0}; di=${ci1}-${ci0}; print('%.1f'%(((dt-di)/dt*${NPROC}) if dt>0 else 0))")
  kill -9 -- "-$BPG" 2>/dev/null || true; kill "$XPID" 2>/dev/null || true
  if [ -f "$BX" ]; then read B_U B_N B_MB B_C <<<"$(analyze "$BX")"
    echo "  x11grab: uniq=$B_U nom=$B_N mb=$B_MB cpu=${B_CPU}/${NPROC} codec=$B_C"
    ROWS="${ROWS}| Xvfb + ffmpeg x11grab (x264 lossless) | ${B_U} | ${B_N} | ${B_MB} | ${B_CPU} | ${B_C} |
"
  else echo "  x11grab: FAILED (see $WORK/b-ff.log)"; fi
fi

# --- report ------------------------------------------------------------------
HOST_CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
{
  echo "# waymux vs Xvfb+ffmpeg: lossless recording head-to-head"
  echo ""
  echo "Same machine, same animated page (software-WebGL waymux logo), ${RES}, ${RECORD_SECS}s, no GPU."
  echo "\`uniq_fps\` = distinct frames/sec (mpdecimate); \`cpu\` = mean busy cores during the record window."
  echo ""
  echo "- host: ${HOST_CPU}, ${NPROC} threads"
  echo ""
  echo "| method | uniq_fps | nom_fps | file_mb | cpu | codec |"
  echo "|--------|----------|---------|---------|-----|-------|"
  printf '%s' "$ROWS"
} | tee "$ARTIFACT_DIR/headtohead.md"

echo ""
echo "artifacts in $ARTIFACT_DIR: headtohead.md, waymux-focused.mkv, x11grab-x264.mkv"
