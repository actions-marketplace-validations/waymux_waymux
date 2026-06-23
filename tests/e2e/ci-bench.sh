#!/usr/bin/env bash
# tests/e2e/ci-bench.sh - small CI benchmark with functional gate.
#
# Sweeps four recording configurations (chromium animated page at two
# resolutions/modes, kwrite at 1920x1080 whole-desktop) and emits:
#   $ARTIFACT_DIR/benchmark.md   - markdown table (one row per config)
#   $ARTIFACT_DIR/benchmark.json - same rows as a JSON array
#   $ARTIFACT_DIR/bench-sample.mkv - one saved recording
#
# Functional gate (only hard failures):
#   - recording codec is not ffv1
#   - unique_fps == 0 (all frames duplicate; capture frozen)
#   - screenshot blank (assert_content fails)
#
# NO latency or fps thresholds; this is a capability check, not a perf gate.
#
# Environment:
#   ARTIFACT_DIR    output directory (default: /tmp/wmx-bench-out)
#   WAYMUX_BINDIR   directory containing waymux/waymuxd (overrides auto-detect)
#   RECORD_SECS     recording duration in seconds per config (default: 5)
#   GALLIUM_DRIVER  Mesa software driver (default: llvmpipe)
#
# SKIP conditions: ffprobe, ffmpeg, python3, or chromium not installed.
#
# Exit 0 = functional gate passed; non-zero = at least one gate failure.

set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd))"

# shellcheck source=tests/e2e/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# --- Dependency checks (SKIP rather than fail on a minimal CI image) ---------
command -v ffprobe >/dev/null || { echo "SKIP: ffprobe not installed"; exit 0; }
command -v ffmpeg  >/dev/null || { echo "SKIP: ffmpeg not installed";  exit 0; }
command -v python3 >/dev/null || { echo "SKIP: python3 not installed"; exit 0; }

CHROMIUM=$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)
[ -n "$CHROMIUM" ] || { echo "SKIP: chromium not installed"; exit 0; }

# --- Force software rendering ------------------------------------------------
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
export WAYMUX_DISABLE_SYNCOBJ=1

# --- Config ------------------------------------------------------------------
RECORD_SECS="${RECORD_SECS:-5}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/wmx-bench-out}"
mkdir -p "$ARTIFACT_DIR"

wmx_pick_bindir

# --- Scratch directories -----------------------------------------------------
export XDG_RUNTIME_DIR
XDG_RUNTIME_DIR="$(mktemp -d /tmp/wmx-bench-run.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"

WORK="$(mktemp -d /tmp/wmx-bench-work.XXXXXX)"
# unique_frames() in lib.sh uses OUT as a temp dir
OUT="$WORK"

SOCK="$XDG_RUNTIME_DIR/waymux.sock"

# --- Tracking PIDs for cleanup -----------------------------------------------
DPID=""
CPGID="" # chromium setsid pgid (for current config)
KPID=""  # kwrite setsid session leader (for current config)
CUR_CFG_IDX=0

cleanup() {
  pkill -9 -f "wmx-bench-prof" 2>/dev/null || true
  [ -n "$CPGID" ] && { kill -9 -- "-$CPGID" 2>/dev/null || true; }
  [ -n "$KPID"  ] && { kill -9 -- "-$KPID"  2>/dev/null || true; kill -9 "$KPID" 2>/dev/null || true; }
  kill "${DPID:-0}" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

# --- Start waymuxd -----------------------------------------------------------
echo "=== ci-bench: start waymuxd (software, syncobj disabled) ==="
waymuxd >"$WORK/daemon.log" 2>&1 &
DPID=$!
for _ in $(seq 1 400); do
  [ -S "$SOCK" ] && break
  sleep 0.05
  kill -0 "$DPID" 2>/dev/null || break
done
if [ -S "$SOCK" ]; then
  ok "daemon socket up (software rendering)"
else
  bad "daemon socket never appeared"
  exit 1
fi

# --- Animated page for Chromium ----------------------------------------------
cat >"$WORK/page.html" <<'HTML'
<!doctype html><meta charset=utf-8><title>waymux bench</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#0a0a0a;font-family:sans-serif}
  /* Full-width fine-stripe band across the exact vertical centre. The non-blank
     screenshot gate samples the centre crop; the band guarantees high-contrast
     edges there in any capture mode, while the counter + boxes provide motion. */
  #band{position:absolute;top:calc(50% - 90px);left:0;width:100%;height:180px;
        background:repeating-linear-gradient(90deg,#fff 0 7px,#101014 7px 14px)}
  #wrap{position:absolute;inset:0;z-index:2;display:flex;flex-direction:column;
        align-items:center;justify-content:center;
        text-shadow:0 0 16px #000,0 0 16px #000,0 4px 12px #000}
  #title{font:bold 52px/1 sans-serif;color:#00d9ff;letter-spacing:3px}
  #frame{font:bold 150px/1 monospace;color:#fff;font-variant-numeric:tabular-nums}
  #sub{font:20px/1 monospace;color:#ddd;margin-top:8px}
  #bar{position:absolute;top:0;left:0;height:10px;background:#00d9ff;z-index:3;width:0}
  .box{position:absolute;width:120px;height:120px;border-radius:18px;z-index:1}
  #b1{background:#e1004b}#b2{background:#0a84ff}
</style>
<div id=band></div>
<div id=wrap>
  <div id=title>WAYMUX CI BENCH</div>
  <div id=frame>00000</div>
  <div id=sub>frames captured under llvmpipe, no GPU</div>
</div>
<div id=bar></div><div class=box id=b1></div><div class=box id=b2></div>
<script>
  // A per-frame counter makes every rendered frame distinct, so the recording's
  // unique-frame count is the real captured fps; the boxes and top bar sweep so
  // motion is obvious in the saved clip.
  let n = 0, t0 = performance.now();
  const fr = document.getElementById('frame'), bar = document.getElementById('bar'),
        b1 = document.getElementById('b1'), b2 = document.getElementById('b2');
  function tick(t) {
    n++; fr.textContent = String(n).padStart(5, '0');
    const W = innerWidth, H = innerHeight, p = (((t - t0) / 3000) % 1);
    bar.style.width = (p * W) + 'px';
    b1.style.left = (p * (W - 120)) + 'px';       b1.style.top = (H * 0.18) + 'px';
    b2.style.left = ((1 - p) * (W - 120)) + 'px'; b2.style.top = (H * 0.72) + 'px';
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
</script>
HTML

# --- KWrite pre-filled document ----------------------------------------------
# 80 long lines packed with characters so glyphs fill the visible area even at
# 1920x1080 and the center crop has non-zero luma contrast.
DOCFILE="$WORK/note.txt"
: >"$DOCFILE"
for i in $(seq 1 80); do
  printf 'line %02d  waymux headless Wayland benchmark llvmpipe (no GPU) XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n' "$i" >>"$DOCFILE"
done

# --- Benchmark state ---------------------------------------------------------
# Space-separated fields: name res mode client
CONFIGS=(
  "chromium-anim-1280x720-whole  1280x720  whole-desktop  chromium"
  "chromium-anim-1920x1080-whole 1920x1080 whole-desktop  chromium"
  "chromium-anim-1920x1080-focus 1920x1080 focused-window chromium"
  "kwrite-1920x1080-whole        1920x1080 whole-desktop  kwrite"
)

JSON_ROWS=""
MD_ROWS=""
SAMPLE_SAVED=0

# --- Helper: time three screenshots and return median ms --------------------
screenshot_ms_median() {
  local sess="$1" png_base="$2"
  local t0 t1 t2 t3 ms0 ms1 ms2
  t0=$(date +%s%3N)
  waymux screenshot-desktop "$sess" -o "${png_base}_0.png" >/dev/null 2>&1
  t1=$(date +%s%3N); ms0=$(( t1 - t0 ))
  waymux screenshot-desktop "$sess" -o "${png_base}_1.png" >/dev/null 2>&1
  t2=$(date +%s%3N); ms1=$(( t2 - t1 ))
  waymux screenshot-desktop "$sess" -o "${png_base}_2.png" >/dev/null 2>&1
  t3=$(date +%s%3N); ms2=$(( t3 - t2 ))
  python3 -c "print(sorted([${ms0},${ms1},${ms2}])[1])"
}

# --- Helper: mkv duration (format-level; stream header may lack it) ---------
mkv_duration() {
  ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null | head -1
}

# --- Run each config ---------------------------------------------------------
for cfg_line in "${CONFIGS[@]}"; do
  cfg_name=$(  echo "$cfg_line" | awk '{print $1}')
  cfg_res=$(   echo "$cfg_line" | awk '{print $2}')
  cfg_mode=$(  echo "$cfg_line" | awk '{print $3}')
  cfg_client=$(echo "$cfg_line" | awk '{print $4}')

  CUR_CFG_IDX=$(( CUR_CFG_IDX + 1 ))
  sess="bench${CUR_CFG_IDX}"

  echo ""
  echo "--- config: $cfg_name ($cfg_res, $cfg_mode) ---"

  # kwrite: check presence (chromium already checked above)
  if [ "$cfg_client" = "kwrite" ]; then
    KWRITE=$(command -v kwrite || true)
    if [ -z "$KWRITE" ]; then
      echo "SKIP: kwrite not installed, skipping $cfg_name"
      continue
    fi
  fi

  # Create session
  waymux --json new "$sess" --size "$cfg_res" >"$WORK/${sess}-new.json" 2>&1
  if [ "$(jget '["ok"]' <"$WORK/${sess}-new.json" 2>/dev/null)" != "True" ]; then
    bad "$cfg_name: session create failed: $(cat "$WORK/${sess}-new.json")"
    continue
  fi

  INNER="$XDG_RUNTIME_DIR/waymux/${sess}/wayland.sock"
  CPGID=""
  KPID=""

  # Launch client
  if [ "$cfg_client" = "chromium" ]; then
    WAYLAND_DISPLAY="$INNER" setsid "$CHROMIUM" \
      --ozone-platform=wayland --no-sandbox --no-first-run \
      --disable-gpu --disable-gpu-compositing --in-process-gpu \
      --disable-dev-shm-usage \
      --user-data-dir="$WORK/wmx-bench-prof-${CUR_CFG_IDX}" \
      --app="file://$WORK/page.html" \
      >"$WORK/${sess}-client.log" 2>&1 &
    CPGID=$!
  else
    WAYLAND_DISPLAY="$INNER" QT_QPA_PLATFORM=wayland setsid "$KWRITE" "$DOCFILE" \
      >"$WORK/${sess}-client.log" 2>&1 &
    KPID=$!
  fi

  # Wait for window
  WN=$(wait_window "$sess" "$cfg_client")
  if ! [ "${WN:-0}" -ge 1 ] 2>/dev/null; then
    bad "$cfg_name: window never appeared (see $WORK/${sess}-client.log)"
    waymux --json rm "$sess" >/dev/null 2>&1 || true
    [ -n "$CPGID" ] && { kill -9 -- "-$CPGID" 2>/dev/null || true; } ; CPGID=""
    [ -n "$KPID"  ] && { kill -9 -- "-$KPID" 2>/dev/null || true; kill -9 "$KPID" 2>/dev/null || true; }; KPID=""
    continue
  fi

  # Settle the compositor before screenshotting.
  waymux idle "$sess" --quiet-ms 700 --timeout-ms 6000 >/dev/null 2>&1 || true

  # For kwrite: move the cursor into the middle of the text so the editor
  # body centre is covered by a blinking cursor or selected text rather than
  # a blank white line gap. Three Page-Down presses (keycode 109) move the
  # view; End (107) then a printable char ensure at least one glyph lands in
  # the crop region regardless of font size or line spacing.
  if [ "$cfg_client" = "kwrite" ]; then
    for kc in 109 109 109; do
      waymux key "$sess" "$kc" >/dev/null 2>&1
      waymux idle "$sess" --quiet-ms 150 --timeout-ms 800 >/dev/null 2>&1 || true
    done
  fi

  # Screenshot x3 (for median latency measurement)
  SCPNG="$WORK/${sess}-shot"
  SHOT_MS=$(screenshot_ms_median "$sess" "$SCPNG")
  assert_content "$cfg_name screenshot" "${SCPNG}_1.png"

  # Start recording
  waymux --json record start "$sess" --codec ffv1 --min-fps 30 --mode "$cfg_mode" \
    >"$WORK/${sess}-rec.json" 2>&1
  RECPATH=$(jget '["data"]["path"]' <"$WORK/${sess}-rec.json" 2>/dev/null || echo "")
  if [ -z "$RECPATH" ] || [ "$RECPATH" = "None" ]; then
    bad "$cfg_name: record start failed: $(cat "$WORK/${sess}-rec.json")"
    waymux --json rm "$sess" >/dev/null 2>&1 || true
    [ -n "$CPGID" ] && { kill -9 -- "-$CPGID" 2>/dev/null || true; }; CPGID=""
    [ -n "$KPID"  ] && { kill -9 -- "-$KPID" 2>/dev/null || true; kill -9 "$KPID" 2>/dev/null || true; }; KPID=""
    continue
  fi
  ok "$cfg_name: recording started ($RECPATH)"

  # For kwrite: inject keystrokes so frames differ during recording.
  # Evdev keycodes: W=17 A=30 Y=21 M=50 U=22 X=45.
  if [ "$cfg_client" = "kwrite" ]; then
    for kc in 17 30 21 50 22 45; do
      waymux key "$sess" "$kc" >/dev/null 2>&1
      waymux idle "$sess" --quiet-ms 200 --timeout-ms 1000 >/dev/null 2>&1 || true
    done
  fi

  sleep "$RECORD_SECS"

  waymux --json record stop "$sess" >/dev/null 2>&1
  waymux idle "$sess" --quiet-ms 800 --timeout-ms 2500 >/dev/null 2>&1 || true

  # Analyze recording
  CODEC="missing"
  NOM_FPS="0"
  UNIQ_FPS="0"
  FILE_MB="0"

  if [ -f "$RECPATH" ]; then
    CODEC=$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=codec_name -of csv=p=0 "$RECPATH" 2>/dev/null || echo "")
    NB_FRAMES=$(ffprobe -v error -count_frames -select_streams v:0 \
      -show_entries stream=nb_read_frames -of csv=p=0 "$RECPATH" 2>/dev/null || echo "0")
    # Format-level duration is reliable for ffv1/mkv when stream header lacks it.
    DURATION=$(mkv_duration "$RECPATH")
    UNIQ_COUNT=$(unique_frames "$RECPATH")
    FILE_BYTES=$(stat -c%s "$RECPATH" 2>/dev/null || echo "0")

    NOM_FPS=$(python3 -c "
d=float('${DURATION:-0}' or 0); n=int('${NB_FRAMES:-0}' or 0)
print('%.2f' % (n/d if d>0 else 0))")
    UNIQ_FPS=$(python3 -c "
d=float('${DURATION:-0}' or 0); u=int('${UNIQ_COUNT:-0}' or 0)
print('%.2f' % (u/d if d>0 else 0))")
    FILE_MB=$(python3 -c "print('%.2f' % (${FILE_BYTES}/1048576))")

    # Functional gate
    if [ "$CODEC" != "ffv1" ]; then
      bad "$cfg_name: codec='$CODEC' want ffv1"
    fi
    if python3 -c "import sys; sys.exit(0 if float('${UNIQ_FPS}')>0 else 1)" 2>/dev/null; then
      ok "$cfg_name: unique_fps=${UNIQ_FPS} nom_fps=${NOM_FPS} file_mb=${FILE_MB}"
    else
      bad "$cfg_name: unique_fps=0 (capture frozen)"
    fi

    # Save the first successful recording as bench-sample.mkv
    if [ "$SAMPLE_SAVED" -eq 0 ]; then
      cp "$RECPATH" "$ARTIFACT_DIR/bench-sample.mkv" 2>/dev/null && SAMPLE_SAVED=1
    fi
  else
    bad "$cfg_name: recording file missing: $RECPATH"
  fi

  # Accumulate output rows
  ROW_JSON="{\"config\":\"${cfg_name}\",\"screenshot_ms\":${SHOT_MS},\"uniq_fps\":${UNIQ_FPS},\"nom_fps\":${NOM_FPS},\"file_mb\":${FILE_MB},\"codec\":\"${CODEC}\"}"
  if [ -z "$JSON_ROWS" ]; then
    JSON_ROWS="$ROW_JSON"
  else
    JSON_ROWS="${JSON_ROWS},${ROW_JSON}"
  fi
  MD_ROWS="${MD_ROWS}| ${cfg_name} | ${SHOT_MS} | ${UNIQ_FPS} | ${NOM_FPS} | ${FILE_MB} | ${CODEC} |
"

  # Teardown this config
  waymux --json rm "$sess" >/dev/null 2>&1 || true
  if [ "$cfg_client" = "chromium" ]; then
    pkill -9 -f "wmx-bench-prof-${CUR_CFG_IDX}" 2>/dev/null || true
    [ -n "$CPGID" ] && { kill -9 -- "-$CPGID" 2>/dev/null || true; }; CPGID=""
  else
    [ -n "$KPID" ] && { kill -9 -- "-$KPID" 2>/dev/null || true; kill -9 "$KPID" 2>/dev/null || true; }; KPID=""
  fi

done

# --- Host specs (makes results reproducible + comparable across runners) -----
HOST_NPROC="$(nproc 2>/dev/null || echo '?')"
HOST_CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || echo '?')"
HOST_MEM="$(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo 2>/dev/null || echo '?')"
HOST_KERNEL="$(uname -sr 2>/dev/null || echo '?')"
HOST_RENDERER="llvmpipe (Mesa software, no GPU)"

# --- Write benchmark.md ------------------------------------------------------
{
  echo "# waymux headless recording benchmark"
  echo ""
  echo "Lossless FFV1, software-rendered (llvmpipe), no GPU. \`uniq_fps\` counts"
  echo "genuinely distinct frames (ffmpeg mpdecimate); \`nom_fps\` is the container's"
  echo "nominal rate (padded toward --min-fps with duplicate frames)."
  echo ""
  echo "- host: ${HOST_CPU}, ${HOST_NPROC} threads, ${HOST_MEM}"
  echo "- kernel: ${HOST_KERNEL}"
  echo "- renderer: ${HOST_RENDERER}"
  echo "- record_secs: ${RECORD_SECS}"
  echo ""
  echo "| config | screenshot_ms | uniq_fps | nom_fps | file_mb | codec |"
  echo "|--------|---------------|----------|---------|---------|-------|"
  printf '%s' "$MD_ROWS"
} >"$ARTIFACT_DIR/benchmark.md"

# --- Write benchmark.json ----------------------------------------------------
{
  printf '{"host":{"cpu":"%s","threads":"%s","mem":"%s","kernel":"%s","renderer":"%s"},' \
    "$HOST_CPU" "$HOST_NPROC" "$HOST_MEM" "$HOST_KERNEL" "$HOST_RENDERER"
  printf '"record_secs":%s,"configs":[%s]}\n' "$RECORD_SECS" "$JSON_ROWS"
} >"$ARTIFACT_DIR/benchmark.json"

echo ""
echo "===== ci-bench: $PASS passed, $FAIL failed ====="
echo "  benchmark.md:       $ARTIFACT_DIR/benchmark.md"
echo "  benchmark.json:     $ARTIFACT_DIR/benchmark.json"
[ "$SAMPLE_SAVED" -eq 1 ] && echo "  bench-sample.mkv:   $ARTIFACT_DIR/bench-sample.mkv"

[ "$FAIL" -eq 0 ]
