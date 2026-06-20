#!/usr/bin/env bash
# waymux end-to-end test harness (Phase 1 + Phase 2).
#
# Builds the binaries, starts waymuxd headless, then drives the full session
# lifecycle through the `waymux` CLI (--json) and the `waymux-mcp` server, with
# real assertions against a live daemon. Spawns `foot` (a Wayland terminal) as
# the inner client. Uses `waymux idle/wait` rather than sleep for timing.
#
#   tests/e2e/run-e2e.sh            # build + run
#   WAYMUX_E2E_NO_BUILD=1 ...       # skip the build
#
# Exit 0 = all checks pass; non-zero = a check failed.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd))"

PASS=0; FAIL=0
ok()   { echo "PASS: $*"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $*"; FAIL=$((FAIL+1)); }
jget() { python3 -c "import sys,json;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]))" "$1"; }

FOOT=$(command -v foot) || { echo "SKIP: foot not installed"; exit 0; }  # absolute path: spawn requires it
command -v ffprobe >/dev/null || { echo "SKIP: ffprobe not installed"; exit 0; }

if [ "${WAYMUX_E2E_NO_BUILD:-0}" != "1" ]; then
  echo "=== build ==="
  CARGO_TERM_COLOR=never cargo build -p waymux-cli -p waymux-daemon -p waymux-session -p waymux-mcp 2>&1 | tail -1
fi

export PATH="$PWD/target/debug:$PATH"
export XDG_RUNTIME_DIR=/tmp/wmx-e2e-run
rm -rf "$XDG_RUNTIME_DIR"; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
SOCK="$XDG_RUNTIME_DIR/waymux.sock"
OUT=/tmp/wmx-e2e-out; rm -rf "$OUT"; mkdir -p "$OUT"

cleanup() { kill "${DPID:-0}" 2>/dev/null; }
trap cleanup EXIT

echo "=== start waymuxd ==="
waymuxd >/tmp/wmx-e2e-daemon.log 2>&1 &
DPID=$!
for _ in $(seq 1 3000000); do [ -S "$SOCK" ] && break; kill -0 "$DPID" 2>/dev/null || break; done
[ -S "$SOCK" ] && ok "daemon socket up" || { bad "daemon socket"; exit 1; }

# --- Phase 1: lifecycle + --json -----------------------------------------
waymux --json new e2e --size 800x600 >"$OUT/new.json" 2>&1
[ "$(jget '["ok"]' <"$OUT/new.json")" = "True" ] && ok "new --json ok" || bad "new: $(cat "$OUT/new.json")"

waymux --json ls >"$OUT/ls.json" 2>&1
[ "$(jget '["data"]["sessions"][0]["name"]' <"$OUT/ls.json")" = "e2e" ] && ok "ls shows session" || bad "ls: $(cat "$OUT/ls.json")"

# --- spawn a real Wayland client + wait for its window -------------------
waymux --json spawn e2e -- "$FOOT" >"$OUT/spawn.json" 2>&1
[ "$(jget '["ok"]' <"$OUT/spawn.json" 2>/dev/null || echo no)" = "True" ] && ok "spawn foot ok" || bad "spawn: $(cat "$OUT/spawn.json")"
waymux wait e2e --app-id foot --timeout-ms 15000 >"$OUT/wait.txt" 2>&1
waymux --json windows e2e >"$OUT/windows.json" 2>&1
WID=$(jget '["data"]["windows"][0]["id"]' <"$OUT/windows.json" 2>/dev/null || echo "")
if [ -n "$WID" ] && [ "$WID" != "None" ]; then ok "spawn+windows: foot window id=$WID"; else bad "windows: $(cat "$OUT/windows.json")"; fi

# --- tag + windows --tag filter -----------------------------------------
if [ -n "$WID" ] && [ "$WID" != "None" ]; then
  waymux --json tag e2e "$WID" demo >"$OUT/tag.json" 2>&1
  [ "$(jget '["ok"]' <"$OUT/tag.json")" = "True" ] && ok "tag --json ok" || bad "tag: $(cat "$OUT/tag.json")"
  waymux --json windows e2e --tag demo >"$OUT/wtag.json" 2>&1
  N=$(python3 -c "import json;print(len(json.load(open('$OUT/wtag.json'))['data']['windows']))" 2>/dev/null || echo 0)
  [ "$N" -ge 1 ] 2>/dev/null && ok "windows --tag demo returns $N tagged window(s)" || bad "windows --tag: $(cat "$OUT/wtag.json")"
fi

# --- screenshot (assert PNG) --------------------------------------------
waymux --json screenshot-desktop e2e -o "$OUT/shot.png" >"$OUT/shot.json" 2>&1
B64LEN=$(python3 -c "import json;d=json.load(open('$OUT/shot.json'));print(len(d['data'].get('png_b64','')) if d.get('ok') else 0)" 2>/dev/null || echo 0)
[ "$B64LEN" -gt 100 ] 2>/dev/null && ok "screenshot --json png_b64 len=$B64LEN" || bad "screenshot: $(head -c 200 "$OUT/shot.json")"
# non-json screenshot writes a file we can probe
waymux screenshot-desktop e2e -o "$OUT/shot2.png" >/dev/null 2>&1
if ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$OUT/shot2.png" >"$OUT/shotdims.txt" 2>&1; then
  ok "screenshot file is a valid image ($(cat "$OUT/shotdims.txt"))"
else bad "screenshot file invalid: $(cat "$OUT/shotdims.txt")"; fi

# --- resize: assert the new output mode propagates -----------------------
# Baseline desktop dims (session was created at 800x600).
ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$OUT/shot2.png" >"$OUT/predims.txt" 2>&1
PREDIMS=$(cat "$OUT/predims.txt")
# Resize to a distinct size, settle (no sleep; wait for the client/compositor
# to quiesce after the configure), then re-screenshot. capture_desktop sizes
# the PNG to the session's logical output, so the dims must follow the resize.
waymux --json resize e2e 640x480 >"$OUT/resize.json" 2>&1
[ "$(jget '["ok"]' <"$OUT/resize.json" 2>/dev/null)" = "True" ] && ok "resize --json ok" || bad "resize: $(cat "$OUT/resize.json")"
waymux idle e2e --quiet-ms 500 --timeout-ms 4000 >/dev/null 2>&1 || true
# info must report the new logical size.
waymux --json info e2e >"$OUT/info.json" 2>&1
IW=$(jget '["data"]["width"]' <"$OUT/info.json" 2>/dev/null || echo "")
IH=$(jget '["data"]["height"]' <"$OUT/info.json" 2>/dev/null || echo "")
if [ "$IW" = "640" ] && [ "$IH" = "480" ]; then ok "info reports resized 640x480"; else bad "info after resize: $(cat "$OUT/info.json")"; fi
# screenshot dims must change from the baseline to the new size.
waymux screenshot-desktop e2e -o "$OUT/shot_resized.png" >/dev/null 2>&1
if ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$OUT/shot_resized.png" >"$OUT/postdims.txt" 2>&1; then
  POSTDIMS=$(cat "$OUT/postdims.txt")
  if [ "$POSTDIMS" = "640,480" ] && [ "$POSTDIMS" != "$PREDIMS" ]; then
    ok "resize propagated: desktop dims $PREDIMS -> $POSTDIMS"
  else
    bad "resize did not propagate to screenshot: before=$PREDIMS after=$POSTDIMS (expected 640,480)"
  fi
else bad "post-resize screenshot invalid: $(cat "$OUT/postdims.txt")"; fi

# --- record (FFV1) then ffprobe -----------------------------------------
waymux --json record start e2e --codec ffv1 --min-fps 10 >"$OUT/rec.json" 2>&1
RECPATH=$(jget '["data"]["path"]' <"$OUT/rec.json" 2>/dev/null || echo "")
[ -n "$RECPATH" ] && [ "$RECPATH" != "None" ] && ok "record start path=$RECPATH" || bad "record start: $(cat "$OUT/rec.json")"
# record status while active: recording=true with the same path + codec.
waymux --json record status e2e >"$OUT/recstat.json" 2>&1
[ "$(jget '["data"]["recording"]' <"$OUT/recstat.json" 2>/dev/null)" = "True" ] && ok "record status reports recording=true" || bad "record status (active): $(cat "$OUT/recstat.json")"
# The recorder waits for the first frame AFTER start; foot is idle, so inject
# keystrokes to force redraws (commits). --min-fps then maintains the stream.
for kc in 30 31 32 33 28; do waymux key e2e "$kc" >/dev/null 2>&1; done
waymux idle e2e --quiet-ms 700 --timeout-ms 3000 >/dev/null 2>&1 || true
for kc in 34 35 36; do waymux key e2e "$kc" >/dev/null 2>&1; done
waymux --json record stop e2e >"$OUT/recstop.json" 2>&1
# record status after stop: recording=false.
waymux --json record status e2e >"$OUT/recstat2.json" 2>&1
[ "$(jget '["data"]["recording"]' <"$OUT/recstat2.json" 2>/dev/null)" = "False" ] && ok "record status reports recording=false after stop" || bad "record status (stopped): $(cat "$OUT/recstat2.json")"
if [ -n "${RECPATH:-}" ] && [ "$RECPATH" != "None" ] && [ -f "$RECPATH" ]; then
  CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$RECPATH" 2>/dev/null)
  [ "$CODEC" = "ffv1" ] && ok "recording is a valid FFV1 MKV" || bad "record codec=$CODEC (expected ffv1)"
else bad "record file missing: $RECPATH"; fi

# --- Phase 2: MCP server live against the daemon ------------------------
MCPREQ='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"waymux_ls","arguments":{}}}'
printf '%s\n' "$MCPREQ" | waymux-mcp >"$OUT/mcp.out" 2>/tmp/wmx-mcp.log || true
if grep -q '"serverInfo"' "$OUT/mcp.out" && grep -q 'waymux_ls' "$OUT/mcp.out"; then
  ok "MCP initialize + tools/list responded"
else bad "MCP init/list: $(head -c 300 "$OUT/mcp.out")"; fi
# the tools/call(waymux_ls) result should contain the e2e session via the live daemon
if grep -q '"e2e"' "$OUT/mcp.out"; then ok "MCP tools/call waymux_ls saw the live session"; else bad "MCP tools/call: $(tail -c 400 "$OUT/mcp.out")"; fi

# --- Phase 3: nested KWin/Plasma whole-desktop record (zero-copy GPU) ----
# A nested compositor needs a GPU + KWin + an animating client, none of which
# exist on CI runners, so this whole block SKIPs cleanly there. Where present
# it proves a nested Plasma/KWin desktop records to a real H.264 file via the
# zero-copy dmabuf tee, exercising the record-start frame priming that lets an
# otherwise-static desktop produce output immediately.
if command -v kwin_wayland >/dev/null && command -v dbus-run-session >/dev/null \
   && command -v glmark2-wayland >/dev/null; then
  echo "=== nested KDE whole-desktop record (h264-vulkan, zero-copy) ==="
  waymux --json new kde --size 1280x720 >"$OUT/kde-new.json" 2>&1
  KINNER="$XDG_RUNTIME_DIR/waymux/kde/wayland.sock"
  # KWin presents its whole composited desktop as one output surface. glmark2
  # animates inside it so KWin keeps committing real dmabuf frames (an idle
  # desktop drops to a single-pixel buffer). nodcc => importable dmabuf modifier.
  cat >"$OUT/kde-launch.sh" <<'LAUNCH'
#!/bin/bash
set +e
exec dbus-run-session -- env \
  AMD_DEBUG=nodcc RADV_DEBUG=nodcc WAYLAND_DISPLAY="$1" \
  KWIN_WAYLAND_NO_PERMISSION_CHECKS=1 XDG_CURRENT_DESKTOP=KDE QT_QPA_PLATFORM=wayland \
  bash -c '
    kwin_wayland --socket wayland-kwin --no-lockscreen &
    KW=$!
    for _ in $(seq 1 150); do [ -S "$XDG_RUNTIME_DIR/wayland-kwin" ] && break; sleep 0.1; done
    WAYLAND_DISPLAY=wayland-kwin glmark2-wayland --run-forever >/dev/null 2>&1 &
    wait $KW
  '
LAUNCH
  setsid bash "$OUT/kde-launch.sh" "$KINNER" >"$OUT/kde-stack.log" 2>&1 &
  # Wait for KWin to present its output surface to the session.
  for _ in $(seq 1 60); do
    n=$(waymux --json windows kde 2>/dev/null | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"]["windows"]))' 2>/dev/null || echo 0)
    [ "$n" = "1" ] && break
    waymux idle kde --quiet-ms 200 --timeout-ms 700 >/dev/null 2>&1 || true
  done
  waymux idle kde --quiet-ms 9000 --timeout-ms 13000 >/dev/null 2>&1 || true
  waymux --json record start kde --codec h264-vulkan --mode whole-desktop --min-fps 30 >"$OUT/kde-rec.json" 2>&1
  KRECPATH=$(jget '["data"]["path"]' <"$OUT/kde-rec.json" 2>/dev/null || echo "")
  # A real wall-clock window: `idle` returns instantly on an already-quiet
  # desktop, so use timeout to guarantee the recorder runs for ~6 s.
  timeout 6 tail -f /dev/null >/dev/null 2>&1
  waymux --json record stop kde >/dev/null 2>&1
  waymux idle kde --quiet-ms 1500 --timeout-ms 2500 >/dev/null 2>&1 || true
  if [ -f "$KRECPATH" ]; then
    KCODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$KRECPATH" 2>/dev/null)
    KFRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$KRECPATH" 2>/dev/null)
    if [ "$KCODEC" = "h264" ] && [ "${KFRAMES:-0}" -gt 10 ]; then
      ok "nested KDE whole-desktop record: $KCODEC, $KFRAMES frames (zero-copy)"
    else bad "nested KDE record: codec=$KCODEC frames=$KFRAMES (expected h264 >10)"; fi
  else bad "nested KDE record file missing: $KRECPATH"; fi
  pkill -9 glmark2-wayland kwin_wayland plasmashell 2>/dev/null || true
  waymux --json rm kde >/dev/null 2>&1
else
  echo "SKIP: nested KDE record (need kwin_wayland + dbus-run-session + glmark2-wayland)"
fi

# --- teardown -----------------------------------------------------------
waymux --json rm e2e >"$OUT/rm.json" 2>&1
[ "$(jget '["ok"]' <"$OUT/rm.json" 2>/dev/null)" = "True" ] && ok "rm --json ok" || bad "rm: $(cat "$OUT/rm.json")"

# --- Quickstart: CLI auto-spawns the daemon ------------------------------
# Self-contained: a FRESH XDG_RUNTIME_DIR with NO daemon pre-started. Running
# a local verb must transparently auto-spawn `waymuxd` (resolved via WAYMUXD_BIN
# / PATH) and succeed. This is the "one-binary onboarding" path. We then assert
# the daemon was left running (it outlives the CLI by design) and tear it down.
echo "=== quickstart: auto-spawn ==="
(
  QS_RUNTIME=/tmp/wmx-e2e-quickstart
  rm -rf "$QS_RUNTIME"; mkdir -p "$QS_RUNTIME"; chmod 700 "$QS_RUNTIME"
  QS_SOCK="$QS_RUNTIME/waymux.sock"
  # Point the resolver at the built daemon explicitly (env override wins),
  # and keep target/debug on PATH so a bare `waymuxd` would also resolve.
  export XDG_RUNTIME_DIR="$QS_RUNTIME"
  export WAYMUXD_BIN="$PWD/target/debug/waymuxd"
  unset WAYMUX_NO_AUTOSPAWN

  # Precondition: no socket, no daemon.
  [ ! -S "$QS_SOCK" ] || { echo "QS_PRE_SOCKET_EXISTS"; }

  # First local verb with no daemon: must auto-spawn and succeed.
  waymux --json ls >"$OUT/qs-ls.json" 2>"$OUT/qs-ls.err"
  if [ "$(jget '["ok"]' <"$OUT/qs-ls.json" 2>/dev/null || echo no)" = "True" ]; then
    echo "QS_LS_OK"
  else
    echo "QS_LS_FAIL: $(cat "$OUT/qs-ls.json" "$OUT/qs-ls.err" 2>/dev/null)"
  fi
  # The auto-spawned daemon should now own a live socket (it outlives the CLI).
  [ -S "$QS_SOCK" ] && echo "QS_SOCKET_LIVE" || echo "QS_SOCKET_MISSING"

  # A second verb must reuse the same daemon (socket present → no re-spawn).
  waymux --json new qs --size 320x240 >"$OUT/qs-new.json" 2>&1
  [ "$(jget '["ok"]' <"$OUT/qs-new.json" 2>/dev/null || echo no)" = "True" ] \
    && echo "QS_NEW_OK" || echo "QS_NEW_FAIL: $(cat "$OUT/qs-new.json")"

  # Tear down ONLY the daemon we auto-spawned (no shutdown verb today → signal
  # it). Match precisely on the per-test XDG_RUNTIME_DIR in the process environ
  # so we never touch the main harness daemon or any unrelated waymuxd.
  for p in $(pgrep -x waymuxd 2>/dev/null); do
    tr '\0' ' ' </proc/"$p"/environ 2>/dev/null | grep -q "XDG_RUNTIME_DIR=$QS_RUNTIME" && kill "$p" 2>/dev/null
  done
  rm -rf "$QS_RUNTIME"
) >"$OUT/qs.out" 2>&1
cat "$OUT/qs.out" | sed 's/^/  /'
grep -q "QS_LS_OK" "$OUT/qs.out"    && ok "quickstart: auto-spawn made 'waymux ls' work with no pre-started daemon" || bad "quickstart auto-spawn ls: $(cat "$OUT/qs.out")"
grep -q "QS_SOCKET_LIVE" "$OUT/qs.out" && ok "quickstart: auto-spawned daemon left a live socket (outlives the CLI)" || bad "quickstart socket: $(cat "$OUT/qs.out")"
grep -q "QS_NEW_OK" "$OUT/qs.out"   && ok "quickstart: second verb reused the auto-spawned daemon" || bad "quickstart reuse: $(cat "$OUT/qs.out")"

# --- Quickstart: serve resolves + would-exec waymuxd ---------------------
# `waymux serve` execs waymuxd. We can't let it replace the test shell, so we
# verify the resolution + error path: with WAYMUXD_BIN pointing at a missing
# file, `serve` must fail fast with a build-it hint (not hang, not exec junk).
WAYMUXD_BIN=/nonexistent/waymuxd waymux serve >"$OUT/serve-missing.out" 2>&1
if grep -q "waymuxd binary not found" "$OUT/serve-missing.out"; then
  ok "quickstart: serve reports a build-it hint when waymuxd is missing"
else
  bad "serve missing-binary hint: $(cat "$OUT/serve-missing.out")"
fi

echo
echo "===== e2e: $PASS passed, $FAIL failed ====="
[ "$FAIL" -eq 0 ]
