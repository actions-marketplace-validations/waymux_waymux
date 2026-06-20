#!/usr/bin/env bash
# Generic waymux launcher: spin up a session, open the attach viewer,
# run a command against the inner Wayland display, clean up on exit.
#
# Usage:
#   scripts/waymux-launch.sh [--size WxH] [--name NAME] -- <command> [args...]
#
# Example:
#   scripts/waymux-launch.sh --size 1024x768 --name term -- foot
#
# Requires the daemon to already be running on $XDG_RUNTIME_DIR/waymux.sock
# (start it in another terminal: cargo run --release -p waymux-daemon).

set -u

SIZE="1280x800"
NAME="scratch"

while [ $# -gt 0 ]; do
    case "$1" in
        --size)
            SIZE="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            echo "usage: $0 [--size WxH] [--name NAME] -- <command> [args...]" >&2
            exit 2
            ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "error: no command given (pass it after '--')" >&2
    echo "usage: $0 [--size WxH] [--name NAME] -- <command> [args...]" >&2
    exit 2
fi

# Resolve the waymux binaries. We prefer release over debug, matching the
# manual test scripts. Falling back to $PATH lets this work after install.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$REPO_ROOT/target/release/waymux" ]; then
    BIN="$REPO_ROOT/target/release"
elif [ -x "$REPO_ROOT/target/debug/waymux" ]; then
    BIN="$REPO_ROOT/target/debug"
else
    BIN=""
fi
WAYMUX="${BIN:+$BIN/}waymux"
WAYMUX_ATTACH="${BIN:+$BIN/}waymux-attach"

export WAYMUX_SOCKET="${WAYMUX_SOCKET:-$XDG_RUNTIME_DIR/waymux.sock}"

if [ ! -S "$WAYMUX_SOCKET" ]; then
    echo "error: daemon socket $WAYMUX_SOCKET not found" >&2
    echo "       start the daemon: cargo run --release -p waymux-daemon" >&2
    exit 3
fi

# Track what we created so the trap only tears down our own resources.
ATTACH_PID=""
CREATED_SESSION=""

cleanup() {
    set +e
    if [ -n "$ATTACH_PID" ]; then
        kill "$ATTACH_PID" 2>/dev/null
        wait "$ATTACH_PID" 2>/dev/null
    fi
    if [ -n "$CREATED_SESSION" ]; then
        "$WAYMUX" rm "$CREATED_SESSION" >/dev/null 2>&1
    fi
}
trap cleanup EXIT INT TERM

# Drop any prior session of the same name so --name is idempotent.
"$WAYMUX" rm "$NAME" >/dev/null 2>&1

if ! "$WAYMUX" new "$NAME" --size "$SIZE" >/dev/null; then
    echo "error: could not create session '$NAME'" >&2
    exit 4
fi
CREATED_SESSION="$NAME"

ATTACH=$("$WAYMUX" attach "$NAME")
if [ -z "$ATTACH" ]; then
    echo "error: could not open attach socket for '$NAME'" >&2
    exit 5
fi

"$WAYMUX_ATTACH" "$ATTACH" &
ATTACH_PID=$!

# Give the attach viewer a beat to map its window before the guest app
# starts committing frames. Without this, the first few frames land
# before the viewer is configured and are silently dropped.
sleep 0.3

# Run the command against the inner display. `env -u` strips any host
# WAYLAND_DISPLAY so the child can't accidentally route to the host
# compositor.
WAYLAND_DISPLAY="$XDG_RUNTIME_DIR/waymux/$NAME/wayland.sock" \
    env -u DISPLAY "$@"
status=$?

exit "$status"
