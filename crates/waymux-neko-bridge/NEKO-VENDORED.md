# Neko vendoring

- **Source:** https://github.com/m1k1o/neko
- **Tag:** `v3.1.0`
- **Commit:** `a597d44680f5b7746123df59ec543256af21af75`
- **Vendored:** 2026-05-14
- **License:** Apache-2.0 (see `LICENSE.neko`)

## What we kept

| Vendored path                       | Upstream path                                        | Notes                                      |
|-------------------------------------|------------------------------------------------------|--------------------------------------------|
| `internal/webrtc/payload/`          | `server/internal/webrtc/payload/`                    | Pure leaf package, no neko-internal deps   |
| `internal/webrtc/pionlog/`          | `server/internal/webrtc/pionlog/`                    | Pion → zerolog adapter, no neko deps       |
| `LICENSE.neko`                      | `LICENSE`                                             | Apache-2.0 notice for the vendored copy    |

## What we dropped (Task 11 scope) — and why

| Upstream path                                  | Why dropped                                                                                                       |
|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `server/internal/webrtc/manager.go` (599 LOC)  | Heavy graph deps on `internal/config`, `pkg/types`, `pkg/types/event`, `pkg/utils`. Task 12 lifts the H.264 branch. |
| `server/internal/webrtc/peer.go` (543 LOC)     | Same — depends on neko `session` + `config`. Task 12 will inline the slim peer.                                    |
| `server/internal/webrtc/handler.go`            | Wires capture/desktop modules that don't exist in our world.                                                       |
| `server/internal/webrtc/legacyhandler.go`      | Pre-v3 fallback, we only target the modern client.                                                                 |
| `server/internal/webrtc/track.go`              | Multi-codec + multi-stream router; we have one h264 source.                                                        |
| `server/internal/webrtc/metrics.go`            | Prometheus surface; outside V1 scope.                                                                              |
| `server/internal/webrtc/cursor/`               | X11 cursor capture; we render cursor on the host compositor.                                                       |
| `server/internal/http/router.go`               | Top-level neko router pulls in auth + REST API + member system.                                                    |
| `server/internal/api/websocket/*`              | Multi-user session signaling, file chooser dialog, chat, watch-party. Task 12 lifts only the SDP/ICE exchange.    |
| `server/internal/session/`                     | Multi-user session manager — single-viewer in waymux V1.                                                           |
| `server/internal/member/`, `member/object/`    | Account/auth model.                                                                                                |
| `server/internal/capture/`                     | X11/GStreamer screen capture — replaced by our pre-encoded NALU stream over Unix socket.                          |
| `server/internal/desktop/`                     | X11 input forwarder — replaced by our `InjectOp` translator.                                                       |
| `server/internal/plugins/{filetransfer,chat}/` | Out of scope.                                                                                                      |
| `client/`                                      | Vue SPA, ~10 MB of node_modules to build. Placeholder HTML in `internal/web/static/index.html` ships today; full client vendoring is Task 12 follow-up. |
| `apps/`, `runtime/`, `webpage/`, Docker        | Distribution & demo apps, not the bridge.                                                                          |

## Plan vs reality divergences (Task 11)

1. **`client/dist/*` does not exist in upstream.** Neko ships a Vue source tree under `client/` that must be built. The plan flagged this as a possibility. We took option (b): ship a minimal placeholder `index.html` that opens a WebSocket and shows a `<video>` element. Full vendoring of the built client is deferred to Task 12.
2. **`//go:embed all:../web/static` does not compile.** `embed` paths cannot traverse `..`. Resolved by putting the embed directive inside `internal/web/embed.go` (which lives next to `static/`) and exposing it via `web.StaticFS()`.
3. **Neko's `internal/webrtc/{manager,peer,handler,...}.go` cannot be vendored stand-alone.** Each pulls in `m1k1o/neko/server/internal/config`, `pkg/types`, `pkg/types/event`, `pkg/utils`, `internal/session`, etc. Trying to fix the import chain pulls in roughly half of neko. For Task 11 we kept only the two genuinely-leaf packages (`payload`, `pionlog`). Task 12 will lift the slim parts inline as needed for the H.264 → Pion path.
4. **WebSocket signaling not wired.** `/ws` currently returns `501 Not Implemented`. The placeholder client logs the close event, which is fine for the static-serve smoke. Task 12 implements signaling.

## Re-vendoring procedure

```bash
cd /tmp && rm -rf neko-upstream
git clone --branch <new-tag> --depth 1 https://github.com/m1k1o/neko neko-upstream

# Re-copy the leaf packages
cp /tmp/neko-upstream/server/internal/webrtc/payload/*.go \
   crates/waymux-neko-bridge/internal/webrtc/payload/
cp /tmp/neko-upstream/server/internal/webrtc/pionlog/*.go \
   crates/waymux-neko-bridge/internal/webrtc/pionlog/
cp /tmp/neko-upstream/LICENSE crates/waymux-neko-bridge/LICENSE.neko

# Update this file's commit / date / tag
( cd /tmp/neko-upstream && git rev-parse HEAD )

# Re-verify
( cd crates/waymux-neko-bridge && go mod tidy && go build ./... && go test ./... )
```

## Outstanding for Task 12

- Vendor or build neko's Vue client into `internal/web/static/`
- Lift H.264 send path from `internal/webrtc/manager.go` + `peer.go`
- Lift SDP/ICE exchange shape from `internal/api/websocket/handler.go`
- Replace the `/ws` 501 placeholder with the real signaling handler
- Wire `naluPumpLoop` → Pion `track.WriteSample`
