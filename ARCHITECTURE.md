# waymux Architecture

waymux is a local-first headless Wayland session manager. A per-user daemon (`waymuxd`) supervises any number of isolated Wayland sessions, each backed by its own in-process compositor that renders to a virtual output. A single `waymux` CLI drives the daemon over a msgpack-RPC Unix socket to create sessions, spawn applications into them, inject input, capture screenshots, record video, and serve a low-latency WebRTC viewer to a browser. Sessions can also be attached to an outer compositor (for example niri) so the virtual output appears as a normal window on the host. The same control surface is reachable remotely over HTTPS for the hosted SaaS, but the canonical, fully open surface is local.

## Component diagram

```
                         +-------------------------------+
                         |          waymux CLI           |
                         |  (23 subcommands, clap)       |
                         +---------------+---------------+
                                         |
              Transport trait: LocalTransport | RemoteTransport
                                         |
         msgpack-RPC over Unix socket    |   HTTPS + Bearer (hosted)
         $XDG_RUNTIME_DIR/waymux.sock    |
                                         v
  +----------------------------------------------------------------------+
  |                          waymuxd (per-user daemon)                    |
  |                                                                      |
  |   Server (accept loop, SO_PEERCRED uid gate, Hello negotiation,      |
  |           reader/writer tasks, dispatch router, error mapping)       |
  |                              |                                       |
  |                              v                                       |
  |   Registry (core engine: session map, lifecycle, broadcast,         |
  |             supervisors, log history, session_control RPC)          |
  |                              |                                       |
  |   SessionBackend trait  --> LocalBackend (subprocess)               |
  |                                                                      |
  |   cgroup / tmpfs quota (best-effort)   usage_events (feature-gated)  |
  +--------------------------------+-------------------------------------+
                                   | spawn + per-session control socket
                                   v
  +----------------------------------------------------------------------+
  |             waymux-session (one process per session)                 |
  |                                                                      |
  |   Compositor thread          Control socket (tokio, msgpack-RPC)     |
  |   - inner Wayland server      - Info/ListWindows/Resize              |
  |   - xdg-shell, wl-shm,        - Inject{Key,Pointer,Touch,Batch}      |
  |     dmabuf, layer-shell       - Screenshot/ScreenshotDesktop         |
  |   - virtual output            - Record{Start,Stop} Viewer{Start..}   |
  |   - SurfaceData / windows     Events socket --> daemon broadcast     |
  |        |            |              |                                 |
  |        v            v              v                                 |
  |   Attach server   Recording    Viewer (neko-bridge child, Go)        |
  |   (waymux_attach  encoders:     - encoder thread -> Annex-B NALUs    |
  |    _v1, fd-pass)  ffv1/nvenc/    - Unix socket -> Pion WebRTC        |
  |        |          vaapi/vulkan)  - WS signaling + data channel       |
  |        v                              |                              |
  |   Outer compositor (niri)        Browser viewer (video + input)      |
  +----------------------------------------------------------------------+
```

## Control plane

The control plane is the daemon plus its wire protocol.

- **Transport.** The daemon binds a per-user `UnixListener` (default `$XDG_RUNTIME_DIR/waymux.sock`, chmod 0600). The CLI connects through a `Transport` trait with two implementations: `LocalTransport` (Unix socket, msgpack-RPC) and `RemoteTransport` (HTTPS with a Bearer token read from `$XDG_CONFIG_HOME/waymux/credentials.toml`). 8 of the 23 CLI verbs are transport-routable over the remote HTTPS transport (`login` also targets the remote but is handled separately); the rest are local-only.
- **Framing and negotiation.** Every frame is a 4-byte big-endian length prefix followed by a msgpack payload, capped at a 20 MiB `MAX_FRAME_SIZE`. The first request on a connection must be `Hello`; the daemon accepts any client protocol from 1 through the current version (4) and replies with its version and capabilities (`subscribe`, `spawn`). A non-Hello first request, protocol 0, or a version newer than the daemon's returns `E_PROTO_VERSION`.
- **Registry.** The Registry is the engine. It holds a `HashMap<String, SessionEntry>` of session metadata, supervisor kill channels, child-PID tracking, rolling per-session log history (1024 lines), and a `broadcast` channel that fans events to all subscribers. Its public methods (`create`, `destroy`, `spawn_child`, `session_control`, `list_windows`, `resize`, `screenshot`, `inject_*`, `record_*`, `viewer_*`, `tag_window`, `wait_for_idle`, `attach`, `detach`, `shutdown_all`) are protocol-agnostic.
- **Backend.** A `SessionBackend` async trait (`create`/`destroy`) abstracts the session-lifecycle path. `LocalBackend` is a thin wrapper over the Registry (subprocess sessions) and is the only implementation; the trait is the seam a future non-local lifecycle target would plug into.
- **Dispatch.** The Server's `dispatch()` is a match over `RequestMethod` that translates each wire request into a Registry call and maps typed engine errors (`CreateError`, `DestroyError`, `SpawnError`, `SessionControlError`, etc.) into stable `ErrorCode` values (`E_NOT_FOUND`, `E_ALREADY_EXISTS`, `E_NOT_IMPLEMENTED`, `E_BACKPRESSURE`, `E_INTERNAL`, and others).

## Data plane

Each session is a separate `waymux-session` process that is a full headless Wayland compositor for one virtual output.

- **Inner compositor.** Advertises xdg-shell, wl-shm, layer-shell, `zwp_linux_dmabuf_v1` (GPU buffer import with modifier negotiation), viewporter, pointer/keyboard/touch, data-device (clipboard), presentation-time, pointer-constraints and relative-pointer (pointer lock), keyboard-shortcuts-inhibit, and KDE-specific protocols. It is observer-only: it tracks surfaces, subsurface trees, toplevels, and damage timestamps without rendering. Composition happens lazily at capture time via a recursive subsurface tree walk that blits into a single ARGB8888 buffer.
- **Capture and screenshots.** Screenshot RPCs run on the control thread, look up the surface by window id, composite, and encode PNG with the `image` crate. The protocol prefers a fd-passed `Dmabuf` format with a `Png` SHM fallback. A buffer-hold ref-count mechanism keeps GPU buffers pinned while a capture or encode is in flight.
- **Recording.** Four backends are available: FFV1/MKV lossless (CPU readback), H.264 via NVENC (subprocess), in-process VAAPI H.264, and in-process Vulkan H.264/HEVC (zero-copy, fastest). A `LatestTaskSlot` lets newer frames evict older ones so a slow encoder never back-pressures the compositor. Dual recording (primary plus `--secondary-codec`) writes two output paths from the same frame.
- **WebRTC viewer.** On `ViewerStart` the session spawns a Go `waymux-neko-bridge` child and an encoder thread that produces Annex-B NALUs tuned for low latency (baseline profile, no B-frames, periodic IDR at 60 fps; Vulkan emits every-frame IDR). NALUs cross a private per-session Unix socket using a typed 5-byte-header protocol (NALU, cursor image, cursor position, force-keyframe, inject-op, set-bitrate, shutdown). The bridge wraps frames into WebRTC with Pion (ICE, DTLS, RTP), signals over WebSocket, and exposes a data channel. Browser input arrives as JSON, is translated to waymux `InjectOp`, and is written back over the socket into the session control loop. Multi-viewer is last-wins: only the primary viewer's input and GCC bandwidth estimate drive the shared encoder; other viewers receive video fan-out only.
- **Attach.** A second Wayland server advertises `waymux_attach_v1`. An attach client passes the outer compositor's display fd via SCM_RIGHTS; the session creates a proxy `wl_surface`/`xdg_toplevel` on the outer compositor and ferries the inner focused window's frames into an outer SHM buffer on each commit. The ferry path is validated for same-format ARGB8888 at 1:1 size and falls back to a placeholder otherwise.

## Process and socket model

- **Same-uid gating.** Both the daemon accept loop and the per-session control socket check `SO_PEERCRED` and reject any connection whose uid differs from the owner. This is the primary local trust boundary.
- **Sockets per session.** Each session is spawned with a set of Unix sockets: the inner Wayland display, a control socket (daemon-to-session RPC), an events socket (session-to-daemon push), an attach socket, and a one-shot ready socket for the startup handshake. The daemon holds a persistent control-socket connection per session and reconnects on error.
- **Lifecycle.** `create` spawns the session subprocess, waits for the ready handshake (5s timeout), sets up best-effort cgroup and tmpfs quota handles, starts a `session_supervisor` task that owns the `Child`, drains stdout/stderr into the log ring, and emits `SessionCreated`. `destroy` removes the session, SIGTERMs tracked child PIDs, signals the supervisor, lazy-unmounts the tmpfs, and cleans up the cgroup. The supervisor also handles natural exit (removing the session and emitting `SessionDestroyed`). `spawn_child` validates that argv[0] is absolute, clears the environment and re-adds only safe variables, optionally applies an fd-limit rlimit, joins the cgroup, and tracks the PID for crash detection.
- **Resource capping.** `SessionCgroup` (cgroup v2) and `SessionTmpfs` are best-effort: if `CAP_SYS_ADMIN` is absent or a write fails, the daemon logs a warning and the session runs uncapped rather than failing.

## Crate layout

- **waymux-cli.** The `waymux` binary. A clap `Cmd` enum of 23 subcommands dispatched through `run_with_transport()` (8 transport-routable verbs) and `run_local_only()` (the rest). Holds the `Transport` trait and its local/remote implementations and the credentials loader.
- **waymux-protocol.** The wire contract: `RequestMethod` (26 variants), `Response`, `EventBody` (10 variants), `SessionCtlMethod` (the simpler daemon-to-session control protocol), supporting enums (`ErrorCode`, `RecordingCodec`, `CaptureMode`, `ScreenshotFormat`, `KeyState`, `TouchPhase`), and `encode_frame`/`decode_frame`. Serialization is rmp-serde with named fields and extensive `#[serde(default)]` for forward/backward compatibility. An extensive round-trip test suite pins the wire contract.
- **waymux-daemon.** `waymuxd`. Contains the Registry engine, the `SessionBackend` trait with `LocalBackend`, the Server (accept loop, per-connection handler, dispatch router, event forwarder, error mapping), and the supporting cgroup/quota/usage-events modules and `main` bootstrap.
- **waymux-session.** The per-session compositor. Subsystems: compositor (Wayland protocol dispatch, surface and window tracking), state (the thread-safe `Arc<State>` shared between compositor and control), control (the session RPC server), recording and its encoder backends, the attach server and outer-view bridge, and the viewer (encoder thread plus bridge supervision).
- **waymux-neko-bridge.** A slim-vendored Go WebRTC bridge (derived from neko, Apache-2.0) spawned as a child of a session. Handles Pion WebRTC, WebSocket signaling, Ed25519 viewer-token validation, multi-viewer fan-out, GCC bandwidth feedback, and input translation.

## Key data flows

**Create a session**
1. CLI `new` sends `Hello` then `CreateSession` over the local socket.
2. Server gates on Hello, then `dispatch()` calls `registry.create()`.
3. Registry spawns `waymux-session` with the socket set and waits for the ready handshake.
4. Registry installs cgroup/tmpfs handles and the supervisor, then emits `SessionCreated`.
5. CLI prints `name (WxH)`.

**Spawn a client**
1. CLI `spawn` sends `Spawn {argv, env, compositor}`.
2. `dispatch()` calls `registry.spawn_child()`, which validates argv[0], sanitizes the environment, joins the cgroup, and starts the process inside the session's `WAYLAND_DISPLAY`.
3. The Registry records the PID, drains its logs, and (on exit) emits `ChildExited` or `SessionCrashed`.
4. CLI prints `pid N`.

**Screenshot**
1. CLI `screenshot` sends `Screenshot {window_id, format}`.
2. The daemon forwards via `session_control()` to the session control socket.
3. The session composites the subsurface tree, encodes PNG, and returns width/height plus the PNG bytes.
4. CLI writes the raw PNG to a file or stdout; metadata goes to stderr.

**Record**
1. CLI `record` sends `RecordStart {path, codec, secondary_codec, mode, min_fps}`.
2. The session selects an encoder, validates the output path (absolute, no `..`), and starts a recording thread fed by the compositor frame tap.
3. CLI prints the primary path (and the secondary path if a secondary codec was set).
4. `RecordStop` finalizes the MKV container.

**Live view**
1. CLI `viewer` sends `ViewerStart {bind, port}`.
2. The session probes a viewer codec (NVENC, then Vulkan), spawns the neko-bridge child, and starts an encoder thread.
3. The browser opens the bridge URL, completes WebSocket signaling, and receives the H.264 WebRTC stream; input flows back over the data channel as `InjectOp`.
4. CLI prints the viewer URL.

## Security model

- **Local-first.** The default and fully open path is a per-user Unix socket. There is no network listener in the local configuration.
- **Same-uid only.** Every connection (daemon socket and per-session control socket) is gated by `SO_PEERCRED`; foreign uids are rejected. The daemon socket is chmod 0600 and the credentials file is enforced at 0700 dir / 0600 file.
- **Process hardening.** `spawn_child` requires an absolute argv[0], clears the environment and re-adds only safe variables before applying user-supplied env, and can apply an fd-limit rlimit. Recording paths are validated to be absolute and free of `..`. Dmabuf imports are capped at 256 MiB.
- **Fail-closed viewer token.** The WebRTC bridge verifies viewer JWTs with an Ed25519 public key only (the control plane holds the private key, so a compromised VM can never forge a token). On a non-loopback bind, a missing public key or an invalid token rejects all viewers; loopback with no key is the developer auth-off path. Tokens require `exp`, an audience containing `viewer`, a UUID subject, and a `vm_session_id` matching the bridge.
- **Bridge DoS hardening.** The bridge caps concurrent viewers (default 8) and in-flight handshakes per source IP (default 4), returning 503 when exceeded, and uses bounded per-connection send queues with a dedicated writer goroutine so a slow viewer is reaped rather than stalling others.

Validated vs experimental: the local control plane, session lifecycle, input injection (key/pointer/touch), screenshots, recording, and the WebRTC viewer are implemented and exercised. `InjectSelector` is the one reserved protocol slot that currently returns `E_NOT_IMPLEMENTED` (resolve the target with `windows`/`wait` and inject with an explicit `window_id` instead). Coordinate scaling for non-1x outputs is not yet implemented.

## Extension points

- **SessionBackend trait.** Implement `create`/`destroy`/`info` to target a new provisioning substrate (the local subprocess backend and the remote VM backend already use this seam).
- **Recording and viewer encoders.** The codec backends are pluggable behind the recording task interface; new encoders slot in alongside ffv1/nvenc/vaapi/vulkan.
- **Protocol evolution.** New `RequestMethod`/`SessionCtlMethod` variants and struct fields are added with `#[serde(default)]` so older peers keep parsing; the version handshake accepts any client protocol from 1 through the daemon's current version.
- **Event subscribers.** Clients subscribe to topic-filtered events (`sessions`, `windows`, `damage`, `logs`, with `:name` scoping and log replay on subscribe), which is the integration point for external monitoring and automation.
- **Attach protocol.** `waymux_attach_v1` is the seam for embedding a session's output into any outer Wayland compositor via display-fd passing.
