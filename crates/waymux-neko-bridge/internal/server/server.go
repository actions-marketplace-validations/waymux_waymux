// SPDX-License-Identifier: Apache-2.0

// Package server is the slim glue that wires a Pion WebRTC peer to the
// waymux session over a typed Unix socket. The peer receives encoded
// H.264 NALUs from the session and serves them over WebRTC to one
// browser viewer. Input events from the browser are translated to
// waymux InjectOp JSON and written back over the same Unix socket.
//
// Signaling is hand-rolled (single peer, no rooms / auth / chat).
// Frames are JSON envelopes over a WebSocket at `/ws`:
//
//	← { "type":"config", "iceServers":[{urls,username,credential}] }  server → client (first, only if configured)
//	→ { "type":"offer",  "sdp":"..."  }   client → server
//	← { "type":"answer", "sdp":"..."  }   server → client
//	← { "type":"ice",    "candidate":<RTCIceCandidateInit> }
//	→ { "type":"ice",    "candidate":<RTCIceCandidateInit> }
//	→ { "type":"event",  "event":"keydown", ... }   neko-format input
//
// The `config` message is sent first when the server has any ICE
// configuration; the browser waits for it before constructing its
// RTCPeerConnection. If the server has no ICE config, the message is
// omitted entirely and the browser falls back to LAN-only behaviour
// (its own default empty config).
package server

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/pion/interceptor"
	"github.com/pion/interceptor/pkg/cc"
	"github.com/pion/interceptor/pkg/gcc"
	"github.com/pion/rtcp"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"

	inputx "github.com/waymuxhq/waymux/crates/waymux-neko-bridge/internal/input"
	sock "github.com/waymuxhq/waymux/crates/waymux-neko-bridge/internal/socket"
	"github.com/waymuxhq/waymux/crates/waymux-neko-bridge/internal/web"
)

// Config carries the runtime parameters chosen by main.go.
//
// SocketR / SocketW are usually the same *net.UnixConn but split here
// so tests can use in-memory pipes.
type Config struct {
	Bind    string
	Port    int
	Width   int
	Height  int
	Logger  *slog.Logger
	SocketR io.Reader
	SocketW io.Writer

	// ICE configuration for WebRTC NAT traversal.
	//
	// Empty values fall back to the LAN-only behaviour that's safe for
	// dev: Pion uses an empty webrtc.Configuration{} and the browser
	// receives no `config` message, so it constructs RTCPeerConnection
	// with its own defaults.
	//
	// Populate via main.go from env vars (WAYMUX_STUN_URLS,
	// WAYMUX_TURN_URL, WAYMUX_TURN_USERNAME, WAYMUX_TURN_CREDENTIAL).
	//
	// The bridge does not mint TURN REST creds from a fleet-wide shared
	// secret. The control plane mints per-session creds and delivers them
	// to the VM, so the bridge only forwards them verbatim to the browser
	// + Pion.
	STUNServers    []string
	TURNServers    []string
	TURNUsername   string // per-session, minted by the control plane
	TURNCredential string // per-session, minted by the control plane

	// Viewer-token verification.
	//
	// ViewerTokenPubKey is the RAW 32-byte ed25519 public key minted by
	// the control plane (which alone holds the private key). Delivered to
	// the VM as base64 `WAYMUX_VIEWER_TOKEN_ED25519_PK` and parsed in
	// main.go into an ed25519.PublicKey (which IS exactly those 32 bytes).
	// When non-empty, every WS upgrade MUST carry a valid `?token=<jwt>`
	// query param whose claims pass:
	//   * EdDSA signature against ViewerTokenPubKey
	//   * exp present and not in the past (WithExpirationRequired)
	//   * aud contains "viewer"
	//   * sub is a valid UUID (well-formed)
	//   * vm_session_id claim == VMSessionID
	//
	// This replaces the earlier HS256 shared-secret scheme. Because the
	// bridge holds only the PUBLIC key, a compromised VM can no longer
	// forge viewer tokens for any tenant.
	//
	// FAIL-CLOSED: when ViewerTokenPubKey is empty/nil, behaviour depends
	// on BoundNonLoopback:
	//   * bound to a non-loopback (public) address → REJECT all upgrades
	//     (a misconfigured production bridge must NOT serve unauthenticated
	//     traffic on the public internet).
	//   * bound to loopback only (dev) → skip validation entirely, so the
	//     local-bridge-binding dev path still works without the control
	//     plane.
	ViewerTokenPubKey ed25519.PublicKey
	VMSessionID       string

	// BoundNonLoopback is true when the bridge's bind address is NOT a
	// loopback address. Used purely to decide the fail-closed behaviour
	// above: an empty pubkey on a public bind is a hard reject; on a
	// loopback bind it's the dev no-auth path. Set in main.go from the
	// resolved --bind flag.
	BoundNonLoopback bool

	// DoS hardening: hard cap on concurrent viewers / PeerConnections.
	//
	// The bridge is bound 0.0.0.0 on a per-VM GPU box; each viewer
	// allocates a Pion peer + interceptor chain + ICE agent + goroutines +
	// an H.264 track that the single frame pump must WriteSample on every
	// frame. Without a cap an attacker opens many WS connections and
	// exhausts the VM's CPU/RAM/FDs, killing the legit session.
	//
	// The /ws handler enforces this AFTER auth but BEFORE NewPeerConnection
	// (rejecting the upgrade with HTTP 503). Atomic w.r.t. concurrent
	// upgrades so a burst can't all slip past. Zero / unset falls back to
	// DefaultMaxViewers. Set in main.go from WAYMUX_MAX_VIEWERS.
	MaxViewers int

	// Cap on concurrent half-open handshakes per source IP. A
	// half-open WS flood (connections that pass auth but never finish
	// signaling) can pin resources before the MaxViewers gate sees a
	// registered viewer. This bounds in-flight handshakes per remote IP.
	// Zero / unset falls back to DefaultMaxHandshakesPerIP. Set in main.go
	// from WAYMUX_MAX_HANDSHAKES_PER_IP.
	MaxHandshakesPerIP int
}

// DoS-hardening defaults. Used when the corresponding Config field
// is zero/unset.
const (
	// DefaultMaxViewers caps concurrent viewers per session. Real usage is
	// 1-2 (one human, maybe a second tab); 8 leaves slack for reconnects /
	// multi-device without letting a flood exhaust the GPU box.
	DefaultMaxViewers = 8
	// DefaultMaxHandshakesPerIP caps concurrent in-flight WS handshakes
	// from a single source IP.
	DefaultMaxHandshakesPerIP = 4
)

// wsWritePolicy selects what the per-connection writer does when its
// bounded send channel is full (head-of-line-blocking remediation).
type wsWritePolicy int

const (
	// dropOnFull silently drops the message when the send buffer is full.
	// Correct for high-rate, individually-disposable cursor frames: a slow
	// viewer just misses a cursor update, the rest keep flowing.
	dropOnFull wsWritePolicy = iota
	// closeOnFull closes (reaps) the connection when the send buffer is
	// full. Correct for control/signaling traffic (config / answer / ICE /
	// stats / ping): if a viewer can't keep up with the low-rate control
	// path it is wedged, and closing it frees its peer + goroutines and
	// promotes the next viewer.
	closeOnFull
)

// wsSendBuffer is the depth of each connection's send channel. Deep enough
// to absorb a normal burst (a flurry of ICE candidates + a config + an
// answer) without dropping, shallow enough that a stuck consumer is
// detected (and reaped / shed) quickly.
const wsSendBuffer = 64

// wsWriteDeadline bounds a single socket write so a stuck TCP/TLS peer
// errors out (and the conn is reaped) instead of hanging the writer
// goroutine forever.
const wsWriteDeadline = 5 * time.Second

// wsOutMsg is one queued write for a connection's writer goroutine. Exactly
// one of json/ping is set: ping=true sends a WebSocket ping control frame,
// otherwise the json payload is written with WriteJSON.
type wsOutMsg struct {
	json any
	ping bool
}

// wsConn wraps a *websocket.Conn with a bounded send channel and a single
// dedicated writer goroutine. All writes to the underlying socket go
// through that one goroutine, so gorilla/websocket's "one concurrent
// writer" rule is satisfied WITHOUT a process-global mutex: a slow or stuck
// consumer can only ever back up ITS OWN channel, never block another
// connection's writes, the broadcast path, the cursor pump, or the
// keepalive goroutine.
type wsConn struct {
	conn wsSocket
	send chan wsOutMsg
	// stopCh is closed by stop() to signal teardown. Producers select on
	// it so they never send on a torn-down conn; the writer goroutine
	// selects on it to exit promptly. NOTE: we never close(send) — multiple
	// producer goroutines (broadcast, ping, stats, ICE, signaling) can call
	// enqueue concurrently, and closing the channel they send on would race
	// into a "send on closed channel" panic. Signalling via stopCh + an
	// unclosed buffered send channel is the panic-free pattern.
	stopCh chan struct{}
	closed chan struct{} // closed once the writer goroutine has exited
	once   sync.Once
	logger *slog.Logger
}

// wsSocket is the minimal write surface wsConn drives. *websocket.Conn
// satisfies it in production; tests inject a fake whose WriteJSON can block
// or fail deterministically (so the stuck-consumer + write-deadline paths
// are exercised without a real flaky network peer).
type wsSocket interface {
	WriteJSON(v any) error
	WriteControl(messageType int, data []byte, deadline time.Time) error
	SetWriteDeadline(t time.Time) error
	Close() error
}

// newWSConn starts the dedicated writer goroutine for c and returns the
// wrapper. The writer drains send, sets a per-write deadline, and tears the
// connection down on the first write error or on stop().
func newWSConn(c wsSocket, logger *slog.Logger) *wsConn {
	w := &wsConn{
		conn:   c,
		send:   make(chan wsOutMsg, wsSendBuffer),
		stopCh: make(chan struct{}),
		closed: make(chan struct{}),
		logger: logger,
	}
	go w.writeLoop()
	return w
}

// writeLoop is the connection's sole writer. It exits (and force-closes the
// underlying conn) on the first write error or when stop() is called,
// signalling via closed so the read loop can unblock and reap the viewer.
//
// On stopCh close it returns IMMEDIATELY and intentionally does NOT drain any
// remaining buffered messages in `send`: stopCh is closed precisely for a
// wedged/dead/reaped connection, so flushing queued frames would only block
// the writer on a socket that is going away. Undelivered messages are dropped
// — correct for this teardown path (disposable cursor/stats traffic; the conn
// is being torn down anyway).
func (w *wsConn) writeLoop() {
	defer close(w.closed)
	defer func() { _ = w.conn.Close() }()
	for {
		select {
		case <-w.stopCh:
			return
		case msg := <-w.send:
			_ = w.conn.SetWriteDeadline(time.Now().Add(wsWriteDeadline))
			var err error
			if msg.ping {
				err = w.conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(wsWriteDeadline))
			} else {
				err = w.conn.WriteJSON(msg.json)
			}
			if err != nil {
				// A stuck/dead socket: tear the conn down so the read loop
				// errors out and the viewer is reaped. Drain nothing further.
				if w.logger != nil {
					w.logger.Debug("ws write failed; closing conn", "err", err)
				}
				return
			}
		}
	}
}

// enqueue hands msg to the writer goroutine without ever blocking the
// caller. On a full buffer it applies policy: drop the message, or close
// the connection (so a wedged control consumer is reaped). Returns false if
// the message was not delivered (dropped or conn already stopping).
func (w *wsConn) enqueue(msg wsOutMsg, policy wsWritePolicy) bool {
	select {
	case <-w.stopCh:
		return false
	default:
	}
	select {
	case w.send <- msg:
		return true
	case <-w.stopCh:
		return false
	default:
		// Buffer full: the consumer is slow/stuck.
		if policy == closeOnFull {
			if w.logger != nil {
				w.logger.Warn("ws send buffer full on control path; reaping slow conn")
			}
			w.stop()
		}
		return false
	}
}

// writeJSON queues a JSON message for the writer goroutine. Non-blocking.
func (w *wsConn) writeJSON(v any, policy wsWritePolicy) bool {
	return w.enqueue(wsOutMsg{json: v}, policy)
}

// writePing queues a ping control frame. Non-blocking; closeOnFull so a
// stuck conn that can't even absorb a ping is reaped (which is the whole
// point of the keepalive).
func (w *wsConn) writePing() bool {
	return w.enqueue(wsOutMsg{ping: true}, closeOnFull)
}

// stop signals the writer goroutine to exit (which force-closes the
// underlying conn). Idempotent and safe to call from multiple goroutines.
// It does NOT close the send channel — concurrent producers must never race
// a close — so teardown is driven entirely by stopCh.
func (w *wsConn) stop() {
	w.once.Do(func() {
		close(w.stopCh)
	})
}

// Server is the bridge process's top-level glue.
// viewer is one connected WebRTC client. The bridge supports
// N concurrent viewers per session; exactly one holds the input "primary"
// slot (its input messages flow to the daemon). Observers receive the same
// video fan-out but their input is silently dropped.
//
// Policy: LAST-WINS. The newest viewer takes the primary slot (demoting any
// prior primary); on primary disconnect, control falls to the newest
// remaining viewer. This means a reload or a new tab always has control —
// avoiding the stale-observer trap where an older or silently-dead connection
// keeps input and every reconnect is a no-input observer. (Multi-viewer
// collaboration with explicit "request control" is future work.)
//
// `primary` is duplicated from the Server.viewers slice position for
// O(1) checks in the input-forward hot path without taking Server.mu.
type viewer struct {
	track     *webrtc.TrackLocalStaticSample
	primary   atomic.Bool
}

type Server struct {
	cfg      Config
	listener net.Listener // captured from net.Listen so the actual port is recoverable

	// Active WebRTC viewers. Multiple concurrent connections are
	// allowed (no single-viewer guard). The slice is ordered by
	// connect time so index 0 is always the current primary.
	// Mutated under mu in addViewer/removeViewer.
	mu      sync.Mutex
	viewers []*viewer

	// reservedViewers counts viewer slots that have been atomically
	// reserved at the pre-upgrade gate (tryReserveViewerSlot) but not yet
	// committed into the viewers slice (addViewer). Guarded by the SAME
	// mutex (mu) as viewers, so the cap is authoritative at reserve time:
	// `len(viewers) + reservedViewers` is the true admitted count and never
	// exceeds maxViewers. A reservation is either committed (converted into a
	// viewer by addViewer) or released on any error/early-return path, so the
	// sum is conserved and NewPeerConnection only ever runs for a connection
	// that already holds a slot.
	reservedViewers int

	// Source-side frame counter. Incremented atomically by
	// naluPumpLoop on every successful WriteSample to the WebRTC
	// track. Sampled once per second by the per-connection stats
	// goroutine in handleWS and pushed to the browser as
	// {"type":"stats","srcFps":N} for the on-screen FPS meter. Not
	// guarded by `mu` — atomic is fine and avoids serialising the
	// hot frame-pump path on the mutex.
	frameCount atomic.Uint64

	// Socket writes (input events + ForceKeyframe) are shared between
	// the WS handler goroutines. The Unix socket is full-duplex and
	// Go net.UnixConn is safe for concurrent Write; the mutex here
	// guards the framing protocol's "write this whole frame in one
	// shot" requirement.
	sockMu sync.Mutex

	// Registry of live viewer WebSocket connections for cursor
	// broadcast. Guarded by a dedicated mutex (wsConnsMu) so the hot
	// naluPumpLoop cursor broadcast path doesn't contend with mu, which
	// is already taken by addViewer/removeViewer/snapshotTracks.
	//
	// Each entry is a *wsConn (bounded send channel + dedicated writer
	// goroutine) so broadcast/cursor/ping/stats/ICE writes hand off
	// non-blocking and a single slow consumer can never stall the others.
	wsConnsMu sync.Mutex
	wsConns   map[*wsConn]struct{}

	// Per-source-IP in-flight handshake counter. Incremented at /ws
	// entry, decremented on handler exit, so a half-open WS flood from one
	// IP is bounded before it can allocate PeerConnections. Guarded by its
	// own mutex (off the hot mu / wsConnsMu paths).
	handshakeMu     sync.Mutex
	handshakesPerIP map[string]int
}

// maxViewers returns the configured cap, or the default when unset.
func (s *Server) maxViewers() int {
	if s.cfg.MaxViewers > 0 {
		return s.cfg.MaxViewers
	}
	return DefaultMaxViewers
}

// maxHandshakesPerIP returns the configured per-IP handshake cap, or the
// default when unset.
func (s *Server) maxHandshakesPerIP() int {
	if s.cfg.MaxHandshakesPerIP > 0 {
		return s.cfg.MaxHandshakesPerIP
	}
	return DefaultMaxHandshakesPerIP
}

// New constructs a Server but does not start any goroutines or open ports.
func New(cfg Config) (*Server, error) {
	if cfg.Logger == nil {
		return nil, errors.New("server.New: Logger is required")
	}
	if cfg.SocketR == nil || cfg.SocketW == nil {
		return nil, errors.New("server.New: SocketR/SocketW are required")
	}
	return &Server{
		cfg:             cfg,
		wsConns:         make(map[*wsConn]struct{}),
		handshakesPerIP: make(map[string]int),
	}, nil
}

// Listen opens the TCP socket but does not start serving.
func (s *Server) Listen() error {
	addr := fmt.Sprintf("%s:%d", s.cfg.Bind, s.cfg.Port)
	l, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", addr, err)
	}
	s.listener = l
	return nil
}

// Addr returns the actual listen address (after Listen() resolves port 0).
func (s *Server) Addr() string {
	if s.listener == nil {
		return ""
	}
	return s.listener.Addr().String()
}

// Run blocks until ctx is cancelled or a fatal error occurs.
func (s *Server) Run(ctx context.Context) error {
	if s.listener == nil {
		if err := s.Listen(); err != nil {
			return err
		}
	}

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(web.StaticFS())))
	mux.HandleFunc("/ws", s.handleWS)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	// Occupancy + viewer count probe. The bridge supports N concurrent
	// viewers, so `occupied` is no longer a hard gate on the portal
	// Connect button: it's purely informational. `viewers` reports the
	// current peer count; `primary` reports whether someone holds the
	// input slot. Backwards-compat: kept `occupied: bool` (= viewers>0)
	// for any existing readers, plus added the richer fields.
	//
	// Require the viewer token, same as /ws. There is no legitimate
	// caller of this endpoint: the control plane probes /healthz (not
	// /status) and hardcodes occupied=false; the viewer page never
	// fetches it. Leaving it open would let anyone who could reach the
	// port enumerate per-session occupancy. validateViewerToken is
	// fail-closed: loopback dev with no key still passes; a non-loopback
	// bind without a key, or a missing/invalid token, is rejected 401.
	// This is defense-in-depth on top of CIDR-scoped viewer ingress.
	mux.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		if err := s.validateViewerToken(r); err != nil {
			s.cfg.Logger.Warn("status probe rejected", "err", err, "remote", r.RemoteAddr)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		s.mu.Lock()
		n := len(s.viewers)
		hasPrimary := n > 0 && s.viewers[0].primary.Load()
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w,
			`{"occupied":%t,"viewers":%d,"primary":%t}`,
			n > 0, n, hasPrimary)
	})

	httpSrv := &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	httpErrCh := make(chan error, 1)
	go func() {
		s.cfg.Logger.Info("http serving", "addr", s.listener.Addr().String())
		err := httpSrv.Serve(s.listener)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			httpErrCh <- err
			return
		}
		httpErrCh <- nil
	}()

	pumpErrCh := make(chan error, 1)
	go func() {
		pumpErrCh <- s.naluPumpLoop(ctx)
	}()

	select {
	case <-ctx.Done():
		shutCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutCtx)
		<-pumpErrCh
		<-httpErrCh
		return nil
	case err := <-httpErrCh:
		return fmt.Errorf("http server: %w", err)
	case err := <-pumpErrCh:
		shutCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutCtx)
		<-httpErrCh
		if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, sock.ErrShutdown) {
			return fmt.Errorf("nalu pump: %w", err)
		}
		return nil
	}
}

// wsUpgrader rejects no origins because the bridge listens only on the
// configured bind address (default 127.0.0.1). For a hosted SaaS we'd
// pin the origin check; loopback dev doesn't need it.
var wsUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// signalingMsg is the JSON envelope used over the WS.
//
// `type` is one of {config, offer, answer, ice, event}. The relevant
// payload field is filled per-type; unused fields stay zero/omit.
type signalingMsg struct {
	Type       string          `json:"type"`
	SDP        string          `json:"sdp,omitempty"`
	Candidate  json.RawMessage `json:"candidate,omitempty"`
	ICEServers []iceServerJSON `json:"iceServers,omitempty"` // only set on "config" messages
	// "event"-typed messages carry the neko-formatted input event in
	// the same envelope. The translator unmarshals from these fields
	// directly via the inputx.Translate(raw) entrypoint.
	Event   string  `json:"event,omitempty"`
	KeyCode int     `json:"keyCode,omitempty"`
	X       float64 `json:"x,omitempty"`
	Y       float64 `json:"y,omitempty"`
	Button  int     `json:"button,omitempty"`
	DeltaX  float64 `json:"deltaX,omitempty"`
	DeltaY  float64 `json:"deltaY,omitempty"`
}

// validateViewerToken decodes and verifies a `?token=<jwt>` query
// parameter against the bridge's configured ViewerTokenPubKey +
// VMSessionID. Returns nil on success; a non-nil error describes the
// rejection reason for logging (token never echoed; only field-shape
// info goes to the client via the HTTP status).
//
// EdDSA verification with a PUBLIC key only, requiring `exp` and
// `aud == "viewer"`. FAIL-CLOSED: see Config.ViewerTokenPubKey /
// Config.BoundNonLoopback docs: an empty pubkey on a non-loopback bind
// is a hard reject; only a loopback bind keeps the dev no-auth path.
func (s *Server) validateViewerToken(r *http.Request) error {
	if len(s.cfg.ViewerTokenPubKey) == 0 {
		// FAIL-CLOSED: no key configured. Refuse on a public bind;
		// allow only the loopback dev path.
		if s.cfg.BoundNonLoopback {
			return errors.New("no viewer-token public key configured and bridge is bound to a non-loopback address (fail-closed)")
		}
		return nil // auth-off mode (dev / loopback only)
	}
	tokenStr := r.URL.Query().Get("token")
	if tokenStr == "" {
		return errors.New("missing ?token= query param")
	}
	tok, err := jwt.Parse(tokenStr, func(t *jwt.Token) (any, error) {
		// Pin EdDSA; reject any other family (e.g. an HS256 token
		// crafted in an alg-confusion attempt). WithValidMethods below
		// is the primary guard; this is defence-in-depth.
		if _, ok := t.Method.(*jwt.SigningMethodEd25519); !ok {
			return nil, fmt.Errorf("unexpected signing method %v", t.Header["alg"])
		}
		return s.cfg.ViewerTokenPubKey, nil
	}, jwt.WithValidMethods([]string{"EdDSA"}), jwt.WithExpirationRequired(), jwt.WithAudience("viewer"))
	if err != nil {
		// jwt.Parse checks exp + nbf and (with WithExpirationRequired)
		// rejects a missing exp. Differentiate the rejection class for
		// log triage without echoing the token.
		return fmt.Errorf("jwt parse/validate: %w", err)
	}
	if !tok.Valid {
		return errors.New("jwt invalid (post-parse)")
	}
	claims, ok := tok.Claims.(jwt.MapClaims)
	if !ok {
		return errors.New("jwt claims not MapClaims")
	}
	// aud must contain "viewer" — distinguishes a viewer token from any
	// other JWT the control plane issues (e.g. a portal session cookie).
	aud, err := claims.GetAudience()
	if err != nil {
		return fmt.Errorf("jwt aud: %w", err)
	}
	hasViewerAud := false
	for _, a := range aud {
		if a == "viewer" {
			hasViewerAud = true
			break
		}
	}
	if !hasViewerAud {
		return fmt.Errorf("jwt aud does not contain %q (got %v)", "viewer", aud)
	}
	// `sub` must be a well-formed UUID — defends against tokens
	// signed under the right key but for a non-customer subject.
	sub, _ := claims["sub"].(string)
	if _, err := uuid.Parse(sub); err != nil {
		return fmt.Errorf("jwt sub not a UUID: %w", err)
	}
	// vm_session_id must match THIS bridge's configured VMSessionID.
	// Without this check, a token minted for a different customer's
	// session would still pass the signature check and grant access
	// to whichever bridge it hit.
	if s.cfg.VMSessionID == "" {
		return errors.New("bridge has no VMSessionID configured (cannot match token)")
	}
	gotVM, _ := claims["vm_session_id"].(string)
	if gotVM != s.cfg.VMSessionID {
		return fmt.Errorf("vm_session_id claim mismatch (want %q got %q)",
			s.cfg.VMSessionID, gotVM)
	}
	return nil
}

// h264ProfileLevelID returns the H.264 profile-level-id for the SDP fmtp.
// Defaults to 42e01f (Constrained Baseline 3.1, what the NVENC cloud path
// emits and every browser decodes). WAYMUX_H264_PROFILE_LEVEL_ID overrides it;
// the local AMD/Vulkan launcher sets 4d0033 (Main 5.1), the profile that
// encoder actually emits, so strict decoders (Firefox) accept the stream
// instead of choking on Main NALUs and black-screening.
func h264ProfileLevelID() string {
	if v := os.Getenv("WAYMUX_H264_PROFILE_LEVEL_ID"); v != "" {
		return v
	}
	return "42e01f"
}

// envBps reads a bits-per-second value from an env var, falling back to def.
// Lets the local LAN launcher lift the WAN-tuned BWE limits — the 8 Mbps cap is
// a WAN setting; on a LAN it just throttles the pacer and adds latency.
func envBps(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// addViewer registers a new viewer and returns it. Last-wins: the new
// viewer becomes primary (input flows), demoting any prior primary, so a
// reload/new tab always has control.
//
// addViewer commits a reservation taken by tryReserveViewerSlot. Under
// mu it consumes one reservation (decrements reservedViewers, the commit) and
// appends to viewers, conserving `len(viewers) + reservedViewers`. The cap was
// already enforced ATOMICALLY at reserve time (pre-upgrade), so this should
// never over-admit; the `len(viewers) >= cap` check below is a defensive
// backstop (also covers direct callers/tests that add without reserving) and
// returns nil if it would exceed the cap.
func (s *Server) addViewer(track *webrtc.TrackLocalStaticSample) *viewer {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Consume this connection's reservation (commit). Clamp at zero so direct
	// callers without a reservation (tests) don't drive the counter negative.
	if s.reservedViewers > 0 {
		s.reservedViewers--
	}
	if len(s.viewers) >= s.maxViewers() {
		// Defensive: the atomic reserve gate should make this unreachable on
		// the normal path. Log so a leak/over-admit would be visible.
		if s.cfg.Logger != nil {
			s.cfg.Logger.Warn("addViewer over cap despite reservation gate",
				"viewers", len(s.viewers), "max", s.maxViewers())
		}
		return nil
	}
	v := &viewer{track: track}
	// Last-wins: the newest viewer takes control, demoting any prior primary,
	// so a reload/new tab always has input (no stale-observer trap).
	for _, x := range s.viewers {
		x.primary.Store(false)
	}
	v.primary.Store(true)
	s.viewers = append(s.viewers, v)
	return v
}

// removeViewer deregisters a viewer. If the removed viewer was primary
// and other viewers exist, the newest remaining auto-promotes (last-wins).
// Returns the count of viewers AFTER removal for caller observability.
func (s *Server) removeViewer(v *viewer) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, x := range s.viewers {
		if x == v {
			s.viewers = append(s.viewers[:i], s.viewers[i+1:]...)
			break
		}
	}
	// Last-wins: if the primary left, hand control to the NEWEST remaining viewer.
	if v.primary.Load() && len(s.viewers) > 0 {
		s.viewers[len(s.viewers)-1].primary.Store(true)
	}
	return len(s.viewers)
}

// snapshotTracks copies the current viewer tracks under the lock so
// the caller can fan out a media sample without holding mu across
// the WriteSample calls (those may block on Pion's internal buffers).
func (s *Server) snapshotTracks() []*webrtc.TrackLocalStaticSample {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*webrtc.TrackLocalStaticSample, len(s.viewers))
	for i, v := range s.viewers {
		out[i] = v.track
	}
	return out
}

// registerConn adds a viewer WebSocket connection to the cursor-broadcast registry.
func (s *Server) registerConn(c *wsConn) {
	s.wsConnsMu.Lock()
	if s.wsConns == nil {
		s.wsConns = make(map[*wsConn]struct{})
	}
	s.wsConns[c] = struct{}{}
	s.wsConnsMu.Unlock()
}

// unregisterConn removes a viewer WebSocket connection from the cursor-broadcast registry.
func (s *Server) unregisterConn(c *wsConn) {
	s.wsConnsMu.Lock()
	delete(s.wsConns, c)
	s.wsConnsMu.Unlock()
}

// snapshotConns copies the current WS connection set under the lock.
func (s *Server) snapshotConns() []*wsConn {
	s.wsConnsMu.Lock()
	defer s.wsConnsMu.Unlock()
	out := make([]*wsConn, 0, len(s.wsConns))
	for c := range s.wsConns {
		out = append(out, c)
	}
	return out
}

// acquireHandshakeSlot reserves an in-flight handshake slot for ip. It
// returns true and a release func on success; false (with a no-op release)
// when the per-IP cap is already reached, so the caller rejects the upgrade
// before allocating anything. Caller MUST invoke release on success.
func (s *Server) acquireHandshakeSlot(ip string) (bool, func()) {
	s.handshakeMu.Lock()
	if s.handshakesPerIP == nil {
		s.handshakesPerIP = make(map[string]int)
	}
	if s.handshakesPerIP[ip] >= s.maxHandshakesPerIP() {
		s.handshakeMu.Unlock()
		return false, func() {}
	}
	s.handshakesPerIP[ip]++
	s.handshakeMu.Unlock()
	var once sync.Once
	release := func() {
		once.Do(func() {
			s.handshakeMu.Lock()
			if n := s.handshakesPerIP[ip]; n <= 1 {
				delete(s.handshakesPerIP, ip)
			} else {
				s.handshakesPerIP[ip] = n - 1
			}
			s.handshakeMu.Unlock()
		})
	}
	return true, release
}

// remoteIP extracts the host portion of an "ip:port" RemoteAddr for the
// per-IP handshake limit. Falls back to the raw string when SplitHostPort
// fails (e.g. an already-bare host), so the limit still keys on something
// stable.
func remoteIP(remoteAddr string) string {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		return remoteAddr
	}
	return host
}

// viewerReservation is a single atomically-acquired viewer slot held between
// the pre-upgrade cap gate (tryReserveViewerSlot) and the commit (addViewer)
// or release. Exactly one of commit/release ever takes effect — they share a
// single sync.Once — so a reserved slot is accounted for exactly once: either
// it becomes a viewer (commit) or it is freed (release). The deferred
// release on the handler's error paths is therefore a no-op after a
// successful commit, and a commit after a release is impossible.
type viewerReservation struct {
	s    *Server
	once sync.Once
}

// release frees a reservation that was never committed (any error/early-return
// path before addViewer). Idempotent; a no-op after commit.
func (r *viewerReservation) release() {
	r.once.Do(func() {
		r.s.mu.Lock()
		if r.s.reservedViewers > 0 {
			r.s.reservedViewers--
		}
		r.s.mu.Unlock()
	})
}

// commit converts the reservation into a registered viewer: it consumes the
// reservation (the same sync.Once that guards release) and calls addViewer,
// which appends to viewers AND decrements reservedViewers under mu. Returns
// the new viewer. Because the cap was already enforced atomically at reserve
// time, this never over-admits. After commit, the deferred release is a no-op.
func (r *viewerReservation) commit(track *webrtc.TrackLocalStaticSample) *viewer {
	var v *viewer
	r.once.Do(func() {
		v = r.s.addViewer(track)
	})
	return v
}

// tryReserveViewerSlot atomically reserves a viewer slot against the
// MaxViewers cap. Under mu (the same lock addViewer/removeViewer use)
// it checks `len(viewers) + reservedViewers`
// against the cap and, if there is room, increments reservedViewers and
// returns (true, reservation). Because the reservation is taken BEFORE the WS
// upgrade / NewPeerConnection / track creation, the cap is authoritative at
// the pre-upgrade gate: a precise burst of exactly-at-cap connections is
// serialized here and the over-cap ones fast-fail (503) without ever
// allocating a Pion peer.
//
// The caller MUST, on success, `defer res.release()` immediately so every
// error/early-return path frees the slot, and commit the reservation on the
// success path via res.commit (which consumes the slot by appending a viewer
// and decrementing reservedViewers under the same lock). The sum
// `len(viewers) + reservedViewers` is thus conserved and never exceeds the
// cap.
//
// On the over-cap path it returns (false, nil) so the caller can send 503
// pre-upgrade without holding a reservation.
func (s *Server) tryReserveViewerSlot() (bool, *viewerReservation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.viewers)+s.reservedViewers >= s.maxViewers() {
		return false, nil
	}
	s.reservedViewers++
	return true, &viewerReservation{s: s}
}

// broadcastJSON sends msg to every connected viewer WS by handing it off to
// each connection's bounded send channel + dedicated writer goroutine.
// The hand-off is NON-BLOCKING: broadcastJSON never waits on any single
// conn's socket, so a slow/stuck consumer (or the cursor pump calling this
// inline from naluPumpLoop) can never back-pressure the frame pump or the
// other viewers. Cursor traffic uses dropOnFull: a slow viewer just misses
// a cursor update; the rest keep flowing and the encoder is never stalled.
func (s *Server) broadcastJSON(msg map[string]interface{}) {
	for _, c := range s.snapshotConns() {
		c.writeJSON(msg, dropOnFull)
	}
}

// cursorImageToWS decodes a TagCursorImage payload into a JSON-ready map.
func cursorImageToWS(f sock.Frame) map[string]interface{} {
	p := f.Payload
	if len(p) < 8 {
		return map[string]interface{}{"type": "cursor_image", "w": 0, "h": 0}
	}
	w := binary.LittleEndian.Uint16(p[0:2])
	h := binary.LittleEndian.Uint16(p[2:4])
	hx := binary.LittleEndian.Uint16(p[4:6])
	hy := binary.LittleEndian.Uint16(p[6:8])
	return map[string]interface{}{
		"type": "cursor_image", "w": w, "h": h,
		"hotspotX": hx, "hotspotY": hy,
		"rgba": base64.StdEncoding.EncodeToString(p[8:]),
	}
}

// cursorPosToWS decodes a TagCursorPos payload into a JSON-ready map.
func cursorPosToWS(f sock.Frame) map[string]interface{} {
	p := f.Payload
	if len(p) < 12 {
		return map[string]interface{}{"type": "cursor_pos", "x": float32(0), "y": float32(0), "seq": uint32(0)}
	}
	x := math.Float32frombits(binary.LittleEndian.Uint32(p[0:4]))
	y := math.Float32frombits(binary.LittleEndian.Uint32(p[4:8]))
	seq := binary.LittleEndian.Uint32(p[8:12])
	return map[string]interface{}{"type": "cursor_pos", "x": x, "y": y, "seq": seq}
}

// viewerCount returns the number of currently-connected viewers.
// Used in connect/disconnect logging.
func (s *Server) viewerCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.viewers)
}

// handleWS upgrades to a WebSocket, sets up a Pion PeerConnection with
// one H.264 video track, drives SDP/ICE signaling, and forwards browser
// input events to the session.
//
// N concurrent viewers allowed. First-come-first-served:
// the first connected peer is "primary" and its input messages flow
// to the daemon; later peers join as observers (video only, input
// silently dropped). On primary disconnect, the next-oldest observer
// auto-promotes.
func (s *Server) handleWS(w http.ResponseWriter, r *http.Request) {
	// Viewer-token JWT validation BEFORE the WS upgrade. A bad token
	// gets a clean HTTP 401 from the existing connection rather than
	// half-completing the WS handshake.
	if err := s.validateViewerToken(r); err != nil {
		s.cfg.Logger.Warn("viewer-token rejected", "err", err, "remote", r.RemoteAddr)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	// Per-source-IP in-flight handshake limit. Bound concurrent
	// half-open handshakes from one IP BEFORE the MaxViewers check and
	// before any Pion allocation, so a half-open WS flood can't exhaust
	// resources by piling up connections that pass auth but never finish
	// signaling. The slot is held for the whole handler lifetime.
	clientIP := remoteIP(r.RemoteAddr)
	ok, releaseHandshake := s.acquireHandshakeSlot(clientIP)
	if !ok {
		s.cfg.Logger.Warn("ws rejected: per-IP handshake limit", "remote", r.RemoteAddr, "ip", clientIP)
		http.Error(w, "too many concurrent connections from your address", http.StatusTooManyRequests)
		return
	}
	defer releaseHandshake()

	// Hard MaxViewers cap. Reject the upgrade with 503 AFTER auth but
	// BEFORE the WS upgrade / NewPeerConnection / addViewer, so an attacker
	// can't allocate a Pion peer + interceptor chain + ICE agent + track
	// past the cap. The check is atomic w.r.t. concurrent upgrades (taken
	// under mu), so a burst can't all slip past.
	reserved, viewerRes := s.tryReserveViewerSlot()
	if !reserved {
		s.cfg.Logger.Warn("ws rejected: MaxViewers cap reached",
			"remote", r.RemoteAddr, "max", s.maxViewers())
		http.Error(w, "session at viewer capacity", http.StatusServiceUnavailable)
		return
	}
	// Free the reserved slot on EVERY error/early-return path before the
	// commit (WS upgrade failure, NewPeerConnection failure, track create/add
	// failure, etc.). The success path commits the reservation via
	// viewerRes.commit(track), which trips the same sync.Once so this deferred
	// release becomes a no-op — the slot is accounted for exactly once.
	defer viewerRes.release()

	rawConn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		s.cfg.Logger.Warn("ws upgrade", "err", err)
		return
	}
	// Wrap in a per-connection writer (bounded send chan + dedicated
	// goroutine). ALL writes to this socket (broadcast, cursor, ping,
	// stats, ICE, config, answer) go through conn.writeJSON/writePing and
	// are serialized by that single goroutine, with a per-write deadline.
	// No process-global write mutex; a stuck consumer stalls only itself.
	conn := newWSConn(rawConn, s.cfg.Logger)
	// registerConn (cursor-broadcast registry) intentionally precedes the
	// peer/track setup and addViewer commit below: a conn is registered for
	// cursor broadcasts before its WebRTC peer is wired up, so it may briefly
	// receive cursor frames before it has a media track. This is harmless —
	// broadcastJSON is non-blocking dropOnFull, so an early cursor frame to a
	// not-yet-ready conn is just enqueued (or dropped), never an error.
	s.registerConn(conn)
	defer s.unregisterConn(conn)
	defer conn.stop()
	s.cfg.Logger.Info("ws connected", "remote", r.RemoteAddr)

	// WebSocket keepalive. A peer that dies WITHOUT sending a close frame
	// (network drop, TURN relay death, phone sleep, abrupt reload) leaves
	// conn.ReadMessage() blocked forever — so the deferred removeViewer never
	// runs and the dead connection keeps the input "primary" slot, making every
	// reconnect a no-input observer. Pings + a read deadline the pong handler
	// extends make the read loop error out within ~pongWait of a silent death,
	// so cleanup runs and the next-oldest viewer auto-promotes to primary.
	const pongWait = 10 * time.Second
	const pingPeriod = (pongWait * 9) / 10
	// Read-side deadlines/handlers operate on the raw conn (the wsConn
	// wrapper owns only the WRITE path). The read loop below also uses
	// rawConn.ReadMessage().
	_ = rawConn.SetReadDeadline(time.Now().Add(pongWait))
	rawConn.SetPongHandler(func(string) error {
		return rawConn.SetReadDeadline(time.Now().Add(pongWait))
	})

	// Stats ticker. Pushes {type:"stats", srcFps:N} every
	// second so the browser can render a live FPS meter. Stops when
	// the WS closes (ctx is cancelled in the deferred cleanup below).
	statsCtx, statsCancel := context.WithCancel(r.Context())
	defer statsCancel()

	// Ping keepalive goroutine (see pongWait note above). Exits when the WS
	// closes (statsCtx cancelled by the deferred cleanup) or the conn's
	// writer goroutine has stopped. The ping is queued non-blocking onto
	// THIS conn's send channel: a stuck peer can never block pings to
	// OTHER viewers. closeOnFull means a conn so wedged it can't absorb a
	// ping is reaped: exactly the dead-peer detection we want.
	go func() {
		ticker := time.NewTicker(pingPeriod)
		defer ticker.Stop()
		for {
			select {
			case <-statsCtx.Done():
				return
			case <-conn.stopCh:
				return
			case <-ticker.C:
				conn.writePing()
			}
		}
	}()
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		var lastCount uint64 = s.frameCount.Load()
		for {
			select {
			case <-statsCtx.Done():
				return
			case <-conn.stopCh:
				return
			case <-ticker.C:
				now := s.frameCount.Load()
				delta := now - lastCount
				lastCount = now
				// Stats are control-rate; closeOnFull reaps a conn
				// that can't keep up with even 1 msg/s.
				conn.writeJSON(map[string]interface{}{
					"type":   "stats",
					"srcFps": delta,
				}, closeOnFull)
			}
		}
	}()

	// Build the ICE server list from env-driven Config. Empty when
	// neither STUN nor TURN env vars are set — the bridge then runs
	// in LAN-only mode (its original behaviour).
	browserICE, pionICE := buildICEServers(s.cfg)

	// Send the `config` message BEFORE anything else so the browser
	// can construct its RTCPeerConnection with the same ICE servers.
	// Skipped entirely when there's nothing to send, preserving the
	// no-env-vars-set fast path for dev.
	if len(browserICE) > 0 {
		out := signalingMsg{Type: "config", ICEServers: browserICE}
		// Control message: closeOnFull (a viewer that can't receive the
		// very first config is wedged). Non-blocking either way.
		if !conn.writeJSON(out, closeOnFull) {
			s.cfg.Logger.Warn("ws write config dropped/conn closing")
			return
		}
	}

	pcCfg := webrtc.Configuration{ICEServers: pionICE}
	mediaEngine := &webrtc.MediaEngine{}
	if err := mediaEngine.RegisterDefaultCodecs(); err != nil {
		s.cfg.Logger.Error("RegisterDefaultCodecs", "err", err)
		return
	}
	// T0/T1: Google Congestion Control adaptive bitrate. Build a
	// send-side BWE controller and TWCC feedback path so Pion can
	// estimate the available bandwidth and we can drive the session's
	// NVENC bitrate to match. The MediaEngine + registry + API are all
	// built per-handleWS (per viewer), so the per-PeerConnection
	// estimator surfaced by OnNewPeerConnection is unambiguously THIS
	// connection's — no id correlation needed.
	interceptorRegistry := &interceptor.Registry{}
	ccFactory, err := cc.NewInterceptor(func() (cc.BandwidthEstimator, error) {
		return gcc.NewSendSideBWE(
			// Floor must be able to drop to real cellular bandwidth. The old 4M
			// floor ("keep 1080p sharp, drop frames rather than soften") is a
			// LAN decision — on a ~1-2 Mbps cellular link it pins the estimate
			// above the link, so the encoder floods and frames drop (choppy).
			// Now the encoder softens (adaptive QP) instead. Env-overridable.
			gcc.SendSideBWEMinBitrate(envBps("WAYMUX_BWE_MIN_BPS", 400_000)),
			// Start conservative and let GCC ramp UP — starting high overshoots
			// a cellular link immediately, causing a loss→crash cycle.
			gcc.SendSideBWEInitialBitrate(envBps("WAYMUX_BWE_INITIAL_BPS", 2_000_000)),
			gcc.SendSideBWEMaxBitrate(envBps("WAYMUX_BWE_MAX_BPS", 8_000_000)),
		)
	})
	if err != nil {
		s.cfg.Logger.Error("cc.NewInterceptor", "err", err)
		return
	}
	// Captured by the OnNewPeerConnection callback below. Read by the
	// bitrate-forwarding goroutine via an atomic load.
	var bwe atomic.Pointer[cc.BandwidthEstimator]
	ccFactory.OnNewPeerConnection(func(_ string, est cc.BandwidthEstimator) {
		bwe.Store(&est)
	})
	interceptorRegistry.Add(ccFactory)
	// transport-cc needs BOTH halves, and the two webrtc helpers each do only
	// one — using just one breaks GCC in a different way:
	//   1. RegisterFeedback(transport-cc) → puts `a=rtcp-fb transport-cc` in the
	//      SDP so the BROWSER sends TWCC feedback (without it GCC gets no data
	//      and sits at the initial bitrate). ConfigureTWCCHeaderExtensionSender
	//      does NOT do this; ConfigureTWCCSender does.
	//   2. ConfigureTWCCHeaderExtensionSender → adds the HeaderExtensionInterceptor
	//      that STAMPS the transport-cc seq-num onto our OUTBOUND packets, which
	//      the cc pacer requires (without it: "pacer ERROR: missing transport
	//      layer cc header extension" on every packet → black video).
	//      ConfigureTWCCSender does NOT add this stamper.
	// So: register the feedback ourselves, then add the header-extension stamper.
	mediaEngine.RegisterFeedback(
		webrtc.RTCPFeedback{Type: webrtc.TypeRTCPFBTransportCC}, webrtc.RTPCodecTypeVideo,
	)
	if err := webrtc.ConfigureTWCCHeaderExtensionSender(mediaEngine, interceptorRegistry); err != nil {
		s.cfg.Logger.Error("ConfigureTWCCHeaderExtensionSender", "err", err)
		return
	}
	// NACK responder: retransmit packets the browser reports lost. Building a
	// custom registry (for GCC) skips the defaults, so this MUST be added back
	// — without it, any packet loss (constant on wifi/internet/relay) is
	// unrecoverable: the decoder stalls and PLIs → regular stutter + input lag.
	if err := webrtc.ConfigureNack(mediaEngine, interceptorRegistry); err != nil {
		s.cfg.Logger.Error("ConfigureNack", "err", err)
		return
	}
	// RTCP sender/receiver reports (RTT + loss stats) — standard, feeds GCC and
	// the browser's stats.
	if err := webrtc.ConfigureRTCPReports(interceptorRegistry); err != nil {
		s.cfg.Logger.Error("ConfigureRTCPReports", "err", err)
		return
	}

	api := webrtc.NewAPI(
		webrtc.WithMediaEngine(mediaEngine),
		webrtc.WithInterceptorRegistry(interceptorRegistry),
	)
	pc, err := api.NewPeerConnection(pcCfg)
	if err != nil {
		s.cfg.Logger.Error("NewPeerConnection", "err", err)
		return
	}
	defer pc.Close()

	// H.264 codec parameters: include packetization-mode=1 so the
	// browser accepts FU-A fragmented packets (Pion's default for
	// any NALU > MTU). Without this fmtp the browser may assume
	// mode 0 (single-NAL-per-packet only), which our encoder
	// violates on the first keyframe → black <video> despite
	// connected ICE. profile-level-id=42e01f = Constrained Baseline
	// level 3.1, universally decoded.
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{
			MimeType:    webrtc.MimeTypeH264,
			ClockRate:   90000,
			SDPFmtpLine: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=" + h264ProfileLevelID(),
		},
		"video", "waymux",
	)
	if err != nil {
		s.cfg.Logger.Error("NewTrackLocalStaticSample", "err", err)
		return
	}
	sender, err := pc.AddTrack(track)
	if err != nil {
		s.cfg.Logger.Error("AddTrack", "err", err)
		return
	}

	// Register as a viewer up-front (last-wins: this connection becomes
	// primary), so the GCC bitrate goroutine below can gate on v.primary —
	// only the primary viewer's estimator drives the single shared encoder.
	// Otherwise multiple open tabs each push their own GCC estimate and the
	// estimators fight over the bitrate (the 4<->11Mbps thrash → stutter +
	// input lag). The role log + deferred removeViewer stay below.
	// Commit the reservation: convert it into a registered viewer. The cap
	// was already enforced atomically at reserve time (pre-upgrade), so this
	// holds a guaranteed slot and should never return nil. Keep the nil guard
	// as a defensive backstop — on a nil commit the deferred viewerRes.release
	// has already run (the commit trips the shared once without admitting), so
	// no reservation leaks; pc is torn down by the deferred cleanup below.
	v := viewerRes.commit(track)
	if v == nil {
		s.cfg.Logger.Warn("ws rejected post-upgrade: MaxViewers cap reached (defensive)",
			"remote", r.RemoteAddr, "max", s.maxViewers())
		return
	}

	// T3: forward browser RTCP PLI/FIR to the session encoder as an
	// on-demand force-keyframe. Without this, a viewer that joins or
	// recovers mid-motion gets no IDR until the session's idle
	// heartbeat fires (which only happens on a static desktop) —
	// leaving the <video> black/garbled until the next scene change.
	//
	// ReadRTCP() blocks until a packet arrives or the sender's
	// transport closes (on pc.Close, which the handleWS defer triggers
	// when this viewer disconnects). On that close it returns an error
	// and the goroutine exits, so it's bounded to this connection's
	// lifetime — no leak across viewers.
	go func() {
		for {
			pkts, _, rerr := sender.ReadRTCP()
			if rerr != nil {
				// io.EOF / closed pipe on viewer disconnect is the
				// normal exit; nothing to log.
				return
			}
			for _, p := range pkts {
				switch p.(type) {
				case *rtcp.PictureLossIndication, *rtcp.FullIntraRequest:
					s.sockMu.Lock()
					werr := sock.WriteFrame(s.cfg.SocketW, sock.Frame{Tag: sock.TagForceKeyframe})
					s.sockMu.Unlock()
					if werr != nil {
						s.cfg.Logger.Warn("force-keyframe write failed", "err", werr)
					}
				}
			}
		}
	}()

	// T0/T1: forward the GCC bandwidth estimate to the session as a
	// TagSetBitrate frame (drives NVENC reconfigure). Rate-limited:
	// each reconfigure forces an IDR on the session, so we poll once a
	// second and only emit a frame on a meaningful change (>=0.5 Mbps
	// or >=10%). Bounded to this connection via statsCtx (cancelled by
	// the handleWS defer on viewer disconnect) — no leak across viewers.
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		// Match the BWE floor/cap. minBitrate must track the cellular-capable
		// floor so the encoder can soften instead of flooding.
		minBitrate := envBps("WAYMUX_BWE_MIN_BPS", 400_000)
		maxBitrate := envBps("WAYMUX_BWE_MAX_BPS", 8_000_000)
		var lastSent int
		for {
			select {
			case <-statsCtx.Done():
				return
			case <-ticker.C:
				// Only the primary viewer's GCC drives the shared encoder
				// bitrate; otherwise multiple tabs' estimators thrash it.
				if !v.primary.Load() {
					continue
				}
				estPtr := bwe.Load()
				if estPtr == nil {
					continue
				}
				// Target 85% of the GCC estimate: the encoder's QP loop holds
				// the average near this target but can peak ~10% over on motion/
				// IDR bursts, so leaving headroom under the measured link
				// capacity prevents those peaks from causing congestion loss.
				target := (*estPtr).GetTargetBitrate() * 85 / 100
				if target < minBitrate {
					target = minBitrate
				}
				if target > maxBitrate {
					target = maxBitrate
				}
				delta := target - lastSent
				if delta < 0 {
					delta = -delta
				}
				// Finer granularity than the old 500k floor: on cellular the
				// whole budget can be ~1 Mbps, so a 500k threshold (50%) is far
				// too coarse. The Vulkan QP encoder doesn't IDR on SetBitrate,
				// so frequent small updates are cheap.
				threshold := lastSent / 10
				if threshold < 150_000 {
					threshold = 150_000
				}
				if lastSent != 0 && delta < threshold {
					continue
				}
				payload := make([]byte, 4)
				binary.LittleEndian.PutUint32(payload, uint32(target)) //nolint:gosec // clamped to [1.5M,12M]
				s.sockMu.Lock()
				werr := sock.WriteFrame(s.cfg.SocketW, sock.Frame{Tag: sock.TagSetBitrate, Payload: payload})
				s.sockMu.Unlock()
				if werr != nil {
					s.cfg.Logger.Warn("set-bitrate write failed", "err", werr)
					continue
				}
				s.cfg.Logger.Info("gcc bitrate change", "bps", target, "prevBps", lastSent)
				lastSent = target
			}
		}
	}()

	// Connect lifecycle logging.
	pc.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		s.cfg.Logger.Info("ice state", "state", state.String())
	})
	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		s.cfg.Logger.Info("peer state", "state", state.String())
	})

	// ICE candidates flow client → server via the WS. Server-side
	// candidates flow back via the same channel.
	pc.OnICECandidate(func(c *webrtc.ICECandidate) {
		if c == nil {
			return
		}
		ci := c.ToJSON()
		b, err := json.Marshal(ci)
		if err != nil {
			s.cfg.Logger.Warn("marshal ice", "err", err)
			return
		}
		out := signalingMsg{Type: "ice", Candidate: b}
		// Signaling message: closeOnFull so a wedged peer is reaped
		// rather than silently losing candidates and hanging ICE.
		conn.writeJSON(out, closeOnFull)
	})

	// Register as a viewer (possibly primary, possibly
	// observer). The fan-out path in naluPumpLoop picks up the track
	// on its next snapshot. Cleanup removes us from the slice and
	// auto-promotes the next viewer to primary if we were primary.
	// (v was registered above so the bitrate goroutine could gate on it.)
	role := "observer"
	if v.primary.Load() {
		role = "primary"
	}
	s.cfg.Logger.Info("ws connected as viewer",
		"role", role, "viewers_total", s.viewerCount())
	defer func() {
		remaining := s.removeViewer(v)
		s.cfg.Logger.Info("ws disconnected",
			"was_primary", v.primary.Load(),
			"viewers_remaining", remaining)
	}()

	// Drive signaling and input loops on the same goroutine; the
	// browser is the offerer (recvonly). Pion drives answer setup
	// when we feed it the offer.
	for {
		_, raw, err := rawConn.ReadMessage()
		if err != nil {
			if !websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.cfg.Logger.Info("ws read closed", "err", err)
			}
			return
		}
		var msg signalingMsg
		if err := json.Unmarshal(raw, &msg); err != nil {
			s.cfg.Logger.Warn("ws bad json", "err", err)
			continue
		}
		switch msg.Type {
		case "offer":
			if err := s.handleOffer(pc, conn, msg.SDP); err != nil {
				s.cfg.Logger.Warn("handle offer", "err", err)
				return
			}
		case "ice":
			if len(msg.Candidate) == 0 {
				continue
			}
			var ci webrtc.ICECandidateInit
			if err := json.Unmarshal(msg.Candidate, &ci); err != nil {
				s.cfg.Logger.Warn("ws bad ice", "err", err)
				continue
			}
			if err := pc.AddICECandidate(ci); err != nil {
				s.cfg.Logger.Warn("AddICECandidate", "err", err)
			}
		case "event":
			// Only the primary viewer's input flows to the daemon.
			// Observer event messages are silently dropped: they
			// still get video, but their mouse/keyboard does nothing
			// remote-side. (A "Request control" UX for observers is
			// future work; today it is silent.)
			if v.primary.Load() {
				s.forwardInput(raw)
			}
		default:
			s.cfg.Logger.Warn("unknown ws type", "type", msg.Type)
		}
	}
}

// handleOffer applies the client's SDP offer, generates an answer,
// waits for ICE gathering to settle (trickle is enabled separately
// via OnICECandidate), and sends the answer over the WS.
func (s *Server) handleOffer(pc *webrtc.PeerConnection, conn *wsConn, offerSDP string) error {
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  offerSDP,
	}); err != nil {
		return fmt.Errorf("SetRemoteDescription: %w", err)
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		return fmt.Errorf("CreateAnswer: %w", err)
	}
	if err := pc.SetLocalDescription(answer); err != nil {
		return fmt.Errorf("SetLocalDescription: %w", err)
	}
	out := signalingMsg{Type: "answer", SDP: answer.SDP}
	// Signaling message: closeOnFull so a wedged peer is reaped rather
	// than hanging the handshake. Non-blocking enqueue.
	if !conn.writeJSON(out, closeOnFull) {
		return errors.New("ws write answer dropped/conn closing")
	}
	return nil
}

// forwardInput translates a neko-format event from the WS into a
// waymux InjectOp and ships it over the Unix socket as TagInjectOp.
//
// Best-effort: a translator error logs and drops the event. The
// translator handles all the neko event kinds we declared support
// for; new kinds need to be added to translator.go.
func (s *Server) forwardInput(raw []byte) {
	s.cfg.Logger.Debug("input event from browser", "raw", string(raw))
	op, err := inputx.Translate(raw)
	if err != nil {
		s.cfg.Logger.Warn("translate input", "err", err, "raw", string(raw))
		return
	}
	s.cfg.Logger.Debug("input → InjectOp", "op", string(op))
	s.sockMu.Lock()
	defer s.sockMu.Unlock()
	if err := sock.WriteFrame(s.cfg.SocketW, sock.Frame{Tag: sock.TagInjectOp, Payload: op}); err != nil {
		s.cfg.Logger.Warn("ws → socket inject", "err", err)
	}
}

// naluPumpLoop reads frames from the Unix socket and pushes each
// `TagNalu` payload into the active Pion track (if any). When no
// viewer is connected, NALUs are dropped on the floor.
//
// 16ms duration matches a nominal 60 fps; Pion uses it as the
// presentation-time delta in the outgoing RTP timestamps. Each
// NALU is a complete coded slice (we shipped one Frame::Nalu per
// NALU from the encoder side), so Pion's chunker can emit it as
// one RTP packet sequence per call.
func (s *Server) naluPumpLoop(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}
		f, err := sock.ReadFrame(s.cfg.SocketR)
		if err != nil {
			if errors.Is(err, io.EOF) {
				s.cfg.Logger.Info("nalu pump: EOF from session")
				return nil
			}
			return err
		}
		switch f.Tag {
		case sock.TagNalu:
			// Fan the sample out to every connected
			// viewer's track. Snapshot under mu, then WriteSample
			// without the lock (Pion may block briefly on its
			// internal queue and we don't want to stall the
			// next viewer's frame on the previous one).
			tracks := s.snapshotTracks()
			if len(tracks) == 0 {
				continue
			}
			sample := media.Sample{
				Data:     f.Payload,
				Duration: 16 * time.Millisecond,
			}
			anyOk := false
			for _, t := range tracks {
				if err := t.WriteSample(sample); err != nil {
					s.cfg.Logger.Warn("WriteSample", "err", err)
					continue
				}
				anyOk = true
			}
			// Bump source-FPS counter once per encoded sample
			// regardless of viewer count (it's an encoder-side
			// metric, not a per-viewer one).
			if anyOk {
				s.frameCount.Add(1)
			}
		case sock.TagCursorImage:
			s.broadcastJSON(cursorImageToWS(f))
		case sock.TagCursorPos:
			s.broadcastJSON(cursorPosToWS(f))
		case sock.TagPtsHint:
			// Future: drive RTP timestamps from upstream PTS. For
			// now the 16 ms duration above is the source of truth.
		case sock.TagShutdown:
			s.cfg.Logger.Info("session sent shutdown", "payload", f.Payload)
			return sock.ErrShutdown
		default:
			s.cfg.Logger.Warn("unexpected tag", "tag", fmt.Sprintf("0x%02X", f.Tag))
		}
	}
}
