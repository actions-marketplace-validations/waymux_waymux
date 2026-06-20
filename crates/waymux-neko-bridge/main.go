// SPDX-License-Identifier: Apache-2.0

// waymux-neko-bridge: WebRTC viewer bridge.
//
// Spawned per session by waymux-session (Rust). Connects to a Unix
// socket for typed NALU/control messages (the Rust side is the
// listener, we are the client) and serves HTML+WebSocket+WebRTC to a
// single browser viewer on the configured bind address.
//
// The WebRTC path is currently stubbed; the binary builds, dials
// the socket, prints READY on stderr after the TCP listener is bound,
// and serves the embedded viewer HTML at "/".
package main

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/waymuxhq/waymux/crates/waymux-neko-bridge/internal/server"
	sock "github.com/waymuxhq/waymux/crates/waymux-neko-bridge/internal/socket"
)

// bindIsLoopback reports whether the bind address resolves to a
// loopback-only address (127.0.0.0/8 or ::1). An empty bind or "0.0.0.0"
// / "::" wildcard is treated as NON-loopback (public): a wildcard bind
// listens on every interface including public ones. A non-IP / unparseable
// value is treated as non-loopback (fail-closed-leaning).
func bindIsLoopback(bind string) bool {
	if bind == "" {
		return false
	}
	ip := net.ParseIP(bind)
	if ip == nil {
		return false
	}
	return ip.IsLoopback()
}

// envInt reads a non-negative integer from env var name. Returns def when
// the var is unset, empty, or not a parseable positive integer (logging a
// warning in the latter case). Used for the DoS-hardening caps (#12).
func envInt(logger *slog.Logger, name string, def int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		logger.Warn("invalid env value; using default", "var", name, "value", raw, "default", def)
		return def
	}
	return n
}

// readICEEnv pulls TURN/STUN configuration from env vars rather than
// CLI flags so the credentials don't show up in `ps`/argv.
//
// The bridge does not mint TURN REST creds from a fleet-wide shared
// secret. The control plane mints per-session creds and delivers them
// as WAYMUX_TURN_USERNAME / WAYMUX_TURN_CREDENTIAL; the bridge just
// forwards them.
//
// Empty result is fine: server falls back to LAN-only ICE behaviour.
func readICEEnv() (stun, turn []string, username, credential string, warn string) {
	splitCSV := func(v string) []string {
		out := []string{}
		for _, p := range strings.Split(v, ",") {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
		return out
	}

	stun = splitCSV(os.Getenv("WAYMUX_STUN_URLS"))
	turn = splitCSV(os.Getenv("WAYMUX_TURN_URL"))
	username = strings.TrimSpace(os.Getenv("WAYMUX_TURN_USERNAME"))
	credential = strings.TrimSpace(os.Getenv("WAYMUX_TURN_CREDENTIAL"))

	// Operator-aid: catch the common misconfiguration where TURN URLs are
	// set without the matching creds (or vice-versa). Don't fail (the
	// bridge stays useful in LAN-only mode), but log a warning so the
	// problem is visible at startup rather than at first-customer-connect.
	haveCreds := username != "" && credential != ""
	switch {
	case len(turn) > 0 && !haveCreds:
		warn = "WAYMUX_TURN_URL set but WAYMUX_TURN_USERNAME/WAYMUX_TURN_CREDENTIAL incomplete — TURN entries will be dropped"
	case len(turn) == 0 && haveCreds:
		warn = "WAYMUX_TURN_USERNAME/WAYMUX_TURN_CREDENTIAL set but WAYMUX_TURN_URL is empty — creds have no effect"
	}
	return
}

func main() {
	bind := flag.String("bind", "127.0.0.1", "bind address for HTTP + WebSocket + WebRTC")
	port := flag.Int("port", 0, "TCP port (0 = ephemeral)")
	socketPath := flag.String("socket", "", "Unix socket path to waymux-session")
	width := flag.Int("width", 1920, "session width (px)")
	height := flag.Int("height", 1080, "session height (px)")
	logLevel := flag.String("log-level", "info", "log level (debug, info, warn, error)")
	flag.Parse()

	level := slog.LevelInfo
	switch *logLevel {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	if *socketPath == "" {
		logger.Error("--socket is required")
		os.Exit(2)
	}

	// Connect to the Unix socket. waymux-session is the listener; we connect.
	conn, err := net.Dial("unix", *socketPath)
	if err != nil {
		logger.Error("dial socket", "path", *socketPath, "err", err)
		os.Exit(3)
	}
	defer conn.Close()

	stun, turn, turnUsername, turnCredential, iceWarn := readICEEnv()
	if iceWarn != "" {
		logger.Warn("ice config", "warn", iceWarn)
	}

	// Viewer-token verification with an
	// asymmetric Ed25519 PUBLIC key. The control plane's
	// /sessions/:id/viewer-token endpoint mints EdDSA-signed JWTs (signed
	// with a private key the control plane alone holds) whose
	// `vm_session_id` claim binds the token to one bridge. The bridge gets
	// only the RAW 32-byte public key, delivered as base64
	// `WAYMUX_VIEWER_TOKEN_ED25519_PK` in /etc/waymux/session.env at VM
	// provision time by build_session_user_data on the control plane
	// (see crates/waymux-api/src/session_lifecycle.rs).
	//
	// Go's ed25519.PublicKey IS exactly those raw 32 bytes, so we decode
	// the base64 and use the slice directly (no DER parsing).
	//
	// FAIL-CLOSED: an unset/malformed key means no auth is possible; the
	// bridge serves unauthenticated only on a loopback bind (dev). On a
	// public bind, validateViewerToken refuses every /ws upgrade (see
	// BoundNonLoopback below).
	var viewerPubKey ed25519.PublicKey
	if raw := os.Getenv("WAYMUX_VIEWER_TOKEN_ED25519_PK"); raw != "" {
		decoded, err := base64.StdEncoding.DecodeString(strings.TrimSpace(raw))
		if err != nil {
			logger.Error("WAYMUX_VIEWER_TOKEN_ED25519_PK is not valid base64; viewer auth disabled",
				"err", err)
		} else if len(decoded) != ed25519.PublicKeySize {
			logger.Error("WAYMUX_VIEWER_TOKEN_ED25519_PK wrong length; viewer auth disabled",
				"got_bytes", len(decoded), "want_bytes", ed25519.PublicKeySize)
		} else {
			viewerPubKey = ed25519.PublicKey(decoded)
		}
	}
	vmSessionID := os.Getenv("WAYMUX_VM_SESSION_ID")
	if (len(viewerPubKey) == 0) != (vmSessionID == "") {
		logger.Warn("viewer-token partial config",
			"WAYMUX_VIEWER_TOKEN_ED25519_PK_valid", len(viewerPubKey) != 0,
			"WAYMUX_VM_SESSION_ID_set", vmSessionID != "")
	}

	// Decide fail-closed posture from the bind address. A bind that is
	// NOT loopback (e.g. 0.0.0.0 or a floating IP) must NOT serve
	// unauthenticated viewer traffic; validateViewerToken enforces this
	// when viewerPubKey is empty.
	boundNonLoopback := !bindIsLoopback(*bind)
	if len(viewerPubKey) == 0 && boundNonLoopback {
		logger.Warn("viewer-token public key unset/invalid AND bound to non-loopback; /ws upgrades will be REJECTED (fail-closed)",
			"bind", *bind)
	}

	srv, err := server.New(server.Config{
		Bind:              *bind,
		Port:              *port,
		Width:             *width,
		Height:            *height,
		Logger:            logger,
		SocketR:           conn,
		SocketW:           conn,
		STUNServers:       stun,
		TURNServers:       turn,
		TURNUsername:      turnUsername,
		TURNCredential:    turnCredential,
		ViewerTokenPubKey: viewerPubKey,
		VMSessionID:       vmSessionID,
		BoundNonLoopback:  boundNonLoopback,
		// #12 DoS hardening: concurrent-viewer cap + per-IP handshake
		// limit. Defaults applied server-side when unset/invalid.
		MaxViewers:         envInt(logger, "WAYMUX_MAX_VIEWERS", server.DefaultMaxViewers),
		MaxHandshakesPerIP: envInt(logger, "WAYMUX_MAX_HANDSHAKES_PER_IP", server.DefaultMaxHandshakesPerIP),
	})
	if err != nil {
		logger.Error("server.New", "err", err)
		os.Exit(4)
	}

	// Pre-bind so we know the actual port before printing READY.
	if err := srv.Listen(); err != nil {
		logger.Error("server.Listen", "err", err)
		os.Exit(5)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sig
		logger.Info("signal received, shutting down")
		cancel()
	}()

	// READY on stderr so the Rust supervisor knows we're up. Include the
	// actual listening address (resolves --port=0). The Rust side may
	// parse this; until then it's also useful for human debugging.
	fmt.Fprintf(os.Stderr, "READY addr=%s\n", srv.Addr())

	if err := srv.Run(ctx); err != nil {
		logger.Error("server.Run", "err", err)
		// Best-effort shutdown notice back to session.
		_ = sock.WriteFrame(conn, sock.Frame{Tag: sock.TagShutdown, Payload: []byte{1}})
		os.Exit(6)
	}
}
