// SPDX-License-Identifier: Apache-2.0

// Unit tests for the viewer-token validator: the switch from an HS256
// shared secret to EdDSA verification with a PUBLIC key only.
//
// The control plane (crates/waymux-api) mints viewer tokens with an
// Ed25519 PRIVATE key and ships the bridge only the 32-byte PUBLIC key.
// These tests mirror the Rust-side `viewer_token_tests` so a wire-shape
// mismatch between the mint endpoint and the bridge validator shows up
// as a CI failure rather than a customer-side 401.
//
// CROSS-LANGUAGE CONTRACT: testSeed is the SAME fixed seed the Rust
// helper `Config::testing_viewer_token_signing_key()` uses (bytes
// 0..31). goldenPubKeyB64 / goldenToken below are captured from the Rust
// side and asserted here to lock Rust→Go interop.
package server

import (
	"crypto/ed25519"
	"encoding/base64"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

// testSeed is the deterministic 32-byte ed25519 seed shared with the
// Rust testing helper (bytes 0,1,2,...,31). Both languages derive the
// same keypair from it.
func testSeed() []byte {
	s := make([]byte, 32)
	for i := range s {
		s[i] = byte(i)
	}
	return s
}

// testPubKey derives the ed25519 public key from testSeed — exactly what
// a VM would receive as WAYMUX_VIEWER_TOKEN_ED25519_PK (raw 32 bytes).
func testPubKey() ed25519.PublicKey {
	priv := ed25519.NewKeyFromSeed(testSeed())
	return priv.Public().(ed25519.PublicKey)
}

func testPrivKey() ed25519.PrivateKey {
	return ed25519.NewKeyFromSeed(testSeed())
}

// slogDiscard returns a logger whose output goes nowhere — keeps test
// stdout clean. The bridge's handleWS calls Logger.Warn on rejection;
// without this we'd noisy-print on every test.
func slogDiscard() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))
}

// mintEdDSA signs a viewer token with the given key + claims. priv being
// the wrong key lets a test exercise the signature-rejection path.
func mintEdDSA(t *testing.T, priv ed25519.PrivateKey, claims jwt.MapClaims) string {
	t.Helper()
	tok := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	signed, err := tok.SignedString(priv)
	if err != nil {
		t.Fatalf("mint EdDSA jwt: %v", err)
	}
	return signed
}

// viewerClaims builds the standard claim set the control plane mints.
func viewerClaims(sub, vmSessionID, aud string, expDelta time.Duration) jwt.MapClaims {
	now := time.Now()
	c := jwt.MapClaims{
		"sub":           sub,
		"vm_session_id": vmSessionID,
		"iat":           now.Unix(),
		"exp":           now.Add(expDelta).Unix(),
	}
	if aud != "" {
		c["aud"] = aud
	}
	return c
}

// newServer builds a Server configured to verify EdDSA viewer tokens
// against the given public key + vm_session_id.
func newServer(t *testing.T, pub ed25519.PublicKey, vmSessionID string) *Server {
	t.Helper()
	return &Server{cfg: Config{
		ViewerTokenPubKey: pub,
		VMSessionID:       vmSessionID,
		Logger:            slogDiscard(),
	}}
}

// Empty pubkey on a LOOPBACK bind skips validation entirely — the V1 dev
// path. Pins backwards-compat for local development.
func TestValidateViewerToken_NoKeyLoopbackSkips(t *testing.T) {
	s := &Server{cfg: Config{BoundNonLoopback: false, Logger: slogDiscard()}}
	r := httptest.NewRequest("GET", "/ws", nil) // no ?token=
	if err := s.validateViewerToken(r); err != nil {
		t.Fatalf("no-key loopback mode must pass without token, got: %v", err)
	}
}

// FAIL-CLOSED: empty pubkey on a NON-loopback (public) bind must reject
// every upgrade — a misconfigured production bridge must not serve
// unauthenticated traffic on the public internet.
func TestValidateViewerToken_NoKeyNonLoopbackRejects(t *testing.T) {
	s := &Server{cfg: Config{BoundNonLoopback: true, Logger: slogDiscard()}}
	r := httptest.NewRequest("GET", "/ws", nil)
	err := s.validateViewerToken(r)
	if err == nil {
		t.Fatal("empty pubkey + non-loopback bind must fail closed")
	}
	if !strings.Contains(err.Error(), "fail-closed") {
		t.Fatalf("expected fail-closed error, got: %v", err)
	}
}

// Token signed with the configured key + aud=viewer + matching
// vm_session_id is accepted.
func TestValidateViewerToken_HappyPath(t *testing.T) {
	vm := uuid.NewString()
	sub := uuid.NewString()
	token := mintEdDSA(t, testPrivKey(), viewerClaims(sub, vm, "viewer", 5*time.Minute))

	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	if err := s.validateViewerToken(r); err != nil {
		t.Fatalf("happy path must pass, got: %v", err)
	}
}

// Missing ?token= when a pubkey is configured → reject.
func TestValidateViewerToken_MissingToken(t *testing.T) {
	s := newServer(t, testPubKey(), uuid.NewString())
	r := httptest.NewRequest("GET", "/ws", nil)
	err := s.validateViewerToken(r)
	if err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("expected 'missing' error, got: %v", err)
	}
}

// Token signed with a WRONG key must fail the EdDSA signature check.
func TestValidateViewerToken_WrongKey(t *testing.T) {
	vm := uuid.NewString()
	// A different seed → different keypair.
	wrongSeed := make([]byte, 32)
	for i := range wrongSeed {
		wrongSeed[i] = byte(255 - i)
	}
	wrongPriv := ed25519.NewKeyFromSeed(wrongSeed)
	token := mintEdDSA(t, wrongPriv, viewerClaims(uuid.NewString(), vm, "viewer", 5*time.Minute))

	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	err := s.validateViewerToken(r)
	if err == nil {
		t.Fatal("token signed with a different key must fail")
	}
	if !strings.Contains(err.Error(), "jwt parse/validate") {
		t.Fatalf("expected parse/validate error, got: %v", err)
	}
}

// Token with NO exp must be rejected (WithExpirationRequired).
func TestValidateViewerToken_NoExp(t *testing.T) {
	vm := uuid.NewString()
	claims := jwt.MapClaims{
		"sub":           uuid.NewString(),
		"vm_session_id": vm,
		"aud":           "viewer",
		"iat":           time.Now().Unix(),
		// no exp
	}
	token := mintEdDSA(t, testPrivKey(), claims)
	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	if err := s.validateViewerToken(r); err == nil {
		t.Fatal("token with no exp must be rejected (WithExpirationRequired)")
	}
}

// Expired token must fail.
func TestValidateViewerToken_Expired(t *testing.T) {
	vm := uuid.NewString()
	token := mintEdDSA(t, testPrivKey(), viewerClaims(uuid.NewString(), vm, "viewer", -1*time.Hour))
	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	if err := s.validateViewerToken(r); err == nil {
		t.Fatal("expired token must fail")
	}
}

// Token with the WRONG audience must be rejected.
func TestValidateViewerToken_WrongAud(t *testing.T) {
	vm := uuid.NewString()
	token := mintEdDSA(t, testPrivKey(), viewerClaims(uuid.NewString(), vm, "portal", 5*time.Minute))
	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	err := s.validateViewerToken(r)
	if err == nil || !strings.Contains(err.Error(), "aud") {
		t.Fatalf("expected aud error, got: %v", err)
	}
}

// Token with NO audience claim must be rejected.
func TestValidateViewerToken_MissingAud(t *testing.T) {
	vm := uuid.NewString()
	token := mintEdDSA(t, testPrivKey(), viewerClaims(uuid.NewString(), vm, "", 5*time.Minute))
	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	if err := s.validateViewerToken(r); err == nil {
		t.Fatal("token with no aud must be rejected")
	}
}

// Token with valid signature + exp + aud but vm_session_id pointing at a
// DIFFERENT session must fail — defends against cross-session reuse.
func TestValidateViewerToken_WrongVMSessionID(t *testing.T) {
	bridgeVM := uuid.NewString()
	token := mintEdDSA(t, testPrivKey(), viewerClaims(uuid.NewString(), uuid.NewString(), "viewer", 5*time.Minute))
	s := newServer(t, testPubKey(), bridgeVM)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	err := s.validateViewerToken(r)
	if err == nil || !strings.Contains(err.Error(), "vm_session_id") {
		t.Fatalf("expected vm_session_id mismatch error, got: %v", err)
	}
}

// sub that isn't a UUID must fail — pins the Rust side's `sub: Uuid`
// serialization contract.
func TestValidateViewerToken_SubNotUUID(t *testing.T) {
	vm := uuid.NewString()
	token := mintEdDSA(t, testPrivKey(), viewerClaims("not-a-uuid", vm, "viewer", 5*time.Minute))
	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	err := s.validateViewerToken(r)
	if err == nil || !strings.Contains(err.Error(), "sub") {
		t.Fatalf("expected sub-not-UUID error, got: %v", err)
	}
}

// An HS256-signed token must be rejected (algorithm-confusion defence):
// EdDSA-only is enforced via WithValidMethods, so an HMAC token never
// reaches the key func.
func TestValidateViewerToken_HS256Rejected(t *testing.T) {
	vm := uuid.NewString()
	claims := viewerClaims(uuid.NewString(), vm, "viewer", 5*time.Minute)
	tok := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	// Sign with the raw public-key bytes as if they were an HMAC secret —
	// the classic alg-confusion attack. Must still be rejected.
	token, err := tok.SignedString([]byte(testPubKey()))
	if err != nil {
		t.Fatalf("mint hs256: %v", err)
	}
	s := newServer(t, testPubKey(), vm)
	r := httptest.NewRequest("GET", "/ws?token="+token, nil)
	if err := s.validateViewerToken(r); err == nil {
		t.Fatal("HS256 token must be rejected (only EdDSA allowed)")
	}
}

// handleWS short-circuits with HTTP 401 when validateViewerToken returns
// an error — confirms the WS upgrade never starts on a bad token.
func TestHandleWS_RejectsBadToken(t *testing.T) {
	s := newServer(t, testPubKey(), uuid.NewString())
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/ws", nil) // no token
	s.handleWS(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d body=%q", w.Code, w.Body.String())
	}
}

// FAIL-CLOSED at the HTTP layer: empty pubkey + non-loopback bind → /ws
// returns 401.
func TestHandleWS_FailClosedNonLoopback(t *testing.T) {
	s := &Server{cfg: Config{BoundNonLoopback: true, Logger: slogDiscard()}}
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/ws", nil)
	s.handleWS(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 (fail-closed), got %d", w.Code)
	}
}

// ---------------------------------------------------------------------
// CROSS-LANGUAGE GOLDEN VECTORS (Rust → Go interop lock)
// ---------------------------------------------------------------------

// goldenPubKeyB64 is the base64 of the public key the Rust testing helper
// `Config::testing_viewer_token_signing_key()` exposes via
// `public_key_b64()` for the fixed seed [0..31]. Captured from the Rust
// side; asserted here so a divergence in either language's seed→pubkey
// derivation fails CI.
const goldenPubKeyB64 = "A6EHv/POEL4dcN0Y50vAmWfk1jCbpQ1fHdyGZBJVMbg="

// goldenToken is a real EdDSA viewer token MINTED BY THE RUST SIDE
// (jsonwebtoken::encode with the testing signing key) with:
//
//	sub           = 11111111-1111-1111-1111-111111111111
//	vm_session_id = 22222222-2222-2222-2222-222222222222
//	exp           = 7258118400 (year 2200 — never expires for this fixture)
//	iat           = 1700000000
//	aud           = "viewer"
//
// Asserting validateViewerToken ACCEPTS it proves a Rust-minted token
// verifies in Go end-to-end.
const (
	goldenToken = "eyJ0eXAiOiJKV1QiLCJhbGciOiJFZERTQSJ9.eyJzdWIiOiIxMTExMTExMS0xMTExLTExMTEtMTExMS0xMTExMTExMTExMTEiLCJ2bV9zZXNzaW9uX2lkIjoiMjIyMjIyMjItMjIyMi0yMjIyLTIyMjItMjIyMjIyMjIyMjIyIiwiZXhwIjo3MjU4MTE4NDAwLCJpYXQiOjE3MDAwMDAwMDAsImF1ZCI6InZpZXdlciJ9.5LTDTm6OQoPKRQtBl6ygBOb4Y_Dh6s5kOv43Um2u-ibYNAV-OETgZCjzAqB7EHwOQNTL1je-mFrXItw541VoAw"
	goldenVM    = "22222222-2222-2222-2222-222222222222"
)

// The pubkey Go derives from the shared seed must equal the base64 the
// Rust side exposes — proves both languages derive the same key.
func TestCrossLang_PubKeyMatchesRust(t *testing.T) {
	goB64 := base64.StdEncoding.EncodeToString(testPubKey())
	if goB64 != goldenPubKeyB64 {
		t.Fatalf("Go-derived pubkey %q != Rust pubkey %q (seed→pubkey divergence)",
			goB64, goldenPubKeyB64)
	}
}

// A token minted by the Rust side must verify in the Go bridge.
func TestCrossLang_RustMintedTokenVerifies(t *testing.T) {
	s := newServer(t, testPubKey(), goldenVM)
	r := httptest.NewRequest("GET", "/ws?token="+goldenToken, nil)
	if err := s.validateViewerToken(r); err != nil {
		t.Fatalf("Rust-minted golden token must verify in Go, got: %v", err)
	}
}
