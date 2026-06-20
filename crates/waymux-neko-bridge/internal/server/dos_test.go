// SPDX-License-Identifier: Apache-2.0

// DoS-hardening tests (#4 head-of-line write stall, #12 viewer cap +
// per-IP handshake limit).
//
// #4 is exercised at the wsConn layer with a fake wsSocket whose WriteJSON
// can block or fail on demand — deterministic, no flaky real network peer.
// #12 is exercised both at the admission-gate layer (addViewer /
// tryReserveViewerSlot / acquireHandshakeSlot) and end-to-end through a real
// httptest server + WS dialer for the 503/429 status assertions.
package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// fakeSocket is a controllable wsSocket. WriteJSON blocks on blockCh (if
// set) and returns failErr (if set). It records the count of completed
// writes. Used to simulate a slow/stuck or dead consumer.
type fakeSocket struct {
	mu        sync.Mutex
	writes    int
	closed    bool
	blockCh   chan struct{} // if non-nil, WriteJSON blocks until it's closed/received
	failErr   error         // if non-nil, WriteJSON returns it (after any block)
	deadlines int
}

func (f *fakeSocket) WriteJSON(v any) error {
	if f.blockCh != nil {
		<-f.blockCh
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failErr != nil {
		return f.failErr
	}
	f.writes++
	return nil
}

func (f *fakeSocket) WriteControl(int, []byte, time.Time) error {
	if f.blockCh != nil {
		<-f.blockCh
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failErr != nil {
		return f.failErr
	}
	f.writes++
	return nil
}

func (f *fakeSocket) SetWriteDeadline(time.Time) error {
	f.mu.Lock()
	f.deadlines++
	f.mu.Unlock()
	return nil
}

func (f *fakeSocket) Close() error {
	f.mu.Lock()
	f.closed = true
	f.mu.Unlock()
	return nil
}

func (f *fakeSocket) writeCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.writes
}

func (f *fakeSocket) isClosed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closed
}

// ---- #4: per-connection write path / head-of-line ----

// A stuck consumer must NOT block another connection's writes nor the
// broadcast path. We register a stuck conn (its writer goroutine is parked
// inside WriteJSON) alongside a healthy one, then fan out via broadcastJSON
// and assert the healthy conn keeps receiving while the stuck one is
// eventually reaped on buffer overflow.
func TestBroadcast_StuckConsumerDoesNotBlockOthers(t *testing.T) {
	s := &Server{wsConns: make(map[*wsConn]struct{}), cfg: Config{Logger: slogDiscard()}}

	// Stuck conn: first write parks forever (until we release it).
	stuckSock := &fakeSocket{blockCh: make(chan struct{})}
	stuck := newWSConn(stuckSock, slogDiscard())
	s.registerConn(stuck)

	// Healthy conn.
	healthySock := &fakeSocket{}
	healthy := newWSConn(healthySock, slogDiscard())
	s.registerConn(healthy)

	// Broadcast more than the send buffer depth so the stuck conn's buffer
	// fills (its single writer goroutine is parked on the first message).
	// dropOnFull (cursor policy) means broadcastJSON must NEVER block here.
	done := make(chan struct{})
	go func() {
		for i := 0; i < wsSendBuffer*4; i++ {
			s.broadcastJSON(map[string]interface{}{"type": "cursor_pos", "seq": i})
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("broadcastJSON blocked on a stuck consumer (head-of-line stall not fixed)")
	}

	// The healthy conn must have drained its writes despite the stuck peer.
	waitFor(t, 2*time.Second, func() bool { return healthySock.writeCount() > 0 })

	// Release the stuck conn so the test goroutine can exit cleanly.
	close(stuckSock.blockCh)
	stuck.stop()
	healthy.stop()
}

// A high-rate (dropOnFull) producer never blocks even when a conn's buffer
// is saturated, and the conn is NOT killed (cursor frames are disposable).
func TestWriteJSON_DropOnFull_DoesNotBlockOrClose(t *testing.T) {
	sock := &fakeSocket{blockCh: make(chan struct{})}
	c := newWSConn(sock, slogDiscard())
	defer func() { close(sock.blockCh); c.stop() }()

	// Fill far past the buffer; every enqueue must return promptly.
	start := time.Now()
	dropped := 0
	for i := 0; i < wsSendBuffer*10; i++ {
		if !c.writeJSON(map[string]interface{}{"n": i}, dropOnFull) {
			dropped++
		}
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("dropOnFull enqueue blocked")
	}
	if dropped == 0 {
		t.Fatal("expected some messages dropped once buffer saturated")
	}
	if c.isClosedConn() {
		t.Fatal("dropOnFull must NOT close the conn")
	}
}

// On a full buffer with closeOnFull (control/signaling policy) the wedged
// conn is reaped so the dead peer is shed.
func TestWriteJSON_CloseOnFull_ReapsWedgedConn(t *testing.T) {
	sock := &fakeSocket{blockCh: make(chan struct{})}
	c := newWSConn(sock, slogDiscard())
	defer func() { close(sock.blockCh) }()

	// Saturate the buffer (writer goroutine parked on the first message).
	for i := 0; i < wsSendBuffer*4; i++ {
		c.writeJSON(map[string]interface{}{"n": i}, closeOnFull)
	}
	// closeOnFull must have signalled teardown immediately (the reap
	// signal). The writer goroutine may still be parked inside the blocked
	// WriteJSON until its deadline fires — in production SetWriteDeadline
	// bounds that to wsWriteDeadline — but the reap (stopCh) is immediate,
	// so producers/keepalive observe it at once.
	waitFor(t, 2*time.Second, func() bool { return c.isStopping() })
}

// A conn whose underlying socket write fails (stuck/dead socket) is reaped:
// the writer goroutine closes the socket and signals `closed`.
func TestWriter_WriteFailure_ReapsConn(t *testing.T) {
	sock := &fakeSocket{failErr: &timeoutErr{}}
	c := newWSConn(sock, slogDiscard())

	if !c.writeJSON(map[string]interface{}{"type": "stats"}, closeOnFull) {
		t.Fatal("first enqueue should succeed (buffer has room)")
	}
	// The writer goroutine attempts the write, gets failErr, closes the
	// socket and signals closed.
	waitFor(t, 2*time.Second, func() bool { return c.isClosedConn() })
	if !sock.isClosed() {
		t.Fatal("underlying socket should be Close()d after a write error")
	}
}

// The writer sets a write deadline before every actual socket write (#4: a
// stuck socket must error rather than hang the writer goroutine forever).
func TestWriter_SetsWriteDeadline(t *testing.T) {
	sock := &fakeSocket{}
	c := newWSConn(sock, slogDiscard())
	defer c.stop()
	c.writeJSON(map[string]interface{}{"x": 1}, closeOnFull)
	waitFor(t, 2*time.Second, func() bool {
		sock.mu.Lock()
		defer sock.mu.Unlock()
		return sock.deadlines >= 1 && sock.writes >= 1
	})
}

// ---- #12: MaxViewers cap ----

// addViewer is the authoritative admission gate: it returns nil once the cap
// is reached, and a slot freed by removeViewer admits a new viewer.
func TestAddViewer_MaxViewersCap(t *testing.T) {
	s := &Server{cfg: Config{MaxViewers: 3}}
	v1 := s.addViewer(newTestTrack(t))
	v2 := s.addViewer(newTestTrack(t))
	v3 := s.addViewer(newTestTrack(t))
	if v1 == nil || v2 == nil || v3 == nil {
		t.Fatal("first 3 viewers (cap=3) should all be admitted")
	}
	// 4th is rejected.
	if v4 := s.addViewer(newTestTrack(t)); v4 != nil {
		t.Fatal("4th viewer over cap=3 must be rejected (nil)")
	}
	// Free a slot; a new viewer is admitted.
	s.removeViewer(v2)
	if v := s.addViewer(newTestTrack(t)); v == nil {
		t.Fatal("after one disconnect, a new viewer should be admitted")
	}
}

func TestTryReserveViewerSlot_AtCap(t *testing.T) {
	s := &Server{cfg: Config{MaxViewers: 1}}
	ok, res := s.tryReserveViewerSlot()
	if !ok || res == nil {
		t.Fatal("empty session should admit")
	}
	// Commit the reservation into a viewer (consumes the reservation).
	if v := res.commit(newTestTrack(t)); v == nil {
		t.Fatal("commit of a reserved slot should succeed")
	}
	ok2, res2 := s.tryReserveViewerSlot()
	if ok2 {
		t.Fatal("at cap=1 with 1 viewer, the reserve check should reject")
	}
	if res2 != nil {
		t.Fatal("a rejected reservation must return a nil reservation")
	}
}

// A reservation that is taken but never committed (any error/early-return
// path before addViewer) must be released, freeing the slot for a later
// viewer. This pins the reserve/release balance the handler relies on.
func TestReserveViewerSlot_ReleaseFreesSlot(t *testing.T) {
	s := &Server{cfg: Config{MaxViewers: 1}}
	ok, res := s.tryReserveViewerSlot()
	if !ok || res == nil {
		t.Fatal("first reservation (cap=1) should succeed")
	}
	// While the reservation is held, a second reserve must be rejected: the
	// slot is accounted for even before any viewer is appended.
	if ok2, _ := s.tryReserveViewerSlot(); ok2 {
		t.Fatal("a held reservation must occupy the only slot")
	}
	// Simulate an error path (e.g. NewPeerConnection failed): release.
	res.release()
	// The slot is now free again.
	ok3, res3 := s.tryReserveViewerSlot()
	if !ok3 || res3 == nil {
		t.Fatal("after release, the freed slot should admit a new reservation")
	}
	// release is idempotent; a double release must not drive the counter
	// negative or free a slot it doesn't own.
	res.release()
	s.mu.Lock()
	got := s.reservedViewers
	s.mu.Unlock()
	if got != 1 {
		t.Fatalf("reservedViewers = %d after held reserve + double-release of a freed res, want 1", got)
	}
}

// commit and release share a single sync.Once: a commit followed by the
// handler's deferred release must NOT double-decrement (no leaked/negative
// slot), and the committed viewer is counted as a viewer, not a reservation.
func TestReserveViewerSlot_CommitThenReleaseIsNoop(t *testing.T) {
	s := &Server{cfg: Config{MaxViewers: 2}}
	ok, res := s.tryReserveViewerSlot()
	if !ok || res == nil {
		t.Fatal("reservation should succeed")
	}
	if v := res.commit(newTestTrack(t)); v == nil {
		t.Fatal("commit should admit")
	}
	// The deferred release in the handler runs after a successful commit; it
	// must be a no-op (the once was consumed by commit).
	res.release()
	s.mu.Lock()
	gotViewers := len(s.viewers)
	gotReserved := s.reservedViewers
	s.mu.Unlock()
	if gotViewers != 1 || gotReserved != 0 {
		t.Fatalf("after commit+release: viewers=%d reserved=%d, want 1 and 0", gotViewers, gotReserved)
	}
}

func TestMaxViewers_DefaultApplied(t *testing.T) {
	s := &Server{}
	if got := s.maxViewers(); got != DefaultMaxViewers {
		t.Fatalf("maxViewers default = %d, want %d", got, DefaultMaxViewers)
	}
}

// End-to-end: with cap=N, N viewers connect, the (N+1)th /ws upgrade is
// rejected with 503; after one disconnects, a new one succeeds.
func TestHandleWS_MaxViewers_503_E2E(t *testing.T) {
	const cap = 2
	s := &Server{
		wsConns:         make(map[*wsConn]struct{}),
		handshakesPerIP: make(map[string]int),
		cfg: Config{
			MaxViewers:         cap,
			MaxHandshakesPerIP: 100, // don't let the per-IP limit interfere
			Logger:             slogDiscard(),
			// no viewer pubkey + loopback bind => auth-off dev path
			BoundNonLoopback: false,
		},
	}
	srv := httptest.NewServer(http.HandlerFunc(s.handleWS))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	var conns []*websocket.Conn
	for i := 0; i < cap; i++ {
		c, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("viewer %d should connect: %v", i, err)
		}
		conns = append(conns, c)
	}
	// Give the handlers time to register their viewers (addViewer happens
	// after NewPeerConnection in the handler goroutine).
	waitFor(t, 3*time.Second, func() bool { return s.viewerCount() == cap })

	// (N+1)th must be rejected with 503.
	_, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err == nil {
		t.Fatal("(N+1)th viewer should be rejected")
	}
	if resp == nil || resp.StatusCode != http.StatusServiceUnavailable {
		code := 0
		if resp != nil {
			code = resp.StatusCode
		}
		t.Fatalf("(N+1)th upgrade status = %d, want 503", code)
	}

	// Disconnect one; a new viewer should now be admitted.
	conns[0].Close()
	waitFor(t, 3*time.Second, func() bool { return s.viewerCount() == cap-1 })
	c, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("after a disconnect, a new viewer should connect: %v", err)
	}
	conns = append(conns[1:], c)
	for _, c := range conns {
		c.Close()
	}
}

// Concurrent burst: a precise burst of more-than-cap simultaneous /ws dials
// must admit exactly `cap` and reject the rest with 503 PRE-UPGRADE (the
// reservation gate runs before wsUpgrader.Upgrade / NewPeerConnection, so a
// rejected dial never completes a WS handshake — proving no Pion peer was
// allocated for it). This is the audit #12 intent: the cap is authoritative
// before any peer allocation, even under a concurrent at-cap burst.
func TestHandleWS_MaxViewers_ConcurrentBurst_503PreUpgrade(t *testing.T) {
	const cap = 3
	const burst = 12
	s := &Server{
		wsConns:         make(map[*wsConn]struct{}),
		handshakesPerIP: make(map[string]int),
		cfg: Config{
			MaxViewers:         cap,
			MaxHandshakesPerIP: 1000, // keep the per-IP limit out of the way
			Logger:             slogDiscard(),
			BoundNonLoopback:   false,
		},
	}
	srv := httptest.NewServer(http.HandlerFunc(s.handleWS))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	type result struct {
		conn    *websocket.Conn
		status  int
		upgrade bool
	}
	results := make(chan result, burst)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < burst; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start // release all dials at once for a true concurrent burst
			c, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
			r := result{}
			if err == nil {
				r.conn = c
				r.upgrade = true
			}
			if resp != nil {
				r.status = resp.StatusCode
			}
			results <- r
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	var admitted []*websocket.Conn
	rejected503 := 0
	for r := range results {
		if r.upgrade {
			admitted = append(admitted, r.conn)
			continue
		}
		// A rejected dial must be a clean pre-upgrade 503 (no WS handshake
		// completed => no PeerConnection allocated).
		if r.status != http.StatusServiceUnavailable {
			t.Errorf("rejected dial status = %d, want 503 (pre-upgrade cap reject)", r.status)
		}
		rejected503++
	}
	defer func() {
		for _, c := range admitted {
			c.Close()
		}
	}()

	if len(admitted) != cap {
		t.Fatalf("admitted %d viewers under a burst of %d, want exactly cap=%d", len(admitted), burst, cap)
	}
	if rejected503 != burst-cap {
		t.Fatalf("rejected %d with 503, want %d", rejected503, burst-cap)
	}
	// The registered viewer count must settle at exactly cap (no over-admit,
	// no leaked reservation that would block a future viewer).
	waitFor(t, 3*time.Second, func() bool { return s.viewerCount() == cap })
	s.mu.Lock()
	reserved := s.reservedViewers
	s.mu.Unlock()
	if reserved != 0 {
		t.Fatalf("reservedViewers = %d after the burst settled, want 0 (no leaked reservation)", reserved)
	}
}

// ---- #12: per-IP handshake limit ----

func TestAcquireHandshakeSlot_PerIPCap(t *testing.T) {
	s := &Server{handshakesPerIP: make(map[string]int), cfg: Config{MaxHandshakesPerIP: 2}}
	ok1, rel1 := s.acquireHandshakeSlot("1.2.3.4")
	ok2, rel2 := s.acquireHandshakeSlot("1.2.3.4")
	if !ok1 || !ok2 {
		t.Fatal("first 2 handshakes from an IP (cap=2) should be admitted")
	}
	ok3, rel3 := s.acquireHandshakeSlot("1.2.3.4")
	if ok3 {
		t.Fatal("3rd concurrent handshake from the same IP (cap=2) must be rejected")
	}
	rel3() // no-op
	// A different IP is unaffected.
	okOther, relOther := s.acquireHandshakeSlot("5.6.7.8")
	if !okOther {
		t.Fatal("a different IP should not be limited by another IP's slots")
	}
	relOther()
	// Releasing a slot lets a new handshake through.
	rel1()
	ok4, rel4 := s.acquireHandshakeSlot("1.2.3.4")
	if !ok4 {
		t.Fatal("after releasing a slot, a new handshake should be admitted")
	}
	rel2()
	rel4()
	// All released: the map entry should be cleaned up (no leak).
	s.handshakeMu.Lock()
	_, present := s.handshakesPerIP["1.2.3.4"]
	s.handshakeMu.Unlock()
	if present {
		t.Fatal("handshake counter for an IP should be deleted once it hits zero")
	}
}

// End-to-end: excess concurrent handshakes from one IP are rejected with
// 429. We hold connections open (each holds a handshake slot for its whole
// lifetime) and assert the over-limit dial gets 429. The MaxViewers cap is
// set high so it's the per-IP limit that trips first.
func TestHandleWS_PerIPHandshakeLimit_429_E2E(t *testing.T) {
	const ipCap = 2
	s := &Server{
		wsConns:         make(map[*wsConn]struct{}),
		handshakesPerIP: make(map[string]int),
		cfg: Config{
			MaxViewers:         100,
			MaxHandshakesPerIP: ipCap,
			Logger:             slogDiscard(),
			BoundNonLoopback:   false,
		},
	}
	srv := httptest.NewServer(http.HandlerFunc(s.handleWS))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	var conns []*websocket.Conn
	for i := 0; i < ipCap; i++ {
		c, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("handshake %d should connect: %v", i, err)
		}
		conns = append(conns, c)
	}
	waitFor(t, 3*time.Second, func() bool { return s.viewerCount() == ipCap })

	_, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err == nil {
		t.Fatal("over-limit handshake from same IP should be rejected")
	}
	if resp == nil || resp.StatusCode != http.StatusTooManyRequests {
		code := 0
		if resp != nil {
			code = resp.StatusCode
		}
		t.Fatalf("over-limit handshake status = %d, want 429", code)
	}
	for _, c := range conns {
		c.Close()
	}
}

// ---- helpers ----

// isClosedConn reports whether the wsConn's writer goroutine has exited.
func (w *wsConn) isClosedConn() bool {
	select {
	case <-w.closed:
		return true
	default:
		return false
	}
}

// isStopping reports whether teardown has been signalled (stop() called or a
// closeOnFull reap). This is the immediate reap signal; the writer goroutine
// exit (closed) may lag if a socket write is mid-flight.
func (w *wsConn) isStopping() bool {
	select {
	case <-w.stopCh:
		return true
	default:
		return false
	}
}

// timeoutErr is a net.Error-shaped error used to stand in for a write
// deadline expiry.
type timeoutErr struct{}

func (*timeoutErr) Error() string   { return "i/o timeout" }
func (*timeoutErr) Timeout() bool   { return true }
func (*timeoutErr) Temporary() bool { return true }

// waitFor polls cond until it's true or the timeout elapses.
func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !cond() {
		t.Fatalf("condition not met within %v", timeout)
	}
}
