// SPDX-License-Identifier: Apache-2.0

package server

import (
	"testing"

	"github.com/pion/webrtc/v4"
)

// Pin the multi-viewer input-control contract (last-wins).
//
// - Newest connect = primary (demotes any prior primary)
// - Primary disconnect with others present = newest remaining auto-promotes
// - Observer disconnect = no change
// - Last viewer disconnect = empty slate, next connect is primary again

func newTestTrack(t *testing.T) *webrtc.TrackLocalStaticSample {
	t.Helper()
	tr, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeH264, ClockRate: 90000},
		"video", "test",
	)
	if err != nil {
		t.Fatalf("NewTrackLocalStaticSample: %v", err)
	}
	return tr
}

func TestAddViewer_FirstIsPrimary(t *testing.T) {
	s := &Server{}
	v := s.addViewer(newTestTrack(t))
	if !v.primary.Load() {
		t.Fatalf("first viewer should be primary, got observer")
	}
	if got := s.viewerCount(); got != 1 {
		t.Fatalf("viewerCount = %d, want 1", got)
	}
}

func TestAddViewer_NewestIsPrimary(t *testing.T) {
	s := &Server{}
	v1 := s.addViewer(newTestTrack(t))
	v2 := s.addViewer(newTestTrack(t))
	if !v2.primary.Load() {
		t.Fatalf("newest viewer should be primary (last-wins), got observer")
	}
	if v1.primary.Load() {
		t.Fatalf("prior viewer should be demoted to observer when a newer one joins")
	}
	if got := s.viewerCount(); got != 2 {
		t.Fatalf("viewerCount = %d, want 2", got)
	}
}

func TestRemoveViewer_PrimaryDisconnect_PromotesNewestRemaining(t *testing.T) {
	s := &Server{}
	v1 := s.addViewer(newTestTrack(t))
	v2 := s.addViewer(newTestTrack(t))
	v3 := s.addViewer(newTestTrack(t))
	// Last-wins: the newest (v3) holds primary; v1/v2 are observers.
	if v1.primary.Load() || v2.primary.Load() || !v3.primary.Load() {
		t.Fatalf("initial: v1 primary=%t v2 primary=%t v3 primary=%t; want false false true",
			v1.primary.Load(), v2.primary.Load(), v3.primary.Load())
	}
	// Primary (v3) leaves — v2 (newest remaining) should auto-promote.
	remaining := s.removeViewer(v3)
	if remaining != 2 {
		t.Fatalf("remaining = %d, want 2", remaining)
	}
	if !v2.primary.Load() {
		t.Fatalf("v2 (newest remaining) should be promoted to primary after v3 left")
	}
	if v1.primary.Load() {
		t.Fatalf("v1 should still be observer; only the newest remaining promotes")
	}
}

func TestRemoveViewer_ObserverDisconnect_NoPromotion(t *testing.T) {
	s := &Server{}
	v1 := s.addViewer(newTestTrack(t)) // observer after v2 joins
	v2 := s.addViewer(newTestTrack(t)) // primary (last-wins)
	remaining := s.removeViewer(v1)    // an observer leaves
	if remaining != 1 {
		t.Fatalf("remaining = %d, want 1", remaining)
	}
	if !v2.primary.Load() {
		t.Fatalf("v2 should still be primary (an observer left, no change)")
	}
}

func TestRemoveViewer_LastViewerLeaves_NextConnectIsPrimary(t *testing.T) {
	s := &Server{}
	v1 := s.addViewer(newTestTrack(t))
	_ = s.removeViewer(v1)
	if got := s.viewerCount(); got != 0 {
		t.Fatalf("viewerCount after removing last = %d, want 0", got)
	}
	v2 := s.addViewer(newTestTrack(t))
	if !v2.primary.Load() {
		t.Fatalf("fresh connect after empty slate should be primary")
	}
}

func TestSnapshotTracks_ReturnsAllInOrder(t *testing.T) {
	s := &Server{}
	t1 := newTestTrack(t)
	t2 := newTestTrack(t)
	t3 := newTestTrack(t)
	_ = s.addViewer(t1)
	_ = s.addViewer(t2)
	_ = s.addViewer(t3)
	tracks := s.snapshotTracks()
	if len(tracks) != 3 {
		t.Fatalf("snapshot len = %d, want 3", len(tracks))
	}
	if tracks[0] != t1 || tracks[1] != t2 || tracks[2] != t3 {
		t.Fatalf("snapshot order wrong; want [t1,t2,t3]")
	}
}
