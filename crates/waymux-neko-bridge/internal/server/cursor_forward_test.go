// SPDX-License-Identifier: Apache-2.0

package server

import (
	"testing"

	sock "github.com/waymuxhq/waymux/crates/waymux-neko-bridge/internal/socket"
)

func TestCursorImageToWSJSON(t *testing.T) {
	// 1x1 RGBA cursor, hotspot (0,0): w=1,h=1,hx=0,hy=0 + 4 rgba bytes
	payload := []byte{1, 0, 1, 0, 0, 0, 0, 0, 9, 8, 7, 6}
	msg := cursorImageToWS(sock.Frame{Tag: sock.TagCursorImage, Payload: payload})
	if msg["type"] != "cursor_image" {
		t.Fatalf("bad type: %v", msg["type"])
	}
	if msg["w"] != uint16(1) || msg["h"] != uint16(1) {
		t.Fatalf("bad dims: %+v", msg)
	}
}

func TestCursorPosToWSJSON(t *testing.T) {
	// x=1.0 (0x3F800000 LE), y=2.0 (0x40000000 LE), seq=3
	payload := []byte{0, 0, 128, 63, 0, 0, 0, 64, 3, 0, 0, 0}
	msg := cursorPosToWS(sock.Frame{Tag: sock.TagCursorPos, Payload: payload})
	if msg["type"] != "cursor_pos" || msg["seq"] != uint32(3) {
		t.Fatalf("bad pos: %+v", msg)
	}
	if msg["x"] != float32(1.0) || msg["y"] != float32(2.0) {
		t.Fatalf("bad coords: %+v", msg)
	}
}
