// SPDX-License-Identifier: Apache-2.0

package socket

import (
	"bytes"
	"testing"
)

func TestNaluRoundtrip(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x01, 0x67, 0x42}
	var buf bytes.Buffer
	if err := WriteFrame(&buf, Frame{Tag: TagNalu, Payload: payload}); err != nil {
		t.Fatal(err)
	}
	f, err := ReadFrame(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if f.Tag != TagNalu {
		t.Errorf("want tag %d got %d", TagNalu, f.Tag)
	}
	if !bytes.Equal(f.Payload, payload) {
		t.Errorf("payload mismatch")
	}
}

func TestForceKeyframeRoundtrip(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteFrame(&buf, Frame{Tag: TagForceKeyframe}); err != nil {
		t.Fatal(err)
	}
	got := buf.Bytes()
	if got[0] != TagForceKeyframe || got[1] != 0 || got[2] != 0 || got[3] != 0 || got[4] != 0 {
		t.Errorf("header bytes wrong: %v", got)
	}
}

func TestUnknownTagRejected(t *testing.T) {
	buf := bytes.NewReader([]byte{0xFF, 0, 0, 0, 0})
	if _, err := ReadFrame(buf); err == nil {
		t.Error("expected error for unknown tag")
	}
}

func TestInjectOpRoundtrip(t *testing.T) {
	payload := []byte(`{"kind":"key","params":{"keycode":30,"state":"pressed"}}`)
	var buf bytes.Buffer
	if err := WriteFrame(&buf, Frame{Tag: TagInjectOp, Payload: payload}); err != nil {
		t.Fatal(err)
	}
	f, err := ReadFrame(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(f.Payload, payload) {
		t.Errorf("payload mismatch: got %v want %v", f.Payload, payload)
	}
}

func TestCursorTagsRoundTrip(t *testing.T) {
	for _, tag := range []uint8{TagCursorImage, TagCursorPos} {
		var buf bytes.Buffer
		if err := WriteFrame(&buf, Frame{Tag: tag, Payload: []byte{1, 2, 3, 4}}); err != nil {
			t.Fatalf("write: %v", err)
		}
		f, err := ReadFrame(&buf)
		if err != nil {
			t.Fatalf("read tag 0x%02X: %v", tag, err)
		}
		if f.Tag != tag || !bytes.Equal(f.Payload, []byte{1, 2, 3, 4}) {
			t.Fatalf("roundtrip mismatch: %+v", f)
		}
	}
}
