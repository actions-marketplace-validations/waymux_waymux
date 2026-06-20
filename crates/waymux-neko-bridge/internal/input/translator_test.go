// SPDX-License-Identifier: Apache-2.0

package input

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// neko's web client sends events like:
//
//	{"event":"keydown","key":"a","code":"KeyA","keyCode":65}
//	{"event":"mousemove","x":120,"y":80}
//	{"event":"mousedown","x":120,"y":80,"button":0}
//
// We translate those to waymux InjectOp JSON:
//
//	{"kind":"key","params":{"keycode":65,"state":"pressed","modifiers":0}}
//	{"kind":"pointer","params":{"x":120,"y":80,"state":"released","button":0}}
//	{"kind":"pointer","params":{"x":120,"y":80,"state":"pressed","button":272}}
func TestKeyDownTranslates(t *testing.T) {
	raw := []byte(`{"event":"keydown","keyCode":65}`)
	out, err := Translate(raw)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	json.Unmarshal(out, &got)
	if got["kind"] != "key" {
		t.Errorf("kind = %v", got["kind"])
	}
	p := got["params"].(map[string]any)
	if int(p["keycode"].(float64)) != 65 {
		t.Errorf("keycode = %v", p["keycode"])
	}
	if p["state"] != "pressed" {
		t.Errorf("state = %v", p["state"])
	}
}

func TestMouseMoveTranslates(t *testing.T) {
	raw := []byte(`{"event":"mousemove","x":120,"y":80}`)
	out, err := Translate(raw)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	json.Unmarshal(out, &got)
	if got["kind"] != "pointer" {
		t.Errorf("kind = %v", got["kind"])
	}
}

func TestMouseButtonCodesMatchEvdev(t *testing.T) {
	cases := []struct{ browserBtn, evdev int }{
		{0, 272}, {1, 274}, {2, 273}, {3, 275}, {4, 276},
	}
	for _, c := range cases {
		raw := []byte(fmt.Sprintf(`{"event":"mousedown","x":1,"y":1,"button":%d}`, c.browserBtn))
		out, err := Translate(raw)
		if err != nil {
			t.Fatal(err)
		}
		var got map[string]any
		json.Unmarshal(out, &got)
		p := got["params"].(map[string]any)
		if int(p["button"].(float64)) != c.evdev {
			t.Errorf("browserBtn=%d: got evdev=%v want %d", c.browserBtn, p["button"], c.evdev)
		}
	}
}

func TestMouseButtonUnknownReturnsError(t *testing.T) {
	if _, err := Translate([]byte(`{"event":"mousedown","button":99}`)); err == nil {
		t.Error("expected error for unknown button")
	}
}

// The mobile soft-keyboard's modifier-strip (e.g. tapping Ctrl then
// "c") sends a single keydown with `modifiers` set, then a matching
// keyup with the same modifiers cleared on the wire. Verify the
// optional field threads through unchanged.
func TestKeyDownPassesModifiers(t *testing.T) {
	raw := []byte(`{"event":"keydown","keyCode":46,"modifiers":4}`)
	out, err := Translate(raw)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	json.Unmarshal(out, &got)
	p := got["params"].(map[string]any)
	if int(p["modifiers"].(float64)) != 4 {
		t.Errorf("modifiers = %v want 4", p["modifiers"])
	}
}

func TestUnknownEventReturnsError(t *testing.T) {
	if _, err := Translate([]byte(`{"event":"flarble"}`)); err == nil {
		t.Error("expected error for unknown event")
	}
}

// Thread the browser input seq through mousemove so the session can
// record CursorPos{seq} for the viewer overlay latency display.
func TestMousemoveCarriesSeq(t *testing.T) {
	raw := []byte(`{"event":"mousemove","x":10,"y":20,"seq":5}`)
	op, err := Translate(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(op), `"seq":5`) {
		t.Fatalf("seq missing or wrong in output: %s", op)
	}
}
