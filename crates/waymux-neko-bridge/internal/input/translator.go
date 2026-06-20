// SPDX-License-Identifier: Apache-2.0

// Package input translates neko web-client events to waymux InjectOp JSON.
// The output is the exact wire shape consumed by
// `waymux inject --ops '[{...}]'` and SessionCtlMethod::InjectBatch.
package input

import (
	"encoding/json"
	"fmt"
)

type nekoEvent struct {
	Event   string  `json:"event"`
	KeyCode int     `json:"keyCode,omitempty"`
	X       float64 `json:"x,omitempty"`
	Y       float64 `json:"y,omitempty"`
	Button  int     `json:"button,omitempty"`
	DeltaX  float64 `json:"deltaX,omitempty"`
	DeltaY  float64 `json:"deltaY,omitempty"`
	// XKB modifier bitmask (Shift=1, Ctrl=4, Alt=8, Meta=64). Optional;
	// older viewers omit it and the daemon decodes the missing field as 0,
	// preserving today's modifier-less behaviour. Used by the mobile
	// modifier-strip so e.g. tapping "Ctrl" then "C" emits
	// `keydown KeyC modifiers=4`.
	Modifiers uint32 `json:"modifiers,omitempty"`
	// Monotonic browser input sequence number for cursor-overlay latency
	// display. Absent in older viewers (decoded as 0 by the daemon's
	// #[serde(default)]).
	Seq uint32 `json:"seq,omitempty"`
}

type injectOp struct {
	Kind   string                 `json:"kind"`
	Params map[string]interface{} `json:"params"`
}

// nekoButtonToEvdev maps a browser MouseEvent.button value (0-4) to the
// corresponding Linux input-event-code. Browser 0=left, 1=middle, 2=right,
// 3=back, 4=forward. Wayland wl_pointer.button expects BTN_* codes from
// <linux/input-event-codes.h>: BTN_LEFT=0x110, BTN_RIGHT=0x111,
// BTN_MIDDLE=0x112, BTN_SIDE=0x113, BTN_EXTRA=0x114.
func nekoButtonToEvdev(b int) (int, bool) {
	switch b {
	case 0:
		return 0x110, true // BTN_LEFT  (272)
	case 1:
		return 0x112, true // BTN_MIDDLE (274)
	case 2:
		return 0x111, true // BTN_RIGHT (273)
	case 3:
		return 0x113, true // BTN_SIDE  (275)
	case 4:
		return 0x114, true // BTN_EXTRA (276)
	default:
		return 0, false
	}
}

func Translate(raw []byte) ([]byte, error) {
	var ev nekoEvent
	if err := json.Unmarshal(raw, &ev); err != nil {
		return nil, fmt.Errorf("parsing neko event: %w", err)
	}
	var op injectOp
	switch ev.Event {
	case "keydown":
		op = injectOp{Kind: "key", Params: map[string]interface{}{
			"keycode":   ev.KeyCode,
			"state":     "pressed",
			"modifiers": ev.Modifiers,
		}}
	case "keyup":
		op = injectOp{Kind: "key", Params: map[string]interface{}{
			"keycode":   ev.KeyCode,
			"state":     "released",
			"modifiers": ev.Modifiers,
		}}
	case "mousemove":
		op = injectOp{Kind: "pointer", Params: map[string]interface{}{
			"x":      ev.X,
			"y":      ev.Y,
			"state":  "released",
			"button": 0,
			"seq":    ev.Seq,
		}}
	case "mousedown":
		evdev, ok := nekoButtonToEvdev(ev.Button)
		if !ok {
			return nil, fmt.Errorf("mousedown: unknown browser button %d", ev.Button)
		}
		op = injectOp{Kind: "pointer", Params: map[string]interface{}{
			"x":      ev.X,
			"y":      ev.Y,
			"state":  "pressed",
			"button": evdev,
		}}
	case "mouseup":
		evdev, ok := nekoButtonToEvdev(ev.Button)
		if !ok {
			return nil, fmt.Errorf("mouseup: unknown browser button %d", ev.Button)
		}
		op = injectOp{Kind: "pointer", Params: map[string]interface{}{
			"x":      ev.X,
			"y":      ev.Y,
			"state":  "released",
			"button": evdev,
		}}
	case "wheel":
		op = injectOp{Kind: "pointer", Params: map[string]interface{}{
			"x":      ev.X,
			"y":      ev.Y,
			"state":  "released",
			"button": 0,
			"axis_x": ev.DeltaX,
			"axis_y": ev.DeltaY,
		}}
	default:
		return nil, fmt.Errorf("unknown neko event: %s", ev.Event)
	}
	return json.Marshal(op)
}
