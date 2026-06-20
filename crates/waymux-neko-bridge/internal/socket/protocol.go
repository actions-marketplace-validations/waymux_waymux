// SPDX-License-Identifier: Apache-2.0

// Package socket implements the typed Unix-socket wire format between
// waymux-session (Rust) and the neko-bridge. Mirrors
// crates/waymux-session/src/viewer/protocol.rs.
package socket

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

const (
	TagNalu          uint8 = 0x01
	TagPtsHint       uint8 = 0x02
	TagCursorImage   uint8 = 0x03
	TagCursorPos     uint8 = 0x04
	TagForceKeyframe uint8 = 0x10
	TagInjectOp      uint8 = 0x11
	TagSetBitrate    uint8 = 0x12
	TagShutdown      uint8 = 0xF0

	MaxPayloadBytes = 1 << 20 // 1 MiB
)

type Frame struct {
	Tag     uint8
	Payload []byte
}

func WriteFrame(w io.Writer, f Frame) error {
	if len(f.Payload) > MaxPayloadBytes {
		return fmt.Errorf("payload %d exceeds %d", len(f.Payload), MaxPayloadBytes)
	}
	header := make([]byte, 5)
	header[0] = f.Tag
	binary.LittleEndian.PutUint32(header[1:], uint32(len(f.Payload)))
	if _, err := w.Write(header); err != nil {
		return err
	}
	if len(f.Payload) > 0 {
		if _, err := w.Write(f.Payload); err != nil {
			return err
		}
	}
	return nil
}

func ReadFrame(r io.Reader) (Frame, error) {
	var hdr [5]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return Frame{}, err
	}
	tag := hdr[0]
	switch tag {
	case TagNalu, TagPtsHint, TagCursorImage, TagCursorPos, TagForceKeyframe, TagInjectOp, TagSetBitrate, TagShutdown:
		// ok
	default:
		return Frame{}, fmt.Errorf("unknown tag 0x%02X", tag)
	}
	length := binary.LittleEndian.Uint32(hdr[1:])
	if length > MaxPayloadBytes {
		return Frame{}, fmt.Errorf("payload %d exceeds %d", length, MaxPayloadBytes)
	}
	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return Frame{}, fmt.Errorf("reading payload: %w", err)
	}
	return Frame{Tag: tag, Payload: payload}, nil
}

var ErrShutdown = errors.New("peer requested shutdown")
