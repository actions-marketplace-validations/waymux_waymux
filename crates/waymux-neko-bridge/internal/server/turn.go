// SPDX-License-Identifier: Apache-2.0

package server

import (
	"github.com/pion/webrtc/v4"
)

// iceServerJSON mirrors the browser's RTCIceServer dictionary so we can
// JSON-marshal it into the WS `config` message that primes the browser
// before it constructs its RTCPeerConnection.
//
// Camel-case JSON tags (not Go's natural CamelCase) match the WebRTC
// JS API exactly: the browser passes `msg.iceServers` straight into
// `new RTCPeerConnection({ iceServers: ... })`.
type iceServerJSON struct {
	URLs       []string `json:"urls"`
	Username   string   `json:"username,omitempty"`
	Credential string   `json:"credential,omitempty"`
}

// buildICEServers turns the per-process config (env-derived in main.go)
// into the JSON-marshalled list we hand the browser and the Pion
// PeerConnection.
//
// Returns (jsonForBrowser, goForPion). Both share the same ICE servers
// but live in different types: the browser wants the dictionary shape
// above; Pion wants []webrtc.ICEServer.
//
// TURN credentials are not minted here from a fleet-wide shared secret.
// The control plane mints per-session creds and delivers them as
// WAYMUX_TURN_USERNAME / WAYMUX_TURN_CREDENTIAL; this function forwards
// them verbatim.
//
// Empty config returns (nil, nil): neko-bridge then runs in LAN-only
// mode exactly as it did before TURN existed. A TURN url present WITHOUT
// both a username and a credential is dropped (STUN-only fallback): an
// unauthenticated TURN entry would only fail auth at the server.
func buildICEServers(cfg Config) ([]iceServerJSON, []webrtc.ICEServer) {
	if len(cfg.STUNServers) == 0 && len(cfg.TURNServers) == 0 {
		return nil, nil
	}

	var (
		browser []iceServerJSON
		pion    []webrtc.ICEServer
	)

	for _, u := range cfg.STUNServers {
		if u == "" {
			continue
		}
		browser = append(browser, iceServerJSON{URLs: []string{u}})
		pion = append(pion, webrtc.ICEServer{URLs: []string{u}})
	}

	if len(cfg.TURNServers) > 0 && cfg.TURNUsername != "" && cfg.TURNCredential != "" {
		for _, u := range cfg.TURNServers {
			if u == "" {
				continue
			}
			browser = append(browser, iceServerJSON{
				URLs:       []string{u},
				Username:   cfg.TURNUsername,
				Credential: cfg.TURNCredential,
			})
			pion = append(pion, webrtc.ICEServer{
				URLs:       []string{u},
				Username:   cfg.TURNUsername,
				Credential: cfg.TURNCredential,
			})
		}
	}

	return browser, pion
}
