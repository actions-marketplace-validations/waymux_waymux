// SPDX-License-Identifier: Apache-2.0

package server

import (
	"testing"
)

func TestBuildICEServers_EmptyConfig(t *testing.T) {
	cfg := Config{}
	browser, pion := buildICEServers(cfg)
	if browser != nil || pion != nil {
		t.Errorf("empty config should yield nil/nil; got %v / %v", browser, pion)
	}
}

func TestBuildICEServers_STUNOnly(t *testing.T) {
	cfg := Config{
		STUNServers: []string{"stun:stun.l.google.com:19302"},
	}
	browser, pion := buildICEServers(cfg)
	if len(browser) != 1 || len(pion) != 1 {
		t.Fatalf("want 1 entry each; got browser=%d pion=%d", len(browser), len(pion))
	}
	if browser[0].Username != "" || browser[0].Credential != "" {
		t.Error("STUN entry must not carry creds")
	}
}

// TURN credentials are provided by the control plane and
// forwarded verbatim. Verify they appear unchanged in both the browser
// JSON and the Pion config.
func TestBuildICEServers_TURNUsesProvidedCreds(t *testing.T) {
	cfg := Config{
		STUNServers:    []string{"stun:stun.l.google.com:19302"},
		TURNServers:    []string{"turns:turn.example.com:5349"},
		TURNUsername:   "1700003600:session-abc",
		TURNCredential: "provided-credential-xyz",
	}
	browser, pion := buildICEServers(cfg)
	if len(browser) != 2 || len(pion) != 2 {
		t.Fatalf("want 2 entries each; got browser=%d pion=%d", len(browser), len(pion))
	}
	// STUN entry first, no creds.
	if browser[0].Username != "" || browser[0].Credential != "" {
		t.Errorf("entry 0 (STUN) must not carry creds; got user=%q cred=%q",
			browser[0].Username, browser[0].Credential)
	}
	// TURN entry second, creds forwarded VERBATIM.
	if browser[1].Username != "1700003600:session-abc" {
		t.Errorf("browser TURN username = %q; want verbatim %q",
			browser[1].Username, "1700003600:session-abc")
	}
	if browser[1].Credential != "provided-credential-xyz" {
		t.Errorf("browser TURN credential = %q; want verbatim %q",
			browser[1].Credential, "provided-credential-xyz")
	}
	// And the same on the Pion side.
	if pion[1].Username != "1700003600:session-abc" {
		t.Errorf("pion TURN username = %q; want verbatim %q",
			pion[1].Username, "1700003600:session-abc")
	}
	cred, ok := pion[1].Credential.(string)
	if !ok || cred != "provided-credential-xyz" {
		t.Errorf("pion TURN credential = %v; want verbatim %q",
			pion[1].Credential, "provided-credential-xyz")
	}
}

// A TURN url with missing creds must NOT emit a TURN entry (it would
// only fail auth at the server); we fall back to STUN-only.
func TestBuildICEServers_TURNWithoutCredsIsDropped(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{
			name: "both creds missing",
			cfg: Config{
				STUNServers: []string{"stun:stun.l.google.com:19302"},
				TURNServers: []string{"turns:turn.example.com:5349"},
			},
		},
		{
			name: "username only",
			cfg: Config{
				STUNServers:  []string{"stun:stun.l.google.com:19302"},
				TURNServers:  []string{"turns:turn.example.com:5349"},
				TURNUsername: "1700003600:session-abc",
			},
		},
		{
			name: "credential only",
			cfg: Config{
				STUNServers:    []string{"stun:stun.l.google.com:19302"},
				TURNServers:    []string{"turns:turn.example.com:5349"},
				TURNCredential: "provided-credential-xyz",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			browser, pion := buildICEServers(tc.cfg)
			// STUN survives; TURN is dropped → exactly 1 entry each.
			if len(browser) != 1 || len(pion) != 1 {
				t.Fatalf("want STUN-only (1 each); got browser=%d pion=%d", len(browser), len(pion))
			}
			if browser[0].Username != "" || browser[0].Credential != "" {
				t.Errorf("remaining entry must be the STUN entry (no creds); got user=%q cred=%q",
					browser[0].Username, browser[0].Credential)
			}
		})
	}
}

// TURN creds set but no TURN url: creds have no effect, no TURN entry.
func TestBuildICEServers_NoTURNURL(t *testing.T) {
	cfg := Config{
		STUNServers:    []string{"stun:stun.l.google.com:19302"},
		TURNUsername:   "1700003600:session-abc",
		TURNCredential: "provided-credential-xyz",
	}
	browser, pion := buildICEServers(cfg)
	if len(browser) != 1 || len(pion) != 1 {
		t.Fatalf("want STUN-only (1 each); got browser=%d pion=%d", len(browser), len(pion))
	}
	if browser[0].Username != "" {
		t.Errorf("no-TURN-url should leave only the STUN entry; got user=%q", browser[0].Username)
	}
}

func TestBuildICEServers_EmptyURLsFiltered(t *testing.T) {
	cfg := Config{
		STUNServers: []string{"", "stun:stun.l.google.com:19302", ""},
	}
	browser, _ := buildICEServers(cfg)
	if len(browser) != 1 {
		t.Errorf("empty URL strings should be filtered out; got %d entries", len(browser))
	}
}
