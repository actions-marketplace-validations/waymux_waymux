# waymux-protocol

The wire protocol for [waymux](https://github.com/waymux/waymux): the typed
request / response / event messages and the framing codec used between the
`waymux` CLI, the `waymuxd` control daemon, and the per-session compositors,
spoken over a local Unix socket.

This crate is the single source of truth for the control-plane wire format
(MessagePack payloads, length-prefixed frames) and for the stable `E_*`
error-code set the CLI surfaces in its `--json` output.

Part of the waymux workspace. Apache-2.0 licensed. See the
[main repository](https://github.com/waymux/waymux) for the full project,
architecture, and roadmap.
