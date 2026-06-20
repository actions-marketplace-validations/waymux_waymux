# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `record start --codec hevc-vulkan-lossless` now fails fast with a clear error
  when the GPU does not expose Vulkan-video HEVC 4:4:4 (Hi444) encode caps
  (e.g. integrated AMD parts), instead of starting a recording that silently
  writes no file. Use `ffv1-vulkan` for portable GPU-lossless capture.

## [0.2.0]

The first public, open-source release of the waymux CLI, daemon, session
compositor, and MCP server (Apache-2.0).

### Added

- **CLI machine surface (Phase 1).** A stable `--json` envelope on every verb:
  success is `{"ok":true,"verb":"<verb>","data":{...}}`, failure is
  `{"ok":false,"verb":"<verb>","error":{"code":"E_...","message":"..."}}` with a
  non-zero exit, and the streaming verbs (`events`, `logs`) emit
  newline-delimited JSON. Screenshots return the PNG base64 in `data.png_b64`.
- New CLI verbs: `tag` plus `windows --tag` filtering, `record status`
  (symmetric with `viewer status`), and a native `viewer token` mint.
- Batched `inject` (one round trip per op batch) and implemented touch
  injection (`InjectTouch`). `InjectSelector` remains a documented reserved
  slot; resolve the target with `windows` / `wait` and an explicit `window_id`.
- Single-command onboarding: `waymux serve` runs the daemon from the one binary,
  and local verbs auto-spawn a background `waymuxd` when its socket is absent
  (opt out with `WAYMUX_NO_AUTOSPAWN`). The neko-bridge binary is auto-located.
- **MCP server (Phase 2).** `waymux-mcp` exposes every discrete request/response
  CLI verb to AI agents over the Model Context Protocol by executing the CLI
  through an argument vector (no shell string, so no command injection). The
  streaming verbs (`events`, `logs`) and the credential-writing `login` are
  intentionally excluded. Contract tests pin the MCP tool set to the CLI verbs,
  and a regression test locks in the no-shell guarantee.

### Changed

- niri (Smithay) is validated as a nested inner compositor (hardware-rendered;
  on AMD it needs `AMD_DEBUG=nodcc` / `RADV_DEBUG=nodcc`, the same untiled-output
  workaround KWin uses), alongside KWin / Plasma 6 and direct Wayland clients.

### Fixed

- End-to-end triage fixes from validating the recording and live-viewer paths
  with a nested compositor.

[Unreleased]: https://github.com/waymux/waymux/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/waymux/waymux/releases/tag/v0.2.0
