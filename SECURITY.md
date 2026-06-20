# Security Policy

waymux runs entirely on your machine: there is no account, no telemetry, and no
phone-home. The local trust boundary is a per-user Unix socket gated by
`SO_PEERCRED` (same-uid only). Network exposure is opt-in (binding the viewer to
a LAN address, which is fail-closed without a signed viewer token, or `login` /
`--remote` against a server you name). We take security seriously and welcome
responsible disclosure.

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

waymux is an early public release. Security fixes land on the latest 0.2.x
line.

## Reporting a vulnerability

Please do not open a public issue, pull request, or discussion for a security
vulnerability. Instead, use one of these private channels:

- Open a private security advisory on GitHub:
  <https://github.com/waymux/waymux/security/advisories/new>.
- Or email **security@waymux.cloud**.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce, or a proof of concept.
- The affected version or commit, and your environment (OS, GPU, compositor) if
  relevant.

## Response window

- We aim to acknowledge a report within **3 business days**.
- We aim to provide an initial assessment within **7 business days**.
- We will keep you informed as we work on a fix, and will coordinate a
  disclosure timeline with you. Please give us reasonable time to release a fix
  before any public disclosure.

We appreciate your effort to disclose responsibly and will credit reporters who
wish to be acknowledged.
