# Contributing to waymux

Thanks for your interest in waymux. Issues and pull requests are welcome. This
guide covers how to build, how to run the same checks CI runs, and the
conventions we follow.

## Building

waymux is a single Cargo workspace of Rust crates plus one Go module (the web
viewer).

Toolchains:

- Rust 1.88 or newer. `rust-toolchain.toml` pins the channel to `stable`, so a
  fresh clone fetches a current compiler. 1.88 is the MSRV (minimum supported
  Rust version), enforced in CI.
- Go 1.26 or newer, only to build the `waymux-neko-bridge` WebRTC viewer.

System libraries (Debian/Ubuntu package names; install the equivalents on your
distro):

```sh
sudo apt-get install -y \
  libwayland-dev libgbm-dev libvulkan-dev libxkbcommon-dev \
  libavutil-dev libavformat-dev libavcodec-dev libavfilter-dev \
  libavdevice-dev libswscale-dev libswresample-dev \
  pkg-config build-essential
```

ffmpeg 6.1+ provides the FFV1 and basic H.264 hardware paths. The lossless
Vulkan codecs need newer ffmpeg (`hevc-vulkan-lossless` requires ffmpeg 8.0).

Build everything:

```sh
# Rust binaries: daemon, session, CLI, attach client, MCP server.
cargo build --release

# The Go WebRTC viewer.
( cd crates/waymux-neko-bridge && go build -o waymux-neko-bridge . )
```

## Running the CI gate locally

CI must be green before a pull request merges. You can run the same gate
locally before you push:

```sh
# Formatting (must be clean).
cargo fmt --all -- --check

# Lints (warnings are errors).
cargo clippy --workspace --all-targets -- -D warnings

# Tests.
cargo test --workspace --all-targets

# Dependency and license policy.
cargo install --locked cargo-deny   # one time
cargo deny check

# The Go viewer builds and vets cleanly.
( cd crates/waymux-neko-bridge && go build -o waymux-neko-bridge . && go vet ./... )
```

CI additionally runs a `cargo publish --dry-run` on the publishable library
crates, a paid-dependency isolation guard, a secret scan, and an anti-jargon
grep over the sources. Keeping the local checks above clean covers the common
cases.

## Branch and pull-request conventions

- Branch off `main`. Use a short, descriptive branch name (for example
  `fix/viewer-token-audience` or `feat/inject-batch`).
- Keep pull requests focused: one logical change per PR where practical.
- Write clear commit messages: a concise subject line, then a body explaining
  the why when it is not obvious.
- Add or update tests for behavior changes. The protocol crate has a
  round-trip test suite; wire-format changes should extend it.
- Run the CI gate locally first. CI runs on every push and pull request, and
  must pass before merge.

## Reporting bugs and proposing features

Open a GitHub issue with a clear title, what you expected, what happened, and steps to reproduce. For security
vulnerabilities, do not open a public issue: follow [SECURITY.md](./SECURITY.md).
