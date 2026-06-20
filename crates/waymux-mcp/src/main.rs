// SPDX-License-Identifier: Apache-2.0

//! `waymux-mcp` binary entry point.
//!
//! Speaks MCP (JSON-RPC 2.0) over stdio: reads newline-delimited requests on
//! stdin and writes newline-delimited responses on stdout. It executes the
//! `waymux` CLI to fulfill tool calls (pi-harness model; see the crate docs).
//!
//! Binary resolution for the `waymux` CLI (see `exec::resolve_waymux_bin`):
//!   1. `WAYMUX_BIN` env var, if set.
//!   2. A `waymux` sibling next to this `waymux-mcp` executable.
//!   3. `waymux` on `$PATH`.

use std::io::{self, BufReader};

use waymux_mcp::server;

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    // Lock both streams for the lifetime of the loop; the server writes one
    // compact JSON object per line and flushes after each.
    let reader = BufReader::new(stdin.lock());
    let writer = stdout.lock();
    server::serve(reader, writer)
}
