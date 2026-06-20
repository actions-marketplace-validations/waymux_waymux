// SPDX-License-Identifier: Apache-2.0

//! `waymux-mcp`: a Model Context Protocol (MCP) server for waymux.
//!
//! Design (pi-harness): the `waymux` CLI is the canonical capability surface.
//! This server is a THIN wrapper. Each MCP tool maps to exactly one discrete
//! (request/response) CLI verb and executes the `waymux` binary with
//! `--json <verb> <args...>`, parsing the resulting JSON envelope back into an
//! MCP tool result.
//!
//! SECURITY: the server builds an argument VECTOR and runs the binary via
//! `std::process::Command` (argv only). It NEVER constructs a shell string and
//! NEVER passes arguments through a shell, so no client-supplied argument value
//! can inject a command. See `exec.rs`.
//!
//! The two STREAMING verbs (`events`, `logs`) are deliberately NOT exposed as
//! tools: they are not request/response and must stay CLI-only. `login` is also
//! not exposed (it writes credentials and is not a session-control capability).

pub mod exec;
pub mod registry;
pub mod server;
