// SPDX-License-Identifier: Apache-2.0

// crates/waymux-daemon/src/backend/mod.rs

// SessionBackend is the session-lifecycle extension point. `server::run` routes
// the `create` and `destroy` dispatch arms through it (every other op stays on
// the `Registry`), so the trait + `LocalBackend` are live in the default build.
// `LocalBackend` is the only implementation; the trait is the seam a future
// non-local lifecycle target would plug into.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::BTreeMap;
use waymux_protocol::CreateSessionResult;

/// Stable session identifier: the session name (matches `Registry::create`'s
/// `name` parameter).
pub type SessionId = String;

#[derive(Debug, Clone)]
pub struct CreateRequest {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub env: BTreeMap<String, String>,
    pub share_clipboard: bool,
    pub share_audio: bool,
    pub mem_cap_mb: Option<u32>,
    pub cpu_cap_pct: Option<u32>,
    pub disk_quota_mb: Option<u32>,
    pub fd_limit: Option<u32>,
    pub api_key_id: Option<String>,
}

#[async_trait]
pub trait SessionBackend: Send + Sync + 'static {
    /// Create a session. Returns the same `CreateSessionResult` the wire
    /// `create_session` response carries, so routing this through the backend
    /// is byte-identical to the previous direct `Registry::create` call on the
    /// local path.
    async fn create(&self, req: CreateRequest) -> Result<CreateSessionResult>;
    async fn destroy(&self, id: SessionId) -> Result<()>;
}

pub mod local;
pub use local::LocalBackend;

/// Pick a backend implementation from the `--backend` flag value. `local`
/// wraps the daemon's `Registry`; it is the only backend in this build.
pub fn build(
    choice: BackendChoice,
    registry: crate::registry::Registry,
) -> anyhow::Result<std::sync::Arc<dyn SessionBackend>> {
    match choice {
        BackendChoice::Local => Ok(std::sync::Arc::new(LocalBackend::new(registry))),
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BackendChoice {
    Local,
}
