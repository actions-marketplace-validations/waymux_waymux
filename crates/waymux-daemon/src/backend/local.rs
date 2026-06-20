// SPDX-License-Identifier: Apache-2.0

//! `LocalBackend` — wraps the daemon's existing `Registry` for the in-process
//! subprocess-spawn path. `create` / `destroy` delegate straight to
//! `Registry::create` / `Registry::destroy` on the SAME registry that
//! `server::run` hands to every other op, so a session created through the
//! backend is visible to `list`/`inject`/`screenshot`/`record`/`rm` exactly as
//! before this was wired.

use crate::backend::{CreateRequest, SessionBackend, SessionId};
use crate::registry::Registry;
use anyhow::Result;
use async_trait::async_trait;
use waymux_protocol::CreateSessionResult;

pub struct LocalBackend {
    registry: Registry,
}

impl LocalBackend {
    pub fn new(registry: Registry) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl SessionBackend for LocalBackend {
    async fn create(&self, req: CreateRequest) -> Result<CreateSessionResult> {
        // Delegate straight to the shared registry and return its result
        // UNWRAPPED. The returned `CreateSessionResult` is the exact value the
        // previous direct `Registry::create` dispatch arm serialized, so the
        // wire response is byte-identical and the new session lands in the same
        // registry every other op reads. We intentionally do NOT add an anyhow
        // context layer here: the dispatcher downcasts the error to the typed
        // `CreateError` (AlreadyExists / InvalidName) to pick the wire error
        // code, and a context wrapper would bury that type and regress those
        // codes to E_INTERNAL.
        self.registry
            .create(
                req.name,
                req.width,
                req.height,
                req.scale,
                req.env,
                req.share_clipboard,
                req.share_audio,
                req.mem_cap_mb,
                req.cpu_cap_pct,
                req.disk_quota_mb,
                req.fd_limit,
                req.api_key_id,
            )
            .await
    }

    async fn destroy(&self, id: SessionId) -> Result<()> {
        // Return the registry's result UNWRAPPED so the dispatcher can still
        // downcast `DestroyError::NotFound` to the E_NOT_FOUND wire code (a
        // context wrapper would regress it to E_INTERNAL).
        self.registry.destroy(&id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;
    use tempfile::tempdir;

    #[tokio::test]
    async fn destroy_missing_session_errors() {
        let tmp = tempdir().unwrap();
        #[cfg(feature = "metering")]
        let registry = Registry::new(
            tmp.path().to_path_buf(),
            tmp.path().join("waymux-session"),
            None,
            "test-run-id".into(),
        );
        #[cfg(not(feature = "metering"))]
        let registry = Registry::new(tmp.path().to_path_buf(), tmp.path().join("waymux-session"));
        let backend = LocalBackend::new(registry);
        let err = backend.destroy("never-existed".into()).await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.to_lowercase().contains("not found") || msg.contains("destroy"),
            "expected destroy/not-found in {msg:?}"
        );
    }

    /// Write a stub `waymux-session` that satisfies the daemon's ready
    /// handshake (connect to `--ready-socket`, write a byte) and then blocks,
    /// so `Registry::create` reaches the "ready" outcome without a real
    /// compositor. Returns the script path. Skips (returns `None`) when no
    /// `python3` is on `PATH` — the script uses it for the AF_UNIX connect.
    fn write_stub_session(dir: &std::path::Path) -> Option<std::path::PathBuf> {
        use std::os::unix::fs::PermissionsExt;
        if std::process::Command::new("python3")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| !s.success())
            .unwrap_or(true)
        {
            return None;
        }
        let path = dir.join("waymux-session-stub.sh");
        // Parse argv for --ready-socket, connect + write one byte, then sleep
        // forever (the daemon SIGKILLs us on destroy via kill_on_drop).
        let script = r#"#!/bin/sh
ready=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ready-socket) ready="$2"; shift 2 ;;
    *) shift ;;
  esac
done
python3 -c "import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); s.send(b'\x01')" "$ready"
exec sleep 3600
"#;
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        Some(path)
    }

    fn make_registry(state_dir: std::path::PathBuf, session_bin: std::path::PathBuf) -> Registry {
        #[cfg(feature = "metering")]
        {
            Registry::new(state_dir, session_bin, None, "test-run-id".into())
        }
        #[cfg(not(feature = "metering"))]
        {
            Registry::new(state_dir, session_bin)
        }
    }

    /// The wiring contract: a session created through `LocalBackend::create`
    /// lands in the SAME `Registry` that every other op (`list`, and therefore
    /// inject/screenshot/record/rm) reads. We hold one registry, hand a clone
    /// to the backend, and assert the create is visible through the registry
    /// handle we kept. Also confirms `destroy` removes it.
    #[tokio::test]
    async fn create_via_backend_is_visible_to_registry() {
        let tmp = tempdir().unwrap();
        let Some(stub) = write_stub_session(tmp.path()) else {
            eprintln!("skipping: python3 not available for the ready-handshake stub");
            return;
        };
        let registry = make_registry(tmp.path().join("state"), stub);
        let backend = LocalBackend::new(registry.clone());

        // Before: empty.
        assert!(registry.list().await.is_empty());

        let cr = backend
            .create(CreateRequest {
                name: "shared".into(),
                width: 640,
                height: 480,
                scale: 1,
                env: Default::default(),
                share_clipboard: false,
                share_audio: false,
                mem_cap_mb: None,
                cpu_cap_pct: None,
                disk_quota_mb: None,
                fd_limit: None,
                api_key_id: None,
            })
            .await
            .expect("create via backend");
        // The wire result is the registry's `CreateSessionResult` verbatim.
        assert_eq!(cr.name, "shared");
        assert!(!cr.inner_socket_path.is_empty());

        // Visible through the registry handle we kept (NOT the backend's),
        // proving they share one registry.
        let list = registry.list().await;
        assert_eq!(list.len(), 1, "session not visible to registry.list()");
        assert_eq!(list[0].name, "shared");
        assert_eq!(list[0].width, 640);
        assert!(list[0].pid > 0);

        // destroy through the backend removes it from the same registry.
        backend.destroy("shared".into()).await.expect("destroy");
        assert!(registry.list().await.is_empty());

        registry.shutdown_all().await;
    }
}
