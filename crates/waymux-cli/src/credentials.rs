// SPDX-License-Identifier: Apache-2.0

//! Credentials store for `waymux login` / `--remote`.
//!
//! On-disk layout (TOML):
//!
//! ```toml
//! [default]
//! base_url = "http://localhost:8080"
//! api_key  = "wmx_…"
//! ```
//!
//! The file lives at `$XDG_CONFIG_HOME/waymux/credentials.toml` (falling back
//! to `~/.config/waymux/credentials.toml`). The directory is `0700`, the file
//! is `0600`.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_PROFILE: &str = "default";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Profile {
    pub base_url: String,
    pub api_key: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Credentials {
    pub profiles: BTreeMap<String, Profile>,
}

impl Credentials {
    pub fn default_profile(&self) -> Option<&Profile> {
        self.profiles.get(DEFAULT_PROFILE)
    }

    pub fn set_default(&mut self, p: Profile) {
        self.profiles.insert(DEFAULT_PROFILE.to_string(), p);
    }
}

/// Resolve the canonical credentials file path. Honors `WAYMUX_CREDENTIALS`
/// (used by tests) and `XDG_CONFIG_HOME` before falling back to `~/.config`.
pub fn credentials_path() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("WAYMUX_CREDENTIALS") {
        return Ok(PathBuf::from(p));
    }
    let dir = if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(xdg)
    } else {
        let home = std::env::var_os("HOME")
            .context("neither XDG_CONFIG_HOME nor HOME set; cannot locate credentials")?;
        PathBuf::from(home).join(".config")
    };
    Ok(dir.join("waymux").join("credentials.toml"))
}

/// Load credentials from `path`. Returns `Ok(None)` if the file does not exist.
pub fn load_from(path: &Path) -> Result<Option<Credentials>> {
    match std::fs::read_to_string(path) {
        Ok(s) => {
            let creds: Credentials = toml::from_str(&s)
                .with_context(|| format!("parse credentials at {}", path.display()))?;
            Ok(Some(creds))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow!("read credentials at {}: {}", path.display(), e)),
    }
}

/// Convenience: load from the canonical location.
pub fn load() -> Result<Option<Credentials>> {
    load_from(&credentials_path()?)
}

/// Atomically write `creds` to `path`, ensuring the parent dir is `0700` and
/// the file itself is `0600`.
pub fn save_to(path: &Path, creds: &Credentials) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create dir {}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(parent)
                .with_context(|| format!("stat {}", parent.display()))?
                .permissions();
            perm.set_mode(0o700);
            std::fs::set_permissions(parent, perm)
                .with_context(|| format!("chmod 0700 {}", parent.display()))?;
        }
    }
    let body = toml::to_string_pretty(creds).context("serialize credentials")?;

    // Write to a sibling temp file with 0600, then rename into place. This
    // prevents transient world-readable windows even on FSes that don't honor
    // umask the way we'd want.
    let tmp = match path.parent() {
        Some(parent) => parent.join(format!(".waymux-credentials.{}.tmp", std::process::id())),
        None => PathBuf::from(format!(".waymux-credentials.{}.tmp", std::process::id())),
    };
    {
        use std::io::Write;
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)
                .with_context(|| format!("open {}", tmp.display()))?;
            f.write_all(body.as_bytes())?;
            f.sync_all()?;
        }
        #[cfg(not(unix))]
        {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)
                .with_context(|| format!("open {}", tmp.display()))?;
            f.write_all(body.as_bytes())?;
            f.sync_all()?;
        }
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = std::fs::metadata(path)
            .with_context(|| format!("stat {}", path.display()))?
            .permissions();
        perm.set_mode(0o600);
        std::fs::set_permissions(path, perm)
            .with_context(|| format!("chmod 0600 {}", path.display()))?;
    }
    Ok(())
}

/// Convenience: save to the canonical location.
pub fn save(creds: &Credentials) -> Result<PathBuf> {
    let path = credentials_path()?;
    save_to(&path, creds)?;
    Ok(path)
}

/// Show only the first 8 chars of an api key, with a trailing "…".
pub fn redact_key(api_key: &str) -> String {
    let prefix: String = api_key.chars().take(8).collect();
    if api_key.chars().count() > 8 {
        format!("{}…", prefix)
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_short_key() {
        assert_eq!(redact_key("wmx_abc"), "wmx_abc");
    }

    #[test]
    fn redact_long_key() {
        assert_eq!(redact_key("wmx_abcdefghijklmnop"), "wmx_abcd…");
    }

    #[test]
    fn roundtrip_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("creds.toml");
        let mut creds = Credentials::default();
        creds.set_default(Profile {
            base_url: "http://localhost:8080".into(),
            api_key: "wmx_xyz".into(),
        });
        save_to(&path, &creds).unwrap();
        let loaded = load_from(&path).unwrap().unwrap();
        assert_eq!(loaded.default_profile(), creds.default_profile());
    }
}
