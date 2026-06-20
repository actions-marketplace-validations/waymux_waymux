// SPDX-License-Identifier: Apache-2.0

//! Transport abstraction so subcommands can run against either the local
//! Unix-socket daemon or a remote HTTPS endpoint (`waymux-api`).
//!
//! Wire spec for the remote impl is fixed by MVP-B:
//!
//! ```text
//! POST   /sessions                                 → 201 { name, width, height, created_at }
//! GET    /sessions                                 → 200 { sessions: [...] }
//! DELETE /sessions/:name                           → 204
//! POST   /sessions/:name/inject                    → 200 {}
//! POST   /sessions/:name/screenshot                → 200 { png_b64, width, height, window_id }
//! GET    /sessions/:name/wait_for_idle?…           → 200 { idle: bool }
//! GET    /sessions/:name/windows                   → 200 { windows: [...] }
//! ```
//!
//! Errors map: 401 → "not authenticated", 402 → "quota exceeded",
//! 404 → "no such session", 503 → "daemon unavailable".

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::Path;

use waymux_protocol::{
    CreateSessionParams, CreateSessionResult, InjectOp as WireInjectOp, KeyState, RequestMethod,
    ScreenshotFormat, SessionCtlScreenshot, SessionInfo, WindowInfo,
};

/// Compact session record returned by `list_sessions` / `get_session` over both
/// transports. Keeps the surface narrow so the remote impl doesn't need to
/// fabricate fields the API doesn't return.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// ISO-8601 string. Local transport leaves this empty.
    #[serde(default)]
    pub created_at: String,
    /// Daemon PID. Remote transport leaves this 0.
    #[serde(default)]
    pub pid: i32,
    /// Attached flag. Remote transport leaves this false.
    #[serde(default)]
    pub attached: bool,
}

#[derive(Debug, Clone)]
pub struct ScreenshotResult {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub window_id: u32,
}

/// One inject op. Mirrors the JSON the API expects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InjectOp {
    Key {
        keycode: u32,
        #[serde(default)]
        release: bool,
        #[serde(default)]
        modifiers: u32,
    },
    Pointer {
        x: f64,
        y: f64,
        #[serde(default)]
        button: u32,
        /// "press" or "release"; "release" with button=0 means motion only.
        #[serde(default = "default_state")]
        state: String,
    },
}

fn default_state() -> String {
    "release".to_string()
}

#[async_trait]
pub trait Transport: Send {
    async fn list_sessions(&mut self) -> Result<Vec<SessionSummary>>;
    #[allow(clippy::too_many_arguments)]
    async fn create_session(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        scale: u32,
        share_audio: bool,
        mem_cap_mb: Option<u32>,
        cpu_cap_pct: Option<u32>,
        disk_quota_mb: Option<u32>,
        fd_limit: Option<u32>,
        api_key_id: Option<String>,
    ) -> Result<SessionSummary>;
    async fn destroy_session(&mut self, name: &str) -> Result<()>;
    async fn get_session(&mut self, name: &str) -> Result<Option<SessionSummary>>;
    async fn screenshot(&mut self, name: &str, window_id: Option<u32>) -> Result<ScreenshotResult>;
    async fn list_windows(&mut self, name: &str) -> Result<Vec<WindowInfo>>;
    async fn inject(&mut self, name: &str, ops: &[InjectOp]) -> Result<()>;
}

// ---------- LocalTransport ----------

/// Wraps the existing Unix-socket `Connection`.
pub struct LocalTransport {
    conn: crate::Connection,
}

impl LocalTransport {
    pub async fn connect(socket: &Path) -> Result<Self> {
        let mut conn = crate::Connection::connect(socket).await?;
        conn.hello().await?;
        Ok(Self { conn })
    }

    pub fn into_connection(self) -> crate::Connection {
        self.conn
    }

    pub fn connection_mut(&mut self) -> &mut crate::Connection {
        &mut self.conn
    }
}

#[async_trait]
impl Transport for LocalTransport {
    async fn list_sessions(&mut self) -> Result<Vec<SessionSummary>> {
        let sessions: Vec<SessionInfo> = self
            .conn
            .request(RequestMethod::ListSessions)
            .await?
            .decode_result()
            .map_err(|e| anyhow!("decode list_sessions result: {e}"))?;
        Ok(sessions
            .into_iter()
            .map(|s| SessionSummary {
                name: s.name,
                width: s.width,
                height: s.height,
                created_at: String::new(),
                pid: s.pid,
                attached: s.attached,
            })
            .collect())
    }

    async fn create_session(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        scale: u32,
        share_audio: bool,
        mem_cap_mb: Option<u32>,
        cpu_cap_pct: Option<u32>,
        disk_quota_mb: Option<u32>,
        fd_limit: Option<u32>,
        api_key_id: Option<String>,
    ) -> Result<SessionSummary> {
        let result: CreateSessionResult = self
            .conn
            .request(RequestMethod::CreateSession(CreateSessionParams {
                name: name.to_string(),
                width,
                height,
                scale,
                env: Default::default(),
                share_clipboard: false,
                share_audio,
                mem_cap_mb,
                cpu_cap_pct,
                disk_quota_mb,
                fd_limit,
                api_key_id,
                codec: None,
                gpu_type: None,
            }))
            .await?
            .decode_result()
            .map_err(|e| anyhow!("decode create_session result: {e}"))?;
        Ok(SessionSummary {
            name: result.name,
            width,
            height,
            created_at: String::new(),
            pid: 0,
            attached: false,
        })
    }

    async fn destroy_session(&mut self, name: &str) -> Result<()> {
        self.conn
            .request(RequestMethod::DestroySession {
                name: name.to_string(),
            })
            .await?;
        Ok(())
    }

    async fn get_session(&mut self, name: &str) -> Result<Option<SessionSummary>> {
        let all = self.list_sessions().await?;
        Ok(all.into_iter().find(|s| s.name == name))
    }

    async fn screenshot(&mut self, name: &str, window_id: Option<u32>) -> Result<ScreenshotResult> {
        let req = match window_id {
            Some(_) => RequestMethod::Screenshot {
                name: name.to_string(),
                window_id,
                format: ScreenshotFormat::Png,
            },
            None => RequestMethod::ScreenshotDesktop {
                name: name.to_string(),
                format: ScreenshotFormat::Png,
            },
        };
        let shot: SessionCtlScreenshot = self
            .conn
            .request(req)
            .await?
            .decode_result()
            .map_err(|e| anyhow!("decode screenshot: {e}"))?;
        Ok(ScreenshotResult {
            png: shot.png,
            width: shot.width,
            height: shot.height,
            window_id: window_id.unwrap_or(0),
        })
    }

    async fn list_windows(&mut self, name: &str) -> Result<Vec<WindowInfo>> {
        let windows: Vec<WindowInfo> = self
            .conn
            .request(RequestMethod::ListWindows {
                name: name.to_string(),
            })
            .await?
            .decode_result()
            .map_err(|e| anyhow!("decode list_windows result: {e}"))?;
        Ok(windows)
    }

    async fn inject(&mut self, name: &str, ops: &[InjectOp]) -> Result<()> {
        // Audit H10: carry all ops in a single InjectBatch RPC instead of one
        // round-trip per op, matching RemoteTransport::inject. The daemon
        // dispatches the batch in order with no inter-op delay.
        let wire_ops: Vec<WireInjectOp> = ops
            .iter()
            .map(|op| match op {
                InjectOp::Key {
                    keycode,
                    release,
                    modifiers,
                } => WireInjectOp::Key {
                    keycode: *keycode,
                    state: if *release {
                        KeyState::Released
                    } else {
                        KeyState::Pressed
                    },
                    modifiers: *modifiers,
                },
                InjectOp::Pointer {
                    x,
                    y,
                    button,
                    state,
                } => {
                    let st = match state.as_str() {
                        "press" | "pressed" => KeyState::Pressed,
                        _ => KeyState::Released,
                    };
                    WireInjectOp::Pointer {
                        x: *x,
                        y: *y,
                        button: *button,
                        state: st,
                        axis_x: 0.0,
                        axis_y: 0.0,
                        // CLI surface doesn't yet expose window_id /
                        // content; pass v1-compatible defaults.
                        window_id: None,
                        content: false,
                        seq: 0,
                    }
                }
            })
            .collect();
        self.conn
            .request(RequestMethod::InjectBatch {
                name: name.to_string(),
                ops: wire_ops,
            })
            .await?;
        Ok(())
    }
}

// ---------- RemoteTransport ----------

pub struct RemoteTransport {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl RemoteTransport {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("waymux-cli/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("build http client")?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            client,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header("Authorization", format!("Bearer {}", self.api_key))
    }

    /// Map HTTP errors to friendly anyhow messages per the spec.
    async fn check(resp: reqwest::Response) -> Result<reqwest::Response> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let body = resp.text().await.unwrap_or_default();
        match status.as_u16() {
            401 => bail!("not authenticated, run `waymux login`"),
            402 => bail!("quota exceeded"),
            404 => bail!("no such session"),
            503 => bail!("daemon unavailable, retry"),
            _ => bail!("HTTP {}: {}", status, body),
        }
    }
}

#[derive(Deserialize)]
struct ListSessionsBody {
    sessions: Vec<SessionSummary>,
}

#[derive(Serialize)]
struct CreateSessionBody<'a> {
    name: &'a str,
    width: u32,
    height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_cap_mb: Option<u32>,
}

#[derive(Deserialize)]
struct ScreenshotBody {
    png_b64: String,
    width: u32,
    height: u32,
    window_id: u32,
}

#[derive(Deserialize)]
struct WindowsBody {
    windows: Vec<WindowInfo>,
}

#[async_trait]
impl Transport for RemoteTransport {
    async fn list_sessions(&mut self) -> Result<Vec<SessionSummary>> {
        let resp = self
            .auth(self.client.get(self.url("/sessions")))
            .send()
            .await
            .context("GET /sessions")?;
        let resp = Self::check(resp).await?;
        let body: ListSessionsBody = resp.json().await.context("decode /sessions body")?;
        Ok(body.sessions)
    }

    async fn create_session(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        _scale: u32,
        _share_audio: bool,
        mem_cap_mb: Option<u32>,
        _cpu_cap_pct: Option<u32>,
        _disk_quota_mb: Option<u32>,
        _fd_limit: Option<u32>,
        _api_key_id: Option<String>,
    ) -> Result<SessionSummary> {
        // api_key_id is ignored on the remote path: the waymux-api proxy
        // already knows which customer is calling from the bearer token,
        // so it stamps usage events itself rather than trusting client input.
        // The new cpu/disk/fd caps are local-daemon-only for now — the API
        // surface for customer-set per-session caps is post-launch.
        let body = CreateSessionBody {
            name,
            width,
            height,
            mem_cap_mb,
        };
        let resp = self
            .auth(self.client.post(self.url("/sessions")).json(&body))
            .send()
            .await
            .context("POST /sessions")?;
        let resp = Self::check(resp).await?;
        let summary: SessionSummary = resp.json().await.context("decode create body")?;
        Ok(summary)
    }

    async fn destroy_session(&mut self, name: &str) -> Result<()> {
        let path = format!("/sessions/{}", name);
        let resp = self
            .auth(self.client.delete(self.url(&path)))
            .send()
            .await
            .with_context(|| format!("DELETE {}", path))?;
        Self::check(resp).await?;
        Ok(())
    }

    async fn get_session(&mut self, name: &str) -> Result<Option<SessionSummary>> {
        let all = self.list_sessions().await?;
        Ok(all.into_iter().find(|s| s.name == name))
    }

    async fn screenshot(&mut self, name: &str, window_id: Option<u32>) -> Result<ScreenshotResult> {
        #[derive(Serialize)]
        struct Body<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            window_id: Option<u32>,
            format: &'a str,
        }
        let body = Body {
            window_id,
            format: "png",
        };
        let path = format!("/sessions/{}/screenshot", name);
        let resp = self
            .auth(self.client.post(self.url(&path)).json(&body))
            .send()
            .await
            .with_context(|| format!("POST {}", path))?;
        let resp = Self::check(resp).await?;
        let s: ScreenshotBody = resp.json().await.context("decode screenshot body")?;
        let png = base64::engine::general_purpose::STANDARD
            .decode(&s.png_b64)
            .context("decode png_b64")?;
        Ok(ScreenshotResult {
            png,
            width: s.width,
            height: s.height,
            window_id: s.window_id,
        })
    }

    async fn list_windows(&mut self, name: &str) -> Result<Vec<WindowInfo>> {
        let path = format!("/sessions/{}/windows", name);
        let resp = self
            .auth(self.client.get(self.url(&path)))
            .send()
            .await
            .with_context(|| format!("GET {}", path))?;
        let resp = Self::check(resp).await?;
        let body: WindowsBody = resp.json().await.context("decode windows body")?;
        Ok(body.windows)
    }

    async fn inject(&mut self, name: &str, ops: &[InjectOp]) -> Result<()> {
        #[derive(Serialize)]
        struct Body<'a> {
            ops: &'a [InjectOp],
        }
        let path = format!("/sessions/{}/inject", name);
        let resp = self
            .auth(self.client.post(self.url(&path)).json(&Body { ops }))
            .send()
            .await
            .with_context(|| format!("POST {}", path))?;
        Self::check(resp).await?;
        Ok(())
    }
}
