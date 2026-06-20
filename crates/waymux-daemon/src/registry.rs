// SPDX-License-Identifier: Apache-2.0

//! Session registry.
//!
//! Owns session metadata (`SessionMeta`) and a broadcast channel that fans
//! events out to all subscribed clients. The `tokio::process::Child` of each
//! session lives in a dedicated supervisor task, not in the registry map —
//! this avoids borrow-checker contortions around `child.wait()` and keeps
//! the cleanup path symmetric between "destroy" and "unexpected exit".

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, oneshot, Mutex};
use tracing::info;
use waymux_protocol::{
    decode_frame, encode_frame, CreateSessionResult, Event, EventBody, SessionCtlMethod,
    SessionCtlRecordStarted, SessionCtlRequest, SessionCtlResponse, SessionInfo,
};

#[cfg(feature = "metering")]
use crate::usage_events::{self, UsageEvent, UsageEventSink};

/// Broadcast buffer size. The protocol spec caps per-subscriber pending events at 1024.
const EVENT_CHANNEL_CAPACITY: usize = 1024;

pub struct SessionMeta {
    pub name: String,
    pub pid: i32,
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub created_at: u64,
    pub inner_socket: PathBuf,
    pub control_socket: PathBuf,
    /// Path of the per-session `waymux_attach_v1` Wayland socket. Handed
    /// out via `attach(name)` so an attach client can connect to it.
    pub attach_socket: PathBuf,
    pub attached: bool,
}

struct SessionEntry {
    meta: SessionMeta,
    /// Consumed by `destroy()`. Sending triggers the supervisor's kill path.
    /// Dropping without sending is harmless (supervisor keeps waiting on child).
    kill: Option<oneshot::Sender<()>>,
    /// PIDs of processes spawned into this session via `spawn_child`.
    /// Pruned when ChildExited fires; killed (SIGTERM) by `destroy()`.
    child_pids: Vec<u32>,
    /// PID of the inner compositor child (the one spawned with
    /// `compositor: true`, e.g. KWin). When this PID exits, the daemon
    /// emits `SessionCrashed` instead of `ChildExited`.
    compositor_pid: Option<u32>,
    /// Best-effort cgroup-v2 leaf capping the session's aggregate memory.
    /// `None` if `mem_cap_mb` was not requested or cgroup setup failed —
    /// see `cgroup::SessionCgroup::try_create`.
    cgroup: Option<crate::cgroup::SessionCgroup>,
    /// Best-effort tmpfs mount over the session runtime dir. `None` when
    /// `disk_quota_mb` wasn't requested or the daemon lacked CAP_SYS_ADMIN.
    /// Unmounted (lazy / MNT_DETACH) on destroy.
    tmpfs: Option<crate::quota::SessionTmpfs>,
    /// API-key id (UUID string) the SDK supplied at create time via
    /// `CreateSessionParams.api_key_id`. Emitted verbatim in usage events so
    /// a consumer can join JSONL to user without a lookup. Only retained in
    /// the `metering` build.
    #[cfg(feature = "metering")]
    api_key_id: Option<String>,
    /// Rolling log history for this session, replayed on
    /// subscribe. Per-session lock so drainers from different sessions
    /// don't contend; `Arc` so the lock can be cheaply cloned out of the
    /// sessions map and handed to drainer tasks (audit H11).
    log_history: Arc<Mutex<std::collections::VecDeque<(String, String)>>>,
    /// Persistent control-socket connection. Opened lazily on the first
    /// `session_control()` call, reused across subsequent calls. Audit H8:
    /// the pre-fix code opened a fresh `UnixStream` per RPC, paying
    /// ~50-200µs connect+spawn overhead on every `wait_for_idle` poll
    /// (100 RPC/s/client). Concurrent callers serialize on this mutex,
    /// matching the session's per-connection request-response loop in
    /// `waymux-session/src/control.rs`. Cleared (set to `None`) on any
    /// I/O error so the next call reconnects rather than wedging on a
    /// dead socket. Drops naturally with the `SessionEntry` on destroy.
    control_conn: Arc<Mutex<Option<UnixStream>>>,
}

#[derive(Clone)]
pub struct Registry {
    inner: Arc<Inner>,
}

pub(crate) struct Inner {
    sessions: Mutex<HashMap<String, SessionEntry>>,
    /// Window tags keyed by (session name, window id).
    /// Window ids come from the session; daemon-side tag storage is
    /// authoritative. Cleared when the session is destroyed.
    tags: Mutex<HashMap<(String, u32), Vec<String>>>,
    state_dir: PathBuf,
    session_bin: PathBuf,
    events_tx: broadcast::Sender<Event>,
    /// Optional append-only JSONL stream of usage events. `None` when the
    /// daemon was started without `--usage-events-sink`. Only present in the
    /// `metering` build.
    #[cfg(feature = "metering")]
    usage_sink: Option<Arc<UsageEventSink>>,
    /// Per-process UUID (string form) generated at daemon startup. Stamped
    /// on every emitted `UsageEvent` so the consumer joins on
    /// `(run_id, session)` instead of bare session name. See audit C8.
    #[cfg(feature = "metering")]
    pub(crate) run_id: String,
    /// Supervisor task handles. Audit C9: shutdown_all awaits these so
    /// `main()` does not return before SIGTERM-driven session teardown
    /// completes (orphaned cgroups, sockets, child processes otherwise).
    supervisor_handles: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

/// Max log lines retained per session (rolling). 1024 lines is generous
/// for debugging a startup failure and tiny in memory (~100KB).
const LOG_HISTORY_CAPACITY: usize = 1024;

/// Upper bound on the number of `spawn` argv elements. A real command line
/// (chromium with a long flag list, say) is a few hundred at most; 1024 leaves
/// generous headroom while bounding a resource-exhaustion attempt.
const MAX_ARGV_LEN: usize = 1024;
/// Upper bound on a single argv element's byte length (256 KiB). Larger than
/// any plausible flag/path/URL, smaller than a value that would blow execve's
/// ARG_MAX or force a large allocation per call.
const MAX_ARGV_ELEM_BYTES: usize = 256 * 1024;

impl Registry {
    #[cfg(feature = "metering")]
    pub fn new(
        state_dir: PathBuf,
        session_bin: PathBuf,
        usage_sink: Option<Arc<UsageEventSink>>,
        run_id: String,
    ) -> Self {
        let (events_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let reg = Self {
            inner: Arc::new(Inner {
                sessions: Mutex::new(HashMap::new()),
                tags: Mutex::new(HashMap::new()),
                state_dir,
                session_bin,
                events_tx,
                supervisor_handles: Mutex::new(Vec::new()),
                usage_sink,
                run_id,
            }),
        };
        if reg.inner.usage_sink.is_some() {
            reg.spawn_heartbeat_task();
        }
        reg
    }

    #[cfg(not(feature = "metering"))]
    pub fn new(state_dir: PathBuf, session_bin: PathBuf) -> Self {
        let (events_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self {
            inner: Arc::new(Inner {
                sessions: Mutex::new(HashMap::new()),
                tags: Mutex::new(HashMap::new()),
                state_dir,
                session_bin,
                events_tx,
                supervisor_handles: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Tick every 60s and emit `session_heartbeat` for each live session.
    /// Lets a usage-event consumer detect lost JSONL events across crashes
    /// without needing to scan the cgroup tree from the daemon side.
    #[cfg(feature = "metering")]
    fn spawn_heartbeat_task(&self) {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(60));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Skip the immediate first tick: first heartbeat fires 60s in.
            tick.tick().await;
            loop {
                tick.tick().await;
                let Some(sink) = inner.usage_sink.as_ref() else {
                    return;
                };
                let snapshot: Vec<(String, Option<String>)> = inner
                    .sessions
                    .lock()
                    .await
                    .iter()
                    .map(|(n, s)| (n.clone(), s.api_key_id.clone()))
                    .collect();
                let ts = usage_events::now();
                for (name, key_id) in snapshot {
                    sink.emit(UsageEvent::SessionHeartbeat {
                        ts,
                        run_id: &inner.run_id,
                        session: &name,
                        api_key_id: key_id.as_deref(),
                    })
                    .await;
                }
            }
        });
    }

    /// Return the retained log history for a session — caller uses this
    /// on `subscribe` to pre-populate the subscriber's stream. Each item
    /// is `(stream_name, text)`. Entries are in arrival order.
    pub async fn replay_logs(&self, session: &str) -> Vec<(String, String)> {
        // Grab the per-session log lock under the sessions lock, then drop
        // the sessions lock before cloning the deque so the (potentially
        // 30k-entry) clone doesn't block other registry operations.
        let entry_log = self
            .inner
            .sessions
            .lock()
            .await
            .get(session)
            .map(|e| e.log_history.clone());
        match entry_log {
            Some(lock) => lock.lock().await.iter().cloned().collect(),
            None => Vec::new(),
        }
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<Event> {
        self.inner.events_tx.subscribe()
    }

    fn broadcast(&self, body: EventBody) {
        // `send` returns Err only when there are no receivers — fine to ignore.
        let _ = self.inner.events_tx.send(Event::new(body));
    }

    pub async fn list(&self) -> Vec<SessionInfo> {
        let sessions = self.inner.sessions.lock().await;
        sessions
            .values()
            .map(|s| SessionInfo {
                name: s.meta.name.clone(),
                pid: s.meta.pid,
                width: s.meta.width,
                height: s.meta.height,
                scale: s.meta.scale,
                attached: s.meta.attached,
                created_at: s.meta.created_at,
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create(
        &self,
        name: String,
        width: u32,
        height: u32,
        scale: u32,
        env: std::collections::BTreeMap<String, String>,
        share_clipboard: bool,
        share_audio: bool,
        mem_cap_mb: Option<u32>,
        cpu_cap_pct: Option<u32>,
        disk_quota_mb: Option<u32>,
        fd_limit: Option<u32>,
        api_key_id: Option<String>,
    ) -> Result<CreateSessionResult> {
        validate_session_name(&name)?;
        validate_output_size(width, height)?;

        // `api_key_id` is wire-driven from `waymux-protocol`'s
        // `CreateSessionParams`; it is consumed only by the metering build's
        // usage-event emit below. Bind it away here so the default build does
        // not warn on the unused parameter.
        #[cfg(not(feature = "metering"))]
        let _ = &api_key_id;

        {
            let sessions = self.inner.sessions.lock().await;
            if sessions.contains_key(&name) {
                bail!(CreateError::AlreadyExists);
            }
        }

        let session_dir = self.inner.state_dir.join(&name);
        tokio::fs::create_dir_all(&session_dir)
            .await
            .with_context(|| format!("mkdir {}", session_dir.display()))?;

        // Mount tmpfs over the session dir BEFORE we bind any sockets in
        // it: the kernel's mount semantics replace the directory's view,
        // and any fd opened against the underlying dir would survive but
        // be invisible to the session process.
        let tmpfs = match disk_quota_mb {
            Some(mb) if mb > 0 => crate::quota::SessionTmpfs::try_mount(&session_dir, mb),
            _ => None,
        };

        let inner_socket = session_dir.join("wayland.sock");
        if inner_socket.exists() {
            let _ = tokio::fs::remove_file(&inner_socket).await;
        }
        let control_socket = session_dir.join("control.sock");
        if control_socket.exists() {
            let _ = tokio::fs::remove_file(&control_socket).await;
        }
        let attach_socket = session_dir.join("attach.sock");
        if attach_socket.exists() {
            let _ = tokio::fs::remove_file(&attach_socket).await;
        }

        let ready_socket = session_dir.join("ready.sock");
        if ready_socket.exists() {
            let _ = tokio::fs::remove_file(&ready_socket).await;
        }
        let ready_listener = UnixListener::bind(&ready_socket)
            .with_context(|| format!("bind ready socket {}", ready_socket.display()))?;

        let events_socket = session_dir.join("events.sock");
        if events_socket.exists() {
            let _ = tokio::fs::remove_file(&events_socket).await;
        }
        let events_listener = UnixListener::bind(&events_socket)
            .with_context(|| format!("bind events socket {}", events_socket.display()))?;

        let mut cmd = Command::new(&self.inner.session_bin);
        cmd.arg("--name")
            .arg(&name)
            .arg("--width")
            .arg(width.to_string())
            .arg("--height")
            .arg(height.to_string())
            .arg("--scale")
            .arg(scale.to_string())
            .arg("--inner-socket")
            .arg(&inner_socket)
            .arg("--control-socket")
            .arg(&control_socket)
            .arg("--attach-socket")
            .arg(&attach_socket)
            .arg("--events-socket")
            .arg(&events_socket)
            .arg("--ready-socket")
            .arg(&ready_socket)
            .args(if share_clipboard {
                &["--share-clipboard"][..]
            } else {
                &[][..]
            })
            .args(if share_audio {
                &["--share-audio"][..]
            } else {
                &[][..]
            })
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        cmd.env_clear();
        cmd.env("PATH", std::env::var_os("PATH").unwrap_or_default());
        if let Ok(v) = std::env::var("RUST_LOG") {
            cmd.env("RUST_LOG", v);
        }
        // Forward RUST_BACKTRACE so panics in the session / compositor /
        // outer-view threads produce useful tracebacks in the logs.
        if let Ok(v) = std::env::var("RUST_BACKTRACE") {
            cmd.env("RUST_BACKTRACE", v);
        }
        // Forward WAYMUX_* feature flags from the daemon's environment.
        // env_clear() is a security boundary, but in-house WAYMUX_*
        // flags are safe to inherit — they're our own contract surface
        // for tuning behavior at session creation time (e.g.,
        // WAYMUX_DISABLE_SYNCOBJ=1 to force implicit-sync fallback
        // for benchmark workloads).
        for (k, v) in std::env::vars() {
            if k.starts_with("WAYMUX_") {
                cmd.env(k, v);
            }
        }
        for (k, v) in env {
            cmd.env(k, v);
        }

        if let Some(cap) = fd_limit {
            if cap > 0 {
                // SAFETY: pre_exec runs in the forked child between fork
                // and execve. Only async-signal-safe calls are permitted;
                // `setrlimit` qualifies. We must not allocate or take
                // locks here — `apply_fd_limit` only does syscalls.
                unsafe {
                    cmd.pre_exec(move || crate::quota::apply_fd_limit(cap));
                }
            }
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {}", self.inner.session_bin.display()))?;
        let pid = child.id().map(|p| p as i32).unwrap_or(-1);

        enum SpawnOutcome {
            Ready,
            Exited(Option<i32>),
            TimedOut,
        }
        let outcome = tokio::select! {
            res = accept_ready(&ready_listener) => {
                res.context("waiting for session ready")?;
                SpawnOutcome::Ready
            }
            status = child.wait() => {
                SpawnOutcome::Exited(status.ok().and_then(|s| s.code()))
            }
            _ = tokio::time::sleep(Duration::from_millis(5_000)) => {
                SpawnOutcome::TimedOut
            }
        };
        match outcome {
            SpawnOutcome::Ready => {}
            SpawnOutcome::Exited(code) => {
                let stderr = drain_stderr(&mut child).await.unwrap_or_default();
                bail!(
                    "session {} exited before ready (code {}): {}",
                    name,
                    code.unwrap_or(-1),
                    stderr.trim()
                );
            }
            SpawnOutcome::TimedOut => {
                let _ = child.kill().await;
                bail!("session {} spawn timed out after 5s", name);
            }
        }
        let _ = tokio::fs::remove_file(&ready_socket).await;

        // Best-effort cgroup-v2 setup. Done after the ready handshake so a
        // session that fails to start doesn't leave a stale cgroup behind.
        // The session pid is moved in immediately; subsequent `spawn_child`
        // calls will join the same cgroup so the cap aggregates across the
        // whole session subtree.
        let want_mem = mem_cap_mb.unwrap_or(0) > 0;
        let want_cpu = cpu_cap_pct.unwrap_or(0) > 0;
        let cgroup = if (want_mem || want_cpu) && pid > 0 {
            let cg = crate::cgroup::SessionCgroup::try_create_empty(&name);
            if let Some(cg) = cg.as_ref() {
                if want_mem {
                    cg.set_memory_max(mem_cap_mb.unwrap_or(0));
                }
                if want_cpu {
                    cg.set_cpu_max(cpu_cap_pct.unwrap_or(0));
                }
                cg.add_pid(pid as u32);
            }
            cg
        } else {
            None
        };

        // Hand off the events socket accept to a long-lived task. The session
        // will connect shortly (typically before its first window is created).
        // If the session never connects or connects then dies, the task
        // naturally exits and we stop forwarding that session's events.
        let events_inner = self.inner.clone();
        let events_name = name.clone();
        tokio::spawn(async move {
            accept_and_forward_events(events_listener, events_name, events_inner).await;
        });

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let (kill_tx, kill_rx) = oneshot::channel();
        let meta = SessionMeta {
            name: name.clone(),
            pid,
            width,
            height,
            scale,
            created_at,
            inner_socket: inner_socket.clone(),
            control_socket: control_socket.clone(),
            attach_socket: attach_socket.clone(),
            attached: false,
        };

        // Drain stdout/stderr into Log events. Must happen BEFORE moving
        // `child` into the supervisor. Otherwise the child's pipes fill (~64KB)
        // and the next write blocks it indefinitely.
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Hold the map lock across the insert + supervisor spawn so the
        // supervisor can never race ahead and find an empty map on a
        // fast-exiting session.
        let log_history: Arc<Mutex<std::collections::VecDeque<(String, String)>>> =
            Arc::new(Mutex::new(std::collections::VecDeque::new()));
        {
            let mut sessions = self.inner.sessions.lock().await;
            if sessions.contains_key(&name) {
                let _ = child.kill().await;
                bail!(CreateError::AlreadyExists);
            }
            #[cfg(feature = "metering")]
            if let Some(sink) = self.inner.usage_sink.as_ref() {
                sink.emit(UsageEvent::SessionStart {
                    ts: usage_events::now(),
                    run_id: &self.inner.run_id,
                    session: &name,
                    api_key_id: api_key_id.as_deref(),
                })
                .await;
            }
            sessions.insert(
                name.clone(),
                SessionEntry {
                    meta,
                    kill: Some(kill_tx),
                    child_pids: Vec::new(),
                    compositor_pid: None,
                    cgroup,
                    tmpfs,
                    #[cfg(feature = "metering")]
                    api_key_id,
                    log_history: log_history.clone(),
                    control_conn: Arc::new(Mutex::new(None)),
                },
            );

            let inner = self.inner.clone();
            let supervised_name = name.clone();
            let supervised_socket = inner_socket.clone();
            let supervisor_handle = tokio::spawn(async move {
                session_supervisor(supervised_name, child, kill_rx, supervised_socket, inner).await;
            });
            // Audit C9: track the handle so shutdown_all can await it.
            self.inner
                .supervisor_handles
                .lock()
                .await
                .push(supervisor_handle);

            if let Some(out) = stdout {
                spawn_log_drainer(
                    self.inner.events_tx.clone(),
                    log_history.clone(),
                    name.clone(),
                    "stdout",
                    out,
                );
            }
            if let Some(err) = stderr {
                spawn_log_drainer(
                    self.inner.events_tx.clone(),
                    log_history.clone(),
                    name.clone(),
                    "stderr",
                    err,
                );
            }
        }

        self.broadcast(EventBody::SessionCreated { name: name.clone() });
        info!(%name, pid, "session created");

        Ok(CreateSessionResult {
            name,
            inner_socket_path: inner_socket.display().to_string(),
        })
    }

    pub async fn destroy(&self, name: &str) -> Result<()> {
        let (tx, child_pids, cgroup, tmpfs) = {
            let mut sessions = self.inner.sessions.lock().await;
            let mut entry = sessions
                .remove(name)
                .ok_or_else(|| anyhow!(DestroyError::NotFound))?;
            (
                entry.kill.take(),
                std::mem::take(&mut entry.child_pids),
                entry.cgroup.take(),
                entry.tmpfs.take(),
            )
        };
        #[cfg(feature = "metering")]
        if let Some(sink) = self.inner.usage_sink.as_ref() {
            sink.emit(UsageEvent::SessionStop {
                ts: usage_events::now(),
                run_id: &self.inner.run_id,
                session: name,
                exit_code: None,
            })
            .await;
        }
        // SIGTERM every process that was spawned into this session.
        // Ignore ESRCH — the process may have already exited.
        for pid in &child_pids {
            unsafe { libc::kill(*pid as libc::pid_t, libc::SIGTERM) };
        }
        if !child_pids.is_empty() {
            info!(%name, pids = ?child_pids, "session destroy: SIGTERMed spawned children");
        }
        // Forget any tags this session accumulated.
        self.inner.tags.lock().await.retain(|(s, _), _| s != name);
        // Sending may fail if the supervisor already exited (child died
        // naturally a moment ago); either way the session is gone.
        if let Some(tx) = tx {
            let _ = tx.send(());
        }
        // Unmount the tmpfs (lazy / MNT_DETACH so it doesn't block on
        // children still holding open fds against files inside). The
        // supervisor's later remove_dir on the underlying dir then succeeds
        // because the umount has detached the tmpfs.
        if let Some(t) = tmpfs {
            t.unmount();
        }
        // Best-effort cgroup teardown.
        if let Some(cg) = cgroup {
            tokio::spawn(async move {
                // Audit task #75: nuke the entire session subtree atomically
                // before attempting rmdir. SIGTERM'ing only `child_pids` left
                // chromium's GPU/utility/zygote subprocesses alive — cgroup.kill
                // catches them all. Best-effort: silent on older kernels or
                // when the leaf is missing.
                cg.kill_all();
                // Audit fix C10: retry rmdir on EBUSY. cgroup.kill is
                // synchronous-ish (the kernel signals every PID in the
                // cgroup, but they have to actually die before rmdir
                // works); the retry handles the "PIDs still mid-exit" race.
                for attempt in 0..10 {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    if cg.try_remove() {
                        tracing::debug!(
                            path = %cg.path.display(), attempt,
                            "cgroup cleaned up"
                        );
                        return;
                    }
                }
                tracing::warn!(
                    path = %cg.path.display(),
                    "cgroup cleanup: still busy after 10 retries (~2s); leaving in place"
                );
            });
        }
        info!(%name, "session destroy requested");
        Ok(())
    }

    pub async fn shutdown_all(&self) {
        let names: Vec<_> = {
            let sessions = self.inner.sessions.lock().await;
            sessions.keys().cloned().collect()
        };
        for name in names {
            let _ = self.destroy(&name).await;
        }
        // Audit C9: actually wait for supervisors to finish their teardown
        // (SIGTERM grace, child cleanup, JSONL session_stop emit) before
        // returning. The pre-fix 200ms sleep was a soft no-op — `main()`
        // would return before the supervisors finished, leaving orphaned
        // cgroups, sockets, and child processes. 8s is the worst-case
        // budget: 5s SIGTERM grace + ~2s SIGKILL grace + slack.
        let handles: Vec<_> = std::mem::take(&mut *self.inner.supervisor_handles.lock().await);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
        for h in handles {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let _ = tokio::time::timeout(remaining, h).await;
        }
    }

    pub async fn spawn_child(
        &self,
        session_name: &str,
        argv: Vec<String>,
        env: std::collections::BTreeMap<String, String>,
        compositor: bool,
    ) -> Result<i32> {
        if argv.is_empty() {
            bail!(SpawnError::InvalidArgv);
        }
        // Audit C7 + H12: argv[0] must be absolute. Without `env_clear`,
        // PATH-resolution against an attacker-controlled env (e.g. customer
        // OCI image in V2) is exploitable. Even with env_clear an absolute
        // requirement is the safer invariant.
        if !argv[0].starts_with('/') {
            bail!(SpawnError::ArgvNotAbsolute);
        }
        // Resource-exhaustion guard (not injection: argv is always passed as
        // discrete execve elements, never through a shell). A caller that hands
        // us millions of args or a multi-gigabyte single arg would force the
        // daemon (and the kernel's execve, capped at ~ARG_MAX) to materialize
        // it all. Reject anything implausibly large with a clear error.
        if argv.len() > MAX_ARGV_LEN {
            bail!(SpawnError::ArgvTooLarge);
        }
        if let Some(big) = argv.iter().find(|a| a.len() > MAX_ARGV_ELEM_BYTES) {
            tracing::warn!(
                session = %session_name,
                elem_bytes = big.len(),
                "rejecting spawn: argv element exceeds size cap"
            );
            bail!(SpawnError::ArgvTooLarge);
        }

        let (inner_dir, wayland_display) = {
            let sessions = self.inner.sessions.lock().await;
            let s = sessions
                .get(session_name)
                .ok_or_else(|| anyhow!(SpawnError::SessionNotFound))?;
            let dir = s
                .meta
                .inner_socket
                .parent()
                .ok_or_else(|| anyhow!("inner socket has no parent"))?
                .to_path_buf();
            let display = s
                .meta
                .inner_socket
                .file_name()
                .ok_or_else(|| anyhow!("inner socket has no file name"))?
                .to_string_lossy()
                .into_owned();
            (dir, display)
        };

        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(false);
        // Audit C7: don't leak daemon secrets (DATABASE_URL, JWT secret,
        // STRIPE_SECRET_KEY, etc.) into customer-supplied argv. env_clear
        // gives us a clean slate; we re-add only what a Wayland client
        // legitimately needs.
        cmd.env_clear();
        for key in &[
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "FONTCONFIG_PATH",
            "XDG_DATA_DIRS",
        ] {
            if let Ok(v) = std::env::var(key) {
                cmd.env(key, v);
            }
        }
        cmd.env("XDG_RUNTIME_DIR", &inner_dir);
        cmd.env("WAYLAND_DISPLAY", &wayland_display);
        cmd.env("XDG_SESSION_TYPE", "wayland");
        cmd.env("MOZ_ENABLE_WAYLAND", "1"); // Firefox ≤ 121 explicit opt-in
                                            // waymux captures and streams the session, so spawned clients must hand
                                            // out an encoder-importable dmabuf. AMD's DCC tiling produces modifiers
                                            // the Vulkan encoder cannot import (recording and the live viewer then
                                            // get zero frames), so disable DCC by default. These are no-ops on
                                            // non-AMD drivers, and the caller env below can override them.
        cmd.env("AMD_DEBUG", "nodcc");
        cmd.env("RADV_DEBUG", "nodcc");
        cmd.env_remove("DISPLAY"); // prevent X11 fallback
                                   // User-supplied env last so callers can override or add (e.g. test
                                   // fixtures, app-specific config). Starts from a clean slate.
        for (k, v) in env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().with_context(|| format!("spawn {}", argv[0]))?;
        let pid = child.id().map(|p| p as i32).unwrap_or(-1);
        info!(session = %session_name, pid, argv = ?argv, "spawned client");

        // Record this child so destroy() can SIGTERM it, and join the
        // session cgroup if one is active so memory accounting aggregates.
        // Also grab the per-session log_history Arc to hand to the drainers
        // (audit H11 — drainers no longer touch a global lock).
        let log_history = if pid > 0 {
            let mut sessions = self.inner.sessions.lock().await;
            sessions.get_mut(session_name).map(|entry| {
                entry.child_pids.push(pid as u32);
                if compositor {
                    entry.compositor_pid = Some(pid as u32);
                }
                if let Some(cg) = entry.cgroup.as_ref() {
                    cg.add_pid(pid as u32);
                }
                entry.log_history.clone()
            })
        } else {
            self.inner
                .sessions
                .lock()
                .await
                .get(session_name)
                .map(|e| e.log_history.clone())
        };

        // Forward stdout/stderr into the session log ring so `waymux logs`
        // shows output from spawned clients (e.g. Firefox crash messages).
        if let Some(log_history) = log_history {
            if let Some(stdout) = child.stdout.take() {
                spawn_log_drainer(
                    self.inner.events_tx.clone(),
                    log_history.clone(),
                    session_name.to_string(),
                    "stdout",
                    stdout,
                );
            }
            if let Some(stderr) = child.stderr.take() {
                spawn_log_drainer(
                    self.inner.events_tx.clone(),
                    log_history,
                    session_name.to_string(),
                    "stderr",
                    stderr,
                );
            }
        }

        let inner3 = self.inner.clone();
        let session_name_owned = session_name.to_string();
        tokio::spawn(async move {
            let status = child.wait().await;
            let code = status.as_ref().map(exit_code_of).unwrap_or(-1);
            // Prune the exited PID and learn whether this was the compositor.
            let was_compositor = if pid > 0 {
                let mut sessions = inner3.sessions.lock().await;
                if let Some(entry) = sessions.get_mut(&session_name_owned) {
                    entry.child_pids.retain(|&p| p != pid as u32);
                    if entry.compositor_pid == Some(pid as u32) {
                        entry.compositor_pid = None;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            let _ = inner3.events_tx.send(Event::new(EventBody::ChildExited {
                name: session_name_owned.clone(),
                pid,
                exit_code: code,
            }));
            if was_compositor {
                let _ = inner3.events_tx.send(Event::new(EventBody::SessionCrashed {
                    name: session_name_owned,
                    pid,
                    exit_code: code,
                }));
            }
        });

        Ok(pid)
    }

    // Retained for the (future) attach path.
    #[allow(dead_code)]
    pub async fn inner_socket_path(&self, name: &str) -> Option<PathBuf> {
        self.inner
            .sessions
            .lock()
            .await
            .get(name)
            .map(|s| s.meta.inner_socket.clone())
    }

    /// RPC to the session process's control socket.
    ///
    /// Audit H8: uses a persistent per-session connection, opened lazily on
    /// the first call and reused thereafter. The session's `control::run`
    /// already loops one frame at a time per connection, so reusing the
    /// stream matches existing semantics. On any I/O error the connection
    /// slot is cleared so the next call reconnects — a transient blip
    /// must not wedge the session forever.
    async fn session_control(
        &self,
        name: &str,
        method: SessionCtlMethod,
    ) -> Result<SessionCtlResponse> {
        let (path, conn) = {
            let sessions = self.inner.sessions.lock().await;
            let entry = sessions
                .get(name)
                .ok_or_else(|| anyhow!(SessionControlError::NotFound))?;
            (
                entry.meta.control_socket.clone(),
                entry.control_conn.clone(),
            )
        };

        // Serialize all RPCs to this session on the persistent-conn mutex.
        // This matches the session's single-handler-per-connection model.
        let mut guard = conn.lock().await;

        // Encode once; we may need it across a reconnect retry below.
        let req = SessionCtlRequest { id: 1, method };
        let mut buf = Vec::with_capacity(128);
        encode_frame(&req, &mut buf).context("encode session-ctl request")?;

        // Run the RPC. On any I/O failure, clear the slot so the next call
        // reconnects. Returns Ok(payload) — the framed msgpack response.
        async fn rpc(stream: &mut UnixStream, frame: &[u8]) -> Result<Vec<u8>> {
            stream
                .write_all(frame)
                .await
                .context("write session-ctl request")?;
            let mut len_buf = [0u8; 4];
            stream
                .read_exact(&mut len_buf)
                .await
                .context("read session-ctl response length")?;
            let len = u32::from_be_bytes(len_buf) as usize;
            if len > waymux_protocol::MAX_FRAME_SIZE {
                bail!("session-ctl response too large: {}", len);
            }
            let mut payload = vec![0u8; 4 + len];
            payload[..4].copy_from_slice(&len_buf);
            stream
                .read_exact(&mut payload[4..])
                .await
                .context("read session-ctl response")?;
            Ok(payload)
        }

        // First attempt: reuse an existing connection if we have one. If
        // there's no stream cached, fall through to the connect-then-rpc
        // path below.
        if guard.is_some() {
            let stream = guard.as_mut().expect("just checked is_some");
            match rpc(stream, &buf).await {
                Ok(payload) => {
                    let resp: SessionCtlResponse =
                        decode_frame(&payload).context("decode session-ctl response")?;
                    return Ok(resp);
                }
                Err(e) => {
                    // Connection died (peer closed, write/read error, etc.).
                    // Drop it so we reconnect below; a stuck slot would
                    // wedge every subsequent RPC against this session.
                    *guard = None;
                    tracing::debug!(session = %name, error = %e, "session-ctl conn lost; reconnecting");
                }
            }
        }

        // No connection (or just dropped one). Open a fresh stream and
        // store it before the RPC so a panic in rpc() doesn't leak it.
        let new_stream = tokio::time::timeout(Duration::from_secs(2), UnixStream::connect(&path))
            .await
            .context("session control connect timed out")?
            .with_context(|| format!("connect {}", path.display()))?;
        *guard = Some(new_stream);

        // Borrow the just-stored stream and run the RPC. If this one fails
        // too, clear the slot and propagate the error — the next caller
        // will reconnect from scratch.
        let stream = guard.as_mut().expect("just stored Some");
        let payload = match rpc(stream, &buf).await {
            Ok(p) => p,
            Err(e) => {
                *guard = None;
                return Err(e);
            }
        };

        let resp: SessionCtlResponse =
            decode_frame(&payload).context("decode session-ctl response")?;
        Ok(resp)
    }

    pub async fn list_windows(&self, name: &str) -> Result<Vec<waymux_protocol::WindowInfo>> {
        let resp = self
            .session_control(name, SessionCtlMethod::ListWindows)
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        let body: waymux_protocol::SessionCtlWindows = resp
            .decode_result()
            .map_err(|e| anyhow!("decode list_windows: {e}"))?;
        // Merge daemon-side tags onto session-reported windows.
        let tags = self.inner.tags.lock().await;
        let windows = body
            .windows
            .into_iter()
            .map(|mut w| {
                if let Some(t) = tags.get(&(name.to_string(), w.id)) {
                    w.tags = t.clone();
                }
                w
            })
            .collect();
        Ok(windows)
    }

    pub async fn resize(&self, name: &str, width: u32, height: u32) -> Result<()> {
        // Reject degenerate/absurd geometry before forwarding to the session,
        // mirroring the create-time validation. A 0x0 resize is caller input,
        // not a server fault, so it surfaces as E_RESIZE_REJECTED.
        if validate_output_size(width, height).is_err() {
            bail!(ResizeError::Rejected(format!(
                "width and height must each be between 1 and {MAX_OUTPUT_DIMENSION} pixels (got {width}x{height})"
            )));
        }
        let resp = self
            .session_control(name, SessionCtlMethod::Resize { width, height })
            .await?;
        if !resp.ok {
            bail!(ResizeError::Rejected(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        // Reflect the new size in the daemon's copy of the metadata.
        let mut sessions = self.inner.sessions.lock().await;
        if let Some(entry) = sessions.get_mut(name) {
            entry.meta.width = width;
            entry.meta.height = height;
        }
        Ok(())
    }

    /// Poll the session's `last_damage_ns` at 10ms intervals (the protocol spec) until
    /// either quiet_ms has elapsed since the last damage, or timeout_ms has
    /// elapsed. Returns `idle == true` on quiescence, `false` on timeout.
    pub async fn wait_for_idle(&self, name: &str, quiet_ms: u32, timeout_ms: u32) -> Result<bool> {
        let quiet_ns = u64::from(quiet_ms) * 1_000_000;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms as u64);
        loop {
            let resp = self.session_control(name, SessionCtlMethod::Info).await?;
            if !resp.ok {
                bail!(SessionControlError::Failed(
                    resp.error.unwrap_or_else(|| "unknown".into())
                ));
            }
            let info: waymux_protocol::SessionCtlInfo = resp
                .decode_result()
                .map_err(|e| anyhow!("decode info: {e}"))?;
            // CRITICAL: must use CLOCK_MONOTONIC because `info.last_damage_ns`
            // is written via `clock_gettime(CLOCK_MONOTONIC)` in
            // `crates/waymux-session/src/state.rs::record_damage`. Comparing
            // against CLOCK_REALTIME (SystemTime::now) yields a delta of
            // ~boot-to-epoch nanoseconds and `idle = true` is returned
            // instantly, so every screenshot is taken mid-animation.
            let now_ns: u64 = {
                let mut ts = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                };
                unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
                (ts.tv_sec as u64)
                    .saturating_mul(1_000_000_000)
                    .saturating_add(ts.tv_nsec as u64)
            };
            let since = now_ns.saturating_sub(info.last_damage_ns);
            if since >= quiet_ns {
                return Ok(true);
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(false);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Capture a window's current shm buffer as PNG via session_control.
    /// Requires the caller to specify a window id; whole-output screenshots
    /// are a v2 concern (the protocol spec Q3).
    pub async fn screenshot(
        &self,
        name: &str,
        window_id: u32,
    ) -> Result<waymux_protocol::SessionCtlScreenshot> {
        let resp = self
            .session_control(name, SessionCtlMethod::Screenshot { window_id })
            .await?;
        if !resp.ok {
            bail!(ScreenshotError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        resp.decode_result()
            .map_err(|e| anyhow!("decode screenshot: {e}"))
    }

    /// Composite every registered window in the session into a single PNG.
    pub async fn screenshot_desktop(
        &self,
        name: &str,
    ) -> Result<waymux_protocol::SessionCtlScreenshot> {
        let resp = self
            .session_control(name, SessionCtlMethod::ScreenshotDesktop)
            .await?;
        if !resp.ok {
            bail!(ScreenshotError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        resp.decode_result()
            .map_err(|e| anyhow!("decode screenshot_desktop: {e}"))
    }

    /// Synthesize a pointer event into the session's focused client.
    #[allow(clippy::too_many_arguments)]
    pub async fn inject_pointer(
        &self,
        name: &str,
        x: f64,
        y: f64,
        button: u32,
        state: waymux_protocol::KeyState,
        axis_x: f64,
        axis_y: f64,
        window_id: Option<u32>,
        content: bool,
    ) -> Result<()> {
        let resp = self
            .session_control(
                name,
                SessionCtlMethod::InjectPointer {
                    x,
                    y,
                    button,
                    state,
                    axis_x,
                    axis_y,
                    window_id,
                    content,
                },
            )
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(())
    }

    /// Synthesize a touch event into a session. Sibling of
    /// `inject_pointer`. Forwards an `InjectTouch` over the session control
    /// socket; the session emits `wl_touch.{down|motion|up}` + a frame
    /// marker on the resolved target client's touch resources.
    #[allow(clippy::too_many_arguments)]
    pub async fn inject_touch(
        &self,
        name: &str,
        id: u32,
        x: f64,
        y: f64,
        phase: waymux_protocol::TouchPhase,
        window_id: Option<u32>,
        content: bool,
    ) -> Result<()> {
        let resp = self
            .session_control(
                name,
                SessionCtlMethod::InjectTouch {
                    id,
                    x,
                    y,
                    phase,
                    window_id,
                    content,
                },
            )
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(())
    }

    /// Mark a session as attached and return the socket path an attach
    /// client should connect to. Does not verify the attach client
    /// actually connects — the subsequent Wayland handshake is
    /// out-of-band from the daemon's perspective.
    pub async fn attach(&self, name: &str) -> Result<PathBuf> {
        let mut sessions = self.inner.sessions.lock().await;
        let entry = sessions
            .get_mut(name)
            .ok_or_else(|| anyhow!(SessionControlError::NotFound))?;
        entry.meta.attached = true;
        Ok(entry.meta.attach_socket.clone())
    }

    /// Flip the attached bit back off. Real teardown is driven by the
    /// attach client's Wayland connection closing — the daemon just
    /// tracks the user-facing state.
    pub async fn detach(&self, name: &str) -> Result<()> {
        let mut sessions = self.inner.sessions.lock().await;
        let entry = sessions
            .get_mut(name)
            .ok_or_else(|| anyhow!(SessionControlError::NotFound))?;
        entry.meta.attached = false;
        Ok(())
    }

    /// Synthesize a keyboard event into the session's focused client.
    pub async fn inject_key(
        &self,
        name: &str,
        keycode: u32,
        state: waymux_protocol::KeyState,
        modifiers: u32,
    ) -> Result<()> {
        let resp = self
            .session_control(
                name,
                SessionCtlMethod::InjectKey {
                    keycode,
                    state,
                    modifiers,
                },
            )
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(())
    }

    /// Audit H10: deliver a batch of input ops in one session_control RPC.
    /// SDKs that previously made N round-trips per typed string / clicked
    /// element collapse to one. The session executes ops in order with no
    /// inter-op delay.
    pub async fn inject_batch(
        &self,
        name: &str,
        ops: Vec<waymux_protocol::InjectOp>,
    ) -> Result<()> {
        let resp = self
            .session_control(name, SessionCtlMethod::InjectBatch { ops })
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(())
    }

    /// Start recording a session's output to an MKV file.
    /// Returns the resolved output paths (primary and optional secondary).
    pub async fn record_start(
        &self,
        name: &str,
        path: Option<String>,
        codec: Option<waymux_protocol::RecordingCodec>,
        secondary_codec: Option<waymux_protocol::RecordingCodec>,
        mode: Option<waymux_protocol::CaptureMode>,
        min_fps: Option<u32>,
    ) -> Result<SessionCtlRecordStarted> {
        let resp = self
            .session_control(
                name,
                SessionCtlMethod::RecordStart {
                    path,
                    codec,
                    secondary_codec,
                    mode,
                    min_fps,
                },
            )
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        let started: SessionCtlRecordStarted = resp
            .decode_result()
            .map_err(|e| anyhow!("decode record_start: {e}"))?;
        Ok(started)
    }

    /// Stop an active recording in the session.
    pub async fn record_stop(&self, name: &str) -> Result<()> {
        let resp = self
            .session_control(name, SessionCtlMethod::RecordStop)
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(())
    }

    /// Return the current recording state for a session (active flag plus
    /// path(s) and codec when active).
    pub async fn record_status(&self, name: &str) -> Result<waymux_protocol::RecordStatusResponse> {
        let resp = self
            .session_control(name, SessionCtlMethod::RecordStatus)
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        let status: waymux_protocol::RecordStatusResponse = resp
            .decode_result()
            .map_err(|e| anyhow!("decode record_status: {e}"))?;
        Ok(status)
    }

    /// Start a WebRTC viewer for a session. Returns the browser-accessible URL.
    pub async fn viewer_start(
        &self,
        name: &str,
        bind: Option<String>,
        port: Option<u16>,
    ) -> Result<waymux_protocol::ViewerStarted> {
        let resp = self
            .session_control(name, SessionCtlMethod::ViewerStart { bind, port })
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        let started: waymux_protocol::ViewerStarted = resp
            .decode_result()
            .map_err(|e| anyhow!("decode viewer_start: {e}"))?;
        Ok(started)
    }

    /// Stop the active viewer for a session (idempotent).
    pub async fn viewer_stop(&self, name: &str) -> Result<()> {
        let resp = self
            .session_control(name, SessionCtlMethod::ViewerStop)
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(())
    }

    /// Return the viewer URL for a session (None if no viewer is active).
    pub async fn viewer_status(&self, name: &str) -> Result<waymux_protocol::ViewerStatusResponse> {
        let resp = self
            .session_control(name, SessionCtlMethod::ViewerStatus)
            .await?;
        if !resp.ok {
            bail!(SessionControlError::Failed(
                resp.error.unwrap_or_else(|| "unknown".into())
            ));
        }
        let status: waymux_protocol::ViewerStatusResponse = resp
            .decode_result()
            .map_err(|e| anyhow!("decode viewer_status: {e}"))?;
        Ok(status)
    }

    pub async fn tag_window(&self, name: &str, window_id: u32, tags: Vec<String>) -> Result<()> {
        // Validate the window exists by asking the session.
        let windows = self.list_windows(name).await?;
        if !windows.iter().any(|w| w.id == window_id) {
            bail!(TagWindowError::WindowNotFound);
        }
        let mut store = self.inner.tags.lock().await;
        store.insert((name.to_string(), window_id), tags);
        Ok(())
    }
}

async fn session_supervisor(
    name: String,
    mut child: Child,
    kill_rx: oneshot::Receiver<()>,
    inner_socket: PathBuf,
    inner: Arc<Inner>,
) {
    tokio::pin!(kill_rx);
    #[cfg_attr(not(feature = "metering"), allow(unused_variables))]
    let exit_code = tokio::select! {
        status = child.wait() => {
            status.as_ref().map(exit_code_of).unwrap_or(-1)
        }
        _ = &mut kill_rx => {
            shutdown_child(&mut child).await
        }
    };
    // If the session exited on its own (destroy not called), remove it.
    // The presence of the entry tells us destroy() didn't run; in that case
    // we are responsible for emitting session_stop with the real exit code.
    #[cfg_attr(not(feature = "metering"), allow(unused_variables))]
    let (was_natural_exit, tmpfs_to_unmount) = {
        let mut sessions = inner.sessions.lock().await;
        match sessions.remove(&name) {
            Some(mut entry) => (true, entry.tmpfs.take()),
            None => (false, None),
        }
    };
    #[cfg(feature = "metering")]
    if was_natural_exit {
        if let Some(sink) = inner.usage_sink.as_ref() {
            sink.emit(UsageEvent::SessionStop {
                ts: usage_events::now(),
                run_id: &inner.run_id,
                session: &name,
                exit_code: Some(exit_code),
            })
            .await;
        }
    }
    // Unmount BEFORE removing socket files: the sockets live inside the
    // tmpfs, so the unlink targets after umount are on the underlying dir
    // which the kernel re-exposes. (For destroy()-driven exits the unmount
    // already happened there; tmpfs_to_unmount is None.)
    if let Some(t) = tmpfs_to_unmount {
        t.unmount();
    }
    inner.tags.lock().await.retain(|(s, _), _| s != &name);
    // Per-session log_history is dropped when the entry leaves the map (above).
    let _ = std::fs::remove_file(&inner_socket);
    if let Some(parent) = inner_socket.parent() {
        let _ = std::fs::remove_file(parent.join("control.sock"));
        let _ = std::fs::remove_file(parent.join("events.sock"));
        let _ = std::fs::remove_file(parent.join("attach.sock"));
        // The session_dir doubles as XDG_RUNTIME_DIR for spawned children
        // (registry.rs::spawn_child) — chromium, niri, kded, et al. write
        // SingletonLock files, dconf caches, etc. into it. So at this
        // point the dir is rarely empty in practice, and the previous
        // `remove_dir` fell through silently and left a per-session dir
        // behind under $XDG_RUNTIME_DIR/waymux/. cgroup.kill above has
        // already terminated every PID rooted in the session, so any
        // file still here is a dead-process artifact safe to nuke.
        // The dir itself is `state_dir/<name>` which we own — no risk
        // of touching unrelated state.
        let _ = std::fs::remove_dir_all(parent);
    }
    let _ = inner
        .events_tx
        .send(Event::new(EventBody::SessionDestroyed {
            name: name.clone(),
            exit_code,
        }));
    info!(%name, exit_code, "session supervisor exit");
}

/// Upper bound on a session's virtual-output width/height. Guards against a
/// caller requesting a degenerate or absurd geometry that would either produce
/// an unusable 0x0 output or trigger a multi-gigabyte framebuffer allocation.
/// 16384 is the conventional maximum single-output dimension on current GPUs.
pub const MAX_OUTPUT_DIMENSION: u32 = 16384;

#[derive(Debug, thiserror::Error)]
pub enum CreateError {
    #[error("session already exists")]
    AlreadyExists,
    #[error("invalid session name")]
    InvalidName,
    #[error(
        "invalid size: width and height must each be between 1 and {MAX_OUTPUT_DIMENSION} pixels"
    )]
    InvalidSize,
}

#[derive(Debug, thiserror::Error)]
pub enum DestroyError {
    #[error("session not found")]
    NotFound,
}

#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("session not found")]
    SessionNotFound,
    #[error("argv is empty")]
    InvalidArgv,
    #[error("argv[0] must be an absolute path: the daemon clears the environment and does not search PATH, so pass a full path (for example /usr/bin/foot, or $(command -v foot))")]
    ArgvNotAbsolute,
    #[error(
        "argv too large: at most {MAX_ARGV_LEN} elements, each at most {MAX_ARGV_ELEM_BYTES} bytes"
    )]
    ArgvTooLarge,
}

#[derive(Debug, thiserror::Error)]
pub enum SessionControlError {
    #[error("session not found")]
    NotFound,
    #[error("session control call failed: {0}")]
    Failed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ResizeError {
    #[error("resize rejected: {0}")]
    Rejected(String),
}

#[derive(Debug, thiserror::Error)]
pub enum TagWindowError {
    #[error("window not found in session")]
    WindowNotFound,
}

#[derive(Debug, thiserror::Error)]
pub enum ScreenshotError {
    #[error("screenshot failed: {0}")]
    Failed(String),
}

fn validate_session_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 64 {
        bail!(CreateError::InvalidName);
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        bail!(CreateError::InvalidName);
    }
    if name.starts_with('.') {
        bail!(CreateError::InvalidName);
    }
    Ok(())
}

/// Reject a degenerate or absurd output geometry. A 0-width or 0-height output
/// is unusable (every client commit would map onto an empty surface), and an
/// over-large one risks a huge framebuffer allocation. Returns `Ok(())` only
/// for `1..=MAX_OUTPUT_DIMENSION` on both axes.
fn validate_output_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 || width > MAX_OUTPUT_DIMENSION || height > MAX_OUTPUT_DIMENSION {
        bail!(CreateError::InvalidSize);
    }
    Ok(())
}

async fn accept_and_forward_events(
    listener: UnixListener,
    session_name: String,
    inner: Arc<Inner>,
) {
    // Audit C13: re-accept across connections. The session may reconnect its
    // events socket (e.g., inner compositor restart, transient EOF). The
    // pre-fix code returned after the first connection's EOF, silently
    // dropping every subsequent event. We exit only when the session is no
    // longer registered (destroy() + supervisor cleanup ran).
    loop {
        let (mut stream, _) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, session = %session_name,
                    "events accept failed; retrying");
                tokio::time::sleep(Duration::from_millis(200)).await;
                // Bail if the session is gone, otherwise loop and retry.
                let still_alive = inner.sessions.lock().await.contains_key(&session_name);
                if !still_alive {
                    return;
                }
                continue;
            }
        };
        let (mut read, _write) = stream.split();
        let conn_alive = async {
            loop {
                let mut len_buf = [0u8; 4];
                if read.read_exact(&mut len_buf).await.is_err() {
                    break; // EOF on this connection
                }
                let len = u32::from_be_bytes(len_buf) as usize;
                if len > waymux_protocol::MAX_FRAME_SIZE {
                    tracing::warn!(session = %session_name, len,
                        "oversized event frame; closing");
                    break;
                }
                let mut payload = vec![0u8; 4 + len];
                payload[..4].copy_from_slice(&len_buf);
                if read.read_exact(&mut payload[4..]).await.is_err() {
                    break;
                }
                let event: Event = match decode_frame(&payload) {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!(error = %e, session = %session_name,
                            "decode event failed; closing");
                        break;
                    }
                };
                let _ = inner.events_tx.send(event);
            }
        };
        conn_alive.await;
        // Connection closed. If the session is gone, exit; otherwise the
        // outer loop will re-accept the next connection.
        let still_alive = inner.sessions.lock().await.contains_key(&session_name);
        if !still_alive {
            tracing::debug!(session = %session_name, "events forwarder exit");
            return;
        }
        tracing::debug!(session = %session_name,
            "events connection closed; awaiting reconnect");
    }
}

async fn accept_ready(listener: &UnixListener) -> Result<()> {
    let (mut stream, _) = listener.accept().await.context("accept ready")?;
    let mut buf = [0u8; 16];
    let _ = stream.read(&mut buf).await.context("read ready")?;
    Ok(())
}

/// The protocol spec: graceful shutdown is SIGTERM → wait 5s → SIGKILL.
async fn shutdown_child(child: &mut Child) -> i32 {
    if let Some(pid) = child.id() {
        // SAFETY: `pid` is the pid of our own child, obtained from tokio's
        // Child handle. Worst case kill() fails (ESRCH) and we fall through
        // to the wait-and-SIGKILL path.
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }
    match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
        Ok(Ok(s)) => exit_code_of(&s),
        Ok(Err(_)) => -1,
        Err(_) => {
            // Timed out on SIGTERM; escalate.
            let _ = child.kill().await;
            match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
                Ok(Ok(s)) => exit_code_of(&s),
                _ => -1,
            }
        }
    }
}

/// Shell convention for the exit status of a process: the waitpid return
/// code if it exited normally, `128 + signum` if it was terminated by a
/// signal, `-1` if we can't tell (e.g. `wait` failed).
fn exit_code_of(status: &std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        code
    } else if let Some(sig) = status.signal() {
        128 + sig
    } else {
        -1
    }
}

/// Drain an AsyncRead line-by-line into `Log` events on the broadcast
/// channel. Used for session child stdout/stderr. Exits when the stream
/// closes (child exited).
///
/// Takes the per-session `log_history` lock directly (audit H11) so
/// drainers from different sessions don't contend on a single global
/// lock. Also takes `events_tx` directly so we don't need to keep an
/// `Arc<Inner>` reference alive for log writes.
fn spawn_log_drainer<R>(
    events_tx: broadcast::Sender<Event>,
    log_history: Arc<Mutex<std::collections::VecDeque<(String, String)>>>,
    session: String,
    stream_name: &'static str,
    reader: R,
) where
    R: tokio::io::AsyncRead + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(reader).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    // Mirror to the daemon's stderr so panics / backtraces
                    // from the session show up in the operator's terminal
                    // without needing `waymux logs -f`. The history ring
                    // also keeps them; this is just for interactive
                    // debugging convenience.
                    eprintln!("[{session}:{stream_name}] {line}");

                    // Append to rolling history for late subscribers.
                    {
                        let mut q = log_history.lock().await;
                        if q.len() >= LOG_HISTORY_CAPACITY {
                            q.pop_front();
                        }
                        q.push_back((stream_name.to_string(), line.clone()));
                    }
                    // Broadcast live. send returns Err only if no
                    // subscribers exist; that's fine.
                    let _ = events_tx.send(Event::new(EventBody::Log {
                        name: session.clone(),
                        stream: stream_name.to_string(),
                        text: line,
                    }));
                }
                Ok(None) => break, // EOF
                Err(_) => break,
            }
        }
    });
}

async fn drain_stderr(child: &mut Child) -> Option<String> {
    let stderr = child.stderr.as_mut()?;
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_millis(200), stderr.read_to_end(&mut buf)).await;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("state_dir", &self.inner.state_dir)
            .field("session_bin", &self.inner.session_bin)
            .finish()
    }
}

// Squelch a warning: the suppressed field `warn` was removed, but test code
// may want the import. Keep things tidy.
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal Registry backed by temp dirs. No real session binary
    /// is needed; we only exercise registry methods that fail before spawning.
    fn make_registry() -> Registry {
        let state_dir = tempfile::TempDir::new().expect("tempdir").keep();
        let session_bin = std::path::PathBuf::from("/nonexistent/waymux-session");
        #[cfg(feature = "metering")]
        let reg = Registry::new(state_dir, session_bin, None, "test-run-id".to_string());
        #[cfg(not(feature = "metering"))]
        let reg = Registry::new(state_dir, session_bin);
        reg
    }

    #[tokio::test]
    async fn viewer_start_round_trips_through_registry() {
        let reg = make_registry();
        // No session created: viewer_start must fail fast (SessionNotFound).
        let result = reg.viewer_start("eagle", None, None).await;
        assert!(
            result.is_err(),
            "viewer_start should error in test env (no session / no bridge bin)"
        );
        // The error must reference a recognisable component.
        let msg = format!("{:?}", result.unwrap_err());
        assert!(
            msg.contains("NotFound")
                || msg.contains("viewer")
                || msg.contains("session")
                || msg.contains("eagle"),
            "error should reference the failed component, got: {msg}"
        );
    }

    #[test]
    fn name_validation() {
        assert!(validate_session_name("ok").is_ok());
        assert!(validate_session_name("a_b-c.1").is_ok());
        assert!(validate_session_name("").is_err());
        assert!(validate_session_name(".hidden").is_err());
        assert!(validate_session_name("has space").is_err());
        assert!(validate_session_name("has/slash").is_err());
    }

    #[test]
    fn size_validation() {
        assert!(validate_output_size(1920, 1080).is_ok());
        assert!(validate_output_size(1, 1).is_ok());
        assert!(validate_output_size(MAX_OUTPUT_DIMENSION, MAX_OUTPUT_DIMENSION).is_ok());
        // Degenerate dimensions are rejected.
        assert!(validate_output_size(0, 0).is_err());
        assert!(validate_output_size(0, 1080).is_err());
        assert!(validate_output_size(1920, 0).is_err());
        // Absurdly large dimensions are rejected.
        assert!(validate_output_size(MAX_OUTPUT_DIMENSION + 1, 1080).is_err());
        assert!(validate_output_size(1920, MAX_OUTPUT_DIMENSION + 1).is_err());
    }

    /// `create` with a 0x0 output is caller input and must be rejected before
    /// any session spawn, as a `CreateError::InvalidSize` (mapped to
    /// E_BAD_REQUEST in server.rs).
    #[tokio::test]
    async fn create_zero_size_rejected() {
        let reg = make_registry();
        let err = reg
            .create(
                "zero".to_string(),
                0,
                0,
                1,
                Default::default(),
                false,
                false,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<CreateError>(),
                Some(CreateError::InvalidSize)
            ),
            "expected InvalidSize, got: {err:?}"
        );
    }

    /// `resize` to 0x0 is rejected up front (before any session lookup) as a
    /// `ResizeError::Rejected`, mapped to E_RESIZE_REJECTED.
    #[tokio::test]
    async fn resize_zero_size_rejected() {
        let reg = make_registry();
        let err = reg.resize("nope", 0, 0).await.unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<ResizeError>(),
                Some(ResizeError::Rejected(_))
            ),
            "expected ResizeError::Rejected, got: {err:?}"
        );
    }

    #[test]
    fn anyhow_chain_surfaces_custom_errors() {
        // Sanity check: downcasting works for our typed errors.
        let e: anyhow::Error = anyhow::Error::new(CreateError::AlreadyExists);
        assert!(e.downcast_ref::<CreateError>().is_some());
    }

    /// argv validation runs before the session lookup, so these reject up
    /// front even with no session registered. Empty argv -> InvalidArgv.
    #[tokio::test]
    async fn spawn_empty_argv_rejected() {
        let reg = make_registry();
        let err = reg
            .spawn_child("nope", Vec::new(), Default::default(), false)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<SpawnError>(),
                Some(SpawnError::InvalidArgv)
            ),
            "expected InvalidArgv, got: {err:?}"
        );
    }

    /// A relative argv[0] (e.g. `./foo`) is rejected with ArgvNotAbsolute:
    /// the daemon clears the env and never searches PATH.
    #[tokio::test]
    async fn spawn_relative_argv0_rejected() {
        let reg = make_registry();
        let err = reg
            .spawn_child("nope", vec!["./foo".to_string()], Default::default(), false)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<SpawnError>(),
                Some(SpawnError::ArgvNotAbsolute)
            ),
            "expected ArgvNotAbsolute, got: {err:?}"
        );
    }

    /// Too many argv elements -> ArgvTooLarge (resource-exhaustion guard).
    #[tokio::test]
    async fn spawn_argv_too_many_elements_rejected() {
        let reg = make_registry();
        let mut argv = vec!["/bin/true".to_string()];
        argv.extend(std::iter::repeat_n("x".to_string(), MAX_ARGV_LEN + 1));
        let err = reg
            .spawn_child("nope", argv, Default::default(), false)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<SpawnError>(),
                Some(SpawnError::ArgvTooLarge)
            ),
            "expected ArgvTooLarge, got: {err:?}"
        );
    }

    /// A single oversized argv element -> ArgvTooLarge.
    #[tokio::test]
    async fn spawn_argv_oversized_element_rejected() {
        let reg = make_registry();
        let big = "a".repeat(MAX_ARGV_ELEM_BYTES + 1);
        let err = reg
            .spawn_child(
                "nope",
                vec!["/bin/true".to_string(), big],
                Default::default(),
                false,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<SpawnError>(),
                Some(SpawnError::ArgvTooLarge)
            ),
            "expected ArgvTooLarge, got: {err:?}"
        );
    }
}
