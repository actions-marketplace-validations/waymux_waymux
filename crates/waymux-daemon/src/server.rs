// SPDX-License-Identifier: Apache-2.0

//! Control-socket accept loop and per-connection handler.
//!
//! Each connection is split into read and write halves:
//!
//! - The **reader task** parses requests, dispatches to the registry, and
//!   sends responses to the writer via an mpsc.
//! - The **writer task** drains the mpsc and frames each message onto the
//!   socket. Responses and server-pushed events share the same queue, so
//!   ordering on the wire is simple FIFO.
//! - The **event forwarder task** (spawned on `subscribe`) subscribes to the
//!   registry's broadcast channel, filters by topic, and enqueues events
//!   onto the same mpsc. One per connection; replaces the previous forwarder
//!   if `subscribe` is called again.
//!
//! A connection terminates when the reader returns EOF/error or the writer
//! fails to flush. All child tasks are then aborted or drained.

use crate::backend::{CreateRequest, SessionBackend};
use crate::registry::{
    CreateError, DestroyError, Registry, ResizeError, ScreenshotError, SessionControlError,
    SpawnError, TagWindowError,
};
use anyhow::{Context, Result};
use serde::Serialize;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use waymux_protocol::{
    decode_frame, encode_frame, AttachResult, ErrorCode, Event, EventBody, HelloResult, Request,
    RequestMethod, Response, ResponseError, CURRENT_PROTOCOL_VERSION,
};

const MAX_FRAME_SIZE: usize = waymux_protocol::MAX_FRAME_SIZE;
const OUT_QUEUE_CAPACITY: usize = 1024;

/// Run the control-socket accept loop.
///
/// `registry` backs every per-session op (spawn/inject/screenshot/record/
/// viewer/tag/list/…). `backend` owns the session-LIFECYCLE ops: the
/// `create_session` and `destroy_session` dispatch arms route through
/// `SessionBackend::create` / `::destroy` instead of calling the registry
/// directly. For the local path `backend` is a `LocalBackend` wrapping THIS
/// SAME registry, so a created session is visible to every other op exactly as
/// before; the abstraction is the seam a future non-local backend would use to
/// swap the lifecycle target without touching the per-session dispatch arms.
pub async fn run(
    listener: UnixListener,
    registry: Registry,
    backend: Arc<dyn SessionBackend>,
) -> Result<()> {
    let daemon_uid = unsafe { libc::getuid() };
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "accept failed");
                continue;
            }
        };

        // SO_PEERCRED: same-uid only.
        match stream.peer_cred() {
            Ok(cred) if cred.uid() == daemon_uid => {}
            Ok(cred) => {
                warn!(
                    peer_uid = cred.uid(),
                    "rejecting connection from foreign uid"
                );
                continue;
            }
            Err(e) => {
                warn!(error = %e, "peer_cred failed; closing connection");
                continue;
            }
        }

        let reg = registry.clone();
        let backend = Arc::clone(&backend);
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, reg, backend).await {
                debug!(error = %e, "connection closed with error");
            }
        });
    }
}

enum Outgoing {
    Response(Response),
    Event(Event),
}

async fn handle_connection(
    stream: UnixStream,
    registry: Registry,
    backend: Arc<dyn SessionBackend>,
) -> Result<()> {
    let (read_half, write_half) = stream.into_split();
    let (out_tx, out_rx) = mpsc::channel::<Outgoing>(OUT_QUEUE_CAPACITY);

    let writer = tokio::spawn(writer_task(write_half, out_rx));
    let result = reader_task(read_half, out_tx.clone(), registry, backend).await;
    drop(out_tx); // signal writer to exit cleanly
    let _ = writer.await;
    result
}

async fn writer_task(mut w: OwnedWriteHalf, mut rx: mpsc::Receiver<Outgoing>) {
    let mut buf = Vec::with_capacity(256);
    while let Some(msg) = rx.recv().await {
        buf.clear();
        let res = match &msg {
            Outgoing::Response(r) => encode_frame(r, &mut buf),
            Outgoing::Event(e) => encode_frame(e, &mut buf),
        };
        if let Err(e) = res {
            warn!(error = %e, "encode failed; dropping connection");
            break;
        }
        if w.write_all(&buf).await.is_err() {
            break;
        }
    }
    let _ = w.shutdown().await;
}

async fn reader_task(
    mut r: OwnedReadHalf,
    out_tx: mpsc::Sender<Outgoing>,
    registry: Registry,
    backend: Arc<dyn SessionBackend>,
) -> Result<()> {
    let mut negotiated = false;
    let mut forwarder: Option<JoinHandle<()>> = None;

    loop {
        let frame = match read_frame(&mut r).await? {
            Some(f) => f,
            None => break, // clean EOF
        };
        let request: Request = match decode_frame(&frame) {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "malformed request; closing connection");
                break;
            }
        };

        let response = if !negotiated {
            handle_hello_gate(&request, &mut negotiated)
        } else {
            dispatch(
                request.id,
                request.method,
                &registry,
                backend.as_ref(),
                &out_tx,
                &mut forwarder,
            )
            .await
        };

        if out_tx.send(Outgoing::Response(response)).await.is_err() {
            // Writer is gone — peer hung up.
            break;
        }
    }

    if let Some(h) = forwarder {
        h.abort();
    }
    Ok(())
}

fn handle_hello_gate(request: &Request, negotiated: &mut bool) -> Response {
    match &request.method {
        RequestMethod::Hello { client_protocol } => {
            if *client_protocol == 0 || *client_protocol > CURRENT_PROTOCOL_VERSION {
                Response::failure(
                    request.id,
                    ResponseError::new(
                        ErrorCode::ProtoVersion,
                        format!(
                            "client protocol {} not supported (server: {})",
                            client_protocol, CURRENT_PROTOCOL_VERSION
                        ),
                    ),
                )
            } else {
                *negotiated = true;
                let result = HelloResult {
                    server_protocol: CURRENT_PROTOCOL_VERSION,
                    capabilities: vec!["subscribe".into(), "spawn".into()],
                };
                Response::success(request.id, &result)
                    .unwrap_or_else(|e| internal_err(request.id, e))
            }
        }
        _ => Response::failure(
            request.id,
            ResponseError::new(ErrorCode::ProtoVersion, "first request must be hello"),
        ),
    }
}

async fn dispatch(
    id: u32,
    method: RequestMethod,
    registry: &Registry,
    backend: &dyn SessionBackend,
    out_tx: &mpsc::Sender<Outgoing>,
    forwarder: &mut Option<JoinHandle<()>>,
) -> Response {
    match method {
        RequestMethod::Hello { .. } => Response::failure(
            id,
            ResponseError::new(ErrorCode::ProtoVersion, "hello already completed"),
        ),

        RequestMethod::ListSessions => {
            let sessions = registry.list().await;
            Response::success(id, &sessions).unwrap_or_else(|e| internal_err(id, e))
        }

        RequestMethod::CreateSession(params) => {
            // Session-lifecycle op: route through the backend. For the local
            // path `backend` is a `LocalBackend` wrapping the same `registry`,
            // so this delegates to `registry.create(...)` and the result is the
            // identical `CreateSessionResult`. `codec`/`gpu_type` stay advisory
            // (unused here, exactly as before this was wired).
            let req = CreateRequest {
                name: params.name,
                width: params.width,
                height: params.height,
                scale: params.scale,
                env: params.env,
                share_clipboard: params.share_clipboard,
                share_audio: params.share_audio,
                mem_cap_mb: params.mem_cap_mb,
                cpu_cap_pct: params.cpu_cap_pct,
                disk_quota_mb: params.disk_quota_mb,
                fd_limit: params.fd_limit,
                api_key_id: params.api_key_id,
            };
            match backend.create(req).await {
                Ok(result) => {
                    info!(name = %result.name, "create_session ok");
                    Response::success(id, &result).unwrap_or_else(|e| internal_err(id, e))
                }
                Err(e) => Response::failure(id, map_create_err(&e)),
            }
        }

        RequestMethod::Attach { name } => match registry.attach(&name).await {
            Ok(path) => Response::success(
                id,
                &AttachResult {
                    attach_socket_path: path.display().to_string(),
                },
            )
            .unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::Detach { name } => match registry.detach(&name).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        // Session-lifecycle op: route through the backend. For the local path
        // this delegates to `registry.destroy(&name)` on the shared registry.
        RequestMethod::DestroySession { name } => match backend.destroy(name).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_destroy_err(&e)),
        },

        RequestMethod::Subscribe { topics } => {
            // Replace any previous forwarder on this connection.
            if let Some(h) = forwarder.take() {
                h.abort();
            }
            // Subscribe BEFORE replay so live events after replay don't
            // slip through the gap.
            let rx = registry.subscribe_events();
            // Replay any retained log history for the requested sessions
            // (the protocol spec). Each `logs:<name>` or bare `logs` topic pulls the
            // matching session's rolling buffer onto the wire.
            for topic in &topics {
                if let Some(session) = topic.strip_prefix("logs:") {
                    for (stream, text) in registry.replay_logs(session).await {
                        let ev = waymux_protocol::Event::new(waymux_protocol::EventBody::Log {
                            name: session.to_string(),
                            stream,
                            text,
                        });
                        if out_tx.send(Outgoing::Event(ev)).await.is_err() {
                            break;
                        }
                    }
                }
                // Bare `logs` topic: replay every session's history.
                if topic == "logs" {
                    let sessions: Vec<String> =
                        registry.list().await.into_iter().map(|s| s.name).collect();
                    for name in sessions {
                        for (stream, text) in registry.replay_logs(&name).await {
                            let ev = waymux_protocol::Event::new(waymux_protocol::EventBody::Log {
                                name: name.clone(),
                                stream,
                                text,
                            });
                            if out_tx.send(Outgoing::Event(ev)).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            }
            let sender = out_tx.clone();
            *forwarder = Some(tokio::spawn(async move {
                event_forwarder(rx, topics, sender).await;
            }));
            Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e))
        }

        RequestMethod::Spawn {
            name,
            argv,
            env,
            compositor,
        } => match registry.spawn_child(&name, argv, env, compositor).await {
            Ok(pid) => {
                Response::success(id, &SpawnResult { pid }).unwrap_or_else(|e| internal_err(id, e))
            }
            Err(e) => Response::failure(id, map_spawn_err(&e)),
        },

        RequestMethod::ListWindows { name } => match registry.list_windows(&name).await {
            Ok(windows) => Response::success(id, &windows).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::Resize {
            name,
            width,
            height,
        } => match registry.resize(&name, width, height).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_resize_err(&e)),
        },

        RequestMethod::TagWindow {
            name,
            window_id,
            tags,
        } => match registry.tag_window(&name, window_id, tags).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_tag_err(&e)),
        },

        RequestMethod::WaitForIdle {
            name,
            quiet_ms,
            timeout_ms,
        } => match registry.wait_for_idle(&name, quiet_ms, timeout_ms).await {
            Ok(idle) => Response::success(id, &WaitForIdleResult { idle })
                .unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::InjectKey {
            name,
            keycode,
            state,
            modifiers,
        } => match registry.inject_key(&name, keycode, state, modifiers).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::InjectPointer {
            name,
            x,
            y,
            button,
            state,
            axis_x,
            axis_y,
            window_id,
            content,
        } => match registry
            .inject_pointer(
                &name, x, y, button, state, axis_x, axis_y, window_id, content,
            )
            .await
        {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::InjectBatch { name, ops } => match registry.inject_batch(&name, ops).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::InjectTouch {
            name,
            id: touch_id,
            x,
            y,
            phase,
            window_id,
            content,
        } => match registry
            .inject_touch(&name, touch_id, x, y, phase, window_id, content)
            .await
        {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::RecordStart {
            name,
            path,
            codec,
            secondary_codec,
            mode,
            min_fps,
        } => {
            match registry
                .record_start(&name, path, codec, secondary_codec, mode, min_fps)
                .await
            {
                Ok(started) => {
                    Response::success(id, &started).unwrap_or_else(|e| internal_err(id, e))
                }
                Err(e) => Response::failure(id, map_session_ctl_err(&e)),
            }
        }

        RequestMethod::RecordStop { name } => match registry.record_stop(&name).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::RecordStatus { name } => match registry.record_status(&name).await {
            Ok(status) => Response::success(id, &status).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::ViewerStart {
            session,
            bind,
            port,
        } => match registry.viewer_start(&session, bind, port).await {
            Ok(started) => Response::success(id, &started).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::ViewerStop { session } => match registry.viewer_stop(&session).await {
            Ok(()) => Response::success(id, &UnitResult {}).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::ViewerStatus { session } => match registry.viewer_status(&session).await {
            Ok(status) => Response::success(id, &status).unwrap_or_else(|e| internal_err(id, e)),
            Err(e) => Response::failure(id, map_session_ctl_err(&e)),
        },

        RequestMethod::Screenshot {
            name,
            window_id,
            format: _,
        } => {
            let wid = match window_id {
                Some(w) => w,
                None => {
                    return Response::failure(
                        id,
                        ResponseError::new(
                            ErrorCode::NotImplemented,
                            "whole-output screenshots are a v2 concern (the protocol spec Q3)",
                        ),
                    );
                }
            };
            match registry.screenshot(&name, wid).await {
                Ok(s) => Response::success(id, &s).unwrap_or_else(|e| internal_err(id, e)),
                Err(e) => Response::failure(id, map_screenshot_err(&e)),
            }
        }

        RequestMethod::ScreenshotDesktop { name, format: _ } => {
            match registry.screenshot_desktop(&name).await {
                Ok(s) => Response::success(id, &s).unwrap_or_else(|e| internal_err(id, e)),
                Err(e) => Response::failure(id, map_screenshot_err(&e)),
            }
        }

        // Reserved protocol slot with no engine implementation. Not a gap a
        // caller hits today: the CLI and MCP resolve a target with
        // `windows`/`wait` and inject with an explicit window_id. Give a
        // clear, actionable message instead of the generic catch-all.
        RequestMethod::InjectSelector { .. } => Response::failure(
            id,
            ResponseError::new(
                ErrorCode::NotImplemented,
                "inject_selector is a reserved slot: resolve the target with `windows` \
                 (or `wait`) and inject with an explicit window_id",
            ),
        ),

        other => Response::failure(
            id,
            ResponseError::new(
                ErrorCode::NotImplemented,
                format!("{} not implemented yet", method_name(&other)),
            ),
        ),
    }
}

#[derive(Serialize)]
struct UnitResult {}

#[derive(Serialize)]
struct SpawnResult {
    pid: i32,
}

#[derive(Serialize)]
struct WaitForIdleResult {
    idle: bool,
}

fn method_name(m: &RequestMethod) -> &'static str {
    match m {
        RequestMethod::Hello { .. } => "hello",
        RequestMethod::ListSessions => "list_sessions",
        RequestMethod::CreateSession(_) => "create_session",
        RequestMethod::DestroySession { .. } => "destroy_session",
        RequestMethod::Attach { .. } => "attach",
        RequestMethod::Detach { .. } => "detach",
        RequestMethod::Resize { .. } => "resize",
        RequestMethod::Spawn { .. } => "spawn",
        RequestMethod::ListWindows { .. } => "list_windows",
        RequestMethod::TagWindow { .. } => "tag_window",
        RequestMethod::Screenshot { .. } => "screenshot",
        RequestMethod::ScreenshotDesktop { .. } => "screenshot_desktop",
        RequestMethod::WaitForIdle { .. } => "wait_for_idle",
        RequestMethod::Subscribe { .. } => "subscribe",
        RequestMethod::InjectKey { .. } => "inject_key",
        RequestMethod::InjectPointer { .. } => "inject_pointer",
        RequestMethod::InjectBatch { .. } => "inject_batch",
        // Reserved wire slot; handler not yet implemented. Until then the
        // dispatcher's `other =>` catch-all returns ErrorCode::NotImplemented
        // via this name.
        RequestMethod::InjectSelector { .. } => "inject_selector",
        // Wired to `registry.inject_touch` above; this entry is kept for the
        // structured-logging method_name fall-through.
        RequestMethod::InjectTouch { .. } => "inject_touch",
        RequestMethod::RecordStart { .. } => "record_start",
        RequestMethod::RecordStop { .. } => "record_stop",
        RequestMethod::RecordStatus { .. } => "record_status",
        RequestMethod::StreamLogs { .. } => "stream_logs",
        RequestMethod::ViewerStart { .. } => "viewer_start",
        RequestMethod::ViewerStop { .. } => "viewer_stop",
        RequestMethod::ViewerStatus { .. } => "viewer_status",
    }
}

fn map_create_err(e: &anyhow::Error) -> ResponseError {
    if let Some(c) = e.downcast_ref::<CreateError>() {
        match c {
            CreateError::AlreadyExists => {
                ResponseError::new(ErrorCode::AlreadyExists, c.to_string())
            }
            // An invalid session name is caller-input, not a server fault.
            CreateError::InvalidName => ResponseError::new(ErrorCode::BadRequest, c.to_string()),
            // A degenerate/absurd output size is also caller-input.
            CreateError::InvalidSize => ResponseError::new(ErrorCode::BadRequest, c.to_string()),
        }
    } else {
        ResponseError::new(ErrorCode::Internal, e.to_string())
    }
}

fn map_destroy_err(e: &anyhow::Error) -> ResponseError {
    if let Some(c) = e.downcast_ref::<DestroyError>() {
        match c {
            DestroyError::NotFound => ResponseError::new(ErrorCode::NotFound, c.to_string()),
        }
    } else {
        ResponseError::new(ErrorCode::Internal, e.to_string())
    }
}

fn map_spawn_err(e: &anyhow::Error) -> ResponseError {
    if let Some(c) = e.downcast_ref::<SpawnError>() {
        match c {
            SpawnError::SessionNotFound => ResponseError::new(ErrorCode::NotFound, c.to_string()),
            // The argv-validation arms are all caller-input errors, so they map
            // to E_BAD_REQUEST (not E_INTERNAL, which is reserved for server
            // faults). The descriptive message is carried through so the cause
            // stays visible.
            SpawnError::InvalidArgv => ResponseError::new(ErrorCode::BadRequest, c.to_string()),
            SpawnError::ArgvNotAbsolute => ResponseError::new(ErrorCode::BadRequest, c.to_string()),
            SpawnError::ArgvTooLarge => ResponseError::new(ErrorCode::BadRequest, c.to_string()),
        }
    } else {
        ResponseError::new(ErrorCode::Internal, e.to_string())
    }
}

fn map_session_ctl_err(e: &anyhow::Error) -> ResponseError {
    if let Some(c) = e.downcast_ref::<SessionControlError>() {
        return match c {
            SessionControlError::NotFound => ResponseError::new(ErrorCode::NotFound, c.to_string()),
            SessionControlError::Failed(_) => {
                ResponseError::new(ErrorCode::Internal, c.to_string())
            }
        };
    }
    ResponseError::new(ErrorCode::Internal, e.to_string())
}

fn map_resize_err(e: &anyhow::Error) -> ResponseError {
    if let Some(c) = e.downcast_ref::<SessionControlError>() {
        if matches!(c, SessionControlError::NotFound) {
            return ResponseError::new(ErrorCode::NotFound, c.to_string());
        }
    }
    if let Some(c) = e.downcast_ref::<ResizeError>() {
        return ResponseError::new(ErrorCode::ResizeRejected, c.to_string());
    }
    ResponseError::new(ErrorCode::Internal, e.to_string())
}

fn map_screenshot_err(e: &anyhow::Error) -> ResponseError {
    if let Some(c) = e.downcast_ref::<SessionControlError>() {
        if matches!(c, SessionControlError::NotFound) {
            return ResponseError::new(ErrorCode::NotFound, c.to_string());
        }
    }
    if let Some(c) = e.downcast_ref::<ScreenshotError>() {
        // A missing buffer / unknown window id surfaces as "not found" for the
        // purposes of client retry.
        return ResponseError::new(ErrorCode::NotFound, c.to_string());
    }
    ResponseError::new(ErrorCode::Internal, e.to_string())
}

fn map_tag_err(e: &anyhow::Error) -> ResponseError {
    if let Some(c) = e.downcast_ref::<SessionControlError>() {
        if matches!(c, SessionControlError::NotFound) {
            return ResponseError::new(ErrorCode::NotFound, c.to_string());
        }
    }
    if let Some(c) = e.downcast_ref::<TagWindowError>() {
        return ResponseError::new(ErrorCode::NotFound, c.to_string());
    }
    ResponseError::new(ErrorCode::Internal, e.to_string())
}

fn internal_err(id: u32, err: impl std::fmt::Display) -> Response {
    Response::failure(id, ResponseError::new(ErrorCode::Internal, err.to_string()))
}

async fn event_forwarder(
    mut rx: broadcast::Receiver<Event>,
    topics: Vec<String>,
    tx: mpsc::Sender<Outgoing>,
) {
    loop {
        match rx.recv().await {
            Ok(event) => {
                if topic_matches(&event, &topics) && tx.send(Outgoing::Event(event)).await.is_err()
                {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(
                    lagged = n,
                    "subscriber lagged; dropping connection (the protocol spec backpressure)"
                );
                // Close our side; the writer task will drain and exit; the
                // reader task sees the mpsc close on next send and exits too.
                break;
            }
        }
    }
}

fn topic_matches(event: &Event, topics: &[String]) -> bool {
    if topics.is_empty() {
        return false;
    }
    match &event.body {
        EventBody::SessionCreated { .. }
        | EventBody::SessionDestroyed { .. }
        | EventBody::SessionCrashed { .. }
        | EventBody::Occluded { .. }
        | EventBody::ChildExited { .. } => topics.iter().any(|t| t == "sessions"),
        EventBody::WindowCreated { .. }
        | EventBody::WindowDestroyed { .. }
        | EventBody::WindowChanged { .. } => topics.iter().any(|t| t == "windows"),
        // Audit H9: avoid `format!` allocation per event per subscriber.
        // At 30 sessions × 60 fps × 10 subscribers = 18 000 allocs/sec.
        // `strip_prefix` does the same comparison without a heap alloc.
        EventBody::Damage { name, .. } => topics.iter().any(|t| {
            t == "damage"
                || t.strip_prefix("damage:")
                    .is_some_and(|suffix| suffix == name)
        }),
        EventBody::Log { name, .. } => topics
            .iter()
            .any(|t| t == "logs" || t.strip_prefix("logs:").is_some_and(|suffix| suffix == name)),
    }
}

async fn read_frame(r: &mut OwnedReadHalf) -> Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e).context("read frame length"),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME_SIZE {
        anyhow::bail!("frame length {} exceeds {}", len, MAX_FRAME_SIZE);
    }
    let mut payload = vec![0u8; 4 + len];
    payload[..4].copy_from_slice(&len_buf);
    r.read_exact(&mut payload[4..])
        .await
        .context("read frame payload")?;
    Ok(Some(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(body: EventBody) -> Event {
        Event::new(body)
    }

    #[test]
    fn empty_topic_list_matches_nothing() {
        let e = ev(EventBody::SessionCreated { name: "a".into() });
        assert!(!topic_matches(&e, &[]));
    }

    #[test]
    fn sessions_topic_matches_lifecycle_events() {
        let topics = vec!["sessions".to_string()];
        assert!(topic_matches(
            &ev(EventBody::SessionCreated { name: "a".into() }),
            &topics
        ));
        assert!(topic_matches(
            &ev(EventBody::SessionDestroyed {
                name: "a".into(),
                exit_code: 0
            }),
            &topics
        ));
        assert!(topic_matches(
            &ev(EventBody::ChildExited {
                name: "a".into(),
                pid: 1,
                exit_code: 0
            }),
            &topics
        ));
        assert!(!topic_matches(
            &ev(EventBody::Damage {
                name: "a".into(),
                serial: 1,
                timestamp_ns: 1
            }),
            &topics
        ));
    }

    #[test]
    fn damage_topic_is_session_scoped() {
        let e = ev(EventBody::Damage {
            name: "foo".into(),
            serial: 1,
            timestamp_ns: 1,
        });
        assert!(topic_matches(&e, &["damage:foo".to_string()]));
        assert!(!topic_matches(&e, &["damage:bar".to_string()]));
        assert!(topic_matches(&e, &["damage".to_string()]));
    }

    #[test]
    fn logs_topic_is_session_scoped() {
        let e = ev(EventBody::Log {
            name: "foo".into(),
            stream: "stdout".into(),
            text: "hi".into(),
        });
        assert!(topic_matches(&e, &["logs:foo".to_string()]));
        assert!(!topic_matches(&e, &["logs:bar".to_string()]));
        assert!(topic_matches(&e, &["logs".to_string()]));
        assert!(!topic_matches(&e, &["sessions".to_string()]));
    }

    #[test]
    fn caller_input_argv_errors_map_to_bad_request() {
        // Caller-input validation failures must surface as E_BAD_REQUEST,
        // not E_INTERNAL (which is reserved for server faults).
        for err in [
            SpawnError::InvalidArgv,
            SpawnError::ArgvNotAbsolute,
            SpawnError::ArgvTooLarge,
        ] {
            let mapped = map_spawn_err(&anyhow::Error::new(err));
            assert_eq!(mapped.code, ErrorCode::BadRequest);
        }
        // A genuine not-found is NOT a bad request.
        let nf = map_spawn_err(&anyhow::Error::new(SpawnError::SessionNotFound));
        assert_eq!(nf.code, ErrorCode::NotFound);
    }

    #[test]
    fn invalid_session_name_maps_to_bad_request() {
        let mapped = map_create_err(&anyhow::Error::new(CreateError::InvalidName));
        assert_eq!(mapped.code, ErrorCode::BadRequest);
        // AlreadyExists keeps its own code.
        let exists = map_create_err(&anyhow::Error::new(CreateError::AlreadyExists));
        assert_eq!(exists.code, ErrorCode::AlreadyExists);
    }

    #[test]
    fn invalid_session_size_maps_to_bad_request() {
        // A degenerate/absurd output size is caller-input, not a server fault.
        let mapped = map_create_err(&anyhow::Error::new(CreateError::InvalidSize));
        assert_eq!(mapped.code, ErrorCode::BadRequest);
    }

    #[test]
    fn rejected_resize_maps_to_resize_rejected() {
        let mapped = map_resize_err(&anyhow::Error::new(ResizeError::Rejected(
            "0x0 is degenerate".to_string(),
        )));
        assert_eq!(mapped.code, ErrorCode::ResizeRejected);
    }
}
