// SPDX-License-Identifier: Apache-2.0

//! Session→daemon event push.
//!
//! The session emits `Event` frames (per waymux-protocol) to an events
//! socket the daemon binds at session-create time. A tokio task drains an
//! mpsc channel and writes framed msgpack onto the socket. The compositor
//! thread (std::thread) pushes events via
//! [`EventSink::emit`], which uses [`tokio::sync::mpsc::Sender::blocking_send`].

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tracing::{debug, warn};
use waymux_protocol::{encode_frame, Event, EventBody};

const QUEUE_CAPACITY: usize = 256;

#[derive(Clone)]
pub struct EventSink {
    inner: Arc<EventSinkInner>,
}

struct EventSinkInner {
    session_name: String,
    tx: mpsc::Sender<EventBody>,
}

impl EventSink {
    pub fn new(session_name: String, tx: mpsc::Sender<EventBody>) -> Self {
        Self {
            inner: Arc::new(EventSinkInner { session_name, tx }),
        }
    }

    /// Send an event body to the daemon.
    ///
    /// Called from the compositor thread. Drops the event silently if the
    /// channel is full (daemon is slow) — losing a window update is better
    /// than blocking the Wayland event loop. The daemon's broadcast layer
    /// has its own backpressure handling for subscribers.
    pub fn emit(&self, body: EventBody) {
        if let Err(e) = self.inner.tx.try_send(body) {
            debug!(error = %e, "event channel full or closed; dropping event");
        }
    }

    pub fn session_name(&self) -> &str {
        &self.inner.session_name
    }
}

/// Connect to the daemon's events socket and drain `rx` onto it until the
/// socket closes or the channel is exhausted.
pub async fn run(
    path: &Path,
    mut rx: mpsc::Receiver<EventBody>,
    on_disconnect: tokio::sync::oneshot::Sender<()>,
) -> Result<()> {
    let mut stream = UnixStream::connect(path)
        .await
        .with_context(|| format!("connect events socket {}", path.display()))?;
    let mut buf = Vec::with_capacity(256);
    while let Some(body) = rx.recv().await {
        buf.clear();
        let event = Event::new(body);
        if let Err(e) = encode_frame(&event, &mut buf) {
            warn!(error = %e, "encode event frame failed; dropping");
            continue;
        }
        if let Err(e) = stream.write_all(&buf).await {
            warn!(error = %e, "events socket write failed; daemon probably gone");
            break;
        }
    }
    let _ = stream.shutdown().await;
    let _ = on_disconnect.send(());
    Ok(())
}

/// Helper: create an unbounded-ish mpsc pair suitable for session event flow.
pub fn channel() -> (mpsc::Sender<EventBody>, mpsc::Receiver<EventBody>) {
    mpsc::channel(QUEUE_CAPACITY)
}
