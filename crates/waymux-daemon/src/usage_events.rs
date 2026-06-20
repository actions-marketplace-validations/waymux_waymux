// SPDX-License-Identifier: Apache-2.0

//! Optional append-only JSONL stream describing session lifecycle.
//! Compiled only under the non-default `metering` cargo feature.
//!
//! Three event kinds: `session_start`, `session_stop`, `session_heartbeat`.
//! Output is one JSON object per line, flushed after every write so a
//! consumer can tail it concurrently without lossy buffering.
//!
//! The daemon never speaks HTTP from this path: JSONL is the boundary.
//! The sink is optional; when `--usage-events-sink` is unset, all
//! `emit` calls are no-ops.

use serde::Serialize;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use time::OffsetDateTime;
use tokio::sync::Mutex;
use tracing::warn;

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageEvent<'a> {
    SessionStart {
        #[serde(with = "time::serde::rfc3339")]
        ts: OffsetDateTime,
        /// Per-process UUID generated once at daemon startup. The reporter
        /// joins on `(run_id, session)` so that two sessions sharing a
        /// name across daemon restarts don't merge billing windows. See
        /// audit C8.
        run_id: &'a str,
        session: &'a str,
        api_key_id: Option<&'a str>,
    },
    SessionStop {
        #[serde(with = "time::serde::rfc3339")]
        ts: OffsetDateTime,
        run_id: &'a str,
        session: &'a str,
        exit_code: Option<i32>,
    },
    SessionHeartbeat {
        #[serde(with = "time::serde::rfc3339")]
        ts: OffsetDateTime,
        run_id: &'a str,
        session: &'a str,
        api_key_id: Option<&'a str>,
    },
}

pub struct UsageEventSink {
    inner: Mutex<Box<dyn Write + Send>>,
}

impl UsageEventSink {
    pub fn open(path: &Path) -> anyhow::Result<Arc<Self>> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let writer: Box<dyn Write + Send> = Box::new(BufWriter::new(file));
        Ok(Arc::new(Self {
            inner: Mutex::new(writer),
        }))
    }

    pub async fn emit(&self, event: UsageEvent<'_>) {
        let line = match serde_json::to_string(&event) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "usage_events: serialize failed");
                return;
            }
        };
        let mut guard = self.inner.lock().await;
        if let Err(e) = writeln!(*guard, "{line}").and_then(|_| guard.flush()) {
            warn!(error = %e, "usage_events: write failed");
        }
    }
}

pub fn now() -> OffsetDateTime {
    OffsetDateTime::now_utc()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct VecWriter(Arc<std::sync::Mutex<Vec<u8>>>);
    impl Write for VecWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn emits_one_line_per_event() {
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = UsageEventSink {
            inner: Mutex::new(Box::new(VecWriter(buf.clone()))),
        };
        let ts = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        sink.emit(UsageEvent::SessionStart {
            ts,
            run_id: "test-run-id",
            session: "demo",
            api_key_id: Some("aaaa1111-bbbb-2222-cccc-333344445555"),
        })
        .await;
        sink.emit(UsageEvent::SessionStop {
            ts,
            run_id: "test-run-id",
            session: "demo",
            exit_code: Some(0),
        })
        .await;
        sink.emit(UsageEvent::SessionHeartbeat {
            ts,
            run_id: "test-run-id",
            session: "demo",
            api_key_id: Some("aaaa1111-bbbb-2222-cccc-333344445555"),
        })
        .await;
        let bytes = buf.lock().unwrap().clone();
        let lines: Vec<serde_json::Value> = std::str::from_utf8(&bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["kind"], "session_start");
        assert_eq!(lines[0]["session"], "demo");
        assert_eq!(lines[0]["run_id"], "test-run-id");
        assert_eq!(
            lines[0]["api_key_id"],
            "aaaa1111-bbbb-2222-cccc-333344445555"
        );
        assert_eq!(lines[1]["kind"], "session_stop");
        assert_eq!(lines[1]["run_id"], "test-run-id");
        assert_eq!(lines[1]["exit_code"], 0);
        assert_eq!(lines[2]["kind"], "session_heartbeat");
        assert_eq!(lines[2]["run_id"], "test-run-id");
    }
}
