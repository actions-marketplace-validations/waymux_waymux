// SPDX-License-Identifier: Apache-2.0

//! Local Unix-socket connection to `waymuxd`. Speaks the msgpack-RPC frame
//! format used by every other client.

use anyhow::{anyhow, bail, Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use waymux_protocol::{
    decode_frame, encode_frame, HelloResult, Request, RequestMethod, Response,
    CURRENT_PROTOCOL_VERSION,
};

pub struct Connection {
    stream: UnixStream,
    next_id: u32,
    /// Event frames (raw payload including 4-byte header) received during a
    /// request/response round-trip, to be replayed by `read_raw_frame`.
    pending_events: std::collections::VecDeque<Vec<u8>>,
}

impl Connection {
    pub async fn connect(path: &std::path::Path) -> Result<Self> {
        let stream = UnixStream::connect(path)
            .await
            .with_context(|| format!("connect {}", path.display()))?;
        Ok(Self {
            stream,
            next_id: 1,
            pending_events: std::collections::VecDeque::new(),
        })
    }

    pub async fn hello(&mut self) -> Result<HelloResult> {
        let resp = self
            .request(RequestMethod::Hello {
                client_protocol: CURRENT_PROTOCOL_VERSION,
            })
            .await?;
        resp.decode_result()
            .map_err(|e| anyhow!("decode hello result: {e}"))
    }

    /// Return the next event frame (header + payload). Drains queued events
    /// before reading from the socket. Blocks until one is available.
    pub async fn read_raw_frame(&mut self) -> Result<Vec<u8>> {
        if let Some(f) = self.pending_events.pop_front() {
            return Ok(f);
        }
        self.read_socket_frame().await
    }

    async fn read_socket_frame(&mut self) -> Result<Vec<u8>> {
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .await
            .context("read frame length")?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut payload = vec![0u8; 4 + len];
        payload[..4].copy_from_slice(&len_buf);
        self.stream
            .read_exact(&mut payload[4..])
            .await
            .context("read frame payload")?;
        Ok(payload)
    }

    pub async fn request(&mut self, method: RequestMethod) -> Result<Response> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).unwrap_or(1);
        let req = Request { id, method };
        let mut buf = Vec::with_capacity(256);
        encode_frame(&req, &mut buf)?;
        self.stream.write_all(&buf).await.context("write request")?;

        // Drain frames until we see the matching Response. Events that land
        // in the meantime get queued for later `read_raw_frame` consumers.
        loop {
            let payload = self.read_socket_frame().await?;
            match decode_frame::<Response>(&payload) {
                Ok(resp) if resp.id == id => {
                    if !resp.ok {
                        let err = resp
                            .error
                            .ok_or_else(|| anyhow!("response not ok but no error body"))?;
                        bail!("{:?}: {}", err.code, err.message);
                    }
                    return Ok(resp);
                }
                _ => {
                    // Not a matching response — treat as an Event and queue.
                    self.pending_events.push_back(payload);
                }
            }
        }
    }
}
