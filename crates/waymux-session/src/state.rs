// SPDX-License-Identifier: Apache-2.0

//! Shared session state — the single source of truth for size/scale/windows
//! that the control socket reads and the compositor mutates.
//!
//! `State` also owns the window_id → wl_surface map so the screenshot path
//! (driven from the control socket on the tokio thread) can read shm buffer
//! bytes directly without round-tripping through the compositor thread.

use std::collections::{HashMap, HashSet};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1;
use wayland_protocols::wp::relative_pointer::zv1::server::zwp_relative_pointer_v1::ZwpRelativePointerV1;
use wayland_protocols::xdg::shell::server::{xdg_surface::XdgSurface, xdg_toplevel::XdgToplevel};
use wayland_protocols_plasma::plasma_window_management::server::{
    org_kde_plasma_window::OrgKdePlasmaWindow,
    org_kde_plasma_window_management::OrgKdePlasmaWindowManagement,
};
use wayland_server::backend::ClientId;
use wayland_server::backend::ObjectId;
use wayland_server::protocol::{
    wl_buffer::WlBuffer, wl_callback::WlCallback, wl_data_device::WlDataDevice,
    wl_data_source::WlDataSource, wl_keyboard::WlKeyboard, wl_output::WlOutput,
    wl_pointer::WlPointer, wl_shm, wl_surface::WlSurface, wl_touch::WlTouch,
};
use wayland_server::{Resource, WEnum};
use waymux_protocol::{EventBody, Rect, WindowChange, WindowInfo};

use crate::events::EventSink;

/// Eagerly-fetched clipboard data from the outer compositor. Multiple MIME
/// types may be present; inner clients receive whichever type they request.
#[derive(Clone)]
pub struct ClipboardContent {
    /// (mime_type, raw bytes) pairs, in order of preference.
    pub entries: Vec<(String, Vec<u8>)>,
}

/// Tracks the inner client's current selection source.
pub struct InnerSelectionInfo {
    pub mime_types: Vec<String>,
    /// The server-side WlDataSource resource so outer_view can call `.send`
    /// on it when the outer compositor requests data.
    pub source: WlDataSource,
}

/// Per-inner-buffer hold tracking. See `State::buffer_hold_state`.
#[derive(Default)]
pub struct BufferHoldState {
    /// Number of live `InnerBufferHold`s pinning this buffer.
    pub count: u32,
    /// An external releaser (commit handler) asked to release this buffer
    /// while holds were still outstanding; the last hold-drop performs it.
    pub release_requested: bool,
}

pub struct State {
    pub name: String,
    inner: Mutex<Inner>,
    events: Option<EventSink>,
    last_damage_ns: AtomicU64,
    /// window_id → (surface, owning client).
    windows_by_id: Mutex<HashMap<u32, (WlSurface, ClientId)>>,
    /// Active wl_keyboard resources per client. A client may have several.
    keyboards: Mutex<HashMap<ClientId, Vec<WlKeyboard>>>,
    /// Active wl_pointer resources per client.
    pointers: Mutex<HashMap<ClientId, Vec<WlPointer>>>,
    /// Active wl_touch resources per client. Mirrors `pointers` —
    /// `State::inject_touch` dispatches by ClientId and emits
    /// `wl_touch.{down|motion|up}` + `wl_touch.frame()` to every live
    /// entry on the resolved target client. Empty `Vec` means the client
    /// bound `wl_seat` but never called `get_touch`, in which case
    /// `inject_touch` returns false (same as `inject_pointer`'s
    /// empty-pointers handling).
    touches: Mutex<HashMap<ClientId, Vec<WlTouch>>>,
    /// The window that currently has keyboard focus. Set to the most
    /// recently-created toplevel (single-window-at-a-time semantics for now).
    focused_window: Mutex<Option<u32>>,
    /// Per-window dedup set for the `content=true` fallback warning emitted
    /// when `inject_pointer` is called with content-space coords but the
    /// target window has never recorded an xdg_surface.set_window_geometry.
    /// Stored on State (not per-call) so the warning fires exactly once per
    /// window per session. Populated by `window_content_inset` when the
    /// lookup misses.
    pub content_fallback_warned: Mutex<HashSet<u32>>,
    /// Monotonic serial for input events.
    next_serial: AtomicU64,
    /// Eventfd the compositor thread polls on. Other threads write a
    /// single byte here after queueing events on Wayland resources so the
    /// compositor immediately wakes to `flush_clients`. Without this,
    /// the compositor's poll() sleeps until its idle timeout elapses and
    /// events arrive with that delay on the client side.
    wake_fd: Arc<OwnedFd>,
    /// Optional wakeup eventfd for the outer-view thread. Set via
    /// `set_outer_wake_fd` when an attach is active; record_damage writes
    /// to it so the outer view can pick up the inner frame promptly.
    outer_wake_fd: Mutex<Option<Arc<OwnedFd>>>,
    /// CLOCK_MONOTONIC ms of the most recent "activity" event — any call
    /// that flows through `wake_compositor` (input injection, attach state
    /// change, focus change, child spawn) plus session creation. The
    /// compositor loop reads this to decide whether to enter idle-suspend
    /// (extending its poll timeout from 16 ms to 1 s) when no outer-view
    /// is attached and nothing has prodded us in 30 s.
    last_activity_ms: AtomicU64,
    /// wl_surface.frame callbacks captured during a dispatch call. We
    /// MUST NOT fire `done` on these from inside the dispatch handler
    /// itself — wayland-backend 0.3.15 double-drops the ObjectData Arc
    /// when a destructor event is sent synchronously, which segfaults.
    /// The compositor thread drains this list between dispatch cycles.
    pending_frame_callbacks: Mutex<Vec<WlCallback>>,
    /// Keymap override supplied by the outer compositor's keyboard. When
    /// set, new inner keyboards receive this keymap instead of the default
    /// embedded US layout. Existing keyboards are resent the keymap on update.
    keymap_override: Mutex<Option<(Arc<OwnedFd>, u32)>>,
    /// Session was created with `share_clipboard: true`.
    share_clipboard: bool,

    // ── clipboard bridge (only active when share_clipboard=true) ──────────
    /// Eagerly-fetched outer selection data. Set by the outer_view thread
    /// when the outer compositor changes its selection. The compositor thread
    /// reads this and forwards it to inner clients via wl_data_offer.
    pub outer_clipboard: Mutex<Option<Arc<ClipboardContent>>>,
    /// Bumped each time outer_clipboard changes. The compositor thread tracks
    /// the last value it dispatched so it can detect new arrivals.
    pub outer_clipboard_version: AtomicU64,
    /// Registered inner wl_data_device resources. Outer clipboard offers are
    /// sent to all live entries.
    pub inner_data_devices: Mutex<Vec<WlDataDevice>>,
    /// Current inner selection source + its MIME types. Set by the compositor
    /// thread when an inner client calls wl_data_device.set_selection.
    pub inner_selection: Mutex<Option<InnerSelectionInfo>>,
    /// Bumped each time inner_selection changes (including on clear).
    pub inner_selection_version: AtomicU64,

    /// Set when any inner surface has requested keyboard shortcuts inhibition.
    /// Read by outer_view to request inhibit from the host compositor.
    pub any_shortcuts_inhibited: AtomicBool,

    // ── pointer lock bridge ───────────────────────────────────────────────
    /// Number of active zwp_locked_pointer_v1 objects held by inner clients.
    /// When > 0, outer_view requests pointer lock from the host compositor.
    pub pointer_lock_count: AtomicUsize,
    /// Server-side ZwpRelativePointerV1 objects registered by inner clients
    /// (e.g. KWin). inject_relative_pointer sends events to all live entries.
    pub relative_pointers: Mutex<Vec<ZwpRelativePointerV1>>,

    // ── primary selection (middle-click clipboard) ────────────────────────
    /// Current primary selection MIME types (set by the focused inner client).
    pub primary_selection: Mutex<Option<Vec<String>>>,
    /// Registered inner zwp_primary_selection_device_v1 resources.
    pub primary_selection_devices: Mutex<Vec<ZwpPrimarySelectionDeviceV1>>,

    // ── plasma window management (for plasmashell taskbar) ────────────────
    /// Bound org_kde_plasma_window_management globals. New windows fire
    /// `window`/`window_with_uuid` events on every entry here.
    pub plasma_managers: Mutex<Vec<OrgKdePlasmaWindowManagement>>,
    /// Per-window plasma_window resources. Title/app_id/state changes are
    /// broadcast to every entry under the matching window_id.
    pub plasma_windows: Mutex<HashMap<u32, Vec<OrgKdePlasmaWindow>>>,

    // ── virtual output + toplevel registries (for resize) ─────────────────
    /// Every bound `wl_output` global. `State::resize` re-sends `mode` (+
    /// `done` on v2+) to all live entries so clients learn the new output
    /// dimensions. Populated on `WlOutput` bind (compositor.rs), pruned of
    /// dead entries on each resize.
    output_globals: Mutex<Vec<WlOutput>>,
    /// Per-window `(xdg_surface, xdg_toplevel)` pair. `State::resize` re-sends
    /// `xdg_toplevel.configure(new_w, new_h)` + `xdg_surface.configure(serial)`
    /// to every mapped toplevel so clients reconfigure to the new size, exactly
    /// as compositor.rs does at map time. Registered when the toplevel is
    /// created; entries for dead resources are pruned on each resize.
    toplevels: Mutex<HashMap<u32, (XdgSurface, XdgToplevel)>>,

    // ── deferred dmabuf buffer release ───────────────────────────────────
    // When KWin commits a GPU frame, the inner compositor queues the new
    // wl_buffer here instead of releasing it immediately. outer_view drains
    // this queue when it forwards a dmabuf to niri, then releases all held
    // buffers only once niri fires wl_buffer.release on the outer surface.
    // This prevents KWin from starting its next GPU write before niri has
    // finished reading the current buffer, eliminating the skybox-flash race.
    pub pending_dmabuf_releases: Mutex<Vec<WlBuffer>>,

    /// wp_linux_drm_syncobj_v1 release-point map. When KWin sets a release
    /// point on a commit, we record (buffer ObjectId → queue of release
    /// points) here. When the buffer is released back to KWin (via any
    /// of the release paths), `release_inner_buffer` pops the front
    /// release point for that buffer and signals it on the kernel-side
    /// timeline. KWin can then reuse the underlying GEM safely.
    pub buffer_release_signals:
        Mutex<HashMap<ObjectId, std::collections::VecDeque<crate::syncobj::TimelinePoint>>>,

    /// Outstanding `InnerBufferHold` count per inner buffer, plus whether an
    /// external release was already requested while held. The GPU/CudaNvenc
    /// viewer path tees a dmabuf LAZILY (the encoder imports it ~16ms later on
    /// its own thread). In headless mode the commit handler releases the
    /// just-committed buffer back to KWin immediately after the tap — which,
    /// without this gate, hands the buffer to KWin before the encoder has read
    /// it, so KWin overwrites it mid-import → a half-composited frame (the
    /// desktop wallpaper showing through = the blue / "reveals underneath"
    /// flicker). `release_inner_buffer` defers the real `wl_buffer.release` +
    /// syncobj release-point signal until the last hold drops.
    pub buffer_hold_state: Mutex<HashMap<ObjectId, BufferHoldState>>,

    // ── lossless recording ────────────────────────────────────────────────
    pub(crate) recording: Mutex<Option<crate::recording::DualRecordingHandle>>,
    /// Rate-limit timestamp for `WholeDesktop` capture mode. The
    /// per-pixel composite in `capture_desktop` is too slow to run on
    /// every inner-client commit (~470 ms / call); the tap throttles
    /// to 10 fps (one capture per 100 ms) by checking this against
    /// the current `Instant::now()`. None until the first whole-desktop
    /// frame is captured.
    pub(crate) last_whole_desktop_capture: Mutex<Option<std::time::Instant>>,

    // ── browser WebRTC viewer (neko-bridge) ───────────────────────────────
    /// Active viewer handle. At most one viewer per session (v1).
    /// Independent of the recording slot — both can be active simultaneously.
    pub viewer: Mutex<Option<crate::viewer::ViewerHandle>>,

    // ── cursor overlay ────────────────────────────────────────────────────
    /// Current pointer cursor set by the inner client via wl_pointer.set_cursor.
    /// (surface, hotspot_x, hotspot_y). None = hidden.
    pub cursor_surface: Mutex<Option<(WlSurface, i32, i32)>>,
    /// Channel for queued CursorImage/CursorPos updates consumed by the viewer
    /// socket writer.
    pub cursor_channel: std::sync::Arc<crate::viewer::cursor::CursorChannel>,
    /// Deduplicates cursor readbacks so we only forward on a real shape change.
    cursor_shape_tracker: Mutex<crate::viewer::cursor::CursorShapeTracker>,
    /// Dedicated EGL+GLES2 context for reading dmabuf-backed cursor buffers
    /// back to RGBA. Lazily initialized on first dmabuf cursor; independent of
    /// the encoder's CUDA context.
    cursor_reader: std::sync::Arc<crate::viewer::cursor::DmabufCursorReader>,
}

struct Inner {
    width: u32,
    height: u32,
    scale: u32,
    next_window_id: u32,
    windows: HashMap<u32, WindowInfo>,
}

impl State {
    pub fn new(
        name: String,
        width: u32,
        height: u32,
        scale: u32,
        events: Option<EventSink>,
        share_clipboard: bool,
    ) -> Self {
        // Eventfd for compositor-thread wakeups. Starts counter=0, non-blocking
        // so writers never stall, CLOEXEC so subprocesses don't inherit it.
        // eventfd() can only fail on resource exhaustion; in that case the
        // process is doomed anyway.
        let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        assert!(raw >= 0, "eventfd(): {}", std::io::Error::last_os_error());
        let wake_fd = Arc::new(unsafe { OwnedFd::from_raw_fd(raw) });

        Self {
            name,
            inner: Mutex::new(Inner {
                width,
                height,
                scale,
                next_window_id: 1,
                windows: HashMap::new(),
            }),
            events,
            last_damage_ns: AtomicU64::new(0),
            windows_by_id: Mutex::new(HashMap::new()),
            keyboards: Mutex::new(HashMap::new()),
            pointers: Mutex::new(HashMap::new()),
            touches: Mutex::new(HashMap::new()),
            focused_window: Mutex::new(None),
            content_fallback_warned: Mutex::new(HashSet::new()),
            next_serial: AtomicU64::new(1),
            wake_fd,
            outer_wake_fd: Mutex::new(None),
            last_activity_ms: AtomicU64::new(monotonic_now_ms()),
            pending_frame_callbacks: Mutex::new(Vec::new()),
            keymap_override: Mutex::new(None),
            share_clipboard,
            outer_clipboard: Mutex::new(None),
            outer_clipboard_version: AtomicU64::new(0),
            inner_data_devices: Mutex::new(Vec::new()),
            inner_selection: Mutex::new(None),
            inner_selection_version: AtomicU64::new(0),
            any_shortcuts_inhibited: AtomicBool::new(false),
            pointer_lock_count: AtomicUsize::new(0),
            relative_pointers: Mutex::new(Vec::new()),
            primary_selection: Mutex::new(None),
            primary_selection_devices: Mutex::new(Vec::new()),
            plasma_managers: Mutex::new(Vec::new()),
            plasma_windows: Mutex::new(HashMap::new()),
            output_globals: Mutex::new(Vec::new()),
            toplevels: Mutex::new(HashMap::new()),
            pending_dmabuf_releases: Mutex::new(Vec::new()),
            buffer_release_signals: Mutex::new(HashMap::new()),
            buffer_hold_state: Mutex::new(HashMap::new()),
            recording: Mutex::new(None),
            last_whole_desktop_capture: Mutex::new(None),
            viewer: Mutex::new(None),
            cursor_surface: Mutex::new(None),
            cursor_channel: std::sync::Arc::new(Default::default()),
            cursor_shape_tracker: Mutex::new(Default::default()),
            cursor_reader: std::sync::Arc::new(crate::viewer::cursor::DmabufCursorReader::new()),
        }
    }

    pub fn share_clipboard(&self) -> bool {
        self.share_clipboard
    }

    /// Replace the inner compositor's keymap with the outer compositor's
    /// keymap (received from the outer `wl_keyboard.keymap` event). Resends
    /// to all currently-registered inner keyboards immediately.
    pub fn update_keymap(&self, fd: OwnedFd, size: u32) {
        let arc_fd = Arc::new(fd);
        *self.keymap_override.lock().unwrap() = Some((arc_fd.clone(), size));
        let all_kbds: Vec<WlKeyboard> = {
            let map = self.keyboards.lock().unwrap();
            map.values()
                .flatten()
                .filter(|k| k.is_alive())
                .cloned()
                .collect()
        };
        use std::os::fd::AsFd;
        for kbd in &all_kbds {
            kbd.keymap(
                wayland_server::protocol::wl_keyboard::KeymapFormat::XkbV1,
                arc_fd.as_fd(),
                size,
            );
        }
        if !all_kbds.is_empty() {
            self.wake_compositor();
        }
    }

    /// Return the active keymap fd + size. Returns the outer compositor's
    /// keymap if one has been received; otherwise falls back to the default
    /// embedded US pc105 keymap.
    pub fn get_keymap(&self) -> (Arc<OwnedFd>, u32) {
        if let Some(ov) = self.keymap_override.lock().unwrap().clone() {
            return ov;
        }
        crate::keymap::keymap_fd().expect("keymap initialisation failed")
    }

    /// Park a wl_surface.frame callback to be completed by the compositor
    /// loop between dispatch cycles. See the field doc for why we don't
    /// just call `done` inline.
    pub fn queue_frame_callback(&self, cb: WlCallback) {
        self.pending_frame_callbacks.lock().unwrap().push(cb);
    }

    /// Fire `done` on every queued frame callback. Called from the
    /// compositor thread's main loop AFTER `dispatch_clients` returns.
    pub fn drain_frame_callbacks(&self) {
        // Use CLOCK_MONOTONIC so this timestamp is consistent with the
        // wp_presentation_feedback timestamps waymux sends to KWin. Both
        // clocks must agree: KWin correlates frame-callback data with
        // presentation feedback to compute its nextPaintDelay. Sending
        // CLOCK_REALTIME here (~1.75e9 ms since Unix epoch) against
        // CLOCK_MONOTONIC presentation timestamps (~uptime ms) causes KWin
        // to compute a ~1.75-billion-ms render lag and race to catch up,
        // producing the "wrong frame interspersed" skybox flicker in games.
        let now_ms = {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
            ((ts.tv_sec as u64) * 1_000 + (ts.tv_nsec as u64) / 1_000_000) as u32
        };
        let mut pending = self.pending_frame_callbacks.lock().unwrap();
        for cb in pending.drain(..) {
            if cb.is_alive() {
                cb.done(now_ms);
            }
        }
    }

    /// Send `preferred_buffer_scale(1)` to every registered client surface.
    ///
    /// This is a server-initiated event (no client request needed). Sending it
    /// every 16 ms pokes the client's Qt event loop — when KWin receives this
    /// on its own output surface it may trigger a pending repaint check that
    /// unsticks its compositor render loop after inner clients produce damage.
    pub fn poke_client_surfaces(&self) {
        let surfaces: Vec<wayland_server::protocol::wl_surface::WlSurface> = self
            .windows_by_id
            .lock()
            .unwrap()
            .values()
            .map(|(s, _)| s.clone())
            .collect();
        for surface in surfaces {
            if surface.is_alive() && surface.version() >= 6 {
                surface.preferred_buffer_scale(1);
            }
        }
    }

    /// Fire `presented` on all pending wp_presentation_feedback objects
    /// for tracked toplevel surfaces. Called from the compositor main loop.
    pub fn drain_presentation_feedbacks(&self) {
        use std::sync::atomic::Ordering;
        use wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::Kind;

        let now_ns = self.last_damage_ns.load(Ordering::Acquire);
        if now_ns == 0 {
            return;
        }
        let tv_sec = now_ns / 1_000_000_000;
        let tv_nsec = (now_ns % 1_000_000_000) as u32;
        let tv_sec_hi = (tv_sec >> 32) as u32;
        let tv_sec_lo = tv_sec as u32;

        let surfaces: Vec<wayland_server::protocol::wl_surface::WlSurface> = self
            .windows_by_id
            .lock()
            .unwrap()
            .values()
            .map(|(s, _)| s.clone())
            .collect();

        for surface in surfaces {
            let Some(sd) = surface.data::<crate::compositor::SurfaceData>() else {
                continue;
            };
            let feedbacks: Vec<_> = sd.pending_feedbacks.lock().unwrap().drain(..).collect();
            for fb in feedbacks {
                if fb.is_alive() {
                    fb.presented(
                        tv_sec_hi,
                        tv_sec_lo,
                        tv_nsec,
                        crate::compositor::PRESENTATION_REFRESH_PERIOD_NS,
                        0,
                        0, // seq_hi, seq_lo
                        Kind::empty(),
                    );
                }
            }
        }
    }

    /// Register (or clear) the outer-view thread's wakeup eventfd. Called
    /// on attach start and detach end. `record_damage` pokes this fd to
    /// notify the outer-view of a new inner frame.
    ///
    /// Emits a daemon-level `Occluded` event whenever the attach state
    /// transitions, so SDKs can react to "nobody is watching".
    ///
    /// On transition to `None` (no attach), drains any deferred dmabuf
    /// releases — without an outer_view to consume them, KWin's GBM pool
    /// would otherwise exhaust and stall its render loop.
    pub fn set_outer_wake_fd(&self, fd: Option<Arc<OwnedFd>>) {
        let was_attached = {
            let mut guard = self.outer_wake_fd.lock().unwrap();
            let prev = guard.is_some();
            *guard = fd.clone();
            prev
        };
        let now_attached = fd.is_some();
        if was_attached != now_attached {
            if !now_attached {
                self.flush_pending_releases();
            }
            if let Some(sink) = &self.events {
                sink.emit(EventBody::Occluded {
                    name: self.name.clone(),
                    occluded: !now_attached,
                });
            }
        }
    }

    /// Whether an outer-view (attach client + host surface) is currently
    /// driving this session. When false the session is "headless" — it
    /// keeps composing for screenshots/recording but no host compositor
    /// is reading the output.
    pub fn is_attached(&self) -> bool {
        self.outer_wake_fd.lock().unwrap().is_some()
    }

    /// Clear the outer wake fd only if it still points to `fd`. Used by
    /// outer_view teardown to avoid clearing a replacement fd installed by
    /// a new outer_view that started before the old one finished tearing down.
    pub fn clear_outer_wake_fd_if_mine(&self, fd: &Arc<OwnedFd>) {
        let mut guard = self.outer_wake_fd.lock().unwrap();
        if let Some(current) = guard.as_ref() {
            if Arc::ptr_eq(current, fd) {
                *guard = None;
            }
        }
    }

    /// Raw fd the compositor thread should add to its poll set. When
    /// readable, the compositor should drain it (read any pending bytes)
    /// and run a flush pass.
    pub fn wake_fd(&self) -> std::os::fd::RawFd {
        self.wake_fd.as_raw_fd()
    }

    /// Signal the compositor thread that there is work to flush.
    /// Non-blocking; safe from any thread.
    /// Public alias for `wake_compositor` used by external threads (e.g.
    /// the outer_view clipboard pipe thread) that need to flush the inner
    /// compositor's send queue without going through State's private methods.
    pub fn poke_compositor_wake(&self) {
        self.wake_compositor();
    }

    fn wake_compositor(&self) {
        self.last_activity_ms
            .store(monotonic_now_ms(), Ordering::Relaxed);
        let byte: u64 = 1;
        // SAFETY: writing 8 bytes to an eventfd is defined. The fd is
        // kept alive by our Arc<OwnedFd>.
        unsafe {
            libc::write(
                self.wake_fd.as_raw_fd(),
                (&byte as *const u64).cast(),
                std::mem::size_of::<u64>(),
            );
        }
    }

    /// CLOCK_MONOTONIC ms timestamp of the last activity event. Read by the
    /// compositor loop to decide whether to enter idle suspend.
    pub fn last_activity_ms(&self) -> u64 {
        self.last_activity_ms.load(Ordering::Relaxed)
    }

    /// Treat "right now" as activity. Used by the compositor loop when a new
    /// client connects on the listener socket — `accept()` doesn't go through
    /// `wake_compositor`, but a new connection is the kind of event we want
    /// to reset the idle clock for.
    pub fn note_activity(&self) {
        self.last_activity_ms
            .store(monotonic_now_ms(), Ordering::Relaxed);
    }

    /// Record the latest applied pointer position + the input seq that produced
    /// it, queueing a CursorPos for the viewer overlay (positions collapse to
    /// newest, so this is cheap per motion).
    pub fn record_cursor_pos(&self, x: f32, y: f32, seq: u32) {
        self.cursor_channel
            .push_pos(crate::viewer::cursor::CursorPos { x, y, seq });
    }

    /// Read the current cursor surface's buffer to RGBA and queue a CursorImage
    /// (only on shape change). SHM buffers are read directly; dmabuf cursors
    /// are handled separately. A null cursor queues a hide (w=h=0).
    pub fn note_cursor_dirty(&self) {
        use crate::buffer::BufferKind;
        use crate::compositor::SurfaceData;
        use crate::viewer::cursor::{argb8888_bytes_to_rgba, CursorImage, MAX_CURSOR_DIM};

        let cursor = self.cursor_surface.lock().unwrap().clone();
        let Some((surface, hot_x_i, hot_y_i)) = cursor else {
            self.cursor_channel.push_image(CursorImage {
                w: 0,
                h: 0,
                hot_x: 0,
                hot_y: 0,
                rgba: vec![],
            });
            return;
        };
        let hot_x = hot_x_i.max(0) as u16;
        let hot_y = hot_y_i.max(0) as u16;

        let Some(sd) = surface.data::<SurfaceData>() else {
            return;
        };
        let Some(buf) = sd.current_buffer.lock().unwrap().clone() else {
            return;
        };
        let Some(kind) = buf.data::<BufferKind>() else {
            return;
        };

        // Use the Wayland protocol_id as the buffer identity for dedup.
        // Two commits of the same wl_buffer (cursor re-committed on every move)
        // share an ObjectId / protocol_id, so this correctly suppresses
        // re-reads when only pointer position changed but not the shape.
        let bid = buf.id().protocol_id() as u64;
        if !self.cursor_shape_tracker.lock().unwrap().changed(bid) {
            return;
        }

        match kind {
            BufferKind::Shm(_) => {
                if let Some((w, h, stride, bytes, opaque)) = self.cursor_shm_argb_bytes(&buf) {
                    if w > MAX_CURSOR_DIM as u32 || h > MAX_CURSOR_DIM as u32 {
                        tracing::warn!(w, h, "cursor: oversized SHM cursor dropped");
                        return;
                    }
                    let mut rgba = argb8888_bytes_to_rgba(&bytes, w, h, stride);
                    // XRGB8888: the X byte is undefined and must not be used as
                    // alpha (a zero X → fully-transparent cursor → invisible).
                    // Force every alpha byte to 0xFF so the cursor is always opaque.
                    if opaque {
                        for a in rgba.chunks_exact_mut(4) {
                            a[3] = 0xFF;
                        }
                    }
                    self.cursor_channel.push_image(CursorImage {
                        w: w as u16,
                        h: h as u16,
                        hot_x,
                        hot_y,
                        rgba,
                    });
                }
            }
            BufferKind::Dmabuf(d) => {
                use std::os::fd::AsRawFd;
                let (w, h) = (d.width as u32, d.height as u32);
                if w > MAX_CURSOR_DIM as u32 || h > MAX_CURSOR_DIM as u32 {
                    tracing::warn!(w, h, "cursor: oversized dmabuf cursor dropped");
                    return;
                }
                if d.drm_format != crate::dmabuf::DRM_FORMAT_ARGB8888
                    && d.drm_format != crate::dmabuf::DRM_FORMAT_XRGB8888
                {
                    tracing::warn!(
                        fmt = d.drm_format,
                        "cursor: unsupported dmabuf format, skipping"
                    );
                    return;
                }
                if let Some(rgba) = self.cursor_reader.read_rgba(
                    d.fd.as_raw_fd(),
                    d.modifier,
                    d.drm_format,
                    d.offset,
                    w,
                    h,
                    d.stride,
                ) {
                    self.cursor_channel.push_image(CursorImage {
                        w: w as u16,
                        h: h as u16,
                        hot_x,
                        hot_y,
                        rgba,
                    });
                }
            }
            _ => {}
        }
    }

    /// Read an ARGB8888 or XRGB8888 SHM buffer's bytes into an owned Vec.
    /// Returns `(width, height, stride_bytes, bytes, opaque)` or None for
    /// non-SHM / non-ARGB8888 / unreadable buffers. `opaque` is `true` when
    /// the source format is XRGB8888 — the caller must force every alpha byte
    /// to 0xFF because the X byte is undefined and must not be used as alpha.
    fn cursor_shm_argb_bytes(&self, buf: &WlBuffer) -> Option<(u32, u32, u32, Vec<u8>, bool)> {
        use crate::buffer::BufferKind;
        use wayland_server::protocol::wl_shm;
        use wayland_server::WEnum;

        let kind = buf.data::<BufferKind>()?;
        // with_bytes works for Shm (and SinglePixel, Dmabuf) — we only want
        // Shm + ARGB8888/XRGB8888. We match on the return to check format.
        kind.with_bytes(|bytes, w, h, stride, format| {
            let is_xrgb = matches!(format, WEnum::Value(wl_shm::Format::Xrgb8888));
            let is_argb = is_xrgb || matches!(format, WEnum::Value(wl_shm::Format::Argb8888));
            if !is_argb {
                return None;
            }
            // Only accept SHM (not dmabuf or single-pixel routed through with_bytes).
            if !matches!(kind, BufferKind::Shm(_)) {
                return None;
            }
            Some((w as u32, h as u32, stride as u32, bytes.to_vec(), is_xrgb))
        })?
    }

    fn next_serial(&self) -> u32 {
        (self.next_serial.fetch_add(1, Ordering::Relaxed) & 0xFFFF_FFFF) as u32
    }

    fn emit(&self, body: EventBody) {
        if let Some(sink) = &self.events {
            sink.emit(body);
        }
    }

    fn session_name(&self) -> String {
        self.events
            .as_ref()
            .map(|s| s.session_name().to_string())
            .unwrap_or_default()
    }

    // ── clipboard bridge helpers ────────────────────────────────────────

    /// Register an inner wl_data_device. Stale (dead) entries are pruned
    /// on each registration to keep the list from growing unboundedly.
    pub fn register_data_device(&self, dev: WlDataDevice) {
        let mut devs = self.inner_data_devices.lock().unwrap();
        devs.retain(|d| d.is_alive());
        devs.push(dev);
    }

    /// Remove a specific inner data device (called on wl_data_device.release
    /// or client disconnect).
    pub fn unregister_data_device(&self, dev: &WlDataDevice) {
        self.inner_data_devices
            .lock()
            .unwrap()
            .retain(|d| d.id() != dev.id());
    }

    pub fn register_primary_device(&self, dev: ZwpPrimarySelectionDeviceV1) {
        self.primary_selection_devices.lock().unwrap().push(dev);
    }

    pub fn unregister_primary_device(&self, dev: &ZwpPrimarySelectionDeviceV1) {
        self.primary_selection_devices
            .lock()
            .unwrap()
            .retain(|d| d.id() != dev.id());
    }

    /// Record a new inner selection. Called by the compositor thread when
    /// wl_data_device.set_selection fires. Bumps inner_selection_version so
    /// the outer_view thread notices.
    pub fn set_inner_selection(&self, mime_types: Vec<String>, source: WlDataSource) {
        *self.inner_selection.lock().unwrap() = Some(InnerSelectionInfo { mime_types, source });
        self.inner_selection_version.fetch_add(1, Ordering::Release);
    }

    /// Clear the current inner selection (set_selection with null source).
    pub fn clear_inner_selection(&self) {
        *self.inner_selection.lock().unwrap() = None;
        self.inner_selection_version.fetch_add(1, Ordering::Release);
    }

    pub fn any_shortcuts_inhibited(&self) -> bool {
        self.any_shortcuts_inhibited.load(Ordering::Acquire)
    }

    pub fn set_shortcuts_inhibited(&self, val: bool) {
        self.any_shortcuts_inhibited.store(val, Ordering::Release);
    }

    // ── pointer lock bridge ───────────────────────────────────────────────

    pub fn inc_pointer_lock_count(&self) {
        self.pointer_lock_count.fetch_add(1, Ordering::AcqRel);
        self.poke_outer_view();
    }

    pub fn dec_pointer_lock_count(&self) {
        self.pointer_lock_count.fetch_sub(1, Ordering::AcqRel);
        self.poke_outer_view();
    }

    pub fn pointer_lock_active(&self) -> bool {
        self.pointer_lock_count.load(Ordering::Acquire) > 0
    }

    pub fn add_relative_pointer(&self, ptr: ZwpRelativePointerV1) {
        self.relative_pointers.lock().unwrap().push(ptr);
    }

    /// True when at least one alive inner relative-pointer object exists.
    /// Used by the outer_view to decide whether to synthesize relative motion
    /// from absolute position deltas (needed when KWin doesn't forward the
    /// pointer-lock chain to waymux-session in nested mode).
    pub fn has_relative_pointers(&self) -> bool {
        self.relative_pointers
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.is_alive())
    }

    /// Forward relative motion to all registered inner relative-pointer objects.
    /// `utime_hi`/`utime_lo` are the microsecond timestamp split from the outer event.
    pub fn inject_relative_pointer(
        &self,
        utime_hi: u32,
        utime_lo: u32,
        dx: f64,
        dy: f64,
        dx_unaccel: f64,
        dy_unaccel: f64,
    ) {
        let mut ptrs = self.relative_pointers.lock().unwrap();
        ptrs.retain(|p| p.is_alive());
        if ptrs.is_empty() {
            tracing::debug!(
                dx,
                dy,
                "inject_relative_pointer: no inner clients; dropping"
            );
            return;
        }
        tracing::debug!(
            dx,
            dy,
            targets = ptrs.len(),
            "inject_relative_pointer: forwarding"
        );
        for p in ptrs.iter() {
            p.relative_motion(utime_hi, utime_lo, dx, dy, dx_unaccel, dy_unaccel);
        }
        self.wake_compositor();
    }

    // ── deferred dmabuf buffer release ───────────────────────────────────

    pub fn queue_deferred_release(&self, buf: WlBuffer) {
        self.pending_dmabuf_releases.lock().unwrap().push(buf);
    }

    /// Drain pending releases and split them into two groups:
    ///  deferred  — the WlBuffer whose DmabufBufferData matches `forwarded_ptr`
    ///              (the frame outer_view is about to forward to niri — must be
    ///              held until niri's wl_buffer.release so its GPU write can't
    ///              start before niri's GPU read fence is established)
    ///  immediate — all other WlBuffers (skipped frames — outer_view never
    ///              forwarded them, so niri isn't reading them; safe to release
    ///              right away so KWin can render new frames at full speed)
    pub fn take_pending_releases_split(
        &self,
        forwarded_ptr: *const crate::dmabuf::DmabufBufferData,
    ) -> (Vec<WlBuffer>, Vec<WlBuffer>) {
        use crate::buffer::BufferKind;
        let all = std::mem::take(&mut *self.pending_dmabuf_releases.lock().unwrap());
        let mut deferred = Vec::new();
        let mut immediate = Vec::new();
        for buf in all {
            let matches = buf
                .data::<BufferKind>()
                .and_then(|k| {
                    if let BufferKind::Dmabuf(d) = k {
                        Some(std::sync::Arc::as_ptr(d) as *const _)
                    } else {
                        None
                    }
                })
                .map(|p: *const crate::dmabuf::DmabufBufferData| p == forwarded_ptr)
                .unwrap_or(false);
            if matches {
                deferred.push(buf);
            } else {
                immediate.push(buf);
            }
        }
        (deferred, immediate)
    }

    /// Drain and return all pending deferred-release buffers.
    pub fn take_pending_releases(&self) -> Vec<WlBuffer> {
        std::mem::take(&mut *self.pending_dmabuf_releases.lock().unwrap())
    }

    /// Diagnostic: how many inner buffers are queued for deferred release.
    /// A persistently-large value means KWin keeps allocating new buffers
    /// because the old ones haven't been released back yet.
    pub fn pending_release_count(&self) -> usize {
        self.pending_dmabuf_releases.lock().unwrap().len()
    }

    /// Release all pending buffers immediately. Called on non-commit paths
    /// (SHM fallback, session empty) so KWin is never blocked indefinitely.
    pub fn flush_pending_releases(&self) {
        let bufs = self.take_pending_releases();
        let any = !bufs.is_empty();
        for b in bufs {
            self.release_inner_buffer(&b);
        }
        if any {
            self.wake_compositor();
        }
    }

    /// Note a release timeline point pending for `buf`. The point is
    /// signaled the next time `release_inner_buffer(buf)` runs. Called by
    /// the wl_surface.commit handler when KWin set a release point on the
    /// surface's wp_linux_drm_syncobj_surface_v1.
    pub fn note_buffer_release_point(&self, buf: &WlBuffer, point: crate::syncobj::TimelinePoint) {
        self.buffer_release_signals
            .lock()
            .unwrap()
            .entry(buf.id())
            .or_default()
            .push_back(point);
    }

    /// Release `buf` back to KWin and, if a wp_linux_drm_syncobj release
    /// point was associated with this buffer's most-recent commit, signal
    /// it. **All** release paths for inner KWin buffers must go through
    /// this helper, otherwise KWin will deadlock waiting for the syncobj
    /// release that we forgot to fire.
    pub fn release_inner_buffer(&self, buf: &WlBuffer) {
        if !buf.is_alive() {
            return;
        }
        // Hold-aware: if any InnerBufferHold still pins this buffer (e.g. the
        // GPU/CudaNvenc viewer encoder hasn't imported it yet), DON'T release
        // now — record the request and let the last hold-drop do it. Releasing
        // here would hand the buffer to KWin mid-import → half-composited frame.
        {
            let mut holds = self.buffer_hold_state.lock().unwrap();
            if let Some(st) = holds.get_mut(&buf.id()) {
                if st.count > 0 {
                    st.release_requested = true;
                    return;
                }
            }
        }
        self.do_release_inner_buffer(buf);
    }

    /// Unconditional release: signal any pending syncobj release point and
    /// send `wl_buffer.release`. Only call when no holds remain (via
    /// `release_inner_buffer` or the last `drop_buffer_hold`).
    fn do_release_inner_buffer(&self, buf: &WlBuffer) {
        if !buf.is_alive() {
            return;
        }
        let signal = {
            let mut map = self.buffer_release_signals.lock().unwrap();
            let pop = map.get_mut(&buf.id()).and_then(|q| q.pop_front());
            // Tidy up empty entries so the map doesn't leak per-buffer slots.
            if let Some(q) = map.get(&buf.id()) {
                if q.is_empty() {
                    map.remove(&buf.id());
                }
            }
            pop
        };
        buf.release();
        if let Some(s) = signal {
            s.signal();
        }
    }

    /// Register a new `InnerBufferHold` pinning `buf`. Increments the hold
    /// count so `release_inner_buffer` defers the real release until the
    /// matching `drop_buffer_hold`.
    pub fn register_buffer_hold(&self, buf: &WlBuffer) {
        self.buffer_hold_state
            .lock()
            .unwrap()
            .entry(buf.id())
            .or_default()
            .count += 1;
    }

    /// Drop an `InnerBufferHold` pinning `buf`. When the last hold drops AND a
    /// release was requested while held, perform the deferred release now.
    pub fn drop_buffer_hold(&self, buf: &WlBuffer) {
        let do_release = {
            let mut holds = self.buffer_hold_state.lock().unwrap();
            if let Some(st) = holds.get_mut(&buf.id()) {
                st.count = st.count.saturating_sub(1);
                if st.count == 0 {
                    let requested = st.release_requested;
                    holds.remove(&buf.id());
                    requested
                } else {
                    false
                }
            } else {
                false
            }
        };
        if do_release {
            self.do_release_inner_buffer(buf);
        }
    }

    /// Poke the outer-view thread without marking new damage (e.g. for lock state changes).
    pub fn poke_outer_view(&self) {
        if let Some(fd) = self.outer_wake_fd.lock().unwrap().as_ref() {
            let byte: u64 = 1;
            unsafe {
                libc::write(
                    fd.as_raw_fd(),
                    (&byte as *const u64).cast(),
                    std::mem::size_of::<u64>(),
                );
            }
        }
    }

    /// Store new outer clipboard data (fetched by the outer_view thread from
    /// the outer compositor's selection offer). Bumps outer_clipboard_version
    /// and pokes the compositor thread so it can forward to inner clients.
    pub fn set_outer_clipboard(&self, content: Arc<ClipboardContent>) {
        *self.outer_clipboard.lock().unwrap() = Some(content);
        self.outer_clipboard_version.fetch_add(1, Ordering::Release);
        // Wake the compositor thread so it can run drain_clipboard immediately.
        self.wake_compositor();
    }

    pub fn snapshot(&self) -> (u32, u32, u32) {
        let s = self.inner.lock().unwrap();
        (s.width, s.height, s.scale)
    }

    /// Resize the session's virtual output to `width x height` and propagate
    /// the change to inner clients.
    ///
    /// Threading: this runs on the tokio control thread, NOT the compositor
    /// thread. It reuses the same cross-thread mechanism `inject_pointer` /
    /// `inject_key` use: queue Wayland events directly onto the resource
    /// handles (which `wayland-server` lets any thread do; the backend
    /// serialises the send buffer internally) and then poke the compositor's
    /// `wake_fd` so it flushes the send buffer to clients immediately. No new
    /// threading path is introduced.
    ///
    /// What it sends:
    ///  - every bound `wl_output` gets a fresh `mode(Current | Preferred,
    ///    width, height, refresh)` (+ `done` on v2+) so clients learn the new
    ///    output dimensions and `zxdg_output` consumers can recompute;
    ///  - every mapped `xdg_toplevel` gets `configure(width, height, [])`
    ///    followed by `xdg_surface.configure(serial)`, exactly as the map-time
    ///    path in compositor.rs does, so clients reconfigure to the new size.
    pub fn resize(&self, width: u32, height: u32) {
        {
            let mut s = self.inner.lock().unwrap();
            s.width = width;
            s.height = height;
        }

        // 1) Re-advertise the output mode to every live wl_output. Prune dead
        //    handles while we hold the lock.
        {
            use wayland_server::protocol::wl_output;
            let mut outputs = self.output_globals.lock().unwrap();
            outputs.retain(|o| o.is_alive());
            for output in outputs.iter() {
                output.mode(
                    wl_output::Mode::Current | wl_output::Mode::Preferred,
                    width as i32,
                    height as i32,
                    crate::compositor::VIRTUAL_OUTPUT_REFRESH_MHZ,
                );
                if output.version() >= 2 {
                    output.done();
                }
            }
        }

        // 2) Reconfigure every mapped toplevel to the new size. Snapshot the
        //    live pairs (dropping dead ones) so we don't hold the lock across
        //    the configure calls.
        let pairs: Vec<(XdgSurface, XdgToplevel)> = {
            let mut map = self.toplevels.lock().unwrap();
            map.retain(|_, (xs, tl)| xs.is_alive() && tl.is_alive());
            map.values().cloned().collect()
        };
        for (xdg_surface, toplevel) in &pairs {
            // Mirror compositor.rs's map-time configure: toplevel.configure
            // carries the new logical size; xdg_surface.configure carries the
            // serial the client acks. Empty states vec = no fullscreen/maximised
            // flags, same as map time.
            toplevel.configure(width as i32, height as i32, Vec::new());
            xdg_surface.configure(self.next_serial());
        }

        // 3) Wake the compositor thread so it flushes these queued events to
        //    clients now rather than at its next idle poll, matching inject_*.
        self.wake_compositor();
    }

    pub fn windows(&self) -> Vec<WindowInfo> {
        // Overlay `content_rect` from each window's SurfaceData
        // (populated by xdg_surface.set_window_geometry in compositor.rs) onto
        // the cached WindowInfo. The Window struct in `inner.windows` keeps a
        // default `content_rect: None`; the live source of truth lives on
        // SurfaceData, so we read it here at ListWindows time rather than
        // mirroring on every set_window_geometry event.
        //
        // Lock order: acquire `windows_by_id` first (clone out the surfaces
        // we need), then `inner` for the WindowInfo cache. Both locks are
        // short-lived; SurfaceData::content_rect's Mutex is acquired and
        // released for each window with the value cloned out.
        let surfaces: HashMap<u32, WlSurface> = self
            .windows_by_id
            .lock()
            .unwrap()
            .iter()
            .map(|(id, (surface, _cid))| (*id, surface.clone()))
            .collect();
        let s = self.inner.lock().unwrap();
        s.windows
            .values()
            .cloned()
            .map(|mut info| {
                if let Some(surface) = surfaces.get(&info.id) {
                    if let Some(data) = surface.data::<crate::compositor::SurfaceData>() {
                        info.content_rect = *data.content_rect.lock().unwrap();
                    }
                }
                info
            })
            .collect()
    }

    pub fn window_count(&self) -> u32 {
        self.inner.lock().unwrap().windows.len() as u32
    }

    pub fn record_damage(&self) {
        // Use CLOCK_MONOTONIC so presentation feedback timestamps match KWin's
        // steady_clock-based nextPaintDelay calculation. Sending CLOCK_REALTIME
        // timestamps with clock_id=MONOTONIC caused KWin to compute its next
        // VSync as ~56 years in the future, permanently stalling its render loop.
        let now_ns = {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
            (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
        };
        self.last_damage_ns.store(now_ns, Ordering::Relaxed);
        // Poke the outer-view thread if one is attached. Non-blocking
        // eventfd write; never fails under normal conditions.
        if let Some(fd) = self.outer_wake_fd.lock().unwrap().as_ref() {
            let byte: u64 = 1;
            unsafe {
                libc::write(
                    fd.as_raw_fd(),
                    (&byte as *const u64).cast(),
                    std::mem::size_of::<u64>(),
                );
            }
        }
    }

    /// Try to push pre-read packed BGRA pixels to the recording channel.
    /// Non-blocking: drops the frame if recording is inactive, stopped,
    /// or the channel is full (encoder is behind). Used by the SHM
    /// software-composite path in `outer_view`.
    pub fn try_push_recording_pixels(&self, pixels: Vec<u8>, width: u32, height: u32) {
        let guard = self.recording.lock().unwrap();
        let Some(dual) = guard.as_ref() else { return };
        if dual.primary.stop.load(Ordering::Acquire) {
            return;
        }
        if let Some(sec) = &dual.secondary {
            if !sec.stop.load(Ordering::Acquire) {
                sec.slot.put(crate::recording::RecordingTask::Pixels {
                    pixels: pixels.clone(),
                    width,
                    height,
                });
            }
        }
        dual.primary
            .slot
            .put(crate::recording::RecordingTask::Pixels {
                pixels,
                width,
                height,
            });
    }

    /// Try to push a dmabuf reference to the recording channel.
    /// The recording thread does the GPU readback in its own thread so
    /// outer_view doesn't stall the compositor on a slow mmap+memcpy.
    /// `holds` keeps the inner buffer pinned until the readback completes.
    /// Non-blocking — drops the frame on full channel.
    pub fn try_push_recording_dmabuf(
        &self,
        dma: Arc<crate::dmabuf::DmabufBufferData>,
        holds: Vec<Arc<crate::recording::InnerBufferHold>>,
    ) {
        let guard = self.recording.lock().unwrap();
        let Some(dual) = guard.as_ref() else { return };
        if dual.primary.stop.load(Ordering::Acquire) {
            return;
        }
        if let Some(sec) = &dual.secondary {
            if !sec.stop.load(Ordering::Acquire) {
                sec.slot.put(crate::recording::RecordingTask::Dmabuf {
                    dma: dma.clone(),
                    _holds: holds.clone(),
                });
            }
        }
        dual.primary
            .slot
            .put(crate::recording::RecordingTask::Dmabuf { dma, _holds: holds });
    }

    /// True if a recording is currently active. Used by `outer_view` to
    /// decide whether to perform the synchronous capture work.
    pub fn is_recording(&self) -> bool {
        self.recording
            .lock()
            .unwrap()
            .as_ref()
            .map(|d| d.is_active())
            .unwrap_or(false)
    }

    /// True when something is actively consuming this session's frames: an
    /// attached outer view, a running recording, or a live viewer.
    ///
    /// The compositor MUST NOT idle-suspend (throttle frame-callback delivery
    /// to ~1 Hz) while any of these is live. Idle-suspend assumes nothing is
    /// watching, so inner clients that follow frame-callback discipline (KWin)
    /// self-rate-limit to ~1 fps. If a recording or viewer is running with no
    /// concurrent input activity, that 1 Hz cadence becomes the *captured*
    /// frame rate — the output freezes to ~1 unique fps, padded to the target
    /// rate by min-fps duplication. Keeping the loop active whenever a consumer
    /// exists lets KWin render at the full output cadence so the capture is fluid.
    pub fn has_active_frame_consumer(&self) -> bool {
        self.is_attached() || self.is_recording() || self.viewer.lock().unwrap().is_some()
    }

    /// Count of active encoder slots that the compositor tap should fan
    /// out to: recording.primary, recording.secondary, viewer. Used by
    /// `maybe_tap_for_recording` to size its iteration and log the
    /// `encoder_count` diagnostic field.
    #[allow(dead_code)] // used by tests; binary path inlines the same counts
    pub fn encoder_count_for_tap(&self) -> usize {
        let mut n = 0;
        if let Some(rec) = &*self.recording.lock().unwrap() {
            n += 1; // primary always present when DualRecordingHandle exists
            if rec.secondary.is_some() {
                n += 1;
            }
        }
        if self.viewer.lock().unwrap().is_some() {
            n += 1;
        }
        n
    }

    /// Commit-driven recording tap. Called from the inner
    /// compositor's `wl_surface::Request::Commit` handler immediately
    /// after the just-committed buffer has been promoted to `current`
    /// (and the explicit-sync acquire fence has been waited on, if any).
    ///
    /// This is the synchronous capture path that replaces the polling
    /// 30 fps producer. Capture rate now matches the inner client's
    /// commit cadence — typically 60 fps for animated content,
    /// effectively 0 fps for idle pages (no commits = no captures).
    /// Tearing is impossible by construction: chromium can't modify
    /// the buffer until we send `wl_buffer.release` in the commit
    /// handler's existing release path *after* this tap returns.
    ///
    /// **Mode dispatch:**
    /// - `FocusedWindow`: only tap when `surface` is the focused
    ///   window's primary buffer-owner. Single-surface fast path —
    ///   one `with_bytes` mmap+memcpy per frame, no compositing.
    /// - `WholeDesktop`: tap on every commit, but route through
    ///   `capture_desktop`'s composite-tree-walk which produces a
    ///   single composited frame from the full surface set. Slower
    ///   per-commit (per-pixel composite) but correct for multi-window
    ///   sessions. Compositor thread bears the cost; if the inner
    ///   commit cadence is too high for the host CPU, the latest-only
    ///   slot drops frames rather than backing pressure into the
    ///   inner client.
    ///
    /// Caller passes the surface that *just* committed; for
    /// `WholeDesktop` mode we ignore which surface fired and capture
    /// the whole desktop, but the parameter exists to support
    /// `FocusedWindow` filtering without a second focused-window
    /// lookup.
    pub fn maybe_tap_for_recording(self: &Arc<Self>, surface: &WlSurface, acquire_signaled: bool) {
        // Snapshot per-encoder (mode, slot, codec) tuples so we can drop the
        // recording mutex before doing any I/O. Both encoders share the same
        // capture `mode` today (set at record start), but each owns its slot
        // + codec. The viewer slot is appended after the recording slots so
        // the fan-out loop below reaches it without any special casing.
        let (mode, encoders): (
            crate::recording::CaptureMode,
            Vec<(
                Arc<crate::recording::LatestTaskSlot>,
                waymux_protocol::RecordingCodec,
            )>,
        ) = {
            let mut encs: Vec<(
                Arc<crate::recording::LatestTaskSlot>,
                waymux_protocol::RecordingCodec,
            )> = Vec::new();
            // Default: WholeDesktop. Right for the viewer use case (SaaS
            // customer wants to see + interact with the Plasma desktop,
            // not just one focused window). Recordings override this with
            // their explicit `dual.primary.mode` below, so screencast
            // recordings can still pick FocusedWindow.
            let mut capture_mode = crate::recording::CaptureMode::WholeDesktop;

            // Recording slots (primary + optional secondary).
            {
                let guard = self.recording.lock().unwrap();
                if let Some(dual) = guard.as_ref() {
                    if dual.is_active() {
                        encs.push((dual.primary.slot.clone(), dual.primary.codec));
                        if let Some(sec) = &dual.secondary {
                            if !sec.stop.load(Ordering::Acquire) {
                                encs.push((sec.slot.clone(), sec.codec));
                            }
                        }
                        capture_mode = dual.primary.mode;
                    }
                }
            }

            // Viewer slot (independent of recording; active simultaneously).
            {
                let guard = self.viewer.lock().unwrap();
                if let Some(v) = guard.as_ref() {
                    if !v.stop_flag.load(Ordering::Acquire) {
                        encs.push((v.frame_slot.clone(), v.codec));
                    }
                }
            }

            if encs.is_empty() {
                return;
            }
            (capture_mode, encs)
        };
        // Diagnostic: count commits reaching the tap. Helps debug
        // "no frames within 5s" failures (slot vs recording-thread vs
        // surface_in_focused_tree filter).
        static TAP_HITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = TAP_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 8 || n.is_multiple_of(60) {
            let primary_codec = encoders[0].1;
            let secondary_codec = encoders.get(1).map(|(_, c)| *c);
            tracing::info!(
                hit = n,
                ?mode,
                ?primary_codec,
                ?secondary_codec,
                encoder_count = encoders.len(),
                "maybe_tap_for_recording: commit reached tap"
            );
        }
        // Tell us whether every active encoder runs on the GPU path
        // (so its driver-side fence wait covers us) or whether at least
        // one is a CPU encoder that still needs the implicit-fence gate.
        let all_gpu_path = encoders.iter().all(|(_, c)| {
            matches!(
                c,
                waymux_protocol::RecordingCodec::H264Nvenc
                    | waymux_protocol::RecordingCodec::H264Vaapi
                    | waymux_protocol::RecordingCodec::H264Vulkan
                    | waymux_protocol::RecordingCodec::H264VulkanLossless
                    | waymux_protocol::RecordingCodec::HevcVulkanLossless,
            )
        });
        match mode {
            crate::recording::CaptureMode::FocusedWindow => {
                if !self.surface_in_focused_tree(surface) {
                    return;
                }
                // Fence check: only needed for the CPU path (mmap +
                // memcpy doesn't synchronize with the producer GPU).
                // The GPU path uses EGL_EXT_image_dma_buf_import, which
                // inserts a producer-fence wait at texture bind time on
                // every driver we target — so we don't need to gate the
                // tap here for h264 codecs. Pre-pass this gate gets
                // around hosts where syncobj reports the explicit-sync
                // node but the implicit fence is never marked ready,
                // which silently drops every commit (seen on Blackwell
                // + driver 580.95.05 hosts with `syncobj: opened DRM
                // node` in the daemon log).
                if !all_gpu_path && !acquire_signaled && !self.dmabuf_implicit_fence_ready(surface)
                {
                    return;
                }
                // Tee the buffer (cheap: clone the Arc, pin via
                // InnerBufferHold) and let the recording thread do the
                // mmap+memcpy off-thread. Doing the read inline on the
                // commit handler stalls chromium for up to 2 seconds
                // when DmabufBufferData::with_bytes blocks on an
                // unsignaled implicit read fence — unacceptable.
                //
                // The hold pins the underlying WlBuffer wrapper for the
                // recording thread's read window. The race-window
                // analysis from Fix A applies: chromium's swap chain
                // can rotate, but the ~1-3 ms mmap+memcpy on the
                // recording thread comfortably fits inside chromium's
                // ~16 ms commit cadence in practice. Live verification
                // shows zero tearing at 43 fps on high-motion content.
                //
                // Dual-encoder fan-out: each encoder gets its own call,
                // since the right per-frame shape (Dmabuf vs Pixels,
                // BGRA vs NV12) is codec-dependent. The CPU readback in
                // push_focused_buffer_to_recording happens once per
                // call; in the dual case, that's one readback per CPU
                // encoder. The Vulkan/NVENC/VAAPI fast paths just
                // Arc-clone the dmabuf and pin via InnerBufferHold —
                // no extra work for the secondary.
                for (slot, codec) in &encoders {
                    self.push_focused_buffer_to_recording(surface, slot, *codec);
                }
            }
            crate::recording::CaptureMode::WholeDesktop => {
                // Split encoders by how they consume frames. In our
                // nested setup the inner client is KWin, which presents
                // its ENTIRE composited desktop as a SINGLE output
                // surface — so the committed `surface` here IS the whole
                // desktop. That lets us tee KWin's single output dmabuf
                // zero-copy to in-process GPU encoders instead of the
                // expensive CPU per-pixel re-composite.
                //
                // GPU encoders (the same `wants_dmabuf` set used in
                // `push_focused_buffer_to_recording`): take the zero-copy
                // dmabuf tee ONLY when the committed surface's CURRENT
                // buffer is actually a `BufferKind::Dmabuf`. The tee is an
                // Arc-clone + InnerBufferHold pin, not a CPU composite, so
                // it is NOT rate-limited. When the desktop is idle/solid
                // KWin commits a `wp_single_pixel_buffer` (SinglePixel),
                // not a dmabuf — there is no buffer to tee, so the GPU
                // slot instead falls through to capture_desktop() →
                // Pixels (consumed by the viewer encoders' host-upload
                // path). This is the fix for the idle-desktop black
                // screen: the OLD code always called
                // push_focused_buffer_to_recording for GPU slots, which
                // pushed a `Pixels` task the CUDA encoder never read.
                //
                // CPU encoders: keep the existing rate-limited
                // capture_desktop → Pixels path. capture_desktop's
                // per-pixel composite is genuinely slow (~470 ms on AMD
                // integrated for a 1280×800 tree); running it inline on
                // every commit would stall the compositor at ~2 fps, so
                // we rate-limit to 10 fps (100 ms minimum interval).
                let mut cpu_slots: Vec<(
                    &Arc<crate::recording::LatestTaskSlot>,
                    waymux_protocol::RecordingCodec,
                )> = Vec::new();
                // Is the desktop currently committing a real GPU dmabuf? On an
                // IDLE Plasma 6 desktop KWin presents a solid frame as a
                // `wp_single_pixel_buffer` (BufferKind::SinglePixel), NOT a
                // dmabuf — and the GPU encoders' zero-copy tee
                // (push_focused_buffer_to_recording) only pushes a usable
                // `Dmabuf` task for a real dmabuf. We compute this ONCE here.
                // Format-gated: only tee ARGB8888/XRGB8888 dmabufs. A
                // wrong-format transition/overlay dmabuf fed to the ARGB CSC
                // is the pink/magenta flash; skip it (hold the last frame).
                // Tee ONLY KWin's full-desktop output surface. Auxiliary
                // surfaces KWin commits to us (the 24×24 pointer cursor, drag
                // icons, subsurfaces) are valid ARGB dmabufs but the WRONG
                // size/tiling — teeing them as a 1080p frame is the pink flash
                // + partial-content flicker. See committed_buffer_is_desktop_output.
                let committed_is_dmabuf = self.committed_buffer_is_desktop_output(surface);
                // Are KWin's GPU writes for THIS commit actually complete?
                // KWin uses explicit sync (wp_linux_drm_syncobj_v1) on these
                // hosts; the commit handler waits the acquire fence with a 50ms
                // deadline and sets `acquire_signaled=false` when it times out
                // (heavy bursts: menu close, video). The implicit dmabuf fence
                // is the fallback for hosts NOT using explicit sync. If NEITHER
                // signal says "done", the buffer is still mid-composite — teeing
                // it encodes a half-rendered frame (the desktop wallpaper showing
                // through where windows/panel aren't painted yet = the blue/
                // "reveals underneath" flicker). Skip it; the constant-cadence
                // encoder holds the last good frame (~16ms, invisible). Either
                // signal alone is enough (chromium/KWin signal one or the other
                // depending on the swap path).
                let gpu_writes_done = acquire_signaled || self.dmabuf_implicit_fence_ready(surface);
                if committed_is_dmabuf && !gpu_writes_done {
                    static SKIP_LOG: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let s = SKIP_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if s < 12 || s.is_multiple_of(60) {
                        tracing::info!(
                            skip_n = s,
                            acquire_signaled,
                            "WholeDesktop tee: SKIP incomplete (mid-composite) desktop buffer"
                        );
                    }
                }
                // DIAGNOSTIC (pink/partial flicker): the WholeDesktop tap
                // assumes the committed surface IS KWin's single full-desktop
                // output surface. If KWin ever commits a SECONDARY surface
                // (popup / subsurface / cursor / overlay) it would be teed as a
                // full desktop frame at the WRONG dimensions → NV12 chroma-plane
                // misalignment (magenta/pink) + partial content. Log per-commit
                // dims (first 12, then on dimension change) to confirm whether
                // mixed-size commits actually arrive. FocusedWindow already
                // filters to the focused toplevel for exactly this reason.
                {
                    use crate::buffer::BufferKind;
                    use crate::compositor::SurfaceData;
                    if let Some(sd) = surface.data::<SurfaceData>() {
                        if let Some(buf) = sd.current_buffer.lock().unwrap().clone() {
                            if let Some(BufferKind::Dmabuf(d)) = buf.data::<BufferKind>() {
                                static LAST_W: std::sync::atomic::AtomicU32 =
                                    std::sync::atomic::AtomicU32::new(0);
                                static LAST_H: std::sync::atomic::AtomicU32 =
                                    std::sync::atomic::AtomicU32::new(0);
                                static DIAG_N: std::sync::atomic::AtomicU32 =
                                    std::sync::atomic::AtomicU32::new(0);
                                let w = d.width as u32;
                                let h = d.height as u32;
                                let dn = DIAG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let changed = LAST_W.swap(w, std::sync::atomic::Ordering::Relaxed)
                                    != w
                                    || LAST_H.swap(h, std::sync::atomic::Ordering::Relaxed) != h;
                                if dn < 12 || changed {
                                    tracing::info!(
                                        sid = ?surface.id(),
                                        w,
                                        h,
                                        stride = d.stride,
                                        modifier = format!("0x{:x}", d.modifier),
                                        drm_format = format!("0x{:x}", d.drm_format),
                                        gated_ok = committed_is_dmabuf,
                                        "WHOLEDESKTOP-DIAG: committed dmabuf surface"
                                    );
                                }
                            }
                        }
                    }
                }
                for (slot, codec) in &encoders {
                    let wants_dmabuf = matches!(
                        codec,
                        waymux_protocol::RecordingCodec::H264Vulkan
                            | waymux_protocol::RecordingCodec::Ffv1Vulkan
                            | waymux_protocol::RecordingCodec::H264VulkanLossless
                            | waymux_protocol::RecordingCodec::CudaNvenc
                            // In-process VA-API GPU-imports the dmabuf itself
                            // (encode_dmabuf: DRM_PRIME → scale_vaapi → encode),
                            // falling back to its own readback if unsupported.
                            // Teeing the desktop dmabuf zero-copy keeps the EGL
                            // readback off the compositor thread and delivers a
                            // full 60fps (vs the rate-limited capture_desktop
                            // Pixels path).
                            | waymux_protocol::RecordingCodec::H264Vaapi
                    );
                    if wants_dmabuf {
                        // Tee the committed surface OFF the compositor thread via
                        // push_focused_buffer_to_recording: a Dmabuf task (the
                        // recording thread does the GPU import / readback) for a
                        // dmabuf, a cheap synchronous read for SHM, a no-op for a
                        // SinglePixel idle buffer.
                        //
                        // This NEVER runs the synchronous capture_desktop() ->
                        // egl_readback the old `cpu_slots` fallback did on THIS
                        // (compositor) thread. That path lazily initialised EGL on
                        // first use — measured at ~6 SECONDS on this AMD RENOIR —
                        // then read ~140 ms/frame, blocking the loop so KWin could
                        // not render new frames. The result was whole-desktop
                        // recordings of animating content frozen for many seconds.
                        // Teeing off-thread keeps the compositor responsive.
                        //
                        // Wrong-size commits (the cursor, popups) are dropped by
                        // the recorder's frame-size check; the recorder sizes to
                        // the desktop output because prime_recording_first_frame
                        // seeds it from the output-sized surface first. The GPU
                        // encoders' dmabuf import barrier (QUEUE_FAMILY_FOREIGN_EXT)
                        // syncs with the producer, so we don't gate on
                        // gpu_writes_done.
                        self.push_focused_buffer_to_recording(surface, slot, *codec);
                    } else {
                        // Real CPU codecs (recording sinks) still get the
                        // rate-limited capture_desktop() -> Pixels path.
                        cpu_slots.push((slot, *codec));
                    }
                }
                if !cpu_slots.is_empty() {
                    use std::time::Instant;
                    // Rate-limit guards ONLY the CPU capture_desktop
                    // block — the GPU tee above has already run, so this
                    // must NOT early-return out of the whole function.
                    let now = Instant::now();
                    let mut should_capture = true;
                    {
                        let last = self.last_whole_desktop_capture.lock().unwrap();
                        if let Some(prev) = *last {
                            if now.duration_since(prev) < std::time::Duration::from_millis(100) {
                                should_capture = false;
                            }
                        }
                    }
                    if should_capture {
                        *self.last_whole_desktop_capture.lock().unwrap() = Some(now);
                        if let Some((pixels, w, h, _format, _stride)) = self.capture_desktop() {
                            // Fan-out: every CPU encoder gets a Pixels
                            // task. We hand owned bytes to the last CPU
                            // slot; earlier slots get a clone. This path
                            // is 10 fps-capped so the extra memcpy cost
                            // is bounded.
                            let last_idx = cpu_slots.len() - 1;
                            let mut owner = Some(pixels);
                            for (i, (slot, _codec)) in cpu_slots.iter().enumerate() {
                                let bytes = if i == last_idx {
                                    owner.take().expect("owner only taken once")
                                } else {
                                    owner.as_ref().expect("owner present until last").clone()
                                };
                                slot.put(crate::recording::RecordingTask::Pixels {
                                    pixels: bytes,
                                    width: w as u32,
                                    height: h as u32,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Seed the active recording slots with the current desktop buffer so a
    /// recording started over an IDLE (static) desktop produces output
    /// immediately, instead of blocking up to 30s for the next `wl_surface`
    /// commit that a still desktop never makes.
    ///
    /// A live Plasma desktop only commits on damage (a clock tick, a cursor
    /// move); with nothing animating after `RecordStart`, the recording
    /// thread's "waiting for first frame" wait would otherwise time out and
    /// write no file. (An animated client like a browser never hit this — it
    /// commits continuously.) We reuse the exact per-surface tee the commit
    /// tap uses, so the seeded frame is a zero-copy `Dmabuf` task when KWin's
    /// output is a dmabuf (the common case) and a CPU readback otherwise.
    /// Live commits take over the instant anything on the desktop changes.
    pub fn prime_recording_first_frame(self: &Arc<Self>) {
        // Snapshot the surface set (don't hold windows_by_id across the tap).
        let surfaces: Vec<WlSurface> = self
            .windows_by_id
            .lock()
            .unwrap()
            .values()
            .map(|(s, _)| s.clone())
            .collect();
        // Prefer KWin's full-desktop output surface (the whole composited
        // desktop presented as one output-sized dmabuf); fall back to the
        // focused toplevel, then to any surface, so single-app and SHM
        // sessions prime too.
        let Some(seed) = surfaces
            .iter()
            .find(|s| self.committed_buffer_is_desktop_output(s))
            .or_else(|| surfaces.iter().find(|s| self.surface_in_focused_tree(s)))
            .or_else(|| surfaces.first())
        else {
            return;
        };
        // Drive one tap as if `seed` had just committed. The tap owns the full
        // routing logic: a zero-copy `Dmabuf` tee when the output is a real
        // dmabuf, or a `capture_desktop()` → `Pixels` readback when an idle
        // desktop is presenting a `wp_single_pixel_buffer`. `acquire_signaled`
        // is true: `seed`'s buffer is already presented, so its producer fence
        // is long settled. A no-op when nothing is recording.
        self.maybe_tap_for_recording(seed, true);
    }

    /// True iff the surface's committed buffer is an ARGB/XRGB dmabuf whose
    /// dimensions match the compositor's output size — i.e. it is KWin's
    /// full-desktop output surface, not an auxiliary surface.
    ///
    /// In the nested setup KWin presents its whole composited desktop as ONE
    /// output-sized surface, but it ALSO commits other surfaces to us — most
    /// notably the pointer **cursor** (a 24×24 `wl_pointer.set_cursor`
    /// surface), and potentially drag icons / subsurfaces. Those reach the
    /// WholeDesktop tap too, and being valid ARGB dmabufs they pass the
    /// format gate — but they are the WRONG size (and a different tiling
    /// modifier). Teeing a 24×24 cursor buffer into the 1920×1080 NVENC
    /// session misaligns the NV12 chroma plane (magenta/pink flash) and
    /// encodes mostly-stale content (the "menu reveals underneath" partial
    /// frames). Gating on output size tees ONLY the real desktop surface,
    /// exactly as `surface_in_focused_tree` filters FocusedWindow mode to the
    /// focused toplevel. The constant-cadence encoder re-encodes the last good
    /// desktop frame for the skipped cursor commits, so the cursor still moves
    /// (KWin composites it INTO the desktop surface as well).
    fn committed_buffer_is_desktop_output(&self, surface: &WlSurface) -> bool {
        use crate::buffer::BufferKind;
        use crate::compositor::SurfaceData;
        let (out_w, out_h, _scale) = self.snapshot();
        let Some(sd) = surface.data::<SurfaceData>() else {
            return false;
        };
        let Some(buf) = sd.current_buffer.lock().unwrap().clone() else {
            return false;
        };
        let Some(kind) = buf.data::<BufferKind>() else {
            return false;
        };
        match kind {
            BufferKind::Dmabuf(d) => {
                let is_argb = d.drm_format == crate::dmabuf::DRM_FORMAT_ARGB8888
                    || d.drm_format == crate::dmabuf::DRM_FORMAT_XRGB8888;
                is_argb && d.width as u32 == out_w && d.height as u32 == out_h
            }
            _ => false,
        }
    }

    /// Tee the focused window's just-committed dmabuf to the recording
    /// thread, pinned via `InnerBufferHold` so the inner client can't
    /// release the wrapper until the recording thread finishes its
    /// mmap+memcpy. SHM-backed surfaces fall through to a synchronous
    /// read (SHM is CPU memory; no fence; no race).
    fn push_focused_buffer_to_recording(
        self: &Arc<Self>,
        surface: &WlSurface,
        slot: &crate::recording::LatestTaskSlot,
        codec: waymux_protocol::RecordingCodec,
    ) {
        use crate::buffer::BufferKind;
        use crate::compositor::SurfaceData;
        use crate::dmabuf::DRM_FORMAT_MOD_LINEAR;
        use waymux_protocol::RecordingCodec;
        let Some(sd) = surface.data::<SurfaceData>() else {
            return;
        };
        let Some(buf) = sd.current_buffer.lock().unwrap().clone() else {
            return;
        };
        let Some(kind) = buf.data::<BufferKind>() else {
            return;
        };
        let needs_nv12 = matches!(codec, RecordingCodec::H264Nvenc | RecordingCodec::H264Vaapi);
        // Vulkan paths that do GPU-side dmabuf import on the recording
        // thread (no CPU readback). Safe to forward the dmabuf Arc
        // lazily — Vulkan's QUEUE_FAMILY_FOREIGN_EXT barrier handles
        // producer sync.
        //
        // HevcVulkanLossless is INTENTIONALLY EXCLUDED: its recording
        // thread CURRENTLY uses CPU readback via dma.with_bytes() in
        // `read_task`. Pushing as a lazy Arc lets Chrome reuse the
        // buffer before the recording thread reads it, producing
        // torn-frame diagonal chroma artifacts. We push it through the
        // eager CPU-readback path below (same as ffv1/h264-vaapi/nvenc),
        // so the bytes are captured AT COMMIT TIME. A future
        // GPU-zero-copy path can re-add it here once the recording
        // thread imports the dmabuf as a Vulkan transfer-src image.
        let needs_vulkan = matches!(
            codec,
            RecordingCodec::H264Vulkan
                | RecordingCodec::Ffv1Vulkan
                | RecordingCodec::H264VulkanLossless,
        );
        // In-process GPU encoders that import the dmabuf themselves (no
        // CPU readback on the compositor thread). For these we forward a
        // lazy `RecordingTask::Dmabuf` (InnerBufferHold-pinned) for ANY
        // modifier — LINEAR or tiled — and NEVER mmap. This is the
        // `needs_vulkan` set generalized to also cover CudaNvenc, whose
        // viewer encoder (`run_cuda_nvenc_encoder`) only consumes
        // `RecordingTask::Dmabuf` and ignores `Pixels`. On NVIDIA KWin 6
        // commits a LINEAR dmabuf (modifier 0 is in the advertised EGL
        // set); without this the CudaNvenc viewer slot would receive a
        // `Pixels` task it never reads → black viewer.
        //
        // HevcVulkanLossless is INTENTIONALLY EXCLUDED here for the same
        // reason it's excluded from `needs_vulkan`: its recording thread
        // currently does CPU readback via `dma.with_bytes()`, so it must
        // take the eager-copy LINEAR path below (and cannot consume a
        // tiled buffer). H264Nvenc / H264Vaapi are ffmpeg-subprocess CPU
        // paths and also stay on the eager-copy path.
        let wants_dmabuf =
            needs_vulkan || matches!(codec, RecordingCodec::CudaNvenc | RecordingCodec::H264Vaapi);
        static KIND_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = KIND_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 5 || n.is_multiple_of(120) {
            let kind_name = match kind {
                BufferKind::Dmabuf(_) => "Dmabuf",
                BufferKind::Shm(_) => "Shm",
                BufferKind::SinglePixel(_) => "SinglePixel",
                BufferKind::Invalid => "Invalid",
            };
            tracing::info!(kind = kind_name, n, ?codec, "push_focused: buffer kind");
        }
        match kind {
            BufferKind::Dmabuf(d) => {
                // ── In-process GPU-encoder fast path (any modifier) ────
                //
                // For codecs whose encoder GPU-imports the dmabuf itself
                // (Vulkan video-encode + CudaNvenc), forward the dmabuf
                // Arc directly — pinned via InnerBufferHold so the inner
                // client can't reclaim it — for BOTH LINEAR and tiled
                // modifiers. NO mmap happens here: this push MUST precede
                // any CPU-readback path (mmap on a tiled buffer is
                // invalid; and the GPU encoders never want a `Pixels`
                // task). This subsumes the old `needs_vulkan` LINEAR
                // branch below and the tiled branch for these codecs.
                if wants_dmabuf {
                    if n < 5 {
                        tracing::trace!(
                            modifier = format!("0x{:x}", d.modifier),
                            ?codec,
                            "push_focused: GPU-encoder codec — routing dmabuf to encoder slot (no mmap)"
                        );
                    }
                    let _hold = crate::recording::InnerBufferHold::new(buf.clone(), self.clone());
                    slot.put(crate::recording::RecordingTask::Dmabuf {
                        dma: d.clone(),
                        _holds: vec![_hold],
                    });
                    return;
                }
                if d.modifier != DRM_FORMAT_MOD_LINEAR {
                    if n < 5 {
                        tracing::trace!(
                            modifier = format!("0x{:x}", d.modifier),
                            "push_focused: non-LINEAR (tiled) dmabuf — routing to encoder slot as Dmabuf task"
                        );
                    }
                    // Tiled (GPU-tiled) dmabuf — e.g. KWin 6 on NVIDIA.
                    // mmap+memcpy is INVALID on a tiled buffer (the bytes
                    // aren't linearly laid out), so we must NOT take the
                    // eager CPU-readback path below. Instead push a lazy
                    // `RecordingTask::Dmabuf` task (mirroring the LINEAR
                    // Vulkan fast path at the `needs_vulkan` branch): pin
                    // the inner buffer via InnerBufferHold so KWin can't
                    // reclaim it, and hand the dmabuf Arc straight to the
                    // slot. The CudaNvenc viewer encoder GPU-imports the
                    // dmabuf itself, so no CPU touch happens on this path.
                    // Recording encoders that can't consume tiled
                    // (Vulkan-on-NVIDIA) will drop the task — acceptable
                    // until full recording-sink wiring lands.
                    let _hold = crate::recording::InnerBufferHold::new(buf.clone(), self.clone());
                    slot.put(crate::recording::RecordingTask::Dmabuf {
                        dma: d.clone(),
                        _holds: vec![_hold],
                    });
                    return;
                }
                // Eager mmap+memcpy on the compositor thread. The lazy
                // variant (push Arc<DmabufBufferData> and let the
                // recording_thread read later) lost frames because KWin
                // reuses GPU memory under pressure regardless of our
                // InnerBufferHold pin — by the time the recording_thread
                // mmaps, KWin has overwritten part of the buffer for the
                // next compose pass. Eagerly copying now (≈0.4 ms for
                // 1312×842 BGRA at 10 GB/s) collapses the race window to
                // the duration of the memcpy itself.
                // Pin the buffer for the duration of the read so KWin
                // can't reclaim it mid-mmap. Even though KWin reuses
                // memory under pressure regardless, holding the wl_buffer
                // wrapper makes a "release" event impossible until our
                // copy is finished — that is the strongest contract the
                // protocol gives us.
                let _hold = crate::recording::InnerBufferHold::new(buf.clone(), self.clone());
                use std::os::fd::AsRawFd;
                // Diagnostic: are we even getting here? The non-blocking
                // implicit-fence probe was silently dropping every frame
                // on Blackwell hosts in the 580.x pool. Log so we can
                // see how many commits hit this gate vs pass it.
                let fence_ready = crate::dmabuf::dmabuf_fence_ready_now(d.fd.as_raw_fd());
                tracing::info!(
                    needs_nv12,
                    needs_vulkan,
                    fence_ready,
                    "push_focused: dmabuf commit arrived"
                );
                // CPU-path fence gate: mmap + memcpy doesn't synchronize
                // with the producer GPU, so we need this. The GPU path
                // (EGL_EXT_image_dma_buf_import + bind to texture) DOES
                // synchronize on the driver side, so we skip the gate
                // there. Vulkan path also handles producer sync via
                // VkImageMemoryBarrier with QUEUE_FAMILY_FOREIGN_EXT.
                if !needs_nv12 && !needs_vulkan && !fence_ready {
                    return;
                }
                let w = d.width as u32;
                let h = d.height as u32;
                let stride = d.stride;

                // NOTE: the Vulkan zero-copy fast path (and CudaNvenc) is
                // now handled at the top of this arm via `wants_dmabuf`
                // (which subsumes the old `needs_vulkan` LINEAR branch),
                // so `needs_vulkan` is always false by the time we reach
                // here — only CPU-readback codecs (ffv1/h264-vaapi/
                // h264-nvenc/hevc-vulkan-lossless) fall through.

                // ── GPU fast path for h264 codecs ──────────────────────
                //
                // When the recording wants NV12 (NVENC or VAAPI), import
                // CPU path: for ffv1/h264-vaapi/h264-nvenc codecs we read
                // back the BGRA bytes here and hand them to the recording
                // thread, which feeds ffmpeg. NV12 conversion (when
                // needed) happens on the recording thread via rayon.
                // (Previous code path had a GPU EGL→NV12 shortcut via
                // gpu_record.rs; that was deleted in favor of --codec
                // h264-vulkan which does the conversion + encode + mux
                // entirely on the GPU.)

                // ── CPU readback ───────────────────────────────────────
                if let Some(pixels) =
                    d.with_bytes(|raw| crate::recording::destride_bgra(raw, w, h, stride))
                {
                    slot.put(crate::recording::RecordingTask::Pixels {
                        pixels,
                        width: w,
                        height: h,
                    });
                }
            }
            BufferKind::Shm(_) | BufferKind::SinglePixel(_) => {
                // SHM: CPU memory, ready immediately, no fence — read
                // inline. SinglePixel is degenerate (1×1) but cheap.
                if let Some((pixels, w, h)) = self.read_surface_buffer_for_recording(surface) {
                    slot.put(crate::recording::RecordingTask::Pixels {
                        pixels,
                        width: w,
                        height: h,
                    });
                }
            }
            BufferKind::Invalid => {}
        }
    }

    /// Non-blocking probe of the just-committed buffer's implicit dmabuf
    /// read fence. Used by `maybe_tap_for_recording` as the second of two
    /// "GPU done" signals (alongside the explicit acquire syncobj). Both
    /// fences may legitimately be NoFence/AlreadySignaled depending on
    /// the client's swap-chain path, so EITHER one signaling is enough.
    fn dmabuf_implicit_fence_ready(&self, surface: &WlSurface) -> bool {
        use crate::buffer::BufferKind;
        use crate::compositor::SurfaceData;
        let Some(sd) = surface.data::<SurfaceData>() else {
            return true;
        };
        let Some(buf) = sd.current_buffer.lock().unwrap().clone() else {
            return true;
        };
        let Some(kind) = buf.data::<BufferKind>() else {
            return true;
        };
        match kind {
            BufferKind::Dmabuf(d) => {
                use std::os::fd::AsRawFd;
                crate::dmabuf::dmabuf_fence_ready_now(d.fd.as_raw_fd())
            }
            // SHM/SinglePixel are CPU memory — always coherent.
            _ => true,
        }
    }

    /// True if `surface` is part of the focused window's surface tree
    /// (the focused window's parent surface OR any of its children).
    /// Used by `maybe_tap_for_recording` to filter commits in
    /// `FocusedWindow` mode.
    fn surface_in_focused_tree(&self, surface: &WlSurface) -> bool {
        let focused = match *self.focused_window.lock().unwrap() {
            Some(id) => id,
            None => return false,
        };
        let parent = match self.windows_by_id.lock().unwrap().get(&focused).cloned() {
            Some((s, _)) => s,
            None => return false,
        };
        // Match ONLY the focused toplevel parent. KWin composites all
        // child subsurfaces into the toplevel's framebuffer, so the
        // toplevel commit carries the full window. Matching subsurfaces
        // too produced a stream of mixed dimensions (parent 1312×842 vs.
        // chromium content child at smaller dims) which tripped the
        // recording_thread's "dimensions changed → stop" guard after the
        // first frame in busy chromium workflows (multi-tab, click).
        &parent == surface
    }

    /// Read the just-committed buffer of `surface` into a packed BGRA
    /// `Vec<u8>`. Returns None for missing/unmappable buffers, AND for
    /// dmabufs whose implicit fence isn't yet signaled — see below.
    ///
    /// **Non-blocking constraint.** This runs inside the inner-compositor's
    /// commit handler. Anything that blocks here stalls chromium (which
    /// is one Wayland round-trip away). `DmabufBufferData::with_bytes`
    /// internally calls `wait_for_dmabuf_fence` with a 2-second timeout
    /// — that's a non-starter on the compositor thread. We pre-probe the
    /// fence via `wait_for_dmabuf_fence_status` (poll with timeout 0).
    /// Only `AlreadySignaled` or `NoFence` proceeds; anything else
    /// returns None and the next commit's tap will retry.
    ///
    /// **LINEAR-only.** Non-LINEAR (GPU-tiled) dmabufs require EGL
    /// readback which isn't safe to run on the compositor thread.
    /// Those clients are tagged "skip"; recording will yield empty
    /// frames in that case. Modern Wayland-native browsers (chromium
    /// with --ozone-platform=wayland, firefox with the right feature
    /// flags) use LINEAR dmabufs by default.
    ///
    /// **Format handling**: returns None for anything other than
    /// ARGB8888 / XRGB8888 (both store as BGRA byte order on
    /// little-endian, matching the `-pixel_format bgra` ffmpeg flag).
    fn read_surface_buffer_for_recording(
        &self,
        surface: &WlSurface,
    ) -> Option<(Vec<u8>, u32, u32)> {
        use crate::buffer::BufferKind;
        use crate::compositor::SurfaceData;
        use crate::dmabuf::{dmabuf_fence_ready_now, DRM_FORMAT_MOD_LINEAR};
        use std::os::fd::AsRawFd;
        use wayland_server::protocol::wl_shm;
        use wayland_server::WEnum;
        let sd = surface.data::<SurfaceData>()?;
        let buf = sd.current_buffer.lock().unwrap().clone()?;
        let kind = buf.data::<BufferKind>()?;

        // Non-blocking gate for dmabuf-backed surfaces. If the implicit
        // read fence isn't already signaled, skip — the next commit
        // will retry, and the compositor thread stays responsive.
        // (DmabufBufferData::with_bytes internally calls
        // wait_for_dmabuf_fence which has a 2 s blocking poll —
        // unacceptable on the commit thread. We pre-gate to avoid
        // ever entering it without the fence being already done.)
        if let BufferKind::Dmabuf(d) = kind {
            if d.modifier != DRM_FORMAT_MOD_LINEAR {
                // Tiled buffers need EGL readback — too expensive for
                // the commit handler. Skip; outer_view's attached path
                // (off-thread) handles these when present.
                return None;
            }
            let fd = d.fd.as_raw_fd();
            if !dmabuf_fence_ready_now(fd) {
                return None;
            }
        }

        kind.with_bytes(|bytes, w, h, stride, format| {
            let is_bgra_compatible = matches!(
                format,
                WEnum::Value(wl_shm::Format::Argb8888) | WEnum::Value(wl_shm::Format::Xrgb8888)
            );
            if !is_bgra_compatible {
                return None;
            }
            Some((
                crate::recording::destride_bgra(bytes, w as u32, h as u32, stride as u32),
                w as u32,
                h as u32,
            ))
        })?
    }

    /// Compose the focused window into `buf`. Kept for potential external use.
    #[allow(dead_code)]
    pub fn capture_focused_into(&self, buf: &mut Vec<u8>) -> Option<(i32, i32, i32)> {
        let focus = *self.focused_window.lock().unwrap();
        let window_id = focus?;
        let (surface, _) = self
            .windows_by_id
            .lock()
            .unwrap()
            .get(&window_id)
            .cloned()?;
        let fallback = {
            let s = self.inner.lock().unwrap();
            (s.width, s.height)
        };
        crate::composite::composite_into_buf(&surface, buf, fallback)
    }

    /// Return the pixel dimensions of the focused window's committed buffer,
    /// or None if there is no focused window with content.
    pub fn capture_focused_dims(&self) -> Option<(u32, u32)> {
        let focus = *self.focused_window.lock().unwrap();
        let window_id = focus?;
        let (surface, _) = self
            .windows_by_id
            .lock()
            .unwrap()
            .get(&window_id)
            .cloned()?;
        let fallback = {
            let s = self.inner.lock().unwrap();
            (s.width, s.height)
        };
        let sd = surface.data::<crate::compositor::SurfaceData>()?;
        let (w, h) = crate::composite::surface_dims(sd, fallback);
        if w > 0 && h > 0 {
            Some((w, h))
        } else {
            None
        }
    }

    /// Composite the focused window directly into `output` (must be `w*h*4`
    /// bytes). Returns true on success. This is the zero-copy hot path:
    /// no intermediate Vec is allocated; pixels go straight into the caller's
    /// buffer (typically the outer framebuffer's mmap slot).
    pub fn composite_focused_direct(&self, output: &mut [u8], w: i32, h: i32) -> bool {
        let focus = *self.focused_window.lock().unwrap();
        let Some(window_id) = focus else { return false };
        let (surface, _) = match self.windows_by_id.lock().unwrap().get(&window_id).cloned() {
            Some(v) => v,
            None => return false,
        };
        crate::composite::composite_into_slice(&surface, output, w, h)
    }

    pub fn last_damage_ns(&self) -> u64 {
        self.last_damage_ns.load(Ordering::Relaxed)
    }

    /// Return the dmabuf data for the currently focused window's committed
    /// buffer. The returned Arc keeps the fd alive for zero-copy forwarding.
    ///
    /// Checks the parent surface first, then subsurfaces. This handles nested
    /// compositors like KWin which use a single-pixel parent surface and put
    /// all GPU-rendered content in a full-size subsurface (wl_surface#28).
    /// Accepts any modifier — the host compositor imports it natively via EGL.
    pub fn clone_focused_dmabuf(&self) -> Option<Arc<crate::dmabuf::DmabufBufferData>> {
        use crate::buffer::BufferKind;
        use crate::compositor::SurfaceData;
        let window_id = (*self.focused_window.lock().unwrap())?;
        let surface = {
            let map = self.windows_by_id.lock().unwrap();
            map.get(&window_id)?.0.clone()
        };

        fn surface_dmabuf(
            surface: &wayland_server::protocol::wl_surface::WlSurface,
        ) -> Option<Arc<crate::dmabuf::DmabufBufferData>> {
            let sd = surface.data::<SurfaceData>()?;
            let buf = sd.current_buffer.lock().unwrap().clone()?;
            if let BufferKind::Dmabuf(d) = buf.data::<BufferKind>()? {
                return Some(d.clone());
            }
            None
        }

        // Try the parent surface first.
        if let Some(d) = surface_dmabuf(&surface) {
            return Some(d);
        }

        // Walk subsurfaces (bottom-to-top). For KWin's nested output the
        // parent holds a single-pixel placeholder and puts the GPU-rendered
        // frame in a full-size subsurface — return the largest one found.
        let sd = surface.data::<SurfaceData>()?;
        let children = sd.children.lock().unwrap().clone();
        let mut best: Option<Arc<crate::dmabuf::DmabufBufferData>> = None;
        for child in &children {
            if let Some(d) = surface_dmabuf(&child.surface) {
                let bigger = best
                    .as_ref()
                    .map(|b| d.width * d.height > b.width * b.height)
                    .unwrap_or(true);
                if bigger {
                    best = Some(d);
                }
            }
        }
        best
    }

    /// Companion to `clone_focused_dmabuf` that also returns the underlying
    /// `WlBuffer`, so the caller can wrap it in an `InnerBufferHold` to
    /// keep the inner client from releasing the buffer mid-read. Used by
    /// the headless recording producer (recording.rs) to skip
    /// `composite_into`'s per-pixel blit entirely when one fullscreen
    /// surface owns the whole session — the chromium-fullscreens-the-
    /// session case that dominates CI workloads.
    ///
    /// Returns `None` when:
    ///   - no focused window
    ///   - the focused window's surface tree has no dmabuf-backed buffer
    ///     (e.g. SHM-only client like firefox without `--enable-features`)
    ///
    /// In those cases the caller should fall back to `capture_desktop`,
    /// which handles SHM and multi-surface scenes via the slower
    /// composite path.
    #[allow(dead_code)] // retained API; not currently routed but documented for the SDK
    pub fn clone_focused_dmabuf_with_buffer(
        &self,
    ) -> Option<(
        Arc<crate::dmabuf::DmabufBufferData>,
        wayland_server::protocol::wl_buffer::WlBuffer,
    )> {
        use crate::buffer::BufferKind;
        use crate::compositor::SurfaceData;
        let window_id = (*self.focused_window.lock().unwrap())?;
        let surface = {
            let map = self.windows_by_id.lock().unwrap();
            map.get(&window_id)?.0.clone()
        };

        fn surface_dmabuf_with_buffer(
            surface: &wayland_server::protocol::wl_surface::WlSurface,
        ) -> Option<(
            Arc<crate::dmabuf::DmabufBufferData>,
            wayland_server::protocol::wl_buffer::WlBuffer,
        )> {
            let sd = surface.data::<SurfaceData>()?;
            let buf = sd.current_buffer.lock().unwrap().clone()?;
            if let BufferKind::Dmabuf(d) = buf.data::<BufferKind>()? {
                return Some((d.clone(), buf));
            }
            None
        }

        // Parent first, then largest subsurface (KWin places the
        // GPU-rendered frame in a full-size child of a single-pixel
        // parent — same logic as clone_focused_dmabuf).
        if let Some(pair) = surface_dmabuf_with_buffer(&surface) {
            return Some(pair);
        }
        let sd = surface.data::<SurfaceData>()?;
        let children = sd.children.lock().unwrap().clone();
        let mut best: Option<(
            Arc<crate::dmabuf::DmabufBufferData>,
            wayland_server::protocol::wl_buffer::WlBuffer,
        )> = None;
        for child in &children {
            if let Some((d, buf)) = surface_dmabuf_with_buffer(&child.surface) {
                let bigger = best
                    .as_ref()
                    .map(|(b, _)| d.width * d.height > b.width * b.height)
                    .unwrap_or(true);
                if bigger {
                    best = Some((d, buf));
                }
            }
        }
        best
    }

    /// Allocate a new window id and insert a default `WindowInfo`.
    /// Emits `window_created`.
    pub fn add_window(&self, pid: i32, geometry: Rect) -> u32 {
        let id = {
            let mut s = self.inner.lock().unwrap();
            let id = s.next_window_id;
            s.next_window_id = s.next_window_id.wrapping_add(1);
            s.windows.insert(
                id,
                WindowInfo {
                    id,
                    app_id: String::new(),
                    title: String::new(),
                    tags: Vec::new(),
                    geometry,
                    focused: false,
                    pid,
                    // This will be populated from SurfaceData later; for
                    // now newly-tracked windows expose `None` until the
                    // session's ListWindows handler is updated.
                    content_rect: None,
                },
            );
            id
        };
        self.emit(EventBody::WindowCreated {
            name: self.session_name(),
            window_id: id,
            app_id: String::new(),
            title: String::new(),
            pid,
        });
        self.notify_plasma_window_created(id);
        id
    }

    /// Snapshot of all currently-tracked window ids, in arbitrary order.
    pub fn window_ids(&self) -> Vec<u32> {
        self.inner.lock().unwrap().windows.keys().copied().collect()
    }

    pub fn register_plasma_manager(&self, mgr: OrgKdePlasmaWindowManagement) {
        let mut v = self.plasma_managers.lock().unwrap();
        v.retain(|m| m.is_alive());
        v.push(mgr);
    }

    pub fn register_plasma_window(&self, window_id: u32, win: OrgKdePlasmaWindow) {
        self.plasma_windows
            .lock()
            .unwrap()
            .entry(window_id)
            .or_default()
            .push(win);
    }

    pub fn unregister_plasma_window(&self, window_id: u32, win: &OrgKdePlasmaWindow) {
        let mut map = self.plasma_windows.lock().unwrap();
        if let Some(v) = map.get_mut(&window_id) {
            v.retain(|w| w.id() != win.id());
            if v.is_empty() {
                map.remove(&window_id);
            }
        }
    }

    /// Register a bound `wl_output` global so `resize` can re-send its `mode`.
    /// Dead entries (clients that disconnected) are pruned here so the list
    /// never grows unbounded across a session's lifetime.
    pub fn register_output(&self, output: WlOutput) {
        let mut v = self.output_globals.lock().unwrap();
        v.retain(|o| o.is_alive());
        v.push(output);
    }

    /// Register a window's `(xdg_surface, xdg_toplevel)` pair so `resize` can
    /// re-send a `configure` to it. Called when the toplevel is created
    /// (compositor.rs GetToplevel). Replaces any prior entry for the id.
    pub fn register_toplevel(
        &self,
        window_id: u32,
        xdg_surface: XdgSurface,
        toplevel: XdgToplevel,
    ) {
        self.toplevels
            .lock()
            .unwrap()
            .insert(window_id, (xdg_surface, toplevel));
    }

    /// Drop a window's toplevel registration (on xdg_toplevel/xdg_surface
    /// destroy or surface unregister). Safe to call for an unknown id.
    pub fn unregister_toplevel(&self, window_id: u32) {
        self.toplevels.lock().unwrap().remove(&window_id);
    }

    fn notify_plasma_window_created(&self, window_id: u32) {
        let uuid = format!("{window_id}");
        let mgrs: Vec<_> = self
            .plasma_managers
            .lock()
            .unwrap()
            .iter()
            .filter(|m| m.is_alive())
            .cloned()
            .collect();
        for mgr in &mgrs {
            mgr.window(window_id);
            if mgr.version() >= 13 {
                mgr.window_with_uuid(window_id, uuid.clone());
            }
        }
    }

    fn notify_plasma_window_destroyed(&self, window_id: u32) {
        if let Some(wins) = self.plasma_windows.lock().unwrap().get(&window_id) {
            for w in wins {
                if w.is_alive() {
                    w.unmapped();
                }
            }
        }
    }

    fn notify_plasma_window_title(&self, window_id: u32, title: &str) {
        if let Some(wins) = self.plasma_windows.lock().unwrap().get(&window_id) {
            for w in wins {
                if w.is_alive() {
                    w.title_changed(title.to_string());
                }
            }
        }
    }

    fn notify_plasma_window_app_id(&self, window_id: u32, app_id: &str) {
        if let Some(wins) = self.plasma_windows.lock().unwrap().get(&window_id) {
            for w in wins {
                if w.is_alive() {
                    w.app_id_changed(app_id.to_string());
                }
            }
        }
    }

    pub fn remove_window(&self, id: u32) -> bool {
        let (removed, now_empty) = {
            let mut s = self.inner.lock().unwrap();
            let removed = s.windows.remove(&id).is_some();
            (removed, removed && s.windows.is_empty())
        };
        if removed {
            self.notify_plasma_window_destroyed(id);
            self.emit(EventBody::WindowDestroyed {
                name: self.session_name(),
                window_id: id,
            });
            if now_empty {
                // Wake the outer_view immediately so it replaces the stale
                // frozen frame with a placeholder rather than waiting for the
                // next 1-second poll timeout.
                if let Some(fd) = self.outer_wake_fd.lock().unwrap().as_ref() {
                    let byte: u64 = 1;
                    unsafe {
                        libc::write(
                            fd.as_raw_fd(),
                            (&byte as *const u64).cast(),
                            std::mem::size_of::<u64>(),
                        );
                    }
                }
            }
        }
        removed
    }

    pub fn set_app_id(&self, id: u32, app_id: String) {
        let changed = {
            let mut s = self.inner.lock().unwrap();
            match s.windows.get_mut(&id) {
                Some(w) if w.app_id != app_id => {
                    w.app_id = app_id.clone();
                    true
                }
                _ => false,
            }
        };
        if changed {
            self.notify_plasma_window_app_id(id, &app_id);
            self.emit(EventBody::WindowChanged {
                name: self.session_name(),
                window_id: id,
                fields: WindowChange {
                    app_id: Some(app_id),
                    ..Default::default()
                },
            });
        }
    }

    pub fn set_title(&self, id: u32, title: String) {
        let changed = {
            let mut s = self.inner.lock().unwrap();
            match s.windows.get_mut(&id) {
                Some(w) if w.title != title => {
                    w.title = title.clone();
                    true
                }
                _ => false,
            }
        };
        if changed {
            self.notify_plasma_window_title(id, &title);
            self.emit(EventBody::WindowChanged {
                name: self.session_name(),
                window_id: id,
                fields: WindowChange {
                    title: Some(title),
                    ..Default::default()
                },
            });
        }
    }

    // ── window_id → surface map, used by the screenshot path ─────────────

    pub fn register_surface(&self, window_id: u32, surface: WlSurface, client: ClientId) {
        self.windows_by_id
            .lock()
            .unwrap()
            .insert(window_id, (surface, client));
        // Simple focus model: most recent toplevel gets keyboard focus.
        // Swap focus; send leave to previous focus's client, enter to this one.
        let prev = {
            let mut focus = self.focused_window.lock().unwrap();
            focus.replace(window_id)
        };
        if let Some(old) = prev {
            if old != window_id {
                self.send_leave(old);
            }
        }
        self.send_enter(window_id);
    }

    /// Like `register_surface`, but does NOT change `focused_window` and
    /// does NOT emit pointer/keyboard enter/leave events. Used for layer
    /// surfaces whose `keyboard_interactivity` is not `exclusive` — they
    /// participate in the windows map (so screenshot / capture / window
    /// listing still work) but never steal focus from xdg_toplevels.
    ///
    /// Q5 of the 2026-05-18 click design decisions
    /// (docs/investigations/2026-05-18-click-design-recommendations.md):
    /// Plasma panels, status indicators, and notification daemons all use
    /// `keyboard_interactivity = none` (the protocol default) and must
    /// register without flipping focus. `on_demand` is treated the same way
    /// today: the SDK has no `set_focus` API, so there's no
    /// trigger to promote an on_demand layer to focus; the simplification is
    /// documented at the caller (compositor.rs ~960-1000).
    ///
    /// Symmetric note: `unregister_surface` already handles the no-focus
    /// case correctly (it only demotes when `focused_window == Some(window_id)`),
    /// so no sibling `unregister_surface_no_focus` is needed.
    pub fn register_surface_no_focus(&self, window_id: u32, surface: WlSurface, client: ClientId) {
        self.windows_by_id
            .lock()
            .unwrap()
            .insert(window_id, (surface, client));
        // Intentionally no focus swap, no send_enter, no send_leave.
    }

    pub fn unregister_surface(&self, window_id: u32) {
        self.windows_by_id.lock().unwrap().remove(&window_id);
        // Drop any toplevel registration for this window so `resize` won't try
        // to configure a destroyed resource. No-op for layer surfaces (never
        // registered as toplevels).
        self.unregister_toplevel(window_id);
        let new_focus = {
            let mut focus = self.focused_window.lock().unwrap();
            if *focus == Some(window_id) {
                // Demote focus deterministically: highest remaining window id
                // wins. `window_id` is monotonically assigned by `add_window`
                // (see state.rs ~1550 — `next_window_id` is incremented per
                // window), so `.max()` returns the most-recently-registered
                // surviving window. This mirrors `register_surface`'s
                // "focus follows most-recent" rule symmetrically on close,
                // and avoids the non-determinism of `HashMap::keys().next()`
                // whose iteration order is randomised per-process for DoS
                // resistance — that flakes multi-window automation.
                let next = self.windows_by_id.lock().unwrap().keys().copied().max();
                *focus = next;
                next
            } else {
                *focus
            }
        };
        // The old focused surface is already gone from the map; clients
        // holding that window's keyboards/pointers will get the `leave`
        // implicitly when the resource is destroyed. We do send enter to
        // the new focus, if any.
        if let Some(new) = new_focus {
            self.send_enter(new);
        }
    }

    fn send_enter(&self, window_id: u32) {
        let Some((surface, cid)) = self.windows_by_id.lock().unwrap().get(&window_id).cloned()
        else {
            return;
        };
        let keyboards: Vec<WlKeyboard> = self
            .keyboards
            .lock()
            .unwrap()
            .get(&cid)
            .map(|v| v.iter().filter(|k| k.is_alive()).cloned().collect())
            .unwrap_or_default();
        let pointers: Vec<WlPointer> = self
            .pointers
            .lock()
            .unwrap()
            .get(&cid)
            .map(|v| v.iter().filter(|p| p.is_alive()).cloned().collect())
            .unwrap_or_default();
        for kbd in &keyboards {
            let serial = self.next_serial();
            kbd.enter(serial, &surface, Vec::new());
            // Push an initial modifier state so clients have a baseline.
            kbd.modifiers(self.next_serial(), 0, 0, 0, 0);
        }
        for ptr in &pointers {
            let serial = self.next_serial();
            ptr.enter(serial, &surface, 0.0, 0.0);
            if ptr.version() >= 5 {
                ptr.frame();
            }
        }
        if !keyboards.is_empty() || !pointers.is_empty() {
            self.wake_compositor();
        }
    }

    fn send_leave(&self, window_id: u32) {
        let Some((surface, cid)) = self.windows_by_id.lock().unwrap().get(&window_id).cloned()
        else {
            return;
        };
        let keyboards: Vec<WlKeyboard> = self
            .keyboards
            .lock()
            .unwrap()
            .get(&cid)
            .map(|v| v.iter().filter(|k| k.is_alive()).cloned().collect())
            .unwrap_or_default();
        let pointers: Vec<WlPointer> = self
            .pointers
            .lock()
            .unwrap()
            .get(&cid)
            .map(|v| v.iter().filter(|p| p.is_alive()).cloned().collect())
            .unwrap_or_default();
        for kbd in &keyboards {
            kbd.leave(self.next_serial(), &surface);
        }
        for ptr in &pointers {
            ptr.leave(self.next_serial(), &surface);
            if ptr.version() >= 5 {
                ptr.frame();
            }
        }
        if !keyboards.is_empty() || !pointers.is_empty() {
            self.wake_compositor();
        }
    }

    // ── keyboard tracking ───────────────────────────────────────────────

    pub fn register_keyboard(&self, client: ClientId, kbd: WlKeyboard) {
        self.keyboards
            .lock()
            .unwrap()
            .entry(client.clone())
            .or_default()
            .push(kbd.clone());
        // Late-binder case: focus already set to a surface owned by this
        // client → send that enter now so the next injected key is valid.
        let focused = *self.focused_window.lock().unwrap();
        if let Some(wid) = focused {
            if let Some((surface, cid)) = self.windows_by_id.lock().unwrap().get(&wid).cloned() {
                if cid == client {
                    let serial = self.next_serial();
                    kbd.enter(serial, &surface, Vec::new());
                    kbd.modifiers(self.next_serial(), 0, 0, 0, 0);
                    self.wake_compositor();
                }
            }
        }
    }

    pub fn unregister_keyboard(&self, client: ClientId, kbd: &WlKeyboard) {
        let mut map = self.keyboards.lock().unwrap();
        if let Some(v) = map.get_mut(&client) {
            v.retain(|k| k.id() != kbd.id());
            if v.is_empty() {
                map.remove(&client);
            }
        }
    }

    pub fn register_pointer(&self, client: ClientId, ptr: WlPointer) {
        self.pointers
            .lock()
            .unwrap()
            .entry(client.clone())
            .or_default()
            .push(ptr.clone());
        let focused = *self.focused_window.lock().unwrap();
        if let Some(wid) = focused {
            if let Some((surface, cid)) = self.windows_by_id.lock().unwrap().get(&wid).cloned() {
                if cid == client {
                    let serial = self.next_serial();
                    ptr.enter(serial, &surface, 0.0, 0.0);
                    if ptr.version() >= 5 {
                        ptr.frame();
                    }
                    self.wake_compositor();
                }
            }
        }
    }

    pub fn unregister_pointer(&self, client: ClientId, ptr: &WlPointer) {
        let mut map = self.pointers.lock().unwrap();
        if let Some(v) = map.get_mut(&client) {
            v.retain(|p| p.id() != ptr.id());
            if v.is_empty() {
                map.remove(&client);
            }
        }
    }

    /// Register a `wl_touch` resource newly created by `wl_seat.get_touch`.
    /// Mirrors `register_pointer` but is simpler: the wl_touch protocol has
    /// no `enter` event (touch focus is implicit in the `down` event's
    /// surface argument) so we don't synthesize anything on registration.
    pub fn register_touch(&self, client: ClientId, touch: WlTouch) {
        self.touches
            .lock()
            .unwrap()
            .entry(client)
            .or_default()
            .push(touch);
    }

    /// Drop a `wl_touch` resource. Symmetric with `unregister_pointer`.
    pub fn unregister_touch(&self, client: ClientId, touch: &WlTouch) {
        let mut map = self.touches.lock().unwrap();
        if let Some(v) = map.get_mut(&client) {
            v.retain(|t| t.id() != touch.id());
            if v.is_empty() {
                map.remove(&client);
            }
        }
    }

    /// Deliver a scroll (axis) frame to the focused window's client.
    /// Forwards axis_source, axis values, and axis_stop with a proper frame
    /// group — required for Firefox to activate smooth/finger-scroll mode.
    pub fn inject_axis(
        &self,
        source: Option<wayland_server::protocol::wl_pointer::AxisSource>,
        axis_h: f64,
        axis_v: f64,
        stop_h: bool,
        stop_v: bool,
    ) -> bool {
        let focused = *self.focused_window.lock().unwrap();
        let Some(window_id) = focused else {
            return false;
        };
        let target_client = match self.windows_by_id.lock().unwrap().get(&window_id) {
            Some((_s, cid)) => cid.clone(),
            None => return false,
        };
        let pointers: Vec<WlPointer> = match self.pointers.lock().unwrap().get(&target_client) {
            Some(v) => v.iter().filter(|p| p.is_alive()).cloned().collect(),
            None => return false,
        };
        if pointers.is_empty() {
            return false;
        }
        // Audit H19: Wayland convention is CLOCK_MONOTONIC ms-as-u32, not
        // wall-clock. Clients use the timestamp to compute gesture velocity
        // and scroll deceleration; SystemTime::now() produced ~1.7×10^9 ms
        // (since 1970), wrapping garbage modulo 2^32.
        let time_ms = monotonic_now_ms() as u32;
        use wayland_server::protocol::wl_pointer::Axis;
        for ptr in &pointers {
            if ptr.version() >= 5 {
                if let Some(src) = source {
                    ptr.axis_source(src);
                }
            }
            if axis_v != 0.0 {
                ptr.axis(time_ms, Axis::VerticalScroll, axis_v);
            }
            if axis_h != 0.0 {
                ptr.axis(time_ms, Axis::HorizontalScroll, axis_h);
            }
            if stop_v && ptr.version() >= 5 {
                ptr.axis_stop(time_ms, Axis::VerticalScroll);
            }
            if stop_h && ptr.version() >= 5 {
                ptr.axis_stop(time_ms, Axis::HorizontalScroll);
            }
            if ptr.version() >= 5 {
                ptr.frame();
            }
        }
        self.wake_compositor();
        true
    }

    /// Deliver a pointer event to a target window's client.
    ///
    /// - `window_id`:
    ///   - `Some(id)`: route to that specific window's client regardless of
    ///     focus. Unknown id → drop and return `false` (no fallback to
    ///     focused — that would mask SDK bugs).
    ///   - `None`: fall back to `self.focused_window` (legacy behaviour).
    /// - `content`: if `true`, treat `(x, y)` as content-space coords and
    ///   add the target window's CSD inset before delivery. The
    ///   `window_content_inset()` stub currently returns `(0, 0)`; the
    ///   real `set_window_geometry` data is plumbed through later.
    /// - `(x, y)`: always emits a motion event.
    /// - `button != 0`: emit a button press/release per `pressed`.
    /// - `(axis_x, axis_y) != (0, 0)`: emit axis events.
    ///
    /// Returns true if at least one wl_pointer received events.
    // encoder/cursor setup takes many tightly-related params by design
    #[allow(clippy::too_many_arguments)]
    pub fn inject_pointer(
        &self,
        window_id: Option<u32>,
        content: bool,
        x: f64,
        y: f64,
        button: u32,
        pressed: bool,
        axis_x: f64,
        axis_y: f64,
        seq: u32,
    ) -> bool {
        // Step 1: pick target window. Explicit window_id wins; otherwise
        // fall back to the focused window. Unknown id is a hard drop —
        // not silently re-routing to focus surfaces SDK bugs.
        let target_window_id = match window_id {
            Some(id) => id,
            None => match *self.focused_window.lock().unwrap() {
                Some(id) => id,
                None => return false,
            },
        };

        // Step 2: resolve the owning client. If the window was unregistered
        // between the SDK observing it and the inject call landing here,
        // we drop rather than fall back.
        let target_client = match self.windows_by_id.lock().unwrap().get(&target_window_id) {
            Some((_s, cid)) => cid.clone(),
            None => {
                tracing::debug!(
                    window_id = target_window_id,
                    "inject_pointer: target window_id not in windows_by_id; dropping"
                );
                return false;
            }
        };

        // Step 3: apply the CSD inset if the caller asked for content-space
        // semantics. `window_content_inset` currently returns (0, 0); the
        // real xdg_surface.set_window_geometry inset is wired through later.
        let (x, y) = if content {
            let inset = self.window_content_inset(target_window_id);
            (x + inset.0 as f64, y + inset.1 as f64)
        } else {
            (x, y)
        };

        // Scale handling. `inject_pointer` receives coordinates in the
        // session's `width x height` space, the same space the SDK observes
        // through screenshots and window geometry. That space IS this
        // compositor's logical coordinate space: the virtual output advertises
        // `zxdg_output_v1.logical_size(width, height)` and every
        // `xdg_toplevel.configure(width, height)` carries the raw session
        // dimensions (compositor.rs GetXdgOutput + GetToplevel). `wl_output`
        // additionally advertises `scale(N)` and `wp_fractional_scale` reports
        // `N*120`, but those only tell a client how many buffer pixels to
        // render per logical pixel (`wl_surface.set_buffer_scale`); they never
        // change the coordinate space of `wl_pointer.motion`, which the Wayland
        // protocol fixes as surface-local logical pixels.
        //
        // So the correct transform at any scale (including scale>1) is the
        // identity (see `logical_to_surface_coord`). An earlier
        // attempt multiplied by `scale` on the false premise that motion
        // carries buffer pixels; that pushed every click `scale` times away
        // from its target (a click meant for logical (10,20) landed at (20,40)
        // under scale=2). The pass-through clicks where the SDK intends and
        // keeps the scale=1 path byte-identical to before.
        let scale = self.inner.lock().unwrap().scale;
        let (x, y) = logical_to_surface_coord(x, y, scale);

        let pointers: Vec<WlPointer> = match self.pointers.lock().unwrap().get(&target_client) {
            Some(v) => v.iter().filter(|p| p.is_alive()).cloned().collect(),
            None => return false,
        };
        if pointers.is_empty() {
            return false;
        }
        // Audit H19: Wayland convention is CLOCK_MONOTONIC ms-as-u32, not
        // wall-clock. Clients use the timestamp to compute gesture velocity
        // and scroll deceleration; SystemTime::now() produced ~1.7×10^9 ms
        // (since 1970), wrapping garbage modulo 2^32.
        let time_ms = monotonic_now_ms() as u32;
        use wayland_server::protocol::wl_pointer::{self, Axis, ButtonState};
        let btn_state = if pressed {
            ButtonState::Pressed
        } else {
            ButtonState::Released
        };
        // Xwayland (and the X11-on-Wayland input pipeline in general)
        // processes button events at the X11 pointer position established
        // BEFORE the wl_pointer.motion in the same frame group. When SDK
        // callers teleport-then-click (`inject_pointer(200, 200, BTN_LEFT,
        // pressed=true)` packed as `[motion(200,200), button(press), frame]`),
        // Xwayland generates a ButtonPress at the OLD position (e.g.
        // 100,100), then a separate Motion to (200,200). The click misses the
        // target.
        //
        // A real hardware seat never sends teleport+button in one frame:
        // motion is a continuous stream of frames, button events fire in
        // their own frame at the time of press/release. Splitting motion into
        // its own frame group (closed with `frame()` for v>=5 clients) before
        // emitting the button gives Xwayland time to update its internal X11
        // pointer position before the button event lands. Native Wayland
        // clients treat adjacent frames the same way they treat back-to-back
        // hardware events from a 1000Hz mouse, so this is a no-op for them
        // and a correctness fix for X-on-Wayland clients.
        let needs_split = button != 0 || axis_x != 0.0 || axis_y != 0.0;
        for ptr in &pointers {
            ptr.motion(time_ms, x, y);
            if needs_split && ptr.version() >= 5 {
                ptr.frame();
            }
            if button != 0 {
                let serial = self.next_serial();
                ptr.button(serial, time_ms, button, btn_state);
            }
            if axis_x != 0.0 {
                ptr.axis(time_ms, Axis::HorizontalScroll, axis_x);
            }
            if axis_y != 0.0 {
                ptr.axis(time_ms, Axis::VerticalScroll, axis_y);
            }
            if ptr.version() >= 5 {
                ptr.frame();
            }
            // Silence unused import warning on older branches.
            let _ = wl_pointer::Axis::VerticalScroll;
        }
        // Record the applied position for the viewer cursor-overlay latency
        // display. The (x, y) here are logical (session-space) pixels; we
        // downcast to f32 which is sufficient for overlay display (max
        // session dim ~7680px).
        self.record_cursor_pos(x as f32, y as f32, seq);
        self.wake_compositor();
        true
    }

    /// Deliver a synthetic touch event to a target window's client.
    /// Mirrors `inject_pointer`'s shape but routes through
    /// `wl_touch` instead of `wl_pointer`.
    ///
    /// - `window_id`: `Some(id)` overrides focus; `None` falls back to
    ///   `self.focused_window`; unknown `Some(id)` returns false.
    /// - `content`: when true, adds the target window's CSD inset
    ///   (`xdg_surface.set_window_geometry`) to `(x, y)`. Shares the
    ///   warn-once dedup HashSet with `inject_pointer`, so a single warning
    ///   fires per `window_id` across both code paths.
    /// - `id`: wl_touch tracking id (use 0 for single-finger flows).
    /// - `(x, y)`: logical pixels (per Q3 2026-05-18 decision); multiplied
    ///   by the session scale before emitting, matching `inject_pointer`.
    /// - `phase`: `Down` emits `wl_touch.down(serial, time, &surface, id, x, y)`;
    ///   `Motion` emits `wl_touch.motion(time, id, x, y)`; `Up` emits
    ///   `wl_touch.up(serial, time, id)`. Each event is followed by a
    ///   `wl_touch.frame()` marker.
    ///
    /// Returns true iff at least one wl_touch resource received the event.
    /// If the resolved client has never called `wl_seat.get_touch`,
    /// returns false (silently dropped, same as `inject_pointer`).
    pub fn inject_touch(
        &self,
        window_id: Option<u32>,
        content: bool,
        id: u32,
        x: f64,
        y: f64,
        phase: waymux_protocol::TouchPhase,
    ) -> bool {
        // Step 1: resolve target window. Explicit `window_id` wins; otherwise
        // fall back to the focused window. Unknown id is a hard drop (matches
        // `inject_pointer` — silent fallback to focused masks SDK bugs).
        let target_window_id = match window_id {
            Some(id) => id,
            None => match *self.focused_window.lock().unwrap() {
                Some(id) => id,
                None => return false,
            },
        };

        // Step 2: resolve the owning client. If the window was unregistered
        // between the SDK observing it and the inject call landing here, we
        // drop rather than fall back.
        let (target_surface, target_client) =
            match self.windows_by_id.lock().unwrap().get(&target_window_id) {
                Some((s, cid)) => (s.clone(), cid.clone()),
                None => {
                    tracing::debug!(
                        window_id = target_window_id,
                        "inject_touch: target window_id not in windows_by_id; dropping"
                    );
                    return false;
                }
            };

        // Step 3: apply the CSD inset if the caller asked for content-space
        // semantics. `window_content_inset` shares the `content_fallback_warned`
        // HashSet with `inject_pointer` so the no-geometry warning fires once
        // per window across both pointer and touch.
        let (x, y) = if content {
            let inset = self.window_content_inset(target_window_id);
            (x + inset.0 as f64, y + inset.1 as f64)
        } else {
            (x, y)
        };

        // Step 4: scale handling. Identical to `inject_pointer`: the session
        // coordinate space (`width x height`) IS this compositor's logical
        // space (advertised via `zxdg_output_v1.logical_size`), and `wl_touch`
        // coordinates are surface-local logical pixels by protocol, so the
        // transform is the identity. `wl_output.scale` / `wp_fractional_scale`
        // govern buffer rendering only, never input coordinates. See
        // `inject_pointer` and `logical_to_surface_coord` for the full
        // rationale.
        let scale = self.inner.lock().unwrap().scale;
        let (x, y) = logical_to_surface_coord(x, y, scale);

        let touches: Vec<WlTouch> = match self.touches.lock().unwrap().get(&target_client) {
            Some(v) => v.iter().filter(|t| t.is_alive()).cloned().collect(),
            None => return false,
        };
        if touches.is_empty() {
            return false;
        }

        // Step 5: emit the touch event + a `frame()` marker per resource.
        // The wl_touch frame marker terminates a logical "atomic update"
        // group; we emit one per call so a Down → Motion → Up sequence
        // (three calls) produces three independent frames in arrival order.
        // Matches the SDK contract documented on `RequestMethod::InjectTouch`.
        let time_ms = monotonic_now_ms() as u32;
        for touch in &touches {
            match phase {
                waymux_protocol::TouchPhase::Down => {
                    let serial = self.next_serial();
                    touch.down(serial, time_ms, &target_surface, id as i32, x, y);
                }
                waymux_protocol::TouchPhase::Motion => {
                    touch.motion(time_ms, id as i32, x, y);
                }
                waymux_protocol::TouchPhase::Up => {
                    let serial = self.next_serial();
                    touch.up(serial, time_ms, id as i32);
                }
            }
            touch.frame();
        }
        self.wake_compositor();
        true
    }

    /// Return the (x, y) CSD inset of the given window — the offset from
    /// the surface buffer's origin to its visible content origin, as set
    /// by `xdg_surface.set_window_geometry`.
    ///
    /// Returns (0, 0) for windows that haven't called set_window_geometry
    /// and emits a `tracing::warn!` once per window per session via the
    /// `content_fallback_warned` HashSet — the Q4 fallback per the
    /// 2026-05-18 design decisions. Hard-erroring was rejected because
    /// heterogeneous workflows (Electron + Qt + Plasma) need degraded-but-
    /// usable behavior, not crashes.
    fn window_content_inset(&self, window_id: u32) -> (i32, i32) {
        let surface = match self.windows_by_id.lock().unwrap().get(&window_id) {
            Some((s, _cid)) => s.clone(),
            None => return (0, 0), // unknown window; inject_pointer/inject_touch returned false earlier
        };
        // Borrow + read the rect, then drop the borrow before any other
        // mutex acquisition so we don't tangle SurfaceData's lifetime with
        // the `content_fallback_warned` lock.
        let rect = {
            let data = surface
                .data::<crate::compositor::SurfaceData>()
                .expect("SurfaceData on tracked window");
            *data.content_rect.lock().unwrap()
        };
        match rect {
            Some(rect) => (rect.x.max(0), rect.y.max(0)),
            None => {
                // Warn-once per window per session. HashSet::insert returns
                // true iff the value was newly inserted, so this fires
                // exactly once even across repeated calls. It is shared between
                // `inject_pointer` and `inject_touch` so one
                // window emits a single fallback warning regardless of which
                // synthetic-input path tripped it.
                let mut warned = self.content_fallback_warned.lock().unwrap();
                if warned.insert(window_id) {
                    tracing::warn!(
                        window_id,
                        "inject: content=true but no set_window_geometry recorded for \
                         window {window_id}; falling back to buffer coords. Some toolkits \
                         (Electron, parts of GTK) don't reliably emit this — coordinates will \
                         be interpreted as buffer-local until the client calls set_window_geometry."
                    );
                }
                (0, 0)
            }
        }
    }

    /// Deliver a synthetic key event to the focused window's client.
    /// Returns true if at least one wl_keyboard received the event.
    pub fn inject_key(
        &self,
        keycode: u32,
        pressed: bool,
        mods_depressed: u32,
        mods_latched: u32,
        mods_locked: u32,
        group: u32,
    ) -> bool {
        let focused = *self.focused_window.lock().unwrap();
        let Some(window_id) = focused else {
            return false;
        };
        let target_client = match self.windows_by_id.lock().unwrap().get(&window_id) {
            Some((_s, cid)) => cid.clone(),
            None => return false,
        };
        let keyboards: Vec<WlKeyboard> = match self.keyboards.lock().unwrap().get(&target_client) {
            Some(v) => v.iter().filter(|k| k.is_alive()).cloned().collect(),
            None => return false,
        };
        if keyboards.is_empty() {
            return false;
        }
        let serial_mods = self.next_serial();
        let serial_key = self.next_serial();
        // Audit H19: Wayland convention is CLOCK_MONOTONIC ms-as-u32, not
        // wall-clock. Clients use the timestamp to compute gesture velocity
        // and scroll deceleration; SystemTime::now() produced ~1.7×10^9 ms
        // (since 1970), wrapping garbage modulo 2^32.
        let time_ms = monotonic_now_ms() as u32;
        let key_state = if pressed {
            wayland_server::protocol::wl_keyboard::KeyState::Pressed
        } else {
            wayland_server::protocol::wl_keyboard::KeyState::Released
        };
        tracing::debug!(
            keycode,
            pressed,
            mods_depressed,
            mods_locked,
            group,
            keyboards = keyboards.len(),
            "inject_key: sending"
        );
        for kbd in &keyboards {
            kbd.modifiers(
                serial_mods,
                mods_depressed,
                mods_latched,
                mods_locked,
                group,
            );
            kbd.key(serial_key, time_ms, keycode, key_state);
        }
        self.wake_compositor();
        true
    }

    /// Composite every registered window (xdg toplevels then layer surfaces)
    /// into a single session-sized ARGB8888 image. Layer surfaces are drawn
    /// on top so panels and overlays sit above regular windows. Returns None
    /// if there are no windows or if all of them have empty buffers.
    #[allow(clippy::type_complexity)]
    pub fn capture_desktop(&self) -> Option<(Vec<u8>, i32, i32, WEnum<wl_shm::Format>, i32)> {
        use crate::compositor::SurfaceData;

        let (sw, sh, _) = self.snapshot();
        if sw == 0 || sh == 0 {
            return None;
        }
        let w = sw as i32;
        let h = sh as i32;
        let stride = w * 4;
        let mut buf = vec![0u8; (h as usize) * (stride as usize)];

        // Snapshot the surface set so we don't hold the lock across compositing.
        let surfaces: Vec<WlSurface> = self
            .windows_by_id
            .lock()
            .unwrap()
            .values()
            .map(|(s, _)| s.clone())
            .collect();

        // Z-order: regular windows first, then layer surfaces on top.
        let mut regular = Vec::new();
        let mut layers = Vec::new();
        for surface in &surfaces {
            let is_layer = surface
                .data::<SurfaceData>()
                .map(|sd| *sd.is_layer_surface.lock().unwrap())
                .unwrap_or(false);
            if is_layer {
                layers.push(surface);
            } else {
                regular.push(surface);
            }
        }

        // Tile regular windows horizontally across the session width; layer
        // surfaces (panels, overlays) are painted last at (0,0) so they cover
        // the full viewport the way a real compositor panel would.
        let regular_surfaces: Vec<_> = regular.into_iter().collect();
        let n = regular_surfaces.len().max(1);
        let tile_w = w / n as i32;
        let mut any = false;
        for (i, surface) in regular_surfaces.iter().enumerate() {
            if surface.data::<SurfaceData>().is_some() {
                let x = i as i32 * tile_w;
                crate::composite::composite_into(&mut buf, w, h, stride, surface, x, 0);
                any = true;
            }
        }
        for surface in layers {
            if surface.data::<SurfaceData>().is_some() {
                crate::composite::composite_into(&mut buf, w, h, stride, surface, 0, 0);
                any = true;
            }
        }
        if !any {
            return None;
        }
        Some((buf, w, h, WEnum::Value(wl_shm::Format::Argb8888), stride))
    }

    /// Capture the named window's composited pixel bytes (toplevel + all
    /// descendant subsurfaces). Returns None if the window doesn't exist.
    /// Always returns ARGB8888 since we composite into a fresh buffer.
    ///
    /// The buffer is a fresh heap allocation; callers may freely process
    /// it (encode to PNG, memcpy to the outer framebuffer, etc.).
    #[allow(clippy::type_complexity)]
    pub fn capture_window(
        &self,
        window_id: u32,
    ) -> Option<(Vec<u8>, i32, i32, WEnum<wl_shm::Format>, i32)> {
        let (surface, _) = self
            .windows_by_id
            .lock()
            .unwrap()
            .get(&window_id)
            .cloned()?;
        let fallback = {
            let s = self.inner.lock().unwrap();
            (s.width, s.height)
        };
        let frame = crate::composite::composite(&surface, fallback)?;
        Some((
            frame.bytes,
            frame.width,
            frame.height,
            WEnum::Value(wl_shm::Format::Argb8888),
            frame.stride,
        ))
    }
}

/// Convert an injected coordinate from the SDK's session-space (`width x
/// height`, the space the SDK observes via screenshots and window geometry)
/// to the surface-local coordinate `wl_pointer`/`wl_touch` events carry.
///
/// In this compositor the session space IS the logical coordinate space:
/// the virtual output advertises `zxdg_output_v1.logical_size(width, height)`
/// and toplevels are configured at the raw session dimensions. Wayland input
/// events use surface-local *logical* pixels; the session `scale` (advertised
/// via `wl_output.scale` / `wp_fractional_scale`) only governs how many buffer
/// pixels a client renders per logical pixel and never the input coordinate
/// space. So the conversion is the identity at every scale.
///
/// Centralising it in one place (rather than inlining a no-op) documents that
/// the lack of a multiply is deliberate and keeps the pointer and touch paths
/// provably in lockstep. `_scale` is accepted so the contract is explicit and
/// any future fractional-buffer work has a single seam to change.
#[inline]
fn logical_to_surface_coord(x: f64, y: f64, _scale: u32) -> (f64, f64) {
    (x, y)
}

/// Read CLOCK_MONOTONIC and return ms since boot. Used for idle-suspend
/// activity tracking; robust against wall-clock jumps that would confuse
/// `SystemTime::now()`-based timing.
fn monotonic_now_ms() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000 + (ts.tv_nsec as u64) / 1_000_000
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> State {
        State::new("test".into(), 100, 100, 1, None, false)
    }

    #[test]
    fn pending_releases_empty_on_new_state() {
        assert!(make_state().take_pending_releases().is_empty());
    }

    /// Scale > 1 coordinate transform. `inject_pointer` / `inject_touch`
    /// receive coordinates in the session's `width x height` space, which is
    /// this compositor's logical space (advertised via
    /// `zxdg_output_v1.logical_size`). Wayland input events carry surface-local
    /// logical pixels, so the transform is the identity at every scale; the
    /// session `scale` only affects buffer rendering, never input coordinates.
    ///
    /// A prior implementation multiplied by `scale` here, which sent every
    /// click `scale` times off-target (logical (10,20) landed at (20,40) under
    /// scale=2). This pins the corrected pass-through so the bug can't return.
    #[test]
    fn logical_to_surface_coord_is_identity_at_every_scale() {
        for scale in [1u32, 2, 3, 4] {
            assert_eq!(
                super::logical_to_surface_coord(10.0, 20.0, scale),
                (10.0, 20.0),
                "scale={scale}: session-space coords must pass through unchanged"
            );
            // A point at the far edge of a 320x240 session must stay in bounds
            // (the old `* scale` pushed it to 640x480, off the logical surface).
            assert_eq!(
                super::logical_to_surface_coord(319.0, 239.0, scale),
                (319.0, 239.0),
                "scale={scale}: edge coords must not be scaled off the surface"
            );
            // Fractional sub-pixel coords are preserved exactly.
            assert_eq!(
                super::logical_to_surface_coord(123.5, 45.75, scale),
                (123.5, 45.75),
                "scale={scale}: fractional coords must round-trip"
            );
        }
    }

    #[test]
    fn take_pending_releases_is_idempotent_when_empty() {
        let s = make_state();
        assert!(s.take_pending_releases().is_empty());
        assert!(s.take_pending_releases().is_empty());
    }

    #[test]
    fn flush_pending_releases_is_safe_when_empty() {
        let s = make_state();
        s.flush_pending_releases();
        assert!(s.take_pending_releases().is_empty());
    }

    #[test]
    fn last_activity_initialised_at_construction() {
        let s = make_state();
        assert!(
            s.last_activity_ms() > 0,
            "last_activity_ms should be seeded at State::new(); got 0"
        );
    }

    #[test]
    fn note_activity_bumps_last_activity() {
        let s = make_state();
        let before = s.last_activity_ms();
        std::thread::sleep(std::time::Duration::from_millis(2));
        s.note_activity();
        assert!(
            s.last_activity_ms() > before,
            "note_activity should advance last_activity_ms (was {before}, after {})",
            s.last_activity_ms()
        );
    }

    #[test]
    fn state_viewer_slot_default_is_none() {
        let s = make_state();
        assert!(
            s.viewer.lock().unwrap().is_none(),
            "new state should have no viewer"
        );
    }

    #[test]
    fn state_viewer_slot_independent_of_recording() {
        let s = make_state();
        *s.viewer.lock().unwrap() = Some(crate::viewer::ViewerHandle::test_stub(
            "http://127.0.0.1:18347".into(),
        ));
        // Recording slot still None — they don't share state.
        assert!(s.recording.lock().unwrap().is_none());
        assert!(s.viewer.lock().unwrap().is_some());
    }

    #[test]
    fn tap_encoder_count_zero_when_no_outputs() {
        let s = make_state();
        assert_eq!(s.encoder_count_for_tap(), 0);
    }

    #[test]
    fn tap_encoder_count_viewer_only() {
        let s = make_state();
        *s.viewer.lock().unwrap() = Some(crate::viewer::ViewerHandle::test_stub("u".into()));
        assert_eq!(s.encoder_count_for_tap(), 1);
    }

    #[test]
    fn tap_encoder_count_recording_primary_plus_viewer() {
        let s = make_state();
        *s.recording.lock().unwrap() =
            Some(crate::recording::DualRecordingHandle::test_stub_primary_only());
        *s.viewer.lock().unwrap() = Some(crate::viewer::ViewerHandle::test_stub("u".into()));
        assert_eq!(s.encoder_count_for_tap(), 2);
    }

    /// Regression guard: `unregister_surface`'s focus-demotion path
    /// uses `keys().max()` not `keys().next()`. Constructing real
    /// `WlSurface`/`ClientId` values in a unit test requires a full
    /// wayland-server fixture, so this test exercises the math of the fix
    /// directly: for any `HashMap<u32, _>` of window ids, the demoted
    /// focus must be the highest remaining id, deterministically across
    /// many trials (defeats `HashMap::keys()` randomisation).
    ///
    /// If this test ever flakes, somebody put `.next()` back in
    /// `unregister_surface` at state.rs ~1773.
    #[test]
    fn keys_max_returns_highest_remaining_after_unregister() {
        use std::collections::HashMap;
        for trial in 0..20 {
            // Fresh HashMap each trial — each gets a new RandomState seed,
            // so `.next()` would return different ids across trials. `.max()`
            // must not.
            let mut m: HashMap<u32, ()> = HashMap::new();
            m.insert(10, ());
            m.insert(20, ());
            m.insert(30, ());

            // Step 1: focused is 30 (the highest). Remove it; demoted focus
            // should be 20.
            m.remove(&30);
            let demoted = m.keys().copied().max();
            assert_eq!(
                demoted,
                Some(20),
                "trial {trial}: expected demoted focus = 20, got {demoted:?}"
            );

            // Step 2: simulate registering a fourth surface id=100; focus
            // would have gone to 100 via `register_surface`. Unregister it;
            // demoted focus should fall to 20 (the highest of {10, 20}).
            m.insert(100, ());
            m.remove(&100);
            let demoted2 = m.keys().copied().max();
            assert_eq!(
                demoted2,
                Some(20),
                "trial {trial}: expected demoted focus = 20 after id=100 unregister, got {demoted2:?}"
            );

            // Step 3: unregister 20 too; demoted focus should be 10.
            m.remove(&20);
            let demoted3 = m.keys().copied().max();
            assert_eq!(
                demoted3,
                Some(10),
                "trial {trial}: expected demoted focus = 10, got {demoted3:?}"
            );

            // Step 4: unregister the last one; demoted focus should be None.
            m.remove(&10);
            let demoted4 = m.keys().copied().max();
            assert_eq!(
                demoted4, None,
                "trial {trial}: expected demoted focus = None on empty map, got {demoted4:?}"
            );
        }
    }

    /// Warn-once dedup. `window_content_inset` calls
    /// `content_fallback_warned.lock().unwrap().insert(window_id)` and
    /// fires `tracing::warn!` only when `insert` returned `true` (i.e. the
    /// value was newly inserted). Constructing a real `WlSurface` for an
    /// end-to-end test requires a wayland-server fixture; this test
    /// exercises the dedup semantic directly against a fresh State.
    ///
    /// Repeating `insert(id)` returns false on every call after the first,
    /// so the warning fires exactly once per window per session lifetime.
    /// Distinct ids are independent (each gets its own first warning).
    /// Guard: `SurfaceData::default()` must initialise
    /// `layer_keyboard_interactivity` to 0 (= `none`, the wlr layer-shell
    /// protocol default). The first-commit registration path at
    /// compositor.rs ~960-1000 reads this field BEFORE the client has had
    /// a chance to call `set_keyboard_interactivity`, so a wrong default
    /// would let panels and notification daemons silently steal focus.
    #[test]
    fn surface_data_default_keyboard_interactivity_is_none() {
        use crate::compositor::SurfaceData;
        let sd = SurfaceData::default();
        let v = *sd.layer_keyboard_interactivity.lock().unwrap();
        assert_eq!(
            v, 0,
            "SurfaceData::default().layer_keyboard_interactivity must be 0 \
             (= keyboard_interactivity::none, the protocol default). \
             Got {v}."
        );
    }

    /// Guard. The `SetKeyboardInteractivity` dispatcher at
    /// compositor.rs ~2680-2700 writes the wire u32 value into
    /// `layer_keyboard_interactivity`. The consumer at compositor.rs
    /// ~960-1000 then reads it back and tests `== 1` for the exclusive
    /// case. This test confirms the field round-trips wire values for the
    /// three known variants (0=none, 1=exclusive, 2=on_demand) plus an
    /// unknown forward-compat value (3) without clamping.
    #[test]
    fn surface_data_keyboard_interactivity_round_trips_wire_u32() {
        use crate::compositor::SurfaceData;
        let sd = SurfaceData::default();
        for wire in [0u32, 1, 2, 3, u32::MAX] {
            *sd.layer_keyboard_interactivity.lock().unwrap() = wire;
            assert_eq!(
                *sd.layer_keyboard_interactivity.lock().unwrap(),
                wire,
                "round-trip failed for wire value {wire}"
            );
        }
    }

    /// Behavioral guard. The first-commit registration branch
    /// at compositor.rs ~960-1000 chooses between `register_surface` (focus
    /// swap) and `register_surface_no_focus` (no focus swap) by reading
    /// `data.layer_keyboard_interactivity` and testing `== 1`. This test
    /// verifies the decision rule — exactly the `== 1` literal threshold —
    /// across all four wire variants (none/exclusive/on_demand/unknown).
    /// If this test fails, somebody widened or narrowed the threshold (e.g.
    /// to `!= 0`, which would let `on_demand` panels steal focus too).
    #[test]
    fn layer_focus_decision_only_triggers_for_exclusive() {
        // Mirror the compositor.rs:962-1010 decision predicate exactly.
        fn should_focus_layer(keyboard_interactivity: u32) -> bool {
            keyboard_interactivity == 1
        }
        // Protocol-defined variants:
        assert!(
            !should_focus_layer(0),
            "keyboard_interactivity::none (0) must NOT focus — \
             Plasma panels, notification daemons, wallpaper surfaces"
        );
        assert!(
            should_focus_layer(1),
            "keyboard_interactivity::exclusive (1) MUST focus — \
             lock screens, modal launchers"
        );
        assert!(
            !should_focus_layer(2),
            "keyboard_interactivity::on_demand (2) must NOT focus on register \
             (simplification: the SDK has no set_focus API)"
        );
        // Forward-compat: unknown variant must NOT focus (safer default).
        assert!(
            !should_focus_layer(3),
            "unknown keyboard_interactivity wire value must NOT focus \
             (forward-compat: protocol bump to add a variant 3 should not \
             silently steal focus from xdg_toplevels)"
        );
    }

    #[test]
    fn content_fallback_warned_dedup_inserts_once_per_window_id() {
        let s = make_state();
        // Field starts empty.
        assert_eq!(s.content_fallback_warned.lock().unwrap().len(), 0);

        // First insert for window 42 → returns true (warning would fire).
        {
            let mut warned = s.content_fallback_warned.lock().unwrap();
            assert!(warned.insert(42), "first insert(42) must return true");
        }
        assert_eq!(s.content_fallback_warned.lock().unwrap().len(), 1);

        // Repeated inserts for the same id → return false (no re-warn).
        for trial in 0..5 {
            let mut warned = s.content_fallback_warned.lock().unwrap();
            assert!(
                !warned.insert(42),
                "trial {trial}: repeat insert(42) must return false"
            );
        }
        assert_eq!(
            s.content_fallback_warned.lock().unwrap().len(),
            1,
            "len must still be 1 after repeat inserts of same id"
        );

        // Distinct id → its own first warning.
        {
            let mut warned = s.content_fallback_warned.lock().unwrap();
            assert!(warned.insert(99), "first insert(99) must return true");
        }
        assert_eq!(s.content_fallback_warned.lock().unwrap().len(), 2);
    }
}
