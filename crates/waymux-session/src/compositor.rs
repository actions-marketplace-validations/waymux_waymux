// SPDX-License-Identifier: Apache-2.0

//! Minimum-viable Wayland compositor.
//!
//! Advertises the globals a simple xdg-shell client needs, tracks
//! `xdg_toplevel` lifecycle into `State`, captures shm pool + buffer
//! metadata so we can serve screenshot requests, and records every
//! `wl_surface.commit` as a damage timestamp for `wait_for_idle`.
//!
//! No rendering. No input forwarding. No dmabuf. All those land in
//! later spec weeks; this compositor is a headless observer.

use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

// ─── new protocol imports ────────────────────────────────────────────────
use wayland_protocols::ext::idle_notify::v1::server::{
    ext_idle_notification_v1::{self, ExtIdleNotificationV1},
    ext_idle_notifier_v1::{self, ExtIdleNotifierV1},
};
use wayland_protocols::wp::alpha_modifier::v1::server::{
    wp_alpha_modifier_surface_v1::{self, WpAlphaModifierSurfaceV1},
    wp_alpha_modifier_v1::{self, WpAlphaModifierV1},
};
use wayland_protocols::wp::linux_drm_syncobj::v1::server::{
    wp_linux_drm_syncobj_manager_v1::{self, WpLinuxDrmSyncobjManagerV1},
    wp_linux_drm_syncobj_surface_v1::{self, WpLinuxDrmSyncobjSurfaceV1},
    wp_linux_drm_syncobj_timeline_v1::{self, WpLinuxDrmSyncobjTimelineV1},
};
use wayland_protocols::xdg::activation::v1::server::{
    xdg_activation_token_v1::{self, XdgActivationTokenV1},
    xdg_activation_v1::{self, XdgActivationV1},
};
use wayland_protocols::xdg::foreign::zv2::server::{
    zxdg_exported_v2::{self, ZxdgExportedV2},
    zxdg_exporter_v2::{self, ZxdgExporterV2},
    zxdg_imported_v2::{self, ZxdgImportedV2},
    zxdg_importer_v2::{self, ZxdgImporterV2},
};
use wayland_protocols::xdg::xdg_output::zv1::server::{
    zxdg_output_manager_v1::{self, ZxdgOutputManagerV1},
    zxdg_output_v1::{self, ZxdgOutputV1},
};
use wayland_protocols_plasma::plasma_window_management::server::{
    org_kde_plasma_activation, org_kde_plasma_activation_feedback,
    org_kde_plasma_stacking_order::{self, OrgKdePlasmaStackingOrder},
    org_kde_plasma_window::{self, OrgKdePlasmaWindow},
    org_kde_plasma_window_management::{self, OrgKdePlasmaWindowManagement},
};
use wayland_protocols_wlr::layer_shell::v1::server::{
    zwlr_layer_shell_v1::{self, ZwlrLayerShellV1},
    zwlr_layer_surface_v1::{self, ZwlrLayerSurfaceV1},
};
use wayland_protocols_wlr::screencopy::v1::server::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::{self, ZwlrScreencopyManagerV1},
};

use wayland_protocols::wp::fractional_scale::v1::server::{
    wp_fractional_scale_manager_v1::{self, WpFractionalScaleManagerV1},
    wp_fractional_scale_v1::{self, WpFractionalScaleV1},
};
use wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::server::{
    zwp_keyboard_shortcuts_inhibit_manager_v1::{self, ZwpKeyboardShortcutsInhibitManagerV1},
    zwp_keyboard_shortcuts_inhibitor_v1::{self, ZwpKeyboardShortcutsInhibitorV1},
};
use wayland_protocols::wp::linux_dmabuf::zv1::server::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1,
    zwp_linux_dmabuf_v1::{self, ZwpLinuxDmabufV1},
};
use wayland_protocols::wp::pointer_constraints::zv1::server::{
    zwp_confined_pointer_v1::{self, ZwpConfinedPointerV1},
    zwp_locked_pointer_v1::{self, ZwpLockedPointerV1},
    zwp_pointer_constraints_v1::{self, ZwpPointerConstraintsV1},
};
use wayland_protocols::wp::presentation_time::server::wp_presentation::{self, WpPresentation};
use wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::WpPresentationFeedback;
use wayland_protocols::wp::primary_selection::zv1::server::{
    zwp_primary_selection_device_manager_v1::{self, ZwpPrimarySelectionDeviceManagerV1},
    zwp_primary_selection_device_v1::{self, ZwpPrimarySelectionDeviceV1},
    zwp_primary_selection_offer_v1::{self, ZwpPrimarySelectionOfferV1},
    zwp_primary_selection_source_v1::{self, ZwpPrimarySelectionSourceV1},
};
use wayland_protocols::wp::relative_pointer::zv1::server::{
    zwp_relative_pointer_manager_v1::{self, ZwpRelativePointerManagerV1},
    zwp_relative_pointer_v1::{self, ZwpRelativePointerV1},
};
use wayland_protocols::wp::single_pixel_buffer::v1::server::wp_single_pixel_buffer_manager_v1::{
    self, WpSinglePixelBufferManagerV1,
};
use wayland_protocols::wp::text_input::zv3::server::{
    zwp_text_input_manager_v3::{self, ZwpTextInputManagerV3},
    zwp_text_input_v3::{self, ZwpTextInputV3},
};
use wayland_protocols::wp::viewporter::server::{
    wp_viewport::{self, WpViewport},
    wp_viewporter::{self, WpViewporter},
};
use wayland_protocols::xdg::decoration::zv1::server::{
    zxdg_decoration_manager_v1::{self, ZxdgDecorationManagerV1},
    zxdg_toplevel_decoration_v1::{self, Mode, ZxdgToplevelDecorationV1},
};
use wayland_protocols::xdg::shell::server::{
    xdg_popup::{self, XdgPopup},
    xdg_positioner::XdgPositioner,
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};
use wayland_server::{
    backend::{ClientData, ClientId, DisconnectReason},
    protocol::{
        wl_buffer::{self, WlBuffer},
        wl_callback::{self, WlCallback},
        wl_compositor::{self, WlCompositor},
        wl_data_device::{self, WlDataDevice},
        wl_data_device_manager::{self, WlDataDeviceManager},
        wl_data_offer::{self, WlDataOffer},
        wl_data_source::{self, WlDataSource},
        wl_output::{self, WlOutput},
        wl_region::{self, WlRegion},
        wl_seat::{self, WlSeat},
        wl_shm::{self, WlShm},
        wl_shm_pool::{self, WlShmPool},
        wl_subcompositor::{self, WlSubcompositor},
        wl_subsurface::{self, WlSubsurface},
        wl_surface::{self, WlSurface},
    },
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, WEnum,
};

use crate::buffer::BufferKind;
use crate::dmabuf::{
    DmabufBufferData, ParamsData, Plane, DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888,
};
use crate::shm::{ShmBufferData, ShmPoolData};
use crate::state::{ClipboardContent, State};
use crate::syncobj::{SurfaceSync, SyncobjDevice, Timeline, TimelinePoint};

/// Compositor dispatch state.
pub struct Compositor {
    pub state: Arc<State>,
    /// DRM device handle for explicit-sync (wp_linux_drm_syncobj_v1).
    /// `None` if /dev/dri couldn't be opened, in which case the protocol
    /// is not advertised and clients fall back to implicit sync.
    pub syncobj_device: Option<Arc<SyncobjDevice>>,
}

/// Viewport crop/scale data stored per surface.
#[derive(Clone, Default)]
pub struct ViewportData {
    /// Source crop rectangle in buffer coordinates (x, y, w, h).
    pub src: Option<(f64, f64, f64, f64)>,
    /// Destination size (width, height) in surface coordinates.
    pub dst: Option<(i32, i32)>,
}

/// Per-surface user data.
pub struct SurfaceData {
    pub xdg_toplevel_id: Mutex<Option<u32>>,
    /// Buffer pending until the next commit.
    pub pending_buffer: Mutex<Option<WlBuffer>>,
    /// Most-recently-committed buffer.
    pub current_buffer: Mutex<Option<WlBuffer>>,
    /// Subsurfaces attached to this surface, in stack order (bottom-to-top
    /// = insertion order; place_above/place_below are ignored for now, which
    /// is fine for apps that don't reorder siblings — firefox doesn't).
    pub children: Mutex<Vec<SubsurfaceChild>>,
    /// Viewport crop/scale state set via wp_viewporter.
    pub viewport: Mutex<ViewportData>,
    /// Whether keyboard shortcuts are inhibited for this surface.
    pub shortcuts_inhibited: Mutex<bool>,
    /// Pending presentation feedback objects awaiting a frame.
    pub pending_feedbacks: Mutex<Vec<WpPresentationFeedback>>,
    /// True if this surface was assigned a zwlr_layer_surface_v1 role.
    pub is_layer_surface: Mutex<bool>,
    /// The window_id registered in State for this layer surface, if any.
    pub layer_window_id: Mutex<Option<u32>>,
    /// wl_surface.frame callbacks parked until this surface commits a new
    /// buffer. Draining into the global ready queue only on commit ensures
    /// the callback fires *after* the buffer that paces it, not before.
    pub surface_frame_cbs: Mutex<Vec<WlCallback>>,
    /// wp_linux_drm_syncobj_v1 per-surface state. Set when the client
    /// calls wp_linux_drm_syncobj_manager_v1.get_surface on this surface.
    /// At commit, we drain pending acquire/release points from here.
    pub surface_sync: Mutex<Option<Arc<SurfaceSync>>>,
    /// The window's content rectangle as set by
    /// `xdg_surface.set_window_geometry`. `None` means the client never
    /// emitted the request (some toolkits don't — Electron is inconsistent,
    /// parts of GTK skip it). When `inject_pointer` is called with
    /// `content=true` and this is `None`, State falls back to buffer coords
    /// and emits a warn-once log. See `state.rs::window_content_inset`.
    pub content_rect: Mutex<Option<waymux_protocol::Rect>>,
    /// The most recently received `set_keyboard_interactivity` value from
    /// this layer surface (0=none, 1=exclusive, 2=on_demand per wlr
    /// layer-shell protocol). 0 (none) is the protocol default for newly
    /// created layer surfaces. Only meaningful when is_layer_surface is true.
    ///
    /// Read at first-commit registration (compositor.rs ~960-1000) to decide
    /// whether the layer surface should steal focus from xdg_toplevels.
    /// Only `exclusive` (=1) triggers a focus swap; `none` and `on_demand`
    /// register without focus per Q5 of the 2026-05-18 click design
    /// decisions. See `state.rs::register_surface_no_focus`.
    pub layer_keyboard_interactivity: Mutex<u32>,
}

/// Entry in a parent surface's children list. Position is relative to the
/// parent; we recursively compose the tree at capture time.
#[derive(Clone)]
pub struct SubsurfaceChild {
    pub surface: WlSurface,
    pub x: i32,
    pub y: i32,
}

/// Per-`wl_subsurface` user data — we need the child's own `WlSurface` so
/// `destroy` / `set_position` can find the entry in the parent's list.
pub struct SubsurfaceData {
    pub surface: WlSurface,
    pub parent: WlSurface,
}

impl Default for SurfaceData {
    fn default() -> Self {
        Self {
            xdg_toplevel_id: Mutex::new(None),
            pending_buffer: Mutex::new(None),
            current_buffer: Mutex::new(None),
            children: Mutex::new(Vec::new()),
            viewport: Mutex::new(ViewportData::default()),
            shortcuts_inhibited: Mutex::new(false),
            pending_feedbacks: Mutex::new(Vec::new()),
            is_layer_surface: Mutex::new(false),
            layer_window_id: Mutex::new(None),
            surface_frame_cbs: Mutex::new(Vec::new()),
            surface_sync: Mutex::new(None),
            content_rect: Mutex::new(None),
            layer_keyboard_interactivity: Mutex::new(0),
        }
    }
}

/// Per-xdg_surface user data linking back to its wl_surface.
pub struct XdgSurfaceData {
    pub surface: WlSurface,
}

/// Per-xdg_toplevel user data carrying its session-assigned id.
pub struct ToplevelData {
    pub window_id: u32,
    #[allow(dead_code)]
    pub pid: i32,
}

struct DefaultClientData;
impl ClientData for DefaultClientData {
    fn initialized(&self, id: ClientId) {
        debug!(?id, "wayland client connected");
    }
    fn disconnected(&self, id: ClientId, reason: DisconnectReason) {
        debug!(?id, ?reason, "wayland client disconnected");
    }
}

pub fn run(socket_path: &Path, state: Arc<State>) -> Result<()> {
    let listener = std::os::unix::net::UnixListener::bind(socket_path)
        .with_context(|| format!("bind Wayland socket {}", socket_path.display()))?;
    listener
        .set_nonblocking(true)
        .context("set Wayland listener nonblocking")?;
    info!(path = %socket_path.display(), "Wayland compositor listening");

    let mut display: wayland_server::Display<Compositor> =
        wayland_server::Display::new().context("create wayland Display")?;
    let mut dh = display.handle();

    let _ = dh.create_global::<Compositor, WlCompositor, ()>(6, ());
    let _ = dh.create_global::<Compositor, WlSubcompositor, ()>(1, ());
    let _ = dh.create_global::<Compositor, WlShm, ()>(1, ());
    let _ = dh.create_global::<Compositor, WlSeat, ()>(7, ());
    let _ = dh.create_global::<Compositor, WlOutput, ()>(4, ());
    let _ = dh.create_global::<Compositor, XdgWmBase, ()>(5, ());
    // GTK4 / GDK Wayland backend hard-requires wl_data_device_manager to
    // be advertised — missing it yields "The Wayland compositor does not
    // provide one or more of the required interfaces". Per the protocol spec the
    // inner clipboard is isolated by default; this no-op implementation
    // enforces that isolation (requests are accepted and discarded).
    let _ = dh.create_global::<Compositor, WlDataDeviceManager, ()>(3, ());
    // Advertise dmabuf at version 5. v4 added the feedback mechanism
    // (get_default_feedback / get_surface_feedback). v5 added tranche
    // flags that KWin checks before enabling wp_linux_drm_syncobj_v1
    // explicit-sync — without v5 KWin silently falls back to implicit
    // sync. We send a minimal feedback table advertising ARGB8888 +
    // XRGB8888 with LINEAR modifier.
    let _ = dh.create_global::<Compositor, ZwpLinuxDmabufV1, ()>(5, ());
    // Legacy `wl_drm` advertisement. NVIDIA's libnvidia-egl-wayland v1.1.9
    // (default on Ubuntu 22.04 LTS) doesn't use the modern
    // `zwp_linux_dmabuf_v1` feedback path — it only looks at `wl_drm` to
    // discover the compositor's DRM device. Without this, NVIDIA EGL
    // returns EGL_NO_DISPLAY for Wayland clients and they fall back to
    // Mesa llvmpipe (software). v1.1.13+ uses dmabuf-feedback and works
    // without `wl_drm`, but we advertise both for the long tail.
    let _ = dh.create_global::<Compositor, crate::wl_drm_proto::wl_drm::WlDrm, ()>(2, ());
    let _ = dh.create_global::<Compositor, WpSinglePixelBufferManagerV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, WpViewporter, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZwpKeyboardShortcutsInhibitManagerV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, WpPresentation, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZxdgDecorationManagerV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, WpFractionalScaleManagerV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZwpPrimarySelectionDeviceManagerV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZwpPointerConstraintsV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZwpRelativePointerManagerV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZwpTextInputManagerV3, ()>(1, ());

    // Group 1: layer shell (plasmashell panel)
    let _ = dh.create_global::<Compositor, ZwlrLayerShellV1, ()>(4, ());
    // Group 2: xdg_output (KWin / plasmashell output info)
    let _ = dh.create_global::<Compositor, ZxdgOutputManagerV1, ()>(3, ());
    // Group 3: xdg_activation, ext_idle_notify
    let _ = dh.create_global::<Compositor, XdgActivationV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, ExtIdleNotifierV1, ()>(1, ());
    // wp_linux_drm_syncobj_v1: advertised iff a DRM device could be opened.
    // The implementation must actually wait/signal points or KWin 6.x
    // deadlocks waiting for fence signals that never arrive. See syncobj.rs.
    //
    // Some inner clients (glmark2-wayland, simple test programs) get
    // wedged when the protocol is advertised because they wait on
    // server-side syncobj acks they never receive. Set
    // WAYMUX_DISABLE_SYNCOBJ=1 to force the implicit-sync fallback path
    // even when a DRM node is available. Verified 2026-05-11 on
    // Maryland Blackwell hosts: with syncobj advertised, glmark2
    // commits stop flowing after xdg_toplevel creation; with it
    // disabled, commits flow normally and the recording pipeline
    // captures every frame.
    let syncobj_disabled = std::env::var("WAYMUX_DISABLE_SYNCOBJ")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let syncobj_device = if syncobj_disabled {
        tracing::info!("syncobj: disabled via WAYMUX_DISABLE_SYNCOBJ");
        None
    } else {
        SyncobjDevice::open()
    };
    if syncobj_device.is_some() {
        let _ = dh.create_global::<Compositor, WpLinuxDrmSyncobjManagerV1, ()>(1, ());
    }
    // Group 4: alpha_modifier, xdg_foreign_v2
    let _ = dh.create_global::<Compositor, WpAlphaModifierV1, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZxdgExporterV2, ()>(1, ());
    let _ = dh.create_global::<Compositor, ZxdgImporterV2, ()>(1, ());
    // Group 5: plasma window management — lets plasmashell taskbar enumerate
    // and reflect xdg_toplevels. Version 8 covers the minimum surface area
    // (window/window_with_uuid events, virtual_desktop_entered).
    let _ = dh.create_global::<Compositor, OrgKdePlasmaWindowManagement, ()>(8, ());
    // Group 6: wlr screencopy — lets grim/wayshot capture the session.
    let _ = dh.create_global::<Compositor, ZwlrScreencopyManagerV1, ()>(3, ());

    let mut comp = Compositor {
        state,
        syncobj_device,
    };

    let listener_fd = listener.as_raw_fd();
    let display_fd = display.backend().poll_fd().as_raw_fd();
    let wake_fd = comp.state.wake_fd();
    let mut last_poke_ms: u64 = 0;
    let mut last_drain_cb_ms: u64 = 0;
    loop {
        let mut accepted_any = false;
        loop {
            match listener.accept() {
                Ok((stream, _)) => match dh.insert_client(stream, Arc::new(DefaultClientData)) {
                    Ok(_) => {
                        accepted_any = true;
                    }
                    Err(e) => warn!(error = %e, "insert_client failed"),
                },
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => {
                    warn!(error = %e, "accept failed");
                    break;
                }
            }
        }
        // A new connection on the listener socket is "real" activity —
        // bump the idle clock so we don't immediately throttle a freshly
        // launched client's first frames.
        if accepted_any {
            comp.state.note_activity();
        }

        match display.dispatch_clients(&mut comp) {
            Ok(_) => {}
            Err(e) => warn!(error = %e, "dispatch_clients failed"),
        }

        // Idle-suspend gate. When NOTHING is consuming this session's frames
        // (no attached outer view, no recording, no viewer) AND we haven't seen
        // activity (input, focus, attach, child spawn) in the last 30 s,
        // throttle frame-callback delivery to ~1 Hz. Inner clients that
        // follow Wayland's frame-callback discipline (request a callback,
        // wait for `done` before rendering the next frame) self-rate-limit
        // to that cadence and CPU drops to <5%. As soon as anything pokes
        // wake_compositor (or a new client connects, above), the activity
        // clock resets and we snap back to 60 Hz.
        //
        // `has_active_frame_consumer()` is the critical guard: a recording or
        // viewer running with no concurrent input would otherwise idle-suspend
        // after the grace period and capture/stream ~1 fps (the rest padded by
        // min-fps duplication) — i.e. a frozen-looking video.
        let now_ms_idle = monotonic_now_ms();
        let idle = !comp.state.has_active_frame_consumer()
            && now_ms_idle.saturating_sub(comp.state.last_activity_ms()) >= IDLE_GRACE_MS;
        let drain_interval = if idle { IDLE_FRAME_INTERVAL_MS } else { 0 };
        if now_ms_idle.saturating_sub(last_drain_cb_ms) >= drain_interval {
            // Fire any `wl_surface.frame` callbacks parked during dispatch.
            // Doing this AFTER dispatch_clients returns (rather than inside
            // the dispatch handler) is mandatory — see State field doc.
            comp.state.drain_frame_callbacks();
            comp.state.drain_presentation_feedbacks();
            last_drain_cb_ms = now_ms_idle;
        }

        // Poke all registered client surfaces via preferred_buffer_scale at
        // most once per 16 ms. Without this rate limit, at 180+ fps the loop
        // runs 180+ times/second and sends 180+ preferred_buffer_scale events
        // per second to KWin, thrashing KWin's event loop and destabilising
        // its render schedule. The intended rate is ~60 Hz (once per 16 ms).
        if now_ms_idle.saturating_sub(last_poke_ms) >= 16 {
            comp.state.poke_client_surfaces();
            last_poke_ms = now_ms_idle;
        }
        // Forward any new outer clipboard data to inner clients. Requires
        // dh (DisplayHandle) so it must run here, not inside State.
        drain_clipboard(&mut comp, &dh);
        if let Err(e) = display.flush_clients() {
            warn!(error = %e, "flush_clients failed");
        }

        // Block until ANY of the three fds becomes readable:
        //   - listener: a new Wayland client connects
        //   - display:  a client sent a request
        //   - wake_fd:  another thread queued events on Wayland resources
        //               (inject_key, focus change, etc.) and wants us to flush
        //
        // Before 2026-04-23 this used a 100ms poll ceiling; input latency
        // tailed at the timeout. Infinite-timeout + eventfd drops that to
        // the scheduling floor (≪1ms).
        let timeout = if idle {
            IDLE_POLL_TIMEOUT_MS
        } else {
            ACTIVE_POLL_TIMEOUT_MS
        };
        poll_wakeup(listener_fd, display_fd, wake_fd, timeout);
        drain_wake(wake_fd);
    }
}

/// Window after the last activity event during which we keep polling at the
/// active 16 ms cadence. After this elapses with no further activity AND no
/// outer view attached, the loop drops into idle suspend.
const IDLE_GRACE_MS: u64 = 30_000;
/// Poll timeout while idle. Inner clients following frame-callback discipline
/// self-rate-limit to ~1 Hz.
const IDLE_POLL_TIMEOUT_MS: i32 = 1_000;
/// Active poll timeout. 8 ms ≈ 120 fps — caps the internal frame-callback
/// drain rate (and thus KWin's repaint cadence) at 120 Hz. Was 16 ms (60 fps),
/// which capped the source rate the viewer encoder could ever see; raising it
/// to 120 lets fast content (video, scrolling) reach the encoder at up to
/// 120 fps. The 60 fps wire cap in run_cuda_nvenc_encoder still bounds what
/// actually goes to the browser; this just removes the upstream 60 Hz ceiling.
const ACTIVE_POLL_TIMEOUT_MS: i32 = 8;
/// Min interval between `drain_frame_callbacks` invocations while idle.
const IDLE_FRAME_INTERVAL_MS: u64 = 1_000;

/// Refresh rate (mHz) advertised on the virtual `wl_output`. It gates the rate
/// at which KWin delivers composited frames to waymux-session's output (and
/// thus to the viewer encoder). 240 Hz.
///
/// History: a streaming-tuning attempt lowered this to 60 Hz on the theory that
/// the encoder's 60 fps wire cap made anything higher wasted GPU. A live A/B on
/// an L40 DISPROVED it: at 60 Hz the encoder was starved to ~31 fps (glxgears);
/// at 240 Hz it rose to ~45 fps — same 4.6 ms/frame encode, GPU only ~2% busy.
/// The output mode throttles KWin's frame *supply*, and there is no GPU to save
/// (ample headroom), so a high mode is strictly better here. Keep it at 240.
/// Must stay in lockstep with `PRESENTATION_REFRESH_PERIOD_NS`.
pub const VIRTUAL_OUTPUT_REFRESH_MHZ: i32 = 240_000;
/// `wp_presentation` feedback refresh PERIOD (ns). MUST equal
/// 1e9 / (VIRTUAL_OUTPUT_REFRESH_MHZ / 1000). KWin computes its next-paint
/// delay from this value; if it disagrees with the advertised output mode the
/// render timer drifts. Moves with the mode.
pub const PRESENTATION_REFRESH_PERIOD_NS: u32 = 4_166_667; // round(1e9 / 240)

fn monotonic_now_ms() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000 + (ts.tv_nsec as u64) / 1_000_000
}

fn poll_wakeup(
    a: std::os::fd::RawFd,
    b: std::os::fd::RawFd,
    c: std::os::fd::RawFd,
    timeout_ms: i32,
) {
    let mut fds = [
        libc::pollfd {
            fd: a,
            events: libc::POLLIN,
            revents: 0,
        },
        libc::pollfd {
            fd: b,
            events: libc::POLLIN,
            revents: 0,
        },
        libc::pollfd {
            fd: c,
            events: libc::POLLIN,
            revents: 0,
        },
    ];
    // 16 ms timeout (active) drives frame callbacks at ~60 fps even when
    // KWin has nothing to send between frames. 1 s timeout (idle) lets the
    // loop sleep and drops CPU to <5% when no outer view is attached.
    unsafe {
        libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout_ms);
    }
}

/// Consume any pending bytes on the wakeup eventfd. A single read of 8
/// bytes clears the counter. Looping until EAGAIN is not required (eventfd
/// semantics guarantee one read drains all).
fn drain_wake(fd: std::os::fd::RawFd) {
    let mut buf = [0u8; 8];
    unsafe {
        let _ = libc::read(fd, buf.as_mut_ptr().cast(), buf.len());
    }
}

// ─── GlobalDispatch impls ───────────────────────────────────────────────

impl GlobalDispatch<WlCompositor, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WlCompositor>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl GlobalDispatch<WlSubcompositor, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WlSubcompositor>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl GlobalDispatch<WlShm, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WlShm>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        let shm = di.init(r, ());
        shm.format(wl_shm::Format::Argb8888);
        shm.format(wl_shm::Format::Xrgb8888);
    }
}

impl GlobalDispatch<WlSeat, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WlSeat>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        let seat = di.init(r, ());
        // Advertise Touch alongside Keyboard + Pointer. Clients
        // that ignore the bit (Chromium, KWin without QT_QPA_PLATFORMTHEME)
        // continue to operate identically: wl_seat.get_touch is a separate
        // request and is simply never issued.
        seat.capabilities(
            wl_seat::Capability::Keyboard
                | wl_seat::Capability::Pointer
                | wl_seat::Capability::Touch,
        );
        if seat.version() >= 2 {
            seat.name("waymux-seat".into());
        }
    }
}

impl GlobalDispatch<WlOutput, ()> for Compositor {
    fn bind(
        state: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WlOutput>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        let (w, h, scale) = state.state.snapshot();
        let output = di.init(r, ());
        output.geometry(
            0,
            0,
            (w as i32 * 265) / 1000,
            (h as i32 * 265) / 1000,
            wl_output::Subpixel::Unknown,
            "waymux".into(),
            "virtual".into(),
            wl_output::Transform::Normal,
        );
        // Virtual output refresh — governs KWin's internal RenderLoop / repaint
        // cadence (the real pacing lever; frame callbacks only throttle inner
        // *client* surfaces, not KWin's output). 60 Hz matches the encoder's
        // 60 fps wire cap.
        output.mode(
            wl_output::Mode::Current | wl_output::Mode::Preferred,
            w as i32,
            h as i32,
            VIRTUAL_OUTPUT_REFRESH_MHZ,
        );
        if output.version() >= 2 {
            output.scale(scale as i32);
        }
        // wl_output v4 added name + description events. Modern SDL2 binds v4
        // and rejects outputs as "incomplete" if it never sees them, erroring
        // with "The video driver did not add any displays".
        if output.version() >= 4 {
            output.name("waymux".into());
            output.description("waymux virtual output".into());
        }
        if output.version() >= 2 {
            output.done();
        }
        // Track this output so `State::resize` can re-send its `mode` event
        // when the session is resized.
        state.state.register_output(output);
    }
}

impl GlobalDispatch<WlDataDeviceManager, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WlDataDeviceManager>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<WlDataDeviceManager, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &WlDataDeviceManager,
        request: wl_data_device_manager::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_device_manager::Request::CreateDataSource { id } => {
                // User data holds the MIME types offered by the client. We
                // collect them here so the clipboard bridge can advertise
                // the same types on the outer compositor.
                di.init(id, Arc::new(Mutex::new(Vec::<String>::new())));
            }
            wl_data_device_manager::Request::GetDataDevice { id, .. } => {
                let dev = di.init(id, ());
                if state.state.share_clipboard() {
                    state.state.register_data_device(dev);
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WlDataDevice, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        resource: &WlDataDevice,
        request: wl_data_device::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_device::Request::SetSelection { source, .. } => {
                if state.state.share_clipboard() {
                    if let Some(src) = source {
                        if let Some(mime_list) = src.data::<Arc<Mutex<Vec<String>>>>() {
                            let types = mime_list.lock().unwrap().clone();
                            state.state.set_inner_selection(types, src);
                        }
                    } else {
                        state.state.clear_inner_selection();
                    }
                }
            }
            wl_data_device::Request::Release if state.state.share_clipboard() => {
                state.state.unregister_data_device(resource);
            }
            _ => {}
        }
    }

    fn destroyed(state: &mut Self, _client: ClientId, resource: &WlDataDevice, _data: &()) {
        if state.state.share_clipboard() {
            state.state.unregister_data_device(resource);
        }
    }
}

impl Dispatch<WlDataSource, Arc<Mutex<Vec<String>>>> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlDataSource,
        request: wl_data_source::Request,
        mime_types: &Arc<Mutex<Vec<String>>>,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        if let wl_data_source::Request::Offer { mime_type } = request {
            mime_types.lock().unwrap().push(mime_type);
        }
        // set_actions / destroy — no-op.
    }
}

/// Dispatch for wl_data_offer objects we create for inner clients when
/// forwarding the outer clipboard. The Arc<ClipboardContent> holds the
/// eagerly-fetched outer selection data.
impl Dispatch<WlDataOffer, Arc<ClipboardContent>> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlDataOffer,
        request: wl_data_offer::Request,
        content: &Arc<ClipboardContent>,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        if let wl_data_offer::Request::Receive { mime_type, fd } = request {
            if let Some((_, data)) = content.entries.iter().find(|(m, _)| m == &mime_type) {
                unsafe {
                    libc::write(fd.as_raw_fd(), data.as_ptr().cast(), data.len());
                }
            }
            // fd is dropped here, closing the write end of the pipe so the
            // client sees EOF on its read end.
        }
        // finish / set_actions / destroy: no-op.
    }
}

impl GlobalDispatch<XdgWmBase, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<XdgWmBase>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

// ─── Dispatch impls ─────────────────────────────────────────────────────

impl Dispatch<WlCompositor, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlCompositor,
        request: wl_compositor::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_compositor::Request::CreateSurface { id } => {
                di.init(id, SurfaceData::default());
            }
            wl_compositor::Request::CreateRegion { id } => {
                di.init(id, ());
            }
            _ => {}
        }
    }
}

impl Dispatch<WlSurface, SurfaceData> for Compositor {
    fn request(
        state: &mut Self,
        client: &Client,
        surface_resource: &WlSurface,
        request: wl_surface::Request,
        data: &SurfaceData,
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_surface::Request::Attach { buffer, .. } => {
                *data.pending_buffer.lock().unwrap() = buffer;
            }
            wl_surface::Request::Commit => {
                // ── Explicit-sync (wp_linux_drm_syncobj_v1) ──────────────
                //
                // If KWin set acquire/release points on this surface,
                // drain them now. We BLOCK CPU-side on the acquire point
                // before promoting the buffer to current — otherwise we
                // would forward to niri (or read for recording) while
                // KWin's GPU work is still in flight, which is the
                // entire reason the protocol exists. Then we attach the
                // release point to the just-committed buffer; it will be
                // signaled when the buffer is released back to KWin.
                let pending_sync = data
                    .surface_sync
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|s| s.take_pending());
                let mut acquire_signaled = true;
                if let Some((Some(p), _)) = pending_sync.as_ref() {
                    // Live-UX deadline: 50 ms. When recording is active
                    // we extend the deadline because the alternative is
                    // recording a buffer KWin's GPU is still writing,
                    // visible as a horizontal seam across thumbnails on
                    // fast YouTube-grid scrolls. The compositor stall is
                    // bounded and only affects users who explicitly opted
                    // into recording.
                    let recording_active = state.state.is_recording();
                    let timeout_ns: i64 = if recording_active {
                        500_000_000
                    } else {
                        50_000_000
                    };
                    let waited = p.wait(timeout_ns);
                    if !waited {
                        warn!(
                            timeout_ms = timeout_ns / 1_000_000,
                            "syncobj: acquire-point wait timed out, forwarding anyway"
                        );
                        acquire_signaled = false;
                    }
                }

                // Promote pending → current. Release the PREVIOUS current
                // (if any) so the client can reuse that slot. HOLD the
                // just-accepted buffer as current until a subsequent
                // commit replaces it — this matches the "normal" Wayland
                // double-buffering contract.
                //
                // Earlier versions released the new buffer immediately
                // too, which is protocol-legal but trips GTK/cairo's
                // ref-count assertions: firefox's GDK saw a release on
                // the buffer that backed its still-staged cairo surface.
                // Foot / simple shm clients don't notice the difference.
                let mut pending = data.pending_buffer.lock().unwrap();
                let mut current = data.current_buffer.lock().unwrap();
                let new_buf = pending.take();
                let _has_new_buffer = new_buf.is_some();
                // Stash the cur-release work so we can run it AFTER the
                // commit-driven recording tap. In headless mode with no
                // outer consumer, the cur-release path immediately fires
                // wl_buffer.release on this just-attached buffer — chromium
                // is then free to start writing the next frame to it.
                // Running the recording tap (which mmaps the dmabuf for
                // CPU readback in some codecs) AFTER that release races
                // chromium's next render: torn-frame chroma artifacts
                // visible as diagonal stripes through the recording.
                // Defer the release until after the tap so the eager
                // mmap+memcpy completes against a stable buffer.
                let mut pending_cur_release: Option<(WlBuffer, bool)> = None;
                if let Some(nb) = new_buf {
                    if let Some(prev) = current.replace(nb) {
                        // For SHM/SinglePixel: release immediately (CPU memory,
                        // no GPU race). For dmabuf: do NOT release — prev was
                        // already queued for deferred release when it was
                        // first committed as cur. Releasing again here would
                        // hand the GEM buffer back to KWin before niri has
                        // finished reading it, causing the skybox-flash race.
                        let is_dmabuf = prev
                            .data::<BufferKind>()
                            .map(|k| matches!(k, BufferKind::Dmabuf(_)))
                            .unwrap_or(false);
                        if !is_dmabuf {
                            state.state.release_inner_buffer(&prev);
                        }
                    }
                    // For dmabuf: defer release until outer_view has forwarded
                    // this buffer to niri AND niri has released it. outer_view
                    // calls take_pending_releases() and fires them on niri's
                    // wl_buffer.release event. For SHM/other: release now.
                    //
                    // Headless (no outer_view): no consumer chain exists, so
                    // deferring would let the queue grow unbounded and exhaust
                    // KWin's GBM pool — release after the recording tap.
                    if let Some(cur) = current.as_ref() {
                        let is_dmabuf = cur
                            .data::<BufferKind>()
                            .map(|k| matches!(k, BufferKind::Dmabuf(_)))
                            .unwrap_or(false);
                        // Attach the release-syncobj point to this buffer
                        // (if any) BEFORE deferred-release queues it, so
                        // the release path always finds it.
                        if let Some((_, Some(release_pt))) = pending_sync {
                            state.state.note_buffer_release_point(cur, release_pt);
                        }
                        if is_dmabuf && state.state.is_attached() {
                            state.state.queue_deferred_release(cur.clone());
                        } else {
                            // Defer the release until AFTER the recording
                            // tap so the mmap+memcpy doesn't race chromium's
                            // next render into the same buffer.
                            pending_cur_release = Some((cur.clone(), is_dmabuf));
                        }
                    }
                }
                drop(pending);
                drop(current);
                // Fix C — commit-driven recording tap. Runs after the
                // buffer has been promoted to current, before frame
                // callbacks fire. Synchronous capture: if recording is
                // active and this surface qualifies (focused-window
                // mode) or every commit fires (whole-desktop mode), we
                // mmap+memcpy into the recording slot here. The
                // explicit-sync acquire fence above already drained, so
                // the buffer's GPU writes are flushed; the inner client
                // can't reuse the memory until the release path below
                // runs, so a race-free read is guaranteed for the
                // duration of this call.
                // Tell the recording tap whether the explicit-sync acquire
                // fence actually signaled. The tap uses this together with
                // the implicit dmabuf fence probe to decide whether the
                // buffer's GPU writes are truly complete: when acquire timed
                // out AND the implicit fence is still unsignaled, the buffer
                // is mid-write and would record as a torn frame. Either
                // signal alone is enough — chromium often signals one but
                // not the other depending on the swap-chain path.
                {
                    static COMMIT_COUNTER: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let n = COMMIT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n < 8 || n.is_multiple_of(60) {
                        let rec_active = state.state.is_recording();
                        tracing::info!(
                            commit_n = n,
                            rec_active,
                            "wl_surface::Commit fired (pre-tap diagnostic)"
                        );
                    }
                }
                state
                    .state
                    .maybe_tap_for_recording(surface_resource, acquire_signaled);
                // Now safe to fire the deferred wl_buffer.release for cur:
                // the recording tap has already done its mmap+memcpy
                // against the just-attached buffer.
                if let Some((cur_buf, _is_dmabuf)) = pending_cur_release.take() {
                    state.state.release_inner_buffer(&cur_buf);
                }
                // If this commit is for the current cursor surface, refresh the
                // cursor image (shape-change-gated inside note_cursor_dirty).
                {
                    let is_cursor = state
                        .state
                        .cursor_surface
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|(s, _, _)| s == surface_resource)
                        .unwrap_or(false);
                    if is_cursor {
                        state.state.note_cursor_dirty();
                    }
                }
                // Only promote per-surface frame callbacks to the global
                // ready queue when a new buffer was attached. This ensures
                // callbacks fire *after* the commit that paces them, not
                // before (fixes KWin render-loop stall on AMD DCC path).
                // Fire pending frame callbacks on every commit regardless of
                // whether a new buffer was attached. KWin requests the callback
                // on the parent surface (wl_surface#3) but renders content to a
                // subsurface — the parent commit applies the transaction without
                // attaching a new buffer, so "new-buffer only" firing deadlocks
                // KWin after the first frame.
                {
                    let cbs: Vec<WlCallback> =
                        data.surface_frame_cbs.lock().unwrap().drain(..).collect();
                    for cb in cbs {
                        state.state.queue_frame_callback(cb);
                    }
                }
                state.state.record_damage();
                // Register layer surfaces on first commit (equivalent to
                // xdg_toplevel registration which happens at GetToplevel).
                if *data.is_layer_surface.lock().unwrap() {
                    let mut layer_wid = data.layer_window_id.lock().unwrap();
                    if layer_wid.is_none() {
                        let (w, h, _) = state.state.snapshot();
                        let wid = state.state.add_window(
                            -1,
                            waymux_protocol::Rect {
                                x: 0,
                                y: 0,
                                width: w,
                                height: h,
                            },
                        );
                        // Q5 of the 2026-05-18 click design decisions: layer
                        // surfaces with keyboard_interactivity = none (the
                        // protocol default, used by Plasma panels, status
                        // indicators, notification daemons) or on_demand do
                        // NOT steal focus from xdg_toplevels on register.
                        // Only `exclusive` layer surfaces (lock screens,
                        // modal launchers) get focus. This matches every
                        // modern Wayland compositor; the previous
                        // always-focus behaviour was a bug that broke SDK
                        // clicks against xdg_toplevels whenever a panel
                        // happened to register after them.
                        let interactivity = *data.layer_keyboard_interactivity.lock().unwrap();
                        if interactivity == 1 {
                            state.state.register_surface(
                                wid,
                                surface_resource.clone(),
                                client.id(),
                            );
                        } else {
                            state.state.register_surface_no_focus(
                                wid,
                                surface_resource.clone(),
                                client.id(),
                            );
                        }
                        *layer_wid = Some(wid);
                        info!(
                            window_id = wid,
                            keyboard_interactivity = interactivity,
                            focused = interactivity == 1,
                            "layer_surface registered on first commit"
                        );
                    }
                }
            }
            wl_surface::Request::Frame { callback } => {
                // Clients (foot, ghostty, firefox, KWin) use wl_surface.frame
                // for render pacing. The `callback` new_id MUST be initialised,
                // or wayland-backend aborts the whole process with a
                // non-unwinding panic.
                //
                // We DO NOT call `cb.done()` here. wl_callback.done is a
                // destructor event; calling it inside the dispatch handler
                // makes wayland-backend 0.3.15 double-drop the callback's
                // ObjectData Arc on the same stack, segfaulting in
                // Arc::drop. Park the callback on the surface; Commit
                // promotes it to the global ready queue when a new buffer
                // arrives (see wl_surface::Request::Commit above and
                // State::drain_frame_callbacks).
                let cb = di.init(callback, ());
                // Store per-surface so the Commit path can drain it immediately.
                // Also add to the global timer queue so the 16 ms poll timeout
                // fires it if the surface never commits again (e.g. KWin stops
                // committing when its scene is undamaged, but waymux must still
                // deliver VSync signals so KWin's render loop stays alive when
                // inner clients later produce damage).
                data.surface_frame_cbs.lock().unwrap().push(cb.clone());
                state.state.queue_frame_callback(cb);
            }
            _ => {}
        }
    }
}

impl Dispatch<WlRegion, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlRegion,
        _req: wl_region::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<WlSubcompositor, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlSubcompositor,
        request: wl_subcompositor::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        if let wl_subcompositor::Request::GetSubsurface {
            id,
            surface,
            parent,
        } = request
        {
            // Register the child with the parent so we can composite the
            // tree when capturing frames. Position defaults to (0, 0) until
            // the client calls wl_subsurface.set_position.
            if let Some(pd) = parent.data::<SurfaceData>() {
                pd.children.lock().unwrap().push(SubsurfaceChild {
                    surface: surface.clone(),
                    x: 0,
                    y: 0,
                });
            }
            di.init(id, SubsurfaceData { surface, parent });
        }
    }
}

impl Dispatch<WlSubsurface, SubsurfaceData> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlSubsurface,
        req: wl_subsurface::Request,
        data: &SubsurfaceData,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        match req {
            wl_subsurface::Request::SetPosition { x, y } => {
                if let Some(pd) = data.parent.data::<SurfaceData>() {
                    let mut children = pd.children.lock().unwrap();
                    let child_id = data.surface.id();
                    for child in children.iter_mut() {
                        if child.surface.id() == child_id {
                            child.x = x;
                            child.y = y;
                            break;
                        }
                    }
                }
            }
            wl_subsurface::Request::Destroy => {
                if let Some(pd) = data.parent.data::<SurfaceData>() {
                    let child_id = data.surface.id();
                    pd.children
                        .lock()
                        .unwrap()
                        .retain(|c| c.surface.id() != child_id);
                }
            }
            wl_subsurface::Request::PlaceAbove { sibling } => {
                if let Some(pd) = data.parent.data::<SurfaceData>() {
                    let mut children = pd.children.lock().unwrap();
                    let child_id = data.surface.id();
                    let sibling_id = sibling.id();
                    let parent_id = data.parent.id();
                    if let Some(ci) = children.iter().position(|c| c.surface.id() == child_id) {
                        let child = children.remove(ci);
                        let insert_at = if sibling_id == parent_id {
                            children.len() // sibling is parent → top of stack
                        } else {
                            children
                                .iter()
                                .position(|c| c.surface.id() == sibling_id)
                                .map(|si| si + 1)
                                .unwrap_or(children.len())
                        };
                        children.insert(insert_at, child);
                    }
                }
            }
            wl_subsurface::Request::PlaceBelow { sibling } => {
                if let Some(pd) = data.parent.data::<SurfaceData>() {
                    let mut children = pd.children.lock().unwrap();
                    let child_id = data.surface.id();
                    let sibling_id = sibling.id();
                    let parent_id = data.parent.id();
                    if let Some(ci) = children.iter().position(|c| c.surface.id() == child_id) {
                        let child = children.remove(ci);
                        let insert_at = if sibling_id == parent_id {
                            0 // sibling is parent → bottom of stack
                        } else {
                            children
                                .iter()
                                .position(|c| c.surface.id() == sibling_id)
                                .unwrap_or(0)
                        };
                        children.insert(insert_at, child);
                    }
                }
            }
            // SetSync / SetDesync: we apply children state immediately on
            // commit, which is compatible with sync mode on parent commits
            // and desync at all other times.
            _ => {}
        }
    }

    fn destroyed(_s: &mut Self, _c: ClientId, _r: &WlSubsurface, data: &SubsurfaceData) {
        // If a client goes away without explicit Destroy, prune the child
        // from the parent's list so capture_window doesn't read a dead
        // WlSurface.
        if let Some(pd) = data.parent.data::<SurfaceData>() {
            let child_id = data.surface.id();
            pd.children
                .lock()
                .unwrap()
                .retain(|c| c.surface.id() != child_id);
        }
    }
}

impl Dispatch<WlShm, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlShm,
        request: wl_shm::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        if let wl_shm::Request::CreatePool { id, fd, size } = request {
            let pool = ShmPoolData::new(fd, size);
            di.init(id, pool);
        }
    }
}

impl Dispatch<WlShmPool, Arc<ShmPoolData>> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlShmPool,
        request: wl_shm_pool::Request,
        pool: &Arc<ShmPoolData>,
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_shm_pool::Request::CreateBuffer {
                id,
                offset,
                width,
                height,
                stride,
                format,
            } => {
                let data = ShmBufferData {
                    pool: pool.clone(),
                    offset,
                    width,
                    height,
                    stride,
                    format,
                };
                di.init(id, BufferKind::Shm(data));
            }
            wl_shm_pool::Request::Resize { size } => {
                pool.resize(size);
            }
            wl_shm_pool::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WlCallback, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlCallback,
        _request: wl_callback::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // wl_callback has no client-to-server requests; this impl exists
        // only so `di.init(callback, ())` in wl_surface.frame is legal.
    }
}

impl Dispatch<WlBuffer, BufferKind> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlBuffer,
        _request: wl_buffer::Request,
        _d: &BufferKind,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<WlSeat, ()> for Compositor {
    fn request(
        state: &mut Self,
        client: &Client,
        _r: &WlSeat,
        request: wl_seat::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        use wl_seat::Request;
        match request {
            Request::GetKeyboard { id } => {
                let cid = client.id();
                let kbd = di.init(id, cid.clone());
                // Send the active keymap — either the outer compositor's
                // keymap (passed through for correct non-US layout support)
                // or the default embedded US pc105 keymap.
                {
                    let (fd, size) = state.state.get_keymap();
                    use std::os::fd::AsFd;
                    kbd.keymap(
                        wayland_server::protocol::wl_keyboard::KeymapFormat::XkbV1,
                        fd.as_fd(),
                        size,
                    );
                }
                // Announce key-repeat parameters. Without this, clients
                // assume no repeat ⇒ holding Backspace only deletes one
                // character, holding an arrow only moves once, etc.
                // Values here match the typical Linux desktop defaults.
                if kbd.version() >= 4 {
                    kbd.repeat_info(25, 400);
                }
                state.state.register_keyboard(cid, kbd);
            }
            Request::GetPointer { id } => {
                let cid = client.id();
                let ptr = di.init(id, cid.clone());
                state.state.register_pointer(cid, ptr);
            }
            Request::GetTouch { id } => {
                // Register the wl_touch resource per client so
                // `State::inject_touch` can dispatch synthetic events. The
                // ObjectData is the ClientId (mirrors get_pointer) so
                // `destroyed` can unregister with no extra bookkeeping.
                let cid = client.id();
                let touch = di.init(id, cid.clone());
                state.state.register_touch(cid, touch);
            }
            _ => {}
        }
    }
}

impl Dispatch<wayland_server::protocol::wl_keyboard::WlKeyboard, wayland_server::backend::ClientId>
    for Compositor
{
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &wayland_server::protocol::wl_keyboard::WlKeyboard,
        _req: wayland_server::protocol::wl_keyboard::Request,
        _d: &wayland_server::backend::ClientId,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }

    fn destroyed(
        state: &mut Self,
        _client: wayland_server::backend::ClientId,
        resource: &wayland_server::protocol::wl_keyboard::WlKeyboard,
        data: &wayland_server::backend::ClientId,
    ) {
        state.state.unregister_keyboard(data.clone(), resource);
    }
}

impl Dispatch<wayland_server::protocol::wl_pointer::WlPointer, wayland_server::backend::ClientId>
    for Compositor
{
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &wayland_server::protocol::wl_pointer::WlPointer,
        req: wayland_server::protocol::wl_pointer::Request,
        _d: &wayland_server::backend::ClientId,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        use wayland_server::protocol::wl_pointer::Request as PtrReq;
        if let PtrReq::SetCursor {
            surface,
            hotspot_x,
            hotspot_y,
            ..
        } = req
        {
            *state.state.cursor_surface.lock().unwrap() =
                surface.map(|s| (s, hotspot_x, hotspot_y));
            state.state.note_cursor_dirty();
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: wayland_server::backend::ClientId,
        resource: &wayland_server::protocol::wl_pointer::WlPointer,
        data: &wayland_server::backend::ClientId,
    ) {
        state.state.unregister_pointer(data.clone(), resource);
    }
}

impl Dispatch<wayland_server::protocol::wl_touch::WlTouch, wayland_server::backend::ClientId>
    for Compositor
{
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &wayland_server::protocol::wl_touch::WlTouch,
        _req: wayland_server::protocol::wl_touch::Request,
        _d: &wayland_server::backend::ClientId,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // wl_touch has a single request — `release` (v3+) — and its only
        // effect is to destroy the resource. The `destroyed` callback below
        // is what runs the unregister bookkeeping; nothing to do here.
    }

    fn destroyed(
        state: &mut Self,
        _client: wayland_server::backend::ClientId,
        resource: &wayland_server::protocol::wl_touch::WlTouch,
        data: &wayland_server::backend::ClientId,
    ) {
        state.state.unregister_touch(data.clone(), resource);
    }
}

impl Dispatch<WlOutput, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WlOutput,
        _req: wl_output::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<XdgWmBase, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        resource: &XdgWmBase,
        request: xdg_wm_base::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_wm_base::Request::CreatePositioner { id } => {
                di.init(id, ());
            }
            xdg_wm_base::Request::GetXdgSurface { id, surface } => {
                di.init(id, XdgSurfaceData { surface });
            }
            xdg_wm_base::Request::Pong { .. } => {}
            xdg_wm_base::Request::Destroy => {}
            _ => {
                warn!("unhandled xdg_wm_base request on {:?}", resource);
            }
        }
    }
}

impl Dispatch<XdgPositioner, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &XdgPositioner,
        _req: wayland_protocols::xdg::shell::server::xdg_positioner::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<XdgSurface, XdgSurfaceData> for Compositor {
    fn request(
        state: &mut Self,
        client: &Client,
        resource: &XdgSurface,
        request: xdg_surface::Request,
        data: &XdgSurfaceData,
        dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_surface::Request::GetToplevel { id } => {
                let pid = client.get_credentials(dh).map(|c| c.pid).unwrap_or(-1);
                let (w, h, _) = state.state.snapshot();
                let window_id = state.state.add_window(
                    pid,
                    waymux_protocol::Rect {
                        x: 0,
                        y: 0,
                        width: w,
                        height: h,
                    },
                );
                let toplevel = di.init(id, ToplevelData { window_id, pid });
                toplevel.configure(w as i32, h as i32, Vec::new());
                resource.configure(next_serial());

                if let Some(sd) = data.surface.data::<SurfaceData>() {
                    *sd.xdg_toplevel_id.lock().unwrap() = Some(window_id);
                }
                state
                    .state
                    .register_surface(window_id, data.surface.clone(), client.id());
                // Track the (xdg_surface, xdg_toplevel) pair so `State::resize`
                // can re-send a `configure` to this window.
                state
                    .state
                    .register_toplevel(window_id, resource.clone(), toplevel.clone());
                info!(window_id, pid, "xdg_toplevel created");
            }
            xdg_surface::Request::GetPopup { id, .. } => {
                di.init(id, ());
            }
            xdg_surface::Request::AckConfigure { .. } => {}
            xdg_surface::Request::SetWindowGeometry {
                x,
                y,
                width,
                height,
            } => {
                // Clients almost always send sane values; defensive `.max(0)` on
                // width/height catches hostile or buggy clients. Sentinel
                // (-1, -1, <=0, <=0) means "no geometry, use buffer dims" —
                // treat as None per the xdg-shell convention.
                let is_sentinel = x == -1 && y == -1 && width <= 0 && height <= 0;
                if let Some(sd) = data.surface.data::<SurfaceData>() {
                    *sd.content_rect.lock().unwrap() = if is_sentinel {
                        None
                    } else {
                        Some(waymux_protocol::Rect {
                            x,
                            y,
                            width: width.max(0) as u32,
                            height: height.max(0) as u32,
                        })
                    };
                }
            }
            xdg_surface::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<XdgPopup, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &XdgPopup,
        _req: xdg_popup::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<XdgToplevel, ToplevelData> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &XdgToplevel,
        request: xdg_toplevel::Request,
        data: &ToplevelData,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_toplevel::Request::SetTitle { title } => {
                state.state.set_title(data.window_id, title);
            }
            xdg_toplevel::Request::SetAppId { app_id } => {
                state.state.set_app_id(data.window_id, app_id);
            }
            xdg_toplevel::Request::Destroy => {
                state.state.unregister_surface(data.window_id);
                if state.state.remove_window(data.window_id) {
                    info!(window_id = data.window_id, "xdg_toplevel destroyed");
                }
            }
            _ => {}
        }
    }

    fn destroyed(state: &mut Self, _c: ClientId, _r: &XdgToplevel, data: &ToplevelData) {
        state.state.unregister_surface(data.window_id);
        if state.state.remove_window(data.window_id) {
            info!(
                window_id = data.window_id,
                "xdg_toplevel destroyed (client disconnect)"
            );
        }
    }
}

/// Check if a new outer clipboard arrived and forward it to all inner clients.
///
/// Called from the compositor thread's main loop (after `drain_frame_callbacks`)
/// so it has access to the `DisplayHandle` needed to create wl_data_offer objects.
pub fn drain_clipboard(comp: &mut Compositor, dh: &DisplayHandle) {
    if !comp.state.share_clipboard() {
        return;
    }
    // Use a simple generation counter: if outer_clipboard_version has grown
    // since we last sent it, there is new data to forward.
    // We track the last-sent version in a thread-local so this function has
    // no extra allocations on the happy path (no pending clipboard).
    thread_local! {
        static LAST_SENT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    let current = comp
        .state
        .outer_clipboard_version
        .load(std::sync::atomic::Ordering::Acquire);
    let last = LAST_SENT.with(|c| c.get());
    if current == last {
        return;
    }
    LAST_SENT.with(|c| c.set(current));

    let content = match comp.state.outer_clipboard.lock().unwrap().clone() {
        Some(c) => c,
        None => return,
    };

    let devices: Vec<WlDataDevice> = comp
        .state
        .inner_data_devices
        .lock()
        .unwrap()
        .iter()
        .filter(|d| d.is_alive())
        .cloned()
        .collect();

    for dev in &devices {
        let Some(client) = dev.client() else { continue };
        // version 3 is what we advertise for wl_data_device_manager;
        // clamp to the client's negotiated version.
        let offer_ver = dev.version().min(3);
        match client.create_resource::<WlDataOffer, Arc<ClipboardContent>, Compositor>(
            dh,
            offer_ver,
            content.clone(),
        ) {
            Ok(offer) => {
                dev.data_offer(&offer);
                for (mime, _) in &content.entries {
                    offer.offer(mime.clone());
                }
                dev.selection(Some(&offer));
            }
            Err(e) => {
                debug!(error = ?e, "clipboard: create_resource for wl_data_offer failed");
            }
        }
    }
}

fn next_serial() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SERIAL: AtomicU32 = AtomicU32::new(1);
    SERIAL.fetch_add(1, Ordering::Relaxed)
}

// ─── zwp_linux_dmabuf_v1 (v4) ───────────────────────────────────────────

/// Build a memfd format table for the two formats we support
/// (ARGB8888 + XRGB8888) with every importable modifier. Each entry is 16 bytes:
///   [format: u32 LE] [pad: u32 = 0] [modifier: u64 LE]
fn make_dmabuf_format_table() -> Option<(std::os::fd::OwnedFd, u32)> {
    use std::os::fd::{FromRawFd, OwnedFd};
    let mods = crate::vulkan_record::importable_bgra_modifiers();
    let raw = unsafe { libc::memfd_create(c"waymux-dmabuf-fmt".as_ptr(), libc::MFD_CLOEXEC) };
    if raw < 0 {
        return None;
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let mut table = Vec::<u8>::with_capacity(2 * mods.len() * 16);
    for &fourcc in &[DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888] {
        for &m in mods {
            table.extend_from_slice(&fourcc.to_ne_bytes());
            table.extend_from_slice(&0u32.to_ne_bytes()); // padding
            table.extend_from_slice(&m.to_ne_bytes());
        }
    }
    let len = table.len();
    let written = unsafe { libc::write(raw, table.as_ptr().cast(), len) };
    if written as usize != len {
        return None;
    }
    Some((fd, len as u32))
}

/// Probe the main DRM render node and return its dev_t as 8 LE bytes,
/// falling back to all-zeros if no render node is present.
fn drm_main_device_bytes() -> [u8; 8] {
    use std::os::unix::fs::MetadataExt;
    for path in &[
        "/dev/dri/renderD128",
        "/dev/dri/renderD129",
        "/dev/dri/card0",
    ] {
        if let Ok(meta) = std::fs::metadata(path) {
            return meta.rdev().to_ne_bytes();
        }
    }
    [0u8; 8]
}

/// Send the complete feedback sequence on a newly-created feedback object.
///
/// Advertises ARGB8888 + XRGB8888 with every modifier in
/// `vulkan_record::importable_bgra_modifiers()` — the EGL-importable set
/// (what KWin renders to and `egl_readback` can consume), always LINEAR-
/// inclusive. KWin 6 selects a render modifier from this set; without the
/// tiled entries its NVIDIA GL backend fails "could not find a suitable
/// render format".
fn send_dmabuf_feedback(fb: &ZwpLinuxDmabufFeedbackV1) {
    use std::os::fd::AsFd;
    let device = drm_main_device_bytes();
    let device_vec: Vec<u8> = device.to_vec();

    let Some((fmt_fd, fmt_size)) = make_dmabuf_format_table() else {
        warn!("dmabuf: failed to create format table memfd — feedback skipped");
        fb.done();
        return;
    };
    fb.format_table(fmt_fd.as_fd(), fmt_size);
    fb.main_device(device_vec.clone());
    fb.tranche_target_device(device_vec);
    let entry_count = (2 * crate::vulkan_record::importable_bgra_modifiers().len()) as u16;
    let indices: Vec<u8> = (0..entry_count).flat_map(|i| i.to_ne_bytes()).collect();
    fb.tranche_formats(indices);
    fb.tranche_flags(wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_feedback_v1::TrancheFlags::empty());
    fb.tranche_done();
    fb.done();
}

impl GlobalDispatch<ZwpLinuxDmabufV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwpLinuxDmabufV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        let dmabuf = di.init(r, ());
        // v1: .format() events.  v3: .modifier() events too.  v4+: feedback object.
        // KWin Plasma 5.27 (Ubuntu 24.04 noble's apt-archive Plasma) binds at v3
        // and never calls get_default_feedback — without these events it sees
        // zero supported formats and aborts with "DRM_FORMAT_*8888 is unsupported".
        let v = dmabuf.version();
        if v < 4 {
            let mods = crate::vulkan_record::importable_bgra_modifiers();
            for &fourcc in &[DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888] {
                dmabuf.format(fourcc);
                if v >= 3 {
                    for &m in mods {
                        dmabuf.modifier(fourcc, (m >> 32) as u32, (m & 0xFFFF_FFFF) as u32);
                    }
                }
            }
        }
    }
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpLinuxDmabufV1,
        request: zwp_linux_dmabuf_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_linux_dmabuf_v1::Request::CreateParams { params_id } => {
                di.init(params_id, ParamsData::default());
            }
            zwp_linux_dmabuf_v1::Request::GetDefaultFeedback { id } => {
                let fb = di.init(id, ());
                send_dmabuf_feedback(&fb);
            }
            zwp_linux_dmabuf_v1::Request::GetSurfaceFeedback { id, .. } => {
                let fb = di.init(id, ());
                send_dmabuf_feedback(&fb);
            }
            zwp_linux_dmabuf_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpLinuxDmabufFeedbackV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpLinuxDmabufFeedbackV1,
        _req: wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_feedback_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // Only request from client is Destroy — no-op.
    }
}

// ─── wl_drm (legacy Mesa NVIDIA-compat protocol) ────────────────────────
//
// Why this exists: see docs in `wl_drm_proto.rs` and the comment at the
// global advertisement site above. NVIDIA's libnvidia-egl-wayland v1.1.9
// uses `wl_drm` to discover the compositor's DRM device; without it,
// `eglGetDisplay(wayland_display)` returns EGL_NO_DISPLAY and clients
// fall back to Mesa llvmpipe (software).

/// Resolve the path of the GPU render node we want clients to use. Same
/// path the dmabuf-feedback path uses; this string is what gets sent on
/// the `device` event when a client binds the `wl_drm` global.
fn wl_drm_device_path() -> String {
    for path in &[
        "/dev/dri/renderD128",
        "/dev/dri/renderD129",
        "/dev/dri/renderD130",
        "/dev/dri/renderD134",
    ] {
        if std::path::Path::new(path).exists() {
            return (*path).to_string();
        }
    }
    // Fallback: card0 (primary DRM node). Some NVIDIA clients still
    // accept it and use prime to open the matching render node.
    "/dev/dri/card0".to_string()
}

impl GlobalDispatch<crate::wl_drm_proto::wl_drm::WlDrm, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<crate::wl_drm_proto::wl_drm::WlDrm>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        use crate::wl_drm_proto::wl_drm::{Capability as WlDrmCapability, Format as WlDrmFormat};
        let drm = di.init(r, ());

        // Send device path first — NVIDIA's libnvidia-egl-wayland reads
        // this synchronously after bind to know which DRM device to open.
        drm.device(wl_drm_device_path());

        // Advertise the same fourcc formats we declare in dmabuf
        // feedback: ARGB8888 + XRGB8888 (BGRA / BGRX in DRM little-endian).
        // NVIDIA EGL filters its allocation against this list.
        drm.format(WlDrmFormat::Argb8888 as u32);
        drm.format(WlDrmFormat::Xrgb8888 as u32);

        // PRIME (fd-based buffer transfer) is the modern path. Setting
        // this tells the client to use `create_prime_buffer` rather than
        // the legacy flink-name `create_buffer`. We don't implement
        // either; the bit just changes which code path the client takes
        // when it eventually allocates. By the time it tries, it'll have
        // used `eglGetDisplay` successfully (which is all we need here)
        // and will be allocating via `zwp_linux_dmabuf_v1`.
        drm.capabilities(WlDrmCapability::Prime as u32);
    }
}

impl Dispatch<crate::wl_drm_proto::wl_drm::WlDrm, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        resource: &crate::wl_drm_proto::wl_drm::WlDrm,
        request: crate::wl_drm_proto::wl_drm::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        use crate::wl_drm_proto::wl_drm::{self as wl_drm_iface, Request as WlDrmRequest};
        match request {
            WlDrmRequest::Authenticate { .. } => {
                // DRM master authentication is unused for render-node
                // clients. Reply success immediately so the client can
                // proceed. The `authenticated` event has no args.
                resource.authenticated();
            }
            WlDrmRequest::CreateBuffer { id, .. } | WlDrmRequest::CreatePlanarBuffer { id, .. } => {
                // Legacy flink-name buffer creation paths. Modern NVIDIA
                // EGL doesn't use these (sets the PRIME capability bit
                // we advertise on bind, which steers it to
                // `create_prime_buffer` or to the `zwp_linux_dmabuf_v1`
                // path). We initialize the buffer resource so the wire
                // protocol stays consistent but the buffer is dead — any
                // commit using it would fail downstream.
                let _ = di.init(id, crate::buffer::BufferKind::Invalid);
                resource.post_error(
                    wl_drm_iface::Error::InvalidName as u32,
                    "wl_drm flink-name buffer creation is not supported; \
                     use zwp_linux_dmabuf_v1.create_immed or \
                     wl_drm.create_prime_buffer instead"
                        .to_string(),
                );
            }
            WlDrmRequest::CreatePrimeBuffer {
                id,
                name: _fd,
                width: _w,
                height: _h,
                format: _f,
                ..
            } => {
                // create_prime_buffer is the modern wl_drm path: client
                // passes a prime fd directly. We could route this to the
                // same dmabuf-import logic the `zwp_linux_dmabuf_v1`
                // params.create_immed handler uses — but in practice
                // NVIDIA EGL drivers prefer to allocate via the modern
                // dmabuf protocol once they've gotten past
                // `eglGetDisplay`. Returning a dead buffer here is fine
                // because the client should never get this far unless it
                // really wants to use wl_drm exclusively.
                let _ = di.init(id, crate::buffer::BufferKind::Invalid);
                resource.post_error(
                    wl_drm_iface::Error::InvalidName as u32,
                    "wl_drm.create_prime_buffer not implemented; clients \
                     should use zwp_linux_dmabuf_v1.create_immed instead"
                        .to_string(),
                );
            }
        }
    }
}

// ─── wp_single_pixel_buffer_manager_v1 ──────────────────────────────────

impl GlobalDispatch<WpSinglePixelBufferManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WpSinglePixelBufferManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<WpSinglePixelBufferManagerV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpSinglePixelBufferManagerV1,
        request: wp_single_pixel_buffer_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_single_pixel_buffer_manager_v1::Request::CreateU32RgbaBuffer { id, r, g, b, a } => {
                // Convert 32-bit pre-multiplied channels to 8-bit ARGB byte order [B,G,R,A].
                let b8 = (b >> 24) as u8;
                let g8 = (g >> 24) as u8;
                let r8 = (r >> 24) as u8;
                let a8 = (a >> 24) as u8;
                di.init(id, BufferKind::SinglePixel([b8, g8, r8, a8]));
            }
            wp_single_pixel_buffer_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpLinuxBufferParamsV1, ParamsData> for Compositor {
    fn request(
        s: &mut Self,
        c: &Client,
        params: &ZwpLinuxBufferParamsV1,
        request: zwp_linux_buffer_params_v1::Request,
        data: &ParamsData,
        dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        use zwp_linux_buffer_params_v1::{Error, Request};
        // Negotiated virtual-output size — the upper bound for an accepted
        // dmabuf (audit M6 / #14 defense-in-depth). Snapshot once per request.
        let (out_w, out_h, _scale) = s.state.snapshot();
        match request {
            Request::Add {
                fd,
                plane_idx,
                offset,
                stride,
                modifier_hi,
                modifier_lo,
            } => {
                let modifier = ((modifier_hi as u64) << 32) | (modifier_lo as u64);
                let mut state = data.lock().unwrap();
                if state.used {
                    params.post_error(Error::AlreadyUsed as u32, "params already used");
                    return;
                }
                state.planes.push((
                    plane_idx,
                    Plane {
                        fd,
                        offset,
                        stride,
                        modifier,
                    },
                ));
            }
            Request::Create {
                width,
                height,
                format,
                flags: _,
            } => {
                // `create` is async: on success the SERVER creates the
                // wl_buffer and emits `created(buffer)`; on failure it emits
                // `failed`. Chromium (and other clients) use this async path
                // rather than `create_immed`, so it must actually produce a
                // buffer, otherwise those clients render nothing the recorder
                // or viewer can capture (the surface stays Invalid/blank).
                match finalize_params(data, width, height, format, out_w, out_h) {
                    Ok(buf_data) => {
                        tracing::debug!(
                            w = width,
                            h = height,
                            format = format!("{:#x}", format),
                            modifier = format!("{:#x}", buf_data.modifier),
                            "dmabuf: create (async) ok"
                        );
                        match c.create_resource::<WlBuffer, BufferKind, Compositor>(
                            dh,
                            1,
                            BufferKind::Dmabuf(std::sync::Arc::new(buf_data)),
                        ) {
                            Ok(buffer) => params.created(&buffer),
                            Err(e) => {
                                tracing::warn!(error = ?e, "dmabuf: create_resource failed for async create");
                                params.failed();
                            }
                        }
                    }
                    Err(msg) => {
                        tracing::debug!(error = %msg, "dmabuf: create (async) rejected");
                        params.failed();
                    }
                }
            }
            Request::CreateImmed {
                buffer_id,
                width,
                height,
                format,
                flags: _,
            } => {
                match finalize_params(data, width, height, format, out_w, out_h) {
                    Ok(buf_data) => {
                        tracing::debug!(
                            w = width,
                            h = height,
                            format = format!("{:#x}", format),
                            modifier = format!("{:#x}", buf_data.modifier),
                            "dmabuf: create_immed ok"
                        );
                        di.init(buffer_id, BufferKind::Dmabuf(std::sync::Arc::new(buf_data)));
                    }
                    Err(msg) => {
                        tracing::debug!(error = %msg, "dmabuf: create_immed rejected — using Invalid buffer");
                        // We MUST init the new_id. Init as Invalid so the
                        // capture path silently skips this buffer (renders
                        // transparent). We intentionally do NOT post_error:
                        // a protocol error is fatal and kills the client
                        // connection. Clients that submit an unsupported
                        // modifier (e.g. GPU-tiled formats on AMD) should see
                        // a blank surface and fall back to shm, not crash.
                        di.init(buffer_id, BufferKind::Invalid);
                    }
                }
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

/// Validate an in-flight params object and produce a `DmabufBufferData`.
/// Errors are returned as strings suitable for the protocol error message.
fn finalize_params(
    data: &ParamsData,
    width: i32,
    height: i32,
    format: u32,
    // Negotiated output bounds (from `State::snapshot`). A client-declared
    // dmabuf whose dimensions exceed the output is rejected here as
    // defense-in-depth for audit M6 / #14: the declared `width`/`height` flow
    // through `DmabufBufferData` into the recorder's CSC kernel launch, and the
    // encoder's NV12 output is allocated for the output size. Bounding here
    // stops an oversized buffer before it ever reaches the encoder. `0` (either
    // bound) means "output size not yet known"; the bound is skipped in that
    // case and the encoder-side guard remains the hard backstop.
    max_w: u32,
    max_h: u32,
) -> Result<DmabufBufferData, String> {
    let mut state = data.lock().unwrap();
    if state.used {
        return Err("params already used".into());
    }
    state.used = true;
    if width <= 0 || height <= 0 {
        return Err(format!("bad dimensions {width}x{height}"));
    }
    // Clients legitimately commit buffers a bit larger than the output: e.g.
    // chromium's client-side-decoration shadow makes a 1280x720 window a
    // 1312x762 buffer. Rejecting on the exact output size marked those buffers
    // Invalid and broke capture for chromium/electron. Use a sane absolute cap
    // instead: it still stops the M6/#14 oversized-dmabuf heap overflow (a
    // malicious 100000x100000), and the encoder clamps to the output region at
    // capture time. `max_w`/`max_h` are unused now but kept in the signature
    // as the encoder-side guard's reference bound.
    const MAX_DMABUF_DIM: i32 = 16384;
    let _ = (max_w, max_h);
    if width > MAX_DMABUF_DIM || height > MAX_DMABUF_DIM {
        return Err(format!(
            "dmabuf dimensions {width}x{height} exceed max {MAX_DMABUF_DIM}x{MAX_DMABUF_DIM}"
        ));
    }
    if format != DRM_FORMAT_ARGB8888 && format != DRM_FORMAT_XRGB8888 {
        return Err(format!("unsupported format {format:#x}"));
    }
    // Sort planes by index and collect them all. We accept any modifier
    // (including AMD DCC multi-plane) — the EGL readback path handles
    // non-LINEAR modifiers at capture time.
    let mut planes = std::mem::take(&mut state.planes);
    planes.sort_by_key(|(idx, _)| *idx);
    if planes.is_empty() || planes[0].0 != 0 {
        return Err("no plane 0 supplied".into());
    }
    let sorted: Vec<Plane> = planes.into_iter().map(|(_, p)| p).collect();
    Ok(DmabufBufferData::new(sorted, width, height, format))
}

// ─── wp_viewporter ──────────────────────────────────────────────────────

impl GlobalDispatch<WpViewporter, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WpViewporter>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<WpViewporter, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpViewporter,
        request: wp_viewporter::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_viewporter::Request::GetViewport { id, surface } => {
                di.init(id, surface);
            }
            wp_viewporter::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpViewport, WlSurface> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpViewport,
        request: wp_viewport::Request,
        surface: &WlSurface,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        let Some(sd) = surface.data::<SurfaceData>() else {
            return;
        };
        match request {
            wp_viewport::Request::SetSource {
                x,
                y,
                width,
                height,
            } => {
                let mut vp = sd.viewport.lock().unwrap();
                // x, y, width, height are f64 (wayland-scanner maps wl_fixed → f64)
                if x < 0.0 {
                    vp.src = None;
                } else {
                    vp.src = Some((x, y, width, height));
                }
            }
            wp_viewport::Request::SetDestination { width, height } => {
                let mut vp = sd.viewport.lock().unwrap();
                if width < 0 {
                    vp.dst = None;
                } else {
                    vp.dst = Some((width, height));
                }
            }
            wp_viewport::Request::Destroy => {
                *sd.viewport.lock().unwrap() = ViewportData::default();
            }
            _ => {}
        }
    }
}

// ─── zwp_keyboard_shortcuts_inhibit_v1 ─────────────────────────────────

impl GlobalDispatch<ZwpKeyboardShortcutsInhibitManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwpKeyboardShortcutsInhibitManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZwpKeyboardShortcutsInhibitManagerV1, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &ZwpKeyboardShortcutsInhibitManagerV1,
        request: zwp_keyboard_shortcuts_inhibit_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_keyboard_shortcuts_inhibit_manager_v1::Request::InhibitShortcuts {
                id,
                surface,
                ..
            } => {
                if let Some(sd) = surface.data::<SurfaceData>() {
                    *sd.shortcuts_inhibited.lock().unwrap() = true;
                    state.state.set_shortcuts_inhibited(true);
                }
                let inhibitor = di.init(id, surface);
                inhibitor.active();
            }
            zwp_keyboard_shortcuts_inhibit_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpKeyboardShortcutsInhibitorV1, WlSurface> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpKeyboardShortcutsInhibitorV1,
        _req: zwp_keyboard_shortcuts_inhibitor_v1::Request,
        _surface: &WlSurface,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }

    fn destroyed(
        state: &mut Self,
        _c: ClientId,
        _r: &ZwpKeyboardShortcutsInhibitorV1,
        surface: &WlSurface,
    ) {
        if let Some(sd) = surface.data::<SurfaceData>() {
            *sd.shortcuts_inhibited.lock().unwrap() = false;
        }
        state.state.set_shortcuts_inhibited(false);
    }
}

// ─── wp_presentation ────────────────────────────────────────────────────

impl GlobalDispatch<WpPresentation, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WpPresentation>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        let pres = di.init(r, ());
        // CLOCK_MONOTONIC (1). drain_presentation_feedbacks now uses
        // clock_gettime(CLOCK_MONOTONIC) so timestamps match KWin's
        // steady_clock-based nextPaintDelay(). Earlier we sent CLOCK_REALTIME
        // timestamps (~1.78B s since epoch) which caused KWin's render timer
        // to fire ~56 years in the future, permanently stalling it.
        pres.clock_id(1); // CLOCK_MONOTONIC
    }
}

impl Dispatch<WpPresentation, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpPresentation,
        request: wp_presentation::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_presentation::Request::Feedback { surface, callback } => {
                let feedback = di.init(callback, ());
                if let Some(sd) = surface.data::<SurfaceData>() {
                    sd.pending_feedbacks.lock().unwrap().push(feedback);
                }
            }
            wp_presentation::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpPresentationFeedback, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpPresentationFeedback,
        _req: wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── zxdg_decoration_v1 ─────────────────────────────────────────────────

impl GlobalDispatch<ZxdgDecorationManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZxdgDecorationManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZxdgDecorationManagerV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZxdgDecorationManagerV1,
        request: zxdg_decoration_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zxdg_decoration_manager_v1::Request::GetToplevelDecoration { id, .. } => {
                let deco = di.init(id, ());
                deco.configure(Mode::ClientSide);
            }
            zxdg_decoration_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZxdgToplevelDecorationV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZxdgToplevelDecorationV1,
        _req: zxdg_toplevel_decoration_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── wp_fractional_scale_v1 ─────────────────────────────────────────────

impl GlobalDispatch<WpFractionalScaleManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WpFractionalScaleManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<WpFractionalScaleManagerV1, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &WpFractionalScaleManagerV1,
        request: wp_fractional_scale_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_fractional_scale_manager_v1::Request::GetFractionalScale { id, .. } => {
                let scale_obj = di.init(id, ());
                let (_, _, scale) = state.state.snapshot();
                scale_obj.preferred_scale(scale * 120);
            }
            wp_fractional_scale_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpFractionalScaleV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpFractionalScaleV1,
        _req: wp_fractional_scale_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── zwp_primary_selection_v1 ────────────────────────────────────────────

impl GlobalDispatch<ZwpPrimarySelectionDeviceManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwpPrimarySelectionDeviceManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZwpPrimarySelectionDeviceManagerV1, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &ZwpPrimarySelectionDeviceManagerV1,
        request: zwp_primary_selection_device_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_primary_selection_device_manager_v1::Request::CreateSource { id } => {
                di.init(id, Arc::new(Mutex::new(Vec::<String>::new())));
            }
            zwp_primary_selection_device_manager_v1::Request::GetDevice { id, .. } => {
                let dev = di.init(id, ());
                state.state.register_primary_device(dev);
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwpPrimarySelectionDeviceV1, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        resource: &ZwpPrimarySelectionDeviceV1,
        request: zwp_primary_selection_device_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_primary_selection_device_v1::Request::SetSelection { source, .. } => {
                if let Some(src) = source {
                    if let Some(mime_list) = src.data::<Arc<Mutex<Vec<String>>>>() {
                        let types = mime_list.lock().unwrap().clone();
                        *state.state.primary_selection.lock().unwrap() = Some(types);
                    }
                } else {
                    *state.state.primary_selection.lock().unwrap() = None;
                }
            }
            zwp_primary_selection_device_v1::Request::Destroy => {
                state.state.unregister_primary_device(resource);
            }
            _ => {}
        }
    }

    fn destroyed(state: &mut Self, _c: ClientId, resource: &ZwpPrimarySelectionDeviceV1, _d: &()) {
        state.state.unregister_primary_device(resource);
    }
}

impl Dispatch<ZwpPrimarySelectionSourceV1, Arc<Mutex<Vec<String>>>> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpPrimarySelectionSourceV1,
        request: zwp_primary_selection_source_v1::Request,
        mime_types: &Arc<Mutex<Vec<String>>>,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        if let zwp_primary_selection_source_v1::Request::Offer { mime_type } = request {
            mime_types.lock().unwrap().push(mime_type);
        }
    }
}

impl Dispatch<ZwpPrimarySelectionOfferV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpPrimarySelectionOfferV1,
        _req: zwp_primary_selection_offer_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── zwp_pointer_constraints_v1 ─────────────────────────────────────────

impl GlobalDispatch<ZwpPointerConstraintsV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwpPointerConstraintsV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZwpPointerConstraintsV1, ()> for Compositor {
    fn request(
        s: &mut Self,
        _c: &Client,
        _r: &ZwpPointerConstraintsV1,
        request: zwp_pointer_constraints_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_pointer_constraints_v1::Request::LockPointer { id, .. } => {
                let new_count = s
                    .state
                    .pointer_lock_count
                    .load(std::sync::atomic::Ordering::Acquire)
                    + 1;
                tracing::info!(
                    pointer_lock_count = new_count,
                    "inner client requested pointer lock"
                );
                let lock = di.init(id, ());
                // Optimistically grant the lock immediately — the outer_view will
                // request the real lock from the host compositor and start
                // forwarding relative motion once the host confirms.
                lock.locked();
                s.state.inc_pointer_lock_count();
            }
            zwp_pointer_constraints_v1::Request::ConfinePointer { id, .. } => {
                let confine = di.init(id, ());
                confine.confined();
            }
            zwp_pointer_constraints_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpLockedPointerV1, ()> for Compositor {
    fn request(
        s: &mut Self,
        _c: &Client,
        _r: &ZwpLockedPointerV1,
        req: zwp_locked_pointer_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        if let zwp_locked_pointer_v1::Request::Destroy = req {
            let prev = s
                .state
                .pointer_lock_count
                .load(std::sync::atomic::Ordering::Acquire);
            tracing::info!(
                pointer_lock_count_before = prev,
                "inner client destroyed pointer lock"
            );
            s.state.dec_pointer_lock_count();
        }
    }
}

impl Dispatch<ZwpConfinedPointerV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpConfinedPointerV1,
        _req: zwp_confined_pointer_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── zwp_relative_pointer_v1 ────────────────────────────────────────────

impl GlobalDispatch<ZwpRelativePointerManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwpRelativePointerManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZwpRelativePointerManagerV1, ()> for Compositor {
    fn request(
        s: &mut Self,
        _c: &Client,
        _r: &ZwpRelativePointerManagerV1,
        request: zwp_relative_pointer_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_relative_pointer_manager_v1::Request::GetRelativePointer { id, .. } => {
                tracing::info!("inner client registered relative pointer (Right Ctrl grab active)");
                let ptr = di.init(id, ());
                // Register this relative pointer object so inject_relative_pointer
                // can forward niri's relative_motion events to it. Dead objects
                // are pruned lazily inside inject_relative_pointer.
                s.state.add_relative_pointer(ptr);
            }
            zwp_relative_pointer_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpRelativePointerV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpRelativePointerV1,
        _req: zwp_relative_pointer_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── zwp_text_input_v3 ──────────────────────────────────────────────────

impl GlobalDispatch<ZwpTextInputManagerV3, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwpTextInputManagerV3>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZwpTextInputManagerV3, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwpTextInputManagerV3,
        request: zwp_text_input_manager_v3::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_text_input_manager_v3::Request::GetTextInput { id, .. } => {
                di.init(id, ());
            }
            zwp_text_input_manager_v3::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpTextInputV3, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        resource: &ZwpTextInputV3,
        request: zwp_text_input_v3::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        if let zwp_text_input_v3::Request::Commit = request {
            resource.done(0);
        }
    }
}

// ─── zwlr_layer_shell_v1 ────────────────────────────────────────────────

impl GlobalDispatch<ZwlrLayerShellV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwlrLayerShellV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZwlrLayerShellV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZwlrLayerShellV1,
        request: zwlr_layer_shell_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwlr_layer_shell_v1::Request::GetLayerSurface { id, surface, .. } => {
                // Mark the surface as a layer surface so commit can register it.
                if let Some(sd) = surface.data::<SurfaceData>() {
                    *sd.is_layer_surface.lock().unwrap() = true;
                }
                let layer_surf = di.init(id, surface);
                // Immediately send configure(serial, w=0, h=0) as required by spec.
                layer_surf.configure(next_serial(), 0, 0);
            }
            zwlr_layer_shell_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, WlSurface> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &ZwlrLayerSurfaceV1,
        request: zwlr_layer_surface_v1::Request,
        surface: &WlSurface,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        match request {
            // configuration setters — store in SurfaceData if needed, else no-op
            zwlr_layer_surface_v1::Request::SetSize { .. } => {}
            zwlr_layer_surface_v1::Request::SetAnchor { .. } => {}
            zwlr_layer_surface_v1::Request::SetExclusiveZone { .. } => {}
            zwlr_layer_surface_v1::Request::SetMargin { .. } => {}
            zwlr_layer_surface_v1::Request::SetKeyboardInteractivity {
                keyboard_interactivity,
            } => {
                // Wire format is `type="uint" enum="keyboard_interactivity"`;
                // wayland-rs decodes it as `WEnum<KeyboardInteractivity>`. We
                // round-trip through the raw u32 wire value for storage
                // simplicity — the consumer at compositor.rs ~960-1000 just
                // tests `== 1` for the exclusive case. Unknown variants are
                // forwarded as-is rather than clamped, so a future protocol
                // bump that adds variant 3 keeps the field truthful.
                if let Some(sd) = surface.data::<SurfaceData>() {
                    let wire = match keyboard_interactivity {
                        WEnum::Value(v) => v as u32,
                        WEnum::Unknown(v) => v,
                    };
                    *sd.layer_keyboard_interactivity.lock().unwrap() = wire;
                }
            }
            zwlr_layer_surface_v1::Request::SetLayer { .. } => {}
            zwlr_layer_surface_v1::Request::SetExclusiveEdge { .. } => {}
            zwlr_layer_surface_v1::Request::AckConfigure { .. } => {}
            zwlr_layer_surface_v1::Request::GetPopup { .. } => {}
            zwlr_layer_surface_v1::Request::Destroy => {
                if let Some(sd) = surface.data::<SurfaceData>() {
                    let wid = sd.layer_window_id.lock().unwrap().take();
                    if let Some(window_id) = wid {
                        state.state.unregister_surface(window_id);
                        if state.state.remove_window(window_id) {
                            info!(window_id, "layer_surface destroyed");
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn destroyed(state: &mut Self, _c: ClientId, _r: &ZwlrLayerSurfaceV1, surface: &WlSurface) {
        if let Some(sd) = surface.data::<SurfaceData>() {
            let wid = sd.layer_window_id.lock().unwrap().take();
            if let Some(window_id) = wid {
                state.state.unregister_surface(window_id);
                if state.state.remove_window(window_id) {
                    info!(window_id, "layer_surface destroyed (client disconnect)");
                }
            }
        }
    }
}

// ─── zxdg_output_manager_v1 ─────────────────────────────────────────────

impl GlobalDispatch<ZxdgOutputManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZxdgOutputManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZxdgOutputManagerV1, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &ZxdgOutputManagerV1,
        request: zxdg_output_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zxdg_output_manager_v1::Request::GetXdgOutput { id, .. } => {
                let xdg_out = di.init(id, ());
                let (w, h, _) = state.state.snapshot();
                xdg_out.logical_position(0, 0);
                xdg_out.logical_size(w as i32, h as i32);
                if xdg_out.version() >= 2 {
                    xdg_out.name("waymux".into());
                    xdg_out.description("waymux virtual output".into());
                }
                if xdg_out.version() >= 3 {
                    xdg_out.done();
                }
            }
            zxdg_output_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZxdgOutputV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZxdgOutputV1,
        _req: zxdg_output_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── xdg_activation_v1 ──────────────────────────────────────────────────

impl GlobalDispatch<XdgActivationV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<XdgActivationV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<XdgActivationV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &XdgActivationV1,
        request: xdg_activation_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_activation_v1::Request::GetActivationToken { id } => {
                di.init(id, ());
            }
            xdg_activation_v1::Request::Activate { .. } => {
                // no-op: accept and ignore activation requests
            }
            xdg_activation_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<XdgActivationTokenV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        resource: &XdgActivationTokenV1,
        request: xdg_activation_token_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // set_serial, set_app_id, set_surface, destroy: no-op
        if let xdg_activation_token_v1::Request::Commit = request {
            resource.done("waymux-token".into());
        }
    }
}

// ─── ext_idle_notify_v1 ─────────────────────────────────────────────────

impl GlobalDispatch<ExtIdleNotifierV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ExtIdleNotifierV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ExtIdleNotifierV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ExtIdleNotifierV1,
        request: ext_idle_notifier_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            ext_idle_notifier_v1::Request::GetIdleNotification { id, .. } => {
                di.init(id, ());
            }
            ext_idle_notifier_v1::Request::GetInputIdleNotification { id, .. } => {
                di.init(id, ());
            }
            ext_idle_notifier_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ExtIdleNotificationV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ExtIdleNotificationV1,
        _req: ext_idle_notification_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // destroy — no-op. Never send `idled` or `resumed` (always active).
    }
}

// ─── wp_alpha_modifier_v1 ────────────────────────────────────────────────

impl GlobalDispatch<WpAlphaModifierV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<WpAlphaModifierV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<WpAlphaModifierV1, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpAlphaModifierV1,
        request: wp_alpha_modifier_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_alpha_modifier_v1::Request::GetSurface { id, surface } => {
                di.init(id, surface);
            }
            wp_alpha_modifier_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpAlphaModifierSurfaceV1, WlSurface> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &WpAlphaModifierSurfaceV1,
        _req: wp_alpha_modifier_surface_v1::Request,
        _surface: &WlSurface,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // set_multiplier, destroy — no-op.
    }
}

// ─── zxdg_foreign_v2 ────────────────────────────────────────────────────

impl GlobalDispatch<ZxdgExporterV2, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZxdgExporterV2>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl GlobalDispatch<ZxdgImporterV2, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZxdgImporterV2>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZxdgExporterV2, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZxdgExporterV2,
        request: zxdg_exporter_v2::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zxdg_exporter_v2::Request::ExportToplevel { id, .. } => {
                let exported = di.init(id, ());
                exported.handle("waymux-handle".into());
            }
            zxdg_exporter_v2::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZxdgImporterV2, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZxdgImporterV2,
        request: zxdg_importer_v2::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zxdg_importer_v2::Request::ImportToplevel { id, .. } => {
                di.init(id, ());
            }
            zxdg_importer_v2::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZxdgExportedV2, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZxdgExportedV2,
        _req: zxdg_exported_v2::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // destroy — no-op.
    }
}

impl Dispatch<ZxdgImportedV2, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &ZxdgImportedV2,
        _req: zxdg_imported_v2::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // set_parent_of, destroy — no-op.
    }
}

// ─── org_kde_plasma_window_management ──────────────────────────────────
//
// Lets plasmashell's taskbar enumerate xdg_toplevels in the session.
// Window UUIDs are just the stringified internal window_id.

/// User data attached to OrgKdePlasmaWindow resources so destroy/dispatch
/// know which window the resource is bound to.
pub struct PlasmaWindowData {
    pub window_id: u32,
}

impl GlobalDispatch<OrgKdePlasmaWindowManagement, ()> for Compositor {
    fn bind(
        state: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<OrgKdePlasmaWindowManagement>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        let mgr = di.init(r, ());
        // Initial state: desktop is not in show-desktop mode.
        mgr.show_desktop_changed(org_kde_plasma_window_management::ShowDesktop::Disabled as u32);
        // Replay existing windows so a late-binding plasmashell sees them.
        for wid in state.state.window_ids() {
            let uuid = format!("{wid}");
            mgr.window(wid);
            if mgr.version() >= 13 {
                mgr.window_with_uuid(wid, uuid);
            }
        }
        if mgr.version() >= 11 {
            mgr.stacking_order_changed(Vec::new());
        }
        if mgr.version() >= 12 {
            mgr.stacking_order_uuid_changed(String::new());
        }
        state.state.register_plasma_manager(mgr);
    }
}

impl Dispatch<OrgKdePlasmaWindowManagement, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &OrgKdePlasmaWindowManagement,
        request: org_kde_plasma_window_management::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            org_kde_plasma_window_management::Request::ShowDesktop { .. } => {
                // We don't have a desktop to show or hide.
            }
            org_kde_plasma_window_management::Request::GetWindow {
                id,
                internal_window_id,
            } => {
                let win = di.init(
                    id,
                    PlasmaWindowData {
                        window_id: internal_window_id,
                    },
                );
                send_plasma_window_initial_state(state, &win, internal_window_id);
            }
            org_kde_plasma_window_management::Request::GetWindowByUuid {
                id,
                internal_window_uuid,
            } => {
                let wid: u32 = internal_window_uuid.parse().unwrap_or(0);
                let win = di.init(id, PlasmaWindowData { window_id: wid });
                send_plasma_window_initial_state(state, &win, wid);
            }
            org_kde_plasma_window_management::Request::GetStackingOrder { stacking_order } => {
                di.init(stacking_order, ());
            }
            _ => {}
        }
    }
}

fn send_plasma_window_initial_state(
    state: &mut Compositor,
    win: &OrgKdePlasmaWindow,
    window_id: u32,
) {
    let info_opt = state
        .state
        .windows()
        .into_iter()
        .find(|w| w.id == window_id);
    if let Some(info) = info_opt {
        if !info.title.is_empty() {
            win.title_changed(info.title.clone());
        }
        if !info.app_id.is_empty() {
            win.app_id_changed(info.app_id.clone());
        }
        if win.version() >= 4 {
            win.pid_changed(info.pid as u32);
        }
        if win.version() >= 6 {
            win.geometry(
                info.geometry.x,
                info.geometry.y,
                info.geometry.width,
                info.geometry.height,
            );
        }
        // Minimal flag set — closeable so plasmashell shows a close button.
        let flags = (org_kde_plasma_window_management::State::Closeable as u32)
            | (org_kde_plasma_window_management::State::Maximizable as u32);
        win.state_changed(flags);
        if win.version() >= 4 {
            win.initial_state();
        }
    } else {
        // Unknown window — still send initial_state so the client doesn't hang.
        win.unmapped();
        if win.version() >= 4 {
            win.initial_state();
        }
    }
    state.state.register_plasma_window(window_id, win.clone());
}

impl Dispatch<OrgKdePlasmaWindow, PlasmaWindowData> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &OrgKdePlasmaWindow,
        _request: org_kde_plasma_window::Request,
        _d: &PlasmaWindowData,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        // We don't honour state-change/move/close requests; the host KWin
        // (running as a waymux client) is the actual window manager.
    }

    fn destroyed(state: &mut Self, _c: ClientId, r: &OrgKdePlasmaWindow, data: &PlasmaWindowData) {
        state.state.unregister_plasma_window(data.window_id, r);
    }
}

impl Dispatch<OrgKdePlasmaStackingOrder, ()> for Compositor {
    fn request(
        _s: &mut Self,
        _c: &Client,
        _r: &OrgKdePlasmaStackingOrder,
        _request: org_kde_plasma_stacking_order::Request,
        _d: &(),
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
    }
}

// Activation feedback objects can be created if a future client extends the
// protocol; we don't currently emit them, but the dispatch impls keep us
// compatible if added.
impl Dispatch<wayland_protocols_plasma::plasma_window_management::server::org_kde_plasma_activation_feedback::OrgKdePlasmaActivationFeedback, ()> for Compositor {
    fn request(
        _s: &mut Self, _c: &Client,
        _r: &wayland_protocols_plasma::plasma_window_management::server::org_kde_plasma_activation_feedback::OrgKdePlasmaActivationFeedback,
        _req: org_kde_plasma_activation_feedback::Request,
        _d: &(), _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {}
}

impl Dispatch<wayland_protocols_plasma::plasma_window_management::server::org_kde_plasma_activation::OrgKdePlasmaActivation, ()> for Compositor {
    fn request(
        _s: &mut Self, _c: &Client,
        _r: &wayland_protocols_plasma::plasma_window_management::server::org_kde_plasma_activation::OrgKdePlasmaActivation,
        _req: org_kde_plasma_activation::Request,
        _d: &(), _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {}
}

// ─── zwlr_screencopy_v1 ─────────────────────────────────────────────────
//
// Lets in-session capture tools (grim, wayshot) read the desktop. We
// composite all windows into a session-sized buffer and copy it into the
// client's wl_shm buffer.

/// Per-frame state carried as the wl_screencopy_frame_v1 user data.
pub struct ScreencopyFrameData {
    pub width: i32,
    pub height: i32,
    pub stride: i32,
    /// Once `copy` has run, the frame is single-use (protocol enforced).
    pub used: Mutex<bool>,
}

impl GlobalDispatch<ZwlrScreencopyManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        _c: &Client,
        r: New<ZwlrScreencopyManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        di.init(r, ());
    }
}

impl Dispatch<ZwlrScreencopyManagerV1, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        _r: &ZwlrScreencopyManagerV1,
        request: zwlr_screencopy_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwlr_screencopy_manager_v1::Request::CaptureOutput {
                frame,
                output: _,
                overlay_cursor: _,
            } => {
                init_screencopy_frame(state, frame, di, None);
            }
            zwlr_screencopy_manager_v1::Request::CaptureOutputRegion {
                frame,
                output: _,
                overlay_cursor: _,
                x,
                y,
                width,
                height,
            } => {
                init_screencopy_frame(state, frame, di, Some((x, y, width, height)));
            }
            zwlr_screencopy_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

fn init_screencopy_frame(
    state: &mut Compositor,
    frame_id: New<ZwlrScreencopyFrameV1>,
    di: &mut DataInit<'_, Compositor>,
    region: Option<(i32, i32, i32, i32)>,
) {
    let (sw, sh, _) = state.state.snapshot();
    let (w, h) = match region {
        Some((_, _, rw, rh)) if rw > 0 && rh > 0 => (rw.min(sw as i32), rh.min(sh as i32)),
        _ => (sw as i32, sh as i32),
    };
    let stride = w * 4;
    let frame = di.init(
        frame_id,
        ScreencopyFrameData {
            width: w,
            height: h,
            stride,
            used: Mutex::new(false),
        },
    );
    // Argb8888 — same byte layout we composite into.
    frame.buffer(wl_shm::Format::Argb8888, w as u32, h as u32, stride as u32);
    if frame.version() >= 3 {
        // wl_shm format also accepted via linux_dmabuf path — we don't
        // implement dmabuf imports for screencopy yet.
        frame.buffer_done();
    }
}

impl Dispatch<ZwlrScreencopyFrameV1, ScreencopyFrameData> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        resource: &ZwlrScreencopyFrameV1,
        request: zwlr_screencopy_frame_v1::Request,
        data: &ScreencopyFrameData,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwlr_screencopy_frame_v1::Request::Copy { buffer }
            | zwlr_screencopy_frame_v1::Request::CopyWithDamage { buffer } => {
                let mut used = data.used.lock().unwrap();
                if *used {
                    resource.post_error(
                        zwlr_screencopy_frame_v1::Error::AlreadyUsed,
                        "frame already used".to_string(),
                    );
                    return;
                }
                *used = true;
                drop(used);

                // The buffer must be wl_shm-backed of matching geometry/format.
                let kind = buffer.data::<crate::buffer::BufferKind>();
                let shm = match kind {
                    Some(crate::buffer::BufferKind::Shm(s)) => s,
                    _ => {
                        resource.failed();
                        return;
                    }
                };
                if shm.width != data.width
                    || shm.height != data.height
                    || shm.stride != data.stride
                    || !matches!(
                        shm.format,
                        WEnum::Value(wl_shm::Format::Argb8888)
                            | WEnum::Value(wl_shm::Format::Xrgb8888)
                    )
                {
                    resource.failed();
                    return;
                }

                let pixels = match state.state.capture_desktop() {
                    Some((bytes, w, h, _, _)) if w == data.width && h == data.height => bytes,
                    _ => {
                        resource.failed();
                        return;
                    }
                };

                if shm.pool.write_bytes(shm.offset, &pixels).is_none() {
                    resource.failed();
                    return;
                }

                resource.flags(zwlr_screencopy_frame_v1::Flags::empty());
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let secs = now.as_secs();
                resource.ready(
                    (secs >> 32) as u32,
                    (secs & 0xFFFF_FFFF) as u32,
                    now.subsec_nanos(),
                );
            }
            zwlr_screencopy_frame_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

// ─── wp_linux_drm_syncobj_v1 ────────────────────────────────────────────
//
// Server-side explicit-sync. KWin on AMD attaches no implicit dma-buf
// fence to its rendered output, so without this protocol waymux reads
// buffers mid-render and tears (live + recorded). With it, KWin tells us
// "wait on this syncobj timeline point before reading; signal this other
// point when you're done." See syncobj.rs for the kernel-side glue.

impl GlobalDispatch<WpLinuxDrmSyncobjManagerV1, ()> for Compositor {
    fn bind(
        _s: &mut Self,
        _dh: &DisplayHandle,
        c: &Client,
        r: New<WpLinuxDrmSyncobjManagerV1>,
        _d: &(),
        di: &mut DataInit<'_, Self>,
    ) {
        info!(client = ?c.id(), "syncobj: client bound wp_linux_drm_syncobj_manager_v1");
        di.init(r, ());
    }
}

impl Dispatch<WpLinuxDrmSyncobjManagerV1, ()> for Compositor {
    fn request(
        state: &mut Self,
        _c: &Client,
        resource: &WpLinuxDrmSyncobjManagerV1,
        request: wp_linux_drm_syncobj_manager_v1::Request,
        _d: &(),
        _dh: &DisplayHandle,
        di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_linux_drm_syncobj_manager_v1::Request::Destroy => {}
            wp_linux_drm_syncobj_manager_v1::Request::GetSurface { id, surface } => {
                let sd = match surface.data::<SurfaceData>() {
                    Some(d) => d,
                    None => {
                        resource.post_error(
                            wp_linux_drm_syncobj_manager_v1::Error::SurfaceExists as u32,
                            String::from("wl_surface has no data"),
                        );
                        return;
                    }
                };
                {
                    let mut slot = sd.surface_sync.lock().unwrap();
                    if slot.is_some() {
                        resource.post_error(
                            wp_linux_drm_syncobj_manager_v1::Error::SurfaceExists as u32,
                            String::from("surface already has a sync object"),
                        );
                        return;
                    }
                    let sync = Arc::new(SurfaceSync::new());
                    *slot = Some(sync.clone());
                    di.init(id, sync);
                    info!("syncobj: get_surface — client enabled explicit-sync on a surface");
                }
            }
            wp_linux_drm_syncobj_manager_v1::Request::ImportTimeline { id, fd } => {
                let device = match state.syncobj_device.as_ref() {
                    Some(d) => d.clone(),
                    None => {
                        resource.post_error(
                            wp_linux_drm_syncobj_manager_v1::Error::InvalidTimeline as u32,
                            String::from("syncobj device unavailable"),
                        );
                        return;
                    }
                };
                use std::os::fd::AsRawFd as _;
                let raw = fd.as_raw_fd();
                tracing::info!(
                    fd_raw_value = raw,
                    "compositor: ImportTimeline received from client, about to import"
                );
                let handle = match device.import_timeline(raw) {
                    Some(h) => h,
                    None => {
                        resource.post_error(
                            wp_linux_drm_syncobj_manager_v1::Error::InvalidTimeline as u32,
                            String::from("FD_TO_HANDLE failed"),
                        );
                        return;
                    }
                };
                let timeline = Arc::new(Timeline {
                    device: device.clone(),
                    handle,
                    _fd: fd, // Keep the OwnedFd alive for the lifetime of the Timeline.
                });
                di.init(id, timeline);
            }
            _ => {}
        }
    }
}

impl Dispatch<WpLinuxDrmSyncobjTimelineV1, Arc<Timeline>> for Compositor {
    fn request(
        _state: &mut Self,
        _c: &Client,
        _resource: &WpLinuxDrmSyncobjTimelineV1,
        request: wp_linux_drm_syncobj_timeline_v1::Request,
        _data: &Arc<Timeline>,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        if let wp_linux_drm_syncobj_timeline_v1::Request::Destroy = request {
            // Arc<Timeline> drops with the resource. Per protocol,
            // points already set on a surface remain valid because
            // SurfaceSync clones the Arc.
        }
    }
}

impl Dispatch<WpLinuxDrmSyncobjSurfaceV1, Arc<SurfaceSync>> for Compositor {
    fn request(
        _state: &mut Self,
        _c: &Client,
        _resource: &WpLinuxDrmSyncobjSurfaceV1,
        request: wp_linux_drm_syncobj_surface_v1::Request,
        sync: &Arc<SurfaceSync>,
        _dh: &DisplayHandle,
        _di: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_linux_drm_syncobj_surface_v1::Request::Destroy => {}
            wp_linux_drm_syncobj_surface_v1::Request::SetAcquirePoint {
                timeline,
                point_hi,
                point_lo,
            } => {
                let tl = match timeline.data::<Arc<Timeline>>() {
                    Some(t) => t.clone(),
                    None => return,
                };
                let point = ((point_hi as u64) << 32) | (point_lo as u64);
                *sync.pending_acquire.lock().unwrap() = Some(TimelinePoint {
                    timeline: tl,
                    point,
                });
            }
            wp_linux_drm_syncobj_surface_v1::Request::SetReleasePoint {
                timeline,
                point_hi,
                point_lo,
            } => {
                let tl = match timeline.data::<Arc<Timeline>>() {
                    Some(t) => t.clone(),
                    None => return,
                };
                let point = ((point_hi as u64) << 32) | (point_lo as u64);
                *sync.pending_release.lock().unwrap() = Some(TimelinePoint {
                    timeline: tl,
                    point,
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod dmabuf_advert_tests {
    #[test]
    fn format_table_has_one_entry_per_format_modifier_pair() {
        let mods = crate::vulkan_record::importable_bgra_modifiers();
        let (_, size) = super::make_dmabuf_format_table().expect("format table");
        // 2 formats (ARGB, XRGB) * mods.len() pairs * 16 bytes each.
        assert_eq!(size as usize, 2 * mods.len() * 16);
        assert!(mods.contains(&crate::dmabuf::DRM_FORMAT_MOD_LINEAR));
    }
}

#[cfg(test)]
mod pacing_tests {
    use super::{PRESENTATION_REFRESH_PERIOD_NS, VIRTUAL_OUTPUT_REFRESH_MHZ};

    #[test]
    fn output_refresh_is_240hz() {
        // 240 Hz: a live A/B showed lowering this to 60 Hz starved KWin's frame
        // supply to the encoder (~31 vs ~45 fps) with no GPU saving. Keep high.
        assert_eq!(VIRTUAL_OUTPUT_REFRESH_MHZ, 240_000);
    }

    #[test]
    fn presentation_period_matches_output_refresh() {
        // The wp_presentation refresh PERIOD must equal 1e9 / refresh_hz, or
        // KWin's next-paint timer drifts (the exact class of bug that left the
        // period at 240 Hz while we thought the mode was 120 Hz).
        let refresh_hz = (VIRTUAL_OUTPUT_REFRESH_MHZ as u32) / 1000;
        let expected_ns = 1_000_000_000 / refresh_hz;
        let diff = (PRESENTATION_REFRESH_PERIOD_NS as i64 - expected_ns as i64).abs();
        assert!(
            diff <= 1,
            "presentation period {PRESENTATION_REFRESH_PERIOD_NS} ns must match \
             1e9/{refresh_hz}Hz = {expected_ns} ns (drift breaks KWin's render timer)"
        );
    }
}
