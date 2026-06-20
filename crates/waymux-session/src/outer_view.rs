// SPDX-License-Identifier: Apache-2.0

//! Session-side "outer view" — bridge to the outer compositor.
//!
//! Runs on a dedicated OS thread per attach. The thread owns the received
//! outer-compositor display fd and drives a standard `wayland-client`
//! event loop on it: binds `wl_compositor` / `wl_shm` / `xdg_wm_base`,
//! creates a `wl_surface` + `xdg_toplevel` on the outer compositor.
//!
//! **Frame ferry** (slice 3): on every inner `wl_surface.commit`, the
//! inner compositor calls `State::record_damage()` which pokes this
//! thread's eventfd. The thread wakes, captures the focused inner
//! window's shm bytes, memcpys them into the outer shm pool, and
//! `attach` + `damage_buffer` + `commit`s the outer surface. Works for
//! same-format ARGB8888 at the same size as the session (1:1 mapping);
//! falls back to the magenta-bordered placeholder otherwise.
//!
//! Known simplification: single shm buffer, overwrite-in-place. A
//! well-behaved outer compositor copies the buffer into its own texture
//! on commit so the protocol-strict "don't touch buffer until release"
//! rule is usually fine in practice. Double-buffering is a cleanup for
//! when this runs under a scanout-only compositor.

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use memmap2::MmapMut;
use tracing::{debug, info, warn};

use crate::input_bridge::InputBridge;
use wayland_client::{
    backend::ObjectId,
    globals::{registry_queue_init, GlobalListContents},
    protocol::{
        wl_buffer::WlBuffer,
        wl_callback::{self, WlCallback},
        wl_compositor::WlCompositor,
        wl_data_device::{self, WlDataDevice as ClientDataDevice},
        wl_data_device_manager::WlDataDeviceManager,
        wl_data_offer::{self, WlDataOffer as ClientDataOffer},
        wl_data_source::{self, WlDataSource as ClientDataSource},
        wl_keyboard::{self, WlKeyboard},
        wl_pointer::{self, WlPointer},
        wl_registry::WlRegistry,
        wl_seat::{self, WlSeat},
        wl_shm::{self, WlShm},
        wl_shm_pool::WlShmPool,
        wl_surface::WlSurface,
    },
    Connection, Dispatch, Proxy, QueueHandle, WEnum as ClientWEnum,
};
use wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::{
    zwp_keyboard_shortcuts_inhibit_manager_v1::ZwpKeyboardShortcutsInhibitManagerV1 as ClientInhibitManager,
    zwp_keyboard_shortcuts_inhibitor_v1::ZwpKeyboardShortcutsInhibitorV1 as ClientInhibitor,
};
use wayland_protocols::wp::pointer_constraints::zv1::client::{
    zwp_locked_pointer_v1::{self as locked_ptr_ev, ZwpLockedPointerV1 as ClientLockedPointer},
    zwp_pointer_constraints_v1::ZwpPointerConstraintsV1 as ClientPointerConstraints,
};
use wayland_protocols::wp::relative_pointer::zv1::client::{
    zwp_relative_pointer_manager_v1::ZwpRelativePointerManagerV1 as ClientRelativeManager,
    zwp_relative_pointer_v1::{self as rel_ptr_ev, ZwpRelativePointerV1 as ClientRelativePointer},
};
use wayland_protocols::xdg::shell::client::{
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};

use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1 as OuterDmabufParams},
    zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1 as OuterDmabuf,
};

use crate::dmabuf::DmabufBufferData;
use crate::recording::InnerBufferHold;
use crate::state::{ClipboardContent, State};
use std::sync::Mutex as StdMutex;
use wayland_server::Resource as ServerResource;

pub fn run(outer_fd: OwnedFd, state: Arc<State>, stop: Arc<AtomicBool>) -> Result<()> {
    let (w, h, _scale) = state.snapshot();
    info!(
        outer_fd = outer_fd.as_raw_fd(),
        size = format!("{w}x{h}"),
        "outer view: wrapping outer display fd"
    );

    use std::os::fd::IntoRawFd;
    let stream = unsafe { UnixStream::from_raw_fd(outer_fd.into_raw_fd()) };
    let conn = Connection::from_socket(stream).context("wrap outer fd")?;
    let (globals, mut queue) =
        registry_queue_init::<OuterState>(&conn).context("outer registry init")?;
    let qh = queue.handle();

    let compositor: WlCompositor = globals
        .bind(&qh, 1..=6, ())
        .context("outer compositor missing")?;
    let wm_base: XdgWmBase = globals
        .bind(&qh, 1..=5, ())
        .context("outer xdg_wm_base missing")?;
    let shm: WlShm = globals
        .bind(&qh, 1..=1, ())
        .context("outer wl_shm missing")?;
    // Seat is optional — if the outer compositor has no keyboard/pointer
    // (unlikely on a desktop), input forwarding is silently disabled.
    let seat: Option<WlSeat> = globals.bind(&qh, 1..=9, ()).ok();
    // Data device manager — only bound when clipboard bridging is enabled.
    let ddm: Option<WlDataDeviceManager> = if state.share_clipboard() {
        globals.bind(&qh, 1..=3, ()).ok()
    } else {
        None
    };
    let inhibit_manager: Option<ClientInhibitManager> = globals.bind(&qh, 1..=1, ()).ok();
    let outer_dmabuf: Option<OuterDmabuf> = globals.bind(&qh, 4..=4, ()).ok();
    let pointer_constraints: Option<ClientPointerConstraints> = globals.bind(&qh, 1..=1, ()).ok();
    let relative_manager: Option<ClientRelativeManager> = globals.bind(&qh, 1..=1, ()).ok();
    let data_device: Option<ClientDataDevice> = ddm
        .as_ref()
        .and_then(|ddm| seat.as_ref().map(|s| ddm.get_data_device(s, &qh, ())));

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_app_id("waymux.session".into());
    toplevel.set_title("waymux session".into());
    surface.commit();

    // Input forwarding: grab keyboard + pointer on the outer seat and
    // translate their events into inject_key / inject_pointer on the
    // inner seat. Kept alive by OuterState holding the proxies.
    let keyboard = seat.as_ref().map(|s| s.get_keyboard(&qh, ()));
    let pointer = seat.as_ref().map(|s| s.get_pointer(&qh, ()));

    let input = InputBridge::new(
        state.clone(),
        inhibit_manager,
        pointer_constraints,
        relative_manager,
    );

    let mut outer = OuterState {
        configured: false,
        toplevel_fullscreen: false,
        state: state.clone(),
        input,
        stop: stop.clone(),
        pointer_enter_serial: 0,
        session_empty: false,
        frame_callback_fired: false,
        pending_frame_cb: None,
        has_new_damage: true, // treat startup as "new damage" to render first frame
        last_damage_checked: 0,
        inner_sel_version: 0,
        _outer_source: None,
        pending_offers: HashMap::new(),
        _seat: seat,
        _keyboard: keyboard,
        _pointer: pointer,
        _data_device: data_device,
        _data_device_manager: ddm,
        outer_dmabuf,
        dmabuf_in_flight: Vec::new(),
        fps_window_start: std::time::Instant::now(),
        fps_dmabuf_count: 0,
        fps_shm_count: 0,
        fence_no_fence: 0,
        fence_signaled: 0,
        fence_waited: 0,
        fence_timed_out: 0,
        fence_wait_us_total: 0,
        fence_wait_us_max: 0,
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !outer.configured && std::time::Instant::now() < deadline {
        queue
            .blocking_dispatch(&mut outer)
            .context("blocking_dispatch during initial configure")?;
    }
    if !outer.configured {
        anyhow::bail!("outer compositor never sent configure within 5s");
    }
    info!("outer view: configure received");

    // Allocate the shm pool sized for the session. We reallocate (via
    // Framebuffer::new) when the inner commit size changes — tiling
    // compositors like niri subtract padding from the client area, so
    // foot and friends commit at something like 798×560 even though we
    // told our session "800×600".
    let mut fb = Framebuffer::new(&shm, &qh, w, h)?;
    fb.paint_placeholder();
    // Commit slot 0 as the initial placeholder; slot 1 stays unreleased until
    // we get the first wl_buffer.release so we don't overwrite too early.
    {
        let slot = &fb.slots[0];
        slot.released.store(false, Ordering::Release);
        surface.attach(Some(&slot.buffer), 0, 0);
        surface.damage_buffer(0, 0, fb.width as i32, fb.height as i32);
        surface.commit();
        fb.write_idx = 1;
    }
    info!("outer view: placeholder committed");

    // Register our wake eventfd with State so inner commits poke us.
    let wake_fd = make_eventfd()?;
    let wake_arc = Arc::new(wake_fd);
    state.set_outer_wake_fd(Some(wake_arc.clone()));

    // Main loop: block on {outer display fd, wake eventfd}. On wake from
    // inner damage, try to blit the focused window's current buffer.
    let display_fd = conn.backend().poll_fd().as_raw_fd();
    let wake_raw = wake_arc.as_raw_fd();

    while !stop.load(Ordering::Relaxed) {
        if let Err(e) = queue.flush() {
            warn!(error = %e, "outer view: flush failed; connection closed");
            break;
        }
        if let Some(guard) = conn.prepare_read() {
            let _ = guard.read();
        }
        if let Err(e) = queue.dispatch_pending(&mut outer) {
            warn!(error = %e, "outer view: dispatch_pending failed");
            break;
        }

        // Non-blocking drain: pick up any wake that fired before or during
        // dispatch. Also run damage detection now so we can decide whether to
        // skip the blocking poll entirely.
        {
            let mut buf = [0u8; 8];
            unsafe {
                let _ = libc::read(wake_raw, buf.as_mut_ptr().cast(), buf.len());
            }
        }
        {
            let current_damage = state.last_damage_ns();
            if current_damage != outer.last_damage_checked {
                outer.last_damage_checked = current_damage;
                outer.has_new_damage = true;
            }
        }

        // Skip the blocking poll when we already have everything needed to
        // commit. Without this, a frame callback + damage that both arrive
        // before poll_two would still incur a full extra poll cycle of latency
        // before the commit, causing the frame to miss its vblank window.
        //
        // With multi-buffer dmabuf tracking the old single-buffer deadlock is
        // gone, so it is safe to skip for both cases again.
        let skip_poll = outer.frame_callback_fired
            || (outer.has_new_damage && outer.pending_frame_cb.is_none());
        if !skip_poll {
            poll_two(display_fd, wake_raw, 1000);
            // Drain the wake eventfd after the blocking poll.
            let mut buf = [0u8; 8];
            unsafe {
                let _ = libc::read(wake_raw, buf.as_mut_ptr().cast(), buf.len());
            }
        }

        // ── clipboard bridge: inner → outer ──────────────────────────────
        // When the inner client sets a new selection we create an outer
        // wl_data_source and hand it to the outer compositor.
        if state.share_clipboard() {
            let cur_ver = state.inner_selection_version.load(Ordering::Acquire);
            if cur_ver != outer.inner_sel_version {
                outer.inner_sel_version = cur_ver;
                let sel = state
                    .inner_selection
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|s| (s.mime_types.clone(), s.source.clone()));
                if let (Some((mime_types, inner_src)), Some(ddm)) =
                    (sel, outer._data_device_manager.as_ref())
                {
                    let new_src: ClientDataSource = ddm.create_data_source(&qh, inner_src);
                    for mime in &mime_types {
                        new_src.offer(mime.clone());
                    }
                    if let Some(dd) = outer._data_device.as_ref() {
                        dd.set_selection(Some(&new_src), 0);
                    }
                    outer._outer_source = Some(new_src);
                } else if outer._outer_source.is_some() {
                    // Selection was cleared by inner client.
                    if let Some(dd) = outer._data_device.as_ref() {
                        dd.set_selection(None, 0);
                    }
                    outer._outer_source = None;
                }
                let _ = queue.flush();
            }
        }

        // ── shortcut inhibitor + pointer-lock state machine ──────────────────
        if outer.configured {
            let want_lock = outer.state.pointer_lock_active();
            let want_inhibit = outer.state.any_shortcuts_inhibited() || want_lock;
            if let Some(seat) = outer._seat.as_ref() {
                outer
                    .input
                    .update_inhibitor(want_inhibit, &surface, seat, &qh);
            }
            let pointer_clone = outer._pointer.clone();
            outer
                .input
                .update_lock(&surface, pointer_clone.as_ref(), &qh);

            // When the pointer lock is active, explicitly hide niri's cursor so
            // the user doesn't see a frozen cursor overlaid on the session window.
            // We call set_cursor every iteration while locked so it takes effect
            // even if niri re-sends an enter event (e.g. after a surface reconfigure).
            // When the lock is released, we stop calling set_cursor; niri will use
            // its default cursor on the next pointer enter / motion event.
            if want_lock {
                if let Some(ptr) = outer._pointer.as_ref() {
                    ptr.set_cursor(outer.pointer_enter_serial, None, 0, 0);
                }
            }
        }

        // ── Detect new inner damage ──────────────────────────────────────
        {
            let current_damage = state.last_damage_ns();
            if current_damage != outer.last_damage_checked {
                outer.last_damage_checked = current_damage;
                outer.has_new_damage = true;
            }
        }

        // ── Forwarded-fps reporter (1 s window) ──────────────────────────
        {
            let elapsed = outer.fps_window_start.elapsed();
            if elapsed >= std::time::Duration::from_secs(1) {
                let secs = elapsed.as_secs_f32();
                let dmabuf_fps = outer.fps_dmabuf_count as f32 / secs;
                let shm_fps = outer.fps_shm_count as f32 / secs;
                if outer.fps_dmabuf_count + outer.fps_shm_count > 0 {
                    let avg_wait_us = if outer.fence_waited > 0 {
                        outer.fence_wait_us_total / outer.fence_waited as u64
                    } else {
                        0
                    };
                    info!(
                        "forwarded fps: dmabuf={:.1} shm={:.1} in_flight={} pending_releases={} \
                         fence[no={} signaled={} waited={} timeout={}] \
                         wait_avg_us={} wait_max_us={}",
                        dmabuf_fps,
                        shm_fps,
                        outer.dmabuf_in_flight.len(),
                        state.pending_release_count(),
                        outer.fence_no_fence,
                        outer.fence_signaled,
                        outer.fence_waited,
                        outer.fence_timed_out,
                        avg_wait_us,
                        outer.fence_wait_us_max,
                    );
                }
                outer.fps_window_start = std::time::Instant::now();
                outer.fps_dmabuf_count = 0;
                outer.fps_shm_count = 0;
                outer.fence_no_fence = 0;
                outer.fence_signaled = 0;
                outer.fence_waited = 0;
                outer.fence_timed_out = 0;
                outer.fence_wait_us_total = 0;
                outer.fence_wait_us_max = 0;
            }
        }

        // ── Commit when ready ────────────────────────────────────────────
        //
        // We commit in two situations:
        //
        //   A. Frame callback fired (paced mode): Niri has presented our last
        //      frame and signalled it is ready for the next one. Committing
        //      here syncs our output to Niri's vblank and avoids over-submit.
        //
        //   B. New damage arrived with no callback in flight (restart mode):
        //      The callback chain went idle (nothing to show) and new content
        //      has arrived. We MUST commit something here to prime the chain —
        //      wl_surface.frame only fires after a commit, so requesting a
        //      callback alone would deadlock.
        //
        // In both cases frame() is called BEFORE commit() per Wayland spec.
        //
        // Anti-tearing: in paced mode (A) we are guaranteed to be between
        // Niri's frames, so Firefox will have finished its full subsurface
        // batch before we composite. Restart (B) may occasionally capture a
        // partial batch on first wake, but subsequent frames are always clean.
        let should_commit = outer.frame_callback_fired
            || (outer.has_new_damage && outer.pending_frame_cb.is_none());

        if should_commit {
            outer.frame_callback_fired = false;

            if !outer.has_new_damage {
                // Callback fired but no new inner frame — nothing to commit.
                // Release any held inner buffers so KWin isn't blocked.
                state.flush_pending_releases();
            }

            if outer.has_new_damage {
                // ── Zero-copy dmabuf fast path ────────────────────────────────
                //
                // Try clone_focused_dmabuf() FIRST so the dmabuf's own dimensions
                // drive fb sizing. This bypasses capture_focused_dims(), which
                // reads the KWin parent surface and can transiently return (1,1)
                // when the parent's SinglePixel buffer has no viewport set — that
                // mismatch would fall through to the SHM path, triggering
                // flush_pending_releases() and reintroducing the GPU race that
                // causes black-bar flashes.
                let mut used_dmabuf = false;
                if let (Some(dmabuf_mgr), Some(dma)) =
                    (outer.outer_dmabuf.as_ref(), state.clone_focused_dmabuf())
                {
                    outer.session_empty = false;
                    // Probe the implicit dma-buf read fence on KWin's buffer
                    // before forwarding. This both blocks until KWin's GPU
                    // write is finished (preventing niri from scanning out a
                    // half-rendered frame) AND tells us via fence_status
                    // whether implicit-sync is actually in effect on this
                    // buffer.
                    {
                        let t0 = std::time::Instant::now();
                        let st = crate::dmabuf::wait_for_dmabuf_fence_status(dma.fd.as_raw_fd());
                        let elapsed_us = t0.elapsed().as_micros() as u64;
                        match st {
                            crate::dmabuf::FenceStatus::NoFence => outer.fence_no_fence += 1,
                            crate::dmabuf::FenceStatus::AlreadySignaled => {
                                outer.fence_signaled += 1
                            }
                            crate::dmabuf::FenceStatus::Waited => {
                                outer.fence_waited += 1;
                                outer.fence_wait_us_total += elapsed_us;
                                if elapsed_us > outer.fence_wait_us_max {
                                    outer.fence_wait_us_max = elapsed_us;
                                }
                            }
                            crate::dmabuf::FenceStatus::TimedOut => outer.fence_timed_out += 1,
                        }
                    }
                    {
                        let iw = dma.width as u32;
                        let ih = dma.height as u32;
                        if iw != fb.width || ih != fb.height {
                            debug!(
                                "outer view: resizing fb {}x{} → {}x{}",
                                fb.width, fb.height, iw, ih
                            );
                            match Framebuffer::new(&shm, &qh, iw, ih) {
                                Ok(new_fb) => {
                                    fb = new_fb;
                                }
                                Err(e) => {
                                    warn!(error = %e, "outer view: fb resize failed");
                                }
                            }
                        }
                    }
                    {
                        // Prune already-released in-flight entries, explicitly
                        // destroying each WlBuffer so niri can free its object.
                        // retain() only gets &T so we drain-and-rebuild for ownership.
                        let mut i = 0;
                        while i < outer.dmabuf_in_flight.len() {
                            if outer.dmabuf_in_flight[i].released.load(Ordering::Acquire) {
                                let d = outer.dmabuf_in_flight.swap_remove(i);
                                d.buf.destroy();
                            } else {
                                i += 1;
                            }
                        }

                        // Split pending inner releases:
                        // - deferred: only the buffer being forwarded to
                        //   niri — held until niri fires wl_buffer.release
                        //   so its GPU write can't race niri's GPU read.
                        // - immediate: all skipped frames committed by
                        //   KWin since the last outer commit — niri never
                        //   reads these, so we release them right away and
                        //   KWin can reuse the slots at full speed.
                        let dma_ptr = std::sync::Arc::as_ptr(&dma);
                        let (inner_releases, skipped) = state.take_pending_releases_split(dma_ptr);
                        let had_skipped = !skipped.is_empty();
                        for buf in skipped {
                            state.release_inner_buffer(&buf);
                        }
                        if had_skipped {
                            state.poke_compositor_wake();
                        }

                        use std::os::fd::BorrowedFd;
                        let modifier = dma.modifier;
                        let released = Arc::new(AtomicBool::new(false));
                        let params = dmabuf_mgr.create_params(&qh, ());
                        params.add(
                            unsafe { BorrowedFd::borrow_raw(dma.fd.as_raw_fd()) },
                            0,
                            dma.offset,
                            dma.stride,
                            (modifier >> 32) as u32,
                            (modifier & 0xFFFF_FFFF) as u32,
                        );
                        for (idx, plane) in dma.extra_planes.iter().enumerate() {
                            use std::os::fd::AsRawFd as _;
                            params.add(
                                unsafe { BorrowedFd::borrow_raw(plane.fd.as_raw_fd()) },
                                (idx + 1) as u32,
                                plane.offset,
                                plane.stride,
                                (modifier >> 32) as u32,
                                (modifier & 0xFFFF_FFFF) as u32,
                            );
                        }
                        let (dw, dh, dfmt) = (dma.width, dma.height, dma.drm_format);

                        // Recording-active path uses `InnerReleases::Refcounted`
                        // so the recording reader can pin the buffer alongside
                        // niri (preventing KWin from overwriting mid-readback).
                        // Recording-inactive path uses `InnerReleases::Direct`
                        // — exactly the proven Vec<WlBuffer> behavior from
                        // commit 7cc908e, no extra allocations or atomics in
                        // the hot non-recording forwarding path.
                        let recording_active = state.is_recording();
                        let dma_for_record = if recording_active {
                            Some(dma.clone())
                        } else {
                            None
                        };

                        let (kind_inner, recording_holds): (
                            InnerReleases,
                            Option<Vec<Arc<InnerBufferHold>>>,
                        ) = if recording_active {
                            let pinned: Vec<Arc<InnerBufferHold>> = inner_releases
                                .into_iter()
                                .map(|b| InnerBufferHold::new(b, state.clone()))
                                .collect();
                            let niri_holds: Vec<Arc<InnerBufferHold>> = pinned.to_vec();
                            let rec_holds: Vec<Arc<InnerBufferHold>> = pinned.to_vec();
                            drop(pinned);
                            (
                                InnerReleases::Refcounted(StdMutex::new(niri_holds)),
                                Some(rec_holds),
                            )
                        } else {
                            (InnerReleases::Direct(inner_releases), None)
                        };

                        let buf = params.create_immed(
                            dw,
                            dh,
                            dfmt,
                            zwp_linux_buffer_params_v1::Flags::empty(),
                            &qh,
                            OuterBufKind::Dmabuf {
                                released: released.clone(),
                                _data: dma,
                                inner_releases: kind_inner,
                            },
                        );
                        // Protocol requires destroy after create_immed.
                        // Dropping without destroy() leaves the object alive
                        // in niri; at 180fps this leaks ~180 objects/second.
                        params.destroy();
                        surface.attach(Some(&buf), 0, 0);
                        surface.damage_buffer(0, 0, dw, dh);
                        outer
                            .dmabuf_in_flight
                            .push(InFlightDmabuf { buf, released });
                        outer.pending_frame_cb = Some(surface.frame(&qh, ()));
                        surface.commit();
                        outer.has_new_damage = false;
                        used_dmabuf = true;
                        outer.fps_dmabuf_count += 1;
                        debug!(
                            "outer view: zero-copy dmabuf frame in_flight={}",
                            outer.dmabuf_in_flight.len()
                        );
                        let _ = queue.flush();

                        // ── Recording capture (async dmabuf path) ─────
                        //
                        // Hand the dmabuf Arc + recording-side inner-buffer
                        // refs to the recording thread. The slow GPU readback
                        // happens off this critical path; the buffer is kept
                        // pinned (KWin can't reuse the GEM) until both niri
                        // and the recording reader have dropped their refs.
                        if let (Some(dma_rec), Some(rec_holds)) = (dma_for_record, recording_holds)
                        {
                            state.try_push_recording_dmabuf(dma_rec, rec_holds);
                        }
                    }
                }

                // SHM copy fallback for sessions without a gpu dmabuf
                // (non-KWin, software-rendered clients). The dmabuf path
                // above already handled the GPU case; we only reach here
                // when clone_focused_dmabuf() returned None.
                if !used_dmabuf {
                    if let Some((iw, ih)) = state.capture_focused_dims() {
                        outer.session_empty = false;
                        if iw != fb.width || ih != fb.height {
                            match Framebuffer::new(&shm, &qh, iw, ih) {
                                Ok(new_fb) => {
                                    fb = new_fb;
                                }
                                Err(e) => {
                                    warn!(error = %e, "outer view: shm fb resize failed");
                                }
                            }
                        }
                        state.flush_pending_releases();
                        if let Some((slot_mmap, sw, sh)) = fb.write_slot_mmap() {
                            if state.composite_focused_direct(slot_mmap, sw, sh) {
                                // ── Recording capture (SHM path) ──────────
                                // Pixels are already CPU-composited in the
                                // SHM slot; just memcpy out before the slot
                                // is committed (the mmap stays valid past
                                // commit, but copying now keeps the lifetime
                                // of recording_pixels short).
                                if state.is_recording() {
                                    let len = (sw as usize) * (sh as usize) * 4;
                                    let pixels = slot_mmap[..len].to_vec();
                                    state.try_push_recording_pixels(pixels, sw as u32, sh as u32);
                                }
                                outer.has_new_damage = false;
                                outer.fps_shm_count += 1;
                                outer.pending_frame_cb = Some(surface.frame(&qh, ()));
                                fb.commit_write_slot(&surface);
                                let _ = queue.flush();
                            }
                            // Slot busy: retry next wake.
                        }
                    } else if !outer.session_empty {
                        outer.session_empty = true;
                        outer.has_new_damage = false;
                        state.flush_pending_releases();
                        info!("outer view: session empty — committing placeholder");
                        outer.pending_frame_cb = Some(surface.frame(&qh, ()));
                        if fb.commit_placeholder(&surface) {
                            let _ = queue.flush();
                        }
                    } else {
                        outer.has_new_damage = false;
                    }
                }
            }
        }
    }

    // Release all held keys and buttons before detaching. The normal
    // wl_keyboard.leave / wl_pointer.leave path only fires when the outer
    // compositor explicitly revokes focus while the keyboard/pointer are alive.
    // On a normal detach the surface is destroyed while it still has focus, so
    // niri sends leave to the dying keyboard object which we never dispatch.
    // Calling these here guarantees the inner session never has stuck input
    // regardless of which exit path is taken.
    outer.input.on_keyboard_leave();
    outer.input.on_pointer_leave();

    // Explicitly destroy inhibitor, relative pointer, and lock objects before
    // tearing down the surface. The inhibitor in particular must be destroyed
    // (not just dropped) so niri clears its state — otherwise the next attach
    // would fail with `already_inhibited` when trying to create a new inhibitor
    // for the same surface/seat pair.
    outer.input.cleanup();
    let _ = queue.flush();

    // Only clear the wake fd if it still points to ours. If a new outer_view
    // started before we finished tearing down, it has already installed its
    // own wake fd — clearing unconditionally would silence that one too.
    state.clear_outer_wake_fd_if_mine(&wake_arc);
    debug!("outer view: shutting down");
    toplevel.destroy();
    xdg_surface.destroy();
    surface.destroy();
    let _ = queue.roundtrip(&mut outer);
    Ok(())
}

/// How the inner-client buffer release is sequenced relative to the
/// outer-compositor's `wl_buffer.release` (niri).
///
/// The non-recording case uses `Direct` to preserve the proven anti-tearing
/// behavior from commit 7cc908e: when niri releases the outer buffer we
/// immediately release the inner wl_buffer back to KWin.
///
/// The recording case uses `Refcounted` so the same inner buffer is also
/// pinned by the recording reader. The buffer is released to KWin only
/// when both niri AND the recording reader have dropped their refs,
/// preventing KWin from overwriting the GEM mid-readback (which causes
/// the bottom-right cutoff in recordings).
enum InnerReleases {
    Direct(Vec<wayland_server::protocol::wl_buffer::WlBuffer>),
    Refcounted(StdMutex<Vec<Arc<InnerBufferHold>>>),
}

/// User-data tag on outer `WlBuffer` objects.
enum OuterBufKind {
    Shm(Arc<AtomicBool>),
    /// Dmabuf forwarding. `released` is set true on wl_buffer.release so the
    /// in-flight entry can be pruned. `_data` keeps the dmabuf fd alive.
    Dmabuf {
        released: Arc<AtomicBool>,
        _data: Arc<DmabufBufferData>,
        inner_releases: InnerReleases,
    },
}

/// One dmabuf buffer forwarded to the outer compositor and awaiting release.
struct InFlightDmabuf {
    buf: WlBuffer,
    released: Arc<AtomicBool>,
}

/// One slot in the double-buffer. Tracks whether the outer compositor has
/// released the buffer (i.e. it is free to overwrite).
struct ShmSlot {
    mmap: MmapMut,
    buffer: WlBuffer,
    /// True when the outer compositor has sent `wl_buffer.release`, i.e.
    /// the slot is safe to write to. Both slots start as released.
    released: Arc<AtomicBool>,
    _pool: WlShmPool,
}

/// Double-buffered outer-side shm framebuffer.
///
/// We keep two slots and alternate between them on each commit. Before writing
/// to a slot we check that it has been released by the outer compositor — this
/// prevents tearing on scanout-only compositors that hold the buffer until the
/// next vblank.
struct Framebuffer {
    width: u32,
    height: u32,
    stride: i32,
    slots: [ShmSlot; 2],
    /// Index (0 or 1) of the slot we will write to next.
    write_idx: usize,
}

fn alloc_slot(
    shm: &WlShm,
    qh: &QueueHandle<OuterState>,
    w: u32,
    h: u32,
    stride: i32,
) -> Result<ShmSlot> {
    let size = (stride as usize) * (h as usize);
    let raw = unsafe { libc::memfd_create(c"waymux-outer".as_ptr(), libc::MFD_CLOEXEC) };
    anyhow::ensure!(
        raw >= 0,
        "memfd_create: {}",
        std::io::Error::last_os_error()
    );
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    anyhow::ensure!(
        unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) } == 0,
        "ftruncate: {}",
        std::io::Error::last_os_error()
    );
    let mmap = unsafe { MmapMut::map_mut(&fd) }.context("mmap outer framebuffer")?;
    let released = Arc::new(AtomicBool::new(true));
    let pool = shm.create_pool(fd.as_fd(), size as i32, qh, ());
    let buffer = pool.create_buffer(
        0,
        w as i32,
        h as i32,
        stride,
        wl_shm::Format::Argb8888,
        qh,
        OuterBufKind::Shm(released.clone()),
    );
    Ok(ShmSlot {
        mmap,
        buffer,
        released,
        _pool: pool,
    })
}

impl Framebuffer {
    fn new(shm: &WlShm, qh: &QueueHandle<OuterState>, w: u32, h: u32) -> Result<Self> {
        let stride = (w * 4) as i32;
        let slot0 = alloc_slot(shm, qh, w, h, stride)?;
        let slot1 = alloc_slot(shm, qh, w, h, stride)?;
        Ok(Self {
            width: w,
            height: h,
            stride,
            slots: [slot0, slot1],
            write_idx: 0,
        })
    }

    fn paint_placeholder(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        let stride = self.stride as usize;
        // Paint both slots so the placeholder is visible regardless of
        // which one the outer compositor first scans out.
        for slot in &mut self.slots {
            for y in 0..h {
                for x in 0..w {
                    let p = y * stride + x * 4;
                    let (b, g, r, a) = if x < 4 || y < 4 || x + 4 >= w || y + 4 >= h {
                        (0xFF, 0x00, 0xFF, 0xFF) // magenta border — attach alive
                    } else {
                        (0x40, 0x40, 0x40, 0xFF) // gray — no inner frame yet
                    };
                    slot.mmap[p] = b;
                    slot.mmap[p + 1] = g;
                    slot.mmap[p + 2] = r;
                    slot.mmap[p + 3] = a;
                }
            }
        }
    }

    /// Commit a placeholder frame (dark gray + magenta border) into the next
    /// available slot. Returns false if the slot hasn't been released yet.
    fn commit_placeholder(&mut self, surface: &WlSurface) -> bool {
        let slot = &mut self.slots[self.write_idx];
        if !slot.released.load(Ordering::Acquire) {
            return false;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let stride = self.stride as usize;
        for y in 0..h {
            for x in 0..w {
                let p = y * stride + x * 4;
                let (b, g, r, a) = if x < 4 || y < 4 || x + 4 >= w || y + 4 >= h {
                    (0xFF, 0x00, 0xFF, 0xFF) // magenta border
                } else {
                    (0x20, 0x20, 0x20, 0xFF) // dark — session empty
                };
                slot.mmap[p] = b;
                slot.mmap[p + 1] = g;
                slot.mmap[p + 2] = r;
                slot.mmap[p + 3] = a;
            }
        }
        slot.released.store(false, Ordering::Release);
        surface.attach(Some(&slot.buffer), 0, 0);
        surface.damage_buffer(0, 0, self.width as i32, self.height as i32);
        surface.commit();
        self.write_idx = 1 - self.write_idx;
        true
    }

    /// Borrow the write slot's mmap for direct compositing without an
    /// intermediate copy. Returns None if the slot has not been released by
    /// the outer compositor yet.
    fn write_slot_mmap(&mut self) -> Option<(&mut MmapMut, i32, i32)> {
        let slot = &mut self.slots[self.write_idx];
        if !slot.released.load(Ordering::Acquire) {
            return None;
        }
        Some((&mut slot.mmap, self.width as i32, self.height as i32))
    }

    /// Commit whatever was last written into the write slot.
    /// Must be called after a successful `write_slot_mmap` write.
    fn commit_write_slot(&mut self, surface: &WlSurface) {
        let slot = &self.slots[self.write_idx];
        slot.released.store(false, Ordering::Release);
        surface.attach(Some(&slot.buffer), 0, 0);
        surface.damage_buffer(0, 0, self.width as i32, self.height as i32);
        surface.commit();
        self.write_idx = 1 - self.write_idx;
    }
}

fn make_eventfd() -> Result<OwnedFd> {
    let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
    anyhow::ensure!(raw >= 0, "eventfd: {}", std::io::Error::last_os_error());
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

fn poll_two(a: RawFd, b: RawFd, timeout_ms: i32) {
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
    ];
    unsafe {
        libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout_ms);
    }
}

/// Copy up to `size` bytes from `src_fd` into a fresh anonymous `memfd`.
/// Used to own the keymap data independently of the outer compositor's fd
/// lifetime.
pub(crate) fn copy_fd_to_memfd(src_fd: &OwnedFd, size: usize) -> Option<OwnedFd> {
    if size == 0 {
        return None;
    }
    let src = unsafe { memmap2::MmapOptions::new().len(size).map(src_fd) }.ok()?;
    let raw = unsafe { libc::memfd_create(c"waymux-keymap-outer".as_ptr(), libc::MFD_CLOEXEC) };
    if raw < 0 {
        return None;
    }
    let new_fd = unsafe { OwnedFd::from_raw_fd(raw) };
    if unsafe { libc::ftruncate(new_fd.as_raw_fd(), size as libc::off_t) } != 0 {
        return None;
    }
    let mut dst = unsafe { memmap2::MmapMut::map_mut(&new_fd) }.ok()?;
    dst[..size].copy_from_slice(&src[..size]);
    Some(new_fd)
}

// ─── wayland-client Dispatch ────────────────────────────────────────────

struct OuterState {
    configured: bool,
    toplevel_fullscreen: bool,
    state: Arc<State>,
    input: InputBridge,
    stop: Arc<AtomicBool>,
    /// Serial from the most recent wl_pointer.enter event. Used to call
    /// wl_pointer.set_cursor(serial, null) to hide the cursor during gaming.
    pointer_enter_serial: u32,
    /// True once all inner windows have been closed.
    session_empty: bool,
    /// Set true by WlCallback::Done. Consumed (set false) in the commit block.
    frame_callback_fired: bool,
    /// Keep the pending wl_surface.frame callback alive until Done fires.
    pending_frame_cb: Option<WlCallback>,
    /// True when inner damage arrived since the last outer commit. Set when
    /// last_damage_ns changes; cleared after each commit.
    has_new_damage: bool,
    /// last_damage_ns value seen on the most recent loop iteration, used to
    /// detect whether new inner damage has arrived.
    last_damage_checked: u64,
    /// Last-seen inner_selection_version.
    inner_sel_version: u64,
    /// Currently active outer data source (inner→outer direction). Kept alive
    /// so the outer compositor can request data from it.
    _outer_source: Option<ClientDataSource>,
    /// Pending wl_data_offer objects received from the outer compositor.
    /// Map from ObjectId → accumulated MIME types (filled before selection fires).
    pending_offers: HashMap<ObjectId, Vec<String>>,
    /// Hold the proxies alive for the lifetime of the view.
    _seat: Option<WlSeat>,
    _keyboard: Option<WlKeyboard>,
    _pointer: Option<WlPointer>,
    _data_device: Option<ClientDataDevice>,
    _data_device_manager: Option<WlDataDeviceManager>,
    /// Outer dmabuf manager — bound if the host compositor supports it.
    outer_dmabuf: Option<OuterDmabuf>,
    /// Dmabuf buffers forwarded to the outer compositor and not yet released.
    /// Multiple may be in flight simultaneously; each is pruned when its
    /// released flag is set by the wl_buffer.release handler.
    dmabuf_in_flight: Vec<InFlightDmabuf>,
    fps_window_start: std::time::Instant,
    fps_dmabuf_count: u32,
    fps_shm_count: u32,
    fence_no_fence: u32,
    fence_signaled: u32,
    fence_waited: u32,
    fence_timed_out: u32,
    fence_wait_us_total: u64,
    fence_wait_us_max: u64,
}

impl Dispatch<WlRegistry, GlobalListContents> for OuterState {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: wayland_client::protocol::wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlCompositor, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &WlCompositor,
        _: wayland_client::protocol::wl_compositor::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlSurface, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &WlSurface,
        _: wayland_client::protocol::wl_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlShm, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &WlShm,
        _: wayland_client::protocol::wl_shm::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlShmPool, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &WlShmPool,
        _: wayland_client::protocol::wl_shm_pool::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlBuffer, OuterBufKind> for OuterState {
    fn event(
        state: &mut Self,
        _: &WlBuffer,
        event: wayland_client::protocol::wl_buffer::Event,
        kind: &OuterBufKind,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wayland_client::protocol::wl_buffer::Event::Release = event {
            match kind {
                OuterBufKind::Shm(released) => released.store(true, Ordering::Release),
                OuterBufKind::Dmabuf {
                    released,
                    inner_releases,
                    ..
                } => {
                    released.store(true, Ordering::Release);
                    match inner_releases {
                        InnerReleases::Direct(bufs) => {
                            // Niri is done reading — release the inner GEMs
                            // immediately so KWin can render its next frame.
                            let any = !bufs.is_empty();
                            for buf in bufs {
                                state.state.release_inner_buffer(buf);
                            }
                            if any {
                                state.state.poke_compositor_wake();
                            }
                        }
                        InnerReleases::Refcounted(m) => {
                            // Drop our Arcs. If the recording reader has
                            // already finished its readback (Arc count was
                            // 1 after this drop), InnerBufferHold's Drop
                            // sends wl_buffer.release and pokes the compositor.
                            let drained: Vec<_> = std::mem::take(&mut *m.lock().unwrap());
                            drop(drained);
                        }
                    }
                }
            }
        }
    }
}

impl Dispatch<XdgWmBase, ()> for OuterState {
    fn event(
        _: &mut Self,
        proxy: &XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
}

impl Dispatch<XdgSurface, ()> for OuterState {
    fn event(
        state: &mut Self,
        proxy: &XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            proxy.ack_configure(serial);
            state.configured = true;
        }
        // The shortcut inhibitor is activated unconditionally on first configure
        // so that Super+Left, Alt+Tab, and all other global shortcuts reach the
        // inner session. Activation is deferred to the main loop via the seat ref.
    }
}

impl Dispatch<XdgToplevel, ()> for OuterState {
    fn event(
        state: &mut Self,
        _: &XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Close => {
                // Outer compositor asked us to close (e.g. Alt+Shift+Q in Niri).
                state.stop.store(true, Ordering::Relaxed);
            }
            xdg_toplevel::Event::Configure { states, .. } => {
                // Track whether the session window is fullscreen. The shortcut
                // inhibitor is activated unconditionally in fullscreen so that
                // Super+Left and all other combos reach the inner compositor.
                // In windowed mode the inhibitor is conditional, preserving the
                // user's host compositor shortcuts.
                state.toplevel_fullscreen = states.chunks_exact(4).any(|c| {
                    let v = u32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
                    matches!(
                        xdg_toplevel::State::try_from(v),
                        Ok(xdg_toplevel::State::Fullscreen)
                    )
                });
            }
            _ => {}
        }
    }
}

// ─── Input forwarding: outer seat → inner inject_* ─────────────────────

impl Dispatch<WlSeat, ()> for OuterState {
    fn event(
        state: &mut Self,
        seat: &WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        // niri (and other hosts) re-emit Capabilities mid-session when their
        // seat reseats internally (Touch device hotplug, xkbcommon reload,
        // etc.). Held wl_keyboard / wl_pointer proxies go defunct silently.
        // Worse: any active shortcut inhibitor stays active on the host side,
        // locking the user out of their own hotkeys (only TTY-switch escapes).
        //
        // Strategy: on every Capabilities event, release the old proxies
        // and rebind from the current seat. Always-rebind is slightly
        // wasteful (initial seat enumeration triggers it once redundantly)
        // but bulletproof. Then drop the inhibitor — the loop's next
        // update_inhibitor() will recreate it against the rebound seat
        // if the inner client still wants the inhibit.
        if let wl_seat::Event::Capabilities { capabilities } = &event {
            let caps = match capabilities {
                wayland_client::WEnum::Value(c) => *c,
                _ => return,
            };
            tracing::info!(
                ?caps,
                "outer seat: capabilities event — rebinding input proxies"
            );

            // Tear down old proxies. release() is the spec-correct way to
            // tell the server we're done with them; merely dropping the
            // Rust handle would leak the server-side object.
            if let Some(kb) = state._keyboard.take() {
                kb.release();
            }
            if let Some(p) = state._pointer.take() {
                p.release();
            }

            // Rebind based on current caps. Get the proxies even if the
            // capability bit is not set — wayland-client tolerates that and
            // it preserves the previous always-call behavior on rebind.
            // If the bit is set we'll receive events; otherwise the proxy
            // stays inert until the next Capabilities event re-grants.
            if caps.contains(wl_seat::Capability::Keyboard) {
                state._keyboard = Some(seat.get_keyboard(qh, ()));
            }
            if caps.contains(wl_seat::Capability::Pointer) {
                state._pointer = Some(seat.get_pointer(qh, ()));
            }

            // Reset inhibitor + clear keys_held so the next loop tick
            // re-establishes both against the new keyboard. Without this,
            // niri honors the orphan inhibitor forever and the user can't
            // escape via host shortcuts.
            state.input.reset_for_seat_change();
        }
    }
}

impl Dispatch<WlKeyboard, ()> for OuterState {
    fn event(
        state: &mut Self,
        _: &WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_keyboard::Event::Keymap { format, fd, size } => {
                // Pass the outer compositor's keymap through to inner clients so
                // non-US layouts (e.g. German QWERTZ) work correctly. INFO so we
                // can correlate keymap arrivals with seat caps changes (task #96).
                tracing::info!(
                    ?format,
                    size,
                    "outer keyboard: keymap event (forwarding to inner clients)"
                );
                if matches!(format, ClientWEnum::Value(wl_keyboard::KeymapFormat::XkbV1)) {
                    if let Some(new_fd) = copy_fd_to_memfd(&fd, size as usize) {
                        state.state.update_keymap(new_fd, size);
                    }
                }
            }
            wl_keyboard::Event::Enter { keys, .. } => state.input.on_keyboard_enter(keys),
            wl_keyboard::Event::Leave { .. } => state.input.on_keyboard_leave(),
            wl_keyboard::Event::Modifiers {
                mods_depressed,
                mods_latched,
                mods_locked,
                group,
                ..
            } => {
                state
                    .input
                    .on_keyboard_modifiers(mods_depressed, mods_latched, mods_locked, group);
            }
            wl_keyboard::Event::Key { key, state: ks, .. } => {
                // The 2026-05-07 task-#96 repro confirmed keys flow through here
                // correctly during gameplay; the perception of "no keyboard input"
                // was downstream (Source/Wine ignoring keys delivered via Xwayland —
                // see task #101). Logging individual keys is too noisy for
                // steady-state, so this stays at debug.
                state.input.on_keyboard_key(key, ks)
            }
            _ => {}
        }
    }
}

impl Dispatch<WlCallback, ()> for OuterState {
    fn event(
        state: &mut Self,
        _: &WlCallback,
        event: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            state.pending_frame_cb = None;
            state.frame_callback_fired = true;
            // Wake the inner compositor so drain_frame_callbacks() runs now,
            // firing KWin's pending frame callbacks in sync with the outer
            // vblank rather than waiting for the compositor thread's 16ms poll.
            state.state.poke_compositor_wake();
        }
    }
}

impl Dispatch<WlPointer, ()> for OuterState {
    fn event(
        state: &mut Self,
        pointer: &WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter { serial, .. } => {
                state.pointer_enter_serial = serial;
                // If pointer lock is already active (e.g. re-enter while locked),
                // immediately re-hide the cursor.
                if state.state.pointer_lock_active() {
                    pointer.set_cursor(serial, None, 0, 0);
                }
            }
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                state.input.on_pointer_motion(surface_x, surface_y);
            }
            wl_pointer::Event::Button {
                button, state: btn, ..
            } => {
                state.input.on_pointer_button(button, btn);
            }
            wl_pointer::Event::Leave { .. } => state.input.on_pointer_leave(),
            wl_pointer::Event::AxisSource { axis_source } => {
                state.input.on_pointer_axis_source(axis_source);
            }
            wl_pointer::Event::Axis { axis, value, .. } => {
                state.input.on_pointer_axis(axis, value);
            }
            wl_pointer::Event::AxisStop { axis, .. } => {
                state.input.on_pointer_axis_stop(axis);
            }
            wl_pointer::Event::Frame => state.input.on_pointer_frame(),
            _ => {}
        }
    }
}

// ─── Clipboard bridge Dispatch impls ────────────────────────────────────

impl Dispatch<WlDataDeviceManager, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &WlDataDeviceManager,
        _: wayland_client::protocol::wl_data_device_manager::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ClientDataDevice, ()> for OuterState {
    fn event(
        state: &mut Self,
        _: &ClientDataDevice,
        event: wl_data_device::Event,
        _: &(),
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_device::Event::DataOffer { id } => {
                // Register an empty MIME list for this offer. The offer's MIME
                // types will be filled in by subsequent wl_data_offer.offer events.
                state.pending_offers.insert(id.id(), Vec::new());
            }
            wl_data_device::Event::Selection { id } => {
                // `id` is the selected offer (or None to clear selection).
                let Some(offer) = id else {
                    return;
                };
                let oid = offer.id();
                let mime_types = state.pending_offers.remove(&oid).unwrap_or_default();

                // Prefer UTF-8 text types in order.
                const TEXT_MIMES: &[&str] = &[
                    "text/plain;charset=utf-8",
                    "text/plain;charset=UTF-8",
                    "UTF8_STRING",
                    "text/plain",
                    "TEXT",
                    "STRING",
                ];
                let chosen = TEXT_MIMES
                    .iter()
                    .copied()
                    .find(|m| mime_types.iter().any(|t| t.eq_ignore_ascii_case(m)))
                    .map(str::to_owned);

                if let Some(mime) = chosen {
                    // Create a pipe, ask the outer compositor to write to the
                    // write end, read from the read end.
                    let mut pipe_fds = [0i32; 2];
                    if unsafe { libc::pipe2(pipe_fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                        return;
                    }
                    let read_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[0]) };
                    let write_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[1]) };

                    offer.receive(mime.clone(), write_fd.as_fd());
                    // Flush so the outer compositor sees the receive request
                    // and starts writing.
                    drop(write_fd);
                    let _ = conn.flush();

                    // Drain the selection pipe in a detached thread so a slow
                    // outer compositor / large payload can't stall the
                    // outer_view event loop (audit H17). Mirrors the send-
                    // direction pattern (waymux-clipboard-pipe) above.
                    let state_arc = state.state.clone();
                    std::thread::Builder::new()
                        .name("waymux-clipboard-recv".into())
                        .spawn(move || {
                            let mut data = Vec::new();
                            let mut buf = [0u8; 4096];
                            loop {
                                let n = unsafe {
                                    libc::read(
                                        read_fd.as_raw_fd(),
                                        buf.as_mut_ptr().cast(),
                                        buf.len(),
                                    )
                                };
                                if n <= 0 {
                                    break;
                                }
                                data.extend_from_slice(&buf[..n as usize]);
                            }
                            let content = Arc::new(ClipboardContent {
                                entries: vec![(mime, data)],
                            });
                            state_arc.set_outer_clipboard(content);
                            debug!("outer view: clipboard bridge: fetched outer selection");
                        })
                        .ok();
                }
                offer.destroy();
                let _ = conn.flush();
            }
            _ => {}
        }
        let _ = qh;
    }
}

impl Dispatch<ClientDataOffer, ()> for OuterState {
    fn event(
        state: &mut Self,
        offer: &ClientDataOffer,
        event: wl_data_offer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_data_offer::Event::Offer { mime_type } = event {
            if let Some(list) = state.pending_offers.get_mut(&offer.id()) {
                list.push(mime_type);
            }
        }
    }
}

// ─── Dmabuf forwarding Dispatch impls ───────────────────────────────────

impl Dispatch<OuterDmabuf, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &OuterDmabuf,
        _: wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<OuterDmabufParams, ()> for OuterState {
    fn event(
        _state: &mut Self,
        _: &OuterDmabufParams,
        event: zwp_linux_buffer_params_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // create_immed is synchronous so `created` is never sent. `failed`
        // means the host rejected the buffer; the released flag in OuterBufKind
        // will be set when the (null) buffer's release fires, which prunes it
        // from dmabuf_in_flight automatically. Nothing extra to do here.
        let _ = event;
    }
}

// ─── Keyboard shortcuts inhibit Dispatch impls ──────────────────────────

impl Dispatch<ClientInhibitManager, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &ClientInhibitManager,
        _: wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::zwp_keyboard_shortcuts_inhibit_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ClientInhibitor, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &ClientInhibitor,
        event: wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::zwp_keyboard_shortcuts_inhibitor_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Diagnostic: log niri's Active/Inactive responses to our
        // inhibitor so we can tell whether our re-activation after a caps
        // rebind actually took effect, or whether niri silently dropped it.
        use wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::zwp_keyboard_shortcuts_inhibitor_v1::Event as InhibitEvt;
        match event {
            InhibitEvt::Active => tracing::info!(
                "outer inhibitor: ACTIVE (niri accepted; host shortcuts forwarded to us)"
            ),
            InhibitEvt::Inactive => tracing::info!(
                "outer inhibitor: INACTIVE (niri ignored or revoked; host eats shortcuts)"
            ),
            _ => {}
        }
    }
}

// ─── Pointer lock + relative pointer Dispatch impls ─────────────────────

/// No events on the manager itself — just a factory for lock/confine objects.
impl Dispatch<ClientPointerConstraints, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &ClientPointerConstraints,
        _: wayland_protocols::wp::pointer_constraints::zv1::client::zwp_pointer_constraints_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

/// Locked-pointer events from niri: `locked` (grant) and `unlocked` (revoke).
impl Dispatch<ClientLockedPointer, ()> for OuterState {
    fn event(
        state: &mut Self,
        _: &ClientLockedPointer,
        event: locked_ptr_ev::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            locked_ptr_ev::Event::Locked => {
                let ptr = state._pointer.clone();
                state.input.on_niri_locked(ptr.as_ref(), qh);
            }
            locked_ptr_ev::Event::Unlocked => state.input.on_niri_unlocked(),
            _ => {}
        }
    }
}

/// No events on the relative-pointer manager — just a factory.
impl Dispatch<ClientRelativeManager, ()> for OuterState {
    fn event(
        _: &mut Self,
        _: &ClientRelativeManager,
        _: wayland_protocols::wp::relative_pointer::zv1::client::zwp_relative_pointer_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

/// Relative motion from niri — forwarded to inner relative-pointer objects.
impl Dispatch<ClientRelativePointer, ()> for OuterState {
    fn event(
        state: &mut Self,
        _: &ClientRelativePointer,
        event: rel_ptr_ev::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let rel_ptr_ev::Event::RelativeMotion {
            utime_hi,
            utime_lo,
            dx,
            dy,
            dx_unaccel,
            dy_unaccel,
        } = event
        {
            state
                .input
                .on_relative_motion(utime_hi, utime_lo, dx, dy, dx_unaccel, dy_unaccel);
        }
    }
}

/// Outer wl_data_source: used when an inner client sets a selection.
/// The user data is the corresponding inner WlDataSource (server-side).
impl Dispatch<ClientDataSource, wayland_server::protocol::wl_data_source::WlDataSource>
    for OuterState
{
    fn event(
        state: &mut Self,
        _source: &ClientDataSource,
        event: wl_data_source::Event,
        inner_src: &wayland_server::protocol::wl_data_source::WlDataSource,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_source::Event::Send { mime_type, fd } => {
                if !ServerResource::is_alive(inner_src) {
                    return;
                }
                // Create a pipe: ask the inner source to write to write_end,
                // then copy from read_end to the outer compositor's fd.
                let mut pipe_fds = [0i32; 2];
                if unsafe { libc::pipe2(pipe_fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                    return;
                }
                let read_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[0]) };
                let write_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[1]) };

                // Signal the inner client to write its data to write_fd.
                inner_src.send(mime_type, write_fd.as_fd());
                drop(write_fd);
                // Wake the inner compositor so the client sees the send event
                // and writes its data to write_fd promptly.
                state.state.poke_compositor_wake();

                // Pipe the data to the outer compositor's fd in a detached thread
                // so we don't block the outer_view event loop.
                let outer_raw = unsafe { libc::dup(fd.as_raw_fd()) };
                std::thread::Builder::new()
                    .name("waymux-clipboard-pipe".into())
                    .spawn(move || {
                        if outer_raw < 0 {
                            return;
                        }
                        let outer_fd = unsafe { OwnedFd::from_raw_fd(outer_raw) };
                        let mut buf = [0u8; 65536];
                        loop {
                            let n = unsafe {
                                libc::read(read_fd.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len())
                            };
                            if n <= 0 {
                                break;
                            }
                            let mut off = 0usize;
                            while off < n as usize {
                                let w = unsafe {
                                    libc::write(
                                        outer_fd.as_raw_fd(),
                                        buf[off..].as_ptr().cast(),
                                        (n as usize) - off,
                                    )
                                };
                                if w <= 0 {
                                    return;
                                }
                                off += w as usize;
                            }
                        }
                    })
                    .ok();
            }
            wl_data_source::Event::Cancelled => {
                // Outer compositor cancelled our selection (another app took focus).
                // Clear the inner selection state so we don't try to re-offer.
                state.state.clear_inner_selection();
                state._outer_source = None;
            }
            _ => {}
        }
    }
}
