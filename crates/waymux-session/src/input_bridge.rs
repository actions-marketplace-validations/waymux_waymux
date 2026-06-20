// SPDX-License-Identifier: Apache-2.0

//! Input protocol bridge between the outer compositor (niri) and the inner session.
//!
//! `InputBridge` owns all input-forwarding state: keyboard modifiers, keys/buttons
//! currently held, the shortcut-inhibitor (always active while attached), and the
//! pointer-lock state machine required for FPS/3D applications.
//!
//! # Pointer lock flow (Xonotic example)
//! ```text
//! Xonotic → KWin (zwp_locked_pointer_v1) → waymux compositor
//!   → state.inc_pointer_lock_count() → poke outer_view
//!   → InputBridge::update_lock() → niri (zwp_locked_pointer_v1)
//!   → Locked event → subscribe zwp_relative_pointer_v1
//!   → niri relative_motion → state.inject_relative_pointer() → KWin → Xonotic
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use tracing::{debug, info};
use wayland_client::protocol::{wl_pointer::WlPointer, wl_seat::WlSeat, wl_surface::WlSurface};
use wayland_client::{
    protocol::{wl_keyboard, wl_pointer},
    Dispatch, QueueHandle, WEnum as ClientWEnum,
};
use wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::{
    zwp_keyboard_shortcuts_inhibit_manager_v1::ZwpKeyboardShortcutsInhibitManagerV1,
    zwp_keyboard_shortcuts_inhibitor_v1::ZwpKeyboardShortcutsInhibitorV1,
};
use wayland_protocols::wp::pointer_constraints::zv1::client::{
    zwp_locked_pointer_v1::ZwpLockedPointerV1, zwp_pointer_constraints_v1::ZwpPointerConstraintsV1,
};
use wayland_protocols::wp::relative_pointer::zv1::client::{
    zwp_relative_pointer_manager_v1::ZwpRelativePointerManagerV1,
    zwp_relative_pointer_v1::ZwpRelativePointerV1,
};

use crate::state::State;

// ─── pointer lock state machine ──────────────────────────────────────────────

#[derive(Debug, PartialEq, Default)]
enum LockState {
    #[default]
    Unlocked,
    /// Lock request sent to niri; awaiting the `locked` confirmation event.
    Requesting,
    /// Niri confirmed the lock; relative motion is flowing.
    Active,
    /// Niri sent `unlocked` because the pointer temporarily left the surface
    /// (persistent-lifetime constraint deactivated but NOT destroyed). The
    /// lock object is still alive in niri. We must NOT destroy and re-create
    /// it — that would trigger `already_constrained`. Instead we wait here
    /// until niri re-sends `locked` when the pointer returns.
    Suspended,
}

// ─── accumulated axis (scroll) state ─────────────────────────────────────────

#[derive(Default)]
pub struct PendingAxis {
    pub source: Option<wayland_server::protocol::wl_pointer::AxisSource>,
    pub v: f64,
    pub h: f64,
    pub stop_v: bool,
    pub stop_h: bool,
}

// ─── InputBridge ─────────────────────────────────────────────────────────────

pub struct InputBridge {
    state: Arc<State>,

    // ── keyboard ─────────────────────────────────────────────────────────
    pub current_modifiers: u32,
    pub current_mods_latched: u32,
    pub current_mods_locked: u32,
    pub current_mod_group: u32,
    /// Keys currently pressed. Bulk-released on wl_keyboard.leave to prevent sticking.
    pub keys_held: HashSet<u32>,

    // ── pointer ──────────────────────────────────────────────────────────
    pub last_pointer_x: f64,
    pub last_pointer_y: f64,
    /// Mouse buttons currently held. Released on wl_pointer.leave.
    pub buttons_held: HashSet<u32>,
    pub pending_axis: PendingAxis,

    // ── shortcut inhibitor (always active while attached) ─────────────────
    inhibit_manager: Option<ZwpKeyboardShortcutsInhibitManagerV1>,
    outer_inhibitor: Option<ZwpKeyboardShortcutsInhibitorV1>,

    // ── pointer lock bridge ───────────────────────────────────────────────
    pointer_constraints: Option<ZwpPointerConstraintsV1>,
    relative_manager: Option<ZwpRelativePointerManagerV1>,
    active_lock: Option<ZwpLockedPointerV1>,
    active_relative: Option<ZwpRelativePointerV1>,
    lock_state: LockState,
    /// Flipped true on first relative_motion from niri; prevents log spam.
    relative_motion_seen: bool,
}

impl InputBridge {
    pub fn new(
        state: Arc<State>,
        inhibit_manager: Option<ZwpKeyboardShortcutsInhibitManagerV1>,
        pointer_constraints: Option<ZwpPointerConstraintsV1>,
        relative_manager: Option<ZwpRelativePointerManagerV1>,
    ) -> Self {
        Self {
            state,
            current_modifiers: 0,
            current_mods_latched: 0,
            current_mods_locked: 0,
            current_mod_group: 0,
            keys_held: HashSet::new(),
            last_pointer_x: 0.0,
            last_pointer_y: 0.0,
            buttons_held: HashSet::new(),
            pending_axis: PendingAxis::default(),
            inhibit_manager,
            outer_inhibitor: None,
            pointer_constraints,
            relative_manager,
            active_lock: None,
            active_relative: None,
            lock_state: LockState::default(),
            relative_motion_seen: false,
        }
    }

    // ── shortcut inhibitor ────────────────────────────────────────────────

    /// Update the shortcut inhibitor state. `want` is computed by the caller:
    ///   windowed   → true only when an inner client or FPS game holds a pointer lock
    /// Idempotent — only sends a Wayland request when the state actually changes.
    /// IMPORTANT: always calls `.destroy()` explicitly before dropping the inhibitor
    /// proxy. In wayland-client, dropping a proxy does NOT send the destroy request —
    /// the server keeps the object alive, causing `already_inhibited` on the next
    /// `inhibit_shortcuts` call for the same surface.
    pub fn update_inhibitor<D>(
        &mut self,
        want: bool,
        surface: &WlSurface,
        seat: &WlSeat,
        qh: &QueueHandle<D>,
    ) where
        D: Dispatch<ZwpKeyboardShortcutsInhibitorV1, ()> + 'static,
    {
        let have = self.outer_inhibitor.is_some();
        if want && !have {
            if let Some(mgr) = &self.inhibit_manager {
                self.outer_inhibitor = Some(mgr.inhibit_shortcuts(surface, seat, qh, ()));
                // INFO for task #96 — inhibitor transitions are a prime suspect
                // for "keyboard suddenly stopped working" reports. Knowing the
                // exact moment one toggles helps correlate with the symptom.
                tracing::info!("input_bridge: shortcut inhibitor activated");
            }
        } else if !want && have {
            if let Some(inhibitor) = self.outer_inhibitor.take() {
                inhibitor.destroy();
            }
            tracing::info!("input_bridge: shortcut inhibitor deactivated");
        }
    }

    /// Explicitly destroy all held Wayland protocol objects. Must be called before
    /// the outer surface/toplevel is destroyed so the server can clean up state in
    /// the correct order. Dropping Rust proxies without calling destroy() leaves
    /// server-side objects alive (e.g. the inhibitor stays active on niri).
    pub fn cleanup(&mut self) {
        if let Some(inhibitor) = self.outer_inhibitor.take() {
            inhibitor.destroy();
        }
        if let Some(rel) = self.active_relative.take() {
            rel.destroy();
        }
        if let Some(lock) = self.active_lock.take() {
            lock.destroy();
        }
    }

    /// Drop the shortcut inhibitor and clear keys_held. Call from the wl_seat
    /// Capabilities handler when caps change: niri may have re-emitted caps
    /// because of a device hotunplug, internal reseat, or xkbcommon reload.
    /// At that point our held wl_keyboard reference goes defunct and we must
    /// rebind. Critically, the inhibitor we previously created is bound to
    /// the OLD wl_keyboard's seat — niri will keep honoring it forever
    /// (suppressing the user's host shortcuts) until we destroy it. The next
    /// outer_view loop iteration's update_inhibitor() will recreate it
    /// against the rebound seat if the inner client still wants the inhibit.
    /// This is the symptom that locked the user out of niri shortcuts and
    /// forced them to TTY-switch via Ctrl+Alt+F3.
    pub fn reset_for_seat_change(&mut self) {
        if let Some(inhibitor) = self.outer_inhibitor.take() {
            inhibitor.destroy();
            tracing::info!(
                "input_bridge: shortcut inhibitor destroyed (seat caps changed; will reactivate next tick if needed)"
            );
        }
        // Don't touch active_relative / active_lock — those belong to the
        // inner client's pointer-lock state, not to the outer seat.
        // Releasing held keys is the same logic as a focus-leave.
        let dropped = self.keys_held.len();
        self.keys_held.clear();
        if dropped > 0 {
            tracing::info!(
                dropped,
                "input_bridge: cleared keys_held (seat caps changed)"
            );
        }
    }

    // ── main-loop pointer-lock state machine ──────────────────────────────

    /// Drive the pointer-lock state machine. Call once per outer_view loop
    /// iteration. Transitions are O(1) with no allocation on the hot path.
    pub fn update_lock<D>(
        &mut self,
        surface: &WlSurface,
        pointer: Option<&WlPointer>,
        qh: &QueueHandle<D>,
    ) where
        D: Dispatch<ZwpLockedPointerV1, ()> + Dispatch<ZwpRelativePointerV1, ()> + 'static,
    {
        let want = self.state.pointer_lock_active();

        match (&self.lock_state, want) {
            // No lock held yet → request one from niri.
            (LockState::Unlocked, true) => {
                if let (Some(pc), Some(ptr)) = (&self.pointer_constraints, pointer) {
                    use wayland_protocols::wp::pointer_constraints::zv1::client::zwp_pointer_constraints_v1::Lifetime;
                    self.active_lock =
                        Some(pc.lock_pointer(surface, ptr, None, Lifetime::Persistent, qh, ()));
                    self.lock_state = LockState::Requesting;
                    info!("input_bridge: outer pointer lock requested from niri");
                } else {
                    info!(
                        has_constraints = self.pointer_constraints.is_some(),
                        has_pointer = pointer.is_some(),
                        "input_bridge: cannot request outer lock — missing constraint mgr or pointer"
                    );
                }
            }
            // Want to release: destroy the lock constraint but KEEP the relative
            // pointer subscription alive. The lock controls cursor hiding; the
            // relative pointer is independent — niri sends relative_motion on it
            // whether or not the pointer is locked. Destroying active_relative here
            // causes a gap where no relative motion arrives between a lock release
            // and the next niri `locked` confirmation, which can be indefinitely long
            // if the cursor drifted off-surface while unlocked. Keeping it avoids that
            // gap; forwarding is a no-op when no inner clients have relative pointers.
            (LockState::Active | LockState::Suspended | LockState::Requesting, false) => {
                if let Some(lock) = self.active_lock.take() {
                    lock.destroy();
                }
                self.lock_state = LockState::Unlocked;
                // During the locked period on_relative_motion() accumulates
                // dx/dy into last_pointer_x/y without bounds. The value can
                // drift arbitrarily far outside the session area. When the
                // lock releases, KWin uses our next inject_pointer call to
                // position the menu cursor; if last_pointer_x/y is huge,
                // the cursor lands off-screen and the user thinks the mouse
                // is broken. Snap to the session centre so the menu cursor
                // starts at a sane position. niri will correct the absolute
                // position on the next wl_pointer.motion event anyway.
                let (w, h, _) = self.state.snapshot();
                self.last_pointer_x = w as f64 / 2.0;
                self.last_pointer_y = h as f64 / 2.0;
                self.state.inject_pointer(
                    None,
                    false,
                    self.last_pointer_x,
                    self.last_pointer_y,
                    0,
                    false,
                    0.0,
                    0.0,
                    0,
                );
                debug!("input_bridge: pointer lock released; cursor snapped to session centre");
            }
            _ => {}
        }
    }

    // ── events from niri ──────────────────────────────────────────────────

    /// Niri confirmed or re-confirmed the pointer lock.
    /// Called from both fresh `Requesting` state and after a `Suspended` resume.
    pub fn on_niri_locked<D>(&mut self, pointer: Option<&WlPointer>, qh: &QueueHandle<D>)
    where
        D: Dispatch<ZwpRelativePointerV1, ()> + 'static,
    {
        match self.lock_state {
            LockState::Requesting => {
                // Subscribe to relative motion on first lock grant only.
                // active_relative is kept alive across lock/unlock cycles so we
                // don't create a duplicate subscription here on subsequent grants.
                if self.active_relative.is_none() {
                    if let (Some(rm), Some(ptr)) = (&self.relative_manager, pointer) {
                        self.active_relative = Some(rm.get_relative_pointer(ptr, qh, ()));
                        info!("input_bridge: niri granted lock; outer relative pointer subscribed");
                    } else {
                        info!(
                            has_mgr = self.relative_manager.is_some(),
                            has_ptr = pointer.is_some(),
                            "input_bridge: niri granted lock but cannot subscribe relative pointer"
                        );
                    }
                } else {
                    info!("input_bridge: niri granted lock (relative pointer already alive)");
                }
                self.lock_state = LockState::Active;
            }
            LockState::Suspended => {
                // Pointer returned to surface — lock reactivated without a new request.
                self.lock_state = LockState::Active;
                info!("input_bridge: outer pointer lock resumed after suspension");
            }
            _ => {}
        }
    }

    /// Niri deactivated the persistent lock (pointer temporarily left surface).
    /// The lock OBJECT is still alive in niri — do NOT destroy or re-create it.
    /// Transition to Suspended and wait for `locked` to fire when pointer returns.
    pub fn on_niri_unlocked(&mut self) {
        if self.lock_state == LockState::Active {
            self.lock_state = LockState::Suspended;
            debug!("input_bridge: pointer lock suspended (pointer left surface)");
        }
        // If Requesting/Unlocked/Suspended: ignore spurious unlocked events.
    }

    /// Relative motion from niri (received when niri has locked the pointer).
    /// Forward to KWin's ZwpRelativePointerV1 objects (registered when Right Ctrl
    /// grabbed the pointer). Do NOT send synthetic wl_pointer.motion — sending
    /// absolute motion while the pointer is locked confuses KWin's input pipeline.
    pub fn on_relative_motion(
        &mut self,
        utime_hi: u32,
        utime_lo: u32,
        dx: f64,
        dy: f64,
        dx_unaccel: f64,
        dy_unaccel: f64,
    ) {
        if !self.relative_motion_seen {
            self.relative_motion_seen = true;
            info!(
                dx,
                dy, "input_bridge: first outer relative_motion received from niri"
            );
        }
        // KWin's nested-mode PointerGrab (Right Ctrl) only routes motion to locked
        // clients (Xonotic) via the absolute wl_pointer.motion delta path, NOT via
        // zwp_relative_pointer_v1 events injected directly. When niri has locked the
        // outer pointer it stops sending wl_pointer.motion, so we synthesise an
        // absolute position by accumulating the relative deltas and inject it. KWin
        // computes the per-frame delta as (new_pos - old_pos) and delivers that to
        // Xonotic's relative pointer subscription. last_pointer_x/y was last set by
        // on_pointer_motion (niri's absolute motion before the lock was granted), so
        // the first injected frame has the correct delta too.
        self.last_pointer_x += dx;
        self.last_pointer_y += dy;
        self.state.inject_pointer(
            None,
            false,
            self.last_pointer_x,
            self.last_pointer_y,
            0,
            false,
            0.0,
            0.0,
            0,
        );
        // Also forward via relative pointer chain as a fallback for compositors that
        // correctly process zwp_relative_pointer_v1 from the host.
        self.state
            .inject_relative_pointer(utime_hi, utime_lo, dx, dy, dx_unaccel, dy_unaccel);
    }

    // ── keyboard events ───────────────────────────────────────────────────

    pub fn on_keyboard_enter(&mut self, keys: Vec<u8>) {
        self.keys_held.clear();
        for chunk in keys.chunks_exact(4) {
            let code = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            self.keys_held.insert(code);
        }
        tracing::info!(
            held = self.keys_held.len(),
            "input_bridge: keyboard enter (focus arrived)"
        );
    }

    pub fn on_keyboard_leave(&mut self) {
        let held: Vec<u32> = self.keys_held.drain().collect();
        let count = held.len();
        for key in held {
            self.state.inject_key(key, false, 0, 0, 0, 0);
        }
        // INFO so task #96 (keyboard-loss-during-attach) can be narrowed:
        // a Leave with no following Enter == the inner session just lost
        // keyboard input. Cross-reference timestamps with the user's repro.
        tracing::info!(
            released = count,
            "input_bridge: keyboard leave (focus departed; released held keys)"
        );
    }

    pub fn on_keyboard_modifiers(
        &mut self,
        mods_depressed: u32,
        mods_latched: u32,
        mods_locked: u32,
        group: u32,
    ) {
        self.current_modifiers = mods_depressed;
        self.current_mods_latched = mods_latched;
        self.current_mods_locked = mods_locked;
        self.current_mod_group = group;
    }

    pub fn on_keyboard_key(&mut self, key: u32, key_state: ClientWEnum<wl_keyboard::KeyState>) {
        let pressed = matches!(
            key_state,
            ClientWEnum::Value(wl_keyboard::KeyState::Pressed)
        );
        if pressed {
            self.keys_held.insert(key);
        } else {
            self.keys_held.remove(&key);
        }
        debug!(
            key,
            pressed,
            modifiers = self.current_modifiers,
            "outer view: forwarding key"
        );
        self.state.inject_key(
            key,
            pressed,
            self.current_modifiers,
            self.current_mods_latched,
            self.current_mods_locked,
            self.current_mod_group,
        );
    }

    // ── pointer events ────────────────────────────────────────────────────

    pub fn on_pointer_motion(&mut self, x: f64, y: f64) {
        let dx = x - self.last_pointer_x;
        let dy = y - self.last_pointer_y;
        self.last_pointer_x = x;
        self.last_pointer_y = y;
        self.state
            .inject_pointer(None, false, x, y, 0, false, 0.0, 0.0, 0);
        // Synthesize relative motion from absolute deltas when inner clients have
        // registered relative-pointer objects (e.g. KWin subscribed at startup).
        // This is the fallback path for FPS games: KWin in nested mode may not
        // forward zwp_locked_pointer_v1 to waymux-session (so the niri lock chain
        // never activates), but it DOES subscribe to relative pointer from
        // waymux-session. By injecting deltas here, KWin receives relative motion
        // it can forward to Xonotic without requiring the full lock chain.
        if (dx != 0.0 || dy != 0.0) && self.state.has_relative_pointers() {
            self.state.inject_relative_pointer(0, 0, dx, dy, dx, dy);
        }
    }

    pub fn on_pointer_button(
        &mut self,
        button: u32,
        btn_state: ClientWEnum<wl_pointer::ButtonState>,
    ) {
        let pressed = matches!(
            btn_state,
            ClientWEnum::Value(wl_pointer::ButtonState::Pressed)
        );
        if pressed {
            self.buttons_held.insert(button);
        } else {
            self.buttons_held.remove(&button);
        }
        self.state.inject_pointer(
            None,
            false,
            self.last_pointer_x,
            self.last_pointer_y,
            button,
            pressed,
            0.0,
            0.0,
            0,
        );
    }

    pub fn on_pointer_axis_source(&mut self, source: ClientWEnum<wl_pointer::AxisSource>) {
        use wayland_server::protocol::wl_pointer::AxisSource as SA;
        self.pending_axis.source = match source {
            ClientWEnum::Value(wl_pointer::AxisSource::Finger) => Some(SA::Finger),
            ClientWEnum::Value(wl_pointer::AxisSource::Wheel) => Some(SA::Wheel),
            ClientWEnum::Value(wl_pointer::AxisSource::Continuous) => Some(SA::Continuous),
            ClientWEnum::Value(wl_pointer::AxisSource::WheelTilt) => Some(SA::WheelTilt),
            _ => None,
        };
    }

    pub fn on_pointer_axis(&mut self, axis: ClientWEnum<wl_pointer::Axis>, value: f64) {
        match axis {
            ClientWEnum::Value(wl_pointer::Axis::VerticalScroll) => self.pending_axis.v += value,
            ClientWEnum::Value(wl_pointer::Axis::HorizontalScroll) => self.pending_axis.h += value,
            _ => {}
        }
    }

    pub fn on_pointer_axis_stop(&mut self, axis: ClientWEnum<wl_pointer::Axis>) {
        match axis {
            ClientWEnum::Value(wl_pointer::Axis::VerticalScroll) => self.pending_axis.stop_v = true,
            ClientWEnum::Value(wl_pointer::Axis::HorizontalScroll) => {
                self.pending_axis.stop_h = true
            }
            _ => {}
        }
    }

    pub fn on_pointer_frame(&mut self) {
        let ax = &self.pending_axis;
        if ax.v != 0.0 || ax.h != 0.0 || ax.stop_v || ax.stop_h {
            self.state
                .inject_axis(ax.source, ax.h, ax.v, ax.stop_h, ax.stop_v);
        }
        self.pending_axis = PendingAxis::default();
    }

    /// Release all held buttons — prevents stuck buttons when the outer
    /// compositor's pointer leaves our surface (e.g. on detach).
    pub fn on_pointer_leave(&mut self) {
        let held: Vec<u32> = self.buttons_held.drain().collect();
        for btn in held {
            self.state.inject_pointer(
                None,
                false,
                self.last_pointer_x,
                self.last_pointer_y,
                btn,
                false,
                0.0,
                0.0,
                0,
            );
        }
    }
}
