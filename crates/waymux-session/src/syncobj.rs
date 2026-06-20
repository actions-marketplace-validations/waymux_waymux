// SPDX-License-Identifier: Apache-2.0

//! Server-side wp_linux_drm_syncobj_v1 implementation.
//!
//! Why: AMD KWin uses explicit-sync internally and attaches no implicit
//! dma-buf fence to its committed buffers. Without this protocol, waymux
//! reads/forwards buffers while KWin's GPU work is still in flight,
//! producing the matching live + recorded tearing observed in xonotic.
//!
//! Strategy: advertise wp_linux_drm_syncobj_manager_v1. Each surface
//! sync object stores the pending acquire/release points. On
//! wl_surface.commit, we block on the acquire point's syncobj timeline
//! (CPU-side, via DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT) before promoting the
//! buffer to current. When the resulting buffer is released back to
//! KWin, we signal the matching release point with
//! DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL.
//!
//! The previous attempt at this protocol was a stub that never signaled
//! release timelines, which made KWin deadlock waiting for fence
//! completion. Always signal release on every release path, even error
//! cases, or KWin will hang the same way again.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex};

use tracing::{debug, warn};

// DRM syncobj ioctls (from <drm/drm.h>). Numbers are stable across kernels.
//
// _IOWR('d', NR, sizeof(STRUCT)):
//   FD_TO_HANDLE: NR=0xC2, struct drm_syncobj_handle = 24 bytes  → 0xC018_64C2
//   DESTROY:      NR=0xC0, struct drm_syncobj_destroy = 8 bytes  → 0xC008_64C0
//   TIMELINE_WAIT:NR=0xCA, struct drm_syncobj_timeline_wait = 48 bytes → 0xC030_64CA
//   TIMELINE_SIGNAL: NR=0xCD, struct drm_syncobj_timeline_array = 24 bytes → 0xC018_64CD
const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE: libc::c_ulong = 0xC018_64C2;
const DRM_IOCTL_SYNCOBJ_DESTROY: libc::c_ulong = 0xC008_64C0;
const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT: libc::c_ulong = 0xC030_64CA;
const DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL: libc::c_ulong = 0xC018_64CD;

const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 0;

#[repr(C)]
struct DrmSyncobjHandle {
    handle: u32,
    flags: u32,
    fd: i32,
    pad: u32,
    point: u64,
}

#[repr(C)]
struct DrmSyncobjDestroy {
    handle: u32,
    pad: u32,
}

#[repr(C)]
struct DrmSyncobjTimelineArray {
    handles_ptr: u64,
    points_ptr: u64,
    count_handles: u32,
    flags: u32,
}

#[repr(C)]
struct DrmSyncobjTimelineWait {
    handles_ptr: u64,
    points_ptr: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
}

/// One DRM device file kept open for the lifetime of the session, used
/// to import client-provided syncobj fds, wait on points, and signal
/// release points.
pub struct SyncobjDevice {
    drm_fd: OwnedFd,
}

impl SyncobjDevice {
    /// Open a DRM render/primary node. Tries card1 first (hypothesis: RADV opens
    /// card1, not renderD128), then renderD128, then card0. Returns None if no
    /// node could be opened — the protocol is just not advertised.
    pub fn open() -> Option<Arc<Self>> {
        for path in [
            "/dev/dri/card1\0",
            "/dev/dri/renderD128\0",
            "/dev/dri/card0\0",
        ] {
            let fd = unsafe {
                libc::open(
                    path.as_ptr() as *const libc::c_char,
                    libc::O_RDWR | libc::O_CLOEXEC,
                )
            };
            if fd >= 0 {
                let path_str = path.trim_end_matches('\0');
                let mut sb = unsafe { std::mem::zeroed::<libc::stat>() };
                unsafe { libc::fstat(fd, &mut sb) };
                let major = libc::major(sb.st_rdev);
                let minor = libc::minor(sb.st_rdev);
                tracing::info!(path = path_str, major, minor, "syncobj: opened DRM node");
                return Some(Arc::new(Self {
                    drm_fd: unsafe { OwnedFd::from_raw_fd(fd) },
                }));
            }
        }
        warn!("syncobj: no DRM node available — explicit-sync disabled");
        None
    }

    /// Import a client-provided syncobj fd into this device. Returns the
    /// kernel handle that subsequent wait/signal calls take. The handle
    /// must be destroyed via `destroy_timeline` when no longer needed.
    pub fn import_timeline(&self, fd: RawFd) -> Option<u32> {
        // Diagnostic: check if fd is valid before ioctl
        let mut stat_buf = unsafe { std::mem::zeroed::<libc::stat>() };
        let fstat_ret = unsafe { libc::fstat(fd, &mut stat_buf) };

        // Try to read the fd type from /proc/self/fd
        let fd_type = {
            let path = format!("/proc/self/fd/{}", fd);
            std::fs::read_link(&path)
                .ok()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        };

        // Try to read fd info from /proc/self/fdinfo to get more details
        let fd_info = {
            let path = format!("/proc/self/fdinfo/{}", fd);
            std::fs::read_to_string(&path)
                .ok()
                .unwrap_or_else(|| "".to_string())
        };

        // Check device major/minor from st_rdev (for device files)
        let drm_fd_stat = unsafe {
            let mut sb = std::mem::zeroed::<libc::stat>();
            libc::fstat(self.drm_fd.as_raw_fd(), &mut sb);
            sb
        };

        let client_fd_major = libc::major(stat_buf.st_rdev);
        let client_fd_minor = libc::minor(stat_buf.st_rdev);
        let device_fd_major = libc::major(drm_fd_stat.st_rdev);
        let device_fd_minor = libc::minor(drm_fd_stat.st_rdev);

        // Log a safe version of fd_info (first 200 chars to avoid spam)
        let fd_info_short = fd_info.lines().take(2).collect::<Vec<_>>().join("; ");

        tracing::info!(
            raw_fd = fd,
            fstat_ret,
            fd_type,
            fd_info = fd_info_short,
            client_fd_major,
            client_fd_minor,
            device_fd_major,
            device_fd_minor,
            "syncobj: import_timeline called"
        );

        let mut req = DrmSyncobjHandle {
            handle: 0,
            flags: 0,
            fd,
            pad: 0,
            point: 0,
        };
        let ret = unsafe {
            libc::ioctl(
                self.drm_fd.as_raw_fd(),
                DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE,
                &mut req,
            )
        };
        if ret == 0 && req.handle != 0 {
            tracing::info!(
                raw_fd = fd,
                handle = req.handle,
                "syncobj: import_timeline success"
            );
            Some(req.handle)
        } else {
            let e = unsafe { *libc::__errno_location() };
            let errno_name = match e {
                libc::ENOENT => "ENOENT",
                libc::EINVAL => "EINVAL",
                libc::EBADF => "EBADF",
                libc::EACCES => "EACCES",
                _ => "OTHER",
            };
            tracing::info!(
                raw_fd = fd,
                fd_type,
                fstat_ret,
                errno = e,
                errno_name,
                ioctl_ret = ret,
                "syncobj: FD_TO_HANDLE failed"
            );
            None
        }
    }

    fn destroy_timeline(&self, handle: u32) {
        let req = DrmSyncobjDestroy { handle, pad: 0 };
        unsafe {
            libc::ioctl(self.drm_fd.as_raw_fd(), DRM_IOCTL_SYNCOBJ_DESTROY, &req);
        }
    }

    /// Block (with timeout) until `point` on `handle` is signaled. Uses
    /// WAIT_FOR_SUBMIT so it's safe to wait on a point that hasn't been
    /// submitted yet — the kernel parks us until submission, then signals.
    /// Returns true if signaled, false if timeout/error.
    pub fn wait_timeline_point(&self, handle: u32, point: u64, timeout_ns: i64) -> bool {
        let handles = [handle];
        let points = [point];
        // Get current CLOCK_MONOTONIC time and add timeout to compute absolute deadline.
        // The kernel's drm_syncobj_timeline_wait uses deadline_nsec as an absolute
        // CLOCK_MONOTONIC deadline, not a relative timeout. timeout_nsec is deprecated.
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let now_ns = if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } == 0 {
            ts.tv_sec * 1_000_000_000 + ts.tv_nsec
        } else {
            // Fallback: if clock_gettime fails, use a large deadline (10 seconds in future).
            // This should never happen under normal conditions.
            10_000_000_000i64
        };
        let deadline = now_ns.saturating_add(timeout_ns);
        let mut req = DrmSyncobjTimelineWait {
            handles_ptr: handles.as_ptr() as u64,
            points_ptr: points.as_ptr() as u64,
            timeout_nsec: 0,
            count_handles: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: deadline as u64,
        };
        let ret = unsafe {
            libc::ioctl(
                self.drm_fd.as_raw_fd(),
                DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT,
                &mut req,
            )
        };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            debug!(
                "syncobj: wait_timeline_point failed handle={} point={} timeout_ns={} deadline={} errno={}",
                handle, point, timeout_ns, deadline, err
            );
        }
        ret == 0
    }

    /// Signal `point` on `handle`. The kernel auto-signals every previous
    /// point on the timeline as well, per syncobj timeline semantics.
    pub fn signal_timeline_point(&self, handle: u32, point: u64) -> bool {
        let handles = [handle];
        let points = [point];
        let req = DrmSyncobjTimelineArray {
            handles_ptr: handles.as_ptr() as u64,
            points_ptr: points.as_ptr() as u64,
            count_handles: 1,
            flags: 0,
        };
        let ret = unsafe {
            libc::ioctl(
                self.drm_fd.as_raw_fd(),
                DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL,
                &req,
            )
        };
        ret == 0
    }
}

/// A single client-imported timeline. Drop releases the kernel handle.
/// Cloned (Arc) into per-surface SyncState so destroying the
/// wp_linux_drm_syncobj_timeline_v1 wayland object doesn't yank the
/// handle out from under in-flight commits.
pub struct Timeline {
    pub device: Arc<SyncobjDevice>,
    pub handle: u32,
    pub _fd: OwnedFd, // Keep the client's syncobj fd alive. The kernel needs this reference for the syncobj to remain valid.
}

impl Drop for Timeline {
    fn drop(&mut self) {
        self.device.destroy_timeline(self.handle);
    }
}

/// One entry on a timeline (handle + 64-bit point). Lightweight clone.
#[derive(Clone)]
pub struct TimelinePoint {
    pub timeline: Arc<Timeline>,
    pub point: u64,
}

impl TimelinePoint {
    pub fn wait(&self, timeout_ns: i64) -> bool {
        self.timeline
            .device
            .wait_timeline_point(self.timeline.handle, self.point, timeout_ns)
    }

    pub fn signal(&self) -> bool {
        self.timeline
            .device
            .signal_timeline_point(self.timeline.handle, self.point)
    }
}

/// Per-surface explicit-sync state. Holds the pending acquire/release
/// points set since the last commit. Fields are Mutex-wrapped so the
/// wayland-server dispatch handlers (which only get &state) can update.
pub struct SurfaceSync {
    pub pending_acquire: Mutex<Option<TimelinePoint>>,
    pub pending_release: Mutex<Option<TimelinePoint>>,
}

impl SurfaceSync {
    pub fn new() -> Self {
        Self {
            pending_acquire: Mutex::new(None),
            pending_release: Mutex::new(None),
        }
    }

    /// Take the pending acquire/release pair atomically. Called at commit
    /// time after the wl_surface.attach has been processed. Returns
    /// `(acquire, release)` — both `None` for a commit that didn't set
    /// new points (e.g. a re-commit with the same buffer state).
    pub fn take_pending(&self) -> (Option<TimelinePoint>, Option<TimelinePoint>) {
        let a = self.pending_acquire.lock().unwrap().take();
        let r = self.pending_release.lock().unwrap().take();
        (a, r)
    }
}
