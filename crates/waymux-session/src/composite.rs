// SPDX-License-Identifier: Apache-2.0

//! Capture-time subsurface compositing.
//!
//! GTK4 / Firefox / most modern toolkits commit the toplevel with a tiny
//! (often transparent) buffer and render all the actual content into
//! `wl_subsurface` children. Without composition, `capture_focused` would
//! return only the toplevel's buffer — a blank rectangle — which is what
//! the attach viewer shows before this module existed.
//!
//! We walk the surface tree recursively at capture time and blit each
//! surface's current shm buffer into a single ARGB8888 output, bottom-up.
//! Cost: O(total pixels in the tree) on every frame. Acceptable at our
//! target sizes; dmabuf + GPU composition is planned.

use wayland_server::protocol::{wl_shm::Format, wl_surface::WlSurface};
use wayland_server::{Resource, WEnum};

use crate::buffer::BufferKind;
use crate::compositor::SurfaceData;

pub struct CompositedFrame {
    pub bytes: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub stride: i32,
}

/// Compose the given surface + all its descendant subsurfaces into a fresh
/// ARGB8888 buffer. Output size is the root's buffer size if the root has
/// committed one; otherwise falls back to `fallback_size` (usually the
/// session's virtual output size).
pub fn composite(root: &WlSurface, fallback_size: (u32, u32)) -> Option<CompositedFrame> {
    let mut bytes = Vec::new();
    let (w, h, stride) = composite_into_buf(root, &mut bytes, fallback_size)?;
    Some(CompositedFrame {
        bytes,
        width: w,
        height: h,
        stride,
    })
}

/// Compose into a caller-supplied buffer. The buffer is resized and zeroed
/// only when the required size changes; on repeated calls with the same
/// surface dimensions, no allocation occurs. Returns `(width, height, stride)`
/// on success, None if no buffer is available.
pub fn composite_into_buf(
    root: &WlSurface,
    buf: &mut Vec<u8>,
    fallback_size: (u32, u32),
) -> Option<(i32, i32, i32)> {
    let root_data = root.data::<SurfaceData>()?;
    let (surf_w, surf_h) = surface_dims(root_data, fallback_size);
    if surf_w == 0 || surf_h == 0 {
        return None;
    }
    // For tiny root surfaces (e.g. layer-shell using a 1×1 SinglePixel
    // background with all content in subsurfaces), fall back to the session
    // size so the subsurface tree isn't clipped to a 1×1 box.
    let (w, h) = if surf_w < 4 || surf_h < 4 {
        (fallback_size.0, fallback_size.1)
    } else {
        (surf_w, surf_h)
    };
    let w = w as i32;
    let h = h as i32;
    let stride = w * 4;
    let needed = (h as usize) * (stride as usize);
    if buf.len() != needed {
        buf.resize(needed, 0);
    }
    buf.fill(0);
    composite_into(buf, w, h, stride, root, 0, 0);
    Some((w, h, stride))
}

/// Return the pixel dimensions of a surface's committed buffer (or fallback).
pub fn surface_dims(root_data: &SurfaceData, fallback: (u32, u32)) -> (u32, u32) {
    // Viewport destination size takes priority over buffer size.
    {
        let vp = root_data.viewport.lock().unwrap();
        if let Some((dw, dh)) = vp.dst {
            if dw > 0 && dh > 0 {
                return (dw as u32, dh as u32);
            }
        }
    }
    let cur = root_data.current_buffer.lock().unwrap().clone();
    let dims = cur
        .as_ref()
        .and_then(|b| b.data::<BufferKind>())
        .and_then(|k| match k {
            BufferKind::Shm(s) => Some((s.width, s.height)),
            BufferKind::Dmabuf(d) => Some((d.width, d.height)),
            BufferKind::SinglePixel(_) => Some((1, 1)),
            BufferKind::Invalid => None,
        });
    match dims {
        Some((w, h)) if w > 0 && h > 0 => (w as u32, h as u32),
        _ => fallback,
    }
}

/// Compose directly into a caller-owned pre-sized slice (e.g. a mmap).
/// Slice must be exactly `w * h * 4` bytes. Returns false if the focused
/// surface has no buffer or its dimensions don't match.
pub fn composite_into_slice(root: &WlSurface, output: &mut [u8], w: i32, h: i32) -> bool {
    if root.data::<SurfaceData>().is_none() {
        return false;
    }
    let stride = w * 4;
    let expected = (h as usize) * (stride as usize);
    if output.len() != expected {
        return false;
    }
    output.fill(0);
    composite_into(output, w, h, stride, root, 0, 0);
    true
}

pub(crate) fn composite_into(
    dst: &mut [u8],
    dst_w: i32,
    dst_h: i32,
    dst_stride: i32,
    surface: &WlSurface,
    off_x: i32,
    off_y: i32,
) {
    let Some(sd) = surface.data::<SurfaceData>() else {
        return;
    };
    // This surface sits below its own children — blit it first.
    let cur = sd.current_buffer.lock().unwrap().clone();
    if let Some(buf) = cur {
        if let Some(kind) = buf.data::<BufferKind>() {
            let vp = sd.viewport.lock().unwrap().clone();
            kind.with_bytes(|bytes, buf_w, buf_h, stride, format| {
                let (sx, sy, sw, sh) = if let Some((x, y, w, h)) = vp.src {
                    (x as i32, y as i32, w as i32, h as i32)
                } else {
                    (0, 0, buf_w, buf_h)
                };
                let (dw, dh) = if let Some((w, h)) = vp.dst {
                    (w, h)
                } else {
                    (sw, sh)
                };
                if sw == dw && sh == dh && sx == 0 && sy == 0 {
                    blit_over(
                        dst, dst_w, dst_h, dst_stride, off_x, off_y, bytes, buf_w, buf_h, stride,
                        format,
                    );
                } else {
                    blit_viewport(
                        dst, dst_w, dst_h, dst_stride, off_x, off_y, bytes, stride, format, sx, sy,
                        sw, sh, dw, dh,
                    );
                }
            });
        }
    }
    // Recurse into children in insertion order (bottom-to-top). We don't
    // track place_above / place_below, which means if an app reorders its
    // subsurfaces we render the wrong stack. Not an issue for firefox.
    let children = sd.children.lock().unwrap().clone();
    for child in children {
        composite_into(
            dst,
            dst_w,
            dst_h,
            dst_stride,
            &child.surface,
            off_x + child.x,
            off_y + child.y,
        );
    }
}

/// Source-over blit. `src` is ARGB8888 or XRGB8888 (little-endian ⇒ byte
/// order B, G, R, A). ARGB pixels are assumed pre-multiplied, which is
/// what Wayland guarantees for wl_shm buffers.
#[allow(clippy::too_many_arguments)]
fn blit_over(
    dst: &mut [u8],
    dst_w: i32,
    dst_h: i32,
    dst_stride: i32,
    off_x: i32,
    off_y: i32,
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: i32,
    format: WEnum<Format>,
) {
    if src_w <= 0 || src_h <= 0 {
        return;
    }
    let x0 = off_x.max(0);
    let y0 = off_y.max(0);
    let x1 = (off_x + src_w).min(dst_w);
    let y1 = (off_y + src_h).min(dst_h);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let is_argb = matches!(format, WEnum::Value(Format::Argb8888));

    for y in y0..y1 {
        let sy = y - off_y;
        let src_row = (sy as usize) * (src_stride as usize);
        let dst_row = (y as usize) * (dst_stride as usize);
        for x in x0..x1 {
            let sx = x - off_x;
            let si = src_row + (sx as usize) * 4;
            let di = dst_row + (x as usize) * 4;
            if si + 4 > src.len() || di + 4 > dst.len() {
                continue;
            }
            let sb = src[si];
            let sg = src[si + 1];
            let sr = src[si + 2];
            let sa = if is_argb { src[si + 3] } else { 0xFF };
            if sa == 0xFF {
                dst[di] = sb;
                dst[di + 1] = sg;
                dst[di + 2] = sr;
                dst[di + 3] = 0xFF;
            } else if sa != 0 {
                // Pre-multiplied src-over.
                let inv = 255 - sa as u32;
                let db = dst[di] as u32;
                let dg = dst[di + 1] as u32;
                let dr = dst[di + 2] as u32;
                let da = dst[di + 3] as u32;
                dst[di] = (sb as u32 + (db * inv) / 255).min(255) as u8;
                dst[di + 1] = (sg as u32 + (dg * inv) / 255).min(255) as u8;
                dst[di + 2] = (sr as u32 + (dr * inv) / 255).min(255) as u8;
                dst[di + 3] = (sa as u32 + (da * inv) / 255).min(255) as u8;
            }
            // sa == 0: leave dst unchanged (child is fully transparent).
        }
    }
}

/// Viewport blit: crop source rect (sx, sy, sw, sh) and scale it to (dw, dh)
/// at destination offset (off_x, off_y). Uses nearest-neighbour sampling.
#[allow(clippy::too_many_arguments)]
fn blit_viewport(
    dst: &mut [u8],
    dst_w: i32,
    dst_h: i32,
    dst_stride: i32,
    off_x: i32,
    off_y: i32,
    src: &[u8],
    src_stride: i32,
    format: WEnum<Format>,
    sx: i32,
    sy: i32,
    sw: i32,
    sh: i32,
    dw: i32,
    dh: i32,
) {
    if sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 {
        return;
    }
    let is_argb = matches!(format, WEnum::Value(Format::Argb8888));
    for dy in 0..dh {
        let src_y = sy + (dy * sh) / dh;
        let dst_y = off_y + dy;
        if dst_y < 0 || dst_y >= dst_h {
            continue;
        }
        for dx in 0..dw {
            let src_x = sx + (dx * sw) / dw;
            let dst_x = off_x + dx;
            if dst_x < 0 || dst_x >= dst_w {
                continue;
            }
            let si = (src_y as usize) * (src_stride as usize) + (src_x as usize) * 4;
            let di = (dst_y as usize) * (dst_stride as usize) + (dst_x as usize) * 4;
            if si + 4 > src.len() || di + 4 > dst.len() {
                continue;
            }
            let sb = src[si];
            let sg = src[si + 1];
            let sr = src[si + 2];
            let sa = if is_argb { src[si + 3] } else { 0xFF };
            if sa == 0xFF {
                dst[di] = sb;
                dst[di + 1] = sg;
                dst[di + 2] = sr;
                dst[di + 3] = 0xFF;
            } else if sa != 0 {
                let inv = 255 - sa as u32;
                let db = dst[di] as u32;
                let dg = dst[di + 1] as u32;
                let dr = dst[di + 2] as u32;
                let da = dst[di + 3] as u32;
                dst[di] = (sb as u32 + (db * inv) / 255).min(255) as u8;
                dst[di + 1] = (sg as u32 + (dg * inv) / 255).min(255) as u8;
                dst[di + 2] = (sr as u32 + (dr * inv) / 255).min(255) as u8;
                dst[di + 3] = (sa as u32 + (da * inv) / 255).min(255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solid 2x2 opaque ARGB8888 source on top of a zero dst: source wins.
    #[test]
    fn blit_opaque_argb_overwrites() {
        let mut dst = vec![0u8; 2 * 2 * 4];
        let src: Vec<u8> = vec![
            10, 20, 30, 255, 11, 21, 31, 255, 12, 22, 32, 255, 13, 23, 33, 255,
        ];
        blit_over(
            &mut dst,
            2,
            2,
            8,
            0,
            0,
            &src,
            2,
            2,
            8,
            WEnum::Value(Format::Argb8888),
        );
        assert_eq!(dst, src);
    }

    /// XRGB8888 source: high byte is "undefined" in the source but MUST be
    /// rendered opaque (0xff) into the ARGB destination, else clients see
    /// the toplevel as fully transparent.
    #[test]
    fn blit_xrgb_forces_opaque_alpha() {
        let mut dst = vec![0u8; 4];
        let src = vec![7, 8, 9, 0]; // alpha byte zero, but XRGB means ignore it
        blit_over(
            &mut dst,
            1,
            1,
            4,
            0,
            0,
            &src,
            1,
            1,
            4,
            WEnum::Value(Format::Xrgb8888),
        );
        assert_eq!(dst, [7, 8, 9, 0xFF]);
    }

    /// Fully-transparent ARGB child (alpha=0) must leave the destination
    /// pixel untouched — that's how firefox's transparent subsurface
    /// regions preserve the parent's content.
    #[test]
    fn blit_fully_transparent_argb_keeps_dst() {
        let mut dst = vec![100, 101, 102, 255];
        let src = vec![0, 0, 0, 0];
        blit_over(
            &mut dst,
            1,
            1,
            4,
            0,
            0,
            &src,
            1,
            1,
            4,
            WEnum::Value(Format::Argb8888),
        );
        assert_eq!(dst, [100, 101, 102, 255]);
    }

    /// Source placed at a positive offset lands in the correct destination
    /// pixel; pixels outside the source rect are untouched.
    #[test]
    fn blit_offset_lands_at_destination() {
        let mut dst = vec![0u8; 4 * 4 * 4];
        let src = vec![1, 2, 3, 255];
        blit_over(
            &mut dst,
            4,
            4,
            16,
            2,
            1,
            &src,
            1,
            1,
            4,
            WEnum::Value(Format::Argb8888),
        );
        // (2, 1) in a 4-wide buffer with stride 16 is byte index 1*16 + 2*4 = 24.
        assert_eq!(&dst[24..28], &[1, 2, 3, 255]);
        // Spot-check a non-target pixel stays zero.
        assert_eq!(&dst[0..4], &[0, 0, 0, 0]);
    }

    /// Negative offset + source spanning beyond destination: clipped on
    /// both sides, no panic, only in-range pixels written.
    #[test]
    fn blit_clips_at_boundaries() {
        let mut dst = vec![0u8; 2 * 2 * 4];
        // 4x4 source, placed at (-1, -1): only the source pixel at (1, 1)
        // maps to dst (0, 0); source (2, 1) → dst (1, 0); etc.
        let src = vec![9u8; 4 * 4 * 4];
        blit_over(
            &mut dst,
            2,
            2,
            8,
            -1,
            -1,
            &src,
            4,
            4,
            16,
            WEnum::Value(Format::Argb8888),
        );
        // All four dst pixels should be 9 (taken from src interior).
        assert!(dst.iter().all(|&b| b == 9));
    }

    #[test]
    fn blit_viewport_scale_2x_opaque() {
        // 1x1 opaque red pixel scaled to 2x2
        let src = vec![0u8, 0, 255, 255];
        let mut dst = vec![0u8; 2 * 2 * 4];
        blit_viewport(
            &mut dst,
            2,
            2,
            8,
            0,
            0,
            &src,
            4,
            WEnum::Value(Format::Argb8888),
            0,
            0,
            1,
            1,
            2,
            2,
        );
        for i in 0..4 {
            let b = i * 4;
            assert_eq!(dst[b], 0);
            assert_eq!(dst[b + 1], 0);
            assert_eq!(dst[b + 2], 255);
            assert_eq!(dst[b + 3], 255);
        }
    }

    #[test]
    fn blit_viewport_crop_top_left() {
        // 2x2 source: TL=red, rest=green. Crop to TL 1x1.
        let src = vec![
            0u8, 0, 255, 255, // TL red
            0, 255, 0, 255, // TR green
            0, 255, 0, 255, // BL green
            0, 255, 0, 255, // BR green
        ];
        let mut dst = vec![0u8; 4];
        blit_viewport(
            &mut dst,
            1,
            1,
            4,
            0,
            0,
            &src,
            8,
            WEnum::Value(Format::Argb8888),
            0,
            0,
            1,
            1,
            1,
            1,
        );
        assert_eq!(&dst, &[0, 0, 255, 255]);
    }
}
