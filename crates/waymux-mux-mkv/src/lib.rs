// SPDX-License-Identifier: Apache-2.0

//! Streaming Matroska/MKV muxer for H.264 video.
//!
//! Designed to replace the ffmpeg subprocess that the legacy recording
//! path pipes raw frames into. The Vulkan zero-copy pipeline produces
//! H.264 NAL units in memory; this muxer accepts those units along with
//! a PTS and writes a valid MKV file to any `impl io::Write`.
//!
//! Single track only. No audio. SimpleBlock for all frames (no B-frames
//! from `VK_KHR_video_encode_h264` at baseline/main profile, so the
//! richer BlockGroup machinery is unnecessary).
//!
//! ## Usage
//!
//! ```ignore
//! use std::fs::File;
//! use std::io::BufWriter;
//! use waymux_mux_mkv::MkvWriter;
//!
//! let file = BufWriter::new(File::create("out.mkv")?);
//! let codec_private = build_avcc(&sps, &pps); // see codec_private_from_sps_pps
//! let mut mux = MkvWriter::new(file, 1920, 1080, &codec_private)?;
//! mux.write_frame(&keyframe_nal, 0, true)?;
//! mux.write_frame(&p_frame_nal, 17, false)?;
//! let buf = mux.finish()?;
//! ```

use std::io::{self, Seek, SeekFrom, Write};

pub mod ebml;

use ebml::{
    write_element_bytes, write_element_f64, write_element_id, write_element_size,
    write_element_str, write_element_u64, UNKNOWN_SIZE,
};

// EBML / Matroska element IDs we use. Sourced from
// https://www.matroska.org/technical/elements.html
const ID_EBML: u32 = 0x1A45_DFA3;
const ID_EBML_VERSION: u32 = 0x4286;
const ID_EBML_READ_VERSION: u32 = 0x42F7;
const ID_EBML_MAX_ID_LENGTH: u32 = 0x42F2;
const ID_EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
const ID_DOC_TYPE: u32 = 0x4282;
const ID_DOC_TYPE_VERSION: u32 = 0x4287;
const ID_DOC_TYPE_READ_VERSION: u32 = 0x4285;

const ID_SEGMENT: u32 = 0x1853_8067;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMECODE_SCALE: u32 = 0x002A_D7B1;
const ID_MUXING_APP: u32 = 0x4D80;
const ID_WRITING_APP: u32 = 0x5741;
const ID_DURATION: u32 = 0x4489;

const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0xAE;
const ID_TRACK_NUMBER: u32 = 0xD7;
const ID_TRACK_UID: u32 = 0x73C5;
const ID_TRACK_TYPE: u32 = 0x83;
const ID_FLAG_LACING: u32 = 0x9C;
const ID_CODEC_ID: u32 = 0x86;
const ID_CODEC_PRIVATE: u32 = 0x63A2;
const ID_VIDEO: u32 = 0xE0;
const ID_PIXEL_WIDTH: u32 = 0xB0;
const ID_PIXEL_HEIGHT: u32 = 0xBA;

const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_TIMECODE: u32 = 0xE7;
const ID_SIMPLE_BLOCK: u32 = 0xA3;

const TRACK_TYPE_VIDEO: u64 = 1;
const TRACK_NUMBER: u64 = 1;
const TIMECODE_SCALE_NS: u64 = 1_000_000; // 1 ms per timecode unit

const CLUSTER_DURATION_MS: i64 = 1000;

/// Streaming MKV muxer. Writes one H.264 video track.
///
/// Not thread-safe — call from a single recording thread.
///
/// `W: Write + Seek`. The `Seek` bound exists for one reason: on
/// `finish()` we seek back to the Info element's Duration field and
/// write the actual recording length. Strict demuxers (LosslessCut,
/// some pro NLE tools) refuse to read a file with `Duration=0.0`.
/// `BufWriter<File>` satisfies both bounds and is the typical
/// consumer.
pub struct MkvWriter<W: Write + Seek> {
    inner: W,
    cluster_open: bool,
    cluster_base_ms: i64,
    frames_total: u64,
    last_pts_ms: i64,
    /// Absolute byte offset of the Duration field's f64 value within
    /// the file. `finish()` seeks here and overwrites the placeholder
    /// 0.0 with the real duration in ms.
    duration_value_offset: u64,
}

impl<W: Write + Seek> MkvWriter<W> {
    /// Write the EBML header + Segment + SeekHead + Info + Tracks
    /// preamble. After this returns, the writer is ready for
    /// `write_frame`.
    ///
    /// `codec_private` is the H.264 `AVCDecoderConfigurationRecord`
    /// (ISO 14496-15 §5.3.3.1) built from the encoder's SPS+PPS.
    pub fn new(inner: W, width: u32, height: u32, codec_private: &[u8]) -> io::Result<Self> {
        Self::new_for_codec(inner, width, height, codec_private, "V_MPEG4/ISO/AVC")
    }

    /// Like [`MkvWriter::new`] but lets the caller pick the Matroska codec
    /// ID. Use `"V_MPEG4/ISO/AVC"` for H.264 (default), `"V_FFV1"` for
    /// libav's FFV1 output, etc. `codec_private` is the codec's
    /// CodecPrivate bytes (avcC for H.264; extradata as-is for FFV1).
    pub fn new_for_codec(
        mut inner: W,
        width: u32,
        height: u32,
        codec_private: &[u8],
        codec_id: &str,
    ) -> io::Result<Self> {
        write_ebml_header(&mut inner)?;
        write_segment_start(&mut inner)?;
        let duration_value_offset = write_info(&mut inner)?;
        write_tracks(&mut inner, width, height, codec_private, codec_id)?;
        Ok(Self {
            inner,
            cluster_open: false,
            cluster_base_ms: 0,
            frames_total: 0,
            last_pts_ms: 0,
            duration_value_offset,
        })
    }

    /// Write one encoded frame. `pts_ms` is milliseconds from
    /// recording start. Opens a new Cluster every CLUSTER_DURATION_MS
    /// of PTS so seeking remains tractable.
    ///
    /// `nal_data` should be the H.264 byte stream in Annex B format
    /// (start codes 00 00 00 01 between NAL units). Matroska's
    /// SimpleBlock holds the bytes verbatim.
    pub fn write_frame(
        &mut self,
        nal_data: &[u8],
        pts_ms: i64,
        is_keyframe: bool,
    ) -> io::Result<()> {
        // Open a new Cluster on the first frame or when the PTS exceeds
        // the current cluster's window. Each cluster is unknown-size
        // (streaming-friendly).
        let need_new_cluster = !self.cluster_open
            || (pts_ms - self.cluster_base_ms) > CLUSTER_DURATION_MS
            // Always start a new cluster on a keyframe — makes seeking
            // robust and matches ffmpeg's mkv mux behavior.
            || (is_keyframe && (pts_ms - self.cluster_base_ms) > 50);
        if need_new_cluster {
            write_cluster_start(&mut self.inner, pts_ms)?;
            self.cluster_open = true;
            self.cluster_base_ms = pts_ms;
        }
        let delta = (pts_ms - self.cluster_base_ms) as i16;
        write_simple_block(&mut self.inner, TRACK_NUMBER, delta, is_keyframe, nal_data)?;
        self.frames_total += 1;
        self.last_pts_ms = pts_ms;
        Ok(())
    }

    /// Flush, patch the Duration field with the real recording length,
    /// and return the underlying writer.
    ///
    /// Closing the Segment itself would require seeking back to the
    /// Segment header to write a known size; we always use UNKNOWN_SIZE
    /// for Segment + Cluster so streaming output works without seek-back
    /// there. The Duration field is the only thing we MUST seek back to
    /// patch — strict demuxers (LosslessCut, some pro NLEs) refuse to
    /// open a file with `Duration=0.0`.
    pub fn finish(mut self) -> io::Result<W> {
        // Compute final duration: last PTS plus an estimated single-frame
        // budget so the duration spans the last frame's display window.
        // The exact value matters less than "non-zero, larger than last PTS".
        let duration_ms = self.last_pts_ms.max(0) as f64 + 16.0;
        // Seek + overwrite the placeholder f64 in the Info block.
        self.inner
            .seek(SeekFrom::Start(self.duration_value_offset))?;
        self.inner.write_all(&duration_ms.to_be_bytes())?;
        // Restore the file pointer to end-of-file so any later writes
        // (e.g. the caller's own appended metadata) continue correctly.
        self.inner.seek(SeekFrom::End(0))?;
        self.inner.flush()?;
        Ok(self.inner)
    }

    /// Frame count written so far. Useful for verifying tests.
    pub fn frames_total(&self) -> u64 {
        self.frames_total
    }
}

fn write_ebml_header<W: Write>(w: &mut W) -> io::Result<()> {
    // EBML element body. Compute it into a buffer first so we can write
    // its exact size (the EBML header is small and known-size).
    let mut body = Vec::with_capacity(64);
    write_element_u64(&mut body, ID_EBML_VERSION, 1)?;
    write_element_u64(&mut body, ID_EBML_READ_VERSION, 1)?;
    write_element_u64(&mut body, ID_EBML_MAX_ID_LENGTH, 4)?;
    write_element_u64(&mut body, ID_EBML_MAX_SIZE_LENGTH, 8)?;
    write_element_str(&mut body, ID_DOC_TYPE, "matroska")?;
    write_element_u64(&mut body, ID_DOC_TYPE_VERSION, 4)?;
    write_element_u64(&mut body, ID_DOC_TYPE_READ_VERSION, 2)?;
    write_element_id(w, ID_EBML)?;
    write_element_size(w, body.len() as u64)?;
    w.write_all(&body)
}

fn write_segment_start<W: Write>(w: &mut W) -> io::Result<()> {
    write_element_id(w, ID_SEGMENT)?;
    write_element_size(w, UNKNOWN_SIZE)
}

/// Write the Info element. Returns the ABSOLUTE file offset of the
/// Duration field's 8-byte f64 value, so the caller can seek back to
/// it on `finish()` and patch in the real recording length.
fn write_info<W: Write + Seek>(w: &mut W) -> io::Result<u64> {
    // Build the Info body in a Vec so we can compute its size before
    // writing the element header. Track where the Duration value
    // lands within the body.
    let mut body = Vec::with_capacity(128);
    write_element_u64(&mut body, ID_TIMECODE_SCALE, TIMECODE_SCALE_NS)?;
    write_element_str(&mut body, ID_MUXING_APP, "waymux-mux-mkv")?;
    write_element_str(&mut body, ID_WRITING_APP, "waymux-mux-mkv")?;
    // The Duration element layout is:
    //   2 bytes  ID 0x4489
    //   1 byte   size VINT (0x88 = 8-byte data)
    //   8 bytes  f64 value
    // The value's offset within the body is (body.len() + 3) at this point.
    let duration_value_offset_in_body = body.len() + 3;
    write_element_f64(&mut body, ID_DURATION, 0.0)?;

    let info_header_pos = w.stream_position()?;
    write_element_id(w, ID_INFO)?;
    write_element_size(w, body.len() as u64)?;
    let body_start = w.stream_position()?;
    let _ = info_header_pos;
    w.write_all(&body)?;
    Ok(body_start + duration_value_offset_in_body as u64)
}

fn write_tracks<W: Write>(
    w: &mut W,
    width: u32,
    height: u32,
    codec_private: &[u8],
    codec_id: &str,
) -> io::Result<()> {
    // Video sub-element.
    let mut video_body = Vec::with_capacity(32);
    write_element_u64(&mut video_body, ID_PIXEL_WIDTH, width as u64)?;
    write_element_u64(&mut video_body, ID_PIXEL_HEIGHT, height as u64)?;

    // TrackEntry body.
    let mut track_body = Vec::with_capacity(128 + codec_private.len());
    write_element_u64(&mut track_body, ID_TRACK_NUMBER, TRACK_NUMBER)?;
    write_element_u64(&mut track_body, ID_TRACK_UID, 0xDEAD_BEEF)?;
    write_element_u64(&mut track_body, ID_TRACK_TYPE, TRACK_TYPE_VIDEO)?;
    write_element_u64(&mut track_body, ID_FLAG_LACING, 0)?;
    write_element_str(&mut track_body, ID_CODEC_ID, codec_id)?;
    write_element_bytes(&mut track_body, ID_CODEC_PRIVATE, codec_private)?;
    write_element_id(&mut track_body, ID_VIDEO)?;
    write_element_size(&mut track_body, video_body.len() as u64)?;
    track_body.extend_from_slice(&video_body);

    // Tracks body.
    let mut tracks_body = Vec::with_capacity(track_body.len() + 8);
    write_element_id(&mut tracks_body, ID_TRACK_ENTRY)?;
    write_element_size(&mut tracks_body, track_body.len() as u64)?;
    tracks_body.extend_from_slice(&track_body);

    write_element_id(w, ID_TRACKS)?;
    write_element_size(w, tracks_body.len() as u64)?;
    w.write_all(&tracks_body)
}

fn write_cluster_start<W: Write>(w: &mut W, timecode_ms: i64) -> io::Result<()> {
    write_element_id(w, ID_CLUSTER)?;
    write_element_size(w, UNKNOWN_SIZE)?;
    write_element_u64(w, ID_TIMECODE, timecode_ms.max(0) as u64)
}

fn write_simple_block<W: Write>(
    w: &mut W,
    track_number: u64,
    pts_delta: i16,
    is_keyframe: bool,
    payload: &[u8],
) -> io::Result<()> {
    // SimpleBlock wire layout:
    //   VINT track_number
    //   i16  timecode delta (relative to Cluster Timecode)
    //   u8   flags (bit 7 = keyframe, bit 0 = discardable)
    //   ...  payload (raw NAL units, Annex B)
    let mut header = [0u8; 12];
    let mut hbuf = &mut header[..];
    let mut hlen: usize;
    {
        let mut tmp = Vec::with_capacity(2);
        ebml::write_vint(&mut tmp, track_number)?;
        hbuf[..tmp.len()].copy_from_slice(&tmp);
        hlen = tmp.len();
    }
    hbuf = &mut header[hlen..];
    let delta_be = pts_delta.to_be_bytes();
    hbuf[..2].copy_from_slice(&delta_be);
    hbuf[2] = if is_keyframe { 0x80 } else { 0x00 };
    hlen += 3;

    let body_len = hlen as u64 + payload.len() as u64;
    write_element_id(w, ID_SIMPLE_BLOCK)?;
    write_element_size(w, body_len)?;
    w.write_all(&header[..hlen])?;
    w.write_all(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Minimal synthetic SPS/PPS-equivalent AVCDecoderConfigurationRecord
    /// for round-trip tests. Real recordings build this from the encoder's
    /// session parameters. This blob is just enough bytes that some
    /// demuxers don't refuse to parse the track.
    fn fake_avcc() -> Vec<u8> {
        // configurationVersion=1, profile=0x42 (Baseline), profile_compat=0x00,
        // level=0x1E, lengthSizeMinusOne=3, numSPS=0, numPPS=0.
        vec![0x01, 0x42, 0x00, 0x1E, 0xFF, 0xE0, 0x00, 0xFC, 0x00]
    }

    #[test]
    fn mux_writes_ebml_magic() {
        let mut buf = Vec::new();
        let _mux = MkvWriter::new(Cursor::new(&mut buf), 1920, 1080, &fake_avcc()).unwrap();
        // EBML header magic: 1A 45 DF A3
        assert_eq!(&buf[0..4], &[0x1A, 0x45, 0xDF, 0xA3]);
    }

    #[test]
    fn mux_writes_segment_after_ebml() {
        let mut buf = Vec::new();
        let _mux = MkvWriter::new(Cursor::new(&mut buf), 1920, 1080, &fake_avcc()).unwrap();
        // Find the Segment ID 0x18 0x53 0x80 0x67 somewhere after the EBML
        // header (must appear before any other content).
        let segment_marker = [0x18, 0x53, 0x80, 0x67];
        assert!(
            buf.windows(4).any(|w| w == segment_marker),
            "no Segment element found in output"
        );
    }

    #[test]
    fn frames_total_counter() {
        let mut buf = Vec::new();
        let mut mux = MkvWriter::new(Cursor::new(&mut buf), 1280, 720, &fake_avcc()).unwrap();
        let frame = vec![0u8; 64];
        mux.write_frame(&frame, 0, true).unwrap();
        mux.write_frame(&frame, 16, false).unwrap();
        mux.write_frame(&frame, 33, false).unwrap();
        assert_eq!(mux.frames_total(), 3);
    }

    #[test]
    fn keyframe_flag_in_simple_block() {
        // Round-trip: scan the output for the SimpleBlock element ID and
        // verify the flags byte after track-number-VINT + 2-byte delta
        // has the keyframe bit set.
        let mut buf = Vec::new();
        let mut mux = MkvWriter::new(Cursor::new(&mut buf), 640, 360, &fake_avcc()).unwrap();
        let frame = b"\x00\x00\x00\x01\x67\x42\x00\x1E"; // fake NAL unit
        mux.write_frame(frame, 0, true).unwrap();
        // Find the SimpleBlock element (ID 0xA3). After:
        //   1 byte: 0xA3
        //   N bytes: size VINT
        //   1 byte: track number VINT (we use track 1 → 0x81)
        //   2 bytes: pts delta
        //   1 byte: flags
        // we should see the keyframe bit (0x80).
        // The element ID 0xA3 also collides with other byte values, but
        // we know SimpleBlocks come last in the file; find the LAST 0xA3.
        let sb_pos = buf
            .iter()
            .rposition(|&b| b == 0xA3)
            .expect("no SimpleBlock byte");
        // Skip ID (1) + size VINT (read it).
        let size_byte = buf[sb_pos + 1];
        let size_vint_len = size_byte.leading_zeros() as usize + 1;
        let header_off = sb_pos + 1 + size_vint_len;
        // Track number VINT (we use track 1, so 1 byte: 0x81).
        let track_vint_len = 1;
        let flags_off = header_off + track_vint_len + 2;
        let flags = buf[flags_off];
        assert_eq!(
            flags & 0x80,
            0x80,
            "keyframe bit not set in SimpleBlock flags byte (flags = 0x{flags:02X})"
        );
    }

    #[test]
    fn multiple_clusters_for_long_recording() {
        // Recording spans 5 seconds; should produce > 1 cluster (cluster
        // duration is 1 second).
        let mut buf = Vec::new();
        let mut mux = MkvWriter::new(Cursor::new(&mut buf), 320, 240, &fake_avcc()).unwrap();
        for i in 0..50 {
            let pts_ms = i * 100; // 100 ms apart → 50 frames over 5 s
            let keyframe = i == 0;
            mux.write_frame(b"frame", pts_ms, keyframe).unwrap();
        }
        // Cluster magic 1F 43 B6 75
        let cluster_count = buf
            .windows(4)
            .filter(|w| w == &[0x1F, 0x43, 0xB6, 0x75])
            .count();
        assert!(
            cluster_count >= 5,
            "expected ≥5 clusters in a 5s recording, got {cluster_count}"
        );
    }

    /// Round-trip the muxer output through ffprobe if available on the
    /// host. ffprobe is a strong "is this a real MKV?" check; it bails
    /// on malformed EBML / element framing.
    #[test]
    fn ffprobe_accepts_output() {
        use std::process::Command;
        if Command::new("ffprobe").arg("-version").output().is_err() {
            eprintln!("ffprobe not available; skipping");
            return;
        }
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut mux = MkvWriter::new(file, 1280, 720, &fake_avcc()).unwrap();
            // Synthetic NAL unit — ffprobe will report codec from the
            // CodecID alone even if the slice data is not valid H.264.
            for i in 0..10 {
                mux.write_frame(b"\x00\x00\x00\x01\x67\x42\x00\x1E", i * 16, i == 0)
                    .unwrap();
            }
            mux.finish().unwrap();
        }
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "default=noprint_wrappers=1",
                path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stdout.contains("codec_name=h264"),
            "ffprobe did not see h264 codec.\nstdout: {stdout}\nstderr: {stderr}"
        );
        assert!(stdout.contains("width=1280"));
        assert!(stdout.contains("height=720"));
    }
}
