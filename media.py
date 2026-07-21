# -*- coding: utf-8 -*-
"""Video metadata inspection and source timestamp extraction."""

from fractions import Fraction
import json
import subprocess
import xml.etree.ElementTree as ET

from config import FFPROBE, MEDIAINFO, MKVEXTRACT


def get_video_info(source):
    """Read coded dimensions and sample aspect ratio from the first video track."""
    cmd = [MEDIAINFO, "--Output=JSON", str(source)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mediainfo failed: {result.stderr}")
    info = json.loads(result.stdout)
    track = next(t for t in info["media"]["track"] if t["@type"] == "Video")
    w = int(track["Width"])
    h = int(track["Height"])
    par_str = track.get("PixelAspectRatio", "1.000")
    par = Fraction(par_str).limit_denominator(100)
    return w, h, par.numerator, par.denominator


def get_video_frame_count(source):
    """Read the actual frame count of the first video track."""
    try:
        import vapoursynth as vs

        core = vs.core
        return int(core.bs.VideoSource(str(source)).num_frames)
    except Exception:
        pass

    cmd = [MEDIAINFO, "--Output=JSON", str(source)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mediainfo failed: {result.stderr}")
    info = json.loads(result.stdout)
    track = next(t for t in info["media"]["track"] if t["@type"] == "Video")
    frame_count = track.get("FrameCount")
    return int(frame_count) if frame_count else None


def extract_source_timecodes(source_path, output_path):
    """Extract v2 timestamps from the source MKV video track."""
    cmd = [MKVEXTRACT, str(source_path), "timestamps_v2", f"0:{output_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mkvextract timestamps failed: {result.stderr}")


def get_video_field_metadata(source_path, frame_count):
    """Read repeat_pict and top_field_first for each frame through ffprobe."""
    cmd = [
        FFPROBE,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=repeat_pict,top_field_first",
        "-of", "json",
        str(source_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe field metadata failed: {result.stderr}")
    frames = json.loads(result.stdout).get("frames", [])
    if len(frames) != frame_count:
        raise RuntimeError(
            f"Invalid field metadata cardinality: {len(frames)} records for {frame_count} frames"
        )

    metadata = []
    for index, frame in enumerate(frames):
        missing = [key for key in ("repeat_pict", "top_field_first") if key not in frame]
        if missing:
            raise RuntimeError(f"Missing field metadata at frame {index}: {', '.join(missing)}")
        repeat_pict = int(frame["repeat_pict"])
        top_field_first = int(frame["top_field_first"])
        if repeat_pict not in (0, 1):
            raise RuntimeError(f"Unsupported repeat_pict at frame {index}: {repeat_pict}")
        if top_field_first not in (0, 1):
            raise RuntimeError(
                f"Invalid top_field_first at frame {index}: {top_field_first}"
            )
        metadata.append({
            "repeat_pict": repeat_pict,
            "top_field_first": bool(top_field_first),
        })
    return metadata


def extract_chapter_ranges(source_path, output_path):
    """Extract Matroska chapters and return time ranges in milliseconds."""
    cmd = [MKVEXTRACT, str(source_path), "chapters", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mkvextract chapters failed: {result.stderr}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        return []

    tree = ET.parse(output_path)
    atoms = tree.findall(".//ChapterAtom")
    starts = []
    for atom in atoms:
        start_el = atom.find("ChapterTimeStart")
        if start_el is not None and start_el.text:
            starts.append(_chapter_timestamp_to_ms(start_el.text.strip()))
    starts = sorted(starts)
    return [(start, starts[i + 1] if i + 1 < len(starts) else None) for i, start in enumerate(starts)]


def _chapter_timestamp_to_ms(value):
    """Convert Matroska chapter timestamps to milliseconds."""
    hms, _, frac = value.partition(".")
    h, m, s = hms.split(":")
    frac = (frac + "000000000")[:9]
    return (
        int(h) * 3600000
        + int(m) * 60000
        + int(s) * 1000
        + int(round(int(frac) / 1000000.0))
    )
