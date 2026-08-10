# -*- coding: utf-8 -*-
"""Video metadata inspection and source timestamp extraction."""

from fractions import Fraction
import json
import subprocess

from config import MEDIAINFO, MKVEXTRACT, MKVMERGE


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


def extract_source_timecodes(source_path, output_path):
    """Extract v2 timestamps from the source MKV video track."""
    track_id = get_video_track_id(source_path)
    cmd = [MKVEXTRACT, str(source_path), "timestamps_v2", f"{track_id}:{output_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mkvextract timestamps failed: {result.stderr}")


def get_video_track_id(source_path):
    """Return the first video track ID reported by MKVToolNix."""
    result = subprocess.run(
        [MKVMERGE, "--identification-format", "json", "--identify", str(source_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mkvmerge identification failed: {result.stderr}")
    try:
        tracks = json.loads(result.stdout)["tracks"]
        return int(next(track["id"] for track in tracks if track["type"] == "video"))
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("mkvmerge returned no usable video track") from exc
