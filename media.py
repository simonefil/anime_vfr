# -*- coding: utf-8 -*-
"""Lettura metadata video ed estrazione dei timestamp sorgente."""

from fractions import Fraction
import json
import subprocess
import xml.etree.ElementTree as ET

from config import MEDIAINFO, MKVEXTRACT


def get_video_info(source):
    """Legge dimensioni codificate e sample aspect ratio della prima traccia video."""
    cmd = [MEDIAINFO, "--Output=JSON", str(source)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mediainfo fallito: {result.stderr}")
    info = json.loads(result.stdout)
    track = next(t for t in info["media"]["track"] if t["@type"] == "Video")
    w = int(track["Width"])
    h = int(track["Height"])
    par_str = track.get("PixelAspectRatio", "1.000")
    par = Fraction(par_str).limit_denominator(100)
    return w, h, par.numerator, par.denominator


def get_video_frame_count(source):
    """Legge il numero reale di frame della prima traccia video."""
    try:
        import vapoursynth as vs

        core = vs.core
        return int(core.bs.VideoSource(str(source)).num_frames)
    except Exception:
        pass

    cmd = [MEDIAINFO, "--Output=JSON", str(source)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mediainfo fallito: {result.stderr}")
    info = json.loads(result.stdout)
    track = next(t for t in info["media"]["track"] if t["@type"] == "Video")
    frame_count = track.get("FrameCount")
    return int(frame_count) if frame_count else None


def calc_square_pixel_res(w, h, sar_num, sar_den):
    """Converte la risoluzione campionata in risoluzione a pixel quadrati."""
    if sar_num == sar_den:
        return w, h
    dar = (w * sar_num) / (h * sar_den)
    square_w = round(h * dar / 2) * 2
    return square_w, h


def extract_source_timecodes(source_path, output_path):
    """Estrae i timestamps_v2 della traccia video da un MKV sorgente."""
    cmd = [MKVEXTRACT, str(source_path), "timestamps_v2", f"0:{output_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mkvextract timestamps fallito: {result.stderr}")


def extract_chapter_ranges(source_path, output_path):
    """Estrae i capitoli Matroska e restituisce range temporali in millisecondi."""
    cmd = [MKVEXTRACT, str(source_path), "chapters", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mkvextract chapters fallito: {result.stderr}")
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
    """Converte hh:mm:ss.nnnnnnnnn dei capitoli Matroska in millisecondi."""
    hms, _, frac = value.partition(".")
    h, m, s = hms.split(":")
    frac = (frac + "000000000")[:9]
    return (
        int(h) * 3600000
        + int(m) * 60000
        + int(s) * 1000
        + int(round(int(frac) / 1000000.0))
    )
