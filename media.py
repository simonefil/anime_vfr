# -*- coding: utf-8 -*-
"""Lettura metadata video ed estrazione dei timestamp sorgente."""

from fractions import Fraction
import json
import subprocess

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
