# -*- coding: utf-8 -*-
"""Codifica video, mappatura dei metadata colore e mux finale MKV."""

import json
from pathlib import Path
import subprocess

from config import ENCODER_BIN, ENCODER_PARAMS, FFMPEG, MEDIAINFO, MKVMERGE, VSPIPE


_MATRIX_MAP = {
    "BT.601": "smpte170m",
    "BT.709": "bt709",
    "BT.470 System B/G": "bt470bg",
    "BT.2020 non-constant": "bt2020nc",
    "BT.2020 constant": "bt2020c",
}
_PRIMARIES_MAP = {
    "BT.601 NTSC": "smpte170m",
    "BT.601 PAL": "bt470bg",
    "BT.709": "bt709",
    "BT.2020": "bt2020",
    "DCI P3": "smpte431",
    "Display P3": "smpte432",
}
_TRANSFER_MAP = {
    "BT.601": "smpte170m",
    "BT.709": "bt709",
    "BT.470 System M": "bt470m",
    "BT.470 System B/G": "bt470bg",
    "PQ": "smpte2084",
    "HLG": "arib-std-b67",
    "sRGB": "iec61966-2-1",
}


def get_color_flags(source_path, is_ffmpeg):
    """Mappa i metadata colore sorgente sui flag dell'encoder selezionato."""
    result = subprocess.run([MEDIAINFO, "--Output=JSON", str(source_path)], capture_output=True, text=True)
    info = json.loads(result.stdout)
    vtrack = next((t for t in info["media"]["track"] if t["@type"] == "Video"), {})
    matrix_raw = vtrack.get("matrix_coefficients", "")
    primaries_raw = vtrack.get("colour_primaries", "")
    transfer_raw = vtrack.get("transfer_characteristics", "")
    matrix = _MATRIX_MAP.get(matrix_raw)
    primaries = _PRIMARIES_MAP.get(primaries_raw)
    transfer = _TRANSFER_MAP.get(transfer_raw)
    if matrix_raw == "BT.601" and (primaries_raw == "BT.601 PAL" or transfer_raw == "BT.470 System B/G"):
        matrix = "bt470bg"
    if not any([matrix, primaries, transfer]):
        return ""
    if is_ffmpeg:
        parts = (
            ([f"-colorspace {matrix}"] if matrix else [])
            + ([f"-color_primaries {primaries}"] if primaries else [])
            + ([f"-color_trc {transfer}"] if transfer else [])
        )
    else:
        parts = (
            ([f"--colormatrix {matrix}"] if matrix else [])
            + ([f"--colorprim {primaries}"] if primaries else [])
            + ([f"--transfer {transfer}"] if transfer else [])
        )
    return " ".join(parts)


def encode(vs_script, encoded_video, color_flags=""):
    """Invia l'output Y4M di VapourSynth all'encoder configurato."""
    vspipe_cmd = [VSPIPE, "-c", "y4m", str(vs_script), "-"]
    is_ffmpeg = "ffmpeg" in ENCODER_BIN.lower()
    params = ENCODER_PARAMS + (f" {color_flags}" if color_flags else "")
    if is_ffmpeg:
        enc_cmd = [ENCODER_BIN] + params.split() + [str(encoded_video)]
    else:
        enc_cmd = [ENCODER_BIN] + params.split() + ["-o", str(encoded_video)]
    print(f"  Encode: vspipe | {Path(ENCODER_BIN).stem}")
    vspipe_proc = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE)
    enc_proc = subprocess.Popen(enc_cmd, stdin=vspipe_proc.stdout)
    vspipe_proc.stdout.close()
    enc_proc.communicate()
    if enc_proc.returncode != 0:
        raise RuntimeError(f"Encoder fallito con codice {enc_proc.returncode}")


def mux_final(encoded_video, source_path, timecodes_path, output_mkv, strip_audio, strip_sub, audio_range=None):
    """Muxa video encodato, timecode VFR e tracce audio/sottotitoli sorgenti."""
    cmd = [
        MKVMERGE,
        "--output", str(output_mkv),
        "--timestamps", f"0:{timecodes_path}",
        "--no-audio",
        "--no-subtitles",
        "--no-chapters",
        str(encoded_video),
    ]
    if audio_range is not None:
        ss_ts, to_ts = audio_range
        trimmed_av = Path(encoded_video).parent / "trimmed_av.mkv"
        av_cmd = [
            FFMPEG,
            "-y", "-hide_banner", "-loglevel", "warning",
            "-ss", ss_ts,
            "-to", to_ts,
            "-i", str(source_path),
            "-map", "0:a?",
            "-map", "0:s?",
            "-c", "copy",
            "-vn",
            str(trimmed_av),
        ]
        print(f"  Trimming audio: {ss_ts} -> {to_ts}")
        result = subprocess.run(av_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio trim fallito (rc={result.returncode})")
        cmd.extend(["--no-video", "--no-chapters", str(trimmed_av)])
    else:
        source_flags = ["--no-video"]
        if strip_audio:
            source_flags.append("--no-audio")
        if strip_sub:
            source_flags.append("--no-subtitles")
        cmd.extend(source_flags)
        cmd.append(str(source_path))

    print(f"  Mux: -> {Path(output_mkv).name}")
    result = subprocess.run(cmd)
    if result.returncode > 1:
        raise RuntimeError(f"mkvmerge mux fallito (rc={result.returncode})")
    if audio_range is not None and trimmed_av.exists():
        trimmed_av.unlink()
