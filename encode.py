# -*- coding: utf-8 -*-
"""Video encoding, color metadata mapping, and final MKV muxing."""

import json
import shlex
import subprocess
from pathlib import Path

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
    """Map source color metadata to flags for the selected encoder."""
    result = subprocess.run(
        [MEDIAINFO, "--Output=JSON", str(source_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mediainfo failed: {result.stderr.strip()}")
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("mediainfo returned invalid JSON") from exc
    vtrack = next((t for t in info["media"]["track"] if t["@type"] == "Video"), {})
    matrix_raw = vtrack.get("matrix_coefficients", "")
    primaries_raw = vtrack.get("colour_primaries", "")
    transfer_raw = vtrack.get("transfer_characteristics", "")
    matrix = _MATRIX_MAP.get(matrix_raw)
    primaries = _PRIMARIES_MAP.get(primaries_raw)
    transfer = _TRANSFER_MAP.get(transfer_raw)
    if matrix_raw == "BT.601" and (
        primaries_raw == "BT.601 PAL" or transfer_raw == "BT.470 System B/G"
    ):
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


def get_par_flags(sar_num, sar_den, is_ffmpeg):
    """Map the source sample aspect ratio to selected-encoder flags."""
    if sar_num == sar_den:
        return ""
    if is_ffmpeg:
        return f"-vf setsar={sar_num}/{sar_den}"
    return f"--sar {sar_num}:{sar_den}"


def get_chroma_flags(output_yuv444, is_ffmpeg):
    """Map the requested output chroma format to selected-encoder flags."""
    if is_ffmpeg:
        return "-pix_fmt yuv444p10le" if output_yuv444 else "-pix_fmt yuv420p10le"
    if output_yuv444:
        return "--output-csp yuv444 --profile main444"
    return "--output-csp yuv420 --profile main10"


def encode(vs_script, encoded_video, color_flags="", par_flags="", chroma_flags=""):
    """Pipe VapourSynth Y4M output to the configured encoder."""
    vspipe_cmd = [VSPIPE, "-c", "y4m", str(vs_script), "-"]
    is_ffmpeg = "ffmpeg" in ENCODER_BIN.lower()
    params = shlex.split(ENCODER_PARAMS)
    for extra_flags in (color_flags, par_flags, chroma_flags):
        if extra_flags:
            params.extend(shlex.split(extra_flags))
    if is_ffmpeg:
        enc_cmd = [ENCODER_BIN] + params + [str(encoded_video)]
    else:
        enc_cmd = [ENCODER_BIN] + params + ["-o", str(encoded_video)]
    print(f"  Encode: vspipe | {Path(ENCODER_BIN).stem}")
    vspipe_proc = None
    encoder_proc = None
    try:
        vspipe_proc = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE)
        encoder_proc = subprocess.Popen(enc_cmd, stdin=vspipe_proc.stdout)
        vspipe_proc.stdout.close()
        encoder_return_code = encoder_proc.wait()
        vspipe_return_code = vspipe_proc.wait()
    finally:
        for process in (encoder_proc, vspipe_proc):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
    if encoder_return_code != 0:
        raise RuntimeError(f"Encoder failed with exit code {encoder_return_code}")
    if vspipe_return_code != 0:
        raise RuntimeError(f"vspipe failed with exit code {vspipe_return_code}")


def mux_final(
    encoded_video,
    source_path,
    timecodes_path,
    output_mkv,
    strip_audio,
    strip_sub,
    audio_range=None,
    default_duration=None,
    display_aspect_ratio=None,
):
    """Mux encoded video and source tracks with VFR timecodes or CFR duration."""
    cmd = [
        MKVMERGE,
        "--output",
        str(output_mkv),
    ]
    if timecodes_path is not None:
        cmd.extend(["--timestamps", f"0:{timecodes_path}"])
    if default_duration is not None:
        cmd.extend(["--default-duration", f"0:{default_duration}"])
    if display_aspect_ratio is not None:
        cmd.extend(["--aspect-ratio", f"0:{display_aspect_ratio}"])
    cmd.extend(
        [
            "--no-audio",
            "--no-subtitles",
            "--no-chapters",
            str(encoded_video),
        ]
    )
    trimmed_av = None
    try:
        if audio_range is not None and not (strip_audio and strip_sub):
            ss_ts, to_ts = audio_range
            trimmed_av = Path(encoded_video).parent / "trimmed_av.mkv"
            av_cmd = [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-ss",
                ss_ts,
            ]
            if to_ts is not None:
                av_cmd.extend(["-to", to_ts])
            av_cmd.extend(["-i", str(source_path)])
            if not strip_audio:
                av_cmd.extend(["-map", "0:a?"])
            if not strip_sub:
                av_cmd.extend(["-map", "0:s?"])
            av_cmd.extend(["-c", "copy", "-vn", str(trimmed_av)])
            print(f"  Trimming source tracks: {ss_ts} -> {to_ts or 'end'}")
            result = subprocess.run(av_cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg source-track trim failed (rc={result.returncode})"
                )
            cmd.extend(["--no-video", "--no-chapters", str(trimmed_av)])
        elif audio_range is None:
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
            raise RuntimeError(f"mkvmerge mux failed (rc={result.returncode})")
    finally:
        if trimmed_av is not None and trimmed_av.exists():
            trimmed_av.unlink()
