# -*- coding: utf-8 -*-
"""Generate VapourSynth scripts for IVTC Vulkan analysis and reconstruction."""

import csv
import re
import subprocess
from pathlib import Path

from config import VSPIPE
from video_source import render_video_source_call


VPY_FMTC_HELPERS = '''\
def fmtc_to_yuv420p8(src):
    src = core.fmtc.bitdepth(src, bits=16)
    src = core.fmtc.resample(src, csp=vs.YUV420P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(src, bits=8)

def fmtc_prepare_output(src, width=None, height=None, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, output_yuv444=False):
    src = core.fmtc.bitdepth(src, bits=16)
    output_format = vs.YUV444P16 if output_yuv444 else vs.YUV420P16
    if output_yuv444:
        src = core.fmtc.resample(src, csp=output_format, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    if crop_left or crop_right or crop_top or crop_bottom:
        src = core.std.CropRel(src, left=crop_left, right=crop_right, top=crop_top, bottom=crop_bottom)
    if width is not None and height is not None:
        src = core.fmtc.resample(src, w=width, h=height, csp=output_format, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    elif not output_yuv444:
        src = core.fmtc.resample(src, csp=output_format, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(src, bits=10)
'''


def _render_qtgmc_assignment(source_name, target_name, tff, opencl):
    return f'''{target_name} = QTempGaussMC(
    {source_name},
    basic_bobber=NNEDI3(nsize=4, nns=4, qual=2, opencl={opencl!r}),
    tff={tff!r},
    basic_tr=3,
    final_tr=2,
    source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED,
).deinterlace()'''


def render_qtgmc_bob(source_name, target_name, tff, backend):
    """Render a QTGMC Bob assignment with the selected NNEDI3 backend."""
    if backend not in ("auto", "opencl", "cpu"):
        raise ValueError(f"Unsupported Bob backend: {backend}")
    if backend == "opencl":
        return _render_qtgmc_assignment(source_name, target_name, tff, True) + "\n"
    if backend == "cpu":
        return _render_qtgmc_assignment(source_name, target_name, tff, False) + "\n"
    opencl = _render_qtgmc_assignment(source_name, target_name, tff, True)
    cpu = _render_qtgmc_assignment(source_name, target_name, tff, False)
    return (
        "try:\n"
        + "\n".join(f"    {line}" for line in opencl.splitlines())
        + f"\n    {target_name}.get_frame(0)\n"
        + "except (vs.Error, AttributeError):\n"
        + "\n".join(f"    {line}" for line in cpu.splitlines())
        + "\n"
    )


def _plugin_load_line(plugin_path):
    return f"core.std.LoadPlugin(path={str(plugin_path)!r})\n" if plugin_path else ""


def artifact_paths(work_dir, stem, source_filter, field_order):
    """Return the analysis TSV and generated VapourSynth artifacts."""
    prefix = f"{stem}_{source_filter}_{field_order}"
    return {
        "decisions": work_dir / f"{prefix}_ivtc.tsv",
        "analysis_vpy": work_dir / f"{prefix}_ivtc_analyze.vpy",
        "vpy": work_dir / f"{prefix}_ivtcvk.vpy",
        "timecodes_full": work_dir / f"{prefix}_tc_full.txt",
        "timecodes_final": work_dir / f"{prefix}_tc_final.txt",
    }


def _write_analysis_script(
    source_path,
    source_filter,
    source_threads,
    field_order,
    threads,
    backend,
    device,
    plugin_path,
    batch,
    decisions_path,
    script_path,
    overwrite,
):
    cache_path = script_path.parent / f"{source_path.stem}_ffms2.ffindex"
    source_call = render_video_source_call(
        source_path,
        source_filter,
        cache_path=cache_path,
        threads=source_threads,
    )
    backend_code = {"auto": 0, "vulkan": 1, "cpu": 2}[backend]
    tff = 1 if field_order == "tff" else 0
    fieldbased = 2 if tff else 1
    script_path.write_text(
        f'''import vapoursynth as vs

core = vs.core
core.num_threads = {threads}
{_plugin_load_line(plugin_path)}\
{VPY_FMTC_HELPERS}
source = {source_call}
source = core.std.SetFrameProps(source, _FieldBased={fieldbased})
source = fmtc_to_yuv420p8(source)
core.ivtcvk.AnalyzeTrack(
    source,
    decisions={str(decisions_path)!r},
    backend={backend_code},
    device={device},
    batch={batch},
    tff={tff},
    overwrite={int(overwrite)},
).set_output(0)
''',
        encoding="utf-8",
    )


def ensure_analysis(
    source_path,
    work_dir,
    source_filter,
    source_threads,
    field_order,
    threads,
    backend="auto",
    device=0,
    batch=512,
    plugin_path=None,
    force=False,
):
    """Create or reuse the single editable IVTC track TSV."""
    paths = artifact_paths(work_dir, source_path.stem, source_filter, field_order)
    if paths["decisions"].exists() and not force:
        print(f"  IVTC Vulkan: reusing {paths['decisions'].name}")
        return paths
    _write_analysis_script(
        source_path,
        source_filter,
        source_threads,
        field_order,
        threads,
        backend,
        device,
        plugin_path,
        batch,
        paths["decisions"],
        paths["analysis_vpy"],
        force,
    )
    print(
        f"  IVTC analysis: {source_filter}, {field_order}, "
        f"backend {backend}, device {device}"
    )
    evaluate_vpy_info(paths["analysis_vpy"])
    return paths


def write_reconstruct_script(
    source_path,
    source_filter,
    source_threads,
    field_order,
    plugin_path,
    decisions_path,
    timecodes_path,
    script_path,
    resize_target,
    crop_margins,
    output_yuv444,
    threads,
    bob_backend,
    config_path=None,
    additional_vpy=None,
    frame_range=None,
):
    """Write the final IVTC Vulkan VPY used by both VSPipe info and encode."""
    cache_path = script_path.parent / f"{source_path.stem}_ffms2.ffindex"
    source_call = render_video_source_call(
        source_path,
        source_filter,
        cache_path=cache_path,
        threads=source_threads,
    )
    tff = field_order == "tff"
    fieldbased = 2 if tff else 1
    crop_left, crop_right, crop_top, crop_bottom = crop_margins or (0, 0, 0, 0)
    width, height = resize_target if resize_target is not None else (None, None)
    config_argument = f", config={str(config_path)!r}" if config_path else ""
    script = f'''import vapoursynth as vs
from vsdeinterlace.qtgmc import QTempGaussMC
from vsaa import NNEDI3

core = vs.core
core.num_threads = {threads}
{_plugin_load_line(plugin_path)}\
{VPY_FMTC_HELPERS}
source = {source_call}
source = core.std.SetFrameProp(source, prop="_FieldBased", intval={fieldbased})
source = fmtc_to_yuv420p8(source)
'''
    script += render_qtgmc_bob("source", "bob", tff, bob_backend)
    script += f'''clip = core.ivtcvk.Reconstruct(
    source,
    decisions={str(decisions_path)!r},
    bob=bob,
    timecodes={str(timecodes_path)!r}{config_argument},
)
clip = fmtc_prepare_output(
    clip,
    width={width!r},
    height={height!r},
    crop_left={crop_left},
    crop_right={crop_right},
    crop_top={crop_top},
    crop_bottom={crop_bottom},
    output_yuv444={output_yuv444!r},
)
'''
    if additional_vpy is not None:
        script += "\n" + Path(additional_vpy).read_text(encoding="utf-8") + "\n"
    if frame_range is not None:
        script += f"clip = clip[{frame_range[0]}:{frame_range[1]}]\n"
    script += '''clip = core.std.AssumeFPS(clip, fpsnum=30000, fpsden=1001)
clip.set_output(0)
'''
    script_path.write_text(script, encoding="utf-8")


def evaluate_vpy_info(script_path):
    """Evaluate a generated VPY and return its output frame count."""
    result = subprocess.run(
        [VSPIPE, "--info", str(script_path)],
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise RuntimeError(f"VapourSynth script evaluation failed:\n{output.strip()}")
    match = re.search(r"^Frames:\s+(\d+)\s*$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"VSPipe did not report an output frame count:\n{output.strip()}")
    return int(match.group(1))


def read_timecodes(path):
    """Read Matroska timecode v2 frame starts."""
    return [
        float(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def trim_timecodes(full_path, final_path, frame_range):
    """Trim and rebase full output timecodes and return source-time bounds."""
    values = read_timecodes(full_path)
    if not values:
        raise RuntimeError(f"Empty IVTC Vulkan timecode file: {full_path}")
    if frame_range is None:
        if final_path != full_path:
            final_path.write_text(full_path.read_text(encoding="utf-8"), encoding="utf-8")
        return final_path, len(values), None
    start, requested_end = frame_range
    end = min(requested_end, len(values))
    if start >= end:
        raise ValueError(f"Output frame range {start}-{requested_end} is empty")
    selected = values[start:end]
    origin = selected[0]
    rebased = [value - origin for value in selected]
    final_path.write_text(
        "# timecode format v2\n" + "".join(f"{value:.6f}\n" for value in rebased),
        encoding="utf-8",
    )
    end_timestamp = _timestamp(values[end]) if end < len(values) else None
    return final_path, len(selected), (_timestamp(values[start]), end_timestamp)


def _timestamp(milliseconds):
    seconds = milliseconds / 1000.0
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def detected_track_counts(decisions_path):
    """Return source-frame counts from the range-based track TSV."""
    counts = {}
    frame_count = 0
    with decisions_path.open("r", encoding="utf-8", newline="") as stream:
        rows = (line for line in stream if not line.startswith("#"))
        for row in csv.DictReader(rows, delimiter="\t"):
            count = int(row["frame_end"]) - int(row["frame_start"]) + 1
            frame_count += count
            section = row["detected_section"]
            counts[section] = counts.get(section, 0) + count
    return frame_count, counts
