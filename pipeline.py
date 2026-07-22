#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main orchestration for the anime_vfr VFR pipeline."""

import argparse
import csv
from fractions import Fraction
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from branches import render_matched_decimated_branch_builder
from config import (
    ENCODER_BIN,
    MM_DEDUP_CAP,
    MM_DEDUP_ENABLED,
    MM_ANALYSIS_MAX_WORKERS,
    MM_BOB_REGION_MAX_ANCHOR_GAP,
    MM_MATCHABLE_BOB_EDGE_GUARD,
    MM_MATCHABLE_MIN_MOTION,
    MM_MATCHABLE_MIN_RUN,
    MM_LOW_INFORMATION_INHERIT_MAX,
    MM_LOW_INFORMATION_MOTION_MAX,
    MM_REDUNDANCY_DROP_DIFF_MAX,
    MM_REDUNDANCY_DROP_RATIO_MAX,
    MM_REDUNDANCY_DROP_RATIO_MIN,
    MM_REDUNDANCY_MIN_RUN,
    MM_TDECIMATE_CYCLE,
    MM_TDECIMATE_CYCLE_DROP,
    MM_TFM_CTHRESH,
    MM_TFM_MI,
    MM_TFM_SLOW,
    MM_TFM_WEAVE_CTHRESH,
    MM_TFM_WEAVE_METRIC,
    MM_TFM_WEAVE_MI,
    MM_TFM_WEAVE_RFF_MI,
    MM_TFM_WEAVE_RFF_MIN_RUN,
    MM_VERTICAL_SCROLL_BEST_MAX,
    MM_VERTICAL_SCROLL_DIRECT_MIN,
    MM_VERTICAL_SCROLL_ENABLED,
    MM_VERTICAL_SCROLL_IMPROVEMENT_MIN,
    MM_VERTICAL_SCROLL_MIN_HITS,
    MM_VERTICAL_SCROLL_MIN_RUN,
    MM_VERTICAL_SCROLL_SOFT_BEST_MAX,
    MM_VERTICAL_SCROLL_SOFT_DIRECT_MIN,
    MM_VERTICAL_SCROLL_SOFT_IMPROVEMENT_MIN,
    MM_VERTICAL_SCROLL_SHIFT,
    MM_VERTICAL_SCROLL_WINDOW,
    PTS_DECIMATE_CYCLE_TOLERANCE_MS,
    PYTHON_BIN,
    VSPIPE,
)
from contracts import AnalysisResult, DedupStats, EpisodeStats, Segment, ShadowResult, validate_analysis_result, validate_dedup_stats, validate_episode_stats, validate_segments, validate_shadow_result
from dedup import run_dedup_detection, run_progressive_dedup_detection
from encode import encode, get_chroma_flags, get_color_flags, get_par_flags, mux_final
from media import (
    extract_chapter_ranges,
    extract_source_timecodes,
    get_video_field_metadata,
    get_video_frame_count,
    get_video_info,
)
from report import print_strategy_analyze_report, run_report
from segments import (
    make_bob_entries_from_source_timecodes,
    make_linear_strategy_segments,
    make_progressive_entries_from_source_timecodes,
    parse_framemap,
    strategies_to_segments,
)
from timecodes import (
    generate_final_timecodes_v2,
    segment_output_frame_count,
    source_end_ms,
    validate_source_timeline,
)
from utils import (
    box_max_16 as _box_max_16,
    read_timecodes_v2,
)

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


def fmtc_to_yuv420p8(core, vs, clip):
    clip = core.fmtc.bitdepth(clip, bits=16)
    clip = core.fmtc.resample(clip, csp=vs.YUV420P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(clip, bits=8)


def _field_order_settings(field_order):
    tff = field_order == "tff"
    return {
        "tff": tff,
        "fieldbased": 2 if tff else 1,
        "tfm_order": 1 if tff else 0,
        "suffix": "" if tff else "_bff",
    }


def _resolve_thread_count(value, default=None):
    """Normalize a CLI or config thread override to at least one worker."""
    if value is None:
        value = default
    if value is None:
        value = os.cpu_count() or 16
    return max(1, int(value))


def run_pass1(source_path, work_dir, field_order="tff", vs_threads=None):
    """Collect TFM matches and mode-4 TDecimate metrics for later passes.

    TFM analyzes 3:2 pulldown through field matching. TDecimate mode 4 records
    decimation metrics without creating an output video timeline.
    """
    stem = source_path.stem
    field = _field_order_settings(field_order)
    suffix = field["suffix"]
    stats_path = work_dir / f"{stem}{suffix}_stats.txt"
    tfm_path = work_dir / f"{stem}{suffix}_tfm.txt"
    script_path = work_dir / f"{stem}{suffix}_pass1.vpy"
    source_esc = str(source_path).replace("\\", "\\\\")
    stats_esc = str(stats_path).replace("\\", "\\\\")
    tfm_esc = str(tfm_path).replace("\\", "\\\\")
    n_threads = _resolve_thread_count(vs_threads)
    content = f'''import vapoursynth as vs
core = vs.core
core.num_threads = {n_threads}
{VPY_FMTC_HELPERS}
clip = core.bs.VideoSource(r"{source_esc}")
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval={field["fieldbased"]})
clip = fmtc_to_yuv420p8(clip)
clip = core.tivtc.TFM(clip, order={field["tfm_order"]}, cthresh={MM_TFM_CTHRESH}, MI={MM_TFM_MI}, slow={MM_TFM_SLOW}, output=r"{tfm_esc}")
clip = core.tivtc.TDecimate(clip, mode=4, output=r"{stats_esc}")
clip.set_output(0)
'''
    if stats_path.exists() and tfm_path.exists():
        print("  Pass 1: existing files found, reusing them...")
        return stats_path, tfm_path
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    cmd = [VSPIPE, "--progress", str(script_path), "--"]
    print(f"  Pass 1: TIVTC analysis (core.num_threads={n_threads})...")
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"Pass 1 failed for {source_path}")
    print(f"  Pass 1 completed in {time.time() - t0:.1f}s")
    return stats_path, tfm_path


def run_pass2a(source_path, stats_path, tfm_path, work_dir, field_order="tff", vs_threads=None):
    """Build the canonical TIVTC candidate timeline and source framemap.

    Each output frame receives ``_OrigFrameNum`` so later stages can recover
    the source frame selected by TFM/TDecimate. A subprocess isolates frame
    iteration from the VapourSynth state of the main process.
    """
    stem = source_path.stem
    field = _field_order_settings(field_order)
    suffix = field["suffix"]
    tc_v1_path = work_dir / f"{stem}{suffix}_tc_v1.txt"
    framemap_path = work_dir / f"{stem}{suffix}_framemap.txt"
    mapper_script = work_dir / f"{stem}{suffix}_mapper.py"

    source_esc = str(source_path).replace("\\", "\\\\")
    stats_esc = str(stats_path).replace("\\", "\\\\")
    tfm_esc = str(tfm_path).replace("\\", "\\\\")
    tc_esc = str(tc_v1_path).replace("\\", "\\\\")
    fm_esc = str(framemap_path).replace("\\", "\\\\")
    n_threads = _resolve_thread_count(vs_threads)

    mapper_content = f'''import vapoursynth as vs
core = vs.core
core.num_threads = {n_threads}
{VPY_FMTC_HELPERS}
clip = core.bs.VideoSource(r"{source_esc}")
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval={field["fieldbased"]})
clip = fmtc_to_yuv420p8(clip)

def set_fn(n, f):
    fout = f.copy()
    fout.props["_OrigFrameNum"] = n
    return fout
clip = core.std.ModifyFrame(clip, clip, set_fn)

clip = core.tivtc.TFM(clip, order={field["tfm_order"]}, cthresh={MM_TFM_CTHRESH}, MI={MM_TFM_MI}, slow={MM_TFM_SLOW}, input=r"{tfm_esc}")
clip = core.tivtc.TDecimate(clip, mode=5, hybrid=2, vfrDec=1, input=r"{stats_esc}", tfmIn=r"{tfm_esc}", mkvOut=r"{tc_esc}")

with open(r"{fm_esc}", "w") as output_file:
    for output_index, frame in enumerate(clip.frames(prefetch={n_threads})):
        source_index = frame.props["_OrigFrameNum"]
        duration_numerator = frame.props["_DurationNum"]
        duration_denominator = frame.props["_DurationDen"]
        combed = frame.props.get("_Combed", 0)
        output_file.write(f"{{output_index}},{{source_index}},{{duration_numerator}},{{duration_denominator}},{{combed}}\\n")
print(f"Framemap: {{clip.num_frames}} frames")
'''
    framemap_is_current = False
    if framemap_path.exists():
        with open(framemap_path, "r", encoding="utf-8") as existing_framemap:
            first_row = next((line.strip() for line in existing_framemap if line.strip()), "")
        framemap_is_current = len(first_row.split(",")) == 5
    if tc_v1_path.exists() and framemap_is_current:
        print("  Pass 2a: existing files found, reusing them...")
        return tc_v1_path, framemap_path
    if framemap_path.exists() and not framemap_is_current:
        print("  Pass 2a: legacy framemap has no exact duration rationals, regenerating it...")
    with open(mapper_script, "w", encoding="utf-8") as f:
        f.write(mapper_content)

    print(f"  Pass 2a: TDecimate mode=5 + framemap (core.num_threads={n_threads}, prefetch={n_threads})...")
    t0 = time.time()
    result = subprocess.run([PYTHON_BIN, str(mapper_script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Pass 2a failed: {result.stderr}")
    print(f"  {result.stdout.strip()}")
    print(f"  Pass 2a completed in {time.time() - t0:.1f}s")

    return tc_v1_path, framemap_path


# ═════════════════════════════════════════════════════════════════════════
# TFM AND TEMPORAL CLASSIFIER
# Operational decisions use structured TFM records, same-parity motion,
# sustained vertical scrolling, source PTS boundaries, and matched-branch
# temporal differences. A sensitive MIC pass vetoes residual combing on the
# exact match selected by the primary TFM pass.
# ═════════════════════════════════════════════════════════════════════════

TFM_RECORD_RE = re.compile(
    r"^\s*(?P<index>\d+)\s+(?P<match>\S+)\s+(?P<combed>[+-])(?:\s+\[(?P<mic>\d+)\])?\s*$"
)
TFM_WEAVABLE_MATCHES = frozenset(("p", "c", "n", "b", "u"))
TFM_MATCH_MIC_INDEX = {
    "p": 0,
    "c": 1,
    "n": 2,
    "b": 3,
    "u": 4,
}


def parse_tfm_records(tfm_path, frame_count):
    """Read the TFM log while preserving source-frame alignment."""
    records = [
        {"index": index, "match": None, "combed": None, "valid": False}
        for index in range(frame_count)
    ]
    diagnostics = {
        "parsed": 0,
        "incomplete": 0,
        "missing": 0,
        "duplicates": 0,
        "malformed": 0,
        "out_of_range": 0,
        "field": None,
    }
    with open(tfm_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("field") and "=" in stripped:
                diagnostics["field"] = stripped.split("=", 1)[1].strip()
                continue
            if stripped.startswith("#"):
                continue
            match = TFM_RECORD_RE.match(stripped)
            if match is None:
                if stripped[0].isdigit():
                    diagnostics["malformed"] += 1
                continue
            index = int(match.group("index"))
            if index >= frame_count:
                diagnostics["out_of_range"] += 1
                continue
            if records[index]["match"] is not None:
                diagnostics["duplicates"] += 1
                continue
            mic_text = match.group("mic")
            records[index] = {
                "index": index,
                "match": match.group("match"),
                "combed": match.group("combed") == "+",
                "valid": mic_text is not None,
            }
            diagnostics["parsed"] += 1
            if mic_text is None:
                diagnostics["incomplete"] += 1
    diagnostics["missing"] = sum(1 for record in records if record["match"] is None)
    return records, diagnostics


def _scan_selected_weave_combing(clip, tfm_records, field, n_threads):
    """Measure combing on the primary TFM match without selecting a new match."""
    import vapoursynth as vs

    core = vs.core
    sensitive = core.tivtc.TFM(
        clip,
        order=field["tfm_order"],
        mode=1,
        PP=1,
        cthresh=MM_TFM_WEAVE_CTHRESH,
        MI=MM_TFM_WEAVE_MI,
        slow=MM_TFM_SLOW,
        metric=MM_TFM_WEAVE_METRIC,
        chroma=False,
        micout=2,
    )
    selected_mics = [None] * len(tfm_records)

    for index, frame in enumerate(sensitive.frames(prefetch=n_threads)):
        match_index = TFM_MATCH_MIC_INDEX.get(tfm_records[index]["match"])
        if match_index is None:
            continue
        mics = frame.props["TFMMics"]
        selected_mic = int(mics[match_index])
        selected_mics[index] = selected_mic

    return selected_mics


def _explicit_rff_run_mask(timeline, tfm_records):
    """Identify sustained clean source runs with an explicit 2/3-field RFF cadence."""
    frame_count = len(tfm_records)
    field_units = timeline["field_units"]
    quantization_valid = timeline["quantization_valid"]
    if len(field_units) != frame_count or len(quantization_valid) != frame_count:
        raise RuntimeError("RFF evidence arrays do not share the TFM record cardinality")

    eligible = [
        (
            quantization_valid[index] and
            field_units[index] in {2, 3} and
            record["valid"] and
            not record["combed"] and
            record["match"] in TFM_WEAVABLE_MATCHES
        )
        for index, record in enumerate(tfm_records)
    ]
    rff_mask = [False] * frame_count
    index = 0
    while index < frame_count:
        if not eligible[index]:
            index += 1
            continue
        start = index
        previous_units = field_units[index]
        index += 1
        while (
            index < frame_count and
            eligible[index] and
            field_units[index] != previous_units
        ):
            previous_units = field_units[index]
            index += 1
        if index - start >= MM_TFM_WEAVE_RFF_MIN_RUN:
            rff_mask[start:index] = [True] * (index - start)
    return rff_mask


def _selected_weave_vetoes(selected_mics, explicit_rff_mask):
    """Apply the sensitive MIC veto with a relaxed limit on explicit clean RFF runs."""
    if len(selected_mics) != len(explicit_rff_mask):
        raise RuntimeError("Selected-weave MIC and RFF evidence cardinalities do not match")
    thresholds = [
        MM_TFM_WEAVE_RFF_MI if is_explicit_rff else MM_TFM_WEAVE_MI
        for is_explicit_rff in explicit_rff_mask
    ]
    vetoes = [
        mic is not None and mic > threshold
        for mic, threshold in zip(selected_mics, thresholds)
    ]
    return vetoes, thresholds


def _source_frame_metrics(arr, previous_arr, field_order_tff):
    """Compute all independent source-frame metrics in one worker job."""
    arr_int32 = arr.astype(np.int32)
    top = arr_int32[0::2]
    bottom = arr_int32[1::2]
    first = top if field_order_tff else bottom
    second = bottom if field_order_tff else top

    first_motion = 0.0
    second_motion = 0.0
    vertical_scroll_hit = _is_vertical_scroll_hit(second, first)
    if previous_arr is not None:
        previous_int32 = previous_arr.astype(np.int32)
        previous_top = previous_int32[0::2]
        previous_bottom = previous_int32[1::2]
        previous_first = previous_top if field_order_tff else previous_bottom
        previous_second = previous_bottom if field_order_tff else previous_top
        first_motion = _box_max_16(np.abs(first - previous_first))
        second_motion = _box_max_16(np.abs(second - previous_second))
        vertical_scroll_hit = (
            vertical_scroll_hit or
            _is_vertical_scroll_hit(first, previous_second)
        )

    return (
        first_motion,
        second_motion,
        int(vertical_scroll_hit),
    )


def _matched_temporal_difference(arr, previous_arr):
    """Measure temporal difference from the previous frame on the TFM branch."""
    temporal_difference = 0.0
    if previous_arr is not None:
        temporal_difference = _box_max_16(np.abs(arr - previous_arr))
    return temporal_difference


def _persistent_run_mask(mask, minimum_run):
    """Keep only true values belonging to a sustained contiguous run."""
    persistent = [False] * len(mask)
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        start = index
        while index < len(mask) and mask[index]:
            index += 1
        if index - start >= minimum_run:
            persistent[start:index] = [True] * (index - start)
    return persistent


def _detect_decision_boundaries(frame_count, timeline):
    """Split decisions after source frames whose durations are discontinuous."""
    boundaries = [False] * frame_count
    if frame_count:
        boundaries[0] = True
    for index in timeline["discontinuities"]:
        next_index = index + 1
        if 0 <= next_index < frame_count:
            boundaries[next_index] = True
    return boundaries


def _build_multimetric_evidence(motion_arr, vertical_scroll_mask, timeline):
    """Convert raw metrics into explainable evidence without selecting a strategy."""
    frame_count = len(vertical_scroll_mask)
    if len(motion_arr) != frame_count * 2:
        raise RuntimeError("Multi-metric evidence arrays do not share the source-frame cardinality")
    same_parity = [max(motion_arr[index * 2:index * 2 + 2]) for index in range(frame_count)]
    raw_low_information = [value <= MM_LOW_INFORMATION_MOTION_MAX for value in same_parity]
    low_information = _persistent_run_mask(raw_low_information, MM_MATCHABLE_MIN_RUN)
    return {
        "same_parity": same_parity,
        "low_information": low_information,
        "decision_boundaries": _detect_decision_boundaries(frame_count, timeline),
        "vertical_scroll": list(vertical_scroll_mask),
    }


def _build_shadow_strategy(tfm_records, sensitive_combed_vetoes, motion_arr, evidence) -> ShadowResult:
    """Build matchability anchors from sustained agreement between independent metrics."""
    frame_count = len(tfm_records)
    matchability = ["unknown"] * frame_count
    redundancy = [None] * frame_count
    redundancy_origins = [None] * frame_count
    redundancy_mapping = [None] * frame_count
    strategies = ["bob_expand"] * frame_count
    shadow_origins = ["fallback_unknown"] * frame_count
    locked_matchable = [False] * frame_count
    locked_bob = [False] * frame_count

    candidates = [False] * frame_count
    for index, record in enumerate(tfm_records):
        if evidence["vertical_scroll"][index]:
            matchability[index] = "not_matchable"
            shadow_origins[index] = "vertical_scroll_veto"
            locked_bob[index] = True
            continue
        if record["combed"]:
            matchability[index] = "not_matchable"
            shadow_origins[index] = "tfm_combed_veto"
            locked_bob[index] = True
            continue
        if sensitive_combed_vetoes[index]:
            matchability[index] = "not_matchable"
            shadow_origins[index] = "tfm_sensitive_combed_veto"
            locked_bob[index] = True
            continue
        candidates[index] = (
            record["valid"] and
            record["match"] in TFM_WEAVABLE_MATCHES
        )
        if record["match"] is None:
            shadow_origins[index] = "tfm_missing"
        elif not record["valid"]:
            shadow_origins[index] = "tfm_incomplete"
        elif record["match"] not in TFM_WEAVABLE_MATCHES:
            shadow_origins[index] = "tfm_match_not_weavable"

    edge_guard = max(0, MM_MATCHABLE_BOB_EDGE_GUARD)
    if edge_guard:
        guarded = [False] * frame_count
        for index, is_locked in enumerate(locked_bob):
            if not is_locked:
                continue
            start = max(0, index - edge_guard)
            end = min(frame_count, index + edge_guard + 1)
            for frame in range(start, end):
                guarded[frame] = True
        for index, is_guarded in enumerate(guarded):
            if is_guarded and candidates[index]:
                candidates[index] = False
                shadow_origins[index] = "bob_anchor_edge_guard"

    index = 0
    while index < frame_count:
        if not candidates[index]:
            index += 1
            continue
        start = index
        while index < frame_count and candidates[index] and (index == start or not evidence["decision_boundaries"][index]):
            index += 1
        end = index
        run_length = end - start
        informative_frames = [frame for frame in range(start, end) if not evidence["low_information"][frame]]
        if not informative_frames:
            for frame in range(start, end):
                shadow_origins[frame] = "clean_tfm_low_information"
            continue
        run_motion = [motion_arr[field] for frame in informative_frames for field in (frame * 2, frame * 2 + 1)]
        average_motion = sum(run_motion) / len(run_motion) if run_motion else 0.0
        if run_length < MM_MATCHABLE_MIN_RUN:
            for frame in range(start, end):
                shadow_origins[frame] = "clean_tfm_run_too_short"
            continue
        if average_motion < MM_MATCHABLE_MIN_MOTION:
            for frame in range(start, end):
                shadow_origins[frame] = "clean_tfm_low_information"
            continue

        for frame in range(start, end):
            matchability[frame] = "matchable"
            redundancy[frame] = "unknown"
            redundancy_origins[frame] = "mapping_not_evaluated"
            strategies[frame] = "match_keep_pts"
            if evidence["low_information"][frame]:
                shadow_origins[frame] = "low_information_absorbed_by_matchable_subrun"
            else:
                shadow_origins[frame] = "sustained_clean_tfm_multimetric"
                locked_matchable[frame] = True

    shadow: ShadowResult = {
        "matchability": matchability,
        "redundancy": redundancy,
        "redundancy_origins": redundancy_origins,
        "redundancy_mapping": redundancy_mapping,
        "strategies": strategies,
        "origins": shadow_origins,
        "locked_matchable": locked_matchable,
        "locked_bob": locked_bob,
    }
    validate_shadow_result(shadow)
    return shadow


def _bridge_hard_bob_regions(shadow: ShadowResult, decision_boundaries) -> tuple[ShadowResult, list[tuple[int, int]]]:
    """Bridge nearby residual-combing anchors without weakening hard bob evidence."""
    validate_shadow_result(shadow)
    frame_count = len(shadow["strategies"])
    if len(decision_boundaries) != frame_count:
        raise RuntimeError("Bob-region boundaries do not share the shadow cardinality")
    hard_anchors = [index for index, locked in enumerate(shadow["locked_bob"]) if locked]
    regions = []
    if hard_anchors:
        region_start = hard_anchors[0]
        previous = hard_anchors[0]
        for anchor in hard_anchors[1:]:
            gap = anchor - previous - 1
            crosses_boundary = any(decision_boundaries[previous + 1:anchor + 1])
            if gap <= MM_BOB_REGION_MAX_ANCHOR_GAP and not crosses_boundary:
                previous = anchor
                continue
            if previous > region_start:
                regions.append((region_start, previous))
            region_start = anchor
            previous = anchor
        if previous > region_start:
            regions.append((region_start, previous))

    for start, end in regions:
        for frame in range(start + 1, end):
            if shadow["locked_bob"][frame]:
                continue
            _set_shadow_bob_frame(
                shadow,
                frame,
                "hard_bob_region_bridge",
                "invalidated_by_residual_combing_region",
            )

    validate_shadow_result(shadow)
    return shadow, regions


def _resolve_low_information_runs(shadow: ShadowResult, low_information_mask, decision_boundaries) -> tuple[ShadowResult, list[tuple[int, int, str]]]:
    """Inherit low-information runs from compatible anchors without crossing boundaries."""
    validate_shadow_result(shadow)
    frame_count = len(shadow["strategies"])
    resolved_runs = []
    index = 0
    while index < frame_count:
        if not low_information_mask[index] or shadow["origins"][index] != "clean_tfm_low_information":
            index += 1
            continue
        start = index
        while index < frame_count and low_information_mask[index] and shadow["origins"][index] == "clean_tfm_low_information" and (index == start or not decision_boundaries[index]):
            index += 1
        end = index

        left = None
        if start > 0 and not decision_boundaries[start]:
            candidate = start - 1
            while candidate >= 0:
                if shadow["matchability"][candidate] != "unknown":
                    left = candidate
                    break
                if decision_boundaries[candidate]:
                    break
                candidate -= 1
        right = None
        if end < frame_count and not decision_boundaries[end]:
            candidate = end
            while candidate < frame_count:
                if shadow["matchability"][candidate] != "unknown":
                    right = candidate
                    break
                candidate += 1
                if candidate < frame_count and decision_boundaries[candidate]:
                    break
        left_state = shadow["matchability"][left] if left is not None else None
        right_state = shadow["matchability"][right] if right is not None else None
        inherited_state = None
        if left_state in {"matchable", "not_matchable"} and left_state == right_state:
            inherited_state = left_state
        elif end - start <= MM_LOW_INFORMATION_INHERIT_MAX:
            available = [(left_state, left), (right_state, right)]
            available = [(state, anchor) for state, anchor in available if state in {"matchable", "not_matchable"}]
            if len(available) == 1 and min(abs(available[0][1] - start), abs(available[0][1] - end)) <= MM_LOW_INFORMATION_INHERIT_MAX:
                inherited_state, _anchor = available[0]
        if inherited_state is None:
            continue

        origin = "low_information_inherited_matchable" if inherited_state == "matchable" else "low_information_inherited_bob"
        for frame in range(start, end):
            shadow["matchability"][frame] = inherited_state
            shadow["strategies"][frame] = "match_keep_pts" if inherited_state == "matchable" else "bob_expand"
            shadow["origins"][frame] = origin
            if inherited_state == "matchable":
                shadow["redundancy"][frame] = "unknown"
                shadow["redundancy_origins"][frame] = "mapping_not_evaluated"
        resolved_runs.append((start, end - 1, inherited_state))

    validate_shadow_result(shadow)
    return shadow, resolved_runs


def _set_shadow_bob_frame(shadow: ShadowResult, index, origin, redundancy_origin):
    """Assign one source frame to the bob branch and clear stale match mappings."""
    changed = shadow["strategies"][index] != "bob_expand"
    shadow["strategies"][index] = "bob_expand"
    shadow["matchability"][index] = "not_matchable"
    shadow["redundancy"][index] = None
    shadow["redundancy_origins"][index] = redundancy_origin
    shadow["redundancy_mapping"][index] = None
    shadow["origins"][index] = origin
    shadow["locked_matchable"][index] = False
    shadow["locked_bob"][index] = True
    return changed


def _apply_bob_strategy_overrides(shadow: ShadowResult, timeline, ranges) -> tuple[ShadowResult, int]:
    """Force explicit source-time ranges to bob before redundancy is evaluated."""
    validate_shadow_result(shadow)
    if not ranges:
        return shadow, 0

    frame_count = len(shadow["strategies"])
    source_end = timeline["pts_ms"][-1] + timeline["duration_ms"][-1] if frame_count else 0.0
    normalized = [(start, source_end if end is None else end) for start, end in ranges]
    changed = 0
    for index in range(frame_count):
        start = timeline["pts_ms"][index]
        end = start + timeline["duration_ms"][index]
        if not any(end > range_start and start < range_end for range_start, range_end in normalized):
            continue
        changed += int(_set_shadow_bob_frame(shadow, index, "explicit_bob_override", "invalidated_by_explicit_bob_override"))
    validate_shadow_result(shadow)
    return shadow, changed


def _validate_bob_field_units(shadow: ShadowResult, timeline, field_metadata):
    """Require quantized field durations only where the final strategy expands fields."""
    validate_shadow_result(shadow)
    frame_count = len(shadow["strategies"])
    if len(timeline["field_units"]) != frame_count or len(field_metadata) != frame_count:
        raise RuntimeError("Bob field-unit validation inputs do not share the source-frame cardinality")
    for index, strategy in enumerate(shadow["strategies"]):
        if strategy != "bob_expand":
            continue
        expected_units = 2 + field_metadata[index]["repeat_pict"]
        actual_units = timeline["field_units"][index]
        if actual_units != expected_units:
            raise RuntimeError(
                f"Cannot bob source frame {index}: duration is not consistent with field metadata "
                f"(field_units={actual_units}, repeat_pict={field_metadata[index]['repeat_pict']})"
            )


def _normalize_field_safe_transitions(shadow: ShadowResult, tfm_records) -> tuple[ShadowResult, list[dict]]:
    """Expand bob boundaries until no matched frame borrows a field across them."""
    validate_shadow_result(shadow)
    frame_count = len(shadow["strategies"])
    if len(tfm_records) != frame_count:
        raise ValueError("TFM record cardinality does not match shadow strategies")

    adjustments = []
    index = 1
    while index < frame_count:
        previous_is_bob = shadow["strategies"][index - 1] == "bob_expand"
        current_is_bob = shadow["strategies"][index] == "bob_expand"
        if previous_is_bob == current_is_bob:
            index += 1
            continue

        if previous_is_bob:
            original_boundary = index
            matches = []
            while index < frame_count and shadow["strategies"][index] != "bob_expand" and tfm_records[index]["match"] in {"p", "b"}:
                matches.append(tfm_records[index]["match"])
                _set_shadow_bob_frame(shadow, index, "field_dependency_boundary_bob", "invalidated_by_field_dependency_boundary")
                index += 1
            if matches:
                adjustments.append({"direction": "bob_to_match", "original_boundary": original_boundary, "new_boundary": index, "start": original_boundary, "end": index - 1, "matches": matches})
            index += 1
            continue

        original_boundary = index
        frame = index - 1
        matches = []
        while frame >= 0 and shadow["strategies"][frame] != "bob_expand" and tfm_records[frame]["match"] in {"n", "u"}:
            matches.append(tfm_records[frame]["match"])
            _set_shadow_bob_frame(shadow, frame, "field_dependency_boundary_bob", "invalidated_by_field_dependency_boundary")
            frame -= 1
        if matches:
            adjustments.append({"direction": "match_to_bob", "original_boundary": original_boundary, "new_boundary": frame + 1, "start": frame + 1, "end": original_boundary - 1, "matches": list(reversed(matches))})
        index += 1

    unsafe = []
    for index in range(1, frame_count):
        previous_is_bob = shadow["strategies"][index - 1] == "bob_expand"
        current_is_bob = shadow["strategies"][index] == "bob_expand"
        if previous_is_bob and not current_is_bob and tfm_records[index]["match"] in {"p", "b"}:
            unsafe.append((index, "bob_to_match", tfm_records[index]["match"]))
        elif not previous_is_bob and current_is_bob and tfm_records[index - 1]["match"] in {"n", "u"}:
            unsafe.append((index, "match_to_bob", tfm_records[index - 1]["match"]))
    if unsafe:
        raise RuntimeError(f"Field-unsafe strategy boundaries remain after normalization: {unsafe[:10]}")

    validate_shadow_result(shadow)
    return shadow, adjustments


def _apply_shadow_redundancy(shadow: ShadowResult, timeline, entries, matched_temporal_diff) -> ShadowResult:
    """Validate only candidate decimation blocks proposed by TDecimate."""
    validate_shadow_result(shadow)
    frame_count = len(shadow["strategies"])
    retained_sources = set()
    candidate_mask = [False] * frame_count
    ordered_entries = sorted(entries, key=lambda entry: entry[0])
    for _dec_idx, src_idx, _dur_num, dur_den, _combed in ordered_entries:
        if src_idx >= frame_count:
            continue
        retained_sources.add(src_idx)
        if dur_den == 24000:
            candidate_mask[src_idx] = True
    for left, right in zip(ordered_entries, ordered_entries[1:]):
        _left_dec, left_src, _left_num, left_den, _left_combed = left
        _right_dec, right_src, _right_num, right_den, _right_combed = right
        if left_den != 24000 or right_den != 24000:
            continue
        for src_idx in range(left_src + 1, min(right_src, frame_count)):
            candidate_mask[src_idx] = True

    def drop_difference(src_idx):
        adjacent = []
        if src_idx < len(matched_temporal_diff):
            adjacent.append(matched_temporal_diff[src_idx])
        if src_idx + 1 < len(matched_temporal_diff):
            adjacent.append(matched_temporal_diff[src_idx + 1])
        return min(adjacent) if adjacent else float("inf")

    unverified_drops = {
        src_idx
        for src_idx in range(frame_count)
        if (
            candidate_mask[src_idx] and
            shadow["matchability"][src_idx] == "matchable" and
            timeline["field_units"][src_idx] == 2 and
            src_idx not in retained_sources and
            drop_difference(src_idx) >= MM_REDUNDANCY_DROP_DIFF_MAX
        )
    }

    def is_eligible(src_idx):
        return (
            candidate_mask[src_idx] and
            shadow["matchability"][src_idx] == "matchable" and
            timeline["field_units"][src_idx] == 2 and
            src_idx not in unverified_drops
        )

    for src_idx in range(frame_count):
        if (
            candidate_mask[src_idx] and
            shadow["matchability"][src_idx] == "matchable" and
            timeline["field_units"][src_idx] != 2
        ):
            shadow["redundancy_origins"][src_idx] = "source_pts_exposes_multi_field_sample"
    for src_idx in unverified_drops:
        shadow["redundancy_origins"][src_idx] = "candidate_drop_not_duplicate"

    index = 0
    while index < frame_count:
        if not is_eligible(index):
            index += 1
            continue
        start = index
        while index < frame_count and is_eligible(index):
            index += 1
        end = index
        run_length = end - start
        dropped = [src_idx for src_idx in range(start, end) if src_idx not in retained_sources]
        drop_ratio = len(dropped) / run_length if run_length else 0.0
        drop_diffs = [drop_difference(src_idx) for src_idx in dropped]
        drops_verified = bool(dropped) and all(
            difference < MM_REDUNDANCY_DROP_DIFF_MAX for difference in drop_diffs
        )

        failure = None
        if run_length < MM_REDUNDANCY_MIN_RUN:
            failure = "candidate_run_too_short"
        elif not (MM_REDUNDANCY_DROP_RATIO_MIN <= drop_ratio <= MM_REDUNDANCY_DROP_RATIO_MAX):
            failure = "candidate_drop_density_invalid"
        elif not drops_verified:
            failure = "candidate_drop_not_duplicate"

        if failure is not None:
            for src_idx in range(start, end):
                shadow["redundancy_origins"][src_idx] = failure
            continue

        for src_idx in range(start, end):
            shadow["redundancy"][src_idx] = "redundant_with_valid_map"
            shadow["strategies"][src_idx] = "match_decimate"
            shadow["redundancy_origins"][src_idx] = "tdecimate_mapping_multimetric_validated"
            shadow["redundancy_mapping"][src_idx] = (
                "retained" if src_idx in retained_sources else "dropped"
            )
    validate_shadow_result(shadow)
    return shadow


def _normalize_complete_decimate_cycles(shadow: ShadowResult, timeline, entries) -> tuple[ShadowResult, list[tuple[int, int, int]]]:
    """Keep IVTC only for complete cycles with a valid TDecimate duration map."""
    validate_shadow_result(shadow)
    frame_count = len(shadow["strategies"])
    source_to_entry = {
        src_idx: (dec_idx, duration_num, duration_den)
        for dec_idx, src_idx, duration_num, duration_den, _combed in entries
        if src_idx < frame_count
    }
    expected_retained = MM_TDECIMATE_CYCLE - MM_TDECIMATE_CYCLE_DROP
    adjustments = []

    for start in range(0, frame_count, MM_TDECIMATE_CYCLE):
        end = min(start + MM_TDECIMATE_CYCLE, frame_count)
        decimated_frames = [
            index
            for index in range(start, end)
            if shadow["strategies"][index] == "match_decimate"
        ]
        if not decimated_frames:
            continue

        retained_frames = [
            index
            for index in range(start, end)
            if shadow["redundancy_mapping"][index] == "retained"
        ]
        complete_cycle = (
            end - start == MM_TDECIMATE_CYCLE and
            len(decimated_frames) == MM_TDECIMATE_CYCLE and
            len(retained_frames) == expected_retained and
            all(index in source_to_entry for index in retained_frames)
        )
        if complete_cycle:
            branch_indices = [source_to_entry[index][0] for index in retained_frames]
            complete_cycle = all(
                branch_indices[position] + 1 == branch_indices[position + 1]
                for position in range(len(branch_indices) - 1)
            )
        if complete_cycle:
            tdecimate_duration_ms = sum(
                source_to_entry[index][1] * 1000.0 / source_to_entry[index][2]
                for index in retained_frames
            )
            source_duration_ms = sum(timeline["duration_ms"][start:end])
            complete_cycle = (
                abs(tdecimate_duration_ms - source_duration_ms)
                <= PTS_DECIMATE_CYCLE_TOLERANCE_MS
            )
        if complete_cycle:
            continue

        for index in decimated_frames:
            shadow["strategies"][index] = "match_keep_pts"
            shadow["redundancy"][index] = "unknown"
            shadow["redundancy_origins"][index] = "mapping_not_evaluated"
            shadow["redundancy_mapping"][index] = None
        adjustments.append((start, end - 1, len(decimated_frames)))

    index = 0
    while index < frame_count:
        if shadow["strategies"][index] != "match_decimate":
            index += 1
            continue
        start = index
        while index < frame_count and shadow["strategies"][index] == "match_decimate":
            index += 1
        end = index
        retained_frames = [
            frame
            for frame in range(start, end)
            if shadow["redundancy_mapping"][frame] == "retained"
        ]
        tdecimate_duration_ms = sum(
            source_to_entry[frame][1] * 1000.0 / source_to_entry[frame][2]
            for frame in retained_frames
        )
        source_duration_ms = sum(timeline["duration_ms"][start:end])
        if abs(tdecimate_duration_ms - source_duration_ms) <= PTS_DECIMATE_CYCLE_TOLERANCE_MS:
            continue
        for frame in range(start, end):
            shadow["strategies"][frame] = "match_keep_pts"
            shadow["redundancy"][frame] = "unknown"
            shadow["redundancy_origins"][frame] = "mapping_not_evaluated"
            shadow["redundancy_mapping"][frame] = None
        adjustments.append((start, end - 1, end - start))

    validate_shadow_result(shadow)
    return shadow, adjustments


def _observed_cadence(timeline, start, end):
    """Summarize run cadence without using it to select a strategy."""
    durations = timeline["duration_ms"][start:end]
    units = timeline["field_units"][start:end]
    total_duration = sum(durations)
    observed_rate = len(durations) * 1000.0 / total_duration if total_duration > 0 else 0.0
    unique_units = sorted(set(units), key=lambda unit: (unit is None, unit if unit is not None else 0))
    cadence_kind = "cfr" if len(unique_units) == 1 else "vfr"
    if cadence_kind == "vfr":
        cadence_label = "vfr"
    else:
        known_rates = (
            (24000.0 / 1001.0, "23.976p"),
            (30000.0 / 1001.0, "29.97p"),
            (60000.0 / 1001.0, "59.94p"),
        )
        cadence_label = "other_cfr"
        for rate, label in known_rates:
            if rate and abs(observed_rate - rate) / rate <= 0.01:
                cadence_label = label
                break
    return observed_rate, cadence_kind, cadence_label, unique_units


def _write_shadow_run_diagnostics(work_dir, stem, timeline, field_metadata, shadow):
    """Write a run-level summary of the selected shadow strategies."""
    output_path = work_dir / f"{stem}_classification_shadow_runs_v1.tsv"
    columns = [
        "source_start", "source_end", "pts_start_ms", "pts_end_ms", "source_frames", "output_frames",
        "strategy", "matchability", "redundancy", "origin", "redundancy_origin",
        "locked_matchable", "locked_bob", "branch", "observed_rate", "cadence_kind",
        "cadence_label", "field_units", "warning",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        index = 0
        frame_count = len(shadow["strategies"])
        while index < frame_count:
            start = index
            key = (
                shadow["strategies"][index],
                shadow["matchability"][index],
                shadow["origins"][index],
                shadow["redundancy"][index],
                shadow["redundancy_origins"][index],
            )
            while index < frame_count and (
                shadow["strategies"][index],
                shadow["matchability"][index],
                shadow["origins"][index],
                shadow["redundancy"][index],
                shadow["redundancy_origins"][index],
            ) == key:
                index += 1
            end = index - 1
            observed_rate, cadence_kind, cadence_label, units = _observed_cadence(
                timeline, start, index
            )
            warning = ""
            if key[0] == "match_decimate":
                output_frames = sum(
                    shadow["redundancy_mapping"][frame] == "retained"
                    for frame in range(start, index)
                )
            elif key[0] == "match_keep_pts":
                output_frames = index - start
            else:
                output_frames = sum(
                    2 + field_metadata[frame]["repeat_pict"]
                    for frame in range(start, index)
                )
            branch = {
                "match_keep_pts": "matched",
                "match_decimate": "decimated",
                "bob_expand": "bobbed",
            }[key[0]]
            writer.writerow({
                "source_start": start,
                "source_end": end,
                "pts_start_ms": f"{timeline['pts_ms'][start]:.6f}",
                "pts_end_ms": f"{timeline['pts_ms'][end] + timeline['duration_ms'][end]:.6f}",
                "source_frames": index - start,
                "output_frames": output_frames,
                "strategy": key[0],
                "matchability": key[1],
                "redundancy": shadow["redundancy"][start],
                "origin": key[2],
                "redundancy_origin": shadow["redundancy_origins"][start],
                "locked_matchable": int(all(shadow["locked_matchable"][start:index])),
                "locked_bob": int(all(shadow["locked_bob"][start:index])),
                "branch": branch,
                "observed_rate": f"{observed_rate:.6f}",
                "cadence_kind": cadence_kind,
                "cadence_label": cadence_label,
                "field_units": ",".join(str(unit) for unit in units),
                "warning": warning,
            })
    return output_path


def _write_classification_shadow_diagnostics(work_dir, stem, timeline, field_metadata, tfm_records, selected_weave_mics, explicit_rff_mask, selected_weave_thresholds, sensitive_combed_vetoes, motion_arr, matched_temporal_diff, vertical_scroll_hits, evidence, shadow):
    """Write per-frame metrics, evidence, and operational decisions."""
    output_path = work_dir / f"{stem}_classification_shadow_v1.tsv"
    columns = [
        "source_index", "pts_ms", "duration_ms", "field_units", "repeat_pict",
        "top_field_first", "pts_quantized",
        "tfm_match", "tfm_combed", "tfm_valid",
        "selected_weave_mic", "explicit_rff", "selected_weave_mi",
        "sensitive_combed_veto",
        "same_parity_first", "same_parity_second", "vertical_scroll",
        "pts_boundary", "low_information",
        "shadow_matchability", "shadow_redundancy", "shadow_strategy", "shadow_origin",
        "shadow_redundancy_origin", "shadow_mapping",
        "locked_matchable", "locked_bob", "matched_temporal_diff",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for index, record in enumerate(tfm_records):
            writer.writerow({
                "source_index": index,
                "pts_ms": f"{timeline['pts_ms'][index]:.6f}",
                "duration_ms": f"{timeline['duration_ms'][index]:.6f}",
                "field_units": timeline["field_units"][index],
                "repeat_pict": field_metadata[index]["repeat_pict"],
                "top_field_first": int(field_metadata[index]["top_field_first"]),
                "pts_quantized": int(timeline["quantization_valid"][index]),
                "tfm_match": record["match"],
                "tfm_combed": "" if record["combed"] is None else int(record["combed"]),
                "tfm_valid": int(record["valid"]),
                "selected_weave_mic": "" if selected_weave_mics[index] is None else selected_weave_mics[index],
                "explicit_rff": int(explicit_rff_mask[index]),
                "selected_weave_mi": selected_weave_thresholds[index],
                "sensitive_combed_veto": int(sensitive_combed_vetoes[index]),
                "same_parity_first": f"{motion_arr[index * 2]:.6f}",
                "same_parity_second": f"{motion_arr[index * 2 + 1]:.6f}",
                "vertical_scroll": vertical_scroll_hits[index],
                "pts_boundary": int(evidence["decision_boundaries"][index]),
                "low_information": int(evidence["low_information"][index]),
                "shadow_matchability": shadow["matchability"][index],
                "shadow_redundancy": shadow["redundancy"][index],
                "shadow_strategy": shadow["strategies"][index],
                "shadow_origin": shadow["origins"][index],
                "shadow_redundancy_origin": shadow["redundancy_origins"][index],
                "shadow_mapping": shadow["redundancy_mapping"][index],
                "locked_matchable": int(shadow["locked_matchable"][index]),
                "locked_bob": int(shadow["locked_bob"][index]),
                "matched_temporal_diff": f"{matched_temporal_diff[index]:.6f}",
            })
    return output_path


def _vertical_shift_match(field, prev_field):
    """Measure the expected vertical shift for interlaced scrolling.

    Non-overlapping 16x16 blocks are sufficient because this detector only
    needs a local region consistent with vertical scrolling; it does not need
    the pixel-perfect maximum block.
    """
    def block_max_16(arr):
        h = (arr.shape[0] // 16) * 16
        w = (arr.shape[1] // 16) * 16
        if h < 16 or w < 16:
            return float(np.mean(arr))
        blocks = arr[:h, :w].reshape(h // 16, 16, w // 16, 16)
        return float(blocks.mean(axis=(1, 3)).max())

    direct = block_max_16(np.abs(field - prev_field))
    shift = MM_VERTICAL_SCROLL_SHIFT
    if shift > 0:
        a = field[shift:]
        b = prev_field[:-shift]
    else:
        a = field[:shift]
        b = prev_field[-shift:]
    if a.shape[0] < 16:
        return direct, direct, 0
    shifted = block_max_16(np.abs(a - b))
    return direct, shifted, shift


def _is_vertical_scroll_hit(field, prev_field):
    """Identify one interlaced vertical-scroll transition."""
    direct, best, shift = _vertical_shift_match(field, prev_field)
    improvement = direct - best
    return (
        shift == MM_VERTICAL_SCROLL_SHIFT and (
            (
                direct >= MM_VERTICAL_SCROLL_DIRECT_MIN and
                best <= MM_VERTICAL_SCROLL_BEST_MAX and
                improvement >= MM_VERTICAL_SCROLL_IMPROVEMENT_MIN
            ) or (
                direct >= MM_VERTICAL_SCROLL_SOFT_DIRECT_MIN and
                best <= MM_VERTICAL_SCROLL_SOFT_BEST_MAX and
                improvement >= MM_VERTICAL_SCROLL_SOFT_IMPROVEMENT_MIN
            )
        )
    )


def _vertical_scroll_force_mask(n_frames, vertical_scroll_hits):
    """Build a mask for frames with reliable 60i vertical scrolling."""
    if not MM_VERTICAL_SCROLL_ENABLED or not vertical_scroll_hits:
        return [False] * n_frames

    win = max(1, MM_VERTICAL_SCROLL_WINDOW)
    half = win // 2
    min_hits = max(1, MM_VERTICAL_SCROLL_MIN_HITS)
    cum = [0] * (n_frames + 1)
    for i, hit in enumerate(vertical_scroll_hits[:n_frames]):
        cum[i + 1] = cum[i] + (1 if hit else 0)
    for i in range(len(vertical_scroll_hits), n_frames):
        cum[i + 1] = cum[i]

    force_flags = [False] * n_frames
    for i in range(n_frames):
        ws = max(0, i - half)
        we = min(n_frames, i + half + 1)
        force_flags[i] = cum[we] - cum[ws] >= min_hits

    min_run = max(1, MM_VERTICAL_SCROLL_MIN_RUN)
    force_mask = [False] * n_frames
    i = 0
    while i < n_frames:
        if not force_flags[i]:
            i += 1
            continue
        start = i
        while i < n_frames and force_flags[i]:
            i += 1
        if i - start < min_run:
            continue
        for j in range(start, i):
            force_mask[j] = True
    return force_mask


def run_multimetric_classification(source_path, work_dir, tfm_path, src_tc_path, framemap_path, field_order="tff", vs_threads=None, analysis_workers=None, forced_bob_ranges=None) -> AnalysisResult:
    """Run multi-metric classification and final consistency verification.

    The main scan decodes the source once and computes per-frame metrics in
    parallel. Each job receives the previous frame, preserving deterministic
    temporal dependencies.
    """
    import vapoursynth as vs
    core = vs.core
    from concurrent.futures import ThreadPoolExecutor

    n_threads = _resolve_thread_count(vs_threads)
    core.num_threads = n_threads
    if analysis_workers is None:
        metric_workers = n_threads if MM_ANALYSIS_MAX_WORKERS is None else max(1, min(n_threads, MM_ANALYSIS_MAX_WORKERS))
    else:
        metric_workers = _resolve_thread_count(analysis_workers)
    print(f"  TFM and temporal analysis — core.num_threads={n_threads}, metric_workers={metric_workers}...")
    field = _field_order_settings(field_order)
    field_order_tff = field["tff"]

    # Classifier metrics only need the luma plane at reduced bit depth.
    clip = core.bs.VideoSource(str(source_path), threads=0)
    clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=field["fieldbased"])
    clip = fmtc_to_yuv420p8(core, vs, clip)
    source_frame_count = clip.num_frames

    timeline = validate_source_timeline(src_tc_path, source_frame_count)
    field_metadata = get_video_field_metadata(source_path, source_frame_count)
    field_duration = timeline["field_duration_ms"]
    if field_duration is None:
        print("  PTS timeline: field duration cannot be estimated")
    else:
        valid_count = sum(timeline["quantization_valid"])
        print(
            f"  PTS timeline: {source_frame_count} frames, field={field_duration:.6f} ms, "
            f"quantized durations={valid_count}/{source_frame_count}"
        )
    for warning in timeline["warnings"]:
        print(f"    Warning PTS: {warning}")

    tfm_records, tfm_diagnostics = parse_tfm_records(tfm_path, source_frame_count)
    print(
        f"  Structured TFM: {tfm_diagnostics['parsed']}/{source_frame_count} records, "
        f"missing={tfm_diagnostics['missing']}, incomplete={tfm_diagnostics['incomplete']}, "
        f"duplicates={tfm_diagnostics['duplicates']}, "
        f"malformed={tfm_diagnostics['malformed']}, out-of-range={tfm_diagnostics['out_of_range']}"
    )

    print(
        "    Checking the selected TFM weave for residual combing "
        f"(metric={MM_TFM_WEAVE_METRIC}, cthresh={MM_TFM_WEAVE_CTHRESH}, "
        f"MI={MM_TFM_WEAVE_MI}, explicit-RFF MI={MM_TFM_WEAVE_RFF_MI}, luma only)..."
    )
    t0 = time.time()
    selected_weave_mics = _scan_selected_weave_combing(
        clip,
        tfm_records,
        field,
        n_threads,
    )
    explicit_rff_mask = _explicit_rff_run_mask(timeline, tfm_records)
    sensitive_combed_vetoes, selected_weave_thresholds = _selected_weave_vetoes(
        selected_weave_mics,
        explicit_rff_mask,
    )
    sensitive_veto_count = sum(sensitive_combed_vetoes)
    print(
        f"    Selected-weave check completed in {time.time() - t0:.1f}s: "
        f"{sensitive_veto_count} residual-combing vetoes, "
        f"{sum(explicit_rff_mask)} explicit-RFF frames"
    )

    print("    Single parallel scan of source metrics...")
    t0 = time.time()
    motion_arr = [0.0] * (source_frame_count * 2)
    vertical_scroll_hits = [0] * source_frame_count
    prev_arr = None
    max_pending = metric_workers * 4

    def store_source_metrics(index, values):
        (
            motion_arr[2 * index],
            motion_arr[2 * index + 1],
            vertical_scroll_hits[index],
        ) = values

    with ThreadPoolExecutor(max_workers=metric_workers) as executor:
        metric_futures = {}
        for i, fr in enumerate(clip.frames(prefetch=n_threads)):
            arr = np.asarray(fr[0]).copy()
            metric_futures[i] = executor.submit(
                _source_frame_metrics,
                arr,
                prev_arr,
                field_order_tff,
            )
            prev_arr = arr
            drain_index = i - max_pending
            if drain_index >= 0:
                store_source_metrics(
                    drain_index,
                    metric_futures.pop(drain_index).result(),
                )
        for index, future in metric_futures.items():
            store_source_metrics(index, future.result())
    print(f"    Scan completed in {time.time()-t0:.1f}s")
    print("    Scanning temporal differences on the full-length TFM branch...")
    t0 = time.time()
    matched_temporal_diff = [0.0] * source_frame_count
    previous_matched = None
    matched = core.tivtc.TFM(
        clip,
        order=field["tfm_order"],
        cthresh=MM_TFM_CTHRESH,
        MI=MM_TFM_MI,
        slow=MM_TFM_SLOW,
        input=str(tfm_path),
    )
    with ThreadPoolExecutor(max_workers=metric_workers) as executor:
        metric_futures = {}
        for i, fr in enumerate(matched.frames(prefetch=n_threads)):
            matched_arr = np.asarray(fr[0]).astype(np.int32)
            metric_futures[i] = executor.submit(_matched_temporal_difference, matched_arr, previous_matched)
            previous_matched = matched_arr
            drain_index = i - max_pending
            if drain_index >= 0:
                matched_temporal_diff[drain_index] = metric_futures.pop(drain_index).result()
        for index, future in metric_futures.items():
            matched_temporal_diff[index] = future.result()
    print(f"    Matched temporal scan completed in {time.time()-t0:.1f}s")

    locked_60i_mask = _vertical_scroll_force_mask(source_frame_count, vertical_scroll_hits)
    evidence = _build_multimetric_evidence(motion_arr, locked_60i_mask, timeline)
    shadow = _build_shadow_strategy(
        tfm_records,
        sensitive_combed_vetoes,
        motion_arr,
        evidence,
    )
    shadow, resolved_low_information_runs = _resolve_low_information_runs(shadow, evidence["low_information"], evidence["decision_boundaries"])
    if resolved_low_information_runs:
        resolved_count = sum(end - start + 1 for start, end, _state in resolved_low_information_runs)
        print(f"  Low-information inheritance: resolved {resolved_count} frames across {len(resolved_low_information_runs)} runs")
    shadow, bridged_bob_regions = _bridge_hard_bob_regions(
        shadow,
        evidence["decision_boundaries"],
    )
    if bridged_bob_regions:
        bridged_count = sum(end - start + 1 for start, end in bridged_bob_regions)
        range_text = ", ".join(f"{start}-{end}" for start, end in bridged_bob_regions)
        print(
            f"  Residual-combing regions: bobbed {bridged_count} source frames "
            f"across {len(bridged_bob_regions)} regions ({range_text})"
        )
    shadow, forced_count = _apply_bob_strategy_overrides(shadow, timeline, forced_bob_ranges)
    if forced_bob_ranges:
        source_end = timeline["pts_ms"][-1] + timeline["duration_ms"][-1]
        range_text = ", ".join(
            f"{start / 1000.0:.3f}-{(end if end is not None else source_end) / 1000.0:.3f}s"
            for start, end in forced_bob_ranges
        )
        print(f"  Explicit bob override: forced {forced_count} source frames to bob_expand ({range_text})")
    shadow, boundary_adjustments = _normalize_field_safe_transitions(shadow, tfm_records)
    if boundary_adjustments:
        adjusted_count = sum(adjustment["end"] - adjustment["start"] + 1 for adjustment in boundary_adjustments)
        print(f"  Field-safe boundaries: extended bob by {adjusted_count} source frames across {len(boundary_adjustments)} transitions")
        for adjustment in boundary_adjustments:
            direction = adjustment["direction"].replace("_to_", "->")
            matches = "".join(adjustment["matches"])
            print(f"    {direction}: boundary {adjustment['original_boundary']} -> {adjustment['new_boundary']}, bobbed {adjustment['start']}-{adjustment['end']} (matches={matches})")
    framemap_entries = parse_framemap(framemap_path)
    shadow = _apply_shadow_redundancy(
        shadow,
        timeline,
        framemap_entries,
        matched_temporal_diff,
    )
    cycle_adjustments = []
    post_redundancy_boundary_adjustments = []
    while True:
        shadow, current_cycle_adjustments = _normalize_complete_decimate_cycles(
            shadow,
            timeline,
            framemap_entries,
        )
        cycle_adjustments.extend(current_cycle_adjustments)
        shadow, current_boundary_adjustments = _normalize_field_safe_transitions(
            shadow,
            tfm_records,
        )
        post_redundancy_boundary_adjustments.extend(current_boundary_adjustments)
        if not current_cycle_adjustments and not current_boundary_adjustments:
            break
    if cycle_adjustments:
        adjusted_count = sum(count for _start, _end, count in cycle_adjustments)
        print(
            f"  Complete IVTC cycles: kept {adjusted_count} source frames without decimation "
            f"across {len(cycle_adjustments)} partial cycles or incompatible runs"
        )
    if post_redundancy_boundary_adjustments:
        adjusted_count = sum(
            adjustment["end"] - adjustment["start"] + 1
            for adjustment in post_redundancy_boundary_adjustments
        )
        print(
            f"  Post-IVTC field-safe boundaries: extended bob by {adjusted_count} source frames "
            f"across {len(post_redundancy_boundary_adjustments)} transitions"
        )
    _validate_bob_field_units(shadow, timeline, field_metadata)
    from collections import Counter
    shadow_strategy_counts = Counter(shadow["strategies"])
    shadow_matchability_counts = Counter(shadow["matchability"])
    print(f"  Matchability shadow: {dict(shadow_matchability_counts.most_common())}")
    print(f"  Strategy shadow: {dict(shadow_strategy_counts.most_common())}")

    diagnostic_path = _write_classification_shadow_diagnostics(
        work_dir,
        source_path.stem,
        timeline,
        field_metadata,
        tfm_records,
        selected_weave_mics,
        explicit_rff_mask,
        selected_weave_thresholds,
        sensitive_combed_vetoes,
        motion_arr,
        matched_temporal_diff,
        vertical_scroll_hits,
        evidence,
        shadow,
    )
    run_diagnostic_path = _write_shadow_run_diagnostics(
        work_dir, source_path.stem, timeline, field_metadata, shadow
    )
    print(f"  Shadow diagnostics: {diagnostic_path.name}, {run_diagnostic_path.name}")

    analysis_result: AnalysisResult = {
        "timeline": timeline,
        "field_metadata": field_metadata,
        **shadow,
        "diagnostic_path": diagnostic_path,
        "run_diagnostic_path": run_diagnostic_path,
    }
    validate_analysis_result(analysis_result)
    return analysis_result


def generate_pass2b_script(source_path, tfm_path, stats_path, segments: list[Segment], script_path, resize_w, resize_h, additional_vpy=None, frame_range=None, progressive_source=False, assume_fps_num=30000, assume_fps_den=1001, resize_enabled=False, crop_margins=None, output_yuv444=False, field_order="tff", vs_threads=None):
    """Generate the final VPY that assembles matched, decimated, and bob branches.

    AssumeFPS is only a technical tag for VFR output. Global ``--bob`` uses the
    requested rate so clip metadata agrees with CFR muxing.
    """
    source_esc = str(source_path).replace("\\", "\\\\")
    n_threads = _resolve_thread_count(vs_threads)
    dummy_path = Path(script_path).parent / f"{Path(script_path).stem}_dummy_tc.txt"
    need_matched = any(s["branch"] in ("matched", "decimated") for s in segments)
    need_decimated = any(s["branch"] == "decimated" for s in segments)
    need_bob = any(s["branch"] == "bobbed" for s in segments)
    need_metadata_bob = any(s.get("bob_frame_specs") for s in segments)
    field = _field_order_settings(field_order)
    field_order_tff = field["tff"]
    fieldbased = 0 if progressive_source else field["fieldbased"]
    crop_left, crop_right, crop_top, crop_bottom = crop_margins or (0, 0, 0, 0)
    output_width = resize_w if resize_enabled else None
    output_height = resize_h if resize_enabled else None
    prepare_output_line = f"{{name}} = fmtc_prepare_output({{name}}, width={output_width!r}, height={output_height!r}, crop_left={crop_left}, crop_right={crop_right}, crop_top={crop_top}, crop_bottom={crop_bottom}, output_yuv444={output_yuv444!r})\n"
    matched_decimated_branch_builder = render_matched_decimated_branch_builder()

    script = f'''import vapoursynth as vs
core = vs.core
core.num_threads = {n_threads}
{VPY_FMTC_HELPERS}
{matched_decimated_branch_builder}
clip = core.bs.VideoSource(r"{source_esc}")
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval={fieldbased})
clip = fmtc_to_yuv420p8(clip)

def splice_many(parts, chunk_size=512):
    if not parts:
        raise ValueError("splice_many: empty parts")
    while len(parts) > 1:
        parts = [core.std.Splice(parts[i:i + chunk_size]) for i in range(0, len(parts), chunk_size)]
    return parts[0]

def frames_from_clip(src, indexes):
    return splice_many([src[i:i + 1] for i in indexes])

def frames_from_bob(tff_clip, bff_clip, specs):
    return splice_many([(tff_clip if is_tff else bff_clip)[index:index + 1] for is_tff, index in specs])

segments = []
'''
    if progressive_source:
        script += f'''
progressive = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
{prepare_output_line.format(name="progressive")}\
'''
    elif need_matched:
        if tfm_path is None or stats_path is None:
            raise ValueError("Matched and decimated branches require tfm_path and stats_path")
        tfm_esc = str(tfm_path).replace("\\", "\\\\")
        stats_esc = str(stats_path).replace("\\", "\\\\")
        dummy_esc = str(dummy_path).replace("\\", "\\\\")
        script += f'''
matched_decimated = build_matched_decimated_branches(core, clip, r"{tfm_esc}", {field["tfm_order"]}, r"{stats_esc}", r"{dummy_esc}", {need_decimated}, {MM_TFM_CTHRESH}, {MM_TFM_MI}, {MM_TFM_SLOW})
matched = matched_decimated["matched"]
{prepare_output_line.format(name="matched")}\
'''
        if need_decimated:
            script += f'''
decimated = matched_decimated["decimated"]
{prepare_output_line.format(name="decimated")}\
'''
    if need_bob:
        if need_metadata_bob:
            script += f'''
from vsdeinterlace.qtgmc import QTempGaussMC
from vsaa import NNEDI3
bob_source_tff = core.std.SetFrameProp(clip, prop="_FieldBased", intval=2)
bob_source_bff = core.std.SetFrameProp(clip, prop="_FieldBased", intval=1)
bobbed_tff = QTempGaussMC(bob_source_tff, basic_bobber=NNEDI3(nsize=4, nns=4, qual=2, opencl=True), tff=True, basic_tr=3, final_tr=2, source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED).deinterlace()
bobbed_bff = QTempGaussMC(bob_source_bff, basic_bobber=NNEDI3(nsize=4, nns=4, qual=2, opencl=True), tff=False, basic_tr=3, final_tr=2, source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED).deinterlace()
{prepare_output_line.format(name="bobbed_tff")}\
{prepare_output_line.format(name="bobbed_bff")}\
'''
        else:
            script += f'''
from vsdeinterlace.qtgmc import QTempGaussMC
from vsaa import NNEDI3
bobbed = QTempGaussMC(clip, basic_bobber=NNEDI3(nsize=4, nns=4, qual=2, opencl=True), tff={field_order_tff}, basic_tr=3, final_tr=2, source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED).deinterlace()
{prepare_output_line.format(name="bobbed")}\
'''
    for seg in segments:
        if seg["strategy"] in ("match_keep_pts", "match_decimate"):
            if "kept_positions" in seg:
                positions = [position for position, _run_len in seg["kept_positions"]]
                indexes = [seg["branch_indices"][position] for position in positions]
            else:
                indexes = seg["branch_indices"]
            script += f'segments.append(frames_from_clip({seg["branch"]}, {indexes!r}))\n'
        elif seg["strategy"] == "bob_expand":
            if seg.get("bob_frame_specs"):
                script += (
                    f'segments.append(frames_from_bob(bobbed_tff, bobbed_bff, '
                    f'{seg["bob_frame_specs"]!r}))\n'
                )
            else:
                indexes = []
                for si in seg["src_indices"]:
                    indexes.extend((si * 2, si * 2 + 1))
                script += f'segments.append(frames_from_clip(bobbed, {indexes!r}))\n'
        else:
            raise ValueError(f"Unsupported segment strategy: {seg['strategy']}")

    script += '\nclip = core.std.Splice(segments)\n'

    if additional_vpy is not None:
        with open(additional_vpy, "r", encoding="utf-8") as af:
            script += '\n' + af.read() + '\n'

    if frame_range is not None:
        script += f'clip = clip[{frame_range[0]}:{frame_range[1]}]\n'

    script += f'''clip = core.std.AssumeFPS(clip, fpsnum={assume_fps_num}, fpsden={assume_fps_den})
clip.set_output(0)
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


def _print_progressive_dedup_report(stem, film_dec, film_out, total_out, dedup_stats, tc_final):
    """Print a compact report for progressive-dedup mode."""
    run_hist = dedup_stats.get("run_hist", []) if dedup_stats else []
    run = lambda n: run_hist[n] if n < len(run_hist) else 0
    print("")
    print(f"{'=' * 80}")
    print(f"PROGRESSIVE DEDUP REPORT - {stem}")
    print(f"{'=' * 80}")
    print(f"  Input frames:    {film_dec:8d}")
    print(f"  Full output:     {film_out:8d}")
    print(f"  Removed frames:  {film_dec - film_out:8d} ({(film_dec - film_out) / max(film_dec, 1) * 100:6.2f}%)")
    print(f"  Selected output: {total_out:8d}")
    print("  Dedup runs:")
    for n in range(1, len(run_hist)):
        print(f"    {n}-in-1: {run(n)}")
    print(f"  TC: {tc_final.name}")


def _parse_time_ms(value):
    """Convert hh:mm:ss.xxx, mm:ss.xxx, or seconds to milliseconds."""
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0]) * 1000.0
    if len(parts) == 2:
        return (int(parts[0]) * 60000.0) + (float(parts[1]) * 1000.0)
    if len(parts) == 3:
        return (int(parts[0]) * 3600000.0) + (int(parts[1]) * 60000.0) + (float(parts[2]) * 1000.0)
    raise ValueError(f"Invalid timestamp: {value}")


def _parse_bob_range_spec(spec):
    """Convert comma-separated START-END values to millisecond ranges."""
    ranges = []
    if not spec:
        return ranges
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid bob range: {item}")
        start, end = item.split("-", 1)
        start_ms = _parse_time_ms(start)
        end_ms = _parse_time_ms(end)
        if end_ms <= start_ms:
            raise ValueError(f"Bob range end must be greater than its start: {item}")
        ranges.append((start_ms, end_ms))
    return ranges


def _parse_chapter_list(spec):
    """Convert comma-separated chapter numbers to one-based indices."""
    chapters = []
    if not spec:
        return chapters
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        chapter = int(item)
        if chapter <= 0:
            raise ValueError(f"Invalid chapter: {item}")
        chapters.append(chapter)
    return chapters


def _parse_dedup_cap(value, option_name):
    """Validate a dedup cap supplied through the CLI."""
    if value is None:
        return None
    cap = int(value)
    if cap < 1:
        raise ValueError(f"{option_name} must be >= 1")
    return cap


def _parse_positive_int(value, option_name):
    """Validate an optional positive integer supplied through the CLI."""
    if value is None:
        return None
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a positive integer") from exc
    if number < 1:
        raise ValueError(f"{option_name} must be >= 1")
    return number


def _parse_frame_range(value):
    """Validate an optional output-frame range and return half-open bounds."""
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", value.strip())
    if match is None:
        raise ValueError(f"Invalid --frames value: {value}; expected N or START-END")
    start = int(match.group(1)) if match.group(2) is not None else 0
    end = int(match.group(2)) if match.group(2) is not None else int(match.group(1))
    if end <= start:
        raise ValueError(f"Invalid --frames range: {start}-{end}; END must be greater than START")
    return start, end


def _parse_resize_spec(value):
    """Validate a CLI resolution in WIDTHxHEIGHT form."""
    if value is None:
        return None
    parts = value.lower().split("x", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--resize must use WIDTHxHEIGHT form, for example 768x576")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise ValueError("--resize must contain numbers only, for example 768x576") from exc
    if width <= 0 or height <= 0:
        raise ValueError("--resize requires width and height greater than zero")
    return width, height


def _parse_crop_spec(value):
    """Validate crop margins in LEFT:RIGHT:TOP:BOTTOM form."""
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError("--crop must use LEFT:RIGHT:TOP:BOTTOM form, for example 8:8:0:0")
    try:
        margins = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--crop must contain four integer pixel margins") from exc
    if any(margin < 0 for margin in margins):
        raise ValueError("--crop margins cannot be negative")
    return margins


def _validate_crop_geometry(crop_margins, width, height, output_yuv444):
    """Return cropped dimensions after validating format alignment and bounds."""
    if crop_margins is None:
        return width, height
    left, right, top, bottom = crop_margins
    if not output_yuv444 and any(margin % 2 for margin in crop_margins):
        raise ValueError("--crop margins must be even for YUV 4:2:0 output; use --yuv444 to allow odd margins")
    cropped_width = width - left - right
    cropped_height = height - top - bottom
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError(f"--crop {left}:{right}:{top}:{bottom} removes the complete {width}x{height} frame")
    return cropped_width, cropped_height


def _parse_fps_ratio(value, option_name):
    """Validate a CLI frame-rate ratio such as 60000/1001."""
    raw = value.strip()
    if not raw:
        raise ValueError(f"{option_name} requires a valid frame-rate ratio")

    parts = raw.split("/", 1)
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"{option_name} must use NUM/DEN form, for example 60000/1001 or 50/1")

    num = int(parts[0])
    den = int(parts[1])

    if num <= 0 or den <= 0:
        raise ValueError(f"{option_name} must be greater than zero")
    return num, den


def _format_fps_ratio(fps_ratio):
    num, den = fps_ratio
    return f"{num}/{den}"


def _bob_ranges_from_chapters(source_path, work_dir, chapters):
    """Extract chapter time ranges that must be forced to bob."""
    if not chapters:
        return []
    chapters_xml = work_dir / f"{source_path.stem}_chapters.xml"
    chapter_ranges = extract_chapter_ranges(source_path, chapters_xml)
    ranges = []
    for chapter in chapters:
        idx = chapter - 1
        if idx < 0 or idx >= len(chapter_ranges):
            raise RuntimeError(f"Chapter {chapter} is not present in the source file")
        ranges.append(chapter_ranges[idx])
    return ranges


def process_episode(source_path, output_path, work_dir, strip_audio, strip_sub, additional_vpy=None, frame_range=None, bob=False, analyze_only=False, progressive_dedup=False, dedup_enabled=False, dedup_cap=None, bob_chapters=None, bob_ranges=None, crop_margins=None, resize_target=None, output_yuv444=False, field_order="tff", bob_fps=(60000, 1001), vs_threads=None, analysis_workers=None) -> EpisodeStats:
    """Run source analysis, classification, segmentation, timecodes, and output."""
    stem = source_path.stem
    print(f"\n{'=' * 60}")
    print(f"Processing: {output_path.name}")
    if frame_range is not None:
        print(f"  Frame range: {frame_range[0]}-{frame_range[1]}")
    print(f"{'=' * 60}")

    w, h, sar_num, sar_den = get_video_info(source_path)
    source_frame_count = get_video_frame_count(source_path)
    cropped_w, cropped_h = _validate_crop_geometry(crop_margins, w, h, output_yuv444)
    crop_description = ""
    if crop_margins is not None and any(crop_margins):
        crop_description = f" -> crop {cropped_w}x{cropped_h}"
    if resize_target is None:
        resize_w, resize_h = cropped_w, cropped_h
        resize_enabled = False
        par_flags_enabled = sar_num != sar_den
        display_aspect_ratio = None
        if sar_num != sar_den:
            dar = Fraction(cropped_w * sar_num, cropped_h * sar_den)
            display_aspect_ratio = f"{dar.numerator}/{dar.denominator}"
        print(f"  Source: {w}x{h} SAR {sar_num}:{sar_den}{crop_description} -> no resize")
    else:
        resize_w, resize_h = resize_target
        resize_enabled = True
        par_flags_enabled = False
        display_aspect_ratio = None
        print(f"  Source: {w}x{h} SAR {sar_num}:{sar_den}{crop_description} -> {resize_w}x{resize_h}")

    src_tc_path = work_dir / f"{stem}_src_timecodes.txt"
    if not src_tc_path.exists():
        extract_source_timecodes(source_path, src_tc_path)

    stats_path = None
    tfm_path = None
    analysis = None
    if progressive_dedup:
        print("  --progressive-dedup enabled: skipping TIVTC/classifier/bob and deduplicating the progressive source directly")
        entries = make_progressive_entries_from_source_timecodes(src_tc_path, source_frame_count)
        segments = make_linear_strategy_segments(source_frame_count, "match_keep_pts", "progressive")
    elif bob:
        print(f"  --bob enabled: skipping TIVTC/classifier/dedup and forcing video_bob for the entire source ({_format_fps_ratio(bob_fps)}fps)")
        entries = make_bob_entries_from_source_timecodes(src_tc_path, source_frame_count)
        segments = make_linear_strategy_segments(source_frame_count, "bob_expand", "bobbed")
    else:
        stats_path, tfm_path = run_pass1(
            source_path,
            work_dir,
            field_order=field_order,
            vs_threads=vs_threads,
        )
        _tc_v1_path, framemap_path = run_pass2a(
            source_path,
            stats_path,
            tfm_path,
            work_dir,
            field_order=field_order,
            vs_threads=vs_threads,
        )
        entries = parse_framemap(framemap_path)
        forced_ranges = []
        if bob_chapters:
            forced_ranges.extend(_bob_ranges_from_chapters(source_path, work_dir, bob_chapters))
        if bob_ranges:
            forced_ranges.extend(bob_ranges)
        analysis = run_multimetric_classification(
            source_path,
            work_dir,
            tfm_path,
            src_tc_path,
            framemap_path,
            field_order=field_order,
            vs_threads=vs_threads,
            analysis_workers=analysis_workers,
            forced_bob_ranges=forced_ranges,
        )
        segments = strategies_to_segments(
            analysis["strategies"],
            entries,
            redundancy_mapping=analysis["redundancy_mapping"],
            origins=analysis["origins"],
            field_units=analysis["timeline"]["field_units"],
            field_metadata=analysis["field_metadata"],
            matchability=analysis["matchability"],
            redundancy=analysis["redundancy"],
            redundancy_origins=analysis["redundancy_origins"],
            locked_matchable=analysis["locked_matchable"],
            locked_bob=analysis["locked_bob"],
        )

    segment_types = sorted({s["strategy"] for s in segments})
    invalid_types = [
        strategy for strategy in segment_types
        if strategy not in ("match_keep_pts", "match_decimate", "bob_expand")
    ]
    if invalid_types:
        raise RuntimeError(f"Unsupported segment strategies: {invalid_types}")
    validate_segments(segments, source_frame_count)

    # Film-segment deduplication; global bob mode has no film segments.
    effective_dedup_cap = dedup_cap or MM_DEDUP_CAP
    dedup_active = (progressive_dedup or dedup_enabled) and MM_DEDUP_ENABLED and not bob
    dedup_stats: DedupStats = {"input": 0, "output": 0, "saved": 0, "saved_pct": 0.0, "run_hist": [0] * (effective_dedup_cap + 1)}
    if progressive_dedup and dedup_active:
        dedup_stats = run_progressive_dedup_detection(
            source_path,
            segments,
            cap=effective_dedup_cap,
            vs_threads=vs_threads,
        )
    elif dedup_active:
        dedup_stats = run_dedup_detection(source_path, work_dir, tfm_path, stats_path, segments,
                                          cap=effective_dedup_cap, field_order=field_order,
                                          vs_threads=vs_threads)

    source_counts = {
        strategy: sum(
            segment["source_frame_count"]
            for segment in segments if segment["strategy"] == strategy
        )
        for strategy in ("match_keep_pts", "match_decimate", "bob_expand")
    }
    base_output_counts = {
        strategy: sum(
            segment["base_output_frame_count"]
            for segment in segments if segment["strategy"] == strategy
        )
        for strategy in ("match_keep_pts", "match_decimate", "bob_expand")
    }
    output_counts = {
        strategy: sum(
            segment_output_frame_count(segment)
            for segment in segments if segment["strategy"] == strategy
        )
        for strategy in ("match_keep_pts", "match_decimate", "bob_expand")
    }
    matched_base = base_output_counts["match_keep_pts"] + base_output_counts["match_decimate"]
    if not dedup_active:
        dedup_stats = {"input": matched_base, "output": matched_base, "saved": 0, "saved_pct": 0.0, "run_hist": [0] * (effective_dedup_cap + 1)}
        if matched_base and len(dedup_stats["run_hist"]) > 1:
            dedup_stats["run_hist"][1] = matched_base
    validate_segments(segments, source_frame_count)
    validate_dedup_stats(dedup_stats)
    total_out = sum(output_counts.values())
    full_output_frame_count = total_out
    film_dec = matched_base
    film_out = output_counts["match_keep_pts"] + output_counts["match_decimate"]
    bob_dec = source_counts["bob_expand"]
    bob_out = output_counts["bob_expand"]
    if progressive_dedup:
        print(f"  Progressive: {film_dec} -> {film_out} frames (dedup -{film_dec - film_out})")
    else:
        print(
            f"  match_keep_pts: {source_counts['match_keep_pts']} source -> "
            f"{output_counts['match_keep_pts']} output"
        )
        print(
            f"  match_decimate: {source_counts['match_decimate']} source -> "
            f"{output_counts['match_decimate']} output "
            f"(structural drops {source_counts['match_decimate'] - base_output_counts['match_decimate']})"
        )
        print(f"  bob_expand:     {bob_dec} source -> {bob_out} output")
        if dedup_active:
            print(f"  Optional dedup: {dedup_stats['input']} -> {dedup_stats['output']}")
    print(f"  Output:    {total_out} total frames")

    full_timecodes_path = work_dir / (f"{stem}_tc_full.txt" if frame_range is not None else f"{stem}_tc_final.txt")
    num_tc = generate_final_timecodes_v2(segments, src_tc_path, full_timecodes_path, strict_bob_field_units=not bob)
    if num_tc != total_out:
        raise RuntimeError(f"Mismatch: timecodes={num_tc} vs output={total_out}")

    tc_final = full_timecodes_path
    audio_range = None
    if frame_range is not None:
        with open(full_timecodes_path, "r", encoding="utf-8") as f:
            full_timestamps = [float(line.strip()) for line in f if line.strip() and not line.startswith("#")]
        if len(full_timestamps) != num_tc:
            raise RuntimeError(f"Invalid full timecode cardinality: {len(full_timestamps)} timestamps for {num_tc} frames")
        all_tc = full_timestamps
        if frame_range[0] >= len(all_tc):
            raise ValueError(f"Frame range starts beyond the output: {frame_range[0]} >= {len(all_tc)}")
        fr_start = frame_range[0]
        fr_end = min(frame_range[1], len(all_tc))
        if fr_end <= fr_start:
            raise ValueError(f"Frame range is empty after clipping: {fr_start}-{fr_end}")
        selected_tc = all_tc[fr_start:fr_end]
        t_origin = selected_tc[0]
        range_end_ms = (
            all_tc[fr_end]
            if fr_end < len(all_tc)
            else source_end_ms(read_timecodes_v2(src_tc_path), source_frame_count)
        )
        trimmed_tc = [tc - t_origin for tc in selected_tc]
        tc_final = work_dir / f"{stem}_tc_final.txt"
        with open(tc_final, "w", encoding="utf-8") as f:
            f.write("# timecode format v2\n")
            for tc in trimmed_tc:
                f.write(f"{tc:.6f}\n")
        total_out = len(trimmed_tc)
        ss_s = all_tc[fr_start] / 1000.0
        to_s = range_end_ms / 1000.0
        def _fmt_ts(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        audio_range = (_fmt_ts(ss_s), _fmt_ts(to_s))
        print(f"  Frame range: {fr_start}-{fr_end} ({total_out} frames)")

    vpy_path = work_dir / f"{stem}_pass2b.vpy"
    assume_fps_num = bob_fps[0] if bob else 30000
    assume_fps_den = bob_fps[1] if bob else 1001
    generate_pass2b_script(source_path, tfm_path, stats_path, segments, vpy_path, resize_w, resize_h, additional_vpy, frame_range, progressive_source=progressive_dedup, assume_fps_num=assume_fps_num, assume_fps_den=assume_fps_den, resize_enabled=resize_enabled, crop_margins=crop_margins, output_yuv444=output_yuv444, field_order=field_order, vs_threads=vs_threads)

    episode_stats: EpisodeStats = {
        "name": output_path.name,
        "mode": "bob" if bob else ("progressive_dedup" if progressive_dedup else "hybrid"),
        "source_frames": source_frame_count,
        "film_frames_24": film_out,
        "video_frames_60": bob_out,
        "match_keep_pts_frames": output_counts["match_keep_pts"],
        "match_decimate_frames": output_counts["match_decimate"],
        "bob_expand_frames": output_counts["bob_expand"],
        "dedup_saved": dedup_stats.get("saved", 0),
        "full_output_frames": full_output_frame_count,
        "total_out_frames": total_out,
        "film_pct": film_out / max(full_output_frame_count, 1) * 100,
        "video60_pct": bob_out / max(full_output_frame_count, 1) * 100,
        "resolution": f"{resize_w}x{resize_h}",
    }

    validate_episode_stats(episode_stats)

    if analyze_only:
        if progressive_dedup:
            _print_progressive_dedup_report(stem, film_dec, film_out, total_out, dedup_stats, tc_final)
        else:
            print_strategy_analyze_report(stem, segments, src_tc_path, full_timecodes_path, dedup_stats)
        print(f"  Analyze-only: skipping encode/mux. VPY: {vpy_path.name}, TC: {tc_final.name}")
        return episode_stats

    encoded_path = work_dir / f"{stem}_encoded.mkv"
    if encoded_path.exists():
        encoded_path.unlink()
    is_ffmpeg = "ffmpeg" in ENCODER_BIN.lower()
    color_flags = get_color_flags(source_path, is_ffmpeg)
    par_flags = get_par_flags(sar_num, sar_den, is_ffmpeg) if par_flags_enabled else ""
    chroma_flags = get_chroma_flags(output_yuv444, is_ffmpeg)
    if color_flags:
        print(f"  Color tags: {color_flags}")
    if par_flags:
        print(f"  PAR tag: {sar_num}:{sar_den}")
    if chroma_flags:
        print("  Chroma output: yuv444p10")
    if display_aspect_ratio is not None:
        print(f"  Aspect tag: {display_aspect_ratio}")
    encode(vpy_path, encoded_path, color_flags, par_flags, chroma_flags)

    mux_timecodes = None if bob else tc_final
    default_duration = f"{_format_fps_ratio(bob_fps)}fps" if bob else None
    mux_final(
        encoded_path,
        source_path,
        mux_timecodes,
        output_path,
        strip_audio,
        strip_sub,
        audio_range,
        default_duration=default_duration,
        display_aspect_ratio=display_aspect_ratio,
    )

    if encoded_path.exists():
        encoded_path.unlink()
    print(f"  Completed: {output_path.name}")

    return episode_stats


def _output_path_for_source(source, output_dir, output_is_explicit):
    """Build an output name from the source name and anti-overwrite suffix."""
    if output_is_explicit:
        output_path = output_dir / source.name
        if output_path.resolve() == source.resolve():
            raise RuntimeError("The explicit output path matches the source; choose a different directory")
        return output_path
    return source.parent / f"{source.stem}_1{source.suffix}"


def _cleanup_work_dir(work_dir):
    """Remove intermediate files produced in the work directory."""
    for f in work_dir.iterdir():
        if f.is_file():
            f.unlink()
    try:
        work_dir.rmdir()
    except OSError:
        pass


def _make_run_work_dir(work_root, source):
    """Create a unique subdirectory to prevent parallel-run collisions."""
    run_id = uuid.uuid4().hex[:12]
    run_dir = work_root / f"{source.stem}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _print_progressive_summary(all_stats: list[EpisodeStats]):
    """Print a compact summary for one or more progressive-dedup sources."""
    total_in = sum(s["source_frames"] for s in all_stats)
    full_output = sum(s["full_output_frames"] for s in all_stats)
    selected_output = sum(s["total_out_frames"] for s in all_stats)
    total_saved = sum(s["dedup_saved"] for s in all_stats)
    print(f"\n{'=' * 80}\nPROGRESSIVE DEDUP SUMMARY\n{'=' * 80}")
    if len(all_stats) > 1:
        print(f"{'File':<35} {'input':>8} {'full out':>9} {'selected':>9} {'saved':>8} {'saved%':>7}")
        for s in all_stats:
            print(f"  {s['name']:<33} {s['source_frames']:>8} {s['full_output_frames']:>9} {s['total_out_frames']:>9} "
                  f"{s['dedup_saved']:>8} {s['dedup_saved'] / max(s['source_frames'], 1) * 100:>6.1f}%")
        print(f"  {'TOTAL':<33} {total_in:>8} {full_output:>9} {selected_output:>9} {total_saved:>8} "
              f"{total_saved / max(total_in, 1) * 100:>6.1f}%")
    else:
        s = all_stats[0]
        print(f"  Input:       {s['source_frames']:>8} frames")
        print(f"  Full output: {s['full_output_frames']:>8} frames")
        if s["total_out_frames"] != s["full_output_frames"]:
            print(f"  Selected:    {s['total_out_frames']:>8} frames")
        print(f"  Removed:     {s['dedup_saved']:>8} ({s['dedup_saved'] / max(s['source_frames'], 1) * 100:.1f}%)")
        print(f"  Resolution:  {s['resolution']}")


def _print_hybrid_summary(all_stats: list[EpisodeStats]):
    """Print a compact summary of operational strategies."""
    total_keep = sum(s["match_keep_pts_frames"] for s in all_stats)
    total_decimate = sum(s["match_decimate_frames"] for s in all_stats)
    total_bob = sum(s["bob_expand_frames"] for s in all_stats)
    full_output = sum(s["full_output_frames"] for s in all_stats)
    selected_output = sum(s["total_out_frames"] for s in all_stats)
    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    if len(all_stats) > 1:
        print(f"{'File':<35} {'keep':>8} {'decimate':>8} {'bob':>8} {'full out':>9} {'selected':>9}")
        for s in all_stats:
            print(
                f"  {s['name']:<33} {s['match_keep_pts_frames']:>8} "
                f"{s['match_decimate_frames']:>8} {s['bob_expand_frames']:>8} "
                f"{s['full_output_frames']:>9} {s['total_out_frames']:>9}"
            )
        print(
            f"  {'TOTAL':<33} {total_keep:>8} {total_decimate:>8} "
            f"{total_bob:>8} {full_output:>9} {selected_output:>9}"
        )
    else:
        s = all_stats[0]
        print(f"  match_keep_pts: {s['match_keep_pts_frames']:>8}")
        print(f"  match_decimate: {s['match_decimate_frames']:>8}")
        print(f"  bob_expand:     {s['bob_expand_frames']:>8}")
        print(f"  Full output: {s['full_output_frames']} frames")
        if s["total_out_frames"] != s["full_output_frames"]:
            print(f"  Selected output: {s['total_out_frames']} frames")
        print(f"  Resolution: {s['resolution']}")


def main():
    parser = argparse.ArgumentParser(description="PTS-aware VFR pipeline for hybrid sources")
    parser.add_argument("source", help="Source MKV file or directory")
    parser.add_argument("--report", action="store_true", help="Generate only the VFR report for an MKV file or directory; must be used alone")
    parser.add_argument("--output", default=None, help="Output directory; defaults to the source directory with an _1 suffix")
    parser.add_argument("--strip-audio", action="store_true", help="Remove audio tracks from the mux")
    parser.add_argument("--strip-sub", action="store_true", help="Remove subtitle tracks from the mux")
    parser.add_argument("--analyze-only", action="store_true", help="Run analysis, classification, dedup, timecodes, and VPY generation without encode or mux")
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate files")
    parser.add_argument("--work-dir", default=None, help="Work directory; defaults to <output>/work")
    parser.add_argument("--additional-vpy", default=None, help="Additional VPY code appended to pass2b; operates on 'clip'")
    parser.add_argument("--frames", default=None, help="Half-open output-frame range to process, for example 1500 or 100-5000")
    parser.add_argument("--bob", nargs="?", const="60000/1001", default=None, metavar="NUM/DEN", help="Force bob deinterlacing for the whole title and skip classification/dedup; optional CFR mux rate defaults to 60000/1001")
    parser.add_argument("--bob-chapters", default=None, help="Force one-based chapter numbers to bob, for example 4 or 4,5,6")
    parser.add_argument("--bob-range", default=None, help="Force comma-separated START-END time ranges to bob, for example 22:30-23:50,10:00-10:20")
    parser.add_argument("--field-order", choices=("tff", "bff"), default="tff", help="Field order for TFM/QTGMC; defaults to tff")
    parser.add_argument("--progressive-dedup", nargs="?", const="2", default=None, metavar="N", help="Deduplicate a progressive source; optional N is the maximum merged run and defaults to 2")
    parser.add_argument("--dedup", nargs="?", const="2", default=None, metavar="N", help="Enable dedup on matched/decimated segments; optional N is the maximum merged run and defaults to 2")
    parser.add_argument("--yuv444", action="store_true", help="Produce YUV 4:4:4 10-bit video instead of YUV 4:2:0 10-bit")
    parser.add_argument("--crop", default=None, metavar="L:R:T:B", help="Crop pixels from left, right, top, and bottom before resize")
    parser.add_argument("--resize", default=None, metavar="WIDTHxHEIGHT", help="Resize output to WIDTHxHEIGHT, for example 768x576")
    parser.add_argument("--threads", default=None, metavar="N", help="Override VapourSynth and prefetch threads; defaults to os.cpu_count()")
    parser.add_argument("--analysis-workers", default=None, metavar="N", help="Override NumPy metric workers; defaults to --threads")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: {source} was not found")
        sys.exit(1)
    progressive_dedup_enabled = args.progressive_dedup is not None
    dedup_enabled = args.dedup is not None
    bob_enabled = args.bob is not None

    if bob_enabled and progressive_dedup_enabled:
        print("Error: --bob and --progressive-dedup are mutually exclusive")
        sys.exit(1)
    if bob_enabled and (args.bob_chapters or args.bob_range):
        print("Error: --bob cannot be combined with --bob-chapters or --bob-range")
        sys.exit(1)
    if progressive_dedup_enabled and (args.bob_chapters or args.bob_range):
        print("Error: --progressive-dedup cannot be combined with --bob-chapters or --bob-range")
        sys.exit(1)

    if args.report:
        forbidden = []
        if args.strip_audio: forbidden.append("--strip-audio")
        if args.strip_sub: forbidden.append("--strip-sub")
        if args.analyze_only: forbidden.append("--analyze-only")
        if args.keep_work: forbidden.append("--keep-work")
        if args.output is not None: forbidden.append("--output")
        if args.work_dir is not None: forbidden.append("--work-dir")
        if args.additional_vpy is not None: forbidden.append("--additional-vpy")
        if args.frames is not None: forbidden.append("--frames")
        if args.threads is not None: forbidden.append("--threads")
        if args.analysis_workers is not None: forbidden.append("--analysis-workers")
        if bob_enabled: forbidden.append("--bob")
        if args.bob_chapters is not None: forbidden.append("--bob-chapters")
        if args.bob_range is not None: forbidden.append("--bob-range")
        if args.field_order != "tff": forbidden.append("--field-order")
        if progressive_dedup_enabled: forbidden.append("--progressive-dedup")
        if dedup_enabled: forbidden.append("--dedup")
        if args.yuv444: forbidden.append("--yuv444")
        if args.crop is not None: forbidden.append("--crop")
        if args.resize is not None: forbidden.append("--resize")
        if forbidden:
            print(f"Error: --report cannot be combined with: {', '.join(forbidden)}")
            sys.exit(1)
        run_report(source)
        return

    try:
        bob_chapters = _parse_chapter_list(args.bob_chapters)
        bob_ranges = _parse_bob_range_spec(args.bob_range)
        bob_fps = _parse_fps_ratio(args.bob, "--bob") if bob_enabled else (60000, 1001)
        progressive_dedup_cap = _parse_dedup_cap(args.progressive_dedup, "--progressive-dedup")
        hybrid_dedup_cap = _parse_dedup_cap(args.dedup, "--dedup")
        crop_margins = _parse_crop_spec(args.crop)
        resize_target = _parse_resize_spec(args.resize)
        vs_threads = _parse_positive_int(args.threads, "--threads")
        analysis_workers = _parse_positive_int(args.analysis_workers, "--analysis-workers")
        frame_range = _parse_frame_range(args.frames)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    dedup_cap = progressive_dedup_cap if progressive_dedup_enabled else hybrid_dedup_cap

    output_is_explicit = args.output is not None
    output_dir = Path(args.output) if output_is_explicit else (source if source.is_dir() else source.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir) if args.work_dir else output_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        sources = sorted(source.glob("*.mkv"))
        if not sources:
            print(f"No MKV files found in {source}")
            sys.exit(1)
        print(f"Directory: {source} ({len(sources)} files)")
        print(f"Output:    {output_dir}")
    else:
        sources = [source]
        print(f"Source: {source}")
        print("Mode:   single file")

    all_stats = []
    for current_source in sources:
        output_path = _output_path_for_source(current_source, output_dir, output_is_explicit)
        work_dir = _make_run_work_dir(work_root, current_source)
        print(f"  {current_source.name} -> {output_path.name}")
        episode_stats = process_episode(current_source, output_path, work_dir, args.strip_audio, args.strip_sub, args.additional_vpy, frame_range, bob=bob_enabled, analyze_only=args.analyze_only, progressive_dedup=progressive_dedup_enabled, dedup_enabled=dedup_enabled, dedup_cap=dedup_cap, bob_chapters=bob_chapters, bob_ranges=bob_ranges, crop_margins=crop_margins, resize_target=resize_target, output_yuv444=args.yuv444, field_order=args.field_order, bob_fps=bob_fps, vs_threads=vs_threads, analysis_workers=analysis_workers)
        all_stats.append(episode_stats)
        if not args.keep_work:
            _cleanup_work_dir(work_dir)

    if progressive_dedup_enabled:
        _print_progressive_summary(all_stats)
    else:
        _print_hybrid_summary(all_stats)


if __name__ == "__main__":
    main()
