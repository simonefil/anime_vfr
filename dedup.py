# -*- coding: utf-8 -*-
"""Mapping-aware deduplication for matched and decimated segments."""

import os
import time

import numpy as np

from branches import build_matched_decimated_branches
from config import MM_DEDUP_CAP, MM_DEDUP_THRESH
from contracts import DedupStats, Segment, validate_dedup_stats, validate_segments
from utils import box_max_16


def _resolve_thread_count(value):
    """Normalize an optional thread override to at least one worker."""
    if value is None:
        value = os.cpu_count() or 16
    return max(1, int(value))


def _fmtc_to_yuv420p8(core, vs, clip):
    clip = core.fmtc.bitdepth(clip, bits=16)
    clip = core.fmtc.resample(
        clip,
        csp=vs.YUV420P16,
        kernel="spline16",
        cplaces="mpeg2",
        cplaced="mpeg2",
    )
    return core.fmtc.bitdepth(clip, bits=8)


def _field_order_settings(field_order):
    tff = field_order == "tff"
    return {
        "fieldbased": 2 if tff else 1,
        "tfm_order": 1 if tff else 0,
    }


def run_dedup_detection(source_path, work_dir, tfm_path, stats_path, segments: list[Segment], threshold=None, cap=None, field_order="tff", vs_threads=None) -> DedupStats:
    """Detect duplicate runs on real branches and attach kept positions."""
    import vapoursynth as vs
    core = vs.core
    if threshold is None:
        threshold = MM_DEDUP_THRESH
    if cap is None:
        cap = MM_DEDUP_CAP

    validate_segments(segments)
    matched_segments = [
        segment
        for segment in segments
        if segment["strategy"] in ("match_keep_pts", "match_decimate")
    ]
    if not matched_segments:
        print("  Dedup: no matched segments, skipping")
        stats: DedupStats = {
            "input": 0,
            "output": 0,
            "saved": 0,
            "saved_pct": 0.0,
            "run_hist": [0] * ((cap or MM_DEDUP_CAP) + 1),
        }
        validate_dedup_stats(stats)
        return stats

    n_threads = _resolve_thread_count(vs_threads)
    core.num_threads = n_threads
    print(
        f"  Dedup detection (cap-{cap}, threshold={threshold}) — "
        f"core.num_threads={n_threads}..."
    )

    dummy_tc = work_dir / "_dedup_dummy_tc.txt"
    branches = _build_strategy_clips(
        core,
        vs,
        source_path,
        tfm_path,
        stats_path,
        dummy_tc,
        field_order,
        need_decimated=any(segment["branch"] == "decimated" for segment in matched_segments),
    )
    stats = _run_dedup_on_branches(branches, segments, n_threads, threshold, cap)

    try:
        if dummy_tc.exists():
            dummy_tc.unlink()
    except OSError:
        pass

    return stats


def run_progressive_dedup_detection(source_path, segments: list[Segment], threshold=None, cap=None, vs_threads=None) -> DedupStats:
    """Detect duplicates directly on a progressive source."""
    import vapoursynth as vs
    core = vs.core
    if threshold is None:
        threshold = MM_DEDUP_THRESH
    if cap is None:
        cap = MM_DEDUP_CAP

    validate_segments(segments)
    matched_segments = [
        segment for segment in segments
        if segment["strategy"] == "match_keep_pts"
    ]
    if not matched_segments:
        print("  Progressive dedup: no matched segments, skipping")
        stats: DedupStats = {
            "input": 0,
            "output": 0,
            "saved": 0,
            "saved_pct": 0.0,
            "run_hist": [0] * ((cap or MM_DEDUP_CAP) + 1),
        }
        validate_dedup_stats(stats)
        return stats

    n_threads = _resolve_thread_count(vs_threads)
    core.num_threads = n_threads
    print(
        f"  Progressive dedup detection (cap-{cap}, threshold={threshold}) — "
        f"core.num_threads={n_threads}..."
    )

    clip = core.bs.VideoSource(str(source_path), threads=0)
    clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
    clip = _fmtc_to_yuv420p8(core, vs, clip)
    return _run_dedup_on_branches({"progressive": clip}, segments, n_threads, threshold, cap)


def _run_dedup_on_branches(branches, segments: list[Segment], n_threads, threshold, cap) -> DedupStats:
    """Apply deduplication to each segment's real branch indices."""
    matched_segments = [
        segment for segment in segments
        if segment["strategy"] in ("match_keep_pts", "match_decimate")
    ]
    print(f"    Computing differences within {len(matched_segments)} matched segment(s)...")
    t0 = time.time()
    segment_diffs = {}
    for segment_index, segment in enumerate(matched_segments):
        indexes = segment["branch_indices"]
        if not indexes:
            raise RuntimeError(
                f"Segment {segment['src_start']}-{segment['src_end']} "
                f"has no branch indices for deduplication"
            )
        branch = branches.get(segment["branch"])
        if branch is None:
            raise RuntimeError(f"Dedup branch was not built: {segment['branch']}")
        prev_arr = None
        diffs = [float("inf")] * len(indexes)
        contiguous = indexes == list(range(indexes[0], indexes[0] + len(indexes)))
        if contiguous:
            frames = branch[indexes[0]:indexes[-1] + 1].frames(prefetch=n_threads)
        else:
            frames = (branch.get_frame(index) for index in indexes)
        for position, fr in enumerate(frames):
            arr = np.asarray(fr[0]).astype(np.int32)
            if prev_arr is not None:
                diffs[position] = box_max_16(np.abs(arr - prev_arr))
            prev_arr = arr
        segment_diffs[segment_index] = diffs
    print(f"    Difference pass completed in {time.time() - t0:.1f}s")

    n_total_in = 0
    n_total_out = 0
    run_hist = [0] * (cap + 1)
    for segment_index, segment in enumerate(matched_segments):
        frame_count = len(segment["branch_indices"])
        diffs = segment_diffs[segment_index]
        kept = []
        position = 0
        while position < frame_count:
            run_len = 1
            while run_len < cap and position + run_len < frame_count:
                d = diffs[position + run_len]
                if d >= threshold:
                    break
                run_len += 1
            kept.append((position, run_len))
            run_hist[run_len] += 1
            position += run_len
        segment["kept_positions"] = kept
        n_total_in += frame_count
        n_total_out += len(kept)

    saved = n_total_in - n_total_out
    pct = 100.0 * saved / max(n_total_in, 1)
    print(f"  Dedup: {n_total_in} -> {n_total_out} matched frames (saved {saved}, -{pct:.1f}%)")
    print("    Run lengths: " + ", ".join(f"{rl}x{run_hist[rl]}" for rl in range(1, cap + 1)))

    stats: DedupStats = {
        "input": n_total_in,
        "output": n_total_out,
        "saved": saved,
        "saved_pct": pct,
        "run_hist": run_hist,
    }
    validate_segments(segments)
    validate_dedup_stats(stats)
    return stats


def _build_strategy_clips(core, vs, source_path, tfm_path, stats_path, mkvout_path, field_order, need_decimated):
    """Build the matched and decimated branches also used by pass2b."""
    field = _field_order_settings(field_order)
    clip = core.bs.VideoSource(str(source_path))
    clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=field["fieldbased"])
    clip = _fmtc_to_yuv420p8(core, vs, clip)
    return build_matched_decimated_branches(core, clip, tfm_path, field["tfm_order"], stats_path, mkvout_path, need_decimated)
