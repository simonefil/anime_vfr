# -*- coding: utf-8 -*-
"""Optional duplicate-run detection for already-progressive sources."""

import os
import time

import numpy as np

from config import MM_DEDUP_CAP, MM_DEDUP_THRESH
from contracts import validate_dedup_stats, validate_segments
from utils import box_max_16
from video_source import BESTSOURCE, open_video_source


def run_progressive_dedup_detection(
    source_path,
    work_dir,
    segments,
    threshold=None,
    cap=None,
    vs_threads=None,
    source_threads=0,
    source_backend=BESTSOURCE,
):
    """Detect duplicate runs directly on an explicitly progressive source."""
    import vapoursynth as vs

    core = vs.core
    threshold = MM_DEDUP_THRESH if threshold is None else threshold
    cap = MM_DEDUP_CAP if cap is None else cap
    validate_segments(segments)
    matched_segments = [
        segment for segment in segments if segment["strategy"] == "match_keep_pts"
    ]
    if not matched_segments:
        stats = {
            "input": 0,
            "output": 0,
            "saved": 0,
            "saved_pct": 0.0,
            "run_hist": [0] * (cap + 1),
        }
        validate_dedup_stats(stats)
        return stats

    threads = max(1, int(vs_threads or os.cpu_count() or 16))
    core.num_threads = threads
    ffindex = work_dir / f"{source_path.stem}_ffms2.ffindex"
    clip = open_video_source(
        core,
        source_path,
        source_backend,
        cache_path=ffindex,
        threads=source_threads,
    )
    clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
    clip = _to_yuv420p8(core, vs, clip)
    print(
        f"  Progressive dedup (cap-{cap}, threshold={threshold}, threads={threads})"
    )
    started = time.time()
    stats = _detect_runs(clip, matched_segments, threads, threshold, cap)
    print(f"  Progressive dedup completed in {time.time() - started:.1f}s")
    return stats


def _to_yuv420p8(core, vs, clip):
    clip = core.fmtc.bitdepth(clip, bits=16)
    clip = core.fmtc.resample(
        clip,
        csp=vs.YUV420P16,
        kernel="spline16",
        cplaces="mpeg2",
        cplaced="mpeg2",
    )
    return core.fmtc.bitdepth(clip, bits=8)


def _detect_runs(clip, segments, threads, threshold, cap):
    total_input = 0
    total_output = 0
    run_hist = [0] * (cap + 1)
    for segment in segments:
        indexes = segment["branch_indices"]
        frames = clip[indexes[0] : indexes[-1] + 1].frames(prefetch=threads)
        previous = None
        differences = [float("inf")] * len(indexes)
        for position, frame in enumerate(frames):
            current = np.asarray(frame[0]).astype(np.int32)
            if previous is not None:
                differences[position] = box_max_16(np.abs(current - previous))
            previous = current

        kept = []
        position = 0
        while position < len(indexes):
            run_length = 1
            while run_length < cap and position + run_length < len(indexes):
                if differences[position + run_length] >= threshold:
                    break
                run_length += 1
            kept.append((position, run_length))
            run_hist[run_length] += 1
            position += run_length
        segment["kept_positions"] = kept
        total_input += len(indexes)
        total_output += len(kept)

    saved = total_input - total_output
    stats = {
        "input": total_input,
        "output": total_output,
        "saved": saved,
        "saved_pct": saved / max(total_input, 1) * 100.0,
        "run_hist": run_hist,
    }
    validate_segments(segments)
    validate_dedup_stats(stats)
    print(f"  Dedup: {total_input} -> {total_output} frames (saved {saved})")
    return stats
