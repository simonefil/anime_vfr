# -*- coding: utf-8 -*-
"""Minimal segment construction for progressive-dedup mode."""

from contracts import validate_segments


def make_linear_strategy_segments(frame_count, strategy, branch):
    """Create one complete linear segment."""
    if strategy != "match_keep_pts" or branch != "progressive":
        raise ValueError("Only progressive match_keep_pts segments are supported")
    indices = list(range(frame_count))
    segments = [
        {
            "type": strategy,
            "strategy": strategy,
            "branch": branch,
            "src_start": 0,
            "src_end": frame_count - 1,
            "src_indices": indices,
            "source_frame_count": frame_count,
            "branch_indices": indices.copy(),
            "output_source_indices": indices.copy(),
            "base_output_frame_count": frame_count,
        }
    ]
    validate_segments(segments, frame_count)
    return segments
