# -*- coding: utf-8 -*-
"""Runtime contracts for progressive-dedup mode."""


def validate_segments(segments, expected_source_frame_count=None):
    """Validate complete progressive source coverage and optional dedup runs."""
    covered = []
    for index, segment in enumerate(segments):
        required = (
            "type",
            "strategy",
            "branch",
            "src_start",
            "src_end",
            "src_indices",
            "source_frame_count",
            "branch_indices",
            "output_source_indices",
            "base_output_frame_count",
        )
        missing = [key for key in required if key not in segment]
        if missing:
            raise ValueError(f"Segment {index} is missing: {', '.join(missing)}")
        if segment["strategy"] != "match_keep_pts" or segment["branch"] != "progressive":
            raise ValueError("Only progressive match_keep_pts segments are supported")
        source_indices = segment["src_indices"]
        expected = list(range(segment["src_start"], segment["src_end"] + 1))
        if source_indices != expected:
            raise ValueError("Segment source indices are not contiguous")
        if segment["branch_indices"] != source_indices:
            raise ValueError("Progressive branch indices must equal source indices")
        if segment["output_source_indices"] != source_indices:
            raise ValueError("Progressive output mapping must preserve source indices")
        if segment["source_frame_count"] != len(source_indices):
            raise ValueError("Segment source-frame count is invalid")
        if "kept_positions" in segment:
            cursor = 0
            for position, run_length in segment["kept_positions"]:
                if position != cursor or run_length < 1:
                    raise ValueError("Invalid progressive dedup run mapping")
                cursor += run_length
            if cursor != len(source_indices):
                raise ValueError("Progressive dedup runs do not cover the segment")
        covered.extend(source_indices)
    expected_count = expected_source_frame_count
    if expected_count is None:
        expected_count = len(covered)
    if covered != list(range(expected_count)):
        raise ValueError("Segments do not cover the complete source")


def validate_dedup_stats(stats):
    """Validate progressive dedup arithmetic."""
    required = ("input", "output", "saved", "saved_pct", "run_hist")
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"Dedup stats are missing: {', '.join(missing)}")
    if stats["input"] < 0 or not 0 <= stats["output"] <= stats["input"]:
        raise ValueError("Invalid dedup input/output counts")
    if stats["saved"] != stats["input"] - stats["output"]:
        raise ValueError("Invalid dedup saved count")
    expected = stats["saved"] / max(stats["input"], 1) * 100.0
    if abs(stats["saved_pct"] - expected) > 1e-9:
        raise ValueError("Invalid dedup percentage")
