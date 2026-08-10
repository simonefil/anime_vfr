# -*- coding: utf-8 -*-
"""Timecode generation for progressive-dedup mode."""

from contracts import validate_segments
from utils import read_timecodes_v2


def segment_output_frame_count(segment):
    if "kept_positions" in segment:
        return len(segment["kept_positions"])
    return len(segment["branch_indices"])


def generate_final_timecodes_v2(segments, source_path, output_path):
    """Keep source PTS for each retained progressive frame."""
    frame_count = sum(segment["source_frame_count"] for segment in segments)
    validate_segments(segments, frame_count)
    source = read_timecodes_v2(source_path)
    if len(source) not in (frame_count, frame_count + 1):
        raise RuntimeError(
            f"Source timecodes contain {len(source)} entries for {frame_count} frames"
        )
    output = []
    for segment in segments:
        positions = (
            [position for position, _ in segment["kept_positions"]]
            if "kept_positions" in segment
            else range(len(segment["output_source_indices"]))
        )
        output.extend(source[segment["output_source_indices"][position]] for position in positions)
    if any(right <= left for left, right in zip(output, output[1:])):
        raise RuntimeError("Progressive output timecodes are not strictly increasing")
    expected = sum(segment_output_frame_count(segment) for segment in segments)
    if len(output) != expected:
        raise RuntimeError("Progressive output/timecode cardinality mismatch")
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("# timecode format v2\n")
        for timestamp in output:
            stream.write(f"{timestamp:.6f}\n")
    return len(output)
