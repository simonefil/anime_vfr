# -*- coding: utf-8 -*-
"""Framemap parsing and strategy-aware segment construction."""

from collections import Counter

from contracts import Branch, Segment, Strategy, validate_segments
from utils import read_timecodes_v2


def parse_framemap(framemap_path):
    """Read pass2a CSV rows as decimated/source/duration/combed tuples."""
    entries = []
    with open(framemap_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",")
                combed = int(parts[3]) if len(parts) > 3 else 0
                entries.append((int(parts[0]), int(parts[1]), int(parts[2]), combed))
    return entries


def strategies_to_segments(strategies: list[Strategy], entries, redundancy_mapping=None, origins=None, confidence=None, field_units=None, field_metadata=None, matchability=None, redundancy=None, redundancy_origins=None, locked_matchable=None, locked_bob=None) -> list[Segment]:
    """Build operational segments from source frames and validated mappings."""
    frame_count = len(strategies)
    if redundancy_mapping is None:
        redundancy_mapping = [None] * frame_count
    if len(redundancy_mapping) != frame_count:
        raise RuntimeError("Redundancy mapping cardinality does not match strategies")
    if field_units is not None and len(field_units) != frame_count:
        raise RuntimeError("Field-unit cardinality does not match strategies")
    if field_metadata is not None and len(field_metadata) != frame_count:
        raise RuntimeError("Field-metadata cardinality does not match strategies")
    parallel_values = {
        "matchability": matchability,
        "redundancy": redundancy,
        "redundancy origins": redundancy_origins,
        "locked matchable": locked_matchable,
        "locked bob": locked_bob,
    }
    for name, values in parallel_values.items():
        if values is not None and len(values) != frame_count:
            raise RuntimeError(f"{name.capitalize()} cardinality does not match strategies")

    source_to_decimated = {}
    for dec_idx, src_idx, _dur_den, _combed in entries:
        if src_idx in source_to_decimated:
            raise RuntimeError(f"Duplicate source index in framemap: {src_idx}")
        source_to_decimated[src_idx] = dec_idx

    segments: list[Segment] = []
    index = 0
    while index < frame_count:
        strategy = strategies[index]
        if strategy not in ("match_keep_pts", "match_decimate", "bob_expand"):
            raise RuntimeError(f"Unsupported strategy at frame {index}: {strategy}")
        start = index
        while index < frame_count and strategies[index] == strategy:
            index += 1
        end = index - 1
        src_indices = list(range(start, index))
        branch: Branch = {"match_keep_pts": "matched", "match_decimate": "decimated", "bob_expand": "bobbed"}[strategy]
        item: Segment = {
            "type": strategy,
            "strategy": strategy,
            "branch": branch,
            "src_start": start,
            "src_end": end,
            "src_indices": src_indices,
            "source_frame_count": len(src_indices),
            "branch_indices": [],
            "output_source_indices": [],
            "base_output_frame_count": 0,
        }

        if strategy == "match_keep_pts":
            item["branch"] = "matched"
            item["branch_indices"] = list(src_indices)
            item["output_source_indices"] = list(src_indices)
        elif strategy == "match_decimate":
            retained_sources = []
            decimated_indices = []
            for src_idx in src_indices:
                mapping = redundancy_mapping[src_idx]
                if mapping == "dropped":
                    continue
                if mapping != "retained":
                    raise RuntimeError(
                        f"Missing match_decimate mapping at source frame {src_idx}: {mapping}"
                    )
                if src_idx not in source_to_decimated:
                    raise RuntimeError(
                        f"Retained source frame {src_idx} is absent from the TDecimate framemap"
                    )
                retained_sources.append(src_idx)
                decimated_indices.append(source_to_decimated[src_idx])
            if not retained_sources:
                raise RuntimeError(f"match_decimate segment {start}-{end} has no retained frames")
            if any(
                decimated_indices[pos] >= decimated_indices[pos + 1]
                for pos in range(len(decimated_indices) - 1)
            ):
                raise RuntimeError(f"Decimated mapping is not increasing in segment {start}-{end}")
            item["branch"] = "decimated"
            item["branch_indices"] = decimated_indices
            item["output_source_indices"] = retained_sources
            item["retained_source_indices"] = retained_sources
            item["dropped_source_indices"] = [
                src_idx for src_idx in src_indices if redundancy_mapping[src_idx] == "dropped"
            ]
        else:
            if field_units is None or field_metadata is None:
                raise RuntimeError(
                    f"bob_expand segment {start}-{end} has no validated field metadata"
                )
            item["branch"] = "bobbed"
            item["branch_indices"] = []
            item["output_source_indices"] = []
            item["bob_field_units"] = []
            item["bob_frame_specs"] = []
            for src_idx in src_indices:
                metadata = field_metadata[src_idx]
                repeat_pict = metadata["repeat_pict"]
                units = field_units[src_idx]
                if units != 2 + repeat_pict:
                    raise RuntimeError(
                        f"Cannot bob source frame {src_idx}: duration is not consistent with field metadata "
                        f"field_units={units}, repeat_pict={repeat_pict}"
                    )
                top_field_first = bool(metadata["top_field_first"])
                bob_indices = [src_idx * 2, src_idx * 2 + 1]
                if repeat_pict:
                    bob_indices.append(src_idx * 2)
                item["branch_indices"].extend(bob_indices)
                item["output_source_indices"].extend([src_idx] * units)
                item["bob_field_units"].append(units)
                item["bob_frame_specs"].extend(
                    (top_field_first, bob_index) for bob_index in bob_indices
                )

        item["base_output_frame_count"] = len(item["branch_indices"])
        if origins is not None:
            item["origin"] = Counter(origins[start:index]).most_common(1)[0][0]
            item["origins"] = sorted(set(origins[start:index]))
            item["frame_origins"] = list(origins[start:index])
        if confidence is not None:
            item["confidence"] = min(confidence[start:index])
            item["frame_confidence"] = list(confidence[start:index])
        if matchability is not None:
            item["matchability"] = Counter(matchability[start:index]).most_common(1)[0][0]
            item["matchability_states"] = sorted(set(matchability[start:index]))
            item["frame_matchability"] = list(matchability[start:index])
        if redundancy is not None:
            item["redundancy"] = Counter(redundancy[start:index]).most_common(1)[0][0]
            item["redundancy_states"] = sorted(set(redundancy[start:index]))
            item["frame_redundancy"] = list(redundancy[start:index])
        if redundancy_origins is not None:
            item["redundancy_origin"] = Counter(
                redundancy_origins[start:index]
            ).most_common(1)[0][0]
            item["redundancy_origins"] = sorted(
                {value for value in redundancy_origins[start:index] if value is not None}
            )
            item["frame_redundancy_origins"] = list(redundancy_origins[start:index])
        if locked_matchable is not None:
            item["locked_matchable"] = all(locked_matchable[start:index])
            item["frame_locked_matchable"] = list(locked_matchable[start:index])
        if locked_bob is not None:
            item["locked_bob"] = all(locked_bob[start:index])
            item["frame_locked_bob"] = list(locked_bob[start:index])
        segments.append(item)

    validate_segments(segments, frame_count)
    return segments


def make_linear_strategy_segments(frame_count: int, strategy: Strategy, branch: Branch) -> list[Segment]:
    """Create one linear segment for an explicit global mode."""
    src_indices = list(range(frame_count))
    segment: Segment = {
        "type": strategy,
        "strategy": strategy,
        "branch": branch,
        "src_start": 0,
        "src_end": frame_count - 1,
        "src_indices": src_indices,
        "source_frame_count": frame_count,
        "branch_indices": [] if strategy == "bob_expand" else list(src_indices),
        "output_source_indices": [] if strategy == "bob_expand" else list(src_indices),
        "base_output_frame_count": frame_count * 2 if strategy == "bob_expand" else frame_count,
    }
    segments = [segment]
    validate_segments(segments, frame_count)
    return segments


def make_bob_entries_from_source_timecodes(src_tc_path, frame_count=None):
    """Create an all-bob framemap aligned with source timestamps."""
    src_tc = read_timecodes_v2(src_tc_path)
    count = min(len(src_tc), frame_count) if frame_count is not None else len(src_tc)
    return [(i, i, 30000, 1) for i in range(count)]


def make_progressive_entries_from_source_timecodes(src_tc_path, frame_count=None):
    """Create a linear progressive framemap aligned with source timestamps."""
    src_tc = read_timecodes_v2(src_tc_path)
    count = min(len(src_tc), frame_count) if frame_count is not None else len(src_tc)
    return [(i, i, 24000, 0) for i in range(count)]
