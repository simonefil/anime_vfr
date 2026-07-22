# -*- coding: utf-8 -*-
"""Typed dictionary contracts and runtime validation for pipeline data."""

from pathlib import Path
from typing import Literal, Optional, TypedDict


Strategy = Literal["match_keep_pts", "match_decimate", "bob_expand"]
Branch = Literal["matched", "decimated", "bobbed", "progressive"]
Matchability = Literal["unknown", "matchable", "not_matchable"]
Redundancy = Literal["unknown", "redundant_with_valid_map"]
RedundancyMapping = Literal["retained", "dropped"]

STRATEGIES = frozenset(("match_keep_pts", "match_decimate", "bob_expand"))
STRATEGY_BRANCHES = {
    "match_keep_pts": frozenset(("matched", "progressive")),
    "match_decimate": frozenset(("decimated",)),
    "bob_expand": frozenset(("bobbed",)),
}


class DurationCluster(TypedDict):
    center_ms: float
    count: int


class SourceTimeline(TypedDict):
    pts_ms: list[float]
    duration_ms: list[Optional[float]]
    field_duration_ms: Optional[float]
    field_units: list[Optional[int]]
    quantization_valid: list[bool]
    duration_clusters: list[DurationCluster]
    discontinuities: list[int]
    warnings: list[str]


class FieldMetadata(TypedDict):
    repeat_pict: int
    top_field_first: bool


class SegmentRequired(TypedDict):
    type: Strategy
    strategy: Strategy
    branch: Branch
    src_start: int
    src_end: int
    src_indices: list[int]
    source_frame_count: int
    branch_indices: list[int]
    output_source_indices: list[int]
    base_output_frame_count: int


class Segment(SegmentRequired, total=False):
    retained_source_indices: list[int]
    dropped_source_indices: list[int]
    decimated_durations: list[tuple[int, int]]
    bob_field_units: list[int]
    bob_frame_specs: list[tuple[bool, int]]
    kept_positions: list[tuple[int, int]]
    origin: str
    origins: list[str]
    frame_origins: list[str]
    matchability: str
    matchability_states: list[str]
    frame_matchability: list[str]
    redundancy: Optional[str]
    redundancy_states: list[Optional[str]]
    frame_redundancy: list[Optional[str]]
    redundancy_origin: Optional[str]
    redundancy_origins: list[str]
    frame_redundancy_origins: list[Optional[str]]
    locked_matchable: bool
    frame_locked_matchable: list[bool]
    locked_bob: bool
    frame_locked_bob: list[bool]


class ShadowResult(TypedDict):
    matchability: list[Matchability]
    redundancy: list[Optional[Redundancy]]
    redundancy_origins: list[Optional[str]]
    redundancy_mapping: list[Optional[RedundancyMapping]]
    strategies: list[Strategy]
    origins: list[str]
    locked_matchable: list[bool]
    locked_bob: list[bool]


class AnalysisResultRequired(ShadowResult):
    timeline: SourceTimeline
    field_metadata: list[FieldMetadata]


class AnalysisResult(AnalysisResultRequired, total=False):
    diagnostic_path: Path
    run_diagnostic_path: Path


class DedupStats(TypedDict):
    input: int
    output: int
    saved: int
    saved_pct: float
    run_hist: list[int]


class EpisodeStats(TypedDict):
    name: str
    mode: Literal["bob", "progressive_dedup", "hybrid"]
    source_frames: int
    film_frames_24: int
    video_frames_60: int
    match_keep_pts_frames: int
    match_decimate_frames: int
    bob_expand_frames: int
    dedup_saved: int
    full_output_frames: int
    total_out_frames: int
    film_pct: float
    video60_pct: float
    resolution: str


def _require_keys(value, required_keys, contract_name):
    missing = [key for key in required_keys if key not in value]
    if missing:
        raise ValueError(f"{contract_name} is missing required keys: {', '.join(sorted(missing))}")


def validate_segment(segment, segment_index=None):
    """Validate one mutable segment at a pipeline boundary."""
    label = f"Segment {segment_index}" if segment_index is not None else "Segment"
    required_keys = ("type", "strategy", "branch", "src_start", "src_end", "src_indices", "source_frame_count", "branch_indices", "output_source_indices", "base_output_frame_count")
    _require_keys(segment, required_keys, label)
    strategy = segment["strategy"]
    if strategy not in STRATEGIES:
        raise ValueError(f"{label} has unsupported strategy: {strategy}")
    if segment["type"] != strategy:
        raise ValueError(f"{label} type does not match its strategy")
    if segment["branch"] not in STRATEGY_BRANCHES[strategy]:
        raise ValueError(f"{label} branch {segment['branch']} is incompatible with {strategy}")
    source_indices = segment["src_indices"]
    if not source_indices:
        raise ValueError(f"{label} has no source indices")
    if source_indices != list(range(segment["src_start"], segment["src_end"] + 1)):
        raise ValueError(f"{label} source indices are not contiguous or do not match its bounds")
    if segment["source_frame_count"] != len(source_indices):
        raise ValueError(f"{label} source-frame count does not match its source indices")
    if segment["base_output_frame_count"] < 0:
        raise ValueError(f"{label} has a negative base output-frame count")
    if strategy != "bob_expand" and segment["base_output_frame_count"] != len(segment["branch_indices"]):
        raise ValueError(f"{label} base output-frame count does not match its branch indices")
    if strategy == "match_keep_pts" and segment["output_source_indices"] != source_indices:
        raise ValueError(f"{label} match_keep_pts mapping does not preserve every source frame")
    if strategy == "match_keep_pts" and segment["branch_indices"] != source_indices:
        raise ValueError(f"{label} match_keep_pts branch does not preserve every source frame")
    if strategy == "match_decimate" and len(segment["output_source_indices"]) != len(segment["branch_indices"]):
        raise ValueError(f"{label} decimated source and branch mappings have different lengths")
    if strategy == "match_decimate" and len(segment.get("decimated_durations", [])) != len(segment["branch_indices"]):
        raise ValueError(f"{label} decimated durations do not match its branch indices")
    if strategy == "match_decimate" and any(
        numerator <= 0 or denominator <= 0
        for numerator, denominator in segment["decimated_durations"]
    ):
        raise ValueError(f"{label} contains an invalid decimated duration")
    if strategy == "match_decimate" and (segment["output_source_indices"] != sorted(set(segment["output_source_indices"])) or segment["branch_indices"] != sorted(set(segment["branch_indices"]))):
        raise ValueError(f"{label} decimated mappings are not strictly increasing")
    if "retained_source_indices" in segment and segment["retained_source_indices"] != segment["output_source_indices"]:
        raise ValueError(f"{label} retained source indices do not match its output mapping")
    if "dropped_source_indices" in segment and sorted(segment["output_source_indices"] + segment["dropped_source_indices"]) != source_indices:
        raise ValueError(f"{label} retained and dropped source indices do not partition the segment")
    if strategy == "bob_expand" and "bob_field_units" in segment:
        if len(segment["bob_field_units"]) != len(source_indices):
            raise ValueError(f"{label} bob field-unit count does not match its source frames")
        if sum(segment["bob_field_units"]) != segment["base_output_frame_count"]:
            raise ValueError(f"{label} bob field units do not match its base output-frame count")
    if strategy == "bob_expand" and "bob_field_units" not in segment and segment["base_output_frame_count"] != len(source_indices) * 2:
        raise ValueError(f"{label} default bob output count is not twice its source-frame count")
    if strategy == "bob_expand" and segment["branch_indices"] and len(segment["branch_indices"]) != segment["base_output_frame_count"]:
        raise ValueError(f"{label} bob branch indices do not match its base output-frame count")
    if "bob_frame_specs" in segment and len(segment["bob_frame_specs"]) != segment["base_output_frame_count"]:
        raise ValueError(f"{label} bob frame specifications do not match its base output-frame count")
    if "kept_positions" in segment:
        expected_position = 0
        for position, run_length in segment["kept_positions"]:
            if position != expected_position or run_length < 1:
                raise ValueError(f"{label} has invalid dedup runs")
            expected_position += run_length
        if expected_position != len(segment["branch_indices"]):
            raise ValueError(f"{label} dedup runs do not cover its branch")
    parallel_segment_keys = ("frame_origins", "frame_matchability", "frame_redundancy", "frame_redundancy_origins", "frame_locked_matchable", "frame_locked_bob")
    for key in parallel_segment_keys:
        if key in segment and len(segment[key]) != len(source_indices):
            raise ValueError(f"{label} {key} cardinality does not match its source frames")


def validate_segments(segments, expected_source_frame_count=None):
    """Validate segment contracts and their complete ordered source coverage."""
    if expected_source_frame_count is not None and expected_source_frame_count > 0 and not segments:
        raise ValueError("No segments were provided for a non-empty source")
    for index, segment in enumerate(segments):
        validate_segment(segment, index)
    covered_source_indices = [source_index for segment in segments for source_index in segment["src_indices"]]
    expected_count = expected_source_frame_count if expected_source_frame_count is not None else len(covered_source_indices)
    if covered_source_indices != list(range(expected_count)):
        raise ValueError("Segments do not provide complete, ordered source coverage")


def validate_analysis_result(analysis):
    """Validate parallel arrays returned by the operational classifier."""
    required_keys = ("timeline", "field_metadata", "matchability", "redundancy", "redundancy_origins", "redundancy_mapping", "strategies", "origins", "locked_matchable", "locked_bob")
    _require_keys(analysis, required_keys, "Analysis result")
    validate_shadow_result(analysis)
    frame_count = len(analysis["strategies"])
    parallel_keys = ("field_metadata",)
    for key in parallel_keys:
        if len(analysis[key]) != frame_count:
            raise ValueError(f"Analysis result {key} cardinality does not match strategies")
    timeline = analysis["timeline"]
    for key in ("pts_ms", "duration_ms", "field_units", "quantization_valid"):
        if len(timeline[key]) != frame_count:
            raise ValueError(f"Analysis timeline {key} cardinality does not match strategies")


def validate_shadow_result(shadow):
    """Validate the classifier's parallel operational decision arrays."""
    required_keys = ("matchability", "redundancy", "redundancy_origins", "redundancy_mapping", "strategies", "origins", "locked_matchable", "locked_bob")
    _require_keys(shadow, required_keys, "Shadow result")
    frame_count = len(shadow["strategies"])
    for key in required_keys:
        if len(shadow[key]) != frame_count:
            raise ValueError(f"Shadow result {key} cardinality does not match strategies")
    if any(strategy not in STRATEGIES for strategy in shadow["strategies"]):
        raise ValueError("Shadow result contains an unsupported strategy")
    if any(value not in ("unknown", "matchable", "not_matchable") for value in shadow["matchability"]):
        raise ValueError("Shadow result contains an unsupported matchability value")
    if any(value not in (None, "unknown", "redundant_with_valid_map") for value in shadow["redundancy"]):
        raise ValueError("Shadow result contains an unsupported redundancy value")
    if any(value not in (None, "retained", "dropped") for value in shadow["redundancy_mapping"]):
        raise ValueError("Shadow result contains an unsupported redundancy mapping")


def validate_dedup_stats(stats):
    """Validate the arithmetic invariants of a deduplication result."""
    _require_keys(stats, ("input", "output", "saved", "saved_pct", "run_hist"), "Dedup stats")
    if stats["input"] < 0 or stats["output"] < 0 or stats["output"] > stats["input"]:
        raise ValueError("Dedup stats contain invalid input/output counts")
    if stats["saved"] != stats["input"] - stats["output"]:
        raise ValueError("Dedup stats saved count does not match input minus output")
    expected_percentage = 100.0 * stats["saved"] / max(stats["input"], 1)
    if abs(stats["saved_pct"] - expected_percentage) > 1e-9:
        raise ValueError("Dedup stats saved percentage does not match its frame counts")


def validate_episode_stats(stats):
    """Validate the final per-episode summary contract."""
    required_keys = ("name", "mode", "source_frames", "film_frames_24", "video_frames_60", "match_keep_pts_frames", "match_decimate_frames", "bob_expand_frames", "dedup_saved", "full_output_frames", "total_out_frames", "film_pct", "video60_pct", "resolution")
    _require_keys(stats, required_keys, "Episode stats")
    count_keys = ("source_frames", "film_frames_24", "video_frames_60", "match_keep_pts_frames", "match_decimate_frames", "bob_expand_frames", "dedup_saved", "full_output_frames", "total_out_frames")
    if any(stats[key] < 0 for key in count_keys):
        raise ValueError("Episode stats contain a negative frame count")
    strategy_output = stats["match_keep_pts_frames"] + stats["match_decimate_frames"] + stats["bob_expand_frames"]
    if strategy_output != stats["full_output_frames"]:
        raise ValueError("Episode strategy counts do not match the full output count")
    if stats["film_frames_24"] != stats["match_keep_pts_frames"] + stats["match_decimate_frames"]:
        raise ValueError("Episode film count does not match its matched strategies")
    if stats["video_frames_60"] != stats["bob_expand_frames"]:
        raise ValueError("Episode video count does not match bob_expand")
    if stats["total_out_frames"] > stats["full_output_frames"]:
        raise ValueError("Episode selected output exceeds its full output")
    expected_film_percentage = stats["film_frames_24"] / max(stats["full_output_frames"], 1) * 100.0
    expected_video_percentage = stats["video_frames_60"] / max(stats["full_output_frames"], 1) * 100.0
    if abs(stats["film_pct"] - expected_film_percentage) > 1e-9 or abs(stats["video60_pct"] - expected_video_percentage) > 1e-9:
        raise ValueError("Episode strategy percentages do not match their frame counts")
