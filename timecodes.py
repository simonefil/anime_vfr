# -*- coding: utf-8 -*-
"""Final VFR timecode generation."""

import math
from fractions import Fraction
from statistics import median

from config import (
    PTS_DECIMATE_CYCLE_TOLERANCE_MS,
    PTS_DURATION_CLUSTER_REL_TOL,
    PTS_FIELD_QUANTIZATION_REL_TOL,
    PTS_MAX_FIELD_UNITS,
    PTS_MIN_CLUSTER_SAMPLES,
)
from contracts import Segment, SourceTimeline, validate_segments
from utils import read_timecodes_v2


def _cluster_durations(durations, relative_tolerance):
    """Cluster nearby durations without relying on nominal frame rates."""
    clusters = []
    for duration in sorted(durations):
        cluster = clusters[-1] if clusters else None
        error = (
            abs(duration - cluster["center_ms"]) / max(cluster["center_ms"], 1e-9)
            if cluster is not None else None
        )
        if cluster is None or error > relative_tolerance:
            clusters.append({
                "center_ms": duration,
                "sum_ms": duration,
                "count": 1,
            })
        else:
            cluster["sum_ms"] += duration
            cluster["count"] += 1
            cluster["center_ms"] = cluster["sum_ms"] / cluster["count"]
    return clusters


def validate_source_timeline(src_tc_path, frame_count=None) -> SourceTimeline:
    """Validate PTS values and quantize source durations into field units.

    The result is diagnostic and does not assign 24/30/60 classes. A missing,
    non-monotonic, or incorrectly sized timeline is rejected because it cannot
    support safe PTS-aware strategies. Positive durations that cannot be
    quantized into fields remain valid for matched output and are exposed as
    discontinuities; bob validation is deferred until the strategy is final.
    """
    extracted_pts = read_timecodes_v2(src_tc_path)
    if not extracted_pts:
        raise RuntimeError(f"Empty source timeline: {src_tc_path}")
    has_terminal_timestamp = frame_count is not None and len(extracted_pts) == frame_count + 1
    if frame_count is not None and len(extracted_pts) not in (frame_count, frame_count + 1):
        raise RuntimeError(
            f"Invalid source timeline cardinality: {len(extracted_pts)} timestamps "
            f"for {frame_count} frames"
        )
    for index, pts in enumerate(extracted_pts):
        if not math.isfinite(pts):
            raise RuntimeError(f"Non-finite source PTS at frame {index}: {pts}")
        if index > 0 and pts <= extracted_pts[index - 1]:
            raise RuntimeError(
                f"Non-increasing source PTS at frame {index}: "
                f"{extracted_pts[index - 1]:.6f} -> {pts:.6f}"
            )

    pts_ms = extracted_pts[:frame_count] if frame_count is not None else extracted_pts
    observed_durations = [
        extracted_pts[i + 1] - extracted_pts[i]
        for i in range(len(extracted_pts) - 1)
    ]
    if not observed_durations:
        return {
            "pts_ms": pts_ms,
            "duration_ms": [None],
            "field_duration_ms": None,
            "field_units": [None],
            "quantization_valid": [False],
            "duration_clusters": [],
            "discontinuities": [0],
            "warnings": ["Only one timestamp: duration and field units cannot be estimated"],
        }

    if has_terminal_timestamp:
        durations = observed_durations
    else:
        recent = observed_durations[-min(120, len(observed_durations)):]
        estimated_last = median(recent)
        durations = observed_durations + [estimated_last]
    clusters = _cluster_durations(observed_durations, PTS_DURATION_CLUSTER_REL_TOL)
    min_samples = min(
        len(observed_durations),
        max(PTS_MIN_CLUSTER_SAMPLES, int(math.ceil(len(observed_durations) * 0.001))),
    )
    significant = [cluster for cluster in clusters if cluster["count"] >= min_samples]
    warnings = []
    if not significant:
        field_duration_ms = None
        warnings.append("No sufficiently stable duration cluster to estimate field units")
    else:
        shortest_cluster = min(significant, key=lambda cluster: cluster["center_ms"])
        field_duration_ms = shortest_cluster["center_ms"] / 2.0

    field_units = []
    quantization_valid = []
    discontinuities = []
    for index, duration in enumerate(durations):
        if field_duration_ms is None or field_duration_ms <= 0:
            field_units.append(None)
            quantization_valid.append(False)
            discontinuities.append(index)
            continue
        units = int(round(duration / field_duration_ms))
        if 2 <= units <= PTS_MAX_FIELD_UNITS:
            expected_duration = units * field_duration_ms
            error = abs(duration - expected_duration) / expected_duration
            valid = error <= PTS_FIELD_QUANTIZATION_REL_TOL
        else:
            valid = False
        field_units.append(units if valid else None)
        quantization_valid.append(valid)
        if not valid:
            discontinuities.append(index)

    if discontinuities:
        warnings.append(
            f"{len(discontinuities)} durations cannot be quantized into 2-{PTS_MAX_FIELD_UNITS} fields"
        )

    return {
        "pts_ms": pts_ms,
        "duration_ms": durations,
        "field_duration_ms": field_duration_ms,
        "field_units": field_units,
        "quantization_valid": quantization_valid,
        "duration_clusters": [
            {"center_ms": cluster["center_ms"], "count": cluster["count"]}
            for cluster in clusters
        ],
        "discontinuities": discontinuities,
        "warnings": warnings,
    }


def source_end_ms(src_tc, frame_count=None):
    """Estimate the source end timestamp from v2 timestamps."""
    if frame_count is not None and len(src_tc) == frame_count + 1:
        return src_tc[-1]
    if len(src_tc) >= 2:
        return src_tc[-1] + (src_tc[-1] - src_tc[-2])
    if len(src_tc) == 1:
        return src_tc[0] + 33.367
    return 0.0


def segment_output_frame_count(segment: Segment) -> int:
    """Return the effective output size of a strategy-aware segment."""
    if segment["strategy"] == "bob_expand":
        if segment.get("branch_indices"):
            return len(segment["branch_indices"])
        return len(segment["src_indices"]) * 2
    if "kept_positions" in segment:
        return len(segment["kept_positions"])
    return len(segment["branch_indices"])


def _fractional_ms(value):
    """Preserve the decimal precision present in an extracted timestamp."""
    return Fraction(str(value))


def generate_final_timecodes_v2(segments: list[Segment], src_tc_path, output_path, strict_bob_field_units=True):
    """Write source-anchored VFR timestamps using each branch's own clock."""
    frame_count = sum(segment["source_frame_count"] for segment in segments)
    validate_segments(segments, frame_count)
    timeline = validate_source_timeline(src_tc_path, frame_count)
    pts_ms = timeline["pts_ms"]
    durations = timeline["duration_ms"]

    covered = [src_idx for segment in segments for src_idx in segment["src_indices"]]
    if covered != list(range(frame_count)):
        raise RuntimeError("Timecode generation received incomplete or unordered source coverage")

    timecodes = []
    for seg in segments:
        strategy = seg["strategy"]
        if strategy == "bob_expand":
            bob_field_units = seg.get("bob_field_units")
            if strict_bob_field_units and (
                bob_field_units is None or len(bob_field_units) != len(seg["src_indices"])
            ):
                raise RuntimeError(
                    f"bob_expand {seg['src_start']}-{seg['src_end']} has no validated field units"
                )
            for position, src_idx in enumerate(seg["src_indices"]):
                units = bob_field_units[position] if strict_bob_field_units else 2
                if strict_bob_field_units and timeline["field_units"][src_idx] != units:
                    raise RuntimeError(
                        f"Inconsistent bob_expand metadata at source frame {src_idx}: "
                        f"timeline={timeline['field_units'][src_idx]}, metadata={units}"
                    )
                cur = _fractional_ms(pts_ms[src_idx])
                field_duration = _fractional_ms(durations[src_idx]) / units
                timecodes.extend(cur + field_duration * offset for offset in range(units))
        elif strategy == "match_decimate":
            branch_timecodes = []
            current = _fractional_ms(pts_ms[seg["src_start"]])
            for duration_num, duration_den in seg["decimated_durations"]:
                branch_timecodes.append(current)
                current += Fraction(duration_num * 1000, duration_den)
            source_end = (
                _fractional_ms(pts_ms[seg["src_end"]]) +
                _fractional_ms(durations[seg["src_end"]])
            )
            if abs(current - source_end) > _fractional_ms(PTS_DECIMATE_CYCLE_TOLERANCE_MS):
                raise RuntimeError(
                    f"Decimated timeline {seg['src_start']}-{seg['src_end']} misses its "
                    f"source boundary by {float(current - source_end):.6f} ms"
                )
            if "kept_positions" in seg:
                positions = [position for position, _run_len in seg["kept_positions"]]
            else:
                positions = range(len(seg["output_source_indices"]))
            for position in positions:
                timecodes.append(branch_timecodes[position])
        elif strategy == "match_keep_pts":
            if "kept_positions" in seg:
                positions = [position for position, _run_len in seg["kept_positions"]]
            else:
                positions = range(len(seg["output_source_indices"]))
            for position in positions:
                src_idx = seg["output_source_indices"][position]
                timecodes.append(_fractional_ms(pts_ms[src_idx]))
        else:
            raise ValueError(f"Unsupported timecode strategy: {strategy}")

    for i in range(1, len(timecodes)):
        if timecodes[i] <= timecodes[i - 1]:
            raise RuntimeError(
                f"Non-increasing final timecodes at frame {i}: "
                f"{float(timecodes[i - 1]):.6f} -> {float(timecodes[i]):.6f}"
            )

    expected_count = sum(segment_output_frame_count(segment) for segment in segments)
    if len(timecodes) != expected_count:
        raise RuntimeError(
            f"Invalid timecode cardinality: {len(timecodes)} for {expected_count} output frames"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# timecode format v2\n")
        for tc in timecodes:
            f.write(f"{float(tc):.6f}\n")
    return len(timecodes)
