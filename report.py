# -*- coding: utf-8 -*-
"""Standard-output reports for pre-encode analysis and post-encode inspection."""

import math
import os
import subprocess
import tempfile

import numpy as np

from config import (
    MKVEXTRACT,
    PTS_DURATION_CLUSTER_REL_TOL,
    REPORT_BUCKETS,
    REPORT_RATE_REL_TOL,
    REPORT_WINDOW_MS,
)
from contracts import DedupStats, Segment, SourceTimeline, validate_dedup_stats, validate_segments
from timecodes import segment_output_frame_count, source_end_ms, validate_source_timeline
from utils import nice_ceil, read_timecodes_v2


STRATEGIES = ("match_keep_pts", "match_decimate", "bob_expand")


def _segment_bounds_ms(segment: Segment, src_tc):
    """Return the half-open source timeline bounds of a strategy segment."""
    start_ms = src_tc[segment["src_start"]]
    next_source_frame = segment["src_end"] + 1
    end_ms = (
        src_tc[next_source_frame]
        if next_source_frame < len(src_tc)
        else source_end_ms(src_tc)
    )
    return start_ms, end_ms


def _print_histogram(title, values, y_max, y_fmt, height=10):
    """Print a fixed-width vertical histogram with numbered buckets."""
    print("")
    print(title)
    if not values:
        print("  no data")
        return
    y_max = max(float(y_max), 1.0)
    col_w = 3
    col_labels = [str(i + 1).rjust(col_w) for i in range(len(values))]
    for row in range(height, 0, -1):
        level = y_max * row / height
        cells = ["  #" if v >= level else "   " for v in values]
        print(f"{y_fmt(level):>7} |" + "".join(cells))
    print(f"{y_fmt(0):>7} +" + "-" * (col_w * len(values)))
    print("        " + "".join(col_labels))


def _analyze_buckets(segments: list[Segment], src_tc, bucket_count=REPORT_BUCKETS):
    """Aggregate pre-dedup rates and optional drops into fixed time buckets."""
    if not segments:
        return []
    total_end = max(_segment_bounds_ms(segment, src_tc)[1] for segment in segments)
    if total_end <= 0:
        total_end = 1.0
    bucket_ms = total_end / bucket_count
    buckets = []
    for i in range(bucket_count):
        buckets.append({
            "start": i * bucket_ms,
            "end": (i + 1) * bucket_ms,
            "fps_weight": 0.0,
            "duration": 0.0,
            "drops": 0,
        })

    for segment in segments:
        start, end = _segment_bounds_ms(segment, src_tc)
        fps = segment["base_output_frame_count"] * 1000.0 / max(end - start, 1e-9)
        a = max(0, min(bucket_count - 1, int(start / bucket_ms)))
        b = max(0, min(bucket_count - 1, int(max(start, end - 0.001) / bucket_ms)))
        for idx in range(a, b + 1):
            bucket = buckets[idx]
            overlap = max(0.0, min(end, bucket["end"]) - max(start, bucket["start"]))
            bucket["fps_weight"] += fps * overlap
            bucket["duration"] += overlap

        # Attribute dropped frames to the bucket containing the retained frame.
        if "kept_positions" in segment:
            for position, run_len in segment["kept_positions"]:
                if run_len <= 1:
                    continue
                src_idx = segment["output_source_indices"][position]
                t = src_tc[src_idx]
                idx = max(0, min(bucket_count - 1, int(t / bucket_ms)))
                buckets[idx]["drops"] += run_len - 1

    for bucket in buckets:
        bucket["fps"] = bucket["fps_weight"] / bucket["duration"] if bucket["duration"] else 0.0
    return buckets


def _fmt_report_time(ms):
    """Format a timestamp with millisecond precision."""
    total_ms = max(0, int(round(ms)))
    hours, remainder = divmod(total_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _cadence_display_ranges(segment: Segment, timeline: SourceTimeline):
    """Split an operational segment only at persistent PTS cadence changes."""
    start = segment["src_start"]
    end = segment["src_end"]
    units = timeline["field_units"]
    durations = timeline["duration_ms"]
    if all(units[index] == 2 for index in range(start, end + 1)):
        return [(start, end)]

    stable_two_field = [False] * (end - start + 1)
    index = start
    while index <= end:
        if units[index] != 2:
            index += 1
            continue
        run_start = index
        while index <= end and units[index] == 2:
            index += 1
        run_duration = sum(durations[run_start:index])
        if run_duration >= REPORT_WINDOW_MS:
            for frame in range(run_start, index):
                stable_two_field[frame - start] = True

    ranges = []
    index = start
    while index <= end:
        range_start = index
        state = stable_two_field[index - start]
        while index <= end and stable_two_field[index - start] == state:
            index += 1
        ranges.append((range_start, index - 1))
    return ranges


def _range_output_counts(segment: Segment, start, end):
    """Count structural and final output frames for a source sub-range."""
    if segment["strategy"] == "bob_expand":
        field_units = segment.get("bob_field_units")
        if field_units is None:
            base_count = (end - start + 1) * 2
        else:
            offset_start = start - segment["src_start"]
            offset_end = end - segment["src_start"] + 1
            base_count = sum(field_units[offset_start:offset_end])
        return base_count, base_count

    output_sources = segment["output_source_indices"]
    base_positions = [
        position for position, source_index in enumerate(output_sources)
        if start <= source_index <= end
    ]
    base_count = len(base_positions)
    if "kept_positions" not in segment:
        return base_count, base_count
    kept = {position for position, _run_length in segment["kept_positions"]}
    return base_count, sum(position in kept for position in base_positions)


def _cadence_label(segment: Segment, timeline: SourceTimeline, start, end, output_rate):
    """Describe observed cadence without changing the selected strategy."""
    known_rates = (
        (24000.0 / 1001.0, "23.976p"),
        (30000.0 / 1001.0, "29.97p"),
        (60000.0 / 1001.0, "59.94p"),
    )
    nearest_rate, label = min(known_rates, key=lambda item: abs(output_rate - item[0]))
    if abs(output_rate - nearest_rate) / nearest_rate > REPORT_RATE_REL_TOL:
        units = timeline["field_units"][start:end + 1]
        kind = "VFR" if len(set(units)) > 1 else "other"
        return f"{kind} {output_rate:.2f}p"

    units = timeline["field_units"][start:end + 1]
    if segment["strategy"] == "match_keep_pts" and label == "23.976p" and any(
        unit != 2 for unit in units
    ):
        return "23.976p/RFF"
    if segment["strategy"] == "match_decimate":
        return f"{label}/dec"
    if segment["strategy"] == "bob_expand":
        return f"{label}/bob"
    return label


def _segment_range_values(segment: Segment, key, start, end):
    """Return per-frame diagnostic values limited to a displayed sub-range."""
    values = segment.get(key)
    if values is None:
        return []
    offset_start = start - segment["src_start"]
    offset_end = end - segment["src_start"] + 1
    return values[offset_start:offset_end]


def _strategy_summary(segments: list[Segment], timeline: SourceTimeline):
    """Build frame- and duration-weighted totals for each strategy."""
    rows = []
    for strategy in STRATEGIES:
        strategy_segments = [
            segment for segment in segments if segment["strategy"] == strategy
        ]
        source_frames = sum(
            segment["source_frame_count"] for segment in strategy_segments
        )
        base_output_frames = sum(
            segment["base_output_frame_count"] for segment in strategy_segments
        )
        final_output_frames = sum(
            segment_output_frame_count(segment) for segment in strategy_segments
        )
        duration_ms = sum(
            sum(timeline["duration_ms"][segment["src_start"]:segment["src_end"] + 1])
            for segment in strategy_segments
        )
        rows.append({
            "strategy": strategy,
            "source_frames": source_frames,
            "duration_ms": duration_ms,
            "base_output_frames": base_output_frames,
            "final_output_frames": final_output_frames,
            "structural_drops": (
                source_frames - base_output_frames
                if strategy == "match_decimate" else 0
            ),
            "dedup_drops": base_output_frames - final_output_frames,
        })
    return rows


def _print_frame_weighted_summary(summary_rows):
    """Print strategy shares weighted by source-frame count."""
    total_source_frames = sum(row["source_frames"] for row in summary_rows)
    total_base_output = sum(row["base_output_frames"] for row in summary_rows)
    total_final_output = sum(row["final_output_frames"] for row in summary_rows)

    print("Decision distribution by source-frame count:")
    print("  decision           source frames    share    pre-dedup       final  structural drops  dedup drops")
    print("  ----------------  ---------------  -------  -----------  ----------  ------------  ----------")
    for row in summary_rows:
        share = row["source_frames"] / max(total_source_frames, 1) * 100.0
        print(
            f"  {row['strategy']:16s} {row['source_frames']:15d} "
            f"{share:6.2f}% {row['base_output_frames']:11d} "
            f"{row['final_output_frames']:10d} {row['structural_drops']:12d} "
            f"{row['dedup_drops']:10d}"
        )
    print(
        f"  {'TOTAL':16s} {total_source_frames:15d} 100.00% "
        f"{total_base_output:11d} {total_final_output:10d}"
    )


def _print_duration_weighted_summary(summary_rows):
    """Print strategy shares weighted by source-timeline duration."""
    total_duration_ms = sum(row["duration_ms"] for row in summary_rows)

    print("")
    print("Decision distribution by source duration:")
    print("  decision                   duration        milliseconds    share")
    print("  ----------------  -----------------  ------------------  -------")
    for row in summary_rows:
        share = row["duration_ms"] / max(total_duration_ms, 1e-9) * 100.0
        print(
            f"  {row['strategy']:16s} {_fmt_report_time(row['duration_ms']):>17s} "
            f"{row['duration_ms']:18.3f} {share:6.2f}%"
        )
    print(
        f"  {'TOTAL':16s} {_fmt_report_time(total_duration_ms):>17s} "
        f"{total_duration_ms:18.3f} 100.00%"
    )


def _joined_values(segment: Segment, key, start, end):
    """Return sorted unique diagnostics as a printable value."""
    values = {
        str(value)
        for value in _segment_range_values(segment, key, start, end)
        if value is not None
    }
    return ",".join(sorted(values)) or "-"


def _lock_description(segment: Segment, start, end):
    """Summarize full or partial classifier locks in a displayed range."""
    locks = []
    for key, label in (
        ("frame_locked_matchable", "matchable"),
        ("frame_locked_bob", "bob"),
    ):
        values = _segment_range_values(segment, key, start, end)
        if values and any(values):
            locks.append(label if all(values) else f"{label}(partial)")
    return ",".join(locks) or "-"


def _build_decision_intervals(segments: list[Segment], timeline: SourceTimeline):
    """Build complete, ordered report intervals from operational segments."""
    intervals = []
    output_cursor = 0
    for segment in segments:
        for start, end in _cadence_display_ranges(segment, timeline):
            start_ms = timeline["pts_ms"][start]
            end_ms = timeline["pts_ms"][end] + timeline["duration_ms"][end]
            duration_ms = max(end_ms - start_ms, 1e-9)
            source_frames = end - start + 1
            base_output_frames, final_output_frames = _range_output_counts(
                segment, start, end
            )
            source_rate = source_frames * 1000.0 / duration_ms
            base_output_rate = base_output_frames * 1000.0 / duration_ms
            final_output_rate = final_output_frames * 1000.0 / duration_ms
            confidence_values = _segment_range_values(
                segment, "frame_confidence", start, end
            )
            confidence = "-"
            if confidence_values:
                minimum = min(confidence_values)
                maximum = max(confidence_values)
                confidence = (
                    f"{minimum:.3f}"
                    if minimum == maximum else f"{minimum:.3f}-{maximum:.3f}"
                )
            output_start = output_cursor
            output_end = output_cursor + final_output_frames - 1
            output_cursor += final_output_frames
            intervals.append({
                "source_start": start,
                "source_end": end,
                "output_start": output_start,
                "output_end": output_end,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "source_frames": source_frames,
                "base_output_frames": base_output_frames,
                "final_output_frames": final_output_frames,
                "source_rate": source_rate,
                "base_output_rate": base_output_rate,
                "final_output_rate": final_output_rate,
                "cadence": _cadence_label(
                    segment, timeline, start, end, base_output_rate
                ),
                "strategy": segment["strategy"],
                "branch": segment["branch"],
                "matchability": _joined_values(
                    segment, "frame_matchability", start, end
                ),
                "redundancy": _joined_values(
                    segment, "frame_redundancy", start, end
                ),
                "classifier_origins": _joined_values(
                    segment, "frame_origins", start, end
                ),
                "redundancy_origins": _joined_values(
                    segment, "frame_redundancy_origins", start, end
                ),
                "confidence": confidence,
                "locks": _lock_description(segment, start, end),
            })
    return intervals


def _print_decision_intervals(intervals):
    """Print every decision interval and the diagnostics supporting it."""
    print("")
    print("Complete decision intervals:")
    print("  Source-frame bounds are inclusive; time ranges are [start, end).")
    print("  #    source frames   source time                         duration  decision          branch      cadence       src/pre/final")
    print("  ---  --------------  ---------------------------------  ------------  ----------------  ----------  ------------  -------------")
    for index, interval in enumerate(intervals, 1):
        output_range = (
            f"{interval['output_start']}-{interval['output_end']}"
            if interval["final_output_frames"] else "-"
        )
        print(
            f"  {index:3d}  {interval['source_start']:6d}-{interval['source_end']:<6d}  "
            f"{_fmt_report_time(interval['start_ms'])}-{_fmt_report_time(interval['end_ms'])}  "
            f"{_fmt_report_time(interval['duration_ms']):>12s}  {interval['strategy']:16s} "
            f"{interval['branch']:10s} {interval['cadence']:12s} "
            f"{interval['source_frames']:4d}/{interval['base_output_frames']:4d}/{interval['final_output_frames']:4d}"
        )
        print(
            f"       output frames={output_range}; source fps={interval['source_rate']:.2f}; "
            f"pre-dedup={interval['base_output_rate']:.2f}; final={interval['final_output_rate']:.2f}"
        )
        print(
            f"       decision: matchability={interval['matchability']}; redundancy={interval['redundancy']}; "
            f"confidence={interval['confidence']}; lock={interval['locks']}"
        )
        print(
            f"       reasons: classifier={interval['classifier_origins']}; mapping={interval['redundancy_origins']}"
        )


def print_strategy_analyze_report(stem, segments: list[Segment], src_tc_path, tc_final, dedup_stats: DedupStats):
    """Print the complete analyze-only report for operational strategies."""
    src_tc = read_timecodes_v2(src_tc_path)
    source_frame_count = sum(segment["source_frame_count"] for segment in segments)
    validate_segments(segments, source_frame_count)
    validate_dedup_stats(dedup_stats)
    timeline = validate_source_timeline(src_tc_path, source_frame_count)
    summary_rows = _strategy_summary(segments, timeline)
    decision_intervals = _build_decision_intervals(segments, timeline)
    buckets = _analyze_buckets(segments, src_tc)
    fps_values = [bucket["fps"] for bucket in buckets]
    drop_values = [bucket["drops"] for bucket in buckets]
    total_output = sum(row["final_output_frames"] for row in summary_rows)
    final_timestamps = read_timecodes_v2(tc_final)
    final_timecode_count = len(final_timestamps) - 1

    if final_timecode_count != total_output:
        raise RuntimeError(
            f"Report mismatch: {final_timecode_count} timecodes for "
            f"{total_output} output frames"
        )
    if not final_timestamps or final_timestamps[-1] <= final_timestamps[-2]:
        raise RuntimeError("Report timecodes have no valid terminal timestamp")
    if decision_intervals and decision_intervals[-1]["output_end"] + 1 != total_output:
        raise RuntimeError("Report intervals do not cover the complete output timeline")

    print("")
    print(f"{'=' * 80}")
    print(f"ANALYZE REPORT PTS-AWARE - {stem}")
    print(f"{'=' * 80}")
    print("Decision legend:")
    print("  match_keep_pts : use the matched TFM branch and preserve every source PTS")
    print("  match_decimate : use the validated TDecimate map and remove structural redundancy")
    print("  bob_expand     : reconstruct every field through the progressive bob branch")
    print("")
    _print_frame_weighted_summary(summary_rows)
    _print_duration_weighted_summary(summary_rows)
    print("")
    print(f"Total output:    {total_output:8d} frames")
    print(f"Final timecodes: {final_timecode_count:8d}")
    _print_decision_intervals(decision_intervals)
    print("")
    print("Optional dedup:")
    print(f"  Input:   {dedup_stats.get('input', 0):8d}")
    print(f"  Output:  {dedup_stats.get('output', 0):8d}")
    print(f"  Dropped: {dedup_stats.get('saved', 0):8d}")

    _print_histogram(
        "Observed pre-dedup FPS",
        fps_values,
        60.0,
        lambda value: f"{value:4.0f}fps",
    )
    max_drop = max(drop_values) if drop_values else 0
    if max_drop:
        _print_histogram(
            "Optional dedup drops",
            drop_values,
            nice_ceil(max_drop),
            lambda value: f"{int(round(value)):5d}",
        )
    else:
        print("")
        print("Optional dedup drops")
        print("  no drops")
def _posterior_distribution(source_path):
    """Classify frame density over time windows for an existing MKV."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = tmp.name
    try:
        cmd = [MKVEXTRACT, str(source_path), "timestamps_v2", f"0:{tmp_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mkvextract timestamps failed: {result.stderr}")
        ptss_ms = []
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    ptss_ms.append(float(line))
                except ValueError:
                    pass
    finally:
        os.unlink(tmp_path)

    if len(ptss_ms) < 2:
        empty = {key: 0.0 for key in ("fps24_pct", "fps30_pct", "fps60_pct", "other_pct", "vfr_pct", "unknown_pct")}
        empty.update({key.replace("_pct", "_duration_pct"): 0.0 for key in empty})
        return {"frames": len(ptss_ms), "duration_ms": 0.0, **empty}

    total_end = source_end_ms(ptss_ms)
    bucket_count = max(1, math.ceil(total_end / REPORT_WINDOW_MS))
    counts = {"fps24": 0, "fps30": 0, "fps60": 0, "other": 0, "vfr": 0, "unknown": 0}
    duration_counts = {key: 0.0 for key in counts}
    known_rates = (
        (24000.0 / 1001.0, "fps24"),
        (30000.0 / 1001.0, "fps30"),
        (60000.0 / 1001.0, "fps60"),
    )
    for bucket_index in range(bucket_count):
        start = bucket_index * REPORT_WINDOW_MS
        end = min(total_end, (bucket_index + 1) * REPORT_WINDOW_MS)
        bucket_ms = end - start
        timestamps = [pts for pts in ptss_ms if start <= pts < end]
        weight = len(timestamps)
        if weight < 2:
            category = "unknown"
        else:
            observed_rate = weight * 1000.0 / bucket_ms
            nearest_rate, nearest_key = min(known_rates, key=lambda item: abs(observed_rate - item[0]))
            if abs(observed_rate - nearest_rate) / nearest_rate <= REPORT_RATE_REL_TOL:
                category = nearest_key
            else:
                deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
                mean_delta = sum(deltas) / len(deltas)
                relative_spread = float(np.std(deltas)) / mean_delta if mean_delta > 0 else float("inf")
                category = "vfr" if relative_spread > PTS_DURATION_CLUSTER_REL_TOL else "other"
        counts[category] += weight
        duration_counts[category] += bucket_ms

    total = sum(counts.values())
    result = {"frames": total, "duration_ms": total_end, **counts}
    result.update({f"{key}_pct": value / max(total, 1) * 100.0 for key, value in counts.items()})
    result.update({f"{key}_duration_pct": value / max(total_end, 1e-9) * 100.0 for key, value in duration_counts.items()})
    return result


def _print_report_distribution_table(results, title, key_suffix):
    """Print one weighting of the post-encode cadence distribution."""
    name_w = max((len(r["name"]) for r in results), default=20)
    name_w = max(name_w, len("File name"))
    col_w = 9
    header = (f"{'File name':<{name_w}}  "
              f"{'24fps':>{col_w}}  "
              f"{'30fps':>{col_w}}  "
              f"{'60fps':>{col_w}}  "
              f"{'other':>{col_w}}  "
              f"{'VFR':>{col_w}}  "
              f"{'unknown':>{col_w}}")
    sep = "-" * len(header)
    print("")
    print(title)
    print(header)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<{name_w}}  {'ERROR':>{col_w}}  "
                  f"{'-':>{col_w}}  {'-':>{col_w}}  {'-':>{col_w}}  {'-':>{col_w}}  "
                  f"{'-':>{col_w}}")
        else:
            print(f"{r['name']:<{name_w}}  "
                  f"{r[f'fps24_{key_suffix}']:>{col_w - 1}.1f}%  "
                  f"{r[f'fps30_{key_suffix}']:>{col_w - 1}.1f}%  "
                  f"{r[f'fps60_{key_suffix}']:>{col_w - 1}.1f}%  "
                  f"{r[f'other_{key_suffix}']:>{col_w - 1}.1f}%  "
                  f"{r[f'vfr_{key_suffix}']:>{col_w - 1}.1f}%  "
                  f"{r[f'unknown_{key_suffix}']:>{col_w - 1}.1f}%")
    print(sep)


def _print_report_table(results):
    """Print frame- and duration-weighted post-encode distributions."""
    print("")
    print("REPORT VFR")
    _print_report_distribution_table(results, "Distribution by output-frame count:", "pct")
    _print_report_distribution_table(results, "Distribution by duration:", "duration_pct")


def run_report(source):
    """Print the post-encode VFR distribution for an MKV or directory."""
    if source.is_dir():
        files = sorted(source.glob("*.mkv"))
        if not files:
            print(f"No .mkv files found in {source}")
            return
        print(f"Directory: {source}")
        print(f"Files found: {len(files)}")
    else:
        files = [source]
        print(f"File: {source}")

    results = []
    for f in files:
        print(f"Analyzing: {f.name}")
        try:
            pct_result = _posterior_distribution(f)
            results.append({"name": f.name, **pct_result})
        except Exception as ex:
            print(f"  Error: {ex}")
            results.append({"name": f.name, "error": str(ex)})

    _print_report_table(results)
