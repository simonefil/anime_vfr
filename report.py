# -*- coding: utf-8 -*-
"""Post-encode VFR cadence reporting."""

import math
import os
import tempfile

import numpy as np

from config import PTS_DURATION_CLUSTER_REL_TOL, REPORT_RATE_REL_TOL, REPORT_WINDOW_MS
from media import extract_source_timecodes


def _source_end_ms(timestamps):
    if len(timestamps) >= 2:
        return timestamps[-1] + timestamps[-1] - timestamps[-2]
    return timestamps[0] + 33.367 if timestamps else 0.0


def _posterior_distribution(source_path):
    """Classify frame density over time windows for an existing MKV."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temporary:
        path = temporary.name
    try:
        extract_source_timecodes(source_path, path)
        with open(path, "r", encoding="utf-8") as stream:
            timestamps = [
                float(line.strip())
                for line in stream
                if line.strip() and not line.startswith("#")
            ]
    finally:
        os.unlink(path)

    keys = (
        "fps24",
        "fps25",
        "fps30",
        "fps50",
        "fps60",
        "other",
        "vfr",
        "unknown",
    )
    if len(timestamps) < 2:
        result = {f"{key}_pct": 0.0 for key in keys}
        result.update({f"{key}_duration_pct": 0.0 for key in keys})
        return {"frames": len(timestamps), "duration_ms": 0.0, **result}

    total_end = _source_end_ms(timestamps)
    bucket_count = max(1, math.ceil(total_end / REPORT_WINDOW_MS))
    counts = {key: 0 for key in keys}
    durations = {key: 0.0 for key in keys}
    known = (
        (24000.0 / 1001.0, "fps24"),
        (25.0, "fps25"),
        (30000.0 / 1001.0, "fps30"),
        (50.0, "fps50"),
        (60000.0 / 1001.0, "fps60"),
    )
    for index in range(bucket_count):
        start = index * REPORT_WINDOW_MS
        end = min(total_end, (index + 1) * REPORT_WINDOW_MS)
        bucket = [value for value in timestamps if start <= value < end]
        if len(bucket) < 2:
            category = "unknown"
        else:
            rate = len(bucket) * 1000.0 / (end - start)
            nearest_rate, nearest_key = min(known, key=lambda item: abs(rate - item[0]))
            if abs(rate - nearest_rate) / nearest_rate <= REPORT_RATE_REL_TOL:
                category = nearest_key
            else:
                deltas = [right - left for left, right in zip(bucket, bucket[1:])]
                mean = sum(deltas) / len(deltas)
                spread = float(np.std(deltas)) / mean if mean > 0 else float("inf")
                category = "vfr" if spread > PTS_DURATION_CLUSTER_REL_TOL else "other"
        counts[category] += len(bucket)
        durations[category] += end - start

    total_frames = sum(counts.values())
    result = {"frames": total_frames, "duration_ms": total_end}
    result.update(
        {f"{key}_pct": counts[key] / max(total_frames, 1) * 100.0 for key in keys}
    )
    result.update(
        {
            f"{key}_duration_pct": durations[key] / max(total_end, 1e-9) * 100.0
            for key in keys
        }
    )
    return result


def _print_table(results, title, suffix):
    width = max([len("File name"), *(len(item["name"]) for item in results)])
    print(f"\n{title}")
    print(
        f"{'File name':<{width}}  {'24fps':>8}  {'25fps':>8}  {'30fps':>8}  "
        f"{'50fps':>8}  {'60fps':>8}  {'other':>8}  {'VFR':>8}  {'unknown':>8}"
    )
    for item in results:
        if "error" in item:
            print(f"{item['name']:<{width}}  ERROR: {item['error']}")
            continue
        print(
            f"{item['name']:<{width}}  {item[f'fps24_{suffix}']:7.1f}%  "
            f"{item[f'fps25_{suffix}']:7.1f}%  {item[f'fps30_{suffix}']:7.1f}%  "
            f"{item[f'fps50_{suffix}']:7.1f}%  {item[f'fps60_{suffix}']:7.1f}%  "
            f"{item[f'other_{suffix}']:7.1f}%  {item[f'vfr_{suffix}']:7.1f}%  "
            f"{item[f'unknown_{suffix}']:7.1f}%"
        )


def run_report(source):
    """Print post-encode VFR distribution for one MKV or a directory."""
    files = sorted(source.glob("*.mkv")) if source.is_dir() else [source]
    if not files:
        print(f"No MKV files found in {source}")
        return
    results = []
    for path in files:
        print(f"Analyzing: {path.name}")
        try:
            results.append({"name": path.name, **_posterior_distribution(path)})
        except Exception as error:
            results.append({"name": path.name, "error": str(error)})
    print("\nREPORT VFR")
    _print_table(results, "Distribution by output-frame count:", "pct")
    _print_table(results, "Distribution by duration:", "duration_pct")
