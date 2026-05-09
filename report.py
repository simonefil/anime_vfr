# -*- coding: utf-8 -*-
"""Report su standard output per analisi pre-encode e ispezione posteriore MKV."""

import os
import subprocess
import tempfile

import numpy as np

from config import MKVEXTRACT, REPORT_BUCKETS
from timecodes import source_end_ms
from utils import fmt_mmss, nice_ceil, pct, read_timecodes_v2


def segment_bounds_ms(entries, seg, src_tc):
    """Restituisce in millisecondi inizio/fine segmento sulla timeline sorgente."""
    seg_s = seg["entry_start"]
    seg_e = seg["entry_end"]
    first_src = entries[seg_s][1]
    t_start = src_tc[first_src] if first_src < len(src_tc) else src_tc[-1]
    if seg_e + 1 < len(entries):
        next_src = entries[seg_e + 1][1]
        t_end = src_tc[next_src] if next_src < len(src_tc) else src_tc[-1]
    else:
        t_end = source_end_ms(src_tc)
    return t_start, t_end


def _print_histogram(title, values, y_max, y_fmt, height=10):
    """Stampa un istogramma verticale a larghezza fissa con bucket numerati."""
    print("")
    print(title)
    if not values:
        print("  nessun dato")
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


def _analyze_buckets(entries, segments, src_tc, bucket_count=REPORT_BUCKETS):
    """Aggrega FPS pre-dedup e drop dedup in bucket temporali fissi."""
    if not segments:
        return []
    total_end = max(segment_bounds_ms(entries, seg, src_tc)[1] for seg in segments)
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

    for seg in segments:
        start, end = segment_bounds_ms(entries, seg, src_tc)
        fps = 60000.0 / 1001.0 if seg["type"] == "video_bob" else 24000.0 / 1001.0
        a = max(0, min(bucket_count - 1, int(start / bucket_ms)))
        b = max(0, min(bucket_count - 1, int(max(start, end - 0.001) / bucket_ms)))
        for idx in range(a, b + 1):
            bucket = buckets[idx]
            overlap = max(0.0, min(end, bucket["end"]) - max(start, bucket["start"]))
            bucket["fps_weight"] += fps * overlap
            bucket["duration"] += overlap

        # Attribuiamo i frame droppati al bucket che contiene il frame tenuto.
        if seg["type"] == "film" and "kept_frames" in seg:
            base_dt = (end - start) / max(seg["num_dec_frames"], 1)
            for dec_idx, run_len in seg["kept_frames"]:
                if run_len <= 1:
                    continue
                t = start + (dec_idx - seg["dec_start"]) * base_dt
                idx = max(0, min(bucket_count - 1, int(t / bucket_ms)))
                buckets[idx]["drops"] += run_len - 1

    for bucket in buckets:
        bucket["fps"] = bucket["fps_weight"] / bucket["duration"] if bucket["duration"] else 0.0
    return buckets


def print_analyze_report(stem, entries, segments, src_tc_path, tc_final,
                         film_dec, film_out, bob_dec, bob_out, dedup_stats):
    """Stampa sullo standard output il report prodotto da analyze-only."""
    src_tc = read_timecodes_v2(src_tc_path)
    pre_total = film_dec + bob_dec * 2
    post_total = film_out + bob_out
    run_hist = dedup_stats.get("run_hist", []) if dedup_stats else []
    run = lambda n: run_hist[n] if n < len(run_hist) else 0
    buckets = _analyze_buckets(entries, segments, src_tc)
    fps_values = [b["fps"] for b in buckets]
    drop_values = [b["drops"] for b in buckets]

    print("")
    print(f"{'=' * 80}")
    print(f"ANALYZE REPORT - {stem}")
    print(f"{'=' * 80}")
    print("Classificazione pre-dedup:")
    print(f"  24p film: {film_dec:8d} frame ({pct(film_dec, pre_total):6.2f}%)")
    print(f"  60p bob:  {bob_dec * 2:8d} frame ({pct(bob_dec * 2, pre_total):6.2f}%)")
    print(f"  Totale:   {pre_total:8d}")
    print("")
    print("Dedup:")
    print(f"  Film input:  {dedup_stats.get('input', 0) if dedup_stats else 0:8d}")
    print(f"  Film output: {dedup_stats.get('output', 0) if dedup_stats else 0:8d}")
    print(f"  Dropped:     {dedup_stats.get('saved', 0) if dedup_stats else 0:8d} ({dedup_stats.get('saved_pct', 0.0) if dedup_stats else 0.0:6.2f}%)")
    for n in range(1, len(run_hist)):
        print(f"  {n}-in-1: {run(n)}")
    print("")
    print("Output post-dedup:")
    print(f"  24p film: {film_out:8d} frame ({pct(film_out, post_total):6.2f}%)")
    print(f"  60p bob:  {bob_out:8d} frame ({pct(bob_out, post_total):6.2f}%)")
    print(f"  Totale:   {post_total:8d}")
    print(f"  Timecode finali: {len(read_timecodes_v2(tc_final))}")

    _print_histogram("FPS pre-dedup", fps_values, 60.0, lambda v: f"{v:4.0f}fps")
    max_drop = max(drop_values) if drop_values else 0
    if max_drop:
        _print_histogram("Drop dedup", drop_values, nice_ceil(max_drop), lambda v: f"{int(round(v)):5d}")
    else:
        print("")
        print("Drop dedup")
        print("  nessun drop dedup")

    print("")
    print("Bucket:")
    print("  #   intervallo      fps-pre   drop")
    for i, bucket in enumerate(buckets, 1):
        print(f"  {i:2d}  {fmt_mmss(bucket['start'])}-{fmt_mmss(bucket['end'])}  "
              f"{bucket['fps']:7.2f}  {bucket['drops']:5d}")


def _posterior_distribution(source_path):
    """Classifica gli intervalli frame di un MKV gia' prodotto."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = tmp.name
    try:
        cmd = [MKVEXTRACT, str(source_path), "timestamps_v2", f"0:{tmp_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mkvextract timestamps fallito: {result.stderr}")
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
        return {"lt24_pct": 0.0, "film_pct": 0.0, "mid_pct": 0.0, "video60_pct": 0.0, "gt60_pct": 0.0}

    dur_24 = 1001.0 / 24.0
    dur_60 = 1001.0 / 60.0
    tol = 1.5
    lt24 = film = mid = v60 = gt60 = 0
    for i in range(1, len(ptss_ms)):
        dt_ms = ptss_ms[i] - ptss_ms[i - 1]
        fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
        if abs(dt_ms - dur_24) < tol:
            film += 1
        elif abs(dt_ms - dur_60) < tol:
            v60 += 1
        elif fps < 24.0:
            lt24 += 1
        elif 24.0 < fps < 60.0:
            mid += 1
        elif fps > 60.0:
            gt60 += 1
        else:
            mid += 1

    total = lt24 + film + mid + v60 + gt60
    denom = total if total > 0 else 1
    return {
        "frames": total,
        "lt24": lt24,
        "film": film,
        "mid": mid,
        "video60": v60,
        "gt60": gt60,
        "lt24_pct": lt24 / denom * 100.0,
        "film_pct": film / denom * 100.0,
        "mid_pct": mid / denom * 100.0,
        "video60_pct": v60 / denom * 100.0,
        "gt60_pct": gt60 / denom * 100.0,
    }


def _print_report_table(results):
    """Stampa i risultati del report posteriore in una tabella riassuntiva."""
    name_w = max((len(r["name"]) for r in results), default=20)
    name_w = max(name_w, len("Nome file"))
    col_w = 9
    header = (f"{'Nome file':<{name_w}}  "
              f"{'<24':>{col_w}}  "
              f"{'24fps':>{col_w}}  "
              f"{'24-60':>{col_w}}  "
              f"{'60fps':>{col_w}}  "
              f"{'>60':>{col_w}}")
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("REPORT VFR")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<{name_w}}  {'ERRORE':>{col_w}}  "
                  f"{'-':>{col_w}}  {'-':>{col_w}}  {'-':>{col_w}}  {'-':>{col_w}}")
        else:
            print(f"{r['name']:<{name_w}}  "
                  f"{r['lt24_pct']:>{col_w - 1}.1f}%  "
                  f"{r['film_pct']:>{col_w - 1}.1f}%  "
                  f"{r['mid_pct']:>{col_w - 1}.1f}%  "
                  f"{r['video60_pct']:>{col_w - 1}.1f}%  "
                  f"{r['gt60_pct']:>{col_w - 1}.1f}%")
    print(sep)


def run_report(source):
    """Stampa la distribuzione VFR posteriore per un MKV o una cartella di MKV."""
    if source.is_dir():
        files = sorted(source.glob("*.mkv"))
        if not files:
            print(f"Nessun file .mkv trovato in {source}")
            return
        print(f"Cartella: {source}")
        print(f"File trovati: {len(files)}")
    else:
        files = [source]
        print(f"File: {source}")

    results = []
    for f in files:
        print(f"Analisi: {f.name}")
        try:
            pct_result = _posterior_distribution(f)
            results.append({"name": f.name, **pct_result})
        except Exception as ex:
            print(f"  Errore: {ex}")
            results.append({"name": f.name, "error": str(ex)})

    _print_report_table(results)
