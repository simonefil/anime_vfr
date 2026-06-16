# -*- coding: utf-8 -*-
"""Dedup dei frame nei segmenti film."""

import os
import time

import numpy as np

from config import MM_DEDUP_CAP, MM_DEDUP_THRESH
from utils import box_max_16


def _fmtc_to_yuv420p8(core, vs, clip):
    clip = core.fmtc.bitdepth(clip, bits=16)
    clip = core.fmtc.resample(clip, csp=vs.YUV420P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(clip, bits=8)


def _field_order_settings(field_order):
    tff = field_order == "tff"
    return {
        "fieldbased": 2 if tff else 1,
        "tfm_order": 1 if tff else 0,
    }


def run_dedup_detection(source_path, work_dir, tfm_path, stats_path, segments,
                        threshold=None, cap=None, field_order="tff"):
    """Rileva run di duplicati nei segmenti film e aggiunge kept_frames."""
    import vapoursynth as vs
    core = vs.core
    if threshold is None:
        threshold = MM_DEDUP_THRESH
    if cap is None:
        cap = MM_DEDUP_CAP

    film_segs = [s for s in segments if s["type"] == "film"]
    if not film_segs:
        print("  Dedup: nessun segmento film, salto")
        return {"input": 0, "output": 0, "saved": 0, "saved_pct": 0.0, "run_hist": [0] * ((cap or MM_DEDUP_CAP) + 1)}

    n_threads = os.cpu_count() or 16
    core.num_threads = n_threads
    print(f"  Rilevamento dedup (cap-{cap}, soglia={threshold}) — core.num_threads={n_threads}...")

    dummy_tc = work_dir / "_dedup_dummy_tc.txt"
    decimated = _build_decimated_clip(core, vs, source_path, tfm_path, stats_path, dummy_tc, field_order)

    stats = _run_dedup_on_clip(decimated, segments, n_threads, threshold, cap)

    try:
        if dummy_tc.exists():
            dummy_tc.unlink()
    except Exception:
        pass

    return stats


def run_progressive_dedup_detection(source_path, segments, threshold=None, cap=None):
    """Rileva duplicati direttamente su una sorgente progressiva."""
    import vapoursynth as vs
    core = vs.core
    if threshold is None:
        threshold = MM_DEDUP_THRESH
    if cap is None:
        cap = MM_DEDUP_CAP

    film_segs = [s for s in segments if s["type"] == "film"]
    if not film_segs:
        print("  Dedup progressivo: nessun segmento film, salto")
        return {"input": 0, "output": 0, "saved": 0, "saved_pct": 0.0, "run_hist": [0] * ((cap or MM_DEDUP_CAP) + 1)}

    n_threads = os.cpu_count() or 16
    core.num_threads = n_threads
    print(f"  Rilevamento dedup progressivo (cap-{cap}, soglia={threshold}) — core.num_threads={n_threads}...")

    clip = core.bs.VideoSource(str(source_path), threads=0)
    clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
    clip = _fmtc_to_yuv420p8(core, vs, clip)
    return _run_dedup_on_clip(clip, segments, n_threads, threshold, cap)


def _run_dedup_on_clip(clip, segments, n_threads, threshold, cap):
    """Applica il dedup a un clip gia' allineato agli indici dei segmenti film."""
    film_segs = [s for s in segments if s["type"] == "film"]
    print(f"    Calcolo diff dentro {len(film_segs)} segmento/i film...")
    t0 = time.time()
    diff_arr = {}
    for seg in film_segs:
        ds, de = seg["dec_start"], seg["dec_end"]
        prev_arr = None
        sub = clip[ds:de + 1]
        for offset, fr in enumerate(sub.frames(prefetch=n_threads)):
            i = ds + offset
            arr = np.asarray(fr[0]).astype(np.int32)
            if prev_arr is not None:
                diff_arr[i] = box_max_16(np.abs(arr - prev_arr))
            prev_arr = arr
    print(f"    Pass diff completata in {time.time() - t0:.1f}s")

    n_total_in = 0
    n_total_out = 0
    run_hist = [0] * (cap + 1)
    for seg in film_segs:
        ds, de = seg["dec_start"], seg["dec_end"]
        kept = []
        i = ds
        while i <= de:
            run_len = 1
            while run_len < cap and (i + run_len) <= de:
                d = diff_arr.get(i + run_len, float("inf"))
                if d >= threshold:
                    break
                run_len += 1
            kept.append((i, run_len))
            run_hist[run_len] += 1
            i += run_len
        seg["kept_frames"] = kept
        n_total_in += de - ds + 1
        n_total_out += len(kept)

    saved = n_total_in - n_total_out
    pct = 100.0 * saved / max(n_total_in, 1)
    print(f"  Dedup: {n_total_in} -> {n_total_out} frame film (risparmiati {saved}, -{pct:.1f}%)")
    print("    Lunghezze run: " + ", ".join(f"{rl}x{run_hist[rl]}" for rl in range(1, cap + 1)))

    try:
        if dummy_tc.exists():
            dummy_tc.unlink()
    except Exception:
        pass

    return {
        "input": n_total_in,
        "output": n_total_out,
        "saved": saved,
        "saved_pct": pct,
        "run_hist": run_hist,
    }


def _build_decimated_clip(core, vs, source_path, tfm_path, stats_path, mkvout_path, field_order):
    """Costruisce lo stesso stream film decimato che verra' usato dal pass2b."""
    field = _field_order_settings(field_order)
    clip = core.bs.VideoSource(str(source_path))
    clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=field["fieldbased"])
    clip = _fmtc_to_yuv420p8(core, vs, clip)
    decimated = core.tivtc.TFM(clip, order=field["tfm_order"], cthresh=8, input=str(tfm_path))
    decimated = core.tivtc.TDecimate(
        decimated,
        mode=5,
        hybrid=2,
        vfrDec=1,
        input=str(stats_path),
        tfmIn=str(tfm_path),
        mkvOut=str(mkvout_path),
    )
    decimated_vinv = core.vinverse.vinverse(decimated, sstr=2.7, amnt=255, scl=0.25)
    return core.std.ModifyFrame(
        decimated,
        [decimated, decimated_vinv],
        lambda n, f: f[1].copy() if f[0].props.get("_Combed", 0) else f[0].copy(),
    )
