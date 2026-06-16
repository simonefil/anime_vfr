#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrazione principale della pipeline VFR anime_vfr."""

import argparse
from fractions import Fraction
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from config import (
    ENCODER_BIN,
    MM_DEDUP_CAP,
    MM_DEDUP_ENABLED,
    MM_FFT_HIGH,
    MM_FFT_PROMOTION_ENABLED,
    MM_FFT_PROMOTION_MIN_MOTION,
    MM_FFT_VERY_LOW,
    MM_BOB_GAP_MAX,
    MM_INHERITANCE_DENSITY_WIN,
    MM_INHERITANCE_DOMINANCE,
    MM_ISOLATION_RATIO,
    MM_MATCH_THRESH,
    MM_MIN_CLUSTER,
    MM_TELECINE_NEIGHBOR_REQ,
    MM_VERIFY_COMBED_THRESH,
    MM_VERIFY_MIN_MOTION,
    MM_VERIFY_MIN_SIZE,
    MM_VERTICAL_SCROLL_BEST_MAX,
    MM_VERTICAL_SCROLL_DIRECT_MIN,
    MM_VERTICAL_SCROLL_ENABLED,
    MM_VERTICAL_SCROLL_IMPROVEMENT_MIN,
    MM_VERTICAL_SCROLL_MIN_HITS,
    MM_VERTICAL_SCROLL_MIN_RUN,
    MM_VERTICAL_SCROLL_SOFT_BEST_MAX,
    MM_VERTICAL_SCROLL_SOFT_DIRECT_MIN,
    MM_VERTICAL_SCROLL_SOFT_IMPROVEMENT_MIN,
    MM_VERTICAL_SCROLL_SHIFT,
    MM_VERTICAL_SCROLL_WINDOW,
    PYTHON_BIN,
    VSPIPE,
)
from dedup import run_dedup_detection, run_progressive_dedup_detection
from encode import encode, get_chroma_flags, get_color_flags, get_par_flags, mux_final
from media import (
    extract_chapter_ranges,
    extract_source_timecodes,
    get_video_frame_count,
    get_video_info,
)
from report import print_analyze_report, run_report
from segments import (
    apply_classification_overrides,
    framemap_to_segments,
    make_bob_entries_from_source_timecodes,
    make_progressive_entries_from_source_timecodes,
    parse_framemap,
)
from timecodes import generate_final_timecodes_v2, source_end_ms
from utils import (
    box_max_16 as _box_max_16,
    read_timecodes_v2,
)

VPY_FMTC_HELPERS = '''\
def fmtc_to_yuv420p8(src):
    src = core.fmtc.bitdepth(src, bits=16)
    src = core.fmtc.resample(src, csp=vs.YUV420P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(src, bits=8)

def fmtc_to_yuv420p10(src, width=None, height=None):
    src = core.fmtc.bitdepth(src, bits=16)
    if width is None or height is None:
        src = core.fmtc.resample(src, csp=vs.YUV420P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    else:
        src = core.fmtc.resample(src, w=width, h=height, csp=vs.YUV420P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(src, bits=10)

def fmtc_to_yuv444p10(src):
    src = core.fmtc.bitdepth(src, bits=16)
    src = core.fmtc.resample(src, csp=vs.YUV444P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(src, bits=10)
'''


def fmtc_to_yuv420p8(core, vs, clip):
    clip = core.fmtc.bitdepth(clip, bits=16)
    clip = core.fmtc.resample(clip, csp=vs.YUV420P16, kernel="spline16", cplaces="mpeg2", cplaced="mpeg2")
    return core.fmtc.bitdepth(clip, bits=8)


def _field_order_settings(field_order):
    tff = field_order == "tff"
    return {
        "tff": tff,
        "fieldbased": 2 if tff else 1,
        "tfm_order": 1 if tff else 0,
        "suffix": "" if tff else "_bff",
    }


def run_pass1(source_path, work_dir, field_order="tff"):
    # TFM analizza il pulldown 3:2 tramite match tra campi. TDecimate mode=4 conta i pattern
    # di decimazione senza produrre output video. Genera stats + tfm per le pass successive.
    stem = source_path.stem
    field = _field_order_settings(field_order)
    suffix = field["suffix"]
    stats_path = work_dir / f"{stem}{suffix}_stats.txt"
    tfm_path = work_dir / f"{stem}{suffix}_tfm.txt"
    script_path = work_dir / f"{stem}{suffix}_pass1.vpy"
    source_esc = str(source_path).replace("\\", "\\\\")
    stats_esc = str(stats_path).replace("\\", "\\\\")
    tfm_esc = str(tfm_path).replace("\\", "\\\\")
    content = f'''import vapoursynth as vs
core = vs.core
{VPY_FMTC_HELPERS}
clip = core.bs.VideoSource(r"{source_esc}")
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval={field["fieldbased"]})
clip = fmtc_to_yuv420p8(clip)
clip = core.tivtc.TFM(clip, order={field["tfm_order"]}, cthresh=8, output=r"{tfm_esc}")
clip = core.tivtc.TDecimate(clip, mode=4, output=r"{stats_esc}")
clip.set_output(0)
'''
    if stats_path.exists() and tfm_path.exists():
        print(f"  Pass 1: file esistenti, riutilizzo...")
        return stats_path, tfm_path
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    cmd = [VSPIPE, "--progress", str(script_path), "--"]
    print(f"  Pass 1: analisi tivtc...")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"Pass 1 fallita per {source_path}")
    return stats_path, tfm_path


def run_pass2a(source_path, stats_path, tfm_path, work_dir, field_order="tff"):
    # Costruisce lo stream ibrido TIVTC usato come timeline decimata canonica.
    # Ogni frame output riceve _OrigFrameNum, cosi' le fasi successive possono
    # risalire al frame sorgente selezionato da TFM/TDecimate.
    # Viene eseguito in un sottoprocesso per isolare il ciclo get_frame() di VapourSynth
    # dallo stato del processo principale.
    stem = source_path.stem
    field = _field_order_settings(field_order)
    suffix = field["suffix"]
    tc_v1_path = work_dir / f"{stem}{suffix}_tc_v1.txt"
    framemap_path = work_dir / f"{stem}{suffix}_framemap.txt"
    mapper_script = work_dir / f"{stem}{suffix}_mapper.py"

    source_esc = str(source_path).replace("\\", "\\\\")
    stats_esc = str(stats_path).replace("\\", "\\\\")
    tfm_esc = str(tfm_path).replace("\\", "\\\\")
    tc_esc = str(tc_v1_path).replace("\\", "\\\\")
    fm_esc = str(framemap_path).replace("\\", "\\\\")

    mapper_content = f'''import vapoursynth as vs
core = vs.core
{VPY_FMTC_HELPERS}
clip = core.bs.VideoSource(r"{source_esc}")
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval={field["fieldbased"]})
clip = fmtc_to_yuv420p8(clip)

def set_fn(n, f):
    fout = f.copy()
    fout.props["_OrigFrameNum"] = n
    return fout
clip = core.std.ModifyFrame(clip, clip, set_fn)

clip = core.tivtc.TFM(clip, order={field["tfm_order"]}, cthresh=8, input=r"{tfm_esc}")
clip = core.tivtc.TDecimate(clip, mode=5, hybrid=2, vfrDec=1, input=r"{stats_esc}", tfmIn=r"{tfm_esc}", mkvOut=r"{tc_esc}")

with open(r"{fm_esc}", "w") as f:
    for i in range(clip.num_frames):
        fr = clip.get_frame(i)
        orig = fr.props["_OrigFrameNum"]
        dur_den = fr.props["_DurationDen"]
        combed = fr.props.get("_Combed", 0)
        f.write(f"{{i}},{{orig}},{{dur_den}},{{combed}}\\n")
print(f"Framemap: {{clip.num_frames}} frame")
'''
    if tc_v1_path.exists() and framemap_path.exists():
        print(f"  Pass 2a: file esistenti, riutilizzo...")
        return tc_v1_path, framemap_path
    with open(mapper_script, "w", encoding="utf-8") as f:
        f.write(mapper_content)

    print(f"  Pass 2a: TDecimate mode=5 + framemap...")
    result = subprocess.run([PYTHON_BIN, str(mapper_script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Pass 2a fallita: {result.stderr}")
    print(f"  {result.stdout.strip()}")

    return tc_v1_path, framemap_path


# ═════════════════════════════════════════════════════════════════════════
# CLASSIFICATORE MULTI-METRICA
# Classificazione per frame sorgente combinando segnali multipli:
#  - pattern ciclico dei match tra campi;
#  - flag combed di TFM;
#  - energia FFT a Nyquist verticale;
#  - movimento per campo.
# Dopo la classificazione iniziale vengono applicati filtri di consistenza,
# isolamento cluster, ereditarieta' dalle ancore vicine e verifica IVTC speculativa.
# ═════════════════════════════════════════════════════════════════════════

def _fft_nyquist_score_callback(n, f):
    """Funzione callback VapourSynth: calcola il rapporto FFT a Nyquist sul luma.
    L'energia verticale a Nyquist indica pattern alternati tra righe,
    cioe' una firma tipica di interlacciamento con movimento.
    """
    fout = f.copy()
    arr = np.asarray(f[0])  # piano luma H x W
    fft = np.fft.rfft(arr.astype(np.float32), axis=0)
    magnitudes = np.abs(fft)
    nyquist_energy = float(np.mean(magnitudes[-1]))
    low_energy = float(np.mean(magnitudes[0:3]))
    fout.props['_FFTNyquistRatio'] = nyquist_energy / (low_energy + 1e-9)
    return fout


def _detect_telecine_phase(window_matches):
    """Rileva la fase 3:2 pulldown. Modalita' stretta: esattamente 2 match a distanza 5.
    Tolleranze (3-4 match) producono falsi positivi.
    """
    matched = [i for i, m in enumerate(window_matches) if m]
    if len(matched) != 2:
        return -1
    if (matched[1] - matched[0]) % 10 == 5:
        return matched[0]
    return -1


def _classify_source_frame(s, match_arr, motion_arr, combed_flags, fft_scores, return_origin=False):
    """Classificatore a cascata con FFT bidirezionale.
    Restituisce una classe o una coppia (classe, origin) se return_origin e' attivo.
    Il campo origin identifica quale regola ha deciso: 'static', 'cycle', 'cycle_combed',
    'combed_only', 'fft_promotion', 'mostly_static', 'default_60i'.
    """
    f = s * 2
    total_fields = len(match_arr)
    win_start = f - 4
    win_end = win_start + 10
    if win_start < 0 or win_end > total_fields:
        ws = max(0, win_start)
        we = min(total_fields, win_end)
        window = list(match_arr[ws:we])
        motion_window = list(motion_arr[ws:we])
        while len(window) < 10:
            window.append(False)
            motion_window.append(0.0)
    else:
        window = match_arr[win_start:win_end]
        motion_window = motion_arr[win_start:win_end]

    n_matches = sum(window)
    avg_motion = sum(motion_window) / len(motion_window) if motion_window else 0.0
    max_motion = max(motion_window) if motion_window else 0.0
    is_combed = combed_flags.get(s, False)
    phase = _detect_telecine_phase(window)
    fft_score = fft_scores[s] if s < len(fft_scores) else 0.0

    # 1. Statico: quasi tutti i campi matchano e il movimento e' basso.
    if n_matches >= 9 and max_motion < 5:
        result = ("static", "static")
    # 2. Pattern ciclico: se non e' combed lo trattiamo come telecine 24p.
    # Se e' combed, il flag TFM prevale e il frame va nel ramo 60i.
    elif phase >= 0:
        if is_combed:
            result = ("interlaced_60i", "cycle_combed")
        else:
            result = ("telecine_24p", "cycle")
    # 3. Nessun ciclo e frame combed: probabile interlacciato reale.
    elif is_combed:
        result = ("interlaced_60i", "combed_only")
    # 4. Promozione FFT: FFT molto bassa con movimento sufficiente indica
    # materiale progressivo/telecine anche se il pattern non e' completo.
    elif MM_FFT_PROMOTION_ENABLED and fft_score < MM_FFT_VERY_LOW and avg_motion > MM_FFT_PROMOTION_MIN_MOTION:
        result = ("ambiguous_to_telecine", "fft_promotion")
    # 5. Molti match ma pattern incompleto: scena quasi statica che eredita
    # la decisione dai frame vicini.
    elif n_matches >= 7:
        result = ("mostly_static", "mostly_static")
    # 6. Caso residuo prudente: resta ambiguo con bias verso 60i.
    else:
        result = ("ambiguous_to_60i", "default_60i")

    if return_origin:
        return result
    return result[0]


def _vertical_shift_match(field, prev_field):
    """Misura lo shift verticale atteso per scroll interlacciati.

    Usiamo blocchi 16x16 non sovrapposti invece della finestra sliding: per
    questo rilevatore serve solo capire se esiste una zona locale coerente con
    lo scroll verticale, non trovare il massimo blocco pixel-perfect.
    """
    def block_max_16(arr):
        h = (arr.shape[0] // 16) * 16
        w = (arr.shape[1] // 16) * 16
        if h < 16 or w < 16:
            return float(np.mean(arr))
        blocks = arr[:h, :w].reshape(h // 16, 16, w // 16, 16)
        return float(blocks.mean(axis=(1, 3)).max())

    direct = block_max_16(np.abs(field - prev_field))
    shift = MM_VERTICAL_SCROLL_SHIFT
    if shift > 0:
        a = field[shift:]
        b = prev_field[:-shift]
    else:
        a = field[:shift]
        b = prev_field[-shift:]
    if a.shape[0] < 16:
        return direct, direct, 0
    shifted = block_max_16(np.abs(a - b))
    return direct, shifted, shift


def _is_vertical_scroll_hit(field, prev_field):
    """Riconosce una transizione da scroll verticale interlacciato."""
    direct, best, shift = _vertical_shift_match(field, prev_field)
    improvement = direct - best
    return (
        shift == MM_VERTICAL_SCROLL_SHIFT and (
            (
                direct >= MM_VERTICAL_SCROLL_DIRECT_MIN and
                best <= MM_VERTICAL_SCROLL_BEST_MAX and
                improvement >= MM_VERTICAL_SCROLL_IMPROVEMENT_MIN
            ) or (
                direct >= MM_VERTICAL_SCROLL_SOFT_DIRECT_MIN and
                best <= MM_VERTICAL_SCROLL_SOFT_BEST_MAX and
                improvement >= MM_VERTICAL_SCROLL_SOFT_IMPROVEMENT_MIN
            )
        )
    )


def _vertical_scroll_filter(classifications, vertical_scroll_hits):
    """Forza a 60i le zone con scroll verticale sostenuto tra field."""
    if not MM_VERTICAL_SCROLL_ENABLED:
        return list(classifications), 0
    if not vertical_scroll_hits:
        return list(classifications), 0

    force_mask = _vertical_scroll_force_mask(len(classifications), vertical_scroll_hits)
    new_cls = list(classifications)
    changed = 0
    for i, force in enumerate(force_mask):
        if force and new_cls[i] != "interlaced_60i":
            new_cls[i] = "interlaced_60i"
            changed += 1
    return new_cls, changed


def _vertical_scroll_force_mask(n_frames, vertical_scroll_hits):
    """Costruisce la maschera dei frame con scroll verticale 60i affidabile."""
    if not MM_VERTICAL_SCROLL_ENABLED or not vertical_scroll_hits:
        return [False] * n_frames

    win = max(1, MM_VERTICAL_SCROLL_WINDOW)
    half = win // 2
    min_hits = max(1, MM_VERTICAL_SCROLL_MIN_HITS)
    cum = [0] * (n_frames + 1)
    for i, hit in enumerate(vertical_scroll_hits[:n_frames]):
        cum[i + 1] = cum[i] + (1 if hit else 0)
    for i in range(len(vertical_scroll_hits), n_frames):
        cum[i + 1] = cum[i]

    force_flags = [False] * n_frames
    for i in range(n_frames):
        ws = max(0, i - half)
        we = min(n_frames, i + half + 1)
        force_flags[i] = cum[we] - cum[ws] >= min_hits

    min_run = max(1, MM_VERTICAL_SCROLL_MIN_RUN)
    force_mask = [False] * n_frames
    i = 0
    while i < n_frames:
        if not force_flags[i]:
            i += 1
            continue
        start = i
        while i < n_frames and force_flags[i]:
            i += 1
        if i - start < min_run:
            continue
        for j in range(start, i):
            force_mask[j] = True
    return force_mask


def _phase_consistency_filter(classifications):
    """Scarta ancore telecine isolate con pochi vicini telecine."""
    new_cls = list(classifications)
    n_rejected = 0
    WIN = 5
    for i in range(len(classifications)):
        if classifications[i] != "telecine_24p":
            continue
        ws = max(0, i - WIN)
        we = min(len(classifications), i + WIN + 1)
        n_telecine = sum(1 for j in range(ws, we) if j != i and classifications[j] == "telecine_24p")
        if n_telecine < MM_TELECINE_NEIGHBOR_REQ:
            new_cls[i] = "ambiguous_to_60i"
            n_rejected += 1
    return new_cls, n_rejected


def _cluster_isolation_filter(classifications):
    """Riclassifica cluster telecine piccoli e isolati come ambiguous_to_60i."""
    runs = []
    i = 0
    while i < len(classifications):
        cur = classifications[i]
        s = i
        while i < len(classifications) and classifications[i] == cur:
            i += 1
        runs.append({"type": cur, "start": s, "end": i-1, "len": i-s})

    new_cls = list(classifications)
    n_demoted = 0
    for j, run in enumerate(runs):
        if run["type"] != "telecine_24p":
            continue
        prev_run = runs[j-1] if j > 0 else None
        next_run = runs[j+1] if j+1 < len(runs) else None
        prev_60i = prev_run["len"] if prev_run and prev_run["type"] == "interlaced_60i" else 0
        next_60i = next_run["len"] if next_run and next_run["type"] == "interlaced_60i" else 0
        surround = prev_60i + next_60i
        ratio = surround / run["len"] if run["len"] > 0 else 0
        if run["len"] < MM_MIN_CLUSTER and ratio > MM_ISOLATION_RATIO:
            for k in range(run["start"], run["end"]+1):
                new_cls[k] = "ambiguous_to_60i"
                n_demoted += 1
    return new_cls, n_demoted


def _short_bob_gap_filter(classifications):
    """Assorbe piccole isole 24p chiuse tra due sezioni 60i."""
    max_gap = max(0, MM_BOB_GAP_MAX)
    if max_gap == 0:
        return list(classifications), 0

    runs = []
    i = 0
    while i < len(classifications):
        cur = classifications[i]
        start = i
        while i < len(classifications) and classifications[i] == cur:
            i += 1
        runs.append({"type": cur, "start": start, "end": i - 1, "len": i - start})

    new_cls = list(classifications)
    changed = 0
    for idx, run in enumerate(runs):
        if run["type"] != "telecine_24p" or run["len"] > max_gap:
            continue
        prev_run = runs[idx - 1] if idx > 0 else None
        next_run = runs[idx + 1] if idx + 1 < len(runs) else None
        if (
            prev_run and prev_run["type"] == "interlaced_60i" and
            next_run and next_run["type"] == "interlaced_60i"
        ):
            for frame in range(run["start"], run["end"] + 1):
                new_cls[frame] = "interlaced_60i"
                changed += 1
    return new_cls, changed


def _nearest_anchor_inheritance(classifications):
    """Risolve i frame ambigui usando ancore telecine/60i vicine."""
    decisive = {"telecine_24p", "interlaced_60i"}
    N = len(classifications)
    WIN = MM_INHERITANCE_DENSITY_WIN
    DOM = MM_INHERITANCE_DOMINANCE

    # Calcoliamo la densita' locale delle ancore con somme cumulative: dopo una
    # preparazione O(N), ogni frame viene valutato in O(1).
    is_tel = [1 if c == "telecine_24p" else 0 for c in classifications]
    is_60i = [1 if c == "interlaced_60i" else 0 for c in classifications]
    cum_tel = [0] * (N + 1)
    cum_60i = [0] * (N + 1)
    for i in range(N):
        cum_tel[i+1] = cum_tel[i] + is_tel[i]
        cum_60i[i+1] = cum_60i[i] + is_60i[i]

    def density(i):
        ws = max(0, i - WIN)
        we = min(N, i + WIN + 1)
        n_tel_w = cum_tel[we] - cum_tel[ws]
        n_60i_w = cum_60i[we] - cum_60i[ws]
        # Se il frame corrente e' gia' un'ancora, lo escludiamo dal conteggio.
        if classifications[i] == "telecine_24p":
            n_tel_w -= 1
        elif classifications[i] == "interlaced_60i":
            n_60i_w -= 1
        return n_tel_w, n_60i_w

    nearest_before = [None] * N
    nearest_after = [None] * N
    last = None
    for i in range(N):
        if classifications[i] in decisive:
            last = (classifications[i], i)
        nearest_before[i] = last
    last = None
    for i in range(N - 1, -1, -1):
        if classifications[i] in decisive:
            last = (classifications[i], i)
        nearest_after[i] = last

    new_cls = list(classifications)
    for i in range(N):
        cur = classifications[i]
        if cur in decisive:
            continue

        # Se nella finestra locale un tipo domina chiaramente, forziamo
        # quella decisione senza affidarci solo alla distanza dall'ancora piu' vicina.
        n_tel_w, n_60i_w = density(i)
        if n_60i_w > DOM * n_tel_w and n_60i_w > 0:
            new_cls[i] = "interlaced_60i"
            continue
        if n_tel_w > DOM * n_60i_w and n_tel_w > 0:
            new_cls[i] = "telecine_24p"
            continue

        # Se nessun tipo domina, ereditiamo dall'ancora piu' vicina con un
        # piccolo bias coerente con il tipo ambiguo di partenza.
        prefer_telecine = cur == "ambiguous_to_telecine"
        tie_pref = "telecine_24p" if prefer_telecine else "interlaced_60i"
        before = nearest_before[i]
        after = nearest_after[i]
        chosen = None
        if before is None and after is None:
            chosen = tie_pref
        elif before is None:
            chosen = after[0]
        elif after is None:
            chosen = before[0]
        else:
            db = i - before[1]
            da = after[1] - i
            if db < da:
                chosen = before[0]
            elif da < db:
                chosen = after[0]
            else:
                chosen = tie_pref if tie_pref in (before[0], after[0]) else before[0]
        if chosen:
            new_cls[i] = chosen
    return new_cls


def _speculative_ivtc_verification(source_clip, classifications, motion_arr, work_dir,
                                  locked_60i_mask=None, log_prefix="", field_order="tff"):
    """Verifica se alcuni cluster 60i possono essere recuperati come 24p.
    Per ogni cluster sufficientemente lungo applica TFM speculativo e misura
    quanti frame restano combed sull'output.
    Riclassifica il cluster come telecine solo se:
      1. il rapporto combed e' sotto soglia, quindi IVTC pulisce il cluster;
      2. il movimento medio e' sopra soglia, quindi il test non e' solo staticita'.
    Il vincolo sul movimento evita che campi quasi identici in scene statiche
    vengano interpretati come recupero IVTC affidabile.
    """
    import vapoursynth as vs
    core = vs.core
    tfm_order = _field_order_settings(field_order)["tfm_order"]

    n_threads = core.num_threads
    runs = []
    i = 0
    while i < len(classifications):
        cur = classifications[i]
        s = i
        while i < len(classifications) and classifications[i] == cur:
            i += 1
        runs.append({"type": cur, "start": s, "end": i-1, "len": i-s})

    new_cls = list(classifications)
    n_recovered = 0
    n_verified = 0
    n_skipped_low_motion = 0
    n_skipped_locked = 0
    for run in runs:
        if run["type"] != "interlaced_60i" or run["len"] < MM_VERIFY_MIN_SIZE:
            continue
        if locked_60i_mask is not None and any(locked_60i_mask[run["start"]:run["end"] + 1]):
            n_skipped_locked += 1
            continue
        # Calcoliamo il movimento medio del cluster. motion_arr e' indicizzato
        # per campo, quindi ogni frame sorgente occupa due posizioni.
        f_start = 2 * run["start"]
        f_end = 2 * (run["end"] + 1)
        cluster_motion = motion_arr[f_start:f_end]
        avg_m = sum(cluster_motion) / len(cluster_motion) if cluster_motion else 0.0
        if avg_m < MM_VERIFY_MIN_MOTION:
            n_skipped_low_motion += 1
            continue
        n_verified += 1
        sub = source_clip[run["start"]:run["end"]+1]
        try:
            matched = core.tivtc.TFM(sub, order=tfm_order, cthresh=8, slow=2)
            n_total = matched.num_frames
            n_combed = 0
            for fr in matched.frames(prefetch=n_threads):
                if fr.props.get('_Combed', 0):
                    n_combed += 1
            ratio = n_combed / n_total if n_total > 0 else 1.0
            if ratio < MM_VERIFY_COMBED_THRESH:
                for k in range(run["start"], run["end"]+1):
                    new_cls[k] = "telecine_24p"
                n_recovered += run["len"]
                print(f"{log_prefix}    Recuperato cluster sorgente {run['start']}-{run['end']} ({run['len']} fr): combed={ratio:.3f}, movimento={avg_m:.1f}")
        except Exception as e:
            print(f"{log_prefix}    Verifica fallita per cluster {run['start']}-{run['end']}: {e}")
    print(
        f"{log_prefix}  Verificati {n_verified} cluster "
        f"({n_skipped_low_motion} saltati per basso movimento, "
        f"{n_skipped_locked} protetti da scroll verticale), "
        f"recuperati {n_recovered} frame"
    )
    return new_cls


def run_multimetric_classification(source_path, work_dir, tfm_path, field_order="tff"):
    """Esegue classificazione multi-metrica e verifica finale.
    Restituisce una lista con una classificazione per frame sorgente.

    La scansione principale decodifica la sorgente una sola volta: la FFT viene
    calcolata in un pool di thread, mentre il ciclo principale mantiene i campi
    del frame precedente necessari alle metriche di movimento.
    """
    import vapoursynth as vs
    core = vs.core
    from concurrent.futures import ThreadPoolExecutor
    import os

    n_threads = os.cpu_count() or 16
    core.num_threads = n_threads
    print(f"  Analisi multi-metrica (FFT + ciclo + combed) — core.num_threads={n_threads}...")
    field = _field_order_settings(field_order)
    field_order_tff = field["tff"]

    # Decodifichiamo i frame sorgente in formato luma/chroma ridotto: le metriche
    # del classificatore lavorano sul piano luma e non richiedono profondita' elevata.
    clip = core.bs.VideoSource(str(source_path), threads=0)
    clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=field["fieldbased"])
    clip = fmtc_to_yuv420p8(core, vs, clip)
    N_src = clip.num_frames

    # Leggiamo i flag combed prodotti da TFM nella pass1.
    combed_flags = {}
    with open(tfm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                combed_flags[int(parts[0])] = (parts[2] == "+")

    # Calcoliamo la FFT in un pool di thread. Nel frattempo il ciclo principale
    # conserva i campi del frame precedente per movimento e confronto duplicati.
    def _fft_compute(arr_copy):
        fft = np.fft.rfft(arr_copy.astype(np.float32), axis=0)
        mags = np.abs(fft)
        nyquist = float(np.mean(mags[-1]))
        low = float(np.mean(mags[0:3]))
        return nyquist / (low + 1e-9)

    print(f"    Scansione singola FFT parallela + confronto campi...")
    t0 = time.time()
    fft_scores = [0.0] * N_src
    motion_arr = [0.0] * (N_src * 2)
    vertical_scroll_hits = [0] * N_src
    prev_top = None
    prev_bot = None
    prev_field = None
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        fft_futures = {}
        for i, fr in enumerate(clip.frames(prefetch=n_threads)):
            arr = np.asarray(fr[0])
            # Inviamo la FFT al pool: NumPy rilascia il GIL durante il lavoro pesante.
            fft_futures[i] = ex.submit(_fft_compute, arr.copy())
            # Il confronto tra campi resta sequenziale perche' dipende dal frame precedente.
            top = arr[0::2].astype(np.int32)
            bot = arr[1::2].astype(np.int32)
            first = top if field_order_tff else bot
            second = bot if field_order_tff else top
            if prev_top is not None:
                motion_arr[2*i] = _box_max_16(np.abs(first - (prev_top if field_order_tff else prev_bot)))
                motion_arr[2*i+1] = _box_max_16(np.abs(second - (prev_bot if field_order_tff else prev_top)))
            if prev_field is not None and _is_vertical_scroll_hit(first, prev_field):
                vertical_scroll_hits[i] = 1
            prev_field = first
            if _is_vertical_scroll_hit(second, prev_field):
                vertical_scroll_hits[i] = 1
            prev_field = second
            prev_top = top
            prev_bot = bot
            # Recuperiamo i risultati asincroni piu' vecchi per limitare la memoria pendente.
            drain_until = i - n_threads * 4
            if drain_until >= 0 and drain_until in fft_futures:
                fft_scores[drain_until] = fft_futures.pop(drain_until).result()
        # Recuperiamo le FFT rimaste in coda a fine scansione.
        for j, fut in list(fft_futures.items()):
            fft_scores[j] = fut.result()
    print(f"    Scansione completata in {time.time()-t0:.1f}s")
    N_fields = len(motion_arr)

    # Convertiamo le metriche di movimento in una maschera booleana di match.
    match_arr = [m < MM_MATCH_THRESH for m in motion_arr]
    if N_fields >= 2:
        match_arr[0] = match_arr[1] = False  # padding iniziale senza frame precedente

    # Classificazione iniziale con tracking diagnostico della regola che decide.
    classifications = []
    origins = [None] * N_src
    for s in range(N_src):
        cls, orig = _classify_source_frame(s, match_arr, motion_arr, combed_flags, fft_scores, return_origin=True)
        classifications.append(cls)
        origins[s] = orig

    # Passate di rifinitura in ordine esplicito. L'isolamento cluster viene dopo
    # la prima ereditarieta' per vedere un contesto gia' risolto e non ambiguo.
    pre_phase = list(classifications)
    classifications, n_rej = _phase_consistency_filter(classifications)
    for i in range(N_src):
        if pre_phase[i] == "telecine_24p" and classifications[i] != "telecine_24p":
            origins[i] = "rejected_by_phase"
    print(f"  Consistenza fase: scartate {n_rej} ancore telecine isolate")

    # Prima ereditarieta': risolve gli ambigui prima dell'isolamento cluster.
    pre_inherit = list(classifications)
    classifications = _nearest_anchor_inheritance(classifications)
    for i in range(N_src):
        if pre_inherit[i] not in ("telecine_24p", "interlaced_60i") and classifications[i] in ("telecine_24p", "interlaced_60i"):
            if classifications[i] == "telecine_24p":
                if pre_inherit[i] == "ambiguous_to_telecine":
                    origins[i] = "inherit_from_telecine_via_fftpromotion"
                else:
                    origins[i] = f"inherit_from_telecine_via_{pre_inherit[i]}"
            else:
                origins[i] = f"inherit_from_60i_via_{pre_inherit[i]}"

    # Isolamento cluster: riclassifica piccoli cluster telecine circondati da 60i.
    pre_cluster = list(classifications)
    classifications, n_dem = _cluster_isolation_filter(classifications)
    for i in range(N_src):
        if pre_cluster[i] == "telecine_24p" and classifications[i] != "telecine_24p":
            origins[i] = "demoted_by_cluster"
    print(f"  Isolamento cluster: riclassificati {n_dem} frame da cluster telecine isolati")

    # Seconda ereditarieta': risolve i frame appena demotati a ambiguous_to_60i.
    classifications = _nearest_anchor_inheritance(classifications)

    # Verifica IVTC speculativa: tenta di recuperare cluster 60i che diventano
    # puliti dopo TFM lento.
    print(f"  Verifica IVTC speculativa sui cluster 60i >= {MM_VERIFY_MIN_SIZE} frame...")
    locked_60i_mask = _vertical_scroll_force_mask(N_src, vertical_scroll_hits)
    pre_verify = list(classifications)
    classifications = _speculative_ivtc_verification(
        clip,
        classifications,
        motion_arr,
        work_dir,
        locked_60i_mask=locked_60i_mask,
        log_prefix="  ",
        field_order=field_order,
    )
    for i in range(N_src):
        if pre_verify[i] == "interlaced_60i" and classifications[i] == "telecine_24p":
            origins[i] = "recovered_by_verification"

    # Scroll verticali ad alto contrasto: sono una firma forte di 60i reale.
    # Li applichiamo alla fine come veto locale, non come ancore per ereditarieta'
    # e isolamento cluster, cosi' una coda scroll non assorbe anche il film vicino.
    pre_scroll_final = list(classifications)
    classifications, n_scroll_final = _vertical_scroll_filter(classifications, vertical_scroll_hits)
    for i in range(N_src):
        if pre_scroll_final[i] != "interlaced_60i" and classifications[i] == "interlaced_60i":
            origins[i] = "vertical_scroll"
    if n_scroll_final:
        print(f"  Scroll verticale: forzati {n_scroll_final} frame a 60i")

    pre_gap_final = list(classifications)
    classifications, n_gap_final = _short_bob_gap_filter(classifications)
    for i in range(N_src):
        if pre_gap_final[i] != "interlaced_60i" and classifications[i] == "interlaced_60i":
            origins[i] = "short_gap_between_60i"
    if n_gap_final:
        print(f"  Gap corti tra sezioni 60i: forzati {n_gap_final} frame a 60i")

    # Riepilogo finale della classificazione sorgente.
    from collections import Counter
    c = Counter(classifications)
    print(f"  Classificazione finale: {dict(c.most_common())}")

    # Distribuzione diagnostica delle regole che hanno prodotto i frame telecine.
    tel_origins = Counter(origins[i] for i in range(N_src) if classifications[i] == "telecine_24p")
    if tel_origins:
        print(f"  Distribuzione origine dei frame finali classificati TELECINE:")
        for orig, n in tel_origins.most_common():
            print(f"    {orig:45s}: {n:6d} ({100*n/sum(tel_origins.values()):5.1f}%)")

    return classifications


def generate_pass2b_script(source_path, tfm_path, stats_path, segments, script_path, resize_w, resize_h,
                           additional_vpy=None, frame_range=None, progressive_source=False,
                           assume_fps_num=30000, assume_fps_den=1001, resize_enabled=False,
                           output_yuv444=False, field_order="tff"):
    # Genera lo script VPY che assembla il clip VFR finale.
    # Il ramo decimato produce i frame film post-IVTC e applica Vinverse sui
    # residui combed. Il ramo bob produce due frame progressivi per ogni frame
    # sorgente nei segmenti interlacciati.
    # AssumeFPS e' solo un tag tecnico nella pipeline VFR; in --bob globale
    # viene impostato al rapporto FPS richiesto per combaciare con il mux CFR.
    source_esc = str(source_path).replace("\\", "\\\\")
    dummy_path = Path(script_path).parent / f"{Path(script_path).stem}_dummy_tc.txt"
    need_decimated = any(s["type"] == "film" for s in segments)
    need_bob = any(s["type"] == "video_bob" for s in segments)
    film_clip_name = "progressive" if progressive_source else "decimated"
    field = _field_order_settings(field_order)
    field_order_tff = field["tff"]
    fieldbased = 0 if progressive_source else field["fieldbased"]
    final_420_line = (
        f"{{name}} = fmtc_to_yuv420p10({{name}}, {resize_w}, {resize_h})\n"
        if resize_enabled
        else "{name} = fmtc_to_yuv420p10({name})\n"
    )
    yuv444_line = (
        'clip = fmtc_to_yuv444p10(clip)\n'
        if output_yuv444 else ""
    )

    script = f'''import vapoursynth as vs
core = vs.core
{VPY_FMTC_HELPERS}
clip = core.bs.VideoSource(r"{source_esc}")
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval={fieldbased})
clip = fmtc_to_yuv420p8(clip)

def splice_many(parts, chunk_size=512):
    if not parts:
        raise ValueError("splice_many: empty parts")
    while len(parts) > 1:
        parts = [core.std.Splice(parts[i:i + chunk_size]) for i in range(0, len(parts), chunk_size)]
    return parts[0]

def frames_from_clip(src, indexes):
    return splice_many([src[i:i + 1] for i in indexes])

segments = []
'''
    if progressive_source:
        script += f'''
progressive = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
{final_420_line.format(name="progressive")}\
'''
    elif need_decimated:
        if tfm_path is None or stats_path is None:
            raise ValueError("La branch film richiede tfm_path e stats_path")
        tfm_esc = str(tfm_path).replace("\\", "\\\\")
        stats_esc = str(stats_path).replace("\\", "\\\\")
        dummy_esc = str(dummy_path).replace("\\", "\\\\")
        script += f'''
decimated = core.tivtc.TFM(clip, order={field["tfm_order"]}, cthresh=8, input=r"{tfm_esc}")
decimated = core.tivtc.TDecimate(decimated, mode=5, hybrid=2, vfrDec=1, input=r"{stats_esc}", tfmIn=r"{tfm_esc}", mkvOut=r"{dummy_esc}")
decimated_vinv = core.vinverse.vinverse(decimated, sstr=2.7, amnt=255, scl=0.25)
decimated = core.std.ModifyFrame(decimated, [decimated, decimated_vinv], lambda n, f: f[1].copy() if f[0].props.get('_Combed', 0) else f[0].copy())
{final_420_line.format(name="decimated")}\
'''
    if need_bob:
        script += f'''
from vsdeinterlace.qtgmc import QTempGaussMC
from vsaa import NNEDI3
bobbed = QTempGaussMC(clip, basic_bobber=NNEDI3(nsize=4, nns=4, qual=2, opencl=True), tff={field_order_tff}, basic_tr=3, final_tr=2, source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED).deinterlace()
{final_420_line.format(name="bobbed")}\
'''
    for seg in segments:
        if seg["type"] == "film":
            if "kept_frames" in seg:
                indexes = [idx for (idx, _rl) in seg["kept_frames"]]
                script += f'segments.append(frames_from_clip({film_clip_name}, {indexes!r}))\n'
            else:
                d_s = seg["dec_start"]
                d_e = seg["dec_end"] + 1
                script += f'segments.append({film_clip_name}[{d_s}:{d_e}])\n'
        elif seg["type"] == "video_bob":
            # Per i segmenti bob usiamo gli indici sorgente continui ricostruiti
            # dalla segmentazione. Il ramo 60i non puo' dipendere dalla timeline
            # TDecimate, altrimenti i frame saltati dal decimatore diventerebbero
            # buchi temporali nel bob.
            indexes = []
            for si in seg["src_indices"]:
                indexes.extend((si * 2, si * 2 + 1))
            script += f'segments.append(frames_from_clip(bobbed, {indexes!r}))\n'
        else:
            raise ValueError(f"Segment type non supportato: {seg['type']}")

    script += f'\nclip = core.std.Splice(segments)\n{yuv444_line}'

    if additional_vpy is not None:
        with open(additional_vpy, "r", encoding="utf-8") as af:
            script += '\n' + af.read() + '\n'

    if frame_range is not None:
        script += f'clip = clip[{frame_range[0]}:{frame_range[1]}]\n'

    script += f'''clip = core.std.AssumeFPS(clip, fpsnum={assume_fps_num}, fpsden={assume_fps_den})
clip.set_output(0)
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


def _print_progressive_dedup_report(stem, film_dec, film_out, total_out, dedup_stats, tc_final):
    """Stampa un report compatto per la modalita' progressive-dedup."""
    run_hist = dedup_stats.get("run_hist", []) if dedup_stats else []
    run = lambda n: run_hist[n] if n < len(run_hist) else 0
    print("")
    print(f"{'=' * 80}")
    print(f"PROGRESSIVE DEDUP REPORT - {stem}")
    print(f"{'=' * 80}")
    print(f"  Frame input:     {film_dec:8d}")
    print(f"  Frame output:    {film_out:8d}")
    print(f"  Frame rimossi:   {film_dec - film_out:8d} ({(film_dec - film_out) / max(film_dec, 1) * 100:6.2f}%)")
    print(f"  Timecode finali: {total_out:8d}")
    print("  Run dedup:")
    for n in range(1, len(run_hist)):
        print(f"    {n}-in-1: {run(n)}")
    print(f"  TC: {tc_final.name}")


def _parse_time_ms(value):
    """Converte hh:mm:ss.xxx, mm:ss.xxx o secondi in millisecondi."""
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0]) * 1000.0
    if len(parts) == 2:
        return (int(parts[0]) * 60000.0) + (float(parts[1]) * 1000.0)
    if len(parts) == 3:
        return (int(parts[0]) * 3600000.0) + (int(parts[1]) * 60000.0) + (float(parts[2]) * 1000.0)
    raise ValueError(f"Timestamp non valido: {value}")


def _parse_bob_range_spec(spec):
    """Converte START-END,START-END in range millisecondi."""
    ranges = []
    if not spec:
        return ranges
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Range bob non valido: {item}")
        start, end = item.split("-", 1)
        start_ms = _parse_time_ms(start)
        end_ms = _parse_time_ms(end)
        if end_ms <= start_ms:
            raise ValueError(f"Range bob con fine <= inizio: {item}")
        ranges.append((start_ms, end_ms))
    return ranges


def _parse_chapter_list(spec):
    """Converte 4,5,6 in indici capitolo 1-based."""
    chapters = []
    if not spec:
        return chapters
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        chapter = int(item)
        if chapter <= 0:
            raise ValueError(f"Capitolo non valido: {item}")
        chapters.append(chapter)
    return chapters


def _parse_dedup_cap(value, option_name):
    """Valida il cap dedup passato da CLI."""
    if value is None:
        return None
    cap = int(value)
    if cap < 1:
        raise ValueError(f"{option_name} deve essere >= 1")
    return cap


def _parse_resize_spec(value):
    """Valida una risoluzione CLI nel formato WIDTHxHEIGHT."""
    if value is None:
        return None
    parts = value.lower().split("x", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--resize deve essere nel formato WIDTHxHEIGHT, es. 768x576")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise ValueError("--resize deve contenere solo numeri, es. 768x576") from exc
    if width <= 0 or height <= 0:
        raise ValueError("--resize richiede width e height > 0")
    return width, height


def _parse_fps_ratio(value, option_name):
    """Valida un rapporto FPS NUM/DEN passato da CLI, ad esempio 60000/1001."""
    raw = value.strip()
    if not raw:
        raise ValueError(f"{option_name} richiede un rapporto FPS valido")

    parts = raw.split("/", 1)
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"{option_name} deve essere nel formato NUM/DEN, es. 60000/1001 o 50/1")

    num = int(parts[0])
    den = int(parts[1])

    if num <= 0 or den <= 0:
        raise ValueError(f"{option_name} deve essere > 0")
    return num, den


def _format_fps_ratio(fps_ratio):
    num, den = fps_ratio
    return f"{num}/{den}"


def _bob_ranges_from_chapters(source_path, work_dir, chapters):
    """Estrae i range temporali dei capitoli da forzare a bob."""
    if not chapters:
        return []
    chapters_xml = work_dir / f"{source_path.stem}_chapters.xml"
    chapter_ranges = extract_chapter_ranges(source_path, chapters_xml)
    ranges = []
    for chapter in chapters:
        idx = chapter - 1
        if idx < 0 or idx >= len(chapter_ranges):
            raise RuntimeError(f"Capitolo {chapter} non presente nel file sorgente")
        ranges.append(chapter_ranges[idx])
    return ranges


def _apply_bob_time_overrides(classifications, src_tc_path, ranges):
    """Forza a 60i i frame sorgente che intersecano range temporali espliciti."""
    if not ranges:
        return list(classifications), 0

    src_tc = read_timecodes_v2(src_tc_path)
    source_end = source_end_ms(src_tc)
    normalized = [(start, source_end if end is None else end) for start, end in ranges]

    new_cls = list(classifications)
    changed = 0
    for idx, start in enumerate(src_tc):
        end = src_tc[idx + 1] if idx + 1 < len(src_tc) else source_end
        if any(end > range_start and start < range_end for range_start, range_end in normalized):
            if new_cls[idx] != "interlaced_60i":
                new_cls[idx] = "interlaced_60i"
                changed += 1
    return new_cls, changed


def process_episode(source_path, output_path, work_dir, strip_audio, strip_sub, additional_vpy=None,
                    frame_range=None, bob=False, analyze_only=False, progressive_dedup=False,
                    dedup_enabled=False, dedup_cap=None, bob_chapters=None, bob_ranges=None,
                    resize_target=None, output_yuv444=False,
                    field_order="tff", bob_fps=(60000, 1001)):
    # Pipeline completa per un episodio: analisi sorgente, classificazione,
    # segmentazione, dedup, generazione timecode, script VPY, encode e mux.
    stem = source_path.stem
    print(f"\n{'=' * 60}")
    print(f"Processamento: {output_path.name}")
    if frame_range is not None:
        print(f"  Frame range: {frame_range[0]}-{frame_range[1]}")
    print(f"{'=' * 60}")

    w, h, sar_num, sar_den = get_video_info(source_path)
    source_frame_count = get_video_frame_count(source_path)
    if resize_target is None:
        resize_w, resize_h = w, h
        resize_enabled = False
        par_flags_enabled = sar_num != sar_den
        display_aspect_ratio = None
        if sar_num != sar_den:
            dar = Fraction(w * sar_num, h * sar_den)
            display_aspect_ratio = f"{dar.numerator}/{dar.denominator}"
        print(f"  Sorgente: {w}x{h} SAR {sar_num}:{sar_den} -> no resize")
    else:
        resize_w, resize_h = resize_target
        resize_enabled = True
        par_flags_enabled = False
        display_aspect_ratio = None
        print(f"  Sorgente: {w}x{h} SAR {sar_num}:{sar_den} -> {resize_w}x{resize_h}")

    src_tc_path = work_dir / f"{stem}_src_timecodes.txt"
    if not src_tc_path.exists():
        extract_source_timecodes(source_path, src_tc_path)

    stats_path = None
    tfm_path = None
    if progressive_dedup:
        print(f"  --progressive-dedup attivo: salto TIVTC/classificatore/bob, dedup diretto su sorgente progressiva")
        entries = make_progressive_entries_from_source_timecodes(src_tc_path, source_frame_count)
    elif bob:
        print(f"  --bob attivo: salto TIVTC/classificatore/dedup, forzo tutto a video_bob ({_format_fps_ratio(bob_fps)}fps)")
        entries = make_bob_entries_from_source_timecodes(src_tc_path, source_frame_count)
    else:
        stats_path, tfm_path = run_pass1(source_path, work_dir, field_order=field_order)
        _tc_v1_path, framemap_path = run_pass2a(source_path, stats_path, tfm_path, work_dir, field_order=field_order)
        entries = parse_framemap(framemap_path)
        # Classificazione multi-metrica dei frame sorgente e applicazione al framemap.
        classifications = run_multimetric_classification(source_path, work_dir, tfm_path, field_order=field_order)
        forced_ranges = []
        if bob_chapters:
            forced_ranges.extend(_bob_ranges_from_chapters(source_path, work_dir, bob_chapters))
        if bob_ranges:
            forced_ranges.extend(bob_ranges)
        if forced_ranges:
            classifications, forced_count = _apply_bob_time_overrides(classifications, src_tc_path, forced_ranges)
            src_end = source_end_ms(read_timecodes_v2(src_tc_path))
            range_text = ", ".join(
                f"{start / 1000.0:.3f}-{(end if end is not None else src_end) / 1000.0:.3f}s"
                for start, end in forced_ranges
            )
            print(f"  Override bob esplicito: {forced_count} frame sorgente forzati a 60i ({range_text})")
        entries = apply_classification_overrides(entries, classifications)

    segments = framemap_to_segments(entries)
    segment_types = sorted({s["type"] for s in segments})
    invalid_types = [t for t in segment_types if t not in ("film", "video_bob")]
    if invalid_types:
        raise RuntimeError(f"Segmenti non binari trovati: {invalid_types}")

    # Dedup dei segmenti film. In modalita' bob non esistono segmenti film.
    effective_dedup_cap = dedup_cap or MM_DEDUP_CAP
    dedup_active = (progressive_dedup or dedup_enabled) and MM_DEDUP_ENABLED and not bob
    dedup_stats = {"input": 0, "output": 0, "saved": 0, "saved_pct": 0.0, "run_hist": [0] * (effective_dedup_cap + 1)}
    if progressive_dedup and dedup_active:
        dedup_stats = run_progressive_dedup_detection(source_path, segments, cap=effective_dedup_cap)
    elif dedup_active:
        dedup_stats = run_dedup_detection(source_path, work_dir, tfm_path, stats_path, segments,
                                          cap=effective_dedup_cap, field_order=field_order)

    film_dec = sum(s["num_dec_frames"] for s in segments if s["type"] == "film")
    bob_dec = sum(s["num_dec_frames"] for s in segments if s["type"] == "video_bob")
    if not dedup_active:
        dedup_stats = {"input": film_dec, "output": film_dec, "saved": 0, "saved_pct": 0.0, "run_hist": [0] * (effective_dedup_cap + 1)}
        if film_dec and len(dedup_stats["run_hist"]) > 1:
            dedup_stats["run_hist"][1] = film_dec
    # Conteggio output: il film tiene conto dei frame rimasti dopo dedup,
    # mentre ogni entry bob produce due frame output.
    film_out = sum(len(s["kept_frames"]) if "kept_frames" in s else s["num_dec_frames"]
                   for s in segments if s["type"] == "film")
    bob_out = bob_dec * 2
    total_out = film_out + bob_out
    if progressive_dedup:
        print(f"  Progressivo: {film_dec} -> {film_out} frame (dedup -{film_dec - film_out})")
    else:
        print(f"  Film:      {film_dec} -> {film_out} frame @ 23.976fps (dedup -{film_dec - film_out})")
        bob_fps_text = _format_fps_ratio(bob_fps) if bob else "59.94"
        print(f"  Video bob: {bob_dec} -> {bob_out} frame @ {bob_fps_text}fps")
    print(f"  Output:    {total_out} frame totali")

    tc_final = work_dir / f"{stem}_tc_final.txt"
    num_tc = generate_final_timecodes_v2(entries, segments, src_tc_path, tc_final)
    if num_tc != total_out:
        raise RuntimeError(f"Mismatch: timecodes={num_tc} vs output={total_out}")

    audio_range = None
    if frame_range is not None:
        with open(tc_final, "r", encoding="utf-8") as f:
            all_tc = [float(line.strip()) for line in f if line.strip() and not line.startswith("#")]
        fr_start = min(frame_range[0], len(all_tc) - 1)
        fr_end = min(frame_range[1], len(all_tc))
        trimmed_tc = all_tc[fr_start:fr_end]
        t_origin = trimmed_tc[0]
        trimmed_tc = [tc - t_origin for tc in trimmed_tc]
        with open(tc_final, "w", encoding="utf-8") as f:
            f.write("# timecode format v2\n")
            for tc in trimmed_tc:
                f.write(f"{tc:.6f}\n")
        total_out = len(trimmed_tc)
        ss_s = all_tc[fr_start] / 1000.0
        to_s = (all_tc[fr_end - 1] + 50) / 1000.0
        def _fmt_ts(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        audio_range = (_fmt_ts(ss_s), _fmt_ts(to_s))
        print(f"  Frame range: {fr_start}-{fr_end} ({total_out} frame)")

    vpy_path = work_dir / f"{stem}_pass2b.vpy"
    assume_fps_num = bob_fps[0] if bob else 30000
    assume_fps_den = bob_fps[1] if bob else 1001
    generate_pass2b_script(source_path, tfm_path, stats_path, segments, vpy_path, resize_w, resize_h,
                           additional_vpy, frame_range, progressive_source=progressive_dedup,
                           assume_fps_num=assume_fps_num, assume_fps_den=assume_fps_den,
                           resize_enabled=resize_enabled, output_yuv444=output_yuv444,
                           field_order=field_order)

    if analyze_only:
        if progressive_dedup:
            _print_progressive_dedup_report(stem, film_dec, film_out, total_out, dedup_stats, tc_final)
        else:
            print_analyze_report(stem, entries, segments, src_tc_path, tc_final,
                                 film_dec, film_out, bob_dec, bob_out, dedup_stats)
        print(f"  Analyze-only: salto encode/mux. VPY: {vpy_path.name}, TC: {tc_final.name}")
        return {
            "name": output_path.name,
            "mode": "bob" if bob else ("progressive_dedup" if progressive_dedup else "hybrid"),
            "source_frames": entries[-1][1] + 1 if entries else 0,
            "film_frames_24": film_out,
            "video_frames_60": bob_out,
            "dedup_saved": film_dec - film_out,
            "total_out_frames": total_out,
            "film_pct": film_out / max(total_out, 1) * 100,
            "video60_pct": bob_out / max(total_out, 1) * 100,
            "resolution": f"{resize_w}x{resize_h}",
        }

    encoded_path = work_dir / f"{stem}_encoded.mkv"
    if encoded_path.exists():
        encoded_path.unlink()
    is_ffmpeg = "ffmpeg" in ENCODER_BIN.lower()
    color_flags = get_color_flags(source_path, is_ffmpeg)
    par_flags = get_par_flags(sar_num, sar_den, is_ffmpeg) if par_flags_enabled else ""
    chroma_flags = get_chroma_flags(output_yuv444, is_ffmpeg)
    if color_flags:
        print(f"  Color tags: {color_flags}")
    if par_flags:
        print(f"  PAR tag: {sar_num}:{sar_den}")
    if chroma_flags:
        print("  Chroma output: yuv444p10")
    if display_aspect_ratio is not None:
        print(f"  Aspect tag: {display_aspect_ratio}")
    encode(vpy_path, encoded_path, color_flags, par_flags, chroma_flags)

    mux_timecodes = None if bob else tc_final
    default_duration = f"{_format_fps_ratio(bob_fps)}fps" if bob else None
    mux_final(
        encoded_path,
        source_path,
        mux_timecodes,
        output_path,
        strip_audio,
        strip_sub,
        audio_range,
        default_duration=default_duration,
        display_aspect_ratio=display_aspect_ratio,
    )

    if encoded_path.exists():
        encoded_path.unlink()
    print(f"  Completato: {output_path.name}")

    return {
        "name": output_path.name,
        "mode": "bob" if bob else ("progressive_dedup" if progressive_dedup else "hybrid"),
        "source_frames": entries[-1][1] + 1 if entries else 0,
        "film_frames_24": film_out,
        "video_frames_60": bob_out,
        "dedup_saved": film_dec - film_out,
        "total_out_frames": total_out,
        "film_pct": film_out / max(total_out, 1) * 100,
        "video60_pct": bob_out / max(total_out, 1) * 100,
        "resolution": f"{resize_w}x{resize_h}",
    }


def _output_path_for_source(source, output_dir, output_is_explicit):
    """Decide il nome output usando solo nome sorgente e suffisso anti-overwrite."""
    if output_is_explicit:
        output_path = output_dir / source.name
        if output_path.resolve() == source.resolve():
            raise RuntimeError("L'output esplicito coincide con il sorgente: scegli una cartella diversa")
        return output_path
    return source.parent / f"{source.stem}_1{source.suffix}"


def _cleanup_work_dir(work_dir):
    """Rimuove i file intermedi prodotti nella cartella di lavoro."""
    for f in work_dir.iterdir():
        if f.is_file():
            f.unlink()
    try:
        work_dir.rmdir()
    except OSError:
        pass


def _make_run_work_dir(work_root, source):
    """Crea una sottocartella univoca per evitare collisioni tra run parallele."""
    run_id = uuid.uuid4().hex[:12]
    run_dir = work_root / f"{source.stem}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _print_progressive_summary(all_stats):
    """Stampa un riepilogo compatto per una o piu' sorgenti progressive-dedup."""
    total_in = sum(s["source_frames"] for s in all_stats)
    total_out = sum(s["total_out_frames"] for s in all_stats)
    total_saved = sum(s["dedup_saved"] for s in all_stats)
    print(f"\n{'=' * 80}\nRIEPILOGO PROGRESSIVE DEDUP\n{'=' * 80}")
    if len(all_stats) > 1:
        print(f"{'File':<35} {'input':>8} {'output':>8} {'saved':>8} {'saved%':>7}")
        for s in all_stats:
            print(f"  {s['name']:<33} {s['source_frames']:>8} {s['total_out_frames']:>8} "
                  f"{s['dedup_saved']:>8} {s['dedup_saved'] / max(s['source_frames'], 1) * 100:>6.1f}%")
        print(f"  {'TOTALE':<33} {total_in:>8} {total_out:>8} {total_saved:>8} "
              f"{total_saved / max(total_in, 1) * 100:>6.1f}%")
    else:
        s = all_stats[0]
        print(f"  Input:       {s['source_frames']:>8} frame")
        print(f"  Output:      {s['total_out_frames']:>8} frame")
        print(f"  Rimossi:     {s['dedup_saved']:>8} ({s['dedup_saved'] / max(s['source_frames'], 1) * 100:.1f}%)")
        print(f"  Risoluzione: {s['resolution']}")


def _print_hybrid_summary(all_stats):
    """Stampa un riepilogo compatto per una o piu' sorgenti ibride."""
    total_24 = sum(s["film_frames_24"] for s in all_stats)
    total_60 = sum(s["video_frames_60"] for s in all_stats)
    total_all = total_24 + total_60
    print(f"\n{'=' * 80}\nRIEPILOGO\n{'=' * 80}")
    if len(all_stats) > 1:
        print(f"{'File':<35} {'24fps':>8} {'60fps':>8} {'24%':>6} {'60%':>6}")
        for s in all_stats:
            t = s["film_frames_24"] + s["video_frames_60"]
            print(f"  {s['name']:<33} {s['film_frames_24']:>8} {s['video_frames_60']:>8} "
                  f"{s['film_frames_24'] / max(t, 1) * 100:>5.1f}% "
                  f"{s['video_frames_60'] / max(t, 1) * 100:>5.1f}%")
        print(f"  {'TOTALE':<33} {total_24:>8} {total_60:>8} "
              f"{total_24 / max(total_all, 1) * 100:>5.1f}% "
              f"{total_60 / max(total_all, 1) * 100:>5.1f}%")
    else:
        s = all_stats[0]
        print(f"  24fps: {s['film_frames_24']:>8} ({s['film_pct']:.1f}%)")
        print(f"  60fps: {s['video_frames_60']:>8} ({s['video60_pct']:.1f}%)")
        print(f"  Totale: {s['total_out_frames']} frame | Risoluzione: {s['resolution']}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline VFR DVD NTSC (23.976/59.94fps)")
    parser.add_argument("source", help="File MKV sorgente o cartella di MKV")
    parser.add_argument("--report", action="store_true", help="Genera solo il report VFR (no encode). Accetta file .mkv o cartella. Deve essere usato da solo.")
    parser.add_argument("--output", default=None, help="Cartella output; se omessa usa la cartella sorgente con suffisso _1")
    parser.add_argument("--strip-audio", action="store_true", help="Rimuovi tracce audio dal mux")
    parser.add_argument("--strip-sub", action="store_true", help="Rimuovi sottotitoli dal mux")
    parser.add_argument("--analyze-only", action="store_true", help="Esegue pass/classificazione/dedup/timecode/VPY ma salta encode e mux")
    parser.add_argument("--keep-work", action="store_true", help="Non cancellare i file intermedi")
    parser.add_argument("--work-dir", default=None, help="Cartella di lavoro (default: <output>/work)")
    parser.add_argument("--additional-vpy", default=None, help="Script VPY aggiuntivo da appendere al pass2b (opera su 'clip')")
    parser.add_argument("--frames", default=None, help="Range di frame output da elaborare (es. 1500 o 100-5000)")
    parser.add_argument("--bob", nargs="?", const="60000/1001", default=None, metavar="NUM/DEN", help="Forza bob deinterlace di tutto il titolo e salta classifier/dedup. Rapporto FPS opzionale per il mux CFR, default 60000/1001; es. --bob 50/1.")
    parser.add_argument("--bob-chapters", default=None, help="Forza a bob capitoli 1-based separati da virgola, es. 4 o 4,5,6.")
    parser.add_argument("--bob-range", default=None, help="Forza a bob range temporali START-END separati da virgola, es. 22:30-23:50,10:00-10:20.")
    parser.add_argument("--field-order", choices=("tff", "bff"), default="tff", help="Ordine field per TFM/QTGMC: tff di default, bff per sorgenti bottom-field-first.")
    parser.add_argument("--progressive-dedup", nargs="?", const="2", default=None, metavar="N", help="Deduplica una sorgente progressiva; N opzionale indica il massimo run da unificare (default se omesso: 2).")
    parser.add_argument("--dedup", nargs="?", const="2", default=None, metavar="N", help="Abilita il dedup sui segmenti film; N opzionale indica il massimo run da unificare (default se omesso: 2).")
    parser.add_argument("--yuv444", action="store_true", help="Produce output video YUV 4:4:4 10-bit invece del default YUV 4:2:0 10-bit.")
    parser.add_argument("--resize", default=None, metavar="WIDTHxHEIGHT", help="Ridimensiona l'output alla risoluzione indicata, es. 768x576.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Errore: {source} non trovato")
        sys.exit(1)
    progressive_dedup_enabled = args.progressive_dedup is not None
    dedup_enabled = args.dedup is not None
    bob_enabled = args.bob is not None

    if bob_enabled and progressive_dedup_enabled:
        print("Errore: --bob e --progressive-dedup sono mutuamente esclusivi")
        sys.exit(1)
    if bob_enabled and (args.bob_chapters or args.bob_range):
        print("Errore: --bob non puo' essere combinato con --bob-chapters o --bob-range")
        sys.exit(1)
    if progressive_dedup_enabled and (args.bob_chapters or args.bob_range):
        print("Errore: --progressive-dedup non puo' essere combinato con --bob-chapters o --bob-range")
        sys.exit(1)

    if args.report:
        forbidden = []
        if args.strip_audio: forbidden.append("--strip-audio")
        if args.strip_sub: forbidden.append("--strip-sub")
        if args.analyze_only: forbidden.append("--analyze-only")
        if args.keep_work: forbidden.append("--keep-work")
        if args.output is not None: forbidden.append("--output")
        if args.work_dir is not None: forbidden.append("--work-dir")
        if args.additional_vpy is not None: forbidden.append("--additional-vpy")
        if args.frames is not None: forbidden.append("--frames")
        if bob_enabled: forbidden.append("--bob")
        if args.bob_chapters is not None: forbidden.append("--bob-chapters")
        if args.bob_range is not None: forbidden.append("--bob-range")
        if args.field_order != "tff": forbidden.append("--field-order")
        if progressive_dedup_enabled: forbidden.append("--progressive-dedup")
        if dedup_enabled: forbidden.append("--dedup")
        if args.yuv444: forbidden.append("--yuv444")
        if args.resize is not None: forbidden.append("--resize")
        if forbidden:
            print(f"Errore: --report non puo' essere combinato con: {', '.join(forbidden)}")
            sys.exit(1)
        run_report(source)
        return

    output_is_explicit = args.output is not None
    output_dir = Path(args.output) if output_is_explicit else (source if source.is_dir() else source.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir) if args.work_dir else output_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    frame_range = None
    if args.frames is not None:
        if "-" in args.frames:
            parts = args.frames.split("-", 1)
            frame_range = (int(parts[0]), int(parts[1]))
        else:
            frame_range = (0, int(args.frames))

    try:
        bob_chapters = _parse_chapter_list(args.bob_chapters)
        bob_ranges = _parse_bob_range_spec(args.bob_range)
        bob_fps = _parse_fps_ratio(args.bob, "--bob") if bob_enabled else (60000, 1001)
        progressive_dedup_cap = _parse_dedup_cap(args.progressive_dedup, "--progressive-dedup")
        hybrid_dedup_cap = _parse_dedup_cap(args.dedup, "--dedup")
        resize_target = _parse_resize_spec(args.resize)
    except ValueError as exc:
        print(f"Errore: {exc}")
        sys.exit(1)
    dedup_cap = progressive_dedup_cap if progressive_dedup_enabled else hybrid_dedup_cap

    if source.is_dir():
        sources = sorted(source.glob("*.mkv"))
        if not sources:
            print(f"Nessun file .mkv trovato in {source}")
            sys.exit(1)
        print(f"Cartella: {source} ({len(sources)} file)")
        print(f"Output:   {output_dir}")
        all_stats = []
        for src in sources:
            output_path = _output_path_for_source(src, output_dir, output_is_explicit)
            work_dir = _make_run_work_dir(work_root, src)
            print(f"  {src.name} -> {output_path.name}")
            ep_stats = process_episode(src, output_path, work_dir, args.strip_audio, args.strip_sub, args.additional_vpy,
                                       frame_range, bob=bob_enabled, analyze_only=args.analyze_only,
                                       progressive_dedup=progressive_dedup_enabled,
                                       dedup_enabled=dedup_enabled,
                                       dedup_cap=dedup_cap,
                                       bob_chapters=bob_chapters,
                                       bob_ranges=bob_ranges,
                                       resize_target=resize_target,
                                       output_yuv444=args.yuv444,
                                       field_order=args.field_order,
                                       bob_fps=bob_fps)
            all_stats.append(ep_stats)
            if not args.keep_work:
                _cleanup_work_dir(work_dir)
        if progressive_dedup_enabled:
            _print_progressive_summary(all_stats)
            return
        _print_hybrid_summary(all_stats)
        return

    output_path = _output_path_for_source(source, output_dir, output_is_explicit)
    work_dir = _make_run_work_dir(work_root, source)
    print(f"Sorgente: {source}")
    print(f"Modalita: file singolo")
    print(f"Output:   {output_path}")
    ep_stats = process_episode(source, output_path, work_dir, args.strip_audio, args.strip_sub, args.additional_vpy,
                               frame_range, bob=bob_enabled, analyze_only=args.analyze_only,
                               progressive_dedup=progressive_dedup_enabled,
                               dedup_enabled=dedup_enabled,
                               dedup_cap=dedup_cap,
                               bob_chapters=bob_chapters,
                               bob_ranges=bob_ranges,
                               resize_target=resize_target,
                               output_yuv444=args.yuv444,
                               field_order=args.field_order,
                               bob_fps=bob_fps)
    all_stats = [ep_stats]
    if progressive_dedup_enabled:
        _print_progressive_summary(all_stats)
        if not args.keep_work:
            _cleanup_work_dir(work_dir)
        return
    _print_hybrid_summary(all_stats)
    if not args.keep_work:
        _cleanup_work_dir(work_dir)


if __name__ == "__main__":
    main()
