# -*- coding: utf-8 -*-
"""Piccoli helper condivisi tra i moduli della pipeline."""

import numpy as np


def pct(part, total):
    """Calcola una percentuale evitando divisioni per zero."""
    return (part / total * 100.0) if total else 0.0


def read_timecodes_v2(path):
    """Legge timestamp Matroska v2 ignorando commenti e righe vuote."""
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    return values


def box_max_16(arr_int32):
    """Restituisce il massimo diff medio 16x16 su un piano luma."""
    f = arr_int32.astype(np.float32)
    cs = np.cumsum(np.cumsum(f, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    bs = cs[16:, 16:] - cs[16:, :-16] - cs[:-16, 16:] + cs[:-16, :-16]
    return float(bs.max() / 256.0)


def ms_to_timecode(ms):
    """Formatta millisecondi come timestamp capitolo Matroska."""
    h = int(ms // 3600000)
    r = ms - h * 3600000
    m = int(r // 60000)
    r -= m * 60000
    s = int(r // 1000)
    ns = int((r - s * 1000) * 1000000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ns:09d}"


def fmt_mmss(ms):
    """Formatta millisecondi come mm:ss per report compatti."""
    total = max(0, int(round(ms / 1000.0)))
    return f"{total // 60:02d}:{total % 60:02d}"


def nice_ceil(value):
    """Arrotonda un valore positivo verso l'alto per scale grafiche leggibili."""
    if value <= 0:
        return 1
    exp = 10 ** int(np.floor(np.log10(value)))
    norm = value / exp
    if norm <= 1:
        nice = 1
    elif norm <= 2:
        nice = 2
    elif norm <= 5:
        nice = 5
    else:
        nice = 10
    return int(nice * exp)
