# -*- coding: utf-8 -*-
"""Small helpers shared by pipeline modules."""

import numpy as np


def read_timecodes_v2(path):
    """Read Matroska v2 timestamps while ignoring comments and blank lines."""
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    return values


def box_max_16(arr_int32):
    """Return the maximum 16x16 mean difference over a luma plane."""
    f = arr_int32.astype(np.float32)
    cs = np.cumsum(np.cumsum(f, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    bs = cs[16:, 16:] - cs[16:, :-16] - cs[:-16, 16:] + cs[:-16, :-16]
    return float(bs.max() / 256.0)
