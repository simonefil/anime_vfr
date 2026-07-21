# -*- coding: utf-8 -*-
"""Shared VapourSynth branch construction for runtime and generated scripts."""

import inspect


def build_matched_decimated_branches(core, clip, tfm_path, tfm_order, stats_path=None, mkvout_path=None, need_decimated=False):
    """Build matched and optional decimated branches from the same TIVTC recipe."""
    from vsdeinterlace import vinverse

    matched_raw = core.tivtc.TFM(clip, order=tfm_order, cthresh=8, input=str(tfm_path))
    matched_vinverse = vinverse(matched_raw, contra_str=2.7, amnt=255, scl=0.25)
    matched = core.std.ModifyFrame(matched_raw, [matched_raw, matched_vinverse], lambda n, f: f[1].copy() if f[0].props.get("_Combed", 0) else f[0].copy())
    branches = {"matched": matched}
    if need_decimated:
        if stats_path is None or mkvout_path is None:
            raise ValueError("The decimated branch requires stats_path and mkvout_path")
        decimated_raw = core.tivtc.TDecimate(matched_raw, mode=5, hybrid=2, vfrDec=1, input=str(stats_path), tfmIn=str(tfm_path), mkvOut=str(mkvout_path))
        decimated_vinverse = vinverse(decimated_raw, contra_str=2.7, amnt=255, scl=0.25)
        branches["decimated"] = core.std.ModifyFrame(decimated_raw, [decimated_raw, decimated_vinverse], lambda n, f: f[1].copy() if f[0].props.get("_Combed", 0) else f[0].copy())
    return branches


def render_matched_decimated_branch_builder():
    """Return the shared builder source for a standalone generated VPY."""
    return inspect.getsource(build_matched_decimated_branches)
