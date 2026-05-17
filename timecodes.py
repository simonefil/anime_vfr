# -*- coding: utf-8 -*-
"""Generazione dei timecode VFR finali."""

from utils import read_timecodes_v2


def source_end_ms(src_tc):
    """Stima il timestamp finale della sorgente partendo dai timestamps_v2."""
    if len(src_tc) >= 2:
        return src_tc[-1] + (src_tc[-1] - src_tc[-2])
    if len(src_tc) == 1:
        return src_tc[0] + 33.367
    return 0.0


def _source_timestamp(entry, src_tc):
    src_idx = entry[1]
    if src_idx >= len(src_tc):
        raise RuntimeError(f"Timecode sorgente mancante per frame {src_idx}")
    return src_tc[src_idx]


def generate_final_timecodes_v2(entries, segments, src_tc_path, output_path):
    """Scrive i timestamps_v2 finali sulla timeline reale del sorgente."""
    src_tc = read_timecodes_v2(src_tc_path)

    timecodes = []
    for seg in segments:
        seg_s = seg["entry_start"]
        seg_e = seg["entry_end"]
        seg_type = seg["type"]

        if seg_type == "video_bob" and "src_indices" in seg:
            for src_idx in seg["src_indices"]:
                if src_idx >= len(src_tc):
                    continue
                cur = src_tc[src_idx]
                if src_idx + 1 < len(src_tc):
                    nxt = src_tc[src_idx + 1]
                else:
                    nxt = source_end_ms(src_tc)
                half = (nxt - cur) / 2.0
                timecodes.append(cur)
                timecodes.append(cur + half)
        elif seg_type == "film" and "kept_frames" in seg:
            entry_by_dec = {entries[i][0]: entries[i] for i in range(seg_s, seg_e + 1)}
            for dec_idx, _run_len in seg["kept_frames"]:
                timecodes.append(_source_timestamp(entry_by_dec[dec_idx], src_tc))
        elif seg_type == "film":
            for i in range(seg_s, seg_e + 1):
                timecodes.append(_source_timestamp(entries[i], src_tc))
        else:
            raise ValueError(f"Segment type non supportato: {seg_type}")

    for i in range(1, len(timecodes)):
        if timecodes[i] <= timecodes[i - 1]:
            raise RuntimeError(
                f"Timecode finali non crescenti a frame {i}: "
                f"{timecodes[i - 1]:.6f} -> {timecodes[i]:.6f}"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# timecode format v2\n")
        for tc in timecodes:
            f.write(f"{tc:.6f}\n")

    return len(timecodes)
