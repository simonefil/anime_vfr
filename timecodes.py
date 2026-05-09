# -*- coding: utf-8 -*-
"""Generazione dei timecode VFR finali."""


def source_end_ms(src_tc):
    """Stima il timestamp finale della sorgente partendo dai timestamps_v2."""
    if len(src_tc) >= 2:
        return src_tc[-1] + (src_tc[-1] - src_tc[-2])
    if len(src_tc) == 1:
        return src_tc[0] + 33.367
    return 0.0


def generate_final_timecodes_v2(entries, segments, src_tc_path, output_path):
    """Scrive i timestamps_v2 finali in base a tipo segmento e run dedup."""
    src_tc = []
    with open(src_tc_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src_tc.append(float(line))

    timecodes = []
    for seg in segments:
        seg_s = seg["entry_start"]
        seg_e = seg["entry_end"]
        seg_type = seg["type"]

        first_src = entries[seg_s][1]
        t_start = src_tc[first_src] if first_src < len(src_tc) else src_tc[-1]

        if seg_e + 1 < len(entries):
            next_src = entries[seg_e + 1][1]
            t_end = src_tc[next_src] if next_src < len(src_tc) else src_tc[-1]
        else:
            last_src = entries[seg_e][1]
            t_end = source_end_ms(src_tc) if src_tc else 0.0

        num_dec = seg["num_dec_frames"]
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
            base_dt = (t_end - t_start) / num_dec
            cum = 0
            for _idx, run_len in seg["kept_frames"]:
                timecodes.append(t_start + cum * base_dt)
                cum += run_len
        else:
            num_out = num_dec * 2 if seg_type == "video_bob" else num_dec
            delta = (t_end - t_start) / num_out
            for f in range(num_out):
                timecodes.append(t_start + f * delta)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# timecode format v2\n")
        for tc in timecodes:
            f.write(f"{tc:.6f}\n")

    return len(timecodes)
