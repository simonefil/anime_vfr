# -*- coding: utf-8 -*-
"""Parsing del framemap e costruzione dei segmenti finali film/bob."""

from utils import read_timecodes_v2


def parse_framemap(framemap_path):
    """Legge il CSV pass2a come tuple (indice decimato, indice sorgente, dur_den, combed)."""
    entries = []
    with open(framemap_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",")
                combed = int(parts[3]) if len(parts) > 3 else 0
                entries.append((int(parts[0]), int(parts[1]), int(parts[2]), combed))
    return entries


def apply_classification_overrides(entries, classifications):
    """Applica le classificazioni dei frame sorgente alla timeline decimata."""
    new_entries = []
    n_to_film = 0
    n_to_bob = 0
    for dec_idx, src_idx, dur_den, combed in entries:
        cls = classifications[src_idx] if src_idx < len(classifications) else None
        if cls == "interlaced_60i":
            if (dur_den, combed) != (30000, 1):
                n_to_bob += 1
            new_entries.append((dec_idx, src_idx, 30000, 1))
        else:
            if (dur_den, combed) != (24000, 0):
                n_to_film += 1
            new_entries.append((dec_idx, src_idx, 24000, 0))
    if n_to_bob or n_to_film:
        print(f"  Override binario: {n_to_film} entries -> film, {n_to_bob} entries -> video_bob")
    return new_entries


def framemap_to_segments(entries):
    """Raggruppa entry adiacenti del framemap in segmenti finali film/video_bob."""
    segments = []
    i = 0
    while i < len(entries):
        _dec_f, _src_f, _dur_den, combed = entries[i]
        seg_type = "video_bob" if combed == 1 else "film"
        seg_start = i
        while i < len(entries):
            _d, _s, _dd, c = entries[i]
            t = "video_bob" if c == 1 else "film"
            if t != seg_type:
                break
            i += 1
        segments.append({"type": seg_type, "start": seg_start, "end": i - 1})

    result = []
    for seg in segments:
        s = seg["start"]
        e = seg["end"]
        item = {
            "type": seg["type"],
            "entry_start": s,
            "entry_end": e,
            "dec_start": entries[s][0],
            "dec_end": entries[e][0],
            "src_start": entries[s][1],
            "src_end": entries[e][1],
            "num_dec_frames": e - s + 1,
        }
        # Nei segmenti bob salviamo gli indici sorgente esatti: TDecimate puo'
        # aver saltato frame sorgente dentro la stessa sezione visiva.
        if seg["type"] == "video_bob":
            item["src_indices"] = [entries[i][1] for i in range(s, e + 1)]
        result.append(item)
    return result


def make_bob_entries_from_source_timecodes(src_tc_path):
    """Crea un framemap tutto-bob allineato ai timestamp sorgente."""
    src_tc = read_timecodes_v2(src_tc_path)
    return [(i, i, 30000, 1) for i in range(len(src_tc))]


def make_progressive_entries_from_source_timecodes(src_tc_path):
    """Crea un framemap progressivo lineare allineato ai timestamp sorgente."""
    src_tc = read_timecodes_v2(src_tc_path)
    return [(i, i, 24000, 0) for i in range(len(src_tc))]
