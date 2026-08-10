# -*- coding: utf-8 -*-
"""Anime_VFR orchestration using IVTC Vulkan as the hybrid-video backend."""

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

from config import (
    ENCODER_BIN,
    IVTCVK_PLUGIN,
)
from dedup import run_progressive_dedup_detection
from encode import encode, get_chroma_flags, get_color_flags, get_par_flags, mux_final
from ivtcvk_backend import (
    VPY_FMTC_HELPERS,
    detected_track_counts,
    ensure_analysis,
    evaluate_vpy_info,
    read_timecodes,
    render_qtgmc_bob,
    trim_timecodes,
    write_reconstruct_script,
)
from media import extract_source_timecodes, get_video_info
from report import run_report
from segments import make_linear_strategy_segments
from timecodes import generate_final_timecodes_v2
from video_source import BESTSOURCE, FFMS2, LSMAS, render_video_source_call


def _resolve_thread_count(value):
    return max(1, int(value or os.cpu_count() or 16))


def _parse_positive_int(value, option_name, allow_zero=False):
    if value is None:
        return 0 if allow_zero else None
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{option_name} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if number < minimum:
        raise ValueError(f"{option_name} must be >= {minimum}")
    return number


def _parse_frame_range(value):
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", value.strip())
    if match is None:
        raise ValueError(f"Invalid --frames value: {value}; expected N or START-END")
    start = int(match.group(1)) if match.group(2) is not None else 0
    end = int(match.group(2)) if match.group(2) is not None else int(match.group(1))
    if end <= start:
        raise ValueError("--frames END must be greater than START")
    return start, end


def _parse_resize_spec(value):
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if match is None:
        raise ValueError("--resize must use WIDTHxHEIGHT, for example 768x576")
    width, height = map(int, match.groups())
    if width <= 0 or height <= 0:
        raise ValueError("--resize dimensions must be positive")
    return width, height


def _parse_crop_spec(value):
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError("--crop must use LEFT:RIGHT:TOP:BOTTOM")
    try:
        margins = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--crop margins must be integers") from exc
    if any(margin < 0 for margin in margins):
        raise ValueError("--crop margins cannot be negative")
    return margins


def _parse_fps_ratio(value, option_name):
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", value.strip())
    if match is None:
        raise ValueError(f"{option_name} must use NUM/DEN")
    numerator, denominator = map(int, match.groups())
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"{option_name} values must be positive")
    return numerator, denominator


def _parse_dedup_cap(value):
    if value is None:
        return None
    cap = int(value)
    if cap < 1:
        raise ValueError("--progressive-dedup must be >= 1")
    return cap


def _validate_geometry(crop_margins, resize_target, width, height, output_yuv444):
    margins = crop_margins or (0, 0, 0, 0)
    if not output_yuv444 and any(value % 2 for value in margins):
        raise ValueError("Odd crop margins require --yuv444")
    left, right, top, bottom = margins
    cropped_width = width - left - right
    cropped_height = height - top - bottom
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError("Crop removes the complete source frame")
    final_width, final_height = resize_target or (cropped_width, cropped_height)
    if not output_yuv444 and (final_width % 2 or final_height % 2):
        raise ValueError("YUV 4:2:0 output dimensions must be even")
    return cropped_width, cropped_height


def _output_path_for_source(source, output_dir, output_is_explicit):
    if output_is_explicit:
        output = output_dir / source.name
        if output.resolve() == source.resolve():
            raise RuntimeError("Output path matches the source")
        return output
    return source.parent / f"{source.stem}_1{source.suffix}"


def _work_dir_for_source(work_root, source):
    identity = str(source.resolve()).casefold().encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:12]
    path = work_root / f"{source.stem}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_call(source_path, source_filter, source_threads, work_dir):
    return render_video_source_call(
        source_path,
        source_filter,
        cache_path=work_dir / f"{source_path.stem}_ffms2.ffindex",
        threads=source_threads,
    )


def _write_source_probe(source_path, source_filter, source_threads, work_dir, threads):
    script = work_dir / f"{source_path.stem}_source_probe.vpy"
    script.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        f"core.num_threads = {threads}\n"
        f"clip = {_source_call(source_path, source_filter, source_threads, work_dir)}\n"
        "clip.set_output(0)\n",
        encoding="utf-8",
    )
    return script


def _prepare_output_line(name, resize_target, crop_margins, output_yuv444):
    width, height = resize_target if resize_target is not None else (None, None)
    left, right, top, bottom = crop_margins or (0, 0, 0, 0)
    return (
        f"{name} = fmtc_prepare_output({name}, width={width!r}, height={height!r}, "
        f"crop_left={left}, crop_right={right}, crop_top={top}, crop_bottom={bottom}, "
        f"output_yuv444={output_yuv444!r})\n"
    )


def _write_global_bob_script(
    source_path,
    source_filter,
    source_threads,
    field_order,
    script_path,
    resize_target,
    crop_margins,
    output_yuv444,
    threads,
    fps,
    bob_backend,
    additional_vpy,
    frame_range,
):
    source_call = _source_call(source_path, source_filter, source_threads, script_path.parent)
    tff = field_order == "tff"
    fieldbased = 2 if tff else 1
    script = f'''import vapoursynth as vs
from vsdeinterlace.qtgmc import QTempGaussMC
from vsaa import NNEDI3

core = vs.core
core.num_threads = {threads}
{VPY_FMTC_HELPERS}
source = {source_call}
source = core.std.SetFrameProp(source, prop="_FieldBased", intval={fieldbased})
source = fmtc_to_yuv420p8(source)
'''
    script += render_qtgmc_bob("source", "clip", tff, bob_backend)
    script += _prepare_output_line("clip", resize_target, crop_margins, output_yuv444)
    if additional_vpy is not None:
        script += "\n" + Path(additional_vpy).read_text(encoding="utf-8") + "\n"
    if frame_range is not None:
        script += f"clip = clip[{frame_range[0]}:{frame_range[1]}]\n"
    script += f"clip = core.std.AssumeFPS(clip, fpsnum={fps[0]}, fpsden={fps[1]})\n"
    script += "clip.set_output(0)\n"
    script_path.write_text(script, encoding="utf-8")


def _write_progressive_script(
    source_path,
    source_filter,
    source_threads,
    script_path,
    kept_indexes,
    resize_target,
    crop_margins,
    output_yuv444,
    threads,
    additional_vpy,
    frame_range,
):
    source_call = _source_call(source_path, source_filter, source_threads, script_path.parent)
    script = f'''import vapoursynth as vs
core = vs.core
core.num_threads = {threads}
{VPY_FMTC_HELPERS}
source = {source_call}
source = core.std.SetFrameProp(source, prop="_FieldBased", intval=0)
source = fmtc_to_yuv420p8(source)

def splice_many(parts, chunk_size=512):
    while len(parts) > 1:
        parts = [core.std.Splice(parts[i:i + chunk_size]) for i in range(0, len(parts), chunk_size)]
    return parts[0]

clip = splice_many([source[index:index + 1] for index in {kept_indexes!r}])
'''
    script += _prepare_output_line("clip", resize_target, crop_margins, output_yuv444)
    if additional_vpy is not None:
        script += "\n" + Path(additional_vpy).read_text(encoding="utf-8") + "\n"
    if frame_range is not None:
        script += f"clip = clip[{frame_range[0]}:{frame_range[1]}]\n"
    script += "clip = core.std.AssumeFPS(clip, fpsnum=30000, fpsden=1001)\n"
    script += "clip.set_output(0)\n"
    script_path.write_text(script, encoding="utf-8")


def _encode_and_mux(
    source_path,
    output_path,
    work_dir,
    vpy_path,
    timecodes,
    strip_audio,
    strip_sub,
    audio_range,
    sar,
    output_yuv444,
    default_duration=None,
    display_aspect_ratio=None,
):
    encoded_path = work_dir / f"{source_path.stem}_encoded.mkv"
    if encoded_path.exists():
        encoded_path.unlink()
    is_ffmpeg = "ffmpeg" in ENCODER_BIN.lower()
    color_flags = get_color_flags(source_path, is_ffmpeg)
    par_flags = get_par_flags(*sar, is_ffmpeg) if sar[0] != sar[1] else ""
    chroma_flags = get_chroma_flags(output_yuv444, is_ffmpeg)
    encode(vpy_path, encoded_path, color_flags, par_flags, chroma_flags)
    mux_final(
        encoded_path,
        source_path,
        timecodes,
        output_path,
        strip_audio,
        strip_sub,
        audio_range,
        default_duration=default_duration,
        display_aspect_ratio=display_aspect_ratio,
    )
    if encoded_path.exists():
        encoded_path.unlink()


def _process_hybrid(source_path, output_path, work_dir, options, geometry):
    paths = ensure_analysis(
        source_path,
        work_dir,
        options.source_filter,
        options.source_threads,
        options.field_order,
        options.threads,
        backend=options.backend,
        device=options.device,
        batch=options.analysis_batch,
        plugin_path=options.ivtc_plugin,
        force=options.reanalyze,
    )
    source_frames, detected = detected_track_counts(paths["decisions"])
    print(
        "  Detected source frames: "
        + ", ".join(f"{name}={count}" for name, count in sorted(detected.items()))
    )
    print(f"  Decisions: {paths['decisions']}")
    if options.analyze_only:
        print(
            "  Analyze-only complete; edit the TSV, then rerun the same command "
            "without --analyze-only"
        )
        return {
            "name": output_path.name,
            "mode": "hybrid",
            "source_frames": source_frames,
            "output_frames": 0,
            "detected": detected,
        }

    write_reconstruct_script(
        source_path,
        options.source_filter,
        options.source_threads,
        options.field_order,
        options.ivtc_plugin,
        paths["decisions"],
        paths["timecodes_full"],
        paths["vpy"],
        geometry["resize"],
        geometry["crop"],
        options.yuv444,
        options.threads,
        options.bob_backend,
        config_path=options.ivtc_config,
        additional_vpy=options.additional_vpy,
        frame_range=options.frame_range,
    )
    output_frames = evaluate_vpy_info(paths["vpy"])
    full_timecodes = read_timecodes(paths["timecodes_full"])
    if options.frame_range is None:
        expected_output = len(full_timecodes)
    else:
        start, requested_end = options.frame_range
        end = min(requested_end, len(full_timecodes))
        if start >= end:
            raise ValueError(f"Output frame range {start}-{requested_end} is empty")
        expected_output = end - start
    if output_frames != expected_output:
        raise RuntimeError(
            f"VPY output has {output_frames} frames but IVTC timeline has {expected_output}; "
            "additional-vpy must not change frame count or order"
        )
    tc_final, selected_count, audio_range = trim_timecodes(
        paths["timecodes_full"],
        paths["timecodes_final"],
        options.frame_range,
    )
    if selected_count != output_frames:
        raise RuntimeError("Trimmed timecodes do not match VSPipe output")
    _encode_and_mux(
        source_path,
        output_path,
        work_dir,
        paths["vpy"],
        tc_final,
        options.strip_audio,
        options.strip_sub,
        audio_range,
        geometry["sar"],
        options.yuv444,
        display_aspect_ratio=geometry["dar"],
    )
    print(f"  Completed: {output_path.name}")
    return {
        "name": output_path.name,
        "mode": "hybrid",
        "source_frames": source_frames,
        "output_frames": output_frames,
        "detected": detected,
    }


def _process_bob(source_path, output_path, work_dir, options, geometry):
    vpy_path = work_dir / f"{source_path.stem}_bob.vpy"
    _write_global_bob_script(
        source_path,
        options.source_filter,
        options.source_threads,
        options.field_order,
        vpy_path,
        geometry["resize"],
        geometry["crop"],
        options.yuv444,
        options.threads,
        options.bob_fps,
        options.bob_backend,
        options.additional_vpy,
        options.frame_range,
    )
    output_frames = evaluate_vpy_info(vpy_path)
    if output_frames <= 0:
        raise ValueError("Global bob frame selection produced no output frames")
    audio_range = None
    if options.frame_range is not None:
        start, _ = options.frame_range
        end = start + output_frames
        frame_seconds = options.bob_fps[1] / options.bob_fps[0]
        audio_range = (
            _seconds_timestamp(start * frame_seconds),
            _seconds_timestamp(end * frame_seconds),
        )
    if options.analyze_only:
        print(f"  Analyze-only: global bob VPY has {output_frames} output frames")
    else:
        _encode_and_mux(
            source_path,
            output_path,
            work_dir,
            vpy_path,
            None,
            options.strip_audio,
            options.strip_sub,
            audio_range,
            geometry["sar"],
            options.yuv444,
            default_duration=f"{options.bob_fps[0]}/{options.bob_fps[1]}fps",
            display_aspect_ratio=geometry["dar"],
        )
    return {
        "name": output_path.name,
        "mode": "bob",
        "source_frames": output_frames // 2,
        "output_frames": output_frames,
        "detected": {},
    }


def _process_progressive(source_path, output_path, work_dir, options, geometry):
    source_probe = _write_source_probe(
        source_path, options.source_filter, options.source_threads, work_dir, options.threads
    )
    source_frames = evaluate_vpy_info(source_probe)
    source_tc = work_dir / f"{source_path.stem}_src_timecodes.txt"
    if not source_tc.exists() or options.reanalyze:
        extract_source_timecodes(source_path, source_tc)
    segments = make_linear_strategy_segments(source_frames, "match_keep_pts", "progressive")
    dedup_stats = run_progressive_dedup_detection(
        source_path,
        work_dir,
        segments,
        cap=options.progressive_dedup,
        vs_threads=options.threads,
        source_threads=options.source_threads,
        source_backend=options.source_filter,
    )
    full_tc = work_dir / f"{source_path.stem}_tc_full.txt"
    generate_final_timecodes_v2(segments, source_tc, full_tc)
    kept_indexes = [
        segment["branch_indices"][position]
        for segment in segments
        for position, _run_length in segment["kept_positions"]
    ]
    vpy_path = work_dir / f"{source_path.stem}_progressive.vpy"
    _write_progressive_script(
        source_path,
        options.source_filter,
        options.source_threads,
        vpy_path,
        kept_indexes,
        geometry["resize"],
        geometry["crop"],
        options.yuv444,
        options.threads,
        options.additional_vpy,
        options.frame_range,
    )
    output_frames = evaluate_vpy_info(vpy_path)
    final_tc = work_dir / f"{source_path.stem}_tc_final.txt"
    final_tc, selected_count, audio_range = trim_timecodes(full_tc, final_tc, options.frame_range)
    if selected_count != output_frames:
        raise RuntimeError("Progressive dedup timecodes do not match VSPipe output")
    if not options.analyze_only:
        _encode_and_mux(
            source_path,
            output_path,
            work_dir,
            vpy_path,
            final_tc,
            options.strip_audio,
            options.strip_sub,
            audio_range,
            geometry["sar"],
            options.yuv444,
            display_aspect_ratio=geometry["dar"],
        )
    return {
        "name": output_path.name,
        "mode": "progressive_dedup",
        "source_frames": source_frames,
        "output_frames": output_frames,
        "detected": {"dedup_saved": dedup_stats["saved"]},
    }


def _seconds_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    return f"{hours:02d}:{minutes:02d}:{seconds % 60:06.3f}"


def process_episode(source_path, output_path, work_dir, options):
    width, height, sar_num, sar_den = get_video_info(source_path)
    cropped_width, cropped_height = _validate_geometry(
        options.crop, options.resize, width, height, options.yuv444
    )
    if options.resize is None:
        resize = None
        dar = (
            f"{cropped_width * sar_num}/{cropped_height * sar_den}"
            if sar_num != sar_den
            else None
        )
        final_resolution = f"{cropped_width}x{cropped_height}"
    else:
        resize = options.resize
        dar = None
        final_resolution = f"{resize[0]}x{resize[1]}"
    geometry = {
        "crop": options.crop,
        "resize": resize,
        "sar": (sar_num, sar_den) if resize is None else (1, 1),
        "dar": dar,
    }
    print(f"\n{'=' * 72}\nProcessing: {source_path.name}\nOutput: {final_resolution}\n{'=' * 72}")
    if options.bob_fps is not None:
        return _process_bob(source_path, output_path, work_dir, options, geometry)
    if options.progressive_dedup is not None:
        return _process_progressive(source_path, output_path, work_dir, options, geometry)
    return _process_hybrid(source_path, output_path, work_dir, options, geometry)


def _print_summary(stats):
    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    for item in stats:
        details = ", ".join(f"{key}={value}" for key, value in sorted(item["detected"].items()))
        print(
            f"  {item['name']}: mode={item['mode']}, source={item['source_frames']}, "
            f"output={item['output_frames']}" + (f", {details}" if details else "")
        )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="PTS-aware VFR pipeline using IVTC Vulkan"
    )
    parser.add_argument("source", help="Source MKV file or directory")
    parser.add_argument("--report", action="store_true", help="Report an already encoded MKV")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--work-dir", help="Persistent work root")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Replace existing IVTC analysis and TSV",
    )
    parser.add_argument("--strip-audio", action="store_true")
    parser.add_argument("--strip-sub", action="store_true")
    parser.add_argument("--additional-vpy")
    parser.add_argument("--frames")
    parser.add_argument("--bob", nargs="?", const="60000/1001", metavar="NUM/DEN")
    parser.add_argument("--progressive-dedup", nargs="?", const="2", metavar="N")
    parser.add_argument("--field-order", choices=("tff", "bff"), default="tff")
    parser.add_argument(
        "--source-filter",
        choices=(FFMS2, LSMAS, BESTSOURCE),
        default=FFMS2,
    )
    parser.add_argument("--source-threads", default="0", metavar="N")
    parser.add_argument("--threads", metavar="N")
    parser.add_argument("--device", default="0", metavar="N")
    parser.add_argument(
        "--backend",
        choices=("auto", "vulkan", "cpu"),
        default="auto",
        help="IVTC analysis backend; auto falls back to native CPU without a GPU",
    )
    parser.add_argument("--analysis-batch", default="512", metavar="N")
    parser.add_argument(
        "--bob-backend",
        choices=("auto", "opencl", "cpu"),
        default="auto",
    )
    parser.add_argument("--ivtc-plugin", default=IVTCVK_PLUGIN, type=Path)
    parser.add_argument("--ivtc-config", type=Path)
    parser.add_argument("--crop", metavar="L:R:T:B")
    parser.add_argument("--resize", metavar="WIDTHxHEIGHT")
    parser.add_argument("--yuv444", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    source = Path(args.source)
    if not source.exists():
        parser.error(f"source was not found: {source}")
    if args.report:
        incompatible = [
            name
            for name, active in (
                ("--analyze-only", args.analyze_only),
                ("--bob", args.bob is not None),
                ("--progressive-dedup", args.progressive_dedup is not None),
                ("--reanalyze", args.reanalyze),
            )
            if active
        ]
        if incompatible:
            parser.error(f"--report cannot be combined with {', '.join(incompatible)}")
        run_report(source)
        return
    if args.bob is not None and args.progressive_dedup is not None:
        parser.error("--bob and --progressive-dedup are mutually exclusive")
    try:
        args.frame_range = _parse_frame_range(args.frames)
        args.resize = _parse_resize_spec(args.resize)
        args.crop = _parse_crop_spec(args.crop)
        args.bob_fps = _parse_fps_ratio(args.bob, "--bob")
        args.progressive_dedup = _parse_dedup_cap(args.progressive_dedup)
        args.threads = _resolve_thread_count(_parse_positive_int(args.threads, "--threads"))
        args.source_threads = _parse_positive_int(
            args.source_threads, "--source-threads", allow_zero=True
        )
        args.device = _parse_positive_int(args.device, "--device", allow_zero=True)
        args.analysis_batch = _parse_positive_int(
            args.analysis_batch, "--analysis-batch"
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.ivtc_plugin is not None:
        args.ivtc_plugin = args.ivtc_plugin.resolve()
        if (
            args.bob_fps is None
            and args.progressive_dedup is None
            and not args.ivtc_plugin.exists()
        ):
            parser.error(f"IVTC Vulkan plugin was not found: {args.ivtc_plugin}")
    if args.ivtc_config is not None:
        args.ivtc_config = args.ivtc_config.resolve()
        if not args.ivtc_config.exists():
            parser.error(f"IVTC config was not found: {args.ivtc_config}")
    if args.additional_vpy is not None:
        args.additional_vpy = Path(args.additional_vpy).resolve()
        if not args.additional_vpy.exists():
            parser.error(f"additional VPY was not found: {args.additional_vpy}")

    output_is_explicit = args.output is not None
    output_dir = Path(args.output) if output_is_explicit else (
        source if source.is_dir() else source.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir) if args.work_dir else output_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(source.glob("*.mkv")) if source.is_dir() else [source]
    if not sources:
        parser.error(f"no MKV files found in {source}")

    all_stats = []
    for current_source in sources:
        output_path = _output_path_for_source(current_source, output_dir, output_is_explicit)
        work_dir = _work_dir_for_source(work_root, current_source)
        try:
            stats = process_episode(current_source, output_path, work_dir, args)
            all_stats.append(stats)
        except Exception:
            print(f"  Work preserved after failure: {work_dir}", file=sys.stderr)
            raise
        if not args.keep_work and not args.analyze_only:
            resolved_work = work_dir.resolve()
            resolved_root = work_root.resolve()
            if resolved_root not in resolved_work.parents:
                raise RuntimeError(
                    f"Refusing to clean work directory outside root: {resolved_work}"
                )
            shutil.rmtree(resolved_work)
    _print_summary(all_stats)


if __name__ == "__main__":
    main()
