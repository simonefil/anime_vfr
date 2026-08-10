# -*- coding: utf-8 -*-
"""Shared VapourSynth source selection for runtime and generated scripts."""

BESTSOURCE = "bestsource"
FFMS2 = "ffms2"
LSMAS = "lsmas"


def open_video_source(core, source_path, backend=BESTSOURCE, cache_path=None, threads=0):
    """Open a video with the selected source backend."""
    options = {"threads": threads} if threads else {}
    if backend == BESTSOURCE:
        return core.bs.VideoSource(str(source_path), **options)
    if backend == FFMS2:
        if cache_path is not None:
            options["cache"] = 1
            options["cachefile"] = str(cache_path)
        else:
            options["cache"] = 0
        return core.ffms2.Source(str(source_path), **options)
    if backend == LSMAS:
        options.update({"cache": 0, "repeat": 0})
        return core.lsmas.LWLibavSource(str(source_path), **options)
    raise ValueError(f"Unsupported video source backend: {backend}")


def render_video_source_call(source_path, backend=BESTSOURCE, cache_path=None, threads=0):
    """Render a source expression for generated Python and VPY files."""
    options = []
    if threads:
        options.append(f"threads={threads}")
    if backend == BESTSOURCE:
        suffix = f", {', '.join(options)}" if options else ""
        return f"core.bs.VideoSource({str(source_path)!r}{suffix})"
    if backend == FFMS2:
        if cache_path is None:
            options.append("cache=0")
            return f"core.ffms2.Source({str(source_path)!r}, {', '.join(options)})"
        options.append("cache=1")
        options.append(f"cachefile={str(cache_path)!r}")
        return f"core.ffms2.Source({str(source_path)!r}, {', '.join(options)})"
    if backend == LSMAS:
        options.extend(("cache=0", "repeat=0"))
        return f"core.lsmas.LWLibavSource({str(source_path)!r}, {', '.join(options)})"
    raise ValueError(f"Unsupported video source backend: {backend}")
