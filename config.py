# -*- coding: utf-8 -*-
"""Load Anime_VFR settings from config.toml."""

import tomllib
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("config.toml")
DEFAULT_ENCODER_PARAMS = (
    "--y4m -i - --codec hevc --preset p7 --tune uhq "
    "--output-depth 10 "
    "--max-bitrate 80000 --lookahead 32 --bframes 5 --ref 6 "
    "--cqp 12:14:16 --tf-level 0 --colorrange limited"
)


def _load_config():
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


_CONFIG = _load_config()
_TOOLS = _CONFIG.get("tools", {})
_ENCODER = _CONFIG.get("encoder", {})

MKVMERGE = _TOOLS.get("mkvmerge", "mkvmerge")
MKVEXTRACT = _TOOLS.get("mkvextract", "mkvextract")
MEDIAINFO = _TOOLS.get("mediainfo", "mediainfo")
FFMPEG = _TOOLS.get("ffmpeg", "ffmpeg")
VSPIPE = _TOOLS.get("vspipe", "vspipe")
IVTCVK_PLUGIN = _TOOLS.get("ivtc_plugin") or None
ENCODER_BIN = _TOOLS.get("encoder", "NVEncC64")
ENCODER_PARAMS = _ENCODER.get("params", DEFAULT_ENCODER_PARAMS)

# Optional progressive-only dedup mode.
MM_DEDUP_THRESH = 5
MM_DEDUP_CAP = 4

# Post-encode VFR report windows and tolerances.
REPORT_RATE_REL_TOL = 0.02
REPORT_WINDOW_MS = 2000.0
PTS_DURATION_CLUSTER_REL_TOL = 0.06
