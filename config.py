# -*- coding: utf-8 -*-
"""Static configuration for external binaries and analysis parameters."""

MKVMERGE = "/opt/homebrew/bin/mkvmerge"
MKVEXTRACT = "/opt/homebrew/bin/mkvextract"
VSPIPE = "/opt/homebrew/bin/vspipe"
PYTHON_BIN = "/opt/homebrew/bin/python3.14"

MEDIAINFO = "/opt/homebrew/bin/mediainfo"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"

ENCODER_BIN = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
ENCODER_PARAMS = (
    "-hide_banner -y -i - -map 0:v:0 -an -sn -dn "
    "-c:v libx265 -preset fast -crf 20"
)

# TFM comb detection. MI is the maximum allowed concentration of combed pixels
# in a 16x16 block before the selected match becomes a hard bob veto.
MM_TFM_CTHRESH = 8
MM_TFM_MI = 32
MM_TFM_SLOW = 2

# Recheck every primary TFM weave with the more sensitive alternate comb
# metric. This pass never selects a match: it only vetoes the match already
# selected by the primary TFM pass when its luma MIC exceeds this limit.
MM_TFM_WEAVE_CTHRESH = 8
MM_TFM_WEAVE_MI = 16
MM_TFM_WEAVE_METRIC = 1
MM_TFM_WEAVE_RFF_MI = 25
MM_TFM_WEAVE_RFF_MIN_RUN = 12

# Per-frame NumPy metrics release the GIL and scale well in parallel.
# None uses the same worker count selected for VapourSynth and prefetch.
MM_ANALYSIS_MAX_WORKERS = None

# Conservative matchable anchors used during initial shadow classification.
# An anchor requires a sustained run of valid, clean TFM records, bounded MIC,
# non-static motion, and no local bob veto. PTS cadence does not decide this.
MM_MATCHABLE_MIN_RUN = 12
MM_MATCHABLE_MIN_MOTION = 5
MM_MATCHABLE_BOB_EDGE_GUARD = 3

# Residual-combing anchors close enough to belong to one temporal region are
# bridged through low-information or clean-match gaps. Never absorb more than
# about one second of clean NTSC source frames between hard bob anchors.
MM_BOB_REGION_MAX_ANCHOR_GAP = 30

# Low-information runs may inherit a surrounding deterministic state.
MM_LOW_INFORMATION_MOTION_MAX = 2
MM_LOW_INFORMATION_INHERIT_MAX = 30

# Candidate TDecimate mapping validation. Short or irregular runs remain keep;
# decimation requires a persistent block, a compatible drop density, and
# locally verified duplicates on the matched branch.
MM_REDUNDANCY_MIN_RUN = 50
MM_REDUNDANCY_DROP_RATIO_MIN = 0.12
MM_REDUNDANCY_DROP_RATIO_MAX = 0.28
MM_REDUNDANCY_DROP_DIFF_MAX = 5

# TDecimate mode 5 uses the default five-frame IVTC cycle and removes one
# duplicate from every film cycle. Partial boundary cycles fall back to bob.
# Extracted Matroska source timestamps are normally quantized to milliseconds,
# so a complete NTSC cycle can differ from its exact rational duration by just
# under one millisecond.
MM_TDECIMATE_CYCLE = 5
MM_TDECIMATE_CYCLE_DROP = 1
PTS_DECIMATE_CYCLE_TOLERANCE_MS = 1.0

# Detection of interlaced vertical scrolling, typically white-on-black credits.
# The metric looks for consecutive fields that align much better after a
# one-field-line vertical shift, a signature of real inter-field motion that
# must be preserved at 59.94p.
MM_VERTICAL_SCROLL_ENABLED = True
MM_VERTICAL_SCROLL_DIRECT_MIN = 50
MM_VERTICAL_SCROLL_BEST_MAX = 60
MM_VERTICAL_SCROLL_IMPROVEMENT_MIN = 20
MM_VERTICAL_SCROLL_SOFT_DIRECT_MIN = 35
MM_VERTICAL_SCROLL_SOFT_BEST_MAX = 20
MM_VERTICAL_SCROLL_SOFT_IMPROVEMENT_MIN = 25
MM_VERTICAL_SCROLL_SHIFT = -1
MM_VERTICAL_SCROLL_WINDOW = 31
MM_VERTICAL_SCROLL_MIN_HITS = 8
MM_VERTICAL_SCROLL_MIN_RUN = 45

# Default dedup parameters for cadences containing duplicate frames.
MM_DEDUP_ENABLED = True
MM_DEDUP_THRESH = 5
MM_DEDUP_CAP = 4

# Layout of text reports printed to standard output.
REPORT_BUCKETS = 20
REPORT_RATE_REL_TOL = 0.02
REPORT_WINDOW_MS = 2000.0

# Source timeline validation. Durations are clustered by relative proximity;
# the shortest significant cluster represents the nominal duration of a
# two-field frame. Relative thresholds keep the logic independent of 29.97 fps
# and of title-specific timestamp values.
PTS_DURATION_CLUSTER_REL_TOL = 0.06
PTS_FIELD_QUANTIZATION_REL_TOL = 0.08
PTS_MIN_CLUSTER_SAMPLES = 8
PTS_MAX_FIELD_UNITS = 6
