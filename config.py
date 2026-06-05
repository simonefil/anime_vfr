# -*- coding: utf-8 -*-
"""Configurazione statica di binari esterni e parametri di analisi."""

MKVMERGE = "/opt/homebrew/bin/mkvmerge"
MKVEXTRACT = "/opt/homebrew/bin/mkvextract"
VSPIPE = "/opt/homebrew/bin/vspipe"
PYTHON_BIN = "/opt/homebrew/bin/python3.14"

MEDIAINFO = "/opt/homebrew/bin/mediainfo"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

ENCODER_BIN = r"C:\Users\Simone\Programs\nvencc\NVEncC64.exe"
ENCODER_PARAMS = (
    "--y4m -i - --codec hevc --preset p7 --tune uhq "
    "--lookahead 32 --bframes 5 --ref 6 --bref-mode middle "
    "--aq --aq-temporal --cqp 12:14:16 --output-depth 10"
)

# Parametri del classificatore multi-metrica. Definiscono il comportamento
# predefinito pubblico e vanno modificati solo confrontando campioni di regressione.
MM_FFT_HIGH = 0.025
MM_FFT_VERY_LOW = 0.002
MM_FFT_PROMOTION_MIN_MOTION = 5
MM_FFT_PROMOTION_ENABLED = True
MM_MATCH_THRESH = 5
MM_MIN_CLUSTER = 150
MM_ISOLATION_RATIO = 3.0
MM_TELECINE_NEIGHBOR_REQ = 6
MM_INHERITANCE_DENSITY_WIN = 100
MM_INHERITANCE_DOMINANCE = 1.5
MM_VERIFY_MIN_SIZE = 50
MM_VERIFY_COMBED_THRESH = 0.01
MM_VERIFY_MIN_MOTION = 5
MM_BOB_GAP_MAX = 30

# Riconoscimento di scroll verticali interlacciati, tipici dei credit a testo
# bianco su nero. La metrica cerca campi consecutivi che combaciano molto meglio
# con uno shift verticale di una riga-field: firma di movimento verticale reale
# tra field, da trattare a 59.94p.
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

# Parametri predefiniti del dedup per cadenze con frame duplicati.
MM_DEDUP_ENABLED = True
MM_DEDUP_THRESH = 5
MM_DEDUP_CAP = 4

# Layout dei report testuali stampati sullo standard output.
REPORT_BUCKETS = 20
