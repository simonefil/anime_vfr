#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point for the anime_vfr pipeline."""

import sys
from pathlib import Path

# Vapourkit's isolated Python omits the script directory from sys.path.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import main


if __name__ == "__main__":
    main()
