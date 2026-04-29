#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrypoint da linea di comando per la pipeline anime_vfr."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import main


if __name__ == "__main__":
    main()
