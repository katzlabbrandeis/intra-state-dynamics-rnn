#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:14:01 2025

@author: vincentcalia-bogan

run.py
======

Dev runner for your orchestration CLI *without installing*.
NOTE: Not for general usage. For running this pipeline regularly, consider using bootstrap.py

Usage:
  python run.py --help
  python run.py preprocess --dry-run
  # (anything you’d pass to core.scripts.orchestrate)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure <repo>/src is importable for THIS process.
SRC = Path(__file__).resolve().parent / "src"
if not (SRC / "core").exists():
    sys.stderr.write("run.py: expected src/core next to this file\n")
    raise SystemExit(2)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Dispatch to your CLI
from core.scripts.orchestrate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
