#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:07:09 2025

src/core/scripts/orchestrate.py

bid: Blech Intra-state Dynamics

@author: vincentcalia-bogan
"""

# src/core/scripts/orchestrate.py
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bid", description="Blech orchestration CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_pre = sub.add_parser("preprocess", help="Run preprocessing pipelines")
    p_pre.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "preprocess":
        if args.dry_run:
            print("[bid] preprocess: dry run OK")
            return 0
        # TODO: call your pipeline(s)-- espeically for setting things up
        # TODO: really, with this orchestration, the idea is going to be setup. We'll worry about that in the future a bit.
        print("[bid] preprocess: running…")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
