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
from pathlib import Path

from core.config.preprocess import PreprocConfig
from core.pre_processing.orchestrate_h5_input import orchestrate_h5_input


# Pretty, actionable help output
class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Pretty, actionable help output."""
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bid",
        description="Blech orchestration CLI",
        formatter_class=_HelpFmt,
        epilog=(
            "Examples:\n"
            "  # Quick import check (no data work)\n"
            "  bid preprocess_dry\n\n"
            "  # H5→NPZ, copy .info, read CPs from RAW PKL (dry run)\n"
            "  bid preprocess_h5_input first-input \\\n"
            "      --h5-root \"/Volumes/T7 Shield/spikesorting\" \\\n"
            "      --pkl-root \"/path/to/raw_pkl_dir\" \\\n"
            "      --dry-run\n\n"
            "  # Real run, re-run steps even if outputs exist; disable CP caching\n"
            "  bid preprocess_h5_input first-input \\\n"
            "      --h5-root \"/Volumes/T7 Shield/spikesorting\" \\\n"
            "      --pkl-root \"/path/to/raw_pkl_dir\" \\\n"
            "      --force --no-cache-changepoints\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- Smoke test: ensures imports & CLI wiring are OK
    p_pre = sub.add_parser(
        "preprocess_dry",
        help="Run to ensure that imports are working. No actual data will be pre-processed.",
        formatter_class=_HelpFmt,
    )

    # --- H5 input pipeline group
    p = sub.add_parser(
        "preprocess_h5_input",
        help="Pre-processing for H5 data: H5→NPZ, copy .info, read changepoints from RAW PKL.",
        formatter_class=_HelpFmt,
        epilog=(
            "Outputs:\n"
            "  output/intermediate_data/\n"
            "    ├─ spike_trains_npz/    # H5→NPZ artifacts\n"
            "    ├─ info_files/          # copied .info files\n"
            "    └─ pkl_cache/           # cached changepoints (may be disabled)\n"
        ),
    )
    # Make the pipeline subcommand OPTIONAL and default to 'first-input'
    p_sub = p.add_subparsers(dest="pipeline")
    p.set_defaults(pipeline="first-input")

    p_first = p_sub.add_parser(
        "first-input",
        help="Process H5→NPZ, copy .info, unpickle CPs (RAW PKL is authoritative).",
        formatter_class=_HelpFmt,
    )
    p_first.add_argument("--h5-root", required=True, help="Root directory containing .h5 trees")
    p_first.add_argument("--pkl-root", required=True, help="Directory containing RAW .pkl files")
    p_first.add_argument("--spike-trains-path", default="/spike_trains", help="HDF5 group path")
    p_first.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist")
    p_first.add_argument("--dry-run", action="store_true", help="Describe actions; do not write")

    # BooleanOptionalAction gives --cache-changepoints / --no-cache-changepoints
    try:
        BoolOpt = argparse.BooleanOptionalAction  # py>=3.9
    except AttributeError:
        BoolOpt = None

    if BoolOpt:
        p_first.add_argument(
            "--cache-changepoints", "--cache_changepoints",
            dest="cache_changepoints",
            action=BoolOpt,
            default=True,
            help="Cache per-dataset CPs under output/intermediate_data/pkl_cache/ (default: enabled).",
        )
    else:
        p_first.add_argument(
            "--cache-changepoints", "--cache_changepoints",
            dest="cache_changepoints",
            action="store_true",
            default=True,
            help="Cache per-dataset CPs under output/intermediate_data/pkl_cache/.",
        )

    args = parser.parse_args(argv)

    # --- Dispatch
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "preprocess_dry":
        print("[bid] preprocess_dry: OK — CLI & imports are wired correctly.")
        return 0

    if args.cmd == "preprocess_h5_input":
        # Default pipeline is 'first-input' if omitted
        if args.pipeline == "first-input":
            repo_root = Path(__file__).resolve().parents[3]  # .../src/core/scripts -> repo root
            cfg = PreprocConfig.from_cli(args, repo_root)
            orchestrate_h5_input(
                cfg,
                force=args.force,
                dry_run=args.dry_run,
                cache_changepoints=getattr(args, "cache_changepoints", True),
            )
            return 0

    # Fallback: show help for the chosen command
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
