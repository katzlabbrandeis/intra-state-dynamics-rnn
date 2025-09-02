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
from core.pre_processing.orchestrate_convolved_fr import orchestrate_convolved_fr
from core.pre_processing.orchestrate_h5_input import orchestrate_h5_input
from core.pre_processing.orchestrate_pred_fr import orchestrate_pred_fr
from core.pre_processing.orchestrate_rnn_latents import orchestrate_rnn_latents


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

    # Accept BOTH spellings; both map to the same dest
    p_first.add_argument("--h5-root", "--h5_root", dest="h5_root",
                         help="H5 root (remembered in roots.json). Required on first run.")
    p_first.add_argument("--pkl-root", "--pkl_root", dest="pkl_root",
                         help="RAW PKL root (remembered). If omitted, uses stored value or falls back to cache.")
    p_first.add_argument("--rnn-latent-parquet-root", "--rnn_latent_parquet_root",
                         dest="rnn_latent_parquet_root",
                         help="Root of precomputed RNN latent Parquets (remembered).")
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

    # --- FR step
    p2 = sub.add_parser(
        "preprocess_intermediate_fr",
        help="Compute FR parquet (lite if needed), clean columns, and run latent processor on FR.",
        formatter_class=_HelpFmt,
    )
    p2.add_argument("--pkl-root", "--pkl_root", dest="pkl_root",
                    help="RAW PKL directory (authoritative). If omitted, uses stored roots.json or cache.")
    p2.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    p2.add_argument("--dry-run", action="store_true", help="Describe actions; do not write")

    # -- RNN Latents step
    p3 = sub.add_parser(
        "preprocess_rnn_latents",
        help="Run RNNLatentProcessor on precomputed RNN latent Parquets.",
        formatter_class=_HelpFmt,
    )
    p3.add_argument("--rnn-latent-parquet-root", "--rnn_latent_parquet_root",
                    dest="latent_parquet_root",
                    help="Root of RNN latent Parquets (remembered in roots.json).")
    p3.add_argument("--pkl-root", "--pkl_root", dest="pkl_root",
                    help="RAW PKL directory (authoritative). If omitted, uses stored roots.json or cache.")
    p3.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    p3.add_argument("--dry-run", action="store_true", help="Describe actions; do not write")

    # -- RNN predicted FR step
    p4 = sub.add_parser(
        "preprocess_pred_fr",
        help="Process predicted FR Parquets: clean schema & run RNNLatentProcessor.",
        formatter_class=_HelpFmt,
        epilog=(
            "Inputs:\n"
            "  --rnn-pred-fr-parquet-root  (remembered in roots.json)\n"
            "  --pkl-root              RAW PKL dir (authoritative; falls back to cache if omitted)\n\n"
            "Outputs under output/intermediate_data/:\n"
            "  pred_fr_clean/          # cleaned Parquets\n"
            "  PRED_FR_RNN/            # processor outputs (PCA, derivatives, warped/unwarped, changepoints)\n"
        ),
    )
    p4.add_argument("--rnn-pred-fr-parquet-root", "--rnn_pred_fr_parquet_root",
                    dest="rnn_pred_fr_parquet_root",
                    help="Root of predicted FR Parquets (remembered).")
    p4.add_argument("--pkl-root", "--pkl_root", dest="pkl_root",
                    help="RAW PKL directory (authoritative). If omitted, uses stored roots.json or cache.")
    p4.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    p4.add_argument("--dry-run", action="store_true", help="Describe actions; do not write")

    args = parser.parse_args(argv)

    # --- Dispatch
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "preprocess_dry":
        print("[bid] preprocess_dry: OK — CLI & imports are wired correctly.")
        return 0

    if args.cmd == "preprocess_intermediate_fr":
        repo_root = Path(__file__).resolve().parents[3]
        cfg = PreprocConfig.from_cli(args, repo_root, require_h5=False)
        orchestrate_convolved_fr(repo_root, pkl_root=cfg.pkl_root, force=args.force, dry_run=args.dry_run)
        return 0

    if args.cmd == "preprocess_h5_input":
        # Default pipeline is 'first-input' if omitted
        if args.pipeline == "first-input":
            repo_root = Path(__file__).resolve().parents[3]  # .../src/core/scripts -> repo root
            cfg = PreprocConfig.from_cli(args, repo_root, require_h5=True)
            orchestrate_h5_input(
                cfg,
                force=args.force,
                dry_run=args.dry_run,
                cache_changepoints=getattr(args, "cache_changepoints", True),
            )
            return 0
    # -- dispatch for the RNN latents
    if args.cmd == "preprocess_rnn_latents":
        repo_root = Path(__file__).resolve().parents[3]
        cfg = PreprocConfig.from_cli(repo_root=repo_root, args=args, require_h5=False)
        orchestrate_rnn_latents(
            repo_root,
            latent_parquet_root=getattr(args, "latent_parquet_root", None),  # resolved or remembered
            pkl_root=cfg.pkl_root,                          # resolved or cache fallback
            force=args.force,
            dry_run=args.dry_run,
        )
        return 0
    # -- Dispatch for the predicted firing rate by the RNN:
    if args.cmd == "preprocess_pred_fr":
        repo_root = Path(__file__).resolve().parents[3]
        orchestrate_pred_fr(
            repo_root,
            rnn_pred_fr_root=getattr(args, "rnn_pred_fr_parquet_root", None),
            pkl_root=getattr(args, "pkl_root", None),
            force=args.force,
            dry_run=args.dry_run,
        )
        return 0

    # Fallback: show help for the chosen command
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
