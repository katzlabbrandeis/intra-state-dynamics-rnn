#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 12:24:54 2025

@author: vincentcalia-bogan

src/core/pre_processing/orchestrate_convolved_fr.py

"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from core.config.params_io import load_pipeline_params
from core.config.roots_io import resolve_roots
from core.io.standard_paths import ProjectPaths
from core.pre_processing.FRPipeline import FRPipeline_lite
from core.pre_processing.RNNLatentprocessing import RNNLatentProcessor
from core.utils.read_parquets import read_parquet_files_into_dict


def _fix_trial_time_order(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure column order matches what downstream expects:
      [neuron_* ... , taste, trial, time]
    Also swap trial/time if trial appears before time.
    """
    cols = list(df.columns)
    if "trial" in cols and "time" in cols:
        trial_idx = cols.index("trial")
        time_idx = cols.index("time")
        if trial_idx < time_idx:
            # swap their names (cheap way to reorder without reconstructing)
            new_cols = cols[:]
            new_cols[trial_idx], new_cols[time_idx] = "time", "trial"
            df = df.rename(dict(zip(cols, new_cols)))
    signal_cols = [c for c in df.columns if c.startswith("neuron_")]
    final_order = signal_cols + ["taste", "trial", "time"]
    # If any columns are missing (e.g., empty df), select will error; guard it:
    present = [c for c in final_order if c in df.columns]
    return df.select(present) if present else df


def _dir_has_parquets(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.parquet"))


def orchestrate_convolved_fr(repo_root: Path, *, pkl_root: Path | None = None,
                             force: bool = False, dry_run: bool = False) -> dict[str, Path]:
    """
    Stage-2 FR orchestrator:
      1) Reads params.json for FR + RNN latent settings and taste replacements
      2) Uses intermediate NPZ/INFO from standard paths
      3) Computes lite FR parquet files if missing
      4) Cleans column order (trial/time) into *_cleaned
      5) Runs RNNLatentProcessor on the cleaned FR (it supports neuron_* inputs)
      6) Saves all outputs under .../FR_PROCESSING_PARQUETS/rr_firing_uw_lite_outputs/

    Returns a dict of key directories used.
    """
    params = load_pipeline_params(repo_root)
    paths = ProjectPaths.from_repo_root(repo_root)
    paths.ensure()

    roots = resolve_roots(
        repo_root,
        cli_pkl_root=pkl_root,      # may be None; will pick stored or cache
        pkl_cache_dir=paths.pkl_dir,
        require_h5=False,
    )
    pkl_source = roots.pkl_root    # either RAW PKL or cache dir

    # ---- Folder layout (your original names)
    fr_base = paths.intermediate_dir / "FR_PROCESSING_PARQUETS"
    input_name = "rr_firing_uw_lite"
    cleaned_name = f"{input_name}_cleaned"
    save_name = f"{input_name}_outputs"  # the files we will want to use

    rr_input_dir = fr_base / input_name
    rr_cleaned_dir = fr_base / cleaned_name
    rr_save_dir = fr_base / save_name
    rr_cleaned_dir.mkdir(parents=True, exist_ok=True)
    rr_save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Detect whether RNN outputs already exist (per your new save structure from RNNLatentProcessor)
    var_thr = int(params.get("rnn_latent", {}).get("variance_threshold", 95))
    existing = any([
        _dir_has_parquets(rr_save_dir / "raw_output_unwarped"),
        _dir_has_parquets(rr_save_dir / f"pca_{var_thr}_pct_var_unwarped"),
        _dir_has_parquets(rr_save_dir / "pca_full_var_unwarped"),
    ])

    if existing and not force:
        print("[RR FR] Output .parquet files found — skipping recompute and using existing artifacts.")
        return {
            "input_dir": rr_input_dir,
            "cleaned_dir": rr_cleaned_dir,
            "save_dir": rr_save_dir,
        }

    # =========================
    # 1) Ensure lite FR parquets exist (if rr_input_dir is empty)
    # =========================
    if not _dir_has_parquets(rr_input_dir):
        print("[RR FR] No FR input parquets found — running FRPipeline_lite.")
        fr_cfg = params.get("fr_pipeline", {})
        # NOTE: your prior workflow added ~200ms padding to end_time to align bins.
        end_time = int(fr_cfg.get("end_time", 4500))
        end_time += 200  # replicate your manual hack for window boundary alignment

        if dry_run:
            print(f"[dry-run] Would compute FR lite → {rr_input_dir}")
        else:
            pipe = FRPipeline_lite(
                npz_path=str(paths.npz_dir),
                info_path=str(paths.info_dir),
                taste_replacements=params.get("taste_replacements", {}),
                output_dir=str(fr_base),
                window_length=int(fr_cfg.get("window_length", 250)),
                step_size=int(fr_cfg.get("step_size", 25)),
                start_time=int(fr_cfg.get("start_time", 1500)),
                end_time=end_time,
                save_outputs=True,
            )
            pipe.full_pipeline()

    # =========================
    # 2) Clean column order into *_cleaned
    # =========================
    print("[RR FR] Cleaning column order (trial/time) → *_cleaned …")
    pred_fr = read_parquet_files_into_dict(rr_input_dir)
    if dry_run:
        print(f"[dry-run] Would rewrite {len(pred_fr)} parquet(s) into {rr_cleaned_dir}")
    else:
        for key, df in pred_fr.items():
            fixed = _fix_trial_time_order(df)
            fixed.write_parquet(rr_cleaned_dir / f"{key}.parquet")  # below line could have been buggeD?
           # (rr_cleaned_dir / f"{key}.parquet").write_bytes(fixed.write_parquet())

    # =========================
    # 3) Run RNNLatentProcessor on cleaned FR (neuron_* supported)
    # =========================
    print("[RR FR] Running RNNLatentProcessor on cleaned FR …")
    rnn_cfg = params.get("rnn_latent", {})  # all the params
    if dry_run:
        print(
            f"[dry-run] Would run RNNLatentProcessor(parquet_dir={rr_cleaned_dir}, "
            f"npz_path={paths.npz_dir}, info_path={paths.info_dir}, pkl_path=<RAW PKL from stage-1>, "
            f"save_dir={rr_save_dir}, bin_size_ms={rnn_cfg.get('bin_size_ms', 25)}, "
            f"start_time_ms={rnn_cfg.get('start_time_ms', 1500)}, max_time_ms={rnn_cfg.get('max_time_ms', 4500)}, "
            f"warp_length={rnn_cfg.get('warp_length', 1000)}, variance_threshold={rnn_cfg.get('variance_threshold', 95.0)})"
        )
    else:
        proc = RNNLatentProcessor(
            parquet_dir=str(rr_cleaned_dir),
            npz_path=str(paths.npz_dir),
            info_path=str(paths.info_dir),
            # RAW PKL remains authoritative; we keep location under intermediate/pkl_cache as a cache,
            # but you should point this to the raw PKL root set in stage-1 (h5 pipeline).
            # If you prefer, add that path into params.json and read here.
            pkl_path=str(pkl_source),  # fallback to cache; swap to RAW PKL root if desired
            taste_replacements=params.get("taste_replacements", {}),
            save_dir=str(rr_save_dir),
            bin_size_ms=int(rnn_cfg.get("bin_size_ms", 25)),
            start_time_ms=int(rnn_cfg.get("start_time_ms", 1500)),
            max_time_ms=int(rnn_cfg.get("max_time_ms", 4500)),
            warp_length=int(rnn_cfg.get("warp_length", 1000)),
            variance_threshold=float(rnn_cfg.get("variance_threshold", 95.0)),
        )
        proc.full_pipeline(
            compute_first_derivative=True,
            compute_second_derivative=True,
            derivative_source=str(rnn_cfg.get("derivative_source", "threshold")),
            return_derivatives=True,
            save_outputs=True,
        )
        # persist CP pickles (for convenience)
        proc.extract_changepoints_dict(save_outputs=True)

    print(f"[RR FR] Done. Outputs in: {rr_save_dir}")
    return {
        "input_dir": rr_input_dir,
        "cleaned_dir": rr_cleaned_dir,
        "save_dir": rr_save_dir,
    }
