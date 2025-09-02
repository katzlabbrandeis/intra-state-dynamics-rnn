#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 13:25:17 2025

@author: vincentcalia-bogan

# /scr/core/pre_processing/orchestrate_pred_fr.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import polars as pl

from core.config.params_io import load_pipeline_params
from core.config.roots_io import resolve_roots
from core.io.standard_paths import ProjectPaths
from core.pre_processing.RNNLatentprocessing import RNNLatentProcessor
from core.utils.read_parquets import read_parquet_files_into_dict


def _fix_trial_time_order_and_scale(
    df: pl.DataFrame,
    *,
    bin_size_ms: int = 25,
    scale_bin_to_hz: bool = True,
    clip_negative: bool = False,
) -> pl.DataFrame:
    """
    Ensure schema & order: [neuron_*..., taste, trial, time]
    - Optionally convert counts/bin → Hz by dividing by (bin_size_ms/1000).
    - Optionally clip negatives to 0.
    - Enforce dtypes: neuron_* -> Float64, taste/trial/time -> Int64
    """
    cols = list(df.columns)
    if "trial" in cols and "time" in cols:
        t_idx, tm_idx = cols.index("trial"), cols.index("time")
        if t_idx < tm_idx:  # swap names to reorder without reconstructing
            new_cols = cols[:]
            new_cols[t_idx], new_cols[tm_idx] = "time", "trial"
            df = df.rename(dict(zip(cols, new_cols)))

    signal = [c for c in df.columns if c.startswith("neuron_")]
    if not signal or not {"taste", "trial", "time"}.issubset(set(df.columns)):
        return df  # let caller decide; we won’t crash

    # cast meta columns to Int64
    df = df.with_columns([
        pl.col("taste").cast(pl.Int64, strict=False),
        pl.col("trial").cast(pl.Int64, strict=False),
        pl.col("time").cast(pl.Int64, strict=False),
    ])

    # cast neuron_* to Float64
    df = df.with_columns([pl.col(signal).cast(pl.Float64, strict=False)])

    if scale_bin_to_hz:
        sec = max(bin_size_ms / 1000.0, 1e-9)
        df = df.with_columns([(pl.col(c) / sec).alias(c) for c in signal])

    if clip_negative:
        df = df.with_columns([pl.col(c).clip(lower_bound=0, upper_bound=None).alias(c) for c in signal])

    final_order = signal + ["taste", "trial", "time"]
    present = [c for c in final_order if c in df.columns]
    return df.select(present)


def _dir_has_parquets(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.parquet"))


def orchestrate_pred_fr(
    repo_root: Path,
    *,
    rnn_pred_fr_root: str | None = None,
    pkl_root: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Path]:
    """
    Orchestrate predicted-FR processing:
      1) Read Parquets from pred_fr_root (remembered in roots.json)
      2) Clean column order/dtypes, optional Hz scaling/negative clipping → pred_fr_clean/
      3) Run RNNLatentProcessor (supports neuron_* schema) → PRED_FR_RNN/

    Returns dict of important directories.
    """
    params = load_pipeline_params(repo_root)
    paths = ProjectPaths.from_repo_root(repo_root)
    paths.ensure()

    roots = resolve_roots(
        repo_root=repo_root,
        cli_rnn_pred_fr_root=rnn_pred_fr_root,
        cli_pkl_root=pkl_root,
        pkl_cache_dir=paths.pkl_dir,
        require_h5=False,
    )
    if roots.rnn_pred_fr_parquet_root is None:
        raise RuntimeError(
            "[pred_fr] No predicted-FR parquet root set. Provide --pred-fr-parquet-root or set it in roots.json"
        )

    # Folder layout under /output/intermediate_data
    base = paths.intermediate_dir
    fr_clean_dir = base / "pred_fr_clean"
    fr_save_dir = base / "PRED_FR_RNN"
    fr_clean_dir.mkdir(parents=True, exist_ok=True)
    fr_save_dir.mkdir(parents=True, exist_ok=True)

    # Already processed?
    var_thr = int(params.get("rnn_latent", {}).get("variance_threshold", 95))
    processed = any([
        _dir_has_parquets(fr_save_dir / "raw_output_unwarped"),
        _dir_has_parquets(fr_save_dir / f"robust_pca_{var_thr}_unwarped"),
        _dir_has_parquets(fr_save_dir / "robust_pca_full_unwarped"),
    ])
    if processed and not force:
        print("[Pred FR] Output .parquet files found — skipping recompute and using existing artifacts.")
        return {"cleaned_dir": fr_clean_dir, "save_dir": fr_save_dir}

    # 1) Load & clean
    pred_cfg = params.get("pred_fr", {})
    bin_ms = int(pred_cfg.get("bin_size_ms", params.get("rnn_latent", {}).get("bin_size_ms", 25)))
    scale_hz = bool(pred_cfg.get("scale_bin_to_hz", True))
    clip_neg = bool(pred_cfg.get("clip_negative", False))

    src_dir = roots.rnn_pred_fr_parquet_root
    pred_map = read_parquet_files_into_dict(src_dir)

    if dry_run:
        print(f"[dry-run] Would clean {len(pred_map)} parquet(s) from {src_dir} → {fr_clean_dir}")
    else:
        print("[Pred FR] Cleaning column order (trial/time), scaling, and dtypes → pred_fr_clean/ …")
        for key, df in pred_map.items():
            fixed = _fix_trial_time_order_and_scale(
                df,
                bin_size_ms=bin_ms,
                scale_bin_to_hz=scale_hz,
                clip_negative=clip_neg,
            )
            (fr_clean_dir / f"{key}.parquet").parent.mkdir(parents=True, exist_ok=True)
            fixed.write_parquet(fr_clean_dir / f"{key}.parquet")

    # 2) Run RNNLatentProcessor
    rnn_cfg = params.get("rnn_latent", {})
    if dry_run:
        print(
            f"[dry-run] Would run RNNLatentProcessor(parquet_dir={fr_clean_dir}, "
            f"npz_path={paths.npz_dir}, info_path={paths.info_dir}, pkl_path={roots.pkl_root or paths.pkl_dir}, "
            f"save_dir={fr_save_dir}, bin_size_ms={rnn_cfg.get('bin_size_ms', 25)}, "
            f"start_time_ms={rnn_cfg.get('start_time_ms', 1500)}, max_time_ms={rnn_cfg.get('max_time_ms', 4500)}, "
            f"warp_length={rnn_cfg.get('warp_length', 1000)}, variance_threshold={rnn_cfg.get('variance_threshold', 95.0)})"
        )
        return {"cleaned_dir": fr_clean_dir, "save_dir": fr_save_dir}

    print("[Pred FR] Running RNNLatentProcessor on cleaned predicted FR …")
    proc = RNNLatentProcessor(
        parquet_dir=str(fr_clean_dir),
        npz_path=str(paths.npz_dir),
        info_path=str(paths.info_dir),
        pkl_path=str(roots.pkl_root or paths.pkl_dir),
        taste_replacements=params.get("taste_replacements", {}),
        save_dir=str(fr_save_dir),
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
    proc.extract_changepoints_dict(save_outputs=True)

    print(f"[Pred FR] Done. Outputs in: {fr_save_dir}")
    return {"cleaned_dir": fr_clean_dir, "save_dir": fr_save_dir}
