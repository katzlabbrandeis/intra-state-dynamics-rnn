#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:42:00 2025

@author: vincentcalia-bogan

src/core/pre_processing/orchestrate_rnn_latents.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl  # not actually imported but will definitely be useful in the future

from core.config.params_io import load_pipeline_params
from core.config.roots_io import resolve_roots
from core.io.standard_paths import ProjectPaths
from core.pre_processing.RNNLatentprocessing import RNNLatentProcessor


# simple helper
def _dir_has_parquets(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.parquet"))


def orchestrate_rnn_latents(
    repo_root: Path,
    *,
    latent_parquet_root: Path | None = None,  # for the rnn latents
    pkl_root: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Path]:
    """
    Stage-2: RNN latent orchestration
      • Reads RNN latent Parquet input (precomputed model outputs with latent_dim_* columns)
      • Uses RAW PKL (authoritative) for changepoints; falls back to cache if unset
      • Runs RNNLatentProcessor end-to-end and persists outputs under:
          output/intermediate_data/RNN_PROCESSING_PARQUETS/latent_outputs/

    Returns a dict of key directories used:
      { "input_dir": ..., "save_dir": ... }
    """
    params = load_pipeline_params(repo_root)
    paths = ProjectPaths.from_repo_root(repo_root)
    paths.ensure()

    # Resolve all roots, remembering them in roots.json (and reusing on subsequent runs)
    roots = resolve_roots(
        repo_root=repo_root,
        cli_rnn_latent_root=latent_parquet_root,  # corrected name
        cli_pkl_root=pkl_root,
        pkl_cache_dir=paths.pkl_dir,
        require_h5=False,
        require_rnn_latent=True,     # <- we do need this
    )
    # quick error handling:
    lat_root = roots.latent_parquet_root       # <<< correct attribute
    if lat_root is None:
        raise RuntimeError(
            "RNN latent Parquet root is not configured. Provide "
            "--rnn-latent-parquet-root once (it will be remembered in roots.json)."
        )

    rnn_input_dir = roots.latent_parquet_root  # fixed, rnn root
    pkl_source = roots.pkl_root    # RAW PKL if set; else cache dir

    # Output layout for this step
    base = paths.intermediate_dir / "RNN_PROCESSING_PARQUETS"
    save = base / "latent_outputs"
    save.mkdir(parents=True, exist_ok=True)

    # Detect if results already exist (avoid recompute unless --force)
    var_thr = int(params.get("rnn_latent", {}).get("variance_threshold", 95))
    already = any([
        _dir_has_parquets(save / "raw_output_unwarped"),
        _dir_has_parquets(save / f"robust_pca_{var_thr}_unwarped"),
        _dir_has_parquets(save / "robust_pca_full_unwarped"),
    ])

    if already and not force:
        print("[RNN] Existing latent outputs detected — reusing (use --force to recompute).")
        return {"input_dir": rnn_input_dir, "save_dir": save}

    rnn_cfg = params.get("rnn_latent", {})

    if dry_run:
        print(
            f"[dry-run] Would run RNNLatentProcessor("
            f"parquet_dir={rnn_input_dir}, "
            f"npz_path={paths.npz_dir}, info_path={paths.info_dir}, pkl_path={pkl_source}, "
            f"save_dir={save}, bin_size_ms={rnn_cfg.get('bin_size_ms', 25)}, "
            f"start_time_ms={rnn_cfg.get('start_time_ms', 1500)}, "
            f"max_time_ms={rnn_cfg.get('max_time_ms', 4500)}, "
            f"warp_length={rnn_cfg.get('warp_length', 1000)}, "
            f"variance_threshold={rnn_cfg.get('variance_threshold', 95.0)})"
        )
        return {"input_dir": rnn_input_dir, "save_dir": save}

    # --- run the processor
    proc = RNNLatentProcessor(
        parquet_dir=str(rnn_input_dir),
        npz_path=str(paths.npz_dir),
        info_path=str(paths.info_dir),
        pkl_path=str(pkl_source),  # RAW PKL authoritative; cache fallback
        taste_replacements=params.get("taste_replacements", {}),
        save_dir=str(save),
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
    # Also persist per-dataset CP pickles in cache (convenience; RAW PKL remains source of truth)
    proc.extract_changepoints_dict(save_outputs=True)

    print(f"[RNN] Done. Outputs in: {save}")
    return {"input_dir": rnn_input_dir, "save_dir": save}
