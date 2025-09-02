#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:52:14 2025

# src/core/pre_processing/orchestrate_h5_input.py


@author: vincentcalia-bogan
"""

from __future__ import annotations

import pickle
from pathlib import Path

from core.config.preprocess import PreprocConfig
from core.config.roots_io import resolve_roots
from core.io.standard_paths import ProjectPaths
from core.utils.extract_npz import extract_from_npz
from core.utils.find_extract_info import find_copy_h5info, modify_tastes, process_info_files

# your existing utils (already commented earlier)
from core.utils.spike_train_to_npz import extract_to_npz, find_h5_files
from core.utils.unpkl_generator import unpickle_changepoints  # or extract_valid_changepoints


def _has_ext(directory: Path, suffix: str) -> bool:
    """Return True if directory contains at least one file with given suffix."""
    return any(p.suffix == suffix for p in directory.iterdir() if p.is_file())


def orchestrate_h5_input(cfg: PreprocConfig, *, force: bool = False, dry_run: bool = False,
                         cache_changepoints: bool = True) -> None:
    # by default, cachce changepoints
    """
    Stage A (spike trains): ensure .npz exist under <output>/intermediate_data/spike_trains_npz
    Stage B (info files):   ensure .info copies exist under <output>/intermediate_data/info_files
    Stage C: for each .npz dataset, read changepoints from RAW PKL (authoritative);
             optionally cache per-dataset changepoints under output/intermediate_data/pkl_cache.
    """
    paths: ProjectPaths = cfg.paths
    paths.ensure()

    # Persist/resolve roots once; precedence: CLI (cfg.*_root) > stored roots.json > (PKL only) cache fallback
    h5_root = cfg.h5_root              # guaranteed non-None here
    pkl_root = cfg.pkl_root or paths.pkl_dir           # RAW PKL if given, else cache dir

    # ---------- Stage A: NPZ (spike trains)
    if force or not _has_ext(paths.npz_dir, ".npz"):
        h5_files = find_h5_files(h5_root)
        if dry_run:
            print(f"[DRY] would extract {len(h5_files)} h5 files to {paths.npz_dir}")
        else:
            print(f"[A] extracting {len(h5_files)} h5 files -> {paths.npz_dir}")
            extract_to_npz(h5_files, h5_root, cfg.spike_trains_path, paths.npz_dir)
    else:
        print(f"[A] .npz already present in {paths.npz_dir}, skipping (use --force to re-run)")

    # ---------- Stage B: INFO
    if force or not _has_ext(paths.info_dir, ".info"):
        if dry_run:
            print(f"[DRY] would copy .info from {h5_root} -> {paths.info_dir}")
        else:
            infos = find_copy_h5info(h5_root, paths.info_dir)
            print(f"[B] copied {len(infos)} .info files -> {paths.info_dir}")
    else:
        print(f"[B] .info already present in {paths.info_dir}, skipping (use --force to re-run)")

    # --- Stage C: changepoints via RAW PKL (authoritative)
    if dry_run:
        print(f"[DRY] would scan {paths.npz_dir} and unpickle using RAW PKL at {pkl_root}")
        return

    npz_count = 0
    for tup in extract_from_npz(paths.npz_dir):
        if not (isinstance(tup, tuple) and len(tup) == 4):
            continue
        spike_array, dataset_num, index, key = tup
        npz_count += 1

        # Optional taste normalization
        try:
            dataset_tastes = process_info_files(paths.info_dir, dataset_num)
            _ = modify_tastes(dataset_tastes, cfg.taste_replacements or {})
        except Exception as e:
            print(f"[C] warn: process_info_files failed for {dataset_num}: {e}")

        # Read from RAW PKL (authoritative)
        try:
            extracted = unpickle_changepoints(pkl_root, [(spike_array, dataset_num, index, key)])
            if extracted is not None and extracted.size > 0 and cache_changepoints:
                # Cache only the changepoints slice (column 3) per legacy convention
                try:
                    changepoints = extracted[:, 3]
                except Exception:
                    # fallback if extracted is a list-like rather than 2D array
                    changepoints = [row[3] for row in extracted]

                clean_name = Path(str(dataset_num)).stem  # npz filename without extension
                out_path = paths.pkl_dir / f"{clean_name}_changepoints.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump(changepoints, f)
                print(f"[C] cached changepoints -> {out_path}")
        except Exception as e:
            print(f"[C] warn: unpickle_changepoints failed for {dataset_num}: {e}")

    print(f"[C] processed changepoints for {npz_count} dataset(s) from RAW PKL at {pkl_root}")
