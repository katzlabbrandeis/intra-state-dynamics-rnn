#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:25:43 2024

@author: vincentcalia-bogan

read_parquets.py
================

Helpers for loading per-timestep parquet files produced by upstream processes.

Functions
---------
- all_nrns_to_df(all_parquet_path)
    Eagerly read **all** *.parquet files in a directory and concatenate them into
    a single Polars DataFrame.

- read_parquet_files_into_dict(parquet_path)
    Eagerly read each *.parquet file into an individual Polars DataFrame and return
    a dict mapping <filename-without-extension> -> DataFrame.

Notes
-----
- Files are processed in **sorted** order for reproducibility.
- These are eager readers. For very large directories, consider Polars **lazy** API
  (e.g., `pl.scan_parquet(...)`) to defer execution until you filter/aggregate.

"""

from __future__ import annotations

import os
import os.path
from pathlib import Path
from typing import Dict

import polars as pl


def all_nrns_to_df(all_parquet_path: str | os.PathLike) -> pl.DataFrame:
    """
    Read every *.parquet file in a directory and concatenate into a single DataFrame.

    Parameters
    ----------
    all_parquet_path
        Directory containing parquet files.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame (empty if no parquet files were found).

    Behavior
    --------
    - Reads files in **sorted** order for stability.
    - Uses `rechunk=True` on concatenation to produce contiguous memory layout.
    """
    root = Path(all_parquet_path)
    if not root.exists():
        print(f"[read_parquets] path does not exist: {root}")
        return pl.DataFrame()
    if not root.is_dir():
        print(f"[read_parquets] not a directory: {root}")
        return pl.DataFrame()
    # Deterministic order helps reproducibility and testing.
    parquet_files = sorted(f for f in os.listdir(root) if f.endswith(".parquet"))

    frames: list[pl.DataFrame] = []
    for parquet_file in parquet_files:
        file_path = root / parquet_file
        try:
            df = pl.read_parquet(file_path)
            frames.append(df)
        except Exception as exc:
            # Skip unreadable files without crashing the whole load.
            print(f"[read_parquets] failed to read {file_path}: {exc!r}")
    if frames:
        all_nrns_df = pl.concat(frames, rechunk=True)
        return all_nrns_df
    else:
        return pl.DataFrame()


def read_parquet_files_into_dict(parquet_path: str | os.PathLike) -> Dict[str, pl.DataFrame]:
    """
    Reads all .parquet files in a directory into a dictionary.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the folder containing parquet files.

    Returns
    -------
    dict
        Dictionary where keys are filenames (no extension) and values are Polars DataFrames.
    """
    root = Path(parquet_path)
    if not root.exists():
        print(f"[read_parquets] path does not exist: {root}")
        return {}
    if not root.is_dir():
        print(f"[read_parquets] not a directory: {root}")
        return {}
    dataframes_dict = Dict[str, pl.DataFrame] = {}
    # `sorted` to ensure deterministic key insertion order.
    for file in sorted(root.glob("*.parquet")):  # using glob which is performant
        key = file.stem  # file.stem gives you filename without extension
        try:
            dataframes_dict[key] = pl.read_parquet(file)
        except Exception as exc:
            print(f"[read_parquets] failed to read {file}: {exc!r}")
            # skip this file but continue others
            continue
    return dataframes_dict
