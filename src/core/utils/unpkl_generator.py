#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:17:38 2024

@author: vincentcalia-bogan

unpkl_generator.py
==================

Changepoint I/O compatibility layer (raw-PKL + per-dataset cache)

This module makes the downstream code (which expects a very specific, legacy
shape) work with TWO different sources of changepoints:

1) RAW PKL DATAFRAME(S)  (e.g., tau_frame.pkl)
   - Each pickle is a pandas DataFrame; each relevant row has an array in
     column 3 describing per-epoch changepoints per trial.
   - We NORMALIZE those rows so that each epoch entry is coerced to 4 values:
       [τ1, τ2, τ3, τ3 + 2000]
     If a source epoch has fewer than 3 values, we pad with NaN before
     adding the 4th (τ3 + 2000). Because of NaN padding, these arrays are
     necessarily FLOAT (Float64).
   - Output contract for the raw path:
       an object-dtype ndarray shaped (N_rows, 4) where ONLY column 3 holds
       the (R x 4) per-trial array for that dataset/epoch block.

2) PER-DATASET CACHED FILES  (e.g., AM35_..._changepoints.pkl)
   - Each file holds the changepoints for a *single* dataset, typically as
     an ndarray of pure integers either shaped:
       (T, R, 4)   OR   object array of length T with each element (R x 4)
     where T = tastes, R = trials.
   - We WRAP this into the same legacy structure the downstream expects:
       an object-dtype ndarray shaped (T, 4) where ONLY column 3 holds the
       (R x 4) array for that taste.
   - **Dtype policy for cached files:** we cast the (R x 4) blocks to Int64
     whenever possible (i.e., when no NaNs are present). This prevents later
     Polars concat failures due to mixed Float64/Int64 schemas in time-like
     columns. If NaNs are detected (rare in cache), we leave as float.

LEGACY CONTRACT (why this shape exists):
---------------------------------------
Downstream code takes the result and does:   extracted[:, 3]
i.e., it expects a 2D object array with the usable payload living entirely
in column 3. We keep that contract for both raw and cached inputs so the rest
of the pipeline does not need to change.

Name matching:
--------------
We map incoming dataset identifiers to cache/rows via two helpers:
- clean_dataset_key(name): drop trailing "_repacked" and file extension to
  get a clean dataset key for cache filenames.
- truncate_name(name): keep only underscore-separated parts that contain at
  least one digit; used to robustly match raw DataFrame basenames against
  dataset IDs.

Loader choice & fallbacks:
--------------------------
- If the provided path is a directory and contains "*_changepoints.pkl"
  files, we prefer PER-DATASET CACHE (fast path).
- If cache for a dataset is missing/unreadable, we FALL BACK to scanning
  RAW PKL DATAFRAME(S) in that directory and normalizing rows (float arrays
  with potential NaNs).
- When reading cache files we use `pandas.read_pickle` to support both
  numpy arrays and pandas pickles seamlessly.

Shapes at a glance:
-------------------
Cached file (common):
    cp_cached:         (T, R, 4) int  or  object length T with (R x 4) int
    wrapped_for_legacy: (T, 4) object where wrapped[:, 3] is (R x 4)

Raw PKL after normalization:
    extracted_raw:     (N_rows, 4) object where extracted_raw[:, 3] is (R x 4) float
                       (float due to NaN padding for short epochs)

The goal is to be completely transparent to callers: both sources return a
2D object array that you can slice as `arr[:, 3]` to obtain per-taste/per-row
(R x 4) changepoint matrices for warping/epoching, while avoiding dtype
mismatches down the line.

"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from core.utils.cp_io import clean_dataset_key  # canonical helper for cleaning

# further patching to help with the datatype of the cached changepoints
# anon helpers for this and this only:
# note: must be consistent with dtype (use only float?) to avoid issues with polars concatenation down the line


def _truncate_name(name: str) -> str:
    """Keep underscore-separated parts containing digits (legacy matcher)."""
    parts = str(name).split("_")
    keep = [p for p in parts if any(ch.isdigit() for ch in p)]
    return "_".join(keep) if keep else str(name)
# save these as int64 rather than f64 for compatibility reasons down the line...


def _as_int_if_possible(a):
    """
    Best-effort convert to Int64 when no NaN present; otherwise return as-is.
    We only use this for the cached-CP path, which is all integers in your examples.
    """
    arr = np.asarray(a)
    # If it's floating but contains NaNs, keep as float (cannot cast NaN -> int).
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            return arr
        # no NaN → safe to cast
        return arr.astype(np.int64, copy=False)
    # Already integer or object-of-integers → try to cast cleanly
    try:
        return arr.astype(np.int64, copy=False)
    except Exception:
        return arr


def _wrap_cached_cp_for_legacy(cp_obj) -> np.ndarray:
    """
    Cached per-dataset file shape (typical):
        object array length T (tastes), each element is (R x 4) ndarray.

    We must return a 2D object array (T x 4) with only column 3 populated so
    that downstream `extracted_pkl[:, 3]` yields a length-T array of (R x 4).
    """
    # Normalize to object array
    cp = np.asarray(cp_obj, dtype=object)

    # Case A: cp is already (T, R, 4) numeric — pack to object by taste
    if cp.ndim == 3 and cp.shape[-1] == 4:
        T = cp.shape[0]
        out = np.empty((T, 4), dtype=object)
        out[:, :3] = None
        for t in range(T):
            out[t, 3] = _as_int_if_possible(cp[t])
        return out

    # Case B: cp is object array (T,) each entry (R x 4)
    if cp.ndim == 1 and all(isinstance(x, (np.ndarray, list)) for x in cp):
        T = len(cp)
        out = np.empty((T, 4), dtype=object)
        out[:, :3] = None
        for t in range(T):
            out[t, 3] = _as_int_if_possible(cp[t])
        return out

    # Fallback: single (R x 4) — treat as 1 taste
   # cp2 = np.array(cp, dtype=float)
    out = np.empty((1, 4), dtype=object)
    out[:, :3] = None
    out[0, 3] = _as_int_if_possible(cp)
    return out


def _try_load_cached_dataset_cp(cache_dir: Path, dataset_num: str) -> np.ndarray | None:
    """
    If `cache_dir` holds per-dataset cache files ('*_changepoints.pkl'),
    read the matching one with pandas (so it can be ndarray/dict/whatever),
    and wrap to legacy 2D object array where col 3 holds (R x 4) per taste.
    """
    if not cache_dir.is_dir():
        return None
    if not any(cache_dir.glob("*_changepoints.pkl")):
        return None

    clean_key = clean_dataset_key(dataset_num)
    preferred = cache_dir / f"{clean_key}_changepoints.pkl"
    wanted_trunc = _truncate_name(dataset_num)

    # Prefer exact match; else try truncated match on any cache file
    candidates = [preferred] if preferred.exists() else list(cache_dir.glob("*_changepoints.pkl"))
    for fp in candidates:
        key = fp.stem.rsplit("_changepoints", 1)[0]
        if fp is preferred or _truncate_name(key) == wanted_trunc:
            try:
                obj = pd.read_pickle(fp)  # <-- critical: pandas loader
                return _wrap_cached_cp_for_legacy(obj)
            except Exception as e:
                print(f"[cp cache] Failed to read {fp}: {e!r}")
                continue
    return None


def _row_basename(row) -> str:
    """Prefer a 'basename' column; else fall back to the first column."""
    if isinstance(row, dict) or hasattr(row, "__getitem__"):
        if "basename" in row:
            return str(row["basename"])
    # pandas Series: .iloc[0] is fine
    try:
        return str(row.iloc[0])
    except Exception:
        return str(row[0])  # last-ditch fallback


def _row_epoch_list(row, df_columns) -> list | np.ndarray | None:
    """
    Find the per-trial epoch list for a row:
      - prefer 'tau' or 'tau_std' column if present
      - else use legacy `row.values[3]`
    """
    if "tau" in df_columns and isinstance(row["tau"], (list, np.ndarray)):
        return row["tau"]
    if "tau_std" in df_columns and isinstance(row["tau_std"], (list, np.ndarray)):
        return row["tau_std"]
    vals = getattr(row, "values", None)
    if vals is not None and len(vals) > 3 and isinstance(vals[3], (list, np.ndarray)):
        return vals[3]
    return None


def extract_valid_changepoints(
    pkl_path: str | os.PathLike,
    spike_array: np.ndarray,
    dataset_num: str,
    index: int,
    key: str,
) -> Optional[np.ndarray]:
    """
    Single-dataset resolver.
    - First try per-dataset cache in `pkl_path` (if that directory contains *_changepoints.pkl).
    - Otherwise scan raw PKL DataFrames in `pkl_path` and normalize to legacy format.

    Returns: 2D object array (T x 4) where col 3 contains (R x 4) arrays (one per taste).
    """
    pdir = Path(pkl_path).expanduser().resolve()

    # Fast path: use cache if present
    try:
        cached = _try_load_cached_dataset_cp(pdir, dataset_num)
    except Exception:
        cached = None
    if cached is not None:
        return cached  # already legacy-shaped

    # Legacy raw-PKL scan
    target = _truncate_name(dataset_num)
    raw_rows = []

    try:
        for fp in sorted(pdir.glob("*.pkl"), key=lambda x: x.name.lower()):
            try:
                obj = pd.read_pickle(fp)  # could be DF (raw) or ndarray (cache)
            except Exception as e:
                print(f"Error reading {fp.name}: {e!r}")
                continue

            # Skip non-DataFrames during raw scan (these are likely cached arrays)
            if not hasattr(obj, "iterrows"):
                continue

            cols = list(obj.columns)
            for _, row in obj.iterrows():
                base = os.path.basename(_row_basename(row))
                if _truncate_name(base) != target:
                    continue
                epoch_list = _row_epoch_list(row, cols)
                if not isinstance(epoch_list, (list, np.ndarray)):
                    continue
                # normalize each per-trial entry to 4 values (add 4th=third+2000; pad NaNs if <3)
                new_sub = []
                for entry in epoch_list:
                    if isinstance(entry, (list, np.ndarray)):
                        e = list(entry)
                        if len(e) < 3:
                            e += [np.nan] * (3 - len(e))
                        e4 = e[:3] + [e[2] + 2000]
                        new_sub.append(np.array(e4))
                # store into row-style object where column 3 is the (R x 4)
                new_vals = [None, None, None, np.array(new_sub)]
                raw_rows.append(np.array(new_vals, dtype=object))

        if not raw_rows:
            print(f"Skipping dataset {target} due to invalid data structures.")
            return None

        # Turn into legacy 2D array (T tastes x 4 columns; only col 3 used)
        out = np.empty((len(raw_rows), 4), dtype=object)
        out[:, :3] = None
        for i, r in enumerate(raw_rows):
            out[i, 3] = r[3]
        print(f"Finished processing valid dataset {target}.")
        return out

    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# new type that deals with how we cache pkl
def unpickle_changepoints(pkl_path, generator):
    """
    Aggregate normalized CPs across many datasets.

    Accepts either:
      - RAW PKL directory (DataFrames) — we scan/normalize rows, or
      - Cache directory with per-dataset '*_changepoints.pkl' — we load & wrap.

    Returns: 2D object array (N x 4) where col 3 holds (R x 4) arrays (grouped across tastes).
    """
    pdir = Path(pkl_path).expanduser().resolve()
    all_rows: list[np.ndarray] = []

    try:
        print("Starting to process pkl files.")
        for spike_array, dataset_num, index, key in generator:
            # Cache first
            try:
                cached = _try_load_cached_dataset_cp(pdir, dataset_num)
            except Exception:
                cached = None
            if cached is not None:
                # append one row per taste (each taste goes into a separate row)
                for t_idx in range(cached.shape[0]):
                    row = np.array([None, None, None, cached[t_idx, 3]], dtype=object)
                    all_rows.append(row)
                print(f"[unpickle_changepoints] used cache for {clean_dataset_key(dataset_num)}")
                continue

            # Raw scan
            target = _truncate_name(dataset_num)
            matched_rows = []

            for fp in sorted(pdir.glob("*.pkl"), key=lambda x: x.name.lower()):
                try:
                    obj = pd.read_pickle(fp)
                except Exception as e:
                    print(f"Error reading {fp.name}: {e!r}")
                    continue
                if not hasattr(obj, "iterrows"):
                    # cached array or non-DF content — skip in raw scan
                    continue

                cols = list(obj.columns)
                for _, row in obj.iterrows():
                    base = os.path.basename(_row_basename(row))
                    if _truncate_name(base) != target:
                        continue
                    epoch_list = _row_epoch_list(row, cols)
                    if not isinstance(epoch_list, (list, np.ndarray)):
                        continue
                    new_sub = []
                    for entry in epoch_list:
                        if isinstance(entry, (list, np.ndarray)):
                            e = list(entry)
                            if len(e) < 3:
                                e += [np.nan] * (3 - len(e))
                            e4 = e[:3] + [e[2] + 2000]
                            new_sub.append(np.array(e4))
                    matched_rows.append(np.array([None, None, None, np.array(new_sub)], dtype=object))

            if not matched_rows:
                print(f"Skipping dataset {target} due to invalid data structures.")
                continue

            all_rows.extend(matched_rows)
            print(f"Finished processing valid dataset {target}.")

        if not all_rows:
            return None

        out = np.empty((len(all_rows), 4), dtype=object)
        out[:, :3] = None
        for i, r in enumerate(all_rows):
            out[i, 3] = r[3]
        return out

    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
