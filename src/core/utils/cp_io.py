#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:37:06 2025

@author: vincentcalia-bogan

# src/core/utils/cp_io.py

This is a helper to deal with the fact that I'm doing two different types of changepoint pkl data storage
and it's deeply annoying-- as the raw pkl is one file, but the cached files are multiple different pkl files.
as such this is a fixer that will help with the multiple pkl files that are genearted.

tbh this should probably be the standard, but hey, here we are lol.

"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------- name helpers --------------------------

_DIGIT_PART = re.compile(r"\d")


def truncate_name(name: str) -> str:
    """
    Keep only underscore-separated parts that contain at least one digit and re-join.
    Matches your existing matching logic used against PKL rows.
    """
    parts = str(name).split("_")
    keep = [p for p in parts if _DIGIT_PART.search(p)]
    return "_".join(keep) if keep else str(name)


def clean_dataset_key(name: str) -> str:
    """
    Turn a dataset filename/path into a canonical key used for cache filenames.
    Example: 'AM35_4Tastes_201229_150307_repacked.npz' -> 'AM35_4Tastes_201229_150307'
    """
    stem = Path(str(name)).stem
    stem = re.sub(r"_repacked$", "", stem)
    return stem


# -------------------------- normalization --------------------------

def _normalize_cp_rows(raw_rows: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Given a list of np.ndarray 'row.values' where row[3] contains a per-epoch sequence,
    coerce each epoch-entry to length 4: first up to 3 original values (pad NaN if <3),
    + a fourth value equal to original third value + 2000.
    Returns an object-dtype array shape (n_rows,) with element [3] being a (trials x 4) array.
    """
    if not raw_rows:
        return None

    extracted = []
    for vals in raw_rows:
        # Expect vals[3] iterable of epoch rows
        cp_src = vals[3]
        if not isinstance(cp_src, (list, tuple, np.ndarray)):
            # malformed; skip row
            continue
        new_sub = []
        for entry in cp_src:
            if isinstance(entry, (list, np.ndarray)):
                entry = list(entry)
            else:
                # unexpected; skip this epoch
                continue

            # pad to 3
            if len(entry) < 3:
                entry = entry + [np.nan] * (3 - len(entry))
            # ensure we can index [2] for +2000
            if len(entry) >= 3:
                entry4 = entry[:3] + [entry[2] + 2000]
                new_sub.append(np.array(entry4))
        new_vals = list(vals)
        new_vals[3] = np.array(new_sub, dtype=float)
        extracted.append(np.array(new_vals, dtype=object))

    if not extracted:
        return None

    return np.array(extracted, dtype=object)


# -------------------------- loaders --------------------------

def _load_cached_per_dataset(cache_dir: Path, clean_key: str) -> Optional[np.ndarray]:
    """
    Look for <clean_key>_changepoints.pkl in cache_dir and return the loaded object if present.
    """
    p = cache_dir / f"{clean_key}_changepoints.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _scan_raw_dir_for_dataset(raw_dir: Path, dataset_num: str) -> Optional[np.ndarray]:
    """
    Search a directory of raw PKL DataFrames and extract normalized rows for a single dataset.
    """
    target = truncate_name(dataset_num)
    raw_rows: List[np.ndarray] = []

    for p in sorted(raw_dir.glob("*.pkl"), key=lambda x: x.name.lower()):
        try:
            df = pd.read_pickle(p)
        except Exception as e:
            print(f"[cp_io] Unable to read {p}: {e!r}")
            continue

        # Each row: col0 path-like, col3 array-like
        for _, row in df.iterrows():
            base = os.path.basename(str(row.iloc[0]))
            if truncate_name(base) == target:
                vals = row.values
                if isinstance(vals[3], np.ndarray):
                    raw_rows.append(vals)
                else:
                    # malformed row for this dataset; ignore
                    pass

    return _normalize_cp_rows(raw_rows)


def _scan_raw_file_group_by_dataset(raw_file: Path) -> Dict[str, np.ndarray]:
    """
    Read a single raw PKL DataFrame and produce a dict clean_key -> normalized CP array.
    """
    out: Dict[str, np.ndarray] = {}
    try:
        df = pd.read_pickle(raw_file)
    except Exception as e:
        print(f"[cp_io] Unable to read {raw_file}: {e!r}")
        return out

    # group rows by dataset (using truncated filename of col0)
    groups: Dict[str, List[np.ndarray]] = {}
    for _, row in df.iterrows():
        base = os.path.basename(str(row.iloc[0]))
        key = clean_dataset_key(base)  # nicer display/caching key
        groups.setdefault(key, []).append(row.values)

    for key, rows in groups.items():
        arr = _normalize_cp_rows(rows)
        if arr is not None:
            out[key] = arr[:, 3]  # keep column-3 convention
    return out


def resolve_changepoints_for_dataset(pkl_location: str | os.PathLike, dataset_num: str) -> Optional[np.ndarray]:
    """
    Unified resolver:
      - If pkl_location is a directory:
          * If a per-dataset cached file exists: use it.
          * Else scan raw PKL DataFrames in that dir and normalize rows.
      - If pkl_location is a file:
          * Read it as a raw PKL DataFrame and group; return the matching dataset if present.

    Returns np.ndarray (typically the column-3 array per your convention) or None.
    """
    p = Path(pkl_location).expanduser().resolve()
    clean_key = clean_dataset_key(dataset_num)

    if p.is_dir():
        cached = _load_cached_per_dataset(p, clean_key)
        if cached is not None:
            return cached
        # fall back to scanning raw DataFrames
        arr = _scan_raw_dir_for_dataset(p, dataset_num)
        if arr is not None:
            return arr[:, 3]  # follow your existing “take column 3” convention
        return None

    if p.is_file():
        # Single raw PKL (DataFrame) that holds all datasets
        grouped = _scan_raw_file_group_by_dataset(p)
        # try both “clean” and “truncated” forms
        if clean_key in grouped:
            return grouped[clean_key]
        trunc = truncate_name(dataset_num)
        # Some raw files may key closer to truncated; try to match by suffix
        for k in grouped:
            if truncate_name(k) == trunc:
                return grouped[k]
        return None

    print(f"[cp_io] PKL location not found: {p}")
    return None


def load_standardized_changepoints_flex(pkl_location: str | os.PathLike) -> Dict[str, Any]:
    """
    Flexible bulk loader:
      - If a directory: read all '*_changepoints.pkl' entries into a dict {clean_key: obj}
      - If a file:
          * If it unpickles to a dict[str, Any], return as-is.
          * Else if it is a raw DataFrame, group+normalize into the same dict form.

    Returned values match your downstream expectation: mapping clean_key -> per-dataset CP array.
    """
    p = Path(pkl_location).expanduser().resolve()
    result: Dict[str, Any] = {}

    if p.is_dir():
        for fp in sorted(p.glob("*_changepoints.pkl"), key=lambda x: x.name.lower()):
            clean = fp.stem.rsplit("_changepoints", 1)[0]
            with open(fp, "rb") as f:
                result[clean] = pickle.load(f)
        return result

    if p.is_file():
        try:
            obj = pd.read_pickle(p)
        except Exception:
            # may not be a pandas object; try generic pickle
            with open(p, "rb") as f:
                obj = pickle.load(f)

        if isinstance(obj, dict):
            return obj

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return _scan_raw_file_group_by_dataset(p)

        print(f"[cp_io] Unsupported PKL object type: {type(obj)} in {p}")
        return {}

    print(f"[cp_io] PKL location not found: {p}")
    return {}


def materialize_cache_from_raw(
    pkl_location: str | os.PathLike,
    cache_dir: str | os.PathLike,
) -> int:
    """
    Create/refresh per-dataset cache files '<clean_key>_changepoints.pkl' from a raw PKL file/dir.
    Returns count of files written.
    """
    p = Path(pkl_location).expanduser().resolve()
    out = Path(cache_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    if p.is_file():
        mapping = load_standardized_changepoints_flex(p)
    elif p.is_dir():
        # Build from raw directory by grouping each dataset found across files
        mapping: Dict[str, Any] = {}
        for fp in sorted(p.glob("*.pkl"), key=lambda x: x.name.lower()):
            try:
                df = pd.read_pickle(fp)
            except Exception:
                continue
            for _, row in df.iterrows():
                base = os.path.basename(str(row.iloc[0]))
                key = clean_dataset_key(base)
                mapping.setdefault(key, []).append(row.values)
        # normalize per key
        new_map: Dict[str, Any] = {}
        for k, rows in mapping.items():
            arr = _normalize_cp_rows(rows)
            if arr is not None:
                new_map[k] = arr[:, 3]
        mapping = new_map
    else:
        return 0

    for key, obj in mapping.items():
        fp = out / f"{key}_changepoints.pkl"
        with open(fp, "wb") as f:
            pickle.dump(obj, f)
        written += 1
    return written
