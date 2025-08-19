#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:17:38 2024

@author: vincentcalia-bogan

unpkl_generator.py
==================

Utilities for extracting **changepoints** from a directory of Pandas pickles.

Expected pickle format (per file)
---------------------------------
Each .pkl is expected to be a Pandas DataFrame where:
- `row.iloc[0]` is a path-like string whose basename encodes the dataset id
  (e.g., ".../dataset_ABC123_240101_123456_repacked.npz").
- `row.values[3]` is an array-like structure for that dataset containing per-epoch
  entries. Each entry is typically a sequence of **3** numbers; this code augments
  it by appending a **4th** value equal to `entry[2] + 2000` (ms). If an entry has
  fewer than 3 numbers, it pads with `NaN` up to length 3, then appends the 4th.

Matching logic
--------------
Datasets are matched by a *truncated name* constructed from parts of the basename
that contain digits. For example, "dataset_ABC_240101_123456_repacked.npz" ->
"dataset_ABC_240101_123456_repacked.npz" (filtered to parts containing digits).
This is a fuzzy-but-stable way of linking .npz dataset ids to rows in the .pkl.

Functions
---------
- extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
    Scan all .pkl files in `pkl_path` and return a NumPy array (dtype=object)
    of rows for the dataset indicated by `dataset_num`, after normalizing the
    4th column as described above. Returns None if not found/invalid.

- unpickle_changepoints(pkl_path, generator)
    Iterate over `(spike_array, dataset_num, index, key)` tuples (e.g., from
    `core.utils.extract_npz.extract_from_npz`) and accumulate normalized
    rows across all datasets, returning a single object array (or None).

Notes & Caveats
---------------
- This module **prints** status/progress and intentionally returns `None`
  on any fatal issue. Adjust to logging/raising as needed.
- The literal `+ 2000` ms augmentation is domain-specific. If that logic
  changes, update both functions below consistently.

"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def extract_valid_changepoints(
    pkl_path: str | os.PathLike,
    spike_array: np.ndarray,
    dataset_num: str,
    index: int,
    key: str,
) -> Optional[np.ndarray]:
    """
    Find and normalize changepoints for a single dataset.

    Parameters
    ----------
    pkl_path
        Directory containing .pkl files (Pandas DataFrames) with changepoint info.
    spike_array
        The spike array corresponding to this dataset (not used by this function,
        but kept for API compatibility with upstream code).
    dataset_num
        The dataset identifier (often the .npz filename). Used to match rows in the pkls.
    index, key
        Extra metadata from the NPZ iterator; not used here.

    Returns
    -------
    np.ndarray | None
        An object-dtype NumPy array of rows (copied from the pickles) where column 3
        (the per-epoch structure) has been coerced so that each element is length-4:
          - first up to three values from the source (padded with NaN if <3)
          - the 4th value is `row[2] + 2000`
        Returns None if no valid rows were found or errors occurred.

    Implementation details
    ----------------------
    - Matching is performed by comparing a 'truncated' name (parts with digits only).
    - Only rows where `row.values[3]` is a NumPy array-like will be kept.
    - Files are scanned in directory order; consider sorting for reproducibility.
    """
    def truncate_name(name):
        """Keep only underscore-separated parts that contain digits; re-join with '_'."""
        numbers = []
        parts = name.split("_")
        truncated_parts = []
        for part in parts:
            if any(char.isdigit() for char in part):
                numbers.extend(char for char in part if char.isdigit())
                truncated_parts.append(part)
        return "_".join(truncated_parts)

    try:
        truncated_dataset_num = truncate_name(dataset_num)
        print(f"Processing dataset number: {truncated_dataset_num}")
        raw_pkl = []
        pkl_files = [f for f in os.listdir(pkl_path) if f.endswith(".pkl")]
        for pkl_file in pkl_files:
            pkl_file_path = os.path.join(pkl_path, pkl_file)
            try:
                df = pd.read_pickle(pkl_file_path)
                # Iterate rows; each row should have a path-like in col 0 and an array-like in col 3.
                for idx, row in df.iterrows():
                    row_basename = os.path.basename(row.iloc[0])
                    truncated_basename = truncate_name(row_basename)
                    if truncated_basename == truncated_dataset_num:
                        extracted_row = row.values
                        if isinstance(extracted_row[3], np.ndarray):
                            raw_pkl.append(extracted_row)
                        else:
                            print(
                                f"Skipping dataset {truncated_dataset_num} due to invalid data structure at index {idx}: {extracted_row[3]}"
                            )
                            raw_pkl = []
                            break  # stop scanning this file for this dataset
            except Exception as e:
                print(f"Error processing file {pkl_file}: {e}")
                raw_pkl = []
                break  # bail on this dataset if any .pkl is malformed

        if not raw_pkl:
            print(
                f"Skipping dataset {truncated_dataset_num} due to invalid data structures."
            )
            return None

        print(
            f"Finished processing dataset {truncated_dataset_num}. Starting to process raw_pkl."
        )
        raw_pkl = np.array(raw_pkl, dtype=object)
        # Create extracted_pkl by ensuring correct length and adding the fourth number
        #   (up to first 3 original values, padded with NaN if needed) + (val[2] + 2000)
        extracted_pkl = []
        for i in range(len(raw_pkl)):
            new_row = list(raw_pkl[i])
            new_sub_array = []
            # new_row[3] is expected to be an iterable of per-epoch sequences
            for j in range(len(new_row[3])):
                row = new_row[3][j]
                if isinstance(row, (list, np.ndarray)) and len(row) == 3:
                    row = np.append(row, row[2] + 2000)
                elif isinstance(row, (list, np.ndarray)) and len(row) < 3:
                    padding = [np.nan] * (3 - len(row))
                    row = np.append(row, padding)
                    row = np.append(row, row[2] + 2000)
                else:
                    print(
                        f"Unexpected data structure for row {j} in new_row[{i}]: {row}"
                    )
                    continue
                new_sub_array.append(row)
            new_row[3] = np.array(new_sub_array)
            extracted_pkl.append(new_row)

        extracted_pkl = np.array(extracted_pkl, dtype=object)
        print(f"Finished processing valid dataset {truncated_dataset_num}.")
        return extracted_pkl if len(extracted_pkl) > 0 else None
    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def unpickle_changepoints(pkl_path, generator):
    """
    Aggregate normalized changepoints across many datasets.

    Parameters
    ----------
    pkl_path
        Directory containing .pkl files (Pandas DataFrames) with changepoint info.
    generator
        An iterable (or generator) yielding tuples `(spike_array, dataset_num, index, key)`,
        typically produced by `core.utils.extract_npz.extract_from_npz`. Only `dataset_num`
        is used for matching here; the other fields are passed through for API consistency.

    Returns
    -------
    np.ndarray | None
        A single object-dtype NumPy array containing all normalized rows for all datasets
        processed, or None if nothing valid was found.

    Notes
    -----
    Logic mirrors `extract_valid_changepoints` but loops over all datasets provided by
    `generator`. Duplicated code is intentional for clarity; consider refactoring to
    share internals if you plan to evolve the normalization rules.
    """
    def truncate_name(name):
        """Keep only underscore-separated parts that contain digits; re-join with '_'."""
        numbers = []
        parts = name.split("_")
        truncated_parts = []
        for part in parts:
            if any(char.isdigit() for char in part):
                numbers.extend(char for char in part if char.isdigit())
                truncated_parts.append(part)
        return "_".join(truncated_parts)

    all_extracted_pkl = []

    try:
        print("Starting to process pkl files.")
        for spike_array, dataset_num, index, key in generator:
            truncated_dataset_num = truncate_name(dataset_num)
            print(f"Processing dataset number: {truncated_dataset_num}")
            raw_pkl = []
            pkl_files = [f for f in os.listdir(pkl_path) if f.endswith(".pkl")]
            for pkl_file in pkl_files:
                pkl_file_path = os.path.join(pkl_path, pkl_file)
                try:
                    df = pd.read_pickle(pkl_file_path)
                    for idx, row in df.iterrows():
                        row_basename = os.path.basename(row.iloc[0])
                        truncated_basename = truncate_name(row_basename)
                        if truncated_basename == truncated_dataset_num:
                            extracted_row = row.values
                            if isinstance(extracted_row[3], np.ndarray):
                                raw_pkl.append(extracted_row)
                            else:
                                print(
                                    f"Skipping dataset {truncated_dataset_num} due to invalid data structure at index {idx}: {extracted_row[3]}"
                                )
                                raw_pkl = []
                                break
                except Exception as e:
                    print(f"Error processing file {pkl_file}: {e}")
                    raw_pkl = []
                    break

            if not raw_pkl:
                print(
                    f"Skipping dataset {truncated_dataset_num} due to invalid data structures."
                )
                continue

            print(
                f"Finished processing dataset {truncated_dataset_num}. Starting to process raw_pkl."
            )
            raw_pkl = np.array(raw_pkl, dtype=object)

            # Same normalization as above: enforce length 4 by padding + (val[2] + 2000)
            extracted_pkl = []
            for i in range(len(raw_pkl)):
                new_row = list(raw_pkl[i])
                new_sub_array = []
                for j in range(len(new_row[3])):
                    row = new_row[3][j]
                    if isinstance(row, (list, np.ndarray)) and len(row) == 3:
                        row = np.append(row, row[2] + 2000)
                    elif isinstance(row, (list, np.ndarray)) and len(row) < 3:
                        padding = [np.nan] * (3 - len(row))
                        row = np.append(row, padding)
                        row = np.append(row, row[2] + 2000)
                    else:
                        print(
                            f"Unexpected data structure for row {j} in new_row[{i}]: {row}"
                        )
                        continue
                    new_sub_array.append(row)
                new_row[3] = np.array(new_sub_array)
                extracted_pkl.append(new_row)

            extracted_pkl = np.array(extracted_pkl, dtype=object)
            all_extracted_pkl.extend(extracted_pkl)
            print(f"Finished processing valid dataset {truncated_dataset_num}.")

        all_extracted_pkl = np.array(all_extracted_pkl, dtype=object)
        return all_extracted_pkl if len(all_extracted_pkl) > 0 else None
    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
