#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:02:02 2024

@author: vincentcalia-bogan

Yield spike arrays from every `.npz` file in a directory.

Parameters
----------
save_path
    Directory containing .npz files. If None, you will be prompted
    interactively. This function treats `save_path` as a **directory**,
    not a single .npz file.

Yields
------
(spike_array, npz_filename, index, key)
    - spike_array : np.ndarray
        The array stored under a key that starts with "spike_array ".
    - npz_filename : str
        Basename of the source .npz file (e.g., 'dataset_001_repacked.npz').
        Callers that need a numeric dataset id should parse it from this.
    - index : int
        0-based index of the key within the npz file’s key order.
    - key : str
        The exact key name inside the npz (e.g., 'spike_array 0').

Notes
-----
- This is a **generator**; it streams arrays across all files.
- Only keys beginning with "spike_array " are yielded.
- Uses a context manager to close each NPZ file promptly.
- Files are processed in sorted order for reproducibility.

Examples
--------
for spike, fname, idx, key in extract_from_npz("/path/to/npz_dir"):
    ...

"""

import os
from pathlib import Path
from typing import Generator, Tuple

import numpy as np


def extract_from_npz(save_path: str | os.PathLike | None = None
                     ) -> Generator[Tuple[np.ndarray, str, int, str], None, None]:
    # Resolve interactive path if not provided
    if save_path is None:
        save_path = input("Enter the save path: ")
    root = Path(save_path)
    # Check if save_path exists
    if not root.exists():
        print(f"Error: Save path '{root}' does not exist.")
        return
    # Check if save_path is a directory
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        return

    # Stable, filtered file list
    npz_files = sorted(f for f in os.listdir(root) if f.endswith(".npz"))
    if not npz_files:
        print(f"No .npz files found in '{root}'.")
        return

    # Iterate over .npz files in save_path
    for npz_file in npz_files:
        file_path = root / npz_file
        # Only yield arrays under keys that match our convention
        try:
            with np.load(file_path) as npz_data:
                for index, key in enumerate(npz_data.keys()):
                    if key.startswith(
                        "spike_array "
                    ):  # indexing data correctly using a dictionary; the key command
                        spike_array = npz_data[
                            key
                        ]  # create a generator for a given spike array
                        yield spike_array, npz_file, index, key  # decide if index and key are needed
        except Exception as exc:
            # Don’t crash the whole scan if one file is malformed
            print(f"Error reading '{file_path}': {exc!r}")
            continue


### ONCE THE .NPZ FILES ARE GENERATED, YOU CAN WORK OFF THE LOCAL STORAGE SO LONG AS YOU DON'T CALL ANY OF THE ABOVE ###
