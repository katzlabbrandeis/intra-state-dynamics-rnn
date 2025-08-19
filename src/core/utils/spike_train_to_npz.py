#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:54:40 2024

@author: vincentcalia-bogan

spike_train_to_npz.py
=====================

Utilities to:
1) Recursively discover `.h5` files that contain spike trains.
2) Extract spike train arrays from each HDF5 file and save them into `.npz` files.

HDF5 expectation
----------------
Each HDF5 file is expected to contain a group at `spike_trains_path` (default: "/spike_trains")
with child nodes that each expose a **`spike_array`** dataset/attribute. We read every node
under that group and stack their `spike_array` values.

Outputs
-------
For each input `.h5`, we create one `.npz` file beside your chosen `save_path`, containing a
single array with a key like `"spike_array <N>"`. The exact number `<N>` matches the code below.

Functions
---------
- find_h5_files(file_path=None) -> list[str]
    Recursively walk `file_path` and return all `.h5` file paths discovered.

- extract_to_npz(h5_files, file_path, spike_trains_path="/spike_trains", save_path=None)
    For each `.h5` path in `h5_files`, read all spike arrays under `spike_trains_path`,
    stack them, and write to `<save_path>/<basename>.npz` with a key `"spike_array <N>"`.

Notes
-----
- These are eager operations and will load entire arrays into memory.
- The second parameter `file_path` in `extract_to_npz` is **unused** (kept for API compatibility).
- The function currently **raises** on missing `/spike_trains` (see comment)—this will abort
  the entire batch. Change to `print(...); continue` if you prefer skip-on-error behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import tables  # PyTables

## ITERATING THROUGH H5 FILES AND SAVING SPIKE ARRAYS TO NPZ FILE ##


# iterate through all files, finding the .h5 files and saving the path to a list
def find_h5_files(file_path: str | os.PathLike | None = None) -> List[str]:
    """
    Recursively collect all `.h5` files under a directory.

    Parameters
    ----------
    file_path
        Root folder to search. If None, prompts interactively.

    Returns
    -------
    list[str]
        Absolute paths of all discovered `.h5` files (depth-first order).

    Notes
    -----
    - Uses os.scandir for speed; recurses into subdirectories.
    - Does not validate the internal structure of HDF5 files—only finds them.
    """
    if file_path is None:
        file_path = input("Enter the pathname for h5 files: ")
    root = Path(file_path)
    if not root.exists() or not root.is_dir():
        print(f"[find_h5_files] invalid directory: {root}")
        return []

    h5_files: List[str] = []  # init list
    for entry in sorted(os.scandir(root), key=lambda e: e.name.lower()):
        try:
            if entry.is_dir():
                subdir_h5_files = find_h5_files(entry.path)
                if subdir_h5_files:
                    h5_files.extend(subdir_h5_files)
                else:
                    pass  # If no .h5 files found in the subdirectory, do nothing
            elif entry.is_file() and entry.name.endswith(".h5"):
                h5_files.append(entry.path)
        except PermissionError as pe:
            print(f"[find_h5_files] permission denied: {entry.path} ({pe})")
        except Exception as exc:
            print(f"[find_h5_files] error reading {entry.path}: {exc!r}")
    return h5_files


# now, for each .h5 file, saving locations of spike trains and associated data to several 4-D arrays

def extract_to_npz(
    h5_files: Sequence[str | os.PathLike],
    file_path,
    spike_trains_path: str | None = None,
    save_path: str | os.PathLike | None = None,
) -> None:
    """
    For each `.h5` file, read spike trains under `spike_trains_path` and save to `.npz`.

    Parameters
    ----------
    h5_files
        Iterable of `.h5` file paths to process.
    file_path
        Unused parameter (legacy). Accepted but ignored.
    spike_trains_path
        HDF5 path to the spike trains group. Defaults to "/spike_trains".
    save_path
        Directory where output `.npz` files will be written. If None, prompts interactively.

    Behavior
    --------
    - For each HDF5 file:
        * open in read-only mode
        * list nodes under `spike_trains_path`
        * read each node's `spike_array`
        * stack into a single NumPy array with `np.stack`
        * save to `<save_path>/<basename>.npz` as:
              { f"spike_array {num_h5_files + i + 1}": <stacked array> }
      (The numeric suffix mirrors the original code; adjust if you want contiguous numbering
       from 1..N rather than offset by `num_h5_files`.)

    Raises
    ------
    Exception
        If `spike_trains_path` does not exist in a file, an Exception is raised and the
        entire batch aborts. Change the `raise` to `print(...); continue` to only skip that file.
    """
    if spike_trains_path is None:
        spike_trains_path = "/spike_trains"
    if save_path is None:
        save_path = input("Enter the save path: ")

    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_h5_files = len(h5_files)

    for i, h5_path in enumerate(sorted(map(str, h5_files))):  # i for number of files
        with tables.open_file(h5_path, mode="r") as f:  # same as before
            try:
                dig_ins = f.list_nodes(
                    spike_trains_path
                )  # if the .h5 file hasn't been processed to generate spike trains
            except tables.NoSuchNodeError:
                raise Exception(
                    f"Spike trains not found at {spike_trains_path} for file {h5_path}, skipping file."
                )  # raise the exception
            try:
                # collect the spike trains from the H5
                spike_trains = [x.spike_array[:] for x in dig_ins]
            except Exception as exc:
                raise Exception(f"Failed to read spike arrays from {h5_path}: {exc!r}")
        spike_array = np.stack(spike_trains)

        # Build NPZ payload: keep original numbering scheme
        spike_arrays_dict = {
            f"spike_array {num_h5_files + i + 1}": spike_array
        }  # + 1 to ensure we start at number 1

        # Construct file name for saving
        h5_file_name = os.path.splitext(os.path.basename(h5_path))[0]
        file_name = f"{h5_file_name}.npz"  # numbering
        out_path = os.path.join(out_dir, file_name)

        # Save the spike arrays to the file
        np.savez(out_path, **spike_arrays_dict)  # ** for arbitrary number of arrays

    print(f"Data saved successfully at {out_dir}")
