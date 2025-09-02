#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:47:40 2024

@author: vincentcalia-bogan

find_extract_info.py
====================

Helpers for:
- Recursively finding `.info` metadata files and copying them to a destination.
- Parsing dataset-specific `.info` files to extract taste labels.
- Normalizing taste labels via a replacement map.

Conventions & Matching
----------------------
Datasets are matched to `.info` files using the first **two** 6-digit tokens
found in the dataset identifier and in the `.info` filename. For example:

    dataset_num: ".../dataset_240101_123456_repacked.npz"
    info file:   ".../session_240101_123456.info"

Both produce the tokens ["240101", "123456"] → match.

Functions
---------
- find_copy_h5info(source_path=None, destination_path=None) -> list[str]
    Recursively scan `source_path` for `.info` files and copy them into
    `destination_path`. Returns the list of **original** `.info` paths found.

- process_info_files(info_path, dataset_num) -> list[str]
    Within `info_path`, find the `.info` file matching `dataset_num`, parse JSON,
    and return `taste_params.tastes` as a list of strings.

- modify_tastes(dataset_tastes, replacements) -> list[str]
    Apply a mapping (old → new) to taste labels, leaving non-mapped labels intact.

Notes
-----
- This module prints human-friendly errors and returns defaults when possible.
  Replace `print` with logging if you prefer.
- `find_copy_h5info` creates `destination_path` if it does not exist.

"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def find_copy_h5info(source_path: str | os.PathLike | None = None,
                     destination_path: str | os.PathLike | None = None) -> List[str]:
    """
    Recursively find all `.info` files under `source_path` and copy them into `destination_path`.

    Parameters
    ----------
    source_path
        Root directory to search. If None, prompts interactively.
    destination_path
        Directory to copy `.info` files into. If None, prompts interactively.

    Returns
    -------
    list[str]
        A list of the **original** `.info` file paths found (not the copied paths).

    Behavior
    --------
    - Creates `destination_path` if it does not exist.
    - Traverses subdirectories recursively.
    - Uses `shutil.copy2` to preserve file metadata.
    - Continues on errors (prints a message and moves on).

    Example
    -------
    >>> found = find_copy_h5info("/data/raw/sessions", "/data/info_cache")
    >>> len(found)
    42
    """
    if source_path is None:
        source_path = input("Enter the source directory path for .info files: ")
    if destination_path is None:
        destination_path = input(
            "Enter the destination directory path for copying .info files: "
        )

    src = Path(source_path).expanduser().resolve()
    dst = Path(destination_path).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)  # new
    if not src.exists() or not src.is_dir():
        print(f"[find_copy_h5info] Invalid source directory: {src}")
        return []
    # Ensure destination directory exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    info_files: List[str] = []  # Initialize list to store .info file paths
    for entry in sorted(src.rglob("*.info"), key=lambda e: e.name.lower()):  # old: os.scandir
        try:
            if entry.is_dir():
                # Recurse into subdirectories
                subdir_info_files = find_copy_h5info(entry.path, dst)
                if subdir_info_files:
                    info_files.extend(subdir_info_files)
            elif entry.is_file() and entry.name.endswith(".info"):
                info_files.append(entry.path)
                # Copy file to the destination directory
                try:
                    shutil.copy(entry.path, destination_path)
                except Exception as copy_exc:
                    print(f"[find_copy_h5info] Failed to copy '{entry}' → '{dst}': {copy_exc!r}")
        except PermissionError as pe:
            print(f"[find_copy_h5info] Permission denied: {entry} ({pe})")
        except Exception as exc:
            print(f"[find_copy_h5info] Error reading {entry}: {exc!r}")
    return info_files


def process_info_files(info_path, dataset_num) -> List[str]:
    """
    Find the .info file that matches `dataset_num` (by the first two YYMMDD tokens)
    and return its taste list.

    Matching rule:
      - Extract the first two 6-digit numbers from the dataset name (YYMMDD_YYMMDD...)
      - Find the first *.info whose filename contains the same first two 6-digit numbers in order.

    Parameters
    ----------
    info_path : str | Path
        Directory containing .info files (flat, after copying).
    dataset_num : str
        Typically an .npz filename like 'AM35_4Tastes_201229_150307_repacked.npz'.

    Returns
    -------
    list[str]
        Tastes read from the file, or [] if missing.
    """
    root = Path(info_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Info dir not found: {root}")

    # normalize dataset name (strip any path, keep name)
    ds_name = Path(str(dataset_num)).name
    ds_tokens = re.findall(r"\d{6}", ds_name)
    if len(ds_tokens) < 2:
        raise ValueError(
            f"Expected at least two 6-digit tokens in dataset name: {ds_name}"
        )
    key2 = tuple(ds_tokens[:2])

    # iterate deterministically over *.info filenames
    candidates = sorted(root.glob("*.info"), key=lambda p: p.name.lower())

    match_path = None
    for fp in candidates:
        file_tokens = re.findall(r"\d{6}", fp.name)
        if len(file_tokens) >= 2 and tuple(file_tokens[:2]) == key2:
            match_path = fp
            break

    if match_path is None:
        raise FileNotFoundError(
            f"No matching .info file in {root} for dataset tokens {key2} (from {ds_name})"
        )

    # read tastes
    try:
        with match_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        tastes = data.get("taste_params", {}).get("tastes", [])
        if not isinstance(tastes, list):
            tastes = []
        return tastes
    except Exception as e:
        print(f"[process_info_files] Failed to parse {match_path.name}: {e}")
        return []


def modify_tastes(dataset_tastes: Iterable[str], replacements: Dict[str, str]) -> List[str]:
    """
    Apply a replacement map to taste labels.

    Parameters
    ----------
    dataset_tastes
        Iterable of taste names from an `.info` file.
    replacements
        Mapping of old → new names (e.g., {"NaCl": "salt"}).

    Returns
    -------
    list[str]
        A new list with replacements applied where keys matched; otherwise original values.

    Example
    -------
    >>> modify_tastes(["NaCl", "Sucrose"], {"NaCl": "salt"})
    ['salt', 'Sucrose']
    """
    modified_tastes = []
    for taste in dataset_tastes:
        if taste in replacements:
            modified_tastes.append(replacements[taste])
        else:
            modified_tastes.append(taste)
    return modified_tastes
