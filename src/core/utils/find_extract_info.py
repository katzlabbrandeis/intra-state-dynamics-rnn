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

    src = Path(source_path)
    dst = Path(destination_path)
    if not src.exists() or not src.is_dir():
        print(f"[find_copy_h5info] Invalid source directory: {src}")
        return []
    # Ensure destination directory exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    info_files: List[str] = []  # Initialize list to store .info file paths
    for entry in sorted(os.scandir(src, key=lambda e: e.name.lower())):
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
                    print(f"[find_copy_h5info] Failed to copy '{entry.path}' → '{dst}': {copy_exc!r}")
        except PermissionError as pe:
            print(f"[find_copy_h5info] Permission denied: {entry.path} ({pe})")
        except Exception as exc:
            print(f"[find_copy_h5info] Error reading {entry.path}: {exc!r}")
    return info_files


def process_info_files(info_path: str | os.PathLike, dataset_num: str) -> List[str]:
    """
    Find the `.info` file matching `dataset_num` and extract taste labels.

    Matching rule
    -------------
    Extract the first two 6-digit tokens from `dataset_num` and from each `.info`
    filename in `info_path`. The first file whose first two tokens match wins.

    Parameters
    ----------
    info_path
        Directory containing `.info` files (JSON).
    dataset_num
        Dataset identifier string (often an `.npz` filename). Must contain at least
        two 6-digit tokens (e.g., '240101_123456').

    Returns
    -------
    list[str]
        The `tastes` list from the `.info` JSON (usually under `taste_params.tastes`).
        Returns an empty list if the file is found but does not contain the key.

    Raises
    ------
    ValueError
        If `dataset_num` contains fewer than two 6-digit tokens.
    FileNotFoundError
        If no `.info` file in `info_path` matches the tokens from `dataset_num`.
    """
    def find_matching_info_file() -> Optional[Path]:
        # Look for two YYMMDD-like tokens in dataset_num
        dataset_numbers = re.findall(r"\d{6}", dataset_num)
        if len(dataset_numbers) < 2:
            raise ValueError(
                "The dataset_num should contain at least two 6-digit numbers."
            )

        for entry in sorted(os.scandir(info_path)):
            if entry.is_file() and entry.name.endswith(".info"):
                file_numbers = re.findall(r"\d{6}", entry.name)
                if len(file_numbers) >= 2 and file_numbers[:2] == dataset_numbers[:2]:
                    return entry.path
        return None

    def extract_tastes_from_info(file_path: Path) -> List[str]:
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
                # Typical structure: { "taste_params": { "tastes": ["NaCl", "Sucrose", ...] } }
                # print("JSON content:", data)  # Debug print to inspect the JSON content
                if "taste_params" in data and "tastes" in data["taste_params"]:
                    return data["taste_params"]["tastes"]
                else:
                    print(
                        "No 'tastes' key found in the 'taste params' key of the JSON data."
                    )  # Debug print if key is missing
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")
        return []

    matching_info_file = find_matching_info_file()
    if matching_info_file:
        dataset_tastes = extract_tastes_from_info(matching_info_file)
        return dataset_tastes
    else:
        raise FileNotFoundError(
            f"No matching .info file found for dataset_num: {dataset_num}"
        )


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
