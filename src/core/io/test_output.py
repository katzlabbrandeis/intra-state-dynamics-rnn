#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 12:55:50 2025

@author: vincentcalia-bogan
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple, Union

# ---------- internals ----------


def _repo_root() -> Path:
    """
    Resolve the repository root assuming this file lives at:
      repo/src/core/io/test_output.py
    """
    return Path(__file__).resolve().parents[3]


_VALID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_component(s: str) -> str:
    """
    Make a safe path component: spaces -> '_', strip illegal chars,
    collapse repeats. Avoids directory traversal and OS-specific issues.
    """
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_").strip()
    s = _VALID_CHARS.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "unnamed"


def _ensure_parents(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _timestamp(utc: bool = True) -> str:
    dt = datetime.now(timezone.utc if utc else None)
    return dt.strftime("%Y%m%d_%H%M%S")


def _with_ext(name: str, ext: str | None) -> str:
    if not ext:
        return name
    if not ext.startswith("."):
        ext = "." + ext
    if not name.endswith(ext):
        return name + ext
    return name


# ---------- public helpers ----------

def get_tests_output_dir(*subdirs: str, create: bool = True) -> Path:
    """
    Return the base directory for test outputs: <repo>/output/tests_output[/subdirs...].

    Parameters
    ----------
    *subdirs : str
        Optional nested folders (each will be sanitized).
    create : bool
        Create the directory path if it does not exist.

    Notes
    -----
    This is intended for all test/introspection scripts EXCEPT the
    environment exporter/differ (export_or_diff_env), which is exempt
    by your project rule.
    """
    base = _repo_root() / "output" / "tests_output"
    for s in subdirs:
        base = base / _sanitize_component(s)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def build_tests_output_path(
    name: str,
    *,
    subdirs: Iterable[str] | None = None,
    ext: str | None = None,
    timestamp: bool = False,
    utc: bool = True,
    create_parents: bool = True,
) -> Path:
    """
    Construct a file path under <repo>/output/tests_output with optional subdirs,
    sanitized filename, optional extension, and optional timestamp suffix.

    Examples
    --------
    >>> build_tests_output_path("tree", ext="json")
    <repo>/output/tests_output/tree.json

    >>> build_tests_output_path("integrity_report", subdirs=("integrity",), ext=".json", timestamp=True)
    <repo>/output/tests_output/integrity/integrity_report_20250818_112233.json
    """
    subdirs = tuple(subdirs or ())
    dir_path = get_tests_output_dir(*subdirs, create=create_parents)

    base = _sanitize_component(name)
    if timestamp:
        base = f"{base}_{_timestamp(utc=utc)}"
    filename = _with_ext(base, ext)

    path = dir_path / filename
    if create_parents:
        _ensure_parents(path)
    return path


def write_json_test_report(
    data: object,
    name: str,
    *,
    subdirs: Iterable[str] | None = None,
    ext: str = ".json",
    timestamp: bool = True,
    utc: bool = True,
    indent: int = 2,
) -> Path:
    """
    Serialize `data` to JSON in output/tests_output and return the written Path.
    Uses a simple atomic write (write to tmp, then replace).

    This is a convenience wrapper—useful for scripts like:
      - check_package_integrity
      - describe_tree
    """
    path = build_tests_output_path(
        name, subdirs=subdirs, ext=ext, timestamp=timestamp, utc=utc, create_parents=True
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def write_text_test_report(
    text: str,
    name: str,
    *,
    subdirs: Iterable[str] | None = None,
    ext: str = ".txt",
    timestamp: bool = True,
    utc: bool = True,
) -> Path:
    """
    Write plain text to output/tests_output and return the written Path.
    """
    path = build_tests_output_path(
        name, subdirs=subdirs, ext=ext, timestamp=timestamp, utc=utc, create_parents=True
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path
