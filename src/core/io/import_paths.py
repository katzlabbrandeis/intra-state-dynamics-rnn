#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:00:55 2025

@author: vincentcalia-bogan


import_paths — tiny helpers to make <repo>/src and nearby directories importable.
This is designed as a transparent drop-in alternative to hardcoded pathing.
As such, importing things from dirs outside of /src is covered by wiring them in.
Similarly, if importing something from a child of /src, eg:
    src/core/analysis/foo.py → from core.analysis import foo
    (standard handling)

Ideally, calling this should not be nessecary if pip install -e has been called for the env.
This helper is for extra scripts where you haven't set or installed pythonpath.

If needed, this will work:

from core.io.import_paths import ensure_src_on_path  # or wire_imports
ensure_src_on_path()
from core.analysis import foo

What this does
--------------
- Finds the repo root (prefers `pyproject.toml`, falls back to `.git`), then
  ensures `<repo>/src` is on `sys.path` so `import core...` works anywhere.
- Lets you add sibling folders that live next to the repo (e.g. "other_func" one dir over),
  absolute paths, paths relative to the repo or its parent, or from env vars.
- Idempotent: never duplicates entries on `sys.path`. You can choose front/back insertion.

Quick start
-----------
from core.io.import_paths import wire_imports
# Make <repo>/src importable and add a sibling folder beside the repo:
    eg.
wire_imports(siblings=["your/sibling/dir"])
from spike_train_to_npz import find_h5_files, extract_to_npz

Other patterns
--------------
from core.io.import_paths import ensure_src_on_path, add_sibling, temp_sys_path

ensure_src_on_path()                      # just <repo>/src
add_sibling("sibling_repo")       # <repo parent>/underlying functions
with temp_sys_path("tools", relative_to="repo_parent"):
    import some_legacy_module

Anchors & options
-----------------
- relative_to: 'repo' | 'repo_parent' | 'cwd' | 'this_file' | 'abs'
- env-based: add_from_env("BLECH_EXTRA_PATHS")  # colon-separated paths
- must_exist=True (default): silently ignore missing paths
- position="front"|"back": where to insert on sys.path

Notes
-----
This is a lightweight bridging layer. For long-term
reuse, prefer packaging external code as installable modules instead of path hacks.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union

Pathish = Union[str, os.PathLike]
_ANCHOR = Literal["repo", "repo_parent", "cwd", "this_file", "abs"]

# ---- repo location helpers -------------------------------------------------


def _this_file() -> Path:
    # this module lives at repo/src/core/io/import_paths.py
    return Path(__file__).resolve()


def find_repo_root(markers: Tuple[str, ...] = ("pyproject.toml", ".git")) -> Path:
    """
    Ascend from this file until a marker file/dir is found.
    Falls back to the expected repo root inferred from src/core/io.
    """
    cur = _this_file().parent
    for p in [cur, *cur.parents]:
        for m in markers:
            if (p / m).exists():
                # if we found markers under src/core/io/... keep walking to the repo root
                # typically pyproject.toml lives at the repo root
                if (p / "pyproject.toml").exists():
                    return p
        # If we hit the filesystem root, break
        if p == p.parent:
            break
    # Fallback: repo/src/core/io -> repo
    return _this_file().parents[3]


def repo_src() -> Path:
    return find_repo_root() / "src"

# ---- sys.path manipulation --------------------------------------------------


def _norm(p: Pathish) -> str:
    return str(Path(p).resolve())


def _insert_sys_path(p: Path, *, position: Literal["front", "back"] = "front") -> Optional[str]:
    s = str(p)
    # idempotent: avoid duplicates (compare normalized)
    existing = {_norm(x) for x in sys.path if isinstance(x, (str, os.PathLike))}
    if _norm(s) in existing:
        return None
    if position == "front":
        sys.path.insert(0, s)
    else:
        sys.path.append(s)
    return s


def ensure_src_on_path(*, position: Literal["front", "back"] = "front") -> Optional[str]:
    """
    Ensure <repo>/src is importable (so `import core` works anywhere).
    """
    return _insert_sys_path(repo_src(), position=position)


def add_import_paths(
    *paths: Pathish,
    relative_to: _ANCHOR = "abs",
    position: Literal["front", "back"] = "front",
    must_exist: bool = True,
    anchor_file: Optional[Pathish] = None,
) -> List[str]:
    """
    Add one or more paths to sys.path. Paths may be absolute or resolved
    relative to an anchor:

      relative_to='repo'        → <repo>/<path>
      relative_to='repo_parent' → <repo>/../<path>
      relative_to='cwd'         → <cwd>/<path>
      relative_to='this_file'   → <dir_of_anchor_file>/<path>
      relative_to='abs'         → treat entries as absolute

    Returns the list of strings actually inserted (idempotent; may be empty).
    """
    added: List[str] = []
    if relative_to == "repo":
        base = find_repo_root()
    elif relative_to == "repo_parent":
        base = find_repo_root().parent
    elif relative_to == "cwd":
        base = Path.cwd()
    elif relative_to == "this_file":
        if anchor_file is None:
            anchor_file = _this_file()
        base = Path(anchor_file).resolve().parent
    else:  # "abs"
        base = None

    for p in paths:
        pp = Path(p)
        target = (base / pp) if base is not None else pp
        if must_exist and not target.exists():
            continue
        inserted = _insert_sys_path(target, position=position)
        if inserted:
            added.append(inserted)
    return added


def add_sibling(*names: str, position: Literal["front", "back"] = "front", must_exist: bool = True) -> List[str]:
    """
    Add sibling directories by name relative to the repo's parent directory.
    Example: if your repo is .../Senior thesis work/Blech_intra_state_dynamics,
    then add_sibling('underlying functions') adds:
      .../Senior thesis work/underlying functions
    (you'll never guess where I got that example from, hehe)
    """
    return add_import_paths(*names, relative_to="repo_parent", position=position, must_exist=must_exist)


def add_from_env(varname: str, *, sep: Optional[str] = None, position: Literal["front", "back"] = "front", must_exist: bool = True) -> List[str]:
    """
    Add colon/OS-sep separated paths from an environment variable (e.g., BLECH_EXTRA_PATHS).
    """
    raw = os.environ.get(varname, "")
    if not raw:
        return []
    paths = [p for p in (raw.split(sep or os.pathsep)) if p]
    return add_import_paths(*paths, relative_to="abs", position=position, must_exist=must_exist)


@contextmanager
def temp_sys_path(
    *paths: Pathish,
    relative_to: _ANCHOR = "abs",
    position: Literal["front", "back"] = "front",
) -> None:
    """
    Temporarily add paths to sys.path (restored on exit).
    """
    before = list(sys.path)
    try:
        add_import_paths(*paths, relative_to=relative_to, position=position, must_exist=False)
        yield
    finally:
        sys.path[:] = before


def wire_imports(
    *,
    src: bool = True,
    siblings: Iterable[str] = (),
    repo_rel: Iterable[Pathish] = (),
    parent_rel: Iterable[Pathish] = (),
    abs_paths: Iterable[Pathish] = (),
    env_vars: Iterable[str] = (),
    position: Literal["front", "back"] = "front",
    must_exist: bool = True,
) -> List[str]:
    """
    One-shot convenience to set up imports at the top of a script/notebook.

    Example:
        from core.io.import_paths import wire_imports
        wire_imports(
            siblings=["your/sibling/path"],   # your use-case
            abs_paths=["/some/absolute/path"],   # optional
            env_vars=["BLECH_EXTRA_PATHS"],      # optional
        )
        from spike_train_to_npz import find_h5_files

    Returns a list of actually added path strings.
    """
    added: List[str] = []
    if src:
        s = ensure_src_on_path(position=position)
        if s:
            added.append(s)
    if siblings:
        added += add_sibling(*siblings, position=position, must_exist=must_exist)
    if repo_rel:
        added += add_import_paths(*repo_rel, relative_to="repo", position=position, must_exist=must_exist)
    if parent_rel:
        added += add_import_paths(*parent_rel, relative_to="repo_parent", position=position, must_exist=must_exist)
    if abs_paths:
        added += add_import_paths(*abs_paths, relative_to="abs", position=position, must_exist=must_exist)
    for var in env_vars:
        added += add_from_env(var, position=position, must_exist=must_exist)
    return added


__all__ = [
    "find_repo_root",
    "repo_src",
    "ensure_src_on_path",
    "add_import_paths",
    "add_sibling",
    "add_from_env",
    "temp_sys_path",
    "wire_imports",
]
