#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 12:39:36 2025

@author: vincentcalia-bogan

Describe the repository tree as JSON and also emit a `tree`-style ASCII directory view.

Describe the repository tree as JSON and also emit a `tree`-style ASCII directory view.
The ASCII view is saved to BOTH a .txt and a .json file under output/tests_output/structure/.

Usage examples:
  python -m tests.describe_tree                       # JSON stdout + ASCII files (dirs only)
  python -m tests.describe_tree --include-files       # include files in both views
  python -m tests.describe_tree --max-depth 4
  python -m tests.describe_tree --no-ascii            # skip ASCII side outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Repo root default (tests/ is one level under repo/)
REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

DEFAULT_EXCLUDES = {
    ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    "build", "dist", ".idea", ".vscode",
}
# do NOT exclude the output file

# Optional writers (best-effort; stdout remains JSON for the orchestrator)
try:
    _SRC = REPO_ROOT_DEFAULT / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from core.io.test_output import write_json_test_report, write_text_test_report  # noqa: F401
    write_text_test_report = None  # skip writing the text for now
except Exception:
    write_json_test_report = None
    write_text_test_report = None


# ---------- JSON tree (dirs-only unless include_files=True) ----------

def build_tree(
    root: Path,
    *,
    exclude: Iterable[str],
    include_files: bool,
    max_depth: int | None,
    _depth: int = 0,
) -> Dict[str, Any]:
    node: Dict[str, Any] = {"name": root.name, "type": "dir", "children": []}
    if max_depth is not None and _depth >= max_depth:
        return node

    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return node

    for p in entries:
        if any(part in exclude for part in p.parts):
            continue
        if p.is_dir():
            node["children"].append(
                build_tree(
                    p,
                    exclude=exclude,
                    include_files=include_files,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
            )
        elif include_files:
            node["children"].append({"name": p.name, "type": "file"})
    return node


# ---------- ASCII tree (like `tree`; dirs-only by default) ----------

def _iter_dir_entries(
    root: Path,
    *,
    exclude: set[str],
    include_files: bool,
) -> Tuple[List[Path], List[Path]]:
    """Return (dirs, files) lists sorted by name, excluding paths containing an excluded part."""
    try:
        it = list(root.iterdir())
    except PermissionError:
        return [], []
    keep = [p for p in it if not any(part in exclude for part in p.parts)]
    dirs = sorted([p for p in keep if p.is_dir()], key=lambda p: p.name.lower())
    files = sorted([p for p in keep if p.is_file()],
                   key=lambda p: p.name.lower()) if include_files else []
    return dirs, files


def build_ascii_lines_and_counts(
    root: Path,
    *,
    exclude: set[str],
    include_files: bool,
    max_depth: int | None,
) -> Tuple[List[str], int, int]:
    """
    Build the ASCII tree lines and return (lines, n_dirs, n_files).
    By default behaves like `tree -d` (directories only).
    """
    lines: List[str] = [str(root)]
    n_dirs, n_files = 0, 0

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        nonlocal n_dirs, n_files
        if max_depth is not None and depth >= max_depth:
            return
        dirs, files = _iter_dir_entries(dir_path, exclude=exclude, include_files=include_files)
        entries = dirs + files
        for idx, child in enumerate(entries):
            is_last = (idx == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{child.name}")
            if child.is_dir():
                n_dirs += 1
                extension = "    " if is_last else "│   "
                walk(child, prefix + extension, depth + 1)
            else:
                n_files += 1

    walk(root, prefix="", depth=0)
    return lines, n_dirs, n_files


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe project tree as JSON and ASCII.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT_DEFAULT,
                        help="Root directory to describe (default: repo root).")
    parser.add_argument("--include-files", action="store_true",
                        help="Include files as well as directories (default: dirs only).")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Limit recursion depth.")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDES),
                        help="Names to exclude anywhere in the path.")
    parser.add_argument("--no-ascii", action="store_true",
                        help="Do not produce the ASCII tree side-outputs.")
    args = parser.parse_args()

    exclude_set = set(args.exclude)

    # Build directory JSON (printed to stdout for the orchestrator)
    tree = build_tree(
        args.root,
        exclude=exclude_set,
        include_files=args.include_files,
        max_depth=args.max_depth,
    )
    print(json.dumps(tree, indent=2))

    # Best-effort: write directory JSON to tests_output
    if write_json_test_report is not None:
        try:
            write_json_test_report(
                tree, "project_tree", subdirs=("structure",), timestamp=True
            )
        except Exception:
            pass

    # Build ASCII tree (and write to BOTH .txt and .json)
    if not args.no_ascii:
        ascii_lines, n_dirs, n_files = build_ascii_lines_and_counts(
            args.root,
            exclude=exclude_set,
            include_files=args.include_files,
            max_depth=args.max_depth,
        )
        ascii_text = "\n".join(ascii_lines) + "\n"

        # .txt output
        if write_text_test_report is not None:
            try:
                write_text_test_report(
                    ascii_text, "project_tree_ascii", subdirs=("structure",), ext=".txt", timestamp=True
                )
            except Exception:
                pass

        # .json output (contains string, per-line list, and counts)
        ascii_payload = {
            "ascii_text": ascii_text,
            "ascii_lines": ascii_lines,
            "counts": {"dirs": n_dirs, "files": n_files},
            "include_files": args.include_files,
            "max_depth": args.max_depth,
            "root": str(args.root),
            "excluded": sorted(exclude_set),
        }
        if write_json_test_report is not None:
            try:
                write_json_test_report(
                    ascii_payload, "project_tree_ascii", subdirs=("structure",), ext=".json", timestamp=True
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()
