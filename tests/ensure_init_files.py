#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:05:21 2025

@author: vincentcalia-bogan

Ensure __init__.py exists in all desired package directories.
Useful for ensuring imports are without issue-- as things need to be importable.
Unclear if this is strictly needed, but can't help to have.

Default target: repo/src/core/**

Usage:
  python -m tests.ensure_init_files
  python -m tests.ensure_init_files --base src/core --also src/scripts --dry-run

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

# dirs whose names we never touch
SKIP_DIR_NAMES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "output", "build", "dist", ".idea", ".vscode",
}

DEFAULT_INIT_CONTENT = "# Auto-generated to mark this directory as a Python package.\n"


def should_skip_dir(p: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in p.parts)


def find_target_dirs(bases: Iterable[Path], include_leaf_empty: bool = True) -> list[Path]:
    """
    Return all directories under bases that should have __init__.py.
    - We include the base dir itself.
    - We include all subdirs (except skipped).
    - If include_leaf_empty is False, only include dirs that contain at least one .py file or subpackage.
    """
    targets: list[Path] = []
    for base in bases:
        base = base.resolve()
        if not base.is_dir():
            continue
        for d in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
            if should_skip_dir(d):
                continue
            if include_leaf_empty:
                targets.append(d)
            else:
                has_py = any((d.glob("*.py")))
                has_pkg = any((sub.is_dir() and not should_skip_dir(sub))
                              for sub in d.iterdir() if sub.is_dir())
                if has_py or has_pkg:
                    targets.append(d)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for d in targets:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


def ensure_inits(dirs: list[Path], *, content: str, dry_run: bool) -> tuple[list[Path], list[Path]]:
    created: list[Path] = []
    existed: list[Path] = []
    for d in dirs:
        init_file = d / "__init__.py"
        if init_file.exists():
            existed.append(init_file)
            continue
        if dry_run:
            created.append(init_file)  # mark as would-create
        else:
            init_file.write_text(content, encoding="utf-8")
            created.append(init_file)
    return created, existed


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="append", default=["src/core"],
                        help="Base directory to process (repeatable). Default: src/core")
    parser.add_argument("--also", action="append", default=[],
                        help="Additional base directories to process (repeatable), e.g., src/scripts")
    parser.add_argument("--only-nonempty", action="store_true",
                        help="Only add __init__.py to dirs that contain .py files or subpackages.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created without writing files.")
    parser.add_argument("--content", default=DEFAULT_INIT_CONTENT,
                        help="Content to write into new __init__.py files.")
    args = parser.parse_args()

    base_dirs = [repo_root / b for b in (args.base + args.also)]
    targets = find_target_dirs(base_dirs, include_leaf_empty=not args.only_nonempty)
    created, existed = ensure_inits(targets, content=args.content, dry_run=args.dry_run)

    print(f"[init-sweep] bases: {', '.join(str(b) for b in base_dirs)}")
    print(f"[init-sweep] target dirs: {len(targets)}")
    print(f"[init-sweep] existing __init__.py: {len(existed)}")
    if args.dry_run:
        print(f"[init-sweep] would create: {len(created)} files")
    else:
        print(f"[init-sweep] created: {len(created)} files")

    # list a few examples for visibility
    for p in created[:10]:
        print(f"  + {p.relative_to(repo_root)}")
    if len(created) > 10:
        print(f"  ... and {len(created) - 10} more")


if __name__ == "__main__":
    main()
