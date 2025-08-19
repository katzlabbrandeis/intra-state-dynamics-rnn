#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:11:58 2025

@author: vincentcalia-bogan

bootstrap.py
============

Make '<repo>/src' importable in THIS Python / conda environment.
This is so that all the imports work in this runtime

Usage:
  python bootstrap.py                  # write a .pth file (default)
  python bootstrap.py --method editable  # pip install -e .
  python bootstrap.py --method conda-activate  # write conda activate hook
  python bootstrap.py --dry-run

After running (without --dry-run), it sanity-checks: `import core`.

If you (the user) needs to sanity-check:
python -c "import core, sys; print('OK:', core.__file__); print('src on path? ', any(p.endswith('/src') for p in sys.path))"

"""

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
import sysconfig
from pathlib import Path


def _find_site_dir(prefer_user: bool = False) -> Path:
    # Try env site-packages first (works in conda/venv), then user site, then sysconfig fallback.
    candidates = []
    if not prefer_user:
        try:
            candidates.extend(site.getsitepackages())
        except Exception:
            pass
    candidates.append(site.getusersitepackages())
    try:
        candidates.append(sysconfig.get_paths()["purelib"])
    except Exception:
        pass

    for d in candidates:
        p = Path(d)
        if p.exists() and os.access(p, os.W_OK):
            return p
    # last ditch: user site even if not writable check passed earlier
    return Path(site.getusersitepackages())


def _write_pth(src: Path, site_dir: Path, name: str = "blech_src.pth") -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    pth = site_dir / name
    pth.write_text(str(src) + "\n", encoding="utf-8")
    print(f"[bootstrap] wrote {pth} -> {src}")
    return pth


def _write_conda_activate(src: Path) -> Path | None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        print("[bootstrap] CONDA_PREFIX not set; skipping conda activate hook")
        return None
    act_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    act_dir.mkdir(parents=True, exist_ok=True)
    sh = act_dir / "blech_src.sh"
    sh.write_text(f'export PYTHONPATH="{src}:${{PYTHONPATH:-}}"\n', encoding="utf-8")
    print(f"[bootstrap] wrote conda activate hook: {sh}")
    return sh


def _install_editable(repo_root: Path) -> None:
    print("[bootstrap] running: pip install -e .")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_root)])


def _sanity_import() -> None:
    import importlib
    m = importlib.import_module("core")
    print("[bootstrap] import core OK ->", Path(m.__file__).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure <repo>/src is importable")
    parser.add_argument("--method", choices=["pth", "editable", "conda-activate"], default="pth")
    parser.add_argument("--name", default="blech_src.pth", help="Name of the .pth file (method=pth)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src"
    if not (src / "core").exists():
        print("[bootstrap] ERROR: expected src/core to exist next to bootstrap.py")
        return 2

    if args.method == "pth":
        site_dir = _find_site_dir()
        if args.dry_run:
            print(f"[bootstrap] would write {site_dir / args.name} -> {src}")
        else:
            _write_pth(src, site_dir, args.name)
    elif args.method == "editable":
        if args.dry_run:
            print("[bootstrap] would run: pip install -e .")
        else:
            _install_editable(repo_root)
    elif args.method == "conda-activate":
        if args.dry_run:
            print("[bootstrap] would write conda activate hook")
        else:
            _write_conda_activate(src)

    if not args.dry_run:
        _sanity_import()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
