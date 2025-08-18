#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 12:41:00 2025

@author: vincentcalia-bogan

#!/usr/bin/env python3

Export or diff the current conda environment against environment.yml.

Usage:
  # Dry-run diff
  python -m tests.export_or_diff_environment --mode diff

  # Update environment.yml in place (re-writes file if it changed)
  python -m tests.export_or_diff_environment --mode update
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
from pathlib import Path


def export_env_text(conda_env: str | None) -> str:
    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError("Conda not found on PATH. Activate your env first.")

    cmd = [conda, "env", "export", "--no-builds"]
    if conda_env:
        cmd += ["-n", conda_env]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if not ln.startswith("prefix:")]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices={"diff", "update"}, default="diff")
    parser.add_argument("--env", dest="env_name", default=None,
                        help="Conda env name (defaults to current).")
    parser.add_argument("--path", type=Path, default=Path("environment.yml"))
    args = parser.parse_args()

    new_text = export_env_text(args.env_name)
    old_text = args.path.read_text() if args.path.exists() else ""

    if args.mode == "diff":
        if new_text == old_text:
            print("No changes to environment.yml")
        else:
            print("Differences vs environment.yml:")
            for ln in difflib.unified_diff(
                old_text.splitlines(), new_text.splitlines(),
                fromfile="environment.yml (old)", tofile="environment.yml (new)", lineterm=""
            ):
                print(ln)
    else:  # update
        if new_text != old_text:
            args.path.write_text(new_text)
            print(f"Updated {args.path.resolve()}")
        else:
            print("environment.yml already up to date.")


if __name__ == "__main__":
    main()
