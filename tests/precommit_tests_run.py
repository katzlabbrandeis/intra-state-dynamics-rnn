#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:10:59 2025

@author: vincentcalia-bogan

Run repo-introspection tests in one place for pre-commit.
Consolidates various tests so that they're run all at once, allowing me control over args.

Included tests:
  1) ensure_init_files       -> creates missing __init__.py in src/core (configurable)
  2) check_package_integrity -> validates imports/graph/cycles/duplicates; FAILS on issues
  3) describe_tree           -> emits repo tree JSON (informational by default)
  4) export_or_diff_env      -> optional; OFF by default; use --env-mode diff|update

Reports for (2) and (3) are written to output/tests_output/ via core.io.test_output.

Exit code:
  - Non-zero if integrity test finds problems (missing imports, cycles, duplicates)
  - Non-zero if any invoked subprocess fails
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Resolve repo root relative to this file
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]

# Add src to the path so `import core...` works for helper writers
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Safe import of the output helpers
try:
    from core.io.test_output import write_json_test_report
except Exception:
    write_json_test_report = None  # we’ll guard usage


def run(cmd):
    """Run a command with PYTHONPATH=src and without writing .pyc files."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{env.get('PYTHONPATH','')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # no __pycache__/pyc for these runs-- unneeded
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(REPO_ROOT), env=env)
    return proc.returncode, proc.stdout, proc.stderr


def json_or_raise(stdout: str) -> Dict[str, Any]:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Expected JSON from sub-tool, got:\n{stdout[:500]}...\n({e})")


def main() -> None:
    parser = argparse.ArgumentParser()
    # Which tests to run
    parser.add_argument("--no-init", action="store_true", help="Skip ensure_init_files")
    parser.add_argument("--no-integrity", action="store_true", help="Skip package integrity check")
    parser.add_argument("--no-tree", action="store_true", help="Skip tree JSON generation")
    parser.add_argument("--env-mode", choices=["off", "diff", "update"], default="off",
                        help="Run export_or_diff_env with mode (default: off)")
    # ensure_init_files tuning
    parser.add_argument("--init-only-nonempty", action="store_true",
                        help="Only create __init__.py in dirs that have .py or subpackages")
    parser.add_argument("--init-also", action="append", default=[],
                        help="Additional base dirs to include (repeatable), e.g., src/scripts")
    # tree tuning
    parser.add_argument("--tree-include-files", action="store_true",
                        help="Include files in tree output (default: dirs only)")
    parser.add_argument("--tree-max-depth", type=int, default=None,
                        help="Max depth for tree traversal (default: unlimited)")
    parser.add_argument("--tree-exclude", nargs="*", default=[
        ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
        "build", "dist", ".idea", ".vscode",
    ], help="Names to exclude anywhere in path")  # updated so that output is not skipped

    # integrity policy: treat tree output as informational (no fail) by default
    parser.add_argument("--tree-enforce", action="store_true",
                        help="Fail if tree generation command fails (still informational content-wise)")

    args = parser.parse_args()

    failures: List[str] = []
    summaries: Dict[str, Any] = {}

    # 1) ensure_init_files (may modify working tree; like a formatter)
    if not args.no_init:
        init_cmd = [
            sys.executable, "-m", "tests.ensure_init_files",
            # now slightly quieter
        ]
        for extra in args.init_also:
            init_cmd += ["--also", extra]
        if args.init_only_nonempty:
            init_cmd.append("--only-nonempty")

        code, out, err = run(init_cmd)
        print("[ensure_init_files]\n" + out.strip())
        if err.strip():
            print("[ensure_init_files:stderr]\n" + err.strip(), file=sys.stderr)
        if code != 0:
            failures.append(f"ensure_init_files exited with {code}")

    # 2) check_package_integrity (MUST return valid JSON on stdout)
    if not args.no_integrity:
        integ_cmd = [sys.executable, "-m", "tests.check_package_integrity"]
        code, out, err = run(integ_cmd)
        if err.strip():
            # integrity script prints JSON to stdout; anything on stderr is suspicious but not fatal yet
            print("[check_package_integrity:stderr]\n" + err.strip(), file=sys.stderr)
        if code != 0:
            failures.append(f"check_package_integrity exited with {code}")
        else:
            try:
                report = json_or_raise(out)
            except RuntimeError as e:
                print(f"[check_package_integrity] {e}", file=sys.stderr)
                failures.append("check_package_integrity produced non-JSON output")
                report = None

            if report is not None:
                # Persist a timestamped JSON report (best effort)
                if write_json_test_report is not None:
                    try:
                        path = write_json_test_report(
                            report, "package_integrity", subdirs=("integrity",), timestamp=True
                        )
                        print(f"[check_package_integrity] wrote report -> {path}")
                    except Exception as werr:
                        print(f"[check_package_integrity] warn: could not write report ({werr})")

                # Enforce failure policy
                n_missing = len(report.get("missing_imports", []))
                n_cycles = len(report.get("import_cycles", []))
                n_dupes = len(report.get("duplicate_symbols", {}))
                summaries["integrity"] = {
                    "missing_imports": n_missing,
                    "import_cycles": n_cycles,
                    "duplicate_symbols": n_dupes,
                    "summary": report.get("summary", {}),
                }
                if n_missing or n_cycles or n_dupes:
                    msg = f"Integrity failures: missing={n_missing}, cycles={n_cycles}, duplicates={n_dupes}"
                    failures.append(msg)
                    print("[check_package_integrity] " + msg, file=sys.stderr)

    # 3) describe_tree (JSON to stdout; write to tests_output; non-fatal content-wise)
    if not args.no_tree:
        tree_cmd = [
            sys.executable, "-m", "tests.describe_tree",
            "--root", str(REPO_ROOT),
        ]
        if args.tree_include_files:
            tree_cmd.append("--include-files")
        if args.tree_max_depth is not None:
            tree_cmd += ["--max-depth", str(args.tree_max_depth)]
        if args.tree_exclude:
            tree_cmd += ["--exclude", *args.tree_exclude]

        code, out, err = run(tree_cmd)
        if err.strip():
            print("[describe_tree:stderr]\n" + err.strip(), file=sys.stderr)
        if code != 0:
            msg = f"describe_tree exited with {code}"
            print("[describe_tree] " + msg, file=sys.stderr)
            if args.tree_enforce:
                failures.append(msg)
        else:
            try:
                tree = json_or_raise(out)
            except RuntimeError as e:
                print(f"[describe_tree] {e}", file=sys.stderr)
                if args.tree_enforce:
                    failures.append("describe_tree produced non-JSON output")
                tree = None
            if tree is not None and write_json_test_report is not None:
                try:
                    path = write_json_test_report(
                        tree, "project_tree", subdirs=("structure",), timestamp=True
                    )
                    print(f"[describe_tree] wrote tree -> {path}")
                except Exception as werr:
                    print(f"[describe_tree] warn: could not write tree report ({werr})")

    # 4) export_or_diff_env (OFF by default; never writes to tests_output)
    if args.env_mode != "off":
        env_cmd = [sys.executable, "-m", "tests.export_or_diff_env", "--mode", args.env_mode]
        code, out, err = run(env_cmd)
        print("[export_or_diff_env]\n" + out.strip())
        if err.strip():
            print("[export_or_diff_env:stderr]\n" + err.strip(), file=sys.stderr)
        if code != 0:
            failures.append(f"export_or_diff_env exited with {code}")

    # Final summary
    if summaries:
        print("\n=== precommit_tests_run summary ===")
        print(json.dumps(summaries, indent=2))

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for f in failures:
            print(" - " + f, file=sys.stderr)
        sys.exit(1)

    print("\nAll selected pre-commit tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
