#!/usr/bin/env python3
"""
Integrity + topology checker for the Blech_intra_state_dynamics codebase.

Outputs JSON with:
  • missing_imports        – imports that could not be resolved inside `core.*`
  • duplicate_symbols      – public names (funcs/classes/methods) defined in >1 module
  • file_tree              – every tracked .py (relative to repo root)
  • symbols_by_file        – per-file { "funcs": [...], "classes": {C: [methods...]}}
  • import_graph           – adjacency list for imports among `core.*` modules
  • import_cycles          – list of cycles (each a list of module names)
  • summary                – counts + sha256 digest (stable across runs if no changes)

Assumptions:
  - Your package lives under:   repo_root / "src" / "core"
  - You import your code as:    import core.analysis.foo
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# standardizing how things are consistently saved
# from core.io.test_output import write_json_test_report
try:
    from core.io.test_output import write_json_test_report  # noqa: F401
except Exception:
    write_json_test_report = None
# --------------------------------------------------------------------------------------
# Paths / constants
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]             # repo/
SRC_ROOT = REPO_ROOT / "src"                 # repo/src/
PKG_ROOT = SRC_ROOT / "core"                 # repo/src/core/
PKG_NAME = "core"

# Ensure `importlib.util.find_spec("core...")` can resolve packages under src/
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EXCLUDE_DIR_NAMES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "build", "dist", ".idea", ".vscode",
}
# do not exclude output
# --------------------------------------------------------------------------------------
# Utilities


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts)


def rel_to_repo(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def module_name_from_path(pyfile: Path) -> Optional[str]:
    """
    Turn repo/src/core/analysis/foo.py into 'core.analysis.foo'.
    Return None if the file is not under PKG_ROOT.
    """
    try:
        rel = pyfile.relative_to(SRC_ROOT)
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts or parts[0] != PKG_NAME:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    if not parts:
        return PKG_NAME
    return ".".join(parts)


def find_spec_exists(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

# --------------------------------------------------------------------------------------
# Collect files


ALL_PY_FILES: List[Path] = []
for p in REPO_ROOT.rglob("*.py"):
    if is_excluded(p):
        continue
    ALL_PY_FILES.append(p)

FILE_TREE: List[str] = sorted(rel_to_repo(p) for p in ALL_PY_FILES)

# Limit module/introspection scanning to package code
PKG_PY_FILES: List[Path] = [p for p in ALL_PY_FILES if str(p).startswith(str(PKG_ROOT))]

# Map: module_name -> file_path
MODULE_MAP: Dict[str, Path] = {}
for f in PKG_PY_FILES:
    mn = module_name_from_path(f)
    if mn:
        MODULE_MAP[mn] = f

# --------------------------------------------------------------------------------------
# Import resolution helpers (incl. relative imports)


def resolve_from_import(current_module: str, node: ast.ImportFrom) -> Optional[str]:
    """
    Resolve 'from X import ...' to an import edge target at module granularity.
    For graph purposes we use node.module (if present) after resolving relativity.
    """
    level = getattr(node, "level", 0) or 0  # 0 == absolute
    base = node.module or ""
    if level == 0:
        # Absolute import. Use base (could be 'core.analysis', or stdlib/third-party).
        return base
    # Relative import: strip `level` segments from the current package
    cur_parts = current_module.split(".")
    # If we're in a module (not a package), its package is all but the last component
    if MODULE_MAP.get(current_module, None):
        # module path exists; package is everything except trailing leaf
        cur_parts = cur_parts[:-1] if len(cur_parts) > 1 else cur_parts
    # Go up `level` ancestors
    if level > 0:
        cur_parts = cur_parts[:-level] if level <= len(cur_parts) else []
    target = ".".join(cur_parts)
    if base:
        target = f"{target}.{base}" if target else base
    return target or None

# --------------------------------------------------------------------------------------
# Data holders


missing_imports: List[Dict[str, Any]] = []
public_symbol_index: Dict[str, List[str]] = {}  # name -> [module_paths...]
symbols_by_file: Dict[str, Dict[str, Any]] = {}
import_graph: Dict[str, Set[str]] = {}          # mod -> set(deps)
all_modules_seen: Set[str] = set()


def add_public(name: str, module_rel_path: str) -> None:
    if not name or name.startswith("_"):
        return
    public_symbol_index.setdefault(name, []).append(module_rel_path)

# --------------------------------------------------------------------------------------
# Scan package files


for py in PKG_PY_FILES:
    rel = rel_to_repo(py)
    src = py.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src, filename=rel)
    except SyntaxError as exc:
        # treat syntax errors as a "missing" problem
        missing_imports.append({"file": rel, "line": exc.lineno or 0,
                               "import": "<syntax>", "error": str(exc)})
        continue

    modname = module_name_from_path(py)
    if not modname:
        continue

    all_modules_seen.add(modname)
    import_graph.setdefault(modname, set())

    # --- imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name  # e.g., "core.analysis.foo" or "numpy"
                # Only graph/check inside our package namespace
                if target.startswith(PKG_NAME + "."):
                    import_graph[modname].add(target.split(".", maxsplit=1)[0] if target == PKG_NAME else target.rsplit(
                        ".", 0)[0] if False else target.split(".")[0] if target.count(".") == 0 else target)
                    # verify resolvable
                    if not find_spec_exists(target) and target.split(".")[0] == PKG_NAME:
                        # mark if not in MODULE_MAP nor importable
                        if target not in MODULE_MAP and not find_spec_exists(target):
                            missing_imports.append({
                                "file": rel, "line": node.lineno, "import": target, "error": "Module not found"
                            })
        elif isinstance(node, ast.ImportFrom):
            target = resolve_from_import(modname, node)
            if target and target.startswith(PKG_NAME):
                # Graph at the package/module granularity we resolved to
                import_graph[modname].add(target)
                # Verify resolvable
                # Accept either a real module (find_spec) or a known source file in MODULE_MAP
                ok = find_spec_exists(target) or any(
                    t == target or t.startswith(target + ".") for t in MODULE_MAP.keys()
                )
                if not ok:
                    missing_imports.append({
                        "file": rel, "line": node.lineno, "import": target, "error": "Module not found"
                    })

    # --- symbol inventory
    funcs: List[str] = []
    classes: Dict[str, List[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
            add_public(node.name, rel)
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes[node.name] = methods
            add_public(node.name, rel)
            for m in methods:
                add_public(f"{node.name}.{m}", rel)

    symbols_by_file[rel] = {"funcs": funcs, "classes": classes}

# --------------------------------------------------------------------------------------
# Duplicate public symbol detection
duplicate_symbols = {name: paths for name, paths in public_symbol_index.items() if len(paths) > 1}

# --------------------------------------------------------------------------------------
# Import graph normalization: ensure nodes exist
for mod in list(import_graph.keys()):
    for dep in list(import_graph[mod]):
        if dep not in import_graph:
            import_graph.setdefault(dep, set())

# --------------------------------------------------------------------------------------
# Cycle detection (simple DFS with stack tracking)


def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    color: Dict[str, int] = {}  # 0=unseen, 1=visiting, 2=done
    stack: List[str] = []
    cycles: Set[Tuple[str, ...]] = set()

    def dfs(u: str):
        color[u] = 1
        stack.append(u)
        for v in graph.get(u, ()):
            c = color.get(v, 0)
            if c == 0:
                dfs(v)
            elif c == 1:
                # found a back-edge; extract cycle from stack
                if v in stack:
                    i = stack.index(v)
                    cyc = stack[i:] + [v]
                    # canonicalize to avoid dupes: rotate to lexicographically smallest
                    body = cyc[:-1]
                    k = min(range(len(body)), key=lambda idx: body[idx])
                    rotated = tuple(body[k:] + body[:k])
                    cycles.add(rotated)
        stack.pop()
        color[u] = 2

    for n in graph.keys():
        if color.get(n, 0) == 0:
            dfs(n)
    # materialize as list of lists (closing the cycle by repeating first at end)
    return [list(c) + [c[0]] for c in sorted(cycles)]


import_cycles = find_cycles(import_graph)

# --------------------------------------------------------------------------------------
# Summary + digest (stable over runs if nothing changed)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


summary = {
    "files_scanned": len(FILE_TREE),
    "pkg_modules": len([m for m in MODULE_MAP.keys()]),
    "graph_nodes": len(import_graph),
    "graph_edges": sum(len(v) for v in import_graph.values()),
    "cycles": len(import_cycles),
    "missing_imports": len(missing_imports),
    "duplicate_symbols": len(duplicate_symbols),
}

digest_payload = {
    "file_tree": FILE_TREE,
    "symbols_by_file": symbols_by_file,
    "import_graph": {k: sorted(v) for k, v in import_graph.items()},
}
digest = hashlib.sha256(stable_json(digest_payload).encode("utf-8")).hexdigest()
summary["digest_sha256"] = digest

# --------------------------------------------------------------------------------------
# Emit JSON

report = {
    "missing_imports": missing_imports,
    "duplicate_symbols": duplicate_symbols,
    "file_tree": FILE_TREE,
    "symbols_by_file": symbols_by_file,
    "import_graph": {k: sorted(v) for k, v in import_graph.items()},
    "import_cycles": import_cycles,
    "summary": summary,
}
print(json.dumps(report, indent=2))

if write_json_test_report is not None:
    try:
        write_json_test_report(tree, "project_tree", subdirs=("structure",), timestamp=True)
    except Exception:
        pass
