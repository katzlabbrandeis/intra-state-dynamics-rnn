#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 14:27:56 2025

@author: vincentcalia-bogan

# src/core/config/roots_io.py

This function is designed to cache the given roots of data (eg. pkl files, hdf5 files, eventually the rnn input)
so that when you call whatever mapping, it just works.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

ROOTS_FILENAME = "roots.json"  # lives next to pipeline_params.json


@dataclass
class Roots:
    h5_root: Optional[Path] = None
    pkl_root: Optional[Path] = None
    latent_parquet_root: Optional[Path] = None

    def to_jsonable(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @staticmethod
    def from_mapping(m: Dict[str, Any]) -> "Roots":
        def _p(x: Any) -> Optional[Path]:
            if x is None or x == "":
                return None
            return Path(str(x)).expanduser().resolve()
        return Roots(
            h5_root=_p(m.get("h5_root")),
            pkl_root=_p(m.get("pkl_root")),
            latent_parquet_root=_p(m.get("latent_parquet_root")),
        )


def _roots_path(repo_root: Path) -> Path:
    # Keep alongside pipeline_params.json
    cfg_dir = Path(repo_root).expanduser().resolve() / "src" / "core" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / ROOTS_FILENAME


def load_roots(repo_root: Path) -> Roots:
    p = _roots_path(repo_root)
    if not p.exists():
        return Roots()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # backup and return empty
        ts = time.strftime("%Y%m%d_%H%M%S")
        p.rename(p.with_suffix(f".invalid_{ts}.json"))
        return Roots()
    return Roots.from_mapping(data if isinstance(data, dict) else {})


def save_roots(repo_root: Path, roots: Roots) -> None:
    p = _roots_path(repo_root)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(roots.to_jsonable(), indent=2), encoding="utf-8")
    tmp.replace(p)


def update_roots(repo_root: Path, **overrides: Optional[str | Path]) -> Roots:
    """
    Merge provided overrides into stored roots and persist.
    Values can be str/Path/None. None means "leave as is".
    """
    cur = load_roots(repo_root)
    new = Roots(
        h5_root=_norm(overrides.get("h5_root", cur.h5_root)),
        pkl_root=_norm(overrides.get("pkl_root", cur.pkl_root)),
        latent_parquet_root=_norm(overrides.get("latent_parquet_root", cur.latent_parquet_root)),
    )
    save_roots(repo_root, new)
    return new


def _norm(x: Optional[str | Path]) -> Optional[Path]:
    if x is None or x == "":
        return None
    return Path(x).expanduser().resolve()


def resolve_roots(
    repo_root: Path,
    *,
    cli_h5_root: Optional[str | Path] = None,
    cli_pkl_root: Optional[str | Path] = None,
    cli_latent_root: Optional[str | Path] = None,
    pkl_cache_dir: Optional[Path] = None,
    require_h5: bool = False,
) -> Roots:
    """
    Precedence:
      CLI > stored roots.json > (PKL only) cache fallback.

    - If require_h5=True and no H5 root available, raises RuntimeError.
    - If PKL root unavailable, falls back to cache (if provided) and prints a warning.
    - Any CLI values are persisted immediately.
    """
    stored = load_roots(repo_root)

    # Prefer CLI when provided (and persist change)
    h5 = _norm(cli_h5_root) or stored.h5_root
    pkl = _norm(cli_pkl_root) or stored.pkl_root
    lat = _norm(cli_latent_root) or stored.latent_parquet_root

    # H5: must exist if required
    if require_h5 and h5 is None:
        raise RuntimeError(
            "[roots] H5 root is not configured. Provide --h5-root or set it in src/core/config/roots.json"
        )

    # PKL: allow fallback to cache if missing
    if pkl is None and pkl_cache_dir is not None:
        print(f"[roots] No PKL root configured — falling back to cache: {pkl_cache_dir}")
        pkl = pkl_cache_dir

    # Persist if anything changed
    changed = (
        (h5 != stored.h5_root) or
        (pkl != stored.pkl_root) or
        (lat != stored.latent_parquet_root)
    )
    if changed:
        save_roots(repo_root, Roots(h5_root=h5, pkl_root=pkl, latent_parquet_root=lat))

    return Roots(h5_root=h5, pkl_root=pkl, latent_parquet_root=lat)
