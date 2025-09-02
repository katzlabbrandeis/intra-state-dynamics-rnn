#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:09:31 2025

@author: vincentcalia-bogan

# src/core/config/params_io.py

"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

_DEFAULTS: Dict[str, Any] = {
    "taste_replacements": {"nacl": "NaCl", "suc": "Sucrose", "ca": "Citric Acid", "qhcl": "Quinine"},
    "epoch_labels": ["Identification", "Palatability", "Decision", "2000 ms Post-Stimulus"],
    "fr_pipeline": {
        "mode": "lite", "window_length": 250, "step_size": 25,
        "start_time": 1500, "end_time": 4500, "fixed_warp_duration": 1000,
        "save_outputs": True
    },
    "rnn_latent": {
        "bin_size_ms": 25, "start_time_ms": 1500, "max_time_ms": 4500,
        "warp_length": 1000, "variance_threshold": 95.0,
        "compute_first_derivative": False, "compute_second_derivative": False,
        "derivative_source": "threshold", "save_outputs": True
    }
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_COMMENT_LINE = re.compile(r"^\s*//.*$")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)


def _decomment(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _COMMENT_LINE.match(ln)]
    no_line = "\n".join(lines)
    return _COMMENT_BLOCK.sub("", no_line)


def load_pipeline_params(repo_root: Path) -> Dict[str, Any]:
    """
    Load pipeline params from src/core/config/pipeline_params.json.
    If missing, create it with baked-in defaults and return those.
    """
    cfg_path = repo_root / "src" / "core" / "config" / "pipeline_params.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(_DEFAULTS, indent=2), encoding="utf-8")
        return _DEFAULTS.copy()

    raw = cfg_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        try:
            data = json.loads(_decomment(raw))
        except json.JSONDecodeError:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup = cfg_path.with_suffix(f".invalid_{ts}.json")
            backup.write_text(raw, encoding="utf-8")
            cfg_path.write_text(json.dumps(_DEFAULTS, indent=2), encoding="utf-8")
            print(f"[params_io] Invalid JSON; backed up to {backup.name} and rewrote defaults.")
            return _DEFAULTS.copy()

    return _deep_merge(_DEFAULTS, data if isinstance(data, dict) else {})
