#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:09:31 2025

@author: vincentcalia-bogan

# src/core/config/params_io.py

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

_DEFAULTS = {
    "taste_replacements": {
        "nacl": "NaCl",
        "suc": "Sucrose",
        "ca":  "Citric Acid",
        "qhcl": "Quinine"
    },
    "epoch_labels": [
        "Identification",
        "Palatability",
        "Decision",
        "2000 ms Post-Stimulus"
    ]
}


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
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Merge with defaults to tolerate future keys
    merged = _DEFAULTS.copy()
    merged.update(data or {})
    return merged
