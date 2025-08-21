#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:47:57 2025

# src/core/config/preprocess.py

@author: vincentcalia-bogan
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from core.config.params_io import load_pipeline_params
from core.io.standard_paths import ProjectPaths


@dataclass
class PreprocConfig:
    # Inputs (edit to match your machine; they can be passed via CLI, too)
    h5_root: Path                 # e.g., "/Volumes/T7 Shield/spikesorting. Typically the .h5 files come from elsewhere"
    pkl_root: Path                # directory of source .pkl files
    spike_trains_path: str = "/spike_trains"

    # Taste map (optional)
    taste_replacements: Mapping[str, str] = None
    epoch_labels: Sequence[str] = ()

    # Outputs (standardized under /output/intermediate_data)
    paths: ProjectPaths = None

    @staticmethod
    def from_cli(args, repo_root: Path) -> "PreprocConfig":
        params = load_pipeline_params(repo_root)
        paths = ProjectPaths.from_repo_root(repo_root)
        return PreprocConfig(
            h5_root=Path(args.h5_root).expanduser().resolve(),
            pkl_root=Path(args.pkl_root).expanduser().resolve(),
            spike_trains_path=args.spike_trains_path,
            taste_replacements=params.get("taste_replacements", {}),
            epoch_labels=tuple(params.get("epoch_labels", ())),
            paths=paths,
        )
