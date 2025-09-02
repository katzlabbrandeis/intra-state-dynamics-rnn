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
from typing import Mapping, Optional, Sequence

from core.config.params_io import load_pipeline_params
from core.config.roots_io import resolve_roots
from core.io.standard_paths import ProjectPaths


@dataclass
class PreprocConfig:
    # Inputs (edit to match your machine; they can be passed via CLI, too)
    h5_root: Path                 # e.g., "/Volumes/T7 Shield/spikesorting. Typically the .h5 files come from elsewhere"
    pkl_root: Path                # directory of source .pkl files
    rnn_latent_parquet_root: Optional[Path] = None      # optional: precomputed latent Parquet root

    # HDF5 internals

    spike_trains_path: str = "/spike_trains"

    # Taste map (optional)
    taste_replacements: Mapping[str, str] = None
    epoch_labels: Sequence[str] = ()

    # Outputs (standardized under /output/intermediate_data)
    paths: ProjectPaths = None

    @staticmethod
    def from_cli(args, repo_root: Path, require_h5: bool = False) -> "PreprocConfig":
        """
        Build a config by:
        - Loading params.json (taste map, epoch labels)
        - Ensuring ProjectPaths (output/intermediate_data/*)
        - Resolving & persisting roots with precedence:
            CLI > stored roots.json > (PKL only) cache fallback
        """
        params = load_pipeline_params(repo_root)
        paths = ProjectPaths.from_repo_root(repo_root)
        paths.ensure()

        roots = resolve_roots(
            repo_root=repo_root,
            cli_h5_root=getattr(args, "h5_root", None),
            cli_pkl_root=getattr(args, "pkl_root", None),
            cli_rnn_latent_root=getattr(args, "rnn_latent_parquet_root", None),
            pkl_cache_dir=paths.pkl_dir,
            require_h5=require_h5,
        )
        return PreprocConfig(
            h5_root=roots.h5_root,
            pkl_root=roots.pkl_root,
            rnn_latent_parquet_root=roots.latent_parquet_root,
            spike_trains_path=getattr(args, "spike_trains_path", "/spike_trains"),
            taste_replacements=params.get("taste_replacements", {}),
            epoch_labels=tuple(params.get("epoch_labels", ())),
            paths=paths,
        )
