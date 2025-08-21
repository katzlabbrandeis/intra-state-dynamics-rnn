#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:44:41 2025

Standardized output paths for the very standardized intermediate files we're working with.

# src/core/io/standard_paths.py
@author: vincentcalia-bogan
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    output_dir: Path
    intermediate_dir: Path
    npz_dir: Path  # formerly npz_path
    info_dir: Path  # formerly info_path
    pkl_dir: Path  # formerly pkl_path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "ProjectPaths":
        out = repo_root / "output"
        interm = out / "intermediate_data"
        return ProjectPaths(
            repo_root=repo_root,
            output_dir=out,
            intermediate_dir=interm,
            npz_dir=interm / "spike_trains_npz",
            info_dir=interm / "info_files",
            # IMPORTANT: raw PKL is authoritative and lives under cfg.pkl_root (input).
            # This pkl_dir is OPTIONAL cache space under outputs; won’t write to it by default.
            pkl_dir=interm / "pkl_cache",
        )

    def ensure(self) -> None:
        for p in (self.output_dir, self.intermediate_dir, self.npz_dir, self.info_dir, self.pkl_dir):
            p.mkdir(parents=True, exist_ok=True)
