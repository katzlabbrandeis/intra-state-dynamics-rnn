#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 17:52:39 2025

Attempting to modernize my firing rate calculation class so I can finally get to re-making some plots. Ugh this is taking far too long.

@author: vincentcalia-bogan

DEPRECATION NOTE: Though FRPipeline is still somewhat relevant, ALL warping should now be consolidated to
RNNLatentprocessing.py -- hence the creation of FRPipelineLite.
FRPipeline is still peripherally relevant, but FRPipelineLite is really the important one.

fr_pipeline.py
==============

Pipelines for computing firing-rate (FR) and spike-train summaries aligned to
changepoints derived from model outputs.

This module provides two classes:

1) FRPipeline
   ----------
   Full pipeline that:
     • parses changepoints from .pkl files (matched to .npz datasets),
     • slices spike trains into epochs per (taste, trial, changepoint),
     • computes rolling-window FR,
     • optionally time-warps FR and spikes to a fixed duration,
     • accumulates results into **Polars** DataFrames (per dataset),
     • optionally writes parquet files under configurable subfolders.

   Typical outputs (per dataset key):
     - self.fr_unwarped_df[f"{clean}_rr_firing_unwarped"]     → FR per-bin, unwarped
     - self.spikes_unwarped_df[f"{clean}_spikes_unwarped"]    → spike trains, unwarped
     - self.fr_warped_df[f"{clean}_rr_firing_warped"]         → FR warped to fixed duration
     - self.spikes_warped_df[f"{clean}_spikes_warped"]        → spike trains warped to fixed duration

   DataFrame schema (per row, unwarped/warped):
     - signal columns: neuron_0 … neuron_{N-1} (Float64)
     - meta columns: taste (Int64), trial (Int64), changepoint (Int64) [FRPipeline only], time (Int64)

   Array shape expectation:
     spike_array.shape == (T, R, N, time)
       T = tastes, R = trials, N = neurons, time = samples in ms (or per-bin)
   Upstream code is assumed to produce this orientation. This module does NOT
   attempt to transpose/repair axis order.

2) FRPipeline_lite
   ----------------
   Simpler pipeline that:
     • computes unwarped FR over a (start_time, end_time) window across full trials,
     • returns spike trains and FR (no warping, no changepoint slicing),
     • optionally writes parquet.

Notes
-----
- This file prints progress/warnings. Swap to `logging` if desired.
- Time-warp settings rely on linear interpolation via SciPy's `interp1d`.
- Where you see “double-check +1,” that reflects inclusive/exclusive fencepost decisions.
  Keep as-is if current outputs match your analyses; adjust consistently otherwise.

Dependencies (if local)
--------------------
- extract_npz.extract_from_npz
- unpkl_generator.extract_valid_changepoints
- find_extract_info.{process_info_files, modify_tastes}

If installed as a package (hint: it is), prefer:
    from core.utils.extract_npz import extract_from_npz
    from core.utils.unpkl_generator import extract_valid_changepoints
    from core.utils.find_extract_info import modify_tastes, process_info_files
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
from scipy.interpolate import interp1d

# Module imports (for when loading as local, see above.)
from core.io.import_paths import ensure_src_on_path
from core.utils.extract_npz import extract_from_npz
from core.utils.find_extract_info import modify_tastes, process_info_files
from core.utils.unpkl_generator import extract_valid_changepoints


class FRPipeline:
    """
    Unified pipeline: extract changepoints, compute spike segments & firing rates,
    accumulate results directly into Polars DataFrames, and optionally save.

    Parameters
    ----------
    npz_path : str
        Directory containing dataset .npz files (iterated by `extract_from_npz`).
    info_path : str
        Directory containing `.info` files with taste metadata.
    pkl_path : str
        Directory of .pkl files containing changepoint rows.
    taste_replacements : dict
        Mapping to normalize taste labels (used only during changepoint extraction).
    output_dir : str
        Root directory where parquet outputs are written if `save_outputs=True`.
    window_length : int, default 250
        Rolling window (ms) for FR computation.
    step_size : int, default 25
        Step (ms) between FR windows.
    fixed_warp_duration : int, default 1000
        Target duration (ms) to which warped outputs are interpolated. If None, warping
        uses the native epoch duration.
    start_time : int, default 1500
        Epoch slicing starts at this ms within each trial.
    end_time : int | None, default None
        If provided, truncate epoch slicing to this ms within each trial.
    save_outputs : bool, default False
        If True, parquet files are written under `output_dir` subfolders.
    """

    def __init__(
        self,
        npz_path: str,
        info_path: str,
        pkl_path: str,
        taste_replacements: dict,
        output_dir: str,
        window_length: int = 250,
        step_size: int = 25,
        fixed_warp_duration: int = 1000,
        start_time: int = 1500,
        end_time: int | None = None,
        save_outputs: bool = False,
    ):
        self.npz_path = npz_path
        self.info_path = info_path
        self.pkl_path = pkl_path
        self.taste_replacements = taste_replacements
        self.output_dir = output_dir
        self.window_length = window_length
        self.step_size = step_size
        self.fixed_warp_duration = fixed_warp_duration
        self.start_time = start_time
        self.end_time = end_time
        self.save_outputs = save_outputs
        # in-memory outputs keyed by a dataset-specific string (e.g., "<clean>_rr_firing_unwarped")
        self.changepoints_dict: dict[str, np.ndarray] = {}
        self.fr_unwarped_df: dict[str, pl.DataFrame] = {}
        self.spikes_unwarped_df: dict[str, pl.DataFrame] = {}
        self.fr_warped_df: dict[str, pl.DataFrame] = {}
        self.spikes_warped_df: dict[str, pl.DataFrame] = {}

    def extract_changepoints(self) -> None:
        """
        Populate `self.changepoints_dict` with rows from matched .pkl entries.

        Iterates datasets yielded by `extract_from_npz(self.npz_path)`. For each dataset:
          1) Reads taste list from info files and normalizes via `taste_replacements`.
             (Currently only used as a side-effect / validation.)
          2) Loads changepoint rows via `extract_valid_changepoints`.
          3) Stores the 4th column (index 3) of the returned object-array under:
                self.changepoints_dict[f"dataset_{dataset_num}"] = extracted_pkl[:, 3]
             (This preserves the per-epoch arrays; consumers unpack per taste/trial.)
        """
        for data in extract_from_npz(self.npz_path):
            if isinstance(data, tuple):
                spike_array, dataset_num, index, key = data
                dataset_name = f"dataset_{dataset_num}"
                # Pull tastes; retained primarily for cross-checks or future labeling.
                dataset_tastes = process_info_files(self.info_path, dataset_num)
                modified_tastes = modify_tastes(dataset_tastes, self.taste_replacements)
                extracted_pkl = extract_valid_changepoints(
                    self.pkl_path, spike_array, dataset_num, index, key
                )

                if extracted_pkl is not None:
                    try:
                        changepoints = extracted_pkl[:, 3]  # Keep full array
                        self.changepoints_dict[dataset_name] = changepoints
                    except IndexError:
                        print(
                            f"Index error with dataset {dataset_name}: shape {extracted_pkl.shape}"
                        )
                else:
                    print(f"No valid changepoints found for dataset {dataset_name}")

    def extract_changepoints_dict(self, save_outputs: bool = False) -> dict:
        """
        Extract only the changepoints for each dataset (populates self.changepoints_dict).
        Optionally saves them under <output_dir>/changepoints as pickled arrays.

        Returns
        -------
        dict
            A shallow copy (reference) of `self.changepoints_dict`.
        """
        # (re)initialize
        self.changepoints_dict = {}
        # delegate to extract_changepoints
        self.extract_changepoints()

        if save_outputs:
            save_path = Path(self.output_dir) / "changepoints"
            save_path.mkdir(parents=True, exist_ok=True)
            for dataset_name, changepoints in self.changepoints_dict.items():
                clean_name = dataset_name.replace("dataset_", "").split("_repacked.npz")[0]
                out_path = save_path / f"{clean_name}_changepoints.pkl"
                if not out_path.exists():
                    with open(out_path, "wb") as f:
                        pickle.dump(changepoints, f)

        return self.changepoints_dict

    def _ensure_dirs(self):
        """Create output subfolders if `save_outputs=True`."""
        for sub in ("rr_firing_unwarped", "spikes_unwarped", "rr_firing_warped", "spikes_warped"):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

    def _calc_fr_rr(self, seg: np.ndarray) -> np.ndarray:
        """
        Compute rolling-window firing rates for a neuron×time segment.

        Parameters
        ----------
        seg : np.ndarray, shape = (N, Tseg)
            Binary or spike-count time series per neuron.

        Returns
        -------
        fr : np.ndarray, shape = (N, nbins)
            Sum over each window divided by window duration (in seconds).

        Notes
        -----
        - Uses inclusive-start, exclusive-end windows: [s, e).
        - If `tlen < window_length`, returns a single bin using min(e, tlen).
        """
        wl, ss = self.window_length, self.step_size
        nrn, tlen = seg.shape
        nbins = max((tlen - wl) // ss + 1, 1)
        fr = np.zeros((nrn, nbins))
        for b in range(nbins):
            s = b * ss
            e = min(s + wl, tlen)
            fr[:, b] = seg[:, s:e].sum(axis=1) / (wl / 1000.0)  # axis of 1 or -1? date: 7/8
        return fr

    def process_all(self):
        """
        For each dataset:
          • Iterate tastes/trials,
          • Slice spike trains into epochs delimited by changepoints,
          • Build unwarped spike & FR DataFrames,
          • Time-warp both spike trains and FR to a fixed duration (optional),
          • Save parquet outputs if configured.

        Produces/updates:
          self.spikes_unwarped_df, self.fr_unwarped_df,
          self.spikes_warped_df,   self.fr_warped_df
        """
        self._ensure_dirs()
        for data in extract_from_npz(self.npz_path):
            if not (isinstance(data, tuple) and len(data) == 4):
                continue
            spike_array, ds_num, idx, key = data
            ds_name = f"dataset_{ds_num}"
            clean = ds_name.replace("dataset_", "").split("_repacked.npz")[0]
            if ds_name not in self.changepoints_dict:
                continue
            cps_all = self.changepoints_dict[ds_name]

            spu_list, fru_list, spw_list, frw_list = [], [], [], []
            T, R, N, time = spike_array.shape
           # T, R, N, _ = spike_array.shape
            # setting a bin max if the warp len is not consist.
            max_bins = (time - self.window_length) // self.step_size + 1
            if self.fixed_warp_duration is not None:
                max_bins = (self.fixed_warp_duration - self.window_length) // self.step_size + 1

            for t in range(T):
                if t >= len(cps_all):
                    continue  # taste-trial mismatch
                taste_cps = cps_all[t]
                if not hasattr(taste_cps, '__len__') or len(taste_cps) != R:
                    print('Taste CPs are inhomogenous!')
                    continue
                # finding the max segments across all trials
                # Unused but potentially useful diagnostic stats:
                first_val = taste_cps[:, 0]
                second_val = taste_cps[:, 1]
                third_val = taste_cps[:, 2]
                fourth_val = taste_cps[:, 3]
                max_1 = np.max(first_val)
                max_2 = np.max(second_val)
                max_3 = np.max(third_val)
                max_4 = np.max(fourth_val)
                largest_cps = np.array([max_1, max_2, max_3, max_4])

                for tr in range(R):
                    trial_cps = np.array(taste_cps[tr], dtype=float)
                    if np.isnan(trial_cps).any():
                        continue
                    start = int(self.start_time)

                    # --- Build list of epoch boundaries (optionally forced to end_time)
                    cp_list = list(trial_cps)
                    final_cp = self.end_time if self.end_time is not None else trial_cps[-1]
                    if cp_list[-1] < final_cp:
                        cp_list.append(final_cp)
                    # Sanity clamp start and each cp by end_time (if provided)
                    start = min(start, self.end_time) if self.end_time is not None else start

                    # Ensure all ends don’t exceed end_time-- this is done below a bit, but segment hendling is down below
                    cp_list = [min(cp, self.end_time) if self.end_time is not None else cp for cp in cp_list]

                    for si, cp in enumerate(cp_list):
                        raw_end = int(cp)
                        end = min(raw_end, self.end_time) if self.end_time is not None else raw_end
                        # this line keeps end_time from being uniformly 4500 ..?

                        if end <= start:
                            continue  # skip invalid segments

                        segment = spike_array[t, tr][:, start:end]
                        # debug statement below:
                        # print(f"Epoch t={t}, trial={tr}, segment {si}: start={start}, end={end}, duration={end-start}")  # debug statement
                        # ---------------------- unwarped spikes (per-ms rows)
                        # unwarped spikes
                        if segment.size:
                            times = list(range(start, start + segment.shape[1]))
                            spu_list.append(
                                pl.DataFrame({
                                    **{f"neuron_{i}": segment[i].tolist() for i in range(N)},
                                    "taste": [t] * segment.shape[1],
                                    "trial": [tr] * segment.shape[1],
                                    "changepoint": [si] * segment.shape[1],
                                    "time": times,
                                })
                            )
                            # ---------------------- unwarped FR (rolling window)
                            fr_seg = self._calc_fr_rr(segment)
                            # removing the + b bug? no, that's not it..
                            fr_times = [start + b * self.step_size for b in range(fr_seg.shape[1])]

                          #  fr_times = [start + b * self.step_size for b in range(fr_seg.shape[1])] # TODO: the plus b is causing some issues
                            fru_list.append(
                                pl.DataFrame({
                                    **{f"neuron_{i}": fr_seg[i].tolist() for i in range(N)},
                                    "taste": [t] * fr_seg.shape[1],
                                    "trial": [tr] * fr_seg.shape[1],
                                    "changepoint": [si] * fr_seg.shape[1],
                                    "time": fr_times,
                                })
                            )
                            # ---------------------- warped FR (interpolate to fixed duration)
                            if fr_seg.shape[1] > 1:
                                # ms-length of the epoch you want to warp to
                                if self.fixed_warp_duration is not None:  # bug fix soem weird stuff?
                                    wlen_ms = self.fixed_warp_duration
                                else:
                                    wlen_ms = 1000  # hard-coded to 1000 ms warp duration
                                # how many FR-bins that becomes, given your wl/ss
                                nbins_w = max(wlen_ms // self.step_size + 1, 1)  # + 1 is effectively fencepost
                                # build normalized axes for interpolation
                                orig_axes = np.linspace(0, 1, fr_seg.shape[1])
                                target_axes = np.linspace(0, 1, nbins_w)
                                # interpolate each neuron’s firing‐rate curve
                                frw_seg = np.vstack([
                                    interp1d(orig_axes, fr_seg[n], kind='linear',  # we want to try linear
                                             bounds_error=False, fill_value='extrapolate')(target_axes)  # warp to this
                                    for n in range(N)
                                ])
                                # recompute the “time” column for these warped‐FR bins
                                # eliminated adding the start time?
                                frw_times = [b * self.step_size for b in range(nbins_w)]
                                # Warped time column is local (0, step_size, 2*step_size, …)
                                frw_list.append(
                                    pl.DataFrame({
                                        **{f"neuron_{i}": frw_seg[i].tolist() for i in range(N)},
                                        "taste":       [t] * nbins_w,
                                        "trial":       [tr] * nbins_w,
                                        "changepoint": [si] * nbins_w,
                                        "time":        frw_times,
                                    })
                                )
                            # ---------------------- warped spikes (per-ms rows, interpolated)
                            if segment.shape[1] > 1:
                                norm = np.linspace(0, 1, segment.shape[1])
                                wlen = self.fixed_warp_duration or int(cp - start)
                                wnorm = np.linspace(0, 1, wlen)
                                war_seg = np.vstack([
                                    interp1d(norm, segment[n], kind='linear', bounds_error=False,
                                             fill_value='extrapolate')(wnorm)  # warp to wnorm
                                    for n in range(N)
                                ])
                                # added self.start_time to adjust for plotting woes
                                wtimes = list(range(war_seg.shape[1]))
                                spw_list.append(
                                    pl.DataFrame({
                                        **{f"neuron_{i}": war_seg[i].tolist() for i in range(N)},
                                        "taste": [t] * war_seg.shape[1],
                                        "trial": [tr] * war_seg.shape[1],
                                        "changepoint": [si] * war_seg.shape[1],
                                        "time": wtimes,
                                    })
                                )

                        start = end  # next epoch starts where the last ended

            # ---- collect results under consistent keys per dataset
            # concatenate and save with the analysis type
            analysis_type = ("rr_firing_unwarped", "spikes_unwarped", "rr_firing_warped", "spikes_warped")
            if spu_list:
                self.spikes_unwarped_df[f"{clean}_{analysis_type[1]}"] = pl.concat(spu_list)
            if fru_list:
                self.fr_unwarped_df[f"{clean}_{analysis_type[0]}"] = pl.concat(fru_list)
            if spw_list:
                self.spikes_warped_df[f"{clean}_{analysis_type[3]}"] = pl.concat(spw_list)
            if frw_list:
                self.fr_warped_df[f"{clean}_{analysis_type[2]}"] = pl.concat(frw_list)
            # ---- optional writes
            if self.save_outputs:
                for name, df in [
                    ('spikes_unwarped', self.spikes_unwarped_df.get(f"{clean}_{analysis_type[1]}")),
                    ('rr_firing_unwarped',     self.fr_unwarped_df.get(f"{clean}_{analysis_type[0]}")),
                    # bug: seems to be stepping at odd intervals for unwarped firing; which shouldn't be happening. not sure why, will investigate later.
                    ('spikes_warped',   self.spikes_warped_df.get(f"{clean}_{analysis_type[3]}")),
                    ('rr_firing_warped',       self.fr_warped_df.get(f"{clean}_{analysis_type[2]}")),
                ]:
                    if df is not None:
                        path = os.path.join(self.output_dir, name, f"{clean}_{name}.parquet")
                        df.write_parquet(path)

    def full_pipeline(self) -> tuple[dict, dict, dict, dict]:
        """
        Run `extract_changepoints_dict` then `process_all`.

        Returns
        -------
        tuple(dict, dict, dict, dict)
            (fr_unwarped_df, spikes_unwarped_df, fr_warped_df, spikes_warped_df)
        """
        self.extract_changepoints_dict(self.save_outputs)
        self.process_all()
        return (
            self.fr_unwarped_df,
            self.spikes_unwarped_df,
            self.fr_warped_df,
            self.spikes_warped_df,
        )

# -----------------------------------------------------------------------------
# A "lite" pipeline that skips changepoints and warping.
# Produces unwarped FR and spike-train tables over [start_time, end_time).
# -----------------------------------------------------------------------------


class FRPipeline_lite:
    """
    Stripped-down pipeline: compute unwarped firing rates across full trials
    with bin-aligned time, taste numbers, and window slicing.

    Parameters mirror FRPipeline where applicable, without changepoint/pkl args.

    Written for increased performance, especially as calculating warped firing rates is now partially deprecated
    (as all warping is beind done in RNNLatentprocessing.py)
    """

    def __init__(
        self,
        npz_path: str,
        info_path: str,
        taste_replacements: dict,
        output_dir: str,
        window_length: int = 250,
        step_size: int = 25,
        start_time: int = 1500,
        end_time: int | None = None,
        save_outputs: bool = False,
    ):
        self.npz_path = npz_path
        self.info_path = info_path
        self.taste_replacements = taste_replacements
        self.output_dir = output_dir
        self.window_length = window_length
        self.step_size = step_size
        self.start_time = start_time
        self.end_time = end_time
        self.save_outputs = save_outputs

        self.fr_unwarped_df: dict[str, pl.DataFrame] = {}
        self.spikes_unwarped_df: dict[str, pl.DataFrame] = {}

    def _ensure_dirs(self):
        """Create lite output subfolders if writing parquet."""
        os.makedirs(os.path.join(self.output_dir, "rr_firing_uw_lite"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "spikes_trains"), exist_ok=True)

    def _calc_fr_rr(self, seg: np.ndarray) -> np.ndarray:
        """
        Same rolling-window FR computation as FRPipeline._calc_fr_rr.
        """
        wl, ss = self.window_length, self.step_size
        nrn, tlen = seg.shape
        nbins = max((tlen - wl) // ss + 1, 1)
        fr = np.zeros((nrn, nbins))
        for b in range(nbins):
            s = b * ss
            e = min(s + wl, tlen)
            fr[:, b] = seg[:, s:e].sum(axis=1) / (wl / 1000.0)
        return fr

    def process_unwarped_only(self):
        """
        Compute unwarped firing rates from start_time to end_time (in ms),
        then divide into time bins relative to start_time. Also emits spike-train tables.
        """
        self._ensure_dirs()

        for data in extract_from_npz(self.npz_path):
            if not (isinstance(data, tuple) and len(data) == 4):
                continue
            spike_array, ds_num, idx, key = data
            ds_name = f"dataset_{ds_num}"
            clean_name = ds_name.replace("dataset_", "").split("_repacked.npz")[0]

            full_df_list = []
            T, R, N, time = spike_array.shape

            # Translate time window into bin indices
            start_idx = self.start_time
            end_idx = self.end_time if self.end_time is not None else time
            # spike df now
            spike_df_list = []
            for t in range(T):
                for tr in range(R):
                    trial_spikes = spike_array[t, tr][:, start_idx:end_idx]
                    # Spike-train table (per ms)
                    n_ms = trial_spikes.shape[1]
                    ms_times = list(range(self.start_time, self.start_time + n_ms))
                    spike_df = pl.DataFrame({
                        **{f"neuron_{i + 1}": trial_spikes[i].tolist() for i in range(N)},
                        "taste": [t] * n_ms,
                        "trial": [tr] * n_ms,
                        "time": ms_times  # time in milliseconds
                    })
                    spike_df_list.append(spike_df)
                    # FR table (per windowed bin)
                    fr_trial = self._calc_fr_rr(trial_spikes)
                    n_bins = fr_trial.shape[1]
                    # Time in bins, relative to start_time
                    time_bins = list(range(n_bins))
                    full_df_list.append(
                        pl.DataFrame({
                            **{f"neuron_{i + 1}": fr_trial[i].tolist() for i in range(N)},
                            "taste": [t] * n_bins,    # <-- Taste now numbered
                            "trial": [tr] * n_bins,
                            "time": time_bins         # <-- Time bins, aligned to start_time
                        })
                    )

            if full_df_list:
                df_final = pl.concat(full_df_list)
                key_name = f"{clean_name}_raw_rr_firing"
                self.fr_unwarped_df[key_name] = df_final

                if self.save_outputs:
                    out_path = os.path.join(self.output_dir, "rr_firing_uw_lite", f"{key_name}.parquet")
                    df_final.write_parquet(out_path)
            if spike_df_list:
                df_spike_final = pl.concat(spike_df_list)
                spike_key = f"{clean_name}_raw_spike_trains"
                self.spikes_unwarped_df[spike_key] = df_spike_final

                if self.save_outputs:
                    out_path = os.path.join(self.output_dir, "spikes_trains", f"{spike_key}.parquet")
                    df_spike_final.write_parquet(out_path)

    def full_pipeline(self) -> dict[str, pl.DataFrame]:
        """
        Run the unwarped-only firing rate pipeline.
        Returns: dict mapping dataset names to polars DataFrames.
        """
        self.process_unwarped_only()
        return self.fr_unwarped_df, self.spikes_unwarped_df
