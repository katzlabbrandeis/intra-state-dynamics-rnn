#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:48:23 2025

New: Class that handles a whole lot:
    - processes the data directly from the RNN
    - now also performs further transforms on firing rate data
    - central area for preprocessing of much of the data we're working with

@author: vincentcalia-bogan

Class: RNNLatentProcessor
==================

Transform RNN latent outputs (or predicted firing rates) + spike-derived changepoints
into analysis-ready tables (epochs), robust PCA summaries, derivatives, and time-warped
versions of each.

Inputs
------
- parquet_dir : Path-like
    Directory of per-timestep parquet files containing either:
      • latent columns:  latent_dim_1, latent_dim_2, ...
      • neuron columns:  neuron_1, neuron_2, ...
    plus: 'taste', 'trial', 'time' columns.
- npz_path : Path-like
    Path to .npz archives with spike arrays (for changepoint extraction).
- info_path : Path-like
    Directory containing metadata ".info" files used to map dataset→tastes.
- pkl_path : Path-like
    Directory with pickled changepoint outputs per dataset (precomputed model).
- taste_replacements : dict or list of (old→new) mappings
    Used to normalize/modify taste labels via `modify_tastes`.

Outputs (in-memory dicts; optionally saved as parquet)
------------------------------------------------------
- epoch_dataframes_dict_unwarped / _warped : dict[str, pl.DataFrame]
    Per dataset tables with rows=(time), columns=(latent/neuron PCs), plus taste/trial/changepoint/time.
- robust_pca_{X}_unwarped/_warped : dict[str, pl.DataFrame]
    PCA cut to cumulative explained variance ≥ X% (per taste×changepoint, split back by trial).
- robust_pca_full_unwarped/_warped : dict[str, pl.DataFrame]
    All PCs kept (per taste×changepoint).
- first_derivatives_{X or 'full'}_* , second_derivatives_{X or 'full'}_*
    Same keying as PCA dicts; derivatives along time axis for each signal column.
- changepoints_dict : dict
    taste×trial changepoint ms for each dataset.

Features
--------
- Detects whether inputs are **latents** or **neuronal** by column names and adapts naming.
- Fixes array shapes consistently as (taste, trial, time, channels), making epoch slicing simple.
- Robust PCA by taste×changepoint with variance threshold, split back by trial.
- PCA named "Robust" (is legacy) -- Robust because of concatenation of trial prior to runtime.
- Optional first/second derivatives on either the thresholded PCA or full PCA.
- Optional time-warp per taste×trial×epoch to a fixed ms length using linear interpolation.
- Structured, timestamped save layout (unwarped/warped, PCA variants, derivatives).

Notes
-----
- Sampling is assumed uniform by `bin_size_ms`. Derivatives therefore ignore actual 'time'.
- Warping uses linear interpolation on the **normalized epoch** (0→1) to `warp_length/bin_size_ms` bins.
- `__init__` initializes flags (`is_latent`, `is_neuron`, `is_rr_fr`) to avoid attribute existence bugs.
- Specialized imports are kept as-is but may be relocated to `core.pre_processing.*` for clarity.

-----------------------------------------------------------------------------
LAYOUT ASSUMPTION (special note on bug propagation-- do not “fix” here)
-----------------------------------------------------------------------------
Upstream exporters now guarantee the array layout expected by this class.
We intentionally DO NOT transpose or reorder axes inside this module.

Expected layout for each dataset array in `self.processed_data`:
    (taste, TIME, CHANNEL, trial)
where:
  - TIME     = the within-trial time axis (bins)
  - CHANNEL  = latent_dim_* or neuron_* columns (features)

Rationale:
  A previous bug placed TIME/CHANNEL in the opposite order. That bug has been
  fixed at the data source. To avoid silent drift, we keep code as-is and
  document the assumption here instead of compensating in code.

If you ever change the exporter and see epoch-slicing errors, first verify
the layout coming from upstream. Only after confirming the upstream change,
update *either* the constants below or add a single `np.transpose` in
`structure_latent_arrays()` (see that method’s comments for guidance).

NOTE: The bug is effectively eliminated by this method, such that the error is not
propagated through from the output of this file. As such, I DO NOT suggest attempting to fix this
as it has already been addressed here.

Example
-------
from core.pre_processing.rnn_latent_processor import RNNLatentProcessor

proc = RNNLatentProcessor(
    parquet_dir="path/to/parquets",
    npz_path="path/to/datasets.npz",
    info_path="path/to/info_dir",
    pkl_path="path/to/pkls",
    taste_replacements={"NaCl": "salt"},
    bin_size_ms=25,
    start_time_ms=1500,
    max_time_ms=4500,
    warp_length=2000,     # optional
    variance_threshold=95 # PCA cumulative variance
)
proc.full_pipeline(
    compute_first_derivative=True,
    compute_second_derivative=False,
    derivative_source="threshold",
    save_outputs=True
)

# Access:
epochs = proc.epoch_dataframes_dict_unwarped
pca95  = getattr(proc, "robust_pca_95_unwarped")

 -----------------------------------------------------------------------------

"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

# general imports
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# specialized imports
# note: must move these to utils
# special pathing import to ensure dependencies: probably not needed...
from core.io.import_paths import ensure_src_on_path
from core.utils.extract_npz import extract_from_npz
from core.utils.find_extract_info import find_copy_h5info, modify_tastes, process_info_files
from core.utils.read_parquets import all_nrns_to_df, read_parquet_files_into_dict
from core.utils.unpkl_generator import extract_valid_changepoints

ensure_src_on_path()


class RNNLatentProcessor:
    """
    See module docstring for details. Only salient param notes here:
    - warp_length: int|None, warped epoch duration in ms (None disables warping)
    - variance_threshold: float, PCA cumulative variance (%)
    """

    def __init__(
        self,
        parquet_dir,
        npz_path,
        info_path,
        pkl_path,
        taste_replacements,
        bin_size_ms: int = 25,  # standardized bin size
        start_time_ms: int = 1500,  # 500 ms pre-stim, which for abu's data, is at 2000 ms
        max_time_ms: int = 4500,  # typically do not train RNN beyond this duration
        save_dir: Optional[str | Path] = None,
        warp_length: Optional[int] = None,           # fixed warp duration; set typically to 1000 ms
        variance_threshold: float = 95.0,    # PCA variance threshold
    ):
        # paths / config -------------------------------------------------------
        self.parquet_dir = Path(parquet_dir)
        self.npz_path = Path(npz_path)
        self.info_path = Path(info_path)
        self.pkl_path = Path(pkl_path)
        self.save_dir = Path(save_dir) if save_dir else None

        self.taste_replacements = taste_replacements
        self.bin_size_ms = int(bin_size_ms)
        self.start_time_ms = int(start_time_ms)
        self.max_time_ms = int(max_time_ms)
        self.warp_length = int(warp_length) if warp_length is not None else None
        self.variance_threshold = float(variance_threshold)
        # working stores -------------------------------------------------------
        self.taste_latent: Dict[str, pl.DataFrame] = {}
        self.processed_data: Dict[str, np.ndarray] = {}
        self.changepoints_dict: Dict[str, np.ndarray] = {}
        # state flags ----------------------------------------------------------
        self.is_latent: bool = False
        self.is_neuron: bool = False
        self.is_rr_fr: bool = False  # special neuron naming for "rr_firing" datasets

        # unwarped outputs
        self.epoch_dataframes_dict_unwarped: Dict[str, pl.DataFrame] = {}
        self.robust_pca_thresh_unwarped: Dict[str, pl.DataFrame] = {}
        self.robust_pca_full_unwarped: Dict[str, pl.DataFrame] = {}
        self.first_derivs_thresh_unwarped: Dict[str, pl.DataFrame] = {}
        self.second_derivs_thresh_unwarped: Dict[str, pl.DataFrame] = {}

        # warped outputs
        self.epoch_dataframes_dict_warped: Dict[str, pl.DataFrame] = {}
        self.robust_pca_thresh_warped: Dict[str, pl.DataFrame] = {}
        self.robust_pca_full_warped: Dict[str, pl.DataFrame] = {}
        self.first_derivs_thresh_warped: Dict[str, pl.DataFrame] = {}
        self.second_derivs_thresh_warped: Dict[str, pl.DataFrame] = {}

    def read_parquet_files(self):
        """Populate `self.taste_latent` as { dataset_name: DataFrame }."""
        self.taste_latent = read_parquet_files_into_dict(self.parquet_dir)

    @staticmethod
    def parse_dataset_name(name: str) -> Tuple[str, str]:
        """
        NOTE: May become it's own util due to usefulness

        Split a compound dataset key into (core_dataset_name, analysis_type).
        Assumes the core name ends with YYMMDD_HHMMSS; returns ('name', '_suffix'|'')

        Examples
        --------
        "dataset_ABC123_240101_123456_latent_unwarped" -> ("dataset_ABC123_240101_123456", "_latent_unwarped")
        """
        pattern = re.compile(r'^(.+?\d{6}_\d{6})(_.+)$')
        m = pattern.match(name)
        if m:
            return m.group(1), m.group(2)
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], f'_{parts[1]}'
        return name, ''

    def get_latent_cols(self, df: pl.DataFrame) -> list[str]:
        """
        Returns a list of signal columns as 'latent_cols', whether neuron or latent.
        Sets flags: self.is_latent or self.is_neuron accordingly.
        Raises an error if both or neither types are present.
        """
        latent_cols = [col for col in df.columns if col.startswith("latent_dim_")]
        neuron_cols = [col for col in df.columns if col.startswith("neuron_")]

        if latent_cols and neuron_cols:
            raise ValueError("DataFrame contains both latent and neuron columns — ambiguous format.")
        elif not latent_cols and not neuron_cols:
            raise ValueError("DataFrame contains neither latent nor neuron columns — unknown format.")
        elif latent_cols:
            self.is_latent = True
            self.is_neuron = False
            return latent_cols
        else:
            self.is_latent = False
            self.is_neuron = True
            latent_cols = neuron_cols
            # this has to be done to maintain var naming compatibility-- unfortunately nessecary patch
            return latent_cols

    def structure_latent_arrays(self):
        """
        Build `self.processed_data` as arrays with shape:
            (num_tastes, num_trials, num_channels, num_time_steps)

        NOTE: I am aware that there is a naming bug with mismatch of channels and time.
        UNDERSTAND that this class will effectively fix it, such that naming schema
        for ALL output dataframes from this script are correct.

        Reasons that I have decided/cannot really NOT to address it here:
          • We rely on upstream to supply columns in the correct orientation.
          • This method DOES NOT transpose or reorder axes to “fix” TIME/CHANNEL.
          • Downstream epoch slicing assumes TIME is the 3rd dimension and CHANNEL
            is the 4th; see `split_epochs_by_changepoints()`.

        If an upstream exporter ever flips TIME and CHANNEL, **do not** hack here.
        Fix the exporter. If you absolutely must compensate locally, replace the
        array assignment with a single transpose at the very end, e.g.:

            # arr currently (taste, trial, CHANNEL, TIME)  → we require (taste, trial, TIME, CHANNEL)
            # arr = arr.transpose(0, 1, 3, 2)  # <-- enable ONLY if upstream layout changes again

        Any local change here must be mirrored anywhere shapes are unpacked.

        """
        # refactored slightly for speed
        for dataset_name, df in self.taste_latent.items():
            unique_tastes = df["taste"].unique()
            unique_trials = df["trial"].unique()
            latent_columns = self.get_latent_cols(df)  # calling the func above now
            latent_dim = len(latent_columns)

            # Find number of time steps
            num_time_steps = len(
                df.filter(
                    (pl.col("trial") == unique_trials[0])
                    & (pl.col("taste") == unique_tastes[0])
                )
            )

            # Dataset array: (taste, trial, latent_dim, time)
            dataset_array = np.empty(
                (len(unique_tastes), len(unique_trials), latent_dim, num_time_steps)
            )

            # --- Pre-group by (taste, trial) once for speed
            trial_lookup = {}
            for taste in unique_tastes:
                for trial in unique_trials:
                    key = (taste, trial)
                    trial_data = df.filter(
                        (pl.col("taste") == taste) & (pl.col("trial") == trial)
                    )
                    if trial_data.height > 0:
                        trial_lookup[key] = (
                            trial_data.select(latent_columns).to_numpy().T
                        )  # (latent_dim, time)

            # --- Now fast loop to fill array
            for taste_idx, taste in enumerate(unique_tastes):
                for trial_idx, trial in enumerate(unique_trials):
                    key = (taste, trial)
                    if key in trial_lookup:
                        dataset_array[taste_idx, trial_idx, :, :] = trial_lookup[key]
                    else:
                        print(f"Warning: Missing data for taste {taste} trial {trial}")

            self.processed_data[dataset_name] = dataset_array
    # -------------------------------------------------------------------------
    # Changepoints
    # -------------------------------------------------------------------------

    def extract_changepoints(self):
        """
        Populate `self.changepoints_dict` per dataset core name.
        Expects `extract_from_npz` to yield tuples: (spike_array, dataset_num, index, key).
        """
        for data in extract_from_npz(self.npz_path):
            if isinstance(data, tuple):
                spike_array, dataset_num, index, key = data
                dataset_name = f"dataset_{dataset_num}"
                dataset_clean = dataset_name.replace("dataset_", "").split("_repacked.npz")[0]
                dataset_tastes = process_info_files(self.info_path, dataset_num)
                modified_tastes = modify_tastes(dataset_tastes, self.taste_replacements)
                extracted_pkl = extract_valid_changepoints(
                    self.pkl_path, spike_array, dataset_num, index, key
                )

                if extracted_pkl is not None:
                    try:
                        # Convention: changepoints at column 3 across trials/tastes (shape should not vary if CPs detected)
                        changepoints = extracted_pkl[
                            :, 3
                        ]
                        self.changepoints_dict[dataset_clean] = changepoints
                    except IndexError:
                        print(
                            f"Index error with dataset {dataset_name}: shape {extracted_pkl.shape}"
                        )
                else:
                    print(f" [RNNlatproce.] No valid changepoints found for dataset {dataset_name}")

    # method that saves changepoints so the entire processor doesn't need to be run again and again
    def extract_changepoints_dict(self, save_outputs: bool = False) -> dict:
        """
        Extract only changepoints (populates self.changepoints_dict). Optionally persist each dataset’s
        changepoints under `<save_dir>/changepoints/<core>_changepoints.pkl`.
        Standalone runner for grabbign changepoints without running the rest of the heavy script.
        """
        # (re)initialize
        self.changepoints_dict = {}
        # delegate to original:
        self.extract_changepoints()
        if save_outputs:
            save_path = Path(self.save_dir) / "changepoints"
            save_path.mkdir(parents=True, exist_ok=True)
            # optionally persist for future runs
            for dataset_name, changepoints in self.changepoints_dict.items():
                clean_name = dataset_name.replace("dataset_", "").split("_repacked.npz")[0]
                out_path = save_path / f"{clean_name}_changepoints.pkl"
                if not out_path.exists():
                    with open(out_path, "wb") as f:
                        pickle.dump(changepoints, f)

        return self.changepoints_dict
    # -------------------------------------------------------------------------
    # Epoch building (unwarped)
    # -------------------------------------------------------------------------

    def split_epochs_by_changepoints(self):
        """
        Build unwarped epoch DataFrames per dataset, keyed as:
          "<core>_latent_unwarped" or "<core>_firing_unwarped"

        Assumes each dataset array is shaped (taste, time, channels, trial).
        (see:  num_tastes, num_time_steps, latent_dim, num_trials = data_array.shape )
        We slice epochs along the TIME axis only:

            epoch_data = data_array[taste_idx, start_idx:end_idx, :, trial_idx]  # -> (time, channels)

        If you see IndexErrors or empty epochs where you expect data, the most
        likely cause is an upstream TIME/CHANNEL swap. Verify one small example:

            # quick diagnostic (run in a notebook):
            dataset_name, data_array = next(iter(self.processed_data.items()))
            print("shape:", data_array.shape)
            print("example row:", data_array[0, 0, :5, :3])  # should show 5 timesteps × 3 channels

        Fix upstream; do not inject transposes here unless absolutely unavoidable.
        """
        standardized_changepoints_dict = {
            key.replace("dataset_", "").split("_repacked.npz")[0]: value
            for key, value in self.changepoints_dict.items()
        }

        for dataset_name, data_array in self.processed_data.items():
            # Mark neuron type special-case from name
            self.is_rr_fr = "rr_firing" in dataset_name
            core_dataset_name, analysis_type = self.parse_dataset_name(dataset_name)

            if self.is_latent or self.is_neuron:
                # `core_dataset_name` already correctly extracted
                pass
            else:
                raise ValueError(
                    "Neither is_latent nor is_neuron is set. See get_latent_cols and structure_latent_arrays to resolve the issue."
                )

            changepoints = standardized_changepoints_dict.get(core_dataset_name, None)
            if changepoints is None:
                print(f"No changepoints for {core_dataset_name}, skipping.")
                continue
            # Array is (taste, time, channels, trials) -- this is not nessecarily correct.
            # I forget why this is built this way, but DO NOT change it bc it works.
            num_tastes, num_time_steps, latent_dim, num_trials = data_array.shape
            all_epochs = []
            time_in_ms = [
                (t * self.bin_size_ms) + self.start_time_ms
                for t in range(num_time_steps)
            ]

            if len(changepoints) != num_tastes:
                print(f"Mismatch in taste dimension for dataset {core_dataset_name}")
                continue

            for taste_idx in range(num_tastes):
                # Per-taste list of arrays of CPs per trial
                if len(changepoints[taste_idx]) < num_trials:
                    print(
                        f"Mismatch in trial dimension for dataset {core_dataset_name}, taste {taste_idx}"
                    )
                    continue

                for trial_idx in range(num_trials):
                    trial_changepoints = changepoints[taste_idx][trial_idx]
                    start_idx = 0
                    # Iterate each epoch boundary within the trial
                    for cp_idx, changepoint_ms in enumerate(trial_changepoints):
                        # end index is first time >= cp_ms, else end
                        end_idx = next(
                            (
                                i
                                for i, t in enumerate(time_in_ms)
                                if t >= changepoint_ms
                            ),
                            num_time_steps,
                        )

                        epoch_data = data_array[
                            taste_idx, start_idx:end_idx, :, trial_idx
                        ]  # (time, latent)
                        if self.is_latent:
                            schema = [f"latent_dim_{i + 1}" for i in range(latent_dim)]  # elim 0 idxing
                        elif self.is_neuron:
                            schema = [f"neuron_{i + 1}" for i in range(latent_dim)]  # elim 0 idxing
                        else:
                            raise ValueError("Data type flags not set. Call get_latent_cols() first.")

                        epoch_df = pl.DataFrame(epoch_data, schema=schema)

                        # epoch_df = pl.DataFrame(
                        #     epoch_data,
                        #     schema=[f"latent_dim_{i}" for i in range(latent_dim)],
                        # )
                        # Build schema based on latents vs neurons
                        epoch_df = epoch_df.with_columns(
                            [
                                pl.Series("taste", [taste_idx] * len(epoch_df)),
                                pl.Series("trial", [trial_idx] * len(epoch_df)),
                                pl.Series("changepoint", [cp_idx] * len(epoch_df)),
                                # really should be state, but naming convention
                                # and backwards compatibility prevents me.
                                pl.Series("time", time_in_ms[start_idx:end_idx]),
                            ]
                        )
                        all_epochs.append(epoch_df)
                        start_idx = end_idx

                    if start_idx < num_time_steps:
                        # Tail segment after last CP
                        epoch_data = data_array[
                            taste_idx, start_idx:num_time_steps, :, trial_idx
                        ]
                        if self.is_latent:
                            schema = [f"latent_dim_{i + 1}" for i in range(latent_dim)]  # eliminate 0 - idx'ing
                        elif self.is_neuron:
                            schema = [f"neuron_{i + 1 }" for i in range(latent_dim)]  # eliminate 0 - idx'ing
                        else:
                            raise ValueError("Data type flags not set. Call get_latent_cols() first.")

                        epoch_df = pl.DataFrame(epoch_data, schema=schema)

                        # epoch_df = pl.DataFrame(
                        #     epoch_data,
                        #     schema=[f"latent_dim_{i}" for i in range(latent_dim)],
                        # )
                        epoch_df = epoch_df.with_columns(
                            [
                                pl.Series("taste", [taste_idx] * len(epoch_df)),
                                pl.Series("trial", [trial_idx] * len(epoch_df)),
                                pl.Series(
                                    "changepoint",
                                    [len(trial_changepoints)] * len(epoch_df),
                                ),
                                pl.Series("time", time_in_ms[start_idx:num_time_steps]),
                            ]
                        )
                        all_epochs.append(epoch_df)
            # if all_epochs:
            #     self.epoch_dataframes_dict_unwarped[
            #         f"{core_dataset_name}_latent_unwarped"
            #     ] = pl.concat(all_epochs)
            if all_epochs:
                if self.is_latent:
                    suffix = "_latent_unwarped"
                elif self.is_neuron:
                    suffix = "_firing_unwarped"
                else:
                    raise ValueError("Data type flags not set. Call get_latent_cols() first.")

                self.epoch_dataframes_dict_unwarped[
                    f"{core_dataset_name}{suffix}"
                ] = pl.concat(all_epochs)
            else:
                print(f"No valid epochs for dataset {dataset_name}, skipping.")
    # -------------------------------------------------------------------------
    # Warping (per taste×trial×epoch)
    # -------------------------------------------------------------------------

    def warp_all_outputs(self):
        """
        If `warp_length` is set, time-warp each unwarped DataFrame (epochs, PCA, derivatives)
        per taste×trial×epoch to fixed duration `warp_length` using linear interpolation.
        """
        if not self.warp_length:
            return

        # number of warped bins = warp_length / bin_size_ms
        n_bins = max(int(self.warp_length // self.bin_size_ms), 1)  # double check +1 -- idk about it

        # standardized changepoints by core name
        standardized_cp = {
            key.replace("dataset_", "").split("_repacked.npz")[0]: value
            for key, value in self.changepoints_dict.items()
        }

        # helper to warp any DataFrame by taste/trial/changepoint segments
        def warp_df(df: pl.DataFrame, core: str) -> pl.DataFrame:
            dims = [c for c in df.columns if c.startswith(
                "PC_") or c.startswith("latent_dim_") or c.startswith("neuron_")]
            warped_segments = []
            tastes = df["taste"].unique().to_list()
            for taste_idx, taste in enumerate(tastes):
                taste_df = df.filter(pl.col("taste") == taste)
                trials = taste_df["trial"].unique().to_list()
                cp_per_taste = standardized_cp[core][taste_idx]
                n_cps = cp_per_taste.shape[1]
                for trial_idx, trial in enumerate(trials):
                    trial_df = taste_df.filter(pl.col("trial") == trial)
                    cps = cp_per_taste[trial_idx]
                    for eid in range(n_cps + 1):
                        if eid == 0:
                            start = self.start_time_ms
                            end = cps[0]
                        elif eid < n_cps:
                            start = cps[eid - 1]
                            end = cps[eid]
                        else:
                            start = cps[-1]
                            end = self.max_time_ms
                        seg = trial_df.filter(
                            (pl.col("time") >= start) & (pl.col("time") < end)
                        )
                        if seg.is_empty():
                            continue
                        t_rel = seg["time"].to_numpy().astype(float) - start
                        if len(t_rel) < 2:
                            continue  # just handle that error rq
                        # Normalize segment to 0..1 → interpolate to n_bins
                        src = np.linspace(0, 1, len(t_rel))  # should scale between 0 and 1...
                        tgt = np.linspace(0, 1, n_bins)
                        warped_vals = {}
                        for dim in dims:
                            v = seg[dim].to_numpy()
                            # linear warp instead of nearest
                            f = interp1d(src, v, kind="linear", fill_value="extrapolate")
                            warped_vals[dim] = f(tgt).tolist()
                        warped_df = pl.DataFrame(warped_vals)
                        # rebuild time in ms increments-- two ways to do it:
                        times = (start + np.arange(n_bins) * self.bin_size_ms).tolist()
                        # or: (should be inclusive of the end bin?)
                        # times = np.linspace(start, end, n_bins).tolist() # eval later...

                        warped_df = warped_df.with_columns([
                            pl.Series("taste", [taste] * n_bins),
                            pl.Series("trial", [trial] * n_bins),
                            pl.Series("changepoint", [eid] * n_bins),
                            pl.Series("time", times),
                        ])
                        warped_segments.append(warped_df)
            return pl.concat(warped_segments) if warped_segments else pl.DataFrame()
        # NOTE: post-fix for "_warped" / "_unwarped" is critically important keyword for
        # plotting functions later on. As such, do not eliminate either postfix.

        # 1) Epochs
        new_epochs = {}
        for key, df in self.epoch_dataframes_dict_unwarped.items():
            # find base dataset name by matching prefix in standardized_cp
            base = next((b for b in standardized_cp if key.startswith(b)), None)
            if base is None:
                raise KeyError(f"Cannot find base dataset for key {key}")
            if self.is_latent:
                new_key = f"{base}_latent_warped"
            elif self.is_neuron:
                new_key = f"{base}_firing_warped"
            else:
                raise ValueError("Data type flags not set. Call get_latent_cols().")
            new_epochs[new_key] = warp_df(df, base)
        self.epoch_dataframes_dict_warped = new_epochs

        # 2) PCA thresholded
        thresh = int(self.variance_threshold)
        pca_unw = getattr(self, f"robust_pca_{thresh}_unwarped", {})
        new_pca = {}
        for key, df in pca_unw.items():
            base = next((b for b in standardized_cp if key.startswith(b)), None)
            if base is None:
                raise KeyError(f"Cannot find base dataset for key {key}")
            new_pca[f"{base}_pca_{thresh}_warped"] = warp_df(df, base)
        setattr(self, f"robust_pca_{thresh}_warped", new_pca)

        # 3) PCA full
        new_full = {}
        for key, df in self.robust_pca_full_unwarped.items():
            base = next((b for b in standardized_cp if key.startswith(b)), None)
            if base is None:
                raise KeyError(f"Cannot find base dataset for key {key}")
            new_full[f"{base}_pca_full_warped"] = warp_df(df, base)
        self.robust_pca_full_warped = new_full

        # 4) Derivatives (if present)
        fd_attr = f"first_derivatives_{thresh}_unwarped"
        if hasattr(self, fd_attr):
            src_fd = getattr(self, fd_attr)
            new_fd = {}
            for key, df in src_fd.items():
                base = next((b for b in standardized_cp if key.startswith(b)), None)
                if base is None:
                    raise KeyError(f"Cannot find base dataset for key {key}")
                new_fd[f"{base}_first_derivatives_{thresh}_warped"] = warp_df(df, base)
            setattr(self, f"first_derivatives_{thresh}_warped", new_fd)

        sd_attr = f"second_derivatives_{thresh}_unwarped"
        if hasattr(self, sd_attr):
            src_sd = getattr(self, sd_attr)
            new_sd = {}
            for key, df in src_sd.items():
                base = next((b for b in standardized_cp if key.startswith(b)), None)
                if base is None:
                    raise KeyError(f"Cannot find base dataset for key {key}")
                new_sd[f"{base}_second_derivatives_{thresh}_warped"] = warp_df(df, base)
            setattr(self, f"second_derivatives_{thresh}_warped", new_sd)

    # -------------------------------------------------------------------------
    # PCA + derivatives
    # -------------------------------------------------------------------------
    def run_robust_pca_analysis(
        self,
        return_dicts: bool = True,
        # variance_threshold=95.0, # now a global
        compute_first_derivative: bool = False,
        compute_second_derivative: bool = False,
        derivative_source: str = "threshold",
    ):
        """
        Perform PCA on concatenated data grouped by taste and changepoint, then split results back accordingly.
        Optionally computes first and second derivatives.

        Parameters
        ----------
        return_dicts : bool, optional
            Whether to return the PCA dictionaries.
        variance_threshold : float, optional
            The cumulative explained variance percentage to retain.
        compute_first_derivative : bool, optional
            Whether to compute and store first derivatives.
        compute_second_derivative : bool, optional
            Whether to compute and store second derivatives.
        derivative_source : str, optional
            "threshold" (default) to compute derivatives on robust_pca_<threshold>;
            "full" to compute derivatives on robust_pca_full.

        Returns
        -------
        (dict, dict) or None
            (robust_pca_threshold, robust_pca_full) if return_dicts=True, else None
        """
        pca_lat_dict_thresh = {}
        pca_lat_dict_full = {}

        for dataset_name, df in self.epoch_dataframes_dict_unwarped.items():
            unique_tastes = df["taste"].unique().to_list()
            unique_changepoints = df["changepoint"].unique().to_list()

            processed_dataframes_thresh = []
            processed_dataframes_full = []

            for taste in unique_tastes:
                for changepoint in unique_changepoints:
                    filtered_df = df.filter(
                        (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )

                    if filtered_df.is_empty():
                        continue
                    # Accepts latent_* or neuron_* as channels
                    latent_columns = [
                        col for col in filtered_df.columns if ("latent_dim_" in col) or ("neuron_" in col)
                    ]
                    if not latent_columns:
                        continue

                    unique_trials = filtered_df["trial"].unique().to_list()
                    trial_data_list = [
                        filtered_df.filter(pl.col("trial") == trial)
                        .select(latent_columns)
                        .to_numpy()
                        for trial in unique_trials
                    ]

                    concatenated_data = np.vstack(trial_data_list)

                    if concatenated_data.shape[0] < 2:
                        continue

                    pca = PCA()
                    transformed_data = pca.fit_transform(concatenated_data)
                    explained_variance = pca.explained_variance_ratio_ * 100

                    cumulative_variance = np.cumsum(explained_variance)
                    num_pcs_thresh = (
                        np.argmax(cumulative_variance > int(self.variance_threshold)) + 1
                    )
                    # Split back by trial
                    split_indices = np.cumsum(
                        [arr.shape[0] for arr in trial_data_list]
                    )[:-1]
                    split_pca_data = np.split(transformed_data, split_indices)

                    pc_names_full = [
                        f"PC_{i+1}" for i in range(transformed_data.shape[1])
                    ]
                    explained_var_cols_full = [
                        f"explained_variance_{pc}" for pc in pc_names_full
                    ]

                    pc_names_thresh = [f"PC_{i+1}" for i in range(num_pcs_thresh)]
                    explained_var_cols_thresh = [
                        f"explained_variance_{pc}" for pc in pc_names_thresh
                    ]

                    for trial, trial_pca_data in zip(unique_trials, split_pca_data):
                        trial_metadata = filtered_df.filter(
                            pl.col("trial") == trial
                        ).select(["trial", "time"])

                        full_pca_df = pl.DataFrame(
                            np.hstack(
                                [
                                    trial_pca_data,
                                    np.tile(
                                        explained_variance, (len(trial_pca_data), 1)
                                    ),
                                ]
                            ),
                            schema=pc_names_full + explained_var_cols_full,
                        ).with_columns(
                            [
                                pl.lit(taste).alias("taste"),
                                pl.lit(changepoint).alias("changepoint"),
                            ]
                        )
                        full_pca_df = pl.concat(
                            [trial_metadata, full_pca_df], how="horizontal"
                        )
                        processed_dataframes_full.append(full_pca_df)

                        pca_df_thresh = pl.DataFrame(
                            np.hstack(
                                [
                                    trial_pca_data[:, :num_pcs_thresh],
                                    np.tile(
                                        explained_variance[:num_pcs_thresh],
                                        (len(trial_pca_data), 1),
                                    ),
                                ]
                            ),
                            schema=pc_names_thresh + explained_var_cols_thresh,
                        ).with_columns(
                            [
                                pl.lit(taste).alias("taste"),
                                pl.lit(changepoint).alias("changepoint"),
                            ]
                        )
                        pca_df_thresh = pl.concat(
                            [trial_metadata, pca_df_thresh], how="horizontal"
                        )
                        processed_dataframes_thresh.append(pca_df_thresh)

            def align_dataframes(dataframes):
                # Ensure all frames share the same columns (fill missing with NaN)
                all_columns = set()
                for df in dataframes:
                    all_columns.update(df.columns)
                all_columns = sorted(all_columns)

                aligned_dfs = []
                for df in dataframes:
                    missing_columns = set(all_columns) - set(df.columns)
                    for col in missing_columns:
                        df = df.with_columns(pl.Series(col, [np.nan] * len(df)))
                    df = df.select(all_columns)
                    aligned_dfs.append(df)
                return aligned_dfs

            if processed_dataframes_thresh:
                aligned_thresh = align_dataframes(processed_dataframes_thresh)
                thr = int(self.variance_threshold)
                if self.is_latent:
                    obj_name_unw_clean = dataset_name.split("_latent_unwarped")[0]
                elif self.is_neuron:
                    obj_name_unw_clean = dataset_name.split("_firing_unwarped")[0]
                else:
                    raise ValueError("Data type flags not set. Call get_latent_cols() first.")
                pca_lat_dict_thresh[f"{obj_name_unw_clean}_pca_{thr}_unwarped"] = pl.concat(aligned_thresh)

            if processed_dataframes_full:
                aligned_full = align_dataframes(processed_dataframes_full)
                if self.is_latent:
                    obj_name_w_clean = dataset_name.split("_latent_unwarped")[0]
                elif self.is_neuron:
                    obj_name_w_clean = dataset_name.split("_firing_unwarped")[0]
                else:
                    raise ValueError("Data type flags not set. Call get_latent_cols() first.")
                # obj_name_w_clean = dataset_name.split("_latent_unwarped")[0]
                pca_lat_dict_full[f"{obj_name_w_clean}_pca_full_unwarped"] = pl.concat(aligned_full)

        setattr(self,
                f"robust_pca_{int(self.variance_threshold)}_unwarped",
                pca_lat_dict_thresh)
        setattr(self,
                "robust_pca_full_unwarped",
                pca_lat_dict_full)

        # Derivative source selector
        if derivative_source == "threshold":
            source_dict = pca_lat_dict_thresh
            source_name = str(int(self.variance_threshold))
        elif derivative_source == "full":
            source_dict = pca_lat_dict_full
            source_name = "full"
        else:
            raise ValueError(
                "Invalid derivative_source. Must be 'threshold' or 'full'."
            )

        if compute_first_derivative:
            first_derivative_dict = {
                f"{key}_first_derivatives_{source_name}_unwarped":
                    self.compute_first_derivative(df)
                for key, df in source_dict.items()
            }
            setattr(
                self,
                f"first_derivatives_{source_name}_unwarped",
                first_derivative_dict
            )

        # Compute second derivatives with descriptive keys
        if compute_second_derivative:
            second_derivative_dict = {
                f"{key}_second_derivatives_{source_name}_unwarped":
                    self.compute_second_derivative(df)
                for key, df in source_dict.items()
            }
            setattr(
                self,
                f"second_derivatives_{source_name}_unwarped",
                second_derivative_dict
            )

        return (pca_lat_dict_thresh, pca_lat_dict_full) if return_dicts else None
    # -------------------------------------------------------------------------
    # Numeric helpers
    # -------------------------------------------------------------------------

    def compute_first_derivative(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the first derivative of latent dimension columns in a Polars DataFrame.
        Assumes uniform time spacing (ignores actual 'time' values).
        Does not touch meta cols.
        """
        meta_cols = ["taste", "trial", "changepoint", "time"]
        latent_cols = [col for col in df.columns if col not in meta_cols]

        data = df.select(latent_cols).to_numpy()
        first_deriv = np.gradient(data, axis=0)

        result_df = df.with_columns(
            [
                pl.Series(name=col, values=first_deriv[:, j])
                for j, col in enumerate(latent_cols)
            ]
        )
        return result_df

    def compute_second_derivative(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the second derivative of latent dimension columns in a Polars DataFrame.
        Assumes uniform time spacing (ignores actual 'time' values).
        """
        meta_cols = ["taste", "trial", "changepoint", "time"]
        latent_cols = [col for col in df.columns if col not in meta_cols]

        data = df.select(latent_cols).to_numpy()
        first_deriv = np.gradient(data, axis=0)
        second_deriv = np.gradient(first_deriv, axis=0)

        result_df = df.with_columns(
            [
                pl.Series(name=col, values=second_deriv[:, j])
                for j, col in enumerate(latent_cols)
            ]
        )
        return result_df
    # -------------------------------------------------------------------------
    # Save / Orchestration
    # -------------------------------------------------------------------------

    def save_analysis_outputs(self, tld):
        """
        Saves all analysis outputs (unwarped and warped) into organized subdirectories as Parquet files.

        Parameters
        ----------
        tld : str or Path
            Top-level directory to save outputs into.
        """
        tld = Path(tld)
        tld.mkdir(parents=True, exist_ok=True)
        thr = int(self.variance_threshold)

        outputs = {
            "raw_output": (  # changed from epoch_dataframes to raw_ouptut; hopefully that doesn't break anything...
                self.epoch_dataframes_dict_unwarped,
                self.epoch_dataframes_dict_warped
            ),
            f"robust_pca_{thr}": (
                getattr(self, f"robust_pca_{thr}_unwarped", {}),
                getattr(self, f"robust_pca_{thr}_warped", {})
            ),
            "robust_pca_full": (
                self.robust_pca_full_unwarped,
                self.robust_pca_full_warped
            ),
            f"first_derivatives_{thr}": (
                getattr(self, f"first_derivatives_{thr}_unwarped", {}),
                getattr(self, f"first_derivatives_{thr}_warped", {})
            ),
            f"second_derivatives_{thr}": (
                getattr(self, f"second_derivatives_{thr}_unwarped", {}),
                getattr(self, f"second_derivatives_{thr}_warped", {})
            ),
        }
        for name, (unw, w) in outputs.items():
            for suffix, d in (("_unwarped", unw), ("_warped", w)):
                subdir = tld / f"{name}{suffix}"
                subdir.mkdir(parents=True, exist_ok=True)
                if not d:
                    print(f"[WARNING] No {suffix[1:]} data for {name}, skipping.")
                    continue
                for ds_key, df in d.items():
                    # Extract core dataset (prefix with YYMMDD_HHMMSS)
                    match = re.match(r"^(.*?\d{6}_\d{6})", ds_key)
                    if match:
                        core = match.group(1)
                    else:
                        raise ValueError(f"Could not extract core dataset name from '{ds_key}'")
                        core = ds_key.split(suffix)[0]
                    if self.is_latent:
                        type = "rnn_latent"
                    elif self.is_neuron:
                        if self.is_rr_fr:
                            type = "rr_fr"
                        else:
                            type = "rnn_pred_fr"
                    else:
                        raise ValueError("Data type flags not set. Call get_latent_cols() first.")
                    # core = ds_key.split(suffix)[0]
                    out = subdir / f"{core}_{type}_{name}{suffix}.parquet"
                    df.write_parquet(out)
                    print(f"Saved {core} [{type}_{name}{suffix}] to {out}")

    def full_pipeline(
        self,
        compute_first_derivative=False,
        compute_second_derivative=False,
        derivative_source="threshold",
        return_derivatives=True,
        save_outputs=False
    ):
        """
        Full pipeline: preprocess, PCA, warp, and save.
        Returns separate dicts for each analysis type (unwarped and, optionally, warped).
        """
        # Preprocessing
        self.read_parquet_files()
        self.structure_latent_arrays()
        self.extract_changepoints()
        self.split_epochs_by_changepoints()

        # PCA and derivatives
        self.run_robust_pca_analysis(
            compute_first_derivative=compute_first_derivative,
            compute_second_derivative=compute_second_derivative,
            derivative_source=derivative_source
        )
        # Warping
        self.warp_all_outputs()

        # Optional saving
        if save_outputs and self.save_dir:
            self.save_analysis_outputs(self.save_dir)

        # Build and return separate dicts
        thresh = int(self.variance_threshold)

        # Epochs
        epochs_unw = self.epoch_dataframes_dict_unwarped
        epochs_w = self.epoch_dataframes_dict_warped if return_derivatives else {}

        # PCA thresholded
        pca_thresh_unw = getattr(self, f"robust_pca_{thresh}_unwarped", {})
        pca_thresh_w = getattr(self, f"robust_pca_{thresh}_warped", {}) if return_derivatives else {}

        # PCA full
        pca_full_unw = self.robust_pca_full_unwarped
        pca_full_w = self.robust_pca_full_warped if return_derivatives else {}

        # Derivatives
        fd_unw = getattr(self, f"first_derivatives_{thresh}_unwarped", {}) if compute_first_derivative else {}
        sd_unw = getattr(self, f"second_derivatives_{thresh}_unwarped", {}) if compute_second_derivative else {}
        fd_w = getattr(self, f"first_derivatives_{thresh}_warped", {}) if (
            compute_first_derivative and return_derivatives) else {}
        sd_w = getattr(self, f"second_derivatives_{thresh}_warped", {}) if (
            compute_second_derivative and return_derivatives) else {}

        return (
            epochs_unw,
            pca_thresh_unw,
            pca_full_unw,
            fd_unw,
            sd_unw,
            epochs_w,
            pca_thresh_w,
            pca_full_w,
            fd_w,
            sd_w,
        )
