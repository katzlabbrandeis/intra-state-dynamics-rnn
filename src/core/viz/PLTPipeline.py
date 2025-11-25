import os
import re
from typing import Dict, List, Tuple, Any

import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from joblib import Parallel, delayed


class PlottingPipeline:
    """
    Pipeline for generating plots from Polars DataFrame outputs, using external changepoints.

    Parameters
    ----------
    standardized_changepoints_dict : dict
        Mapping core dataset names to nested changepoint arrays.
    output_dir : str, optional
        Directory to save generated plots.
    start_time_ms : int, default=1500
        Lower bound of the time window (ms) for plotting.
    end_time_ms : int, default=4500
        Upper bound of the time window (ms) for plotting.
    stim_time_ms : int, default=2000
        Time of stimulus delivery (ms) to mark on plots.
    plot_configs : dict
        Additional configuration for plot styles or parameters.
    """
    # Columns always present in DataFrame
    META_COLS = ["taste", "trial", "changepoint", "time"]

    def __init__(
        self,
        standardized_changepoints_dict: Dict[str, Any],
        output_dir: str = None,
        start_time_ms: int = 1500,
        end_time_ms: int = 4500,
        stim_time_ms: int = 2000,
        **plot_configs
    ):
        self.standardized_changepoints_dict = standardized_changepoints_dict
        self.output_dir = output_dir
        self.start_time_ms = start_time_ms
        self.end_time_ms = end_time_ms
        self.stim_time_ms = stim_time_ms
        self.plot_configs = plot_configs
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize default style parameters
        self._init_plot_style()
    
    def run(self, dataset_dict: Dict[str, pl.DataFrame]):
        """
        Iterate over each dataset key and DataFrame, using external changepoints.
        Detects whether dataset is 'warped' or 'unwarped' from its analysis_type.
        """
        for dataset_name, df in tqdm(dataset_dict.items(), desc='Processing datasets'):
            core_name, analysis_type = self.parse_dataset_name(dataset_name)
            cps = self.standardized_changepoints_dict.get(core_name)
            if cps is None:
                print(f'No changepoints for {core_name}, skipping plots.')
                continue
            data_cols = self.get_data_columns(df)
            meta_cols = self.get_meta_columns(df)
            # Determine warp flags
            kind = analysis_type.lower()
            is_unwarped = kind.endswith('_unwarped')
            is_warped   = kind.endswith('_warped')
            # Call plotting for this dataset; flags reset per-call
            self._analyze_dataset(
                core_name,
                analysis_type,
                df,
                data_cols,
                meta_cols,
                cps,
                ignore_start=False,
                ignore_end=False,
                is_warped=is_warped,
                is_unwarped=is_unwarped
            )

    def _analyze_dataset(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        changepoints: Any,
        ignore_start: bool = False,
        ignore_end: bool = False,
        is_warped: bool = False,
        is_unwarped: bool = False,
    ):
        """
        Per-dataset plotting logic.
        """

        # If this is warped data, call non-stationarity vs similarity plot and return
        if is_warped:
            # self._plot_nonstationarity_vs_similarity(core_name, analysis_type, df,
            #                                           data_cols, meta_cols,
            #                                           changepoints,
            #                                           ignore_start, ignore_end,
            #                                           True, False)
            # self._plot_state_avg_with_ellipses(core_name, analysis_type, df, data_cols, is_warped)
            # self._plot_taste_avg_with_ellipses(core_name, analysis_type, df, data_cols, is_warped)
            #self._plot_non_windowed(core_name, analysis_type, df, data_cols, changepoints, is_warped, is_unwarped)
            self._plot_non_windowed_alt_align(core_name, analysis_type, 
                                              df, data_cols, changepoints, 
                                              is_warped, is_unwarped, alignment="start")


            return
        if is_unwarped:
            self._plot_full_by_trial(
                core_name,
                analysis_type,
                df,
                data_cols,
                meta_cols,
                ignore_start,
                ignore_end,
                is_warped,
                is_unwarped,
            )
            # self._plot_non_windowed(core_name, analysis_type, df, data_cols, changepoints, is_warped, is_unwarped)
            # self._plot_non_windowed_alt_align(core_name, analysis_type, 
            #                                   df, data_cols, changepoints, 
            #                                   is_warped, is_unwarped, alignment="end")
            # self._plot_non_windowed_align_multithread(core_name, analysis_type, 
            #                                   df, data_cols, changepoints, 
            #                                   is_warped, is_unwarped, alignment="none", 
            #                                   parallel = True, n_jobs= 4, backend= "threading")
            # # batching to just run em all: 
            # self._plot_non_windowed_align_multithread(core_name, analysis_type, 
            #                                   df, data_cols, changepoints, 
            #                                   is_warped, is_unwarped, alignment="start", 
            #                                   parallel = True, n_jobs= 4, backend= "threading")
            # self._plot_non_windowed_align_multithread(core_name, analysis_type, 
            #                                   df, data_cols, changepoints, 
            #                                   is_warped, is_unwarped, alignment="end", 
            #                                   parallel = True, n_jobs= 4, backend= "threading")
            

            # self._plot_full_by_trial_3d(core_name, analysis_type, df, data_cols, meta_cols, ignore_start, ignore_end, is_warped, is_unwarped)
            
            # interactivity bugged, work more on later...
            # self._plot_full_by_trial_3d_interactive(core_name, analysis_type, df, data_cols, meta_cols, ignore_start, ignore_end, is_warped, is_unwarped)
            return
    # --- Metrics calculation methods --- discrete logic steps that shouldn't run unless called
    def compute_nonstationarity(self, df: pl.DataFrame, unit: str, seed: int = 42) -> float:
        """
        Normalize each trial vector, shuffle, and measure mean abs deviation.
        """
        rng = np.random.RandomState(seed)
        values = []
        trials = df.select('trial').unique().to_series().to_list()
        for tr in trials:
            sub = df.filter(pl.col('trial') == tr)
            vec = sub.select(unit).to_series().to_numpy().astype(float)
            if vec.size == 0:
                continue
            if vec.max() > vec.min():
                vec = (vec - vec.min()) / (vec.max() - vec.min())
            shuf = rng.permutation(vec)
            values.append(np.mean(np.abs(vec - shuf)))
        return float(np.mean(values)) if values else 0.0
    
    def compute_intertrial_similarity(self, df: pl.DataFrame, unit: str) -> float:
        """
        Compute mean pairwise cosine similarity across valid trial vectors.
        Skips individual trials with empty or all-NaN vectors.
        """
        trials = df.select('trial').unique().to_series().to_list()
        vecs = []
    
        for tr in trials:
            sub = df.filter(pl.col('trial') == tr)
            vec = sub.select(unit).to_series().to_numpy().astype(float)
    
            # Skip if vector is empty or all NaNs
            if vec.size == 0 or np.isnan(vec).all():
                continue
    
            vecs.append(vec)
    
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                v1, v2 = vecs[i], vecs[j]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    
                if n1 > 0 and n2 > 0:
                    sims.append(np.dot(v1, v2) / (n1 * n2))
    
        return float(np.mean(sims)) if sims else 0.0

    # def compute_intertrial_similarity(self, df: pl.DataFrame, unit: str) -> float:
    #     """
    #     Compute mean pairwise cosine similarity across trial vectors.
    #     """
    #     trials = df.select('trial').unique().to_series().to_list()
    #     vecs = []
    #     for tr in trials:
    #         sub = df.filter(pl.col('trial') == tr)
    #         vec = sub.select(unit).to_series().to_numpy().astype(float)
    #         if vec.size:
    #             vecs.append(vec)
    #     sims = []
    #     for i in range(len(vecs)):
    #         for j in range(i+1, len(vecs)):
    #             v1, v2 = vecs[i], vecs[j]
    #             n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    #             if n1 and n2:
    #                 sims.append(np.dot(v1, v2) / (n1 * n2))
    #     return float(np.mean(sims)) if sims else 0.0
    
    # ---- Plotting methods ----- # 
    def _plot_nonstationarity_vs_similarity(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        changepoints: Any,
        ignore_start: bool,
        ignore_end: bool,
        is_warped: bool,
        is_unwarped: bool
    ):
        if not is_warped:
            print(f"Skipping metric plot on unwarped data {core_name}{analysis_type}")
            return
        tastes = sorted(df.select('taste').unique().to_series().to_list())
        fig, axes = plt.subplots(4, len(tastes), figsize=(12, 12), dpi=self.dpi, tight_layout=True, 
                                 sharex=True, sharey=True)
        fig_name = 'Non-Stationarity vs Inter-Trial Similarity'
        fig.suptitle(fig_name, y = self.suptitle_y, **self.title_fontdict)
        title = f"{core_name}{analysis_type}"
        fig.text(0.5, self.subtitle_y, title, ha='center', **self.subtitle_fontdict)
        ann = (
            "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
            f"Epoch 2: 2nd→3rd; Epoch 3: 3rd→end.\n"
            f"Start: {self.start_time_ms} ms, End: {self.end_time_ms} ms. If warped, warping to 1000 ms. excluding trials less than 25 ms.\n"
            "Non-Stationarity is trial average of abs. diff between trials, similarity is trial-average of cos similarity."
        )
        fig.text(0.5, -0.01, ann, ha="center", va="top")
        # fig.subplots_adjust(top = self.subplot_top)
        for i in range(4):
            for j, taste in enumerate(tastes):
                taste_name = self.taste_id_to_name(taste) # for naming
                ax = axes[i, j]
                sub = df.filter((pl.col('changepoint') == i) & (pl.col('taste') == taste))
                x_vals = [self.compute_nonstationarity(sub, unit) for unit in data_cols]
                y_vals = [self.compute_intertrial_similarity(sub, unit) for unit in data_cols]
                labels = [f"{u[0]}_{u.split('_')[-1]}" for u in data_cols]
                ax.scatter(x_vals, y_vals)
                for xi, yi, lbl in zip(x_vals, y_vals, labels): ax.annotate(lbl, (xi, yi))
                ax.set_xlabel('Non-stationarity', **self.label_fontdict)
                ax.set_ylabel('Similarity', **self.label_fontdict)
                ax.set_title(f"State {i}, Taste: {taste_name}", **self.subtitle_fontdict)

        if self.output_dir:
            self._save_figure(fig, fig_name, fig_name, core_name, analysis_type, tastes)
        else:
            plt.show()

            
    def _plot_state_avg_with_ellipses(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        is_warped: bool,
    ):
        if not is_warped:
            print(f"Skipping metric plot on unwarped data {core_name}{analysis_type}")
            return
    
        fig, axes = plt.subplots(
            1, 4, figsize=(20, 5), dpi=self.dpi, tight_layout=True, sharex=True, sharey=True
        )
    
        fig_name = 'Taste Avg: Non-Similarity vs Inter-Trial Similarity (averaging tastes)'
        fig.suptitle(fig_name, y=self.suptitle_y, **self.title_fontdict)
        title = f"{core_name}{analysis_type}"
        fig.text(0.5, self.subtitle_y - 0.05, title, ha='center', **self.subtitle_fontdict)
        ann = (
            "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
            f"Epoch 2: 2nd→3rd; Epoch 3: 3rd→end.\n"
            f"Start: {self.start_time_ms} ms, End: {self.end_time_ms} ms. If warped, warping to 1000 ms. excluding trials less than 25 ms.\n"
            "Non-Stationarity is trial average of abs. diff between trials, similarity is trial-average of cos similarity."
        )
        fig.text(0.5, -0.02, ann, ha="center", va="top")
    
        colors = plt.cm.tab20(np.linspace(0, 1, len(data_cols)))
    
        for state_idx in range(4):
            ax = axes[state_idx]
            sub = df.filter(pl.col('changepoint') == state_idx)
    
            for unit_idx, unit in enumerate(data_cols):
                tastes = sub['taste'].unique().to_list()
    
                x_vals_tastes = [
                    self.compute_nonstationarity(sub.filter(pl.col('taste') == taste), unit)
                    for taste in tastes
                ]
                y_vals_tastes = [
                    self.compute_intertrial_similarity(sub.filter(pl.col('taste') == taste), unit)
                    for taste in tastes
                ]
    
                # Plot individual points for each taste
                ax.scatter(
                    x_vals_tastes, y_vals_tastes,
                    color=colors[unit_idx],
                    alpha=0.5,
                    s=20,
                    label=f"Points_{unit[0]}_{unit.split('_')[-1]}"
                )
    
                # Mean and std for ellipse
                x_mean, y_mean = np.mean(x_vals_tastes), np.mean(y_vals_tastes)
                x_std, y_std = np.std(x_vals_tastes), np.std(y_vals_tastes)
    
                ellipse = Ellipse(
                    (x_mean, y_mean), width=2 * x_std, height=2 * y_std,
                    alpha=0.3, color=colors[unit_idx],
                    label=f"Mean_{unit[0]}_{unit.split('_')[-1]}"
                )
                ax.add_patch(ellipse)
                ax.scatter(x_mean, y_mean, color=colors[unit_idx], edgecolor='black', s=40)
                ax.annotate(
                    f"{unit[0]}_{unit.split('_')[-1]}", (x_mean, y_mean), fontsize=8,
                    ha='center', va='center'
                )
    
            ax.set_xlabel('Non-stationarity', **self.label_fontdict)
            if state_idx == 0:
                ax.set_ylabel('Similarity', **self.label_fontdict)
            ax.set_title(f"State {state_idx}", **self.subtitle_fontdict)
    
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.9))
    
        if self.output_dir:
            self._save_figure(fig, fig_name, fig_name, core_name, analysis_type, [])
        else:
            plt.show()     

    def _plot_taste_avg_with_ellipses(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        is_warped: bool,
    ):
        if not is_warped:
            print(f"Skipping metric plot on unwarped data {core_name}{analysis_type}")
            return
    
        tastes = df['taste'].unique().to_list()
        fig, axes = plt.subplots(
            1, len(tastes), figsize=(5 * len(tastes), 5), dpi=self.dpi, tight_layout=True, sharex=True, sharey=True
        )
    
        fig_name = 'State Avg: Non-Similarity vs Inter-Trial Similarity (averaging states, excluding pre-stim state 0)'
        fig.suptitle(fig_name, y=self.suptitle_y, **self.title_fontdict)
        title = f"{core_name}{analysis_type}"
        fig.text(0.5, self.subtitle_y - 0.05, title, ha='center', **self.subtitle_fontdict)
        ann = (
            "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
            f"Epoch 2: 2nd→3rd; Epoch 3: 3rd→end.\n"
            f"Start: {self.start_time_ms} ms, End: {self.end_time_ms} ms. If warped, warping to 1000 ms. excluding trials less than 25 ms.\n"
            "Non-Stationarity is trial average of abs. diff between trials, similarity is trial-average of cos similarity."
        )
        fig.text(0.5, -0.02, ann, ha="center", va="top")
    
        colors = plt.cm.tab20(np.linspace(0, 1, len(data_cols)))
    
        for taste_idx, taste in enumerate(tastes):
            ax = axes[taste_idx]
            sub = df.filter((pl.col('taste') == taste) & (pl.col('changepoint') != 0))
    
            for unit_idx, unit in enumerate(data_cols):
                states = sub['changepoint'].unique().to_list()
    
                x_vals_states = [
                    self.compute_nonstationarity(sub.filter(pl.col('changepoint') == state), unit)
                    for state in states
                ]
                y_vals_states = [
                    self.compute_intertrial_similarity(sub.filter(pl.col('changepoint') == state), unit)
                    for state in states
                ]
    
                # Plot individual points for each state
                ax.scatter(
                    x_vals_states, y_vals_states,
                    color=colors[unit_idx],
                    alpha=0.5,
                    s=20,
                    label=f"Points_{unit[0]}_{unit.split('_')[-1]}"
                )
    
                # Mean and std for ellipse
                x_mean, y_mean = np.mean(x_vals_states), np.mean(y_vals_states)
                x_std, y_std = np.std(x_vals_states), np.std(y_vals_states)
    
                ellipse = Ellipse(
                    (x_mean, y_mean), width=2 * x_std, height=2 * y_std,
                    alpha=0.3, color=colors[unit_idx],
                    label=f"Mean_{unit[0]}_{unit.split('_')[-1]}"
                )
                ax.add_patch(ellipse)
                ax.scatter(x_mean, y_mean, color=colors[unit_idx], edgecolor='black', s=40)
                ax.annotate(
                    f"{unit[0]}_{unit.split('_')[-1]}", (x_mean, y_mean), fontsize=8,
                    ha='center', va='center'
                )
    
            ax.set_xlabel('Non-stationarity', **self.label_fontdict)
            if taste_idx == 0:
                ax.set_ylabel('Similarity', **self.label_fontdict)
            taste_name = self.taste_id_to_name(taste)
            ax.set_title(f"Taste: {taste_name}", **self.subtitle_fontdict)
    
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.9))
    
        if self.output_dir:
            self._save_figure(fig, fig_name, fig_name, core_name, analysis_type, tastes)
        else:
            plt.show()
   
    def _plot_full_by_trial(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        ignore_start: bool,
        ignore_end: bool,
        is_warped: bool,
        is_unwarped: bool
    ):
        # Only handle un-warped data
        if not is_unwarped:
            print(f"Skipping full-by-trial plot on warped data {core_name}{analysis_type}")
            return

        # Initialize plot style
        self._init_plot_style()        
        # Constant figure name
        fig_name = 'full_trial_plots'

        # Unique sorted tastes
        tastes = sorted(df.select('taste').unique().to_series().to_list())

        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col('taste') == taste)
            taste_name = self.taste_id_to_name(taste) # for naming
            trials = sorted(taste_df.select('trial').unique().to_series().to_list()) # ensure sequential?

            # 4 trials per figure
            for chunk_start in range(0, len(trials), 4):
                fig, axes = plt.subplots(
                    4,
                    1,
                    figsize=(15, 15),
                    dpi=self.dpi,
                    tight_layout=True
                )
                # Main titles
                fig.suptitle(
                    fig_name,
                    y=self.suptitle_y,
                    **self.title_fontdict
                )
                fig.text(
                    0.5,
                    self.subtitle_y,
                    f"{core_name}{analysis_type}_Taste: {taste_name}",
                    ha='center',
                    **self.subtitle_fontdict
                )

                # Plot each trial in the chunk
                for i, trial in enumerate(trials[chunk_start:chunk_start + 4]):
                    ax = axes[i]
                    trial_df = taste_df.filter(pl.col('trial') == trial)
                    time = trial_df['time'].to_numpy()

                    # Build time mask
                    mask = np.ones_like(time, dtype=bool)
                    if not ignore_start:
                        mask &= time >= self.start_time_ms
                    if not ignore_end:
                        mask &= time <= self.end_time_ms

                    # Plot data columns
                    for col in data_cols:
                        series = trial_df[col].to_numpy()[mask]
                        ax.plot(time[mask], series, label=col)

                    # Plot changepoints
                    lower = self.start_time_ms if not ignore_start else time.min()
                    upper = self.end_time_ms if not ignore_end else time.max()
                    for cp in self.get_trial_changepoints(
                        self.standardized_changepoints_dict.get(core_name),
                        taste_idx,
                        trial
                    ):
                        if lower <= cp <= upper:
                            ax.axvline(cp, **self.line_kwargs['changepoint'])

                    # Stimulus delivery line
                    if lower <= self.stim_time_ms <= upper:
                        ax.axvline(self.stim_time_ms, **self.line_kwargs['stim'])

                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Value (see analysis type)")

                # Shared legend below subplots
                handles, labels = axes[-1].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc='lower center',
                    ncol=5,
                    fontsize='small',
                    frameon=False
                )
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])

                # Save figure with chunk identifier
                chunk_id = chunk_start // 4 + 1
                title = f"{fig_name}_{taste_name}_chunk{chunk_id}"
                self._save_figure(
                    fig,
                    fig_name,
                    title,
                    core_name,
                    analysis_type,
                    [taste]
                )

    def _plot_non_windowed(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        dims: list,
        cps: list,
        is_warped: bool,
        is_unwarped: bool,
    ):
        tastes = sorted(df.select('taste').unique().to_series().to_list())
    
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col("taste") == taste)
            trials = taste_df["trial"].unique().to_list()
            trials_cps = cps[taste_idx]
            n_cps = trials_cps.shape[1]
            num_epochs = min(n_cps + 1, 4)
            taste_name = self.taste_id_to_name(taste)
    
            for chunk_start in range(0, len(dims), 6):
                dim_chunk = dims[chunk_start:chunk_start + 6]
    
                fig, axes = plt.subplots(
                    6,
                    num_epochs,
                    figsize=(5 * num_epochs, 4 * 6),
                    dpi=self.dpi,
                    squeeze=False
                )
    
                for row, dim in enumerate(dim_chunk):
                    for eid in range(num_epochs):
                        ax = axes[row, eid]
                        epoch_vals = []
    
                        for trial in trials:
                            trial_df = taste_df.filter(pl.col("trial") == trial)
                            cps_trial = self.get_trial_changepoints(cps, taste_idx, trial)
                    
                            if is_warped:
                                # Filter the current epoch by changepoint index
                                seg = trial_df.filter(
                                    (pl.col("changepoint") == eid)
                                )
                            
                                if seg.is_empty():
                                    continue
                            
                                # Dynamically get start and end time range for this epoch
                                start = int(seg["time"].min())
                                end = int(seg["time"].max())
                            else:
                                if eid == 0:
                                    start, end = self.start_time_ms, cps_trial[0]
                                elif eid < len(cps_trial):
                                    start, end = cps_trial[eid - 1], cps_trial[eid]
                                else:
                                    start, end = cps_trial[-1], self.end_time_ms
                            
                                seg = trial_df.filter(
                                    (pl.col("time") >= start) & (pl.col("time") <= end)
                                )
                            
                            # Common for both warped and unwarped -- temporal alignment 
                            t = seg["time"].to_numpy() - start
                            v = seg[dim].to_numpy()
                            
                            if t.size > 0:
                                ax.plot(t, v, alpha=0.5, linewidth=self.default_linewidth)
                                epoch_vals.append((t.astype(int), v))

    
                        if self.plot_configs.get('plot_average', True) and epoch_vals:
                            grid = np.unique(np.concatenate([t for t, _ in epoch_vals]))
                            mat = np.vstack(
                                [
                                    np.interp(grid, np.ravel(t), np.ravel(v), left=np.nan, right=np.nan)
                                    for t, v in epoch_vals
                                ]
                            )
                            ax.plot(
                                grid,
                                np.nanmean(mat, axis=0),
                                linewidth=2,
                                alpha=1,
                                color="gray",
                            )
    
                        ax.set_title(f"{dim}, Epoch {eid}", fontdict=self.subtitle_fontdict)
                        ax.set_xlabel("Time (ms)", fontdict=self.label_fontdict)
                        ax.set_ylabel("Value (see analysis)", fontdict=self.label_fontdict)
                        ax.tick_params(axis='both', labelsize=self.tick_fontdict['fontsize'])
    
                fig.suptitle(
                    f"Taste: {taste_name}, Mode: {'Warped' if is_warped else 'Unwarped'}, Dataset: {core_name}{analysis_type} (Dims {chunk_start + 1}-{chunk_start + len(dim_chunk)})",
                    y=self.suptitle_y,
                    **self.title_fontdict
                )
    
                ann = (
                    "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
                    f"Epoch 2: 2nd→3rd; Epoch 3: 3rd→end.\n"
                    f"Start: {self.start_time_ms} ms, End: {self.end_time_ms} ms. Average in gray. If warped, warping to 1000 ms. excluding trials less than 25 ms."
                )
                fig.subplots_adjust(top=(self.subplot_top + 0.08))
                fig.subplots_adjust(hspace=0.42) # adjusts the spacing of each sub plot
                fig.text(0.5, 0.1, ann, ha="center", va="top")
    
                self._save_figure(
                    fig=fig,
                    dir_title="State_Windowed_Plots",
                    title=f"State_Win{analysis_type}_{taste_name}_part{chunk_start // 6 + 1}",
                    core_name=core_name,
                    analysis_type=analysis_type,
                    tastes=[taste]
                )
# experimenting with differing modes of alignment: 
    def _plot_non_windowed_alt_align(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        dims: list,
        cps: list,
        is_warped: bool,
        is_unwarped: bool,
        *,
        alignment: str = "start",        #  NEW- pass start, end, or none
    ):
        """
        Plot non-windowed traces with optional temporal alignment.
    
        Parameters
        ----------
        alignment : {"start", "end", "none"}
            • "start" – align epoch starts to 0 ms (legacy behaviour)  
            • "end"   – align epoch ends to a common max-end time  
            • "none"  – no intra-epoch alignment; raw times shown
        """
        if alignment not in {"start", "end", "none"}:
            raise ValueError("alignment must be 'start', 'end', or 'none'.")
    
        tastes = sorted(df.select('taste').unique().to_series().to_list())
    
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col("taste") == taste)
            trials     = taste_df["trial"].unique().to_list()
            trials_cps = cps[taste_idx]
            n_cps      = trials_cps.shape[1]
            num_epochs = min(n_cps + 1, 4)
            taste_name = self.taste_id_to_name(taste)
    
            for chunk_start in range(0, len(dims), 6):
                dim_chunk = dims[chunk_start:chunk_start + 6]
    
                fig, axes = plt.subplots(
                    6, num_epochs,
                    figsize=(5 * num_epochs, 4 * 6),
                    dpi=self.dpi,
                    squeeze=False,
                )
    
                # ──────────────────────────────────────────────────────────────
                # iterate over dimensions (rows) and epochs (cols)
                # ──────────────────────────────────────────────────────────────
                for row, dim in enumerate(dim_chunk):
                    for eid in range(num_epochs):
                        ax          = axes[row, eid]
                        epoch_vals  = []       # (t, v) tuples for averaging
                        end_times   = []       # keep end times for 'end' alignment
                        pending_end = []          # only used when alignment == "end"
    
                        # -------- gather data for all trials in this epoch --------
                        for trial in trials:
                            trial_df   = taste_df.filter(pl.col("trial") == trial)
                            cps_trial  = self.get_trial_changepoints(cps, taste_idx, trial)
    
                            # ── select segment for this epoch ────────────────────
                            if is_warped:
                                seg = trial_df.filter(pl.col("changepoint") == eid)
                                if seg.is_empty():
                                    continue
                                start = int(seg["time"].min())
                                end   = int(seg["time"].max())
                            else:
                                if eid == 0:
                                    start, end = self.start_time_ms, cps_trial[0]
                                elif eid < len(cps_trial):
                                    start, end = cps_trial[eid - 1], cps_trial[eid]
                                else:
                                    start, end = cps_trial[-1], self.end_time_ms
                                seg = trial_df.filter(
                                    (pl.col("time") >= start) & (pl.col("time") <= end)
                                )
    
                            if seg.is_empty():
                                continue
    
                            t_raw = seg["time"].to_numpy()
                            v     = seg[dim].to_numpy()
    
                            # ── alignment transforms ─────────────────────────────
                            if alignment == "start":
                                t = t_raw - start
                                ax.plot(t, v, alpha=0.5, linewidth=self.default_linewidth)
                                epoch_vals.append((t.astype(int), v))
                            
                            elif alignment == "end":
                                pending_end.append((t_raw, v))      # store raw, shift later
                            
                            else:  # alignment == "none"
                                t = t_raw - self.start_time_ms
                                ax.plot(t, v, alpha=0.5, linewidth=self.default_linewidth)
                                epoch_vals.append((t.astype(int), v))
                            
                            # after exiting the trials-loop for this epoch
                            if alignment == "end" and pending_end:
                                max_end = max(t_raw.max() for t_raw, _ in pending_end)
                                for t_raw, v in pending_end:
                                    shift = max_end - t_raw.max()
                                    t = t_raw + shift
                                    ax.plot(t, v, alpha=0.5, linewidth=self.default_linewidth)
                                    epoch_vals.append((t.astype(int), v))
                                    
                        # -------- average trace (optional) ------------------------------
                        if self.plot_configs.get("plot_average", True) and epoch_vals:
                            grid = np.unique(np.concatenate([t for t, _ in epoch_vals]))
                            mat  = np.vstack([
                                np.interp(grid, np.ravel(t), np.ravel(v),
                                          left=np.nan, right=np.nan)
                                for t, v in epoch_vals
                            ])
                            ax.plot(grid, np.nanmean(mat, axis=0),
                                    linewidth=2, color="gray")
    
                        ax.set_title(f"{dim}, Epoch {eid}", fontdict=self.subtitle_fontdict)
                        ax.set_xlabel("Time (ms)",  fontdict=self.label_fontdict)
                        ax.set_ylabel("Value",      fontdict=self.label_fontdict)
                        ax.tick_params(axis='both', labelsize=self.tick_fontdict['fontsize'])
    
                # ───── figure level decorations ─────────────────────────────────
                align_str = {"start": "Start-aligned",
                             "end":   "End-aligned",
                             "none":  "No alignment"}[alignment]
    
                fig.suptitle(
                    f"Taste: {taste_name} · {align_str} · "
                    f"Mode: {'Warped' if is_warped else 'Unwarped'} · "
                    f"Dataset: {core_name}{analysis_type} "
                    f"(Dims {chunk_start + 1}-{chunk_start + len(dim_chunk)})",
                    y=self.suptitle_y,
                    **self.title_fontdict
                )
    
                ann = (
                    "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
                    "Epoch 2: 2nd→3rd; Epoch 3: 3rd→end.\n"
                    f"Start: {self.start_time_ms} ms, End: {self.end_time_ms} ms. "
                    "Average in gray. If warped, 0–1000 ms."
                )
                fig.subplots_adjust(top=self.subplot_top + 0.08, hspace=0.42)
                fig.text(0.5, 0.1, ann, ha="center", va="top")
    
                self._save_figure(
                    fig=fig,
                    dir_title=f"State_Windowed_Plots_{alignment}_aligned",
                    title=f"State_Win{analysis_type}_{taste_name}_part{chunk_start // 6 + 1}_{alignment}",
                    core_name=core_name,
                    analysis_type=analysis_type,
                    tastes=[taste],
                )
# experimental: variable alignmnet, multi-threaded for time: 
    def _plot_non_windowed_align_multithread(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        dims: list,
        cps: list,
        is_warped: bool,
        is_unwarped: bool,
        *,
        alignment: str = "start",
        parallel: bool = True,
        n_jobs: int | None = None, # by default you should use four
        backend: str = "threading",      # "threading"  (default)  or "loky"
    ):
        """
        Plot non-windowed traces per taste, optionally in parallel via joblib.
    
        alignment : {"start", "end", "none"}
        parallel  : bool     → enable/disable Parallel()
        n_jobs    : int|None → #workers (None → len(tastes); -1 → all cores)
        backend   : str      → "threading" (GIL-sharing) or "loky" (multiproc)
        """
        if alignment not in {"start", "end", "none"}:
            raise ValueError("alignment must be 'start', 'end', or 'none'.")
    
        tastes       = sorted(df.select('taste').unique().to_series().to_list())
        n_jobs       = n_jobs or len(tastes)
    
        # ─────────────────── helper that handles ONE taste ────────────────────
        def _plot_one_taste(taste_idx: int):
            taste      = tastes[taste_idx]
            taste_df   = df.filter(pl.col("taste") == taste)
            trials     = taste_df["trial"].unique().to_list()
            trials_cps = cps[taste_idx]
            n_cps      = trials_cps.shape[1]
            num_epochs = min(n_cps + 1, 4)
            taste_name = self.taste_id_to_name(taste)
    
            for chunk_start in range(0, len(dims), 6):
                dim_chunk = dims[chunk_start:chunk_start + 6]
    
                fig, axes = plt.subplots(
                    6, num_epochs,
                    figsize=(5 * num_epochs, 4 * 6),
                    dpi=self.dpi,
                    squeeze=False,
                )
    
                for row, dim in enumerate(dim_chunk):
                    for eid in range(num_epochs):
                        ax          = axes[row, eid]
                        epoch_vals  = []
                        pending_end = []   # only used for "end" alignment
    
                        for trial in trials:
                            trial_df  = taste_df.filter(pl.col("trial") == trial)
                            cps_trial = self.get_trial_changepoints(cps, taste_idx, trial)
    
                            # ── segment selection (unchanged) ───────────────────
                            if is_warped:
                                seg = trial_df.filter(pl.col("changepoint") == eid)
                                if seg.is_empty():
                                    continue
                                start = int(seg["time"].min())
                                end   = int(seg["time"].max())
                            else:
                                if eid == 0:
                                    start, end = self.start_time_ms, cps_trial[0]
                                elif eid < len(cps_trial):
                                    start, end = cps_trial[eid - 1], cps_trial[eid]
                                else:
                                    start, end = cps_trial[-1], self.end_time_ms
    
                                seg = trial_df.filter(
                                    (pl.col("time") >= start) & (pl.col("time") <= end)
                                )
                            if seg.is_empty():
                                continue
    
                            t_raw = seg["time"].to_numpy()
                            v     = seg[dim].to_numpy()
    
                            # ── alignment transforms ─────────────────────────────
                            if alignment == "start":
                                t = t_raw - start
                                ax.plot(t, v, alpha=0.5, linewidth=self.default_linewidth)
                                epoch_vals.append((t.astype(int), v))
    
                            elif alignment == "end":
                                pending_end.append((t_raw, v))
                            else:  # "none"
                                t = t_raw #- self.start_time_ms
                                ax.plot(t, v, alpha=0.5, linewidth=self.default_linewidth)
                                epoch_vals.append((t.astype(int), v))
    
                        # resolve pending traces for "end" alignment
                        if alignment == "end" and pending_end:
                            max_end = max(t_raw.max() for t_raw, _ in pending_end)
                            for t_raw, v in pending_end:
                                shift = max_end - t_raw.max()
                                t = t_raw + shift
                                ax.plot(t, v, alpha=0.5, linewidth=self.default_linewidth)
                                epoch_vals.append((t.astype(int), v))
    
                        # average trace (optional)
                        if self.plot_configs.get("plot_average", True) and epoch_vals:
                            grid = np.unique(np.concatenate([t for t, _ in epoch_vals]))
                            mat  = np.vstack([
                                np.interp(grid, np.ravel(t), np.ravel(v),
                                          left=np.nan, right=np.nan)
                                for t, v in epoch_vals
                            ])
                            ax.plot(grid, np.nanmean(mat, axis=0), linewidth=2, color="gray")
    
                        ax.set_title(f"{dim}, Epoch {eid}", fontdict=self.subtitle_fontdict)
                        ax.set_xlabel("Time (ms)", fontdict=self.label_fontdict)
                        ax.set_ylabel("Value",     fontdict=self.label_fontdict)
                        ax.tick_params(axis='both', labelsize=self.tick_fontdict['fontsize'])
    
                # figure-level decoration & save
                align_str = {"start": "Start-aligned",
                             "end":   "End-aligned",
                             "none":  "No alignment"}[alignment]
    
                fig.suptitle(
                    f"Taste: {taste_name} · {align_str} · "
                    f"Mode: {'Warped' if is_warped else 'Unwarped'} · "
                    f"Dataset: {core_name}{analysis_type} "
                    f"(Dims {chunk_start + 1}-{chunk_start + len(dim_chunk)})",
                    y=self.suptitle_y,
                    **self.title_fontdict,
                )
    
                ann = (
                    "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
                    "Epoch 2: 2nd→3rd; Epoch 3: 3rd→end.\n"
                    f"Start: {self.start_time_ms} ms, End: {self.end_time_ms} ms."
                    "\n window timing relative to alignment sceme."
                    " borders for none-type reflect the extremes of state across all trials"
                )
                fig.subplots_adjust(top=self.subplot_top + 0.08, hspace=0.42)
                fig.text(0.5, 0.1, ann, ha="center", va="top")
    
                self._save_figure(
                    fig=fig,
                    dir_title=f"State_Windowed_Plots_{alignment}_aligned",
                    title=(
                        f"State_Win{analysis_type}_{taste_name}"
                        f"_part{chunk_start // 6 + 1}_{alignment}"
                    ),
                    core_name=core_name,
                    analysis_type=analysis_type,
                    tastes=[taste],
                )
                plt.close(fig)               # mandatory to free memory
    
        # ─────────────────── serial vs. parallel execution ────────────────────
        if parallel and len(tastes) > 1:
            Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_plot_one_taste)(idx) for idx in range(len(tastes))
            )
        else:
            for idx in range(len(tastes)):
                _plot_one_taste(idx)

                
    # experimental plot time: 3D full trial organized by first cp timing...


    def _plot_full_by_trial_3d(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        ignore_start: bool,
        ignore_end: bool,
        is_warped: bool,
        is_unwarped: bool
    ):
        # Only handle un-warped data
        if not is_unwarped:
            print(f"Skipping full-by-trial 3D plot on warped data {core_name}{analysis_type}")
            return
    
        # Initialize plot style
        self._init_plot_style()
    
        # Unique sorted tastes
        tastes = sorted(df.select('taste').unique().to_series().to_list())
    
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col('taste') == taste)
            taste_name = self.taste_id_to_name(taste)
    
            for col in data_cols:
                fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
                ax = fig.add_subplot(111, projection='3d')
    
                fig.suptitle(f"3D plot - {col} - Taste: {taste_name}",
                              y=self.suptitle_y, **self.title_fontdict)
    
                # Get trial and changepoint ordering
                trials = sorted(taste_df.select('trial').unique().to_series().to_list())
                trial_cps = [self.get_trial_changepoints(self.standardized_changepoints_dict.get(core_name), taste_idx, trial)
                              for trial in trials]
    
                # Order trials by first changepoint
                trial_cp_order = sorted(zip(trials, trial_cps), key=lambda x: x[1][0] if len(x[1]) > 0 else float('inf'))
    
                for trial_idx, (trial, cps) in enumerate(trial_cp_order):
                    trial_df = taste_df.filter(pl.col('trial') == trial)
                    time = trial_df['time'].to_numpy()
    
                    # Build time mask
                    mask = np.ones_like(time, dtype=bool)
                    if not ignore_start:
                        mask &= time >= self.start_time_ms
                    if not ignore_end:
                        mask &= time <= self.end_time_ms
    
                    masked_time = time[mask]
                    series = trial_df[col].to_numpy()[mask]
    
                    # Plot the data in 3D
                    x = np.full_like(masked_time, trial_idx)
                    y = masked_time
                    z = series
                    ax.plot(x, y, z, label=f"Trial {trial}")
    
                    # Plot changepoints with progressive shading
                    for cp_idx, cp in enumerate(cps):
                        if masked_time.min() <= cp <= masked_time.max():
                            cp_val = np.interp(cp, masked_time, series)
                            shade = 0.1 + 0.8 * (cp_idx / max(len(cps)-1, 1))
                            ax.scatter(trial_idx, cp, cp_val, c=str(1-shade), s=30)
    
                # Stimulus delivery line across all trials
                stim_y = [self.stim_time_ms, self.stim_time_ms]
                stim_x = [0, len(trials)-1]
                stim_z = [ax.get_zlim()[0], ax.get_zlim()[0]]
                ax.plot(stim_x, stim_y, stim_z, color='red', linestyle='--', label='Stimulus Delivery', linewidth=2)
    
                # Labels
                ax.set_xlabel('Trial (ordered by earliest changepoint)')
                ax.set_ylabel('Time (ms)')
                ax.set_zlabel('Value (see analysis type)')
    
                # ax.legend(loc='upper right') # figure out this placement later
    
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
                # Save figure
                fig_name = f"3D_{col}_Taste_{taste_name}"
                self._save_figure(
                    fig,
                    'full_trial_plots_3D',
                    fig_name,
                    core_name,
                    analysis_type,
                    [taste]
                )
    



    def _plot_full_by_trial_3d_interactive(
        self,
        core_name: str,
        analysis_type: str,
        df: pl.DataFrame,
        data_cols: List[str],
        meta_cols: List[str],
        ignore_start: bool,
        ignore_end: bool,
        is_warped: bool,
        is_unwarped: bool
    ):
        if not is_unwarped:
            print(f"Skipping full-by-trial 3D plot on warped data {core_name}{analysis_type}")
            return
    
        self._init_plot_style()
    
        tastes = sorted(df.select('taste').unique().to_series().to_list())
    
        viewpoint_set = False
        azim, elev = -60, 30  # Default angles
    
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col('taste') == taste)
            taste_name = self.taste_id_to_name(taste)
    
            for col_idx, col in enumerate(data_cols):
                fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
                ax = fig.add_subplot(111, projection='3d')
    
                fig.suptitle(f"3D plot - {col} - Taste: {taste_name}",
                             y=self.suptitle_y, **self.title_fontdict)
    
                trials = sorted(taste_df.select('trial').unique().to_series().to_list())
                trial_cps = [
                    self.get_trial_changepoints(
                        self.standardized_changepoints_dict.get(core_name),
                        taste_idx, trial
                    )
                    for trial in trials
                ]
    
                trial_cp_order = sorted(
                    zip(trials, trial_cps),
                    key=lambda x: x[1][0] if len(x[1]) > 0 else float('inf')
                )
    
                for trial_idx, (trial, cps) in enumerate(trial_cp_order):
                    trial_df = taste_df.filter(pl.col('trial') == trial)
                    time = trial_df['time'].to_numpy()
    
                    mask = np.ones_like(time, dtype=bool)
                    if not ignore_start:
                        mask &= time >= self.start_time_ms
                    if not ignore_end:
                        mask &= time <= self.end_time_ms
    
                    masked_time = time[mask]
                    series = trial_df[col].to_numpy()[mask]
    
                    x = np.full_like(masked_time, trial_idx)
                    y = masked_time
                    z = series
                    ax.plot(x, y, z, label=f"Trial {trial}")
    
                    for cp_idx, cp in enumerate(cps):
                        if masked_time.min() <= cp <= masked_time.max():
                            cp_val = np.interp(cp, masked_time, series)
                            shade = 0.1 + 0.8 * (cp_idx / max(len(cps)-1, 1))
                            ax.scatter(trial_idx, cp, cp_val, c=str(1-shade), s=30)
    
                stim_y = [self.stim_time_ms, self.stim_time_ms]
                stim_x = [0, len(trials)-1]
                stim_z = [ax.get_zlim()[0], ax.get_zlim()[0]]
                ax.plot(stim_x, stim_y, stim_z, color='red', linestyle='--', label='Stimulus Delivery', linewidth=2)
    
                ax.set_xlabel('Trial (ordered by earliest changepoint)')
                ax.set_ylabel('Time (ms)')
                ax.set_zlabel('Value (see analysis type)')
                ax.legend(loc='upper right')
    
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                # plt.show()
    
                # Set viewpoint interactively for first plot
                if not viewpoint_set and col_idx == 0 and taste_idx == 0:
                    plt.show()
                    input("Adjust the viewpoint interactively, then press Enter here to store viewpoint...")
                    azim, elev = ax.azim, ax.elev
                    viewpoint_set = True
                else:
                    ax.view_init(elev=elev, azim=azim)
                    plt.draw()
                    plt.pause(0.5)
    
                fig_name = f"3D_{col}_Taste_{taste_name}"
                self._save_figure(
                    fig,
                    'full_trial_plots_3D',
                    fig_name,
                    core_name,
                    analysis_type,
                    [taste]
                )
                plt.close(fig)
# new ttest venn diagram plots 

    def plot_ttest_venn(self, ttest_dataset_dict: Dict[str, pl.DataFrame]) -> None:
        """
        Create 'Venn-style' diagrams per dataset showing counts of significant units
        (MWU half-split) vs all units, one panel per taste.
    
        Expects ttest-augmented DataFrames (columns like 'mwu_sig_neuron_#' or 'mwu_sig_PC_#').
        Uses only neuron_* and PC_* dimensions as the unit set.
        """
        for dataset_name, df in ttest_dataset_dict.items():
            core_name, analysis_type = self.parse_dataset_name(dataset_name)
    
            # Identify units (neurons or PCs only) and their corresponding mwu_sig columns
            all_dims = self.get_data_columns(df)
            dims = [d for d in all_dims if d.startswith("neuron_") or d.startswith("PC_")]
            if not dims:
                print(f"[Venn] {dataset_name}: no neuron_/PC_ dims found; skipping.")
                continue
    
            dims_with_sig = [d for d in dims if f"mwu_sig_{d}" in df.columns]
            if not dims_with_sig:
                print(f"[Venn] {dataset_name}: no mwu_sig_* columns found; skipping.")
                continue
    
            tastes = sorted(df.select("taste").unique().to_series().to_list())
            if not tastes:
                print(f"[Venn] {dataset_name}: no tastes found; skipping.")
                continue
    
            # Figure scaffolding
            self._init_plot_style()
            n_tastes = len(tastes)
            fig_w = max(4.0 * n_tastes, 8.0)
            fig_h = 4.5
            fig, axes = plt.subplots(
                1, n_tastes, figsize=(fig_w, fig_h), dpi=self.dpi, tight_layout=True
            )
            if n_tastes == 1:
                axes = [axes]  # normalize to list
    
            # Choose a fill color from your configured cycle
            cmap = plt.get_cmap(self.colormaps.get("line", "tab10"))
            fill_color = cmap(0)
    
            # Per-taste rollup and draw
            for ax, taste in zip(axes, tastes):
                taste_df = df.filter(pl.col("taste") == taste)
    
                # Count total units evaluated and #significant across *any* state×trial for this taste
                total_units = len(dims_with_sig)
                n_sig = 0
                for d in dims_with_sig:
                    sc = f"mwu_sig_{d}"
                    # any() over rows; fill_null(False) treats NaNs as non-sig
                    sig_any = bool(taste_df.select(pl.col(sc).fill_null(False).any()).item())
                    if sig_any:
                        n_sig += 1
                n_nonsig = max(total_units - n_sig, 0)
    
                # Draw fully overlapped circles with area ∝ counts
                # Outer (all units) radius = 1; inner radius scales by sqrt(n_sig/total)
                from matplotlib.patches import Circle
                r_all = 1.0
                frac = (n_sig / total_units) if total_units else 0.0
                r_sig = (frac ** 0.5) * r_all
    
                ax.add_patch(Circle((0, 0), r_all, edgecolor="black", facecolor="none", linewidth=2))
                ax.add_patch(Circle((0, 0), r_sig, edgecolor="none", facecolor=fill_color, alpha=0.55))
    
                # Styling / labels
                ax.set_aspect("equal")
                ax.set_xlim(-1.15, 1.15)
                ax.set_ylim(-1.15, 1.15)
                ax.axis("off")
    
                taste_name = self.taste_id_to_name(taste)
                ax.set_title(f"{taste_name}", **self.subtitle_fontdict)
    
                # Center count + small caption
                ax.text(0, 0, f"{n_sig}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white" if frac > 0.15 else "black")
                ax.text(0, -1.22, f"sig: {n_sig} / {total_units}\nnon-sig: {n_nonsig}",
                        ha="center", va="top", fontsize=10)
    
            # Figure title + legend
            fig.suptitle(
                f"Significant units (MWU half-split) • {core_name}{analysis_type}",
                y=self.suptitle_y, **self.title_fontdict
            )
            from matplotlib.lines import Line2D
            import matplotlib.patches as mpatches
            handles = [
                Line2D([0], [0], color="black", lw=2, label="All units (count)"),
                mpatches.Patch(facecolor=fill_color, alpha=0.55, label="Significant (any state)"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize="small")
            plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    
            # Save once per dataset; replicate into taste subfolders for consistency
            self._save_figure(
                fig=fig,
                dir_title="venn_significant_neurons",
                title="venn_sig_vs_nonsig",
                core_name=core_name,
                analysis_type=analysis_type,
                tastes=tastes,
            )



    # -- Auxilliary and helper methods --- # 
    
    def _init_plot_style(self):
        """
        Set up default matplotlib styling parameters for axes labels, ticks, lines, titles, colormaps, and more.
        Can be overridden per-plot by specifying custom parameters in the plotting methods.
        """
        # Uncomment and set fig size per-plot as needed:
        # self.figsize = (8, 6)  # Set on case-by-case basis
        self.dpi = 200
        self.plot_average = True

        # Axis label fonts
        self.label_fontdict = {
            'fontsize': 12,
            'family': 'sans-serif',
            'weight': 'bold'
        }
        # Axis tick fonts
        self.tick_fontdict = {
            'fontsize': 10,
            'family': 'sans-serif',
            'weight': 'normal'
        }
        # Default line width and marker size
        self.default_linewidth = 1
        self.default_markersize = 3

        # Grid default off, but style defined
        plt.rcParams['axes.grid'] = False
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.3

        # Legend styling
        self.legend_kwargs = {
            'fontsize': 10,
            'frameon': False,
            'loc': 'best'
        }

        # Spine visibility defaults
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
       #  plt.rcParams['tight_layout'] = True # enforce tight layout by default -- apparently not a valid default??

        # Colormaps
        self.colormaps = {
            'heatmap': 'viridis',
            'line': 'tab10'
        }
        # Color cycle for lines
        cmap = plt.get_cmap(self.colormaps['line'])
        colors = [cmap(i) for i in range(cmap.N)]
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors)

        # Line styles for changepoints and stimulus
        self.line_kwargs = {
            'changepoint': {
                'color': 'black',
                'linestyle': '--',
                'linewidth': 2,
                'label': 'Changepoint'
            },
            'stim': {
                'color': 'black',
                'linestyle': ':',
                'linewidth': 2,
                'label': 'Stimulus Delivery'
            }
        }

        # Title and subtitle fonts
        self.title_fontdict = {
            'fontsize': 14,
            'weight': 'bold'
        }
        self.subtitle_fontdict = {
            'fontsize': 12,
            'weight': 'normal'
        }
        #––– Title / subtitle spacing tweaks –––#
        # How high up the figure the suptitle sits (default ≈0.98)
        self.suptitle_y = 0.998 
        # How high up the subtitle sits (must be < suptitle_y)
        self.subtitle_y = 0.972
        # How far down the axes themselves are pushed (leaves more room at top)
        self.subplot_top = 0.88 # not used anymore with the whole plot-- clashes 
        # You can also set a global default for all figures if you like:
        # plt.rcParams['figure.subplot.top'] = self.subplot_top
        self.taste_replacements = {
            "0": "NaCl",
            "1": "Sucrose",
            "2": "Citric Acid",
            "3": "Quinine",
        }
        self._inv_taste_replacements = {v: k for k, v in self.taste_replacements.items()}
    
    def _save_figure(
        self,
        fig: plt.Figure,
        dir_title: str,
        title: str, 
        core_name: str,
        analysis_type: str,
        tastes: List[Any]
    ):
        """
        General method to save a figure under output_dir/title_folder/core_name/taste/analysis_type/{PNG,SVG}.
        """
        # Normalize title for folder name
        if not tastes:
            print("Warning: No tastes provided; saving to data.")
            tastes = ["data"]

        folder_name = re.sub(r'[^0-9A-Za-z]+', '_', dir_title).lower().strip('_')
        title_dir = os.path.join(self.output_dir, folder_name)
        dataset_dir = os.path.join(title_dir, core_name)
        plot_type = analysis_type.lstrip('_')
        for taste in tastes:
            taste_dir = os.path.join(dataset_dir, f"taste_{taste}")
            plot_dir = os.path.join(taste_dir, plot_type)
            png_dir = os.path.join(plot_dir, 'PNG')
            svg_dir = os.path.join(plot_dir, 'SVG')
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(svg_dir, exist_ok=True)
            png_path = os.path.join(png_dir, f"{title}_taste{taste}.png")
            svg_path = os.path.join(svg_dir, f"{title}_taste{taste}.svg")
            fig.savefig(png_path, dpi=self.dpi)
            fig.savefig(svg_path, dpi=self.dpi)
        plt.close(fig)
    
    def get_trial_changepoints(self, changepoints, taste_idx, trial):
        """
        Retrieve the list of changepoint times for a given taste index and trial.
        """
        try:
            return changepoints[taste_idx][trial]
        except (KeyError, IndexError):
            return []
    
    def taste_id_to_name(self, taste_id: int | str) -> str:
        """
        Convert a taste ID (int or numeric‐string) to its name.
        """
        key = str(taste_id)
        try:
            return self.taste_replacements[key]
        except KeyError:
            raise ValueError(f"Unknown taste ID: {taste_id!r}")

    def taste_name_to_id(self, taste_name: str) -> int:
        """
        Convert a taste name back to its integer ID.
        """
        try:
            return int(self._inv_taste_replacements[taste_name])
        except KeyError:
            raise ValueError(f"Unknown taste name: {taste_name!r}")


    # ---- static utils --- # 
    @staticmethod
    def get_meta_columns(df: pl.DataFrame) -> List[str]:
        """
        Identify and return meta columns in the DataFrame, mapping 'state' to 'changepoint'.
        """
        cols = df.columns
        if 'state' in cols and 'changepoint' not in cols:
            df.rename({'state': 'changepoint'})
            cols = df.columns
        return [c for c in PlottingPipeline.META_COLS if c in cols]

    @staticmethod
    def get_data_columns(df: pl.DataFrame) -> List[str]:
        """
        Return a list of column names corresponding to data vectors.
        """
        return [
            col
            for col in df.columns
            if col.startswith('PC_') or col.startswith('latent_dim_') or col.startswith('neuron_')
        ]

    @staticmethod
    def parse_dataset_name(name: str) -> Tuple[str, str]:
        """
        Split a compound key into (core_dataset_name, analysis_type).
        Assumes the core name ends with a timestamp YYMMDD_HHMMSS.
        """
        pattern = re.compile(r'^(.+?\d{6}_\d{6})(_.+)$')
        m = pattern.match(name)
        if m:
            return m.group(1), m.group(2)
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], f'_{parts[1]}'
        return name, ''
    
    

