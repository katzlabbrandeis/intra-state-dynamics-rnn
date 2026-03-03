"""
Match inferred states across multiple datasets
Arguably, if a state is of similar onset time and duration across a different neural population, it should have similar dynamics
By this logic, we should be able to pool units into a pseudo-population to visualize it's dynamics
"""

from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data import visualize as vz
from tqdm import tqdm
from pprint import pprint as pp
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import pingouin as pg
import cloudpickle
from cloudpickle import load, dump
import matplotlib.pyplot as plt

import sys
import os
base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src', 'abu', 'model_fitting', 'population_analysis')
sys.path.append(src_dir)

from utils import SpikeRasterIO

##############################
output_dir = os.path.join(base_dir, 'output')
artifacts_dir = os.path.join(output_dir, 'artifacts')

artifacts_subdir = os.path.join(artifacts_dir, 'population_analysis')
change_out_dir = os.path.join(artifacts_subdir, 'models')

# plot_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/plots/changepoint_analysis'
plot_dir = os.path.join(base_dir, 'output', 'plots', 'changepoint_analysis')

##############################
# # Save best model info dataframe
# best_models_df_fp = os.path.join(
#     artifacts_subdir, 'best_model_info_df.pkl'
#     )
# with open(best_models_df_fp, 'wb') as f:
#     dump(best_models_df, f)
#
# Load best model info dataframe
best_models_df_fp = os.path.join(
    artifacts_dir, 'ephys_changepoint_models', 'best_model_info_df.pkl'
    )
with open(best_models_df_fp, 'rb') as f:
    best_models_df = load(f)

# For each model, extract spike-trains and inferred transitions
# Also append changepoints to best_fit_data_df
model_save_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/artifacts/ephys_changepoint_models/models'
plot_subdir = os.path.join(plot_dir, 'changepoint_rasters')
os.makedirs(plot_subdir, exist_ok=True)
for row_ind, this_row in tqdm(best_models_df.iterrows()):

    basename = this_row['basename']
    taste_name = this_row['taste_name']
    n_states = int(this_row['n_states'])

    save_name = this_row['save_path']
    save_path = os.path.join(model_save_dir, save_name)
    with open(save_path, 'rb') as f:
        model_tuple = load(f)
    spike_trains = model_tuple[-3]
    changepoints = model_tuple[-2]

    # Model params:
    # time_lims = [2000, 4000]
    # bin_size = 50

    all_changes = np.concatenate(
            [
                np.zeros(len(changepoints))[:,None],
                changepoints,
                np.ones(len(changepoints))[:,None] * (2000//50)
                ], axis=1
            )
    state_durations = np.diff(all_changes, axis=1)

    # Remove any trials with insufficient duration in any state
    min_state_duration_time = 250 # ms
    min_state_duration_bins = min_state_duration_time // 50
    valid_trials = np.all(state_durations >= min_state_duration_bins, axis=1)

    valid_spike_trains = spike_trains[valid_trials]
    valid_changepoints = changepoints[valid_trials]

    fig, ax = vz.firing_overview(valid_spike_trains, cmap_lims='shared')
    for these_changes, this_ax in zip(valid_changepoints, ax.flatten()):
        for this_change in these_changes:
            this_ax.axvline(this_change, color='yellow', linestyle='--')
    # plt.show()
    fig.suptitle(f'{basename} - {taste_name} - {n_states} states')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_subdir, f'{basename}_{taste_name}_{n_states}states_changepoint_raster.png'))
    plt.close(fig)

