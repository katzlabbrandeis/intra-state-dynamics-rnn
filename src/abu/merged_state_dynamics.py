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

state_df_list = []
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
    valid_changes = all_changes[valid_trials]

    # Extract onset times and durations of each state for valid trials
    valid_state_onsets = valid_changes[:, :-1]
    valid_state_durations = state_durations[valid_trials]
    valid_trial_ids = np.where(valid_trials)[0]

    change_ids = np.tile(np.arange(n_states), (len(valid_trial_ids),1)) 
    this_valid_df = pd.DataFrame(
            dict(
                basename=basename,
                taste_name=taste_name,
                n_states=n_states,
                trial_id=valid_trial_ids,
                valid_state_onsets=list(valid_state_onsets),
                valid_state_durations=list(valid_state_durations),
                change_ids=list(change_ids)
                )
            )
    # Explode
    this_valid_df = this_valid_df.explode(['valid_state_onsets', 'valid_state_durations', 'change_ids'])

    state_df_list.append(this_valid_df)

    # fig, ax = vz.firing_overview(valid_spike_trains, cmap_lims='shared')
    # for these_changes, this_ax in zip(valid_changepoints, ax.flatten()):
    #     for this_change in these_changes:
    #         this_ax.axvline(this_change, color='yellow', linestyle='--')
    # # plt.show()
    # fig.suptitle(f'{basename} - {taste_name} - {n_states} states')
    # plt.tight_layout()
    # fig.savefig(os.path.join(plot_subdir, f'{basename}_{taste_name}_{n_states}states_changepoint_raster.png'))
    # plt.close(fig)
    #

full_state_df = pd.concat(state_df_list, ignore_index=True)
# Add some jitter to onsets and durations for better visualization
jitter_mag = 0.5
full_state_df['valid_state_onsets'] = full_state_df['valid_state_onsets'].apply(lambda x: x + np.random.uniform(-jitter_mag, jitter_mag))
full_state_df['valid_state_durations'] = full_state_df['valid_state_durations'].apply(lambda x: x + np.random.uniform(-jitter_mag, jitter_mag))

# Plot state durations by n_states and taste

# g = sns.relplot(
#         data=full_state_df,
#         x='valid_state_onsets',
#         y='valid_state_durations',
#         hue='change_ids',
#         col='n_states',
#         row='taste_name',
#         kind='scatter',
#         )
# underlay with kde
# g = sns.displot(
#         data=full_state_df,
#         x='valid_state_onsets',
#         y='valid_state_durations',
#         col='n_states',
#         row='taste_name',
#         kind='hist',
#         alpha=0.5,
#         # Adjust the bandwidth for better visualization
#         # bw_adjust=0.5
#         )
# plt.colorbar()
# g.set_axis_labels('State Onset Time (ms)', 'State Duration (ms)')
# plt.tight_layout()
# g.savefig(os.path.join(plot_dir, 'state_durations_by_onset_time.png'))
# plt.close(g.fig)

# Make plot manually
unique_tastes = full_state_df['taste_name'].unique()
unique_n_states = full_state_df['n_states'].unique()
fig, axes = plt.subplots(
    len(unique_tastes), len(unique_n_states),
    figsize=(5*len(unique_n_states), 4*len(unique_tastes)),
    sharex=True, sharey=True
    )
group_data = full_state_df.groupby(['taste_name', 'n_states'])
# First plot histograms of state onsets and durations
for (taste_name, n_states), group in group_data:
    ax = axes[np.where(unique_tastes == taste_name)[0][0], np.where(unique_n_states == n_states)[0][0]]
    ax.hist2d(
        group['valid_state_onsets'], group['valid_state_durations'],
        bins=30, cmap='Blues'
        )
    ax.set_title(f'{taste_name} - {n_states} states')
    # Group by change_id and plot scatter
    for change_id, change_group in group.groupby('change_ids'):
        ax.scatter(
            change_group['valid_state_onsets'], change_group['valid_state_durations'],
            label=f'Change {change_id}', alpha=0.5, s=5
            )
    ax.legend()
    ax.set_xlabel('State Onset Time (ms)')
    ax.set_ylabel('State Duration (ms)')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'state_durations_by_onset_time_manual.png'))
plt.close(fig)

##############################
