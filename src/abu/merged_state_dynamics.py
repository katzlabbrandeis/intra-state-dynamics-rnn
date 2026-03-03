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
from sklearn.decomposition import PCA
from scipy import signal

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
                n_units=valid_spike_trains.shape[1],
                trial_id=valid_trial_ids,
                valid_state_onsets=list(valid_state_onsets),
                valid_state_durations=list(valid_state_durations),
                change_ids=list(change_ids),
                spike_trains=list(valid_spike_trains),
                valid_changes=list(valid_changepoints),
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
# Recreate to remove jitter
full_state_df = pd.concat(state_df_list, ignore_index=True)
# Cut by state onsets and duration
tolerance = 1
full_state_df['cut_onset'] = pd.cut(full_state_df['valid_state_onsets'], bins=np.arange(full_state_df['valid_state_onsets'].min(), full_state_df['valid_state_onsets'].max() + tolerance, tolerance))
full_state_df['cut_duration'] = pd.cut(full_state_df['valid_state_durations'], bins=np.arange(full_state_df['valid_state_durations'].min(), full_state_df['valid_state_durations'].max() + tolerance, tolerance))

# Group by taste, n_states, and cut bins
grouped = full_state_df.groupby(['taste_name', 'n_states', 'cut_onset', 'cut_duration'])

for group_ind, this_group in grouped:
    taste_name, n_states, cut_onset, cut_duration = group_ind
    if len(this_group) > 5:
        break

    fig, ax = plt.subplots(len(this_group), 1, sharex=True) 
    for this_ax, (trial_ind, this_trial) in zip(ax,this_group.iterrows()):
        this_ax.imshow(this_trial['spike_trains'], aspect='auto', cmap='jet')
        for this_change in this_trial['valid_changes']:
            this_ax.axvline(this_change, color='yellow', linestyle='--')
    fig.suptitle(f'{taste_name} - {n_states} states - Onset: {cut_onset} - Duration: {cut_duration}')
    plt.tight_layout()
    plt.show()

    unit_counts = this_group.n_units.values 
    unit_ind_vec = np.concatenate([np.ones(unit_count) * ind for ind, unit_count in enumerate(unit_counts)])
    all_spikes = np.concatenate(this_group['spike_trains'].values, axis=0)
    onset = int(this_group.valid_state_onsets.values[0])
    duration = int(this_group.valid_state_durations.values[0])
    state_spikes = all_spikes[:, onset:onset+duration] 
    zscore_spikes = stats.zscore(state_spikes, axis=1)
    # Drop any units with NaNs (e.g. zero variance)
    valid_units = ~np.isnan(zscore_spikes).any(axis=1)
    zscore_spikes = zscore_spikes[valid_units]
    pca_obj = PCA(3).fit(zscore_spikes.T)
    pca_spikes = pca_obj.transform(zscore_spikes.T).T
    loadings = pca_obj.components_.T

    # Filter pca_spikes using Savitzky-Golay filter for better visualization
    filter_window = 3
    filter_polyorder = 2
    filt_pca_spikes = signal.savgol_filter(pca_spikes, filter_window, filter_polyorder, axis=1)

    cmap = plt.get_cmap('tab10')
    fig,ax = plt.subplots(5,1, figsize=(4,15))
    ax[0].imshow(zscore_spikes, aspect='auto', cmap='viridis', interpolation='nearest')
    for pca_ind in range(pca_spikes.shape[0]):
        ax[1].plot(filt_pca_spikes[pca_ind], c=cmap(pca_ind), linewidth=2)
        ax[1].plot(pca_spikes.T, alpha=0.1, linestyle='--', c=cmap(pca_ind))
    ax[2].imshow(loadings, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[3].bar(np.arange(loadings.shape[1]), pca_obj.explained_variance_ratio_)
    ax[4].imshow(unit_ind_vec[:,None], aspect='auto', cmap='tab20', interpolation='nearest')
    ax[0].set_title('Z-scored Spikes')
    ax[1].set_title('PCA Projection')
    ax[2].set_title('PCA Loadings')
    ax[3].set_title('Explained Variance Ratio')
    ax[4].set_title('Unit Grouping')
    plt.tight_layout()
    plt.show()


    

