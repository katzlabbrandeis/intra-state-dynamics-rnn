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

##############################
# data_dir_file = os.path.join(base_dir, 'src', 'GC_EMG_alignment', 'data_dir_list.txt')
data_dir_file = '/media/bigdata/abu_resorted_rolling/abu_all_datasetes.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

pp(data_dir_list)

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
    artifacts_subdir, 'best_model_info_df.pkl'
    )
with open(best_models_df_fp, 'rb') as f:
    best_models_df = load(f)

##############################

fit_data_list = []
for this_dir in tqdm(data_dir_list, desc='Processing data dirs'):

    this_dat = ephys_data.ephys_data(this_dir)
    this_dat.profile_units(recalculate=False)
    basename = os.path.basename(this_dir)

    if 'spikes' not in dir(this_dat):
        this_dat.get_spikes()
        this_dat.get_info_dict()

    this_dat.check_laser()
    if this_dat.laser_exists:
        this_dat.separate_laser_spikes()
        off_spikes = this_dat.off_spikes
    else:
        off_spikes = this_dat.spikes

    # Only stable units
    stable_alpha = 0.005
    stable_units = this_dat.unit_profile[
            this_dat.unit_profile['stable_pval'] > stable_alpha
            ]['neuron_num'].values


    selected_units = stable_units  # For now, just use stable units for both conditions

    taste_names = this_dat.info_dict['taste_params']['tastes']

    for taste_ind, taste_name in enumerate(taste_names):
        # Extract spike trains for selected units
        spike_trains = off_spikes[taste_ind]
        time_lims = [1500, 4000]
        spike_trains = spike_trains[:, :, time_lims[0]:time_lims[1]]  # Trim to time window
        only_stable_spikes = spike_trains[:, stable_units]
        selected_spikes = spike_trains[:, selected_units]

        fit_data_list.append({
            'basename': basename,
            'taste_name': taste_name,
            'dat_type': 'stable_only',
            'spike_trains': only_stable_spikes,
        })

# Combine into a dataframe and write to disk
fit_data_df = pd.DataFrame(fit_data_list)
fit_data_df['n_neurons'] = fit_data_df['spike_trains'].apply(lambda x: x.shape[1])
# Drop any datasets with <8 neurons
fit_data_df = fit_data_df[fit_data_df['n_neurons'] >= 8].reset_index(drop=True)

##############################
# Keep only rows that are in best_models_df
best_fit_data_df = fit_data_df.merge(
    best_models_df,
    on=['basename', 'taste_name', 'dat_type'],
    how='inner',
    suffixes=('_fit_data', '_model_info')
)

# Save best fit data dataframe
best_fit_data_df_fp = os.path.join(
    artifacts_subdir, 'best_fit_data_df.pkl'
    )
with open(best_fit_data_df_fp, 'wb') as f:
    dump(best_fit_data_df, f)

# Write out a note in artifacts_subdir to indicate what library and version dump function is from
dump_note_fp = os.path.join(artifacts_subdir, 'dump_note.txt')
with open(dump_note_fp, 'w') as f:
    f.write(f'This directory contains dataframes dumped using cloudpickle version {cloudpickle.__version__}')

##############################

# Load best fit data dataframe and save spike rasters as spike-times
best_fit_data_df_fp = os.path.join(
    artifacts_subdir, 'best_fit_data_df.pkl'
    )
best_fit_data_df = pd.read_pickle(best_fit_data_df_fp)


# Generate splits and save to disk
n_splits = 10
train_test_frac = 0.75 # Splitting along neuron dimension, so this is the fraction of neurons to use for training
split_data_list = []
for idx, row in tqdm(best_fit_data_df.iterrows()):
    spike_trains = row['spike_trains']
    original_array_shape = spike_trains.shape

    n_neurons = spike_trains.shape[1]
    neuron_indices = np.arange(n_neurons)
    for split_ind in range(n_splits):
        np.random.shuffle(neuron_indices)
        train_neurons = neuron_indices[:int(train_test_frac * n_neurons)]
        test_neurons = neuron_indices[int(train_test_frac * n_neurons):]

        train_spike_trains = spike_trains[:, train_neurons, :]
        test_spike_trains = spike_trains[:, test_neurons, :]

        train_spike_times, train_shape = SpikeRasterIO.spike_train_to_spike_times(train_spike_trains)
        test_spike_times, test_shape = SpikeRasterIO.spike_train_to_spike_times(test_spike_trains)

        split_data_list.append({
            'basename': row['basename'],
            'taste_name': row['taste_name'],
            'dat_type': row['dat_type'],
            'split_ind': split_ind,
            'train_neurons': train_neurons,
            'test_neurons': test_neurons,
            'train_spike_times': train_spike_times,
            'train_shape': train_shape,
            'test_spike_times': test_spike_times,
            'test_shape': test_shape
        })

# Combine into a dataframe and write to disk
split_data_df = pd.DataFrame(split_data_list)
split_data_df_fp = os.path.join(
    artifacts_subdir, 'neuron_split_data_df.pkl'
    )
with open(split_data_df_fp, 'wb') as f:
    dump(split_data_df, f)

##############################
# Also generate a temporal split (across trials)

n_splits = 10
train_test_frac = 0.75 # Splitting along neuron dimension, so this is the fraction of neurons to use for training
split_data_list = []
for idx, row in tqdm(best_fit_data_df.iterrows()):
    spike_trains = row['spike_trains']
    original_array_shape = spike_trains.shape

    n_trials = spike_trains.shape[0]
    trial_indices = np.arange(n_trials)
    for split_ind in range(n_splits):
        np.random.shuffle(trial_indices)
        train_trials = trial_indices[:int(train_test_frac * n_trials)]
        test_trials = trial_indices[int(train_test_frac * n_trials):]

        train_spike_trains = spike_trains[train_trials, :, :]
        test_spike_trains = spike_trains[test_trials, :, :]

        train_spike_times, train_shape = SpikeRasterIO.spike_train_to_spike_times(train_spike_trains)
        test_spike_times, test_shape = SpikeRasterIO.spike_train_to_spike_times(test_spike_trains)

        split_data_list.append({
            'basename': row['basename'],
            'taste_name': row['taste_name'],
            'dat_type': row['dat_type'],
            'split_ind': split_ind,
            'train_trials': train_trials,
            'test_trials': test_trials,
            'train_spike_times': train_spike_times,
            'train_shape': train_shape,
            'test_spike_times': test_spike_times,
            'test_shape': test_shape
        })

# Combine into a dataframe and write to disk
split_data_df = pd.DataFrame(split_data_list)
split_data_df_fp = os.path.join(
    artifacts_subdir, 'trial_split_data_df.pkl'
    )
with open(split_data_df_fp, 'wb') as f:
    dump(split_data_df, f)
