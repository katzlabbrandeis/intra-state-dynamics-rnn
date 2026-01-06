"""
Analysis and plots for single neuron analysis of Calia-Bogan 2025
1- Paired test on first vs last half of states for single-neurons
2- Plot of warped single-neuron firing rates
"""

import blech_clust as bc
import pandas as pd
from glob import glob
import numpy as np
import os
from tqdm import tqdm
from pprint import pprint as pp

tqdm.pandas()

import sys
sys.path.append('/media/bigdata/projects/pytau/')
import pytau
from pytau.changepoint_analysis import get_state_snippets

cp_file_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/pkl_files/tau_frame.pkl'
tau_frame = pd.read_pickle(cp_file_path)

##############################
# Get spike-trains

spike_train_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/spike_trains_npz'
spike_train_files = glob(f'{spike_train_dir}/*.npz')
basename_list = [os.path.basename(f).replace('_repacked.npz', '') for f in spike_train_files]

# From both basename_list and tau_frame['basename'], drop any "_repacked" or "_copy" suffixes
basename_list = [name.replace('_repacked', '').replace('_copy', '') for name in basename_list]
tau_frame['basename'] = tau_frame['basename'].str.replace('_repacked', '').str.replace('_copy', '')

# Load all spike train files into a dict
spike_train_dict = {this_name: np.load(this_file) for this_name, this_file in zip(basename_list, spike_train_files)}

# All files have only 1 array but with different key names
spike_train_dict = {k: v[list(v.files)[0]] for k, v in spike_train_dict.items()}

##############################
# Get state snippets for all neurons
state_snippet_list = []
for ind, row in tqdm(tau_frame.iterrows()):
    session_name = row['basename']
    # Shape: (trials, changepoints)
    change_points = row['tau']
    
    if np.isnan(change_points).all():
        print(f"Skipping {session_name} taste {row['taste_num']} due to all NaN change points")
        continue

    taste_ind = row['taste_num']
    # Shape: (trials, neurons, time)
    spike_trains = spike_train_dict[session_name][int(taste_ind)]
    # get_state_snippets(spike_array, tau_array)
    # Extract neural activity snippets for each state and trial without averaging
    # 
    # Returns raw neural activity for each state as a ragged array structure,
    # where each state can have different durations across trials.
    # 
    # Args:
    #     spike_array (np.ndarray): Neural activity data
    #         Shape: (n_trials, n_neurons, n_bins)
    #     tau_array (np.ndarray): Changepoint positions for each trial
    #         Shape: (n_trials, n_changepoints)
    # 
    # Returns:
    #     list: Nested list structure organized as [state][trial]
    #         - Outer list length: n_states (n_changepoints + 1)
    #         - Inner list length: n_trials
    #         - Each element shape: (n_neurons, bins_in_state)
    #         Note: bins_in_state varies by trial and state
    state_snippets = get_state_snippets(spike_trains, change_points) 

    # Add to dict individually for each neuron
    for state_ind in range(len(state_snippets)):
        for trial_ind in range(len(state_snippets[state_ind])):
            for neuron_ind in range(spike_trains.shape[1]):
                state_snippet_list.append({
                    'basename': session_name,
                    'taste_num': taste_ind,
                    'state_ind': state_ind,
                    'trial_ind': trial_ind,
                    'neuron_ind': neuron_ind,
                    'spike_data': state_snippets[state_ind][trial_ind][neuron_ind]
                })

# Convert to pandas DataFrame for easier handling
state_snippet_df = pd.DataFrame(state_snippet_list)

##############################
# For each trial, neuron, state: compute firing rates in first vs last half of state snippet
def mean_rate_halves(row):
    spike_data = row['spike_data']
    n_bins = len(spike_data)
    if n_bins < 2:
        return pd.Series({'mean_rate_first_half': np.nan, 'mean_rate_last_half': np.nan})
    
    half_point = n_bins // 2
    first_half = spike_data[:half_point]
    last_half = spike_data[half_point:]
    
    mean_rate_first_half = np.mean(first_half)
    mean_rate_last_half = np.mean(last_half)
    
    # Add results to row
    return pd.Series({'mean_rate_first_half': mean_rate_first_half, 'mean_rate_last_half': mean_rate_last_half})

state_snippet_df[['mean_rate_first_half', 'mean_rate_last_half']] = state_snippet_df.progress_apply(mean_rate_halves, axis=1)
