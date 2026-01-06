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
from scipy.stats import ttest_rel
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

tqdm.pandas()

import sys
sys.path.append('/media/bigdata/projects/pytau/')
import pytau
from pytau.changepoint_analysis import get_state_snippets

cp_file_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/pkl_files/tau_frame.pkl'
tau_frame = pd.read_pickle(cp_file_path)

##############################
base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
artifacts_sup_dir = os.path.join(base_dir, 'output', 'artifacts')
artifacts_dir = os.path.join(artifacts_sup_dir, 'single_neuron_analysis')
plot_dir = os.path.join(base_dir, 'output', 'plots', 'single_neuron_analysis')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)
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
time_lims = [2000, 4000]
state_snippet_list = []
for ind, row in tqdm(tau_frame.iterrows()):
    session_name = row['basename']
    # Shape: (trials, changepoints)
    change_points = row['tau']
    change_points -= time_lims[0]  # adjust to start at 0
    
    if np.isnan(change_points).all():
        print(f"Skipping {session_name} taste {row['taste_num']} due to all NaN change points")
        continue

    taste_ind = row['taste_num']
    # Shape: (trials, neurons, time)
    spike_trains = spike_train_dict[session_name][int(taste_ind)]
    # Cut to time_lims
    spike_trains = spike_trains[:, :, time_lims[0]:time_lims[1]]
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

# Write intermediate DataFrame to artifact
intermediate_artifact_path = os.path.join(artifacts_dir, 'state_snippet_frame.pkl')
state_snippet_df.to_pickle(intermediate_artifact_path)

# For each basenames, taste_num, state_ind, neuron_ind: perform paired t-test across trials
def paired_t_test(group):
    # Drop NaN values
    group = group.dropna(subset=['mean_rate_first_half', 'mean_rate_last_half'])
    if len(group) < 2:
        return pd.Series({'t_stat': np.nan, 'p_value': np.nan})
    
    t_stat, p_value = ttest_rel(group['mean_rate_first_half'], group['mean_rate_last_half'])

    mean_rate = np.mean(group[['mean_rate_first_half', 'mean_rate_last_half']].values.flatten())

    return pd.Series({'t_stat': t_stat, 'p_value': p_value, 'mean_rate': mean_rate})

paired_test_results = \
        state_snippet_df.groupby(['basename', 'taste_num', 'state_ind', 'neuron_ind']).progress_apply(paired_t_test).reset_index()

# Calculate number of repeated measures for each neuron (tastes and states)
def count_repeated_measures(group):
    return len(group)
repeated_measures = \
    paired_test_results.groupby(['basename', 'neuron_ind']).progress_apply(count_repeated_measures).reset_index(name='n_repeated_measures')

# Merge so we can calculate corrected alpha
paired_test_results = paired_test_results.merge(repeated_measures, on=['basename', 'neuron_ind'])
# Bonferroni correction for multiple comparisons per neuron
paired_test_results['corrected_alpha'] = 0.05 / paired_test_results['n_repeated_measures']

# Convert mean_rate to Hz (current bins are 1ms)
paired_test_results['mean_rate_Hz'] = paired_test_results['mean_rate'] * 1000

# Save results as artifact
paired_test_artifact_path = os.path.join(artifacts_dir, 'paired_test_results.pkl')
paired_test_results.to_pickle(paired_test_artifact_path)

############################################################
# Make plots

if 'paired_test_results' not in globals():
    paired_test_artifact_path = os.path.join(artifacts_dir, 'paired_test_results.pkl')
    paired_test_results = pd.read_pickle(paired_test_artifact_path)

if 'state_snippet_df' not in globals():
    intermediate_artifact_path = os.path.join(artifacts_dir, 'state_snippet_frame.pkl')
    state_snippet_df = pd.read_pickle(intermediate_artifact_path)


paired_test_results['sig'] = paired_test_results['p_value'] < paired_test_results['corrected_alpha']

# For each neuron, check if any state shows significant change
def neuron_significance(group):
    any_sig = group['sig'].any()
    lowest_p = group['p_value'].min()
    n_sig = group['sig'].sum()
    return pd.Series({'any_significant': any_sig, 'lowest_p_value': lowest_p, 'n_significant_states': n_sig})
neuron_sig_results = \
    paired_test_results.groupby(['basename', 'neuron_ind']).progress_apply(neuron_significance).reset_index()

grand_mean_rate_df = \
        paired_test_results.groupby(['basename', 'neuron_ind']).agg({'mean_rate_Hz': 'mean'}).reset_index()
neuron_sig_results = neuron_sig_results.merge(grand_mean_rate_df, on=['basename', 'neuron_ind'])

# make venn diagram with groups:
# 1- all
# 2- units with > 2Hz mean firing rate
# 3- significant units 
neuron_sig_results['id'] = neuron_sig_results['basename'] + '_neuron_' + neuron_sig_results['neuron_ind'].astype(str) 
all_units = set(neuron_sig_results['id'])
high_rate_units = set(neuron_sig_results[neuron_sig_results['mean_rate_Hz'] > 2]['id'])
significant_units = set(neuron_sig_results[neuron_sig_results['any_significant']]['id'])

fig, ax = plt.subplots(figsize=(4, 4))
# Distinct colors for each set
set_colors = ('#1f77b4', '#ff7f0e', '#2ca02c')
v = venn3([all_units, high_rate_units, significant_units],
      set_labels = ('All Units', '>2Hz Mean Rate', 'Significant Change'),
      ax=ax, set_colors=set_colors, alpha=0.7)
c = venn3_circles([all_units, high_rate_units, significant_units], ax=ax)
# Draw outlines
for circle in c:
    circle.set_lw(1.0)
# # Hide subset labels
# for text in v.subset_labels:
#     text.set_visible(False)
# Set text color to same as circle
for text, color in zip(v.set_labels, set_colors):
    text.set_color(color)
plt.title('Venn Diagram of Single Neuron Analysis')
venn_plot_path = os.path.join(plot_dir, 'single_neuron_analysis_venn.svg')
plt.savefig(venn_plot_path, bbox_inches='tight')
plt.close(fig)

###############
# Plot traces of warped firing rates for significant neurons

# For each neuron, plot both warped and unwarped firing rates for all states for a single taste
n_plots = 20
# Sort by highest mean firing rate and significance
sorted_neurons = neuron_sig_results.sort_values(by=['n_significant_states', 'mean_rate_Hz'], ascending=False).head(n_plots) 
# Add rank for each neuron for easier plotting
sorted_neurons['rank'] = range(1, len(sorted_neurons) + 1)

wanted_snippets = state_snippet_df.merge(
    sorted_neurons[['basename', 'neuron_ind', 'rank']],
    on=['basename', 'neuron_ind']
)



grouped_snippets = wanted_snippets.groupby(['basename', 'neuron_ind','taste_num'])

this_plot_dir = os.path.join(plot_dir, 'rate_plots') 
os.makedirs(this_plot_dir, exist_ok=True)

# Plot layout: rows = (unwarped, warped), columns = states
# bin_size = 50  # in ms
kernel_width = 100  # in ms
for (basename, neuron_ind, taste_num), group in grouped_snippets:
    states = sorted(group.state_ind.unique())
    n_states = len(states)
    
    # Create subplot grid: 2 rows (unwarped, warped) x n_states columns
    fig, axs = plt.subplots(2, n_states, figsize=(4*n_states, 6)) 
    
    # Handle case where there's only one state
    if n_states == 1:
        axs = axs.reshape(2, 1)
    
    # Process each state
    state_data = {}
    for col_idx, this_state in enumerate(states):
        state_group = group[group['state_ind'] == this_state]
        # Get spike_data as list of arrays
        spike_data_list = state_group['spike_data'].tolist()

        binned_spike_data_list = []
        for arr in spike_data_list:
            # Bin the spike data
            n_bins = int(np.ceil(len(arr) / bin_size))
            binned = np.array([np.mean(arr[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
            binned_spike_data_list.append(binned)
            # # Smooth with boxcar kernel
            # if len(arr) < kernel_width:
            #     smoothed = arr
            # else:
            #     kernel = np.ones(kernel_width) / kernel_width
            #     smoothed = np.convolve(arr, kernel, mode='same')
            # binned_spike_data_list.append(smoothed)

        # Smooth firing rates with Savitzky-Golay filter
        smoothed_spike_data_list = []
        for arr in binned_spike_data_list:
            if len(arr) < 5:
                smoothed = arr
            else:
                smoothed = savgol_filter(arr, window_length=5, polyorder=2)
            smoothed_spike_data_list.append(smoothed)
        
        # Warp firing rates to mean length
        lengths = [len(arr) for arr in smoothed_spike_data_list]
        mean_length = int(np.mean(lengths))
        
        warped_firing_rates = []
        for arr in smoothed_spike_data_list:
            if len(arr) < 2:
                warped = np.full(mean_length, np.nan)
            else:
                warped = np.interp(
                    np.linspace(0, len(arr)-1, mean_length),
                    np.arange(len(arr)),
                    arr
                )
            warped_firing_rates.append(warped)
        
        warped_firing_rates = np.array(warped_firing_rates)
        mean_warped_rate = np.nanmean(warped_firing_rates, axis=0)
        
        # Plot unwarped firing rates (top row)
        for trial_rate in smoothed_spike_data_list:
            axs[0, col_idx].plot(trial_rate, color='gray', alpha=0.5)
        axs[0, col_idx].set_title(f'State {this_state}')
        if col_idx == 0:
            axs[0, col_idx].set_ylabel('Firing Rate (spikes/ms)')
        
        # Plot warped firing rates (bottom row)
        for trial_rate in warped_firing_rates:
            axs[1, col_idx].plot(trial_rate, color='gray', alpha=0.5)
        axs[1, col_idx].plot(mean_warped_rate, color='red', linewidth=2, label='Mean' if col_idx == 0 else '')
        axs[1, col_idx].set_xlabel('Warped Time Bins')
        if col_idx == 0:
            axs[1, col_idx].set_ylabel('Firing Rate (spikes/ms)')
            axs[1, col_idx].legend()
    
    # Add overall title
    suptitle_str = f'{basename} Neuron {neuron_ind} Taste {taste_num}\nTop: Unwarped, Bottom: Warped'
    # Also add mean firing rate info and significance info
    neuron_info = sorted_neurons[
        (sorted_neurons['basename'] == basename) & 
        (sorted_neurons['neuron_ind'] == neuron_ind)
    ].iloc[0]
    nrn_rank = neuron_info['rank']

    suptitle_str += f'\nMean Firing Rate: {neuron_info["mean_rate_Hz"]:.2f} Hz, Significant States: {neuron_info["n_significant_states"]}'

    fig.suptitle(suptitle_str, fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(
        this_plot_dir,
        f'rank{nrn_rank}_{basename}_neuron_{neuron_ind}_taste_{taste_num}_all_states_firing_rates.svg'
    )
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
