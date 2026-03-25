"""
ENV: pytau0.3

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
from scipy.stats import ttest_rel, percentileofscore
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from cloudpickle import dump, load


tqdm.pandas()

import sys
sys.path.append('/media/bigdata/projects/pytau/')
import pytau
from pytau.changepoint_analysis import get_state_snippets
import pymc as pm

cp_file_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/pkl_files/tau_frame.pkl'
tau_frame = pd.read_pickle(cp_file_path)

sys.path.append('/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/src/abu/model_fitting/population_analysis/')
import utils

from importlib import reload
reload(utils)

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

reload_data_bool = False

if reload_data_bool:
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

##############################
# For each basenames, taste_num, state_ind, neuron_ind: perform paired t-test across trials
def paired_t_test(group):
    # Drop NaN values
    group = group.dropna(subset=['mean_rate_first_half', 'mean_rate_last_half'])
    if len(group) < 2:
        return pd.Series({'t_stat': np.nan, 'p_value': np.nan})
    
    t_stat, p_value = ttest_rel(group['mean_rate_first_half'], group['mean_rate_last_half'])

    mean_rate = np.mean(group[['mean_rate_first_half', 'mean_rate_last_half']].values.flatten())

    return pd.Series({'t_stat': t_stat, 'p_value': p_value, 'mean_rate': mean_rate})

if reload_data_bool:
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

##############################
# Bar plot for fold-changes before vs after state change for significant neurons
significant_rows = paired_test_results[paired_test_results['sig']]

significant_rows = significant_rows.merge(
        state_snippet_df.groupby(['basename', 'neuron_ind', 'state_ind']).agg({
            'mean_rate_first_half': 'mean',
            'mean_rate_last_half': 'mean'
        }).reset_index(),
        on=['basename', 'neuron_ind', 'state_ind']
        )

significant_rows['fold_change'] = (significant_rows['mean_rate_last_half'] + 1e-6) / (significant_rows['mean_rate_first_half'] + 1e-6)

# calculate log2 fold change for better visualization
significant_rows['log2_fold_change'] = np.log2(significant_rows['fold_change'])
# calculate -log10 p-value for better visualization
significant_rows['neg_log10_p'] = -np.log10(significant_rows['p_value'] + 1e-10)  # add small value to avoid log(0)

# Calculate count of significant changes for each absolute log2 fold change bin
bins = np.arange(0, np.ceil(np.abs(significant_rows['log2_fold_change']).max()) + 1, 0.25)
# Plot but formatted as an inset
fig, ax = plt.subplots(figsize=(2, 2))
bin_counts, bin_edges = np.histogram(significant_rows['log2_fold_change'].abs(), bins=bins)
mode_bin_indices = np.where(bin_counts == bin_counts.max())[0][0]
mode_bin_values = (bin_edges[mode_bin_indices] + bin_edges[mode_bin_indices + 1]) / 2
# ax.bar(bin_centers, bin_counts, width=0.25, edgecolor='black', histtype='step') 
ax.hist(significant_rows['log2_fold_change'].abs(), bins=bins, edgecolor='black', histtype='stepfilled', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Absolute Log2 Fold Change\n(Last Half / First Half)')
ax.set_ylabel('Count of Significant Changes')
ax.set_title('Distribution of Fold Changes for Significant Neurons')
# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Set 0, 2, 4 as x-ticks for better readability
ax.set_xticks([0, 2, 4])
# Draw arrow pointing x-axis position of mode bin
# Not text
ax.annotate('',xy=(mode_bin_values, 0), 
            xytext=(mode_bin_values, 5), 
            arrowprops=dict(facecolor='black', shrink=0.02),
            fontsize=8)
bin_plot_path = os.path.join(plot_dir, 'significant_neurons_fold_change_distribution.svg')
plt.savefig(bin_plot_path, bbox_inches='tight')
plt.close(fig)

# Plot volcano plot of fold changes
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(significant_rows['log2_fold_change'], significant_rows['neg_log10_p'], alpha=0.7) 
ax.set_xlabel('Log2 Fold Change\n(Last Half / First Half)')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Fold Change vs Significance for Significant Neurons')
# ax.set_ylim(0, significant_rows['neg_log10_p'].max() + 1)
corrected_alpha = significant_rows['corrected_alpha'].iloc[0]  # same for all rows of a neuron, just take the first one
ax.axhline(-np.log10(corrected_alpha), 
           color='red', linestyle='--', label=f'Corrected Alpha ({corrected_alpha:.4f})')
ax.legend()
volcano_plot_path = os.path.join(plot_dir, 'significant_neurons_fold_change_volcano.svg')
plt.savefig(volcano_plot_path, bbox_inches='tight')
plt.close(fig)

##############################
# Distrubution of significance by state
state_sig_counts = paired_test_results.groupby('state_ind')['sig'].sum().reset_index(name='n_significant_neurons')
# Normalize by total number of neurons tested in each state
state_sig_frac = state_sig_counts['frac_significant'] = state_sig_counts['n_significant_neurons'] / state_sig_counts['n_significant_neurons'].sum()
fig, ax = plt.subplots(figsize=(2, 2))
ax.bar(state_sig_counts['state_ind'], state_sig_frac, color='skyblue', edgecolor='black')
ax.set_xlabel('State Index')
ax.set_ylabel('Number of Significant Neurons')
ax.set_title('Number of Neurons with Significant Change by State')
ax.set_xticks(state_sig_counts['state_ind'])
# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
state_sig_plot_path = os.path.join(plot_dir, 'significant_frac_neurons_by_state.svg')
plt.savefig(state_sig_plot_path, bbox_inches='tight')
plt.close(fig)

##############################
significant_snippets = significant_rows.merge(
    state_snippet_df,
    on=['basename', 'neuron_ind', 'state_ind', 'taste_num']
    )
significant_snippets.drop(
        columns=[
            'mean_rate_first_half_x', 
            'mean_rate_first_half_y',
            'mean_rate_last_half_x',
            'mean_rate_last_half_y', 
            'fold_change', 
            'log2_fold_change', 
            'neg_log10_p'
            ],
        inplace=True
        )

# Group
significant_snippets_grouped = significant_snippets.groupby(
        ['basename', 'neuron_ind', 'taste_num', 'state_ind'])

this_plot_dir = os.path.join(plot_dir, 'sig_rate_plots')
this_artifact_dir = os.path.join(artifacts_dir, 'sig_rate_plots')
os.makedirs(this_plot_dir, exist_ok=True)
os.makedirs(this_artifact_dir, exist_ok=True)

# ind = 0
# for ind in range(len(significant_snippets_grouped)):
#     this_group = list(significant_snippets_grouped)[ind][1]
all_warped_arrays = dict()
for group_ind, this_group in significant_snippets_grouped:
    group_name_str = "_".join([str(x) for x in group_ind])
    this_spikes = this_group['spike_data'].tolist()

    # Plot raster
    trial_durations = [len(trial) for trial in this_spikes]
    # Sort trials by duration
    sort_inds = np.argsort(trial_durations)
    sorted_durations = [trial_durations[i] for i in sort_inds]
    sorted_spikes = [this_spikes[i] for i in sort_inds]
    sorted_spike_times = [np.where(trial)[0] for trial in sorted_spikes]
    sorted_trial_inds = [np.full_like(times, i) for i, times in enumerate(sorted_spike_times)]
    flat_times = np.concatenate(sorted_spike_times)
    flat_trial_inds = np.concatenate(sorted_trial_inds)

    # Interpolate to same length
    warp_len = 20
    all_interp_spike_times = []
    for i, trial in enumerate(sorted_spike_times):
        interp_spike_times = (trial / sorted_durations[i]) * (warp_len - 1) 
        # Convert to int
        interp_spike_times = np.round(interp_spike_times).astype(int)
        all_interp_spike_times.append(interp_spike_times)

    flat_interp_spike_times = np.concatenate(all_interp_spike_times)
    assert len(flat_interp_spike_times) == len(flat_times)  # should have same number of spikes before and after warping

    # Infer firing rate with PyMC using Gaussian Random Walk prior on warped spikes 
    n_trials = len(this_group)
    n_bins = warp_len  # after warping to 100 bins
    # Convert warped spikes to array
    wapred_array = np.zeros((n_trials, n_bins))
    for i, interp_times in enumerate(all_interp_spike_times):
        for t in interp_times:
            wapred_array[i, t] += 1

    all_warped_arrays[group_name_str] = wapred_array

    with pm.Model() as model:
        hyper_step = pm.Exponential("hyper_step", 0.05)
        step_size = pm.Exponential("step_size", hyper_step)
        lambda_latent = pm.GaussianRandomWalk("volatility", sigma=step_size, 
                        shape=(n_trials, n_bins))
        lambda_ = pm.Deterministic('lambda_', np.exp(lambda_latent))
        data = pm.Data("data", wapred_array)
        rate = pm.Poisson("rate", lambda_, observed=data)

    with model:
        # trace = pm.sample(nuts_sampler="numpyro")
        # trace = pm.sample(draws =500, chains=8, cores=8)
        # Fit with ADVI for speed
        fit = pm.fit(n=50000, method='advi', progressbar=True)
        trace = fit.sample(1000)

    out_dict = {
            'model': model,
            'trace': trace,
            'fit': fit,
            }
    artifact_path = os.path.join(this_artifact_dir, f'{group_name_str}_model_trace.pkl')
    with open(artifact_path, 'wb') as f:
        dump(out_dict, f)

    ppc_list = pm.sample_posterior_predictive(trace, model = model, var_names = ['lambda_'])
    mean_ppc = ppc_list.posterior_predictive.lambda_.mean(axis=(0,1)).values
    grand_mean_rate = mean_ppc.mean(axis=0)

    fig, ax = plt.subplots(2,2,figsize=(4, 4), sharey='row', sharex='col')
    ax[0,0].scatter(flat_times, flat_trial_inds, marker='|')
    for trial_idx, duration in enumerate(sorted_durations):
        ax[0,0].plot(duration, trial_idx, color='red', marker = 'o', alpha=0.5)  # Mark end of trial with red dot
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,0].set_ylabel('Trial Index')
    # Plot warped spikes
    # ax[0,1].scatter(flat_interp_spike_times, flat_trial_inds, marker='|')
    ax[0,1].imshow(wapred_array, aspect='auto', cmap='Greys', origin='lower')
    ax[0,1].set_xlabel('Warped Time Bins')
    ax[0,0].set_title('Unwarped Spike Raster')
    ax[0,1].set_title('Warped Spike Raster')
    ax[1,1].plot(mean_ppc.T, color='red', alpha=0.2)
    ax[1,1].plot(grand_mean_rate, color='black', linewidth=2, label='Grand Mean Rate')
    ax[1,1].set_xlabel('Warped Time Bins')
    ax[1,1].set_title('Inferred Firing Rate from Warped Spikes')
    # ax[1,1].legend()
    # Scale [1,1] to match grand mean rate range
    ax[1,1].set_ylim(grand_mean_rate.min() * 0.9, grand_mean_rate.max() * 1.1)
    fig.suptitle(f'{this_group["basename"].iloc[0]}\nNeuron {this_group["neuron_ind"].iloc[0]} Taste {this_group["taste_num"].iloc[0]} State {this_group["state_ind"].iloc[0]}')
    plt.tight_layout()
    fig.savefig(os.path.join(this_plot_dir, f'{group_name_str}_firing_rate_inference.svg'), bbox_inches='tight')
    plt.close(fig)
    # plt.show()

# Also calcualte rates for shuffled data to compare deviation from flatness
with pm.Model() as shuffled_model:
    hyper_step = pm.Exponential("hyper_step", 0.05)
    step_size = pm.Exponential("step_size", hyper_step)
    lambda_latent = pm.GaussianRandomWalk("volatility", sigma=step_size, 
                    shape=(n_trials, n_bins))
    lambda_ = pm.Deterministic('lambda_', np.exp(lambda_latent))
    data = pm.Data("data", wapred_array)
    rate = pm.Poisson("rate", lambda_, observed=data)

all_shuffled_rates = dict()
n_shuffles = 10
for group_ind, this_group in tqdm(significant_snippets_grouped):
    group_name_str = "_".join([str(x) for x in group_ind])
    warped_array = all_warped_arrays[group_name_str]

    shuffle_list = []
    for i in range(n_shuffles):

        shuffled_array = np.copy(warped_array)
        for i in range(shuffled_array.shape[0]):
            np.random.shuffle(shuffled_array[i])

        # fig, ax = plt.subplots(1,2,figsize=(4, 2), sharey=True)
        # ax[0].imshow(warped_array, aspect='auto', cmap='Greys', origin='lower')
        # ax[0].set_title(f'{group_name_str} Warped Spike Raster')
        # ax[0].set_ylabel('Trial Index')
        # ax[1].imshow(shuffled_array, aspect='auto', cmap='Greys', origin='lower')
        # ax[1].set_title(f'{group_name_str} Shuffled Warped Spike Raster')
        # ax[1].set_xlabel('Warped Time Bins')
        # plt.show()

        with shuffled_model:
            pm.set_data({"data": shuffled_array})
            fit = pm.fit(n=50000, method='advi', progressbar=True)
            trace = fit.sample(1000)

        ppc_list = pm.sample_posterior_predictive(trace, model = shuffled_model, var_names = ['lambda_'])
        mean_ppc = ppc_list.posterior_predictive.lambda_.mean(axis=(0,1)).values
        shuffle_list.append(mean_ppc)
    all_shuffled_rates[group_name_str] = shuffle_list
    # Dump to have checkpoint in case of long runtime
    shuffle_artifact_path = os.path.join(this_artifact_dir, f'{group_name_str}_shuffled_rates.pkl')
    with open(shuffle_artifact_path, 'wb') as f:
        dump(shuffle_list, f)


# Calculate bits-per-spike for each grand_mean_rate
all_grand_mean_rates = dict()
for group_ind, this_group in tqdm(significant_snippets_grouped):
    group_name_str = "_".join([str(x) for x in group_ind])
    artifact_path = os.path.join(this_artifact_dir, f'{group_name_str}_model_trace.pkl')
    with open(artifact_path, 'rb') as f:
        out_dict = load(f)
    trace = out_dict['trace']
    ppc_list = pm.sample_posterior_predictive(trace, model = out_dict['model'], var_names = ['lambda_'])
    mean_ppc = ppc_list.posterior_predictive.lambda_.mean(axis=(0,1)).values
    grand_mean_rate = mean_ppc.mean(axis=0)
    all_grand_mean_rates[group_name_str] = grand_mean_rate

def calc_flat_deviation(ts, n_shuffles = 10_000):
    """
    Calculates deviation of time-seroes from unformity

    Args:
        ts (np.ndarray): Time-series data
    Returns:
        stat (float): Deviation statistic (e.g., standard deviation, entropy, etc.)
        p_value (float): P-value from statistical test comparing to null distribution
    """
    mean_val = np.mean(ts)
    dev = np.cumsum(ts - mean_val)
    # Generate null distribution by shuffling the time-series
    shuffles = np.array([np.random.permutation(ts) for _ in range(n_shuffles)])
    shuffle_devs = np.array([np.cumsum(shuffle - mean_val) for shuffle in shuffles])

    stat = np.abs(dev).max()
    shuffle_stats = np.abs(shuffle_devs).max(axis=1)

    # fig, ax = plt.subplots(3,1,sharex=True)
    # ax[0].plot(ts, color='blue', label='Original')
    # ax[0].axhline(mean_val, color='red', linestyle='--', label='Mean Value')
    # ax[1].plot(dev, color='blue', label='Original')
    # ax[2].imshow(shuffle_devs, aspect='auto', cmap='Reds', alpha=0.5)
    # plt.show()

    p_value = 1 - percentileofscore(shuffle_stats, stat) / 100.0  # Convert to proportion
    return stat, p_value


# all_bits_per_spike = dict()
all_deviation_stats = dict()
for group_ind, this_group in tqdm(significant_snippets_grouped):
    group_name_str = "_".join([str(x) for x in group_ind])
    grand_mean_rate = all_grand_mean_rates[group_name_str]
    warped_array = all_warped_arrays[group_name_str]

    dev_stat, p_value = calc_flat_deviation(grand_mean_rate)
    all_deviation_stats[group_name_str] = {'dev_stat': dev_stat, 'p_value': p_value}

plt.hist([stat['p_value'] for stat in all_deviation_stats.values()], edgecolor='black')
plt.show()

# Plot top and bottom n rates
dev_stats_df = pd.DataFrame(all_deviation_stats).T.reset_index().rename(columns={'index': 'group_name'})
# Add mean rate info to dev_stats_df
dev_stats_df['mean_rate'] = dev_stats_df['group_name'].map(lambda x: all_grand_mean_rates[x])

n = 10
top_deviation = dev_stats_df.sort_values(by='p_value', ascending=True).head(n).reset_index(drop=True)
bottom_deviation = dev_stats_df.sort_values(by='p_value', ascending=False).head(n).reset_index(drop=True)

fig, ax = plt.subplots(n, 2, figsize=(8, 4*n), sharex=True)
for i, row in top_deviation.iterrows():
    group_name_str = row['group_name']
    rate = row['mean_rate']
    ax[i, 0].plot(rate, color='blue')
    ax[i, 0].set_ylabel(f'p={row["p_value"]:.4f}')
for i, row in bottom_deviation.iterrows():
    group_name_str = row['group_name']
    rate = row['mean_rate']
    ax[i, 1].plot(rate, color='orange')
    ax[i, 1].set_ylabel(f'p={row["p_value"]:.4f}')
ax[0, 0].set_title('Top Deviation from Flatness')
ax[0, 1].set_title('Bottom Deviation from Flatness')
fig.suptitle('Grand Mean Firing Rates for Groups with Highest and Lowest Deviation from Flatness', fontsize=16)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'deviation_from_flatness_top_bottom_rates.svg'), bbox_inches='tight')
plt.close(fig)
# plt.show()


    # fig, ax = plt.subplots(2,1,figsize=(4, 4), sharex=True)
    # ax[0].imshow(warped_array, aspect='auto', cmap='Greys', origin='lower')
    # ax[0].set_title(f'{group_name_str} Warped Spike Raster')
    # ax[0].set_ylabel('Trial Index')
    # ax[1].plot(grand_mean_rate, color='black', linewidth=2)
    # ax[1].set_title(f'{group_name_str} Inferred Firing Rate')
    # ax[1].set_xlabel('Warped Time Bins')
    # plt.tight_layout()
    # plt.show()
    
    # # Repeat grand_mean_rate along axis 0 to match warped_array shape
    # grand_mean_rate_repeated = np.tile(grand_mean_rate, (warped_array.shape[0], 1))
    #
    # bits_per_spike = utils.calc_bits_per_spike(warped_array, grand_mean_rate_repeated)
    # all_bits_per_spike[group_name_str] = bits_per_spike

# plt.hist(list(all_bits_per_spike.values()), edgecolor='black')
# plt.show()

##############################
# For each neuron, plot both warped and unwarped firing rates for all states for a single taste
# Plot traces of warped firing rates for significant neurons
n_plots = 50
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
bin_size = 50  # in ms
kernel_width = 250  # in ms
for (basename, neuron_ind, taste_num), group in grouped_snippets:
    states = sorted(group.state_ind.unique())
    n_states = len(states)
    
    # Create subplot grid: 2 rows (unwarped, warped) x n_states columns
    fig, axs = plt.subplots(2, n_states, figsize=(4*n_states, 6), sharey='col') 
    
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
            # # Bin the spike data
            # n_bins = int(np.ceil(len(arr) / bin_size))
            # binned = np.array([np.mean(arr[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
            # binned_spike_data_list.append(binned)
            # Smooth with boxcar kernel
            if len(arr) < kernel_width:
                smoothed = arr
            else:
                kernel = np.ones(kernel_width) / kernel_width
                smoothed = np.convolve(arr, kernel, mode='valid')
            binned_spike_data_list.append(smoothed)

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

        # Set y_lim to a robust range across both plots
        all_rates = np.concatenate(smoothed_spike_data_list + [mean_warped_rate])
        perc_lims = [5, 95]
        y_min, y_max = np.percentile(all_rates, perc_lims)
        axs[0, col_idx].set_ylim(y_min, y_max)

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
