"""
# Quality of latents
- Check for:
    - stability across trials
    - bits/spike
    - bits/spike on trials-matched data > bits/spike on trial-shuffled data
    - Comparison of timescales / frequencies bewteen kernel smoothing and latents
"""

import os
import sys
from pprint import pprint as pp
from matplotlib import pyplot as plt
import numpy as np
from itertools import product
import pandas as pd
import pingouin as pg
from tqdm import tqdm
from glob import glob
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA

base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

pop_analysis_src_dir = os.path.join(src_dir, 'abu', 'model_fitting','population_analysis')
sys.path.append(pop_analysis_src_dir)
import utils

rel_data_path = 'output/intermediate_data/RNN_PROCESSING_PARQUETS/latent_outputs/raw_output_unwarped'
abs_data_path = f"{base_dir}/{rel_data_path}"

artifacts_dir = f"{base_dir}/output/artifacts/population_analysis"
array_artifacts_dir = f"{artifacts_dir}/latent_arrays"
dfs_artifacts_dir = f"{artifacts_dir}/latent_dfs"
fr_artifacts_dir = f"{artifacts_dir}/firing_rate_arrays"

plot_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/plots'
pop_analysis_plot_dir = os.path.join(plot_dir, 'population_analysis')

##############################
# Get spike-trains

spike_train_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/spike_trains_npz'
spike_train_files = glob(f'{spike_train_dir}/*.npz')
basename_list = [os.path.basename(f).replace('_repacked.npz', '') for f in spike_train_files]

# From both basename_list and tau_frame['basename'], drop any "_repacked" or "_copy" suffixes
basename_list = [name.replace('_repacked', '').replace('_copy', '') for name in basename_list]

# Load all spike train files into a dict
spike_train_dict = {this_name: np.load(this_file) for this_name, this_file in zip(basename_list, spike_train_files)}

# All files have only 1 array but with different key names
# Each values array has shape: (taste, trial, neuron, time)
spike_train_dict = {k: v[list(v.files)[0]] for k, v in spike_train_dict.items()}

# Cut and bin spike_trains to same params as latents
bin_size = 25 # ms
ind_lims = [1500, 4500] 

binned_spike_trains = [] 
for session_name, spike_train in spike_train_dict.items():
    # Cut to same time window as latents
    spike_train = spike_train[:, :, :, ind_lims[0]:ind_lims[1]]
    # Bin by summing over bins of size bin_size
    n_bins = (ind_lims[1] - ind_lims[0]) // bin_size
    binned_spike_train = spike_train.reshape(
            spike_train.shape[0], 
            spike_train.shape[1], 
            spike_train.shape[2], 
            n_bins, 
            bin_size
            ).sum(axis=-1)
    # binned_spike_trains[session_name] = binned_spike_train
    for taste in range(binned_spike_train.shape[0]):
        out_dict = {
                'session_name': session_name,
                'taste': taste,
                'binned_spike_train': binned_spike_train[taste]
                }
        binned_spike_trains.append(out_dict)

binned_spike_trains_df = pd.DataFrame(binned_spike_trains)



##############################
##############################
# Load dfs to perform anova for consistency across trials
latent_df_list = os.listdir(dfs_artifacts_dir)
# Drop any non-csv files
latent_df_list = [f for f in latent_df_list if f.endswith('.csv')]
session_names = ["_".join(f.split('_')[:4]) for f in latent_df_list]

latent_dfs = [
    pd.read_csv(os.path.join(dfs_artifacts_dir, f)) for f in latent_df_list
]

# columns for each df: taste, trial, latent_dim, time, latent_value
# Bin time into bins and average
bin_size = 500 # ms
binned_latent_dfs = []
for df in latent_dfs:
    df['time_bin'] = (df['time'] // bin_size) * bin_size
    binned_df = df.groupby(['taste', 'trial', 'latent_dim', 'time_bin'])['latent_value'].mean().reset_index()
    binned_latent_dfs.append(binned_df)

# For each taste per session, perform 2-way anova across trials and time bins for each latent dimension to check for consistency across trials
anova_results = []
for session_name, df in tqdm(zip(session_names, binned_latent_dfs)): 
    for taste in df['taste'].unique():
        taste_df = df[df['taste'] == taste]
        for latent_dim in taste_df['latent_dim'].unique():
            latent_dim_df = taste_df[taste_df['latent_dim'] == latent_dim]
            aov = pg.anova(
                    dv='latent_value', 
                    between=['trial', 'time_bin'], 
                    data=latent_dim_df, 
                    detailed=True
                    )
            aov['taste'] = taste
            aov['latent_dim'] = latent_dim
            anova_results.append(aov)

all_aov_df = pd.concat(anova_results, ignore_index=True).dropna()


##############################

latent_file_list = os.listdir(array_artifacts_dir)
# Drop any non-npy files
latent_file_list = [f for f in latent_file_list if f.endswith('.npy')]

# Load all latent arrays
latent_arrays = [
    np.load(os.path.join(array_artifacts_dir, f), allow_pickle=True) for f in latent_file_list
]

# Shape of each array: (taste, trial, latents, time)
# Break down by taste
inds = []
taste_latent_arrays = []
for session_name, latent_array in zip(session_names, latent_arrays):
    for taste in range(len(latent_array)): 
        inds.append((session_name, taste))
        taste_latent_arrays.append(latent_array[taste])

# Convert to pandas array for easier handling
data_list = [
        (session_name, taste, latent_array) for \
                (session_name, taste), latent_array in zip(inds, taste_latent_arrays)
        ]
latent_df = pd.DataFrame(
        columns=['session', 'taste', 'latent_array'],
        data=data_list
        )

# Write out to artifacts as pkl
latent_df.to_pickle(os.path.join(artifacts_dir, 'all_latent_df.pkl'))

###########################################################
# Load RNN-inferred firing rates to calculate bits/spike
###########################################################
fr_file_list = os.listdir(fr_artifacts_dir)
# Drop any non-npy files
fr_file_list = [f for f in fr_file_list if f.endswith('.npy')]
session_names = ["_".join(f.split('_')[:4]) for f in fr_file_list]
# Load all firing rate arrays
fr_arrays = [
    np.load(os.path.join(fr_artifacts_dir, f), allow_pickle=True) for f in fr_file_list
]

# Shape of each array: (taste, trial, neuron, time)
# Convert to pandas array for easier handling
data_list = []
for session_name, fr_array in zip(session_names, fr_arrays):
    for taste in range(len(fr_array)): 
        data_list.append((session_name, taste, fr_array[taste]))
fr_df = pd.DataFrame(
        columns=['session', 'taste', 'fr_array'],
        data=data_list
        )

# Merge with binned_spike_trains_df to get both fr_array and binned_spike_train in same df
spike_fr_df = pd.merge(
        fr_df, 
        binned_spike_trains_df, 
        left_on=['session', 'taste'],
        right_on=['session_name', 'taste']
        )
# Drop redundant session_name column
spike_fr_df = spike_fr_df.drop(columns=['session_name'])

# Make plots of spike-trains and firing rates for sanity check
spike_fr_plot_dir = os.path.join(pop_analysis_plot_dir, 'spike_fr_plots')
os.makedirs(spike_fr_plot_dir, exist_ok=True)

# For each taste, make plots with 2 columns (spike train and firing rate) and rows = neurons
for row_ind, this_row in tqdm(spike_fr_df.iterrows(), total=spike_fr_df.shape[0]):
    session_name = this_row['session']
    taste = this_row['taste']
    binned_spike_train = this_row['binned_spike_train']
    fr_array = this_row['fr_array']

    fig, axes = plt.subplots(2, fr_array.shape[1], figsize=(fr_array.shape[1]*2, 5))
    for neuron in range(fr_array.shape[1]):
        axes[0, neuron].imshow(binned_spike_train[:, neuron], interpolation='nearest', aspect='auto')
        axes[0, neuron].set_title(f'Neuron {neuron}')
        axes[1, neuron].imshow(fr_array[:, neuron], interpolation='nearest', aspect='auto')
        axes[0,0].set_ylabel('Binned Spike Train')
        axes[1,0].set_ylabel('Firing Rate')
    plt.suptitle(f'Session: {session_name}, Taste: {taste}')
    plt.tight_layout()
    plt.savefig(os.path.join(spike_fr_plot_dir, f'{session_name}_taste{taste}_spike_fr.png'), bbox_inches='tight')
    plt.close()

# Check that there are no scaling issues
# Plot binned spikes against firing rates
all_binned_spikes = np.concatenate(spike_fr_df['binned_spike_train'].values, axis=1)[..., :-1] # drop last time bin to match fr_array shape
all_frs = np.concatenate(spike_fr_df['fr_array'].values, axis=1)

flat_binned_spikes = all_binned_spikes.flatten()
flat_frs = all_frs.flatten()

# Get slope of binned spikes vs firing rates to check for scaling issues
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(flat_binned_spikes.reshape(-1, 1), flat_frs)
# Print slop and intercept
print(f"Slope: {model.coef_[0]}, Intercept: {model.intercept_}")

# Perform correlation between binned spikes and firing rates to check for significant correlation
from scipy.stats import pearsonr
corr, p_value = pearsonr(flat_binned_spikes, flat_frs)


# Plot binned spikes against firing rates for a random sample of neurons
n_points = 10_000
plot_inds = np.random.choice(len(flat_binned_spikes), size=n_points, replace=False)
plot_spike_counts = flat_binned_spikes[plot_inds]
plot_frs = flat_frs[plot_inds]
# Correct using slope
plot_frs_corrected = plot_frs / model.coef_[0]
# Jitter everything by 0.1sd noise
plot_spike_counts += np.random.normal(0, plot_spike_counts.std()*0.3, size=plot_spike_counts.shape)
plot_frs_corrected += np.random.normal(0, plot_frs_corrected.std()*0.3, size=plot_frs_corrected.shape)

plt.figure(figsize=(5,5))
plt.scatter(plot_spike_counts, plot_frs_corrected, alpha=0.1)
# Plot x=y line for reference
max_val = max(plot_spike_counts.max(), plot_frs_corrected.max())
plt.plot([0, max_val], [0, max_val], 'r--')
plt.xlabel('Binned Spike Count')
plt.ylabel('Firing Rate')
plt.title(f'Binned Spike Count vs Firing Rate (sample of points)\nPearson r: {corr:.2f}, p-value: {p_value:.2e}')
plt.savefig(os.path.join(spike_fr_plot_dir, f'binned_spikes_vs_firing_rate.png'), bbox_inches='tight')
plt.close()

# Plot KDE of 2D distribution of binned spikes vs firing rates
# sns.kdeplot(plot_spike_counts, plot_frs_corrected, fill=True, thresh=0.05)
# 2D histogram with log color scale

plt.figure(figsize=(5,5))
plt.hist2d(plot_spike_counts, plot_frs_corrected, bins=10, norm=LogNorm())
# PLlot x=y line for reference
max_val = max(plot_spike_counts.max(), plot_frs_corrected.max())
plt.plot([0, max_val], [0, max_val], 'r--')
plt.xlabel('Binned Spike Count')
plt.ylabel('Firing Rate')
plt.title('KDE of Binned Spike Count vs Firing Rate')
plt.savefig(os.path.join(spike_fr_plot_dir, f'binned_spikes_vs_firing_rate_kde.png'), bbox_inches='tight')
plt.close()


# utils.calc_bits_per_spike(spike_train, rate):
# """
# Calculate bits per spike for a given spike train and firing rate.
# Args:
#     spike_train: numpy array of shape (trials, time_bins) with binary values indicating spikes
#     rate: firing rate in Hz (spikes per second)
# Returns:
#     bits per spike
#
# Refs:
#     - https://neuronaldynamics.epfl.ch/online/Ch10.S3.html
#     - https://www.biorxiv.org/content/10.1101/2025.02.07.637062v2.full
# """

# For each neuron and taste, calculate bits per spike using utils.calc_bits_per_spike and compare to trial-shuffled data
# Correct for scaling issues by dividing firing rates by slope from linear regression
bits_per_spike_results = []
for row_ind, this_row in tqdm(spike_fr_df.iterrows(), total=spike_fr_df.shape[0]):
    session_name = this_row['session']
    taste = this_row['taste']
    binned_spike_train = this_row['binned_spike_train'][..., :-1] # drop last time bin to match fr_array shape
    fr_array = this_row['fr_array'] / model.coef_[0] # correct for scaling issues
    # Because network did not have strictly positive output, clip output at 0 to avoid issues with bits/spike calculation 
    fr_array = np.clip(fr_array, a_min=0, a_max=None)
    mean_fr = fr_array.mean(axis=None)

    for neuron in range(fr_array.shape[1]):
        spike_train = binned_spike_train[:, neuron]
        rate = fr_array[:, neuron]
        bits_per_spike = utils.calc_bits_per_spike(spike_train, rate)
        sh_bits_per_spike = utils.calc_bits_per_spike(spike_train, np.random.permutation(rate))
        
        # plt.plot(spike_train.T, label='Binned Spike Train', color='blue', alpha=0.5)
        # plt.plot(rate.T, label='Firing Rate', color='orange', alpha=0.5)
        # plt.show()

        bits_per_spike_results.append({
                'session': session_name,
                'taste': taste,
                'neuron': neuron,
                'bits_per_spike': bits_per_spike,
                'sh_bits_per_spike': sh_bits_per_spike,
                'mean_fr': mean_fr,
                'fr_array': rate,
                'spike_train': spike_train
                })

bps_df = pd.DataFrame(bits_per_spike_results)
bps_df['bits_per_spike_diff'] = bps_df['bits_per_spike'] - bps_df['sh_bits_per_spike']

# Sort by session, taste, neuron
bps_df = bps_df.sort_values(by=['session', 'taste', 'neuron']).reset_index(drop=True)

# Write out to artifacts as pkl
bps_df.to_pickle(os.path.join(artifacts_dir, 'bits_per_spike_df.pkl'))

# Plot bits per spike vs shuffled bits per spike
# Both as scatter and difference histogram
fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.scatterplot(x='bits_per_spike', y='sh_bits_per_spike', data=bps_df, ax=ax[0])
ax[0].plot([bps_df['bits_per_spike'].min(), bps_df['bits_per_spike'].max()], [bps_df['bits_per_spike'].min(), bps_df['bits_per_spike'].max()], 'r--')
ax[0].set_xlabel('Bits per Spike')
ax[0].set_ylabel('Shuffled Bits per Spike')
ax[0].set_title('Bits per Spike vs Shuffled Bits per Spike')
sns.histplot(bps_df['bits_per_spike'] - bps_df['sh_bits_per_spike'], bins=30, ax=ax[1])
# set log scale for y axis
ax[1].set_yscale('log')
# Plot vertical line at 0 for reference
ax[1].axvline(0, color='r', linestyle='--')
# Plot median line for reference and annotate with median value
median_diff = (bps_df['bits_per_spike'] - bps_df['sh_bits_per_spike']).median()
mean_diff = (bps_df['bits_per_spike'] - bps_df['sh_bits_per_spike']).mean()
ax[1].axvline(median_diff, color='g', linestyle='--')
ax[1].annotate(f'Median: {median_diff:.2f}', xy=(median_diff, ax[1].get_ylim()[1]*0.8), xytext=(median_diff+0.5, ax[1].get_ylim()[1]*0.8), arrowprops=dict(arrowstyle='->', color='green'), color='green')
ax[1].axvline(mean_diff, color='b', linestyle='--')
ax[1].annotate(f'Mean: {mean_diff:.2f}', xy=(mean_diff, ax[1].get_ylim()[1]*0.6), xytext=(mean_diff+0.5, ax[1].get_ylim()[1]*0.6), arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
ax[1].set_xlabel('Bits per Spike - Shuffled Bits per Spike')
ax[1].set_title('Distribution of Bits per Spike - Shuffled Bits per Spike')
plt.tight_layout()
plt.savefig(os.path.join(pop_analysis_plot_dir, f'bits_per_spike_vs_shuffled.png'), bbox_inches='tight')
plt.close()

# Look for any relationship bewteen:
# 1) firing rate and bits per spike
# 2) firing rate and bits per spike - shuffled bits per spike
fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.scatterplot(x='mean_fr', y='bits_per_spike', data=bps_df, ax=ax[0])
ax[0].set_xlabel('Mean Firing Rate')
ax[0].set_ylabel('Bits per Spike')
ax[0].set_title('Bits per Spike vs Mean Firing Rate')
sns.scatterplot(x='mean_fr', y='bits_per_spike_diff', data=bps_df, ax=ax[1])
ax[1].set_xlabel('Mean Firing Rate')
ax[1].set_ylabel('Bits per Spike - Shuffled Bits per Spike')
ax[1].set_title('Bits per Spike - Shuffled Bits per Spike vs Mean Firing Rate')
plt.tight_layout()
plt.savefig(os.path.join(pop_analysis_plot_dir, f'bits_per_spike_vs_firing_rate.png'), bbox_inches='tight')
plt.close()

# Check whether bps-diff is session and taste dependent
g = sns.catplot(
        x='session', y='bits_per_spike_diff', hue='taste', 
        data=bps_df, 
        kind='bar', 
        height=5, aspect=1.5,
        )
g.set_axis_labels('Taste', 'Bits per Spike - Shuffled Bits per Spike')
g.set_xticklabels(rotation=45, ha='right')
g.set_titles('Bits per Spike - Shuffled Bits per Spike by Taste and Session')
plt.savefig(os.path.join(pop_analysis_plot_dir, f'bits_per_spike_diff_by_taste_session.png'), bbox_inches='tight')
plt.close()

# Plot n units for highest and lowest diff bps
plot_n = 5
top_bps = bps_df.sort_values(by='bits_per_spike_diff', ascending=False).head(plot_n)
bottom_bps = bps_df.sort_values(by='bits_per_spike_diff', ascending=True).head(plot_n)

fig, ax = plt.subplots(plot_n, 2, figsize=(10, plot_n*3))
for i in range(plot_n):
    top_row = top_bps.iloc[i]
    bottom_row = bottom_bps.iloc[i]

    ax[i, 0].plot(top_row['spike_train'].T, color='blue', alpha=0.5)
    ax[i, 0].plot(top_row['fr_array'].T , color='orange', alpha=0.5)
    ax[i, 0].set_title(f"Top {i+1} - Session: {top_row['session']}, Taste: {top_row['taste']}, Neuron: {top_row['neuron']}\nBits per Spike Diff: {top_row['bits_per_spike_diff']:.2f}")
    ax[i, 0].legend()

    ax[i, 1].plot(bottom_row['spike_train'], color='blue', alpha=0.5)
    ax[i, 1].plot(bottom_row['fr_array'], color='orange', alpha=0.5)
    ax[i, 1].set_title(f"Bottom {i+1} - Session: {bottom_row['session']}, Taste: {bottom_row['taste']}, Neuron: {bottom_row['neuron']}\nBits per Spike Diff: {bottom_row['bits_per_spike_diff']:.2f}")
    ax[i, 1].legend()
plt.tight_layout()
plt.savefig(os.path.join(pop_analysis_plot_dir, f'top_bottom_bits_per_spike_diff.png'), bbox_inches='tight')
plt.close()

# Do the same, but with averaged firing rates and spike counts across time bins to see if relationship is clearer
fig, ax = plt.subplots(plot_n, 2, figsize=(10, plot_n*3))
for i in range(plot_n):
    top_row = top_bps.iloc[i]
    bottom_row = bottom_bps.iloc[i]

    ax[i, 0].plot(top_row['spike_train'].mean(axis=0), color='blue', alpha=0.5)
    ax[i, 0].plot(top_row['fr_array'].mean(axis=0) , color='orange', alpha=0.5)
    ax[i, 0].set_title(f"Top {i+1} - Session: {top_row['session']}, Taste: {top_row['taste']}, Neuron: {top_row['neuron']}\nBits per Spike Diff: {top_row['bits_per_spike_diff']:.2f}")
    ax[i, 0].legend()

    ax[i, 1].plot(bottom_row['spike_train'].mean(axis=0), color='blue', alpha=0.5)
    ax[i, 1].plot(bottom_row['fr_array'].mean(axis=0), color='orange', alpha=0.5)
    ax[i, 1].set_title(f"Bottom {i+1} - Session: {bottom_row['session']}, Taste: {bottom_row['taste']}, Neuron: {bottom_row['neuron']}\nBits per Spike Diff: {bottom_row['bits_per_spike_diff']:.2f}")
    ax[i, 1].legend()
plt.tight_layout()
plt.savefig(os.path.join(pop_analysis_plot_dir, f'top_bottom_bits_per_spike_diff_avg.png'), bbox_inches='tight')
plt.close()

############################################################

# Find which sessions have significantly higher bits per spike than shuffled for all tastes
session_bps_summary = bps_df.groupby(['session','taste'])['bits_per_spike_diff'].agg(['mean', 'median', 'std', 'count'])
session_bps_summary.reset_index(inplace=True)
# For each session and taste, perform one-sample t-test to see if bits_per_spike_diff is significantly greater than 0
p_value_list = []
for session, taste in session_bps_summary[['session', 'taste']].values:
    subset = bps_df[(bps_df['session'] == session) & (bps_df['taste'] == taste)]
    t_stat, p_value = ttest_1samp(subset['bits_per_spike_diff'], popmean=0, alternative='greater')
    p_value_list.append(p_value)

alpha = 0.05
session_bps_summary['significant'] = session_bps_summary['p_value'] < alpha

session_bps_summary.groupby('session')['significant'].mean()

# Write out session_bps_summary to artifacts as csv
session_bps_summary.to_csv(os.path.join(artifacts_dir, 'session_bps_summary.csv'), index=False)

# Get only signifincant sessions/tastes
sig_fits_df = session_bps_summary[session_bps_summary['significant']]

# Extract significant latents
latent_df = pd.merge(
        sig_fits_df[['session', 'taste']],
        latent_df,
        left_on=['session', 'taste'],
        right_on=['session', 'taste'],
        how='inner'
        )

# Plot all significant latents for each session and taste
sig_latent_plot_dir = os.path.join(pop_analysis_plot_dir, 'significant_latents')
os.makedirs(sig_latent_plot_dir, exist_ok=True)

for row_ind, this_row in tqdm(latent_df.iterrows(), total=latent_df.shape[0]):
    session_name = this_row['session']
    taste = this_row['taste']
    latent_array = this_row['latent_array']

    n_plots = latent_array.shape[1]
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    for latent_dim in range(latent_array.shape[1]):
        row = latent_dim // n_cols
        col = latent_dim % n_cols
        axes[row, col].imshow(latent_array[:, latent_dim, :], interpolation='nearest', aspect='auto')
        axes[row, col].set_title(f'Latent Dim {latent_dim}')
    plt.suptitle(f'Session: {session_name}, Taste: {taste}')
    plt.tight_layout()
    plt.savefig(os.path.join(sig_latent_plot_dir, f'{session_name}_taste{taste}_significant_latents.png'), bbox_inches='tight')
    plt.close()

# Also apply PCA to determine useful # of latents
all_exp_var_ratios = []
for row_ind, this_row in tqdm(latent_df.iterrows(), total=latent_df.shape[0]):
    session_name = this_row['session']
    taste = this_row['taste']
    latent_array = this_row['latent_array']

    # Reshape to (trials*time, latent_dim)
    n_trials, n_latent_dims, n_time_bins = latent_array.shape
    reshaped_latent_array = latent_array.transpose(0, 2, 1).reshape(-1, n_latent_dims)

    pca = PCA()
    pca.fit(reshaped_latent_array)
    explained_variance_ratio = pca.explained_variance_ratio_
    all_exp_var_ratios.append({
            'session': session_name,
            'taste': taste,
            'explained_variance_ratio': explained_variance_ratio
            })

# Plot explained variance ratio for each session and taste
exp_var_df = pd.DataFrame(all_exp_var_ratios)

plt.figure(figsize=(5,5))
cumsum_exp_var_ratios = []
for row_ind, this_row in tqdm(exp_var_df.iterrows(), total=exp_var_df.shape[0]):
    session_name = this_row['session']
    taste = this_row['taste']
    explained_variance_ratio = this_row['explained_variance_ratio']
    cumsum_exp_var = np.cumsum(explained_variance_ratio)
    cumsum_exp_var_ratios.append(cumsum_exp_var)
    plt.plot(
            np.arange(1, len(explained_variance_ratio)+1),
            cumsum_exp_var, marker='o', color = 'k', alpha = 0.3
            )
# Also plot mean explained variance ratio across sessions
mean_cumsum_exp_var = np.mean(cumsum_exp_var_ratios, axis=0)
plt.plot(
        np.arange(1, len(mean_cumsum_exp_var)+1),
        mean_cumsum_exp_var, marker='o', color='r', label='Mean across sessions',
        linewidth=2
        )
# Annotate with mean exp_var for each component and dashed line at 90% variance explained
for i, exp_var in enumerate(mean_cumsum_exp_var):
    plt.annotate(f'{exp_var:.2f}', xy=(i+1, 0.1), xytext=(i+1, 0.1), ha='center')
plt.axhline(0.9, color='gray', linestyle='--', label='90% Variance Explained')
plt.ylim(0, 1)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio for Significant Latents')
plt.grid()
plt.savefig(os.path.join(sig_latent_plot_dir, f'{session_name}_taste{taste}_pca_explained_variance.png'), bbox_inches='tight')
plt.close()

##############################
# Compress latents down to 4 components and write to artifacts
compressed_latent_arrays = []
for row_ind, this_row in tqdm(latent_df.iterrows(), total=latent_df.shape[0]):
    session_name = this_row['session']
    taste = this_row['taste']
    latent_array = this_row['latent_array']

    # Reshape to (trials*time, latent_dim)
    n_trials, n_latent_dims, n_time_bins = latent_array.shape
    reshaped_latent_array = latent_array.transpose(0, 2, 1).reshape(-1, n_latent_dims)

    pca = PCA(n_components=4)
    compressed_latent_array = pca.fit_transform(reshaped_latent_array)
    compressed_latent_array = compressed_latent_array.reshape(n_trials, n_time_bins, 4).transpose(0, 2, 1)
    compressed_latent_arrays.append(compressed_latent_array)

latent_df['pca_latents'] = compressed_latent_arrays

# Write out to artifacts as pkl
latent_df.to_pickle(os.path.join(artifacts_dir, 'all_latent_df.pkl'))

# Plot pca latents for each session and taste
for row_ind, this_row in tqdm(latent_df.iterrows(), total=latent_df.shape[0]):
    session_name = this_row['session']
    taste = this_row['taste']
    latent_array = this_row['pca_latents']

    n_plots = latent_array.shape[1]
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    for latent_dim in range(latent_array.shape[1]):
        row = latent_dim // n_cols
        col = latent_dim % n_cols
        axes[row, col].imshow(latent_array[:, latent_dim, :], interpolation='nearest', aspect='auto')
        axes[row, col].set_title(f'PCA Latent Dim {latent_dim}')
    plt.suptitle(f'Session: {session_name}, Taste: {taste} - PCA Compressed Latents')
    plt.tight_layout()
    plt.savefig(os.path.join(sig_latent_plot_dir, f'{session_name}_taste{taste}_pca_compressed_latents.png'), bbox_inches='tight')
    plt.close()
