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

base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

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
plt.title('Binned Spike Count vs Firing Rate (sample of points)')
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
