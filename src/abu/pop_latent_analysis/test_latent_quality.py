"""
# Quality of latents
- Check for:
    - stability across trials
    - bits/spike
    - bits/spike on trials-matched data > bits/spike on trial-shuffled data
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

base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

rel_data_path = 'output/intermediate_data/RNN_PROCESSING_PARQUETS/latent_outputs/raw_output_unwarped'
abs_data_path = f"{base_dir}/{rel_data_path}"

artifacts_dir = f"{base_dir}/output/artifacts/population_analysis"
array_artifacts_dir = f"{artifacts_dir}/latent_arrays"
dfs_artifacts_dir = f"{artifacts_dir}/latent_dfs"

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

