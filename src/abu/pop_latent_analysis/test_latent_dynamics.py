"""
# Non-stationarity of latents
- Check for:
    - Alignment with changepoints
    - Eigenspectrum of LDS fit to latents
    - High correlation with binned spike-counts
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

############################################################
# latent_df.to_pickle(os.path.join(artifacts_dir, 'all_latent_df.pkl'))
latent_df = pd.read_pickle(os.path.join(artifacts_dir, 'all_latent_df.pkl'))

# Latent params 
latent_bin_size = 25 # ms
latent_ind_lims = [1500, 4500] 
stim_t = 2000
t_correction = -1500 # to align with latent time axis (0-3000ms), since start_time is 2000ms, and we want to align with stimulus onset 

# Load changepoint data
chp_pkl_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/pkl_cache'
file_list = glob(os.path.join(chp_pkl_dir, '*_changepoints.pkl'))
chp_dfs = []
for file in file_list:
    chp_array = pd.read_pickle(file)
    # start_time is 2000 ms
    # Correct for 2000ms offset, align with latent time axis, and bin accordingly
    # chp_dfs.append(df)
    chp_array = (chp_array - t_correction) // latent_bin_size 
    basename = os.path.basename(file)
    session_name = basename.split('_repacked')[0]
    for taste_id, chp_array in enumerate(chp_array):
        chp_dfs.append({
            'session': session_name,
            'taste': taste_id,
            'changepoints': chp_array
        })

# Merge with latent_df
chp_df = pd.DataFrame(chp_dfs)
latent_df = latent_df.merge(chp_df, on=['session', 'taste'], how='left')
# Index(['session', 'taste', 'latent_array', 'pca_latents', 'changepoints'], dtype='object')

# Plot PCA latents overlayed with changepoints
sig_latent_plot_dir = os.path.join(pop_analysis_plot_dir, 'significant_latents')
for row_ind, this_row in latent_df.iterrows():
    session = this_row['session']
    taste = this_row['taste']
    changepoints = this_row['changepoints'] # shape: (trials, num_changepoints)
    pca_latents = this_row['pca_latents'] # shape: (trials, latents, time_bins)

    n_plots = pca_latents.shape[1]
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2*n_plots), sharex=True)
    for latent_ind in range(n_plots):
        ax = axes[latent_ind]
        ax.imshow(pca_latents[:, latent_ind, :], aspect='auto', origin='lower', interpolation='none')
        for trial_ind, trial_chps in enumerate(changepoints):
            ax.scatter(trial_chps, [trial_ind]*len(trial_chps), color='red', s=10)
        ax.set_title(f'Session: {session}, Taste: {taste}, Latent {latent_ind}')
    plt.xlabel('Time bins')
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(sig_latent_plot_dir, f'{session}_taste{taste}_pca_latents.png'))
