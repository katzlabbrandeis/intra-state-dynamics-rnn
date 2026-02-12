"""
Analysis and plots for single neuron analysis of Calia-Bogan 2025
1- Paired test on first vs last half of states for single-neurons
2- Plot of warped single-neuron firing rates
"""

import blech_clust as bc
from blech_clust.utils.ephys_data import visualize as vz
from blech_clust.utils.ephys_data.ephys_data import ephys_data

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
import xarray as xr
from pprint import pprint as pp

tqdm.pandas()

##############################
base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
artifacts_sup_dir = os.path.join(base_dir, 'output', 'artifacts')
artifacts_dir = os.path.join(artifacts_sup_dir, 'raw_spikes')
plot_dir = os.path.join(base_dir, 'output', 'plots', 'raw_spikes')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)
##############################
# Get spike-trains

spike_train_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/spike_trains_npz'
spike_train_files = glob(f'{spike_train_dir}/*.npz')
basename_list = [os.path.basename(f).replace('_repacked.npz', '') for f in spike_train_files]

# From both basename_list and tau_frame['basename'], drop any "_repacked" or "_copy" suffixes
basename_list = [name.replace('_repacked', '').replace('_copy', '') for name in basename_list]
# tau_frame['basename'] = tau_frame['basename'].str.replace('_repacked', '').str.replace('_copy', '')

# Load all spike train files into a dict
spike_train_dict = {this_name: np.load(this_file) for this_name, this_file in zip(basename_list, spike_train_files)}

# All files have only 1 array but with different key names
spike_train_dict = {k: v[list(v.files)[0]] for k, v in tqdm(spike_train_dict.items())}

##############################

# Plot all spike_trains

for this_name, this_spike_train in tqdm(spike_train_dict.items()):
    cat_spikes = np.concatenate(this_spike_train, axis=0)  # Concatenate across states
    fig,ax = vz.gen_square_subplots(cat_spikes.shape[1], figsize=(10,10),
                                    sharex=True, sharey=True)
    for unit in range(cat_spikes.shape[1]):
        vz.raster(ax.flatten()[unit], cat_spikes[:, unit], marker='|', color='black')
    fig.suptitle(f'Spike trains for {this_name}')
    fig.savefig(os.path.join(plot_dir, f'spike_trains_{this_name}.png'))
    plt.close(fig)

############################################################
############################################################

# Collect directly from sorted data

data_list_path = '/media/storage/abu_resorted/abu_all_datasetes.txt'
data_list = open(data_list_path, 'r').read().splitlines()

# Test that everything has spikes
error_list = []
for this_dir in tqdm(data_list):
    try:
        dat = ephys_data(this_dir)
        dat.get_spikes()
    except Exception as e:
        error_list.append(this_dir)
        print(e)

