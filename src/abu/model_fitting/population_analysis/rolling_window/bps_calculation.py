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
from matplotlib import pyplot as plt

import sys
import os
base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src', 'abu', 'model_fitting', 'population_analysis')
sys.path.append(src_dir)

from utils import SpikeRasterIO,  calc_bits_per_spike

##############################
output_dir = os.path.join(base_dir, 'output')
artifacts_dir = os.path.join(output_dir, 'artifacts')
artifacts_subdir = os.path.join(artifacts_dir, 'population_analysis')

plots_sup_dir = os.path.join(output_dir, 'plots')
plot_dir = os.path.join(plots_sup_dir, 'population_analysis', 'rolling_window_bps')
os.makedirs(plot_dir, exist_ok=True)
##############################


# Load best fit data dataframe 
best_fit_data_df_fp = os.path.join(
    artifacts_subdir, 'best_fit_data_df.pkl'
    )
best_fit_data_df = pd.read_pickle(best_fit_data_df_fp)

# split_data_df_fp = os.path.join(
#     artifacts_subdir, 'split_data_df.pkl'
#     )
# split_data_df = pd.read_pickle(split_data_df_fp)

##############################
this_raster = best_fit_data_df['spike_trains'].iloc[0]
this_unit = this_raster[:,0]

fig, ax = vz.gen_square_subplots(this_raster.shape[1])
for unit_idx in range(this_raster.shape[1]):
    this_unit = this_raster[:,unit_idx]
    vz.raster(ax.flatten()[unit_idx], this_unit, color='k', marker = '|')
plt.show()

"""
Help on function _calc_conv_rates in module blech_clust.utils.ephys_data.ephys_data:

_calc_conv_rates(step_size, window_size, dt, spike_array)
    Calculate firing rates using convolution with moving window
    
    Args:
        step_size: Step size in milliseconds for moving window
        window_size: Window size in milliseconds for firing rate calculation
        dt: Inter-sample interval in milliseconds
        spike_array: N-D array with time as last dimension, binary spike data
    
    Returns:
        tuple: (firing_rate, time_vector)
            - firing_rate: Calculated firing rates, shape (*spike_array.shape[:-1], n_bins)
            - time_vector: Time vector in ms relative to stimulus delivery
    
    Raises:
        Exception: If step_size or window_size are not integer multiples of dt
"""

step_size = 25
window_size = 250
dt = 1
spike_array = this_raster

firing_rate, time_vector = ephys_data.ephys_data._calc_conv_rates(step_size, window_size, dt, spike_array)

vz.firing_overview(firing_rate.swapaxes(0,1))
plt.show()

binned_spikes = spike_array.reshape(spike_array.shape[0], spike_array.shape[1], -1, step_size).sum(axis=-1)

vz.firing_overview(binned_spikes.swapaxes(0,1))
plt.show()

# Cut binned spikes and firing rates to match in time dimension
min_time_bins = min(binned_spikes.shape[2], firing_rate.shape[2])
binned_spikes = binned_spikes[:,:,:min_time_bins]
firing_rate = firing_rate[:,:,:min_time_bins]

# Calculate bits per spike for each unit using the binned spikes and the firing rates
bps_list = []
for unit_idx in range(this_raster.shape[1]):
    unit_bps = calc_bits_per_spike(binned_spikes[:,unit_idx,:], firing_rate[:,unit_idx,:])
    bps_list.append(unit_bps)

# Calculate random cross-unit bps by shuffling the firing rates across units and calculating bps for each unit with the shuffled rates 
n_samples = 500
cross_unit_bps = []
for sample_idx in range(n_samples):
    binned_ind = np.random.choice(this_raster.shape[1])
    rate_ind = np.random.choice(this_raster.shape[1])
    if binned_ind == rate_ind:
        continue
    shuffled_rate = firing_rate[:,rate_ind,:]
    shuffled_binned = binned_spikes[:,binned_ind,:]
    sample_bps = calc_bits_per_spike(shuffled_binned, shuffled_rate)
    cross_unit_bps.append(sample_bps)

# Plot binned spikes and firing rates for a single unit to verify they are aligned
fig, ax = plt.subplots(2,this_raster.shape[1], figsize=(20,5))
for unit_idx in range(this_raster.shape[1]):
    ax[0,unit_idx].imshow(binned_spikes[:,unit_idx,:], aspect='auto', cmap='viridis', interpolation='nearest') 
    ax[0,unit_idx].set_title(f'Unit {unit_idx}\nBPS: {bps_list[unit_idx]:.2f}')
    ax[1,unit_idx].imshow(firing_rate[:,unit_idx,:], aspect='auto', cmap='viridis', interpolation='nearest') 
fig.suptitle('Binned Spikes and Firing Rates for Each Unit')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'bps_example_unit.png'))
plt.close(fig)

# Plot distribution of bits per spike across units and compare to cross-unit shuffled distribution
fig, ax = plt.subplots(1, figsize=(5,5))
ax.hist(bps_list, bins=20, alpha=0.7, label='Actual BPS', density=True)
ax.hist(cross_unit_bps, bins=20, alpha=0.7, label='Cross-unit shuffled BPS', density=True)
ax.set_xlabel('Bits per Spike')
ax.set_ylabel('Count')
ax.legend()
fig.suptitle('Distribution of Bits per Spike Across Units')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'bps_distribution.png'))
plt.close(fig)
