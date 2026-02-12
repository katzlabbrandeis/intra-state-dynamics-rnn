"""
Refactored as part of blech_clust PR #326

This module uses an Auto-regressive Recurrent Neural Network (RNN) to infer firing rates from electrophysiological data. It processes data for each taste separately, trains an RNN model, and saves the predicted firing rates and latent factors.

- Parses command-line arguments to configure the RNN model, including data directory, training steps, hidden size, bin size, train-test split, PCA usage, retraining option, and time limits.
- Loads configuration from a JSON file if not overridden by command-line arguments.
- Loads spike data using the `ephys_data` class and preprocesses it, including binning and optional PCA.
- Trains an RNN model for each taste, using a specified loss function (MSE) and saves the model and training artifacts.
- Generates and saves various plots, including firing rate overviews, latent factors, and mean firing rates.
- Writes the predicted firing rates and latent outputs to an HDF5 file for each taste.
- Handles file paths and directories for saving models, plots, and outputs, ensuring necessary directories exist.
"""

import argparse  # noqa: E402
import os  # noqa
import matplotlib.pyplot as plt  # noqa
import torch  # noqa
import numpy as np  # noqa
import sys  # noqa
from pprint import pprint  # noqa
import json  # noqa
from itertools import product  # noqa
import pandas as pd  # noqa
import xarray as xr  # noqa
from blech_clust.utils.ephys_data import ephys_data, visualize as vz  # noqa
from cloudpickle import load, dump  # noqa

# Check that blechRNN is on the Desktop, if so, add to path
blechRNN_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'blechRNN')
if os.path.exists(blechRNN_path):
    sys.path.append(blechRNN_path)
else:
    raise FileNotFoundError('blechRNN not found on Desktop')

from src.train import train_model, MSELoss  # noqa
from src.model import autoencoderRNN  # noqa


def prepare_data(spike_data, bin_size):
    # Cut taste_spikes to time limits
    # Bin spikes
    binned_spikes = np.reshape(
        spike_data, (*spike_data.shape[:-1], -1, bin_size)).sum(-1)
    return binned_spikes


def train_rnn_model(inputs, labels, train_steps, hidden_size, output_size, device):
    net = autoencoderRNN(
        input_size=inputs.shape[-1],
        hidden_size=hidden_size,
        output_size=output_size,
        rnn_layers=2,
        dropout=0.2,
    )
    net.to(device)
    net, loss, cross_val_loss = train_model(
        net,
        inputs,
        labels,
        output_size=output_size,
        lr=0.001,
        train_steps=train_steps,
        criterion=MSELoss(),
    )
    return net, loss, cross_val_loss

############################################################
############################################################
# Load collected spike data and prepare for processing
spike_pkl_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/artifacts/population_analysis/best_fit_data_df.pkl'
with open(spike_pkl_path, 'rb') as f:
    best_fit_data_df = load(f)

data_dir_file = '/media/bigdata/abu_resorted_rolling/abu_all_datasetes.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

basenames = [os.path.basename(x) for x in data_dir_list]
basename_map = dict(zip(basenames, data_dir_list))

#Columns: basename, taste_name, spike_trains
best_fit_data_df['data_dir'] = best_fit_data_df['basename'].map(basename_map)

############################################################
############################################################

output_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output'
artifacts_sup_dir = os.path.join(output_path, 'artifacts/population_analysis')
artifacts_dir = os.path.join(artifacts_sup_dir, 'rnn_fits')
plots_sup_dir = os.path.join(output_path, 'plots')
plot_dir = os.path.join(plots_sup_dir, 'population_analysis')

os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

###############
# Load data
data_dir = best_fit_data_df.loc[0, 'data_dir']
basename = os.path.basename(data_dir)
print(f'Processing data from {data_dir}')
# Shape: trials x neurons x time
this_df = best_fit_data_df.loc[best_fit_data_df['basename'] == basename]
taste_order = this_df.taste_name
spike_data = np.stack(this_df['spike_trains'].to_list())

# Get taste open-times using trial-info-frame
dat = ephys_data.ephys_data(data_dir)
dat.get_trial_info_frame()

info_frame = dat.trial_info_frame.copy()
info_frame['taste_duration'] = info_frame['end_taste_ms'] - info_frame['start_taste_ms']
# Get average taste duration from each taste
taste_durations = info_frame.groupby('taste')['taste_duration'].mean()
# Sort by taste order in the data
taste_durations = taste_durations.loc[taste_order]

##############################

# mse loss performs better than poisson loss
loss_name = 'poisson'

spikes_xr = [xr.DataArray(
    x,
    dims=['trials', 'neurons', 'time'],
    coords={
        'trials': np.arange(x.shape[0]),
        'neurons': np.arange(x.shape[1]),
        'time': np.arange(x.shape[2]),
        'region': (['neurons'], range(x.shape[1])),
    }
) for x in spike_data] 

############################################################
############################################################

# Hardcode for now
hidden_size = 8
bin_size = 25
forecast_time = 25
train_test_split = 0.9
time_lims = [0, spike_data.shape[-1]]
stim_start = 500

############################################################
############################################################

binned_spikes = prepare_data(spike_data, bin_size)
trial_num = np.arange(spike_data.shape[1])

inputs = binned_spikes.copy()
# New shape: time_bins x tastes x trials x neurons
inputs = np.moveaxis(inputs, -1, 0)

stim_start_ind = stim_start // bin_size
stim_dur_inds = taste_durations // bin_size
stim_end_ind = stim_start_ind + stim_dur_inds.astype(int)

stim_time = np.zeros(inputs.shape[:3])
for taste_ind, this_end_ind in enumerate(stim_end_ind):
    stim_time[stim_start_ind:this_end_ind, taste_ind] = 1

trial_num_scaled = trial_num / trial_num.max()
trial_num_broad = np.broadcast_to(trial_num_scaled, inputs.shape[:-1])

taste_num_scaled = np.arange(len(taste_order)) / len(taste_order)
taste_num_broad = np.broadcast_to(taste_num_scaled[None,:,None], inputs.shape[:-1])

# Stack trials across tastes so that 2nd dimension is trials*tastes
inputs_long = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1])
stim_time_long = stim_time.reshape(stim_time.shape[0], -1)
trial_num_long = trial_num_broad.reshape(trial_num_broad.shape[0], -1)
taste_num_long = taste_num_broad.reshape(taste_num_broad.shape[0], -1)

inputs_long_plus_context = np.concatenate(
    [
        inputs_long,
        stim_time_long[:, :, None],
        trial_num_long[:, :, None],
        taste_num_long[:, :, None],
    ],
    axis=-1)

vz.firing_overview(inputs_long_plus_context.T)
plt.show()

def prepare_combination_data(spike_data, params_dict):




    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    forecast_bins = int(
        params_dict['forecast_time'] // params_dict['bin_size'])
    inputs_plus_context = inputs_plus_context[:-forecast_bins]
    inputs = inputs[forecast_bins:]

    labels = torch.from_numpy(inputs).type(torch.float32)
    inputs = torch.from_numpy(inputs_plus_context).type(torch.float)

    return spike_data, inputs_plus_context, labels, device, stim_time_val, pca_obj


def train_combination_model(inputs_plus_context, labels, params_dict, device):
    input_size = inputs_plus_context.shape[-1]
    output_size = inputs_plus_context.shape[-1] - 2

    train_inds = np.random.choice(
        np.arange(inputs_plus_context.shape[1]),
        int(params_dict['train_test_split'] * inputs_plus_context.shape[1]),
        replace=False)
    test_inds = np.setdiff1d(
        np.arange(inputs_plus_context.shape[1]), train_inds)

    train_inputs = inputs_plus_context[:, train_inds]
    train_labels = labels[:, train_inds]
    test_inputs = inputs_plus_context[:, test_inds]
    test_labels = labels[:, test_inds]

    train_inputs = train_inputs.to(device)
    train_labels = train_labels.to(device)
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)

    net, loss, cross_val_loss = train_rnn_model(
        train_inputs, train_labels, params_dict['train_steps'], params_dict['hidden_size'], output_size, device
    )

    if loss[-1] > cross_val_loss[max(cross_val_loss.keys())]:
        warning_file_path = os.path.join(artifacts_dir, 'warning.txt')
        warning_str = """
        Final training loss is greater than cross validation loss.
        This indicates something weird is going on (maybe with train-test split or PCA).
        Try retraining the model (to get a new train-test split) or using the --no-pca flag.
        """
        with open(warning_file_path, 'w') as f:
            f.write(warning_str)
        print(warning_str)

    return net, loss, cross_val_loss


