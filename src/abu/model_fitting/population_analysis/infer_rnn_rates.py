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
from blech_clust.utils.ephys_data import ephys_data, visualize as vz  # noqa
from cloudpickle import load, dump  # noqa
from sklearn.decomposition import PCA  # noqa

# Check that blechRNN is on the Desktop, if so, add to path
blechRNN_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'blechRNN')
if os.path.exists(blechRNN_path):
    sys.path.append(blechRNN_path)
else:
    raise FileNotFoundError('blechRNN not found on Desktop')

from src.train import train_model, MSELoss, PoissonLoss
from src.model import autoencoderRNN  # noqa

def prepare_data(spike_data, bin_size):
    # Cut taste_spikes to time limits
    # Bin spikes
    binned_spikes = np.reshape(
        spike_data, (*spike_data.shape[:-1], -1, bin_size)).sum(-1)
    return binned_spikes


def train_rnn_model(
        inputs, 
        labels, 
        train_steps, 
        hidden_size, 
        output_size, 
        device,
        rnn_layers=2,
        dropout=0.2,
        bidirectional=False,
        strictly_positive=False,
        lr=0.001,
        ):
    net = autoencoderRNN(
        input_size=inputs.shape[-1],
        hidden_size=hidden_size,
        output_size=output_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        strictly_positive=strictly_positive,
    )
    net.to(device)
    net, loss, cross_val_loss = train_model(
        net,
        inputs,
        labels,
        output_size=output_size,
        lr=lr,
        train_steps=train_steps,
        criterion=PoissonLoss(),
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
artifacts_dir = os.path.join(artifacts_sup_dir, 'rnn_fits', 'test_fit')
plots_sup_dir = os.path.join(output_path, 'plots', 'population_analysis')
plot_dir = os.path.join(plots_sup_dir, 'rnn_rates')

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

############################################################
############################################################

# Hardcode for now

# Network hyperparameters
hidden_size = 8
rnn_layers = 2
dropout = 0.2
lr = 0.001
bidirectional = False
strictly_positive = True

# Data parameters
bin_size = 25
time_lims = [0, spike_data.shape[-1]]
stim_start = 500
forecast_time = 25

# Training parameters
train_test_split = 0.9
train_steps = 50_000

params_dict = dict(
    hidden_size=hidden_size,
    rnn_layers=rnn_layers,
    dropout=dropout,
    lr=lr,
    bidirectional=bidirectional,
    strictly_positive=strictly_positive,
    bin_size=bin_size,
    time_lims=time_lims,
    stim_start=stim_start,
    forecast_time=forecast_time,
    train_test_split=train_test_split,
    train_steps=train_steps,
    )

############################################################
############################################################

def train_rnn_all_tastes(
        spike_data,
        taste_durations,
        params_dict,
        ):

    """
    Train an RNN model to predict firing rates from spike data for each taste.

    Parameters:
        - spike_data: 4D numpy array of shape (tastes x trials x neurons x time)
        - taste_durations: 1D numpy array of average taste durations for each taste
        - params_dict: dictionary of parameters for training and model configuration
            - hidden_size: int, number of hidden units in the RNN
            - rnn_layers: int, number of RNN layers
            - dropout: float, dropout rate for the RNN
            - lr: float, learning rate for training
            - bidirectional: bool, whether to use a bidirectional RNN
            - strictly_positive: bool, whether to enforce strictly positive outputs
            - bin_size: int, size of time bins for spike data
            - time_lims: list of two ints, start and end times for analysis
            - stim_start: int, time of stimulus onset in ms
            - forecast_time: int, time in ms to forecast ahead
            - train_test_split: float, proportion of data to use for training
            - train_steps: int, number of training steps for the RNN
    Returns:
        - net: trained RNN model
        - loss: list of training losses over time
        - cross_val_loss: list of cross-validation losses over time
    """

    assert spike_data.ndim == 4, "Spike data must be 4D (tastes x trials x neurons x time)"
    assert spike_data.shape[0] == len(taste_durations), "Number of tastes in spike data and taste durations must match"

    # Unload parameters
    hidden_size = params_dict['hidden_size']
    rnn_layers = params_dict['rnn_layers']
    dropout = params_dict['dropout']
    lr = params_dict['lr']
    bidirectional = params_dict['bidirectional']
    strictly_positive = params_dict['strictly_positive']
    bin_size = params_dict['bin_size']
    time_lims = params_dict['time_lims']
    stim_start = params_dict['stim_start']
    forecast_time = params_dict['forecast_time']
    train_test_split = params_dict['train_test_split']

    ############### 
    n_tastes = len(spike_data)
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

    taste_num_scaled = np.arange(n_tastes) / n_tastes
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

    # fig, ax = vz.firing_overview(inputs_long_plus_context.T)
    # fig.savefig(os.path.join(plot_dir, f'{basename}_firing_overview.png'))
    # plt.close(fig)
    # plt.show()

    forecast_bins = int(forecast_time // bin_size)
    # inputs_plus_context = inputs_plus_context[:-forecast_bins]
    inputs_long_plus_context = inputs_long_plus_context[:-forecast_bins]
    # inputs = inputs[forecast_bins:]
    labels = inputs_long[forecast_bins:]


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    labels_torch = torch.from_numpy(labels).type(torch.float32)
    inputs_torch = torch.from_numpy(inputs_long_plus_context).type(torch.float)

    input_size = inputs_long_plus_context.shape[-1]
    output_size = labels.shape[-1]

    train_inds = np.random.choice(
        np.arange(inputs_long_plus_context.shape[1]),
        int(train_test_split * inputs_long_plus_context.shape[1]),
        replace=False)
    test_inds = np.setdiff1d(
        np.arange(inputs_long_plus_context.shape[1]), train_inds)

    train_inputs = inputs_torch[:, train_inds]
    test_inputs = inputs_torch[:, test_inds]
    train_labels = labels_torch[:, train_inds]
    test_labels = labels_torch[:, test_inds]

    train_inputs = train_inputs.to(device)
    train_labels = train_labels.to(device)
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)

    net, loss, cross_val_loss = train_rnn_model(
        train_inputs, 
        train_labels, 
        train_steps, 
        hidden_size, 
        output_size, 
        device,
        rnn_layers=rnn_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        strictly_positive=strictly_positive,
        lr=lr,
    )

    # Get predictions
    outputs, latent = net(inputs_torch.to(device))
    # Shape: time_bins x (tastes*trials) x output_size
    outputs = outputs.detach().cpu().numpy()
    # Shape: time_bins x (tastes*trials) x latent_size
    latent = latent.detach().cpu().numpy()

    return net, loss, cross_val_loss, outputs, latent

net, loss, cross_val_loss, outputs, latent = train_rnn_all_tastes(
    spike_data,
    taste_durations,
    params_dict,
)

# Save the model
model_save_path = os.path.join(artifacts_dir, f'{basename}_rnn_model.pt')
torch.save(net.state_dict(), model_save_path)

# Load model (for testing)
# net.load_state_dict(torch.load(model_save_path)) 

# Plot outputs and latents
fig, ax = vz.firing_overview(outputs.T)
fig.savefig(os.path.join(plot_dir, f'{basename}_rnn_outputs.png'))
plt.close(fig)

fig, ax = vz.firing_overview(latent.T)
fig.savefig(os.path.join(plot_dir, f'{basename}_rnn_latent.png'))
plt.close(fig)

# Plot latents for n random trials
# Also plot PCA of latents
latent_long = latent.reshape(-1, latent.shape[-1])
pca_obj = PCA(n_components=0.9)
latent_pca = pca_obj.fit_transform(latent_long)
latent_pca_trials = latent_pca.reshape(latent.shape[0], -1, latent_pca.shape[-1])

n_random = 10
random_inds = np.random.choice(latent.shape[1], n_random, replace=False)
fig, ax = plt.subplots(n_random, 2, figsize=(10, 2*n_random), sharex=True, sharey=True)
for i, ind in enumerate(random_inds):
    ax[i,0].plot(latent[:, ind])
    ax[i,0].set_title(f'Trial {ind}')
    ax[i,0].axvline(stim_start_ind, color='red', linestyle='--')
    ax[i,1].plot(latent_pca_trials[:, ind])
    ax[i,1].set_title(f'Trial {ind} PCA')
    ax[i,1].axvline(stim_start_ind, color='red', linestyle='--')
fig.suptitle('RNN Latent Factors for Random Trials + PCA of Latents')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, f'{basename}_rnn_latent_random_trials.png'))
plt.close(fig)
