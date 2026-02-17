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

src_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/src/abu/model_fitting/population_analysis'
sys.path.append(src_dir)
from infer_rnn_rates_single import train_rnn_all_tastes 

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
stim_start = 500
forecast_time = 25

# Training parameters
train_steps = 50_000

params_dict = dict(
    hidden_size=hidden_size,
    rnn_layers=rnn_layers,
    dropout=dropout,
    lr=lr,
    bidirectional=bidirectional,
    strictly_positive=strictly_positive,
    bin_size=bin_size,
    stim_start=stim_start,
    forecast_time=forecast_time,
    train_steps=train_steps,
    )

# Bundle all inputs into a single pkl so training can be parallelized
inputs_dict = dict(
    spike_data=spike_data,
    taste_durations=taste_durations,
    params_dict=params_dict,
    )
inputs_pkl_path = os.path.join(artifacts_dir, f'{basename}_rnn_inputs.pkl')
with open(inputs_pkl_path, 'wb') as f:
    dump(inputs_dict, f)

############################################################
############################################################

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

fig, ax = plt.subplots()
ax.plot(loss, label='Training Loss')
# plt.plot(cross_val_loss, label='Cross-Validation Loss')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Loss')
ax.set_title('RNN Training Loss')
plt.legend()
# Make an inset plot zooming in on the last 1000 training steps
inset_ax = fig.add_axes([0.5, 0.5, 0.4, 0.4])
inset_ax.plot(loss[-5000:], label='Training Loss')
fig.savefig(os.path.join(plot_dir, f'{basename}_rnn_training_loss.png'))
plt.close(fig)

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
