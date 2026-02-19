"""
Perform network hyperparameter optimization using optuna
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

import optuna  # noqa
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour  # noqa 

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
import infer_rnn_rates_single as infer_rates 
# from utils import SpikeRasterIO

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

output_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output'
artifacts_sup_dir = os.path.join(output_path, 'artifacts/population_analysis')
artifacts_dir = os.path.join(artifacts_sup_dir, 'rnn_fits', 'test_fit')

# split_data_df_fp = os.path.join(
#     artifacts_sup_dir, 'trial_split_data_df.pkl'
#     )
# # with open(split_data_df_fp, 'wb') as f:
# #     dump(split_data_df, f)
# with open(split_data_df_fp, 'rb') as f:
#     split_data_df = load(f)
#
############################################################
############################################################

plots_sup_dir = os.path.join(output_path, 'plots', 'population_analysis')
plot_dir = os.path.join(plots_sup_dir, 'rnn_rates')

os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

###############
# Load data
# row_ind = 0
# this_row = split_data_df.loc[row_ind]
# basename = this_row['basename']
# train_spike_array = SpikeRasterIO.spike_times_to_spike_train(
#         this_row['train_spike_times'], this_row['train_shape'])
# test_spike_array = SpikeRasterIO.spike_times_to_spike_train(
#         this_row['test_spike_times'], this_row['test_shape'])

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

# Specify test_trials for cross-validation during training
test_fraction = 0.25
trial_inds = np.arange(spike_data.shape[0]*spike_data.shape[1]) # Trials are flattened internally during training, so trial_inds is just a range of the total number of trial-neuron combinations 
test_inds = np.random.choice(trial_inds, size=int(test_fraction * len(trial_inds)), replace=False)

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
# Run test training to make sure everything is working before running optuna optimization
from importlib import reload
reload(infer_rates)

net, loss, cross_val_loss, best_cross_val_loss, outputs, latent, split_dict = infer_rates.train_rnn_all_tastes(
    spike_data,
    taste_durations,
    params_dict,
    test_trials=test_inds,
)

############################################################
# Create optuna study and optimize hyperparameters

def objective(trial):
    # Sample hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 4, 64)
    rnn_layers = trial.suggest_int('rnn_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    bidirectional = trial.suggest_categorical('bidirectional', [False, True])
    strictly_positive = trial.suggest_categorical('strictly_positive', [False, True])

    # Update params_dict with sampled hyperparameters
    params_dict.update(
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        lr=lr,
        bidirectional=bidirectional,
        strictly_positive=strictly_positive,
    )

    # Print sampled hyperparameters for debugging
    print(f'Trial {trial.number}:')
    print(f'  hidden_size: {hidden_size}')
    print(f'  rnn_layers: {rnn_layers}')
    print(f'  dropout: {dropout}')
    print(f'  lr: {lr}')
    print(f'  bidirectional: {bidirectional}')
    print(f'  strictly_positive: {strictly_positive}')

    # Train the model and get the final loss
    net, loss, cross_val_loss, best_cross_val_loss, outputs, latent = train_rnn_all_tastes(
        spike_data,
        taste_durations,
        params_dict,
        test_trials=test_inds,
    )
    
    # final_loss = loss[-1]  # Use final training loss as objective value
    final_loss = best_cross_val_loss  # Use best cross-validation loss as objective value
    return final_loss

study = optuna.create_study(direction='minimize',
                            # storage="sqlite:///optuna_study.db",
                            # study_name="rnn_hyperparameter_optimization",
                            )
study.optimize(objective, n_trials=20)

# Save study as pkl
study_pkl_path = os.path.join(artifacts_dir, f'{basename}_optuna_study.pkl')
# with open(study_pkl_path, 'wb') as f:
#     dump(study, f)
with open(study_pkl_path, 'rb') as f:
    study = load(f)


fig = plot_optimization_history(study)
fig.write_html(os.path.join(plot_dir, f'{basename}_optuna_optimization_history.html'))
fig = plot_param_importances(study)
fig.write_html(os.path.join(plot_dir, f'{basename}_optuna_param_importances.html'))
# fig.show()

fig = plot_contour(study) 
fig.write_html(os.path.join(plot_dir, f'{basename}_optuna_contour.html'))
# fig.show()

study_df = study.trials_dataframe()

############################################################

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
