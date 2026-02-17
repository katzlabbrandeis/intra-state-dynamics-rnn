"""
Script that takes path to pkl as input, trains an RNN on the data, and outputs the predicted rates to a new pkl file.
Usage:
    python infer_rnn_rates_single.py --input_path path/to/input.pkl --output_dir path/to/output/dir

Output name will be the same as the input name, but with "_predicted" appended before the .pkl extension. For example, if the input file is "data.pkl", the output file will be "data_predicted.pkl".
"""

import argparse  # noqa: E402
import os  # noqa
import torch  # noqa
import numpy as np  # noqa
import sys  # noqa
from pprint import pprint  # noqa
from cloudpickle import load, dump  # noqa

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
    print(f"Binning spike data with bin_size={bin_size}ms...")
    binned_spikes = np.reshape(
        spike_data, (*spike_data.shape[:-1], -1, bin_size)).sum(-1)
    print(f"Binned spike data shape: {binned_spikes.shape}")
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
            - stim_start: int, time of stimulus onset in ms
            - forecast_time: int, time in ms to forecast ahead
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
    stim_start = params_dict['stim_start']
    forecast_time = params_dict['forecast_time']

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

    train_inputs = inputs_torch.to(device)
    train_labels = labels_torch.to(device)

    print(f"Training RNN model for {train_steps} steps...")
    print(f"Model configuration: hidden_size={hidden_size}, rnn_layers={rnn_layers}, "
          f"bidirectional={bidirectional}, dropout={dropout}")
    
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
    print("Generating predictions from trained model...")
    outputs, latent = net(inputs_torch.to(device))
    # Shape: time_bins x (tastes*trials) x output_size
    outputs = outputs.detach().cpu().numpy()
    # Shape: time_bins x (tastes*trials) x latent_size
    latent = latent.detach().cpu().numpy()

    return net, loss, cross_val_loss, outputs, latent

# # Bundle all inputs into a single pkl so training can be parallelized
# inputs_dict = dict(
#     spike_data=spike_data,
#     taste_durations=taste_durations,
#     params_dict=params_dict,
#     )
# inputs_pkl_path = os.path.join(artifacts_dir, f'{basename}_rnn_inputs.pkl')
# with open(inputs_pkl_path, 'wb') as f:
#     dump(inputs_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an RNN to predict firing rates from spike data for each taste.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input pkl file containing spike data, taste durations, and parameters.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output pkl file with predicted rates.')
    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir

    print(f"Loading input data from: {input_path}")
    with open(input_path, 'rb') as f:
        inputs_dict = load(f)
    print("Input data loaded successfully")

    spike_data = inputs_dict['spike_data']
    taste_durations = inputs_dict['taste_durations']
    params_dict = inputs_dict['params_dict']

    print(f"Spike data shape: {spike_data.shape}")
    print(f"Number of tastes: {len(taste_durations)}")
    print("Training parameters:")
    pprint(params_dict)
    print("\nStarting RNN training...")
    
    net, loss, cross_val_loss, outputs, latent = train_rnn_all_tastes(
        spike_data,
        taste_durations,
        params_dict,
    )
    print("Training completed successfully")
    print(f"Final training loss: {loss[-1]:.6f}")
    print(f"Final cross-validation loss: {cross_val_loss[-1]:.6f}")
    
    # Save outputs to new pkl
    print("\nPreparing outputs for saving...")
    output_dict = dict(
        outputs=outputs,
        latent=latent,
        loss=loss,
        cross_val_loss=cross_val_loss,
    )
    input_basename = os.path.basename(input_path)
    output_basename = input_basename.replace('.pkl', '_predicted.pkl')
    output_path = os.path.join(output_dir, output_basename)
    
    print(f"Saving results to: {output_path}")
    with open(output_path, 'wb') as f:
        dump(output_dict, f)
    print("Results saved successfully")
    print(f"\nOutput shapes:")
    print(f"  - outputs: {outputs.shape}")
    print(f"  - latent: {latent.shape}")

