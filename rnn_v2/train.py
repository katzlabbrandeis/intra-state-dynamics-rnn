"""
Class to handle model training and prediction
"""

############################################################
# Imports
############################################################

from torch.nn import init
from torch.nn import functional as F
from scipy.stats import poisson, zscore
from model import autoencoderRNN
import torch.optim as optim
import torch.nn as nn
import torch
import math
import os
import sys
import time

import numpy as np
import pylab as plt
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm, trange

file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(file_path)
sys.path.append(src_dir)

############################################################
# Define Model
############################################################
# Define networks


class smooth_MSELoss(nn.Module):
    """
    MSE loss with temporal smoothness constraint
    """

    def __init__(self, alpha=0.05):
        super(smooth_MSELoss, self).__init__()
        self.loss1 = nn.MSELoss()
        self.alpha = alpha

    def mean_diffrence(self, x):
        """
        Calculate the mean difference between adjacent elements

        Args:
            x: (seq_len, batch, output_size)
        """
        return torch.mean(torch.abs(x[1:] - x[:-1])) * self.alpha

    def forward(self, input, target):
        """
        Args:
            input: (seq_len,batch, output_size)
            target: (seq_len,batch, output_size)
        """
        loss = self.loss1(input, target) + self.mean_diffrence(input)
        return loss


def MSELoss():
    return nn.MSELoss()

# a few utils for better model comparisons


def poisson_log_likelihood(predicted_rates, actual_counts):
    """
    Poisson log-likelihood: sum( y*log(r) - r - log(y!) )

    Args:
        predicted_rates: tensor, model output (must be > 0)
        actual_counts: tensor, observed spike counts

    Returns:
        float, total log-likelihood
    """
    r = torch.clamp(predicted_rates, min=1e-8)
    y = actual_counts
    ll = torch.sum(y * torch.log(r) - r - torch.lgamma(y + 1))
    return ll.item()


def count_parameters(net):
    """Count trainable parameters in a network."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

# now to compute AIC and BIC (Alkine & Baysiean information cirterion-- relevant to prediction error).
# when comparing across models, lower is better.


def compute_aic_bic(net, inputs, labels):
    """
    Compute AIC and BIC using Poisson log-likelihood.

    Args:
        net: trained model
        inputs: (seq_len, batch, input_size)
        labels: (seq_len, batch, output_size) — observed counts/rates

    Returns:
        dict with aic, bic, log_likelihood, n_params, n_observations

    Significance of the following values:
    - n_Params:
    - Observations:
    - Log-Liklihood
    - AIC:
    - BIC:

    """
    net.eval()
    with torch.no_grad():
        output, _ = net(inputs)
        output = torch.clamp(output, min=1e-8)

    ll = poisson_log_likelihood(output, labels)
    k = count_parameters(net)
    n = labels.numel()

    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    net.train()

    return dict(
        aic=aic,
        bic=bic,
        log_likelihood=ll,
        n_params=k,
        n_observations=n,
    )


def train_model(
        net,
        inputs,
        labels,
        output_size,
        train_steps=1000,
        lr=0.01,
        delta_loss=0.01,
        device=None,
        criterion=MSELoss(),
        test_inputs=None,
        test_labels=None,
        patience=10,  # New parameter for early stopping
        quiet=False
):
    """Simple helper function to train the model.

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
        inputs: shape (seq_len, batch, input_size)
        labels: shape (seq_len * batch, output_size)

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    cross_val_bool = np.logical_and(
        test_inputs is not None,
        test_labels is not None
    )

    loss_history = []
    cross_val_loss = {}
    best_loss = float('inf')
    patience_counter = 0
    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(train_steps):
        # labels = labels.reshape(-1, output_size)
        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # # Reshape to (SeqLen x Batch, OutputSize)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Only compute cross_val_loss every 100 steps
        # because it's expensive
        if cross_val_bool and (i % 100 == 99):
            test_out, _ = net(test_inputs)
            test_out_flat = test_out.reshape(-1, output_size)
            test_labels_flat = test_labels.reshape(-1, output_size)
            # consider the potential that the below is a bug-- is test labels getting reshaped every single loop here? hmmm
            # test_labels = test_labels.reshape(-1, output_size)
            test_loss = criterion(test_out_flat, test_labels_flat)
            cross_val_loss[i] = test_loss.item()
            # Early stopping check
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at step {i+1}')
                break

            cross_str = f'Cross Val Loss: {test_loss.item():0.4f}'
        else:
            cross_str = ''

        # Compute the running loss every 100 steps
        current_loss = loss.item()
        loss_history.append(current_loss)
        running_loss += current_loss
        if i % 100 == 99:
            running_loss /= 100
            if not quiet:
                print('Step {}, Loss {:0.4f}, {}, Time {:0.1f}s'.format(
                    i+1, running_loss, cross_str, time.time() - start_time))
            running_loss = 0
    # now calculating AIC/BIC at the end of every training:
    if cross_val_bool:
        eval_inputs = test_inputs
        eval_labels = test_labels
        eval_set = 'test'
    else:
        eval_inputs = inputs
        eval_labels = labels
        eval_set = 'train'

    info_criteria = compute_aic_bic(net, eval_inputs, eval_labels)
    info_criteria['eval_set'] = eval_set
    # print some info at the end of training just bc?
    print(f"\n--- Information Criteria ({eval_set} set) ---")
    print(f"  Params:         {info_criteria['n_params']}")
    print(f"  Observations:   {info_criteria['n_observations']}")
    print(f"  Log-Likelihood: {info_criteria['log_likelihood']:.2f}")
    print(f"  AIC:            {info_criteria['aic']:.2f}")
    print(f"  BIC:            {info_criteria['bic']:.2f}")
    return net, loss_history, cross_val_loss, info_criteria
