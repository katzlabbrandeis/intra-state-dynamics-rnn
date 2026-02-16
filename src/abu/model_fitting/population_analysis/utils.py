import numpy as np
from scipy.special import gammaln

class SpikeRasterIO:
    @staticmethod
    def spike_train_to_spike_times(spike_train):
        """
        Convert spike trains (binary arrays) to spike times (lists of spike timestamps).
        Args:
            spike_trains: numpy array of shape (trials, neurons, time_bins)
        Returns:
            array of shape (dims x spike_times), where dims is the number of dimensions in the original spike_trains (e.g. trials x neurons) 
        """
        original_array_shape = spike_train.shape
        return np.array(np.where(spike_train)), original_array_shape

    @staticmethod
    def spike_times_to_spike_train(spike_times, original_array_shape):
        """
        Convert spike times (lists of spike timestamps) back to spike trains (binary arrays).
        Args:
            spike_times: array of shape (dims x spike_times), where dims is the number of dimensions in the original spike_trains (e.g. trials x neurons)
            original_array_shape: tuple indicating the shape of the original spike_trains array (trials, neurons, time_bins)
        Returns:
            numpy array of shape (trials, neurons, time_bins) with binary values indicating spikes
        """
        spike_train = np.zeros(original_array_shape, dtype=int)
        spike_train[tuple(spike_times)] = 1
        return spike_train


def poisson_ll(lam, k):
    """
    Poisson log likelihood
    # Note: This has been tested against scipy.stats.poisson.logpmf

    Inputs:
        lam: lambda parameter of poisson distribution
        k: observed counts

    Outputs:
        ll: log likelihood

    # From /media/bigdata/firing_space_plot/Mahmood_2025_directional_influence/poisson_glm/src/utils/utils.py
    Ref: https://sherrytowers.com/2014/07/10/poisson-likelihood/
    """
    lam += 1e-10 # To ensure there is no log(0)
    assert len(lam) == len(k), 'lam and k must be same length'
    assert all(lam > 0), 'lam must be non-negative'
    assert all(k >= 0), 'k must be non-negative'
    return np.sum(k*np.log(lam) - lam - gammaln(k+1))

def calc_bits_per_spike(spike_train, rate):
    """
    Calculate bits per spike for a given spike train and firing rate.
    Args:
        spike_train: numpy array of shape (trials, neurons, time_bins) with binary values indicating spikes
        rate: firing rate in Hz (spikes per second)
    Returns:
        bits per spike

    Refs:
        - https://neuronaldynamics.epfl.ch/online/Ch10.S3.html
        - https://www.biorxiv.org/content/10.1101/2025.02.07.637062v2.full
    """
    
    mean_rate_ll = poisson_ll(rate, np.mean(spike_train)) 
    given_rate_ll = poisson_ll(rate, spike_train)

    bits_per_spike = (given_rate_ll - mean_rate_ll) / (np.log(2) * np.sum(spike_train)) 
    
    return bits_per_spike
