import numpy as np

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
