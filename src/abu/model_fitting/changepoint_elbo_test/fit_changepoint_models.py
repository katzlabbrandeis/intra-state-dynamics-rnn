"""
Test best # of changepoint states for datasets to simultaneously determine extent of dynamics and drift 

Fit models:
    - Across range of states
    - Using only stable neurons
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Import modules
base_dir = '/media/bigdata/projects/pytau'
import sys
sys.path.append(base_dir)
from pytau.changepoint_io import FitHandler
import pylab as plt
# from pytau.utils import ephys_data
from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data import visualize as vz
from tqdm import tqdm
import pytau.changepoint_model as models
from pprint import pprint as pp
import pandas as pd
from ast import literal_eval
import numpy as np
import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
from cloudpickle import load, dump
from scipy import stats

def model_generator(train_dat, n_states):
    """
    Generate a CategoricalChangepoint2D model for the given training data.
    
    Parameters:
    -----------
    train_dat : numpy.ndarray
        Training data in the shape (trials, time_points, emission_dim)
    n_states : int
        Number of states for the HMM
    
    Returns:
    --------
    model : CategoricalChangepoint2D
        The generated changepoint model
    """
    # change_instance = models.CategoricalChangepoint2D(
    change_instance = models.SingleTastePoisson(
        train_dat,
        n_states=n_states,  # Initial number of states
    )
    return change_instance.generate_model()

# Use the first session's data for model training

def fit_model(model, n_retry=3, conv_tol=1e-2, n_iter=100_000):
    """
    Fit the model using ADVI with retries in case of convergence issues.
    
    Parameters:
    -----------
    model : pymc3.Model
        The PyMC3 model to fit
    n_retry : int
        Number of retries if convergence fails
    
    Returns:
    --------
    approx : pymc3.variational.approximations.ADVI
        The fitted approximation
    """
    retry_count = 0
    success = False
    while (retry_count < n_retry) and not success:
        try:
            seed = np.random.randint(0, 10000)  # Random seed for reproducibility
            with model:
                inference = pm.ADVI("full-rank", random_seed=seed)
                approx = pm.fit(
                    n=n_iter, 
                    method=inference,
                    callbacks=[
                        CheckParametersConvergence(
                            diff='absolute',
                            tolerance=conv_tol,
                        )
                    ]
                )
            # Check that all of elbo history is not inf or nan
            elbo_hist = np.array(approx.hist)
            elbo_hist = elbo_hist[~np.isnan(elbo_hist)]
            elbo_hist = elbo_hist[~np.isinf(elbo_hist)]
            if len(elbo_hist) == 0: 
                print(f"ELBO is inf or NaN. Retrying {retry_count + 1}/{n_retry}...")
                success = False
                retry_count += 1
                continue
            else:
                success = True
        except Exception as e:
            print(f"Error fitting model: {e}. Retrying {retry_count + 1}/{n_retry}...")
            retry_count += 1
            continue
    if retry_count == n_retry:
        print("Model fitting failed after maximum retries.")
        return None, None
    else:
        return approx, retry_count

##############################
# Data Dirs
# data_dir_file = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/data_dir_list.txt'
# with open(data_dir_file, 'r') as f:
#     data_dir_list = f.read().splitlines()
base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
# base_dir = '/media/bigdata/firing_space_plot/NBT_EMB_Classifier_Analyses'
# artifacts_dir = '/media/bigdata/firing_space_plot/NBT_EMB_Classifier_Analyses/artifacts'
artifacts_dir = os.path.join(base_dir, 'output', 'artifacts')

artifacts_subdir = os.path.join(artifacts_dir, 'ephys_changepoint_models')
os.makedirs(artifacts_subdir, exist_ok=True)
change_out_dir = os.path.join(artifacts_subdir, 'models')
os.makedirs(change_out_dir, exist_ok=True)

##############################
# data_dir_file = os.path.join(base_dir, 'src', 'GC_EMG_alignment', 'data_dir_list.txt')
data_dir_file = '/media/bigdata/abu_resorted_rolling/abu_all_datasetes.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

pp(data_dir_list)

##############################
# Collect data to fit

# from importlib import reload
# reload(ephys_data)

recreate_data_bool = True

if recreate_data_bool:
    fit_data_list = []
    for this_dir in tqdm(data_dir_list, desc='Processing data dirs'):

        this_dat = ephys_data.ephys_data(this_dir)
        this_dat.profile_units(recalculate=False)
        basename = os.path.basename(this_dir)

        if 'spikes' not in dir(this_dat):
            this_dat.get_spikes()
            this_dat.get_info_dict()

        this_dat.check_laser()
        if this_dat.laser_exists:
            this_dat.separate_laser_spikes()
            off_spikes = this_dat.off_spikes
        else:
            off_spikes = this_dat.spikes

        # Only stable units
        stable_alpha = 0.005
        stable_units = this_dat.unit_profile[
                this_dat.unit_profile['stable_pval'] > stable_alpha
                ]['neuron_num'].values


        # # responsive, dynamic, or discriminative units
        # col_names = ['responsive','discriminative','dynamic']
        # wanted_profile = this_dat.unit_profile[
        #         this_dat.unit_profile[col_names].any(axis=1)
        #         ]['neuron_num'].values
        #
        # selected_units = np.intersect1d(stable_units, wanted_profile)

        selected_units = stable_units  # For now, just use stable units for both conditions

        # vz.firing_overview(this_dat.all_normalized_firing[stable_units],
        #                    subplot_labels=stable_units)
        # vz.firing_overview(this_dat.all_normalized_firing[selected_units],
        #                    subplot_labels=selected_units)
        # plt.show()

        taste_names = this_dat.info_dict['taste_params']['tastes']

        for taste_ind, taste_name in enumerate(taste_names):
            # Extract spike trains for selected units
            spike_trains = off_spikes[taste_ind]
            time_lims = [2000, 4000]
            spike_trains = spike_trains[:, :, time_lims[0]:time_lims[1]]  # Trim to time window
            only_stable_spikes = spike_trains[:, stable_units]
            selected_spikes = spike_trains[:, selected_units]

            fit_data_list.append({
                'basename': basename,
                'taste_name': taste_name,
                'dat_type': 'stable_only',
                'spike_trains': only_stable_spikes,
            })

            # fit_data_list.append({
            #     'basename': basename, 
            #     'taste_name': taste_name,
            #     'dat_type': 'stable_responsive_dynamic_discriminative',
            #     'spike_trains': selected_spikes,
            #     })

    # Combine into a dataframe and write to disk
    fit_data_df = pd.DataFrame(fit_data_list)
    fit_data_df['n_neurons'] = fit_data_df['spike_trains'].apply(lambda x: x.shape[1])
    # Drop any datasets with <8 neurons
    fit_data_df = fit_data_df[fit_data_df['n_neurons'] >= 8].reset_index(drop=True)

    fit_out_path = os.path.join(artifacts_subdir, 'ephys_fit_data_all_sessions.pkl')
    dump(fit_data_df, open(fit_out_path, 'wb'))
else:
    fit_out_path = os.path.join(artifacts_subdir, 'ephys_fit_data_all_sessions.pkl')
    fit_data_df = load(open(fit_out_path, 'rb'))


# For each dataset, print how many stable neurons there are
for ind, this_row in fit_data_df.iterrows():
    this_basename = this_row['basename']
    this_taste_name = this_row['taste_name']
    this_dat_type = this_row['dat_type']
    this_dat = this_row['spike_trains']
    n_neurons = this_dat.shape[1]
    print(f"Dataset: {this_basename}, Taste: {this_taste_name}, Type: {this_dat_type}, Neurons: {n_neurons}")

##############################


force_refit = True
n_states_vec = np.arange(2,8)  # Fit models with 2 to 7 states

# for group_ind, group_df in tqdm(grouped_mtm, total=len(grouped_mtm), desc='Processing sessions'):

for ind, this_row in tqdm(fit_data_df.iterrows(), total=len(fit_data_df), desc='Fitting models'):

    # this_basename, this_taste_name, this_section = group_ind
    this_basename = this_row['basename']
    this_taste_name = this_row['taste_name']
    this_dat_type = this_row['dat_type']
    # this_section = int(this_section)

    this_dat = this_row['spike_trains']  # Shape: (n_trials, n_neurons, n_timepoints) 

    # Change so that there are no negative values
    # this_dat = this_dat - np.min(this_dat)  # Shift to non-negative values
    # Bin spike-trains
    bin_width = 50  # in ms
    n_trials, n_neurons, n_timepoints = this_dat.shape
    n_bins = n_timepoints // bin_width
    binned_dat = np.reshape(
            this_dat[:, :, :n_bins * bin_width],
            (n_trials, n_neurons, n_bins, bin_width)
            ).sum(axis=-1)

    # Reshape data to match expected format: (batches, timesteps, emission_dim)
    # train_dat = this_dat[:,:, np.newaxis]  # Add a new axis for emission_dim
    train_dat = binned_dat.copy()  

    for n_states in n_states_vec:

        model_info_dict = {
            'basename': this_basename,
            'taste_name': this_taste_name,
            'dat_type': this_dat_type,
            'n_states': n_states,
        }

        save_name = f'{this_basename}_{this_taste_name}_{this_dat_type}_{n_states}_states_model.pkl'
        if os.path.exists(os.path.join(change_out_dir, save_name)) and not force_refit:
            print(f"Model already exists: {save_name}, skipping...") 
            continue

        model = model_generator(train_dat, int(n_states))
        approx, retry_count = fit_model(model, n_retry=5, conv_tol=1e-3, n_iter=50_000)
        if approx is None:
            print(f"Skipping model with {n_states} states due to fitting failure.")
            continue
        else:
            print(f"Model fitted successfully")
            pp(model_info_dict)
        # Take last non-NaN or non-inf elbo
        elbo = approx.hist[~np.isnan(approx.hist)]
        elbo = elbo[~np.isinf(elbo)][-1]

        # plt.plot(approx.hist);plt.show()

        # Perform sampling
        with model:
            posterior_samples = approx.sample(1000)
        tau_samples = posterior_samples.posterior['tau'].values[0]
        mode_tau = stats.mode(np.round(tau_samples).astype(int), axis=0, keepdims=False).mode
        # If mode_tau is 1D, make it 2D
        if mode_tau.ndim == 1:
            mode_tau = mode_tau[np.newaxis, :]

        dump(
            (model, approx, elbo, retry_count, train_dat, mode_tau, model_info_dict),
            open(os.path.join(change_out_dir, save_name), 'wb')
            )

        del model, approx, posterior_samples, tau_samples
