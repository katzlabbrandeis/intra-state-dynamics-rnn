"""
Part of a concerted effort to clean this script up in general to make it that much
easier to do model comparisons, etc

run_rnn.py — Main orchestration script.

Loops over H5 datasets and tastes, calling modular functions for:
    - Config loading
    - Preprocessing
    - Training / loading
    - Postprocessing (inverse PCA/scaling)
    - Visualization
    - Saving outputs

NOTE: On the params json:
Config parameter "validation_mode":
    - "split" (default): standard train/test split
    - "loo": leave-one-out CV for AIC/BIC, then retrain on all trials

"""
from ephys_data import ephys_data
import os

import numpy as np
import torch
from config_loader import load_config
from postprocessing import reconstruct_firing
from preprocessing import preprocess_taste, train_test_split_trials
from run_training import loo_then_train, run_prediction, train_or_load
from save_outputs import save_firing_parquet, save_latents_parquet, save_to_hdf5
from visualizations import (
    plot_aic_bic_summary,
    plot_firing_overview,
    plot_individual_neurons,
    plot_inputs,
    plot_latent_factors,
    plot_loo_diagnostics,
    plot_loss_curves,
    plot_mean_firing,
    plot_mean_neurons_across_tastes,
    plot_pred_vs_true_neurons,
    plot_trial_latents,
)

# load in the configs:
config_path = '/home/vincent/Senior thesis work/blechRNN-master/config/blechrnn_config.json'

config, paths, params, criterion = load_config(config_path)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
validation_mode = params.get('validation_mode', 'split')  # note 'split' is the default if nothing is specified.
print(f"Validation mode: {validation_mode}")
# ----------------------------------------------------------------
# Loop over datasets
# ----------------------------------------------------------------
for subdir in sorted(os.listdir(paths['h5_dir'])):
    full_subdir_path = os.path.join(paths['h5_dir'], subdir)
    if not os.path.isdir(full_subdir_path):
        continue

    h5_files = [f for f in os.listdir(full_subdir_path) if f.endswith(".h5")]
    if len(h5_files) != 1:
        print(f"Skipping {subdir} — expected 1 .h5 file, found {len(h5_files)}")
        continue

    dataset_name = os.path.splitext(h5_files[0])[0]
    print(f"\nProcessing: {dataset_name}")

    # --- Output directories ---
    output_path = os.path.join(paths['output_base_dir'], dataset_name)
    plots_dir = os.path.join(output_path, 'plots')
    artifacts_dir = os.path.join(output_path, 'artifacts')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # --- Load spikes ---
    data = ephys_data(full_subdir_path)
    data.get_spikes()
    spike_array = np.stack(data.spikes)

    # --- Accumulators ---
    pred_firing_list = []
    latent_out_list = []
    binned_spikes_list = []
    conv_rate_list = []
    conv_x_list = []
    info_criteria_all = {}

    # ----------------------------------------------------------------
    # Loop over tastes
    # ----------------------------------------------------------------
    for taste_ind, taste_spikes in enumerate(spike_array):
        print(f"\n  Taste {taste_ind}")

        # --- Time slice ---
        taste_spikes = taste_spikes[..., params['time_lims'][0]:params['time_lims'][1]]

        # --- Preprocess ---
        prep = preprocess_taste(
            taste_spikes,
            bin_size=params['bin_size'],
            stim_time_val=paths['stim_time_val'],
            use_pca=params['use_pca'],
        )

        # --- Debug ---
        print(f"    Input shape: {prep['inputs_plus_context'].shape}")
        print(f"    input_size={prep['input_size']}, output_size={prep['output_size']}")

        # --- Plot inputs ---
        plot_inputs(
            prep['inputs_plus_context'], dataset_name, taste_ind, plots_dir
        )

        # --- Model path ---
        model_name = (f'taste_{taste_ind}_hidden_{params["hidden_size"]}'
                      f'_loss_{params["loss_name"]}')
        model_save_path = os.path.join(artifacts_dir, f'{model_name}.pt')

        # ==============================================================
        # Branch on validation mode
        # ==============================================================
        if validation_mode == 'loo':
            # LOO: no manual split needed — loo_then_train handles it
            net, loss, cross_val_loss, info_criteria = loo_then_train(
                inputs_tensor=prep['inputs_tensor'],
                labels_tensor=prep['labels_tensor'],
                input_size=prep['input_size'],
                hidden_size=params['hidden_size'],
                output_size=prep['output_size'],
                device=device,
                criterion=criterion,
                train_steps=params['train_steps'],
                patience=params['patience'],
                retrain=params['retrain'],
                model_save_path=model_save_path,
                artifacts_dir=artifacts_dir,
                taste_ind=taste_ind,
                loo_train_steps=params.get('loo_train_steps'),
                loo_patience=params.get('loo_patience')
            )

        else:
            # Standard split
            (train_inputs, train_labels,
             test_inputs, test_labels,
             train_inds, test_inds) = train_test_split_trials(
                prep['inputs_tensor'], prep['labels_tensor'],
                split_ratio=params['train_test_split'],
            )

            net, loss, cross_val_loss, info_criteria = train_or_load(
                input_size=prep['input_size'],
                hidden_size=params['hidden_size'],
                output_size=prep['output_size'],
                train_inputs=train_inputs.to(device),
                train_labels=train_labels.to(device),
                test_inputs=test_inputs.to(device),
                test_labels=test_labels.to(device),
                device=device,
                criterion=criterion,
                train_steps=params['train_steps'],
                patience=params['patience'],
                retrain=params['retrain'],
                model_save_path=model_save_path,
                artifacts_dir=artifacts_dir,
                taste_ind=taste_ind,
            )

        info_criteria_all[taste_ind] = info_criteria

        # --- LOO diagnostics (only in LOO mode) ---
        if validation_mode == 'loo':
            plot_loo_diagnostics(info_criteria, dataset_name, taste_ind, plots_dir)

        # --- Predict on full data ---
        outs, latent_outs = run_prediction(
            net, prep['inputs_tensor'], device
        )
        latent_out_list.append(latent_outs)

        # --- Reconstruct firing rates ---
        pred_firing = reconstruct_firing(
            outs,
            scaler=prep['scaler'],
            pca_obj=prep['pca_obj'],
            num_neurons=prep['num_neurons'],
            use_pca=params['use_pca'],
        )
        pred_firing_list.append(pred_firing)
        binned_spikes_list.append(prep['binned_spikes'])

        # --- Convolved firing rate (for comparison plots) ---
        conv_kern = np.ones(250) / 250
        conv_rate = np.apply_along_axis(
            lambda m: np.convolve(m, conv_kern, mode='valid'),
            axis=-1, arr=taste_spikes
        ) * params['bin_size']
        conv_x = np.convolve(
            np.arange(taste_spikes.shape[-1]), conv_kern, mode='valid'
        )
        conv_rate_list.append(conv_rate)
        conv_x_list.append(conv_x)

        # --- Per-taste plots ---
        plot_loss_curves(loss, cross_val_loss, dataset_name, taste_ind, plots_dir)
        plot_firing_overview(pred_firing, prep['binned_spikes'],
                             dataset_name, taste_ind, plots_dir)
        plot_mean_firing(pred_firing, prep['binned_spikes'],
                         dataset_name, taste_ind, plots_dir)
        plot_latent_factors(latent_outs, dataset_name, taste_ind, plots_dir)
        plot_trial_latents(latent_outs, dataset_name, taste_ind, plots_dir)
        plot_individual_neurons(
            taste_spikes, prep['binned_spikes'], pred_firing,
            conv_rate, conv_x, params['bin_size'],
            paths['stim_time_val'], dataset_name, taste_ind, plots_dir
        )

    # ----------------------------------------------------------------
    # Post-taste-loop: cross-taste plots and saving
    # ----------------------------------------------------------------
    plot_mean_neurons_across_tastes(
        spike_array, pred_firing_list, binned_spikes_list,
        conv_rate_list, conv_x_list, params['bin_size'],
        paths['stim_time_val'], params['time_lims'],
        dataset_name, plots_dir
    )
    plot_pred_vs_true_neurons(pred_firing_list, binned_spikes_list, plots_dir)
    plot_aic_bic_summary(info_criteria_all, dataset_name, plots_dir)

    # --- Save outputs ---
    save_to_hdf5(
        data.hdf5_path, pred_firing_list, latent_out_list,
        params['bin_size']
    )
    save_latents_parquet(
        latent_out_list, dataset_name, artifacts_dir,
        paths['pred_lat_dir']
    )
    save_firing_parquet(
        pred_firing_list, dataset_name, artifacts_dir,
        paths['pred_fr_dir']
    )

    print(f"\n  Done: {dataset_name}")
