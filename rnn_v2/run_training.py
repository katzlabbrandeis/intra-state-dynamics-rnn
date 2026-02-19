"""
Model training and logic. 
"""
import os
import json
import time
import numpy as np
import torch
from model import autoencoderRNN
from train import (
    train_model, compute_aic_bic,
    poisson_log_likelihood, count_parameters, MSELoss,
)


# ----------------------------------------------------------------
# Mode 1: Standard train/test split
# ----------------------------------------------------------------

def train_or_load(
        input_size,
        hidden_size,
        output_size,
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        device,
        criterion,
        train_steps,
        patience,
        lr=0.001,
        rnn_layers=2,
        dropout=0.2,
        retrain=True,
        model_save_path=None,
        artifacts_dir=None,
        taste_ind=None,
        ):
    """
    Train a new model or load an existing one.
    Computes AIC/BIC in both cases.

    Returns:
        net: trained/loaded model
        loss: list of per-step training losses
        cross_val_loss: dict of step -> test loss
        info_criteria: dict with AIC/BIC stats
    """
    net = autoencoderRNN(
        input_size, hidden_size, output_size,
        rnn_layers=rnn_layers, dropout=dropout
    )
    net.to(device)

    if retrain or not os.path.exists(model_save_path):
        net, loss, cross_val_loss, info_criteria = train_model(
            net, train_inputs, train_labels, output_size,
            train_steps=train_steps, lr=lr,
            criterion=criterion,
            test_inputs=test_inputs, test_labels=test_labels,
            patience=patience,
        )
        # Save model and loss histories
        if model_save_path:
            torch.save(net, model_save_path)
        if artifacts_dir and taste_ind is not None:
            with open(os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(loss, f)
            with open(os.path.join(artifacts_dir, f'cross_val_loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(cross_val_loss, f)
    else:
        net = torch.load(model_save_path)
        info_criteria = compute_aic_bic(
            net, test_inputs.to(device), test_labels.to(device)
        )
        info_criteria['eval_set'] = 'test'
        # Load saved loss histories
        loss_path = os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json')
        cv_path = os.path.join(artifacts_dir, f'cross_val_loss_taste_{taste_ind}.json')
        loss = json.load(open(loss_path)) if os.path.exists(loss_path) else []
        cross_val_loss = json.load(open(cv_path)) if os.path.exists(cv_path) else {}

    # Save info criteria
    if artifacts_dir and taste_ind is not None:
        with open(os.path.join(artifacts_dir, f'info_criteria_taste_{taste_ind}.json'), 'w') as f:
            json.dump(info_criteria, f, indent=2)

    return net, loss, cross_val_loss, info_criteria



# ----------------------------------------------------------------
# Mode 2: LOO evaluation + final retrain on all data
# ----------------------------------------------------------------


def loo_then_train(
        inputs_tensor,
        labels_tensor,
        input_size,
        hidden_size,
        output_size,
        device,
        criterion,
        train_steps,
        patience,
        lr=0.001,
        rnn_layers=2,
        dropout=0.2,
        retrain=True,
        model_save_path=None,
        artifacts_dir=None,
        taste_ind=None,
        verbose=True,
        loo_train_steps=None,
        loo_patience=None,
        ):
    """
    Phase 1: LOO cross-validation to get robust AIC/BIC.
             Uses loo_train_steps/loo_patience if provided (faster folds).
    Phase 2: Retrain a single model on ALL trials for downstream use.
             Uses full train_steps/patience.

    Returns:
        net: final model trained on all data
        loss: training loss history (from final retrain)
        cross_val_loss: {} (no held-out set for final model)
        info_criteria: dict with LOO-based AIC/BIC + per-trial LL
    """
    n_trials = inputs_tensor.shape[1]

    # LOO fold settings — default to full settings if not specified
    fold_train_steps = loo_train_steps if loo_train_steps is not None else train_steps
    fold_patience = loo_patience if loo_patience is not None else patience

    # --- Check for cached LOO results ---
    loo_cache_path = None
    if artifacts_dir and taste_ind is not None:
        loo_cache_path = os.path.join(
            artifacts_dir, f'loo_results_taste_{taste_ind}.json'
        )

    run_loo = retrain or (loo_cache_path is None) or not os.path.exists(loo_cache_path)

    if run_loo:
        # ============================================================
        # Phase 1: LOO cross-validation
        # ============================================================
        print(f"  Running LOO ({n_trials} folds)...")

        # Parameter count (same for all folds)
        dummy_net = autoencoderRNN(
            input_size, hidden_size, output_size,
            rnn_layers=rnn_layers, dropout=dropout
        )
        n_params = count_parameters(dummy_net)
        del dummy_net

        per_trial_ll = []
        per_trial_train_loss = []
        per_fold_loss_history = []
        per_fold_n_steps = []
        total_start = time.time()

        for j in range(n_trials):
            fold_start = time.time()

            # Hold out trial j
            train_mask = [i for i in range(n_trials) if i != j]
            fold_train_inputs = inputs_tensor[:, train_mask].to(device)
            fold_train_labels = labels_tensor[:, train_mask].to(device)
            fold_test_inputs = inputs_tensor[:, j:j+1].to(device)
            fold_test_labels = labels_tensor[:, j:j+1].to(device)

            # Fresh model
            fold_net = autoencoderRNN(
                input_size, hidden_size, output_size,
                rnn_layers=rnn_layers, dropout=dropout
            )
            fold_net.to(device)

            # Train (quiet mode — LOO folds don't print per-step output)
            fold_net, fold_loss, _, _ = train_model(
                fold_net, fold_train_inputs, fold_train_labels, output_size,
                train_steps=fold_train_steps, lr=lr,
                criterion=criterion,
                test_inputs=fold_test_inputs, test_labels=fold_test_labels,
                patience=fold_patience,
                quiet=True,
            )

            # Evaluate held-out trial
            fold_net.eval()
            with torch.no_grad():
                pred, _ = fold_net(fold_test_inputs)
                pred = torch.clamp(pred, min=1e-8)
            ll = poisson_log_likelihood(pred, fold_test_labels)
            per_trial_ll.append(ll)
            per_trial_train_loss.append(fold_loss[-1])
            per_fold_loss_history.append(fold_loss)
            per_fold_n_steps.append(len(fold_loss))

            if verbose:
                elapsed = time.time() - fold_start
                print(f"    Fold {j+1}/{n_trials} | "
                      f"LL: {ll:.2f} | "
                      f"Final loss: {fold_loss[-1]:.4f} | "
                      f"{elapsed:.1f}s")

            del fold_net

        # Aggregate LOO results
        total_ll = sum(per_trial_ll)
        n_obs = labels_tensor.numel()
        aic = 2 * n_params - 2 * total_ll
        bic = n_params * np.log(n_obs) - 2 * total_ll
        total_elapsed = time.time() - total_start

        info_criteria = dict(
            aic=aic,
            bic=bic,
            log_likelihood=total_ll,
            per_trial_ll=per_trial_ll,
            per_trial_train_loss=per_trial_train_loss,
            per_fold_loss_history=per_fold_loss_history,
            per_fold_n_steps=per_fold_n_steps,
            n_params=n_params,
            n_observations=n_obs,
            n_trials=n_trials,
            hidden_size=hidden_size,
            eval_set='loo',
            total_time_s=total_elapsed,
        )

        print(f"\n  --- LOO Summary (taste {taste_ind}, hidden={hidden_size}) ---")
        print(f"    Folds:         {n_trials}")
        print(f"    Fold settings: train_steps={fold_train_steps}, patience={fold_patience}")
        print(f"    Params:        {n_params}")
        print(f"    Total LL:      {total_ll:.2f}")
        print(f"    Mean LL/trial: {np.mean(per_trial_ll):.2f} "
              f"+/- {np.std(per_trial_ll):.2f}")
        print(f"    AIC:           {aic:.2f}")
        print(f"    BIC:           {bic:.2f}")
        print(f"    Time:          {total_elapsed:.1f}s")

        # Cache LOO results
        if loo_cache_path:
            save_dict = {k: v for k, v in info_criteria.items()
                         if not isinstance(v, list)}
            save_dict['per_trial_ll'] = per_trial_ll
            save_dict['per_trial_train_loss'] = per_trial_train_loss
            with open(loo_cache_path, 'w') as f:
                json.dump(save_dict, f, indent=2)

    else:
        print(f"  Loading cached LOO results from {loo_cache_path}")
        with open(loo_cache_path, 'r') as f:
            info_criteria = json.load(f)

    # ============================================================
    # Phase 2: Retrain on ALL trials for downstream outputs
    # ============================================================
    should_retrain_final = retrain or (
        model_save_path is not None and not os.path.exists(model_save_path)
    )

    if should_retrain_final:
        print(f"  Retraining final model on all {n_trials} trials...")
        net = autoencoderRNN(
            input_size, hidden_size, output_size,
            rnn_layers=rnn_layers, dropout=dropout
        )
        net.to(device)

        net, loss, _, _ = train_model(
            net,
            inputs_tensor.to(device),
            labels_tensor.to(device),
            output_size,
            train_steps=train_steps, lr=lr,
            criterion=criterion,
            patience=patience,
        )

        if model_save_path:
            torch.save(net, model_save_path)
        if artifacts_dir and taste_ind is not None:
            with open(os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(loss, f)
    else:
        print(f"  Loading final model from {model_save_path}")
        net = torch.load(model_save_path)
        loss_path = os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json')
        loss = json.load(open(loss_path)) if os.path.exists(loss_path) else []

    cross_val_loss = {}
    return net, loss, cross_val_loss, info_criteria

# ----------------------------------------------------------------
# Shared: forward pass for predictions
# ----------------------------------------------------------------

def run_prediction(net, inputs_tensor, device):
    """
    Forward pass through trained model.

    Args:
        net: trained model
        inputs_tensor: (time, trials, input_size)
        device: torch device

    Returns:
        outs: numpy array (time, trials, output_size)
        latent_outs: numpy array (time, trials, hidden_size)
    """
    net.eval()
    with torch.no_grad():
        outs, latent_outs = net(inputs_tensor.to(device))
    outs = outs.cpu().numpy()
    latent_outs = latent_outs.cpu().numpy()
    return outs, latent_outs