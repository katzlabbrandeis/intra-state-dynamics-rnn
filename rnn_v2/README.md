# blechRNN — Refactored

NOTE: this code has been partially ai-generated, subject to full human review by yours truly (Vincent).
This is a WIP, as LOO will ultimately be used to optimize this model further.

Autoencoder-RNN for predicting firing rates from binned spike trains. Encoder compresses inputs to a latent space, an RNN learns temporal dynamics in that space, and a decoder projects back to firing rate predictions.

## Files

| File | Purpose |
|---|---|
| `run_rnn.py` | Main script. Loops over datasets and tastes, calls everything else. |
| `config_loader.py` | Parses `blechrnn_config.json`, sets up paths, parameters, and loss function. |
| `preprocessing.py` | Spike binning, z-scoring, PCA, stimulus/trial context concatenation, tensor creation, train/test splitting. |
| `run_training.py` | Two training modes: `train_or_load` (standard split) and `loo_then_train` (LOO cross-validation + final retrain). |
| `training.py` | Core `train_model` loop, loss functions (`MSELoss`, `smooth_MSELoss`), and AIC/BIC utilities (`compute_aic_bic`, `poisson_log_likelihood`, `count_parameters`). |
| `model.py` | Network architectures: `CTRNN`, `CTRNN_plus_output`, `autoencoderRNN`. |
| `postprocessing.py` | Inverse PCA and inverse z-scoring to reconstruct predicted firing rates back into neuron space. |
| `visualizations.py` | All plotting: loss curves, firing rate heatmaps, latent factors, per-neuron rasters, AIC/BIC summaries, LOO diagnostics. |
| `save_outputs.py` | Saves predicted firing rates and latent vectors to HDF5 and Parquet. |

## Config parameters

Set in `blechrnn_config.json` under `"parameters"`:

```json
{
    "train_steps": 12000,
    "hidden_size": 8,
    "bin_size": 25,
    "train_test_split": 0.75,
    "use_pca": true,
    "retrain": true,
    "time_lims": [1500, 7000],
    "patience": 12000,
    "loss_name": "mse",
    "validation_mode": "split",
    "loo_train_steps": 3000,
    "loo_patience": 15
}
```

- `validation_mode`: `"split"` (default) or `"loo"` — see below.
- `loo_train_steps` / `loo_patience`: Shorter training settings used only for LOO folds, not the final model. Reduces LOO runtime without affecting the quality of the final model.

## Model evaluation metrics

### n_params
Total number of trainable parameters in the network (weights + biases). Computed as `sum(p.numel() for p in net.parameters() if p.requires_grad)`. This is `k` in the AIC/BIC formulas. A model with more parameters can fit data better but risks overfitting — AIC and BIC penalize this.

### n_observations
Total number of individual data points the model is evaluated on. For a tensor of shape `(time_steps, trials, neurons)`, this is `time_steps × trials × neurons`. This is `n` in the BIC formula.

### Log-likelihood (Poisson)
How well the model's predicted firing rates explain the observed spike counts, assuming spikes follow a Poisson process. For each time bin, neuron, and trial:

```
LL = sum( y * log(r) - r - log(y!) )
```

where `y` = observed count and `r` = predicted rate. Higher (less negative) is better. A model that predicts rates close to the true underlying rates will have a higher log-likelihood.

### AIC (Akaike Information Criterion)
```
AIC = 2k - 2LL
```
Balances model fit (log-likelihood) against complexity (parameter count). Lower is better. Penalizes each additional parameter by 2, so a larger model needs to improve LL enough to justify its extra parameters. Tends to favor slightly more complex models compared to BIC.

### BIC (Bayesian Information Criterion)
```
BIC = k * ln(n) - 2LL
```
Same idea as AIC but the complexity penalty scales with `ln(n_observations)`. For large datasets (many time bins × trials × neurons), BIC penalizes complexity more heavily than AIC. Lower is better. Tends to favor simpler models.

### Interpreting AIC/BIC
- Compare models trained on the **same data** with the **same validation scheme**.
- Only the **difference** between models matters, not the absolute values.
- If AIC and BIC disagree (AIC picks a larger model, BIC picks a smaller one), it's a judgment call — BIC is more conservative.

## Validation modes

### `"split"` — Train/test split (default)

Randomly holds out 25% of trials as a test set. Trains on the remaining 75%. Computes AIC/BIC on the held-out test set. The saved model has only ever seen 75% of the data.

**Pros:** Fast — one training run per taste.
**Cons:** With small trial counts (~30 trials, 7–8 in the test set), results are noisy and depend on which trials happen to land in the test set. Run it again with a different random seed and you may get different AIC/BIC values. The saved model is also suboptimal because it never trained on the held-out trials.

### `"loo"` — Leave-one-out cross-validation

Two phases:

**Phase 1 (evaluation):** For each of the N trials, hold it out, train on the other N-1, and compute the Poisson log-likelihood on the single held-out trial. Repeat for all N trials. Sum the per-trial log-likelihoods to get a total LOO log-likelihood, then compute AIC/BIC from that. Every trial is evaluated exactly once on a model that never saw it. No randomness in the split — deterministic up to training stochasticity.

**Phase 2 (final model):** After LOO evaluation is complete, train a single model on ALL N trials. This is the model that gets saved and used for all downstream outputs (predicted firing rates, latent trajectories, plots, HDF5, Parquet).

The logic: LOO answers "is this architecture good?" (Phase 1), then "give me the best possible model from this architecture" (Phase 2). These are separate questions — once you've validated the architecture, there's no reason to withhold data from the final model.

**Pros:** Robust AIC/BIC with no sensitivity to split randomness. The final model uses all available data. Well-suited for small trial counts.
**Cons:** Trains N models per taste (30 folds × 4 tastes = 120 training runs per dataset). Mitigated by using `loo_train_steps` and `loo_patience` for faster folds, plus `quiet=True` to suppress fold output.

### LOO diagnostics plot

When running in LOO mode, a 6-panel diagnostic figure is generated per taste:

1. **Per-trial held-out LL** — which trials the model struggles with (outliers flagged in red).
2. **Per-fold final training loss** — consistency of convergence across folds.
3. **Fold loss curves overlaid** — overfitting diagnostic. Diverging or rising curves indicate overfitting.
4. **Train loss vs held-out LL scatter** — whether good training loss predicts good generalization. Positive correlation is healthy; no correlation suggests memorization.
5. **Convergence steps histogram** — whether folds are hitting the step ceiling or converging early.
6. **Summary statistics** — AIC, BIC, LL stats, timing, all in one place.
