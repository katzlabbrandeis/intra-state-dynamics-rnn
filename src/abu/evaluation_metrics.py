"""
Perform comparison of RNN-inferred rates against rolling window rates using various evaluation metrics.

Metrics included:
    - R2 Score
    - Bits per Spike
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
import sklearn.metrics as metrics

from core.config.roots_io import resolve_roots
from core.utils.read_parquets import read_parquet_files_into_dict
from core.pre_processing.RNNLatentprocessing import RNNLatentProcessor
from core.io.standard_paths import ProjectPaths


def load_rnn_data(
    parquet_dir: str | Path,
    data_type: str = 'latent',
    verbose: bool = False
) -> Dict[str, pl.DataFrame]:
    """
    Load RNN data (latents or firing rates) from a directory of parquet files.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing parquet files with RNN data
    data_type : str, optional
        Type of data to load. Either 'latent' or 'firing_rate'
    verbose : bool, optional
        Print additional information during loading

    Returns
    -------
    Dict[str, pl.DataFrame]
        Dictionary of dataset names and their corresponding DataFrames
    """
    if data_type not in ['latent', 'firing_rate']:
        raise ValueError("data_type must be either 'latent' or 'firing_rate'")

    data = read_parquet_files_into_dict(parquet_dir)

    if verbose:
        print(f"Loaded {len(data)} {data_type} datasets")
        for name, df in data.items():
            print(f"{name}: {df.shape}")

    return data


def compute_r2_score(
    true_rates: np.ndarray,
    predicted_rates: np.ndarray
) -> float:
    """
    Compute R2 (coefficient of determination) score between true and predicted rates.

    Parameters
    ----------
    true_rates : np.ndarray
        Ground truth firing rates
    predicted_rates : np.ndarray
        Predicted firing rates

    Returns
    -------
    float
        R2 score
    """
    return metrics.r2_score(true_rates, predicted_rates)


def compute_bits_per_spike(
    true_rates: np.ndarray,
    predicted_rates: np.ndarray
) -> float:
    """
    Compute bits per spike metric.

    Parameters
    ----------
    true_rates : np.ndarray
        Ground truth firing rates
    predicted_rates : np.ndarray
        Predicted firing rates

    Returns
    -------
    float
        Bits per spike metric
    """
    # Placeholder implementation - will need refinement
    mse = metrics.mean_squared_error(true_rates, predicted_rates)
    return -np.log2(mse) if mse > 0 else np.inf


def _resolve_rnn_parquet_path(repo_root: Path, relative_path: str) -> Path:
    """
    Resolve a parquet path relative to the repository root.
    
    Parameters
    ----------
    repo_root : Path
        Root directory of the repository
    relative_path : str
        Relative path to the parquet directory
    
    Returns
    -------
    Path
        Resolved absolute path to the parquet directory
    """
    # Remove the user-specific prefix and use the last part of the path
    path_parts = Path(relative_path).parts
    return repo_root / "output" / "intermediate_data" / path_parts[-2] / path_parts[-1]


def evaluate_rnn_performance(
    repo_root: Path,
    output_dir: Optional[str | Path] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate RNN performance across multiple metrics.

    Parameters
    ----------
    repo_root : Path
        Root directory of the repository
    output_dir : str or Path, optional
        Directory to save evaluation results
    verbose : bool, optional
        Print detailed information during evaluation

    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    # Resolve roots to get the RNN parquet paths
    roots = resolve_roots(repo_root)
    
    # Resolve paths relative to repository root
    true_rates_dir = _resolve_rnn_parquet_path(repo_root, roots.rnn_pred_fr_parquet_root)
    predicted_rates_dir = _resolve_rnn_parquet_path(repo_root, roots.rnn_pred_fr_parquet_root)

    true_rates = load_rnn_data(true_rates_dir, data_type='firing_rate', verbose=verbose)
    predicted_rates = load_rnn_data(predicted_rates_dir, data_type='firing_rate', verbose=verbose)

    performance_metrics = {}

    for dataset_name, true_df in true_rates.items():
        if dataset_name not in predicted_rates:
            print(f"Warning: No predicted rates for {dataset_name}")
            continue

        pred_df = predicted_rates[dataset_name]

        # Assuming neuron columns start with 'neuron_'
        true_neuron_cols = [col for col in true_df.columns if col.startswith('neuron_')]
        pred_neuron_cols = [col for col in pred_df.columns if col.startswith('neuron_')]

        # Ensure column names match
        if set(true_neuron_cols) != set(pred_neuron_cols):
            print(f"Warning: Neuron column mismatch for {dataset_name}")
            continue

        dataset_metrics = {}
        for col in true_neuron_cols:
            true_rates_col = true_df[col].to_numpy()
            pred_rates_col = pred_df[col].to_numpy()

            dataset_metrics[f'{col}_r2'] = compute_r2_score(true_rates_col, pred_rates_col)
            dataset_metrics[f'{col}_bits_per_spike'] = compute_bits_per_spike(true_rates_col, pred_rates_col)

        performance_metrics[dataset_name] = dataset_metrics

        if verbose:
            print(f"Performance for {dataset_name}:")
            for metric, value in dataset_metrics.items():
                print(f"  {metric}: {value}")

    if output_dir is not None:
        output_path = Path(output_dir) / "rnn_performance_metrics.json"
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)

    return performance_metrics
