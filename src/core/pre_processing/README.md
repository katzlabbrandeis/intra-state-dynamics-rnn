README for `src/core/pre_processing/README.md`.
---
# Preprocessing

Utilities and pipelines for turning raw model outputs + spike metadata into analysis-ready tables. The primary entrypoint is **`RNNLatentProcessor`**.

## Inputs

* **Parquet directory** (`parquet_dir`)
  Per-timestep tables with either:

  * `latent_dim_1, latent_dim_2, …` **or** `neuron_1, neuron_2, …`
  * plus meta columns: `taste`, `trial`, `time` (ms)

* **Spike / metadata paths**

  * `npz_path`: NPZ with spike arrays (for changepoint extraction)
  * `info_path`: dataset info files (taste mapping, etc.)
  * `pkl_path`: pickled changepoint outputs (model-derived)
  * `taste_replacements`: mapping to normalize taste labels

* **Config**
  `bin_size_ms`, `start_time_ms`, `max_time_ms`, `warp_length=None`, `variance_threshold=95.0`, `save_dir=None`

> **Layout assumption:** The upstream exporters guarantee the expected array/time orientation. This module does **not** transpose to “fix” TIME↔CHANNEL; if something looks off, check the exporter first.

## What it produces (in memory)

All outputs are dictionaries keyed by dataset, with **Polars DataFrames** as values.

* **Epochs (unwarped / warped)**
  Keys:

  * `<core>_latent_unwarped` or `<core>_firing_unwarped`
  * `<core>_latent_warped` or `<core>_firing_warped` (if `warp_length` set)

  Schema (per row = one time bin):

  * signal columns: `latent_dim_*` **or** `neuron_*`
  * meta: `taste` (int), `trial` (int), `changepoint` (epoch id), `time` (ms)

* **PCA (per taste × changepoint, split back by trial)**

  * **Thresholded:** `robust_pca_<thr>_unwarped` (and `_warped` if enabled)
    Columns: `PC_1..PC_k` + `explained_variance_PC_1..PC_k` + meta (`taste`, `changepoint`, `trial`, `time`)
  * **Full:** `robust_pca_full_unwarped` (and `_warped`)
    Columns: `PC_1..PC_m` + `explained_variance_PC_*` + meta

* **Derivatives (optional)**

  * First: `first_derivatives_<thr|full>_unwarped` (and `_warped` if enabled)
  * Second: `second_derivatives_<thr|full>_unwarped` (and `_warped`)
    Schema mirrors the PCA source dict; values are time-derivatives of signal columns.

* **Changepoints**
  `changepoints_dict[<core>] -> array/list` per taste × trial (ms boundaries)


## Output schemas (Polars dataframes)

All tables are **per-dataset** DataFrames stored in dictionaries (see processor docs). Unless noted, numeric columns are `Float64` and meta columns are `Int64`.

Note: If `save_dir` is set and `save_outputs=True`, parquet files contain exactly 1 polars dataframe per dataset, and are named as such for whatever type.

### 1) Epochs (unwarped / warped)
> Name `epochs` is in refrence to **unprocessed** data (eg. no PCA, no derivs) output which has been transformed to the correct data format. Usable "raw" RNN latents will come from the `epoch` name. This is unfortunately a legacy name, which if eliminated, may break something.

**Key names**

* Unwarped: `<core>_latent_unwarped` **or** `<core>_firing_unwarped`
* Warped (if `warp_length` set): `<core>_latent_warped` **or** `<core>_firing_warped`

**Columns**

* **Signal** (one of):

  * `latent_dim_1 … latent_dim_K` (`Float64`)
  * `neuron_1 … neuron_M` (`Float64`)
* **Meta**

  * `taste` (`Int64`) – taste index
  * `trial` (`Int64`) – trial index
  * `changepoint` (`Int64`) – epoch id within a trial (0..N)
  * `time` (`Int64`) – ms since trial start

**Row meaning**

* One **row = one time bin** within a `(taste, trial, changepoint)` segment.
* Within each segment, `time` is monotonic, step ≈ `bin_size_ms`.
* Warped outputs have a fixed rows-per-segment: `warp_length / bin_size_ms`.

---

### 2) PCA (thresholded & Full)

**Key name**

* `robust_pca_<thr>_unwarped` (and `…_warped` if enabled), where `<thr>` is the integer variance threshold (e.g., `95`).
* if there is no integer variance threshold, consider this to be PCA (full). Schemas are identical, save variance threshold.
**Columns**

* **Scores**: `PC_1 … PC_k` (`Float64`) – k chosen so cumulative explained variance ≥ `<thr>%`
* **Explained variance (per component)**:
  `explained_variance_PC_1 … explained_variance_PC_k` (`Float64`)
  (constant within each group; repeated for convenience)
* **Meta**:

  * `taste` (`Int64`)
  * `changepoint` (`Int64`)
  * `trial` (`Int64`)
  * `time` (`Int64`)

**Row meaning**

* One row per original **time bin** after grouping by `(taste, changepoint)` and splitting back by `trial`.

---

### 4) Derivatives (optional)

**Key names**

* First derivative: `first_derivatives_<thr|full>_unwarped` (and `…_warped`)
* Second derivative: `second_derivatives_<thr|full>_unwarped` (and `…_warped`)

**Columns**

* Same layout as their **source** tables (thresholded/full PCA).
  All **non-meta** numeric columns are differentiated along **time**:

  * For PCA sources, derivatives are computed for `PC_*` **and** `explained_variance_PC_*` columns (the latter are typically \~0 across time since EV is constant within a group).

**Meta**

* `taste`, `changepoint`, `trial`, `time` unchanged.

---

### 5) Changepoints (reference structure)

Not a DataFrame, but used to build epochs.

```python
changepoints_dict[core]  # → array/list indexed as [taste][trial] -> sequence of ms boundaries
```

---

### Invariants

* Each DataFrame can be partitioned by `(taste, trial, changepoint)`.
* Within a partition, `time` is ascending; step is \~`bin_size_ms` (warped: exactly `warp_length / bin_size_ms` rows).
* Signal columns are mutually exclusive: you’ll have **either** `latent_dim_*` **or** `neuron_*` in a given table.


## Optional on-disk outputs

If `save_dir` is set and `save_outputs=True`, parquet files are written as:

```
<save_dir>/
  raw_output_unwarped/ | raw_output_warped/
  robust_pca_<thr>_unwarped/ | robust_pca_<thr>_warped/
  robust_pca_full_unwarped/  | robust_pca_full_warped/
  first_derivatives_<thr>_unwarped/ | ..._warped/
  second_derivatives_<thr>_unwarped/ | ..._warped/
```

File names follow:

```
<core>_<type>_<block>_<unwarped|warped>.parquet
# where:
#   <type>  = rnn_latent | rnn_pred_fr | rr_fr
#   <block> = raw_output | robust_pca_<thr> | robust_pca_full | first_derivatives_<thr> | second_derivatives_<thr>
```

## Quick start

```python
from core.pre_processing.rnn_latent_processor import RNNLatentProcessor

proc = RNNLatentProcessor(
    parquet_dir="data/parquets",
    npz_path="data/spikes.npz",
    info_path="data/info/",
    pkl_path="data/changepoints/",
    taste_replacements={"NaCl": "salt"},
    bin_size_ms=25,
    start_time_ms=1500,
    max_time_ms=4500,
    warp_length=2000,          # optional: enables *_warped outputs
    variance_threshold=95.0,
    save_dir="output/intermediate_data/preproc"  # optional
)

# end-to-end:
(
    epochs_unw,
    pca95_unw,
    pca_full_unw,
    fd_unw,
    sd_unw,
    epochs_w,
    pca95_w,
    pca_full_w,
    fd_w,
    sd_w,
) = proc.full_pipeline(
    compute_first_derivative=True,
    compute_second_derivative=False,
    derivative_source="threshold",
    return_derivatives=True,
    save_outputs=True
)

# Access dicts directly later:
epochs = proc.epoch_dataframes_dict_unwarped
pca95  = getattr(proc, "robust_pca_95_unwarped")
```

## Dependencies

`numpy`, `polars`, `scikit-learn`, `scipy`, plus local readers: `extract_npz`, `find_extract_info`, `read_parquets`, `unpkl_generator`.
