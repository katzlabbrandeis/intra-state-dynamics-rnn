Blech_intra_state_dynamics README

Output by: Vincent Calia-Bogan
This repo is a work in progress.


# Initialization (one-time) & Running

This repo uses a `src/` layout. Do a **one-time bootstrap per Conda env** so Python can import `core.*` everywhere (scripts, notebooks, CLI).

> **Note on `bid` CLI:** It’s provided for convenience and a clean UX. Consider it largely a tech-demo path; if it misbehaves on your setup, use `run.py` or direct scripts.
> In the future, `bid` in CLI can be used to run all pre-processing steps automatically.

## 1) Create & activate the Conda env

```bash
# from the repo root
conda env create -f environment.yml          # uses the name inside the file
# or, if you want to name it:
# conda env create -f environment.yml -n blech
conda activate <your-env-name>
```

## 2) Bootstrap imports (one-time per env)

```bash
python bootstrap.py
```

This writes a small `.pth` file in your **active env** that points to `<repo>/src` and verifies `import core`.

> If you later **move/rename** the repo or switch to a **new env**, run `python bootstrap.py` again.
> To undo: delete the printed `.../site-packages/blech_src.pth` file.

## 3) Sanity check (optional)

```bash
python -c "import core; print('core OK at:', core.__file__)"
# verbose:
# python -c "import core, sys; print('OK:', core.__file__); print('src on path?', any(p.endswith('/src') for p in sys.path))"
```

You should see a path under `<repo>/src/core`.

---

## Running things

# Pre-processing via CLI


## Pre-processing orchestration

Run the H5 input pipeline from the CLI. This will:

1) **Extract spike trains** from `.h5` into `.npz` under `output/intermediate_data/spike_trains_npz/`
2) **Copy `.info` files** into `output/intermediate_data/info_files/`
3) **Read changepoints** from **RAW `.pkl`** (authoritative, read-only). No caching is written by default.

### Basic usage

Zero-install (after `bootstrap.py`):

```
python run.py preprocess_h5_input first-input \
  --h5-root "/path/to/spikesorting" \
  --pkl-root "/path/to/raw_pkl_dir" \
  --dry-run
````

Installed CLI (`bid`), after `pip install -e .`:

```bash
bid preprocess_h5_input first-input \
  --h5-root "/path/to/spikesorting" \
  --pkl-root "/path/to/raw_pkl_dir" \
  --force
```

Flags:

* `--spike-trains-path` (default `/spike_trains`) – HDF5 group where spike arrays live
* `--dry-run` – show actions without writing files
* `--force` – re-run steps even if outputs already exist

### Where outputs go

> By default the pipeline caches per-dataset changepoints (extracted from RAW `.pkl`) under:

`output/intermediate_data/pkl_cache/<npz_stem>_changepoints.pkl`

> RAW .pkl at --pkl-root remains the authoritative source; the cache is derived and disposable.

```
output/intermediate_data/
  ├─ spike_trains_npz/     # H5→NPZ artifacts
  ├─ info_files/           # copied .info files
  └─ pkl_cache/            # (reserved; will cache by defalt)
```

Changepoint (pkl file) Caching is **on by default**. You can disable it:

```bash
bid preprocess_h5_input first-input \
  --h5-root "/path/to/spikesorting" \
  --pkl-root "/path/to/raw_pkl_dir" \
  --no-cache-changepoints
```

### Parameters & labels

Human-friendly names (taste replacements) and epoch labels are stored in:

```
src/core/config/pipeline_params.json
```

On first run, the file is created with defaults:

```json
{
  "taste_replacements": {
    "nacl": "NaCl",
    "suc": "Sucrose",
    "ca":  "Citric Acid",
    "qhcl":"Quinine"
  },
  "epoch_labels": [
    "Identification",
    "Palatability",
    "Decision",
    "2000 ms Post-Stimulus"
  ]
}

> more params are to be added in the future, as this is a WIP.

```

You can edit this JSON; the orchestrator loads it automatically.

> **Note:** RAW `.pkl` under `--pkl-root` remains the **authoritative** source for changepoints. The pipeline reads from it and does not modify or cache changepoints unless you add such a step later.

### Option A — Zero-install (recommended for development)

```bash
python run.py --help
python run.py preprocess_dry --dry-run
```
> preprocess_dry will ensure that all imports work. Please understand that this will not actually process any such data.

To init the intermediate files off of a given H5 dir:

```
bid preprocess first-input --h5-root "/path/to/h5/files" \
  --pkl-root "/path/to/pkl files" --force
```

`run.py` temporarily adds `<repo>/src` to `sys.path` **for this process only** and dispatches to the orchestration CLI.

### Option B — Installed CLI (`bid`)

Add (or ensure) this in `pyproject.toml`:

```toml
[project.scripts]
bid = "core.scripts.orchestrate:main"
```

Install in the active env, then use `bid`:

```bash
python -m pip install -e .
bid --help
bid preprocess_dry --dry-run
```

> If `bid` isn’t found, you likely installed to a different env: check `which python` and `which bid`.

---

## What `bid` does today

* **Installed `bid` (entry point):**
  When you run `bid …` after `pip install -e .`, it simply dispatches to `core.scripts.orchestrate:main`. It **does not** modify `sys.path` or your environment—imports work because the package is installed (or you ran `bootstrap.py` earlier).

* **Dev wrapper (`bin/bid`) and `run.py`:**
  These **do** add `<repo>/src` to the import path **for that process only**, so you can run without installing:

  * `./bin/bid …` exports `PYTHONPATH=<repo>/src:$PYTHONPATH` then runs `python -m core.scripts.orchestrate …`.
  * `python run.py …` inserts `<repo>/src` into `sys.path` then calls the same CLI.

* **Current CLI behavior:**
  The CLI is a thin orchestrator. It supports `--help` and the initial subcommands (e.g., `preprocess --dry-run`) and will grow as pipelines are added.
  Note that the CLI is not intended for more active development: at most, it will only ever auto-execute pre-processing and generate intermediate files.
  The extent of CLI functionality is TBD: as I have pursued it as an educational exercise. This package is designed to be run in an IDE for full functionality.

---

## Quick troubleshooting

* **`ModuleNotFoundError: core`**
  Run `python bootstrap.py` again (new env? moved repo?). For zero-install runs, prefer `python run.py ...`.

* **`bid: command not found`**
  Re-run `python -m pip install -e .` in the **active** env; confirm `which bid` points into that env.

* **Jupyter**
  Ensure the notebook kernel is the same Conda env you bootstrapped/installed into.


## Repo checks & formatting — quick start. More information can be found in the readme under /tests

* **Hooks** run on every commit (and push):

  * `precommit_tests_run` (repo health checks)
  * `isort`, `autopep8`, and small hygiene fixes
* **First-time setup**

  ```bash
  python -m pip install --upgrade pre-commit
  python -m pre_commit install
  python -m pre_commit run --all-files   # baseline the repo
  ```
* **What’s enforced**

  * Import hygiene (sorted imports, consistent style)
  * PEP8 fixes (120-char line length)
  * Package integrity (no missing `core.*` imports, no circular imports, no duplicate public symbols)
* **If a hook modifies files:** stage and commit again:

  ```bash
  git add -A && python -m pre_commit run --all-files
  git commit -m "apply formatting"
  ```
* **Manual checks**

  ```bash
  # Orchestrated health checks (what pre-commit runs)
  python -m tests.precommit_tests_run --tree-max-depth 9 --init-only-nonempty --env-mode off
  # Quick env export when needed (not part of hooks)
  python -m tests.export_or_diff_env --mode diff
  ```

---
