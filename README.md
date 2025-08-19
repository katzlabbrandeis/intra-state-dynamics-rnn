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

### Option A — Zero-install (recommended for development)

```bash
python run.py --help
python run.py preprocess --dry-run
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
bid preprocess --dry-run
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
