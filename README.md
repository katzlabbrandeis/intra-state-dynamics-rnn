Blech_intra_state_dynamics README

Output by: Vincent Calia-Bogan
This repo is a work in progress.


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
