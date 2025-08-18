
# Detailed README for repo checks and formatting

## Overview

This directory contains **repo-introspection tests** and a **single orchestrator** that pre-commit runs. Outputs (JSON/TXT) go under `output/tests_output/` (ignored by Git).

### Orchestrator

**`precommit_tests_run.py`**
Runs all checks in one go and fails the commit on structural issues.

```bash
python -m tests.precommit_tests_run --tree-max-depth 9 --init-only-nonempty --env-mode off
```

* **Fails commit** if `check_package_integrity` reports:

  * missing imports within `core.*`
  * circular imports within `core.*`
  * duplicate public symbols
* Writes timestamped reports to `output/tests_output/…`

Flags:

* `--no-init` / `--no-integrity` / `--no-tree` — skip specific checks
* `--tree-include-files` — include files in the tree views
* `--env-mode [off|diff|update]` — run environment export (default: `off`)
* Also forwards `--init-only-nonempty`, `--init-also`, `--tree-max-depth`, `--tree-exclude`

---

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

## Individual scripts

### `ensure_init_files.py`

Ensures `__init__.py` exists in package dirs (default scope: `src/core/**`).

```bash
python -m tests.ensure_init_files
python -m tests.ensure_init_files --only-nonempty
python -m tests.ensure_init_files --also src/scripts
```

* Keeps imports predictable across tools/IDEs.

### `check_package_integrity.py`

Static analysis (no module execution). Prints **JSON** to stdout with:

* `missing_imports`: unresolved `core.*` imports
* `duplicate_symbols`: public names defined in >1 module
* `file_tree`: all `.py` files (relative paths)
* `symbols_by_file`: per-file functions/classes
* `import_graph`: adjacency among `core.*`
* `import_cycles`: detected cycles
* `summary`: counts + SHA256 digest

```bash
python -m tests.check_package_integrity > /tmp/pkg_integrity.json
```

Also writes a timestamped JSON report to `output/tests_output/integrity/`.

### `describe_tree.py`

Emits a nested **JSON** dir tree (stdout) and saves both JSON and ASCII versions to `output/tests_output/structure/`.

```bash
# Dirs only (default)
python -m tests.describe_tree --max-depth 6
# Include files (like `tree`)
python -m tests.describe_tree --include-files
```
> Note: Including files in tree structure can be overwhelmingly verbose.

Artifacts:

* `project_tree_*.json` — structured directory tree
* `project_tree_ascii_*.txt` and `project_tree_ascii_*.json` — ASCII tree + counts
>  Note: .txt saves are currently disabled by default. Comment out `line 42` of `describe_tree.py` under `/tests` to enable .txt saves.

### `export_or_diff_env.py` (manual)

Exports current Conda env to `environment.yml` (prefix removed) or shows a diff.

```bash
python -m tests.export_or_diff_env --mode diff
python -m tests.export_or_diff_env --mode update
```

> Not part of the pre-commit flow by default.

---

## Pre-commit configuration (summary)

`.pre-commit-config.yaml` relevant bits:

```yaml
exclude: '^output/'

- repo: local
  hooks:
    - id: precommit-tests-run
      name: precommit_tests_run (repo health checks)
      entry: python -m tests.precommit_tests_run --tree-max-depth 9 --init-only-nonempty --env-mode off
      language: system
      pass_filenames: false
      always_run: true
      stages: [commit, push]

- repo: https://github.com/pycqa/isort
  rev: '6.0.1'
  hooks: [ { id: isort } ]

- repo: https://github.com/pre-commit/mirrors-autopep8
  rev: 'v2.0.4'
  hooks:
    - id: autopep8
      args: ['-i', '--exit-code', '--max-line-length=100']

- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: 'v4.5.0'
  hooks:
    - { id: end-of-file-fixer }
    - { id: trailing-whitespace }
    - { id: mixed-line-ending, args: ['--fix=lf'] }
```

`pyproject.toml` (isort):

```toml
[tool.isort]
line_length = 100
multi_line_output = 3
include_trailing_comma = true
use_parentheses = true
ensure_newline_before_comments = true
known_first_party = ["core"]
src_paths = ["src"]
```

---

## Troubleshooting

* **Pre-commit using wrong Python** (Homebrew shebang): prefer `python -m pre_commit …` from your active Conda env.
* **Hooks modify files:** this is expected. Stage and re-run:

  ```bash
  git add -A && python -m pre_commit run --all-files
  ```
* **JSON parse errors in orchestrator:** ensure the sub-tools print **JSON-only** to stdout (no extra prints). All status messages should be written to files or stderr, not stdout.

---

## Extending / Adding new tests and scripts:

* Add new scripts under `tests/` and wire them into `precommit_tests_run.py`.
* Use `core.io.test_output` helpers to write reports:

  * `write_json_test_report(data, name, subdirs=(...), timestamp=True)`
  * `write_text_test_report(text, name, subdirs=(...), timestamp=True)`
