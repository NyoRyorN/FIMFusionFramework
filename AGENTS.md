# AGENTS.md — TNSRE2026FIMFusion Development Guidelines (for Codex)

## Commands to run when verifying behavior
Copy `path_example.sh` to `path.sh`, fill in the placeholders, then source it to activate the virtual environment.
```bash
source path.sh
```

## Comment / docstring granularity

### Basic policy
- **Write WHAT by default.** Identifier names alone often don't convey the full picture of what the code does, so briefly describe what it does.
- **Add WHY where it isn't obvious.** Supplement with the design rationale, constraints, or background so that future readers can judge whether a change is safe or appropriate.
- If both WHAT and WHY are needed, they may be combined in the same comment.
- Comments in this project are written in English.

---

### Module docstring
At the top of each script, include the following:

```python
"""
filename - one-line purpose description

Overview of algorithm/framework (method name, key components, training flow).

Usage:
    $ python exam_xxx.py

Config: config/config_xxx.yaml
Env: KEY_NAME (description) in config/.env
"""
```

Include:
- File name and one-line purpose
- Overview of the algorithm/framework (method name, key components, training flow)
- Usage example
- Dependent config files / environment variables

---

### Function docstring
Write this for functions such as `main()` whose processing spans multiple phases.

```python
def main():
    """One-line summary.

    Processing flow:
        1. Step A
        2. Step B
            a. Sub-step
        3. Step C

    Args:
        arg1: description (only if args exist)

    Returns:
        description (only if return value exists)

    Raises:
        ExceptionType: condition under which it is raised.
    """
```

- Not needed for simple helper functions. Write it for functions with 3 or more phases.
- `Args` / `Returns` may be omitted if self-evident. `Raises` must always be documented when an exception is explicitly raised.

---

### Inline comments
Write these only in the following cases, to explain hidden constraints, workarounds, or surprising behavior.

| When to write | Example |
|---|---|
| Reason for an environment-dependent setting | `# required for environments without write access to home dir` |
| Non-obvious meaning of a flag value | `# -1 disables augmentation` |
| Reason for choosing one option among several | `# always use seed 42 when seed_fix_mode is "model"` |
| Intent behind a batching/sampling strategy | `# WeightedRandomSampler to compensate for class imbalance` |
| Preventing a side effect | `# explicitly finalize to avoid mixing logs from the next seed` |

Not needed:
- Explaining processing that's already clear from reading the code (e.g. `# create DataLoader`)
- Repeating what's already clear from argument/variable names

---

### Section-separator comments
When splitting logic blocks within a long function, use the following format:

```python
# ------------------------------------------------------------------
# Section name (e.g. Build dataset for pre-training)
# ------------------------------------------------------------------
```

Guideline: use when a function has 4 or more logical phases.

---

### Language
- Comments and docstrings are written in **English** (project convention).
- Code identifiers (variable/function names) remain in English.

---

## Experiment / script implementation policy

- Split processing into functions of appropriate granularity. Don't put everything in a single file.
- Place new functions in the `src/` folder/file that already contains similar processing.
- Reuse existing functions in `src/` rather than reimplementing them.
- **When creating a new directory, always create a `README.md` and `README_ja.md` in it.**
  - Follow the format of existing per-folder READMEs (e.g. `src/model/baselines/README.md` and its `README_ja.md`). Include a heading (`# path — one-line role`), an overview, and a file/subfolder table.
  - For model-implementation folders, note the source paper; for code folders such as `src/training/`, note which experiments use it.
  - Also add a description of the new directory to the subfolder table in the parent folder's `README.md` / `README_ja.md` (describe it in both the parent folder and the new folder itself).
  - Output-only or deprecated folders may omit an individual README, but this should be noted in the parent folder's README.
- **When adding, modifying, or deleting files, update all related `README.md` files (and the corresponding `README_ja.md`) on the spot.**
  - This applies to every folder across the project, including `src/`, `reports/`, `experiments/`, etc.
  - Always check and update the `README.md` of the folder the changed file belongs to (e.g. `reports/skeleton_plots/README.md`).
  - Also update the parent folder's `README.md` if it references that folder (e.g. `reports/README.md`).
  - Update the root `README.md` as well if it describes that part of the project.
  - Update the README not only for file additions/deletions/role changes but also whenever a script's arguments or execution method change.

---

## src/ folder structure

| Folder | Role |
|---|---|
| `data/feeder/` | PyTorch Dataset/DataLoader. Lazy loading of skeleton HDF5 data, data augmentation, experiment result management |
| `data/dataset/` | HDF5 dataset generation (train.h5 / test.h5) |
| `data/preprocessing/` | Raw data preprocessing pipeline (temporal interpolation → filtering/feature computation → skeleton point reduction) |
| `model/` | Neural network definitions. Main encoder (STGCLSATT), fusion models (early_fusion / middle_fusion / late_fusion), subnets (gcn / rnn / attention), baselines (`baselines/`: LITEMV / ConvTran / VanTran from DeepRehabPile, SgcnLstm from STGCN-rehab, DDGCN from DD-GCN), encoder selection factory (build_model.py) |
| `training/` | Training/evaluation loop management. Train / Test / Run classes |
| `utils/` | Experiment utilities. Name definitions, font settings, confusion matrix, post-experiment cleanup helpers |
| `plot/` | Visualization. Evaluation metrics, per-class accuracy, ablation summary plots |
| `results_process.py` | Post-experiment processing pipeline (remove unnecessary files → output summary → box plots) |
| `results_process_batch.py` | Post-processing of experiment results targeting a specific directory |
