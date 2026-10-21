# src/data/dataset — HDF5 dataset generation

Builds the `train.h5` / `test.h5` datasets consumed by the feeder from the preprocessed
`ConciseData`. Handles both single-action datasets and multi-action (fusion) datasets, with
StratifiedGroupKFold splitting that keeps a subject's samples within a single fold.

## Files

| File | Role |
|---|---|
| `make_single_dataset.py` | Builds a dataset for one (action task, FIM item) pair; applies StratifiedGroupKFold cross-validation splitting by subject. |
| `make_some_tasks_dataset.py` | Builds a fused dataset over multiple action tasks, aligning/complementing per-action samples before merging. |

## Subfolders

| Subfolder | Role |
|---|---|
| [`utils/`](utils/) | Dataset-build helpers: movement-onset detection and DataFrame→NumPy conversion. |
