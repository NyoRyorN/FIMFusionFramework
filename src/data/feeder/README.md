# src/data/feeder — PyTorch Dataset / DataLoader and result management

The runtime data layer: a PyTorch `Dataset` that lazily reads skeleton sequences from the
HDF5 datasets, applies augmentation, and yields `(sequence, FIM label)` samples, plus a
result-aggregation helper used by the training loops to collect per-fold metrics.

## Files

| File | Role |
|---|---|
| `feeder.py` | PyTorch `Dataset` with lazy HDF5 loading, augmentation, and sample metadata; feeds the `DataLoader`. |
| `results.py` | Result-management class: aggregates per-class/balanced accuracy, saves predictions/true labels, computes mean/std across folds and seeds. |
| `tools.py` | Preprocessing/transform helpers for the feeder (channel selection, temporal cropping, 3D coordinate transforms). |
