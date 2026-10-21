# src/data — Skeleton data pipeline

The full path from raw measurements to model-ready tensors. Three stages, each in its own
subfolder: raw-data **preprocessing** (organize → complement → filter/feature → reduce
joints), HDF5 **dataset** generation, and the PyTorch **feeder** that lazily loads those
datasets during training.

## Subfolders

| Subfolder | Role |
|---|---|
| [`preprocessing/`](preprocessing/) | Raw-data preprocessing pipeline: file/dir organization, time complementation, filtering + feature computation (velocity, angle, bone length), reduced-joint output. |
| [`dataset/`](dataset/) | Builds `train.h5` / `test.h5` HDF5 datasets (single-action and multi-action) with StratifiedGroupKFold splitting. |
| [`feeder/`](feeder/) | PyTorch `Dataset`/`DataLoader` with lazy HDF5 loading and augmentation, plus per-experiment result management. |
