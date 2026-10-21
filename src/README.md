# src — Core library (models, data, training, evaluation)

Python package holding everything the experiment entrypoints in `experiments/` import:
the neural-network definitions, the skeleton data pipeline, the training/evaluation loops,
and the post-processing/plotting helpers. Nothing here is run directly for an experiment;
instead `experiments/exam_fusion.py` and `experiments/exam_each_seed.py` compose these modules.

Every subfolder below carries its own `README.md` / `README_ja.md` describing its files.

## Subfolders

| Subfolder | Role |
|---|---|
| [`data/`](data/) | Skeleton data pipeline: raw-data preprocessing, HDF5 dataset generation, and the PyTorch `Dataset`/`DataLoader` feeder. |
| [`model/`](model/) | Network definitions: the proposed STGCLSATT encoder, the Early/Middle/Late fusion models, their GCN/RNN/attention subnets, and ported baselines. |
| [`training/`](training/) | Training and evaluation loop management (`Train` / `Test` / `Run`). Consumed by both experiment entrypoints. |
| [`plot/`](plot/) | Metric calculation and figure generation (per-class boxplots, indicator computation, summary CSV output). |
| [`utils/`](utils/) | Experiment utilities: name/label definitions, plot fonts, confusion matrices, post-run cleanup. |

## Top-level scripts

| File | Role |
|---|---|
| `ensemble_mean.py` | Averages the predictions of several experiments to compute ensemble accuracy. |
| `results_process.py` | Post-experiment pipeline: delete unneeded files → emit summaries → box plots. |
| `results_process_batch.py` | Result post-processing scoped to a specific directory. |
