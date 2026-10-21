# experiments — Experiment entrypoints

The two top-level scripts that run the experiments. Each reads a YAML config from `config/`,
builds the data feeder, and drives the training/evaluation loops in `src/training/`. Outputs
are written under `results/` (override the directory with `--output/-o`).

## Files

| File | Role |
|---|---|
| `exam_each_seed.py` | Single-action, multi-seed experiment: predicts a FIM item from one action, over combinations of `action_task_indices` × `fim_item_indices`, with K-fold or holdout validation and statistical result analysis. Uses `src/training/Run`. |
| `exam_fusion.py` | Multi-action fusion experiment: combines the actions in `combined_action_indices` with Early / Middle / Late Fusion (or a baseline). Uses `src/training/Run`. |

## Run

```bash
source path.sh
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml
python experiments/exam_fusion.py   -c config/fusion_strategy_ablations/early_data.yaml
# -o/--output <dir> to override the results directory
```

Config folders under `config/` (each with its own README) list the available conditions and
their run commands.
