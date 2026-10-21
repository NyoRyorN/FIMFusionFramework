# Multi-Action Fusion for Skeleton-Based FIM Score Prediction

[Japanese version](README_ja.md)

This repository contains experiment code for predicting FIM
(Functional Independence Measure) item scores from rehabilitation skeleton sequences.
It supports both single-action baselines and multi-action Early / Middle / Late Fusion
experiments.

Paper:

> **Skeleton-Based Multi-Action Fusion Framework for Binary Classification of FIM Score from Simplified Rehabilitation Actions**
>
> Jun Masaki, Ariaki Higashi, Naoko Shinagawa, Kazuhiko Hirata, Yuichi Kurita, and Akira Furui
>
> *IEEE Transactions on Neural Systems and Rehabilitation Engineering(TNSRE), 2026*
>
> DOI: 10.1109/TNSRE.2026.3744185

## Overview

The core model family combines ST-GCN, RNN/AGC-LSTM, and Spatio-Temporal Attention
components for skeleton-sequence classification. The repository handles experiments
that predict a FIM item from one action, and experiments that fuse multiple actions
into one FIM-item prediction.

| Workflow | Entry Point | Description |
| --- | --- | --- |
| Single-action experiment | `experiments/exam_each_seed.py` | Runs multi-seed 5-fold evaluation for combinations of `action_task_indices` and `fim_item_indices`. |
| Fusion experiment | `experiments/exam_fusion.py` | Combines actions listed in `combined_action_indices` with Early / Middle / Late Fusion. |
| Cross-subject aggregation | `reports/calculates/calc_cross_subject_variables.py` | Concatenates fold-level `PredictTruelabel.csv` files and summarizes per-subject match rates. |
| Ablation plotting | `reports/ablation_boxplot/ablation_cross_subject_variables.py` | Generates condition-wise violin + box plots from cross-subject summaries. |
| CAM weight visualization | `reports/skeleton_plots/` | Visualizes saved Attention / GradCAM weights as skeleton videos, combined fusion views, and GradCAM distribution plots. |

See also:

- [Single-action experiment: Japanese](docs/each_seed_ja.md)
- [Single-action experiment: English](docs/each_seed.md)
- [Fusion experiment: Japanese](docs/fusion_ja.md)
- [Fusion experiment: English](docs/fusion.md)
- [CAM weight visualization: Japanese](docs/cam_weights_ja.md)
- [CAM weight visualization: English](docs/cam_weights.md)
- [Experiment docs index](docs/README.md)

## Environment

For this checkout, load the project environment with:

```bash
source ./path.sh
```

`path.sh` adds the repository root to `PYTHONPATH`, activates the
project virtual environment, and sets the PyTorch CUDA allocation option.
It is machine-specific and gitignored; on a fresh clone, copy
`path_example.sh` to `path.sh` and fill in the placeholders.

If you need to install dependencies from scratch, use `pyproject.toml` or
`requirements.txt`.

```bash
# With uv
uv sync

# With pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main requirements:

- Python >= 3.11
- PyTorch 2.8.0 series
- For CUDA runs, confirm that the PyTorch index in `pyproject.toml` matches the GPU environment.

## Directory Structure

```text
TNSRE2026FIMFusion/
├── config/                          # Experiment YAML configs (each folder has its own README.md / README_ja.md listing the comparison conditions and run commands)
│   ├── action_selection/            # Action-subset ablation configs
│   ├── complement_ablation/         # Missing-action complementation ablation configs
│   ├── fusion_strategy_ablations/   # Single-action and fusion strategy configs
│   └── model_ablation/              # Model ablation configs (module variants + baselines, shared non-model conditions)
├── data/                            # Input and processed data
│   ├── OriginalData/                # Raw data
│   ├── ComplementedData/            # Time-complemented data
│   ├── FilteredData/                # Filtered and feature-computed data
│   ├── ConciseData/                 # Reduced-joint data
│   └── Info/                        # Metadata such as data lists
├── docs/                            # Detailed experiment READMEs
├── experiments/                     # Experiment entrypoints (exam_each_seed.py, exam_fusion.py)
├── reports/                         # Result aggregation and visualization scripts
│   ├── calculates/                  # Cross-subject variable aggregation
│   ├── ablation_boxplot/            # Ablation box/violin plots
│   ├── check_fim_dist/                  # FIM distribution checks
│   └── skeleton_plots/               # Skeleton and Attention/GradCAM visualization
├── results/                         # Experiment outputs
├── src/                             # Core library imported by experiments/
│   ├── data/                        # Skeleton data pipeline
│   │   ├── preprocessing/           # Raw-data preprocessing (order/, utils/)
│   │   ├── dataset/                 # HDF5 dataset generation (utils/)
│   │   └── feeder/                  # PyTorch Dataset/DataLoader + result management
│   ├── model/                       # Network definitions
│   │   ├── gcn/                     # ST-GCN subnet (components/)
│   │   ├── rnn/                     # LSTM / AGC-LSTM subnets
│   │   ├── attention/              # Attention subnet + CAM/GradCAM (components/)
│   │   ├── baselines/               # Ported baselines (LITEMV/ConvTran)
│   │   └── params/                  # Model parameter helpers
│   ├── training/                    # Train / Test / Run (used by experiments/)
│   ├── plot/                        # Metric calculation and figure generation
│   └── utils/                       # Names, fonts, confusion matrices, cleanup
├── path_example.sh                  # Template for path.sh (gitignored, machine-specific)
├── pyproject.toml                   # uv project definition
└── requirements.txt                 # pip-compatible dependency list
```

Every folder under `src/`, `experiments/`, `reports/`, and `config/` carries its own
`README.md` / `README_ja.md` describing its files and subfolders.

### Folder Roles and Dependencies

The repository forms one pipeline from raw data to figures:

```text
data/OriginalData
   └─ src/data/preprocessing ──► data/ConciseData
                                     └─ src/data/dataset ──► train.h5 / test.h5
                                                                 └─ src/data/feeder
                                                                        ▼
config/*.yaml ──► experiments/ ──► src/training ◄── src/model
                                        │
                                        ▼
                                    results/ ──► reports/ ──► figures & summaries
```

- **`src/data/`** — the data pipeline. `preprocessing/` reads `data/OriginalData/` and
  writes the reduced-joint `data/ConciseData/` (file organization in `order/`, numeric
  helpers in `utils/`). `dataset/` builds `train.h5` / `test.h5` from `ConciseData` with
  subject-aware K-fold splitting. `feeder/` wraps those HDF5 files as a PyTorch
  `Dataset`/`DataLoader` with augmentation, and also holds the result-aggregation class
  used during training. Label/joint definitions come from `src/utils/names.py`.
- **`src/model/`** — network definitions, independent of the data pipeline. The proposed
  encoder `stgclsatt.py` composes the three subnets `gcn/` → `rnn/` → `attention/`;
  the Early/Middle/Late fusion models wrap one or more encoders and share the
  missing-action primitives in `mask_token.py`. `build_model.py` is the factory that both
  `src/training` entry classes call to resolve `model.name` (including `baselines/`).
- **`src/training/`** — glue between feeder and model. `Run` builds the model via
  `build_model.py` (a single encoder, or an Early/Middle/Late fusion model dispatched by
  `fusion.type`), then drives the `Train`/`Test` loops; `Test` computes metrics via
  `src/plot/calc_indicator.py`, confusion matrices via `src/utils/`, and GradCAM weights
  via `src/model/attention/cam_fusion.py`.
- **`experiments/`** — the only scripts run to launch experiments. They read a YAML from
  `config/`, build feeders, call `src/training`, and write everything under `results/`.
  Both `exam_each_seed.py` and `exam_fusion.py` use `Run`.
- **`reports/`** — post-hoc only; depends on `results/` output files, never on training
  code. `calculates/` aggregates per-fold `PredictTruelabel.csv` into per-subject rates,
  which `ablation_boxplot/` turns into comparison figures; `skeleton_plots/` renders the
  Attention/GradCAM weights saved by `Test`.
- **`config/` and `docs/`** — configs select model/fusion/actions per experiment batch;
  docs hold the detailed per-experiment READMEs referenced above.

## Data Preparation

Place raw skeleton data under `data/OriginalData/` as subject or measurement directories.

```text
data/OriginalData/
├── NEDO001-0001_...
├── NEDO001-0002_...
└── ...
```

The current preprocessing implementation lives under `src/data/preprocessing/`.
`orderliness_data.py` moves and deletes files, so back up raw data first when needed.

```bash
# Organize raw data
python src/data/preprocessing/orderliness_data.py

# Full pipeline: time complementation -> filtering plus velocity, angle,
# and bone-length computation -> reduced-joint skeleton data generation
python src/data/preprocessing/make_concise_data.py
```

`make_concise_data.py` builds `data/ConciseData/` directly from
`data/OriginalData/` in one command — time complementation, filtering, and
velocity/angle/bone-length computation all run in memory, so no intermediate
directories are written to disk. Centering is applied later at dataset-build
time (`norm_centering_mode: First`), so only the non-centered dataset is
produced.

The run also classifies every (subject, motion) slot and writes the
per-FIM-item missing-motion distribution to `data/Distributions/`. Skeleton
quality failures are identified only from existing `Unused/` evidence;
`make_concise_data.py` itself does not move raw CSV files into `Unused/` — see
[docs/missing_motion.md](docs/missing_motion.md).

## Running Experiments

The default paths embedded in the scripts, `config/config.yaml` for
`experiments/exam_each_seed.py` and `config/configSomeTask.yaml` for
`experiments/exam_fusion.py`, are not included in the current checkout.
Normally, pass an existing config with `-c` / `--config`. Both scripts also
accept `-o` / `--output` to save results under a directory other than the
default `results/`.

Each ablation folder under `config/` has its own `README.md` /
`README_ja.md` that enumerates every comparison condition and its run command:
[`action_selection/`](config/action_selection/README.md),
[`complement_ablation/`](config/complement_ablation/README.md),
[`fusion_strategy_ablations/`](config/fusion_strategy_ablations/README.md),
[`model_ablation/`](config/model_ablation/README.md).

### Single-Action Experiment

```bash
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml
```

### Fusion Experiments

```bash
# Early Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_data.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_graph.yaml

# Middle Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_shallow.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_deep.yaml

# Late Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_feature.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_probability.yaml
```

### Model and Action-Selection Ablations

Model ablation configs are under `config/model_ablation/`. The folder holds
both the proposed-model module variants (`stgclsatt.yaml`, `stgcn.yaml`, ...)
and the baseline models (`litemv.yaml`, `convtran.yaml`) so that they run as
one ablation batch; every setting except the `model` section is kept
identical across the configs.
Action-selection ablation configs are under `config/action_selection/`.
Run them with `exam_fusion.py -c <config>`.

### Missing-Action Complementation Ablation

Configs under `config/complement_ablation/` compare how missing actions are
handled, using `config/action_selection/stand_actions.yaml` as the base;
every setting except the complementation keys is identical across the
configs. Run them with `exam_fusion.py -c <config>`. See
[`complement_ablation/README.md`](config/complement_ablation/README.md) for
the full comparison table (complete-case, zero padding, padding mask, and
modality-dropout variants).

| Config | Setting changed from the base | Behavior |
| --- | --- | --- |
| `complete_case.yaml` | `preprocess.is_fill_actionless: false` | Subjects missing any action are excluded (complete-case, `_CompleteCase` suffix) |
| `zero_fill.yaml` | none (baseline) | Missing actions stay all-zero slices (`_ZeroPad` suffix) |
| `zero_fill_modality_dropout.yaml` | `zero_fill.yaml` + `trainings.modality_dropout_p: 0.15` | Same as `zero_fill.yaml`, plus training-time modality dropout at the real dataset's average missing-action rate (`_ZeroPad_MD` suffix) |
| `pad_mask.yaml` | `fusion.missing_representation: pad_mask` | Missing actions are not forwarded through the encoder and are excluded by masked-mean aggregation (`_Masking` suffix) |
| `pad_mask_modality_dropout.yaml` | `pad_mask.yaml` + `trainings.modality_dropout_p: 0.15` | Same as `pad_mask.yaml`, plus the same 0.15 modality dropout rate (`_Masking_MD` suffix) |

### Baseline Models

PyTorch ports of two baselines live in `src/model/baselines/`, both from
[DeepRehabPile](https://github.com/MSD-IRIMAS/DeepRehabPile): `LITEMV`
(Ismail-Fawaz et al., arXiv:2409.02869) and `ConvTran` (Foumani et al., DMKD
2024, following the official implementation). They are selected by
`model.name` and trained with the project-standard protocol; the model
hyperparameters keep the official defaults.

```bash
# fusion experiments (fusion.type: late_* only; early and middle are not supported)
python experiments/exam_fusion.py -c config/model_ablation/litemv.yaml
python experiments/exam_fusion.py -c config/model_ablation/convtran.yaml
```

GradCAM output (`model.cam_type: gradcam`) is not supported for baselines.

## Main Configuration Keys

| Key | Description |
| --- | --- |
| `model.name` | Model name, such as `STGCN`, `STGCLSAtt`, or `STGCAGLS`. Baselines: `LITEMV`, `ConvTran`. |
| `fusion.type` | Fusion strategy: `early_chn_stack`, `early_exp_graph`, `middle_shallow`, `middle_deep`, `late_feature`, or `late_probability`. Not used by the single-action config. |
| `fusion.missing_representation` | Missing-action representation: `zero` (default) or `pad_mask` (Late Fusion `mean` only). See `docs/fusion.md`. |
| `trainings.modality_dropout_p` | Training-time modality dropout probability (default `0.0` = disabled). |
| `action_task_indices` | Action indices for single-action experiments. They map to `Tasks()` in `src/utils/names.py`. |
| `combined_action_indices` | Action indices combined by fusion experiments. |
| `fim_item_indices` | Target FIM item indices. |
| `classes` | FIM score classes to classify. Internal labels are converted to 0-based indices. |
| `seed_fix_mode` | `split`, `model`, or `change`; controls split and model-training seed behavior. |
| `preprocess.*` | Frame length, centering, moving-frame extraction, short-sequence filling, and missing-action filling. |
| `trainings.*` | Epochs, optimizer, scheduler, batch size, weighted sampler, and cleanup behavior. |
| `wandb.use_wandb` | Enables or disables Weights & Biases logging. |
| `output_weights` | Saves Attention / Grad-CAM weights when supported by the model. |

## Experiment Outputs

Experiment outputs are written under `results/<experiment_name>/`.

Typical single-action output structure:

```text
results/<experiment_name>/
├── situations.yaml
├── Accuracies.csv
├── results.csv
└── <Action>_to_<ItemIndex>-<FIMItem>/
    ├── Accuracies.csv
    └── Stratified_Seed<seed>/
        ├── Accuracies.csv
        └── Fold<fold>/
            ├── checkpoint.pth
            ├── ELLA_ep<N>.csv
            ├── confusion_matrix.png
            ├── PredictTruelabel.csv
            ├── trainDir.csv
            ├── train_distribution.csv
            ├── testDir.csv
            ├── test_distribution.csv
            ├── weights_Att.h5        # optional
            └── weights_CAM.h5        # optional
```

Fusion experiments create `Stratified_Seed<seed>/Fold<fold>/` directories under each
FIM-item directory.

Main CSV files:

| File | Contents |
| --- | --- |
| `situations.yaml` | Copy of the runtime config. |
| `Accuracies.csv` | Aggregated metrics by seed, fold, and FIM item: `Accuracy`, `Balanced_Accuracy`, per-class `Class<c>_Accuracy` [%], and `F1Score` / `MCC` / `Precision` / `Recall` / `AUCROC` (0-1). All metrics are computed regardless of the `wandb.use_wandb` setting. |
| `results.csv` | Summary table extracted from `Seed == "Mean"` rows. |
| `PredictTruelabel.csv` | Sample-level fold predictions with `Directory`, `Predicted`, `True`, `RealScore`, and `Prob_*` columns. |
| `ELLA_ep<N>.csv` | Epoch-wise training log. |
| `trainDir.csv`, `testDir.csv` | Split metadata such as `Directory`, `FrameStart`, and `RealScore`. |

### Backfilling metrics into older results

Experiment directories created before per-class accuracy and the detailed metrics
were recorded independently of wandb have empty `Class<c>_Accuracy` columns and no
`F1Score` / `MCC` / `Precision` / `Recall` / `AUCROC` columns. Recompute and fill
them from the per-fold `PredictTruelabel.csv` files with:

```bash
python -m src.backfill_metrics results/<experiment_name>          # fill missing columns
python -m src.backfill_metrics results/<experiment_name> --dry-run  # compare only, no writes
```

Columns that already contain values are skipped unless `--force` is given. The
aggregation follows the experiment scripts (fold mean -> seed mean).

## Cross-Subject Aggregation

Use the aggregation script to convert fold-level `PredictTruelabel.csv` files into
per-subject match-rate summaries.

```bash
# Explicit experiment root(s)
python reports/calculates/calc_cross_subject_variables.py results/<experiment_name>

# Scan direct children under a parent directory
python reports/calculates/calc_cross_subject_variables.py --scan results/<parent_dir>
```

Main outputs:

| Output | Location |
| --- | --- |
| `PredictTruelabel_all_folds.csv` | Directly under each `Stratified_Seed<seed>/` directory |
| `CrossSubjectVariables.csv` | Under each FIM-item or `<Action>_to_<Item>` directory |
| `CrossSubjectVariables_all_actions.csv` | Under item directories that contain action subdirectories |
| `CrossSubjectVariables_action_to_item/<Item>.csv` | Under the experiment root |
| `CrossSubjectVariables_all_items.csv` | Under the experiment root |

## Ablation Plots

Three scripts in `reports/ablation_boxplot/` compare ablation conditions side-by-side.
The wrapper `reports/ablation_process.py` runs the first two in sequence.

| Script | Output directory | Description |
| --- | --- | --- |
| `ablation_sum_plots.py` | `AblationPlots/` | Balanced-Accuracy box + bar plots per FIM item and cross-item |
| `ablation_plots_each_class.py` | `AblationPlotsEachClass/` | Class-wise accuracy box plots per FIM item |
| `ablation_cross_subject_variables.py` | `AblationPlotsCrossSubject/` | Per-subject match-rate violin + box plots |

See [docs/ablation_boxplot.md](docs/ablation_boxplot.md) /
[docs/ablation_boxplot_ja.md](docs/ablation_boxplot_ja.md) for the full
workflow, directory layout, and output file reference.

### Directory layout and `order.csv`

Place target experiment result directories under one ablation root. Name each condition
directory starting with its `order.csv` prefix followed by `_`:

```text
<ablation_root>/
├── order.csv            ← one prefix per row, no header; controls plot order
├── <prefix_A>_<rest>/   ← condition A
├── <prefix_B>_<rest>/   ← condition B
└── ...
```

### Typical workflow

```bash
# After src/results_process_batch.py and calc_cross_subject_variables.py have been run:
python reports/ablation_process.py
# Enter rootDir: results/<ablation_root>

python reports/ablation_boxplot/ablation_cross_subject_variables.py results/<ablation_root>
```

### Output files

```text
AblationPlots/
├── Balanced_Accuracy_<Item>.{png,pdf,svg}
├── Balanced_Accuracy_<Item>_BarPlot.{png,pdf,svg}
└── Balanced_Accuracy_CrossActions.{png,pdf,svg}

AblationPlotsEachClass/
└── Class-wise_Accuracy_<Item>.{png,pdf,svg}

AblationPlotsCrossSubject/
├── AverageMatchRate_AllItems.{csv,png,pdf,svg}
└── MatchRate_<Item>.{csv,png,pdf,svg}
```

`MaxAction` first uses `Max_Balanced_Acc_Action` from `Summary.csv`, and falls back to
balanced-accuracy columns in `results.csv` / `Accuracies.csv`. Supported column names
are `BalancedAccuracy`, `Balanced_Accuracy`, and `Balanced Accuracy`.

## Weights & Biases

Set `wandb.use_wandb: true` to log fold-level training and evaluation to Weights & Biases.
Set `wandb.entity` to the target entity or team name.

```yaml
wandb:
  use_wandb: true
  entity: "your-team-name"
```

## Citation

If you find this code useful in your research, please cite:

```bibtex
@article{author2026multifusion,
  title     = {Skeleton-Based Multi-Action Fusion Framework for Binary Classification of FIM Score from Simplified Rehabilitation Actions},
  author    = {Jun Masaki, Ariaki Higashi, Naoko Shinagawa, Kazuhiko Hirata, Yuichi Kurita, and Akira Furui},
  journal   = {IEEE Transactions on Neural Systems and Rehabilitation Engineering(TNSRE)},
  year      = {2026},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {10.1109/TNSRE.2026.3744185},
  memo      = {IEEE Trans. Neural Syst. Rehabil. Eng. (Abbreviation of journal name)}
}
```

## License

This project is provided for research purposes only. Please contact the authors for
licensing information.
