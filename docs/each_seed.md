# exam_each_seed.py: Single-Action FIM Prediction Experiment

`experiments/exam_each_seed.py` runs the single-action baseline experiment. It trains a model that predicts one FIM item from one rehabilitation action. For every `action_task_indices` and `fim_item_indices` combination, the script runs multiple seeds with 5-fold cross-validation and saves results at the fold, seed, action-to-item, and experiment levels under `results/`.

## How To Run

In the current checkout, the single-action experiment config is `config/fusion_strategy_ablations/single_action.yaml`. The script still has `config/config.yaml` as its parser default, but that file is not part of the standard config tree, so pass `-c` explicitly.

```bash
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml
```

Pass `--output` / `-o` to save results under a directory other than the default `results/`:

```bash
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml -o /path/to/output
```

When `wandb.use_wandb: true`, each fold is tracked as a Weights & Biases run. Set `wandb.entity` to the target entity or team.

## Key Config Fields

| Key | Meaning |
| --- | --- |
| `model.name` | Model architecture, such as `STGCN`, `STGCLSAtt`, or `STGCAGLS`. Baselines: `LITEMV`, `ConvTran`, `VanTran` (DeepRehabPile), `SgcnLstm` (STGCN-rehab), `DDGCN` (DD-GCN). |
| `action_task_indices` | Single input actions. The indices correspond to `Tasks()` in `src/utils/names.py`. |
| `fim_item_indices` | Target FIM items. The indices correspond to `FIMitems()` in `src/utils/names.py`. |
| `seeds` | Random seeds used for repeated splits and model training. |
| `classes` | FIM score classes to classify. Internal labels are converted to 0-based class indices. |
| `data_list_path` | Optional CSV listing subject directories. If empty, the data directory is scanned. |
| `seed_fix_mode` | `split` fixes the split seed to 42, `model` fixes the model seed to 42, and `change` uses each configured seed for both. |
| `preprocess.*` | Normalization, frame length, moving-frame extraction, and short-sequence filling settings. |
| `trainings.*` | Epochs, optimizer, scheduler, batch size, weighted sampler, and related training settings. |
| `output_weights` | Enables Attention or GradCAM weight output when the selected model exposes weights. |

## Experiment Flow

1. Load the YAML config and resolve target actions, FIM items, seeds, and classes.
2. Build `results/<experiment_name>/` through `decideExamName()` and copy the config to `situations.yaml`.
3. For each `action -> FIM item` pair, run `makeSingleDataset()` and create 5 train/test splits with `StratifiedGroupKFold(n_splits=5)`, grouped by subject ID.
4. Train the configured model (STGCLSATT family or a DeepRehabPile baseline) for each fold and evaluate it on that fold's test split.
5. Aggregate fold results into seed results, seed results into action-to-item results, and all pairs into experiment-level summary files.

## Internal Runtime Processing

### 1. Entrypoint And Config Loading

| Processing Step | File / Function | Details |
| --- | --- | --- |
| CLI parsing | `experiments/exam_each_seed.py` `__main__` | Parses `--config` / `-c` and `--output` / `-o` with `argparse` and calls `main(configPath=..., outputDir=...)`. |
| Experiment orchestration | `experiments/exam_each_seed.py` `main()` | Controls YAML loading, action/FIM/seed/fold loops, training, evaluation, aggregation, and file output. |
| Plot font setup | `src/utils/set_plt_font.py` `setPltFont()` | Initializes matplotlib font settings before figures are created. |
| Name resolution | `src/utils/names.py` `Tasks()` / `FIMitems()` | Converts `action_task_indices` and `fim_item_indices` into action names and FIM item names. |
| Experiment name generation | `src/utils/decides.py` `decideExamName()` | Builds `results/<experiment_name>/` from the model name, ST-GCN layer count, seed-fixing mode, attention setting, and start timestamp. |
| Seed control | `src/utils/decides.py` `decideSeed()` | Sets Python, NumPy, PyTorch, and CUDA seeds and deterministic cuDNN behavior. `seed_fix_mode` determines whether the split seed or model seed is fixed to 42. |
| W&B config construction | `src/utils/decides.py` `decideWandbConfig()` | Builds a W&B config dict containing seed, fold, action, FIM item, preprocessing settings, model settings, and training settings. |

### 2. Single-Action Dataset Creation

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Dataset creation entrypoint | `src/data/dataset/make_single_dataset.py` `makeSingleDataset()` | For one action and one FIM item, scans data, reads FIM labels, converts skeleton CSVs to NumPy arrays, creates 5-fold splits, and writes split files. |
| Input data root selection | `makeSingleDataset()` | The input root is always `data/ConciseData`; centering is applied at load time (`norm_centering_mode: First`). |
| Subject directory selection | `makeSingleDataset()` | Uses the `Directory` column from `data_list_path` when available; otherwise scans the selected data root. |
| FIM score loading | `makeSingleDataset()` | Reads each subject's `FIM.csv` and extracts the target FIM item's `Score`. Scores that are non-positive or NaN are skipped. |
| Class-label mapping | `makeSingleDataset()` | Maps the original FIM score into a 0-based class index according to `classes`. For example, `classes: [6,7]` maps scores 6 and 7 to internal labels 0 and 1. |
| Skeleton CSV conversion | `src/data/dataset/utils/dataframe_to_numpy.py` `dataframeToNumpy()` | Reads `p_x.csv`, `p_y.csv`, `p_z.csv`, and other axis CSVs for requested position / velocity / acceleration attributes and converts them into model input arrays. |
| Movement onset detection | `src/data/dataset/utils/detect_movement_onset.py` `detectMovementOnset()` | When `use_moving_frame: true`, estimates the window start frame from velocity signals and extracts a `frame_length` window. |
| Short-sequence handling | `dataframeToNumpy()` | If a sequence is shorter than `frame_length`, `is_fill_short: true` pads it with zeros; otherwise the sample is treated as invalid. |
| Coordinate centering | `dataframeToNumpy()` | With `norm_centering_mode: First`, position data are centered by the first frame's `norm_center_point`. |
| Fold splitting | `makeSingleDataset()` | Uses `StratifiedGroupKFold(n_splits=5)` to preserve class balance while grouping samples by subject ID, derived from the part of `Directory` before `_`. |
| Split output | `makeSingleDataset()` | Writes `train/data.npy`, `train/target.npy`, `train/trainDir.csv`, `train/train_distribution.csv`, `test/data.npy`, `test/target.npy`, `test/testDir.csv`, and `test/test_distribution.csv` for each fold. |

### 3. DataLoader And Sampling

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Dataset initialization | `src/data/feeder/feeder.py` `Feeder.__init__()` | Loads skeleton data and labels from `.npy`, plus `Directory` and `RealScore` from `trainDir.csv` or `testDir.csv`. Stops if NaNs are found in the data. |
| Sample retrieval | `src/data/feeder/feeder.py` `Feeder.__getitem__()` | Returns `data, label, dir, real_score` to the PyTorch DataLoader. |
| Data augmentation | `src/data/feeder/tools.py` `random_move()` | During training only, applies a random movement transform with 50% probability when `preprocess.use_data_transform: true`. Evaluation keeps augmentation disabled. |
| Imbalance-aware sampling | `experiments/exam_each_seed.py` `main()` | Computes class counts from `train/target.npy` and uses `WeightedRandomSampler` when `trainings.is_weighted_random_sampler: true`. |
| DataLoader creation | `experiments/exam_each_seed.py` `main()` | Creates the train DataLoader with weighted sampling or shuffling, and creates the test DataLoader with `shuffle=False`. |

### 4. Model Training

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Train/evaluate entrypoint | `src/training/run.py` `Run()` | Constructs the model via `build_encoder()`, plus loss, optimizer, `Train`, and `Test`, then runs one fold of training and evaluation. |
| Model construction | `src/model/build_model.py` `build_encoder()` | Resolves `model.name` to `STGCLSATT` (`src/model/stgclsatt.py`) or a DeepRehabPile baseline (`src/model/baselines/`). |
| Optimizer creation | `src/training/run.py` `Run()` | Creates Adam, SGD, or AdamW according to `trainings.optimizer`, using `lr`, `momentum`, and `weight_decay`. |
| Training loop | `src/training/train.py` `Train.train()` | Runs forward pass, NLLLoss, optional AGC-LSTM attention regularization, backpropagation, optimizer step, and scheduler step for each epoch. |
| Scheduler setup | `src/training/train.py` `Train.set_scheduler()` | Configures `OneCycleLR`, `ReduceLROnPlateau`, `MultiStepLR`, or no scheduler. |
| Training log output | `src/training/train.py` `Train.train()` | Writes epoch-level `Epoch`, `TrainLoss`, `LearningRate`, and `TrainAccuracy` to `ELLA_ep<N>.csv`. |
| Checkpoint output | `src/training/train.py` `Train.train()` | Saves the final model `state_dict` to `checkpoint.pth`. |

### 5. Evaluation And Fold Outputs

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Evaluation entrypoint | `src/training/test.py` `Test.test()` | Loads `checkpoint.pth` and runs inference on the test DataLoader. |
| Metric calculation | `Test.test()` | Computes accuracy, balanced accuracy, per-class accuracy, F1, MCC, precision, recall, and AUCROC. Some metrics are logged to W&B when enabled. |
| Fold confusion matrix | `Test.test()` | Saves the fold-level rate confusion matrix as `confusion_matrix.png`. |
| Prediction CSV output | `Test.test()` | Writes `Directory`, `Predicted`, `True`, `RealScore`, and `Prob_*` columns to `PredictTruelabel.csv`. |
| Attention / CAM output | `Test.output_weights()` | Writes `weights_Att.h5` or `weights_CAM.h5` when `output_weights: true` and the model returns weights. |
| In-memory result collection | `src/data/feeder/results.py` `Results.append()` | Accumulates fold accuracy, balanced accuracy, per-class accuracy, predicted labels, and true labels. |

### 6. Aggregation And Cleanup

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Fold mean calculation | `src/data/feeder/results.py` `Results.appendMean()` | Computes the mean of five folds and prepares the `Mean` row for `Stratified_Seed<seed>/Accuracies.csv`. |
| Seed-level confusion matrices | `src/utils/make_confusion_matrix.py` `makeConfMatrixOnExam()` | Builds count and rate confusion matrices from all fold predictions for the seed and writes CSV plus `png/pdf/svg` outputs. |
| Seed result CSV | `experiments/exam_each_seed.py` `main()` | Writes `Stratified_Seed<seed>/Accuracies.csv` with fold rows 0-4 and a `Mean` row. |
| Action-to-item aggregation | `experiments/exam_each_seed.py` `main()` and `Results` | Aggregates each seed's fold mean into `<Action>_to_<FIMItem>/Accuracies.csv` and action-to-item confusion matrices. |
| Experiment-level aggregation | `experiments/exam_each_seed.py` `main()` | Concatenates all action-to-item results into `results/<experiment_name>/Accuracies.csv`, then writes `Seed == "Mean"` rows to `results.csv`. |
| Post-fold cleanup | `experiments/exam_each_seed.py` `main()` | Deletes `*_*.pth`, moves CSV files from `train/` and `test/` to the fold directory, and removes the temporary split directories. |

## Output Directory Structure

`<experiment_name>` is generated from the model name, ST-GCN layer count, seed-fixing mode, attention mode, and start timestamp.

```text
results/<experiment_name>/
├── situations.yaml
├── Accuracies.csv
├── results.csv
└── <Action>_to_<ItemIndex>-<FIMItem>/
    ├── Accuracies.csv
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.csv
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.pdf
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.svg
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.png
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.csv
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.pdf
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.svg
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.png
    └── Stratified_Seed<seed>/
        ├── Accuracies.csv
        ├── <Action>to<FIMItem>_<Model>_countConfMtx.*
        ├── <Action>to<FIMItem>_<Model>_rateConfMtx.*
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

After each fold finishes, `exam_each_seed.py` moves CSV files from the temporary `train/` and `test/` directories to the fold directory and removes the temporary split directories. The final fold directory therefore contains the CSV metadata and evaluation outputs shown above.

## Output File Details

### `situations.yaml`

A copy of the YAML config used for the run. It is written when `results/<experiment_name>/` is created and then made read-only. It preserves model, preprocessing, action/item selection, seed, class, and W&B settings for reproducibility.

### `results/<experiment_name>/Accuracies.csv`

Seed-level results for all `action -> FIM item` pairs.

| Column | Meaning |
| --- | --- |
| `Seed` | Seed value. Each pair also has a trailing `Mean` row. |
| `FromAction` | Input action name. |
| `ToFIM` | Target FIM item name. |
| `Accuracy` | Mean accuracy across the 5 folds for that seed, in percent. |
| `Balanced_Accuracy` | Mean balanced accuracy across the 5 folds, in percent. |
| `Class<score>_Accuracy` | Per-class accuracy for each configured FIM score class. |

### `results/<experiment_name>/results.csv`

The `Seed == "Mean"` rows from the root `Accuracies.csv`, with the `Seed` column removed. Use this file to compare average performance across action-to-item pairs.

### `<Action>_to_<ItemIndex>-<FIMItem>/Accuracies.csv`

Seed-level results for one action-to-FIM-item pair. Each seed row is the mean of that seed's five folds, and the final row is the mean across seeds.

### `Stratified_Seed<seed>/Accuracies.csv`

Fold-level results for one seed.

| Column | Meaning |
| --- | --- |
| `Fold` | Fold number `0` to `4`, plus a trailing `Mean` row. |
| `Accuracy` | Fold accuracy in percent. |
| `Balanced_Accuracy` | Fold balanced accuracy in percent. |
| `Class<score>_Accuracy` | Fold per-class accuracy for each configured FIM score class. |

### `Fold<fold>/PredictTruelabel.csv`

Sample-level predictions on the fold's test split.

| Column | Meaning |
| --- | --- |
| `Directory` | Source subject or trial directory. |
| `Predicted` | Predicted 0-based class index. |
| `True` | Ground-truth 0-based class index. |
| `RealScore` | Original FIM score before mapping into configured classes. |
| `Prob_0`, `Prob_1`, ... | Model output probability for each class index. The number of columns equals `len(classes)`. |

This file is also consumed by report scripts for cross-subject aggregation and ablation plotting. If the aggregation report is run, `PredictTruelabel_all_folds.csv` is written directly under `Stratified_Seed<seed>/`.

### `Fold<fold>/ELLA_ep<N>.csv`

Training log. `<N>` is the final epoch index.

| Column | Meaning |
| --- | --- |
| `Epoch` | Epoch number. |
| `TrainLoss` | Mean training loss for the epoch. |
| `LearningRate` | Learning rate at the end of the epoch. |
| `TrainAccuracy` | Training accuracy for the epoch. |

### `checkpoint.pth`

The trained model `state_dict` for the fold. In the normal `exam_each_seed.py` flow, the final single-action model remains directly under the fold directory.

### `trainDir.csv` / `testDir.csv`

Metadata for samples assigned to each split.

| Column | Meaning |
| --- | --- |
| `Directory` | Source data directory. |
| `FrameStart` | Start frame of the extracted moving window. |
| `RealScore` | Original FIM score. |

### `train_distribution.csv` / `test_distribution.csv`

Class-count summaries for each split. The index contains configured class scores plus `Total`; the values are the sample counts.

### `confusion_matrix.png`

Fold-level rate confusion matrix. The x-axis is predicted FIM score and the y-axis is true FIM score. The title includes `Accuracy` and `Balanced Accuracy`.

### `*_countConfMtx.*`

Aggregated confusion matrices at the seed level or action-to-item level. The `.csv` file stores counts, while `.png/.pdf/.svg` store the heatmap.

### `*_rateConfMtx.*`

Row-normalized versions of the aggregated confusion matrices. Values are percentages within each true class.

### `weights_Att.h5` / `weights_CAM.h5`

Written only when `output_weights: true` and the selected model returns Attention or GradCAM weights. HDF5 keys correspond to `Directory` values in the test split. The stored arrays are model-specific spatio-temporal weight tensors.
