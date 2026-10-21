# exam_fusion.py: Multi-Action Fusion FIM Prediction Experiment

`experiments/exam_fusion.py` runs multi-action fusion experiments for FIM item prediction. It treats the actions listed in `combined_action_indices` as one multi-action input set and trains Early Fusion, Middle Fusion, or Late Fusion models according to `fusion.type`. All fusion models train end-to-end in a single pass, combining per-action outputs by averaging over the action axis.

## How To Run

In the current checkout, fusion experiment configs live under `config/fusion_strategy_ablations/`, `config/model_ablation/`, `config/action_selection/`, and `config/complement_ablation/`. The script still has `config/configSomeTask.yaml` as its parser default, but that file is not part of the standard config tree, so pass `-c` explicitly.

```bash
# Early Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_data.yaml

# Middle Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_shallow.yaml

# Late Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_feature.yaml
```

Pass `--output` / `-o` to save results under a directory other than the default `results/`:

```bash
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_data.yaml -o /path/to/output
```

When `wandb.use_wandb: true`, each FIM item, seed, and fold run is logged to Weights & Biases. The W&B config includes fusion type, fusion mode, fold index, target FIM item, seed, preprocessing settings, and training settings.

## Key Config Fields

| Key | Meaning |
| --- | --- |
| `fusion.type` | The fusion strategy, naming both the family and its variant: `early_chn_stack` / `early_exp_graph` (input level, channel stack or graph expansion), `middle_shallow` / `middle_deep` (intermediate features, insertion point), `late_feature` / `late_probability` (pre-FC features or per-action output probabilities). |
| `fusion.missing_representation` | How missing actions are represented: `zero` (zero padding, default) or `pad_mask` (Late Fusion only: each branch is forwarded only over present samples so missing slots never touch the encoder). See "Missing-Action Representation" below. |
| `trainings.modality_dropout_p` | Modality dropout probability (default `0.0` = disabled). During training only, each present action is additionally marked as missing with this probability; at least one action is always kept per sample. |
| `combined_action_indices` | Action indices to combine. They correspond to `Tasks()` in `src/utils/names.py`. |
| `fim_item_indices` | Target FIM item indices. |
| `trainings.delete_model_after_training` | If `true`, removes some temporary models and intermediate directories after each fold and moves split CSV files to the fold directory. |
| `output_weights` | Enables Attention or GradCAM weight output when the model exposes weights. |

## Missing-Action Representation And Modality Dropout

Missing actions are stored as all-zero `(C, T, V)` slices by `makeSomeTasksDataset()`. `Feeder` derives a per-sample per-action availability mask (1 = present, 0 = missing) from these all-zero slices and returns it as the fifth item of each sample. The `Train`/`Test` loops forward the mask only to fusion models (`accepts_mask = True`); the single-action `STGCLSATT` path is unchanged.

Two representations are selectable via `fusion.missing_representation` (helpers in `src/model/mask_token.py`):

| Method | Behavior |
| --- | --- |
| `zero` | Baseline. Missing slots stay all-zero; pseudo-missing slots from modality dropout are also zeroed. |
| `pad_mask` | Late Fusion only. Each per-action branch is forwarded only for present samples (`f[present] = model_list[a](x[present, a])`); missing slots never reach the encoder, so BatchNorm statistics are never contaminated and no gradient flows through them. Aggregated with a masked mean. |

`trainings.modality_dropout_p > 0` enables modality dropout: during training, present actions are randomly marked as pseudo-missing and merged with the real mask into an effective mask consumed by the active representation. At least one action is always kept per sample; inference uses only the real mask. Experiment names use `_ZeroPad` or `_Masking` for the representation and add `_MD` when modality dropout is enabled. Complete-case preprocessing uses `_CompleteCase`.

Known limitations: per-action BatchNorm running statistics still absorb zero slices (the zero-padding baseline; `pad_mask` is the exception, see above), and GradCAM re-invokes the model without a mask, so CAMs are computed under all-present semantics. Prefer `output_weights: false` for representation-comparison runs.

## Experiment Flow

1. Load the YAML config and resolve the combined actions, target FIM items, seeds, and fusion settings.
2. Create `results/<experiment_name>/` and copy the config to `situations.yaml`.
3. For each FIM item, run `makeSomeTasksDataset()` to build 5-fold train/test splits containing all selected actions.
4. Train the fusion model end-to-end on the combined-action split, then evaluate it on the test split.
5. Aggregate fold results into seed results, seed results into FIM-item results, and all items into experiment-level summary files.

## Internal Runtime Processing

### 1. Entrypoint And Config Loading

| Processing Step | File / Function | Details |
| --- | --- | --- |
| CLI parsing | `experiments/exam_fusion.py` `__main__` | Parses `--config` / `-c` and `--output` / `-o` with `argparse` and calls `main(configPath=..., outputDir=...)`. |
| Experiment orchestration | `experiments/exam_fusion.py` `main()` | Controls YAML loading, FIM item, seed, and fold loops, fusion training, evaluation, aggregation, and output writing. |
| Plot font setup | `src/utils/set_plt_font.py` `setPltFont()` | Initializes matplotlib font settings before figures are created. |
| Action and FIM name resolution | `src/utils/names.py` `Tasks()` / `FIMitems()` | Converts `combined_action_indices` and `fim_item_indices` into action names and FIM item names. |
| Experiment name generation | `src/utils/decides.py` `decideExamName()` | Builds `results/<experiment_name>/` from model, ST-GCN layer count, seed-fixing mode, fusion type/mode, attention setting, and timestamp. |
| Seed control | `src/utils/decides.py` `decideSeed()` | Sets split and model seeds according to `seed_fix_mode`, including Python, NumPy, PyTorch, and CUDA seed state. |
| W&B config construction | `src/utils/decides.py` `decideWandbConfig()` | Builds a W&B config dict containing seed, fold, target FIM item, fusion settings, combined actions, preprocessing settings, model settings, and training settings. |
| Model dispatch | `src/training/run.py` `_build_model()` | Builds the model selected by `config["fusion"]["type"]` (Early/Middle/Late Fusion); for Middle Fusion also splits ST-GCN channels into independent and post-fusion channel lists. |
| Middle Fusion channel split | `src/model/params/separate_channels.py` `separate_channels()` | Splits `model.st-gcn.output_channel_list` according to the insertion point in `fusion.type` (`middle_shallow` / `middle_deep`), deciding which layers run before and after feature fusion. |

### 2. Multi-Action Dataset Creation

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Dataset creation entrypoint | `src/data/dataset/make_some_tasks_dataset.py` `makeSomeTasksDataset()` | For one FIM item, loads all actions in `combined_action_indices`, creates multi-action arrays, and writes 5-fold splits. |
| Input data root selection | `makeSomeTasksDataset()` | The input root is always `data/ConciseData`; centering is applied at load time (`norm_centering_mode: First`). |
| Subject directory selection | `makeSomeTasksDataset()` | Uses the `Directory` column from `data_list_path` when available; otherwise scans the selected data root. |
| FIM score loading | `makeSomeTasksDataset()` | Reads each subject's `FIM.csv`, extracts the target FIM item's `Score`, and skips invalid scores. |
| Class-label mapping | `makeSomeTasksDataset()` | Maps the original FIM score into a 0-based class index according to `classes`. |
| Missing-action handling | `makeSomeTasksDataset()` | If a selected action is missing, `is_fill_actionless` and `complement_actionless` determine whether to substitute a fallback action (via `preprocess.complement_map`, an ordered per-task candidate list; defaults to the original seated/standing pairing when unset), fill zeros, or drop the sample. |
| Skeleton CSV conversion | `src/data/dataset/utils/dataframe_to_numpy.py` `dataframeToNumpy()` | Reads position / velocity / acceleration axis CSVs for each action and converts them into fixed-length arrays. |
| Movement onset detection | `src/data/dataset/utils/detect_movement_onset.py` `detectMovementOnset()` | When `use_moving_frame: true`, estimates the start frame from velocity signals. |
| Multi-action stacking | `makeSomeTasksDataset()` | Stacks each subject's per-action arrays along the action axis to form the fusion model input. |
| Fold splitting | `makeSomeTasksDataset()` | Uses `StratifiedGroupKFold(n_splits=5)` to preserve class balance while grouping by subject ID. |
| Split output | `makeSomeTasksDataset()` | Writes `train/data.npy`, `train/target.npy`, `train/trainDir.csv`, `train/train_distribution.csv`, `test/data.npy`, `test/target.npy`, `test/testDir.csv`, and `test/test_distribution.csv` for each fold. |

### 3. DataLoader And Sampling

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Dataset initialization | `src/data/feeder/feeder.py` `Feeder.__init__()` | Loads skeleton arrays, labels, `Directory`, and `RealScore` from `.npy` and `.csv` files, and derives the per-sample per-action availability mask from all-zero slices. |
| Sample retrieval | `src/data/feeder/feeder.py` `Feeder.__getitem__()` | Returns `data, label, dir, real_score, mask`. In fusion experiments, `data` contains multiple actions and `mask` flags which actions are present. |
| Data augmentation | `src/data/feeder/tools.py` `random_move()` | During training, applies a random movement transform with 50% probability when `preprocess.use_data_transform: true`. |
| Imbalance-aware sampling | `experiments/exam_fusion.py` `main()` | Computes class counts from `target.npy` and uses `WeightedRandomSampler` when `trainings.is_weighted_random_sampler: true`. |
| DataLoader creation | `experiments/exam_fusion.py` `main()` | Creates the fusion train/test loaders from the combined-action split. |

### 4. Fusion Model Training

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Fusion training entrypoint | `src/training/run.py` `Run()` | Builds the model according to `fusion.type`, trains it end-to-end, and evaluates it. Shared with the single-model driver (`exam_each_seed.py`). |
| Late Fusion model | `src/model/late_fusion.py` `LateFusion` | Averages action outputs (`late_probability`), or pre-FC features (`late_feature`). |
| Middle Fusion model | `src/model/middle_fusion.py` `MiddleFusion` | Processes each action through early ST-GCN blocks, fuses intermediate features, and passes them through post-fusion ST-GCN blocks and the classifier. |
| Early Fusion model | `src/model/early_fusion.py` `EarlyFusion` | Combines actions at the input level using channel stacking (`early_chn_stack`) or graph expansion (`early_exp_graph`). |
| Fusion training loop | `src/training/train.py` `Train.train()` | Runs epoch-level forward pass, loss, backpropagation, optimizer step, and scheduler step using `trainings` hyperparameters. |
| Missing-action mask handling | `src/training/train.py` `Train.train()` and `src/model/mask_token.py` | Applies `apply_modality_dropout()` to the availability mask and passes the effective mask to the fusion model's `forward(x, mask)`. |
| Fusion training log | `src/training/train.py` `Train.train()` | Writes `ELLA_ep<N>.csv` directly under the fold directory. |
| Fusion checkpoint output | `src/training/train.py` `Train.train()` | Saves the final fusion model `state_dict` to `checkpoint.pth`. |

### 5. Evaluation And Fold Outputs

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Evaluation entrypoint | `src/training/test.py` `Test.test()` | Loads `checkpoint.pth` and runs inference on the test DataLoader. Fusion models receive the real availability mask (no modality dropout at inference). |
| Metric calculation | `Test.test()` | Computes accuracy, balanced accuracy, per-class accuracy, F1, MCC, precision, recall, and AUCROC. |
| Fold confusion matrix | `Test.test()` | Saves the fold-level rate confusion matrix as `confusion_matrix.png`. |
| Prediction CSV output | `Test.test()` | Writes `Directory`, `Predicted`, `True`, `RealScore`, and `Prob_*` columns to `PredictTruelabel.csv`. |
| Attention / CAM output | `Test.output_weights()` | Writes `weights_Att.h5` or `weights_CAM.h5` when `output_weights: true` and the model returns weights. Fusion models create one group per `Directory`, with action-index datasets under each group. |
| In-memory result collection | `src/data/feeder/results.py` `Results.append()` | Accumulates fold accuracy, balanced accuracy, per-class accuracy, predicted labels, and true labels. |

### 6. Aggregation And Cleanup

| Processing Step | File / Function | Details |
| --- | --- | --- |
| Fold mean calculation | `src/data/feeder/results.py` `Results.appendMean()` | Computes the mean of five folds and prepares the `Mean` row for `Stratified_Seed<seed>/Accuracies.csv`. |
| Seed-level confusion matrices | `src/utils/make_confusion_matrix.py` `makeConfMatrixOnExam()` | Builds count and rate confusion matrices from all fold predictions for the seed and writes CSV plus `png/pdf/svg` outputs. |
| Seed result CSV | `experiments/exam_fusion.py` `main()` | Writes `Stratified_Seed<seed>/Accuracies.csv` with fold rows 0-4 and a `Mean` row. |
| FIM-item aggregation | `experiments/exam_fusion.py` `main()` and `Results` | Aggregates each seed's fold mean into `<ItemIndex>-<FIMItem>/Accuracies.csv` and FIM-item confusion matrices. |
| Experiment-level aggregation | `experiments/exam_fusion.py` `main()` | Concatenates all FIM item results into `results/<experiment_name>/Accuracies.csv`, then writes `Seed == "Mean"` rows to `results.csv`. |
| Optional cleanup | `experiments/exam_fusion.py` `main()` | If `trainings.delete_model_after_training: true`, removes temporary directories and moves split CSV files to the fold directory. |

## Output Directory Structure

`<experiment_name>` is generated from the model name, ST-GCN layer count, seed-fixing mode, fusion settings, attention mode, and start timestamp.

```text
results/<experiment_name>/
├── situations.yaml
├── Accuracies.csv
├── results.csv
└── <ItemIndex>-<FIMItem>/
    ├── Accuracies.csv
    ├── <FIMItem>_<Model>_countConfMtx.csv
    ├── <FIMItem>_<Model>_countConfMtx.pdf
    ├── <FIMItem>_<Model>_countConfMtx.svg
    ├── <FIMItem>_<Model>_countConfMtx.png
    ├── <FIMItem>_<Model>_rateConfMtx.csv
    ├── <FIMItem>_<Model>_rateConfMtx.pdf
    ├── <FIMItem>_<Model>_rateConfMtx.svg
    ├── <FIMItem>_<Model>_rateConfMtx.png
    └── Stratified_Seed<seed>/
        ├── Accuracies.csv
        ├── <FIMItem>_<Model>_countConfMtx.*
        ├── <FIMItem>_<Model>_rateConfMtx.*
        └── Fold<fold>/
            ├── train/
            │   ├── data.npy
            │   ├── target.npy
            │   ├── trainDir.csv
            │   └── train_distribution.csv
            ├── test/
            │   ├── data.npy
            │   ├── target.npy
            │   ├── testDir.csv
            │   └── test_distribution.csv
            ├── checkpoint.pth
            ├── ELLA_ep<N>.csv
            ├── confusion_matrix.png
            ├── PredictTruelabel.csv
            ├── weights_Att.h5        # optional
            └── weights_CAM.h5        # optional
```

`<ActionIndex>` is the configured action index plus one, formatted as two digits. For example, action index `0` becomes `01`. It appears in the `use_<ActionName>` column of `trainDir.csv`/`testDir.csv` and in the per-action groups of `weights_Att.h5`/`weights_CAM.h5`.

If `trainings.delete_model_after_training: true`, intermediate directories such as `train/` and `test/` are removed after the fold. The split CSV files `trainDir.csv`, `testDir.csv`, `train_distribution.csv`, and `test_distribution.csv` are moved directly under the fold directory.

## Output File Details

### `situations.yaml`

A copy of the YAML config used for the run. It records the fusion type, fusion mode, combined actions, target FIM items, seeds, preprocessing settings, training settings, and W&B settings.

### `results/<experiment_name>/Accuracies.csv`

Seed-level results for all target FIM items.

| Column | Meaning |
| --- | --- |
| `Seed` | Seed value. Each FIM item also has a trailing `Mean` row. |
| `ToFIM` | Target FIM item name. |
| `Accuracy` | Mean accuracy across the 5 folds for that seed, in percent. |
| `Balanced_Accuracy` | Mean balanced accuracy across the 5 folds, in percent. |
| `Class<score>_Accuracy` | Per-class accuracy for each configured FIM score class. |

### `results/<experiment_name>/results.csv`

The `Seed == "Mean"` rows from the root `Accuracies.csv`, with the `Seed` column removed. Use this file to compare average FIM-item performance across fusion conditions.

### `<ItemIndex>-<FIMItem>/Accuracies.csv`

Seed-level results for one FIM item. Each seed row is the mean of that seed's five folds, and the final row is the mean across seeds.

### `Stratified_Seed<seed>/Accuracies.csv`

Fold-level results for one seed.

| Column | Meaning |
| --- | --- |
| `Fold` | Fold number `0` to `4`, plus a trailing `Mean` row. |
| `Accuracy` | Fold accuracy in percent. |
| `Balanced_Accuracy` | Fold balanced accuracy in percent. |
| `Class<score>_Accuracy` | Fold per-class accuracy for each configured FIM score class. |

### `Fold<fold>/train/data.npy`

Training split for the fusion model. It stores multi-action skeleton data with dimensions corresponding to samples, actions, channels, time, and joints. The exact shape depends on `attributes`, `frame_length`, skeleton layout, and `combined_action_indices`.

### `target.npy`

0-based class-label array for the train or test split. The original FIM scores remain in `trainDir.csv` and `testDir.csv` as `RealScore`.

### `trainDir.csv` / `testDir.csv`

Metadata for samples assigned to each split.

| Column | Meaning |
| --- | --- |
| `Directory` | Source data directory. |
| `<ActionName>` | Start frame of each action window when `use_moving_frame: true`. |
| `RealScore` | Original FIM score. |
| `use_<ActionName>` | The action actually used for the sample. Missing-action filling can record a substitute action or `None`. |

### `train_distribution.csv` / `test_distribution.csv`

Class-count summaries for each split. The index contains configured class scores plus `Total`.

### `Fold<fold>/ELLA_ep<N>.csv`

Training log for the fusion model. `<N>` is the final epoch index.

| Column | Meaning |
| --- | --- |
| `Epoch` | Epoch number. |
| `TrainLoss` | Mean training loss for the epoch. |
| `LearningRate` | Learning rate at the end of the epoch. |
| `TrainAccuracy` | Training accuracy for the epoch. |

### `checkpoint.pth`

The final fusion model `state_dict` for the fold. Evaluation loads this checkpoint.

### `PredictTruelabel.csv`

Sample-level predictions on the fold's test split.

| Column | Meaning |
| --- | --- |
| `Directory` | Source subject or trial directory. |
| `Predicted` | Predicted 0-based class index. |
| `True` | Ground-truth 0-based class index. |
| `RealScore` | Original FIM score. |
| `Prob_0`, `Prob_1`, ... | Model output probability for each class index. |

This file is used by cross-subject aggregation and ablation plotting report scripts. If the fold-concatenation report is run, `PredictTruelabel_all_folds.csv` is written directly under `Stratified_Seed<seed>/`.

### `confusion_matrix.png`

Fold-level rate confusion matrix. The x-axis is predicted FIM score and the y-axis is true FIM score. The title includes `Accuracy` and `Balanced Accuracy`.

### `*_countConfMtx.*`

Aggregated confusion matrices at the seed level or FIM-item level. The `.csv` file stores counts, while `.png/.pdf/.svg` store the heatmap.

### `*_rateConfMtx.*`

Row-normalized versions of the aggregated confusion matrices. Values are percentages within each true class.

### `weights_Att.h5` / `weights_CAM.h5`

Written only when `output_weights: true` and the selected model returns Attention or GradCAM weights. For fusion models, the HDF5 file stores one group per `Directory`; each group contains datasets keyed by the combined action index. Values are model-specific spatio-temporal weight tensors.
