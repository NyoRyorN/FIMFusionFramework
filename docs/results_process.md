# Post-Experiment Processing Pipeline

This document describes the post-experiment processing scripts
`src/results_process.py` and `src/results_process_batch.py`.

## Overview

After training completes, the experiment root directory contains raw CSV files
and checkpoint files. The post-processing pipeline:

1. **DeleteAfter** — removes `.pth` checkpoint files to free disk space
2. **SummaryOutPuts** — aggregates per-seed/per-fold results into a summary CSV
3. **all_calc_indicator** *(mode C only)* — calculates additional classification
   metrics from raw prediction CSVs
4. **BoxPlotsEachClass** — generates box plots and bar plots of accuracy per
   FIM item and per class

---

## Scripts

### `src/results_process.py` — Single-directory processing

Runs the full pipeline on one experiment result directory.

**Entry point (interactive):**

```bash
python -m src.results_process
```

Prompts:

| Prompt | Description |
|---|---|
| `rootPath` | Path to the experiment result directory |
| `mode` | `N` (Normal: 3-class, score 1–5 vs 6 vs 7), `A` (Assistance: 2-class, 1–5 vs 6–7), `C` (Complete independence: 2-class, 1–6 vs 7). Select to match the class configuration used during training. |
| `flag_deleteAfter` | `y` to delete `.pth` files, `n` to skip |
| `flag_summaryOutputs` | `y` to run `SummaryOutPuts` (and `all_calc_indicator` for mode C) |
| `flag_detailPlots` | `y` to run `BoxPlotsEachClass` |
| Y-axis range for plots | `y` to fix Y-axis at 0–100, `n` to auto-scale |

**Programmatic use:**

```python
from src.results_process import ResultsProcess

ResultsProcess(
    rootPath="/path/to/results/experiment_name",
    mode="C",
    flag_deleteAfter="y",
    flag_summaryOutputs="y",
    flag_detailPlots="y",
    range_detailPlots="y",   # "y" = 0-100 fixed; anything else = auto-scale
)
```

---

### `src/results_process_batch.py` — Batch processing across multiple directories

Applies `ResultsProcess` to every immediate subdirectory of a given root.
Useful when multiple experiment result directories have been moved under a
single parent directory for comparison.

**Usage:**

```bash
python -m src.results_process_batch
```

Prompts:

| Prompt | Description |
|---|---|
| `rootDir` | Parent directory containing experiment subdirectories |
| `mode` | `N` / `A` / `C` — must match the class configuration used during training |
| `flag_deleteAfter` | `y` / `n` |
| `flag_summaryOutputs` | `y` / `n` |
| `flag_detailPlots` | `y` / `n` |

> **Note:** Directories whose name contains `"AblationPlots"` are skipped
> automatically. The Y-axis for plots defaults to auto-scale.

**Typical workflow for batch processing:**

```
results/
├── experiment_A/      ← processed
├── experiment_B/      ← processed
└── AblationPlots/     ← skipped automatically
```

```bash
# Move experiment results into a parent directory
mv results/experiment_A /path/to/my_experiments/
mv results/experiment_B /path/to/my_experiments/

# Then run batch processing
python -m src.results_process_batch
# Enter rootDir: /path/to/my_experiments
```

---

## Processing steps in detail

### Step 1 — DeleteAfter (`src/utils/delete_after.py`)

Recursively walks `rootPath` with `os.walk` and deletes every file with a
`.pth` extension. Each deleted path is printed to stdout. If no `.pth` files
are found, a message is printed and the step exits cleanly. Model checkpoints
are no longer needed after training and can occupy significant disk space.

---

### Step 2 — SummaryOutPuts (`src/plot/summary_outputs.py`)

Reads `Accuracies.csv` from `rootPath` and filters out the summary row
(`Seed == "Mean"`), leaving only per-seed/per-fold rows.

#### Single-action experiments (`FromAction` column present)

**Cross-item bar plot** (`PlotCrossItemPlots`):

1. For each (Action, Seed) pair, compute the mean of `Balanced_Accuracy` across
   all FIM items → saved as `CrossItem_Balanced_Accuracy_Data.csv`.
2. Group those seed-level means by Action → compute mean and std across seeds →
   saved as `CrossItem_Balanced_Accuracy_Stats.csv`.
3. Draw a bar plot (mean ± std per action, overall mean as a red dashed line)
   sorted in the canonical action order defined by `plotTasks()` →
   saved as `CrossItem_Balanced_Accuracy_BarPlot_MeanStd.{png,pdf,svg}`.

**Per-item statistics** (written to `Summary.csv`):

For each FIM item the following columns are computed:

| Column | Calculation |
|---|---|
| `Accuracy_Mean` | Mean over all rows for that item (all seeds × all actions) |
| `Accuracy_Std` | Per-seed average across actions first, then std of those seed means |
| `Acc_Max_Action` | Action with the highest mean Accuracy for that item |
| `Acc_Max_Accuracy` | Mean Accuracy of the best action |
| `Acc_Max_Std` | Std of Accuracy for the best action across seeds |
| `Balanced_Accuracy_Mean` | Same as above for Balanced Accuracy |
| `Balanced_Accuracy_Std` | Same as above for Balanced Accuracy |
| `Max_Balanced_Acc_Action` | Action with the highest mean Balanced Accuracy |
| `Max_Balanced_Acc_Accuracy` | Mean Balanced Accuracy of that best action |
| `Max_Balanced_Acc_Std` | Std of Balanced Accuracy for that best action |

The final row (`FIMItem = "All"`) summarises across all items. For the "Max"
columns, the per-item best actions are used: scores for those (item, best-action)
combinations are averaged per seed first, then mean and std are taken across
seeds (`CalcAllMaxActionStd`).

#### Multi-action Fusion experiments (`FromAction` column absent)

For each FIM item, mean and std are computed for every numeric column in
`Accuracies.csv` (all columns except `Seed` and `ToFIM`). The std for the
"All" row uses the same seed-level averaging as the single-action case
(`CalcAllStd`). Results are saved to `Summary.csv`.

---

### Step 3 — all_calc_indicator (`src/plot/calc_indicator.py`) — mode C only

Triggered only in mode `C` (2-class: score 1–6 vs 7). Calculates a richer set
of classification metrics by reading `PredictTruelabel.csv` directly from each
Fold directory.

**Directory traversal:**

```
rootPath/
└── <NN>-<Item>                         → action = "All",    item = <Item>
    or <NN>-<Action>_to_<NN>-<Item>     → action = <Action>, item = <Item>
    └── Stratified_Seed<seed>/
        └── Fold<fold>/
            └── PredictTruelabel.csv    ← source
```

**Metrics computed per fold** (`calc_indicator`):

| Metric | Description |
|---|---|
| `Accuracy` | Overall classification accuracy |
| `BalancedAccuracy` | Balanced accuracy (mean recall per class) |
| `AccClass6` | Accuracy on samples with FIM score ≤ 6 (class 1–6) |
| `AccClass7` | Accuracy on samples with FIM score > 6 (class 7) |
| `F1Score` | Binary F1 score (positive class = 0, i.e. class 1–6) |
| `MCC` | Matthews correlation coefficient |
| `Precision` | Binary precision (positive class = 0) |
| `AUCROC` | AUC-ROC computed from `Prob_0` column (probability for class 1–6) |

**Aggregation pipeline:**

1. All per-fold rows are saved to `Indicator_Lists.csv`.
2. `aggregate_indicators_each_fold`: groups by (Action, FIMItem, Seed) and takes
   the mean across folds → saved to `Aggregated_Indicator_Lists.csv`.
3. `summary_indicator` produces two outputs depending on the number of unique
   actions:

   **Single action (`action = "All"` only)**:
   - Groups by FIMItem → mean and std across seeds.
   - Adds a `MeanOfItems` row: for each seed, average all items first, then
     compute mean and std of those seed means.
   - Saved to `Summary_Indicator_Lists.csv`. `Each_Summary_Indicator_Lists.csv`
     is not produced.

   **Multiple actions**:
   - Groups by (Action, FIMItem) → mean and std across seeds →
     `Each_Summary_Indicator_Lists.csv`.
   - Selects the best action per FIM item (highest `BalancedAccuracy_Mean`) →
     forms `BestAction` rows.
   - `BestOfActions` row: for each seed, average the per-item best-action scores
     across items, then compute mean and std.
   - `MeanOfActions` rows: for each (FIMItem, Seed), average across all actions
     first, then aggregate across seeds. Includes a `MeanOfItems` row computed
     the same way.
   - All rows are concatenated and saved to `Summary_Indicator_Lists.csv`.

---

### Step 4 — BoxPlotsEachClass (`src/plot/box_plots_each_class.py`)

Reads `Accuracies.csv`, renames class columns according to mode, and saves all
plots to `rootPath/results/`.

**Mode and class column mapping:**

| Mode | Classification | Class columns renamed to | Box colors |
|---|---|---|---|
| `N` (Normal) | 3-class: score 1–5 vs 6 vs 7 | `Class:1-5`, `Class:6`, `Class:7` | lightcoral, Yellow, Lime |
| `A` (Assistance) | 2-class: score 1–5 vs 6–7 | `Class:1-5`, `Class:6-7` | lightcoral, Yellow, Lime |
| `C` (Complete independence) | 2-class: score 1–6 vs 7 | `Class:1-6`, `Class:7` | Yellow, Lime |

#### Single-action experiments (`FromAction` column present)

For each FIM item, three plots are generated:

1. **Box plot of Balanced Accuracy** (`{item}.{png,pdf,svg}`):
   Seaborn boxplot with x = action (sorted by `plotTasks()` order),
   y = `Balanced_Accuracy`. Shows mean marker (red `+`), median (dashed black
   line), whiskers, and individual outliers.

2. **Bar plot of Balanced Accuracy** (`{item}_BarPlot_MeanStd.{png,pdf,svg}`):
   Groups by action → computes mean ± std across seeds. Mean values are
   annotated above each bar. A red dashed horizontal line marks the overall
   mean. Y-axis is auto-scaled to the data range.

3. **Class-wise accuracy box plot** (`{item}_ClassWiseAccuracies.{png,pdf,svg}`):
   Reshapes class columns to long format and draws a grouped Seaborn boxplot
   with x = action, hue = class. Y-axis is either 0–100 (if
   `range_detailPlots == "y"`) or auto-scaled to the nearest 5-point boundary.

#### Multi-action Fusion experiments (`FromAction` column absent)

Two plots are generated over all FIM items:

1. **Box plot of Balanced Accuracy** (`EachItemBalancedAccuracies.{png,pdf,svg}`):
   x = FIM item (`ToFIM`), y = `Balanced_Accuracy`. Same boxplot style as above.

2. **Class-wise accuracy box plot** (`All_ClassWiseAccuracies.{png,pdf,svg}`):
   x = FIM item, hue = class. Same class renaming and Y-axis logic as above.

All plots are saved in three formats: `.png`, `.pdf`, and `.svg`.

---

## Output files

All output is written relative to the experiment result directory (`rootPath`).

### Always generated (when corresponding flags are set)

| File | Location | Description |
|---|---|---|
| *(`.pth` files deleted)* | `rootPath/**/*.pth` | Checkpoint files removed |
| `Summary.csv` | `rootPath/` | Per-item accuracy statistics (mean, std, best action) |

### Generated only when `FromAction` is present (single-action experiments)

| File | Location | Description |
|---|---|---|
| `CrossItem_Balanced_Accuracy_Data.csv` | `rootPath/results/` | Per-seed per-action cross-item mean balanced accuracy |
| `CrossItem_Balanced_Accuracy_Stats.csv` | `rootPath/results/` | Action-wise mean ± std of cross-item balanced accuracy |
| `CrossItem_Balanced_Accuracy_BarPlot_MeanStd.{png,pdf,svg}` | `rootPath/results/` | Bar plot of cross-item balanced accuracy |
| `{item}.{png,pdf,svg}` | `rootPath/results/` | Box plot of Balanced Accuracy per action for each FIM item |
| `{item}_BarPlot_MeanStd.{png,pdf,svg}` | `rootPath/results/` | Bar plot (mean ± std) of Balanced Accuracy per action |
| `{item}_ClassWiseAccuracies.{png,pdf,svg}` | `rootPath/results/` | Class-wise accuracy box plot per action |

### Generated only for multi-action Fusion experiments (no `FromAction`)

| File | Location | Description |
|---|---|---|
| `EachItemBalancedAccuracies.{png,pdf,svg}` | `rootPath/results/` | Box plot of Balanced Accuracy across FIM items |
| `All_ClassWiseAccuracies.{png,pdf,svg}` | `rootPath/results/` | Class-wise accuracy box plot across all FIM items |

### Generated only for mode C (`all_calc_indicator`)

| File | Location | Description |
|---|---|---|
| `Indicator_Lists.csv` | `rootPath/` | Raw per-fold indicators (Action, FIMItem, Seed, Fold, metrics) |
| `Aggregated_Indicator_Lists.csv` | `rootPath/` | Fold-aggregated per-seed indicators (mean across folds) |
| `Summary_Indicator_Lists.csv` | `rootPath/` | Seed-aggregated summary (mean ± std), rows for MeanOfActions and BestOfActions |
| `Each_Summary_Indicator_Lists.csv` | `rootPath/` | Per-Action × FIMItem indicator summary (mean ± std across seeds) |

### `Summary_Indicator_Lists.csv` row types

| `BestAction` value | Description |
|---|---|
| `MeanOfActions` | Metrics averaged over all actions, then over seeds |
| `<action_name>` | Best action per FIM item (highest mean Balanced Accuracy) |
| `BestOfActions` | Cross-item mean using the per-item best action |
