# Ablation Box-plot Pipeline

This document describes the ablation comparison plot scripts in
`reports/ablation_boxplot/` and the batch wrapper `reports/ablation_process.py`.

## Overview

These scripts compare multiple ablation conditions side-by-side using box plots,
bar plots, and violin plots. Each ablation condition is one experiment result
directory that has already been processed by the post-processing pipeline.

| Script | Entry function | Input | Output directory |
| --- | --- | --- | --- |
| `ablation_sum_plots.py` | `ablationSumPlots()` | `Accuracies.csv`, `results.csv` | `<root>/AblationPlots/` |
| `ablation_plots_each_class.py` | `ablationPlotsEachClass()` | `Accuracies.csv`, `Summary.csv` | `<root>/AblationPlotsEachClass/` |
| `ablation_cross_subject_variables.py` | `ablationCrossSubjectVariables()` | `CrossSubjectVariables_all_items.csv`, `CrossSubjectVariables_action_to_item/` | `<root>/AblationPlotsCrossSubject/` |
| `ablation_process.py` *(wrapper)* | — | — | — |

See also:

- [Post-experiment processing: English](results_process.md)
- [Post-experiment processing: Japanese](results_process_ja.md)

---

## Prerequisites

### Post-processing pipeline

Each condition directory must contain the outputs of `src/results_process.py`
(or `src/results_process_batch.py`):

- `Accuracies.csv` — aggregated per-seed / per-fold metrics
- `Summary.csv` — per-item statistics including `Max_Balanced_Acc_Action`
- `results.csv` — summary rows extracted from `Accuracies.csv`

### Cross-subject aggregation (for `ablation_cross_subject_variables.py` only)

Run `reports/calculates/calc_cross_subject_variables.py` on every experiment
before running `ablation_cross_subject_variables.py`. This produces:

- `CrossSubjectVariables_all_items.csv` — per-subject match rates across all FIM items
- `CrossSubjectVariables_action_to_item/<Item>.csv` — per-subject per-action match rates
  (single-action experiments only)

---

## Directory Layout and `order.csv`

Create one **ablation root** directory and place the target experiment result
directories inside it. Each condition directory name must **start with** its
`order.csv` prefix followed by `_`:

```text
<ablation_root>/
├── order.csv                              ← manually created; one prefix per row
├── <prefix_A>_<rest_of_name>/             ← condition A
│   ├── Accuracies.csv
│   ├── Summary.csv
│   ├── results.csv
│   └── CrossSubjectVariables_all_items.csv     ← if using ablationCrossSubjectVariables
│   └── CrossSubjectVariables_action_to_item/   ← if using ablationCrossSubjectVariables
├── <prefix_B>_<rest_of_name>/             ← condition B
│   └── ...
└── <prefix_C>_<rest_of_name>/             ← condition C
    └── ...
```

`order.csv` is a headerless single-column CSV. Each row is one prefix in the
desired plot order (left to right on the x-axis):

```csv
prefix_A
prefix_B
prefix_C
```

The scripts match each prefix using `glob("<ablation_root>/<prefix>*")`, so the
prefix must be an unambiguous start of the directory name. The `_` separator
after the prefix is the project convention.

> **Note** — Directories whose names contain `"AblationPlots"` are automatically
> skipped by `src/results_process_batch.py`, so the ablation root can be placed under
> a parent processed by that script without creating a processing loop.

---

## Workflow

Move the target experiment result directories into the ablation root **before**
running post-processing, then run `src/results_process_batch.py` on the root to
post-process all conditions at once.

```
1. Create <ablation_root>/
2. Move experiment dirs in, naming each  <prefix>_<rest>
3. Create order.csv
4. python -m src.results_process_batch
   # Enter rootDir: <ablation_root>
   # (AblationPlots* directories are skipped automatically)
5. python reports/calculates/calc_cross_subject_variables.py --scan <ablation_root>
   # Required only for ablation_cross_subject_variables.py
6. python reports/ablation_process.py
   # Enter rootDir: <ablation_root>
   # Generates AblationPlots/ and AblationPlotsEachClass/
7. python reports/ablation_boxplot/ablation_cross_subject_variables.py <ablation_root>
   # Generates AblationPlotsCrossSubject/
```

---

## Scripts in detail

### `ablation_sum_plots.py` — Balanced-Accuracy plots

**Entry point:**

```bash
python reports/ablation_process.py      # recommended (also runs ablationPlotsEachClass)
# or standalone
python reports/ablation_boxplot/ablation_sum_plots.py
```

Prompts for the ablation root directory.

**Processing steps:**

1. Reads `order.csv` for condition prefixes and plot order.
2. For each FIM item in `FIMitems()[8:-1]`, iterates over conditions:
   - Reads `Accuracies.csv` from the matched condition directory.
   - **Single-action experiments** (`FromAction` column present):
     - **MeanAction** — for each seed, averages `Balanced_Accuracy` across all
       actions. Plot label: `<prefix>_MeanAction`.
     - **MaxAction** — reads `results.csv`, identifies the action with the
       highest `Balanced_Accuracy` for that item, then filters `Accuracies.csv`
       to that action. Plot label: `<prefix>_MaxAction`.
   - **Fusion experiments** (`FromAction` absent): uses `Balanced_Accuracy`
     directly. Plot label: `<prefix>`.
3. For each item, generates:
   - A box plot with x = condition, y = Balanced Accuracy, mean annotated in
     red above each box.
   - A bar plot of mean ± std across seeds, with mean values annotated.
4. After all items, averages Balanced Accuracy per (Seed, condition) across items
   and generates a cross-item summary box plot and bar plot.

**Output files** (written to `<root>/AblationPlots/`):

| File | Description |
| --- | --- |
| `result_all_<item>.csv` | Raw per-seed Balanced-Accuracy data for the item box plot |
| `Balanced_Accuracy_<item>.{png,pdf,svg}` | Box plot per item |
| `Balanced_Accuracy_<item>_BarPlot.{png,pdf,svg}` | Bar plot (mean ± std) per item |
| `Balanced_Accuracy_<item>_Stats.csv` | Descriptive statistics (mean, std) per condition |
| `result_all_CrossActions.csv` | Per-seed means averaged across all items |
| `Balanced_Accuracy_CrossActions.{png,pdf,svg}` | Cross-item box plot |
| `Balanced_Accuracy_CrossActions_BarPlot.{png,pdf,svg}` | Cross-item bar plot (mean ± std) |
| `Balanced_Accuracy_CrossActions_Stats.csv` | Cross-item statistics |

---

### `ablation_plots_each_class.py` — Class-wise accuracy plots

**Entry point:**

```bash
python reports/ablation_process.py      # recommended (also runs ablationSumPlots)
# or standalone
python reports/ablation_boxplot/ablation_plots_each_class.py
```

Prompts for the ablation root directory.

**Processing steps:**

1. Reads `order.csv` for condition prefixes and plot order.
2. For each FIM item in `FIMitems()[8:-1]`, iterates over conditions:
   - Reads `Accuracies.csv` from the matched condition directory.
   - **Single-action experiments** (`FromAction` column present):
     - **MaxAction** — reads `Summary.csv`, looks up `Max_Balanced_Acc_Action`
       for the item, filters `Accuracies.csv` to that action, and extracts
       class-wise accuracy columns. Plot label: `<prefix>_MaxAction`.
     - **MeanAction** — for each seed, averages class-wise accuracy columns
       across all actions. Plot label: `<prefix>_MeanAction`.
   - **Fusion experiments** (`FromAction` absent): extracts class-wise accuracy
     columns directly. Plot label: `<prefix>`.
   - Class columns are those whose names contain `"Class"`. They are reshaped
     to long format and treated as `situation` groups on the x-axis.
3. Draws a grouped box plot with x = class label, hue = condition, with a
   legend placed at the right of the plot.

**Output files** (written to `<root>/AblationPlotsEachClass/`):

| File | Description |
| --- | --- |
| `result_class-wise_<item>.csv` | Long-format per-seed class accuracy data |
| `Class-wise_Accuracy_<item>.{png,pdf,svg}` | Grouped box plot: x = class, hue = condition |

---

### `ablation_cross_subject_variables.py` — Per-subject match-rate plots

**Entry point:**

```bash
python reports/ablation_boxplot/ablation_cross_subject_variables.py <ablation_root>
# or interactive
python reports/ablation_boxplot/ablation_cross_subject_variables.py
```

**Processing steps:**

1. Reads `order.csv` from the ablation root if present; otherwise auto-discovers
   condition directories in alphabetical order, excluding `AblationPlots*`.
2. For each condition, reads:
   - `CrossSubjectVariables_all_items.csv` — per-subject match rates for each
     FIM item.
   - `CrossSubjectVariables_action_to_item/<Item>.csv` — per-subject per-action
     match rates (present for single-action experiments only).
3. Generates the **overall average plot** (`AverageMatchRate_AllItems`):
   - Per subject, averages match rate across all FIM item columns.
   - For single-action conditions, also adds `<prefix>_MeanAction`
     (`AverageMatchRate` across all actions per subject) and `<prefix>_MaxAction`
     (match rate of the per-item best action, averaged across items) variants.
   - Best action is determined by `find_best_action()`: checks `Summary.csv`
     first, then `results.csv` / `Accuracies.csv` as fallbacks.
4. Generates one **per-item plot** (`MatchRate_<Item>`) for each FIM item column
   found in any condition's all-items CSV.
   - For single-action conditions, also adds `MeanAction` and `MaxAction` variants
     from `CrossSubjectVariables_action_to_item/<Item>.csv`.

Each data point in the plots is one subject's match rate (not a seed aggregate).
Plots use a violin + box overlay with the mean annotated in red.

**Output files** (written to `<root>/AblationPlotsCrossSubject/`):

| File | Description |
| --- | --- |
| `AverageMatchRate_AllItems.csv` | Per-subject data used for the overall plot |
| `AverageMatchRate_AllItems_Stats.csv` | Descriptive statistics per condition |
| `AverageMatchRate_AllItems.{png,pdf,svg}` | Overall violin + box plot across all FIM items |
| `MatchRate_<Item>.csv` | Per-subject data for one FIM item |
| `MatchRate_<Item>_Stats.csv` | Per-item descriptive statistics per condition |
| `MatchRate_<Item>.{png,pdf,svg}` | Per-item violin + box plot |

---

### `ablation_process.py` — Wrapper

Runs `ablationSumPlots` and `ablationPlotsEachClass` in sequence. Does **not**
include `ablationCrossSubjectVariables`.

```bash
python reports/ablation_process.py
# Enter rootDir: <ablation_root>
```

---

## Output directory summary

| Output directory | Produced by | Contents |
| --- | --- | --- |
| `<root>/AblationPlots/` | `ablation_sum_plots.py` | Balanced-Accuracy box + bar plots per item and cross-item |
| `<root>/AblationPlotsEachClass/` | `ablation_plots_each_class.py` | Class-wise accuracy box plots per item |
| `<root>/AblationPlotsCrossSubject/` | `ablation_cross_subject_variables.py` | Per-subject match-rate violin + box plots |

All plot files are saved in three formats: `.png`, `.pdf`, `.svg`.

---

## Notes

- **FIM items scope** — `ablationSumPlots` and `ablationPlotsEachClass` process
  only `FIMitems()[8:-1]` (a predefined subset). `ablationCrossSubjectVariables`
  processes whatever item columns appear in the CSVs.
- **Prefix matching** — condition directories are located by
  `glob("<prefix>*")`. If multiple directories share a prefix, the first match
  in sorted order is used. Keep prefixes unambiguous.
- **Color palette** — up to 15 distinct base colors are available. Conditions
  beyond 15 reuse earlier colors.
- **MaxAction source differs by script** — `ablationSumPlots` reads `results.csv`;
  `ablationPlotsEachClass` reads `Summary.csv["Max_Balanced_Acc_Action"]`;
  `ablationCrossSubjectVariables` tries `Summary.csv` → `results.csv` →
  `Accuracies.csv` in that order.
