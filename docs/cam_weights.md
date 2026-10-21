# CAM Weight Visualization Pipeline

This document describes the CAM / GradCAM weight visualization scripts under
`reports/skeleton_plots/`:

- `makeWeightPlots.py`
- `makeWeightPlotsBulk.py`
- `makeWeightPlotsBulkCompact.py`
- `makeWeightDists.py`
- `plot2D.py`
- `plot3D.py`

## Overview

After an experiment writes model weights, these scripts visualize how each
skeleton joint contributes to the prediction.

1. **Frame-level skeleton videos/images** - overlay weights on the original
   2D or 3D skeleton sequence.
2. **Bulk fusion videos/images** - draw every available fusion action for one
   test subject in one combined 2D or 3D view.
3. **GradCAM representative-value CSVs** - summarize GradCAM weights by action,
   joint, and body-part group, then save seed-level CSV tables.

Supported experiment entrypoints are intentionally different by script:

| Script family | Supported experiment output |
|---|---|
| `makeWeightPlots.py` | `experiments/exam_each_seed.py` and `experiments/exam_fusion.py` |
| `makeWeightPlotsBulk.py` | `experiments/exam_fusion.py` only |
| `makeWeightPlotsBulkCompact.py` | `experiments/exam_fusion.py` only |
| `makeWeightDists.py` | `experiments/exam_fusion.py` only |

`plot2D.py` and `plot3D.py` are the low-level drawing helpers used by
`makeWeightPlots.py`. The bulk scripts contain their own combined-view drawing
functions, but they follow the same idea: build the skeleton graph from
`Graph(skl_type).neighbor_link`, normalize the weights to a red color map, and
write either MP4 video or per-frame SVG files.

---

## Preconditions

### Experiment settings

The experiment must have been run with weight output enabled.

```yaml
output_weights: true
model:
  cam_type: gradcam   # required when weights_CAM.h5 is needed
```

Attention weights are written only when the selected model exposes attention
weights. GradCAM weights are written only when `model.cam_type: gradcam` is set
and the model can run the GradCAM path.

### Required fold-level files

The visualization scripts read files from the experiment result directory.

| File | Required by | Description |
|---|---|---|
| `weights_Att.h5` | `makeWeightPlots.py` only | Attention weights. Not supported by the bulk or distribution scripts. |
| `weights_CAM.h5` | `makeWeightPlots.py`, plus the `exam_fusion.py`-only bulk and distribution scripts | GradCAM weights. Required by `makeWeightPlotsBulk.py`, `makeWeightPlotsBulkCompact.py`, and `makeWeightDists.py`. |
| `testDir.csv` | video/image scripts | Maps test `Directory` values to start frames or action-specific start-frame columns. `makeWeightDists.py` does not read it. |
| `PredictTruelabel.csv` | `makeWeightDists.py` | Supplies `Directory` and prediction-result columns for the representative-value CSVs. Prediction-probability columns whose names start with `Prob_` are not saved. |
| Skeleton CSVs | video/image scripts | `p_x.csv`, `p_y.csv`, and `p_z.csv` under the source skeleton data directory. |

### Script support limitations

`makeWeightPlotsBulk.py`, `makeWeightPlotsBulkCompact.py`, and
`makeWeightDists.py` are GradCAM-only and `exam_fusion.py`-only in the current
implementation. They read fusion-style `weights_CAM.h5` and do not process
`weights_Att.h5` or single-action `exam_each_seed.py` result directories.

`makeWeightPlots.py` is the flexible script: it tries `weights.h5`,
`weights_Att.h5`, and `weights_CAM.h5` if each file exists, and it has separate
paths for single-action and fusion outputs.

### Skeleton-data assumption

The current entrypoints default to the reduced 17-joint skeleton:

| Value | Default |
|---|---|
| `headers` | `SimpleJoints()` |
| `skl_type` | `simple_ntu-rgb+d` |
| source data | `data/ConciseData` |
| `is_center` | `False` |

The scripts also contain `plot_data_dict` entries for the full NTU layout and
centered variants, but the active `__main__` blocks are currently fixed to the
simple skeleton defaults.

Run commands from the repository root after loading the project environment:

```bash
source ./path.sh
```

---

## Scripts

### `reports/skeleton_plots/makeWeightPlots.py` - per-sample skeleton weight plots

Generates 2D and/or 3D videos or SVG frame sequences for each test sample and
each available weight file.

**Usage:**

```bash
python reports/skeleton_plots/makeWeightPlots.py
```

Prompts:

| Prompt | Description |
|---|---|
| root path | Experiment root whose basename starts with `Ex_...`. |
| task | Single-action mode only. Action index `1--11`. |
| item | FIM item index `0--12`. |
| seed | Seed string such as `100`. |
| K-Fold / fold index | Uses `Fold0` to `Fold4` when the root path contains `kfold`, or when the fusion prompt answers `y`. |
| mode | `2D`, `3D`, or `Both`. Invalid input falls back to `Both`. |
| output format | `mp4` or `svg`. Invalid input falls back to `mp4`. |

The script chooses the path pattern from the root path:

| Root path condition | Processing path |
|---|---|
| contains `multi-model` or `Step` | Fusion mode: `rootpath/<Item>/Stratified_Seed<seed>/[FoldN]/` |
| otherwise | Single-action mode: `rootpath/<Task>_to_<Item>/Stratified_Seed<seed>/[FoldN]/` |

It attempts these weight files in order:

| Shift | Input file | Output directory prefix |
|---|---|---|
| empty | `weights.h5` | `Weights/` |
| `_Att` | `weights_Att.h5` | `Weights_Att/` |
| `_CAM` | `weights_CAM.h5` | `Weights_CAM/` |

Missing weight files are skipped with `No weights: <path>`.

---

### `reports/skeleton_plots/makeWeightPlotsBulk.py` - combined GradCAM plots for `exam_fusion.py`

Generates one combined view per test subject from `exam_fusion.py` outputs. Each
available action stored in `weights_CAM.h5` is drawn as a separate subplot in
the same frame.

**Usage:**

```bash
python reports/skeleton_plots/makeWeightPlotsBulk.py
```

Prompts:

| Prompt | Description |
|---|---|
| root path | `exam_fusion.py` result root whose basename starts with `Ex_...`. |
| item | FIM item index `0--12`. |
| seed | Seed string such as `100`. |
| K-Fold / fold index | Optional `Fold0` to `Fold4`. |
| mode | `2D`, `3D`, or `Both`. |
| output format | `mp4` or `svg`. |

This script is GradCAM-only and supports only the fusion result layout produced
by `experiments/exam_fusion.py`. It reads:

```text
rootpath/
└── <Item>/
    └── Stratified_Seed<seed>/
        └── [FoldN]/
            ├── testDir.csv
            └── weights_CAM.h5
```

For each test `Directory`, the HDF5 group keys are treated as global action
indices. The script reads each corresponding skeleton sequence, trims it with
the action-specific start frame from `testDir.csv`, and normalizes all actions
for that subject with one shared min/max color scale.

---

### `reports/skeleton_plots/makeWeightPlotsBulkCompact.py` - compact combined GradCAM plots for `exam_fusion.py`

Runs the same GradCAM-only, `exam_fusion.py`-only workflow as
`makeWeightPlotsBulk.py`, but uses a more compact canvas:

| Difference | `makeWeightPlotsBulk.py` | `makeWeightPlotsBulkCompact.py` |
|---|---|---|
| 2D layout | Subplots, one axes per action | One shared axes; each action is translated to a grid position |
| 3D layout | Subplots, one axes per action | One shared 3D axes; each skeleton is translated to a grid position |
| Centering order | Centers before trimming | Trims first, then centers with the first plotted frame |
| Processing scope | Single item/seed/fold | Three modes: all seeds in an item dir, all seeds under a root dir, or single interactive |

Use this script when the subplot version wastes too much space or when a more
dense side-by-side comparison is easier to inspect.

**Usage:**

```bash
python reports/skeleton_plots/makeWeightPlotsBulkCompact.py
```

The script first asks for the plot mode and output format, then opens a mode
selection prompt:

| Prompt | Values | Description |
|---|---|---|
| Plot mode | `2D`, `3D`, `Both` (default) | Selects which combined views to generate. |
| Output format | `mp4` (default), `svg` | Video or per-frame SVG sequence. |

| Mode | Prompt | Description |
|---|---|---|
| **1** | Item directory path | Parent directory containing `Stratified_Seed*` subdirectories. All seeds found under this path are processed in order. K-Fold is auto-detected per seed. |
| **2** | Root directory path | Any ancestor directory. The script recursively finds every `Stratified_Seed*` directory that contains `weights_CAM.h5` (directly or inside `Fold*/`) and processes them all. |
| **3** | Root path → item → seed → fold | Original interactive mode: prompts for `exam_fusion.py` result root, FIM item index `0--12`, seed string, and optional fold index. |

K-Fold layout is **auto-detected** in Modes 1 and 2: if `Fold*` subdirectories
exist directly under `Stratified_Seed<seed>/`, they are enumerated and processed;
otherwise the seed directory itself is used as the single fold target.

**Output paths:**

| Mode | Output location |
|---|---|
| 1 or 2 | `Stratified_Seed<seed>/CAM_Animations/[FoldN/]<data_dir>_{2D,3D}_combined.{mp4,…}` |
| 3 | `rootpath/Weights_CAM/<Item>_Seed<seed>[_FoldN]/<data_dir>_{2D,3D}_combined.{mp4,…}` |

---

### `reports/skeleton_plots/makeWeightDists.py` - GradCAM distribution analysis for `exam_fusion.py`

This script also processes GradCAM weights from `exam_fusion.py` outputs, but it
does not generate frame-level skeleton videos. Instead, it vertically
concatenates subject rows from all folds and writes seed-level unnormalized
representative-value CSVs, label-group action violin plots, and correct-class
skeleton/body-part violin plots.

**Usage:**

```bash
python reports/skeleton_plots/makeWeightDists.py
```

The script opens with a mode selection prompt before asking for paths:

| Mode | Prompt | Description |
|---|---|---|
| **1** | Item directory path | Parent directory containing `Stratified_Seed*` subdirectories. All seeds found under this path are processed in order. |
| **2** | Root directory path | Any ancestor directory. The script recursively finds every `Stratified_Seed*` directory that contains `weights_CAM.h5` (directly or inside `Fold*/`) and processes them all. |
| **3** | Root path → item → seed | Original interactive mode: prompts for `exam_fusion.py` result root, FIM item index `0--12`, and seed string such as `100`. |

K-Fold layout is **auto-detected**: if `Fold*` subdirectories exist directly under
`Stratified_Seed<seed>/`, they are enumerated and processed; otherwise the seed
directory itself is used as the single fold target. No K-Fold prompt is shown.

This script is GradCAM-only and supports only the fusion result layout produced
by `experiments/exam_fusion.py`. It reads `weights_CAM.h5` and
`PredictTruelabel.csv`. Subject names come from the `Directory` column in
`PredictTruelabel.csv`; `testDir.csv` is not read.

The main difference from the video scripts is the aggregation level:

| Level | Column shape | Meaning |
|---|---|---|
| Action | `<Action>` | One scalar per subject and action, aggregating all frames and joints. |
| Joint | `<Action>_<Joint>` | One scalar per subject, action, and joint, aggregating over frames. |
| Body-part group | `<Action>_<Group>` | One scalar per subject, action, and body-part group. |

For each level, the script writes Max, Mean, and Median statistics under
`CAM_Dists`. The saved CSV values remain unnormalized; min-max normalization is
applied only when building the plots.

---

## Processing Details

### Step 1 - Read weight files

`makeWeightPlots.py` reads a single-action HDF5 dataset directly, or fusion HDF5
groups keyed by action index.

The bulk and distribution scripts read only the fusion-style `weights_CAM.h5`
produced by `exam_fusion.py`:

```text
weights_CAM.h5
└── <Directory>/
    ├── <ActionIndex>
    ├── <ActionIndex>
    └── ...
```

For `exam_fusion.py` outputs, the action-index key is converted through
`Tasks()` from `src/utils/names.py`.

### Step 2 - Align weights and skeleton frames

Weights are reshaped to `(T, V)` where `T` is frame count and `V` is joint count.
When the weight sequence is shorter than the default frame length (`150`), the
rows are repeated four times and center-cropped. This preserves Attention-style
short sequences in `makeWeightPlots.py`.

`makeWeightDists.py` does **not** apply this padding. The raw GradCAM output
length is used as-is so that representative statistics reflect the actual model
output without artificial frame duplication.

Skeleton coordinates are read from:

```text
<data_path>/<Directory>/<Task>/
├── p_x.csv
├── p_y.csv
└── p_z.csv
```

The frame start comes from `testDir.csv`:

| Experiment type | Start-frame column |
|---|---|
| Single-action `makeWeightPlots.py` | `FrameStart` |
| Fusion scripts | one column per action name |

### Step 3 - Draw 2D / 3D frame-level plots

`plot2D.py` creates a `networkx.Graph`, places each joint at `(x, y)`, draws
skeleton edges from `Graph(skl_type).neighbor_link`, maps the per-frame weight
values to the `Reds` color map, and writes either:

| Format | Output |
|---|---|
| `mp4` | one video named `<Task>_2D.mp4` |
| `svg` | one file per frame under `figures_2D/` |

`plot3D.py` uses the same skeleton edges and color mapping, but draws each frame
as a 3D scatter plot with coordinates `(x, z, y)`. It writes:

| Format | Output |
|---|---|
| `mp4` | one video named `<Task>_3D.mp4` |
| `svg` | one file per frame under `figures_3D/` |

The bulk scripts use equivalent combined-view functions:

| Function | Output |
|---|---|
| `PlotAllTasks2D_Combined` | `<Directory>_2D_combined.mp4` or `figures_2D_combined/frameNNN_2d_combined.svg` |
| `PlotAllTasks3D_Combined` | `<Directory>_3D_combined.mp4` or `figures_3D_combined/frameNNN_3d_combined.svg` |

### Step 4 - Calculate GradCAM representative-value CSVs

`makeWeightDists.py` reads `Directory` values from each fold-level
`PredictTruelabel.csv` and keeps only subjects that also exist as groups in
`weights_CAM.h5`. It does not read `testDir.csv`. Rows that appear in
`PredictTruelabel.csv` but not in `weights_CAM.h5` are skipped.

For each subject/action pair, the raw HDF5 array is first converted to
`new_weights` with shape `(T, V)`, where:

| Symbol | Meaning |
|---|---|
| `T` | Number of time frames in the raw GradCAM output, used as-is without padding or cropping. |
| `V` | Number of skeleton joints. The current entrypoint uses `SimpleJoints()` by default. |

The script does not apply frame-length adjustment or padding. NaN values in
`new_weights` propagate into the saved representative values without being
explicitly dropped.

The three representative values are computed from the adjusted `new_weights`
array without normalization:

| Statistic | Calculation | Output file |
|---|---|---|
| `Max` | Largest GradCAM value in the aggregation target. Captures the strongest single activation. | `GradCAM_Max.csv` |
| `Mean` | Arithmetic average over the aggregation target. Captures the overall magnitude. | `GradCAM_Mean.csv` |
| `Median` | Median over the aggregation target after flattening if needed. Captures a robust central value. | `GradCAM_Median.csv` |

Each CSV is a seed-level table made by vertically concatenating subject rows
from all folds. Subjects are expected not to overlap between folds; duplicated
subject indexes are reported as a warning but kept as-is.

| Column prefix | Column name | Aggregation target for one subject/action | Exact representative-value calculation |
|---|---|---|---|
| `Action_` | `Action_<Action>` | All frames and all joints, i.e. the full `(T, V)` matrix. | `Max = max(new_weights.values)`, `Mean = mean(new_weights.values)`, `Median = median(new_weights.values)`. |
| `Joint_` | `Joint_<Action>_<Joint>` | One joint column across all frames, i.e. a vector of length `T`. | `Max = max(new_weights[Joint])`, `Mean = mean(new_weights[Joint])`, `Median = median(new_weights[Joint])`. |
| `BodypartGroup_` | `BodypartGroup_<Action>_<Group>` | All frames and all joints belonging to one body-part group, i.e. a `(T, |Group|)` submatrix. | `Max = max(new_weights[group_joints].values)`, `Mean = mean(new_weights[group_joints].values)`, `Median = median(new_weights[group_joints].values)`. |

Each output row corresponds to one `Directory` value, stored as the CSV index
named `Subject`. The leading columns come from `PredictTruelabel.csv` after
dropping prediction-probability columns whose names start with `Prob_`.
The Action / Joint / BodypartGroup representative-value columns follow those
label columns, and the final column is `Fold`. The `Fold` column stores the
numeric part extracted from fold directory names such as `Fold0`.
Saved representative values are per-subject unnormalized values, not averages
across subjects. Before writing, the completed table is sorted in ascending
order by the index values corresponding to `Directory`.

Body-part groups are:

| Group | Joints |
|---|---|
| `LeftArm` | `ShoulderLeft`, `ElbowLeft`, `WristLeft` |
| `RightArm` | `ShoulderRight`, `ElbowRight`, `WristRight` |
| `Torso` | `SpineBase`, `SpineMid`, `SpineShoulder` |
| `LeftLeg` | `HipLeft`, `KneeLeft`, `AnkleLeft` |
| `RightLeg` | `HipRight`, `KneeRight`, `AnkleRight` |

Only joints that exist in the active `headers` list are used. If none of a
group's joints exist in the selected skeleton layout, that group is skipped.

### Step 5 - Create action label-group violin plots

After saving the CSVs, the script creates
`GradCAM_Action_LabelGroup_Violin.png` and `.svg` from the `Action_` columns.
For `Max`, `Mean`, and `Median`, the action representative values are first
row-normalized per subject across all actions. The values saved in the CSVs
remain unnormalized.

The figure has four panels for `(True, Predicted) = (0, 0), (0, 1), (1, 0),
(1, 1)`, so incorrect prediction groups are preserved rather than filtered
out. Within each panel, `Max`, `Mean`, and `Median` violins are drawn for each
action using `seaborn.violinplot`, with an inner boxplot configured through
`inner_kws` and the subject-direction mean overlaid as a separate marker.

### Step 6 - Create correct-class skeleton / violin plots

After saving the CSVs, `makeWeightDists.py` creates
`GradCAM_Skeleton_{Max,Mean,Median}.png` and `.svg` from the same seed-level
tables. Each statistic produces one figure whose columns are actions and whose
rows are:

| Row | Content |
|---|---|
| 1 | Class 0 skeleton plot, using only subjects with `True == Predicted == 0`. |
| 2 | Body-part group violin plots for the same Class 0 subjects. |
| 3 | Class 1 skeleton plot, using only subjects with `True == Predicted == 1`. |
| 4 | Body-part group violin plots for the same Class 1 subjects. |

For the skeleton rows, the script first applies row-wise min-max normalization
to all `Joint_` columns for each subject across all actions. It then filters to
correctly classified subjects for each class and averages in the subject
direction for each action/joint pair. The color range is shared within each
class row across action columns, rather than forced to be global across both
classes.

For the violin rows, the script applies the same row-wise normalization idea to
all `BodypartGroup_` columns. Each seaborn violin treats one subject as one
value for a given action/body-part group, shows an inner boxplot through
`inner_kws`, and draws the subject-direction mean as a separate marker inside
the violin panel.
Body-group violin colors follow a high-contrast palette close to the ablation
plots in `reports/ablation_boxplot/ablation_cross_subject_variables.py`.
The x-axis order of groups follows `BODYPART_GROUP_DISPLAY_ORDER` to mirror the
skeleton's anatomical front-view layout: `RightArm`, `RightLeg`, `Torso`,
`LeftLeg`, `LeftArm` (viewer's left = subject's right side).

Skeleton bone edges are colored by body-part group: edges whose two endpoint
joints belong to the **same** group use that group's palette color; edges that
cross group boundaries (e.g. ShoulderLeft → SpineShoulder) are drawn in gray.
The `edgelist` passed to `nx.draw_networkx_edges` explicitly follows the
`Graph.neighbor_link` order so that each color maps to the correct bone.

### Step 7 - Save seed-level CSVs and plots

For K-Fold results, the DataFrames from `Fold*` directories directly under
`Stratified_Seed<seed>/` are vertically concatenated before saving. Non-K-Fold
results use the same output format and write the single seed directory's
contents under `CAM_Dists`.
The CSVs and all violin/skeleton plots are saved to the same `CAM_Dists`
directory.

---

## Output Files

### `makeWeightPlots.py`

Single-action output:

```text
rootpath/
└── Weights[_Att|_CAM]/
    └── <Task>_to_<Item>_Seed<seed>[_FoldN]/
        └── <Directory>/
            ├── <TaskIndex>_<Task>_2D.mp4
            ├── <TaskIndex>_<Task>_3D.mp4
            └── <TaskIndex>_<Task>/
                ├── figures_2D/frameNNN.svg
                └── figures_3D/frameNNN_3d.svg
```

Fusion output:

```text
rootpath/
└── Weights[_Att|_CAM]/
    └── <Item>_Seed<seed>[_FoldN]/
        └── <Directory>/
            ├── <ActionIndex>_<Task>_2D.mp4
            ├── <ActionIndex>_<Task>_3D.mp4
            └── <ActionIndex>_<Task>/
                ├── figures_2D/frameNNN.svg
                └── figures_3D/frameNNN_3d.svg
```

The `.mp4` files are created only when `output_fmt` is `mp4`. The `figures_*`
directories are created only when `output_fmt` is `svg`.

### `makeWeightPlotsBulk.py` / `makeWeightPlotsBulkCompact.py` (`exam_fusion.py` only)

```text
rootpath/
└── Weights_CAM/
    └── <Item>_Seed<seed>[_FoldN]/
        ├── <Directory>_2D_combined.mp4
        ├── <Directory>_3D_combined.mp4
        └── <Directory>/
            ├── figures_2D_combined/frameNNN_2d_combined.svg
            └── figures_3D_combined/frameNNN_3d_combined.svg
```

### `makeWeightDists.py` (`exam_fusion.py` only)

Seed-level output:

```text
rootpath/
└── <Item>/
    └── Stratified_Seed<seed>/
        └── CAM_Dists/
            ├── GradCAM_Max.csv
            ├── GradCAM_Mean.csv
            ├── GradCAM_Median.csv
            ├── GradCAM_Action_LabelGroup_Violin.png
            ├── GradCAM_Action_LabelGroup_Violin.svg
            ├── GradCAM_Skeleton_Max.png
            ├── GradCAM_Skeleton_Max.svg
            ├── GradCAM_Skeleton_Mean.png
            ├── GradCAM_Skeleton_Mean.svg
            ├── GradCAM_Skeleton_Median.png
            └── GradCAM_Skeleton_Median.svg
```

For K-Fold results, per-fold `Weight_Distributions/` directories are not
created. Only the three all-fold-concatenated seed-level CSVs and the
action label-group plus statistic-specific skeleton/violin plots are written
under `CAM_Dists/`.

---

## Runtime Function Map

| Step | File / function | Role |
|---|---|---|
| Select root, item, seed, fold, mode, output format | `makeWeightPlots.py::makePlotsSingle`, `makeWeightPlots.py::makePlotsFusion`, `makeWeightPlotsBulk.py::makePlotsFusionBulk`, `makeWeightPlotsBulkCompact.py::makePlotsFusionBulk`, `makeWeightDists.py::__main__` | Interactive entrypoints that construct the result paths. Both BulkCompact and Dists `__main__` offer 3 modes: item directory, recursive root, or manual item/seed. |
| Resolve seed directories (BulkCompact and Dists) | `makeWeightPlotsBulkCompact.py::_list_seed_dirs`, `makeWeightPlotsBulkCompact.py::_find_seed_dirs`, `makeWeightDists.py::_list_seed_dirs`, `makeWeightDists.py::_find_seed_dirs` | Mode 1 lists `Stratified_Seed*` directly under one item directory; Mode 2 recursively finds all seed directories that contain `weights_CAM.h5`. |
| Process one seed directory (BulkCompact and Dists) | `makeWeightPlotsBulkCompact.py::_process_seed_dir_animate`, `makeWeightDists.py::_process_seed_dir` | Auto-detects K-Fold by scanning `Fold*` subdirectories and delegates to the per-fold processing function. |
| Read and reshape HDF5 weights | `makeWeightPlots.py::makeWeightPlots`, `makeWeightPlotsBulk*.py::_process_single_fold_animate`, `makeWeightDists.py::_process_single_fold` | Converts stored arrays to `(T, V)` and assigns joint names. Bulk, BulkCompact, and Dists read only fusion-style `weights_CAM.h5`. Dists uses the raw CAM output length without padding. |
| Read skeleton coordinates | `makeWeightPlots.py::makeWeightPlots`, `makeWeightPlotsBulk*.py::_process_single_fold_animate` | Loads `p_x.csv`, `p_y.csv`, and `p_z.csv` from `data/ConciseData` by default. |
| Draw per-action 2D frames | `plot2D.py::Plot2D` | Draws the skeleton graph in 2D and writes MP4 or SVG frames. |
| Draw per-action 3D frames | `plot3D.py::Plot3D` | Draws 3D skeleton scatter/edge frames and writes MP4 or SVG frames. |
| Draw combined fusion frames | `makeWeightPlotsBulk.py::PlotAllTasks2D_Combined`, `makeWeightPlotsBulk.py::PlotAllTasks3D_Combined`, `makeWeightPlotsBulkCompact.py::PlotAllTasks2D_Combined`, `makeWeightPlotsBulkCompact.py::PlotAllTasks3D_Combined` | Draws all available actions for one subject with a shared color scale. |
| Calculate GradCAM representative values | `makeWeightDists.py::_process_single_fold` | Reads each fold's `weights_CAM.h5` and `PredictTruelabel.csv`, then calculates unnormalized Max, Mean, and Median values at action, joint, and body-part group levels. |
| Save seed-level CSVs | `makeWeightDists.py::_save_seed_representative_csvs` | Vertically concatenates all fold subject rows and writes `CAM_Dists/GradCAM_{Max,Mean,Median}.csv`. |
| Draw action label-group violin plots | `makeWeightDists.py::_plot_label_group_violins` | Applies per-subject min-max normalization to the seed-level `Action_` tables, keeps all four `True` / `Predicted` groups, and saves action-wise violins to `CAM_Dists/GradCAM_Action_LabelGroup_Violin.{png,svg}`. |
| Draw correct-class skeleton / violin plots | `makeWeightDists.py::_plot_skeleton_weights` | Applies per-subject min-max normalization to the seed-level tables, keeps correctly classified subjects, and saves joint-mean skeleton rows plus body-part group violin rows to `CAM_Dists/GradCAM_Skeleton_{Max,Mean,Median}.{png,svg}`. |
