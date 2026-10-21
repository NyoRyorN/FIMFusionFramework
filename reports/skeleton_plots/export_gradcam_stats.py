"""export_gradcam_stats.py - export seed-level GradCAM representative values.

The script reads fusion GradCAM weights from each fold, calculates per-subject
Max, Mean, and Median values for every action at action, joint, and body-part
group levels, writes one unnormalized CSV per statistic under the seed
directory, and creates action-label and class-wise skeleton/violin summary
plots.

Usage:
    $ python reports/skeleton_plots/export_gradcam_stats.py

Config: none
Env: none
"""

import os
import sys
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.model.gcn.components.graph import Graph
from src.utils.names import SimpleJoints, FIMitems, Tasks, plotTasks
from src.utils.set_plt_font import setPltFont

setPltFont()
all_tasks = Tasks()
plot_tasks = plotTasks()
REPRESENTATIVE_STATS = ("Max", "Mean", "Median")
CAM_DIST_DIRNAME = "CAM_Dists"
VIOLIN_INNER_KWS = {
    # box_width and whis_width are in points (screen units), not data units.
    # 5 pt box keeps the inner box visible without hiding the KDE shape;
    # 1.5 pt whisker is thin enough to distinguish from the box edge.
    "box_width": 5,
    "whis_width": 1.5,
    "color": "#333333",
}

# Body-part group summaries intentionally use coarse anatomical regions so the
# violin rows remain readable across many action columns.
BODYPART_GROUPS = {
    "LeftArm": ["ShoulderLeft", "ElbowLeft", "WristLeft"],
    "RightArm": ["ShoulderRight", "ElbowRight", "WristRight"],
    "Torso": ["SpineBase", "SpineMid", "SpineShoulder"],
    "LeftLeg": ["HipLeft", "KneeLeft", "AnkleLeft"],
    "RightLeg": ["HipRight", "KneeRight", "AnkleRight"],
}
BODYPART_GROUP_COLORS = {
    # Match the high-contrast ablation palette used by ablation_cross_subject_variables.py.
    "LeftArm": "c",
    "RightArm": "m",
    "Torso": "y",
    "LeftLeg": "g",
    "RightLeg": "b",
}
# Display order mirrors the skeleton's front-view layout: viewer's left is the
# subject's right side, center is Torso, viewer's right is the subject's left.
BODYPART_GROUP_DISPLAY_ORDER = [
    "RightArm", "RightLeg", "Torso", "LeftLeg", "LeftArm"]

# Fixed 2D template for the simple_ntu-rgb+d skeleton used by this entrypoint.
_TEMPLATE_POS_17 = {
    0: (0.0, -0.3),     # SpineBase
    1: (0.0, -0.1),    # SpineMid
    2: (0.0, 0.25),     # Neck
    3: (0.0, 0.4),     # Head
    4: (-0.25, 0.1),    # ShoulderLeft
    5: (-0.45, -0.05),  # ElbowLeft
    6: (-0.6, -0.2),    # WristLeft
    7: (0.25, 0.1),     # ShoulderRight
    8: (0.45, -0.05),   # ElbowRight
    9: (0.6, -0.2),     # WristRight
    10: (-0.25, -0.3),  # HipLeft
    11: (-0.3, -0.5),  # KneeLeft
    12: (-0.35, -0.7),   # AnkleLeft
    13: (0.25, -0.3),   # HipRight
    14: (0.3, -0.5),   # KneeRight
    15: (0.35, -0.7),    # AnkleRight
    16: (0.0, 0.1),     # SpineShoulder
}


def _fold_output_value(fold_idx: str | None) -> int | str | None:
    """Convert `FoldN` directory names to compact CSV values."""
    if fold_idx is None:
        return None
    suffix = fold_idx.removeprefix("Fold")
    if suffix.isdigit():
        return int(suffix)
    return suffix


def _process_single_fold(seed_dir: str, fold_idx: str | None,
                         headers: list[str]):
    """Calculate unnormalized GradCAM representative values for one fold.

    Processing flow:
        1. Read fold-level `PredictTruelabel.csv`.
        2. Keep only rows whose `Directory` also exists in `weights_CAM.h5`.
        3. Calculate Max, Mean, and Median values without normalization for
           each subject/action at action, joint, and body-part group levels.

    Args:
        seed_dir: `Stratified_Seed<seed>` directory for the target run.
        fold_idx: Fold directory name, such as `Fold0`; `None` for non-fold.
        headers: Joint-name list matching the stored GradCAM arrays.

    Returns:
        Tuple of action, joint, body-part group, and prediction-label
        DataFrames, or `None` when required fold files do not exist.

    Raises:
        ValueError: If `PredictTruelabel.csv` does not provide `Directory`.
    """
    # ------------------------------------------------------------------
    # Resolve required fold files
    # ------------------------------------------------------------------
    main_path = os.path.join(
        seed_dir, fold_idx) if fold_idx is not None else seed_dir

    temp_weights_path = os.path.join(
        main_path, "weights_CAM.h5")
    if not os.path.exists(temp_weights_path):
        print("No weights: ", temp_weights_path)
        return None

    predict_path = os.path.join(main_path, "PredictTruelabel.csv")
    if not os.path.exists(predict_path):
        print("No predictions: ", predict_path)
        return None

    prediction_df = pd.read_csv(predict_path, header=0)
    if "Directory" not in prediction_df.columns:
        raise ValueError(f"`Directory` column is required in {predict_path}")
    prediction_df["Directory"] = prediction_df["Directory"].astype(str)
    probability_cols = [
        col for col in prediction_df.columns if col.startswith("Prob_")]
    prediction_df = prediction_df.drop(columns=probability_cols)

    action_max_records = {}
    action_mean_records = {}
    action_median_records = {}
    joint_max_records = {}
    joint_mean_records = {}
    joint_median_records = {}
    group_max_records = {}
    group_mean_records = {}
    group_median_records = {}

    # ------------------------------------------------------------------
    # Read weights and calculate per-subject representative values
    # ------------------------------------------------------------------
    with h5py.File(temp_weights_path, 'r') as f:
        data_df = prediction_df[
            prediction_df["Directory"].map(lambda data_dir: data_dir in f)
        ].copy()
        missing_weight_count = len(prediction_df) - len(data_df)
        if missing_weight_count > 0:
            print(
                f"Skipped {missing_weight_count} prediction rows without "
                f"matching weights in {temp_weights_path}"
            )
        data_dirs = data_df["Directory"].values

        fold_label = fold_idx if fold_idx else "NoFold"
        for data_dir in tqdm(
                data_dirs, desc=f"Processing subjects ({fold_label})"):
            temp_data = f[data_dir]
            available_tasks = set(temp_data.keys())

            action_max_row = {}
            action_mean_row = {}
            action_median_row = {}
            joint_max_row = {}
            joint_mean_row = {}
            joint_median_row = {}
            group_max_row = {}
            group_mean_row = {}
            group_median_row = {}

            for task_key in sorted(available_tasks, key=lambda x: int(x)):
                task_global_idx = int(task_key)
                task = all_tasks[task_global_idx]

                # Normalize to shape (joints, frames): HDF5 stores either
                # orientation depending on the model output path.
                weights_raw = pd.DataFrame(temp_data[task_key])
                if weights_raw.shape[0] != len(headers):
                    # ensure shape = (joints, frames)
                    weights_raw = weights_raw.T
                weights_raw.index = headers

                # Transpose to (frames, joints); columns inherit joint names
                # from weights_raw.index, so no further column assignment needed.
                # Use the raw CAM output length without padding or cropping so
                # that the statistics reflect the actual model output.
                new_weights = weights_raw.T

                # Action-level: aggregate over all frames AND all joints.
                # One scalar per action captures the overall activation strength.
                vals = new_weights.values
                action_max_row[task] = float(np.max(vals))
                action_mean_row[task] = float(np.mean(vals))
                action_median_row[task] = float(np.median(vals))

                # Joint-level: aggregate over frames only, keep joints separate.
                # Produces one scalar per (action, joint) pair used for
                # skeleton coloring.
                for joint in headers:
                    col_name = f"{task}_{joint}"
                    joint_vals = new_weights[joint].values
                    joint_max_row[col_name] = float(np.max(joint_vals))
                    joint_mean_row[col_name] = float(np.mean(joint_vals))
                    joint_median_row[col_name] = float(np.median(joint_vals))

                # Body-part group level: average over frames AND the joints
                # belonging to each anatomical group (3 joints per group).
                # Groups with no matching joint in headers are skipped silently.
                for grp_name, grp_joints in BODYPART_GROUPS.items():
                    valid_joints = [j for j in grp_joints if j in headers]
                    if len(valid_joints) == 0:
                        continue
                    grp_vals = new_weights[valid_joints].values
                    col_name = f"{task}_{grp_name}"
                    group_max_row[col_name] = float(np.max(grp_vals))
                    group_mean_row[col_name] = float(np.mean(grp_vals))
                    group_median_row[col_name] = float(np.median(grp_vals))

            action_max_records[data_dir] = action_max_row
            action_mean_records[data_dir] = action_mean_row
            action_median_records[data_dir] = action_median_row
            joint_max_records[data_dir] = joint_max_row
            joint_mean_records[data_dir] = joint_mean_row
            joint_median_records[data_dir] = joint_median_row
            group_max_records[data_dir] = group_max_row
            group_mean_records[data_dir] = group_mean_row
            group_median_records[data_dir] = group_median_row

    # ------------------------------------------------------------------
    # Convert dictionaries to aligned DataFrames
    # ------------------------------------------------------------------
    label_df = data_df.set_index("Directory")
    label_df.index.name = "Subject"
    label_df["Fold"] = _fold_output_value(fold_idx)
    action_max_df = pd.DataFrame.from_dict(action_max_records, orient='index')
    action_mean_df = pd.DataFrame.from_dict(
        action_mean_records, orient='index')
    action_median_df = pd.DataFrame.from_dict(
        action_median_records, orient='index')
    joint_max_df = pd.DataFrame.from_dict(joint_max_records, orient='index')
    joint_mean_df = pd.DataFrame.from_dict(joint_mean_records, orient='index')
    joint_median_df = pd.DataFrame.from_dict(
        joint_median_records, orient='index')
    group_max_df = pd.DataFrame.from_dict(group_max_records, orient='index')
    group_mean_df = pd.DataFrame.from_dict(group_mean_records, orient='index')
    group_median_df = pd.DataFrame.from_dict(
        group_median_records, orient='index')
    for df in (
        action_max_df, action_mean_df, action_median_df,
        joint_max_df, joint_mean_df, joint_median_df,
        group_max_df, group_mean_df, group_median_df,
    ):
        df.index.name = "Subject"

    return (action_max_df, action_mean_df, action_median_df,
            joint_max_df, joint_mean_df, joint_median_df,
            group_max_df, group_mean_df, group_median_df,
            label_df)


def _make_representative_csv_df(
        label_df: pd.DataFrame,
        action_df: pd.DataFrame,
        joint_df: pd.DataFrame,
        group_df: pd.DataFrame) -> pd.DataFrame:
    """Combine prediction labels and all GradCAM aggregation levels.

    The `Fold` column is moved to the final column so prediction labels stay at
    the front and value columns keep the `Action_`, `Joint_`, and
    `BodypartGroup_` prefixes grouped together.
    """
    # Extract Fold before concatenation so it can be appended at the end.
    # Keeping prediction labels at the front and Fold at the back makes the CSV
    # easier to inspect: label columns come first, feature columns are grouped
    # by prefix (Action_, Joint_, BodypartGroup_), and Fold is the last column.
    fold_col = label_df["Fold"] if "Fold" in label_df.columns else None
    label_without_fold = label_df.drop(columns=["Fold"], errors="ignore")
    value_dfs = [
        action_df.add_prefix("Action_"),
        joint_df.add_prefix("Joint_"),
        group_df.add_prefix("BodypartGroup_"),
    ]
    output_df = pd.concat([label_without_fold, *value_dfs], axis=1)
    if fold_col is not None:
        output_df["Fold"] = fold_col
    # mergesort preserves the original fold order for subjects with equal keys.
    return output_df.sort_index(kind="mergesort")


def _save_seed_representative_csvs(
        seed_save_dir: str,
        label_df: pd.DataFrame,
        action_dfs: dict[str, pd.DataFrame],
        joint_dfs: dict[str, pd.DataFrame],
        group_dfs: dict[str, pd.DataFrame]) -> None:
    """Write one unnormalized seed-level CSV for each statistic.

    Args:
        seed_save_dir: Destination `CAM_Dists` directory.
        label_df: Prediction-result table indexed by subject directory.
        action_dfs: Per-subject action-level representative values.
        joint_dfs: Per-subject action/joint representative values.
        group_dfs: Per-subject action/body-part group representative values.
    """
    os.makedirs(seed_save_dir, exist_ok=True)
    for stat_name in REPRESENTATIVE_STATS:
        output_df = _make_representative_csv_df(
            label_df=label_df,
            action_df=action_dfs[stat_name],
            joint_df=joint_dfs[stat_name],
            group_df=group_dfs[stat_name])
        output_df.index.name = "Subject"
        output_path = os.path.join(seed_save_dir, f"GradCAM_{stat_name}.csv")
        output_df.to_csv(output_path)
        print(f"Saved: {output_path}")


def _fold_sort_key(fold_name: str) -> tuple[int, int | str]:
    """Sort `Fold0`, `Fold1`, ... numerically before non-numeric suffixes."""
    suffix = fold_name.removeprefix("Fold")
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, suffix)


def _list_fold_dirs(seed_dir: str) -> list[str]:
    """List fold directories directly under one `Stratified_Seed*` directory."""
    if not os.path.isdir(seed_dir):
        return []
    return sorted(
        [
            name for name in os.listdir(seed_dir)
            if name.startswith("Fold")
            and os.path.isdir(os.path.join(seed_dir, name))
        ],
        key=_fold_sort_key)


def _process_seed_dir(
        seed_dir: str,
        skl_type: str,
        headers: list[str],
        title_suffix: str = "") -> None:
    """Process one `Stratified_Seed*` directory: collect folds, write CSVs and plots.

    Processing flow:
        1. Auto-detect K-Fold layout by scanning `Fold*` subdirectories.
        2. Calculate fold-level representative values without writing fold
           outputs.
        3. Vertically concatenate all fold rows and write three seed-level CSVs
           under `<seed_dir>/CAM_Dists`.
        4. Create an action-level label-group violin plot.
        5. Create one skeleton-plus-body-group violin plot per representative
           statistic from the seed-level tables.

    Fold rows are expected to be subject-disjoint, so duplicate subjects are
    reported but not merged.

    Args:
        seed_dir: `Stratified_Seed<seed>` directory for the target run.
        skl_type: Skeleton layout name used to resolve joint positions.
        headers: Joint-name list matching the stored GradCAM arrays.
        title_suffix: String appended to figure titles (e.g. item and seed).
    """
    # ------------------------------------------------------------------
    # Auto-detect K-Fold layout
    # ------------------------------------------------------------------
    # If Fold* subdirectories exist, treat them as K-Fold splits.
    # Otherwise, expect weights_CAM.h5 directly inside seed_dir.
    fold_indices = _list_fold_dirs(seed_dir)
    if fold_indices:
        print(f"K-Fold detected: {fold_indices}")
    else:
        if not os.path.exists(os.path.join(seed_dir, "weights_CAM.h5")):
            print(f"No CAM weights found in {seed_dir}; skip.")
            return
        fold_indices = [None]

    action_tables = {stat_name: [] for stat_name in REPRESENTATIVE_STATS}
    joint_tables = {stat_name: [] for stat_name in REPRESENTATIVE_STATS}
    group_tables = {stat_name: [] for stat_name in REPRESENTATIVE_STATS}
    label_tables = []

    # ------------------------------------------------------------------
    # Process each fold without writing fold-local outputs
    # ------------------------------------------------------------------
    for fold_idx in fold_indices:
        fold_label = fold_idx if fold_idx is not None else "NoFold"
        print(f"\n{'='*50}")
        print(f"Processing {fold_label}...")
        print(f"{'='*50}")
        result = _process_single_fold(seed_dir, fold_idx, headers)
        if result is None:
            continue

        (max_df, mean_df, median_df,
         jmax_df, jmean_df, jmedian_df,
         gmax_df, gmean_df, gmedian_df,
         label_df) = result
        action_tables["Max"].append(max_df)
        action_tables["Mean"].append(mean_df)
        action_tables["Median"].append(median_df)
        joint_tables["Max"].append(jmax_df)
        joint_tables["Mean"].append(jmean_df)
        joint_tables["Median"].append(jmedian_df)
        group_tables["Max"].append(gmax_df)
        group_tables["Mean"].append(gmean_df)
        group_tables["Median"].append(gmedian_df)
        label_tables.append(label_df)

    if not label_tables:
        print("No GradCAM representative values were generated.")
        return

    combined_labels = pd.concat(label_tables)
    duplicate_count = int(combined_labels.index.duplicated().sum())
    if duplicate_count > 0:
        print(
            f"Warning: found {duplicate_count} duplicated subject rows across "
            "folds. Rows are kept as-is."
        )

    # ------------------------------------------------------------------
    # Build and save seed-level CSVs and plots
    # ------------------------------------------------------------------
    combined_action_tables = {
        stat_name: pd.concat(action_tables[stat_name])
        for stat_name in REPRESENTATIVE_STATS
    }
    combined_joint_tables = {
        stat_name: pd.concat(joint_tables[stat_name])
        for stat_name in REPRESENTATIVE_STATS
    }
    combined_group_tables = {
        stat_name: pd.concat(group_tables[stat_name])
        for stat_name in REPRESENTATIVE_STATS
    }

    seed_save_dir = os.path.join(seed_dir, CAM_DIST_DIRNAME)
    _save_seed_representative_csvs(
        seed_save_dir=seed_save_dir,
        label_df=combined_labels,
        action_dfs=combined_action_tables,
        joint_dfs=combined_joint_tables,
        group_dfs=combined_group_tables)
    _plot_label_group_violins(
        action_dfs=combined_action_tables,
        label_df=combined_labels,
        save_dirs=[seed_save_dir],
        title_suffix=title_suffix)
    _plot_skeleton_weights(
        joint_dfs=combined_joint_tables,
        group_dfs=combined_group_tables,
        label_df=combined_labels,
        headers=headers,
        skl_type=skl_type,
        save_dirs=[seed_save_dir],
        title_suffix=title_suffix)


def _list_seed_dirs(item_dir: str) -> list[str]:
    """List `Stratified_Seed*` directories directly under one item directory."""
    if not os.path.isdir(item_dir):
        return []
    return sorted(
        os.path.join(item_dir, name)
        for name in os.listdir(item_dir)
        if name.startswith("Stratified_Seed")
        and os.path.isdir(os.path.join(item_dir, name))
    )


def _find_seed_dirs(root_dir: str) -> list[str]:
    """Recursively find all `Stratified_Seed*` directories that contain CAM weights.

    A directory qualifies if it contains `weights_CAM.h5` directly or has at
    least one `Fold*/weights_CAM.h5`.  Recursion stops when a qualifying
    directory is found so nested seeds are not double-counted.
    """
    seed_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not os.path.basename(dirpath).startswith("Stratified_Seed"):
            continue
        has_cam = "weights_CAM.h5" in filenames or any(
            os.path.exists(os.path.join(dirpath, d, "weights_CAM.h5"))
            for d in dirnames
            if d.startswith("Fold")
        )
        if has_cam:
            seed_dirs.append(dirpath)
            dirnames.clear()  # do not recurse further into this seed directory
    return sorted(seed_dirs)


def makeWeightsDists(rootpath: str,
                     skl_type: str,
                     headers: list[str]) -> None:
    """Export CSVs and plots for one interactively chosen item/seed combination.

    Asks for item (0--12) and seed, constructs the `Stratified_Seed*` path,
    and delegates all processing to `_process_seed_dir`.  K-Fold layout is
    detected automatically from directory contents.

    Args:
        rootpath: Parent directory containing FIM item subdirectories.
        skl_type: Skeleton layout name used to resolve joint positions.
        headers: Joint-name list matching the stored GradCAM arrays.
    """
    item = int(input("Enter the item(0--12): "))
    if item not in range(0, 13):
        sys.exit("Invalid item")
    item = "{:0>2d}-{}".format(item, FIMitems()[item])
    seed = input("Enter the seed(100): ")

    seed_dir = os.path.join(rootpath, item, "Stratified_Seed" + seed)
    _process_seed_dir(
        seed_dir=seed_dir,
        skl_type=skl_type,
        headers=headers,
        title_suffix=f" ({item}, Seed {seed})")


def _minmax_normalize_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Apply row-wise Min-Max normalization across all value columns.

    The distribution plots compare each subject's within-subject relative
    GradCAM pattern, so normalization is performed per row rather than over the
    whole cohort.
    """
    row_min = df.min(axis=1)
    row_max = df.max(axis=1)
    denom = row_max - row_min
    denom = denom.replace(0, 1)  # avoid division by zero
    return df.sub(row_min, axis=0).div(denom, axis=0)


def _plot_label_group_violins(
        action_dfs: dict[str, pd.DataFrame],
        label_df: pd.DataFrame,
        save_dirs: list[str],
        title_suffix: str = "") -> None:
    """Create action-level violin plots for all true/predicted label groups.

    Processing flow:
        1. Row-normalize each statistic table across action columns.
        2. Split subjects into the four `(True, Predicted)` groups.
        3. Draw action-wise Max, Mean, and Median seaborn violins with inner
           boxplots while keeping correct and incorrect groups visible.

    Action representative values are row-normalized per subject across all
    actions before grouping. The plot keeps correct and incorrect prediction
    groups so that `(True, Predicted) = (0, 1)` and `(1, 0)` remain visible.
    """
    if not action_dfs:
        print("No action values; skip label-group violin plot.")
        return

    action_cols = action_dfs[REPRESENTATIVE_STATS[0]].columns.tolist()
    if not action_cols:
        print("No action columns; skip label-group violin plot.")
        return

    # reset_index(drop=False) retains subject directory names as a column so
    # "True"/"Predicted" columns remain accessible by name after the reset.
    # The value tables use drop=True to align on a shared integer index with
    # the labels DataFrame.
    labels = label_df.reset_index(drop=False)
    normalized_tables = {}
    for stat_name in REPRESENTATIVE_STATS:
        action_values = action_dfs[stat_name].reset_index(drop=True)
        normalized_tables[stat_name] = _minmax_normalize_per_subject(
            action_values)

    label_combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    label_titles = {
        (0, 0): "True=0, Predicted=0",
        (0, 1): "True=0, Predicted=1",
        (1, 0): "True=1, Predicted=0",
        (1, 1): "True=1, Predicted=1",
    }
    stat_colors = {
        "Max": "#ED7D31",
        "Mean": "#70AD47",
        "Median": "#5B9BD5",
    }
    # Offset each statistic violin by ±0.24 from the action center.
    # With violin width=0.22 (half-width=0.11), adjacent violins just touch
    # at the boundary and do not visually overlap.
    stat_offsets = {
        "Max": -0.24,
        "Mean": 0.0,
        "Median": 0.24,
    }
    # Space action groups 1.25 apart so three offset violins fit without
    # colliding with the neighboring action group.
    centers = np.arange(len(action_cols)) * 1.25
    fig, axes = plt.subplots(
        2, 2, figsize=(max(16, len(action_cols) * 1.7), 12))
    axes = axes.flatten()

    true_values = pd.to_numeric(labels["True"], errors="coerce")
    predicted_values = pd.to_numeric(labels["Predicted"], errors="coerce")
    for ax, (true_val, pred_val) in zip(axes, label_combos):
        mask = (true_values == true_val) & (predicted_values == pred_val)
        n_subjects = int(mask.sum())
        for stat_name in REPRESENTATIVE_STATS:
            stat_df = normalized_tables[stat_name]
            positions = centers + stat_offsets[stat_name]
            plot_records = []
            mean_positions = []
            means = []
            for pos, col in zip(positions, action_cols):
                vals = stat_df.loc[mask, col].dropna().values
                # seaborn needs at least 2 samples to compute the KDE.
                if len(vals) < 2:
                    continue
                mean_positions.append(pos)
                means.append(float(np.mean(vals)))
                plot_records.extend(
                    {"Position": pos, "Value": val} for val in vals)
            if not plot_records:
                continue

            sns.violinplot(
                data=pd.DataFrame(plot_records),
                x="Position",
                y="Value",
                native_scale=True,
                color=stat_colors[stat_name],
                saturation=0.75,
                width=0.22,
                inner="box",
                inner_kws=VIOLIN_INNER_KWS.copy(),
                cut=0,
                linewidth=0.8,
                ax=ax)
            ax.scatter(
                mean_positions, means, marker="D", s=18,
                color="#111111", zorder=3)

        if n_subjects == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
        ax.set_xticks(centers)
        ax.set_xticklabels(
            [_task_plot_label(col) for col in action_cols],
            rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{label_titles[(true_val, pred_val)]} (n={n_subjects})")
        ax.set_ylabel("Normalized Action Value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=stat_colors[stat_name], alpha=0.65)
        for stat_name in REPRESENTATIVE_STATS
    ]
    mean_handle = Line2D(
        [0], [0], marker="D", color="w", markerfacecolor="#111111",
        markeredgecolor="#111111", markersize=5,
        label="Subject mean")
    axes[0].legend(
        [*legend_handles, mean_handle],
        [*REPRESENTATIVE_STATS, "Subject mean"],
        loc="upper right", fontsize=8)

    fig.suptitle(
        "GradCAM Action Violin Plot by Label Group"
        f"{title_suffix}\n"
        "Action values are Min-Max normalized within each subject across all "
        "actions before grouping.",
        fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for save_dir in save_dirs:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(
            save_dir, "GradCAM_Action_LabelGroup_Violin.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {plot_path}")
        svg_path = os.path.join(
            save_dir, "GradCAM_Action_LabelGroup_Violin.svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved plot: {svg_path}")
    plt.close(fig)


def _extract_available_tasks(value_df: pd.DataFrame) -> list[str]:
    """Extract action names from `<Action>_<Feature>` value columns."""
    # dict.fromkeys preserves insertion order and eliminates duplicates in O(n).
    task_names = (col.rsplit("_", 1)[0] for col in value_df.columns)
    return list(dict.fromkeys(task_names))


def _task_plot_label(task: str) -> str:
    """Return the compact action label used in multi-panel figures."""
    try:
        task_idx = all_tasks.index(task)
        return plot_tasks[task_idx]
    except ValueError:
        return task


def _get_skeleton_template_pos(
        skl_type: str,
        n_joints: int) -> dict[int, tuple[float, float]]:
    """Return fixed 2D joint positions for the active skeleton layout."""
    if skl_type == "simple_ntu-rgb+d":
        return {idx: _TEMPLATE_POS_17[idx] for idx in range(n_joints)}
    raise ValueError(
        f"Unsupported skeleton layout for summary plot: {skl_type}")


def _edge_group_colors(edges: list[tuple[int, int]],
                       headers: list[str],
                       fallback: str = "gray") -> list[str]:
    """Return one color per skeleton edge matching the body-part group violin palette.

    An edge that connects two joints within the same body-part group is colored
    with that group's palette color.  Cross-group edges (e.g. SpineBase→Hip)
    use the fallback color so they are visually neutral.
    """
    joint_to_group = {}
    for group_name, joints in BODYPART_GROUPS.items():
        for j in joints:
            joint_to_group[j] = group_name

    colors = []
    for i, j in edges:
        gi = joint_to_group.get(headers[i]) if i < len(headers) else None
        gj = joint_to_group.get(headers[j]) if j < len(headers) else None
        if gi is not None and gi == gj:
            colors.append(BODYPART_GROUP_COLORS[gi])
        else:
            colors.append(fallback)
    return colors


def _prepare_plot_inputs(
        label_df: pd.DataFrame,
        value_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reset row indexes so duplicated subject names do not affect plotting."""
    labels = label_df.reset_index(drop=False)
    values = value_df.reset_index(drop=True)
    return labels, values


def _correct_mask(labels: pd.DataFrame, cls: int) -> pd.Series:
    """Select correctly classified subjects for one class."""
    true_values = pd.to_numeric(labels["True"], errors="coerce")
    predicted_values = pd.to_numeric(labels["Predicted"], errors="coerce")
    return (true_values == cls) & (predicted_values == cls)


def _finite_range(values: list[np.ndarray]) -> tuple[float, float]:
    """Calculate a finite color range, falling back to `[0, 1]` if empty."""
    finite_values = [
        arr[np.isfinite(arr)]
        for arr in values
        if arr is not None and np.isfinite(arr).any()
    ]
    if not finite_values:
        # Fall back to [0, 1] when all values are NaN or no data exists so
        # the colorbar is still drawn with a valid normalization range.
        return 0.0, 1.0
    merged = np.concatenate(finite_values)
    vmin = float(np.nanmin(merged))
    vmax = float(np.nanmax(merged))
    if vmax - vmin < 1e-6:
        # Constant-weight edge case: expand the range by 1.0 to avoid a
        # degenerate colormap where every node maps to the same color.
        vmax = vmin + 1.0
    return vmin, vmax


def _draw_skeleton_panel(
        ax: Axes,
        colors: np.ndarray | None,
        edges: list[tuple[int, int]],
        template_pos: dict[int, tuple[float, float]],
        n_joints: int,
        vmin: float,
        vmax: float,
        edge_colors: list[str] | str = "gray") -> None:
    """Draw one fixed-layout skeleton colored by joint representative values."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_joints))
    graph.add_edges_from(edges)
    # edgelist must match the order used in _edge_group_colors so that each
    # color maps to the correct bone; graph.edges() uses a different iteration
    # order than the neighbor_link list we pass in.
    nx.draw_networkx_edges(
        graph, template_pos, edgelist=list(edges),
        edge_color=edge_colors, ax=ax, width=3.0)
    if colors is None:
        nx.draw_networkx_nodes(
            graph, template_pos, node_color="#F0F0F0", ax=ax,
            edgecolors="black", node_size=150)
    else:
        nx.draw_networkx_nodes(
            graph, template_pos, node_color=colors, cmap="Reds", ax=ax,
            edgecolors="black", node_size=150, vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.5)
    ax.axis("off")
    # Invert x so the skeleton appears as seen from the front: the subject's
    # left arm is on the viewer's right, matching anatomical convention.
    ax.invert_xaxis()


def _plot_bodypart_violin_panel(
        ax: Axes,
        group_norm: pd.DataFrame,
        labels: pd.DataFrame,
        task: str,
        cls: int) -> int:
    """Draw body-part group distributions for one action and class.

    Each seaborn violin treats one subject as one observation and uses an inner
    boxplot. The diamond marker shows the subject-direction mean requested for
    direct comparison with the skeleton's subject-averaged joint colors.
    """
    mask = _correct_mask(labels, cls)
    # Follow BODYPART_GROUP_DISPLAY_ORDER so the x-axis matches the skeleton's
    # front-view layout (subject's right on the viewer's left).
    group_names = [
        group_name
        for group_name in BODYPART_GROUP_DISPLAY_ORDER
        if f"{task}_{group_name}" in group_norm.columns
    ]
    values = [
        group_norm.loc[mask, f"{task}_{group_name}"].dropna().values
        for group_name in group_names
    ]
    # seaborn needs at least 2 samples to compute the KDE; single-sample
    # groups are omitted to prevent a crash.
    non_empty = [(name, vals) for name, vals in zip(group_names, values)
                 if len(vals) >= 2]
    if not non_empty:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
        ax.set_ylim(-0.05, 1.05)
        return int(mask.sum())

    group_names, values = zip(*non_empty)
    positions = np.arange(1, len(values) + 1)
    for group_name, pos, vals in zip(group_names, positions, values):
        plot_df = pd.DataFrame({
            "Position": np.full(len(vals), pos),
            "Value": vals,
        })
        sns.violinplot(
            data=plot_df,
            x="Position",
            y="Value",
            native_scale=True,
            color=BODYPART_GROUP_COLORS.get(group_name, "gray"),
            saturation=0.75,
            width=0.8,
            inner="box",
            inner_kws=VIOLIN_INNER_KWS.copy(),
            cut=0,
            linewidth=0.9,
            ax=ax)
    means = [float(np.mean(vals)) for vals in values]
    ax.scatter(
        positions, means, marker="D", s=28, color="#222222", zorder=3,
        label="Subject mean")
    ax.set_xticks(positions)
    ax.set_xticklabels(group_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    return int(mask.sum())


def _plot_skeleton_weights(
        joint_dfs: dict[str, pd.DataFrame],
        group_dfs: dict[str, pd.DataFrame],
        label_df: pd.DataFrame,
        headers: list[str],
        skl_type: str,
        save_dirs: list[str],
        title_suffix: str = "") -> None:
    """Create class-wise skeleton and body-part violin plots.

    Processing flow:
        1. Row-normalize Joint and BodypartGroup values per subject across all
           actions.
        2. For each class, keep only correctly classified subjects.
        3. Average normalized Joint values across subjects for skeleton colors.
        4. Plot BodypartGroup distributions as one value per subject, with the
           subject-direction mean marked separately.
    """
    edges = Graph(skl_type).neighbor_link
    n_joints = len(headers)
    template_pos = _get_skeleton_template_pos(skl_type, n_joints)
    # Pre-compute once; same color assignment used for every action panel.
    bone_edge_colors = _edge_group_colors(edges, headers)
    class_rows = {
        0: (0, 1, "Class 0 (Dependent)"),
        1: (2, 3, "Class 1 (Independent)"),
    }

    for stat_name in REPRESENTATIVE_STATS:
        joint_df = joint_dfs[stat_name]
        group_df = group_dfs[stat_name]
        if joint_df.empty:
            print(f"No joint values for {stat_name}; skip summary plot.")
            continue

        available_tasks = _extract_available_tasks(joint_df)
        if not available_tasks:
            print(f"No action columns for {stat_name}; skip summary plot.")
            continue

        labels, joint_values = _prepare_plot_inputs(label_df, joint_df)
        _, group_values = _prepare_plot_inputs(label_df, group_df)
        joint_norm = _minmax_normalize_per_subject(joint_values)
        group_norm = _minmax_normalize_per_subject(group_values)

        # For each (class, action) pair, average the row-normalized joint
        # weights across correctly classified subjects.  The result is one
        # value per joint used as the skeleton node color.
        class_task_means = {}
        for cls in class_rows:
            mask = _correct_mask(labels, cls)
            for task in available_tasks:
                task_joint_cols = [
                    f"{task}_{joint}" for joint in headers
                    if f"{task}_{joint}" in joint_norm.columns]
                if not task_joint_cols or not mask.any():
                    continue
                mean_series = joint_norm.loc[mask, task_joint_cols].mean(
                    axis=0)
                mean_values = np.array([
                    mean_series.get(f"{task}_{joint}", np.nan)
                    for joint in headers
                ])
                if np.isfinite(mean_values).any():
                    class_task_means[(cls, task)] = mean_values

        # Color range is computed per class (not globally across both classes)
        # so each skeleton row uses its own vmin/vmax.  This highlights
        # within-class joint importance patterns rather than forcing a shared
        # scale that would wash out the weaker class.
        class_ranges = {
            cls: _finite_range([
                class_task_means[(cls, task)]
                for task in available_tasks
                if (cls, task) in class_task_means
            ])
            for cls in class_rows
        }

        n_cols = len(available_tasks)
        fig_width = max(12, 3.2 * n_cols)
        # constrained_layout lets matplotlib automatically reserve space for
        # colorbars and row labels; calling subplots_adjust after colorbar
        # creation would reset the layout and cause overlap.
        fig, axes = plt.subplots(
            4, n_cols, figsize=(fig_width, 12),
            gridspec_kw={"height_ratios": [2.4, 1.45, 2.4, 1.45]},
            squeeze=False,
            constrained_layout=True)

        # n_subjects depends only on the class label, not on the task column,
        # so it is computed once per class before iterating over tasks.
        n_subjects_per_class = {
            cls: int(_correct_mask(labels, cls).sum())
            for cls in class_rows
        }

        for col_idx, task in enumerate(available_tasks):
            axes[0, col_idx].set_title(
                _task_plot_label(task), fontsize=10, pad=6)
            for cls, (skeleton_row, violin_row, class_label) in (
                    class_rows.items()):
                skeleton_ax = axes[skeleton_row, col_idx]
                violin_ax = axes[violin_row, col_idx]
                colors = class_task_means.get((cls, task))
                n_subjects = n_subjects_per_class[cls]
                vmin, vmax = class_ranges[cls]
                if colors is None:
                    skeleton_ax.text(
                        0.5, 0.5, "No Data", ha="center", va="center",
                        transform=skeleton_ax.transAxes, fontsize=9)
                _draw_skeleton_panel(
                    ax=skeleton_ax,
                    colors=colors,
                    edges=edges,
                    template_pos=template_pos,
                    n_joints=n_joints,
                    vmin=vmin,
                    vmax=vmax,
                    edge_colors=bone_edge_colors)
                if col_idx == 0:
                    skeleton_ax.text(
                        -0.2, 0.5,
                        f"{class_label}\nSkeleton mean\n(n={n_subjects})",
                        fontsize=10, rotation=90, ha="center", va="center",
                        transform=skeleton_ax.transAxes)

                _plot_bodypart_violin_panel(
                    ax=violin_ax,
                    group_norm=group_norm,
                    labels=labels,
                    task=task,
                    cls=cls)
                if col_idx == 0:
                    violin_ax.set_ylabel("Body group\nviolin", fontsize=9)
                else:
                    violin_ax.set_yticklabels([])

        for cls, (skeleton_row, _, class_label) in class_rows.items():
            vmin, vmax = class_ranges[cls]
            sm = plt.cm.ScalarMappable(
                cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=axes[skeleton_row, :].ravel().tolist(),
                fraction=0.018, pad=0.01)
            cbar.set_label(
                f"{class_label} joint mean", fontsize=8)

        legend_handle = Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="#222222",
            markeredgecolor="#222222", markersize=6,
            label="Body-group subject mean")
        axes[1, -1].legend(
            handles=[legend_handle], loc="upper right", fontsize=8)
        fig.suptitle(
            f"GradCAM {stat_name} Correct-Class Summary{title_suffix}\n"
            "Joint and body-group values are Min-Max normalized within each "
            "subject across all actions before class aggregation.",
            fontsize=13)

        for save_dir in save_dirs:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(
                save_dir, f"GradCAM_Skeleton_{stat_name}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {plot_path}")
            svg_path = os.path.join(
                save_dir, f"GradCAM_Skeleton_{stat_name}.svg")
            fig.savefig(svg_path, bbox_inches="tight")
            print(f"Saved plot: {svg_path}")
        plt.close(fig)


if __name__ == "__main__":
    headers = SimpleJoints()
    skl_type = "simple_ntu-rgb+d"

    print("Select processing mode:")
    print("  1: Specify item directory (parent of Stratified_Seed*)"
          " — process all seeds in that item")
    print("  2: Specify a root directory — find and process all seeds"
          " containing CAM weights")
    print("  3: Specify root directory and enter item/seed interactively")
    _mode = input("Mode (1/2/3): ").strip()

    if _mode == "1":
        # ------------------------------------------------------------------
        # Mode 1: item directory → all Stratified_Seed* inside it
        # ------------------------------------------------------------------
        item_dir = input(
            "Enter item directory path (parent of Stratified_Seed*): ").strip()
        seed_dirs = _list_seed_dirs(item_dir)
        if not seed_dirs:
            sys.exit(
                f"No Stratified_Seed* directories found under: {item_dir}")
        # normpath strips trailing separators so basename works even when the
        # user appends a trailing slash to the path.
        item = os.path.basename(os.path.normpath(item_dir))
        for seed_dir in seed_dirs:
            seed = os.path.basename(seed_dir).removeprefix("Stratified_Seed")
            print(f"\n{'#'*60}")
            print(f"# {item} / Seed {seed}")
            print(f"{'#'*60}")
            _process_seed_dir(
                seed_dir=seed_dir,
                skl_type=skl_type,
                headers=headers,
                title_suffix=f" ({item}, Seed {seed})")

    elif _mode == "2":
        # ------------------------------------------------------------------
        # Mode 2: root directory → recursively find all seeds with CAM weights
        # ------------------------------------------------------------------
        root_dir = input(
            "Enter root directory path to search recursively: ").strip()
        seed_dirs = _find_seed_dirs(root_dir)
        if not seed_dirs:
            sys.exit(
                f"No Stratified_Seed* directories with CAM weights found "
                f"under: {root_dir}")
        print(f"Found {len(seed_dirs)} seed director"
              f"{'y' if len(seed_dirs) == 1 else 'ies'}.")
        for seed_dir in seed_dirs:
            item = os.path.basename(os.path.dirname(seed_dir))
            seed = os.path.basename(seed_dir).removeprefix("Stratified_Seed")
            print(f"\n{'#'*60}")
            print(f"# {item} / Seed {seed}")
            print(f"{'#'*60}")
            _process_seed_dir(
                seed_dir=seed_dir,
                skl_type=skl_type,
                headers=headers,
                title_suffix=f" ({item}, Seed {seed})")

    elif _mode == "3":
        # ------------------------------------------------------------------
        # Mode 3: original interactive mode (rootpath + item + seed prompts)
        # ------------------------------------------------------------------
        rootpath = input(
            "Enter root path (parent of item directories, "
            "basename starts with Ex_...): ").strip()
        makeWeightsDists(rootpath=rootpath, skl_type=skl_type, headers=headers)

    else:
        sys.exit("Invalid mode selection.")
