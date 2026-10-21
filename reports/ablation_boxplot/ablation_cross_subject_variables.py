"""Generate cross-subject ablation boxplots from CrossSubjectVariables output.

Each row in CrossSubjectVariables CSVs represents one subject (Directory),
so each data point in the resulting boxplots is a single subject's match rate.
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setPltFont


def ablation_colors(count: int) -> list[str]:
    """Return *count* distinct color strings for ablation conditions."""
    colors = [
        "c", "m", "y", "g", "b", "r", "k",
        "orange", "purple", "brown", "pink", "gray", "olive", "navy", "crimson",
    ]
    # If more conditions than base colors, tile the list until it is long enough
    while len(colors) < count:
        colors.extend(colors)
    return colors[:count]


def rate_to_percent(series: pd.Series) -> pd.Series:
    """Convert a 0-1 rate series to percentage; leave values already above 1.5 unchanged.

    The threshold 1.5 distinguishes 0-1 rates from already-scaled percentages:
    a value above 1.5 cannot be a valid 0-1 rate, so it is assumed to be in % already.
    """
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return series
    if series.max() <= 1.5:
        return series * 100
    return series


def label_matches(left: object, right: object) -> bool:
    """Return True if two labels are equal or one is a suffix of the other (after normalization).

    Suffix matching handles cases where one label includes extra context that the other omits,
    e.g. "Eating" matches "SelfCare_Eating" because "eating" is a suffix of "selfcareeating".
    """
    # Lowercase and strip non-alphanumeric characters for fuzzy suffix matching
    left_norm = re.sub(r"[^0-9a-z]+", "", str(left).lower())
    right_norm = re.sub(r"[^0-9a-z]+", "", str(right).lower())
    return (
        left_norm == right_norm
        or left_norm.endswith(right_norm)
        or right_norm.endswith(left_norm)
    )


def find_condition_dir(root_dir: Path, condition: str) -> Path | None:
    """Return the first directory whose name starts with *condition*, or None.

    Glob with a trailing wildcard allows condition directories to have version
    suffixes (e.g. "CondA_v2") while still being matched by the bare name "CondA".
    """
    for path in sorted(glob.glob(str(root_dir / f"{condition}*"))):
        condition_dir = Path(path)
        if condition_dir.is_dir():
            return condition_dir
    return None


def find_matching_column(columns: pd.Index, target: str) -> str | None:
    """Find *target* in *columns* by exact match first, then by fuzzy suffix rule.

    The two-pass strategy prefers exact matches to avoid false positives from
    suffix matching when multiple columns share a common suffix.
    """
    for column in columns:
        if str(column) == str(target):
            return str(column)
    for column in columns:
        if label_matches(column, target):
            return str(column)
    return None


def collect_summary_points(
    root_dir: Path,
    conditions: list[str],
    relative_csv_path: Path,
    value_column: str,
) -> pd.DataFrame:
    """Collect per-subject match-rate points from a fixed CSV path across conditions.

    Args:
        root_dir: Ablation root directory containing one subdirectory per condition.
        conditions: Ordered list of condition names to collect.
        relative_csv_path: CSV path relative to each condition directory.
        value_column: Column name to use as the match-rate value.

    Returns:
        Long-format DataFrame with columns [Directory, MatchRate, situation].
    """
    all_df = []
    for condition in conditions:
        condition_dir = find_condition_dir(root_dir, condition)
        if condition_dir is None:
            print(f"No directory found for condition {condition}. Skipping...")
            continue
        csv_path = condition_dir / relative_csv_path
        if not csv_path.exists():
            print(f"File {csv_path} does not exist. Skipping...")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        missing = {"Directory", value_column} - set(df.columns)
        if missing:
            print(f"Missing columns in {csv_path}: {sorted(missing)}")
            continue
        current_df = df[["Directory", value_column]].dropna(subset=[value_column]).copy()
        current_df["MatchRate"] = rate_to_percent(current_df[value_column])
        current_df["situation"] = condition
        all_df.append(current_df[["Directory", "MatchRate", "situation"]])
    if not all_df:
        return pd.DataFrame(columns=["Directory", "MatchRate", "situation"])
    return pd.concat(all_df, ignore_index=True)


def collect_existing_item_points(
    root_dir: Path,
    conditions: list[str],
    all_items_csv_name: str,
    item_name: str,
) -> pd.DataFrame:
    """Collect per-subject match-rate points for a single FIM item across conditions.

    Silently skips conditions where the item column is absent, because not every
    ablation condition necessarily covers every FIM item.

    Args:
        root_dir: Ablation root directory.
        conditions: Ordered list of condition names to collect.
        all_items_csv_name: Filename of the all-items summary CSV inside each condition dir.
        item_name: Column name of the target FIM item in that CSV.

    Returns:
        Long-format DataFrame with columns [Directory, MatchRate, situation].
    """
    all_df = []
    for condition in conditions:
        condition_dir = find_condition_dir(root_dir, condition)
        if condition_dir is None:
            continue
        csv_path = condition_dir / all_items_csv_name
        if not csv_path.exists():
            continue
        # Read only the header row first to avoid loading the full CSV unnecessarily
        columns = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns
        if item_name not in columns:
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        current_df = df[["Directory", item_name]].dropna(subset=[item_name]).copy()
        current_df["MatchRate"] = rate_to_percent(current_df[item_name])
        current_df["situation"] = condition
        all_df.append(current_df[["Directory", "MatchRate", "situation"]])
    if not all_df:
        return pd.DataFrame(columns=["Directory", "MatchRate", "situation"])
    return pd.concat(all_df, ignore_index=True)


def collect_existing_overall_points(
    root_dir: Path,
    conditions: list[str],
    all_items_csv_name: str,
) -> pd.DataFrame:
    """Collect per-subject average match-rate across all FIM items for each condition.

    Averages item-level match rates per subject rather than using the pre-computed
    AverageMatchRate column, so that the denominator is consistent across conditions
    even when some items are missing for a subset of conditions.

    Args:
        root_dir: Ablation root directory.
        conditions: Ordered list of condition names to collect.
        all_items_csv_name: Filename of the all-items summary CSV inside each condition dir.

    Returns:
        Long-format DataFrame with columns [Directory, MatchRate, situation].
    """
    all_df = []
    for condition in conditions:
        condition_dir = find_condition_dir(root_dir, condition)
        if condition_dir is None:
            continue
        csv_path = condition_dir / all_items_csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # Exclude metadata/aggregate columns and <Action>_to_<Item> columns;
        # keep only per-item overall match rate columns (one column per FIM item)
        item_cols = [
            c for c in df.columns
            if c not in {"Directory", "AverageMatchRate"} and "_to_" not in c
        ]
        if not item_cols:
            continue
        current_df = df[["Directory"] + item_cols].copy()
        current_df[item_cols] = current_df[item_cols].apply(rate_to_percent)
        # Compute per-subject mean across all FIM item columns
        current_df["MatchRate"] = current_df[item_cols].mean(axis=1, skipna=True)
        current_df = current_df.dropna(subset=["MatchRate"])
        current_df["situation"] = condition
        all_df.append(current_df[["Directory", "MatchRate", "situation"]])
    if not all_df:
        return pd.DataFrame(columns=["Directory", "MatchRate", "situation"])
    return pd.concat(all_df, ignore_index=True)


def find_best_action(condition_dir: Path, item_name: str) -> str | None:
    """Find the action with the highest balanced accuracy for *item_name*.

    Lookup priority:
        1. Summary.csv — pre-computed best action per FIM item (preferred).
        2. results.csv / Accuracies.csv — derive best action from raw accuracy rows.

    Args:
        condition_dir: Root directory of a single ablation condition.
        item_name: FIM item name to look up.

    Returns:
        Action name string, or None if no matching data is found.
    """
    # Primary: Summary.csv contains a pre-computed Max_Balanced_Acc_Action per FIM item
    summary_path = condition_dir / "Summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
        if {"FIMItem", "Max_Balanced_Acc_Action"} <= set(summary_df.columns):
            item_df = summary_df[
                summary_df["FIMItem"].map(lambda v: label_matches(v, item_name))
            ]
            if not item_df.empty:
                return str(item_df.iloc[0]["Max_Balanced_Acc_Action"])

    # Fallback: derive best action by scanning raw per-action accuracy rows
    for result_file in ("results.csv", "Accuracies.csv"):
        results_path = condition_dir / result_file
        if not results_path.exists():
            continue
        results_df = pd.read_csv(results_path, encoding="utf-8-sig")
        # Support multiple column name conventions for balanced accuracy
        accuracy_col = next(
            (c for c in ("BalancedAccuracy", "Balanced_Accuracy", "Balanced Accuracy")
             if c in results_df.columns),
            None,
        )
        if accuracy_col is None or not ({"FromAction", "ToFIM"} <= set(results_df.columns)):
            continue
        # Filter rows to only those matching the target FIM item
        item_df = results_df[
            results_df["ToFIM"].map(lambda v: label_matches(v, item_name))
        ].copy()
        if "Seed" in item_df.columns:
            # Prefer the pre-aggregated "Mean" seed row when available to avoid
            # picking the best action from a single lucky seed
            mean_rows = item_df[item_df["Seed"].astype(str) == "Mean"].copy()
            item_df = mean_rows if not mean_rows.empty else item_df
        item_df[accuracy_col] = pd.to_numeric(item_df[accuracy_col], errors="coerce")
        item_df = item_df.dropna(subset=[accuracy_col])
        if not item_df.empty:
            return str(item_df.loc[item_df[accuracy_col].idxmax(), "FromAction"])

    return None


def collect_action_to_item_points(
    root_dir: Path,
    conditions: list[str],
    action_to_item_dir_name: str,
    item_file_name: str,
) -> pd.DataFrame:
    """Collect MeanAction and MaxAction match-rate points for a single FIM item.

    Each condition contributes two rows per subject:
    - MeanAction: average match rate across all actions (AverageMatchRate column).
    - MaxAction: match rate of the single best-performing action for that item.

    Args:
        root_dir: Ablation root directory.
        conditions: Ordered list of condition names to collect.
        action_to_item_dir_name: Subdirectory name inside each condition dir that holds
                                  per-item action-to-item CSV files.
        item_file_name: CSV filename for the target FIM item (e.g. "Eating.csv").

    Returns:
        Long-format DataFrame with columns [Directory, MatchRate, situation].
    """
    item_name = Path(item_file_name).stem
    all_df = []
    for condition in conditions:
        condition_dir = find_condition_dir(root_dir, condition)
        if condition_dir is None:
            print(f"No directory found for condition {condition}. Skipping...")
            continue
        csv_path = condition_dir / action_to_item_dir_name / item_file_name
        if not csv_path.exists():
            print(f"File {csv_path} does not exist. Skipping...")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        missing = {"Directory", "AverageMatchRate"} - set(df.columns)
        if missing:
            print(f"Missing columns in {csv_path}: {sorted(missing)}")
            continue

        # MeanAction: AverageMatchRate already aggregates all action columns per subject
        mean_df = df[["Directory", "AverageMatchRate"]].dropna().copy()
        mean_df["MatchRate"] = rate_to_percent(mean_df["AverageMatchRate"])
        mean_df["situation"] = f"{condition}_MeanAction"
        all_df.append(mean_df[["Directory", "MatchRate", "situation"]])

        # MaxAction: use only the column for the single best-performing action
        best_action = find_best_action(condition_dir, item_name)
        if best_action is None:
            print(
                f"Could not find best action for {item_name} under {condition_dir}. "
                "Skipping MaxAction."
            )
            continue
        # Fuzzy column lookup in case the action name has minor formatting differences
        best_action_column = find_matching_column(df.columns, best_action)
        if best_action_column is None:
            print(f"Best action {best_action} is not in {csv_path}. Skipping MaxAction.")
            continue
        max_df = df[["Directory", best_action_column]].dropna().copy()
        max_df["MatchRate"] = rate_to_percent(max_df[best_action_column])
        max_df["situation"] = f"{condition}_MaxAction"
        all_df.append(max_df[["Directory", "MatchRate", "situation"]])

    if not all_df:
        return pd.DataFrame(columns=["Directory", "MatchRate", "situation"])
    return pd.concat(all_df, ignore_index=True)


def collect_action_to_item_overall_points(
    root_dir: Path,
    conditions: list[str],
    action_to_item_dir_name: str,
) -> pd.DataFrame:
    """Collect overall MeanAction/MaxAction match-rate by averaging across all FIM items.

    For each condition:
    - MeanAction: per-subject mean of each item's AverageMatchRate across all items.
    - MaxAction: per-subject mean of each item's best-action match rate across all items.

    Items are joined on Directory with an outer merge so that subjects missing from
    some item CSVs still appear in the result (their missing items contribute NaN,
    which are excluded from the row mean via skipna=True).

    Args:
        root_dir: Ablation root directory.
        conditions: Ordered list of condition names to collect.
        action_to_item_dir_name: Subdirectory name inside each condition dir.

    Returns:
        Long-format DataFrame with columns [Directory, MatchRate, situation].
    """
    all_df = []
    for condition in conditions:
        condition_dir = find_condition_dir(root_dir, condition)
        if condition_dir is None:
            continue
        action_to_item_dir = condition_dir / action_to_item_dir_name
        if not action_to_item_dir.exists():
            continue

        # Accumulate one DataFrame per FIM item for each action type
        mean_parts: list[pd.DataFrame] = []
        max_parts: list[pd.DataFrame] = []
        for item_csv_path in sorted(action_to_item_dir.glob("*.csv")):
            item_name = item_csv_path.stem
            df = pd.read_csv(item_csv_path, encoding="utf-8-sig")
            if {"Directory", "AverageMatchRate"} <= set(df.columns):
                # Rename the value column to the item name so items can be merged on Directory
                mean_parts.append(
                    df[["Directory", "AverageMatchRate"]].rename(
                        columns={"AverageMatchRate": item_name}
                    )
                )
            best_action = find_best_action(condition_dir, item_name)
            best_col = (
                find_matching_column(df.columns, best_action)
                if best_action is not None else None
            )
            if best_col is not None:
                max_parts.append(
                    df[["Directory", best_col]].rename(columns={best_col: item_name})
                )

        for action_type, item_parts in (("MeanAction", mean_parts), ("MaxAction", max_parts)):
            if not item_parts:
                continue
            # Outer-merge all item DataFrames on Directory to build a wide table
            combined = item_parts[0]
            for part in item_parts[1:]:
                combined = combined.merge(part, on="Directory", how="outer")
            item_cols = [c for c in combined.columns if c != "Directory"]
            combined[item_cols] = combined[item_cols].apply(rate_to_percent)
            # Per-subject average across all FIM items; skipna so partial coverage is handled
            combined["MatchRate"] = combined[item_cols].mean(axis=1, skipna=True)
            combined = combined.dropna(subset=["MatchRate"])
            combined["situation"] = f"{condition}_{action_type}"
            all_df.append(combined[["Directory", "MatchRate", "situation"]])

    if not all_df:
        return pd.DataFrame(columns=["Directory", "MatchRate", "situation"])
    return pd.concat(all_df, ignore_index=True)


def plot_subject_boxplot(
    df: pd.DataFrame,
    output_prefix: Path,
    title: str,
    condition_order: list[str],
) -> None:
    """Plot a violin+box chart with one ablation condition per x-axis position.

    Each box represents the distribution of match rates across subjects.
    Mean values are annotated in red above each box.
    Outputs raw data CSV, statistics CSV, and figure files (PNG/PDF/SVG).

    Args:
        df: Long-format DataFrame with columns [Directory, MatchRate, situation].
        output_prefix: Path prefix for output files (suffix is added per format).
        title: Plot title string.
        condition_order: Desired left-to-right order of conditions on the x-axis.
    """
    if df.empty:
        print(f"No data available for {title}. Skipping plot generation.")
        return

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Save the raw per-subject data that was used to draw the plot
    df.to_csv(output_prefix.with_suffix(".csv"), index=False)

    data = df.copy()
    # Filter condition_order to only those actually present in the data
    present = set(data["situation"])
    data["situation"] = pd.Categorical(
        data["situation"],
        categories=[c for c in condition_order if c in present],
        ordered=True,
    )
    data = data.sort_values("situation")
    # Replace underscores with newlines so long condition names wrap on the x-axis
    data["situation_label"] = data["situation"].astype(str).str.replace("_", "\n")

    label_order = [c.replace("_", "\n") for c in condition_order if c in present]
    # Strip _MeanAction/_MaxAction suffix to group conditions by base name for palette assignment,
    # so MeanAction and MaxAction variants of the same condition share the same hue
    base_order = list(dict.fromkeys(
        re.sub(r"_(MeanAction|MaxAction)$", "", c) for c in condition_order
    ))
    base_palette = dict(zip(base_order, ablation_colors(len(base_order))))
    palette = {
        c.replace("_", "\n"): base_palette[re.sub(r"_(MeanAction|MaxAction)$", "", c)]
        for c in condition_order
    }
    # Box fill is white so the colored violin behind it remains visible
    box_palette = {label: "white" for label in label_order}

    fontsize = 42
    # Scale figure width with the number of conditions so labels don't overlap
    fig = plt.figure(figsize=(max(16, len(label_order) * 4), 14))
    plt.rcParams["font.size"] = fontsize
    ax = fig.add_subplot(111)

    sns.violinplot(
        data=data, x="situation_label", y="MatchRate",
        hue="situation_label", order=label_order, hue_order=label_order,
        palette=palette, inner=None, cut=0, linewidth=1.5, alpha=0.35,
        legend=False, ax=ax,
    )
    sns.boxplot(
        data=data, x="situation_label", y="MatchRate",
        hue="situation_label", order=label_order, hue_order=label_order,
        palette=box_palette, showmeans=True,
        medianprops=dict(color="black", linewidth=3, linestyle="dashed"),
        whiskerprops=dict(linewidth=2),
        meanprops=dict(
            color="white", marker="+", markeredgecolor="white",
            markerfacecolor="red", markeredgewidth=5, markersize=18,
        ),
        width=0.28, linewidth=3, fliersize=0, legend=False, ax=ax,
    )

    # Annotate each box with the numeric group mean placed just above the whisker top
    for i, label in enumerate(label_order):
        group = data[data["situation_label"] == label]["MatchRate"]
        if not group.empty:
            ax.text(
                i, group.max() + 1.0, f"{group.mean():.2f}",
                ha="center", va="bottom",
                fontsize=fontsize - 12, fontweight="bold", color="red",
            )

    ax.set_xlabel("")
    ax.set_ylabel("Match Rate (%)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 8)
    ax.tick_params(axis="y", labelsize=fontsize - 4)
    # Clamp y-axis to [0, 105] with 5/8 pt margin, avoiding degenerate ranges
    y_min = max(0, data["MatchRate"].min() - 5)
    y_max = min(105, data["MatchRate"].max() + 8)
    if y_min < y_max:
        ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", linewidth=1, alpha=0.7)

    # Save descriptive statistics alongside the figure for downstream analysis
    stats_df = (
        data.groupby("situation", observed=True)["MatchRate"]
        .agg(
            count="count",
            mean="mean",
            std=lambda values: values.std(ddof=1),
            median="median",
            min="min",
            max="max",
        )
        .reset_index()
    )
    stats_df.to_csv(
        output_prefix.with_name(output_prefix.name + "_Stats.csv"), index=False
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_action_type_boxplot(
    df: pd.DataFrame,
    output_prefix: Path,
    title: str,
    condition_order: list[str],
) -> None:
    """Plot a split violin+box chart with MeanAction vs MaxAction side-by-side.

    Conditions that have neither action type are dropped automatically.
    The split violin shows MeanAction on one half and MaxAction on the other,
    making it easy to compare the two aggregation strategies within each condition.
    Outputs raw data CSV, statistics CSV, and figure files (PNG/PDF/SVG).

    Args:
        df: Long-format DataFrame with columns [Directory, MatchRate, situation].
            situation values must follow the "<condition>_MeanAction" / "<condition>_MaxAction"
            naming convention.
        output_prefix: Path prefix for output files.
        title: Plot title string.
        condition_order: Desired left-to-right order of base condition names.
    """
    if df.empty:
        print(f"No data available for {title}. Skipping plot generation.")
        return

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_prefix.with_suffix(".csv"), index=False)

    data = df.copy()
    # Extract the action type suffix into a separate column for use as hue
    data["ActionType"] = data["situation"].str.extract(r"_(MeanAction|MaxAction)$")
    # Remove the suffix from situation so it holds only the base condition name
    data["situation"] = data["situation"].str.replace(
        r"_(MeanAction|MaxAction)$", "", regex=True
    )
    # Drop rows that had no action type suffix (not MeanAction or MaxAction)
    data = data.dropna(subset=["ActionType"])
    present = set(data["situation"].astype(str))
    data["situation"] = pd.Categorical(
        data["situation"],
        categories=[c for c in condition_order if c in present],
        ordered=True,
    )
    data["situation_label"] = data["situation"].astype(str).str.replace("_", "\n")
    data = data.sort_values(["situation", "ActionType"])

    label_order = [c.replace("_", "\n") for c in condition_order if c in present]
    hue_order = ["MeanAction", "MaxAction"]
    colors = ablation_colors(len(condition_order))
    # Assign the first two distinct colors to MeanAction and MaxAction respectively
    palette = {"MeanAction": colors[0], "MaxAction": colors[1]}
    box_palette = {"MeanAction": "white", "MaxAction": "white"}

    fontsize = 42
    fig = plt.figure(figsize=(max(18, len(label_order) * 5), 14))
    plt.rcParams["font.size"] = fontsize
    ax = fig.add_subplot(111)

    sns.violinplot(
        data=data, x="situation_label", y="MatchRate",
        hue="ActionType", order=label_order, hue_order=hue_order,
        palette=palette, split=True, inner=None, cut=0,
        linewidth=1.5, alpha=0.35, ax=ax,
    )
    sns.boxplot(
        data=data, x="situation_label", y="MatchRate",
        hue="ActionType", order=label_order, hue_order=hue_order,
        palette=box_palette, showmeans=True,
        medianprops=dict(color="black", linewidth=3, linestyle="dashed"),
        whiskerprops=dict(linewidth=2),
        meanprops=dict(
            color="white", marker="+", markeredgecolor="white",
            markerfacecolor="red", markeredgewidth=5, markersize=18,
        ),
        width=0.28, linewidth=3, fliersize=0, ax=ax,
    )

    # Keep only the hue legend entries (seaborn may add duplicate entries)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:len(hue_order)], labels[:len(hue_order)],
        fontsize=fontsize - 12, loc="upper right",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Match Rate (%)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 8)
    ax.tick_params(axis="y", labelsize=fontsize - 4)
    y_min = max(0, data["MatchRate"].min() - 5)
    y_max = min(105, data["MatchRate"].max() + 8)
    if y_min < y_max:
        ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", linewidth=1, alpha=0.7)

    stats_df = (
        data.groupby(["situation", "ActionType"], observed=True)["MatchRate"]
        .agg(
            count="count",
            mean="mean",
            std=lambda values: values.std(ddof=1),
            median="median",
            min="min",
            max="max",
        )
        .reset_index()
    )
    stats_df.to_csv(
        output_prefix.with_name(output_prefix.name + "_Stats.csv"), index=False
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_all_items_summary(
    root_dir: Path,
    save_dir: Path,
    conditions: list[str],
    all_items_csv_name: str,
    action_to_item_dir_name: str,
) -> None:
    """Generate the overall match-rate summary plot across all FIM items.

    Produces two types of plots:
    1. One combined plot averaging all FIM items (overall ablation performance).
    2. One plot per FIM item column found in the all-items CSV.
    """
    # Merge the item-level overall points with the action-to-item overall points
    # so that MeanAction/MaxAction variants appear alongside the base condition in one plot
    overall_df = pd.concat(
        [
            collect_existing_overall_points(root_dir, conditions, all_items_csv_name),
            collect_action_to_item_overall_points(
                root_dir, conditions, action_to_item_dir_name
            ),
        ],
        ignore_index=True,
    )
    # Interleave each base condition with its MeanAction/MaxAction variants so that
    # related conditions appear adjacent on the x-axis; then filter to data-present only
    overall_condition_order = []
    for condition in conditions:
        overall_condition_order.extend(
            [condition, f"{condition}_MeanAction", f"{condition}_MaxAction"]
        )
    present = set(overall_df["situation"])
    overall_condition_order = [c for c in overall_condition_order if c in present]

    plot_subject_boxplot(
        overall_df,
        save_dir / "AverageMatchRate_AllItems",
        "Cross-Subject Ablation Results",
        overall_condition_order,
    )

    # Collect the union of all per-item column names that appear across any condition's CSV;
    # this ensures items present in only a subset of conditions are still plotted
    item_columns: set[str] = set()
    for condition in conditions:
        condition_dir = find_condition_dir(root_dir, condition)
        if condition_dir is None:
            continue
        csv_path = condition_dir / all_items_csv_name
        if not csv_path.exists():
            continue
        columns = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns
        item_columns.update(
            c for c in columns
            if c not in {"Directory", "AverageMatchRate"} and "_to_" not in c
        )

    for item_column in sorted(item_columns):
        item_df = collect_summary_points(
            root_dir, conditions, Path(all_items_csv_name), item_column
        )
        safe_col = re.sub(r'[\\/:*?"<>|]+', "_", item_column)  # strip chars illegal in filenames
        plot_subject_boxplot(
            item_df,
            save_dir / f"MatchRate_{safe_col}",
            f"{item_column}'s Cross-Subject Ablation Results",
            conditions,
        )


def plot_action_to_item_summaries(
    root_dir: Path,
    save_dir: Path,
    conditions: list[str],
    all_items_csv_name: str,
    action_to_item_dir_name: str,
) -> None:
    """Generate per-item match-rate plots that include MeanAction/MaxAction breakdowns.

    For each FIM item that has an action-to-item CSV in any condition, this produces
    a plot that combines the item-level match rate (from all-items CSV) with the
    MeanAction and MaxAction variants (from the action-to-item directory).
    """
    # Collect the union of item CSV filenames across all conditions so that items
    # available in only some conditions are still included in the plot
    item_file_names: set[str] = set()
    for condition in conditions:
        condition_dir = find_condition_dir(root_dir, condition)
        if condition_dir is None:
            continue
        action_to_item_dir = condition_dir / action_to_item_dir_name
        if not action_to_item_dir.exists():
            continue
        item_file_names.update(p.name for p in action_to_item_dir.glob("*.csv"))

    # Interleave each base condition with its action-type variants for x-axis ordering
    action_condition_order = []
    for condition in conditions:
        action_condition_order.extend(
            [condition, f"{condition}_MeanAction", f"{condition}_MaxAction"]
        )

    for item_file_name in sorted(item_file_names):
        item_name = Path(item_file_name).stem
        # Combine item-level points (base condition) with action-type points in one DataFrame
        item_df = pd.concat(
            [
                collect_existing_item_points(
                    root_dir, conditions, all_items_csv_name, item_name
                ),
                collect_action_to_item_points(
                    root_dir, conditions, action_to_item_dir_name, item_file_name
                ),
            ],
            ignore_index=True,
        )
        safe_name = re.sub(r'[\\/:*?"<>|]+', "_", item_name)  # strip chars illegal in filenames
        plot_subject_boxplot(
            item_df,
            save_dir / f"MatchRate_{safe_name}",
            f"{item_name}'s Cross-Subject Ablation Results",
            action_condition_order,
        )


def ablationCrossSubjectVariables(
    root_dir: str,
    all_items_csv_name: str = "CrossSubjectVariables_all_items.csv",
    action_to_item_dir_name: str = "CrossSubjectVariables_action_to_item",
) -> None:
    """Generate all cross-subject ablation boxplots under *root_dir*.

    Plots are written to <root_dir>/AblationPlotsCrossSubject/.
    """
    root_path = Path(root_dir)
    save_dir = root_path / "AblationPlotsCrossSubject"
    save_dir.mkdir(parents=True, exist_ok=True)
    setPltFont()

    # Read condition order from order.csv if present; otherwise sort non-plot subdirectory names
    order_path = root_path / "order.csv"
    if order_path.exists():
        conditions = pd.read_csv(order_path, header=None)[0].dropna().astype(str).tolist()
    else:
        conditions = sorted(
            path.name
            for path in root_path.iterdir()
            if path.is_dir() and not path.name.startswith("AblationPlots")
        )
    if not conditions:
        print(f"No conditions found in {root_path}.")
        return

    plot_all_items_summary(
        root_path, save_dir, conditions, all_items_csv_name, action_to_item_dir_name
    )
    plot_action_to_item_summaries(
        root_path, save_dir, conditions, all_items_csv_name, action_to_item_dir_name
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate ablation boxplots where each Directory row in "
            "CrossSubjectVariables outputs is one subject point."
        )
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        help="Ablation root directory. If omitted, it is read from standard input.",
    )
    parser.add_argument(
        "--all-items-csv-name",
        default="CrossSubjectVariables_all_items.csv",
        help="CSV file name under each ablation condition directory.",
    )
    parser.add_argument(
        "--action-to-item-dir-name",
        default="CrossSubjectVariables_action_to_item",
        help="Directory name under each condition for <Action>_to_<Item> summaries.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the ablation plot pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    setPltFont()

    root_dir = args.root_dir or input("Input the directory path: ")
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return

    ablationCrossSubjectVariables(
        root_dir,
        all_items_csv_name=args.all_items_csv_name,
        action_to_item_dir_name=args.action_to_item_dir_name,
    )
    print(f"Ablation cross-subject plots generated in {root_dir}.")


if __name__ == "__main__":
    main()
