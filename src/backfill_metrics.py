"""
backfill_metrics.py - backfill per-class accuracy and detailed metrics into result CSVs

Recomputes, from the per-fold PredictTruelabel.csv files, the metrics that the
experiment scripts record in Accuracies.csv / results.csv:
    - Class{c}_Accuracy: per-class recall [%] (diagonal of the row-normalized
      confusion matrix, identical to src/training/test.py)
    - F1Score / MCC / Precision / Recall / AUCROC (identical definitions to
      src/training/test.py: binary with pos_label=0 for 2 classes,
      weighted / ovr otherwise; 0-1 scale)
and fills them into the existing CSVs at all three levels
(Stratified_Seed*/Accuracies.csv, <NN>-<Item>/Accuracies.csv, and the
experiment-root Accuracies.csv / results.csv), following the same
fold-mean -> seed-mean aggregation as the experiment scripts.

Intended for experiment directories produced before per-class accuracy and
detailed metrics were computed independently of wandb (their Class columns are
empty and the metric columns are absent).

Usage:
    $ python -m src.backfill_metrics <experiment_root> [--force] [--dry-run]

    <experiment_root>: directory containing situations.yaml and <NN>-<Item>/
    --force:   overwrite columns that already contain values
    --dry-run: only compare recomputed values with existing ones (no writes)

Config: reads classes from <experiment_root>/situations.yaml
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import yaml

from src.data.feeder.results import OTHER_METRIC_KEYS


def calc_fold_metrics(df: pd.DataFrame, classes: list) -> dict:
    """Compute the per-fold metrics from one PredictTruelabel.csv DataFrame.

    Args:
        df: DataFrame with True / Predicted / Prob_* columns.
        classes: class label list from situations.yaml (e.g. [6, 7]);
            label index i in True/Predicted corresponds to classes[i].

    Returns:
        dict with Class{c}_Accuracy [%] and F1Score/MCC/Precision/Recall/AUCROC (0-1).
    """
    true_label = df["True"].to_numpy()
    pred_label = df["Predicted"].to_numpy()
    classes_num = len(classes)
    result = {}

    for i, c in enumerate(classes):
        n_true = np.sum(true_label == i)
        # test.py yields 0 for classes absent from the fold (zero-sum row)
        recall_pct = 0.0 if n_true == 0 else 100.0 * \
            np.sum((true_label == i) & (pred_label == i)) / n_true
        result[f"Class{c}_Accuracy"] = recall_pct

    result["F1Score"] = metrics.f1_score(true_label, pred_label, average='weighted') \
        if classes_num > 2 else metrics.f1_score(true_label, pred_label, average='binary', pos_label=0)
    result["MCC"] = metrics.matthews_corrcoef(true_label, pred_label)
    result["Precision"] = metrics.precision_score(true_label, pred_label, average='weighted') \
        if classes_num > 2 else metrics.precision_score(true_label, pred_label, average='binary', pos_label=0)
    result["Recall"] = metrics.recall_score(true_label, pred_label, average='weighted') \
        if classes_num > 2 else metrics.recall_score(true_label, pred_label, average='binary', pos_label=0)
    prob_cols = [f"Prob_{i}" for i in range(classes_num)]
    try:
        if classes_num > 2:
            result["AUCROC"] = metrics.roc_auc_score(
                true_label, df[prob_cols].to_numpy(), multi_class='ovr')
        else:
            # class 0 is treated as positive, so invert y_true to use Prob_0
            result["AUCROC"] = metrics.roc_auc_score(
                1 - true_label, df["Prob_0"].to_numpy())
    except ValueError as e:
        print(f"Warning: AUC-ROC could not be computed: {e}")
        result["AUCROC"] = float('nan')
    return result


def collect_metrics(rootPath: str, classes: list) -> dict:
    """Scan <NN>-<Item>/Stratified_Seed*/Fold*/PredictTruelabel.csv and compute metrics.

    Returns:
        Nested dict: {item_dir_name: {seed: {fold: {metric: value}}}}.
    """
    all_metrics = {}
    item_dirs = sorted([d for d in os.listdir(rootPath) if os.path.isdir(
        os.path.join(rootPath, d)) and re.match(r'^\d{2}-', d)])
    for item_dir in item_dirs:
        item_path = os.path.join(rootPath, item_dir)
        seed_dirs = sorted([d for d in os.listdir(item_path) if os.path.isdir(
            os.path.join(item_path, d)) and d.startswith('Stratified_Seed')])
        seed_metrics = {}
        for seed_dir in seed_dirs:
            seed = int(re.match(r'Stratified_Seed(\d+)', seed_dir).group(1))
            seed_path = os.path.join(item_path, seed_dir)
            fold_dirs = sorted([d for d in os.listdir(seed_path) if os.path.isdir(
                os.path.join(seed_path, d)) and d.startswith('Fold')])
            fold_metrics = {}
            for fold_dir in fold_dirs:
                fold = int(re.match(r'Fold(\d+)', fold_dir).group(1))
                csv_path = os.path.join(
                    seed_path, fold_dir, 'PredictTruelabel.csv')
                if not os.path.exists(csv_path):
                    print(f"CSV file not found: {csv_path}")
                    continue
                fold_df = pd.read_csv(csv_path, header=0)
                fold_metrics[fold] = calc_fold_metrics(fold_df, classes)
            if fold_metrics:
                seed_metrics[seed] = fold_metrics
        if seed_metrics:
            all_metrics[item_dir] = seed_metrics
    return all_metrics


def _mean_over(dicts: list) -> dict:
    """Mean of each metric over a list of metric dicts (fold or seed level)."""
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def _fill_df(df: pd.DataFrame, key_column: str, values_by_key: dict,
             metric_columns: list, force: bool, dry_run: bool, csv_name: str) -> bool:
    """Fill metric columns of df, matching rows by key_column.

    Args:
        df: target DataFrame (modified in place).
        key_column: column used to match rows ("Fold", "Seed", or "ToFIM").
        values_by_key: {key(str): {metric: value}}.
        metric_columns: columns to fill, in output order.
        force: overwrite columns that already contain values.
        dry_run: compare against existing values instead of writing.
        csv_name: label used in log messages.

    Returns:
        True if df was modified.
    """
    modified = False
    for col in metric_columns:
        if col not in df.columns:
            df[col] = np.nan
        has_values = df[col].notna().any()
        if dry_run:
            if not has_values:
                continue
            # compare recomputed values with the stored ones
            diffs = []
            for idx, row in df.iterrows():
                key = str(row[key_column])
                if key in values_by_key and pd.notna(row[col]):
                    diffs.append(
                        abs(float(row[col]) - values_by_key[key][col]))
            if diffs:
                print(f"  [dry-run] {csv_name} {col}: "
                      f"n={len(diffs)}, max|diff|={max(diffs):.6f}")
            continue
        if has_values and not force:
            print(f"  Skip {csv_name} {col}: already filled (use --force to overwrite)")
            continue
        for idx, row in df.iterrows():
            key = str(row[key_column])
            if key in values_by_key:
                df.at[idx, col] = values_by_key[key][col]
                modified = True
    return modified


def backfill_metrics(rootPath: str, force: bool = False, dry_run: bool = False):
    """Backfill metric columns of the result CSVs under one experiment root.

    Processing flow:
        1. Read classes from situations.yaml
        2. Recompute per-fold metrics from every PredictTruelabel.csv
        3. Fill Stratified_Seed*/Accuracies.csv (fold rows + fold-mean row)
        4. Fill <NN>-<Item>/Accuracies.csv (seed rows = fold means + seed-mean row)
        5. Fill root Accuracies.csv and results.csv (matched by ToFIM)

    Args:
        rootPath: experiment root containing situations.yaml and <NN>-<Item>/.
        force: overwrite columns that already contain values.
        dry_run: only report differences against existing values; write nothing.

    Raises:
        FileNotFoundError: situations.yaml is missing under rootPath.
    """
    situations_path = os.path.join(rootPath, "situations.yaml")
    if not os.path.exists(situations_path):
        raise FileNotFoundError(f"situations.yaml not found in {rootPath}")
    with open(situations_path, "r") as f:
        situations = yaml.safe_load(f)
    classes = situations["classes"]
    class_columns = [f"Class{c}_Accuracy" for c in classes]
    metric_columns = class_columns + OTHER_METRIC_KEYS

    all_metrics = collect_metrics(rootPath, classes)
    if not all_metrics:
        print(f"No PredictTruelabel.csv found under {rootPath}")
        return

    # per-item fold means (per seed) and seed means, reused at every level
    item_seed_means = {}   # {item_dir: {seed: metric dict}}
    item_means = {}        # {item_dir: metric dict}
    for item_dir, seed_metrics in all_metrics.items():
        seed_means = {seed: _mean_over(list(folds.values()))
                      for seed, folds in seed_metrics.items()}
        item_seed_means[item_dir] = seed_means
        item_means[item_dir] = _mean_over(list(seed_means.values()))

    # ------------------------------------------------------------------
    # Fill Stratified_Seed*/Accuracies.csv (fold rows + Mean row)
    # ------------------------------------------------------------------
    for item_dir, seed_metrics in all_metrics.items():
        for seed, fold_metrics in seed_metrics.items():
            csv_path = os.path.join(
                rootPath, item_dir, f"Stratified_Seed{seed}", "Accuracies.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, header=0)
            values = {str(fold): m for fold, m in fold_metrics.items()}
            values["Mean"] = item_seed_means[item_dir][seed]
            if _fill_df(df, "Fold", values, metric_columns, force, dry_run,
                        f"{item_dir}/Stratified_Seed{seed}/Accuracies.csv"):
                df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Fill <NN>-<Item>/Accuracies.csv (seed rows + Mean row)
    # ------------------------------------------------------------------
    for item_dir, seed_means in item_seed_means.items():
        csv_path = os.path.join(rootPath, item_dir, "Accuracies.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, header=0)
        # Seed values are recorded as floats (e.g. "100.0") in the CSV
        values = {}
        for seed, m in seed_means.items():
            values[str(seed)] = m
            values[str(float(seed))] = m
        values["Mean"] = item_means[item_dir]
        if _fill_df(df, "Seed", values, metric_columns, force, dry_run,
                    f"{item_dir}/Accuracies.csv"):
            df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Fill root Accuracies.csv (matched by ToFIM and Seed)
    # ------------------------------------------------------------------
    def item_name(item_dir: str) -> str:
        # "08-Bed_Chair_Wheelchair" -> "Bed_Chair_Wheelchair"
        # "01-Task_to_08-Item" -> "Item"
        tail = item_dir.split('_to_')[-1] if '_to_' in item_dir else item_dir
        return tail.split('-', 1)[-1]

    def action_name(item_dir: str) -> str:
        # "01-Task_to_08-Item" -> "Task"; single-item dirs have no action
        return item_dir.split('_to_')[0].split('-', 1)[-1] if '_to_' in item_dir else None

    root_acc_path = os.path.join(rootPath, "Accuracies.csv")
    if os.path.exists(root_acc_path):
        df = pd.read_csv(root_acc_path, header=0)
        for col in metric_columns:
            if col not in df.columns:
                df[col] = np.nan
        modified = False
        for item_dir, seed_means in item_seed_means.items():
            item_mask = df["ToFIM"] == item_name(item_dir)
            # examEachSeed layout also distinguishes rows by the input action
            if "FromAction" in df.columns and action_name(item_dir) is not None:
                item_mask &= df["FromAction"] == action_name(item_dir)
            sub_df = df[item_mask].copy()
            values = {}
            for seed, m in seed_means.items():
                values[str(seed)] = m
                values[str(float(seed))] = m
            values["Mean"] = item_means[item_dir]
            if _fill_df(sub_df, "Seed", values, metric_columns, force, dry_run,
                        f"Accuracies.csv[{item_name(item_dir)}]"):
                df.loc[item_mask, metric_columns] = sub_df[metric_columns]
                modified = True
        if modified:
            df.to_csv(root_acc_path, index=False)

    # ------------------------------------------------------------------
    # Fill root results.csv (Mean rows only, matched by ToFIM)
    # ------------------------------------------------------------------
    results_path = os.path.join(rootPath, "results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path, header=0)
        for col in metric_columns:
            if col not in df.columns:
                df[col] = np.nan
        modified = False
        for item_dir, m in item_means.items():
            item_mask = df["ToFIM"] == item_name(item_dir)
            if "FromAction" in df.columns and action_name(item_dir) is not None:
                item_mask &= df["FromAction"] == action_name(item_dir)
            sub_df = df[item_mask].copy()
            if _fill_df(sub_df, "ToFIM", {item_name(item_dir): m},
                        metric_columns, force, dry_run,
                        f"results.csv[{item_name(item_dir)}]"):
                df.loc[item_mask, metric_columns] = sub_df[metric_columns]
                modified = True
        if modified:
            df.to_csv(results_path, index=False)

    print(f"{'Checked' if dry_run else 'Backfilled'} {rootPath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill per-class accuracy and detailed metrics into result CSVs")
    parser.add_argument("rootPath", help="experiment root directory "
                        "(contains situations.yaml and <NN>-<Item>/)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite columns that already contain values")
    parser.add_argument("--dry-run", action="store_true",
                        help="compare recomputed values with existing ones without writing")
    args = parser.parse_args()
    backfill_metrics(args.rootPath, force=args.force, dry_run=args.dry_run)
