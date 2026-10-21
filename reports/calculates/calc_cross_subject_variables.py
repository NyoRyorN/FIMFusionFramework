"""Cross-subject variable aggregation for FIM fusion experiments.

Expected directory layout:
    <root>/<Item>/<Stratified_SeedXXX>/FoldX/PredictTruelabel.csv

Fold-level prediction CSVs are concatenated per Stratified directory, then
per-subject match rates are aggregated hierarchically across seeds and items.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


FOLD_DIR_PATTERN = re.compile(r"^Fold(\d+)$")
STRATIFIED_PREFIX = "Stratified_"



def resolve_root_dirs(root_dirs: list[str], scan: bool) -> list[Path]:
    """Validate input paths and return the list of experiment root directories.

    Args:
        root_dirs: Path strings from the command line or stdin.
        scan: When True, each path is treated as a parent and all child
              experiment directories are collected automatically.

    Returns:
        Deduplicated list of resolved experiment root Paths.

    Raises:
        FileNotFoundError: A given path does not exist.
        NotADirectoryError: A given path is not a directory.
        ValueError: No experiment structure found under a path (scan=False only).
    """
    input_paths = [Path(root_dir).expanduser() for root_dir in root_dirs]

    resolved_root_dirs = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {input_path}")
        if not input_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_path}")

        if scan:
            # Collect only direct children that contain at least one Stratified_* subtree
            resolved_root_dirs.extend(
                path for path in sorted(input_path.iterdir())
                if path.is_dir() and any(path.glob(f"**/{STRATIFIED_PREFIX}*"))
            )
        elif any(input_path.glob(f"**/{STRATIFIED_PREFIX}*")):
            resolved_root_dirs.append(input_path)
        else:
            raise ValueError(f"No experiment structure found under {input_path}")

    # Deduplicate using resolve() to handle symlinks pointing to the same target
    unique_root_dirs = []
    seen = set()
    for root_dir in resolved_root_dirs:
        resolved = root_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_root_dirs.append(root_dir)

    return unique_root_dirs


def collect_fold_csvs(stratified_dir: Path, csv_name: str) -> list[Path]:
    """Collect csv_name paths from FoldX subdirectories in ascending fold order.

    Args:
        stratified_dir: A Stratified_SeedXXX directory.
        csv_name: The CSV filename to look for inside each Fold directory.

    Returns:
        List of existing CSV paths ordered by fold index.
    """
    csv_paths = []
    # Sort by fold number so Fold10 comes after Fold9 (integer comparison, not lexicographic);
    # paths not matching the pattern are pushed to the end via the large sentinel value
    fold_key = lambda p: (int(m.group(1)), p.name) if (m := FOLD_DIR_PATTERN.match(p.name)) else (10**9, p.name)  # noqa: E731
    for fold_dir in sorted(stratified_dir.iterdir(), key=fold_key):
        if not fold_dir.is_dir() or FOLD_DIR_PATTERN.match(fold_dir.name) is None:
            continue

        csv_path = fold_dir / csv_name
        if csv_path.exists():
            csv_paths.append(csv_path)

    return csv_paths


def concat_csvs(csv_paths: list[Path], output_path: Path) -> int:
    """Vertically concatenate CSVs, sort by Directory, and write the result.

    Args:
        csv_paths: Source CSV paths to concatenate.
        output_path: Destination path for the combined CSV.

    Returns:
        Number of rows written, or 0 if all inputs are empty.

    Raises:
        ValueError: Headers differ across files, or Directory column is absent.
    """
    csv_dataframe_pairs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if not df.empty:
            csv_dataframe_pairs.append((csv_path, df))

    if not csv_dataframe_pairs:
        return 0

    # Use the first file's header as the reference for consistency checks
    first_columns = list(csv_dataframe_pairs[0][1].columns)
    for csv_path, df in csv_dataframe_pairs:
        if list(df.columns) != first_columns:
            raise ValueError(
                f"Header mismatch in {csv_path}\n"
                f"expected: {first_columns}\n"
                f"actual:   {list(df.columns)}"
            )

    combined_df = pd.concat(
        [df for _, df in csv_dataframe_pairs],
        ignore_index=True,
    )
    if "Directory" not in combined_df.columns:
        raise ValueError(f"Directory column does not exist in {csv_paths[0]}")

    combined_df = combined_df.sort_values("Directory", ascending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    return len(combined_df)


def make_item_summary(
    item_dir: Path,
    combined_csv_name: str,
    summary_output_name: str,
) -> int:
    """Build a per-subject correctness summary across all Stratified seeds in item_dir.

    For each Stratified_* directory, Predicted == True is encoded as 1/0 and
    added as a column. Rows are joined on Directory (outer merge), and
    MatchCount / MatchRate are computed across all seed columns.

    Args:
        item_dir: Directory that directly contains Stratified_* subdirectories.
        combined_csv_name: Filename of the fold-concatenated CSV inside each seed dir.
        summary_output_name: Output filename for the summary CSV.

    Returns:
        Number of rows written, or 0 if no suitable CSV was found.

    Raises:
        ValueError: Required columns missing, duplicate Directory values, or
                    RealScore inconsistency across seeds.
    """
    summary_df = None
    correct_columns = []  # Accumulates one column name per Stratified seed

    for stratified_dir in sorted(item_dir.glob(f"{STRATIFIED_PREFIX}*")):
        if not stratified_dir.is_dir():
            continue

        combined_csv_path = stratified_dir / combined_csv_name
        if not combined_csv_path.exists():
            continue

        df = pd.read_csv(combined_csv_path, encoding="utf-8-sig")
        required_columns = {"Directory", "Predicted", "True", "RealScore"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing columns in {combined_csv_path}: "
                f"{sorted(missing_columns)}"
            )
        if df["Directory"].duplicated().any():
            duplicated = df.loc[df["Directory"].duplicated(), "Directory"].tolist()
            raise ValueError(
                f"Duplicated Directory values in {combined_csv_path}: "
                f"{duplicated[:5]}"
            )

        # Strip the "Stratified_" prefix to get a short label (e.g. "Seed42")
        correct_column = (
            stratified_dir.name[len(STRATIFIED_PREFIX):]
            if stratified_dir.name.startswith(STRATIFIED_PREFIX)
            else stratified_dir.name
        )
        correct_columns.append(correct_column)
        current_df = df[["Directory", "RealScore"]].copy()
        # Encode correctness as 1 (correct) or 0 (incorrect)
        current_df[correct_column] = df["Predicted"].eq(df["True"]).astype(int)

        if summary_df is None:
            summary_df = current_df
            continue

        # Outer-merge on Directory and verify that RealScore is consistent across seeds
        summary_df = summary_df.merge(
            current_df,
            on="Directory",
            how="outer",
            suffixes=("", "__current"),
        )
        current_real_score = "RealScore__current"
        if current_real_score in summary_df.columns:
            inconsistent = (
                summary_df["RealScore"].notna()
                & summary_df[current_real_score].notna()
                & summary_df["RealScore"].ne(summary_df[current_real_score])
            )
            if inconsistent.any():
                directories = summary_df.loc[inconsistent, "Directory"].tolist()
                raise ValueError(
                    f"RealScore mismatch under {item_dir}: {directories[:5]}"
                )
            # Fill NaN values from whichever seed has a value
            summary_df["RealScore"] = summary_df["RealScore"].combine_first(
                summary_df[current_real_score]
            )
            summary_df = summary_df.drop(columns=[current_real_score])

    if summary_df is None:
        return 0

    summary_df = summary_df.sort_values("Directory", ascending=True)
    # MatchRate denominator excludes seeds with NaN (missing data)
    summary_df["MatchCount"] = summary_df[correct_columns].sum(axis=1)
    summary_df["MatchRate"] = summary_df["MatchCount"] / summary_df[
        correct_columns
    ].notna().sum(axis=1)

    output_columns = (
        ["Directory"]
        + correct_columns
        + ["MatchCount", "MatchRate", "RealScore"]
    )
    output_path = item_dir / summary_output_name
    summary_df.to_csv(output_path, columns=output_columns, index=False)
    return len(summary_df)


def make_all_items_summary(
    root_dir: Path,
    item_summary_name: str,
    all_items_output_name: str,
) -> int:
    """Horizontally join per-item MatchRate columns and append AverageMatchRate.

    Reads item_summary_name from each subdirectory of root_dir, renames
    MatchRate to the item directory name, and merges all items on Directory.

    Args:
        root_dir: Directory whose immediate children are Item directories.
        item_summary_name: Summary CSV filename inside each Item directory.
        all_items_output_name: Output filename written to root_dir.

    Returns:
        Number of rows written, or 0 if no summary files were found.
    """
    all_items_df = None
    item_columns = []  # Accumulates one column name per Item directory

    for item_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        item_summary_path = item_dir / item_summary_name
        if not item_summary_path.exists():
            continue

        df = pd.read_csv(item_summary_path, encoding="utf-8-sig")
        required_columns = {"Directory", "MatchRate"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing columns in {item_summary_path}: "
                f"{sorted(missing_columns)}"
            )
        if df["Directory"].duplicated().any():
            duplicated = df.loc[df["Directory"].duplicated(), "Directory"].tolist()
            raise ValueError(
                f"Duplicated Directory values in {item_summary_path}: "
                f"{duplicated[:5]}"
            )

        item_column = item_dir.name
        item_columns.append(item_column)
        current_df = df[["Directory", "MatchRate"]].rename(
            columns={"MatchRate": item_column}
        )

        if all_items_df is None:
            all_items_df = current_df
        else:
            all_items_df = all_items_df.merge(current_df, on="Directory", how="outer")

    if all_items_df is None:
        return 0

    all_items_df = all_items_df.sort_values("Directory", ascending=True)
    all_items_df["AverageMatchRate"] = all_items_df[item_columns].mean(
        axis=1,
        skipna=True,
    )

    output_columns = ["Directory"] + item_columns + ["AverageMatchRate"]
    output_path = root_dir / all_items_output_name
    all_items_df.to_csv(output_path, columns=output_columns, index=False)
    return len(all_items_df)


def write_match_rate_summary(
    summary_paths: list[tuple[str, Path]],
    output_path: Path,
) -> int:
    """Join MatchRate columns from multiple summary CSVs into a single output file.

    Args:
        summary_paths: List of (column_name, csv_path) tuples.
        output_path: Destination path (parent directories are created if needed).

    Returns:
        Number of rows written, or 0 if the input list is empty.

    Raises:
        ValueError: Required columns missing or duplicate Directory values.
    """
    summary_df = None
    summary_columns = []

    for column_name, summary_path in summary_paths:
        df = pd.read_csv(summary_path, encoding="utf-8-sig")
        required_columns = {"Directory", "MatchRate"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing columns in {summary_path}: {sorted(missing_columns)}"
            )
        if df["Directory"].duplicated().any():
            duplicated = df.loc[df["Directory"].duplicated(), "Directory"].tolist()
            raise ValueError(
                f"Duplicated Directory values in {summary_path}: "
                f"{duplicated[:5]}"
            )

        summary_columns.append(column_name)
        current_df = df[["Directory", "MatchRate"]].rename(
            columns={"MatchRate": column_name}
        )

        if summary_df is None:
            summary_df = current_df
        else:
            summary_df = summary_df.merge(current_df, on="Directory", how="outer")

    if summary_df is None:
        return 0

    summary_df = summary_df.sort_values("Directory", ascending=True)
    summary_df["AverageMatchRate"] = summary_df[summary_columns].mean(
        axis=1,
        skipna=True,
    )

    output_columns = ["Directory"] + summary_columns + ["AverageMatchRate"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, columns=output_columns, index=False)
    return len(summary_df)


def make_action_to_item_summaries(
    root_dir: Path,
    item_summary_name: str,
    output_dir_name: str,
) -> int:
    """Group <Action>_to_<Item> directories by Item name and write per-Item summaries.

    Scans root_dir for subdirectories whose names follow the "<Action>_to_<Item>"
    convention, collects each action's MatchRate, and writes one CSV per Item
    into output_dir_name/.

    Args:
        root_dir: Directory containing <Action>_to_<Item> subdirectories.
        item_summary_name: Existing summary CSV filename inside each action-to-item dir.
        output_dir_name: Output directory name created under root_dir.

    Returns:
        Number of CSV files written.
    """
    # Map each Item name to a list of (action_name, summary_path) pairs
    item_summary_paths: dict[str, list[tuple[str, Path]]] = {}

    for action_to_item_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        if "_to_" not in action_to_item_dir.name:
            continue

        action_name, item_name = action_to_item_dir.name.split("_to_", maxsplit=1)
        item_summary_path = action_to_item_dir / item_summary_name
        if not item_summary_path.exists():
            continue

        item_summary_paths.setdefault(item_name, []).append(
            (action_name, item_summary_path)
        )

    output_dir = root_dir / output_dir_name
    output_count = 0
    for item_name, summary_paths in sorted(item_summary_paths.items()):
        row_count = write_match_rate_summary(
            sorted(summary_paths),
            output_dir / f"{item_name}.csv",
        )
        if row_count == 0:
            continue

        output_count += 1
        print(f"Saved {output_dir / f'{item_name}.csv'} ({row_count} directories)")

    return output_count


def process_root_dir(
    root_dir: Path,
    csv_name: str,
    output_name: str,
    summary_output_name: str,
    all_actions_output_name: str,
    action_to_item_output_dir: str,
    all_items_output_name: str,
) -> None:
    """Run the full aggregation pipeline for a single experiment root directory.

    Pipeline steps:
        1. Concatenate Fold CSVs within each Stratified_SeedXXX directory.
        2. Build a per-subject correctness summary for each Item directory.
        3. Merge Item-level summaries within Action directories (all_actions).
        4. Group <Action>_to_<Item> directories by Item and write summaries.
        5. Horizontally join all Item summaries into a root-level final CSV.

    Args:
        root_dir: Experiment root directory to process.
        csv_name: Source CSV filename inside each Fold directory.
        output_name: Filename for the fold-concatenated CSV.
        summary_output_name: Filename for the per-item correctness summary.
        all_actions_output_name: Filename for the all-actions aggregated summary.
        action_to_item_output_dir: Directory name for action-to-item summaries.
        all_items_output_name: Filename for the root-level all-items summary.
    """
    print(f"Processing {root_dir}")

    # Collect all Stratified_* directories anywhere under root_dir, sorted
    stratified_dirs = sorted(
        path for path in root_dir.glob(f"**/{STRATIFIED_PREFIX}*") if path.is_dir()
    )
    if not stratified_dirs:
        print(f"No Stratified_Seed directories found under {root_dir}")
        return

    # Step 1: Concatenate Fold CSVs for each Stratified directory
    combined_count = 0
    for stratified_dir in stratified_dirs:
        csv_paths = collect_fold_csvs(stratified_dir, csv_name)
        if not csv_paths:
            print(f"Skip: no {csv_name} found under {stratified_dir}")
            continue

        output_path = stratified_dir / output_name
        row_count = concat_csvs(csv_paths, output_path)
        combined_count += 1
        print(
            f"Saved {output_path} "
            f"({len(csv_paths)} folds, {row_count} rows)"
        )

    print(f"Done: {combined_count} Stratified_Seed directories processed.")

    # Step 2: Build cross-seed correctness summaries for each Item directory.
    # Derive unique parent directories from already-collected Stratified_* dirs (deduplicated via set)
    stratified_parent_dirs = sorted({d.parent for d in stratified_dirs})
    summary_count = 0
    for summary_dir in stratified_parent_dirs:
        row_count = make_item_summary(
            summary_dir,
            output_name,
            summary_output_name,
        )
        if row_count == 0:
            continue

        summary_count += 1
        print(
            f"Saved {summary_dir / summary_output_name} "
            f"({row_count} directories)"
        )

    print(f"Done: {summary_count} Stratified parent summaries processed.")

    # Step 3: Merge Item summaries inside Action directories
    action_summary_count = 0
    for item_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        row_count = make_all_items_summary(
            item_dir,
            summary_output_name,
            all_actions_output_name,
        )
        if row_count == 0:
            continue

        action_summary_count += 1
        print(
            f"Saved {item_dir / all_actions_output_name} "
            f"({row_count} directories)"
        )

    if action_summary_count:
        print(f"Done: {action_summary_count} Item-to-Action summaries processed.")

    # Step 4: Group <Action>_to_<Item> directories by Item name
    action_to_item_count = make_action_to_item_summaries(
        root_dir,
        summary_output_name,
        action_to_item_output_dir,
    )
    if action_to_item_count:
        print(f"Done: {action_to_item_count} Action-to-Item summaries processed.")

    # Step 5: Produce the root-level summary across all Items
    all_items_row_count = make_all_items_summary(
        root_dir,
        summary_output_name,
        all_items_output_name,
    )
    if all_items_row_count == 0:
        print("No Item summary files found for all-items summary.")
        return

    print(
        f"Saved {root_dir / all_items_output_name} "
        f"({all_items_row_count} directories)"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Concat PredictTruelabel.csv files under "
            "<Ex>/<Item>/Stratified_SeedXXX/FoldX in Fold number order."
        )
    )
    parser.add_argument(
        "root_dirs",
        nargs="*",
        help=(
            "Parent Ex... directories. If omitted, one path is read from "
            "standard input."
        ),
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help=(
            "Treat each specified path as a parent directory and process all "
            "direct child directories that contain Item/Stratified_SeedXXX."
        ),
    )
    parser.add_argument(
        "--csv-name",
        default="PredictTruelabel.csv",
        help="CSV file name under each Fold directory.",
    )
    parser.add_argument(
        "--output-name",
        default="PredictTruelabel_all_folds.csv",
        help="Output file name created directly under each Stratified_SeedXXX directory.",
    )
    parser.add_argument(
        "--summary-output-name",
        default="CrossSubjectVariables.csv",
        help=(
            "Output file name created directly under each directory that "
            "contains Stratified_SeedXXX."
        ),
    )
    parser.add_argument(
        "--all-actions-output-name",
        default="CrossSubjectVariables_all_actions.csv",
        help=(
            "Output file name created directly under each Item directory when "
            "Action subdirectories contain summary files."
        ),
    )
    parser.add_argument(
        "--action-to-item-output-dir",
        default="CrossSubjectVariables_action_to_item",
        help=(
            "Output directory created under the parent Ex directory for "
            "<Action>_to_<Item> summaries grouped by Item name."
        ),
    )
    parser.add_argument(
        "--all-items-output-name",
        default="CrossSubjectVariables_all_items.csv",
        help="Output file name created directly under the parent Ex directory.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the aggregation pipeline for each experiment directory."""
    parser = build_arg_parser()
    args = parser.parse_args()

    root_dir_inputs = args.root_dirs or [input("Input directory path: ")]
    root_dirs = resolve_root_dirs(root_dir_inputs, args.scan)
    if not root_dirs:
        print("No experiment directories found.")
        return

    for root_dir in root_dirs:
        process_root_dir(
            root_dir,
            args.csv_name,
            args.output_name,
            args.summary_output_name,
            args.all_actions_output_name,
            args.action_to_item_output_dir,
            args.all_items_output_name,
        )


if __name__ == "__main__":
    main()
