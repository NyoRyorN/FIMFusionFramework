"""
missing_motion_class_view.py - display MissingMotionCounts.csv aggregated by FIM item and score class

Reads data/Distributions/MissingMotionCounts.csv (produced by
src/data/preprocessing/make_concise_data.py; see docs/missing_motion.md)
and, for a single FIM item, merges the 1-7 score rows into custom classes
(e.g. "1-6,7" for a 6-vs-1 binary split used by class-wise accuracy plots) to
show, per task, the Subjects / Present / MissingSkeletonQuality /
MissingNoVideo counts and the missing rate of each class. It also joins
MotionStatus.csv (per subject x task status) with all_fim.csv (per subject
FIM scores) to append an "AnyMissing" pseudo-task row per class that counts
subjects with at least one missing motion. The whole output (per-task rows
and the AnyMissing row) is restricted to a selectable task subset
(--any-tasks, default all 11 tasks; e.g. "1-10" to exclude 11-Walking when
walking is out of scope).

Usage:
    $ python reports/check_fim_dist/missing_motion_class_view.py Eating 1-6,7
    $ python reports/check_fim_dist/missing_motion_class_view.py 0 1-6,7
    $ python reports/check_fim_dist/missing_motion_class_view.py Bathing 1-3,4-5,6-7 -o out.csv
    $ python reports/check_fim_dist/missing_motion_class_view.py Eating 1-6,7 --any-tasks 1-10

Config: none
Env: none
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils.names import FIMitems, Tasks

STATUS_COLUMNS = ["Present", "MissingSkeletonQuality", "MissingNoVideo"]
DEFAULT_CSV_PATH = "data/Distributions/MissingMotionCounts.csv"
DEFAULT_STATUS_CSV_PATH = "data/Distributions/MotionStatus.csv"
DEFAULT_FIM_CSV_PATH = "data/Distributions/all_fim.csv"
ANY_MISSING_TASK = "AnyMissing"  # pseudo-task: subjects with >=1 missing motion


def parse_class_spec(spec: str) -> list[tuple[str, list[int]]]:
    """Parse a class-grouping spec such as "1-6,7" into (label, scores) groups.

    Groups are comma-separated; within a group, "-" denotes an inclusive
    range and "+" joins multiple ranges/values (e.g. "1-3+5,4,6-7"). The
    group string itself (e.g. "1-6") is kept as the display label.

    Args:
        spec: Class-grouping spec string.

    Returns:
        List of (label, sorted scores) tuples, one per class, in spec order.

    Raises:
        ValueError: A score is outside 1-7, a range is malformed, or a score
            is assigned to more than one class.
    """
    groups = []
    seen_scores = set()
    for group_str in spec.split(","):
        group_str = group_str.strip()
        if not group_str:
            raise ValueError(f"Empty class group in spec: {spec!r}")

        scores = []
        for part in group_str.split("+"):
            part = part.strip()
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start, end = int(start_str), int(end_str)
                if start > end:
                    raise ValueError(
                        f"Invalid range '{part}' in class spec {spec!r}: start > end")
                scores.extend(range(start, end + 1))
            else:
                scores.append(int(part))

        for score in scores:
            if score not in range(1, 8):
                raise ValueError(
                    f"Score {score} in class spec {spec!r} is outside 1-7")
            if score in seen_scores:
                raise ValueError(
                    f"Score {score} assigned to multiple classes in spec {spec!r}")
            seen_scores.add(score)

        groups.append((group_str, sorted(scores)))

    return groups


def parse_task_spec(spec: str) -> list[str]:
    """Parse a task spec such as "1-10" into canonical task names.

    Groups are comma-separated; "-" denotes an inclusive range of 1-based
    task numbers matching the Tasks() order (which equals the numeric
    prefix of each task name, e.g. 11 = "11-Walking").

    Args:
        spec: Task spec string, e.g. "1-10" or "1-9,11".

    Returns:
        Selected task names in canonical Tasks() order, without duplicates.

    Raises:
        ValueError: A task number is out of range, a range is malformed, or
            the spec selects no task.
    """
    tasks = Tasks()
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty task group in spec: {spec!r}")
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}' in task spec {spec!r}: start > end")
            numbers = range(start, end + 1)
        else:
            numbers = [int(part)]
        for number in numbers:
            if number not in range(1, len(tasks) + 1):
                raise ValueError(
                    f"Task number {number} in task spec {spec!r} is outside "
                    f"1-{len(tasks)}")
            selected.add(number)
    return [tasks[number - 1] for number in sorted(selected)]


def resolve_item(item_arg: str, valid_items: list[str]) -> str:
    """Resolve the item CLI argument to a canonical FIM item name.

    Args:
        item_arg: Either an item name or a 0-based index into valid_items
            (matching the FIMitems() order shown in --help).
        valid_items: Canonical FIM item names (FIMitems() without "Sum").

    Returns:
        The canonical item name.

    Raises:
        ValueError: item_arg is neither a known name nor a valid index.
    """
    if item_arg.isdigit():
        index = int(item_arg)
        if index not in range(len(valid_items)):
            raise ValueError(
                f"Item index {index} is out of range 0-{len(valid_items) - 1}")
        return valid_items[index]

    if item_arg not in valid_items:
        raise ValueError(
            f"Unknown Item {item_arg!r}. Valid items: "
            f"{', '.join(f'{i}={name}' for i, name in enumerate(valid_items))}")
    return item_arg


def load_item_table(csv_path: Path, item: str) -> pd.DataFrame:
    """Load MissingMotionCounts.csv and filter it to a single FIM item.

    Args:
        csv_path: Path to MissingMotionCounts.csv.
        item: FIM item name (must be one of FIMitems(), excluding "Sum").

    Returns:
        Rows of csv_path whose Item column matches item.

    Raises:
        ValueError: item is not a known FIM item, or no rows match it.
    """
    valid_items = FIMitems()[:-1]  # "Sum" is not a 1-7 score, never present here
    if item not in valid_items:
        raise ValueError(
            f"Unknown Item {item!r}. Valid items: {', '.join(valid_items)}")

    df = pd.read_csv(csv_path)
    item_df = df[df["Item"] == item].copy()
    if item_df.empty:
        raise ValueError(f"No rows found for Item {item!r} in {csv_path}")
    return item_df


def aggregate_by_class(
        item_df: pd.DataFrame,
        class_groups: list[tuple[str, list[int]]],
        target_tasks: list[str]) -> pd.DataFrame:
    """Merge score rows into classes and sum status counts per task.

    Only the tasks in target_tasks are emitted, so tasks outside the
    experiment's scope (e.g. 11-Walking) can be dropped from the whole
    output.

    Args:
        item_df: Rows of MissingMotionCounts.csv for one Item (see
            load_item_table).
        class_groups: Output of parse_class_spec.
        target_tasks: Task names to include, in output order.

    Returns:
        Long-format DataFrame with columns Task, Class, Subjects, Present,
        MissingSkeletonQuality, MissingNoVideo, MissingRate(%).
    """
    records = []
    for task in target_tasks:
        task_df = item_df[item_df["Task"] == task]
        for label, scores in class_groups:
            matched = task_df[task_df["Score"].isin(scores)]
            record = {"Task": task, "Class": label,
                      "Subjects": int(matched["Subjects"].sum())}
            for col in STATUS_COLUMNS:
                record[col] = int(matched[col].sum())
            records.append(record)
    long_df = pd.DataFrame(records)

    # each row holds one motion slot per subject, so the status total is the
    # missing-rate denominator (equals Subjects for per-task rows)
    slot_count = long_df[STATUS_COLUMNS].sum(axis=1)
    long_df["MissingRate(%)"] = (
        (long_df["MissingSkeletonQuality"] + long_df["MissingNoVideo"])
        / slot_count.replace(0, pd.NA) * 100
    ).round(1).fillna(0.0)

    return long_df


def load_subject_scores(fim_csv: Path, item: str) -> pd.Series:
    """Load all_fim.csv and return the valid per-subject scores of one item.

    Args:
        fim_csv: Path to all_fim.csv (subject ID index, one column per item).
        item: FIM item name.

    Returns:
        Series indexed by subject ID holding that item's score, restricted
        to valid scores (1-7; all_fim.csv stores unassessed items as 0).

    Raises:
        ValueError: item is not a column of fim_csv.
    """
    fim_df = pd.read_csv(fim_csv, index_col=0)
    if item not in fim_df.columns:
        raise ValueError(f"Item {item!r} not found in {fim_csv}")
    scores = fim_df[item]
    return scores[scores.between(1, 7)]


def aggregate_any_missing(
        status_df: pd.DataFrame,
        subject_scores: pd.Series,
        class_groups: list[tuple[str, list[int]]],
        target_tasks: list[str]) -> pd.DataFrame:
    """Count subjects with at least one missing motion, per class.

    Only the motions in target_tasks are inspected (e.g. excluding
    11-Walking when walking is out of the experiment's scope). Each subject
    is counted once: under MissingSkeletonQuality when any target motion is
    a skeleton-quality rejection, else under MissingNoVideo when any target
    recording is missing, else under Present (all target motions present).
    Skeleton-quality takes precedence so the three status columns still add
    up to Subjects, matching the row layout of aggregate_by_class.

    Args:
        status_df: MotionStatus.csv rows (Directory, Task, Status, Reason).
        subject_scores: Output of load_subject_scores.
        class_groups: Output of parse_class_spec.
        target_tasks: Task names whose missing status is considered.

    Returns:
        One "AnyMissing" pseudo-task row per class with columns Task, Class,
        Subjects, Present, MissingSkeletonQuality, MissingNoVideo.
    """
    target_df = status_df[status_df["Task"].isin(target_tasks)]
    status_by_subject = target_df.groupby("Directory")["Status"]
    has_skel = status_by_subject.apply(
        lambda s: bool((s == "MissingSkeletonQuality").any()))
    has_novideo = status_by_subject.apply(
        lambda s: bool((s == "MissingNoVideo").any()))

    records = []
    for label, scores in class_groups:
        subjects = subject_scores[subject_scores.isin(scores)].index
        unknown = subjects.difference(has_skel.index)
        if len(unknown) > 0:
            print(f"Warning: {len(unknown)} subjects of class {label!r} "
                  "have no MotionStatus.csv rows; they are excluded from "
                  "the AnyMissing row.")
            subjects = subjects.intersection(has_skel.index)
        skel = int(has_skel.loc[subjects].sum())
        novideo = int((~has_skel.loc[subjects]
                       & has_novideo.loc[subjects]).sum())
        records.append({
            "Task": ANY_MISSING_TASK, "Class": label,
            "Subjects": len(subjects),
            "Present": len(subjects) - skel - novideo,
            "MissingSkeletonQuality": skel,
            "MissingNoVideo": novideo,
        })
    return pd.DataFrame(records)


def pivot_subjects(long_df: pd.DataFrame, class_labels: list[str]) -> pd.DataFrame:
    """Reshape the long-format table into a Task x Class Subjects overview.

    Args:
        long_df: Output of aggregate_by_class.
        class_labels: Class labels in the desired column order.

    Returns:
        Wide DataFrame indexed by Task (canonical order, "AnyMissing"
        last), one column per class, holding Subjects counts.
    """
    wide = long_df.pivot(index="Task", columns="Class", values="Subjects")
    task_order = Tasks() + [ANY_MISSING_TASK]
    return wide.reindex(index=[t for t in task_order if t in wide.index],
                        columns=class_labels)


def pivot_missing(long_df: pd.DataFrame, class_labels: list[str]) -> pd.DataFrame:
    """Reshape the long-format table into a Task x Class missing-count overview.

    Args:
        long_df: Output of aggregate_by_class (optionally with AnyMissing
            rows appended).
        class_labels: Class labels in the desired column order.

    Returns:
        Wide DataFrame indexed by Task (canonical order, "AnyMissing"
        last), one column per class, holding MissingSkeletonQuality +
        MissingNoVideo counts. Per-task rows count subjects missing that
        motion; the AnyMissing row counts subjects missing at least one of
        the target motions.
    """
    tmp = long_df.copy()
    tmp["Missing"] = tmp["MissingSkeletonQuality"] + tmp["MissingNoVideo"]
    wide = tmp.pivot(index="Task", columns="Class", values="Missing")
    task_order = Tasks() + [ANY_MISSING_TASK]
    return wide.reindex(index=[t for t in task_order if t in wide.index],
                        columns=class_labels)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Display MissingMotionCounts.csv for one FIM item, aggregated "
            "into custom score classes (e.g. '1-6,7')."
        )
    )
    valid_items = FIMitems()[:-1]  # "Sum" is not a 1-7 score, never present here
    parser.add_argument(
        "item",
        help=(
            "FIM item name, or its 0-based index. "
            + ", ".join(f"{i}={name}" for i, name in enumerate(valid_items))
        ),
    )
    parser.add_argument(
        "classes",
        help=(
            "Class-grouping spec. Comma-separates classes, '-' is an "
            "inclusive score range, '+' joins parts within a class, e.g. "
            "'1-6,7' or '1-3+5,4,6-7'."
        ),
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help=f"Path to MissingMotionCounts.csv (default: {DEFAULT_CSV_PATH}).",
    )
    parser.add_argument(
        "--status-csv",
        default=DEFAULT_STATUS_CSV_PATH,
        help=(
            "Path to MotionStatus.csv used for the AnyMissing row "
            f"(default: {DEFAULT_STATUS_CSV_PATH})."
        ),
    )
    parser.add_argument(
        "--fim-csv",
        default=DEFAULT_FIM_CSV_PATH,
        help=(
            "Path to all_fim.csv used for the AnyMissing row "
            f"(default: {DEFAULT_FIM_CSV_PATH})."
        ),
    )
    tasks = Tasks()
    parser.add_argument(
        "--any-tasks",
        default=f"1-{len(tasks)}",
        help=(
            "Tasks included in the output (per-task rows and the "
            "AnyMissing row), as comma-separated 1-based task numbers with "
            "'-' ranges (e.g. '1-10' to exclude 11-Walking). Default: all "
            f"tasks ('1-{len(tasks)}'). "
            + ", ".join(f"{i + 1}={name}" for i, name in enumerate(tasks))
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional path to also save the detail table as a CSV.",
    )
    return parser


def main() -> None:
    """Aggregate one FIM item's missing-motion counts by class and print them.

    Processing flow:
        1. Parse the item / class-spec / task-subset / path arguments.
        2. Aggregate MissingMotionCounts.csv into per-task rows per class,
           restricted to the --any-tasks subset.
        3. Join MotionStatus.csv with all_fim.csv to append an "AnyMissing"
           row per class (subjects with >=1 missing motion among the
           --any-tasks subset) and warn when its Subjects total disagrees
           with the counts CSV.
        4. Print the Subjects and missing-count pivots plus the detail
           table, optionally saving the detail table as a CSV.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    class_groups = parse_class_spec(args.classes)
    item = resolve_item(args.item, FIMitems()[:-1])
    any_tasks = parse_task_spec(args.any_tasks)
    item_df = load_item_table(Path(args.csv), item)
    long_df = aggregate_by_class(item_df, class_groups, any_tasks)
    class_labels = [label for label, _ in class_groups]

    # ------------------------------------------------------------------
    # Append the subject-level AnyMissing rows
    # ------------------------------------------------------------------
    status_df = pd.read_csv(args.status_csv)
    subject_scores = load_subject_scores(Path(args.fim_csv), item)
    any_df = aggregate_any_missing(
        status_df, subject_scores, class_groups, any_tasks)
    slot_count = any_df[STATUS_COLUMNS].sum(axis=1)
    any_df["MissingRate(%)"] = (
        (any_df["MissingSkeletonQuality"] + any_df["MissingNoVideo"])
        / slot_count.replace(0, pd.NA) * 100
    ).round(1).fillna(0.0)

    # subject-level totals must match the counts CSV; a mismatch means the
    # Distributions CSVs were generated from different pipeline runs
    # (Subjects is task-invariant, so any per-task row works as reference)
    for label, _ in class_groups:
        counts_subjects = int(long_df.loc[
            long_df["Class"] == label, "Subjects"].iloc[0])
        any_subjects = int(any_df.loc[
            any_df["Class"] == label, "Subjects"].iloc[0])
        if counts_subjects != any_subjects:
            print(f"Warning: class {label!r} has {counts_subjects} subjects "
                  f"in {args.csv} but {any_subjects} in {args.fim_csv}; "
                  "the Distributions CSVs may be out of sync.")
    long_df = pd.concat([long_df, any_df], ignore_index=True)

    print(f"Item: {item}")
    print("Classes: " + ", ".join(
        f"{label} (scores {scores})" for label, scores in class_groups))
    print(f"Target tasks ({len(any_tasks)}): " + ", ".join(any_tasks))

    print("\n=== Subjects by Task x Class ===")
    print(pivot_subjects(long_df, class_labels).to_string())

    print("\n=== Missing subjects by Task x Class ===")
    print("(per-task rows: subjects missing that motion; AnyMissing: "
          "subjects missing >=1 motion among the target tasks above)")
    print(pivot_missing(long_df, class_labels).to_string())

    print("\n=== Detail (Task x Class) ===")
    print(long_df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(output_path, index=False)
        print(f"\nSaved detail table to {output_path}")


if __name__ == "__main__":
    main()
