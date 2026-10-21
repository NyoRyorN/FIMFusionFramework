"""
missing_motion_output.py - per-FIM-score missing motion distribution output

Aggregates the (subject, task) classifications collected while running
makeConciseData into per-FIM-item distribution tables: for each FIM item,
subjects with a valid score (1-7) are grouped by that score and the motion
slots are counted as Present / MissingSkeletonQuality (skeleton-estimation
quality rejections evidenced by Unused/) / MissingNoVideo
(missing or unprocessable recordings). The output style follows FimDistOutput
(plain CSVs written into data/Distributions/).

Usage:
    Called from makeConciseData.MakeConciseData(); not run directly.

Config: none
Env: none
"""

import os

import numpy as np
import pandas as pd

from src.utils.names import FIMitems, Tasks

# per-motion classification labels recorded by makeConciseData
STATUS_COLUMNS = ["Present", "MissingSkeletonQuality", "MissingNoVideo"]


def MissingMotionOutput(dst_dir: str, motion_status: list, fim_scores: dict,
                        subject_dir_root: str = None):
    """Write the missing-motion distribution CSVs.

    Processing flow:
        1. Dump the raw per-motion classifications (MotionStatus.csv) and,
           when subject_dir_root is given, a per-subject MissingMotion.csv
           inside each subject folder.
        2. Build the valid (Directory, Item, Score) table from the FIM
           scores collected during preprocessing.
        3. Count motion slots per (Item, Task, Score) and per status
           (MissingMotionCounts.csv).
        4. Sum the counts over tasks per (Item, Score)
           (MissingMotionSummary.csv).

    Args:
        dst_dir: output directory (created if missing)
        motion_status: (Directory, Task, Status, Reason) tuples, one per
            (subject, task) slot of subjects that have FIM data
        fim_scores: Directory -> {Item: raw score} from the converted FIM.csv
        subject_dir_root: ConciseData root; when given, each subject's motion
            status is also written to <root>/<Directory>/MissingMotion.csv
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    items = FIMitems()[:-1]  # "Sum" is not a 1-7 score, never aggregated
    tasks = Tasks()
    scores = list(range(1, 8))

    status_df = pd.DataFrame(
        motion_status, columns=["Directory", "Task", "Status", "Reason"])
    status_df.to_csv(os.path.join(dst_dir, "MotionStatus.csv"), index=False)

    if subject_dir_root is not None:
        _WritePerSubjectStatus(status_df, subject_dir_root, tasks)

    score_df = _ValidScoreTable(fim_scores, items)

    # one row per subject x item x task slot, labeled with its status
    merged = score_df.merge(status_df, on="Directory")
    counts = merged.pivot_table(index=["Item", "Task", "Score"],
                                columns="Status", aggfunc="size",
                                fill_value=0)

    # materialize every (item, task, score) combination and status column so
    # zero-count rows are visible in the output
    full_index = pd.MultiIndex.from_product([items, tasks, scores],
                                            names=["Item", "Task", "Score"])
    counts = counts.reindex(index=full_index, columns=STATUS_COLUMNS,
                            fill_value=0).astype(int).reset_index()
    counts.insert(3, "Subjects", counts[STATUS_COLUMNS].sum(axis=1))
    counts.to_csv(os.path.join(dst_dir, "MissingMotionCounts.csv"),
                  index=False)

    # task-summed view; Subjects stays the per-score subject count, so the
    # three status columns add up to Subjects x number of tasks
    summary = counts.groupby(["Item", "Score"], sort=False, as_index=False)[
        STATUS_COLUMNS].sum()
    subjects = score_df.groupby(["Item", "Score"]).size()
    summary.insert(2, "Subjects", [
        int(subjects.get((item, score), 0))
        for item, score in zip(summary["Item"], summary["Score"])])
    summary.to_csv(os.path.join(dst_dir, "MissingMotionSummary.csv"),
                   index=False)
    print("Saved the missing motion distribution.")


def _WritePerSubjectStatus(status_df: pd.DataFrame, subject_dir_root: str,
                           tasks: list):
    """Write each subject's motion status into its own ConciseData folder.

    One MissingMotion.csv per subject (Task, Status, Reason ordered by the
    canonical task list) so the missing-motion breakdown lives next to that
    subject's data; skipped when the subject folder does not exist.
    """
    task_order = {task: i for i, task in enumerate(tasks)}
    for directory, sub_df in status_df.groupby("Directory"):
        subject_dir = os.path.join(subject_dir_root, directory)
        if not os.path.isdir(subject_dir):
            continue
        sub_df = sub_df.drop(columns="Directory").copy()
        sub_df = sub_df.sort_values(
            "Task", key=lambda col: col.map(task_order))
        sub_df.to_csv(os.path.join(subject_dir, "MissingMotion.csv"),
                      index=False)


def _ValidScoreTable(fim_scores: dict, items: list) -> pd.DataFrame:
    """Flatten fim_scores into valid (Directory, Item, Score) rows.

    A score is valid when it is 1-7; NaN and scores <= 0 mean the item was
    not assessed (same rule as makeSomeTasksDataset) and the subject is
    excluded for that item only. Other out-of-range values are reported and
    excluded as well.
    """
    records = []
    for directory, scores in fim_scores.items():
        for item in items:
            score = scores.get(item, np.nan)
            if score is None or np.isnan(score) or score <= 0:
                continue
            if int(score) not in range(1, 8):
                print(directory + ": " + item + " score " + str(score) +
                      " is outside 1-7, excluded.")
                continue
            records.append((directory, item, int(score)))
    return pd.DataFrame(records, columns=["Directory", "Item", "Score"])
