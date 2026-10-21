"""
make_concise_data.py - direct preprocessing from OriginalData to ConciseData

Builds data/ConciseData/ straight from data/OriginalData/ without writing the
former intermediate directories (ComplementedData, FilteredData) to disk.
Per subject and task:
    1. Time complementation: detect the sampling pattern, merge the
       measurements onto the ideal time grid and fill gaps by linear
       interpolation.
    2. Filtering: rolling median + rolling mean smoothing, then velocity
       computation.
    3. Feature output: positions/velocities of the SimpleJoints subset and
       angle/bone-length recomputed on the "simple_ntu-rgb+d" layout.
FIM.csv is converted (Shift-JIS raw sheet -> Item/Score table) per subject,
and QC reports (time-shortage check, FIM distribution, missing-motion
distribution) go to data/Distributions/.

Every (subject, task) slot is classified as Present / MissingSkeletonQuality /
MissingNoVideo and aggregated per FIM item and score by MissingMotionOutput.
Skeleton-quality evidence comes only from raw CSVs already placed in Unused/;
this script never moves raw CSVs to Unused/.

Only the non-centered ConciseData is produced; centering is applied later at
dataset-build time (norm_centering_mode "First" in dfToNumpy).

For accepted tasks the output is bit-identical to the former three-stage
pipeline (timeComplement -> simplifyFilterCalc -> processSimpleSklData). That
pipeline passed data between stages through CSV files, and pandas' default
read_csv float parser is not perfectly round-trip accurate, so some low-order
digits of the former output are artifacts of those CSV reads. To reproduce
them exactly without touching the disk, the position/velocity DataFrames are
round-tripped through an in-memory CSV (_CsvRoundTrip) at the same points the
old pipeline wrote and re-read files, and the full-skeleton filtering is kept
intact (joints are reduced only when writing, matching the old stage order).

Usage:
    $ python src/data/preprocessing/make_concise_data.py

Config: none (directory layout is fixed under data/)
Env: none
"""

import io
import os
import shutil
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
from tqdm import tqdm

from src.utils.names import Tasks, Joints, SimpleJoints, FIMitems
from src.data.preprocessing.utils import (
    CalcVelocity,
    CalcAngleLength,
    plotFIMDist,
    TimeShortageCheck,
    FimDistOutput,
    MissingMotionOutput,
)

# rolling window length shared by the median and mean filters
LEN_FILTER = 15

# base sampling patterns at 30 frame/s; observed timestamp triples repeat
# one of these within every 0.1 s block
TIME_PATTERNS = [[0.000, 0.033, 0.066],
                 [0.000, 0.033, 0.067],
                 [0.000, 0.034, 0.067]]
# degraded patterns at 15 frame/s; index i maps back to TIME_PATTERNS[i]
SHORTAGE_TIME_PATTERNS = [[0.000, 0.066, 0.133],
                          [0.000, 0.067, 0.133],
                          [0.000, 0.067, 0.134]]


def _Quantize3(x: float) -> float:
    # str(x) first so the rounding sees the decimal repr, matching how the
    # measured CSV values were rounded originally
    return float(Decimal(str(x)).quantize(Decimal('0.001'),
                                          rounding=ROUND_HALF_UP))


def _CsvRoundTrip(df: pd.DataFrame) -> pd.DataFrame:
    # replicate the lossy CSV write+read the old pipeline performed between
    # stages: to_csv writes the shortest round-tripping repr, and the default
    # read_csv parser re-reads it into occasionally 1-ULP-different floats.
    # Kept in memory (StringIO) so no intermediate file touches the disk.
    return pd.read_csv(io.StringIO(df.to_csv(index=False)),
                       index_col=None, header=0)


def BuildTimeGrids() -> dict:
    """Precompute the time-grid tables used by ComplementTime.

    The candidate triples and the full ideal grids depend only on the fixed
    patterns above, so they are built once with the same Decimal rounding
    the per-file search used, instead of being recomputed for every file.

    Returns:
        dict with keys:
            "candidates": [pattern][offset] -> frozenset of 3 timestamps
                          (offsets 0.0, 0.1, ..., 10.9 s)
            "shortage_candidates": same for the 15 frame/s patterns
                                   (offsets 0.0, 0.2, ..., 10.8 s)
            "grids": [pattern] -> sorted ideal timestamps in (0.099, 10.100)
    """
    offsets = [_Quantize3(0.1 * i) for i in range(0, 110)]

    candidates = [[frozenset(_Quantize3(x + off) for x in items)
                   for off in offsets]
                  for items in TIME_PATTERNS]

    shortage_candidates = [[frozenset(_Quantize3(x + off) for x in items)
                            for off in offsets[::2]]
                           for items in SHORTAGE_TIME_PATTERNS]

    grids = []
    for items in TIME_PATTERNS:
        times = [_Quantize3(x + off) for off in offsets for x in items]
        grids.append(sorted(t for t in times if 0.099 < t < 10.100))

    return {"candidates": candidates,
            "shortage_candidates": shortage_candidates,
            "grids": grids}


def ComplementTime(df: pd.DataFrame, time_grids: dict) -> tuple:
    """Resample one task recording onto the ideal time grid.

    Processing flow:
        1. Detect which sampling pattern the timestamps follow
           (30 frame/s first, then the 15 frame/s fallback).
        2. Merge the measured rows onto the ideal grid.
        3. Fill the remaining gaps by linear interpolation (interior only).
        4. Round to 3 decimals, trim leading/trailing rows that are still
           NaN, and shift Time back by the 0.1 s grid margin.

    Args:
        df: rounded task DataFrame (Frame/Time + joint coordinate columns)
        time_grids: precomputed tables from BuildTimeGrids()

    Returns:
        tuple: (complemented DataFrame with Frame as a column, drop reason).
        The DataFrame is None when no sampling pattern matches; drop reason is
        then "no_pattern", otherwise None.
    """
    intervals = set(df["Time"])

    # pattern detection: keep scanning all offsets like the original search
    # did, so the last matching offset decides the pattern
    pattern_index = None
    for offset_index in range(len(time_grids["candidates"][0])):
        for p, candidate in enumerate(time_grids["candidates"]):
            if candidate[offset_index] <= intervals:
                pattern_index = p
                break

    # 15 frame/s fallback: a match still selects the 30 frame/s grid of the
    # same index, because the ideal output rate is unchanged
    if pattern_index is None:
        for offset_index in range(len(time_grids["shortage_candidates"][0])):
            for p, candidate in enumerate(time_grids["shortage_candidates"]):
                if candidate[offset_index] <= intervals:
                    pattern_index = p
                    break

    if pattern_index is None:
        return None, "no_pattern"

    # merge onto the ideal grid; unmatched grid rows become NaN
    df_times = pd.DataFrame(time_grids["grids"][pattern_index],
                            columns=["Time"])
    new_df = df_times.merge(df, on="Time", how="left")
    new_df = new_df.reindex(columns=df.columns)
    new_df["Frame"] = new_df.index
    new_df.set_index("Time", inplace=True)

    # linear interpolation by missing-value complementation (interior only,
    # so the leading/trailing gaps stay NaN and are trimmed below)
    new_df.interpolate(method="linear",
                       inplace=True,
                       limit_direction="both",
                       limit_area="inside")
    new_df.reset_index(inplace=True)
    new_df.set_index("Frame", inplace=True)
    new_df = new_df.map(lambda x: float(
        Decimal(str(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)))

    # trim rows that are still NaN at the beginning and the end
    delete_index = []
    nan_counts = new_df.isnull().sum(axis=1)
    for i, v in nan_counts.items():
        if v > 0:
            delete_index.append(i)
        else:
            break
    nan_counts = nan_counts.sort_index(ascending=False)
    for i, v in nan_counts.items():
        if v > 0:
            delete_index.append(i)
        else:
            break
    new_df.drop(index=delete_index, inplace=True)

    # remove the leading 0.1 s grid margin so recordings start near 0 s
    new_df["Time"] = new_df["Time"].map(lambda x: float(
        Decimal(str(x - 0.1)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)))

    return new_df.reset_index(), None


def FilterPoints(df: pd.DataFrame, len_filter: int) -> tuple:
    """Smooth the full-skeleton positions and derive the velocity.

    The filtering runs on all joints (not just the reduced set) so the
    rolling results match the former pipeline bit-for-bit; the reduction to
    SimpleJoints happens only when the CSVs are written.

    Args:
        df: complemented time-series data (Frame/Time + joint columns)
        len_filter: rolling window length for both filters

    Returns:
        tuple: (filtered positions, velocity) DataFrames
    """
    if "Frame" in df.columns:
        df = df.drop(columns="Frame")

    df_p = df.copy()
    filter_index = df_p.columns.drop("Time")

    # median filter first to reject spikes, then moving average to smooth
    df_p[filter_index] = df_p[filter_index].rolling(window=len_filter).median()
    df_p = df_p.dropna()
    df_p[filter_index] = df_p[filter_index].rolling(window=len_filter).mean()
    df_p = df_p.dropna()

    df_v = CalcVelocity(df_p)

    return df_p, df_v


def ConvertFim(path: str, items: list) -> pd.DataFrame:
    """Convert a raw FIM sheet into the Item/Score table.

    Args:
        path: raw FIM CSV path (Shift-JIS encoded)
        items: FIM item names; the last entry receives the total score

    Returns:
        DataFrame indexed by Item with a single Score column.
    """
    fim_df = pd.read_csv(path, header=0, encoding="shift-jis")
    # the first two columns are sheet labels, only the score column remains
    fim_df.drop(columns=fim_df.columns[:2], inplace=True)
    fim_df.columns = ["Score"]
    fim_df["Item"] = items[:-1]
    fim_df = fim_df.set_index("Item")
    fim_df.loc[items[-1]] = fim_df.sum()
    return fim_df


def SaveConciseTask(df_p: pd.DataFrame, df_v: pd.DataFrame,
                    simple_joints: list, dst_dir: str):
    """Write the per-task ConciseData CSVs for the reduced skeleton.

    Processing flow:
        1. Reduce positions/velocities to SimpleJoints and replicate the
           former pipeline's CSV read-back (_CsvRoundTrip).
        2. Save per-axis position (p_*.csv) and velocity (v_*.csv) files
           with plain joint-name headers.
        3. Recompute angle and bone length from the round-tripped positions
           and save them per axis (a_*.csv / b_*.csv).

    Args:
        df_p: full-skeleton filtered positions (Time + joint columns)
        df_v: full-skeleton velocities (joint columns)
        simple_joints: reduced joint-name list
        dst_dir: output task directory (created if missing)
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    suffix_cols = [j + "_" + axis for j in simple_joints for axis in "XYZ"]
    # round-trip mirrors the old stage2->stage3 CSV hop: positions/velocities
    # are written full-precision, then re-read for stage3 output and the
    # angle/bone recomputation
    p_rt = _CsvRoundTrip(df_p[suffix_cols])
    v_rt = _CsvRoundTrip(df_v[suffix_cols])

    for axis in ["x", "y", "z"]:
        columns = [j + "_" + axis.upper() for j in simple_joints]
        p_rt[columns].to_csv(os.path.join(dst_dir, "p_" + axis + ".csv"),
                             header=simple_joints, index=False)
        v_rt[columns].to_csv(os.path.join(dst_dir, "v_" + axis + ".csv"),
                             header=simple_joints, index=False)

    df_a, df_l = CalcAngleLength(p_rt, simple_joints)

    # keep the axis-suffixed headers; downstream loaders read these files
    # positionally and never select by column name
    for axis in ["x", "y", "z"]:
        columns = [j + "_" + axis.upper() for j in simple_joints]
        df_a[columns].to_csv(os.path.join(dst_dir, "a_" + axis + ".csv"),
                             index=False)
        df_l[columns].to_csv(os.path.join(dst_dir, "b_" + axis + ".csv"),
                             index=False)


def MakeConciseData():
    """Build data/ConciseData/ directly from data/OriginalData/.

    Processing flow:
        1. Run the time-shortage QC check on the raw data.
        2. For each NEDO subject: convert FIM.csv (subjects without FIM
           keep an empty output directory and are skipped).
        3. For each task: complement time, filter the full skeleton, and
           save the reduced-joint features.
            a. Every (subject, task) slot is classified as Present /
               MissingSkeletonQuality / MissingNoVideo.
        4. Output QC reports (FIM presence and unprocessable files), the FIM
           score distribution and the per-item missing-motion distribution.

    Raises:
        SystemExit: when data/OriginalData/ does not exist.
    """
    tasks = Tasks()
    joints = Joints()
    simple_joints = SimpleJoints()
    items = FIMitems()

    root_dir = "data/"
    original_dir = root_dir + "OriginalData/"
    src_dir = root_dir + "ConciseData/"
    dst_dir = root_dir + "Distributions/"

    if not os.path.exists(original_dir):
        print(original_dir + " does not exist.")
        exit()
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # columns dropped right after loading, matching the old stage1:
    # device metadata and per-joint status flags
    drop_columns = ["BodyID"] + [j + "_Status" for j in joints]

    time_grids = BuildTimeGrids()

    dirs = sorted([d for d in os.listdir(original_dir) if os.path.isdir(
        os.path.join(original_dir, d)) and "NEDO" in d])

    TimeShortageCheck(tasks, dirs, original_dir, dst_dir)

    # QC records: FIM presence per subject and unprocessable task files
    fim_dirs = []
    not_fim_dirs = []
    many_shortage_files = []
    motion_status = []   # (Directory, Task, Status, Reason) per motion slot
    fim_scores = {}      # Directory -> {Item: raw score}

    for dir in tqdm(dirs, leave=False):
        each_dir_org = os.path.join(original_dir, dir)
        each_dir_sv = os.path.join(src_dir, dir)
        if not os.path.exists(each_dir_sv):
            os.makedirs(each_dir_sv)

        # FIM conversion; task data without a FIM label is never used, so
        # such subjects are skipped after being recorded for the QC report
        org_files = [f for f in os.listdir(each_dir_org)
                     if os.path.isfile(os.path.join(each_dir_org, f))]
        fim_files = [f for f in org_files if "FIM" in f.split("_")[-1]]
        if len(fim_files) == 0:
            not_fim_dirs.append(dir)
            print(dir + " has no FIM data.")
            continue
        fim_dirs.append(dir)
        fim_df = ConvertFim(os.path.join(each_dir_org, fim_files[0]), items)
        fim_df.to_csv(os.path.join(each_dir_sv, "FIM.csv"), index=True)
        fim_scores[dir] = fim_df["Score"].to_dict()

        for task in tasks:
            task_files = [f for f in org_files if task in f]
            if len(task_files) == 0:
                # absent raw recording: skeleton-quality only when the file was
                # already placed in Unused/, otherwise the video never existed
                _DropStaleTaskDir(os.path.join(each_dir_sv, task))
                unused_csv = os.path.join(
                    each_dir_org, "Unused", task + ".csv")
                if os.path.isfile(unused_csv):
                    motion_status.append(
                        (dir, task, "MissingSkeletonQuality", "Unused"))
                else:
                    motion_status.append((dir, task, "MissingNoVideo", ""))
                continue

            df = pd.read_csv(os.path.join(each_dir_org, task_files[0]),
                             header=0)
            df = df.drop(columns=drop_columns)
            df = df.map(lambda x: float(Decimal(str(x)).quantize(
                Decimal('0.001'), rounding=ROUND_HALF_UP)))
            df["Frame"] = df["Frame"].astype(int)

            df, drop_reason = ComplementTime(df, time_grids)
            if df is None:
                _DropStaleTaskDir(os.path.join(each_dir_sv, task))
                many_shortage_files.append(
                    os.path.join(each_dir_org, task + ".csv"))
                motion_status.append(
                    (dir, task, "MissingNoVideo", drop_reason))
                continue

            df_p, df_v = FilterPoints(df, LEN_FILTER)
            SaveConciseTask(df_p, df_v, simple_joints,
                            os.path.join(each_dir_sv, task))
            motion_status.append((dir, task, "Present", ""))

    # QC reports and FIM score distribution; the aggregate FIM tables go to
    # Distributions/ (the old pipeline placed them under FilteredData/, which
    # no longer exists) so ConciseData/ holds only the per-subject data
    FimDistOutput(dst_dir, fim_dirs, not_fim_dirs, root_dir=original_dir)
    with open(os.path.join(dst_dir, "ManyShortageTimeFiles.txt"), "w") as f:
        for file in many_shortage_files:
            f.write(file + "\n")
    MissingMotionOutput(dst_dir, motion_status, fim_scores,
                        subject_dir_root=src_dir)
    plotFIMDist(src_dir, dst_dir)
    print("Plotted the FIM data distribution.")


def _DropStaleTaskDir(task_dir: str):
    """Remove a task's output left by an earlier run so ConciseData holds
    only the tasks accepted in this run."""
    if os.path.isdir(task_dir):
        shutil.rmtree(task_dir)


def main():
    """Run the OriginalData -> ConciseData preprocessing.

    Processing flow:
        1. Run MakeConciseData.
    """
    try:
        print("=== ConciseData preprocessing started ===")
        MakeConciseData()
        print("=== All processing completed ===")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Run processing when the script is executed directly
    main()
