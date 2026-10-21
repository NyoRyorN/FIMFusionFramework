"""
temporal

time-series data temporal /analyze.
evaluate and purpose.

Key features:
- temporal (>0.035seconds)
- value(NaN)
- generate
- visualizecreate
- output
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import sys
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt


def TimeShortageCheck(tasks, dirs, target_dir, distribution_dir):
    """temporalprocessing

    time-series data temporal and value,
    /visualize.

    Args:
        tasks (list): list
        dirs (list): subjectdirectorylist(path)
        target_dir (str): parent directorypath
        distribution_dir (str): result savingdirectorypath

    Processing details:
    1. temporal (>0.035seconds)
    2. valueframe
    3. aggregate
    4. visualizegenerate
    5. CSVoutput
    """

    # output directory set
    dst_dir = distribution_dir + "TimeShortage/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # processing directory
    if "Complemented" in target_dir:
        # complementafter case
        dst_dir = dst_dir + "After/"
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
    elif "Original" in target_dir:
        # case
        dst_dir = dst_dir + "Before/"
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
    else:
        print("The target directory is not correct.")
        sys.exit()

    # data frame and initialize
    sum_df = pd.DataFrame([i for i in range(315)], columns=["Index"])
    plt.figure(figsize=(20, 10))

    # each temporalrun
    for task in tasks:
        # recordlistinitialize
        shortage_dirs = []     # temporaldirectory
        shortage_indexs = []   # temporalframe
        shortage_times = []    # temporalvalue
        Nan_dirs = []         # valuedirectory
        Nan_indexs = []       # valueframe

        # each subjectdirectory processing
        for dir in dirs:
            # task file search
            temp_file = glob.glob(target_dir + dir + "/*" +
                                  task+".csv", recursive=True)
            for f in temp_file:
                # CSV load
                temp_dfs = pd.read_csv(f, index_col=None, header=0)
                # BodyIDcolumn (case)
                if "BodyID" in temp_dfs.columns:
                    temp_dfs = temp_dfs.drop(columns="BodyID", axis=1)

                # temporal calculate(frame)
                temp_times = temp_dfs["Time"].rolling(
                    window=2).agg(lambda x: x.iloc[1] - x.iloc[0])
                # points 3
                temp_times.map(lambda x: float(Decimal(str(x)).quantize(
                    Decimal('0.001'), rounding=ROUND_HALF_UP)))
                # value (row)
                isNanDf = temp_dfs.isnull().any(axis=1)

                # temporal /record(>0.035seconds)
                if len(temp_times[temp_times > 0.035].index) > 0:
                    for i in temp_times[temp_times > 0.035].index:
                        shortage_dirs.append(dir)
                        shortage_indexs.append(i)
                        shortage_times.append(temp_times[i])

                # value /record
                if isNanDf.any():
                    for i in isNanDf[isNanDf].index:
                        Nan_dirs.append(dir)
                        Nan_indexs.append(i)

        # temporal /outputprocessing
        shortage_df = pd.DataFrame(
            {"Directory": shortage_dirs, "Index": shortage_indexs, "Time": shortage_times})
        # temporal save
        shortage_df.to_csv(dst_dir + task + "_times.csv",
                           header=True, index=False)
        # directorytemporal
        shortage_df.value_counts("Directory").to_csv(
            dst_dir + task + "_CountShortageTimes.csv", header=True)
        # temporal(>0.11seconds) extract
        shortage_df[shortage_df["Time"] > 0.11].to_csv(
            dst_dir + task + "_CountLongShortageTimes.csv", header=True, index=False)

        # value /outputprocessing
        Nan_df = pd.DataFrame({"Directory": Nan_dirs, "Index": Nan_indexs})
        # value save
        Nan_df.to_csv(dst_dir + task + "_NanTimes.csv",
                      header=True, index=False)
        # directoryvalue
        Nan_df.value_counts("Directory").to_csv(
            dst_dir + task + "_CountNanTimes.csv", header=True)

        # visualize
        counts = shortage_df.value_counts("Index").sort_index()
        counts.rename("Count_"+task, inplace=True)
        sum_df = pd.merge(sum_df, counts, on="Index", how="outer")
        sum_df = sum_df.fillna(0)
        # bar chart generate
        plt.bar(sum_df["Index"], sum_df["Count_"+task],
                bottom=sum_df.iloc[:, 1:tasks.index(task)+1].sum(axis=1), label=task)
        print(task + " time-quality check completed.")

    # visualize save
    plt.legend()
    plt.title("Distribution of Storage Times")
    plt.savefig(dst_dir + "TimeShortageDist.png")

    # allintegrate generate
    all_shortage_df = pd.DataFrame(
        columns=["Directory", "Task", "Index", "Time"])
    all_nan_df = pd.DataFrame(columns=["Directory", "Task", "Index"])

    # each integrateprocessing
    for task in tasks:
        # temporal load/integrate
        temp_shortage_df = pd.read_csv(dst_dir + task + "_times.csv")
        temp_nan_df = pd.read_csv(dst_dir + task + "_NanTimes.csv")
        # task name append
        temp_shortage_df["Task"] = task
        temp_nan_df["Task"] = task
        # integratedata frame append
        all_shortage_df = pd.concat([all_shortage_df, temp_shortage_df])
        all_nan_df = pd.concat([all_nan_df, temp_nan_df])

    # integratesave results
    all_shortage_df.to_csv(dst_dir + "AllShortageTimes.csv",
                           header=True, index=False)
    all_nan_df.to_csv(dst_dir + "AllNanTimes.csv", header=True, index=False)
    print("------------ Time-quality check completed. ------------")
