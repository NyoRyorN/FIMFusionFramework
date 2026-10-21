"""
FIM

FIM visualize/aggregate.
each subject FIM score integrateanalyze generate.

Key features:
- multipledirectory from FIMintegrate
- subjectIDintegrate
- aggregate
- CSVformat output
- analyze
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from src.utils.names import FIMitems

# FIM itemlist get
items = FIMitems()


def plotFIMDist(rootDir: str, dstDir: str):
    """FIM integrate/aggregateprocessing

    multipledirectory FIM integrate,
    analyze CSV file generate.

    Args:
        rootDir (str): FIMsearchdirectory
        dstDir (str): output directory

    Processing details:
    1. alldirectory from FIMfilesearch
    2. subjectIDintegrate
    3. allFIM integrateCSVgenerate
    4. each calculate
    5. aggregateCSVoutput
    """
    # target directory get(sorted)
    target_dirs = sorted(
        [f for f in glob.glob(rootDir + "*") if os.path.isdir(f)])

    # integratedata frame initialize
    df = pd.DataFrame()

    # eachdirectory FIMprocessing
    for target_dir in target_dirs:
        # subjectID extract(directory from)
        person = target_dir.split("/")[-1]
        # FIMfilepath build
        path = target_dir + "/FIM.csv"

        # FIMfile check/processing
        if os.path.exists(path):
            # FIM load; FIM.csv is "Item,Score" formatted, so use the Item
            # column as index to keep only the numeric Score values
            temp_df = pd.read_csv(path, header=0, index_col=0)
            # processing commented out(beforeprocessing)
            # temp_df.drop(columns = temp_df.columns[:2], inplace = True)
            # temp_df.columns=["Score"]
            # temp_df["Item"] = items[:-1]
            # temp_df = temp_df.set_index("Item")
            # temp_df.loc[items[-1]] = temp_df.sum()

            # subjectIDrow append
            temp_df.loc["ID"] = person
            # transposesubject row convert
            temp_df = temp_df.T
            temp_df = temp_df.set_index("ID")

            # merge data frames
            if df.empty:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])

    # value 0 convert
    df = df.fillna(0).apply(lambda x: x.astype(int))
    # integrateFIM save
    df.to_csv(os.path.join(dstDir, "all_fim.csv"))

    # eachFIM item aggregateprocessing
    for item in items:
        # integrate load
        df = pd.read_csv(os.path.join(dstDir, "all_fim.csv"),
                         header=0, index_col=0)
        # extract
        df = df[item]
        # value(sorted)
        df = df.value_counts().sort_index()
        # row append
        df.loc["Sum"] = df.sum()
        # aggregateCSV save
        df.to_csv(os.path.join(dstDir, "fim-" +
                  str(items.index(item)) + "-" + item + ".csv"))
