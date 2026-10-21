"""
FIMoutput

FIM(Functional Independence Measure) aggregate/analyze run.
evaluate statistical processing and visualizegenerate purpose.

Key features:
- FIM aggregate
- each aggregate
- FIMpresence/absencedirectory
- CSVformat result output
- generate
"""

import numpy as np
import pandas as pd
import os
import glob
import csv
from src.utils.names import FIMitems


def FimDistOutput(dst_dir, fim_dirs, not_fim_dirs, root_dir):
    """FIM outputprocessing

    FIM aggregate/analyze CSVformat output.
    each and statistics generate.

    Args:
        dst_dir (str): result savingdirectorypath
        fim_dirs (list): FIMdirectorylist
        not_fim_dirs (list): FIMdirectorylist
        root_dir (str): directorypath

    Processing details:
    1. allFIM aggregate
    2. each
    3. presence/absencedirectory
    4. filegenerate
    """

    # output directory createcheck
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # FIM item and set
    items = FIMitems()
    dfHeader = ["Directory"] + items

    # allFIM aggregateprocessing
    csv_path = dst_dir + 'all_fim.csv'
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(dfHeader)

        # eachFIMdirectory processing
        for dir in fim_dirs:
            # FIMfile search
            fim_file = glob.glob(root_dir + dir + "/*_FIM.csv", recursive=True)
            if len(fim_file) == 0:
                continue
            else:
                # FIM load/processing
                fim_dfs = []
                df = pd.read_csv(fim_file[0], header=0, encoding="shift-jis")
                fim_dfs.append(dir)

                # eachFIM item get
                for i in range(1, len(dfHeader)-1):
                    if df.iloc[i-1].DataValue in [1, 2, 3, 4, 5, 6, 7]:
                        fim_dfs.append(int(df.iloc[i-1].DataValue))
                    else:
                        fim_dfs.append(0)
                # calculate
                fim_dfs.append(np.sum(fim_dfs[1:len(dfHeader)-1]))
                writer.writerow(fim_dfs)
    print("FIM data writing completed.")

    # eachFIM item statistical processing
    dfAll = pd.read_csv(csv_path, header=0, encoding="shift-jis")
    for i in range(1, len(dfHeader)):
        # each value
        df_count = dfAll[dfHeader[i]].value_counts().sort_index()
        # CSV as save
        df_count.to_csv(dst_dir+'fim-' + str(i) + '-' +
                        dfHeader[i] + '.csv', header=True)
    print("FIM data aggregation completed.")

    # FIMdirectory listoutput
    csv_path = dst_dir + 'non_fim.csv'
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        for dir in not_fim_dirs:
            writer.writerow([root_dir+dir])
        print("Saved the list of directories without FIM data.")

    # FIMdirectory listoutput
    csv_path = dst_dir + 'exist_fim.csv'
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        for dir in fim_dirs:
            writer.writerow([root_dir+dir])
        print("Saved the list of directories with FIM data.")
    print("-------------------------------------")
