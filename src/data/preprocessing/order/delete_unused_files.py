"""
filedeletemodule

NEDOdata directory from file delete,
optimization and dataset row.

deletetarget file:
- file:._ file(macOS file)
- CSV file:.csv child file

Key features:
- filechild by
- generatefile
- dataset

all:
- CSV file
- deleteitems by delete
"""

import os


def deleteUnusedFiles():
    """NEDO from file delete

    specifyNEDOdata directory from, file
    delete.CSV file and file deletetarget.

    deleteitems:
    1. filename "._" (file)
    2. child ".csv" (file)

    Processing flow:
    1. eachNEDOdirectory
    2. filelist get(file)
    3. deleteitems check
    4. items file delete

    target directory:
    - NEDO001, 002, 003, 004, 006, 007, 010

    Example:
        deletetarget:._metadata.txt, README.md, data.json
        target: experiment_data.csv, results.csv
    """
    # data directory pathset
    root_dir = "data/"
    target_dir = root_dir+"OriginalData/"

    # Processing targets NEDOlist
    nedo_lists = ["001", "002", "003", "004", "006", "007", "010"]

    # eachNEDOdirectory from file delete
    for i in range(len(nedo_lists)):
        # currently processing NEDOdirectorypath
        dir_path = target_dir + "NEDO" + nedo_lists[i] + "/"

        # directory filelist get(sorted, file)
        files = sorted([f for f in os.listdir(dir_path)
                       if os.path.isfile(os.path.join(dir_path, f))])

        # eachfile for deleteitems
        for j in files:
            # deleteitems1: file(._)
            # deleteitems2: CSV file(child.csv)
            if j[:2] == "._" or j[-3:] != "csv":
                os.remove(dir_path + j)
                print(f"Deleted: {j}")

    print("Delete unused files done.")


if __name__ == '__main__':
    deleteUnusedFiles()
