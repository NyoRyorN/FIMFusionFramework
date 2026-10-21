"""
directorymodule

CSV file specifydirectory processing run.
directory standardization row.

Key features:
- specifydirectory
- FIMLabelLessdirectory target
- change target listCSV by processing
- by allprocessing
"""

# directoryprocessing
import pandas as pd
import os


def changeDirName():
    """CSVlist based on directoryprocessing

    change target listCSV file load, specify
    directory run.standardization and purpose.

    Processing targets:
    - data/OriginalData/ NEDOdirectory
    - data/OriginalData/FIMLabelLess/ directory

    Used files:
    - data/Deletes/ChangeDirName.csv: change target list
    """
    # change target list and path set
    refDeletePath = 'data/Deletes/ChangeDirName.csv'
    targetRootPath = 'data/OriginalData/'
    targetDirs = sorted([d for d in os.listdir(targetRootPath) if os.path.isdir(
        os.path.join(targetRootPath, d)) and 'NEDO' in d])

    # change target listCSV load
    try:
        df = pd.read_csv(refDeletePath, header=0, index_col=0)
        print(f"Directory rename list loaded: {len(df)} items")
    except FileNotFoundError:
        print(f"Change target list was not found: {refDeletePath}")
        return

    # eachtarget processing
    for task, mode, oldDir, newDir in zip(df["Task"], df["KindChange"], df["OriginalDirName"], df["NewDirName"]):
        oldDirPath = targetRootPath + oldDir + "/"

        # OriginalData does not existcase
        if not os.path.exists(oldDirPath):
            # FIMLabelLess check
            oldDirPath = targetRootPath + "FIMLabelLess/" + oldDir + "/"
            if os.path.exists(oldDirPath):
                targetDirs = sorted([d for d in os.listdir(oldDirPath) if os.path.isdir(
                    os.path.join(oldDirPath, d)) and 'NEDO' in d])

        newDirPath = targetRootPath + newDir + "/"
        if task == "All":
            if not os.path.exists(newDirPath):
                os.rename(oldDirPath, newDirPath)
                print(oldDir + "is renamed to " + newDir)
            # os.rename(oldDirPath, newDirPath)

        else:
            oldFilePath = oldDirPath + task + ".csv"
            newFilePath = newDirPath + task + ".csv"
            if os.path.exists(oldFilePath):
                os.rename(oldFilePath, newFilePath)
                print(oldDir + "is renamed to " + newDir)


if __name__ == "__main__":
    changeDirName()
