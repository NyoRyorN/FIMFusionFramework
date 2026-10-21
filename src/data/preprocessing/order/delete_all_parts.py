"""
all-parts datadeletemodule

file delete, dataset.
CSV file specifydeletetarget alldeleteprocessing run.

Key features:
- deletion target list
- by delete
- all-parts data delete
- dedicateddirectorymove

delete:
- Priority="High" processing
- Task="All" case directoryall deletetarget
- deletebefore Unuseddirectory move
"""

# unused files processing
import pandas as pd
import os
import shutil


def deleteAllParts():
    """CSVlist based on all-parts datadeleteprocessing

    deletion target listCSV file load, specifyitemsbased on
    data directory delete or Unuseddirectory to move run.

    Processing flow:
    1. deletion target listCSV load
    2. Unuseddirectory create()
    3. Priority="High"
    4. Task="All" case alldelete
    5. allmove/deleteprocessing

    Used files:
    - data/Deletes/DeleteAllParts.csv: deletion target list

    Processing targets:
    - Task="All": directoryall
    - Priority="High": deletetarget
    """
    # deletion target list and path set
    refDeletePath = 'data/Deletes/DeleteAllParts.csv'
    targetRootPath = 'data/OriginalData/'
    targetDirs = sorted([d for d in os.listdir(targetRootPath) if os.path.isdir(
        os.path.join(targetRootPath, d)) and 'NEDO' in d])

    # Unuseddirectory create()
    if not os.path.exists(targetRootPath + "Unused"):
        os.makedirs(targetRootPath + "Unused")
        print("Created the Unused directory")

    # deletion target listCSV load
    try:
        df = pd.read_csv(refDeletePath, header=0, index_col=0)
        print(f"Deletion target list loaded: {len(df)} items")
    except FileNotFoundError:
        print(f"Deletion target list was not found: {refDeletePath}")
        return

    # eachdeletetarget processing
    for task, dir, priority in zip(df["Task"], df["DirName"], df["Priority"]):
        # processing
        if priority != "High":
            continue

        tempDir = targetRootPath + dir + "/"

        # alldeletetarget processing
        if task == "All":
            if not os.path.exists(tempDir):
                print(tempDir + "has already deleted.")
                continue

            shutil.move(tempDir, os.path.join(targetRootPath, "Unused"))

            print(tempDir + " is deleted.")

        else:
            if not os.path.exists(tempDir + "Unused"):
                os.makedirs(tempDir + "Unused")

            targetFile = tempDir + task + ".csv"

            if not os.path.exists(targetFile):
                print(targetFile + "has already deleted.")
                continue

            shutil.move(targetFile, os.path.join(tempDir, "Unused"))

            print(targetFile + " is deleted.")

    print("Delete all parts done.")


if __name__ == "__main__":
    deleteAllParts()
