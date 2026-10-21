"""
task filemodule

CSV file specifytask file processing run.
filename row.

Key features:
- specifytask file
- directory filename standardization
- change target listCSV by processing
- filemove by beforerun
"""

# task fileprocessing
import pandas as pd
import os
import shutil


def changeTaskName():
    """CSVlist based on task fileprocessing

    change target listCSV file load, specify
    task file run.filename purpose.

    Processing targets:
    - eachdirectory task file

    Used files:
    - data/Deletes/ChangeTaskName.csv: change target list
    """
    # change target list and path set
    refDeletePath = 'data/Deletes/ChangeTaskName.csv'
    targetRootPath = 'data/OriginalData/'
    targetDirs = sorted([d for d in os.listdir(targetRootPath) if os.path.isdir(
        os.path.join(targetRootPath, d)) and 'NEDO' in d])

    # change target listCSV load
    try:
        df = pd.read_csv(refDeletePath, header=0, index_col=0)
        print(f"Task rename list loaded: {len(df)} items")
    except FileNotFoundError:
        print(f"Change target list was not found: {refDeletePath}")
        return

    # eachtarget task processing
    for oldTask, newTask, dirName in zip(df["OriginalTask"], df["NewTask"], df["DirName"]):
        # target directory path build
        targetPath = targetRootPath + dirName + "/"

        # backup directory set
        oldPath = targetPath + "/TaskMistakes/"

        # backup directory createcheck
        if not os.path.exists(oldPath):
            os.makedirs(oldPath)
            print(f"Backup directory created: {oldPath}")

        # source file backuprun
        oldTaskFile = targetPath + oldTask + ".csv"
        newTaskFile = targetPath + newTask + ".csv"

        # task file backup
        if os.path.exists(oldTaskFile):
            backupOldPath = oldPath + oldTask + ".csv"
            if not os.path.exists(backupOldPath):
                shutil.move(oldTaskFile, backupOldPath)
                print(f"Old task file backed up: {oldTask}.csv")

        # task file backup
        if os.path.exists(newTaskFile):
            backupNewPath = oldPath + newTask + ".csv"
            if not os.path.exists(backupNewPath):
                shutil.move(newTaskFile, backupNewPath)
                print(f"New task file backed up: {newTask}.csv")

        # backupfile filename copy
        if os.path.exists(oldPath + newTask + ".csv"):
            shutil.copy(oldPath + newTask + ".csv",
                        targetPath + oldTask + ".csv")
            print(f"Task rename completed: {oldTask} -> {newTask} (dir: {dirName})")
        else:
            print(f"Target file for change was not found: {newTask}.csv in {dirName}")

    print("All task filename changes completed")


def main():
    """Main processing entry point

    task fileprocessing run and.
    backup allfilename changeprocessing.
    """
    try:
        print("=== Task filename change started ===")
        changeTaskName()
        print("=== All processing completed ===")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Run processing when the script is executed directly
    main()
