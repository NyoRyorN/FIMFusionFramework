"""
Partial data deletion module.

Deletes specified frame ranges from CSV files to remove selected intervals
from time-series data and improve data quality.

Key features:
- Delete data after a specified frame (DeleteAfter)
- Delete data before a specified frame (DeleteBefore)
- Delete one specified frame (DeletePart)
- Back up source files automatically
- Control processing by priority
"""

import pandas as pd
import os
import shutil


def deleteSomeParts():
    """Run partial data deletion.

    Deletes the specified frame range based on the deletion-target CSV.
    The purpose is to improve time-series data quality.

    Deletion modes:
    - DeleteAfter: delete the specified frame and all later frames
    - DeleteBefore: delete the specified frame and all earlier frames
    - DeletePart: delete only the specified frame

    Used files:
    - data/Deletes/DeleteSomeParts.csv: deletion-target list
    """
    """Run partial data deletion.

    Deletes the specified frame range based on the deletion-target CSV.
    The purpose is to improve time-series data quality.

    Deletion modes:
    - DeleteAfter: delete the specified frame and all later frames
    - DeleteBefore: delete the specified frame and all earlier frames
    - DeletePart: delete only the specified frame

    Used files:
    - data/Deletes/DeleteSomeParts.csv: deletion-target list
    """
    # deletion target list and path set
    refDeletePath = 'data/Deletes/DeleteSomeParts.csv'
    targetRootPath = 'data/OriginalData/'
    targetDirs = sorted([d for d in os.listdir(targetRootPath) if os.path.isdir(
        os.path.join(targetRootPath, d)) and 'NEDO' in d])

    # deletion target listCSV load
    try:
        df = pd.read_csv(refDeletePath, header=0, index_col=0)
        print(f"Partial deletion list loaded: {len(df)} items")
    except FileNotFoundError:
        print(f"Deletion target list was not found: {refDeletePath}")
        return
    # eachdeletetarget task processing
    for task, mode, num, dir, priority in zip(df["Task"], df["DeleteKind"], df["FrameNum"], df["DirName"], df["Priority"]):
        # (High processing)
        if priority != "High":
            print(f"Priority Low - skipped: {task} in {dir}")
            continue

        # target directory path build
        tempDir = targetRootPath + dir + "/"

        # backup directory createcheck
        originalDir = tempDir + "Original"
        if not os.path.exists(originalDir):
            os.makedirs(originalDir)
            print(f"Backup directory created: {originalDir}")

        # filepath set
        targetFile = tempDir + task + ".csv"
        oldFilePath = tempDir + "Original/" + task + ".csv"

        # source file backuprun
        if not os.path.exists(oldFilePath):
            if os.path.exists(targetFile):
                shutil.copy(targetFile, tempDir + "Original/")
                print(f"Backup created: {task}.csv")
            else:
                print(f"Target file was not found: {targetFile}")
                continue

        # CSV file load
        try:
            df_data = pd.read_csv(targetFile, header=0, index_col=0)
        except Exception as e:
            print(f"File read error: {targetFile}, error: {e}")
            continue

        # deletion mode processingrun
        if mode == "DeleteAfter":
            # specifyframeand later delete(2columnand later)
            df_data.iloc[num:, 2:] = None
            df_data.to_csv(targetFile, index=True)
            print(f"Frame {num} and later deleted: {task} in {dir}")

        elif mode == "DeleteBefore":
            # specifyframeand earlier delete(2columnand later)
            df_data.iloc[:num+1, 2:] = None
            df_data.to_csv(targetFile, index=True)
            print(f"Frame {num} and earlier deleted: {task} in {dir}")

        # In this experiment, DeletePoints was mistakenly skipped, so it is commented out.
        # Applying DeletePoints will not reproduce the results, so avoid using DeletePoints.
        # elif mode == "DeletePoints":
        #     # frame delete(2columnand later)
        #     df_data.iloc[num, 2:] = None
        #     df_data.to_csv(targetFile, index=True)
        #     print(f"Frame {num} deleted: {task} in {dir}")

        else:
            print(f"Unknown deletion mode: {mode} for {task} in {dir}")

        print(f"{dir} {task} partial deletion completed")

    print("All partial deletion processing completed")


def main():
    """Main processing entry point

    deleteprocessing run and.
    time-series data quality improvement purpose and processing.
    """
    try:
        print("=== Partial data deletion started ===")
        deleteSomeParts()
        print("=== All processing completed ===")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Run processing when the script is executed directly
    main()
