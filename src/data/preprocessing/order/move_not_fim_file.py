"""
non-FIM filesmodule

FIM does not existdata directory dedicated move,
dataset row.FIMevaluate presence/absence by processing.

Key features:
- FIMfile
- FIMdirectory
- dataset
- target data

Processing effects:
- enabled
- all
- dataset
"""

import os
import shutil


def moveNotFimFile():
    """FIMfile does not existdirectory moveprocessing

    eachdirectory FIMfile,
    does not existcase dedicated FIMLabelLessdirectory move.
    purpose.

    Processing flow:
    1. NEDOdirectory list get
    2. movedirectory create()
    3. eachdirectory FIMfilecheck
    4. FIMfile directory move

    moveitems:
    - directory "NEDO"
    - directory FIMfile does not exist

    Example:
        move: data/OriginalData/NEDO001-01_20230101-10-20-30-001/
        move: data/OriginalData/FIMLabelLess/NEDO001-01_20230101-10-20-30-001/
    """
    # directorypath set
    target_dir = 'data/OriginalData/'
    move_dir = target_dir + 'FIMLabelLess/'

    # NEDOdirectory list get(NEDO directory)
    dirs = sorted([f for f in os.listdir(target_dir) if os.path.isdir(
        os.path.join(target_dir, f)) and "NEDO" in f])

    # movedirectory does not existcase create
    if not os.path.exists(move_dir):
        os.makedirs(move_dir)
        print(f"Created the FIMLabelLess directory: {move_dir}")

    # eachNEDOdirectory
    moved_count = 0
    for dir in dirs:
        # directory FIMfile search
        files = sorted([f for f in os.listdir(target_dir + dir + "/")
                        if os.path.isfile(os.path.join(target_dir + dir + "/", f)) and "FIM" in f])

        # FIMfile does not existcase directory and move
        if len(files) == 0:
            try:
                shutil.move(target_dir + dir, move_dir)
                print(f"Moved because the FIM file was missing: {dir}")
                moved_count += 1
            except Exception as e:
                print(f"Movederror: {dir}, error: {str(e)}")

    print(f"Move not FIM files done. (moved: {moved_count} items)")


if __name__ == '__main__':
    moveNotFimFile()
