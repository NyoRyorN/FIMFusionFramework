"""
directorymodule

NEDO directory,
to processing row.

Key features:
- directory file parent directory move
- directory delete
- NEDOdataset standardization

Processing targets:
- NEDO001, 002, 003, 004, 006, 007, 010
- eachNEDO

Notes:
- moveprocessing
- file with the same name case overwrite
"""

import os
import shutil


def moveDeleteDirs():
    """NEDOdata directory processing

    NEDO eachdata directory directory from
    file parent directory move, directory delete.
    by file.

    Processing flow:
    1. targetNEDOdirectory
    2. directory list get
    3. directoryfile parent directory to move
    4. directory delete

    Example:
        before change: data/OriginalData/NEDO001/subdir/file.csv
        after change: data/OriginalData/NEDO001/file.csv
    """
    # data directory pathset
    root_dir = "data/"
    target_dir = root_dir+"OriginalData/"

    # Processing targets NEDOlist
    # NEDO enableddataset
    nedo_lists = ["001", "002", "003", "004", "006", "007", "010"]

    # eachNEDOdirectory for filemoveprocessing run
    for i in range(len(nedo_lists)):
        # currently processing NEDOdirectorypath
        dir_path = target_dir + "NEDO" + nedo_lists[i] + "/"

        # NEDOdirectory directory list get
        # sorted, directory target and
        dirs = sorted([f for f in os.listdir(dir_path)
                      if os.path.isdir(os.path.join(dir_path, f))])

        # eachdirectory file parent directory move
        for j in range(len(dirs)):
            # directory filelist get
            files = sorted([f for f in os.listdir(dir_path + dirs[j] + "/")
                            if os.path.isfile(os.path.join(dir_path + dirs[j] + "/", f))])

            # eachfile NEDOdirectory move
            for k in files:
                shutil.move(dir_path + dirs[j] + "/" + k, dir_path)

        # directory delete
        for j in dirs:
            shutil.rmtree(dir_path + j)

    print("Move and delete dirs done.")


if __name__ == '__main__':
    moveDeleteDirs()
