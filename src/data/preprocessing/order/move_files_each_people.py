"""
files by participantmodule

NEDOfile per subject, directory.
filename from subjectID and information extractdirectory build.

Key features:
- filename by subject
- information extract and
- directory generate
- file /move

filename:
- subjectID: NEDO\d+-\d+ format
- information: \d{8}-\d{2}-\d{2}-\d{2}-\d{3} format
- format: {subjectID}_{information}

by:
-
- time-series data
-
"""

import os
import shutil
import re


def moveFilesEachPeople():
    """NEDOfile processing

    eachNEDOdirectory file subjectID+,
    directory move.directory does not exist
    case create.

    Processing flow:
    1. NEDOdirectory filelistget
    2. filename from subjectID and extract
    3. extractinformation movedirectory generate
    4. according tomovecreate directory
    5. file directory move

    :
    - number: r"NEDO\d+-\d+" (: NEDO001-01)
    - date: r"\d{8}-\d{2}-\d{2}-\d{2}-\d{3}" (: 20230101-10-20-30-001)

    Example:
        inputfile: "NEDO001-01_20230101-10-20-30-001_data.csv"
        move: "data/OriginalData/NEDO001-01_20230101-10-20-30-001/"

    Raises:
        AttributeError: filename format case
    """
    # data directory pathset
    root_dir = "data/"
    target_dir = root_dir+"OriginalData/"

    # Processing targets NEDOlist
    nedo_lists = ["001", "002", "003", "004", "006", "007", "010"]

    # eachNEDOdirectory file
    for i in range(len(nedo_lists)):
        # currently processing NEDOdirectorypath
        dir_path = target_dir + "NEDO" + nedo_lists[i] + "/"

        # directory filelist get(sorted, file)
        files = sorted([f for f in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, f))])

        # eachfile directory move
        for j in files:
            try:
                # filename from subjectIDinformation extract
                # : NEDO001-01 format
                number = re.search(r"NEDO\d+-\d+", j)

                # filename from /information extract
                # : 20230101-10-20-30-001 format
                date = re.search(r"\d{8}-\d{2}-\d{2}-\d{2}-\d{3}", j)

                # movedirectory generate(subjectID_information)
                dst_dir = number.group() + "_" + date.group()

                # movedirectory does not existcase create
                if dst_dir not in os.listdir(target_dir):
                    os.makedirs(target_dir + dst_dir)
                    print(f"Directory created: {dst_dir}")

                # file directory move
                shutil.move(dir_path + j, target_dir + dst_dir)
                print(f"Moved: {j} -> {dst_dir}/")

            except AttributeError as e:
                print(f"Warning: filename does not match the expected format: {j}")
                continue

    print("Move files each people done.")


if __name__ == '__main__':
    moveFilesEachPeople()
