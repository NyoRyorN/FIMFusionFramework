"""
duplicate filesdeletemodule

file details duplicates/delete.
SHA-256value by duplicates,
dataset and optimization.

Key features:
- SHA-256 by duplicate files
- directory delete
- deletefilelist logoutput
- details

all:
- check
- filedetails
- delete logrecord

Processing effects:
- reduce
- duplicates
- quality improvement
"""

import os
import glob
import hashlib
import shutil


def get_body(fname):
    """file details get

    specifyfile load,
    details return.valuecalculate and file.

    Args:
        fname (str): loadtarget file path

    Returns:
        bytes: file details()

    Raises:
        FileNotFoundError: file does not existcase
        PermissionError: fileload case
    """
    with open(fname, "rb") as f:
        return f.read()


def deleteDuplicateFiles():
    """duplicate files /deleteprocessing

    data directory allfile for SHA-256value
    calculate, value file(duplicate files) delete.
    also, directory delete.

    processing:
    1. eachdirectory
    2. directory delete
    3. eachfile SHA-256valuecalculate
    4. value duplicatescheck
    5. duplicates details()
    6. duplicate files delete
    7. deletefilelist output

    duplicates:
    -: SHA-256value
    -: filedetails all

    outputfile:
    - delete_dirs.txt: deletefilelist

    Example:
        value abc123... file 2case:
        1. file: dictionary
        2. 2 file: duplicates and delete
    """
    # data directory pathset
    root_dir = "data/"
    target_dir = root_dir+"OriginalData/"

    # directory directory listget
    dirs = sorted([f for f in os.listdir(target_dir)
                  if os.path.isdir(os.path.join(target_dir, f))])

    # deletefilelist recordlist
    delete_dirs = []

    # eachdirectory processing
    for i in range(len(dirs)):
        dir_path = target_dir + dirs[i] + "/"

        # directory filelist get
        files = glob.glob(dir_path + "*")

        # directory case delete
        if len(files) == 0:
            shutil.rmtree(dir_path)
            print("delete dir: ", dir_path)
        else:
            # value and dictionary(duplicates)
            body_dict = {}

            # directory eachfile processing
            for f in files:
                try:
                    # filedetails get
                    body = get_body(f)

                    # SHA-256value calculate
                    v = hashlib.sha256(body).hexdigest()

                    # duplicates
                    if v in body_dict:
                        # value file
                        f2 = body_dict[v]

                        # : details
                        if body == get_body(f2):
                            # all duplicatesfile delete
                            os.remove(f)
                            print("delete file: ", f)
                            delete_dirs.append(f)
                        else:
                            print(f"Hash collision detected (rare case): {f}")
                    else:
                        # value case dictionary
                        body_dict[v] = f

                except Exception as e:
                    print(f"File processing error: {f}, error: {str(e)}")
                    continue

    # deletefilelist file output
    try:
        with open("delete_dirs.txt", "w", encoding='utf-8') as f:
            f.write("Deleted file list:\n")
            f.write("=" * 50 + "\n")
            for i in delete_dirs:
                f.write(i + "\n")
        print(f"Wrote deleted file list to delete_dirs.txt（{len(delete_dirs)} items）")
    except Exception as e:
        print(f"Log file output error: {str(e)}")

    print("Delete duplicate files done.")


if __name__ == '__main__':
    deleteDuplicateFiles()
