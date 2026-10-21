"""
filename standardizationmodule

NEDOfile filename English notation,
processing automation and accessibility improvement.

Key features:
- Japanese task names English notation to convert
- filename standardization
- per taskfile

convert:
- sitting tasks: Sittingrelated
- standing tasks: Standingrelated
- reach tasks: Forward/Downwarddirection designation
- others: Walking, Squat, evaluate

filenameformat:
- before change: Japanese task names filename
- after change: task name.csv

Processing effects:
- easier programmatic processing
- support for international data sharing
- simplified file management
"""

import os
import re


def rename():
    """NEDOfilename English notationprocessing

    eachdirectory filename from English notation.
    task name standardization by, after automatic processing.

    convert:
    - Japanese task names corresponding English name
    - filechild.csv preserve
    - does not existtask name file skip

    Processing flow:
    1. eachdirectory
    2. directoryfilelist get
    3. filename from task name
    4. task name to convert
    5. filename run

    task nameconvert:
    - "１- " -> "01-Sitting"
    - "３- before" -> "03-SittingForward"
    - "１１- row" -> "11-Walking"
    - "FIM" -> "FIM" (none)

    Notes:
    - file with the same name case overwrite
    - convertout of scope filename processing
    """
    # Japanese task nameslist(source for conversion)
    task_lists = ["１- 座位", "２- 上肢", "３- 座位前方リーチ", "４- 座位下方リーチ（右）", "５- 座位下方リーチ（左）",
                  "６- 立位", "７- 立位前方リーチ", "８- 立位下方リーチ（右）", "９- 立位下方リーチ（左）",
                  "１０- スクワット", "１１- 歩行", "INFO", "BI", "FIM", "REPORT", "WHODAS"]

    # task namelist(conversion target)
    rename_tasks = ["01-Sitting", "02-Upper", "03-SittingForward", "04-SittingDownwardRight", "05-SittingDownwardLeft",
                    "06-Standing", "07-StandingForward", "08-StandingDownwardRight", "09-StandingDownwardLeft",
                    "10-Squat", "11-Walking", "INFO", "BI", "FIM", "REPORT", "WHODAS"]

    # processingtarget directory path
    target_dir = 'data/OriginalData/'

    # directory list get
    dirs = sorted([f for f in os.listdir(target_dir)
                  if os.path.isdir(os.path.join(target_dir, f))])

    # eachdirectory file processing
    for i in range(len(dirs)):
        # currently processing directorypath
        dir_path = target_dir + dirs[i] + "/"

        # directory filelist get
        files = sorted([f for f in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, f))])

        # eachfile for processing run
        for f in files:
            # information and information extract(commented out - currently unused)
            # date = re.search(r"\d{8}-\d{2}-\d{2}-\d{2}-\d{3}", f).group()
            # code = re.search(r"NEDO\d+-\d+",f).group()

            # task index initialize(for not-found cases)
            index = 50

            # filename task name
            for k in range(len(task_lists)):
                if re.search(task_lists[k], f):
                    index = k
                    print(f"Task identified: {f} → {task_lists[k]} (index: {k})")
                    break

            # when found run
            if index < len(task_lists):
                # filename generate
                # dst_dir = code + "_" + date + "_" (commented out)
                name = target_dir+dirs[i]+"/"+rename_tasks[index]+".csv"

                # filename change run
                try:
                    os.rename(target_dir+dirs[i]+"/"+f, name)
                    print(f"Rename succeeded: {f} → {rename_tasks[index]}.csv")
                except OSError as e:
                    print(f"Rename error: {f}, error: {str(e)}")
            else:
                print(f"No matching task: {f} (skipped)")

    print("Rename done.")


if __name__ == '__main__':
    rename()
