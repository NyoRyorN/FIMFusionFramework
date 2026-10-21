"""
/beforeprocessingintegraterunmodule

FIM, format convert to
integrateprocessing.multiple run, quality improvement and
row.

Processing flow:
1. unused directories move/delete
2. unused files delete
3. files by participant
4. duplicate filesdelete
5. filename standardization
6. non-FIM files
7. all-parts datadelete
8. selected-parts datadelete
9. directory
10. task-name change

method:
    python orderliness_data.py

Notes:
- runbefore backup
- processing
- processing in temporal case
"""

# individualmodule(commented out - current integratemodule)
# from src.data.orderlinessOperateFiles.move_delete_dirs import moveDeleteDirs
# from src.data.orderlinessOperateFiles.delete_unused_files import deleteUnusedFiles
# from src.data.orderlinessOperateFiles.move_files_each_people import moveFilesEachPeople
# from src.data.orderlinessOperateFiles.delete_duplicate_files import deleteDuplicateFiles
# from src.data.orderlinessOperateFiles.rename import rename
# from src.data.orderlinessOperateFiles.move_not_fim_file import moveNotFimFile
# from src.data.orderlinessOperateFiles.delete_all_parts import deleteAllParts
# from src.data.orderlinessOperateFiles.delete_some_parts import deleteSomeParts
# from src.data.orderlinessOperateFiles.change_dir_name import changeDirName
# from src.data.orderlinessOperateFiles.change_task_name import changeTaskName

# integratemodule from
from src.data.preprocessing.order import *

if __name__ == '__main__':
    """Main execution block for data organization.

    Runs each processing function sequentially to organize the dataset.
    The order is set according to processing dependencies.

    Execution order:
    1. moveDeleteDirs(): organize unused directories (foundation step)
    2. deleteUnusedFiles(): delete unused files (reduce storage)
    3. moveFilesEachPeople(): organize data by participant (structure data)
    4. deleteDuplicateFiles(): remove duplicates (improve data quality)
    5. rename(): standardize filenames (apply naming rules)
    6. moveNotFimFile(): organize non-FIM files (classification)
    7. deleteAllParts(): delete all-parts data (narrow the dataset)
    8. deleteSomeParts(): delete selected parts (fine-grained cleanup)
    9. changeDirName(): standardize directory names (improve readability)
    10. changeTaskName(): standardize task names (ensure consistency)
    """
    print("Starting data organization...")

    print("1/10: Moving and deleting unused directories...")
    moveDeleteDirs()

    print("2/10: Deleting unused files...")
    deleteUnusedFiles()

    print("3/10: Organizing files by participant...")
    moveFilesEachPeople()

    print("4/10: Deleting duplicate files...")
    deleteDuplicateFiles()

    print("5/10: Standardizing filenames...")
    rename()

    print("6/10: Organizing non-FIM files...")
    moveNotFimFile()

    print("7/10: Deleting all-parts data...")
    deleteAllParts()

    print("8/10: Deleting selected-parts data...")
    deleteSomeParts()

    print("9/10: Renaming directories...")
    changeDirName()

    print("10/10: Renaming task files...")
    changeTaskName()

    print("Data organization completed.")
