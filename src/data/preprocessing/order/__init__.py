"""
processingmoduleintegrate

FIM beforeprocessing each.
quality improvement,, standardization row function integrate.

:
- directory/file
- duplicates/
-
- /

eachmodule:
- changeDirName: directory
- changeTaskName: task name standardization
- deleteAllParts: all delete
- deleteDuplicateFiles: duplicate files /delete
- deleteSomeParts: delete
- deleteUnusedFiles: Used files delete
- moveDeleteDirs: unused directories move/delete
- moveFilesEachPeople: files by participant
- moveNotFimFile: FIM file
- rename: filename
"""

from src.data.preprocessing.order.change_dir_name import changeDirName
from src.data.preprocessing.order.change_task_name import changeTaskName
from src.data.preprocessing.order.delete_all_parts import deleteAllParts
from src.data.preprocessing.order.delete_duplicate_files import deleteDuplicateFiles
from src.data.preprocessing.order.delete_some_parts import deleteSomeParts
from src.data.preprocessing.order.delete_unused_files import deleteUnusedFiles
from src.data.preprocessing.order.move_delete_dirs import moveDeleteDirs
from src.data.preprocessing.order.move_files_each_people import moveFilesEachPeople
from src.data.preprocessing.order.move_not_fim_file import moveNotFimFile
from src.data.preprocessing.order.rename import rename

__all__ = [
    "changeDirName",
    "changeTaskName",
    "deleteAllParts",
    "deleteDuplicateFiles",
    "deleteSomeParts",
    "deleteUnusedFiles",
    "moveDeleteDirs",
    "moveFilesEachPeople",
    "moveNotFimFile",
    "rename"
]
