# src/data/preprocessing/order — Raw file/directory organization scripts

Single-purpose scripts that clean up and reorganize the raw NEDO data tree. They are
orchestrated by `../orderliness_data.py`; each performs one file/directory operation.

## Files

| File | Role |
|---|---|
| `change_dir_name.py` | Renames data directories referenced by a CSV mapping. |
| `change_task_name.py` | Renames task files referenced by a CSV mapping. |
| `delete_all_parts.py` | Deletes all-parts data files not needed for the dataset. |
| `delete_duplicate_files.py` | Detects and deletes duplicate files. |
| `delete_some_parts.py` | Deletes specified frame ranges from CSV files to remove selected intervals. |
| `delete_unused_files.py` | Deletes unused files from the NEDO data directory. |
| `move_delete_dirs.py` | Moves/deletes unneeded directories in the NEDO tree. |
| `move_files_each_people.py` | Groups NEDO files into per-subject directories. |
| `move_not_fim_file.py` | Moves files with no corresponding FIM record into a dedicated location. |
| `rename.py` | Standardizes NEDO filenames to English notation. |
