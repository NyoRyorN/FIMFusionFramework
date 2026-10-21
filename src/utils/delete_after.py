import os
import shutil
import glob


def DeleteAfter(input_dir: str):
    """Delete model files and temporary directories under input_dir.

    Processing flow:
        1. Recursively delete all .pth files.
        2. Find train/test subdirectories at every level and move their CSV
           files to the parent directory, then delete those subdirectories.
        3. Delete any remaining empty directories bottom-up so that parents
           emptied by child removal are also cleaned up.
    """
    # ------------------------------------------------------------------
    # Delete .pth files
    # ------------------------------------------------------------------
    deleted_any = False
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pth"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                    deleted_any = True
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    if not deleted_any:
        print("No .pth files found")

    # ------------------------------------------------------------------
    # Collect train/test subdirectories (snapshot before deletion)
    # ------------------------------------------------------------------
    subdirs_to_process: list[tuple[str, str]] = []
    for root, dirs, _ in os.walk(input_dir):
        for d in dirs:
            if "train" in d or "test" in d:
                subdirs_to_process.append((os.path.join(root, d), root))

    # ------------------------------------------------------------------
    # Move CSVs to parent directory, then delete train/test subdirectories
    # ------------------------------------------------------------------
    for subdir_path, parent_path in subdirs_to_process:
        if not os.path.exists(subdir_path):
            continue
        for f in glob.glob(os.path.join(subdir_path, "*.csv")):
            dest = os.path.join(parent_path, os.path.basename(f))
            try:
                shutil.move(f, dest)
                print(f"Moved {f} -> {dest}")
            except Exception as e:
                print(f"Failed to move {f}: {e}")
        try:
            shutil.rmtree(subdir_path)
            print(f"Deleted directory {subdir_path}")
        except Exception as e:
            print(f"Failed to delete directory {subdir_path}: {e}")

    # ------------------------------------------------------------------
    # Delete remaining empty directories (bottom-up so parents are
    # re-evaluated after their children are removed).
    # Skip directories whose name contains "Fold" to avoid accidental
    # deletion when CSV moves fail and the fold dir becomes empty.
    # ------------------------------------------------------------------
    for root, dirs, files in os.walk(input_dir, topdown=False):
        if root == input_dir:
            continue
        if not os.path.exists(root):
            continue
        if "Fold" in os.path.basename(root):
            continue
        if not os.listdir(root):
            try:
                os.rmdir(root)
                print(f"Deleted empty directory {root}")
            except Exception as e:
                print(f"Failed to delete empty directory {root}: {e}")

    print("==============================================")
    print("DeleteAfter is finished.")
    print("==============================================")


if __name__ == "__main__":
    input_dir = input("Enter the path: ")
    DeleteAfter(input_dir)
