# src/data/preprocessing — Raw-data preprocessing pipeline

Turns the raw NEDO measurements into the reduced-joint `ConciseData` that datasets are built
from. Two entrypoints run in sequence: first `orderliness_data.py` organizes the raw file
tree, then `make_concise_data.py` runs the numeric pipeline (time complementation → filtering
+ feature computation → reduced-joint output). Reusable numeric helpers live in `utils/`;
the one-off file/directory ordering scripts live in `order/`.

## Files

| File | Role |
|---|---|
| `orderliness_data.py` | Organizes the raw data tree: move/delete unused dirs and files, group files per subject, drop duplicates, standardize filenames, separate non-FIM files. |
| `make_concise_data.py` | Builds `data/ConciseData/` straight from `data/OriginalData/`: time complementation → rolling-median/mean filtering + velocity → SimpleJoints positions/velocities with recomputed angle/bone length. |

## Subfolders

| Subfolder | Role |
|---|---|
| [`order/`](order/) | File/directory organization scripts invoked by `orderliness_data.py` (rename, move, delete, deduplicate). |
| [`utils/`](utils/) | Numeric preprocessing helpers: velocity/angle/bone-length computation and FIM-distribution / missing-motion / time-shortage quality checks. |

## Run

```bash
source path.sh
python src/data/preprocessing/orderliness_data.py   # organize raw data
python src/data/preprocessing/make_concise_data.py   # full numeric pipeline
```
