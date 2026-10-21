# Missing-Motion Distribution

While preprocessing runs, `src/data/preprocessing/make_concise_data.py` classifies every (subject, motion) slot of the 11 tasks and writes the per-FIM-item, per-score (1-7) missing-motion distribution to `data/Distributions/`. Skeleton-quality failures are identified only from raw CSVs that are already in `Unused/`; this script does not move raw CSVs into `Unused/`.

## Usage

```bash
source path.sh
python3 src/data/preprocessing/make_concise_data.py
```

## Motion status classification

| Status | Reason | Condition |
| --- | --- | --- |
| `Present` | - | Converted and saved successfully |
| `MissingSkeletonQuality` | `Unused` | Raw CSV already moved to `Unused/` |
| `MissingNoVideo` | `NoPattern` | No sampling pattern matched; also recorded in `ManyShortageTimeFiles.txt` |
| `MissingNoVideo` | - | No raw data and no quality evidence (the recording never existed) |

## Output files (`data/Distributions/`)

| File | Columns | Content |
| --- | --- | --- |
| `MotionStatus.csv` | `Directory,Task,Status,Reason` | Per (subject, motion) detail (subjects with FIM data only) |
| `MissingMotionCounts.csv` | `Item,Task,Score,Subjects,Present,MissingSkeletonQuality,MissingNoVideo` | Counts per FIM item (13) × motion × score (1–7). Each row satisfies `Present + MissingSkeletonQuality + MissingNoVideo = Subjects` |
| `MissingMotionSummary.csv` | `Item,Score,Subjects,Present,MissingSkeletonQuality,MissingNoVideo` | Task-summed view. `Subjects` is the subject count for that score, so the three status columns add up to `Subjects × 11` |

Each subject's status detail is also written inside its own folder.

| File | Columns | Content |
| --- | --- | --- |
| `data/ConciseData/<subject>/MissingMotion.csv` | `Task,Status,Reason` | The 11 motion statuses for that subject (sorted by task). Not written for subjects without FIM data |

## Aggregation scope

- Subjects without FIM data (see `non_fim.csv`) are excluded.
- For each item, subjects whose score is NaN, <= 0 or outside 1–7 are excluded from that item only (same validity rule as `makeSomeTasksDataset`). `Sum` is never aggregated.
- Subjects whose whole session was removed (directly under `data/OriginalData/Unused/`) are not enumerated and stay out of scope.
- To inspect specific FIM items, filter `MissingMotionCounts.csv` by the `Item` column (all items are always emitted).

## Viewing by custom score class

`reports/check_fim_dist/missing_motion_class_view.py` displays `MissingMotionCounts.csv`
for one FIM item, merging the 1–7 score rows into custom classes (e.g. a
6-vs-1 binary split), and prints per-task counts plus a missing rate for each
class. The whole output is restricted to a selectable task subset
(`--any-tasks`, comma-separated 1-based task numbers with `-` ranges; default
all 11 tasks, e.g. `1-10` to exclude `11-Walking` when walking is out of
scope). It also joins `MotionStatus.csv` with `all_fim.csv` (paths
overridable via `--status-csv` / `--fim-csv`) to append an `AnyMissing`
pseudo-task row per class counting subjects with at least one missing motion
among the target tasks: a subject with any skeleton-quality rejection is
counted under `MissingSkeletonQuality`, otherwise a subject with any missing
recording under `MissingNoVideo`, so the three status columns still add up to
`Subjects`. The `MissingRate(%)` of that row is the share of subjects with
≥1 missing motion.

```bash
python reports/check_fim_dist/missing_motion_class_view.py Eating 1-6,7
python reports/check_fim_dist/missing_motion_class_view.py 0 1-6,7      # same item, by FIMitems() index
python reports/check_fim_dist/missing_motion_class_view.py Bathing 1-3,4-5,6-7 -o out.csv
python reports/check_fim_dist/missing_motion_class_view.py Eating 1-6,7 --any-tasks 1-10  # exclude 11-Walking
```

The `item` argument accepts either the item name or its 0-based `FIMitems()`
index (see `--help` for the full mapping). The class spec comma-separates
classes; within a class, `-` is an inclusive score range and `+` joins
multiple ranges/values (e.g. `1-3+5,4,6-7`).
