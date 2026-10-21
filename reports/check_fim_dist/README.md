# reports/check_fim_dist — FIM distribution checks

Checks and outputs of the FIM-score distribution across the dataset.

## Files

| File | Role |
|---|---|
| `fimdist_outputs.py` | Aggregates and outputs the FIM-score distribution per item (using `FIMitems()` from `src/utils/names.py`). |
| `missing_motion_class_view.py` | Displays `data/Distributions/MissingMotionCounts.csv` for one FIM item with the 1-7 scores merged into custom classes (e.g. `1-6,7`): per-task counts/missing rates plus an `AnyMissing` row (from `MotionStatus.csv` + `all_fim.csv`) counting subjects with at least one missing motion. The whole output is restricted to a `--any-tasks` subset (default all 11 tasks). See `docs/missing_motion.md`. |
