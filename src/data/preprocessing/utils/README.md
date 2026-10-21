# src/data/preprocessing/utils — Numeric preprocessing helpers

Reusable numeric routines called by `../make_concise_data.py` (feature computation) and by the
data-quality analyses. Split into two groups: per-frame feature computation and dataset
quality checks / distribution outputs.

## Files

| File | Role |
|---|---|
| `calc_velocity.py` | Computes joint velocities from time-series positions. |
| `calc_angle_length.py` | Computes joint angles and bone lengths from the parent-child graph structure. |
| `fim_dist_output.py` | Aggregates/analyzes the FIM (Functional Independence Measure) score distribution. |
| `plot_fim_dist.py` | Visualizes the aggregated FIM-score distribution. |
| `missing_motion_output.py` | Outputs the per-FIM-score distribution of missing-motion (subject, task) classifications. |
| `time_shortage_check.py` | Checks/analyzes sequences that are too short in the temporal dimension. |
