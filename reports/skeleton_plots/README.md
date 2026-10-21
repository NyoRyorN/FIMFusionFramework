# reports/skeleton_plots — Skeleton and GradCAM visualization

Renders skeleton sequences and overlays Attention / GradCAM weights (from
`src/model/attention/cam_fusion.py`, saved during evaluation) as videos, multi-task
subplots, and distribution statistics. `plot_skeleton.py` and the animation scripts build on
the shared 2D/3D renderers.

## Files

| File | Role |
|---|---|
| `skeleton_renderer_2d.py` | `Plot2D`: shared 2D skeleton renderer (joints/bones per frame). |
| `skeleton_renderer_3d.py` | Shared 3D skeleton renderer. |
| `plot_skeleton.py` | Renders static skeleton figures via the 2D renderer. |
| `animate_gradcam_per_task.py` | Animates GradCAM-weighted skeleton videos, one per action task. |
| `animate_gradcam_all_tasks_subplot.py` | Animates all tasks together as a subplot grid. |
| `animate_gradcam_all_tasks_compact.py` | Animates all tasks together in a compact combined view. |
| `export_gradcam_stats.py` | Exports seed-level GradCAM representative values (per-subject Max/Mean/Median at action/joint/body-part level). |
