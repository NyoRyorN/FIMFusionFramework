# src/utils — Experiment utilities

Cross-cutting helpers shared across the pipeline: canonical name/label definitions, plot
font setup, confusion-matrix rendering, and post-run cleanup.

## Files

| File | Role |
|---|---|
| `names.py` | Canonical definitions: action `Tasks()`, `FIMitems()`, and `SimpleJoints()` — the single source of truth for indices/labels used across configs and datasets. |
| `decides.py` | Decision/selection helpers (e.g. resolving experiment options from config). |
| `make_confusion_matrix.py` | Builds and renders confusion matrices from predictions/true labels. |
| `set_plt_font.py` | Configures matplotlib fonts for consistent figures. |
| `delete_after.py` | `DeleteAfter`: post-run cleanup helper that removes intermediate files after an experiment. |
