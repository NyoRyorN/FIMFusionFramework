# src/plot — Metric calculation and figure generation

Turns per-fold predictions into evaluation metrics and figures. Used inside the training
pipeline and by the result post-processing (`src/results_process.py`). Reported standard
deviations use sample standard deviation (`ddof=1`).

## Files

| File | Role |
|---|---|
| `calc_indicator.py` | Computes evaluation indicators (accuracy and related metrics) for the 2-class / fusion case. |
| `box_plots_each_class.py` | Generates per-class accuracy box plots. |
| `summary_outputs.py` | Collects each experiment's `Accuracies.csv` and writes a combined `Summary.csv`. |
