# reports/ablation_boxplot — Ablation box/violin plots

Generates the ablation comparison figures from the cross-subject summaries produced by
`reports/calculates/`. Each data point is one subject's match rate. Reported standard
deviations use sample standard deviation (`ddof=1`).

## Files

| File | Role |
|---|---|
| `ablation_cross_subject_variables.py` | Generates condition-wise violin + box plots from `CrossSubjectVariables` CSVs (one point per subject). |
| `ablation_plots_each_class.py` | Generates per-class ablation plots. |
| `ablation_sum_plots.py` | Generates summary ablation plots aggregated across conditions. |
