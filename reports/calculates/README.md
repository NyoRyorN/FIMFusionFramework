# reports/calculates — Cross-subject variable aggregation

Aggregates per-fold predictions into per-subject match rates, the input for the ablation
box/violin plots.

## Files

| File | Role |
|---|---|
| `calc_cross_subject_variables.py` | Concatenates fold-level `PredictTruelabel.csv` files (layout `<root>/<Item>/<Stratified_SeedXXX>/FoldX/`) and summarizes per-subject match rates into `CrossSubjectVariables` CSVs. |
