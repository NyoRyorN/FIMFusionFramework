# reports — Result aggregation and visualization

Post-hoc analysis and figure generation over experiment outputs in `results/`. These scripts
are run manually after experiments finish; they do not train models. Each subfolder carries
its own `README.md` / `README_ja.md`.

## Top-level scripts

| File | Role |
|---|---|
| `combine_results.py` | Combines result CSVs from several file paths (interactive prompt) into one table. |
| `ablation_process.py` | Drives the ablation post-processing (aggregation → plotting) over a set of experiment directories. |

## Subfolders

| Subfolder | Role |
|---|---|
| [`calculates/`](calculates/) | Cross-subject variable aggregation from per-fold `PredictTruelabel.csv`. |
| [`ablation_boxplot/`](ablation_boxplot/) | Ablation box/violin plots and per-class plots from cross-subject summaries. |
| [`check_fim_dist/`](check_fim_dist/) | FIM-score distribution checks/outputs. |
| [`skeleton_plots/`](skeleton_plots/) | Skeleton rendering and Attention/GradCAM visualization as videos, subplots, and distribution stats. |
