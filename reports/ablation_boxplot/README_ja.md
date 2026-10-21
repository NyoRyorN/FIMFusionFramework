# reports/ablation_boxplot — アブレーション box/violin プロット

`reports/calculates/` が生成した被験者横断サマリから、アブレーション比較図を生成する。
各データ点は 1 被験者の一致率に対応する。出力する標準偏差は標本標準偏差（`ddof=1`）を使う。

## ファイル

| ファイル | 役割 |
|---|---|
| `ablation_cross_subject_variables.py` | `CrossSubjectVariables` CSV から条件別の violin + box プロットを生成する（1 点 = 1 被験者）。 |
| `ablation_plots_each_class.py` | クラス別のアブレーションプロットを生成する。 |
| `ablation_sum_plots.py` | 条件を横断的に集約したサマリアブレーションプロットを生成する。 |
