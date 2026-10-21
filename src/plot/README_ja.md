# src/plot — 評価指標計算と図生成

fold ごとの予測を評価指標と図に変換する。学習パイプライン内および結果後処理
（`src/results_process.py`）から利用される。出力する標準偏差は標本標準偏差（`ddof=1`）を使う。

## ファイル

| ファイル | 役割 |
|---|---|
| `calc_indicator.py` | 2 クラス / fusion 向けの評価指標（accuracy と関連指標）を計算する。 |
| `box_plots_each_class.py` | クラス別 accuracy のボックスプロットを生成する。 |
| `summary_outputs.py` | 各実験の `Accuracies.csv` を収集し、統合した `Summary.csv` を出力する。 |
