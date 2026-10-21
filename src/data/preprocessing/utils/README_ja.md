# src/data/preprocessing/utils — 数値前処理ヘルパ

`../make_concise_data.py`（特徴量計算）やデータ品質分析から呼ばれる再利用可能な数値処理。
フレームごとの特徴量計算と、データセットの品質チェック/分布出力の 2 グループに分かれる。

## ファイル

| ファイル | 役割 |
|---|---|
| `calc_velocity.py` | 時系列の位置から関節速度を計算する。 |
| `calc_angle_length.py` | 親子グラフ構造から関節角度と骨長を計算する。 |
| `fim_dist_output.py` | FIM（Functional Independence Measure）スコア分布を集計・分析する。 |
| `plot_fim_dist.py` | 集計した FIM スコア分布を可視化する。 |
| `missing_motion_output.py` | FIM スコアごとの欠損動作（被験者, タスク）分類の分布を出力する。 |
| `time_shortage_check.py` | 時間方向に短すぎる系列をチェック/分析する。 |
