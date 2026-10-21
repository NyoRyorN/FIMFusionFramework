# src/data — 骨格データパイプライン

生の計測データからモデル入力テンソルまでの全経路。3 つのステージがそれぞれサブフォルダ
に分かれる。生データの **preprocessing**（整理 → 時間補完 → フィルタ/特徴量 → 骨格点
削減）、HDF5 **dataset** 生成、学習時にそれらを遅延読み込みする PyTorch **feeder**。

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`preprocessing/`](preprocessing/) | 生データ前処理パイプライン。ファイル/ディレクトリ整理・時間補完・フィルタ + 特徴量計算（速度・角度・骨長）・骨格点削減出力。 |
| [`dataset/`](dataset/) | `train.h5` / `test.h5` の HDF5 データセット生成（単一アクション・複数アクション）。StratifiedGroupKFold で分割。 |
| [`feeder/`](feeder/) | HDF5 遅延読み込みとデータ拡張を行う PyTorch `Dataset`/`DataLoader`、および実験ごとの結果管理。 |
