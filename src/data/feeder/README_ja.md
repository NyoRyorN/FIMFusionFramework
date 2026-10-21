# src/data/feeder — PyTorch Dataset / DataLoader と結果管理

実行時のデータ層。HDF5 データセットから骨格系列を遅延読み込みしてデータ拡張を適用し、
`(系列, FIM ラベル)` サンプルを返す PyTorch `Dataset` と、学習ループが fold ごとの指標を
集約するために使う結果集計ヘルパを含む。

## ファイル

| ファイル | 役割 |
|---|---|
| `feeder.py` | HDF5 遅延読み込み・データ拡張・サンプルメタ情報を持つ PyTorch `Dataset`。`DataLoader` へ供給する。 |
| `results.py` | 結果管理クラス。クラス別/balanced accuracy の集約、予測/真ラベルの保存、fold・seed をまたぐ平均/標準偏差の計算。 |
| `tools.py` | feeder 用の前処理/変換ヘルパ（チャンネル選択・時間方向の切り出し・3D 座標変換）。 |
