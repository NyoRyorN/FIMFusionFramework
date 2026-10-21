# src/model/rnn — 再帰サブネット

エンコーダの時間モデリング段。相互に差し替え可能な 2 つの再帰サブネットを提供する。
どれを使うかは `config/model_ablation/` で選ぶ `model.name` 変種に依存する。

## ファイル

| ファイル | 役割 | 出典 |
|---|---|---|
| `agc_lstm.py` | `AGCLSTM` — Attention-Enhanced Graph Convolutional LSTM。グラフ畳み込みの出力を LSTM に入力し、隠れ状態を空間 attention でゲートする。 | AGC-LSTM — Si, Chen, Wang, Wang, Tan, "An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition," CVPR 2019. |
| `lstm_part.py` | `LSTM_part` — フレームごとのグラフ特徴に対する双方向 LSTM ラッパ。 | LSTM（Hochreiter & Schmidhuber, 1997）。自作ラッパ。 |
