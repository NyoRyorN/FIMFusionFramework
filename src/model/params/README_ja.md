# src/model/params — モデルパラメータヘルパ

config からモデルの結線パラメータを導出する自作ヘルパ。Fusion モデルが
（`training/run.py` 経由で）複数アクション入力を正しく分割するために使う。

## ファイル

| ファイル | 役割 |
|---|---|
| `separate_channels.py` | GCN / fusion 構造向けに、アクションごとの入力チャンネルの分割方法を計算する（`separate_channels`）。 |
