# src/model/gcn — 空間時間グラフ畳み込みサブネット（ST-GCN）

エンコーダの空間段。骨格関節グラフ上で空間時間グラフ畳み込みを行い、適応的/学習可能な
グラフ構造とエッジ重要度重み付けをオプションで持つ。構成要素は `components/` に置く。

**元論文:** ST-GCN — Yan, Xiong, Lin, "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition," AAAI 2018.

## ファイル

| ファイル | 役割 |
|---|---|
| `stgcn_part.py` | `ST_GCN_Part`。関節グラフ上に ST-GCN ブロックを積層。適応的グラフ学習とエッジ重要度重み付けに対応し、後段の attention 段と連携する。 |

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`components/`](components/) | ST-GCN 構成要素。グラフ構造・空間グラフ畳み込み・ST-GCN ブロック。 |
