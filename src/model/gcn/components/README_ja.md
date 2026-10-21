# src/model/gcn/components — ST-GCN 構成要素

`stgcn_part.py` が空間時間グラフ畳み込み段に組み立てる部品群。

**元論文:** ST-GCN — Yan, Xiong, Lin, "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition," AAAI 2018.

## ファイル

| ファイル | 役割 |
|---|---|
| `graph.py` | グラフ畳み込みで使う骨格グラフ構造（関節接続 / 隣接パーティション）を構築する。 |
| `sg_conv.py` | 関節グラフ上の空間グラフ畳み込み。 |
| `st_gcn_block.py` | 1 つの ST-GCN ブロック。空間グラフ畳み込み → 時間畳み込み、残差接続付き。 |
