# src/model/gcn — Spatial-temporal graph convolution subnet (ST-GCN)

The spatial stage of the encoder. Applies spatial-temporal graph convolution over the
skeleton joint graph, with optional adaptive/learnable graph structure and edge-importance
weighting. Building blocks live in `components/`.

**Source paper:** ST-GCN — Yan, Xiong, Lin, "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition," AAAI 2018.

## Files

| File | Role |
|---|---|
| `stgcn_part.py` | `ST_GCN_Part`: stacks ST-GCN blocks over the joint graph; supports adaptive graph learning and edge-importance weighting; integrates with the downstream attention stage. |

## Subfolders

| Subfolder | Role |
|---|---|
| [`components/`](components/) | ST-GCN building blocks: graph structure, spatial graph convolution, and the ST-GCN block. |
