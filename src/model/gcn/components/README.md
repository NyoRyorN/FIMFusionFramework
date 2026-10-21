# src/model/gcn/components — ST-GCN building blocks

The pieces `stgcn_part.py` assembles into the spatial-temporal graph convolution stage.

**Source paper:** ST-GCN — Yan, Xiong, Lin, "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition," AAAI 2018.

## Files

| File | Role |
|---|---|
| `graph.py` | Builds the skeleton graph structure (joint connectivity / adjacency partitions) used by the graph convolutions. |
| `sg_conv.py` | Spatial graph convolution over the joint graph. |
| `st_gcn_block.py` | One ST-GCN block: spatial graph convolution followed by temporal convolution, with residual connection. |
