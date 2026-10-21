"""
(GCN)module

row for spatial(ST-GCN).
human skeleton joints graph structure as, spatial/temporal.

:
    stgcn_part: allST-GCNintegratemodule
    components: ST-GCN structure
        - Graph: graph structure
        - SpatialGraphicalConv: spatial
        - STGCN_Block: spatial

:
    - by graph structureoptimization
    - edges by
    - batch normalization
    - fusion and integrate

References:
    Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
    https://arxiv.org/abs/1801.07455
"""
