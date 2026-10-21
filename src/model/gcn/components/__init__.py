"""
(GCN)module

row for ST-GCN structure
.

:
    Graph: graph structure and adjacency matrixgenerate
    STGCN_Block: spatial(spatialGCN + temporalTCN)

combination and,
and temporal build.
"""

from src.model.gcn.components.graph import Graph
from src.model.gcn.components.st_gcn_block import STGCN_Block
