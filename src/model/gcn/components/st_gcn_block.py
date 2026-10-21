"""
spatial(ST-GCN)module

row for ST-GCN structureelement.
spatial(GCN) and temporal(TCN) combination,
spatial and temporal.

Key features:
- spatial by
- temporal by columnextract
- by
- batch normalization and by

References:
    Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
    https://arxiv.org/abs/1801.07455
"""

import torch.nn as nn
import torch.nn.functional as F
from src.model.gcn.components.sg_conv import SpatialGraphicalConv

class STGCN_Block(nn.Module):
    r"""spatial

    input graph for spatial and temporal apply
    ST-GCN structure. processing run:

    1. spatial(GCN): spatial
    2. temporal(TCN): temporal extract
    3.:input and output by
    4. ReLU:

    Args:
        in_channels (int): input sequence number of channels
        out_channels (int): generatenumber of channels
        t_kernel_size (int): temporal()
        s_kernel_size (int): spatial(adjacency matrix)
        stride (int, optional): temporal (default: 1)
        dropout (int, optional): output (default: 0)
        residual (bool, optional): apply(default: True)

    Shape:
        - Input[0]: input graph:math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: inputgraph adjacency matrix:math:`(K, V, V)` format
        - Output[0]: output graph:math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: graph adjacency matrix:math:`(K, V, V)` format(input and)

        Here:
            :math:`N` batch
            :math:`K` spatial(s_kernel_size and)
            :math:`T_{in}/T_{out}` input/output sequence
            :math:`V` number of graph nodes(joints)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 t_kernel_size,
                 s_kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        """ST-GCN initialize

        spatial, temporal,, ST
        combinationspatialextract build.
        """
        super().__init__()

        assert t_kernel_size % 2 == 1, "Temporal kernel size must be odd"
        padding = ((t_kernel_size - 1) // 2, 0)  # temporal dimension calculate

        # spatial(GCN)
        # Translated note
        self.gcn = SpatialGraphicalConv(in_channels,
                                        out_channels,
                                        s_kernel_size)

        # temporal(TCN)
        # column extract
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),      # batch normalization
            nn.ReLU(inplace=True),             # ReLU
            nn.Conv2d(                         # temporal1D
                out_channels,
                out_channels,
                (t_kernel_size, 1),            # (temporal, =1)
                (stride, 1),                   # (temporal, =1)
                padding,                       # temporal
            ),
            nn.BatchNorm2d(out_channels),      # batch normalization
            nn.Dropout(dropout, inplace=True),  # Translated note
        )

        # set
        if not residual:
            # disabled(return)
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            # output channels and case
            self.residual = lambda x: x
        else:
            # number of channels or case 1x1 adjust
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,           # 1x1
                          stride=(stride, 1)),     # temporal adjust
                nn.BatchNorm2d(out_channels),      # batch normalization
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """ST-GCN path

        input for spatial, temporal, apply,
        spatial extract.

        Args:
            x (torch.Tensor): input graph (N, C_in, T_in, V)
            A (torch.Tensor): graph adjacency matrix (K, V, V)

        Returns:
            tuple:
                - torch.Tensor: output (N, C_out, T_out, V)
                - torch.Tensor: adjacency matrix (K, V, V) - input and
        """
        # input save
        res = self.residual(x)

        # spatial by
        x, A = self.gcn(x, A)

        # temporal by column extract
        x = self.tcn(x)

        x = x + res  # Translated note
        return self.relu(x), A
