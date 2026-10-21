"""
spatial(Spatial Graph Convolutional)module

row, joints graph structure for
applystructureelement.adjacency matrix by,
eachjoints and.

module ST-GCN based on.

References:
    Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
    https://arxiv.org/abs/1801.07455
"""

# The based unit of graph convolutional networks.
# This is the original implementation for ST-GCN papers.

import torch
import torch.nn as nn


class SpatialGraphicalConv(nn.Module):
    r""" applymodule

    joints for spatial.
    adjacency matrix graph structurebased on, eachjoints
    information and combination.

    :
    1. input features for 2D apply
    2. generate
    3. each adjacency matrix and
    4. by

    Args:
        in_channels (int): input sequence number of channels
        out_channels (int): generatenumber of channels
        kernel_size (int): (adjacency matrix)
        t_kernel_size (int): temporal
        t_stride (int, optional): temporal (default: 1)
        t_padding (int, optional): temporal zero padding(default: 0)
        t_dilation (int, optional): temporalelement (default: 1)
        bias (bool, optional): trainable append(default: True)

    Shape:
        - Input[0]: input graph:math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: inputgraph adjacency matrix:math:`(K, V, V)` format
        - Output[0]: output graph:math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: outputgraph adjacency matrix:math:`(K, V, V)` format(input and)

        Here:
            :math:`N` batch
            :math:`K` spatial(kernel_size and)
            :math:`T_{in}/T_{out}` input/output sequence
            :math:`V` number of graph nodes(joints)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        """spatial initialize

        2D initialize, graph structure extract row.

        Args:
            each classdocstring
        """
        super().__init__()

        self.kernel_size = kernel_size
        # 2D:(temporal, joints=1)
        # output channels kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        """spatial path

        input features for 2D apply, adjacency matrix
        graph structure based on run.

        Args:
            x (torch.Tensor): input graph (N, C_in, T, V)
            A (torch.Tensor): graph adjacency matrix (K, V, V)

        Returns:
            tuple:
                - torch.Tensor: output (N, C_out, T, V)
                - torch.Tensor: adjacency matrix (K, V, V) - input and
        """
        assert A.size(0) == self.kernel_size, "Adjacency matrix size does not match the kernel size"

        # 2D by convert
        x = self.conv(x)

        # shape structure: (N, K*C, T, V) -> (N, K, C, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)

        # by
        # eachk for adjacency matrix A[k] from
        # 'nkctv,kvw->nctw': (batch,, channels, temporal,) × (,,)
        # -> (batch, channels, temporal,)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
