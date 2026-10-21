# spatial(ST-GCN)integratemodule
# for allST-GCN
# graph structure,, integrate

import torch
import torch.nn as nn
import itertools
from src.model.gcn.components import Graph, STGCN_Block
from typing import Union
import numpy as np


class ST_GCN_Part(nn.Module):
    r"""spatial(Spatial Temporal Graph Convolutional Networks)

    allST-GCN.
    human skeleton joints graph structure and column, extract.

    :
    - graph structure(Graph Adaptive Learning)
    - edges(Edge Importance Weighting)
    - batch normalization by
    - fusion to

    Args:
        config (dict): model settings config dictionary
        in_channels (int): input data number of channels(default: 3)
        use_in_fusion (bool): fusion flag(default: False)

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
          N: batch, T_in: inputcolumn, V_in: joints
        - Output: :math:`(N, out_channels, T_{out}, V_{out})`
          format output
    """

    def __init__(self, config, in_channels=3, use_in_fusion=False, A: Union[None, torch.Tensor] = None):
        """ST-GCN initialize

        graph structure build, ST-GCN structure, and,
        set row.

        Args:
            config (dict): set
                - model: set
                - training: training settings()
            in_channels (int): input data number of channels(dimension count)
            use_in_fusion (bool): fusion True
        """
        super().__init__()

        # and frameset get
        device = config["device"]

        # ST-GCNmodel settings extract
        model_attr = config["model"]["st-gcn"]
        t_kernel_size = model_attr["t_kernel_size"]  # temporal
        dropout = model_attr["dropout"]  # Translated note
        # edges
        edge_importance_weighting = model_attr["edge_importance_weighting"]
        chn_list = model_attr["output_channel_list"]  # each output channelslist

        # graph structureset extract
        graph_attr = model_attr["graph"]
        hop_size = graph_attr["hop_size"]  # graph hop size
        strategy = graph_attr["strategy"]  # graph construction strategy
        dilation = graph_attr["dilation"]  # graph dilation setting
        norm_mode = graph_attr["norm_mode"]  # Translated note

        # graph structure build
        # human skeleton jointsconnection relationships, adjacency matrix A generate
        if A is None:
            self.graph = Graph(max_hop=hop_size,
                               norm_mode=norm_mode,
                               strategy=strategy,
                               dilation=dilation)
            self.A = torch.tensor(self.graph.A,
                                  dtype=torch.float32,
                                  requires_grad=False,
                                  device=device)
        else:
            self.A = A.to(device)

        # spatial(adjacency matrix dimension count)
        s_kernel_size = self.A.size(0)

        # input normalization: batch normalization over channels x joints.
        # Disabled when this trunk runs after a fusion layer, because the
        # per-action encoders have already normalized their inputs.
        self.use_data_bn = not use_in_fusion
        if self.use_data_bn:
            self.data_bn = nn.BatchNorm1d(
                in_channels * self.A.size(1), device=device)

        # ST-GCN build
        # multiple ST-GCN
        # ST-GCN build
        # multiple ST-GCN
        self.st_gcn_networks = nn.ModuleList([
            # 1: inputchannels from to convert
            STGCN_Block(in_channels=in_channels,
                        out_channels=chn_list[0],
                        stride=2 if use_in_fusion else 1,  # fusion 2
                        t_kernel_size=t_kernel_size,
                        s_kernel_size=s_kernel_size,
                        dropout=dropout,
                        residual=use_in_fusion)  # fusion
        ])

        # build: number of channels set
        for i in range(1, len(chn_list)):
            stride = 1
            # number of channels adjust
            if chn_list[i - 1] < chn_list[i]:
                if i > 1 and chn_list[i - 2] == chn_list[i - 1]:
                    stride = 2  # temporal

            self.st_gcn_networks.append(
                STGCN_Block(in_channels=chn_list[i - 1],
                            out_channels=chn_list[i],
                            stride=stride,
                            t_kernel_size=t_kernel_size,
                            s_kernel_size=s_kernel_size,
                            dropout=dropout)
            )

        # edges initialize
        # eachedges trainable as append
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            # case all1 freeze/fix
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        """ST-GCN path

        for and temporal apply,
        extract.

        Args:
            x (torch.Tensor): input
                Shape: (N, C, T, V) - batch, channels, temporal, joints

        Returns:
            torch.Tensor: extract
                Shape: (N, C_out, T_out, V) - processingafter
        """
        # apply
        N, C, T, V = x.size()
        if self.use_data_bn:
            # batch normalization: (N, C, T, V) -> (N, V*C, T) -> ->
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.view(N, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, V, C, T)
            x = x.permute(0, 2, 3, 1).contiguous()

        x = x.view(N, C, T, V)

        # ST-GCN by extract
        for l, gcn, importance in zip(itertools.count(), self.st_gcn_networks, self.edge_importance):
            # adjacency matrix
            x, _ = gcn(x, self.A * importance)

        # shapeadjust
        _, c, t, v = x.size()
        feature = x.view(N, c, t, v).permute(0, 1, 2, 3)

        return feature  # (N, C_out, T_out, V)

    def return_weights(self) -> Union[np.ndarray, None]:
        return self.edge_importance.detach().cpu().numpy() if isinstance(self.edge_importance, nn.ParameterList) else None
