# integrateprocessingmodule
# attention mechanism class
# Translated note

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.attention.components import *


class Attention_part(nn.Module):
    """integrateprocessingclass

    attention mechanism class.
    specify by and,
    and evaluate run. by
    freeze/fix and attention weights output.

    :
    - spatial: spatial
    - temporal: temporal dimension
    - ST-sequential: spatial->temporal
    - ST-parallel: spatial/temporal column
    - ST-joint: spatial
    - ST-sequential-v2: (disabled)
    - ST-self-attention: (Transformer)(disabled)
    - ST-self-attention-sequential: (disabled)

    Args:
        input_dim (int): input features dimension count
        hidden_dim (int): hidden layer dimension count
        mode (str): specify
    """

    def __init__(self, input_dim, hidden_dim, mode):
        super(Attention_part, self).__init__()
        self.mode = mode
        # specifyaccording toattention mechanism initialize
        if mode == 'spatial':
            # spatial
            self.spatial_attn = Attention(in_dim=input_dim, hid_dim=hidden_dim)
        elif mode == 'temporal':
            # temporal dimension
            self.temporal_attn = Attention(
                in_dim=input_dim, hid_dim=hidden_dim)
        elif mode == "ST-joint":
            # spatial()
            self.st_joint_att = STJointAtt(in_dim=input_dim)
        # elif mode == "ST-joint-v2":
        # # spatial()
        #     self.st_joint_att = STJointAttV2(in_dim=input_dim)
        # elif mode == "ST-joint-ln":
        # # spatial(LayerNorm)
        #     self.st_joint_att = STJointAtt(in_dim=input_dim, mode="LN")
        # elif mode == "ST-sequential-v2":
        # # (V2)
        #     self.attention_v2 = AttentionV2(in_dim=input_dim)
        # elif mode == "ST-self-attention":
        # # Transformer
        #     self.self_attention = SelfAttention(in_dim=input_dim, num_heads=4)
        # elif mode == "ST-self-attention-sequential":
        # #
        #     self.self_attention = SelfAttentionSequential(
        #         in_dim=input_dim, num_heads=4)
        elif mode == "ST-sequential" or mode == "ST-parallel":
            # spatial/temporal or column
            self.spatial_attn = Attention(in_dim=input_dim, hid_dim=hidden_dim)
            self.temporal_attn = Attention(
                in_dim=input_dim, hid_dim=hidden_dim)

    def forward(self, x):
        N, C, T, V = x.size()
        weights = None
        feature = None

        if self.mode == 'spatial':
            # spatialprocessing
            # temporal dimension spatial extract
            feature = F.avg_pool2d(x, (T, 1))
            feature = feature.squeeze(dim=2).permute(0, 2, 1)
            # attention weights calculate
            e = self.spatial_attn(feature)
            beta = torch.softmax(e, dim=1)  # (N, V, 1)
            # weighted average aggregate features
            feature = torch.bmm(torch.transpose(beta, 1, 2),
                                feature).permute(0, 2, 1)
            feature = feature.unsqueeze(dim=3)
            weights = beta.permute(
                0, 2, 1).contiguous().expand(-1, T, V)  # (N, T, V)

        elif self.mode == 'temporal':
            # temporal dimensionprocessing
            # spatial temporal extract
            feature = x.permute(0, 1, 3, 2).contiguous()
            feature = F.avg_pool2d(feature, (V, 1))
            feature = feature.squeeze(dim=2).permute(0, 2, 1)
            # attention weights calculate
            e = self.temporal_attn(feature)
            alpha = torch.softmax(e, dim=1)
            # weighted average aggregate features
            feature = torch.bmm(torch.transpose(alpha, 1, 2),
                                feature).permute(0, 2, 1)
            feature = feature.unsqueeze(dim=3)
            weights = alpha.permute(
                0, 2, 1).contiguous().expand(-1, -1, V)  # (N, T, V)

        elif self.mode == 'ST-sequential':
            # spatial->temporal processing
            # first stage: spatial
            feature = x.permute(0, 2, 3, 1).contiguous()
            feature = feature.view(N * T, V, C)
            e = self.spatial_attn(feature)
            beta = torch.softmax(e, dim=1)
            # spatialintegrate
            feature = torch.bmm(torch.transpose(beta, 1, 2),
                                feature).permute(0, 2, 1)
            feature = feature.view(N, T, C, 1)
            beta = beta.squeeze(dim=2)

            # second stage: temporal
            feature = feature.squeeze(dim=3)
            e = self.temporal_attn(feature)
            alpha = torch.softmax(e, dim=1)
            # temporalintegrate
            feature = torch.bmm(torch.transpose(alpha, 1, 2),
                                feature).permute(0, 2, 1)
            feature = feature.unsqueeze(dim=3)
            alpha = alpha.squeeze(dim=2)

            # attention weights
            beta = beta.view(N, T, V)
            # (N, T) -> (N, T, 1) -> (N, T, V)
            alpha = alpha.unsqueeze(-1).expand(-1, -1, V)
            weights = alpha * beta  # (N, T, V) element-wise product by cross attention
            # weights = weights.permute(0, 2, 1).contiguous().view(N, V, T)

        # elif self.mode == "ST-parallel":
        # # spatial/temporal columnprocessing
        # # spatial calculate
        #     feature_spatial = F.avg_pool2d(
        #         x, (T, 1)).squeeze(dim=2).permute(0, 2, 1)
        #     e_spatial = self.spatial_attn(feature_spatial)
        #     # (N, V) -> (N, V, 1) -> (N, V, T) -> (N, T, V)
        #     beta = torch.softmax(
        #         e_spatial, dim=1).expand(-1, -1, T).permute(0, 2, 1)

        # # temporal calculate
        #     feature_temporal = x.permute(0, 1, 3, 2).contiguous()
        #     feature_temporal = F.avg_pool2d(
        #         feature_temporal, (V, 1)).squeeze(dim=2).permute(0, 2, 1)
        #     e_temporal = self.temporal_attn(feature_temporal)
        #     # (N, T) -> (N, T, 1) -> (N, T, V)
        #     alpha = torch.softmax(e_temporal, dim=1).expand(-1, -1, V)

        # # columnattention weights integrate
        # # (N, T, V) element-wise product by cross attention
        #     weights = alpha * beta
        # # apply
        #     # (N, T, V) -> (N, T, V, 1)
        #     weights = weights.unsqueeze(-1).contiguous()
        #     # (N, T, V, 1) -> (N, T * V, 1)
        #     weights = weights.view(N, T * V, 1).contiguous()
        #     # (N, C, T, V) -> (N, C, T * V)
        #     feature = x.view(N, C, T * V)
        #     # (N, C, T * V) * (N, T * V, 1) -> (N, C, 1) -> (N, C, 1, 1)
        #     feature = torch.bmm(feature, weights).unsqueeze(-1)

        #     weights = weights.squeeze(-1).view(N, T, V)
        #     # weights = weights.permute(0, 2, 1).contiguous()

        elif "ST-joint" in self.mode:
            # spatialprocessing
            feature, weights = self.st_joint_att(x)  # (N, C, 1, 1), (N, T, V)
            # Translated note
            feature = F.avg_pool2d(feature, feature.size()[2:])

        # elif self.mode == "ST-sequential-v2":
        # # processing
        #     feature, weights = self.attention_v2(x)
        # #
        #     feature = F.avg_pool2d(feature, feature.size()[2:])

        # elif self.mode == "ST-self-attention":
        # # processing
        #     feature, s_weights, t_weights = self.self_attention(x)
        # #
        #     feature = F.avg_pool2d(feature, feature.size()[2:])
        # # spatial/temporalattention weights return
        #     weights = s_weights, t_weights

        # elif self.mode == "ST-self-attention-sequential":
        # # processing
        #     feature, s_weights, t_weights = self.self_attention(x)
        # # spatial/temporalattention weights return
        #     weights = s_weights, t_weights

        else:
            raise ValueError('Mode not found')

        return feature, weights  # (N, C, 1, 1), (N, T, V)


class Attention_part_fusion(Attention_part):
    """fusionintegrateprocessingclass

    Attention_part class, fusion(fusion learning).
    in row, class in
    processing row, later stage fusionprocessing format output.

    :
    - row, shape preserve
    - attention weights apply
    - fusion as

    Args:
        input_dim (int): input features dimension count
        hidden_dim (int): hidden layer dimension count
        mode (str): specify
    """

    def __init__(self, input_dim, hidden_dim, mode):
        # parentclass initialize
        super().__init__(input_dim, hidden_dim, mode)

    def forward(self, x):
        N, C, T, V = x.size()
        weights = None
        feature = None
        if self.mode == 'ST-sequential':
            # fusion spatial->temporalprocessing
            # first stage: spatial(element-wise product)
            feature = x.permute(0, 2, 3, 1).contiguous()
            feature = feature.view(N * T, V, C)
            e = self.spatial_attn(feature)  # (N * T, V, C) -> (N * T, V, 1)
            beta = torch.softmax(e, dim=1)
            # spatial apply(element-wise product)
            feature = feature * beta.expand(-1, -1, C)
            feature = feature.view(N, T, V, C)
            beta = beta.squeeze(dim=2)  # (N * T, V)

            # second stage: temporal(element-wise product)
            feature = feature.permute(0, 2, 1, 3).contiguous()  # (N, V, T, C)
            feature = feature.view(N * V, T, C)
            e = self.temporal_attn(feature)  # (N * V, T, C) -> (N * V, T, 1)
            alpha = torch.softmax(e, dim=1)
            # temporal apply(element-wise product)
            feature = feature * alpha.expand(-1, -1, C)
            feature = feature.view(N, V, T, C)
            feature = feature.permute(0, 3, 2, 1).contiguous()  # (N, C, T, V)
            alpha = alpha.squeeze(dim=2)  # (N * V, T)

            # attention weights
            beta = beta.view(N, T, V)
            # (N, T) -> (N, T, 1) -> (N, T, V)
            alpha = alpha.view(N, V, T).permute(
                0, 2, 1).contiguous()  # (N, T, V)
            weights = alpha * beta  # (N, T, V) element-wise product by cross attention

        elif self.mode == "ST-parallel":
            # fusion spatial/temporalcolumnprocessing
            # spatial calculate
            feature_spatial = F.avg_pool2d(
                x, (T, 1)).squeeze(dim=2).permute(0, 2, 1)
            e_spatial = self.spatial_attn(feature_spatial)
            # (N, V) -> (N, V, 1) -> (N, V, T) -> (N, T, V)
            beta = torch.softmax(
                e_spatial, dim=1).expand(-1, -1, T).permute(0, 2, 1)

            # temporal calculate
            feature_temporal = x.permute(0, 1, 3, 2).contiguous()
            feature_temporal = F.avg_pool2d(
                feature_temporal, (V, 1)).squeeze(dim=2).permute(0, 2, 1)
            e_temporal = self.temporal_attn(feature_temporal)
            # (N, T) -> (N, T, 1) -> (N, T, V)
            alpha = torch.softmax(e_temporal, dim=1).expand(-1, -1, V)

            # columnattention weights integrate
            # (N, T, V) element-wise product by cross attention
            weights = alpha * beta
            # all apply(fusion)
            # (N, T, V) -> (N, T, V, 1)
            weights = weights.unsqueeze(-1).contiguous()
            # (N, T, V, 1) -> (N, 1, T, V)
            weights = weights.permute(0, 3, 1, 2).contiguous()
            # attention weights allchannels apply
            feature = x * weights.expand(-1, C, -1, -1)

            weights = weights.squeeze(-1).view(N, T, V)
            # weights = weights.permute(0, 2, 1).contiguous()

        elif "ST-joint" in self.mode:
            # spatialprocessing(fusion)
            feature, weights = self.st_joint_att(x)

        # elif self.mode == "ST-sequential-v2":
        # # processing(fusion)
        #     feature, weights = self.attention_v2(x)

        # elif self.mode == "ST-self-attention":
        # # processing(fusion)
        #     feature, s_weights, t_weights = self.self_attention(x)
        # # spatial/temporalattention weights return
        #     weights = s_weights, t_weights

        else:
            raise ValueError('Mode not found')

        return feature, weights
