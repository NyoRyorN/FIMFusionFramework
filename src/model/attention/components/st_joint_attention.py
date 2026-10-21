# spatialattention mechanismmodule
# temporal/spatial processing
# columnframe and joints integrate,

from torch import nn
import torch
import torch.nn.functional as F


class STJointAtt(nn.Module):
    r"""spatial(Spatial-Temporal Joint Attention)module

    , temporal dimension and spatial integrate processing.
    columnframe(temporal) and joints(spatial),
    for attention weights calculate from cross attention apply.

    Key characteristics:
    - temporal/spatial integrateprocessing
    - by cross attentiongenerate
    - LayerNormalization
    - channels

    Args:
        in_dim (int): input data number of channels
        mode (str, optional): ("LN" LayerNorm)

    Shape:
        - Input: :math:`(N, C, T, V)`
          N: batch, C: number of channels, T: temporalframe, V: joints
        - Output: :math:`(N, C, T, V)`, weights: :math:`(N, T, V)`
          apply and attention weights
    """

    def __init__(self, in_dim):
        super(STJointAtt, self).__init__()
        # temporal/spatial to linear transformation
        self.attention_stitch = nn.Linear(in_dim, in_dim)
        # batch normalization(LayerNorm delete)
        self.stitch_bn = nn.BatchNorm1d(in_dim)
        # HardSwishfunction
        self.hardswish = nn.Hardswish()
        # spatial(joints) for attention weightscalculate
        self.attention_spatial = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        # temporal dimension(frame) for attention weightscalculate
        self.attention_temporal = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, feature):
        N, C, T, V = feature.size()

        # temporal dimension by spatialextract
        # embeddings_temporal: (mini_batch_size, T, z_dim)
        # (N, C, T, V) -> (N, C, 1, V) -> (N, C, V) -> (N, V, C)
        embeddings_spatial = F.avg_pool2d(feature, (T, 1)).squeeze(dim=2).permute(
            0, 2, 1)  # (N, C, T, V) -> (N, C, 1, V)-> (N, C, V)  -> (N, V, C)

        # spatial by temporalextract
        # embeddings_spatial: (mini_batch_size, V, z_dim)
        # (N, C, T, V) -> (N, C, T, 1) -> (N, C, T) -> (N, T, C)
        embeddings_temporal = F.avg_pool2d(feature, (1, V)).squeeze(dim=3).permute(
            0, 2, 1)  # (N, C, T, V) -> (N, C, T, 1) -> (N, C, T) -> (N, T, C)

        # temporal and spatial:(N, T, C) + (N, V, C) -> (N, T + V, C)
        embeddings = torch.cat(
            [embeddings_temporal, embeddings_spatial], dim=1)
        # linear transformation apply
        embeddings = self.attention_stitch(embeddings)  # (N, T + V, C)
        # BatchNorm apply
        embeddings = self.stitch_bn(embeddings.permute(0, 2, 1)).permute(
            0, 2, 1)  # (N, T + V, C) -> (N, C, T + V) -> (N, T + V, C)
        # HardSwishfunction apply
        embeddings = self.hardswish(embeddings)

        # spatialattention weights calculate and
        embeddings_spatial = self.attention_spatial(embeddings[:, -V:, :])
        # (N, V, C) -> (N, V, 1, C) -> (N, C, 1, V) -> (N * C, 1, V)
        embeddings_spatial = embeddings_spatial.unsqueeze(2).permute(0, 3, 2, 1).reshape(
            N * C, 1, V)  # (N, V, C) -> (N, V, 1, C) -> (N, C, 1, V) -> (N * C , 1, V)

        # temporalattention weights calculate and
        embeddings_temporal = self.attention_temporal(embeddings[:, :T, :])
        # (N, T, C) -> (N, T, 1, C) -> (N, C, T, 1) -> (N * C, T, 1)
        embeddings_temporal = embeddings_temporal.unsqueeze(2).permute(0, 3, 2, 1).reshape(
            N * C, T, 1)  # (N, T, C) -> (N, T, 1, C) -> (N, C, T, 1) -> (N * C , T, 1)

        # batchrowcolumn by cross attentioncalculate
        # (N * C, T, 1) × (N * C, 1, V) -> (N * C, T, V)
        weights = torch.bmm(embeddings_temporal, embeddings_spatial)
        weights = weights.view(N, C, T, V)

        # attention weights apply
        feature = torch.mul(feature, weights)
        feature = feature.view(N, C, T, V)

        # channels attention weights generate
        # weights = weights.contiguous().permute(0, 3, 2, 1).view(N, V, T, C)
        # weights = F.avg_pool2d(weights, (1, C)).squeeze(dim=3).view(N, V, T)
        weights = weights.mean(dim=1)  # (N, T, V)

        return feature, weights


# class STJointAttV2(nn.Module):
# r"""spatialV2(Spatial-Temporal Joint Attention V2)module

# STJointAtt, attention weightscalculate.
# each(temporal/spatial) attention weights 1 reduce from,
# calculate and optimization.

# points:
# - each attention weights 1 reduce
# - by cross attention
# - apply
# - BatchNormalizationdelete by

#     Args:
# in_dim (int): input data number of channels
# mode (str, optional): (current)

#     Shape:
#         - Input: :math:`(N, C, T, V)`
# N: batch, C: number of channels, T: temporalframe, V: joints
#         - Output: :math:`(N, C, T, V)`, weights: :math:`(N, T, V)`
# apply and attention weights
#     """

#     def __init__(self, in_dim, mode=None):
#         super(STJointAttV2, self).__init__()
#         self.mode = mode
# # temporal/spatial to linear transformation
#         self.attention_stitch = nn.Linear(in_dim, in_dim)
# # V1 and (for)
#         self.stitch_bn = nn.BatchNorm1d(in_dim)
# # spatial attention weights(1output)
#         self.attention_spatial = nn.Sequential(
#             nn.Linear(in_dim, 1),
#             nn.Sigmoid()
#         )
# # temporal dimension attention weights(1output)
#         self.attention_temporal = nn.Sequential(
#             nn.Linear(in_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, feature):
#         N, C, T, V = feature.size()

# # V1 and temporal/spatialextract
#         # embeddings_temporal: (mini_batch_size, T, z_dim)
# # temporal dimension by spatialextract:(N, C, T, V) -> (N, V, C)
#         embeddings_spatial = F.avg_pool2d(feature, (T, 1)).squeeze(dim=2).permute(
#             0, 2, 1)  # (N, C, T, V) -> (N, C, 1, V)-> (N, C, V)  -> (N, V, C)

# # spatial by temporalextract:(N, C, T, V) -> (N, T, C)
#         # embeddings_spatial: (mini_batch_size, V, z_dim)
#         embeddings_temporal = F.avg_pool2d(feature, (1, V)).squeeze(dim=3).permute(
#             0, 2, 1)  # (N, C, T, V) -> (N, C, T, 1) -> (N, C, T) -> (N, T, C)

# # temporal and spatial:(N, T, C) + (N, V, C) -> (N, T + V, C)
#         embeddings = torch.cat(
#             [embeddings_temporal, embeddings_spatial], dim=1)
# # linear transformation apply
#         embeddings = self.attention_stitch(embeddings)  # (N, T + V, C)

# # 1attention weights calculate and
# # spatialattention weights:(N, V, 1) -> (N, V, T)
#         embeddings_spatial = self.attention_spatial(
#             embeddings[:, -V:, :]).expand(-1, -1, T)  # (N, V, 1) -> (N, V, T)
# # temporalattention weights:(N, T, 1) -> (N, T, V)
#         embeddings_temporal = self.attention_temporal(
#             embeddings[:, :T, :]).expand(-1, -1, V)  # (N, T, 1) -> (N, T, V)

# # by cross attentioncalculate:(N, T, V) × (N, V, T) -> (N, T, V)
#         weights = embeddings_spatial.permute(
#             0, 2, 1) * embeddings_temporal  # (N, T, V)

# # attention weights apply
# # channels:(N, T, V) -> (N, C, T, V)
#         feature = feature * weights.unsqueeze(1).expand(-1, C, -1, -1)
#         feature = feature.view(N, C, T, V)

#         return feature, weights
