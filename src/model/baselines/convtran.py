"""
convtran.py - ConvTran baseline classifier

PyTorch port following the official ConvTran implementation
(https://github.com/Navidfoumani/ConvTran, the paper's original code),
also used as a baseline in DeepRehabPile: a disjoint CNN embedding
(temporal conv then spatial conv over all variables), time Absolute
Position Encoding (tAPE), one transformer block with efficient Relative
Position Encoding attention (eRPE, Attention_Rel_Scl), a feed-forward
block, global average pooling over time and a classification head.

As in the official code, the position encoding feeds the attention input
only, while the attention residual keeps the pre-encoding embedding (this
differs from the DeepRehabPile Keras port, which reuses the encoded
tensor for the residual). A softmax is appended to the head so the output
is a probability distribution compatible with the project NLLLoss loop.

The skeleton input (N, C, T, V) is flattened to (N, T, C*V) before the
embedding convolutions.

Reference:
    Foumani et al. "Improving position encoding of transformers for
    multivariate time series classification." Data Mining and Knowledge
    Discovery 38.1 (2024): 22-48. arXiv:2305.16642.

Config: config/model_ablation/convtran.yaml (model.convtran block)
"""

import math

import torch
import torch.nn as nn


class TAPE(nn.Module):
    """time Absolute Position Encoding (tAPE).

    Sinusoidal encoding whose frequency argument is rescaled by
    d_model / max_len so that the embedding distance stays distance-aware
    for short sequences and low embedding dimensions (see the ConvTran
    paper, Sec. 4.1).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term * (d_model / max_len))
        pe[:, 1::2] = torch.cos(position * div_term * (d_model / max_len))
        self.register_buffer("pe", scale_factor * pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttentionRelScl(nn.Module):
    """Multi-head attention with efficient Relative Position Encoding (eRPE).

    A learnable relative bias table indexed by key-query distance is added
    to the attention matrix. Following the original implementation, the
    bias is added AFTER the softmax, and the output projection is a
    LayerNorm instead of a linear layer.
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5

        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(
            torch.zeros(2 * seq_len - 1, num_heads))
        coords = torch.arange(seq_len)
        # distance (i - j) shifted to [0, 2*seq_len-2] to index the table
        relative_index = coords[None, :] - coords[:, None] + seq_len - 1
        self.register_buffer(
            "relative_index", relative_index.flatten().unsqueeze(1))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        k = self.key(x).reshape(
            batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(
            batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(
            batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.num_heads))  # (T*T, H)
        relative_bias = relative_bias.reshape(
            self.seq_len, self.seq_len, self.num_heads
        ).permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)
        attn = attn + relative_bias

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out)


class ConvTran(nn.Module):
    """ConvTran baseline classifier.

    Args:
        config: experiment config dictionary; reads the model.convtran
            block (kernel_size, factor_filters, emb_size, num_heads,
            dim_ff, dropout, epsilon, n_joints) plus attributes / classes
            and preprocess.frame_length (sequence length for the position
            encodings)
        use_fc: if False, return GAP features (N, emb_size) instead of
            class probabilities (for late feature fusion)
        num_actions: input channel multiplier for early fusion; the input
            is expected as (N, num_actions * C, T, V)

    Shape:
        - Input: (N, C, T, V)
        - Output: (N, num_class) softmax probabilities, or (N, emb_size)
          features when use_fc=False
    """

    def __init__(self, config, use_fc=True, num_actions=1):
        super().__init__()

        model_attr = config["model"].get("convtran", {})
        kernel_size = model_attr.get("kernel_size", 8)
        factor_filters = model_attr.get("factor_filters", 4)
        emb_size = model_attr.get("emb_size", 16)
        num_heads = model_attr.get("num_heads", 8)
        dim_ff = model_attr.get("dim_ff", 256)
        dropout = model_attr.get("dropout", 0.01)
        epsilon = model_attr.get("epsilon", 1e-5)
        n_joints = model_attr.get("n_joints", 17)

        num_classes = len(config["classes"])
        seq_len = config["preprocess"]["frame_length"]
        in_features = len(config["attributes"]) * 3 * n_joints * num_actions

        # disjoint CNN embedding: temporal conv over T, then spatial conv
        # covering all input variables at once
        self.embed = nn.Sequential(
            nn.Conv2d(1, emb_size * factor_filters,
                      kernel_size=(kernel_size, 1), padding="same"),
            nn.BatchNorm2d(emb_size * factor_filters),
            nn.GELU(),
            nn.Conv2d(emb_size * factor_filters, emb_size,
                      kernel_size=(1, in_features)),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
        )

        self.position_encoding = TAPE(
            d_model=emb_size, dropout=dropout, max_len=seq_len)
        self.attention = AttentionRelScl(
            emb_size=emb_size, num_heads=num_heads,
            seq_len=seq_len, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_size, eps=epsilon)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(emb_size, eps=epsilon)

        if use_fc:
            self.fc = nn.Linear(emb_size, num_classes)
        self.use_fc = use_fc

    @staticmethod
    def get_feature_dim(config):
        """Feature width after GAP, used by late feature fusion."""
        return config["model"].get("convtran", {}).get("emb_size", 16)

    def forward(self, x: torch.Tensor):
        N, C, T, V = x.size()
        # flatten joints into the variable axis: (N, 1, T, C*V)
        x = x.permute(0, 2, 1, 3).reshape(N, T, C * V).unsqueeze(1)

        x = self.embed(x)  # (N, emb_size, T, 1)
        x_src = x.squeeze(-1).transpose(1, 2)  # (N, T, emb_size)

        # position encoding flows into attention only; the residual keeps the
        # pre-encoding embedding x_src (official ConvTran, not the Keras port)
        x_pos = self.position_encoding(x_src)
        att = self.norm1(x_src + self.attention(x_pos))
        x = self.norm2(att + self.feed_forward(att))

        # global average pooling over time
        x = x.mean(dim=1)  # (N, emb_size)

        if self.use_fc:
            x = self.fc(x)
            x = x.softmax(dim=1)  # (N, num_class)
        return x
