# attention mechanismmodule
# attention weightscalculate
# from attention weights output

from torch import nn


class Attention(nn.Module):
    r"""(Basic Attention)module

    attention weights calculate.
    from value(attention weights) output,
    element evaluate.

    Key characteristics:
    - 2 linear transformation by calculate
    - Tanhfunction by linear transformation
    -
    - attention mechanism

    Args:
        in_dim (int): input dimension count
        hid_dim (int): hidden layer dimension count

    Shape:
        - Input: :math:`(mini_batch_size, number_of_blocks, z_dim)`
          batch × ×
        - Output: :math:`(mini_batch_size, number_of_blocks, 1)`
          batch × × 1(attention weights)
    """

    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        # 2 MLP(Multi-Layer Perceptron) by attention weightscalculate
        self.attention = nn.Sequential(
            # 1: input from hidden layer to linear transformation
            nn.Linear(in_dim, hid_dim),
            # Tanhfunction by linear transformation
            nn.Tanh(),
            # 2: hidden layer from value to convert
            nn.Linear(hid_dim, 1)
        )

    def forward(self, embeddings):
        # : shape(batch_size, attn_len, z_dim)
        # eachelement for attention weights calculate
        e = self.attention(embeddings)
        # attention weights: shape(batch_size, attn_len, 1)
        # eachelement value
        return e
