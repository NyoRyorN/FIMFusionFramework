import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.gcn.components import Graph
from typing import Union

# ===== Graph Convolution (ST-GCN) =====


class GraphConv(nn.Module):
    """
    Y = sum_k  A_hat[k] @ X @ W_k
      X: (B, V, Cin) -> Y: (B, V, Cout)
    A_hat: (K, V, V) normalized (D^-1/2 A D^-1/2)
    """

    def __init__(self, A_hat: torch.Tensor, in_channels: int, out_channels: int, bias=True):
        super().__init__()
        self.register_buffer("A_hat", A_hat)  # (K, V, V)
        self.K, self.V, _ = A_hat.shape
        self.W = nn.Linear(in_channels, out_channels * self.K, bias=bias)
    #     self.W = nn.Parameter(torch.empty(self.K, in_channels, out_channels))
    #     self.b = nn.Parameter(torch.empty(out_channels)) if bias else None
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
    #     if self.b is not None:
    #         bound = 1 / math.sqrt(self.W.shape[1])
    #         nn.init.uniform_(self.b, -bound, bound)

    def forward(self, X):            # X: (B, V, Cin)
        XW = self.W(X)               # (B, V, Cout * K)
        XW = XW.view(X.shape[0], self.V, self.K, -
                     1).permute(2, 0, 1, 3)  # (K, B, V, Cout)
        AXW = torch.einsum("kvu,kbuc->kbvc", self.A_hat, XW)  # (K, B, V, Cout)
        Y = AXW.sum(dim=0)                                     # (B,V,Cout)
        return Y


# ===== spatial((4)-(5)) =====
class SpatialAttention(nn.Module):
    """
    : H_b (B,V,C): alpha (B,V,1), value(0,1)
    q_t allnodes, eachnodes to broadcast.
    """

    def __init__(self, channels):
        super().__init__()
        self.Wq = nn.Linear(channels, channels, bias=False)
        self.Wq_post = nn.Linear(channels, channels, bias=True)
        self.Wh = nn.Linear(channels, channels, bias=True)
        self.Us = nn.Linear(channels, 1, bias=True)

    def forward(self, H_b):                 # (B,V,C)
        B, V, C = H_b.shape
        q = self.Wq(H_b).sum(dim=1)         # (B,C)
        q = self.Wq_post(F.relu(q))         # (B,C)
        q = q.unsqueeze(1).expand(B, V, C)  # (B,V,C)
        s = torch.tanh(self.Wh(H_b) + q)    # (B,V,C)
        alpha = torch.sigmoid(self.Us(s))   # (B,V,1)
        return alpha


# ===== AGC-LSTM ((3) LSTMCell) =====
class AGCLSTMCell(nn.Module):
    """
    each before GraphConv:
      [i,f,o,g] = split( GCN_x(X_t) + GCN_h(H_{t-1}) + b )
      C_t = f*C_{t-1} + i*tanh(g)
      H~_t = o * tanh(C_t)
      H_t = (1 + alpha_t) * H~_t # spatial((4)-(5))
    output:
      X_t: (B,V,Cin),  H_{t-1}: (B,V,Ch),  C_{t-1}: (B,V,Ch)
      : H_t, C_t, alpha_t
    """

    def __init__(self, A_hat, in_channels, hidden_channels, use_bias=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gcn_xi = GraphConv(A_hat, in_channels,
                                hidden_channels, bias=use_bias)
        self.gcn_xf = GraphConv(A_hat, in_channels,
                                hidden_channels, bias=use_bias)
        self.gcn_xo = GraphConv(A_hat, in_channels,
                                hidden_channels, bias=use_bias)
        self.gcn_xu = GraphConv(A_hat, in_channels,
                                hidden_channels, bias=use_bias)
        self.gcn_hi = GraphConv(A_hat, hidden_channels,
                                hidden_channels, bias=use_bias)
        self.gcn_hf = GraphConv(A_hat, hidden_channels,
                                hidden_channels, bias=use_bias)
        self.gcn_ho = GraphConv(A_hat, hidden_channels,
                                hidden_channels, bias=use_bias)
        self.gcn_hu = GraphConv(A_hat, hidden_channels,
                                hidden_channels, bias=use_bias)
        self.attn = SpatialAttention(hidden_channels)

    def forward(self, X_t, H_prev, C_prev):
        # (1) before generate
        i = torch.sigmoid(self.gcn_xi(X_t) + self.gcn_hi(H_prev))
        f = torch.sigmoid(self.gcn_xf(X_t) + self.gcn_hf(H_prev))
        o = torch.sigmoid(self.gcn_xo(X_t) + self.gcn_ho(H_prev))
        u = torch.tanh(self.gcn_xu(X_t) + self.gcn_hu(H_prev))
        # (2)
        C_t = f * C_prev + i * u
        H_hat = o * torch.tanh(C_t)                 # H_hat
        # (3) spatial
        alpha = self.attn(H_hat)                    # (B,V,1)
        H_t = (1.0 + alpha) * H_hat                 # f_att(h_hat) + h_hat
        return H_t, C_t, alpha  # (B,V,H), (B,V,H), (B,V,1)


# ===== AGC-LSTM (temporal) =====
class AGCLSTM(nn.Module):
    """
    : X (B, Cin, T, V)
    : H_all (B,H,T,V), alpha_all (B,T,V)
    """

    def __init__(self, config, in_channels: int, hidden_channels: int, out_fusion: str = "concat", use_tap: bool = False, A: Union[torch.Tensor, None] = None):
        super().__init__()
        assert out_fusion in [
            "concat", "sum"], "out_fusion must be 'concat' or 'sum'"
        device = config["device"]

        agc_lstm_attr = config["model"]["agc-lstm"]
        self.lambda_loss = agc_lstm_attr["loss"]["lambda"]  # Translated note
        self.beta_loss = agc_lstm_attr["loss"]["beta"]    # Translated note
        self.bidirectional = agc_lstm_attr["bidirectional"]

        graph_attr = config["model"]["st-gcn"]["graph"]
        hop_size = graph_attr["hop_size"]  # graph hop size
        strategy = graph_attr["strategy"]  # graph construction strategy
        dilation = graph_attr["dilation"]  # graph dilation setting

        if use_tap:
            kernel_size = (config["model"]["st-gcn"]["t_kernel_size"], 1)
            # self.tap = nn.AvgPool2d(kernel_size=kernel_size, stride=(
            #     2, 1), padding=(kernel_size[0]//2, 0))
            if config["model"]["agc-lstm"]["double_channel"]["mode"] == "tap":
                self.tap = nn.AvgPool2d(kernel_size=kernel_size, stride=(
                    2, 1), padding=(kernel_size[0]//2, 0))
            elif config["model"]["agc-lstm"]["double_channel"]["mode"] == "conv":
                self.tap = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=(
                    2, 1), padding=(kernel_size[0]//2, 0))
        self.use_tap = use_tap  # Temporal Average Pooling

        # graph structure build
        # human skeleton jointsconnection relationships, adjacency matrix A generate
        A_hat = None
        if A is None:
            graph = Graph(max_hop=hop_size,
                          strategy=strategy,
                          dilation=dilation)
            A_hat = torch.tensor(graph.A,
                                 dtype=torch.float32,
                                 requires_grad=False,
                                 device=device)
        else:
            A_hat = A.to(device)

        self.f_cell = AGCLSTMCell(A_hat, in_channels, hidden_channels)
        if self.bidirectional:
            self.b_cell = AGCLSTMCell(A_hat, in_channels, hidden_channels)

        self.out_fusion = out_fusion  # "concat" or "add"

    def _init_state(self, X, hidden_channels):
        B, T, V, _ = X.shape
        H = X.new_zeros(B, V, hidden_channels)
        C = X.new_zeros(B, V, hidden_channels)
        return H, C

    def forward(self, X):
        B, Cin, T, V = X.shape
        if self.use_tap:
            X = self.tap(X)  # (B,Cin,T',V)
            T = X.shape[2]
        X = X.permute(0, 2, 3, 1)  # (B,T,V,Cin)

        # forward
        H, C = self._init_state(X, self.f_cell.hidden_channels)
        outs = []
        alphas = []
        for t in range(T):
            H, C, a = self.f_cell(X[:, t], H, C)
            outs.append(H)  # (T, B, V, H)
            alphas.append(a)  # (T, B, V, 1)
        H_f = torch.stack(outs, dim=1)      # (B,T,V,H)
        A_f = torch.stack(alphas, dim=1).squeeze(-1)  # (B,T,V)

        # backward
        if self.bidirectional:
            H, C = self._init_state(X, self.b_cell.hidden_channels)
            outs = []
            alphas = []
            for t in reversed(range(T)):
                H, C, a = self.b_cell(X[:, t], H, C)
                outs.append(H)
                alphas.append(a)  # (B,V,1)
            outs.reverse()
            alphas.reverse()  # (T, B, V, 1)
            H_b = torch.stack(outs, dim=1)      # (B,T,V,H)
            A_b = torch.stack(alphas, dim=1).squeeze(-1)  # (B,T,V)

            if self.out_fusion == "concat":
                H_f = torch.cat([H_f, H_b], dim=-1)   # (B,T,V,2H)
                A_f = (A_f + A_b) / 2.0                # (B,T,V)
            else:  # "sum"
                H_f = H_f + H_b                        # (B,T,V,H)
                A_f = (A_f + A_b) / 2.0                # (B,T,V)

        return H_f.permute(0, 3, 1, 2), A_f  # (B,H,T,V), (B,T,V)

    def calc_attn_loss(self, A_f: torch.Tensor) -> torch.Tensor:
        """spatial loss

        2 from loss calculate:
        1. temporal: temporal value 1
        2. spatial: spatial value 0

        Args:
            A_f (torch.Tensor): (B, T, V)
                B: batch, T: temporalframe, V: points

        Returns:
            torch.Tensor: lossvalue ()

        :
        - temporal: λ * ||1 - mean_t(alpha)||²
        - spatial: β * ||mean_v(alpha)||²
        """
        tempAf = A_f.clone()
        B, T, V = tempAf.shape

        # all to to (1points attention)
        # (B,T,V) -> (B,V) -> (B,V) -> (B,)
        left_attn_loss = self.lambda_loss * (1.0 - tempAf.mean(dim=1)
                                             ).pow(2).sum(dim=-1)  # points

        # skeletonpoints and (all points attention)
        # (B,T,V) -> (B,T) -> (B,T) -> (B,)
        right_attn_loss = self.beta_loss * (tempAf.sum(dim=2)).pow(
            2).mean(dim=-1)  # temporal dimension

        return (left_attn_loss + right_attn_loss).mean()
