import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.gcn.stgcn_part import ST_GCN_Part as ST_GCN
from src.model.rnn import LSTM_part as LSTM, AGCLSTM
from src.model.attention.attention_part import Attention_part as Attention
import typing
import numpy as np
from src.model.gcn.components import Graph


class FusionGraph():
    def __init__(self, config):
        self.config = config
        graph_config = config["model"]["st-gcn"]["graph"]
        strategy = graph_config["strategy"]
        hop_size = graph_config["hop_size"]
        dilation = graph_config["dilation"]
        action_length = len(config["combined_action_indices"])
        self.graph = Graph(max_hop=hop_size,
                           strategy=strategy,
                           dilation=dilation)
        temp_graph = np.zeros(
            (self.graph.A.shape[0]+1, action_length * self.graph.num_node, action_length * self.graph.num_node))
        for i in range(action_length):
            temp_graph[:-1, i*self.graph.num_node:(i+1)*self.graph.num_node,
                       i*self.graph.num_node:(i+1)*self.graph.num_node] = self.graph.A
            for j in range(action_length):
                if i != j:
                    temp_graph[-1, i*self.graph.num_node:(i+1)
                               * self.graph.num_node, j*self.graph.num_node:(j+1)*self.graph.num_node] = np.eye(self.graph.num_node) / (action_length - 1.0)
        self.A = torch.tensor(temp_graph,
                              dtype=torch.float32,
                              requires_grad=False,
                              device=config["device"])

    def __str__(self):
        return self.A


class EarlyFusion(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        device (torch.device): Device on which the model runs
        t_kernel_size (int): Size of the temporal convolving kernel
        hop_size (int): Number of frames to be hopped
        layer_num (int): Number of layers in the network
        strategy (str):

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
    """

    def __init__(self, config):
        super().__init__()

        # "early_chn_stack" / "early_exp_graph" -> "chn_stack" / "exp_graph"
        self.fusion_mode = config["fusion"]["type"].split("_", 1)[1]
        model_attr = config["model"]
        model_name = model_attr["name"]

        # Early fusion keeps only zero-padding missing-action handling.
        self.missing_representation = config["fusion"].get(
            "missing_representation", "zero")
        if self.missing_representation != "zero":
            raise ValueError("Invalid missing representation: {}".format(
                self.missing_representation))

        num_classes = len(config["classes"])
        # follow the model.rnn config schema used by STGCLSATT
        # (the former model.lstm schema no longer exists in any config)
        rnn_attr = model_attr["rnn"]
        attention_mode = model_attr["attention"]["mode"]

        self.model_name = model_name
        self.only_last = rnn_attr.get("only_last", False)
        in_channels = len(config["attributes"]) * 3
        if self.fusion_mode == "chn_stack":
            in_channels = in_channels * len(config["combined_action_indices"])
        elif self.fusion_mode == "exp_graph":
            fusion_graph = FusionGraph(config)
            self.A = fusion_graph.A
        self.st_gcn = ST_GCN(config, in_channels=in_channels,
                             A=self.A if self.fusion_mode == "exp_graph" else None)

        last_channel = config["model"]["st-gcn"]["output_channel_list"][-1]
        att_hidden_dim = last_channel

        if "AGLS" in model_name:
            att_hidden_dim = last_channel // 2 if config["model"]["agc-lstm"][
                "bidirectional"] and not config["model"]["agc-lstm"]["double_channel"]["use"] else last_channel
            self.agc_lstm = AGCLSTM(
                config,
                in_channels=last_channel,
                hidden_channels=att_hidden_dim,
                out_fusion="concat",
                use_tap=config["model"]["agc-lstm"]["double_channel"]["use"],
                A=self.A if self.fusion_mode == "exp_graph" else None
            )
            last_channel = att_hidden_dim if not config["model"][
                "agc-lstm"]["bidirectional"] else att_hidden_dim * 2
            # att_hidden_dim = last_channel

        else:
            if "LS" in model_name:
                self.lstm_bidirectional = rnn_attr["bidirectional"]
                # keep the output width equal to last_channel (see STGCLSATT)
                lstm_hidden_size = last_channel // 2 if self.lstm_bidirectional else last_channel

                self.lstm = LSTM(
                    input_size=last_channel,
                    hidden_size=lstm_hidden_size,
                    bidirectional=self.lstm_bidirectional,
                    num_layers=rnn_attr["num_layers"],
                )
                att_hidden_dim = last_channel

            if "Att" in model_name:
                self.attention = Attention(
                    input_dim=att_hidden_dim,
                    hidden_dim=att_hidden_dim,
                    mode=attention_mode
                )
            self.att_hidden_dim = att_hidden_dim

        self.fc = nn.Linear(last_channel, num_classes)

    # signals Train/Test loops that forward accepts a mask keyword
    accepts_mask = True

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        N, A, C, T, V = x.size()

        # apply missing-action representation before the fusion reshape
        # mask=None means all actions present (backward compatible)
        if mask is not None:
            x = x * mask.view(N, A, 1, 1, 1)

        if self.fusion_mode == "chn_stack":
            x = x.view(N, A * C, T, V)
        elif self.fusion_mode == "exp_graph":
            x = x.permute(0, 2, 3, 1, 4).contiguous().view(
                N, C, T, A * V)  # (N,C,T,A*V)
        weights = None

        x = self.st_gcn(x)
        if "AGLS" in self.model_name and hasattr(self, 'agc_lstm') and self.agc_lstm is not None:
            x, weights = self.agc_lstm(x)  # N,C,T,V
            x = F.avg_pool2d(x, x.size()[2:])
        else:
            if "LS" in self.model_name and hasattr(self, 'lstm') and self.lstm is not None:
                x = self.lstm(x)
                if self.only_last:
                    feature = x[:, :, -1, :]
                    if self.lstm_bidirectional:
                        feature += x[:, :, 0, :]
                    x = feature.unsqueeze(2)

            if "Att" in self.model_name and hasattr(self, 'attention') and self.attention is not None:
                x, weights = self.attention(x)
            else:
                # global pooling
                x = F.avg_pool2d(x, x.size()[2:])
        self._feature = x.clone().detach()  # Store features for later use

        # flatten
        x = x.view(x.size(0), -1)
        # prediction
        x = self.fc(x)
        x = x.view(N, -1)
        x = torch.softmax(x, dim=1)

        if isinstance(weights, tuple):
            self._weights = (weights[0].detach().cpu().numpy(),
                             weights[1].detach().cpu().numpy())
        elif "AGLS" in self.model_name and hasattr(self, 'agc_lstm') and self.agc_lstm is not None and self.training:
            self._weights = weights  # for loss calculation
        else:
            self._weights = weights.detach().cpu().numpy() if weights is not None else None

        return x

    def get_weights(self) -> typing.Optional[typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray], None]]:
        return self._weights

    def get_stgcn_weights(self):
        return self.st_gcn.return_weights()

    def get_feature(self) -> torch.Tensor:
        # print("Feature shape:", self.feature.shape)
        return self._feature.view(self._feature.size(0), -1) if self._feature is not None else None

    def get_agc_lstm_attn_loss(self):
        if hasattr(self, 'agc_lstm') and self.agc_lstm is not None and self._weights is not None:
            return self.agc_lstm.calc_attn_loss(self._weights)
        else:
            return None


if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import torch

    with open("config/configSomeTask.yaml", "r") as f:
        config = yaml.safe_load(f)

    save_path = input(
        "Enter path to save heatmaps (or press Enter to skip saving): ").strip()

    model = EarlyFusion(config)
    # Arowcolumn get(CPU move)
    if model.fusion_mode == "exp_graph":
        A_matrices = model.A.detach().cpu().numpy()
    else:
        A_matrices = model.st_gcn.A.detach().cpu().numpy()

    # subplotcreate(2x2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Adjacency Matrices Heatmaps', fontsize=16, fontweight='bold')

    # colormapset
    cmap = 'viridis'  # or 'RdYlBu', 'coolwarm', 'plasma'

    for i in range(A_matrices.shape[0]):  # A[0] from A[3]
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # create
        im = ax.imshow(A_matrices[i], cmap=cmap, aspect='auto')

        # titleset
        ax.set_title(f'A[{i}] - {A_matrices[i].shape}', fontweight='bold')

        # axis labels
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Node Index')

        # append
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Edge Weight', rotation=270, labelpad=15)

        # ★ 17x17 and child append
        matrix_size = A_matrices[i].shape[0]

        # 17nodes and
        for grid_line in range(17, matrix_size, 17):
            # ()
            ax.axvline(x=grid_line - 0.5, color='red',
                       linewidth=1.5, alpha=0.7)
            # ()
            ax.axhline(y=grid_line - 0.5, color='red',
                       linewidth=1.5, alpha=0.7)

        # 17x17
        for thick_line in range(17, matrix_size, 17):
            ax.axvline(x=thick_line - 0.5, color='white',
                       linewidth=2.5, alpha=0.9)
            ax.axhline(y=thick_line - 0.5, color='white',
                       linewidth=2.5, alpha=0.9)

        # statistics append
        min_val = np.min(A_matrices[i])
        max_val = np.max(A_matrices[i])
        mean_val = np.mean(A_matrices[i])

        stats_text = f'Min: {min_val:.3f}\nMax: {max_val:.3f}\nMean: {mean_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                                                   facecolor='white', alpha=0.8), fontsize=9)

        # 17x17 append()
        num_blocks = matrix_size // 17
        for block_idx in range(num_blocks):
            center_pos = block_idx * 17 + 8.5  # Translated note
            if center_pos < matrix_size:
                # display
                ax.text(center_pos, -2, f'Block {block_idx}', ha='center', va='top',
                        fontsize=8, color='black', weight='bold')
                ax.text(-2, center_pos, f'Block {block_idx}', ha='right', va='center',
                        fontsize=8, color='black', weight='bold', rotation=90)

    plt.tight_layout()

    # save
    if save_path:
        plt.savefig(save_path + "/heatmaps_with_grid.png",
                    dpi=300, bbox_inches='tight')
        print(f"Heatmaps with 17x17 grid saved to: {save_path}")

    plt.show()

    # statistics display
    print("\n=== Adjacency Matrices Statistics ===")
    for i in range(A_matrices.shape[0]):
        print(f"\nA[{i}] - Shape: {A_matrices[i].shape}")
        print(
            f"  17x17 blocks: {A_matrices[i].shape[0] // 17} x {A_matrices[i].shape[1] // 17}")
        print(f"  Min: {np.min(A_matrices[i]):.6f}")
        print(f"  Max: {np.max(A_matrices[i]):.6f}")
        print(f"  Mean: {np.mean(A_matrices[i]):.6f}")
        print(f"  Element Population Std: {np.std(A_matrices[i], ddof=0):.6f}")
