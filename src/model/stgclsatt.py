import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.gcn.stgcn_part import ST_GCN_Part as ST_GCN
from src.model.rnn import LSTM_part as LSTM, AGCLSTM
from src.model.attention.attention_part import Attention_part as Attention
from src.model.attention.attention_part import Attention_part_fusion as AttentionFusion
import typing
import numpy as np


class STGCLSATT(nn.Module):
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

    def __init__(self, config, use_fc=True):
        super().__init__()

        model_attr = config["model"]
        model_name = model_attr["name"]

        num_classes = len(config["classes"])
        rnn_attr = model_attr["rnn"]
        attention_mode = model_attr["attention"]["mode"]

        self.model_name = model_name
        in_channels = len(config["attributes"]) * 3
        self.st_gcn = ST_GCN(config, in_channels=in_channels)

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
                use_tap=config["model"]["agc-lstm"]["double_channel"]["use"]
            )
            last_channel = att_hidden_dim if not config["model"][
                "agc-lstm"]["bidirectional"] else att_hidden_dim * 2
            # att_hidden_dim = last_channel

        else:
            if "LS" in model_name or "GRU" in model_name:
                self.rnn_bidirectional = rnn_attr["bidirectional"]
                rnn_hidden_size = last_channel // 2 if self.rnn_bidirectional else last_channel

                self.rnn = LSTM(
                    input_size=last_channel,
                    hidden_size=rnn_hidden_size,
                    bidirectional=self.rnn_bidirectional,
                    num_layers=rnn_attr["num_layers"]
                )  # if "LS" in model_name else GRU(
                #     input_size=last_channel,
                #     hidden_size=rnn_hidden_size,
                #     bidirectional=self.rnn_bidirectional,
                #     num_layers=rnn_attr["num_layers"]
                # )
                att_hidden_dim = last_channel

            # elif "TTF" in model_name:
            # # output in, output
            #     self.temp_transformer = TemporalTransformer(
            #         in_channels=last_channel,
            #         t_nhead=model_attr["temp_transformer"]["t_nhead"],
            #         dim_feedforward=model_attr["temp_transformer"]["dim_feedforward"],
            #         num_layers=model_attr["temp_transformer"]["num_layers"],
            #     )
            #     att_hidden_dim = last_channel

            if "Att" in model_name:
                self.attention = Attention(
                    input_dim=att_hidden_dim,
                    hidden_dim=att_hidden_dim,
                    mode=attention_mode
                )
            self.att_hidden_dim = att_hidden_dim

        if use_fc:
            self.fc = nn.Linear(last_channel, num_classes)
        self.use_fc = use_fc

    @staticmethod
    def get_feature_dim(config) -> int:
        """Encoder output width when built with use_fc=False, used by late
        feature fusion. Must mirror the last_channel computation above:
        "AGLS" models double the width via bidirectional AGC-LSTM
        concatenation, other variants (RNN/attention/plain) do not change
        the ST-GCN output width.
        """
        model_name = config["model"]["name"]
        last_channel = config["model"]["st-gcn"]["output_channel_list"][-1]
        if "AGLS" in model_name:
            agc_attr = config["model"]["agc-lstm"]
            att_hidden_dim = last_channel // 2 if agc_attr["bidirectional"] \
                and not agc_attr["double_channel"] else last_channel
            last_channel = att_hidden_dim if not agc_attr["bidirectional"] \
                else att_hidden_dim * 2
        return last_channel

    def forward(self, x: torch.Tensor):
        N, C, T, V = x.size()
        weights = None

        x = self.st_gcn(x)
        if "AGLS" in self.model_name and hasattr(self, 'agc_lstm') and self.agc_lstm is not None:
            x, weights = self.agc_lstm(x)  # N,C,T,V
            x = F.avg_pool2d(x, x.size()[2:])
        else:
            if ("LS" in self.model_name or "GRU" in self.model_name) and hasattr(self, 'rnn') and self.rnn is not None:
                x = self.rnn(x)

            if "Att" in self.model_name and hasattr(self, 'attention') and self.attention is not None:
                x, weights = self.attention(x)
            else:
                # global pooling
                x = F.avg_pool2d(x, x.size()[2:])

        self._feature = x.clone().detach()  # Store features for later use

        if self.use_fc:
            # flatten
            x = x.view(x.size(0), -1)

            # prediction
            x = self.fc(x)
            x = x.view(N, -1)
            x = x.softmax(dim=1)  # (N, num_class)
        else:
            # flatten
            x = x.view(x.size(0), -1)  # (N, C)

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


class STGCLSATT_fusion(STGCLSATT):
    r"""Spatial temporal graph convolutional networks with middle layer attention.

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

    def __init__(self, config, use_fc=True):
        super().__init__(config, use_fc)
        # self.st_gcn = ST_GCN(config, in_channels=in_channels)
        self.attention = AttentionFusion(
            input_dim=self.att_hidden_dim,
            hidden_dim=self.att_hidden_dim,
            mode=config["model"]["attention"]["mode"]
        )

    def forward(self, x):
        N, C, T, V = x.size()
        weights = None

        x = self.st_gcn(x)
        if "AGLS" in self.model_name and hasattr(self, 'agc_lstm') and self.agc_lstm is not None:
            x, weights = self.agc_lstm(x)  # N,C,T,V
        else:
            if ("LS" in self.model_name or "GRU" in self.model_name) and hasattr(self, 'rnn') and self.rnn is not None:
                x = self.rnn(x)

            if "Att" in self.model_name and hasattr(self, 'attention') and self.attention is not None:
                x, weights = self.attention(x)

        if isinstance(weights, tuple):
            self._weights = (weights[0].detach().cpu().numpy(),
                             weights[1].detach().cpu().numpy())
        elif "AGLS" in self.model_name and hasattr(self, 'agc_lstm') and self.agc_lstm is not None and self.training:
            self._weights = weights  # for loss calculation
        else:
            self._weights = weights.detach().cpu().numpy() if weights is not None else None

        return x
