import torch
import copy
import torch.nn as nn
import numpy as np
from typing import Union
from src.model.stgclsatt import STGCLSATT_fusion as STGCLSATT
from src.model.gcn.stgcn_part import ST_GCN_Part as STGCN
from src.model.params.separate_channels import separate_channels


class FusionMean(nn.Module):
    """ by fusionmodule"""

    def __init__(self):
        super(FusionMean, self).__init__()
        # 3(action dimension)
        self.pooling = nn.AdaptiveAvgPool3d((None, None, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (N, A, C, T, V) -> (N, C, T, V, A) convert
        N, A, C, T, V = x.size()
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # N, C, T, V, A

        # action dimension
        x = self.pooling(x)  # N, C, T, V, 1
        x = x.squeeze(dim=-1)  # N, C, T, V
        return x


class MiddleFusion(nn.Module):
    r"""Middle Fusion Module with fusion layer

    This module is used to fuse the features from different action
    classes in the middle of the network. Intermediate features are
    combined by averaging over the action axis, then passed through the
    post-fusion ST-GCN blocks and the classifier.
    """

    def __init__(self, config, gcn_channels: list[int] = None, fusion_channels: list[int] = None):
        super(MiddleFusion, self).__init__()

        # set load
        self.fusion_attr = config["fusion"]
        self.action_length = len(config["combined_action_indices"])
        self.num_classes = len(config["classes"])

        # Middle fusion keeps only zero-padding missing-action handling.
        self.missing_representation = self.fusion_attr.get(
            "missing_representation", "zero")
        if self.missing_representation != "zero":
            raise ValueError("Invalid missing representation: {}".format(
                self.missing_representation))

        # fusion and set
        # "middle_shallow" / "middle_deep" -> "shallow" / "deep"
        self.fusion_position = config["fusion"]["type"].split("_", 1)[1]
        out_channel_list: list = config["model"]["st-gcn"]["output_channel_list"]

        # list fusionbefore/fusionafter
        if gcn_channels is None or fusion_channels is None:
            gcn_channels, fusion_channels = separate_channels(
                out_channel_list, self.fusion_position)

        # actions are always combined by averaging over the action axis
        self.fusion = FusionMean()

        # GCNinitialize the model(set copy)
        temp_config = copy.deepcopy(config)
        temp_config["model"]["st-gcn"]["output_channel_list"] = gcn_channels

        # each action and create
        self.model_list = nn.ModuleList()
        self.BNs = nn.ModuleList()
        for act in range(self.action_length):
            temp_model = STGCLSATT(config=temp_config, use_fc=False)
            # set apply
            if temp_config["device"][:4] == "cuda":
                if torch.cuda.is_available():
                    temp_model = temp_model.cuda(
                        device=temp_config["device"])
                else:
                    temp_model = temp_model.cpu()
            self.model_list.append(temp_model)
            self.BNs.append(nn.BatchNorm2d(
                gcn_channels[-1], device=temp_config["device"]))

        # after fusion ST-GCNinitialize
        # input GCN output
        temp_config["model"]["st-gcn"]["output_channel_list"] = fusion_channels

        # fusion case input adjust
        self.stgcn_model = STGCN(
            config=temp_config, in_channels=gcn_channels[-1], use_in_fusion=True)

        # processing
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Translated note
        self.fc = nn.Linear(fusion_channels[-1], self.num_classes)  # Translated note
        self.gcn_last_channel = gcn_channels[-1]

    # signals Train/Test loops that forward accepts a mask keyword
    accepts_mask = True

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # input tensor shape get: (Batch, Action, Channel, Time, Vertex)
        N, A, C, T, V = x.size()

        # apply missing-action representation at input level
        # mask=None means all actions present (backward compatible)
        if mask is not None:
            x = x * mask.view(N, A, 1, 1, 1)

        # each action and (or weight-sharing model) extract
        feature = []
        for a in range(A):
            # a action data extract: (N, C, T, V)
            f_x = x[:, a, :, :, :]
            # extract, list append
            feature.append(self.BNs[a](self.model_list[a](f_x)))
        # featureslist convert: (A, N, C, T, V) -> (N, A, C, T, V)
        x = torch.stack(feature, dim=0)  # (A, N, C, T, V)
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # (N, A, C, T, V)
        N, A, C, T, V = x.size()  # shape get

        # fusionprocessing run
        if hasattr(self, 'fusion') and self.fusion is not None:
            # setfusion apply: (N, A, C, T, V) -> (N, C, T, V)
            x = self.fusion(x)  # (N, C, T, V)
        else:
            raise ValueError("Invalid fusion layer")

        # after fusion ST-GCN extract
        x = self.stgcn_model(x)  # (N, C, T, V)

        # temporal/points
        x = self.gap(x)  # (N, C, 1, 1)
        x = x.squeeze()  # (N, C)
        x = x.view(N, -1)  # 2 convert

        # # afterprocessing for features save
        # if self.training:
        #     self._features = x.clone()  # Store features for later use

        # value output
        x = self.fc(x)  # (N, num_classes)
        x = x.softmax(dim=-1)  # probability convert
        return x

    # each model weight(Spatial and Temporal) get
    def get_weights(self) -> Union[dict[int, dict[str, np.ndarray]], None]:
        """each model Spatial/Temporal get

        Returns:
            dict: per-action weights keyed by action index
        """
        weights = dict()
        for i, model in enumerate(self.model_list):
            weights[i] = model.get_weights()
        return weights

    # each model trainable adjacency matrix get
    def get_stgcn_weights(self):
        """each model trainableadjacency matrix get

        Returns:
            dict or None: each model adjacency matrixdictionary
        """
        weights = dict()
        for i, model in enumerate(self.model_list):
            weights[i] = model.get_stgcn_weights()
        return weights

    def get_agc_lstm_attn_loss(self):
        """each model AGC-LSTMattention loss get
        Returns:
            torch.float or None: each model AGC-LSTMattention loss, or None
        """
        attn_loss = 0.0
        count = 0
        for model in self.model_list:
            if hasattr(model, 'get_agc_lstm_attn_loss') and model.get_agc_lstm_attn_loss() is not None:
                loss = model.get_agc_lstm_attn_loss()
                if loss is not None:
                    attn_loss += loss
                    count += 1
            else:
                return None
        return attn_loss / count if count > 0 else None


if __name__ == "__main__":
    import yaml
    import io
    import sys
    from torchinfo import summary

    # config file load
    config_path = input("Enter the path to the config file: ")
    config = None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # instance
    model = MiddleFusion(config)

    # model summary generate and save
    txt = summary(model)
    txt = str(txt)

    with open(f"reports/modelStructures/model_MiddleFusion_{model.fusion_position}.txt", "w") as f:
        f.write(txt)
