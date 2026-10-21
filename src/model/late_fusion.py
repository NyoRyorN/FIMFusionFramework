import torch
import torch.nn as nn
from typing import Union
from src.model.build_model import build_encoder, get_encoder_feature_dim
from src.model.mask_token import masked_action_mean_late
import numpy as np


class LateFusion(nn.Module):
    """Late Fusion Module for combining multiple action features.

    This module implements late fusion for combining predictions from multiple
    action-specific models. Per-action outputs are combined by averaging over the
    action axis to handle multi-action skeletal data classification tasks.

        config (dict): Configuration dictionary containing model parameters including:
            - fusion.type: "late_feature" (fuse pre-FC features) or
              "late_probability" (fuse per-action outputs)
            - fusion.missing_representation: How missing actions are represented
            - combined_action_indices: List of action indices to combine
            - classes: List of output classes
            - device: Computing device specification

    Attributes:
        missing_representation (str): How missing actions are represented. Options:
            - 'zero': missing input slices stay all-zero (dataset default)
            - 'pad_mask': missing slots never reach the encoder (forwarded
              only for present samples) and are excluded from the averaging.
              During training, an action is
              additionally treated as fully missing for a batch if fewer than
              2 samples have it present, since BatchNorm inside the encoder
              cannot compute batch statistics from a single sample; eval mode
              has no such restriction (BatchNorm uses running stats there)
        action_length (int): Number of action classes to fuse
        num_classes (int): Number of output classification classes
        model_list (nn.ModuleList): One encoder per action
        fusion (nn.Module): Average pooling over the action axis

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, A, C, T, V) where:
            - N: batch size
            - A: number of actions
            - C: number of channels
            - T: temporal dimension
            - V: number of vertices/joints

    Forward Output:
        torch.Tensor: Fused predictions of shape (N, num_classes)

    Methods:
        forward: Performs forward pass through the fusion module
        get_weights: Returns per-action attention weights
        get_stgcn_weights: Returns ST-GCN adjacency matrix weights

    Raises:
        ValueError: If an invalid missing_representation is specified

    Example:
        >>> config = {
        ...     'fusion': {'type': 'late_probability', 'missing_representation': 'zero'},
        ...     'combined_action_indices': [0, 1, 2],
        ...     'classes': ['class1', 'class2'],
        ...     'device': 'cuda:0'
        ... }
        >>> fusion_module = LateFusion(config)
        >>> x = torch.randn(2, 3, 64, 100, 25)  # (batch, actions, channels, time, joints)
        >>> output = fusion_module(x)  # Shape: (2, 2)
    """

    def __init__(self, config):
        super().__init__()

        # set load
        self.fusion_attr = config["fusion"]
        # "late_feature" fuses pre-FC features, "late_probability" fuses outputs
        self.fusion_before_fc = self.fusion_attr["type"] == "late_feature"
        self.action_length = len(config["combined_action_indices"])
        self.num_classes = len(config["classes"])
        out_channel = get_encoder_feature_dim(config)
        self.out_channel = out_channel

        # missing-action representation: "zero" | "pad_mask"
        self.missing_representation = self.fusion_attr.get(
            "missing_representation", "zero")
        valid_representations = ("zero", "pad_mask")
        if self.missing_representation not in valid_representations:
            raise ValueError("Invalid missing representation: {}".format(
                self.missing_representation))

        # actions are always combined by averaging over the action axis
        self.fusion = nn.AdaptiveAvgPool1d(1)

        # each action and create
        self.model_list = nn.ModuleList()
        for act in range(self.action_length):
            temp_model = build_encoder(
                config, use_fc=not self.fusion_before_fc)

            # set apply
            if config["device"][:4] == "cuda":
                if torch.cuda.is_available():
                    temp_model = temp_model.cuda(device=config["device"])
                else:
                    temp_model = temp_model.cpu()
            self.model_list.append(temp_model)

        if self.fusion_before_fc:
            self.fc = nn.Linear(out_channel, self.num_classes)
            self.BNs = nn.ModuleList()
            for _ in range(self.action_length):
                self.BNs.append(nn.BatchNorm1d(
                    out_channel, device=config["device"]))

        # see _scatter_branch_weights / get_weights
        self._weight_shape_templates = {}
        self._pad_mask_weights = None

    # signals Train/Test loops that forward accepts a mask keyword
    accepts_mask = True

    # pad_mask forwards only present samples through each branch (see the
    # branch loop in forward()), so a branch's get_weights() output is
    # smaller than the batch, or stale, whenever some/all samples miss that
    # action this batch. These caches let _scatter_branch_weights expand
    # such output back to one row per batch sample (NaN where not computed)
    # so output_weights (Test.output_weights) stays aligned with dir_list.
    _weight_shape_templates: dict
    _pad_mask_weights: Union[dict, None]

    def _scatter_branch_weights(self, action_idx, raw, present_np, N):
        """Expand a pad_mask branch's present-only weights to batch size N.

        Args:
            action_idx: index into model_list, used as the shape-template
                cache key (architecture, hence weight shape, is the same
                across batches for a given action)
            raw: this batch's get_weights() output for the branch (already
                a numpy array/tuple, see STGCLSATT.get_weights), or None if
                the branch had too few present samples to run this batch
            present_np: boolean array (N,) selecting which rows were
                forwarded through the branch this batch
            N: full batch size

        Returns:
            structure matching raw (None / ndarray / tuple of ndarray)
            expanded to batch size N with NaN in rows the branch didn't
            compute, or None if no shape template is available yet (only
            possible if no action anywhere has had enough present samples
            in any batch seen so far this run)
        """
        if raw is None:
            is_multi, template = self._weight_shape_templates.get(
                action_idx, (None, None))
            if template is None:
                # all branches share the same architecture (hence weight
                # shape), so borrow another action's template rather than
                # waiting for this specific action to appear in a batch
                is_multi, template = next(
                    iter(self._weight_shape_templates.values()), (None, None))
            if template is None:
                return None
            if is_multi:
                return tuple(np.full((N,) + t, np.nan, dtype=np.float32)
                             for t in template)
            return np.full((N,) + template, np.nan, dtype=np.float32)

        if isinstance(raw, tuple):
            self._weight_shape_templates[action_idx] = (
                True, tuple(r.shape[1:] for r in raw))
            scattered = []
            for r in raw:
                full = np.full((N,) + r.shape[1:], np.nan, dtype=np.float32)
                full[present_np] = r
                scattered.append(full)
            return tuple(scattered)

        self._weight_shape_templates[action_idx] = (False, raw.shape[1:])
        full = np.full((N,) + raw.shape[1:], np.nan, dtype=np.float32)
        full[present_np] = raw
        return full

    def forward(self, x, mask=None):
        # input tensor shape get: (Batch, Action, Channel, Time, Vertex)
        N, A, C, T, V = x.size()

        # apply missing-action representation at input level
        # mask=None means all actions present (backward compatible);
        # pad_mask needs no input substitution since missing slots never
        # reach the encoder (handled in the branch loop below)
        if mask is not None and self.missing_representation == "zero":
            x = x * mask.view(N, A, 1, 1, 1)

        # each action and (or weight-sharing model) extract
        # agg_mask feeds the downstream masked-mean aggregation; it starts as
        # an alias of mask and is only cloned if pad_mask needs to downgrade
        # a column below (avoids mutating the caller's mask tensor)
        agg_mask = mask
        feature = []
        # pad_mask forwards only present samples per branch (below), so a
        # branch's get_weights() output must be scattered back to batch size
        # N before it can be cached for Test.output_weights (see
        # _scatter_branch_weights); only worth the numpy conversion at eval
        # time, since that's the only place get_weights() output is used
        collect_weights = self.missing_representation == "pad_mask" \
            and mask is not None and not self.training
        pad_mask_weights = {} if collect_weights else None
        for a in range(A):
            # a action data extract: (N, C, T, V)
            f_x = x[:, a, :, :, :]
            if self.missing_representation == "pad_mask" and mask is not None:
                # forward only present samples so missing slots never touch
                # the encoder: no BatchNorm contamination, no gradient flow
                present = mask[:, a] > 0
                out_dim = self.out_channel if self.fusion_before_fc else self.num_classes
                f = x.new_zeros(N, out_dim)
                # BatchNorm inside the encoder (and self.BNs[a]) needs batch
                # size > 1 to compute training statistics; eval-mode BN uses
                # running stats so a single present sample is safe there
                min_present = 2 if self.training else 1
                if present.sum() >= min_present:
                    branch_out = self.model_list[a](f_x[present])
                    f[present] = self.BNs[a](branch_out) \
                        if self.fusion_before_fc else branch_out
                    if collect_weights and hasattr(self.model_list[a], "get_weights"):
                        raw_w = self.model_list[a].get_weights()
                        pad_mask_weights[a] = self._scatter_branch_weights(
                            a, raw_w, present.detach().cpu().numpy(), N)
                else:
                    if present.any():
                        # too few present samples to run the encoder this
                        # batch; exclude from aggregation too so the lone
                        # present sample isn't diluted by a phantom zero
                        # contribution
                        if agg_mask is mask:
                            agg_mask = mask.clone()
                        agg_mask[:, a] = 0
                    if collect_weights and hasattr(self.model_list[a], "get_weights"):
                        # branch didn't run this batch (0 or 1 present
                        # sample): NaN-fill from the cached shape template
                        # instead of leaving get_weights() stale/mis-sized
                        pad_mask_weights[a] = self._scatter_branch_weights(
                            a, None, present.detach().cpu().numpy(), N)
                feature.append(f)
            else:
                # run, list append
                feature.append(self.model_list[a](f_x)) if not self.fusion_before_fc else \
                    feature.append(self.BNs[a](self.model_list[a](f_x)))
        # featureslist convert: (A, N, C) -> (N, C, A)
        x = torch.stack(feature, dim=0)
        x = x.permute(1, 2, 0).contiguous()

        # Late Fusion - Probability
        if not self.fusion_before_fc:
            # branch outputs are already probabilities, so averaging them
            # keeps the result a probability distribution (no softmax here)
            if self.missing_representation == "pad_mask" and mask is not None:
                # masked mean: average only over present actions
                x = masked_action_mean_late(x, agg_mask)  # (N, C)
            else:
                x = self.fusion(x)  # (N, C, 1)
                x = x.squeeze(dim=-1)  # (N,C)
        # Late Fusion - Feature
        else:
            # setfusion apply
            if self.missing_representation == "pad_mask" and mask is not None:
                # masked mean: average only over present actions
                x = masked_action_mean_late(x, agg_mask)  # (N, C)
            else:
                x = self.fusion(x)  # (N, C, 1)
                # after deleteoutput shape: (N, C, 1) -> (N, C)
                x = x.squeeze(dim=-1)
            x = self.fc(x)
            x = torch.softmax(x, dim=1)

        self._pad_mask_weights = pad_mask_weights
        return x

    def get_weights(self) -> Union[dict[int, np.ndarray], None]:
        """each model get

        Under pad_mask, branches only forward present samples (see
        forward()), so a branch's raw get_weights() output is smaller than
        the batch, or stale, whenever some/all samples miss that action.
        Returning the NaN-scattered cache from the last forward() call
        instead keeps this aligned with dir_list for Test.output_weights.

        Returns:
            dict: per-action weights keyed by action index
        """
        if self.missing_representation == "pad_mask" and self._pad_mask_weights is not None:
            return self._pad_mask_weights
        weights = dict()
        for i, model in enumerate(self.model_list):
            weights[i] = model.get_weights()
        return weights

    def get_stgcn_weights(self):
        """ST-GCN adjacency matrix get

        Returns:
            dict or torch.Tensor or None: mode format
        """
        # each model dictionaryformat get
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
    from torchinfo import summary
    import io
    import torch

    # config file load
    config_path = input("Enter the path to the config file: ")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # output directory
    output_path = input("Enter the output directory path: ")

    # instance and CPU to move
    model = LateFusion(config)
    model = model.to("cpu")

    # model summary generate and save
    txt = summary(model)
    txt = str(txt)
    with open(f"{output_path}/model_late_fusion.txt", "w") as f:
        f.write(txt)

    # model structure file
    model_str = str(model)
    with open(f"{output_path}/model_late_fusion_structure.txt", "w") as f:
        f.write(model_str)

    print(
        f"Model structure saved to {output_path}/model_late_fusion_structure.txt")
