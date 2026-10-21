"""
litemv.py - LITEMV (LITE MultiVariate) baseline classifier

PyTorch port of the LITEMV classifier from DeepRehabPile
(https://github.com/MSD-IRIMAS/DeepRehabPile,
deep_rehab_pile/classifiers/_lite_mv.py, Keras implementation).
LITE is a lightweight inception-based CNN for time series classification:
one multiplexing inception module (three parallel depthwise-separable
convolutions plus fixed hand-crafted "hybrid" filters) followed by two
dilated depthwise-separable FCN modules, global average pooling over time
and a softmax classification head.

The skeleton input (N, C, T, V) is flattened to a multivariate time
series (N, C*V, T) before the first convolution.

Reference:
    Ismail-Fawaz et al. "Look Into the LITE in Deep Learning for Time
    Series Classification." arXiv:2409.02869 (2024).

Config: config/model_ablation/litemv.yaml (model.litemv block)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution matching Keras SeparableConv1D.

    Both the depthwise and the pointwise convolution are bias-free because
    the reference implementation uses use_bias=False.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            groups=in_channels, dilation=dilation,
            padding="same", bias=False)
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def _build_hybrid_filters(kernel_sizes):
    """Build the fixed hand-crafted 1D kernels of the LITE hybrid layer.

    Returns:
        list of 1D numpy arrays: 6 increasing-detection, 6 decreasing-
        detection and 5 peak-detection kernels (17 in total for the
        default kernel_sizes). Each kernel is shared across all input
        channels (depthwise convolution with identical per-channel weights).
    """
    filters = []

    # increasing detection filters: alternating -1/+1 starting with -1
    for kernel_size in kernel_sizes:
        filt = np.ones(kernel_size)
        filt[np.arange(kernel_size) % 2 == 0] *= -1
        filters.append(filt)

    # decreasing detection filters: alternating +1/-1 starting with +1
    for kernel_size in kernel_sizes:
        filt = np.ones(kernel_size)
        filt[np.arange(kernel_size) % 2 > 0] *= -1
        filters.append(filt)

    # peak detection filters: piecewise-quadratic mexican-hat-like shape
    for kernel_size in kernel_sizes[1:]:
        filt = np.zeros(kernel_size + kernel_size // 2)
        xmesh = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:]
        filter_left = xmesh ** 2
        filter_right = filter_left[::-1]
        filt[0:kernel_size // 4] = -filter_left
        filt[kernel_size // 4:kernel_size // 2] = -filter_right
        filt[kernel_size // 2:3 * kernel_size // 4] = 2 * filter_left
        filt[3 * kernel_size // 4:kernel_size] = 2 * filter_right
        filt[kernel_size:5 * kernel_size // 4] = -filter_left
        filt[5 * kernel_size // 4:] = -filter_right
        filters.append(filt)

    return filters


class HybridLayer(nn.Module):
    """Non-trainable depthwise convolutions with hand-crafted filters.

    Each filter is applied as a bias-free depthwise convolution whose
    weights are registered as buffers (excluded from optimization), then
    all outputs are concatenated on the channel axis and passed through
    ReLU, mirroring the Keras hybrid layer of LITE.
    """

    def __init__(self, in_channels, kernel_sizes=(2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.in_channels = in_channels
        filters = _build_hybrid_filters(list(kernel_sizes))
        self.num_filters = len(filters)
        for i, filt in enumerate(filters):
            weight = torch.tensor(filt, dtype=torch.float32)
            # depthwise weight shape (in_channels, 1, kernel): the same
            # kernel is shared by every input channel
            weight = weight.view(1, 1, -1).repeat(in_channels, 1, 1)
            self.register_buffer(f"filter_{i}", weight)

    @property
    def out_channels(self):
        return self.num_filters * self.in_channels

    def forward(self, x):
        outs = []
        for i in range(self.num_filters):
            weight = getattr(self, f"filter_{i}")
            outs.append(F.conv1d(
                x, weight, groups=self.in_channels, padding="same"))
        return F.relu(torch.cat(outs, dim=1))


class LITEMV(nn.Module):
    """LITEMV baseline classifier.

    Args:
        config: experiment config dictionary; reads the model.litemv block
            (n_filters, kernel_size, n_joints) plus attributes / classes
        use_fc: if False, return GAP features (N, n_filters) instead of
            class probabilities (for late feature fusion)
        num_actions: input channel multiplier for early fusion; the input
            is expected as (N, num_actions * C, T, V)

    Shape:
        - Input: (N, C, T, V)
        - Output: (N, num_class) softmax probabilities, or (N, n_filters)
          features when use_fc=False
    """

    def __init__(self, config, use_fc=True, num_actions=1):
        super().__init__()

        model_attr = config["model"].get("litemv", {})
        n_filters = model_attr.get("n_filters", 32)
        kernel_size = model_attr.get("kernel_size", 40)
        n_joints = model_attr.get("n_joints", 17)

        num_classes = len(config["classes"])
        in_features = len(config["attributes"]) * 3 * n_joints * num_actions

        # Keras BatchNormalization defaults (eps=1e-3, momentum 0.99 in
        # Keras convention == 0.01 in PyTorch convention)
        bn_kwargs = {"eps": 1e-3, "momentum": 0.01}

        # ------------------------------------------------------------------
        # Inception module (multiplexing convolutions + hybrid filters)
        # ------------------------------------------------------------------
        inception_kernels = [kernel_size // (2 ** i) for i in range(3)]
        self.inception_convs = nn.ModuleList([
            SeparableConv1d(in_features, n_filters, k)
            for k in inception_kernels
        ])
        self.hybrid = HybridLayer(in_features)
        inception_out = n_filters * 3 + self.hybrid.out_channels
        self.inception_bn = nn.BatchNorm1d(inception_out, **bn_kwargs)

        # ------------------------------------------------------------------
        # Two dilated FCN modules
        # ------------------------------------------------------------------
        fcn_kernel = kernel_size // 2
        self.fcn1 = nn.Sequential(
            SeparableConv1d(inception_out, n_filters, fcn_kernel, dilation=2),
            nn.BatchNorm1d(n_filters, **bn_kwargs),
            nn.ReLU(),
        )
        self.fcn2 = nn.Sequential(
            SeparableConv1d(n_filters, n_filters, fcn_kernel // 2, dilation=4),
            nn.BatchNorm1d(n_filters, **bn_kwargs),
            nn.ReLU(),
        )

        if use_fc:
            self.fc = nn.Linear(n_filters, num_classes)
        self.use_fc = use_fc

    @staticmethod
    def get_feature_dim(config):
        """Feature width after GAP, used by late feature fusion."""
        return config["model"].get("litemv", {}).get("n_filters", 32)

    def forward(self, x: torch.Tensor):
        N, C, T, V = x.size()
        # flatten joints into the channel axis: (N, C*V, T)
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)

        convs = [conv(x) for conv in self.inception_convs]
        convs.append(self.hybrid(x))
        x = F.relu(self.inception_bn(torch.cat(convs, dim=1)))

        x = self.fcn1(x)
        x = self.fcn2(x)

        # global average pooling over time
        x = x.mean(dim=-1)  # (N, n_filters)

        if self.use_fc:
            x = self.fc(x)
            x = x.softmax(dim=1)  # (N, num_class)
        return x
