"""
build_model.py - encoder factory mapping config model names to classes

Single place that resolves config["model"]["name"] to an encoder class.
STGCLSATT covers the name-driven in-house variants (STGCN / STGCLSAtt /
STGCAGLS, selected by substring inside the class), while the baseline
models (LITEMV / ConvTran from DeepRehabPile) are independent classes
selected by exact name here.

Config: model.name in any experiment yaml
"""

from src.model.stgclsatt import STGCLSATT
from src.model.baselines.litemv import LITEMV
from src.model.baselines.convtran import ConvTran

# exact-name registry; STGCLSATT handles every other (substring-based) name
BASELINE_CLASSES = {
    "LITEMV": LITEMV,
    "ConvTran": ConvTran,
}


def is_baseline(config) -> bool:
    """Return True when config selects a baseline model."""
    return config["model"]["name"] in BASELINE_CLASSES


def build_encoder(config, use_fc=True, num_actions=1):
    """Instantiate the encoder selected by config["model"]["name"].

    Args:
        config: experiment config dictionary
        use_fc: if False, the encoder returns pooled features instead of
            class probabilities (late feature fusion)
        num_actions: input channel multiplier for baseline early fusion;
            must stay 1 for STGCLSATT (its early fusion is handled by the
            dedicated EarlyFusion class)

    Returns:
        nn.Module accepting (N, C, T, V) input

    Raises:
        ValueError: if num_actions != 1 is requested for STGCLSATT.
    """
    name = config["model"]["name"]
    if name in BASELINE_CLASSES:
        return BASELINE_CLASSES[name](
            config, use_fc=use_fc, num_actions=num_actions)
    if num_actions != 1:
        raise ValueError(
            "num_actions != 1 is only supported for baseline models; "
            "use EarlyFusion for {}".format(name))
    return STGCLSATT(config=config, use_fc=use_fc)


def get_encoder_feature_dim(config) -> int:
    """Feature width of the encoder output when built with use_fc=False."""
    name = config["model"]["name"]
    if name in BASELINE_CLASSES:
        return BASELINE_CLASSES[name].get_feature_dim(config)
    return STGCLSATT.get_feature_dim(config)
