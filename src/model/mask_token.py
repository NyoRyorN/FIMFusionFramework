"""
mask_token.py - utilities for missing-action masks in fusion models

Provides the shared helpers for the remaining missing-action handling methods:
zero padding, padding mask, and training-time modality dropout.

The real mask (1 = action present, 0 = missing) is derived in the Feeder
from all-zero input slices; `apply_modality_dropout` merges it with the
pseudo-missing flags into an effective mask consumed by the fusion models.

Config: fusion.missing_representation, trainings.modality_dropout_p
"""

import torch


def apply_modality_dropout(mask: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly mark present actions as pseudo-missing during training.

    Merges real missing (mask == 0) with randomly dropped present actions
    into an effective mask. At least one originally present action is kept
    per sample to avoid zero-division / meaningless gradients in the
    aggregation step.

    Args:
        mask: real availability mask (N, A), 1 = present, 0 = missing
        p: drop probability per present action; p <= 0 returns mask as-is

    Returns:
        effective mask (N, A) with eff <= mask elementwise
    """
    if p <= 0:
        return mask
    keep = (torch.rand_like(mask) >= p).float()
    eff_mask = mask * keep
    # revive one random present action for samples dropped to all-zero
    dead = (eff_mask.sum(dim=1) == 0) & (mask.sum(dim=1) > 0)
    if dead.any():
        # rand * mask makes argmax pick only among originally present actions
        revive = torch.argmax(torch.rand_like(mask) * mask, dim=1)
        eff_mask[dead, revive[dead]] = 1.0
    return eff_mask


def masked_action_mean_late(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the action axis excluding missing actions (late fusion).

    Args:
        x: per-action branch outputs (N, C, A)
        mask: effective mask (N, A)

    Returns:
        masked mean (N, C); denominator is the present-action count,
        clamped to 1 so all-missing samples yield zeros instead of NaN
    """
    m = mask.view(mask.size(0), 1, mask.size(1))  # (N, 1, A)
    return (x * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
