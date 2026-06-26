"""
Tiny shared base for all standalone matching models.

Replaces the lab's `BaseModel` with the absolute minimum needed at inference
time: device resolution and a thin checkpoint-loading helper. No training
metadata, no optimizer plumbing, no `set_input_specs` bookkeeping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from loguru import logger

from ..utils.config import EdgeConfig
from ..utils.io import resolve_device, load_state_dict_from


class EdgeModelBase(nn.Module):
    """Base class for every standalone matching model."""

    #: Subclasses set this to the dotted-key prefix used in the config
    #: (e.g. "crlite", "crlite_vit_exp3").
    model_key: str = ""

    def __init__(self, config: EdgeConfig):
        super().__init__()
        if not self.model_key:
            raise RuntimeError(f"{type(self).__name__}.model_key must be set")
        self.config = config

        # Resolve device: prefer model-specific, then global, default "auto"
        device_spec = (
            config.get(f"{self.model_key}.device")
            or config.get("device")
            or "auto"
        )
        self.device = resolve_device(device_spec)

    def load_weights(
        self,
        weights_path: str | Path | Mapping[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        """Load a checkpoint (any of the 3 lab formats) into this model.

        Accepts either a path on disk or an already-extracted state dict
        (the matcher pre-loads the checkpoint once so it can also pick up
        one-shot adapter weights stored alongside the model state).
        """
        if isinstance(weights_path, Mapping):
            state = weights_path
        else:
            state = load_state_dict_from(weights_path, device="cpu", strict=strict)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            logger.warning(
                f"{type(self).__name__}: {len(missing)} missing keys "
                f"(first 5: {list(missing)[:5]})"
            )
        if unexpected:
            logger.warning(
                f"{type(self).__name__}: {len(unexpected)} unexpected keys "
                f"(first 5: {list(unexpected)[:5]})"
            )
        self.eval()

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        self.eval()
        return self.forward(*args, **kwargs)
