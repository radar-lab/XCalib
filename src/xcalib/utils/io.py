"""
Shared utilities: device resolution and checkpoint state-dict loading.

The checkpoint loader mirrors the three formats handled by the lab's
`BaseModel.load_weights`:
    1. full checkpoint with `state_dict`
    2. alternative format with `model_state_dict`
    3. raw state_dict

Author: Lihao Guo (leolihao@arizona.edu)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from loguru import logger


def package_dir() -> Path:
    """Directory of the installed/checked-out `xcalib` package."""
    return Path(__file__).resolve().parent.parent


def default_config_path(model: str, site: str = "utc4") -> Path:
    """Resolve the reference YAML for a (model, site) pair.

    Configs ship as package data under ``xcalib/cfg/`` — the same path in a
    wheel install and in an (editable) source checkout.
    """
    path = package_dir() / "cfg" / f"{model}_{site}.yaml"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"No reference config found for model='{model}', site='{site}' "
        f"(looked for {path}). Pass an explicit config path instead."
    )


def resolve_device(spec: str | torch.device | None) -> torch.device:
    """Resolve "auto"/"cuda"/"cpu"/etc. to a concrete torch.device."""
    if isinstance(spec, torch.device):
        return spec
    s = (spec or "auto").lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    """Load a checkpoint file and return the full (un-extracted) payload."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unexpected checkpoint root type: {type(checkpoint).__name__}")
    return checkpoint


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Pull the model state_dict out of any of the 3 lab checkpoint shapes:

        - {"state_dict": {...}, "epoch": ..., ...}
        - {"model_state_dict": {...}, "epoch": ..., ...}
        - {parameter_name: tensor, ...}  (raw)
    """
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], Mapping):
        return checkpoint["model_state_dict"]
    return checkpoint


def load_state_dict_from(
    path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = False,
) -> Mapping[str, torch.Tensor]:
    """Load a checkpoint file and return its state_dict.

    `strict` is reported but not enforced here — callers decide whether to
    pass it to `nn.Module.load_state_dict`.
    """
    path = Path(path)
    state = extract_state_dict(load_checkpoint(path, device=device))
    logger.info(f"Loaded checkpoint: {path} (strict={strict}, num_keys={len(state)})")
    return state
