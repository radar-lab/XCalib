"""
Minimal model-name -> class registry.

Used by `Matcher.from_pretrained` and the validation script so that
both the partner and the lab can spell a model the same way (e.g. "crlite",
"calibrefine", "crlite_vit_exp3").
"""

from __future__ import annotations

from typing import Dict, Type

from ..utils.config import EdgeConfig
from . import (
    CalibRefineModel,
    CRLite2DPEModel,
    CRLiteModel,
    CRLiteViTExp1Model,
    CRLiteViTExp3Model,
    CRLiteViTExp4Model,
)
from ._base import EdgeModelBase


_MODEL_REGISTRY: Dict[str, Type[EdgeModelBase]] = {
    "crlite": CRLiteModel,
    "crlite_2dpe": CRLite2DPEModel,
    "crlite_vit_exp1": CRLiteViTExp1Model,
    "crlite_vit_exp3": CRLiteViTExp3Model,
    "crlite_vit_exp4": CRLiteViTExp4Model,
    "calibrefine": CalibRefineModel,
}


def list_models() -> list[str]:
    """Names of every model the registry knows about."""
    return sorted(_MODEL_REGISTRY.keys())


def get_model_class(name: str) -> Type[EdgeModelBase]:
    """Look up a model class by its short name."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {', '.join(list_models())}"
        )
    return _MODEL_REGISTRY[name]


def build_model(name: str, config: EdgeConfig) -> EdgeModelBase:
    """Instantiate `name` from `config` (no weights loaded yet)."""
    cls = get_model_class(name)
    return cls(config)
