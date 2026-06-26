"""
CRLite-2DPE — CRLite with the depth-MLP and 3D-MLP branches of the
position encoder zeroed (only the 2D sinusoidal branch remains active).

This is the R1 #6 ablation in the manuscript. Architecturally identical
to CRLite, so the same `CRLiteModel` is reused; we only force
`pe_mode = "2d_only"` regardless of the YAML.
"""

from __future__ import annotations

from ..utils.config import EdgeConfig
from .crlite import CRLiteModel


class CRLite2DPEModel(CRLiteModel):
    """CRLite locked to the `2d_only` position-encoding ablation."""

    model_key = "crlite_2dpe"

    def __init__(self, config: EdgeConfig):
        config.set("pe_mode", "2d_only")
        super().__init__(config)
