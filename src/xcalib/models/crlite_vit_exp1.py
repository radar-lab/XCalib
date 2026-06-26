"""
CRLite-ViT Exp1 — ViT image encoder + PointNet LiDAR encoder, no
position encoding. Pure dot-product cosine similarity for matching.

Standalone port of src/core/models/crlite_vit_exp1/model.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ._base import EdgeModelBase
from ..utils.config import EdgeConfig
from .backbones import CropViTEncoder, PointNetEmbedding


@dataclass(frozen=True)
class _ViTCfg:
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    pool: str = "cls"


class CRLiteViTExp1Model(EdgeModelBase):
    """ViT + PointNet, dot-product matching (no PE)."""

    model_key = "crlite_vit_exp1"

    def __init__(self, config: EdgeConfig):
        super().__init__(config)

        self.embed_dim = int(config.get("embed_dim", 256))
        self.top_k = int(config.get("top_k", 3))

        self.vit_cfg = _ViTCfg(
            image_size=int(config.get("crop_size", 32)),
            patch_size=int(config.get("vit.patch_size", 4)),
            embed_dim=int(config.get("vit.embed_dim", self.embed_dim)),
            depth=int(config.get("vit.depth", 6)),
            num_heads=int(config.get("vit.num_heads", 8)),
            mlp_ratio=float(config.get("vit.mlp_ratio", 4.0)),
            dropout=float(config.get("vit.dropout", 0.0)),
            attention_dropout=float(config.get("vit.attention_dropout", 0.0)),
            pool=str(config.get("vit.pool", "cls")),
        )

        self.build_model()
        self.to(self.device)

        logger.info(
            f"Built {type(self).__name__} "
            f"(embed_dim={self.embed_dim}, crop={self.vit_cfg.image_size}, "
            f"patch={self.vit_cfg.patch_size}, top_k={self.top_k}) on {self.device}"
        )

    def build_model(self) -> None:
        self.img_backbone = CropViTEncoder(
            image_size=self.vit_cfg.image_size,
            patch_size=self.vit_cfg.patch_size,
            embed_dim=self.vit_cfg.embed_dim,
            depth=self.vit_cfg.depth,
            num_heads=self.vit_cfg.num_heads,
            mlp_ratio=self.vit_cfg.mlp_ratio,
            dropout=self.vit_cfg.dropout,
            attention_dropout=self.vit_cfg.attention_dropout,
            pool=self.vit_cfg.pool,
        )
        self.lidar_backbone = PointNetEmbedding(embed_dim=self.embed_dim)

        self.img_proj: Optional[nn.Module] = None
        if self.vit_cfg.embed_dim != self.embed_dim:
            self.img_proj = nn.Sequential(
                nn.Linear(self.vit_cfg.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
            )

    def extract_features(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,  # unused, kept for API parity
        lid_centers: Optional[torch.Tensor] = None,  # unused, kept for API parity
    ) -> Dict[str, torch.Tensor]:
        img_embed = self.img_backbone(images)
        if self.img_proj is not None:
            img_embed = self.img_proj(img_embed)

        if pcds.dim() != 3:
            raise ValueError(f"Expected pcds [M,P,3] or [M,3,P], got {tuple(pcds.shape)}")
        if pcds.shape[2] == 3:
            pcds = pcds.permute(0, 2, 1)
        lid_embed = self.lidar_backbone(pcds)
        return {"img_embed": img_embed, "lid_embed": lid_embed}

    @staticmethod
    def _dot_similarity(img_embed: torch.Tensor, lid_embed: torch.Tensor) -> torch.Tensor:
        img = F.normalize(img_embed, p=2, dim=1)
        lid = F.normalize(lid_embed, p=2, dim=1)
        return img @ lid.t()

    def forward_train(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del img_centers, lid_centers
        features = self.extract_features(images, pcds)
        sim = self._dot_similarity(features["img_embed"], features["lid_embed"])
        return {"stage1_similarity": sim}

    def forward_inference(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.extract_features(images, pcds)
        sim = self._dot_similarity(features["img_embed"], features["lid_embed"])
        return {"similarity": sim, "stage1_similarity": sim}

    def forward(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_inference(images, pcds)
