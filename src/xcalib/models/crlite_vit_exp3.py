"""
CRLite-ViT Exp3 — ViT image encoder + PointNet LiDAR encoder + Enhanced
3D position encoding. Cosine-similarity matching with position-enhanced
features.

Standalone port of src/core/models/crlite_vit_exp3/model.py.
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
from .position_encoding import Enhanced3DPositionEncoding


@dataclass(frozen=True)
class _ViTCfg:
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    pool: str = "cls"


class CRLiteViTExp3Model(EdgeModelBase):
    """ViT + PointNet + Enhanced 3D position encoding, cosine matching."""

    model_key = "crlite_vit_exp3"

    def __init__(self, config: EdgeConfig):
        super().__init__(config)

        self.embed_dim = int(config.get("embed_dim", 256))
        self.token_len = int(config.get("token_len", 256))
        self.max_depth = float(config.get("max_depth", 100.0))
        self.top_k = int(config.get("top_k", 10))

        self.vit_cfg = _ViTCfg(
            image_size=int(config.get("crop_size", 64)),
            patch_size=int(config.get("vit.patch_size", 8)),
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
            f"(embed_dim={self.embed_dim}, token_len={self.token_len}, "
            f"crop={self.vit_cfg.image_size}, top_k={self.top_k}) on {self.device}"
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

        # Position encoder (matches Exp3 — no pe_mode argument in the lab code,
        # so we keep "full"; the same Enhanced3DPositionEncoding class is used).
        self.position_encoder = Enhanced3DPositionEncoding(
            token_len=self.token_len, max_depth=self.max_depth, pe_mode="full"
        )

        self.img_position_proj = nn.Sequential(
            nn.Linear(self.token_len, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )
        self.lid_position_proj = nn.Sequential(
            nn.Linear(self.token_len, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )

        self.img_backbone_proj: Optional[nn.Module] = None
        if self.vit_cfg.embed_dim != self.embed_dim:
            self.img_backbone_proj = nn.Sequential(
                nn.Linear(self.vit_cfg.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
            )

        self.img_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )
        self.lid_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )

    def extract_features(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        img_backbone_feat = self.img_backbone(images)
        if self.img_backbone_proj is not None:
            img_backbone_feat = self.img_backbone_proj(img_backbone_feat)

        if pcds.dim() == 3 and pcds.shape[2] == 3:
            pcds_t = pcds.permute(0, 2, 1)
        else:
            pcds_t = pcds
        lid_backbone_feat = self.lidar_backbone(pcds_t)

        if img_centers is not None:
            img_pos_feat = self.position_encoder(img_centers[:, :2], lidar_points=None)
            img_pos_proj = self.img_position_proj(img_pos_feat)
            img_embed = self.img_fusion(
                torch.cat([img_backbone_feat, img_pos_proj], dim=1)
            )
        else:
            img_embed = img_backbone_feat

        if lid_centers is not None:
            lid_pos_feat = self.position_encoder(lid_centers[:, :2], pos_3d=lid_centers)
            lid_pos_proj = self.lid_position_proj(lid_pos_feat)
            lid_embed = self.lid_fusion(
                torch.cat([lid_backbone_feat, lid_pos_proj], dim=1)
            )
        else:
            lid_embed = lid_backbone_feat

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
        features = self.extract_features(images, pcds, img_centers, lid_centers)
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
        features = self.extract_features(images, pcds, img_centers, lid_centers)
        sim = self._dot_similarity(features["img_embed"], features["lid_embed"])
        return {"similarity": sim, "stage1_similarity": sim}

    def forward(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_inference(images, pcds, img_centers, lid_centers)
