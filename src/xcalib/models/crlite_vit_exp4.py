"""
CRLite-ViT Exp4 — ViT image encoder + PointNet LiDAR encoder + structure-
aware GNN heads (replaces absolute position encoding).

Standalone port of `src/core/models/crlite_vit_exp4/model.py`. State-dict
keys are preserved exactly so the trained checkpoint loads without
renaming. No dependency on `src/`.

Architecture::

    ViT(image_crops)   -> [N, D] backbone features
    PointNet(pc_crops) -> [M, D] backbone features
        |                   |
        v                   v
    CameraGNN(features, bboxes_2d)
    LiDARGNN(features, centers_3d)
        |                   |
        v                   v
    L2-normalised cosine similarity -> [N, M]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ._base import EdgeModelBase
from ..utils.config import EdgeConfig
from .backbones import CropViTEncoder, PointNetEmbedding
from .gnn import CameraGNN, LiDARGNN


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


@dataclass(frozen=True)
class _GNNCfg:
    edge_dim: int = 8
    hidden_dim: int = 128
    num_layers: int = 2
    k_neighbors: int = 4
    dropout: float = 0.1


class CRLiteViTExp4Model(EdgeModelBase):
    """ViT + PointNet + Structure-Aware GNN (Exp4)."""

    model_key = "crlite_vit_exp4"

    def __init__(self, config: EdgeConfig):
        super().__init__(config)

        self.embed_dim = int(config.get("embed_dim", 256))
        self.top_k = int(config.get("top_k", 10))

        self.image_size: Tuple[int, int] = (
            int(config.get("image_width", 1920)),
            int(config.get("image_height", 1080)),
        )

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

        self.gnn_cfg = _GNNCfg(
            edge_dim=int(config.get("gnn.edge_dim", 8)),
            hidden_dim=int(config.get("gnn.hidden_dim", 128)),
            num_layers=int(config.get("gnn.num_layers", 2)),
            k_neighbors=int(config.get("gnn.k_neighbors", 4)),
            dropout=float(config.get("gnn.dropout", 0.1)),
        )

        self.build_model()
        self.to(self.device)

        logger.info(
            f"Built {type(self).__name__} "
            f"(embed_dim={self.embed_dim}, crop={self.vit_cfg.image_size}, "
            f"patch={self.vit_cfg.patch_size}, "
            f"gnn_layers={self.gnn_cfg.num_layers}, "
            f"k={self.gnn_cfg.k_neighbors}, top_k={self.top_k}) "
            f"on {self.device}"
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

        self.camera_gnn = CameraGNN(
            embed_dim=self.embed_dim,
            edge_dim=self.gnn_cfg.edge_dim,
            hidden_dim=self.gnn_cfg.hidden_dim,
            num_layers=self.gnn_cfg.num_layers,
            k_neighbors=self.gnn_cfg.k_neighbors,
            dropout=self.gnn_cfg.dropout,
        )

        self.lidar_gnn = LiDARGNN(
            embed_dim=self.embed_dim,
            edge_dim=self.gnn_cfg.edge_dim,
            hidden_dim=self.gnn_cfg.hidden_dim,
            num_layers=self.gnn_cfg.num_layers,
            k_neighbors=self.gnn_cfg.k_neighbors,
            dropout=self.gnn_cfg.dropout,
        )

    def extract_backbone_features(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        img_embed = self.img_backbone(images)
        if self.img_proj is not None:
            img_embed = self.img_proj(img_embed)

        if pcds.dim() == 3 and pcds.shape[2] == 3:
            pcds = pcds.permute(0, 2, 1)
        lid_embed = self.lidar_backbone(pcds)
        return {"img_backbone_feat": img_embed, "lid_backbone_feat": lid_embed}

    def extract_features(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        bboxes_2d: Optional[torch.Tensor] = None,
        centers_3d: Optional[torch.Tensor] = None,
        sizes_3d: Optional[torch.Tensor] = None,
        camera_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        backbone = self.extract_backbone_features(images, pcds)
        img_backbone = backbone["img_backbone_feat"]
        lid_backbone = backbone["lid_backbone_feat"]

        if bboxes_2d is not None and bboxes_2d.size(0) > 0:
            if camera_ids is not None:
                img_embed = self._apply_camera_gnn_per_camera(
                    img_backbone, bboxes_2d, camera_ids
                )
            else:
                img_embed = self.camera_gnn(
                    img_backbone, bboxes_2d, image_size=self.image_size
                )
        else:
            img_embed = img_backbone

        if centers_3d is not None and centers_3d.size(0) > 0:
            lid_embed = self.lidar_gnn(lid_backbone, centers_3d, sizes=sizes_3d)
        else:
            lid_embed = lid_backbone

        return {
            "img_embed": img_embed,
            "lid_embed": lid_embed,
            "img_backbone_feat": img_backbone,
            "lid_backbone_feat": lid_backbone,
        }

    def _apply_camera_gnn_per_camera(
        self,
        node_features: torch.Tensor,
        bboxes_2d: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build a separate GNN per camera (used by multi-camera A9 setup).

        UTC datasets are single-camera so the partner won't hit this path,
        but we keep it for API parity with the lab model.
        """
        output = torch.zeros_like(node_features)
        unique_cameras = torch.unique(camera_ids)
        for cam_id in unique_cameras:
            cam_mask = camera_ids == cam_id
            cam_indices = cam_mask.nonzero(as_tuple=True)[0]
            if cam_indices.numel() == 0:
                continue
            cam_features = node_features[cam_indices]
            cam_bboxes = bboxes_2d[cam_indices]
            cam_out = self.camera_gnn(
                cam_features, cam_bboxes, image_size=self.image_size
            )
            output[cam_indices] = cam_out
        return output

    @staticmethod
    def _dot_similarity(
        img_embed: torch.Tensor, lid_embed: torch.Tensor
    ) -> torch.Tensor:
        img = F.normalize(img_embed, p=2, dim=1)
        lid = F.normalize(lid_embed, p=2, dim=1)
        return img @ lid.t()

    def forward_inference(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        bboxes_2d: Optional[torch.Tensor] = None,
        centers_3d: Optional[torch.Tensor] = None,
        sizes_3d: Optional[torch.Tensor] = None,
        camera_ids: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.extract_features(
            images,
            pcds,
            bboxes_2d=bboxes_2d,
            centers_3d=centers_3d,
            sizes_3d=sizes_3d,
            camera_ids=camera_ids,
        )
        sim = self._dot_similarity(features["img_embed"], features["lid_embed"])
        return {
            "similarity": sim,
            "stage1_similarity": sim,
            "img_features": features["img_embed"],
            "lid_features": features["lid_embed"],
        }

    def forward(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        bboxes_2d: Optional[torch.Tensor] = None,
        centers_3d: Optional[torch.Tensor] = None,
        camera_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_inference(
            images,
            pcds,
            bboxes_2d=bboxes_2d,
            centers_3d=centers_3d,
            camera_ids=camera_ids,
        )
