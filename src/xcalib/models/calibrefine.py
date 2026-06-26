"""
CalibRefine — original pairwise discriminator (paper baseline).

Standalone port of the Common Feature Discriminator from CalibRefine:

    Cheng, Guo, Zhang, Bang, Harris, Hajij, Sartipi, Cao —
    "CalibRefine: Deep Learning-Based Online Automatic Targetless
    LiDAR-Camera Calibration with Iterative and Attention-Driven
    Post-Refinement", IEEE Trans. Instrumentation and Measurement,
    vol. 75, 2026. arXiv:2502.17648.
    Original code: https://github.com/radar-lab/Lidar_Camera_Automatic_Calibration

Uses a Bottleneck-ResNet image backbone, a PointNet2 LiDAR backbone, and a
2D sinusoidal position encoding. Inference scores every (image, LiDAR) pair
individually via fc1..fc6 — O(N×M) forward passes per frame.

The 2D sinusoidal computation is rewritten as pure tensor ops (was numpy
on CPU) to keep parity with the trained checkpoint while remaining
ONNX-exportable.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ._base import EdgeModelBase
from ..utils.config import EdgeConfig
from .backbones import Bottleneck, PointNet2, ResNet
from .position_encoding import _get_2d_position_encoding


class CalibRefineModel(EdgeModelBase):
    """Pairwise baseline (ResNet + PointNet2 + 2D sin PE + fc1..fc6)."""

    model_key = "calibrefine"

    def __init__(self, config: EdgeConfig):
        super().__init__(config)

        self.num_class = int(config.get("num_classes", 13))
        self.token_len = int(config.get("token_len", 256))
        # The lab's CalibRefine uses ResNet `num_blocks=[2,2,2,2]` regardless of dataset.
        num_blocks_cfg = config.get("num_blocks", [2, 2, 2, 2])
        self.num_blocks = list(num_blocks_cfg)
        self.dense_embed_dim = int(config.get("dense.embed_dim", 512))
        self.dense_fusion_dim = int(config.get("dense.fusion_dim", 512))
        self.dense_hidden_dim = int(config.get("dense.hidden_dim", 256))
        self.dense_drop_embed = float(config.get("dense.dropout_embed", 0.1))
        self.dense_drop_fusion = float(config.get("dense.dropout_fusion", 0.3))
        self.dense_drop_hidden = float(config.get("dense.dropout_hidden", 0.4))

        # Validate dense dims — 0 or negative would build an invalid Linear layer
        # that only fails opaquely inside forward.
        for name, val in [("dense.embed_dim", self.dense_embed_dim),
                          ("dense.fusion_dim", self.dense_fusion_dim),
                          ("dense.hidden_dim", self.dense_hidden_dim)]:
            if val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")

        self.build_model()
        self.to(self.device)

        logger.info(
            f"Built {type(self).__name__} "
            f"(num_class={self.num_class}, token_len={self.token_len}, "
            f"dense=({self.dense_embed_dim}, {self.dense_fusion_dim}, "
            f"{self.dense_hidden_dim})) on {self.device}"
        )

    def build_model(self) -> None:
        cls_dim = self.num_class
        pos_dim = self.token_len * 2

        self.resnet = ResNet(Bottleneck, self.num_blocks, num_classes=cls_dim)
        self.pointnet2 = PointNet2(cls_dim, normal_channel=False)

        self.fc1 = nn.Linear(1024, self.dense_embed_dim)
        self.bn1 = nn.LayerNorm(self.dense_embed_dim)
        self.drop1 = nn.Dropout(self.dense_drop_embed)

        self.fc2 = nn.Linear(cls_dim * 2, cls_dim * 2)
        self.bn2 = nn.LayerNorm(cls_dim * 2)

        self.fc3 = nn.Linear(pos_dim, pos_dim)
        self.bn3 = nn.LayerNorm(pos_dim)

        self.fc4 = nn.Linear(self.dense_embed_dim + cls_dim * 2 + pos_dim, self.dense_fusion_dim)
        self.bn4 = nn.LayerNorm(self.dense_fusion_dim)
        self.drop4 = nn.Dropout(self.dense_drop_fusion)

        self.fc5 = nn.Linear(self.dense_fusion_dim, self.dense_hidden_dim)
        self.bn5 = nn.LayerNorm(self.dense_hidden_dim)
        self.drop5 = nn.Dropout(self.dense_drop_hidden)

        self.fc6 = nn.Linear(self.dense_hidden_dim, 1)

    # ------------------- helpers -------------------

    def _2d_pe(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _get_2d_position_encoding(x, y, self.token_len)

    # ------------------- forward -------------------

    def forward(
        self,
        image_data: torch.Tensor,
        lidar_data: torch.Tensor,
        image_pos: torch.Tensor,
        lidar_pos: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Pairwise forward.

        Args:
            image_data: [B, 3, 32, 32] image crops.
            lidar_data: [B, N, 3] (or [B, 3, N]) point clouds.
            image_pos:  [B, 2] image bbox centers.
            lidar_pos:  [B, 2] lidar bbox centers (XY projection).
        Returns dict with `main_output` [B, 1] (raw similarity logit).
        """
        img_embed_feat, _, img_cls_res = self.resnet(image_data)
        img_pos_feat = self._2d_pe(image_pos[:, 0], image_pos[:, 1])

        if lidar_data.dim() == 3 and lidar_data.shape[2] == 3:
            lidar_data = lidar_data.permute(0, 2, 1)
        lidar_embed_feat, _, lidar_cls_res, l3_points = self.pointnet2(lidar_data)
        lidar_pos_feat = self._2d_pe(lidar_pos[:, 0], lidar_pos[:, 1])

        embed_feat = torch.cat([img_embed_feat, lidar_embed_feat], dim=1)
        cls_feat = torch.cat([img_cls_res, lidar_cls_res], dim=1)
        pos_feat = torch.cat([img_pos_feat, lidar_pos_feat], dim=1)

        embed_feat = self.drop1(F.relu(self.bn1(self.fc1(embed_feat))))
        cls_feat = self.drop1(F.relu(self.bn2(self.fc2(cls_feat))))
        pos_feat = self.drop1(F.relu(self.bn3(self.fc3(pos_feat))))

        feats = torch.cat([embed_feat, cls_feat, pos_feat], dim=1)
        feats = self.drop4(F.relu(self.bn4(self.fc4(feats))))
        feats = self.drop5(F.relu(self.bn5(self.fc5(feats))))

        main_output = self.fc6(feats)

        return {
            "main_output": main_output,
            "img_cls_output": img_cls_res,
            "lid_cls_output": lidar_cls_res,
            "l3_points": l3_points,
        }
