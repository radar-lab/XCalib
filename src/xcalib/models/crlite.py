"""
CRLite — Hybrid two-stage matching model with Enhanced 3D Position Encoding.

Standalone port of src/core/models/crlite/model.py. Removes the lab's
BaseModel inheritance, the unused training metadata, and auto-load
behaviour. State-dict keys are preserved so existing UTC3/UTC4 checkpoints
load directly.

Configuration is read from a flat YAML (top-level keys), e.g.:

    model: crlite
    embed_dim: 256
    token_len: 256
    top_k: 5
    max_depth: 100.0
    pe_mode: full
    num_classes: 13
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ._base import EdgeModelBase
from ..utils.config import EdgeConfig
from .backbones import PointNetEmbedding, ResNetEmbeddingDeep
from .position_encoding import Enhanced3DPositionEncoding


class CRLiteModel(EdgeModelBase):
    """Hybrid two-stage CRLite (ResNet + PointNet + Enhanced 3D PE)."""

    model_key = "crlite"

    def __init__(self, config: EdgeConfig):
        super().__init__(config)

        self.num_class = int(config.get("num_classes", 13))
        self.embed_dim = int(config.get("embed_dim", 256))
        self.token_len = int(config.get("token_len", 256))
        self.top_k = int(config.get("top_k", 3))
        self.max_depth = float(config.get("max_depth", 100.0))
        self.pe_mode = str(config.get("pe_mode", "full"))
        self.embed_fusion_dropout = float(config.get("embed_fusion.dropout", 0.1))
        self.similarity_dropout = float(config.get("similarity_head.dropout", 0.1))
        self.similarity_hidden_dims = self._int_list_or_none(
            config.get("similarity_head.hidden_dims")
        )

        self.build_model()
        self.to(self.device)

        logger.info(
            f"Built {type(self).__name__} "
            f"(embed_dim={self.embed_dim}, token_len={self.token_len}, "
            f"top_k={self.top_k}, pe_mode={self.pe_mode}, "
            f"similarity_hidden={self._similarity_hidden_dims()}) on {self.device}"
        )

    @staticmethod
    def _int_list_or_none(value: Any) -> Optional[list[int]]:
        """Parse optional architecture lists from YAML/dict config."""
        if value is None:
            return None
        if isinstance(value, str):
            value = [x.strip() for x in value.split(",") if x.strip()]
        if isinstance(value, (int, float)):
            value = [value]
        dims = [int(v) for v in value]
        if not dims or any(v <= 0 for v in dims):
            raise ValueError("similarity_head.hidden_dims must contain positive ints")
        return dims

    def _similarity_hidden_dims(self) -> list[int]:
        return self.similarity_hidden_dims or [
            self.embed_dim // 2,
            self.embed_dim // 4,
            self.embed_dim // 8,
        ]

    @staticmethod
    def _dense_head(
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        *,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def build_model(self) -> None:
        self.img_backbone = ResNetEmbeddingDeep(embed_dim=self.embed_dim)
        self.lidar_backbone = PointNetEmbedding(embed_dim=self.embed_dim)

        self.position_encoder = Enhanced3DPositionEncoding(
            token_len=self.token_len,
            max_depth=self.max_depth,
            pe_mode=self.pe_mode,
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

        # Stage 2: pairwise refinement
        self.embed_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.embed_fusion_dropout),
        )

        # Kept so checkpoints with the aux-classifier load cleanly.
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.num_class),
        )

        if self.similarity_hidden_dims is None:
            # Preserve the exact default module indices for released checkpoints.
            # The old hardcoded nn.Sequential is reproduced verbatim here — do NOT
            # refactor this into a call to _dense_head().  Released UTC3/UTC4/A9
            # state_dicts depend on this precise module index layout, and any
            # change (even functionally equivalent) would break load_state_dict().
            self.similarity_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.LayerNorm(self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.similarity_dropout),
                nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
                nn.LayerNorm(self.embed_dim // 4),
                nn.ReLU(),
                nn.Dropout(self.similarity_dropout),
                nn.Linear(self.embed_dim // 4, self.embed_dim // 8),
                nn.LayerNorm(self.embed_dim // 8),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 8, 1),
            )
        else:
            self.similarity_head = self._dense_head(
                self.embed_dim,
                self.similarity_hidden_dims,
                1,
                dropout=self.similarity_dropout,
            )

    # ----- inference primitives -----

    def extract_features(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        img_backbone_feat = self.img_backbone(images)

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
    def stage1_forward(features: Dict[str, torch.Tensor]) -> torch.Tensor:
        img_norm = F.normalize(features["img_embed"], p=2, dim=1)
        lid_norm = F.normalize(features["lid_embed"], p=2, dim=1)
        return img_norm @ lid_norm.t()

    def stage2_forward_candidates(
        self, features: Dict[str, torch.Tensor], candidates: torch.Tensor
    ) -> torch.Tensor:
        N, k = candidates.shape
        img_embed = features["img_embed"].unsqueeze(1).expand(-1, k, -1)
        lid_embed = features["lid_embed"][candidates]
        combined = torch.cat(
            [img_embed.reshape(N * k, -1), lid_embed.reshape(N * k, -1)], dim=1
        )
        fused = self.embed_fusion(combined)
        scores = self.similarity_head(fused).squeeze(1)
        return scores.reshape(N, k)

    def stage2_forward_all(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper used by the pairwise-style validation script (N×M Stage 2)."""
        N = features["img_embed"].size(0)
        M = features["lid_embed"].size(0)
        img_exp = features["img_embed"].unsqueeze(1).expand(-1, M, -1).reshape(N * M, -1)
        lid_exp = features["lid_embed"].unsqueeze(0).expand(N, -1, -1).reshape(N * M, -1)
        combined = torch.cat([img_exp, lid_exp], dim=1)
        fused = self.embed_fusion(combined)
        scores = self.similarity_head(fused).squeeze(1)
        return scores.reshape(N, M)

    # ----- high-level forward (inference) -----

    def forward_inference(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Two-stage inference matching `src/core/models/crlite/model.py`.

        Stage 1 produces an N×M cosine similarity; we then re-rank the top-k
        candidates per image with Stage 2 and pack the result back into a
        full N×M matrix where non-top-k entries get a sentinel value of -1.
        """
        if top_k is None:
            top_k = self.top_k

        features = self.extract_features(images, pcds, img_centers, lid_centers)
        N = features["img_embed"].size(0)
        M = features["lid_embed"].size(0)
        device = images.device

        stage1_sim = self.stage1_forward(features)
        k = min(top_k, M)
        _, top_indices = stage1_sim.topk(k, dim=1)
        stage2_scores = self.stage2_forward_candidates(features, top_indices)

        # Match the working dtype (FP16 after model.half()) so the per-row
        # index_put below does not raise a Float<->Half mismatch on Thor.
        final_sim = torch.full((N, M), -1.0, device=device, dtype=stage1_sim.dtype)
        for i in range(N):
            row = stage2_scores[i]
            rmin, rmax = row.min(), row.max()
            if rmax > rmin:
                normalized = (row - rmin) / (rmax - rmin)
            else:
                normalized = torch.full_like(row, 0.5)
            final_sim[i, top_indices[i]] = normalized

        return {
            "similarity": final_sim,
            "stage1_similarity": stage1_sim,
            "stage2_scores": stage2_scores,
            "top_indices": top_indices,
        }

    def forward_train(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training tensors for both stages (dense N$\times$M Stage~2 logits)."""
        features = self.extract_features(images, pcds, img_centers, lid_centers)
        stage1_sim = self.stage1_forward(features)
        stage2_sim = self.stage2_forward_all(features)
        return {"stage1_similarity": stage1_sim, "stage2_similarity": stage2_sim}

    def forward(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_inference(images, pcds, img_centers, lid_centers)
