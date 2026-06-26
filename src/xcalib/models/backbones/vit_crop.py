"""
CropViTEncoder — small ViT used by CRLite-ViT (Exp1, Exp3).

Verbatim port of src/core/models/crlite_vit_exp1/backbones/vit_crop.py.
State-dict keys are preserved so the partner can load the trained weights
without renaming.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CropViTEncoder(nn.Module):
    """Lightweight ViT encoder for fixed-size crops.

    Args:
        image_size: square crop size (e.g. 32 or 64)
        patch_size: patch side length (must divide image_size)
        embed_dim: token dimension
        depth: number of transformer encoder layers
        num_heads: number of attention heads
        mlp_ratio: feed-forward hidden ratio inside each encoder layer
        dropout: dropout probability
        attention_dropout: kept for signature parity (unused by nn.TransformerEncoder)
        pool: "cls" | "mean"
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.pool = pool

        grid = self.image_size // self.patch_size
        num_patches = grid * grid

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input [B,3,H,W], got {tuple(x.shape)}")

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, T, D]

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        if x.size(1) != self.pos_embed.size(1):
            raise ValueError(
                f"Token length mismatch: got {x.size(1)} tokens, "
                f"but pos_embed has {self.pos_embed.size(1)}"
            )

        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)

        if self.pool == "cls":
            return x[:, 0]
        if self.pool == "mean":
            return x[:, 1:].mean(dim=1)
        raise ValueError(f"Unknown pool mode: {self.pool} (expected 'cls' or 'mean')")
