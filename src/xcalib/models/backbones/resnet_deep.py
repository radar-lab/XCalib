"""
ResNetEmbeddingDeep — image backbone used by CRLite / CRLite-2DPE.

Verbatim port of src/core/models/crlite/backbones/resnet.py with no
external imports. State-dict keys are preserved so checkpoints trained with
the lab code load directly here.
"""

from __future__ import annotations

import torch.nn as nn


class ResNetEmbeddingDeep(nn.Module):
    """Deeper ResNet-style image backbone (3 -> 256 -> embed_dim)."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.embed_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.embed_head(self.features(x))
