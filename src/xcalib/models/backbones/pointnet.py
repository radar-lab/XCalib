"""
PointNetEmbedding — LiDAR backbone used by CRLite / CRLite-2DPE /
CRLite-ViT (Exp1, Exp3, Exp4).

Verbatim port of src/core/models/crlite/backbones/pointnet.py. State-dict
keys are preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEmbedding(nn.Module):
    """PointNet embedding with spatial attention."""

    def __init__(self, embed_dim: int = 256, input_channel: int = 3):
        super().__init__()

        # Multi-scale feature extraction
        self.conv1 = nn.Conv1d(input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Spatial attention head
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 1, 1),
            nn.Sigmoid(),
        )

        # Projection head
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, embed_dim)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(embed_dim)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # Accept [B,3,N] or [B,N,3]
        if xyz.dim() == 3 and xyz.shape[2] == 3 and xyz.shape[1] != 3:
            xyz = xyz.transpose(1, 2)
        # Drop intensity if present
        if xyz.shape[1] > 3:
            xyz = xyz[:, :3, :]

        x = F.relu(self.bn1(self.conv1(xyz)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 512, N]

        attn = self.spatial_attention(x)            # [B, 1, N]
        x_attended = x * attn                       # [B, 512, N]
        global_feat = torch.sum(x_attended, dim=2) / (torch.sum(attn, dim=2) + 1e-8)

        x = self.drop1(F.relu(self.bn5(self.fc1(global_feat))))
        embed = self.drop2(F.relu(self.bn6(self.fc2(x))))
        return embed
