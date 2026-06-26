"""
PointNet2 — hierarchical LiDAR backbone used by the CalibRefine baseline.

Verbatim port of src/core/models/calibrefine/backbones/pointnet2.py. The
sampling/grouping ops live in `_pointnet2_ops.py` (pure-PyTorch, portable).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._pointnet2_ops import sample_and_group, sample_and_group_all


class PointNetSetAbstraction(nn.Module):
    """A single Set Abstraction layer."""

    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channel: int,
        mlp,
        group_all: bool,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz: torch.Tensor, points):
        # xyz: [B, 3, N], points: [B, D, N] or None
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2(nn.Module):
    """PointNet++ classification network used by the CalibRefine baseline.

    Returns:
        embed_feat: [B, 512] embedding for matching
        cls_feat:   [B, num_class] logits
        cls_res:    [B, num_class] log-softmax
        l3_points:  [B, 1024] global SA3 feature
    """

    def __init__(self, num_class: int, normal_channel: bool = True):
        super().__init__()
        self.normal_channel = normal_channel
        in_channel = 6 if normal_channel else 3

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True,
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, num_class)

    def forward(self, xyz: torch.Tensor):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        embed_feat = self.drop2(F.relu(self.bn2(self.fc2(x))))
        cls_feat = self.fc3(embed_feat)
        cls_res = F.log_softmax(cls_feat, -1)

        return embed_feat, cls_feat, cls_res, l3_points
