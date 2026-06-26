"""
Bottleneck-based ResNet used by the CalibRefine pairwise baseline.

Verbatim port of src/core/models/calibrefine/backbones/resnet.py. The
key arrangement of layers is preserved so trained checkpoints load directly.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 bottleneck residual block (expansion = 4)."""

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """Image backbone for CalibRefine. Outputs (embed_feat, cls_feat, cls_res).

    Architecture: 3x3 stem -> 4 residual stages -> AvgPool(4) -> Linear(2048,512)
    -> Linear(512, num_classes).
    """

    def __init__(self, block, num_blocks, num_strides=(1, 2, 2, 2), num_classes: int = 10):
        super().__init__()
        planes = [64, 128, 256, 512]
        self.in_planes = planes[0]

        # 3x3 stem (stride 1)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu = nn.ReLU(inplace=True)

        # 4 stages built dynamically
        for i in range(len(planes)):
            setattr(
                self,
                f"layer{i+1}",
                self._make_layer(block, planes[i], num_blocks[i], num_strides[i]),
            )

        # Custom two-stage output head matching PointNet2's output format
        self.linear2 = nn.Linear(512 * block.expansion, out_features=512)
        self.linear = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        embed_feat = self.linear2(out)
        cls_feat = self.linear(embed_feat)
        cls_res = F.log_softmax(cls_feat, -1)

        return embed_feat, cls_feat, cls_res
