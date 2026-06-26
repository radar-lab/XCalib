"""Backbones used by the standalone matching models.

Each backbone is a verbatim port (with only docstring/import changes) of the
corresponding file under `src/core/models/`. No upstream imports.
"""

from .resnet_deep import ResNetEmbeddingDeep
from .resnet_bottleneck import Bottleneck, ResNet
from .pointnet import PointNetEmbedding
from .pointnet2 import PointNet2
from .vit_crop import CropViTEncoder

__all__ = [
    "ResNetEmbeddingDeep",
    "Bottleneck",
    "ResNet",
    "PointNetEmbedding",
    "PointNet2",
    "CropViTEncoder",
]
