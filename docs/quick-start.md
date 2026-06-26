# Quick start

Get a camera–LiDAR match in three steps.

## 1. Install

```bash
pip install XCalib
```

Requires Python 3.10+. For ONNX export add `XCalib[onnx]`; for HDF5 training add
`XCalib[train]`.

## 2. Load a pretrained matcher

```python
from xcalib import Matcher

# Downloads and caches the released checkpoint + config on first use.
matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
```

## 3. Match one frame

Pass an RGB image, a LiDAR point cloud, and your detector's 2D/3D boxes:

```python
import numpy as np

result = matcher.match(
    image=image,             # H,W,3 uint8 RGB
    point_cloud=points,      # N,3 float32 (x, y, z) in meters
    bboxes_2d=bboxes_2d,     # K,4 (x1, y1, x2, y2) in pixels
    bboxes_3d=bboxes_3d,     # M,6 (xmin, ymin, zmin, xmax, ymax, zmax) in meters
)

print(result.similarity.shape)  # (K, M) similarity matrix
print(result.matches)           # [(img_idx, lidar_idx, score), ...]
```

That's it — `result.matches` gives the camera box ↔ LiDAR box pairs.

## Next steps

- [Input protocol](protocol.md) — exact shape/dtype contract for real frames.
- [Models and datasets](hub.md) — available checkpoints and sites.
- [Custom models](adding-models.md) — load your own weights or add an architecture.
