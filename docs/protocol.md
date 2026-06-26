# xcalib input protocol — v1.0

This document is the data contract between the partner's perception stack and the `xcalib` package.
Its executable counterpart is `xcalib.protocol.validate_frame_inputs`, which every `Matcher.match()`
call runs by default (`validate="warn"`).

Versioning: the protocol version (`xcalib.protocol.PROTOCOL_VERSION`) is bumped on any breaking
change to this contract and noted in the release notes. Additive relaxations do not bump the
version.

## 1. Per-frame inputs

`matcher.match(image, point_cloud, bboxes_2d, bboxes_3d)` consumes one time-synchronized frame:

| Input         | Shape / dtype        | Contract                                                                                                                                                                                                       |
| ------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image`       | `[H, W, 3]` `uint8`  | **RGB** channel order (convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` — OpenCV decodes BGR). Values 0–255.                                                                                                |
| `point_cloud` | `[P, >=3]` `float32` | Columns 0–2 are X, Y, Z in **meters**; extra columns (intensity, ring, ...) are ignored. No NaN/Inf. Must be in the **same coordinate frame as `bboxes_3d`** (sensor or global — consistency is what matters). |
| `bboxes_2d`   | `[K, 4]` `float32`   | `(x1, y1, x2, y2)` in **pixel coordinates** of `image`, with `x1 < x2`, `y1 < y2`. One row per camera detection.                                                                                               |
| `bboxes_3d`   | `[M, 6]` `float32`   | Either axis-aligned extents `(xmin, ymin, zmin, xmax, ymax, zmax)` or center+dimensions `(cx, cy, cz, dx, dy, dz)`. One row per LiDAR detection.                                                               |

### Concrete file example

The public demo includes a tiny manifest-backed sample under
`demo/frames/a9_sample/`. Each frame directory uses ordinary files:

```text
frame_0000/
  image.png
  point_cloud.pcd
  detections.json
```

`image.png` can be any PNG/JPEG decoded to RGB `uint8`. OpenCV reads BGR, so
convert explicitly:

```python
import cv2

image_bgr = cv2.imread("image.png", cv2.IMREAD_COLOR)
if image_bgr is None:
    raise FileNotFoundError("image.png")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```

Pillow also works and already returns RGB when requested:

```python
from PIL import Image
import numpy as np

image = np.asarray(Image.open("image.png").convert("RGB"), dtype=np.uint8)
```

`point_cloud.pcd` should contain XYZ points in meters. ASCII PCD is easy to
inspect:

```text
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
7.100 -3.250 0.200
7.450 -3.250 0.450
7.800 -3.000 0.700
```

Use Open3D if it is already in your perception stack:

```python
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("point_cloud.pcd")
point_cloud = np.asarray(pcd.points, dtype=np.float32)
```

For minimal ASCII PCD files, the demo uses a small fallback loader:

```python
import numpy as np

def load_pcd_ascii(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() == "data ascii":
            data_start = idx + 1
            break
    else:
        raise ValueError("PCD file does not contain DATA ascii")
    return np.loadtxt(lines[data_start:], dtype=np.float32).reshape(-1, 3)
```

`detections.json` stores the camera detections and LiDAR detections for the
same frame:

```json
{
  "frame_key": "a9_sample_0000",
  "camera_name": "s110_camera_basler_south2_8mm",
  "image": "image.png",
  "point_cloud": "point_cloud.pcd",
  "bboxes_2d": [
    [42.0, 62.0, 96.0, 128.0],
    [138.0, 54.0, 198.0, 124.0]
  ],
  "bboxes_3d": [
    [6.7, -3.6, -0.2, 9.3, -2.4, 1.6],
    [13.7, -1.1, -0.2, 16.3, 0.1, 1.6]
  ]
}
```

The JSON-to-API conversion is:

```python
import json
from pathlib import Path

import numpy as np

from xcalib import Matcher

frame_dir = Path("demo/frames/a9_sample/frame_0000")
det = json.loads((frame_dir / "detections.json").read_text())

image = load_image_rgb(frame_dir / det["image"])
point_cloud = load_pcd_ascii(frame_dir / det["point_cloud"])
bboxes_2d = np.asarray(det["bboxes_2d"], dtype=np.float32).reshape(-1, 4)
bboxes_3d = np.asarray(det["bboxes_3d"], dtype=np.float32).reshape(-1, 6)

matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
result = matcher.match(image, point_cloud, bboxes_2d, bboxes_3d)
print(result.matches)
```

### 3D bbox format auto-detection

The package disambiguates the two `[M, 6]` conventions per row: when `bbox[3:6] >= bbox[0:3]`
element-wise, the row is treated as min/max extents, otherwise as center+dimensions. Mixed
conventions inside a single call are technically handled but **strongly discouraged** — pick one and
keep it. Note a center-style box at e.g. `(30, 5, 1)` with dims `(4, 2, 2)` looks like
`second < first` so it parses correctly, but a center-style box whose dims all exceed its center
coordinates would be misread as extents; with metric, road-scene boxes this does not occur in
practice.

### Synchronization & framing requirements

- Image and point cloud must come from the **same trigger window**; the matchers tolerate normal ITS
  jitter (≤ ~50 ms at 10 Hz) but are not built to match across frames.
- The LiDAR points inside each 3D box are what the model sees — boxes are expanded by
  `bbox_expansion` (default 1.25×, per-model YAML) and points inside are resampled to
  `point_cloud_size` (default 1024).
- Detections are produced upstream (the package does no detection).

## 2. Quality floors (soft — warnings)

| Check                | Floor                           | Why                                                                                                                                                 |
| -------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image resolution     | ≥ 640 × 480                     | Training data was full-HD ITS footage; small frames upsample crops.                                                                                 |
| 2D bbox size         | ≥ 8 px per side                 | Crops are resized to `crop_size` (32 px default); tiny boxes carry no texture.                                                                      |
| Points per 3D box    | ≥ ~50 inside the (expanded) box | PointNet embeddings degrade on near-empty crops; empty crops are dropped.                                                                           |
| Detections per frame | ≤ 32 image, ≤ 32 LiDAR          | Matches the TensorRT dynamic-shape `maxShapes` the Thor engines are built with (`xcalib/engine/trt.py`). PyTorch/ONNX-RT inference has no hard cap. |

Violating a floor logs a warning (once per kind); inference still runs. Hard violations — wrong
rank/dtype, NaN/Inf coordinates — raise `xcalib.ProtocolError` instead.

```python
result = matcher.match(image, pc, b2, b3)                 # validate="warn" (default)
result = matcher.match(image, pc, b2, b3, validate="strict")  # any violation raises
result = matcher.match(image, pc, b2, b3, validate="off")     # trusted hot path
```

To pre-flight a recording without running inference:

```python
from xcalib import validate_frame_inputs
for v in validate_frame_inputs(image, pc, b2, b3):
    print(v)   # [warning] bboxes_2d.small: 2 bbox(es) are smaller than 8px ...
```

## 3. Camera intrinsics (calibrate / one-shot only)

`matcher.match()` never needs intrinsics. `matcher.calibrate()` and `matcher.oneshot()` do — they
solve / use the camera-LiDAR projection:

```python
from xcalib import CameraIntrinsics

K = CameraIntrinsics(fx=2666.7, fy=2666.7, cx=960.0, cy=540.0)        # pixels
K = CameraIntrinsics.from_matrix(K_3x3, distortion=np.array([k1, k2, p1, p2, k3]))
```

- Intrinsics are **fixed and known** (factory calibration); this package estimates only the
  camera↔LiDAR **extrinsics** `[R|t]` and reports `P = K [R|t]`.
- Distortion coefficients follow OpenCV ordering and are optional; if the image stream is already
  rectified, omit them.
- The solved extrinsics map **`bboxes_3d`-frame coordinates to the camera frame** — i.e. whatever
  frame the 3D boxes/point cloud were given in.

## 4. Outputs

`MatchResult.similarity` is `[K', M']` where `K' <= K`, `M' <= M` after degenerate/empty detections
are dropped; `kept_2d_indices` / `kept_3d_indices` map the surviving rows/columns back to the
caller's original indices, and entries of `matches` are already expressed in the caller's numbering.
Score semantics per model: cosine in `[-1, 1]` for the ViT models, normalized Stage-2 scores in
`[0, 1]` (non-top-K entries are `-1`) for `crlite`/`crlite_2dpe`, sigmoid pair probability in
`[0, 1]` for `calibrefine`. Ranking, not absolute magnitude, is the supported signal; threshold per
model after a short on-site calibration run.

## 5. TensorRT engine envelope (Thor deployment)

The shipped engines are built with these dynamic-shape profiles (`min / opt / max`):

| Tensor                        | min | opt | max  |
| ----------------------------- | --- | --- | ---- |
| `image_crops` (N)             | 1   | 8   | 32   |
| `lidar_crops` (M)             | 1   | 12  | 32   |
| stage-2 pairs (B = N·K)       | 1   | 80  | 320  |
| `calibrefine` pairs (B = N·M) | 1   | 96  | 1024 |

Frames outside the envelope must be split or truncated **before** the engine call; the PyTorch and
ONNX-Runtime paths accept any size.
