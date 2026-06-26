# Visualization

`xcalib` includes lightweight visualization helpers for the same ideas shown on
the docs home page: matching links between camera detections and LiDAR
detections, and calibrated LiDAR projection onto the camera image.

The helpers return RGB `uint8` images. Save them with Pillow, show them in a
notebook, or convert to BGR before using OpenCV writers.

## Matching Overlay

```python
from PIL import Image
from xcalib import Matcher, draw_matching_overlay

matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
result = matcher.match(image, point_cloud, bboxes_2d, bboxes_3d, top_k=1)

overlay = draw_matching_overlay(
    image=image,
    point_cloud=point_cloud,
    bboxes_2d=bboxes_2d,
    bboxes_3d=bboxes_3d,
    matches=result.matches,
    match_threshold=0.5,
)
Image.fromarray(overlay).save("matching_overlay.png")
```

This produces a side-by-side panel: camera detections on the left, LiDAR
bird's-eye detections on the right, and green links for selected matches.

## Calibration Projection

Use a solved `CalibrationResult`:

```python
from PIL import Image
from xcalib import CameraIntrinsics, Matcher, draw_calibration_overlay

matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
K = CameraIntrinsics(fx=1418.0, fy=1422.0, cx=976.0, cy=606.0)

calibration = matcher.calibrate(
    image,
    point_cloud,
    bboxes_2d,
    bboxes_3d,
    intrinsics=K,
)

overlay = draw_calibration_overlay(
    image=image,
    point_cloud=point_cloud,
    intrinsics=K,
    rotation=calibration.rotation,
    translation=calibration.translation,
    bboxes_3d=bboxes_3d,
)
Image.fromarray(overlay).save("calibration_overlay.png")
```

Or pass an existing `3x4` projection matrix directly:

```python
overlay = draw_calibration_overlay(
    image=image,
    point_cloud=point_cloud,
    projection=P,
    bboxes_3d=bboxes_3d,
)
```

The projection overlay draws colorized LiDAR points and optional LiDAR
detection centers on the camera image. It is meant for qualitative inspection:
for quantitative validation, use reprojection error from `CalibrationResult`.

## Demo Media

The animated GIFs on the home page are produced by
`scripts/make_demo_media.py`, which adds timeline/HUD effects around the same
matching and projection concepts. Use the package helpers above for notebooks,
reports, and integration tests where a single diagnostic image is enough.
