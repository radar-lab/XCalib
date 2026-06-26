# Public A9 Demo

This folder is a small standalone demo for the accepted-paper public release.
It uses the local wheel under `wheels/`, a tiny committed A9-style sample under
`frames/a9_sample/`, and the released A9 model weights through `xcalib`.

The committed sample is for step-by-step usage walkthroughs. For realistic
paper-scale runs, place `data/a9_test.h5` locally or let `xcalib.load_dataset()`
resolve the released Hugging Face dataset cache. Optional frame extraction
utilities live in the private `radar-lab-ops` workspace.

## Steps

1. `run_00_prepare_assets.py`: check wheel, optional local weights, optional
   local dataset cache, and ONNX outputs.
2. `run_01_match_one_frame.py`: load one committed A9 sample frame and call
   `Matcher.match()`.
3. `run_02_calibrate_stream.py`: stream A9 frames through `matcher.oneshot()`
   and solve calibration.
4. `run_03_full_demo.py`: run matching and calibration together, like a small
   edge loop.

## Install

From inside this `demo/` folder:

```bash
pixi install
```

`pixi.toml` installs the local wheel:

```text
wheels/xcalib-0.1.0-py3-none-any.whl
```

## Run

From inside `demo/`:

```bash
pixi run run-00
pixi run run-01
pixi run run-02
pixi run run-03
```

By default the demo uses the committed sample frames when
`frames/a9_sample/manifest.json` exists:

```text
frames/a9_sample/
  manifest.json
  frame_0000/image.png
  frame_0000/point_cloud.pcd
  frame_0000/detections.json
```

The same scripts fall back to HDF5 loading when the sample manifest is absent.
For full A9 runs, use:

```text
data/a9_test.h5
weights/crlite_a9_dataset_r02_s01_best.pth
onnx_export/crlite_a9_dataset_r02_s01/
```

If the local HDF5 cache is missing, the demo can resolve released Hub artifacts
after the dataset license gate is complete.

## What The Demo Calls

Matching:

```python
result = matcher.match(
    frame.image,
    frame.point_cloud,
    frame.bboxes_2d,
    frame.bboxes_3d,
)
print(result.matches)
```

Calibration:

```python
session = matcher.oneshot(intrinsics, match_threshold=0.5)
report = session.observe(image, point_cloud, bboxes_2d, bboxes_3d)
calibration = session.calibrate(min_pairs=12)
print(calibration.projection)
```

## Intrinsics

`intrinsics_a9_example.json` is an example file for the selected A9 camera:

```json
{
  "fx": 1418.0,
  "fy": 1422.0,
  "cx": 976.0,
  "cy": 606.0,
  "distortion": []
}
```

Before using calibration results as measurements, replace it with the exact
intrinsics for the released A9 camera/cache you are demonstrating.

## Runtime Contract

The real-time pipeline should provide the same four arrays loaded by the demo:

- `image`: `[H, W, 3] uint8` RGB.
- `point_cloud`: `[P, >=3] float32` XYZ in meters.
- `bboxes_2d`: `[K, 4] float32`, `(x1, y1, x2, y2)` pixels.
- `bboxes_3d`: `[M, 6] float32`, extents or center-dim boxes.

If calibration does not solve, collect more frames or lower the demo match
threshold after checking the score distribution for the selected model/site.

## Private UTC Data

Do not place UTC HDF5 caches, UTC extracted frames, or private UTC weights in
this public repository. The `.gitignore` rules block the usual UTC demo paths;
private extraction and partner-delivery utilities belong in `radar-lab-ops`.
