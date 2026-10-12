# Public A9 Demo

This folder is a small standalone demo for the accepted-paper public release.
It installs `xcalib` editable from the repo source (so the env always tracks
`src/xcalib`), a tiny committed A9-style sample under `frames/a9_sample/`, and the
released A9 model weights through `xcalib`.

The committed sample is for step-by-step usage walkthroughs. For realistic
paper-scale runs, place `data/a9_test.h5` locally or let `xcalib.load_dataset()`
resolve the released Hugging Face dataset cache.

The sample frames come from the TUM Traffic (A9) dataset, which carries its own
CC BY-NC-ND 4.0 terms (see the root `README.md`).

## Quickstart notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/radar-lab/XCalib/blob/main/demo/xcalib_quickstart.ipynb)

`xcalib_quickstart.ipynb` is the most direct way to see the API: install →
`Matcher.from_pretrained` (Hugging Face weights) → `load_dataset` (Hugging Face
dataset, with an offline fallback to `frames/a9_sample`) → `match()` →
`oneshot()` + `calibrate()`, with the overlays rendered inline. Sections 6 to 8 cover
what is new in 0.3: one-to-one pairs with the geometric gate, `calibrate(refine=True)`,
`SceneRecorder` with the `xcalib-review` window, and `oriented_extents`.

- **Colab:** click the badge above — it installs `xcalib` and shallow-clones this
  repo for the sample.
- **Locally:** `pixi install` here, then `pixi run notebook` to open it in
  JupyterLab — or open `xcalib_quickstart.ipynb` in VS Code and pick this `pixi`
  environment as the kernel (the env ships `jupyterlab`, `ipykernel`, and
  `matplotlib`).

The numbered scripts below are the same flow as plain, CI-friendly Python.

## Continual-refine notebook

`refine_unification.ipynb` walks the **continual** learning tier end-to-end in the `xcalib`
package: build a `Scene` and a base model from the A9 sample → `match_review` →
`record_verified_pairs` into the durable corpus → inspect it → `scene_frame_loader` →
`run_continual_refine` (gated; the notebook relaxes the gate so a handful of frames can
publish) → load the published revision. It is the worked example for accumulating
human-verified pairs into a corpus and training the matcher from it over time — the durable
complement to one-shot `adapt()`. The [continual refine guide](../docs/continual-refine.md)
explains the corpus, the gate, and published revisions.

Run it locally from a checkout. `pixi run notebook` opens the quickstart, so open
`refine_unification.ipynb` from the JupyterLab file browser. It reads `frames/a9_sample` and
takes the A9 weights from the model hub, or from
`../checkpoints/crlite_a9_dataset_r02_s01_best.pth` when that file exists. One optional cell
launches the live `xcalib-review` window (`%gui qt`) on the demo scene, so you can do the
review by hand — step frames, fix pairs, Apply, **Refine (accumulate)** — then continue in
the notebook. Skip that cell on a headless machine.

## Review window

`xcalib-review` is a Qt window for correcting camera-LiDAR pairs on a saved scene. This
demo environment gets PySide6 as a dependency of its matplotlib package; elsewhere run `pip install "xcalib[gui]"`.

Save a scene with `SceneRecorder` (section 7 of the quickstart notebook does this with the
sample frames), then open it:

```bash
xcalib-review --scene <scene_dir> --camera <camera_name>
```

The window loads the matcher from `<scene_dir>/calib/xcalib/<camera_name>_model/` (write it
with `matcher.save_pretrained(...)`) and, when `<scene_dir>/calib/camera/<camera_name>.json`
exists, uses that calibration for the geometric gate.

## Steps

1. `run_00_prepare_assets.py`: report the `xcalib` install, optional local
   weights, optional local dataset cache, and ONNX outputs.
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

`pixi.toml` installs `xcalib` **editable** from the repo source (`../`), so the
demo env always tracks `src/xcalib` — no wheel to rebuild. (Released installs use
the published wheel / `pip install XCalib`; the notebook's Colab cell uses pip.)

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
if calibration.success and calibration.accepted:
    print(calibration.projection)
```

Pairing, optionally gated by the calibration (`session.calibration` is `None` until one is
accepted):

```python
result = matcher.match(
    frame.image,
    frame.point_cloud,
    frame.bboxes_2d,
    frame.bboxes_3d,
    top_k=1,
    match_threshold=0.5,
    one_to_one=True,
    calibration=session.calibration,
)
print(result.matches)     # best LiDAR box per camera box; a LiDAR box can repeat
print(result.assignment)  # optimal pairs, each LiDAR box used at most once
```

## Intrinsics

`intrinsics_a9_example.json` holds the released A9 intrinsics for the committed sample's
camera (`s110_camera_basler_south1_8mm`):

```json
{
  "fx": 1293.5,
  "fy": 1320.2,
  "cx": 946.1,
  "cy": 622.0,
  "distortion": []
}
```

For another camera or your own rig, use its own intrinsics. With the A9 cache,
`frame.intrinsics[camera]` from `load_dataset()` carries the released `K`.

## Runtime Contract

The real-time pipeline should provide the same four arrays loaded by the demo:

- `image`: `[H, W, 3] uint8` RGB.
- `point_cloud`: `[P, >=3] float32` XYZ in meters.
- `bboxes_2d`: `[K, 4] float32`, `(x1, y1, x2, y2)` pixels.
- `bboxes_3d`: `[M, 6] float32`, extents or center-dim boxes. Pass boxes that enclose
  the object; for oriented detections use `xcalib.data.oriented_extents`.

If calibration does not solve, collect more frames or lower the demo match
threshold after checking the score distribution for the selected model/site.

## Private UTC Data

Do not place UTC HDF5 caches, UTC extracted frames, or private UTC weights in
this public repository. The `.gitignore` rules block the usual UTC demo paths.
