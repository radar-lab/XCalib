# XCalib

Camera-LiDAR cross-modal matching for edge devices: pretrained matchers, targetless extrinsic
calibration, ONNX/TensorRT export, HDF5 training, and label-free one-shot adaptation — behind one
small API.

`xcalib` is the deployment package for the matching front-end studied in _Position Encoding in
Detection-Based LiDAR–Camera Matching: A Diagnostic Study at Infrastructure Sites_ (accepted at
IEEE Sensors Letters; final publisher URL pending). It runs the same weights on a lab
workstation and an NVIDIA Jetson AGX Thor, and builds on the lab's prior calibration framework
[CalibRefine](https://github.com/radar-lab/Lidar_Camera_Automatic_Calibration) (Cheng et al., IEEE
TIM 2026, [arXiv:2502.17648](https://arxiv.org/abs/2502.17648)).

> **Status** — publicly released and actively developed. The public API
> (`xcalib.__all__`) is stable across minor versions; internals may change
> without notice. Broader visibility is expected once the accompanying IEEE
> Sensors Letters paper is published. New versions are cut from `main` as tagged
> releases.

## Install

```bash
pip install XCalib            # inference
pip install "XCalib[onnx]"    # + ONNX export / onnxruntime parity checks
pip install "XCalib[train]"   # + h5py for HDF5 training
```

For normal use, install the PyPI wheel once it is published; a source checkout
is only needed for development, paper validation, or local experiments.

Requires Python 3.10+ (CI tests 3.10 through 3.14). The wheel pins `torch>=2.2` only — on Jetson
keep the JetPack/PyPI torch you already have.

## Quickstart

After installing the wheel, import `Matcher`, load a released checkpoint, and
pass one camera/LiDAR frame with detector boxes:

```python
import numpy as np
from xcalib import Matcher

# First call downloads and caches the released checkpoint/config.
matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")

image = np.zeros((720, 1280, 3), dtype=np.uint8)  # H,W,3 RGB
points = np.random.default_rng(0).uniform(
    low=[-20.0, -20.0, -2.0],
    high=[50.0, 20.0, 4.0],
    size=(4096, 3),
).astype(np.float32)
boxes_2d = np.array([[420.0, 260.0, 520.0, 360.0]], dtype=np.float32)
boxes_3d = np.array([[8.0, -1.5, -1.0, 11.5, 1.5, 1.6]], dtype=np.float32)

result = matcher.match(
    image=image,
    point_cloud=points,
    bboxes_2d=boxes_2d,       # K,4 (x1,y1,x2,y2) in pixels
    bboxes_3d=boxes_3d,       # M,6 (xmin,ymin,zmin,xmax,ymax,zmax) in meters
    top_k=1,
)

print(result.similarity.shape)  # (K, M)
print(result.matches)           # [(img_idx, lid_idx, score), ...]
```

For real frames, replace the arrays above with decoded RGB images, LiDAR points,
and boxes from your detector. See [Input protocol](protocol.md) for the exact
shape/dtype contract.

Everything a deployment needs hangs off the `Matcher`:

| call                                       | does                                                 |
| ------------------------------------------ | ---------------------------------------------------- |
| `Matcher.from_pretrained(model, site=...)` | load packaged config + released/local weights        |
| `.match(...)` / `.pair(...)`               | K x M similarity, top-k matches, latency             |
| `.calibrate(frames, intrinsics=K)`         | targetless PnP/RANSAC extrinsics                     |
| `.oneshot(K)`                              | one-shot adaptation session (pseudo-labels, adapter) |
| `.train(train_h5, val_h5)`                 | fine-tune the loaded weights on HDF5 caches          |
| `.build("onnx" \| "trt")`                  | export whatever weights are loaded                   |
| `.save_pretrained(dir)`                    | persist weights + config locally                     |

Training and dataset helpers:

```python
from xcalib import train, load_dataset

best = train("crlite", "a9_dataset_r02_s01", "a9_dataset_r02_s01",
             site="a9_dataset_r02_s01", epochs=100)
loader = load_dataset("a9_dataset_r02_s01", split="test")
```

### External Configs

Packaged YAMLs under `xcalib/cfg/` are just defaults. For custom weights or
site-specific experiments, pass any external YAML path (or a plain dict /
`EdgeConfig`) through the same `config=` argument:

```python
matcher = Matcher.from_pretrained(
    "crlite",
    weights="runs/site42/best.pth",
    config="runs/site42/crlite_site42.yaml",
)
```

The config is loaded before the model is built, so it can change architecture knobs such as
`embed_dim`, `token_len`, `top_k`, `crop_size`, `point_cloud_size`, `vit.depth`, `vit.num_heads`,
`gnn.num_layers`, `gnn.hidden_dim`, `similarity_head.hidden_dims`, and CalibRefine `dense.*`
widths/dropouts. If a cfg changes tensor shapes, the checkpoint must have been trained with that
same cfg; released A9/UTC checkpoints should use their packaged YAMLs.

## See it in action

Both clips are rendered from the public A9 (TUMTraf intersection) test cache with real checkpoints —
regenerate them any time with `pixi run python scripts/make_demo_media.py`.
For single-frame diagnostics in your own notebooks or reports, use the package
helpers in [Visualization](visualization.md).

### Matching — `matcher.match()`

![Matching demo on an A9 test frame](assets/a9_matching.gif)

One frame, two unregistered sensors. The matcher embeds every camera detection (left, 2D boxes) and
every LiDAR detection (right, bird's-eye view of the native cloud with 3D footprints), scores all
camera x LiDAR pairs by cosine similarity, and takes the argmax per camera detection. Confident rows
lock as green camera-to-LiDAR links; rows whose best score stays below the threshold are left
unmatched on purpose — at an intersection the two sensors never see exactly the same set of objects.

### Calibration — `matcher.calibrate()`

![Calibrated camera-LiDAR projection on A9](assets/a9_calibration.gif)

Calibration is what those matches buy you. `CalibrationSession` accumulates confident (2D box
center, 3D box center) pairs across frames and solves PnP/RANSAC with the camera intrinsics the
protocol requires — recovering the extrinsics with no checkerboard or survey target. The overlay
shows the calibrated projection at this intersection: LiDAR scan lines land on the road surface and
the 3D detections wrap the vehicles. The matcher's 61 confident pairs from this 29-frame clip agree
with the projection shown to a ~30 px median, which is the camera-box-center vs LiDAR-centroid
offset, not noise.

_(Full disclosure: this public test slice is a flat intersection, so the matched vehicle centers are
nearly coplanar and a 6-DoF PnP solution is under-constrained — the figure therefore projects
through the intersection's
[published reference calibration](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit/blob/main/calib/s110_camera_basler_south2_8mm.json).
`calibrate()` recovers the pose itself when the buffered matches span varied depth and height —
longer clips, mixed vehicle sizes, non-flat scenes.)_

### One-shot adaptation — `matcher.oneshot()`

Once a projection matrix exists, it becomes a free supervisor. An `OneShotSession` watches
deployment frames, keeps only the matches that are both confident _and_ geometrically consistent
with the projection (the matched LiDAR center must land near its camera box), and uses those
pseudo-labels to train small identity-initialized adapters on top of the frozen backbone — no human
labels, and the paper-grade weights are never overwritten. `session.save()` /
`matcher.save_pretrained()` persist the site-adapted weights when you are happy with them.

## Model zoo

| Model             | Backbone                  | Position encoding                     | Tier           |
| ----------------- | ------------------------- | ------------------------------------- | -------------- |
| `crlite`          | ResNet + PointNet         | Enhanced 3D (2D-sin + depth + 3D MLP) | paper          |
| `crlite_2dpe`     | ResNet + PointNet         | 2D-only                               | paper          |
| `crlite_vit_exp1` | crop-ViT + PointNet       | none                                  | paper          |
| `crlite_vit_exp3` | crop-ViT + PointNet       | Enhanced 3D                           | paper          |
| `calibrefine`     | ResNet + PointNet2        | 2D sinusoidal (pairwise)              | paper baseline |
| `crlite_vit_exp4` | crop-ViT + PointNet + GAT | edge-aware GAT                        | delivery only  |

`calibrefine` is our standalone port of the Common Feature Discriminator from
[CalibRefine](https://github.com/radar-lab/Lidar_Camera_Automatic_Calibration) (Cheng et al., IEEE
TIM 2026, [arXiv:2502.17648](https://arxiv.org/abs/2502.17648)), kept as the pairwise baseline the
paper compares against.

Released paper checkpoints and the public A9 cache can be loaded from the Hub
or from local files; see [Models and datasets](hub.md).

## Where next

- [Input protocol](protocol.md) — the exact contract `match()` enforces.
- [Visualization](visualization.md) — draw matching links and calibrated LiDAR projections.
- [Models and datasets](hub.md) — load released weights and A9 dataset caches.
- [Custom checkpoints and models](adding-models.md) — load external weights/configs, or add a new packaged architecture.
- [HDF5 cache format](hdf5-format.md) — what `train()` / `load_dataset()` read.
- [API reference](api.md) — the curated public API.
- [Reproducing the paper](paper.md) — validation commands and reference numbers.
- [Paper evidence](evidence/index.md) — frozen JSON evidence and public/private data scope.

## API stability

Only the symbols exported by `xcalib.__all__` (and documented in the [API reference](api.md)) are
stable across minor versions. Anything imported from submodules (`xcalib.engine.*`,
`xcalib.models.*`, ...) is internal and may change without notice.

## Citation

The xcalib paper has been accepted at IEEE Sensors Letters. This entry keeps a
temporary paper URL note until the publisher page is live:

```bibtex
@article{guo2026xcalib,
  author  = {Guo, Lihao and Tang, Jiahao and Bang, Tam and Zhang, Tianya and
             Harris, Austin and Sartipi, Mina and Cao, Siyang},
  title   = {Position Encoding in Detection-Based LiDAR--Camera Matching:
             A Diagnostic Study at Infrastructure Sites},
  journal = {IEEE Sensors Letters},
  year    = {2026},
  note    = {Accepted. Paper URL pending. Code:
             https://github.com/radar-lab/XCalib},
}
```

For the `calibrefine` baseline / prior framework:

```bibtex
@article{cheng2026calibrefine,
  author  = {Cheng, Lei and Guo, Lihao and Zhang, Tianya and Bang, Tam and
             Harris, Austin and Hajij, Mustafa and Sartipi, Mina and
             Cao, Siyang},
  title   = {CalibRefine: Deep Learning-Based Online Automatic Targetless
             LiDAR-Camera Calibration With Iterative and Attention-Driven
             Post-Refinement},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  volume  = {75},
  year    = {2026},
  note    = {arXiv:2502.17648. Code:
             https://github.com/radar-lab/Lidar_Camera_Automatic_Calibration},
}
```
