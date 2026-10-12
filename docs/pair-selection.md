# Pair selection

`Matcher.match` scores every camera box against every LiDAR box and returns a
`[K, M]` similarity matrix. Turning those scores into pairs takes two more steps,
and `match` can run both:

- **Optimal one-to-one assignment** picks the set of pairs with the highest total
  similarity, using each camera box and each LiDAR box at most once.
- **A geometric gate** uses a camera–LiDAR calibration to rule out pairs whose
  LiDAR box cannot land on the camera box in the image.

Both are off by default. A call that passes none of the new arguments returns the
same `similarity`, `top_indices` and `matches` as 0.2.0 (see
[What changed since 0.2.0](#what-changed-since-020)).

## When to use it

| You need | Use |
| --- | --- |
| Ranked candidates for each camera box | `result.matches` and `result.top_indices`. Not one-to-one. |
| Each LiDAR box claimed by at most one camera box | `one_to_one=True` |
| Fewer wrong pairs between look-alike vehicles or on the wrong side of the road | `one_to_one=True` and `calibration=...` |

The gate needs a calibration that is roughly right, not exact. Its radius is
generous by default (see [Parameters](#parameters)), so a few pixels of drift do
not veto a true pair. A pair that projects farther from the box than the radius
is vetoed, however similar the two crops look, unless the gate would veto every
LiDAR box for that camera box (see the row fallback below).

## Use it with `Matcher.match`

```python
result = matcher.match(
    image, point_cloud, bboxes_2d, bboxes_3d,
    one_to_one=True,           # optimal one-to-one assignment
    calibration=calibration,   # optional: gate pairs by geometry first
)

for cam_idx, lidar_idx, score in result.assignment:
    print(cam_idx, lidar_idx, score)
```

`calibration` is a successful `CalibrationResult`: the output of
`matcher.calibrate(...)`, of a one-shot session, or one you build from known
extrinsics (see the [example below](#a-matchermatch-call)). Without `calibration`
there is no gate, and `assignment` comes from the raw similarity.

### Parameters

New keyword-only arguments of `Matcher.match`:

| Argument | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `one_to_one` | `False` | | Fill `result.assignment` with the optimal one-to-one pairs. |
| `calibration` | `None` | | A successful `CalibrationResult`. Turns the gate on and fills `geo_compatible` and `proj_px`. A calibration with `success=False` raises `ValueError`. |
| `geo_gate_min_px` | `None`, which means `150.0` | pixels | Smallest gate radius. Ignored without `calibration`. |
| `geo_gate_diag` | `None`, which means `3.0` | multiple of the 2D box diagonal | Gate radius that scales with the box. Ignored without `calibration`. |

`match_threshold` (default `0.0`, same unit as the similarity) already existed. With
`one_to_one=True` it also drops assignment pairs that score below it.

The gate radius for each camera box is

```text
radius = max(geo_gate_min_px, geo_gate_diag * box_diagonal)      # pixels
box_diagonal = sqrt((x2 - x1)**2 + (y2 - y1)**2)
```

A pair is compatible when the projected center of the LiDAR box lies in front of
the camera and within `radius` pixels of the center of the camera box. The
diagonal term keeps close-up objects, whose reprojection error grows with their
size, from losing their true match to a fixed radius.

If the gate blocks every LiDAR box for a camera box, that camera box is left
ungated: its whole row of `geo_compatible` is `True`. A badly wrong calibration then
degrades to appearance-only matching instead of emptying the row. `match` has no
switch for this; `geometric_gate` exposes it as `row_fallback`.

### Result fields

| Field | Contents | Numbering |
| --- | --- | --- |
| `assignment` | `[(camera_idx, lidar_idx, score), ...]`, highest score first. `score` is the raw similarity. Empty unless `one_to_one=True`. | Your boxes |
| `geo_compatible` | `[K', M']` bool array, or `None` without `calibration` or when no box survives cropping. | Kept boxes |
| `proj_px` | `[K', M']` float array, or `None` without `calibration` or when no box survives cropping. Pixel distance from the projected LiDAR center to the camera box center; `inf` behind the camera. | Kept boxes |

`match` drops a camera box whose integer `x1` is negative or that has no area, and a
LiDAR box whose 1.25x-expanded crop holds no points. `assignment` and `matches` use your original numbering.
`similarity`, `geo_compatible` and `proj_px` have one row per kept camera box and
one column per kept LiDAR box: row `r` belongs to camera box
`result.kept_2d_indices[r]` and column `c` to LiDAR box `result.kept_3d_indices[c]`.

## The two functions

`match` calls these, and you can call them on your own arrays:

```python
from xcalib import geometric_gate, one_to_one_assign
```

```text
one_to_one_assign(similarity, threshold=0.0)
geometric_gate(bboxes_2d, centers_3d, calibration=None, *, intrinsics=None,
               rotation=None, translation=None, min_px=150.0, diag_factor=3.0,
               row_fallback=True)
```

`one_to_one_assign` takes a `[K, M]` similarity matrix and returns
`[(row, col, score), ...]`, highest score first. It maximizes the total similarity
over all one-to-one assignments, then drops pairs below `threshold`. For
`K != M` it returns at most `min(K, M)` pairs.

`geometric_gate` returns `(mask, dist_px)`, both `[K, M]`:

| Argument | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `bboxes_2d` | | pixels | `[K, 4]` camera boxes `(x1, y1, x2, y2)`. |
| `centers_3d` | | meters | `[M, 3]` LiDAR box centers, in the frame the calibration maps to the camera. |
| `calibration` | `None` | | A successful `CalibrationResult`. |
| `intrinsics`, `rotation`, `translation` | `None` | | Pass all three instead of `calibration`: `CameraIntrinsics`, a `[3, 3]` rotation and a `[3]` translation from the LiDAR frame to the camera frame. |
| `min_px` | `150.0` | pixels | Smallest gate radius. |
| `diag_factor` | `3.0` | multiple of the box diagonal | Gate radius that scales with the box. |
| `row_fallback` | `True` | | If the gate blocks every LiDAR box for a camera box, make that whole row compatible. A badly wrong calibration then falls back to appearance only instead of emptying the row. `Matcher.match` always uses `True`. |

## Examples

### Arrays only

This needs no model. The first part shows greedy top-1 sending both camera boxes to
the same LiDAR box. The second part shows a geometric gate: appearance alone prefers
a LiDAR box that is behind the camera, and the gate removes it.

```python
import numpy as np

from xcalib import CalibrationResult, CameraIntrinsics, geometric_gate, one_to_one_assign

# 1. Greedy top-1 versus the optimal one-to-one assignment ---------------------
similarity = np.array([[0.90, 0.98, 0.10],
                       [0.20, 0.95, 0.30]])

print("greedy argmax per row:", similarity.argmax(axis=1).tolist())
print("one_to_one_assign    :", one_to_one_assign(similarity, threshold=0.5))

# 2. A geometric gate -----------------------------------------------------------
# Pinhole camera; LiDAR frame is x forward, y left, z up; camera frame is
# x right, y down, z forward.
intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
R = np.array([[0.0, -1.0, 0.0],
              [0.0, 0.0, -1.0],
              [1.0, 0.0, 0.0]])
t = np.zeros(3)
calibration = CalibrationResult(
    success=True, intrinsics=intrinsics, rotation=R, translation=t,
    projection=intrinsics.K @ np.hstack([R, t.reshape(3, 1)]),
)

# Two camera boxes (x1, y1, x2, y2) in pixels ...
bboxes_2d = np.array([[620.0, 520.0, 700.0, 580.0],     # centre (660, 550)
                      [1160.0, 500.0, 1240.0, 560.0]])  # centre (1200, 530)
# ... and three LiDAR box centres (x, y, z) in meters.
centers_3d = np.array([[20.0, 6.0, 0.0],      # projects near (660, 540)
                       [25.0, -6.0, 0.0],     # projects near (1200, 540)
                       [-10.0, 0.0, 0.0]])    # behind the camera

mask, dist_px = geometric_gate(bboxes_2d, centers_3d, calibration)
print("compatible:\n", mask.astype(int))
print("distance (px):\n", np.round(dist_px, 1))

# Appearance alone prefers the wrong objects: box 0 ends up assigned to the box
# behind the camera.
similarity = np.array([[0.60, 0.97, 0.90],
                       [0.20, 0.95, 0.10]])
print("no gate :", one_to_one_assign(similarity, threshold=0.5))

# Veto the incompatible pairs with a large finite penalty, then assign.
gated = np.where(mask, similarity, -1e9)
print("gated   :", one_to_one_assign(gated, threshold=0.5))
```

Output:

```text
greedy argmax per row: [1, 1]
one_to_one_assign    : [(1, 1, 0.95), (0, 0, 0.9)]
compatible:
 [[1 0 0]
 [0 1 0]]
distance (px):
 [[ 10.  540.1   inf]
 [540.1  10.    inf]]
no gate : [(1, 1, 0.95), (0, 2, 0.9)]
gated   : [(1, 1, 0.95), (0, 0, 0.6)]
```

### A `Matcher.match` call

A randomly initialised model loads and runs, but its scores carry no information.
That is enough to see the arguments, the result fields and the gate, because the
geometry does not depend on the weights. Only the pairs on the `one_to_one` and
`gated (150 px, 3x)` lines depend on the random weights, so yours can differ.
Everything else in the output follows from the geometry. The random scores sit
close to zero, so `match_threshold=-1.0` keeps every pair instead of dropping the
ones below the default `0.0`.

```python
import tempfile
from pathlib import Path

import numpy as np
import torch

from xcalib import CalibrationResult, CameraIntrinsics, Matcher
from xcalib.models.registry import build_model
from xcalib.utils.config import load_yaml
from xcalib.utils.io import default_config_path

# A randomly initialised model: it loads and runs, but its similarity scores carry
# no information. The shapes, the index spaces and the gate do not depend on it.
torch.manual_seed(0)
cfg = load_yaml(default_config_path("crlite_vit_exp1", "a9_dataset_r02_s01"))
cfg.set("device", "cpu")
weights = Path(tempfile.mkdtemp()) / "crlite_vit_exp1.pth"
torch.save({"model_state_dict": build_model("crlite_vit_exp1", cfg).state_dict()}, weights)
matcher = Matcher.from_pretrained(model="crlite_vit_exp1", weights=weights, config=cfg, device="cpu")

# Camera, and a calibration for it (LiDAR frame: x forward, y left, z up).
intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
t = np.zeros(3)
calibration = CalibrationResult(
    success=True, intrinsics=intrinsics, rotation=R, translation=t,
    projection=intrinsics.K @ np.hstack([R, t.reshape(3, 1)]),
)

# A synthetic frame: four objects, each seen by the camera and by the LiDAR.
# Camera box i and LiDAR box i are the same object.
rng = np.random.default_rng(0)
centers = np.array([[20.0, 6.0, 0.8], [25.0, -6.0, 0.8], [32.0, 2.0, 0.8], [45.0, -3.0, 0.8]])
cam = centers @ R.T + t
uv = cam[:, :2] / cam[:, 2:3] * 1000.0 + [960.0, 540.0]
half = (np.clip(2400.0 / cam[:, 2], 30.0, 120.0) / 2.0)[:, None]
bboxes_2d = np.hstack([uv - half, uv + half]).astype(np.float32)
size = np.array([4.4, 1.8, 1.6])
bboxes_3d = np.hstack([centers - size / 2, centers + size / 2]).astype(np.float32)
image = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
point_cloud = np.vstack(
    [rng.uniform(-5, 60, size=(1500, 3))]
    + [rng.uniform(b[:3], b[3:], size=(300, 3)) for b in bboxes_3d]
).astype(np.float32)

# The same call as before 0.3.0: no assignment, no gate.
plain = matcher.match(image, point_cloud, bboxes_2d, bboxes_3d, top_k=1)
print("similarity:", plain.similarity.shape, "| assignment:", plain.assignment,
      "| geo_compatible:", plain.geo_compatible, "| proj_px:", plain.proj_px)

# Optimal one-to-one assignment, no calibration, so no gate. match_threshold=-1.0
# keeps every pair; the random scores sit near zero and some are below the default 0.0.
result = matcher.match(image, point_cloud, bboxes_2d, bboxes_3d,
                       one_to_one=True, match_threshold=-1.0)
print("one_to_one          :", sorted((i, j) for i, j, _ in result.assignment))

# With the calibration, incompatible pairs are vetoed before the assignment.
result = matcher.match(image, point_cloud, bboxes_2d, bboxes_3d, match_threshold=-1.0,
                       one_to_one=True, calibration=calibration)
print("gated (150 px, 3x)  :", sorted((i, j) for i, j, _ in result.assignment))
print("geo_compatible:\n", result.geo_compatible.astype(int))
print("proj_px:\n", np.round(result.proj_px))

# A tighter gate: radius = max(geo_gate_min_px, geo_gate_diag x box diagonal).
result = matcher.match(image, point_cloud, bboxes_2d, bboxes_3d, match_threshold=-1.0,
                       one_to_one=True, calibration=calibration,
                       geo_gate_min_px=40.0, geo_gate_diag=0.5)
print("gated (40 px, 0.5x) :", sorted((i, j) for i, j, _ in result.assignment))
print("geo_compatible:\n", result.geo_compatible.astype(int))

# Dropped boxes: a camera box that starts left of the image is dropped before scoring.
# `assignment` stays in YOUR numbering; the matrices follow the kept boxes.
edge = np.array([[-10.0, 400.0, 60.0, 460.0]], dtype=np.float32)
result = matcher.match(image, point_cloud, np.vstack([edge, bboxes_2d]), bboxes_3d,
                       match_threshold=-1.0, one_to_one=True, calibration=calibration,
                       geo_gate_min_px=40.0, geo_gate_diag=0.5)
print("kept_2d_indices     :", result.kept_2d_indices.tolist())
print("similarity / geo_compatible:", result.similarity.shape, result.geo_compatible.shape)
print("assignment          :", sorted((i, j) for i, j, _ in result.assignment))
```

Output:

```text
similarity: (4, 4) | assignment: [] | geo_compatible: None | proj_px: None
one_to_one          : [(0, 2), (1, 1), (2, 3), (3, 0)]
gated (150 px, 3x)  : [(0, 0), (1, 2), (2, 3), (3, 1)]
geo_compatible:
 [[1 0 1 1]
 [0 1 1 1]
 [1 1 1 1]
 [0 1 1 1]]
proj_px:
 [[  0. 540. 238. 367.]
 [540.   0. 303. 174.]
 [238. 303.   0. 129.]
 [367. 174. 129.   0.]]
gated (40 px, 0.5x) : [(0, 0), (1, 1), (2, 2), (3, 3)]
geo_compatible:
 [[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 0 1]]
kept_2d_indices     : [1, 2, 3, 4]
similarity / geo_compatible: (4, 4) (4, 4)
assignment          : [(1, 0), (2, 1), (3, 2), (4, 3)]
```

With the tight gate, each camera box has exactly one compatible LiDAR box (the
identity matrix above), so the assignment is the true pairing whatever the weights
are. In the last call, the
camera box at row `0` of your input was dropped, so `kept_2d_indices` starts at `1`
and `assignment` still uses your numbering.

## Things to know

- `matches` and `top_indices` are never gated and are not one-to-one. Only
  `assignment` is.
- The threshold is applied after the assignment. The assignment maximizes the total
  over all pairs, including pairs below the threshold, so a pair under the
  threshold can take a row and a column away from a pair above it.
  `one_to_one_assign(np.array([[0.90, 0.45], [0.89, 0.00]]), threshold=0.5)`
  returns `[(1, 0, 0.89)]`: the best total uses `(0, 1)` and `(1, 0)`, and `(0, 1)`
  is then dropped.
- If you gate by hand, mark incompatible pairs with a large finite value such as
  `-1e9`, as `Matcher.match` does. `-inf` is not supported: depending on the shape,
  a row or column that is blocked everywhere makes the problem infeasible.
- Boxes given as `(cx, cy, cz, dx, dy, dz)` are told apart from
  `(xmin, ymin, zmin, xmax, ymax, zmax)` by a per-row rule. A row whose three
  dimensions are all at least its three center coordinates is read as extents, and
  the gate then uses the wrong center. See
  [3D bbox format auto-detection](protocol.md#3d-bbox-format-auto-detection).
- `Matcher.pair` is an alias of `Matcher.match` and takes the same arguments.
- SciPy is optional. `one_to_one_assign` uses it when it is installed and otherwise
  falls back to a built-in exact solver that returns an equally good set of pairs (the same
  total similarity); ties may be broken differently.

## What changed since 0.2.0

| | 0.2.0 | 0.3.0 | Default |
| --- | --- | --- | --- |
| `Matcher.match` arguments | `image, point_cloud, bboxes_2d, bboxes_3d, top_k, match_threshold, return_latency, validate` | Four keyword-only arguments added: `one_to_one`, `calibration`, `geo_gate_min_px`, `geo_gate_diag`. The positional order is unchanged. | `False`, `None`, `None`, `None`: the call behaves as before. |
| `MatchResult` fields | `similarity, top_indices, matches, kept_2d_indices, kept_3d_indices, latency_ms, model, device` | Three fields added at the end: `assignment`, `geo_compatible`, `proj_px`. | `[]`, `None`, `None` |
| Existing outputs of the same call | | `similarity`, `top_indices`, `matches`, `kept_2d_indices` and `kept_3d_indices` are identical for the same weights and inputs. | Unchanged |
| `match_threshold` | Filters `matches`. | Also filters `assignment`. | Matters only with `one_to_one=True`. |
| `calibration` without `one_to_one` | | Fills `geo_compatible` and `proj_px`. `matches` and `assignment` are not affected. | Opt-in |
| Public API | | `xcalib.one_to_one_assign`, `xcalib.geometric_gate`, and the constants `GEO_GATE_MIN_PX = 150.0` and `GEO_GATE_DIAG = 3.0` in `xcalib.assign`. | New |
| Runtime dependencies | | No new ones. SciPy stays optional. | |

Code written for 0.2.0 needs no change. Its `match` calls return the same matches,
because pair selection only runs when you pass `one_to_one=True` or `calibration`.
`Matcher.calibrate` and one-shot sessions call `match` without the new arguments, so
they behave as before.

One change reaches beyond pair selection: `import xcalib` is now lazy. `from xcalib import Matcher`
and every other name in `xcalib.__all__` work as before, but the submodules (`xcalib.engine`,
`xcalib.protocol`, and so on) are attributes of `xcalib` only after you import them, for example
`import xcalib.protocol`.

See also: [Input protocol](protocol.md) for the box formats, and the
[API reference](api.md) for the full signatures.
