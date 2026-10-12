# Deployed-site workflow

`matcher.oneshot()` corrects small drift on its own. This page covers what to do when a deployed
rig drifts further: keep the frames the matcher fails on, have a person review them, and change
the stored camera-LiDAR matrix only when pairs the fit never saw support the change.

You use four pieces, in this order:

| Step | Problem it solves | Entry point |
| --- | --- | --- |
| [Capture](#1-capture-the-frames-the-matcher-is-lost-on) | A stream integration has no annotation tool. When matching fails you need the exact inputs on disk, in a format a person can open. | `SceneRecorder`, `matcher.oneshot(K, recorder=...)` |
| [Review](#2-review-the-captured-frames) | Someone has to say which camera box belongs to which LiDAR box. | `xcalib-review` |
| [Nudge](#3-nudge-the-matrix-in-session) | After a small shift, a fresh PnP solve from a few buffered pairs is not guaranteed to beat the matrix you already have. | `refine_projection`, `session.calibrate(refine=True)` |
| [Gate](#4-gate-a-candidate-on-held-out-data) | The nudge is judged on the pairs it was fitted to. A matrix you deploy needs evidence from pairs the fit never saw, and a cap on how far it moves. | `build_physical_refinement_candidate` |

`SceneRecorder`, `refine_projection` and the five gate names (`CalibrationGateConfig`,
`CalibrationMetrics`, `PhysicalCalibrationCandidate`, `build_physical_refinement_candidate`,
`evaluate_calibration`) are in `xcalib.__all__`. `should_record`, `ObserveReport`,
`CalibrationSession` and everything under `xcalib.gui` are not, so they can change without notice.

## The workflow

1. **Start from a matrix.** `matcher.calibrate(...)`, `session.calibrate()` or a surveyed
   extrinsic gives a `CalibrationResult`. Intrinsics stay fixed throughout; only `[R|t]` moves.
2. **Record.** Attach a `SceneRecorder` to the session. Frames that have detections on both sides
   but no confident pair are written to disk.
3. **Review.** Open the capture directory in `xcalib-review`, fix the pairs, press Apply. Each
   accepted pair writes the LiDAR id onto the camera box.
4. **Collect pairs.** Turn the reviewed boxes into (3D center, 2D center) pairs. Fit on earlier
   frames; keep other frames out of the fit and use them for validation.
5. **Gate.** `build_physical_refinement_candidate` fits a candidate on the fit set and checks it
   on the validation sets. XCalib saves nothing: if `eligible` is true, you save the candidate.

There is no `xcalib` subcommand for any of this. The only command-line entry point is
`xcalib-review`.

## A synthetic site

The examples need no data. A virtual camera with known intrinsics sees boxes through a true pose,
and the site's stored matrix is that pose after the rig moved 1 degree and 0.18 m. Run the blocks
in order in one Python session. The blocks that call `Matcher.from_pretrained` need the released
weights (the first call downloads them); the rest runs offline.

```python
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from xcalib import CalibrationResult, CameraIntrinsics
from xcalib.oneshot.calibration import project_points

K = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)  # pixels, fixed and known

# True LiDAR -> camera pose. LiDAR: x forward, y left, z up. Camera: x right, y down, z forward.
R_true = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
t_true = np.array([0.2, -0.3, 0.5])  # meters
true_extrinsics = np.hstack([R_true, t_true.reshape(3, 1)])


def make_calibration(R, t):
    return CalibrationResult(success=True, intrinsics=K, rotation=R, translation=t,
                             projection=K.K @ np.hstack([R, t.reshape(3, 1)]))


def shifted_pose(deg, shift_m):
    """The true pose after the rig rotated by `deg` degrees and moved by `shift_m` meters."""
    axis = np.array([0.3, -0.5, 0.8])
    rot, _ = cv2.Rodrigues(np.deg2rad(deg) * axis / np.linalg.norm(axis))
    return rot @ R_true, t_true + np.asarray(shift_m)


# The matrix the site still uses: the true pose after a 1 degree / 0.18 m shift.
stored = make_calibration(*shifted_pose(1.0, (0.15, -0.10, 0.05)))
exact = make_calibration(R_true, t_true)


def site_pairs(rng, n, noise_px=1.5, offset_px=(0.0, 0.0), pose=None):
    """n pairs (3D center [m], 2D center [px]) as seen through `pose` (default: the true pose)."""
    R, t = pose or (R_true, t_true)
    points_3d = np.column_stack([rng.uniform(8, 45, n), rng.uniform(-6, 6, n),
                                 rng.uniform(-1, 2, n)])
    uv, _ = project_points(points_3d, K, R, t)
    return points_3d, uv + rng.normal(0, noise_px, uv.shape) + np.asarray(offset_px)


def synth_frame(rng, n=6):
    """One frame: image, cloud [P, 4] (x y z intensity), 2D boxes, 3D boxes, pair_of."""
    centers = []
    while len(centers) < n:  # keep boxes that land inside the image
        c = rng.uniform([8.0, -6.0, -1.0], [45.0, 6.0, 2.0])
        (u, v), = project_points(c[None], K, R_true, t_true)[0]
        if 80 <= u <= 1200 and 80 <= v <= 640:
            centers.append(c)
    centers = np.array(centers)
    uv, depth = project_points(centers, K, R_true, t_true)
    uv = uv + rng.normal(0, 1.5, uv.shape)  # detector jitter, pixels
    half = np.clip(1100.0 / depth, 12, 80)[:, None]
    boxes_2d = np.hstack([uv - half, uv + half]).astype(np.float32)  # x1, y1, x2, y2
    extents = np.hstack([centers - [1.75, 0.9, 0.8], centers + [1.75, 0.9, 0.8]])
    order = rng.permutation(n)  # LiDAR detections arrive in another order
    boxes_3d = extents[order].astype(np.float32)  # xmin, ymin, zmin, xmax, ymax, zmax
    pair_of = np.argsort(order)  # camera box i belongs to LiDAR box pair_of[i]
    ground = rng.uniform([-20, -20, -2], [50, 20, 0], (800, 3))
    inside = [rng.uniform(b[:3], b[3:], (60, 3)) for b in boxes_3d]
    xyz = np.vstack([ground, *inside])
    cloud = np.hstack([xyz, rng.uniform(0, 255, (len(xyz), 1))]).astype(np.float32)
    image = np.zeros((720, 1280, 3), np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 1280).astype(np.uint8)
    return image, cloud, boxes_2d, boxes_3d, pair_of
```

## 1. Capture the frames the matcher is lost on

`SceneRecorder` writes what you passed to `match()` or `observe()` as a SUSTechPOINTS-format
scene directory. `xcalib-review` reads that layout directly.

### Save a frame by hand

```python
from xcalib import SceneRecorder

rng = np.random.default_rng(1)
rec = SceneRecorder("capture/site1", camera="cam_front")  # cloud_format="pcd" (default) or "npy"
rec.save_calibration(stored)  # once per site: K and [R|t]

image, cloud, boxes_2d, boxes_3d, pair_of = synth_frame(rng)
frame = rec.save_frame(
    image, cloud, boxes_2d, boxes_3d,
    ids_3d=[100 + j for j in range(len(boxes_3d))],  # LiDAR box obj_id
    types_3d=["Car"] * len(boxes_3d),
    ids_2d=[10 + i for i in range(len(boxes_2d))],   # the camera tracker's own ids
    types_2d=["Car"] * len(boxes_2d),
)
print(frame, rec.frames())
```

```text
000000 ['000000']
```

`SceneRecorder(root, camera="camera_0", cloud_format="pcd")` takes the scene directory, the
camera name (a folder under `camera/`) and the point-cloud format. Any other `cloud_format`
raises `ValueError`.

| `save_frame` argument | Type | Notes |
| --- | --- | --- |
| `image` | `[H, W, 3]` `uint8`, RGB | The array you pass to `match()`. Written as JPEG, which is lossy. |
| `point_cloud` | `[P, >=3]` float, meters | Columns 0 to 3 are kept. Column 3 is stored as intensity: pass `point_cloud[:, :3]` if your fourth column is something else, such as a ring index. |
| `bboxes_2d` | `[K, 4]` | `(x1, y1, x2, y2)` in pixels. |
| `bboxes_3d` | `[M, 6]` | Extents or center plus size, detected per row as in the [input protocol](protocol.md). |
| `frame` | `str`, optional | Frame id. The default is a six-digit counter that continues from the highest numeric id already on disk. An id that already exists is overwritten. |
| `ids_3d`, `types_3d` | length `M`, optional | LiDAR box `obj_id` and `obj_type`. The default ids are `1..M`: positions within this call, not tracker ids. |
| `ids_2d`, `types_2d` | length `K`, optional | `ids_2d` are your camera tracker's ids and land in `obj_attr` as `t<id>`. |

`save_frame` returns the frame id. `rec.frames()` lists the ids in the `lidar/` folder.

### What is on disk

```python
root = Path("capture/site1")
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    print(path.relative_to(root).as_posix())

boxes = json.loads((root / "label/000000.json").read_text())["objs"]
rects = json.loads((root / "label_fusion/camera/cam_front/000000.json").read_text())["objs"]
b, r = boxes[0], rects[0]
print("3D box:", b["obj_id"], b["obj_type"], {k: round(v, 2) for k, v in b["psr"]["position"].items()})
print("2D rect:", repr(r["obj_id"]), r["obj_attr"], r["annotator"], {k: round(v) for k, v in r["rect"].items()})
```

```text
calib/camera/cam_front.json
camera/cam_front/000000.jpg
label/000000.json
label_fusion/camera/cam_front/000000.json
lidar/000000.pcd
3D box: 100 Car {'x': 9.02, 'y': 3.04, 'z': 0.61}
2D rect: '' t10 xcalib {'x1': 447, 'y1': 327, 'x2': 527, 'y2': 408}
```

Camera boxes are written with an empty `obj_id` and `annotator: "xcalib"`. A camera box gets a
LiDAR id only when a person accepts a pair in review, so a captured frame never contains a model
guess that looks like a verified pair.

### Intensity

A fourth point-cloud column is kept as `intensity`: a binary PCD with `FIELDS x y z intensity`, or
a four-column `.npy`. The matcher reads only x, y and z. The review window shades its top-down
view by intensity. A three-column cloud stays three columns; no intensity is invented.

```python
pcd = root / "lidar/000000.pcd"
header = b""
with open(pcd, "rb") as f:
    while not header.endswith(b"DATA binary\n"):
        header += f.readline()
print(header.decode().splitlines()[2])
points = np.fromfile(pcd, dtype=np.float32, offset=len(header)).reshape(-1, 4)
print(points.shape, "identical to the input:", np.array_equal(points, cloud))

saved = cv2.cvtColor(cv2.imread(str(root / "camera/cam_front/000000.jpg")), cv2.COLOR_BGR2RGB)
print("largest pixel change from JPEG:", int(np.abs(saved.astype(int) - image).max()))
```

```text
FIELDS x y z intensity
(1160, 4) identical to the input: True
largest pixel change from JPEG: 2
```

### Capture automatically

Pass the recorder to `matcher.oneshot()`. `session.observe()` then calls `save_frame` for every
frame where `should_record(report)` is true:

```text
n_detections_2d > 0 and n_detections_3d > 0 and n_confident_matches == 0
```

`n_confident_matches` counts the one-to-one pairs whose score reaches `match_threshold` (default
`0.7`). Score ranges differ per model, so tune `match_threshold` on site data as the
[protocol](protocol.md#4-outputs) advises.

```python
from xcalib import Matcher

matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")  # first call downloads weights
rec = SceneRecorder("capture/site1", camera="cam_front")
session = matcher.oneshot(K, recorder=rec)  # record_lost_frames=True by default
session.set_calibration(stored)             # or session.calibrate() once enough pairs are buffered

for image, point_cloud, boxes_2d, boxes_3d in stream:
    session.observe(image, point_cloud, boxes_2d, boxes_3d)
```

To check the wiring without a drifted rig, set `match_threshold` above every model's score range.
Every frame with detections on both sides is then "lost":

```python
from xcalib import Matcher
from xcalib.capture import should_record
from xcalib.oneshot import ObserveReport

matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")  # first call downloads weights
rec = SceneRecorder("capture/wiring_check", camera="cam_front")
session = matcher.oneshot(K, match_threshold=2.0, recorder=rec)
for seed in range(3):
    image, cloud, boxes_2d, boxes_3d, _ = synth_frame(np.random.default_rng(100 + seed))
    report = session.observe(image, cloud, boxes_2d, boxes_3d)
print(report.n_detections_2d, report.n_detections_3d, report.n_confident_matches)
print(rec.frames())

# record_lost_frames=False keeps the recorder attached but turns the trigger off.
manual = SceneRecorder("capture/manual_only", camera="cam_front")
session = matcher.oneshot(K, match_threshold=2.0, recorder=manual, record_lost_frames=False)
session.observe(image, cloud, boxes_2d, boxes_3d)
print(manual.frames())

def report_with(n_confident, n_2d=3, n_3d=4):
    return ObserveReport(frame_id=0, n_detections_2d=n_2d, n_detections_3d=n_3d,
                         n_confident_matches=n_confident)

print(should_record(report_with(0)), should_record(report_with(2)), should_record(report_with(0, n_2d=0)))
```

```text
6 6 0
['000000', '000001', '000002']
[]
True False False
```

The trigger is a heuristic:

- "Lost" means the matcher found no confident pair. The trigger does not look at the projection
  matrix. A frame the matcher is confidently wrong about is not recorded, and a frame whose camera
  boxes have no LiDAR counterpart (an object outside LiDAR range) is.
- If you have a better alarm, keep `record_lost_frames=False` and call `save_frame` yourself.
- A failure inside `save_frame` is logged as a warning. It does not stop `observe()`.
- The session passes no ids and does not save a calibration. Automatically captured frames get
  positional LiDAR ids `1..M` and no tracker ids, and the scene has no `calib/camera/` file until
  you call `rec.save_calibration(...)`. Call `save_frame` yourself when you have stable ids.

### What capture does not do

- **Limit what it writes.** Every lost frame is saved. A rig that stays lost writes every frame
  until the disk is full. Apply your own policy with `record_lost_frames=False` and `save_frame`.
- **Store box rotation.** Boxes are axis-aligned, as in the matcher input. `psr.rotation` is
  always zero.
- **Preserve pixels exactly.** Images are re-encoded as JPEG (the check above shows the largest
  change on a smooth test image).
- **Store lens distortion.** `save_calibration` writes `K` and `[R|t]` only. Reading it back gives
  intrinsics without distortion. Capture rectified images if your lens is distorted.
- **Check what you save.** `save_calibration` accepts any successful `CalibrationResult`,
  including a gate candidate that was rejected, and overwrites the existing file. Keep a copy of
  the old one.

## 2. Review the captured frames

```bash
pip install "XCalib[gui]"
xcalib-review --scene capture/site1 --camera cam_front
```

The window loads a matcher from `<scene>/calib/xcalib/<camera>_model/`. `SceneRecorder` does not
create that folder, and without it the window reports `no trained model`. Save the matcher you
recorded with:

```python
matcher.save_pretrained("capture/site1/calib/xcalib/cam_front_model")
```

The window shows the image beside a top-down LiDAR view, and one card per camera box with a ranked
list of LiDAR candidates. Each candidate shows its score. When `calib/camera/<camera>.json`
exists, which `save_calibration` writes, it also shows the projected pixel distance, and a
candidate that geometry rules out carries a warning sign and is never picked automatically. That
geometric check is deliberately loose: by default a pair is compatible when the projected center
lands in front of the camera within `max(150 px, 3 x box diagonal)` of the camera box, so a drifted
matrix still helps. If every candidate for a camera box fails the check, none of them is flagged.

The window pre-fills each camera box with the matcher's suggestion and counts it as a pending
decision. On a captured frame those suggestions are the guesses that just failed. Check every card
before you press Apply.

**Apply this frame** and **Apply all** write the pending decisions into
`label_fusion/camera/<camera>/<frame>.json`: an accepted pair sets the camera box `obj_id` to the
LiDAR id, and a rejection sets `no_match: true`.

The two refine buttons train the matcher (see [Continual refine](continual-refine.md)). Neither
changes the camera-LiDAR matrix: the review window does not call `refine_projection` or the gate.

## 3. Nudge the matrix in session

`refine_projection(previous, points_3d, points_2d, *, ransac_reproj_px=8.0, min_points=6)` starts
Levenberg-Marquardt at the stored pose and returns a `CalibrationResult` whose `accepted` says
whether the result beat it.

| Parameter | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `previous` | required | | A successful `CalibrationResult`. A failed one raises `ValueError`. |
| `points_3d` | required | meters, `[N, 3]` | Box centers in the frame of `bboxes_3d`. |
| `points_2d` | required | pixels, `[N, 2]` | The matching camera box centers. |
| `ransac_reproj_px` | `8.0` | pixels | The solver is fitted only on pairs the stored pose already puts within 3 times this value (24 px). After a full run, `n_inliers` and `reproj_error_px` on the result count pairs below this value itself. |
| `min_points` | `6` | pairs | With fewer pairs, or fewer pairs inside the 24 px window, the stored pose is returned unchanged. |

The decision rule: the refined pose is adopted only if its median reprojection error over all
supplied pairs is strictly lower than the stored pose's. Pairs behind the camera count as
infinitely wrong. Otherwise the stored pose is returned with `accepted=False`.
`buffer_reproj_px` holds the median error of the returned pose.

```python
from xcalib import refine_projection

pts3, pts2 = site_pairs(np.random.default_rng(2), 40)
out = refine_projection(stored, pts3, pts2)
before, after = stored.pose_error(true_extrinsics), out.pose_error(true_extrinsics)
print(out.accepted, "|", out.message)
print(f"pose error vs truth: {before[0]:.2f} deg, {before[1]:.3f} m -> {after[0]:.2f} deg, {after[1]:.3f} m")

few = refine_projection(stored, pts3[:3], pts2[:3])
print(few.accepted, "|", few.message)

far = make_calibration(*shifted_pose(25.0, (3.0, 2.0, 1.0)))
hopeless = refine_projection(far, pts3, pts2)
print(hopeless.accepted, "|", hopeless.message)
```

```text
True | refined: median 7.9 -> 1.7 px over 40 pair(s)
pose error vs truth: 1.00 deg, 0.187 m -> 0.04 deg, 0.069 m
False | only 3 pair(s) (< 6) — kept the previous calibration
False | the previous pose explains only 0 of 40 pair(s) — too far gone to nudge; re-solve (estimate_projection) instead
```

`CalibrationSession.refine(previous, min_pairs=6)` and `session.calibrate(refine=True)` call the
same function on the pairs a session has buffered:

```python
session = matcher.oneshot(K)
session.set_calibration(stored)
session.calib_session.add_correspondences(pts2, pts3)  # session.observe() normally fills this
result = session.calibrate(refine=True)                # min_pairs=12 by default
print(result.accepted, session.calibration is result)
```

```text
True True
```

`calibrate(refine=True)` needs an existing calibration and raises `ValueError` without one. It
ignores `disambiguate`, `gate_factor` and `min_frames`. An accepted refinement also updates
`session.best_reproj_px` when its median is lower. A later fresh `session.calibrate()` is gated
against that value. Use `session.calibrate()` without `refine`, or `estimate_projection`, when you
have no matrix you trust.

**What this guarantees, and what it does not.** On the pairs you pass, the returned pose never has
a larger median error than the one you gave. It says nothing about other pairs. The criterion is
"strictly lower", with no minimum margin, so noise alone can make it accept. With the stored pose
already exact:

```python
accepted, rot, trans = 0, [], []
for seed in range(200):
    pts3, pts2 = site_pairs(np.random.default_rng(seed), 40)
    out = refine_projection(exact, pts3, pts2)
    if out.accepted:
        accepted += 1
        r, t = out.pose_error(true_extrinsics)
        rot.append(r)
        trans.append(t)
print(f"accepted {accepted}/200; error of the accepted poses: "
      f"mean {np.mean(rot):.3f} deg / {np.mean(trans):.3f} m, "
      f"max {np.max(rot):.3f} deg / {np.max(trans):.3f} m")
```

```text
accepted 148/200; error of the accepted poses: mean 0.102 deg / 0.030 m, max 0.327 deg / 0.117 m
```

`accepted` means "lower on these pairs", not "closer to the truth". For a matrix you will deploy,
use the gate in the next section.

## 4. Gate a candidate on held-out data

`build_physical_refinement_candidate(previous, fit_points_3d, fit_points_2d, validation_sets, *,
config=None)` fits a candidate pose on one set of pairs and accepts it only from pairs you supply
separately. It never writes a file.

- The candidate comes from `cv2.solvePnPRefineLM`, started at `previous`, on the fit pairs the
  stored pose puts in front of the camera and within `3 x inlier_px` (75 px by default).
  Intrinsics, including distortion, are never changed. With fewer than `min_fit_points` such
  pairs the solver is skipped and the candidate equals `previous`.
- `validation_sets` maps a name to a `(points_3d, points_2d)` tuple. The order is kept for
  display. The set named `primary_validation_set` (default `"held_out"`) must improve. Every set
  you pass must also pass the guardrails. Do not name a set `"fit"`: that key holds the fit
  metrics.
- Point units follow [Refine](#3-nudge-the-matrix-in-session): meters in the frame of
  `bboxes_3d`, and pixels.

### Gate rules

Every rule must hold for `eligible` to be true. `CalibrationGateConfig` is a frozen dataclass;
build one with the fields you want to change and pass it as `config=`.

| Field | Default | Unit | Rule |
| --- | --- | --- | --- |
| `inlier_px` | `25.0` | pixels | A pair is an inlier below this error. Also sets the 3x window used to pick fit pairs. |
| `min_fit_points` | `12` | pairs | The fit set needs this many pairs, and the stored pose must put this many within `3 x inlier_px`. |
| `min_validation_points` | `6` | pairs | Each validation set needs this many pairs. Sets with fewer are rejected and skip their other checks. |
| `min_median_improvement_px` | `0.5` | pixels | Primary set: the median error must drop by at least `max(this, ratio x baseline median)`. |
| `min_median_improvement_ratio` | `0.02` | fraction | See above. It only matters when the baseline median exceeds 25 px (`0.5 / 0.02`). |
| `max_q90_regression_px` | `2.0` | pixels | Every set: the 90th-percentile error may rise by at most this much. |
| `max_inlier_ratio_drop` | `0.02` | fraction | Every set: the share of inliers may fall by at most this much. |
| `min_positive_depth_ratio` | `0.80` | fraction | Every set: at least this share of pairs must project in front of the camera. This is an absolute floor, not a comparison with the baseline. |
| `max_depth_ratio_drop` | `0.01` | fraction | Every set: the in-front share may fall by at most this much. |
| `max_rotation_delta_deg` | `3.0` | degrees | The candidate may differ from `previous` by at most this rotation angle. |
| `max_translation_delta_m` | `1.0` | meters | And by at most this norm of the change in `t`. |
| `primary_validation_set` | `"held_out"` | | The name of the set that must improve. A missing name rejects the candidate. |

The result is a `PhysicalCalibrationCandidate`:

| Field | Meaning |
| --- | --- |
| `eligible` | `True` only if no rule produced a reason. |
| `reasons` | One string per failed rule. Empty when eligible. |
| `candidate` | A `CalibrationResult`. `candidate.accepted` equals `eligible`. `candidate.message` is `eligible physical refinement candidate`, or the reasons joined with `; `. |
| `previous` | The pose you passed in. |
| `before`, `after` | `dict[str, CalibrationMetrics]` for `"fit"` and each validation set, under `previous` and under `candidate`. |
| `rotation_delta_deg`, `translation_delta_m` | How far the candidate moved. |

`CalibrationMetrics` fields: `pairs`; `median_px` and `q90_px` (over pairs in front of the camera);
`rmse_inlier_px` and `inliers` (pairs below `inlier_px`); `inlier_ratio` (inliers over all
`pairs`); `positive_depth` and `positive_depth_ratio` (pairs in front of the camera). You can score
any pose yourself with `evaluate_calibration(calibration, points_3d, points_2d, inlier_px=25.0)`.
An empty set returns infinite errors and zero counts.

The function raises `ValueError` when `previous` is not a successful calibration, when the 3D and
2D arrays differ in length, or when any value is not finite. Everything else, including too few
pairs, comes back as `eligible=False` with a reason.

### An accepted candidate

```python
from xcalib import CalibrationGateConfig, build_physical_refinement_candidate, evaluate_calibration

rng = np.random.default_rng(3)
fit = site_pairs(rng, 80)
validation = {"held_out": site_pairs(rng, 30), "later_window": site_pairs(rng, 30)}

result = build_physical_refinement_candidate(stored, *fit, validation)
print("eligible:", result.eligible, "| reasons:", result.reasons)
print(f"pose change: {result.rotation_delta_deg:.2f} deg, {result.translation_delta_m:.3f} m")
print(f"{'set':<13}{'pairs':>5}  {'median px':>14}  {'q90 px':>14}  {'inlier ratio':>14}")
for name, b in result.before.items():
    a = result.after[name]
    print(f"{name:<13}{b.pairs:>5}  {b.median_px:>6.2f} ->{a.median_px:>5.2f}  "
          f"{b.q90_px:>6.2f} ->{a.q90_px:>5.2f}  {b.inlier_ratio:>6.2f} ->{a.inlier_ratio:>5.2f}")
print(result.candidate.message)

# evaluate_calibration scores any pose on any pairs, with the same metrics.
held_out_metrics = evaluate_calibration(result.candidate, *validation["held_out"])
print("held-out median under the candidate:", round(held_out_metrics.median_px, 2), "px")
```

```text
eligible: True | reasons: ()
pose change: 1.02 deg, 0.171 m
set          pairs       median px          q90 px    inlier ratio
fit             80    8.09 -> 1.78   10.57 -> 3.09    1.00 -> 1.00
held_out        30    8.60 -> 1.60   13.39 -> 2.97    1.00 -> 1.00
later_window    30    8.17 -> 1.82   11.55 -> 3.26    1.00 -> 1.00
eligible physical refinement candidate
held-out median under the candidate: 1.6 px
```

### Rejected candidates

Each case below changes one thing and prints the reasons:

```python
rng = np.random.default_rng(4)
fit = site_pairs(rng, 80)
held_out = site_pairs(rng, 30)
stored_pose = (stored.rotation, stored.translation)
behind = held_out[0].copy()
behind[:9, 0] *= -1  # 9 of 30 held-out points now sit behind the camera
caps = CalibrationGateConfig(max_rotation_delta_deg=0.5, max_translation_delta_m=0.05)
nearly_right = make_calibration(*shifted_pose(0.05, (0.005, 0.0, 0.0)))

cases = {
    "later window agrees with the old matrix": (stored, fit, {
        "held_out": held_out,
        "later_window": site_pairs(rng, 30, noise_px=0.05, pose=stored_pose)}, None),
    "no validation set": (stored, fit, {}, None),
    "only 5 held-out pairs": (stored, fit, {"held_out": site_pairs(rng, 5)}, None),
    "only 8 fit pairs": (stored, site_pairs(rng, 8), {"held_out": held_out}, None),
    "30% of held-out pairs behind the camera": (stored, fit, {"held_out": (behind, held_out[1])}, None),
    "pose caps of 0.5 deg and 0.05 m": (stored, fit, {"held_out": held_out}, caps),
    "stored matrix already right to 0.05 deg": (nearly_right, fit, {"held_out": held_out}, None),
}
for name, (previous, fit_set, validation_sets, config) in cases.items():
    out = build_physical_refinement_candidate(previous, *fit_set, validation_sets, config=config)
    print(f"{name}: eligible={out.eligible}")
    for reason in out.reasons:
        print("   -", reason)
```

```text
later window agrees with the old matrix: eligible=False
   - later_window q90 regresses 10.8px
no validation set: eligible=False
   - missing primary validation set 'held_out'
only 5 held-out pairs: eligible=False
   - held_out has 5 pair(s); need 6
only 8 fit pairs: eligible=False
   - fit has 8 pair(s); need 12
   - baseline explains only 8 fit pair(s); need 12 for a nudge
   - held_out median improves -4.44e-14px; need 0.5px
30% of held-out pairs behind the camera: eligible=False
   - held_out positive-depth ratio 0.700 is below 0.800
pose caps of 0.5 deg and 0.05 m: eligible=False
   - rotation delta 0.853deg exceeds 0.500deg
   - translation delta 0.200m exceeds 0.050m
stored matrix already right to 0.05 deg: eligible=False
   - held_out median improves -0.178px; need 0.5px
```

How to read the reasons:

| Reason | Cause and fix |
| --- | --- |
| `fit has N pair(s); need 12` | Too few fit pairs. Collect more reviewed frames. |
| `baseline explains only N fit pair(s); need 12 for a nudge` | Under the stored pose, too few fit pairs land in front of the camera within 75 px (`3 x inlier_px`). Either there are few pairs or the pose is far off. The solver is skipped and the candidate equals the stored pose, so the primary set usually also reports an improvement of about zero (floating-point noise). If the pose is far off, re-solve with `estimate_projection`. |
| `<set> has N pair(s); need 6` | The set is too small to judge. The set is otherwise unchecked. |
| `<set> candidate projection is non-finite` | Under the candidate, no pair in that set projects in front of the camera. Check the points and their frame. |
| `<set> median improves Xpx; need Ypx` | The primary set did not improve enough. A negative X means the candidate is worse. |
| `<set> q90 regresses Xpx` | The tail got worse, even if the median improved. Often a sign the fit pairs and this set disagree. |
| `<set> inlier ratio drops X` | Fewer pairs fall inside `inlier_px` than under the stored pose. |
| `<set> positive-depth ratio X is below 0.800` | Too many pairs project behind the camera. This floor does not compare with the baseline, so filter such pairs out of the set first. |
| `<set> positive-depth ratio drops X` | The candidate moved pairs behind the camera. |
| `missing primary validation set 'held_out'` | No set carries the name in `primary_validation_set`. Rename the set or pass `config=CalibrationGateConfig(primary_validation_set="...")`. |
| `rotation delta Xdeg exceeds Ydeg` | The candidate moved further than the cap, even if it fits better. Raise the cap only if you expect a large move. |
| `translation delta Xm exceeds Ym` | Same, for `t`. |

### What the gate cannot tell you

`eligible` means every rule above held on the pairs you supplied. It is not a statement about the
physical pose.

- **Pairs are box centers.** The pairs are camera box centers against LiDAR box centers, the same
  pairs `matcher.calibrate()` uses. A camera box center does not generally sit exactly on the
  projection of the LiDAR box center. A systematic offset in the pairs looks like a pose error,
  and the gate accepts a pose that absorbs it.
- **Independence is yours to provide.** The gate cannot tell whether a validation set overlaps the
  fit set. Split by frame or time, not by pair: one vehicle seen in consecutive frames produces
  near-duplicate pairs.
- **The thresholds are fixed numbers.** Pixel thresholds depend on your image resolution and box
  quality. Tune `inlier_px` and the pose caps for the site.

```python
rng = np.random.default_rng(5)
offset = (0.0, -8.0)  # every camera center sits 8 px above the projected LiDAR center
biased_fit = site_pairs(rng, 80, offset_px=offset)
biased_validation = {name: site_pairs(rng, 30, offset_px=offset) for name in ("held_out", "later_window")}
out = build_physical_refinement_candidate(exact, *biased_fit, biased_validation)
rot, trans = out.candidate.pose_error(true_extrinsics)
print(f"exact matrix, biased pairs: eligible={out.eligible}, "
      f"candidate is {rot:.2f} deg / {trans:.3f} m from the truth")

out = build_physical_refinement_candidate(stored, *fit, {"held_out": fit})
print("fit set passed as the held-out set: eligible =", out.eligible)


def eligible_count(previous, offset_px=(0.0, 0.0), trials=200):
    count = 0
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        fit = site_pairs(rng, 80, offset_px=offset_px)
        validation = {name: site_pairs(rng, 30, offset_px=offset_px)
                      for name in ("held_out", "later_window")}
        count += build_physical_refinement_candidate(previous, *fit, validation).eligible
    return count


print("exact matrix, noise only:  ", eligible_count(exact), "/ 200 eligible")
print("matrix 1 deg / 0.18 m off: ", eligible_count(stored), "/ 200 eligible")
print("exact matrix, 8 px offset: ", eligible_count(exact, (0.0, -8.0)), "/ 200 eligible")
```

```text
exact matrix, biased pairs: eligible=True, candidate is 0.58 deg / 0.033 m from the truth
fit set passed as the held-out set: eligible = True
exact matrix, noise only:   0 / 200 eligible
matrix 1 deg / 0.18 m off:  200 / 200 eligible
exact matrix, 8 px offset:  200 / 200 eligible
```

These counts come from synthetic pairs with 1.5 px of noise. They show how the rules behave. They
are not a field validation.

## 5. End to end

This block runs the whole loop on synthetic frames. It records 36 frames, simulates the review by
writing the LiDAR ids a person would accept, collects the reviewed pairs, gates a candidate, and
saves it only if it is eligible.

```python
rng = np.random.default_rng(6)
rec = SceneRecorder("capture/site2", camera="cam_front")
rec.save_calibration(stored)
truth = {}
for _ in range(36):
    image, cloud, boxes_2d, boxes_3d, pair_of = synth_frame(rng)
    truth[rec.save_frame(image, cloud, boxes_2d, boxes_3d)] = pair_of

# Review. xcalib-review's Apply writes these same fields: the LiDAR id on the camera box.
root = Path("capture/site2")
fusion = root / "label_fusion/camera/cam_front"
for frame, pair_of in truth.items():
    path = fusion / f"{frame}.json"
    doc = json.loads(path.read_text())
    for i, rect in enumerate(doc["objs"]):
        rect["obj_id"] = str(pair_of[i] + 1)  # default ids are 1..M
        rect.pop("annotator", None)
    path.write_text(json.dumps(doc))


def reviewed_pairs(frames):
    """(3D centers, 2D centers) for every camera box that carries a LiDAR id."""
    points_3d, points_2d = [], []
    for frame in frames:
        rects = json.loads((fusion / f"{frame}.json").read_text())["objs"]
        labels = json.loads((root / "label" / f"{frame}.json").read_text())["objs"]
        boxes = {box["obj_id"]: box for box in labels}
        for rect in rects:
            box = boxes.get(rect["obj_id"])
            if box is None:  # not reviewed, or rejected
                continue
            x1, y1, x2, y2 = (rect["rect"][k] for k in ("x1", "y1", "x2", "y2"))
            points_2d.append([(x1 + x2) / 2, (y1 + y2) / 2])
            p = box["psr"]["position"]
            points_3d.append([p["x"], p["y"], p["z"]])
    return np.array(points_3d).reshape(-1, 3), np.array(points_2d).reshape(-1, 2)


frames = rec.frames()
fit = reviewed_pairs(frames[:20])
validation = {"held_out": reviewed_pairs(frames[20:28]), "later_window": reviewed_pairs(frames[28:])}
print("pairs:", len(fit[0]), {name: len(v[0]) for name, v in validation.items()})

result = build_physical_refinement_candidate(stored, *fit, validation)
print("eligible:", result.eligible, "| reasons:", result.reasons)
for name, b in result.before.items():
    print(f"  {name:<12} median {b.median_px:5.2f} -> {result.after[name].median_px:5.2f} px")

if result.eligible:
    calib_file = root / "calib/camera/cam_front.json"
    shutil.copy(calib_file, calib_file.with_name("cam_front.previous.json"))  # keep the old matrix
    rec.save_calibration(result.candidate)
    session.set_calibration(result.candidate)  # and hand it to a live session

before, after = stored.pose_error(true_extrinsics), result.candidate.pose_error(true_extrinsics)
print(f"pose error vs truth (known only because the data is synthetic): "
      f"{before[0]:.2f} deg, {before[1]:.3f} m -> {after[0]:.2f} deg, {after[1]:.3f} m")
```

```text
pairs: 120 {'held_out': 48, 'later_window': 48}
eligible: True | reasons: ()
  fit          median  7.91 ->  1.86 px
  held_out     median  7.28 ->  1.80 px
  later_window median  7.86 ->  1.46 px
pose error vs truth (known only because the data is synthetic): 1.00 deg, 0.187 m -> 0.02 deg, 0.015 m
```

Before you deploy a real candidate, look at it. `draw_calibration_overlay` projects the cloud with
the new pose onto a frame ([Visualization](visualization.md)):

```python
from xcalib import draw_calibration_overlay

overlay = draw_calibration_overlay(
    image, cloud, intrinsics=K, rotation=result.candidate.rotation,
    translation=result.candidate.translation, bboxes_3d=boxes_3d,
)
print(overlay.shape, overlay.dtype)
```

```text
(720, 1280, 3) uint8
```

## Guarantees and heuristics

| Piece | Guaranteed by the code | Heuristic |
| --- | --- | --- |
| Capture | The camera box `obj_id` is empty at capture. A failure in `save_frame` never stops `observe()`. | Which frames count as lost. |
| Review | Apply writes only the fusion label files. | The matcher's suggestions and the geometric warning. |
| `refine_projection` | On the pairs you pass, the returned pose's median error is not larger than the stored pose's. | That the refined pose is better on other pairs. |
| Gate | `eligible` is true only if every rule holds. A missing primary set, too few pairs or a pose jump past the caps give `eligible=False`. Intrinsics are unchanged. Nothing is written. | That passing the rules means a better physical pose. That the validation sets are independent. The default thresholds. |

## See also

- [Input protocol](protocol.md): frame formats, box conventions and intrinsics.
- [Continual refine](continual-refine.md): training the matcher from reviewed pairs.
- [Visualization](visualization.md): `draw_calibration_overlay`.
