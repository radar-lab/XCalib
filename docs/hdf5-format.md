# HDF5 cache format (UTC and A9 caches)

Training (`scripts/paper/train_hdf5.py`), validation (`scripts/paper/validate_paper.py`), and ONNX verification
(`scripts/paper/validate_onnx.py`) all consume **the same** preprocessed cache layout. Inputs are expected
to be read-only HDF5 files produced upstream (outside this repo) from raw ITS recordings. The public
A9 r02_s01 caches shipped under `datasets/a9_dataset_r02_s01/hdf5_cache/` follow this exact schema.

---

## Root groups

### `images/`

One subgroup per logical camera (`<camera_name>`).

| Dataset                     | Dtype                       | Shape        | Meaning                                                                                         |
| --------------------------- | --------------------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| `images/<camera_name>/data` | variable-length uint8 bytes | `[n_frames]` | JPEG-compressed RGB frames stored as a 1-D array of byte blobs; indexed by decoded frame index. |

Decoded frames are interpreted as OpenCV-default **BGR**, then converted to RGB in the loader
(`cv2.imdecode`, `cvtColor`). Each blob must decode successfully or the frame is skipped.

### `point_clouds/`

One subgroup per lidar (`<lidar_name>`), then **per-frame** groups keyed by `<frame_key>` (string;
often a zero-padded index such as `"00042"` matching the `/labels/` tree).

| Dataset                                | Dtype   | Shape    | Meaning                                      |
| -------------------------------------- | ------- | -------- | -------------------------------------------- |
| `point_clouds/<lidar>/<frame_key>/xyz` | float32 | `[P, 3]` | Global XYZ lidar coordinates for that sweep. |

Different frames may contain different counts `P`; only the XYZ columns are mandatory.

### `labels/`

All detections live under a sensor tag (typically the primary camera bucket). The loader uses **the
first** key alphabetically inside `labels/` as `label_sensor`; single-camera caches should therefore
expose exactly one subtree.

Inside `labels/<sensor>/<frame_key>/`:

| Dataset                   | Dtype           | Shape      | Meaning                                                                                                                                                                                                                                                                                  |
| ------------------------- | --------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_camera_detections`   | scalar int      | `()`       | `K` bounding boxes projected from the paired camera detection stream.                                                                                                                                                                                                                    |
| `num_lidar_detections`    | scalar int      | `()`       | `M` LiDAR object boxes after projection / association.                                                                                                                                                                                                                                   |
| `camera_bbox_2d`          | float32         | `[>=K, 4]` | Image-plane `(x1, y1, x2, y2)` in **pixel coordinates** aligned with decoded `images[...]` after resize/crop bookkeeping in the authoring pipeline—only `[0:K)` rows are consumed.                                                                                                       |
| `lidar_bbox_3d`           | float32         | `[>=M, 6]` | Either axis-aligned extents `(xmin,ymin,zmin,xmax,ymax,zmax)` or centre-style `(cx,cy,cz, dx,dy,dz)`; only `[0:M)` rows are consumed (see cropping helper heuristics).                                                                                                                   |
| `lidar_centers` (optional) | float32 | `[>=M, 3]` | Centre `(x, y, z)` of each oriented LiDAR cuboid. Present in the A9 caches; read only with `box_source="envelope"` (see [3D boxes](#3d-boxes-stored-and-envelope)). |
| `lidar_dimensions` (optional) | float32 | `[>=M, 3]` | Full size `(length, width, height)` of each cuboid along its own axes. |
| `lidar_rotations` (optional) | float32 | `[>=M, 4]` | Unit quaternion `(x, y, z, w)` turning the cuboid's axes into the world axes (A9: a turn about z). |
| `match_matrix`            | uint8/bool-like | `[K, M]`   | Entry `(i,j)=1` if camera detection `i` matches LiDAR detection `j` for supervised training/eval. Rows may repeat ground truth from multi-object frames.                                                                                                                                 |
| `camera_names` (optional) | bytes / str     | `[K]`      | Per-detection source camera id embedded by the authoring pipeline. The loader decodes **every** referenced camera for the frame and the cropping path picks each 2-D box's own source image (multi-camera caches such as A9 south1+south2). Missing → first camera name under `images/`. |

Splits such as **`utc_train.h5`**, **`utc_val.h5`**, and **`utc_test.h5`** are ordinary files
following this schema; semantics (which temporal slice is withheld) belong in your dataset readme,
not in the HDF5 itself.

`/calibration` is **optional** for this package; UTC caches omit it—the loader never reads
rigid-body extrinsics from HDF5 during matching-only scripts.

---

## Frame keys and iteration order

`UTCFrameLoader` sorts `labels/<sensor>/` keys lexicographically. Image bytes are fetched from
`images/<camera_name>/data[<frame_idx>]`:

- Prefer numeric `frame_key` → interpreted as linear index into the JPEG blob array.
- Non-numeric keys fall back to the sorted position among all frame keys — **consistent but
  fragile** across regenerations.

---

## 3D boxes: `stored` and `envelope`

The point-cloud crop is **axis-aligned**: it keeps the points inside the box, enlarged by
`bbox_expansion`. It therefore holds a vehicle whole only when the box encloses the vehicle, whatever
the vehicle's heading.

In the A9 r02_s01 train, val and test caches we checked, `lidar_bbox_3d` is each cuboid's centre
± `lidar_dimensions` / 2 with the rotation **dropped** (the length always lies on x), not the
extents that enclose it. A vehicle lying along y is then cropped over only part of its length: about
a third of the points inside its box stay in the crop (pooled over the train, val and test caches),
and most vehicles at this intersection are turned 30° or more against the axes. The same caches also carry the full oriented cuboid
(`lidar_centers`, `lidar_dimensions`, `lidar_rotations`), so the enclosing extents can be derived.

`UTCFrameLoader(path, box_source=...)` chooses what the loader hands out as `bboxes_3d`:

| `box_source`           | `bboxes_3d`                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `"stored"` (default)   | `lidar_bbox_3d` exactly as the cache holds it. Every released weight and reported number uses this.   |
| `"envelope"`           | The tight axis-aligned extents around each oriented cuboid (`xcalib.data.oriented_extents`). Raises `ValueError` if the cache has no `lidar_centers`, `lidar_dimensions` or `lidar_rotations`. |

The same choice is available as `box_source=` on `xcalib.train`, `Matcher.train` and
`load_dataset`, as `--box-source` on `scripts/paper/train_hdf5.py` and `scripts/paper/validate_paper.py`,
and as a `box_source` key in a training config. The default stays `stored` because the shipped weights
and the reported numbers were produced with it. Weights trained on one source are not guaranteed to do
best on the other, so measure both. With the shipped A9 `crlite` weights and no retraining, scoring the
r02_s01 test split with `envelope` instead of `stored` boxes raised top-1 from 92.0 % to 97.1 %
(`validate_paper.py --site a9_dataset_r02_s01 --models crlite --box-source envelope`; the other
models were not measured). Callers that run the matcher on their own detections should pass boxes that
enclose the object (for an oriented detector, `oriented_extents`).

---

## What this package derives

Given a `UTCFrame`, `prepare_frame`:

1. Crops patches from the full-resolution image per **2-D** box resized to YAML `crop_size`
   (pixels).
2. Crops globally aligned point clouds inside each enlarged **3-D** bounding box (`bbox_expansion`
   default 1.25×).
3. Sub/zero-pads every LiDAR crop to `point_cloud_size` (~1024) points **with fresh RNG draws each
   call** (`train_hdf5` enables augmentation by seeding RNG per epoch).

---

## Sanity checklist before training

| Check             | Detail                                                                                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Non-empty matches | Rows with zero positives are skipped silently in the loss (`match_matrix`).                                                                                          |
| K / M truncation  | HDF5 bounding-box arrays larger than advertised counts (`num_*`) ignore trailing rows consistently with `[:K]` slices.                                               |
| Camera selection  | Single-camera caches (UTC) stay deterministic by construction; multi-camera caches (A9) must fill `camera_names` so each detection crops from its own source camera. |
| Temporal hygiene  | Mirror the paper splits: train≠val≠test filenames to avoid bleed between supervision and reported Top-1.                                                             |

For matching metrics **after** ONNX export (`validate_onnx.py`), the ONNX graph must ingest tensors
produced via the **exact same** cropping path.
