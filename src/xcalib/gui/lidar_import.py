"""Write vendor LiDAR detections as SUSTechPOINTS 3D box drafts.

A perception CSV reports boxes in the vendor's site frame (z up, heading about
z), while a scene's point clouds are in one sensor's own frame. ``label_lidar``
first checks whether the two already agree. If not, it recovers the rigid
transform from the data: the sensor origin from each row's reported distance
to its primary sensor, the tilt from the ground plane, and the heading by
maximizing how many moving points fall inside the boxes. A transform that does
not explain the points is rejected rather than written.

A delivery can also arrive with its clouds already in the vendor's site frame.
That fits its own boxes, but not a model or an extrinsic made in an earlier
scene's sensor frame: ``label_lidar(frame_of=...)`` rewrites such clouds into
that scene's frame.

Boxes keep the vendor track id and are drafts that still need human review.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .rename_retry import rename_when_free
from .scene import Scene, box3d_corners, load_points_xyz

ANNOTATOR = "vendor-lidar-draft"
_GEOMETRY = ("position_x", "position_y", "position_z", "length", "width", "height", "heading")
# An export with these columns reports the box in metres and radians there;
# its position_x/position_y are then longitude and latitude.
_LOCAL_GEOMETRY = ("x", "y", "z", "length", "width", "height", "yaw")
_TYPES = {"car": "Car", "vehicle": "Car", "van": "Van", "bus": "Bus", "truck": "Truck",
          "trailer": "Truck", "large_vehicle": "Truck", "pedestrian": "Pedestrian",
          "person": "Pedestrian", "bicycle": "BicycleRider"}


def read_vendor_boxes(path):
    """Group vendor rows by frame. ``box`` is x, y, z, length, width, height, yaw [rad].

    A track gets its most frequent known class in every frame, because vendors
    report a new track as unknown until its classification settles.
    """
    frames = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = set(reader.fieldnames or [])
        geometry = _LOCAL_GEOMETRY if names.issuperset(_LOCAL_GEOMETRY) else _GEOMETRY
        missing = ({"timestamp", "id"} | set(geometry)) - names
        if missing:
            raise ValueError(f"LiDAR vendor CSV has missing columns: {sorted(missing)}")
        for row in reader:
            try:
                box = [float(row[key]) for key in geometry]
                distance = float(row.get("distance_to_primary_sensor") or "nan")
                score = float(row.get("classification_confidence") or "nan")
            except (TypeError, ValueError):
                continue
            if not all(map(math.isfinite, box)) or min(box[3:6]) <= 0:
                continue
            if geometry is _GEOMETRY:  # heading in degrees
                box[6] = math.radians(box[6])
            names = (row.get("sub_classification"), row.get("classification"))
            kind = next((_TYPES[n.strip().lower()] for n in names
                         if n and n.strip().lower() in _TYPES), "Unknown")
            frames[row["timestamp"].strip()].append({
                "id": row["id"].strip(), "type": kind, "score": score, "box": box,
                "sensor": (row.get("primary_sensor") or "").strip(), "distance": distance})
    votes = defaultdict(Counter)
    for entries in frames.values():
        for row in entries:
            if row["type"] != "Unknown":
                votes[row["id"]][row["type"]] += 1
    for entries in frames.values():
        for row in entries:
            if votes[row["id"]]:
                row["type"] = votes[row["id"]].most_common(1)[0][0]
    return dict(frames)


def _valid(points):
    """Finite returns only; organized clouds pad missing returns with zeros."""
    points = np.asarray(points, dtype=float)
    return points[np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)]


def _rotation(axis, angle):
    """Rotation about coordinate axis 0, 1 or 2."""
    c, s = math.cos(angle), math.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    matrix = np.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def _inside(points, boxes, margin=0.0):
    """Boolean [points, boxes]: which points lie in each yawed box (same frame)."""
    delta = points[:, None, :] - boxes[None, :, :3]
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    half = boxes[:, 3:6] / 2 + margin
    return ((np.abs(c * delta[..., 0] + s * delta[..., 1]) < half[:, 0])
            & (np.abs(c * delta[..., 1] - s * delta[..., 0]) < half[:, 1])
            & (np.abs(delta[..., 2]) < half[:, 2]))


def _moving_points(clouds, voxel=0.3, static=0.25):
    """Drop points in voxels occupied in at least ``static`` of the clouds."""
    def keys(points):
        cell = np.floor(points / voxel).astype(np.int64) + (1 << 20)
        return (cell[:, 0] << 42) | (cell[:, 1] << 21) | cell[:, 2]

    found = [keys(points) for points in clouds]
    cells, count = np.unique(np.concatenate([np.unique(k) for k in found]), return_counts=True)
    background = cells[count >= max(2, static * len(clouds))]
    return [points[~np.isin(k, background)] for points, k in zip(clouds, found)]


def _ground_normal(points, rng, iterations=400, tolerance=0.1):
    """Unit normal of the dominant plane, pointing to the sensor at the origin."""
    points = points[np.linalg.norm(points, axis=1) < 50]
    if len(points) > 20000:
        points = points[rng.choice(len(points), 20000, replace=False)]
    if len(points) < 3:
        raise ValueError("too few LiDAR points to find the ground plane")
    best, inliers = 0, None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        mask = np.abs((points - a) @ (normal / length)) < tolerance
        if mask.sum() > best:
            best, inliers = mask.sum(), mask
    if inliers is None:
        raise ValueError("no ground plane found in the LiDAR points")
    center = points[inliers].mean(axis=0)
    normal = np.linalg.svd(points[inliers] - center, full_matrices=False)[2][2]
    return -normal if normal @ center > 0 else normal


def _sensor_origin(centers, distances):
    """Multilaterate a sensor from object centers and their reported ranges.

    Objects lie near one plane, so the height has a mirrored second solution;
    both are refined and the better fit wins. Returns (origin, median error).
    """
    planar = np.linalg.lstsq(np.c_[-2 * centers[:, :2], np.ones(len(centers))],
                             distances ** 2 - (centers[:, :2] ** 2).sum(axis=1), rcond=None)[0]
    best = None
    for side in (5.0, -5.0):
        origin = np.array([planar[0], planar[1], np.median(centers[:, 2]) + side])
        for _ in range(30):
            delta = centers - origin
            ranges = np.linalg.norm(delta, axis=1)
            error = ranges - distances
            keep = np.abs(error) <= max(0.3, 2.5 * np.median(np.abs(error)))
            origin = origin + np.linalg.lstsq(-delta[keep] / ranges[keep, None], -error[keep],
                                              rcond=None)[0]
        error = float(np.median(np.abs(np.linalg.norm(centers - origin, axis=1) - distances)))
        if best is None or error < best[1]:
            best = (origin, error)
    return best


def _score(moving, boxes, rotation, origin):
    """Moving points inside the boxes and near them, and boxes that got any point."""
    inside = near = hit = 0
    for points, box in zip(moving, boxes):
        if len(points) and len(box):
            local = points @ rotation + origin  # sensor frame -> vendor frame
            mask = _inside(local, box)
            inside += int(mask.sum())
            near += int(_inside(local, box, margin=0.6).sum())
            hit += int(mask.any(axis=0).sum())
    return inside, near, hit


def _quality(moving, boxes, rotation, origin):
    inside, near, hit = _score(moving, boxes, rotation, origin)
    total = sum(len(box) for box in boxes)
    return {"inside_fraction": round(inside / near, 4) if near else 0.0,
            "boxes_with_points": round(hit / total, 4) if total else 0.0}


def _accepted(quality):
    return quality["inside_fraction"] >= 0.8 and quality["boxes_with_points"] >= 0.5


def _matrix(rotation, origin):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, -rotation @ origin
    return matrix.tolist()


def estimate_transform(scene, *, samples=60, seed=0, progress=print):
    """Find the 4x4 vendor-to-PCD transform of a scene, or raise ``ValueError``.

    Assumes a fixed sensor above the ground at the PCD origin and a z-up
    vendor frame. Returns ``{"matrix", "source", "sensor", quality...}`` where
    ``source`` is ``"identity"`` when the CSV is already in the PCD frame.
    """
    scene = Path(scene)
    rows = read_vendor_boxes(scene / "vendor" / "lidar_perception.csv")
    clouds = Scene(str(scene), "")
    frames = [frame for frame in clouds.frames() if frame in rows]
    if len(frames) < 4:
        raise ValueError("too few LiDAR frames have vendor detections")
    picks = np.unique(np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int))
    frames = [frames[i] for i in picks]
    loaded = [_valid(load_points_xyz(clouds.lidar_path(frame))) for frame in frames]
    moving = _moving_points(loaded)

    def boxes_of(sensor=None):
        return [np.array([r["box"] for r in rows[f] if sensor in (None, r["sensor"])],
                         dtype=float).reshape(-1, 7) for f in frames]

    result = {"source": "identity", "sensor": None, "sampled_frames": len(frames)}
    quality = _quality(moving, boxes_of(), np.eye(3), np.zeros(3))
    if _accepted(quality):
        return {"matrix": np.eye(4).tolist(), **result, **quality}

    ranged = defaultdict(list)
    for entries in rows.values():
        for row in entries:
            if row["sensor"] and math.isfinite(row["distance"]):
                ranged[row["sensor"]].append(row["box"][:3] + [row["distance"]])
    if not ranged:
        raise ValueError("vendor boxes do not fit the point clouds and the CSV has no "
                         "distance_to_primary_sensor column; pass the transform as matrix=")
    rng = np.random.default_rng(seed)
    normal = _ground_normal(loaded[0], rng)
    axis, cosine = np.cross([0.0, 0.0, 1.0], normal), normal[2]
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    upright = np.eye(3) + skew + skew @ skew / (1 + cosine)  # vendor z onto the ground normal

    best = None
    for sensor, values in ranged.items():
        values = np.array(values)
        if len(values) < 30:
            continue
        if len(values) > 20000:
            values = values[rng.choice(len(values), 20000, replace=False)]
        origin = _sensor_origin(values[:, :3], values[:, 3])[0]
        boxes = boxes_of(sensor)
        yaw = max(np.radians(np.arange(0, 360, 2)), key=lambda a: _score(
            moving[::3], boxes[::3], upright @ _rotation(2, a), origin)[0])
        rotation = upright @ _rotation(2, yaw)
        inside = _score(moving, boxes, rotation, origin)[0]
        if best is None or inside > best[0]:
            best = (inside, sensor, rotation, origin, boxes)
    if best is None:
        raise ValueError("too few vendor rows report a distance to their primary sensor")
    _, sensor, rotation, origin, boxes = best
    if progress:
        progress(f"Aligning the vendor LiDAR boxes with the point clouds (sensor {sensor}) ...")

    def contrast(parameters):
        r = rotation
        for axis in range(3):
            r = r @ _rotation(axis, parameters[axis])
        inside, near, _ = _score(moving, boxes, r, origin + parameters[3:])
        return 2 * inside - near, r, origin + parameters[3:]

    parameters = np.zeros(6)
    value = contrast(parameters)[0]
    for angle, shift in ((0.5, 0.2), (0.2, 0.1), (0.1, 0.05), (0.05, 0.02), (0.02, 0.01)):
        improved = True
        while improved:
            improved = False
            for index in range(6):
                for sign in (1, -1):
                    trial = parameters.copy()
                    trial[index] += sign * (math.radians(angle) if index < 3 else shift)
                    trial_value = contrast(trial)[0]
                    if trial_value > value:
                        parameters, value, improved = trial, trial_value, True
    _, rotation, origin = contrast(parameters)
    quality = _quality(moving, boxes, rotation, origin)
    if not _accepted(quality):
        raise ValueError(f"vendor boxes could not be aligned with the point clouds ({quality}); "
                         "pass the vendor-to-PCD transform as matrix=")
    return {"matrix": _matrix(rotation, origin), **result, "source": "estimated",
            "sensor": sensor, **quality}


def _label(row, rotation, origin):
    """One SUSTechPOINTS box: psr in the PCD frame, rotation as intrinsic XYZ euler."""
    box = row["box"]
    position = rotation @ (np.array(box[:3]) - origin)
    m = rotation @ _rotation(2, box[6])
    euler = (math.atan2(-m[1, 2], m[2, 2]), math.asin(max(-1.0, min(1.0, m[0, 2]))),
             math.atan2(-m[0, 1], m[0, 0]))
    label = {"obj_id": row["id"], "obj_type": row["type"],
             "psr": {"position": dict(zip("xyz", map(float, position))),
                     "rotation": dict(zip("xyz", euler)),
                     "scale": dict(zip("xyz", box[3:6]))},
             "annotator": ANNOTATOR}
    if math.isfinite(row["score"]):
        label["score"] = row["score"]
    return label


def _is_draft(path):
    """True when a label file holds only untouched drafts (an editor of this label
    format, such as SUSTechPOINTS, clears the annotator of a box a person edits)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, list) and all(
        isinstance(o, dict) and o.get("annotator") == ANNOTATOR for o in data)


def write_lidar_labels(scene, matrix, *, min_points=5, progress=print):
    """Write ``label/<frame>.json`` drafts; never replace a label a person edited.

    A box is written only when its frame's cloud has ``min_points`` returns
    inside it, so objects seen only by another sensor are left out.
    """
    scene = Path(scene)
    rows = read_vendor_boxes(scene / "vendor" / "lidar_perception.csv")
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    origin = -rotation.T @ matrix[:3, 3]
    clouds = Scene(str(scene), "")
    directory = scene / "label"
    directory.mkdir(exist_ok=True)
    report = {"frames_written": 0, "boxes_written": 0, "boxes_without_points": 0,
              "frames_kept_from_reviewer": 0}
    frames = clouds.frames()
    for index, frame in enumerate(frames, 1):
        target = directory / f"{frame}.json"
        if target.exists() and not _is_draft(target):
            report["frames_kept_from_reviewer"] += 1
        else:
            entries = list({row["id"]: row for row in rows.get(frame, [])}.values())
            labels = []
            if entries:
                points = _valid(load_points_xyz(clouds.lidar_path(frame))) @ rotation + origin
                counts = _inside(points, np.array([row["box"] for row in entries])).sum(axis=0)
                labels = [_label(row, rotation, origin)
                          for row, count in zip(entries, counts) if count >= min_points]
            report["boxes_without_points"] += len(entries) - len(labels)
            if labels:
                target.write_text(json.dumps(labels), encoding="utf-8")
                report["frames_written"] += 1
                report["boxes_written"] += len(labels)
            else:
                target.unlink(missing_ok=True)
        if progress and (index % 500 == 0 or index == len(frames)):
            progress(f"{scene.name}: LiDAR labels {index}/{len(frames)}")
    return report


def _occupied(points, voxel=0.5, reach=40.0):
    """Keys of the voxels that hold a point within ``reach`` of the origin."""
    points = points[np.linalg.norm(points, axis=1) < reach]
    cell = np.floor(points / voxel).astype(np.int64) + (1 << 20)
    return np.unique((cell[:, 0] << 42) | (cell[:, 1] << 21) | cell[:, 2])


def _sample_clouds(scene, count=6):
    clouds = Scene(str(scene), "")
    frames = clouds.frames()
    picks = np.unique(np.linspace(0, len(frames) - 1, min(count, len(frames))).astype(int))
    return [_valid(load_points_xyz(clouds.lidar_path(frames[i]))) for i in picks]


def _move_clouds(scene, matrix, progress):
    """Rewrite every cloud of a scene through ``matrix`` as x, y, z; the old
    folder is replaced only once all clouds are written."""
    from ..capture import _write_pcd

    lidar, staging, old = scene / "lidar", scene / "lidar.moving", scene / "lidar.before"
    if old.exists():
        raise ValueError(f"an earlier move of {lidar} was interrupted; import the data again")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    rotation, shift = matrix[:3, :3], matrix[:3, 3]
    files = sorted(p for p in lidar.iterdir() if p.suffix.lower() in (".pcd", ".npy", ".bin"))
    for index, path in enumerate(files, 1):
        _write_pcd(str(staging / f"{path.stem}.pcd"), _valid(load_points_xyz(str(path))) @ rotation.T + shift)
        if progress and (index % 250 == 0 or index == len(files)):
            progress(f"{scene.name}: point clouds moved {index}/{len(files)}")
    rename_when_free(lidar, old, progress)
    try:
        rename_when_free(staging, lidar, progress)
    except PermissionError:
        rename_when_free(old, lidar, progress)   # the scene keeps its clouds as they were
        raise
    shutil.rmtree(old)


def join_lidar_frame(scenes, reference, *, samples=60, progress=print):
    """Bring scenes into the LiDAR frame of ``reference``; return that frame's transform.

    A model and an extrinsic belong to the LiDAR frame of the scene they were
    made in. A later delivery of the same site can hold its clouds in another
    frame, such as the vendor's site frame instead of one sensor's. Its clouds
    are then rewritten through the reference scene's vendor-to-PCD transform,
    after checking that they come to lie on the reference scene's clouds;
    clouds that do not line up are left alone and ``ValueError`` is raised.
    """
    reference = Path(reference)
    record = reference / "vendor" / "lidar_transform.json"
    if not record.is_file():
        raise ValueError(f"{reference.name} has no vendor/lidar_transform.json: its LiDAR frame is unknown")
    target = np.asarray(json.loads(record.read_text(encoding="utf-8"))["matrix"], dtype=float).reshape(4, 4)
    saved = scenes[0] / "vendor" / "lidar_transform.json"
    own = (json.loads(saved.read_text(encoding="utf-8")) if saved.is_file()
           else estimate_transform(scenes[0], samples=samples, progress=progress))
    move = target @ np.linalg.inv(np.asarray(own["matrix"], dtype=float).reshape(4, 4))
    turn = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(move[:3, :3]) - 1) / 2))))
    if turn < 1.0 and np.linalg.norm(move[:3, 3]) < 0.3:  # the same frame, found twice
        return own
    space = np.unique(np.concatenate([_occupied(cloud) for cloud in _sample_clouds(reference)]))
    # Half a voxel of play, so that points beside a voxel border still count: the
    # same site then matches above 0.9, a frame 2 degrees or 1 m off near 0.7.
    nudges = np.r_[np.zeros((1, 3)), np.eye(3), -np.eye(3)] * 0.25
    moved = np.unique(np.concatenate([_occupied(cloud @ move[:3, :3].T + move[:3, 3] + nudge)
                                      for cloud in _sample_clouds(scenes[0]) for nudge in nudges]))
    overlap = float(np.isin(space, moved).mean())
    if overlap < 0.8:
        raise ValueError(f"the point clouds do not come to lie on those of {reference.name} "
                         f"(only {overlap:.0%} of its space is matched); the two deliveries do not "
                         "share a vendor frame")
    for scene in scenes:
        (scene / "vendor" / "lidar_transform.json").unlink(missing_ok=True)  # stale while clouds move
        _move_clouds(scene, move, progress)
    quality = {key: own[key] for key in ("inside_fraction", "boxes_with_points") if key in own}
    return {"matrix": target.tolist(), "source": "reference scene", "reference": reference.name,
            "cloud_overlap": round(overlap, 4), **quality}


def label_lidar(scenes, *, matrix=None, frame_of=None, min_points=5, samples=60, progress=print):
    """Write vendor LiDAR drafts into every scene's ``label/`` folder.

    The scenes must share one sensor and vendor frame: unless ``matrix``
    (vendor to PCD, 4x4) is given, the transform is found once from the first
    scene. It is saved beside the vendor rows as ``vendor/lidar_transform.json``.
    With ``frame_of``, a scene of the same site (the base scene of a refine
    delivery), the scenes first join that scene's LiDAR frame: see
    ``join_lidar_frame``.
    """
    scenes = [Path(scene) for scene in scenes]
    if frame_of is not None:
        transform = join_lidar_frame(scenes, frame_of, samples=samples, progress=progress)
    elif matrix is None:
        transform = estimate_transform(scenes[0], samples=samples, progress=progress)
    else:
        transform = {"matrix": np.asarray(matrix, dtype=float).reshape(4, 4).tolist(),
                     "source": "provided"}
    report = {"transform": transform, "scenes": {}}
    for scene in scenes:
        report["scenes"][scene.name] = write_lidar_labels(
            scene, transform["matrix"], min_points=min_points, progress=progress)
    for scene in scenes:  # written last: its presence means the drafts are complete
        (scene / "vendor" / "lidar_transform.json").write_text(
            json.dumps(transform, indent=2), encoding="utf-8")
    return report


def preview(scene, frame, camera=None, *, extent=45.0, size=720):
    """RGB check image of the saved labels: top-down points with 3D boxes, and
    the camera image with its 2D boxes when ``camera`` is given."""
    import cv2

    source = Scene(str(scene), camera or "")
    points = _valid(load_points_xyz(source.lidar_path(frame)))

    def pixels(xy):  # x to the right, y up
        return np.round((np.asarray(xy) * [1, -1] + extent) * size / (2 * extent)).astype(np.int32)

    top = np.zeros((size, size, 3), np.uint8)
    at = pixels(points[:, :2])
    at = at[((at >= 0) & (at < size)).all(axis=1)]
    top[at[:, 1], at[:, 0]] = 170
    for box in source.load_boxes(frame):
        corners = box3d_corners(box["psr"])  # index = 4 * x side + 2 * y side + z side
        outline = pixels((corners[[0, 2, 6, 4], :2] + corners[[1, 3, 7, 5], :2]) / 2)
        cv2.polylines(top, [outline], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(top, tuple(map(int, outline.mean(axis=0))),
                 tuple(map(int, outline[2:].mean(axis=0))), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(top, str(box.get("obj_id", "")), tuple(map(int, outline.max(axis=0))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    path = source.image_path(frame) if camera else None
    image = cv2.imread(path) if path else None
    if image is None:
        return top
    for rect in source.load_rects(frame):
        r = rect["rect"]
        corner = (int(r["x1"]), int(r["y1"]))
        cv2.rectangle(image, corner, (int(r["x2"]), int(r["y2"])), (0, 255, 0), 2)
        cv2.putText(image, str(rect.get("obj_attr", "")), corner,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    image = cv2.resize(image, (round(image.shape[1] * size / image.shape[0]), size))
    return np.hstack((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), top))
