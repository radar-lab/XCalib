"""The LiDAR objects that a homography places in the camera image, and pictures of it.

:mod:`xcalib.gui.pair_homography` fits ``[u, v, 1] ~ H @ [position_x, position_y, 1]``
to the saved pairs: from an object's position, as the delivered LiDAR table
reports it, to the pixel where the camera sees the object's centre. This module
reads those objects, applies and draws ``H``, and writes the values a projection
script takes.

Coordinates. ``H`` takes the table's ``position_x, position_y`` columns as
they are, so the delivered table is the input and nothing has to be converted:

- the table reports the vendor's site frame; when the point clouds are in one
  sensor's own frame, the import step saved the transform between the two as
  ``vendor/lidar_transform.json`` (:func:`table_frame`);
- some tables hold longitude and latitude in ``position_x, position_y`` and
  the metres in ``x, y``. Over one site the two are tied by a linear map, read
  from the table itself (:func:`position_to_metres`).

A scene without a delivered table (``vendor/lidar_perception.csv``) falls back
to its own LiDAR boxes, moved back into the table's frame.

Images. The scene's images are rectified: with the saved camera matrix and
zero distortion they need no further correction.

A homography cannot tell an object behind the camera from one in front: it
puts both at some pixel. ``H`` is scaled so that the third coordinate of
``H @ [position_x, position_y, 1]`` is positive in front of the camera and
negative behind it; whoever draws through ``H`` should skip the negative ones.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .scene import Scene

OBJECT_COLUMNS = ("timestamp", "id", "classification", "position_x", "position_y", "position_z",
                  "length", "width", "height")
# A table with these columns reports the box in metres there, and longitude and
# latitude in position_x, position_y (see lidar_import).
_LOCAL = ("x", "y", "z")
_POSITION = ("position_x", "position_y", "position_z")
_SIZE = ("length", "width", "height")


def table_frame(scene: Scene):
    """4x4 transform from the delivered LiDAR table's frame to the scene's point
    clouds, and whether the scene recorded one (identity when it did not)."""
    record = Path(scene.root) / "vendor" / "lidar_transform.json"
    if not record.is_file():
        return np.eye(4), False
    matrix = json.loads(record.read_text(encoding="utf-8"))["matrix"]
    return np.asarray(matrix, dtype=float).reshape(4, 4), True


def table_objects(scene: Scene):
    """The rows of the delivered LiDAR table for the scene's frames, or ``None``
    when the scene has no table.

    Returns ``(objects, columns)``. Each object has ``frame``, ``id``, ``type``,
    ``position`` (the table's position_x, position_y as written), ``centre``
    (x, y, z in metres, the table's frame) and ``size``. ``columns`` is
    ``"metres"`` or ``"longitude, latitude"``.
    """
    path = Path(scene.root) / "vendor" / "lidar_perception.csv"
    if not path.is_file():
        return None
    frames = set(scene.frames())
    objects = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = set(reader.fieldnames or [])
        if not names.issuperset({"timestamp", "id", "position_x", "position_y"}):
            return None
        metres = _LOCAL if names.issuperset(_LOCAL) else _POSITION
        for row in reader:
            frame = (row.get("timestamp") or "").strip()
            if frame not in frames:
                continue
            try:
                centre = [float(row[key]) for key in metres]
                position = [float(row["position_x"]), float(row["position_y"])]
                extent = [float(row.get(key) or "nan") for key in _SIZE]
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(centre + position).all():
                continue
            kind = (row.get("sub_classification") or row.get("classification") or "").strip()
            objects.append({"frame": frame, "id": (row.get("id") or "").strip(), "type": kind,
                            "position": position, "centre": centre, "size": extent})
    return objects, ("longitude, latitude" if metres is _LOCAL else "metres")


def scene_objects(scene: Scene, to_cloud):
    """The scene's own LiDAR boxes as objects, for a scene without a delivered table:
    ``position`` is then the centre's x, y in the table's frame. ``to_cloud`` is the
    transform of :func:`table_frame`."""
    rotation, shift = to_cloud[:3, :3], to_cloud[:3, 3]
    objects = []
    for frame in scene.frames():
        for box in scene.load_boxes(frame):
            psr = box.get("psr") or {}
            try:
                in_cloud = np.array([float(psr["position"][axis]) for axis in "xyz"])
                extent = [float(psr["scale"][axis]) for axis in "xyz"]
            except (KeyError, TypeError, ValueError):
                continue
            centre = ((in_cloud - shift) @ rotation).tolist()  # inverse of x -> rotation @ x + shift
            objects.append({"frame": frame, "id": str(box.get("obj_id") or "").strip(),
                            "type": str(box.get("obj_type") or ""), "position": centre[:2],
                            "centre": centre, "size": extent})
    return objects


def _scaled(homography):
    """Scale a homography so its last entry is 1 or -1, keeping the sign of the third coordinate."""
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("the camera centre lies on the plane; no homography")
    return homography / abs(homography[2, 2])


def position_to_metres(positions, metres, tolerance=0.05):
    """3x3 affine map from the table's ``(position_x, position_y, 1)`` to ``(x, y, 1)``
    in metres, fitted to rows that give both.

    Longitude and latitude are linear in the site's metres over a few hundred
    metres; ``ValueError`` when the rows are further than ``tolerance`` metres
    from one linear map.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    metres = np.asarray(metres, dtype=float).reshape(-1, 2)
    if len(positions) < 3:
        raise ValueError("too few table rows to relate position_x, position_y to metres")
    origin = positions.mean(axis=0)  # degrees differ in the sixth digit: fit about the mean
    design = np.column_stack([positions - origin, np.ones(len(positions))])
    coefficients, *_ = np.linalg.lstsq(design, metres, rcond=None)
    error = float(np.linalg.norm(design @ coefficients - metres, axis=1).max())
    if error > tolerance:
        raise ValueError("position_x, position_y of the LiDAR table are not a linear function of "
                         f"its x, y (off by up to {error:.2f} m)")
    linear = coefficients[:2].T
    matrix = np.eye(3)
    matrix[:2, :2], matrix[:2, 2] = linear, coefficients[2] - linear @ origin
    return matrix


def apply_homography(homography, xy):
    """Map ``[N, 2]`` points through a 3x3 homography."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    mapped = np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(homography, dtype=float).T
    return mapped[:, :2] / mapped[:, 2:3]


def draw_objects(image, objects, homography):
    """Mark each object on ``image`` (BGR, changed in place) where the homography puts
    its ``position``: a red dot with the LiDAR id beside it. An object that lands
    outside the image is left out. Returns how many were drawn.
    """
    import cv2

    height, width = image.shape[:2]
    drawn = 0
    for obj in objects:
        x, y = apply_homography(homography, [obj["position"]])[0]
        if not (np.isfinite([x, y]).all() and 0 <= x < width and 0 <= y < height):
            continue
        u = int(np.clip(round(float(x)), 0, width - 1))
        v = int(np.clip(round(float(y)), 0, height - 1))
        cv2.circle(image, (u, v), 7, (0, 0, 255), -1)
        cv2.putText(image, str(obj["id"]), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA)
        drawn += 1
    return drawn


def spread_frames(frames, count):
    """``count`` of ``frames`` (name -> number of objects) spread over their order: the
    order is cut into ``count`` stretches and each gives the fullest of its middle frames."""
    names = [name for name in frames if frames[name]]
    if len(names) <= count:
        return names
    picked = []
    for stretch in np.array_split(np.arange(len(names)), count):
        middle = stretch[len(stretch) // 2]
        nearby = [names[i] for i in stretch if abs(i - middle) <= 5]
        picked.append(max(nearby, key=lambda name: frames[name]))
    return picked


def _array(rows, form):
    return ",\n".join("        [" + ", ".join(format(float(value), form) for value in row) + "]"
                      for row in rows)


def _inputs(frame, image_dir, table, output, camera_matrix, homography, columns):
    """The results as Python assignments, for a script that reads one image and the
    table rows of one timestamp and maps ``[position_x, position_y, 1]`` to a pixel."""
    return f'''"""Values written by xcalib for one scene: the parameters of a script that projects
the LiDAR table's (position_x, position_y) through a homography. The data stays
as delivered; only these values change."""

from pathlib import Path

import numpy as np

# A timestamp found in both the image file name and the table.
TARGET_TIMESTAMP = "{frame}"

# The delivered images and LiDAR table, as they are.
image_dir = Path(r"{image_dir}")
csv_path = Path(r"{table}")
output_dir = Path(r"{output}")

output_dir.mkdir(parents=True, exist_ok=True)

# The images are already rectified: this camera matrix with zero distortion
# describes them, and undistorting with it leaves them as they are.
K = np.array(
    [
{_array(camera_matrix, ".10f")},
    ],
    dtype=np.float64,
)
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
alpha = 1

# [position_x, position_y, 1] -> [pixel_x, pixel_y, 1] of the object's centre.
# position_x, position_y are the table's columns as written ({columns}).
# The third coordinate is positive in front of the camera and negative behind it.
H_ground_to_rectified = np.array(
    [
{_array(homography, ".17g")},
    ],
    dtype=np.float64,
)
'''
