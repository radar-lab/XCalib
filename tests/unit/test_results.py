"""The LiDAR objects the homography places in the image: the delivered table, a scene's own
boxes in the table's frame, and the dots drawn through the homography."""

import csv
import json

import cv2
import numpy as np
import pytest

from xcalib.gui.results import (
    _scaled, apply_homography, draw_objects, position_to_metres, scene_objects, spread_frames,
    table_frame, table_objects,
)
from xcalib.gui.scene import Scene

PLANE = (0.05, -0.02, -0.7)  # through the box centres, in the point clouds' frame
# The delivered table's frame: turned 90 degrees about z and shifted against the clouds.
TO_CLOUD = np.array([[0.0, -1, 0, 2.0], [1, 0, 0, -1.0], [0, 0, 1, 0.4], [0, 0, 0, 1]])
# Longitude and latitude of the table frame's origin, and metres per degree there.
ORIGIN, PER_DEGREE = (-85.3069, 35.0452), (91137.0, 111319.0)


def _on_plane(xy):
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([xy, PLANE[0] * xy[:, 0] + PLANE[1] * xy[:, 1] + PLANE[2]])


def _in_table(cloud):
    return (np.asarray(cloud, dtype=float) - TO_CLOUD[:3, 3]) @ TO_CLOUD[:3, :3]


def _degrees(metres):
    """x runs north and y west, as in the delivery this mimics."""
    return ORIGIN[0] - metres[1] / PER_DEGREE[0], ORIGIN[1] + metres[0] / PER_DEGREE[1]


def test_degrees_are_tied_to_metres_by_one_linear_map():
    metres = np.array([[3.0, -40.0], [25.0, 10.0], [-18.0, 31.0], [60.0, 2.0]])
    degrees = np.array([_degrees(m) for m in metres])
    matrix = position_to_metres(degrees, metres)
    assert np.allclose((np.column_stack([degrees, np.ones(4)]) @ matrix.T)[:, :2], metres, atol=1e-4)
    with pytest.raises(ValueError, match="not a linear function"):
        position_to_metres(degrees, metres ** 2)


def test_a_scaled_homography_maps_the_same_and_keeps_the_side_of_the_camera():
    homography = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0], [0.01, 0.0, -0.5]])
    scaled = _scaled(homography)
    assert scaled[2, 2] == pytest.approx(-1.0)
    spots = np.array([[10.0, -3.0], [25.0, 4.0], [70.0, 0.5]])
    assert np.allclose(apply_homography(scaled, spots), apply_homography(homography, spots))
    assert np.allclose(apply_homography(np.linalg.inv(scaled), apply_homography(scaled, spots)), spots)
    lifted = np.column_stack([spots, np.ones(3)])
    assert np.array_equal(np.sign(lifted @ scaled.T)[:, 2], np.sign(lifted @ homography.T)[:, 2])
    with pytest.raises(ValueError, match="no homography"):
        _scaled(np.zeros((3, 3)))


def _scene(root, *, frame_file=False, table=None):
    """A scene with four boxes in each of three frames; ``frame_file`` records the table's
    frame, ``table`` writes the delivered LiDAR table with its position columns in
    ``"metres"`` or in ``"degrees"``."""
    for name in ("lidar", "camera/cam", "label", "label_fusion/camera/cam"):
        (root / name).mkdir(parents=True)
    grid = np.array([[x, y] for x in (10, 16, 22, 28) for y in (-3, 0, 3)], dtype=float)
    rows = []
    for index in range(3):
        frame = f"17000000{index:02d}"
        cv2.imwrite(str(root / "camera/cam" / f"{frame}.jpg"), np.full((480, 640, 3), 90, np.uint8))
        np.save(root / "lidar" / f"{frame}.npy", _on_plane(grid).astype(np.float32))
        centres = _on_plane(grid[index::3] + 0.5 * index)
        boxes = [{"obj_id": str(100 + i), "obj_type": "Car",
                  "psr": {"position": dict(zip("xyz", map(float, p))),
                          "scale": {"x": 4.0, "y": 1.8, "z": 1.6}, "rotation": {"x": 0, "y": 0, "z": 0.3}}}
                 for i, p in enumerate(centres)]
        (root / "label" / f"{frame}.json").write_text(json.dumps(boxes), encoding="utf-8")
        (root / "label_fusion/camera/cam" / f"{frame}.json").write_text(json.dumps({"objs": []}),
                                                                         encoding="utf-8")
        rows += [(frame, box["obj_id"], _in_table(p)) for box, p in zip(boxes, centres)]
    if frame_file or table:
        (root / "vendor").mkdir()
        (root / "vendor/lidar_transform.json").write_text(json.dumps({"matrix": TO_CLOUD.tolist()}),
                                                          encoding="utf-8")
    if table:
        with (root / "vendor/lidar_perception.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            local = ["x", "y", "z"] if table == "degrees" else []
            writer.writerow(["timestamp", "id", "classification", *local, "length", "width", "height",
                             "position_x", "position_y", "position_z"])
            for frame, name, metres in rows:
                position = _degrees(metres) if table == "degrees" else metres[:2]
                writer.writerow([frame, name, "VEHICLE", *(metres if local else []), 4.0, 1.8, 1.6,
                                 *(repr(float(v)) for v in position), metres[2]])
            writer.writerow(["1699999999", "5", "VEHICLE", *([1.0] * len(local)), 4.0, 1.8, 1.6,
                             1.0, 1.0, 0.0])   # a timestamp the scene has no frame of
    return Scene(str(root), "cam")


def _rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _cloud_centre(scene, obj):
    box = next(b for b in scene.load_boxes(obj["frame"]) if b["obj_id"] == obj["id"])
    return np.array([box["psr"]["position"][axis] for axis in "xyz"])


def test_a_scene_without_a_table_gives_its_own_boxes(tmp_path):
    scene = _scene(tmp_path / "scene")
    assert table_objects(scene) is None
    matrix, recorded = table_frame(scene)
    assert not recorded and np.allclose(matrix, np.eye(4))
    objects = scene_objects(scene, matrix)
    assert len(objects) == 12 and {obj["frame"] for obj in objects} == set(scene.frames())
    for obj in objects[:4]:
        assert np.allclose(obj["centre"], _cloud_centre(scene, obj))
        assert obj["position"] == obj["centre"][:2]
        assert obj["type"] == "Car" and obj["size"] == [4.0, 1.8, 1.6]


def test_own_boxes_are_given_in_the_recorded_table_frame(tmp_path):
    scene = _scene(tmp_path / "scene", frame_file=True)
    matrix, recorded = table_frame(scene)
    assert recorded and np.allclose(matrix, TO_CLOUD)
    for obj in scene_objects(scene, matrix)[:4]:
        in_cloud = _cloud_centre(scene, obj)
        assert np.allclose(TO_CLOUD[:3, :3] @ obj["centre"] + TO_CLOUD[:3, 3], in_cloud, atol=1e-5)
        assert not np.allclose(obj["position"], in_cloud[:2], atol=0.5)  # really another frame


@pytest.mark.parametrize("columns", ["metres", "degrees"])
def test_the_delivered_table_is_read_as_it_is(tmp_path, columns):
    scene = _scene(tmp_path / "scene", table=columns)
    objects, form = table_objects(scene)
    assert form == ("metres" if columns == "metres" else "longitude, latitude")
    rows = [row for row in _rows(tmp_path / "scene/vendor/lidar_perception.csv")
            if row["timestamp"] in scene.frames()]
    assert len(objects) == len(rows) == 12
    metres = ("x", "y", "z") if columns == "degrees" else ("position_x", "position_y", "position_z")
    for obj, row in zip(objects, rows):
        assert (obj["frame"], obj["id"]) == (row["timestamp"], row["id"])
        assert obj["position"] == [float(row["position_x"]), float(row["position_y"])]  # as written
        assert obj["centre"] == [float(row[key]) for key in metres]
        assert obj["type"] == "VEHICLE" and obj["size"] == [4.0, 1.8, 1.6]


def test_dots_outside_the_image_are_skipped():
    image = np.zeros((100, 200, 3), np.uint8)
    objects = [{"id": "7", "position": [50.0, 40.0]}, {"id": "8", "position": [500.0, 40.0]}]
    assert draw_objects(image, objects, np.eye(3)) == 1
    assert tuple(image[40, 50]) == (0, 0, 255)


def test_the_frames_shown_are_spread_over_the_scene():
    counts = {f"{index:03d}": index % 7 for index in range(60)}
    picked = spread_frames(counts, 3)
    assert len(picked) == 3 and picked == sorted(picked) and all(counts[name] for name in picked)
    assert spread_frames({"a": 1, "b": 0}, 3) == ["a"]   # frames without objects are never shown
