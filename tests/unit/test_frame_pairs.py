"""With an extrinsic, the LiDAR box that projects closest to a camera box is its pair: the
stricter check behind the doubt marks of ``xcalib.gui.model_pairs``."""

import json

import numpy as np
import pytest

from xcalib.gui.model_pairs import frame_pairs
from xcalib.gui.scene import Scene, box3d_corners
from xcalib.gui.workflow import save_calibration

CAMERA = "front"
K = np.array([[800.0, 0, 640], [0, 800, 360], [0, 0, 1]])
POSE = np.eye(4)
POSE[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]  # LiDAR x forward -> camera z
POSE[:3, 3] = [0.2, -0.1, 0.3]


def _box(obj_id, x, y, kind="Car"):
    return {"obj_id": obj_id, "obj_type": kind, "psr": {
        "position": {"x": x, "y": y, "z": 0.0}, "rotation": dict.fromkeys("xyz", 0.0),
        "scale": {"x": 4.0, "y": 2.0, "z": 1.5}}}


def _rect(box, kind="Car", shift=0.0, **extra):
    """The camera box of a LiDAR box: where its corners project, moved sideways by `shift`."""
    camera = box3d_corners(box["psr"]) @ POSE[:3, :3].T + POSE[:3, 3]
    uv = (camera @ K.T)[:, :2] / camera[:, 2:]
    return {"obj_type": kind, "obj_id": "", **extra, "rect": {
        "x1": uv[:, 0].min() + shift, "y1": uv[:, 1].min(),
        "x2": uv[:, 0].max() + shift, "y2": uv[:, 1].max()}}


def _scene(root, frames):
    for folder in ("lidar", "label", f"label_fusion/camera/{CAMERA}"):
        (root / folder).mkdir(parents=True)
    for frame, (boxes, rects) in frames.items():
        np.save(root / "lidar" / f"{frame}.npy", np.zeros((1, 3), np.float32))
        (root / "label" / f"{frame}.json").write_text(json.dumps(boxes))
        (root / "label_fusion" / "camera" / CAMERA / f"{frame}.json").write_text(
            json.dumps({"objs": rects}))
    return Scene(str(root), CAMERA)


def test_the_closest_projection_is_the_pair_and_a_lidar_box_is_used_once(tmp_path):
    near, far, beside, alone = (_box("1", 20, 0), _box("2", 30, 0.6), _box("3", 20, 6),
                                _box("4", 20, -8))
    person = _box("5", 15, -3, "Pedestrian")
    rects = [_rect(far), _rect(near),                      # one car behind the other: both
             _rect(beside, shift=15.0),                    # centres lie inside both boxes
             _rect(_box("x", 25, 12)),                     # no LiDAR box on it
             _rect(person, "Car")]                         # a car box on a pedestrian
    scene = _scene(tmp_path, {"000": ([near, far, beside, alone, person], rects)})
    with pytest.raises(ValueError, match="no extrinsic"):
        frame_pairs(scene, "000")
    save_calibration(scene, K, POSE)

    pairs = frame_pairs(scene, "000")

    by_rect = {round(p["rect"][0], 1): p["lidar_id"] for p in pairs}
    assert by_rect == {round(rects[0]["rect"]["x1"], 1): "2", round(rects[1]["rect"]["x1"], 1): "1",
                       round(rects[2]["rect"]["x1"], 1): "3"}
    assert [p["px"] for p in pairs] == sorted(p["px"] for p in pairs) and pairs[-1]["px"] > 5


def test_decided_camera_boxes_and_the_lidar_boxes_they_hold_are_left_alone(tmp_path):
    one, two, three = _box("1", 20, 0), _box("2", 20, 6), _box("3", 20, -6)
    scene = _scene(tmp_path, {
        "000": ([one, two, three, _box("9", 40, 3)],
                [_rect(one), _rect(two, obj_id="9"), _rect(three, no_match=True)]),
        "001": ([one], [_rect(one, obj_id="1"), _rect(one, shift=4.0)])})  # a second camera box on a paired car
    save_calibration(scene, K, POSE)

    assert [p["lidar_id"] for p in frame_pairs(scene, "000")] == ["1"]
    assert frame_pairs(scene, "001") == []
