"""
SceneRecorder round-trips: captured frames must open as a SUSTechPOINTS-style
scene (the whole point — GUI correction of deployed-site failures).

Run with:
    pixi run python -m pytest tests/unit/test_capture.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from xcalib import CalibrationResult, CameraIntrinsics, SceneRecorder
from xcalib.capture import should_record
from xcalib.gui.scene import Scene, load_points_xyz
from xcalib.oneshot import ObserveReport


def _frame_inputs(seed=0, k=2, m=3, cols=3):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    pts = rng.uniform(-30, 30, size=(400, cols)).astype(np.float32)
    b2 = np.array([[10, 10, 50, 40], [80, 20, 120, 60]], dtype=np.float32)[:k]
    # extents form (lo <= hi)
    b3 = np.array([[18, 1, -1, 22, 3, 1],
                   [23, -9, -1, 27, -7, 1],
                   [30, 4, -1, 34, 6, 1]], dtype=np.float32)[:m]
    return image, pts, b2, b3


def test_save_frame_roundtrips_as_scene(tmp_path):
    rec = SceneRecorder(tmp_path / "cap", camera="cam")
    image, pts, b2, b3 = _frame_inputs()
    fid = rec.save_frame(image, pts, b2, b3,
                         ids_3d=["7", "9", "12"], types_3d=["Car", "Car", "Bus"],
                         ids_2d=[4, 5], types_2d=["Car", "Car"])
    assert fid == "000000"

    scene = Scene(str(tmp_path / "cap"), "cam")
    assert scene.frames() == ["000000"]
    assert scene.image_path(fid).endswith(".jpg")

    got = load_points_xyz(scene.lidar_path(fid))          # binary PCD round-trip
    assert got.shape == pts.shape and np.allclose(got, pts)

    boxes = scene.load_boxes(fid)
    assert [b["obj_id"] for b in boxes] == ["7", "9", "12"]
    # extents form converted to center+size psr
    p0 = boxes[0]["psr"]
    assert p0["position"]["x"] == pytest.approx(20.0)
    assert p0["scale"]["x"] == pytest.approx(4.0)
    ext = Scene.boxes_extents(boxes)
    assert np.allclose(ext[0], [18, 1, -1, 22, 3, 1], atol=1e-5)

    rects = scene.load_rects(fid)
    # the partner's tracker ids are the camera's OWN track (obj_attr), the
    # shared obj_id stays EMPTY — model output never masquerades as truth
    assert [r.get("obj_attr") for r in rects] == ["t4", "t5"]
    assert all(r["obj_id"] == "" for r in rects)
    assert all(r["annotator"] == "xcalib" for r in rects)


def test_save_frame_center_form_and_defaults(tmp_path):
    rec = SceneRecorder(tmp_path / "cap", camera="cam", cloud_format="npy")
    image, pts, b2, _ = _frame_inputs()
    # center+size form: center may exceed size component-wise -> not extents
    b3 = np.array([[20, 2, 0.5, 4, 2, 1.5]], dtype=np.float32)
    fid = rec.save_frame(image, pts, b2, b3)
    scene = Scene(str(tmp_path / "cap"), "cam")
    assert scene.lidar_path(fid).endswith(".npy")
    box = scene.load_boxes(fid)[0]
    assert box["obj_id"] == "1"                            # 1-based enumeration
    assert box["psr"]["position"]["x"] == pytest.approx(20.0)
    assert box["psr"]["scale"]["y"] == pytest.approx(2.0)
    rects = scene.load_rects(fid)
    assert "obj_attr" not in rects[0]                      # no tracker ids given


def test_frame_counter_continues_from_disk(tmp_path):
    rec = SceneRecorder(tmp_path / "cap", camera="cam")
    image, pts, b2, b3 = _frame_inputs()
    assert rec.save_frame(image, pts, b2, b3) == "000000"
    assert rec.save_frame(image, pts, b2, b3) == "000001"
    # a NEW recorder on the same directory keeps counting, never overwrites
    rec2 = SceneRecorder(tmp_path / "cap", camera="cam")
    assert rec2.save_frame(image, pts, b2, b3) == "000002"
    assert rec2.frames() == ["000000", "000001", "000002"]


def test_save_calibration_roundtrip(tmp_path):
    rec = SceneRecorder(tmp_path / "cap", camera="cam")
    intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=80.0, cy=60.0)
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    t = np.array([0.1, -0.2, 0.3])
    calib = CalibrationResult(success=True, intrinsics=intr, rotation=R, translation=t,
                              projection=intr.K @ np.hstack([R, t.reshape(3, 1)]))
    rec.save_calibration(calib)

    got = Scene(str(tmp_path / "cap"), "cam").calibration()
    assert got is not None and got.success
    assert np.allclose(got.rotation, R) and np.allclose(got.translation, t)
    assert got.intrinsics.fx == pytest.approx(800.0)

    with pytest.raises(ValueError, match="SUCCESSFUL"):
        rec.save_calibration(CalibrationResult(success=False))


def test_should_record_trigger():
    lost = ObserveReport(frame_id=0, n_detections_2d=3, n_detections_3d=4,
                         n_confident_matches=0)
    fine = ObserveReport(frame_id=1, n_detections_2d=3, n_detections_3d=4,
                         n_confident_matches=2)
    empty = ObserveReport(frame_id=2, n_detections_2d=0, n_detections_3d=4,
                          n_confident_matches=0)
    assert should_record(lost)
    assert not should_record(fine)       # the matcher is doing its job
    assert not should_record(empty)      # nothing to match — not model failure


def test_cloud_format_validation(tmp_path):
    with pytest.raises(ValueError, match="cloud_format"):
        SceneRecorder(tmp_path, cloud_format="ply")


def test_intensity_roundtrips_through_pcd_and_npy(tmp_path):
    """A 4th (intensity) column survives capture — the GUI colors the BEV by it;
    xyz-only consumers keep working via load_points_xyz."""
    from xcalib.gui.scene import load_points

    for fmt in ("pcd", "npy"):
        rec = SceneRecorder(tmp_path / fmt, camera="cam", cloud_format=fmt)
        image, pts4, b2, b3 = _frame_inputs(cols=4)
        fid = rec.save_frame(image, pts4, b2, b3)
        scene = Scene(str(tmp_path / fmt), "cam")
        got = load_points(scene.lidar_path(fid))
        assert got.shape == (400, 4)
        assert np.allclose(got, pts4)
        assert load_points_xyz(scene.lidar_path(fid)).shape == (400, 3)

    # 3-column clouds stay 3 columns (no fake intensity invented)
    rec3 = SceneRecorder(tmp_path / "xyz", camera="cam")
    image, pts3, b2, b3 = _frame_inputs(cols=3)
    fid = rec3.save_frame(image, pts3, b2, b3)
    got = load_points(Scene(str(tmp_path / "xyz"), "cam").lidar_path(fid))
    assert got.shape == (400, 3) and np.allclose(got, pts3)
