"""Package-native continual frame loader (``xcalib.continual.scene_loader``).

Covers the three things that keep the corpus coherent across front ends: the
box-geometry formulas (b2d/b3d), the v2 feature fingerprint (file
identity, NOT content hash), and the loader's key mapping + staleness guard.
"""

import json
import os

import numpy as np
import pytest

from xcalib.continual.scene_loader import (
    _input_fingerprint,
    _lidar_review_key,
    frame_input_fingerprint,
    scene_frame_loader,
)
from xcalib.continual.state import FeatureWork
from xcalib.continual.store import CorrectionRecord
from xcalib.continual.train import _FrameInputFingerprintMismatch
from xcalib.gui.scene import Scene


def _write_ascii_pcd(path, points):
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7", "FIELDS x y z", "SIZE 4 4 4", "TYPE F F F",
        "COUNT 1 1 1", f"WIDTH {len(points)}", "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0", f"POINTS {len(points)}", "DATA ascii",
    ]
    body = [" ".join(str(float(v)) for v in p) for p in points]
    path.write_text("\n".join(header + body) + "\n")


def _psr(px, py, pz, sx, sy, sz):
    return {"position": {"x": px, "y": py, "z": pz},
            "scale": {"x": sx, "y": sy, "z": sz},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}}


def _make_scene(tmp_path, camera="cam", frame="f0"):
    import cv2

    root = tmp_path / "scene"
    (root / "camera" / camera).mkdir(parents=True)
    (root / "lidar").mkdir(parents=True)
    (root / "label").mkdir(parents=True)
    (root / "label_fusion" / "camera" / camera).mkdir(parents=True)
    cv2.imwrite(str(root / "camera" / camera / f"{frame}.jpg"),
                np.zeros((8, 8, 3), np.uint8))
    _write_ascii_pcd(root / "lidar" / f"{frame}.pcd",
                     [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    boxes = [
        {"obj_id": "10", "obj_type": "Car", "psr": _psr(1, 2, 3, 2, 4, 6)},
        {"obj_id": "", "obj_type": "Pedestrian", "psr": _psr(5, 5, 5, 1, 1, 1)},
    ]
    (root / "label" / f"{frame}.json").write_text(json.dumps({"objs": boxes}))
    rects = [
        {"rect": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
         "obj_attr": "t1", "obj_type": "Car"},
        {"rect": {"x1": 50, "y1": 60, "x2": 70, "y2": 80},
         "obj_attr": "t2", "obj_type": "Pedestrian"},
    ]
    (root / "label_fusion" / "camera" / camera / f"{frame}.json").write_text(
        json.dumps({"objs": rects}))
    return root, camera, frame


def test_box_geometry_follows_the_documented_formula(tmp_path):
    # b2d = [[x1,y1,x2,y2], ...] float32; b3d = rotation-aware AABB [xmin..zmax]
    # float32. The v2 fingerprint hashes exactly these arrays, so another front end
    # that builds different ones would compute a different fingerprint and split the
    # corpus.
    root, camera, frame = _make_scene(tmp_path)
    scene = Scene(str(root), camera)

    b2d = Scene.rects_xyxy(scene.load_rects(frame))
    assert b2d.dtype == np.float32
    np.testing.assert_array_equal(
        b2d, np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float32))

    b3d = Scene.boxes_extents(scene.load_boxes(frame))
    assert b3d.dtype == np.float32
    # box0: center (1,2,3), scale (2,4,6), no rotation -> extents [0,0,0, 2,4,6]
    np.testing.assert_array_equal(b3d[0], np.array([0, 0, 0, 2, 4, 6], np.float32))
    # box1: center (5,5,5), scale (1,1,1) -> [4.5,4.5,4.5, 5.5,5.5,5.5]
    np.testing.assert_array_equal(
        b3d[1], np.array([4.5, 4.5, 4.5, 5.5, 5.5, 5.5], np.float32))


def test_lidar_review_key_follows_the_documented_convention():
    # labelled -> "id:<obj_id>"; anonymous -> "psr:<sha256 of canonical psr>"
    assert _lidar_review_key({"obj_id": "7", "psr": {}}) == "id:7"
    anon = _lidar_review_key({"obj_id": "", "psr": _psr(0, 0, 0, 1, 1, 1)})
    assert anon.startswith("psr:") and len(anon) == len("psr:") + 64


def test_input_fingerprint_is_file_identity_not_content(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"a" * 2048)
    pcd = tmp_path / "f.pcd"
    pcd.write_bytes(b"b" * 2048)
    b2d = np.array([[0, 0, 10, 10]], np.float32)
    b3d = np.zeros((1, 6), np.float32)
    base = _input_fingerprint(str(img), str(pcd), b2d, b3d)

    # rewrite both files' bytes but preserve each (mtime, size) -> unchanged
    for path in (img, pcd):
        stat = path.stat()
        path.write_bytes(b"Z" * 2048)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert _input_fingerprint(str(img), str(pcd), b2d, b3d) == base

    # re-Detect (moved rect) changes it -- geometry is still part of the identity
    moved = np.array([[0, 0, 11, 10]], np.float32)
    assert _input_fingerprint(str(img), str(pcd), moved, b3d) != base

    # a real re-export (new mtime) changes it
    stat = img.stat()
    os.utime(img, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert _input_fingerprint(str(img), str(pcd), b2d, b3d) != base


def _correction(scene, camera, frame, fingerprint):
    return CorrectionRecord(
        sample_key="s1", observation_key="o1", scene=scene, camera=camera,
        frame=frame, review_id="r1", input_fingerprint=fingerprint,
        model_revision="m", camera_track="1",
        rect_json=json.dumps([10.0, 20.0, 30.0, 40.0]),
        final_decision="positive", final_lidar_key="id:10",
        original_proposal_json=None,
        candidates_json=json.dumps([{"lidar_key": "id:10"}]),
        provenance="review", label_revision=1, created_at="t", updated_at="t",
    )


def test_scene_frame_loader_maps_keys_and_builds_input(tmp_path):
    root, camera, frame = _make_scene(tmp_path)
    fingerprint = frame_input_fingerprint(str(root), camera, frame)
    record = _correction("scene", camera, frame, fingerprint)
    work = FeatureWork(frame=frame, input_fingerprint=fingerprint,
                       dataset_split="train")

    frame_input = scene_frame_loader(str(tmp_path), "scene", camera, work, [record])

    # rect t1 (matches the ledger rect_json) -> its sample_key; t2 -> positional
    assert frame_input.rect_keys == ("s1", "rect:1")
    # box0 is labelled id 10; box1 is anonymous -> psr hash
    assert frame_input.lidar_keys[0] == "id:10"
    assert frame_input.lidar_keys[1].startswith("psr:")
    assert frame_input.input_fingerprint == fingerprint
    assert frame_input.image.shape[2] == 3
    assert frame_input.point_cloud.shape[1] == 3
    assert frame_input.bboxes_2d.shape == (2, 4)
    assert frame_input.bboxes_3d.shape == (2, 6)


def test_scene_frame_loader_guards_stale_frames(tmp_path):
    root, camera, frame = _make_scene(tmp_path)
    fingerprint = frame_input_fingerprint(str(root), camera, frame)
    record = _correction("scene", camera, frame, fingerprint)
    stale = FeatureWork(frame=frame, input_fingerprint="not-the-real-fingerprint",
                        dataset_split="train")

    with pytest.raises(_FrameInputFingerprintMismatch):
        scene_frame_loader(str(tmp_path), "scene", camera, stale, [record])
