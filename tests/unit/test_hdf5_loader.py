"""Offline tests for xcalib.data.hdf5_loader (no HDF5 file / network needed)."""

from __future__ import annotations

import math

import cv2
import h5py
import numpy as np
import pytest

from xcalib.data.hdf5_loader import (
    BOX_SOURCES,
    UTCFrame,
    UTCFrameLoader,
    _read_boxes_3d,
    _read_calibration,
)


def _two_camera_frame() -> UTCFrame:
    """A frame carrying both cameras' 2D detections, as A9 caches do."""
    return UTCFrame(
        frame_key="0001",
        image=np.zeros((4, 4, 3), np.uint8),
        point_cloud=np.zeros((10, 3), np.float32),
        bboxes_2d=np.arange(20, dtype=np.float32).reshape(5, 4),
        bboxes_3d=np.zeros((3, 6), np.float32),
        match_matrix=np.zeros((5, 3), bool),
        camera_name="south1",
        images={
            "south1": np.zeros((4, 4, 3), np.uint8),
            "south2": np.ones((4, 4, 3), np.uint8),
        },
        camera_per_det=np.array(["south1", "south2", "south1", "south2", "south2"]),
    )


def test_for_camera_filters_2d_boxes_to_one_camera():
    f = _two_camera_frame()

    img, pc, b2, b3 = f.for_camera("south1")
    # Only the south1 rows survive; the point cloud and 3D boxes are
    # sensor-global and pass through unchanged.
    np.testing.assert_array_equal(b2, f.bboxes_2d[[0, 2]])
    assert b3.shape == (3, 6) and len(pc) == 10
    assert img is f.images["south1"]

    img2, _, b2_s2, _ = f.for_camera("south2")
    assert len(b2_s2) == 3
    assert img2 is f.images["south2"]


def test_for_camera_unknown_camera_raises():
    with pytest.raises(KeyError, match="south3"):
        _two_camera_frame().for_camera("south3")


def test_for_camera_single_camera_passthrough():
    # camera_per_det=None (single-camera UTC caches) -> keep every box.
    f = UTCFrame(
        frame_key="0",
        image=np.zeros((2, 2, 3), np.uint8),
        point_cloud=np.zeros((1, 3), np.float32),
        bboxes_2d=np.zeros((4, 4), np.float32),
        bboxes_3d=np.zeros((2, 6), np.float32),
        match_matrix=np.zeros((4, 2), bool),
        camera_name="cam",
        images={"cam": np.zeros((2, 2, 3), np.uint8)},
        camera_per_det=None,
    )
    _, _, b2, _ = f.for_camera("cam")
    assert len(b2) == 4


def test_read_calibration_normalizes_keys_and_picks_frame(tmp_path):
    p = tmp_path / "mini.h5"
    with h5py.File(p, "w") as f:
        f.create_group("images").create_group("s110_camera_basler_south1_8mm")
        cam = f.create_group("calibration").create_group("camera_basler_south1_8mm")
        cam.create_dataset("intrinsics", data=np.eye(3) * 2.0)
        ex = cam.create_group("extrinsics")
        ex.create_dataset("0001", data=np.eye(4))
        ex.create_dataset("0002", data=np.eye(4) * 3.0)

    with h5py.File(p, "r") as f:
        intr, extr = _read_calibration(f, "0002")

    # the prefix-dropped calibration key is normalised to the image-stream name,
    # and the per-frame extrinsic for frame "0002" is selected.
    cam = "s110_camera_basler_south1_8mm"
    assert set(intr) == {cam} and set(extr) == {cam}
    np.testing.assert_array_equal(intr[cam], np.eye(3) * 2.0)
    np.testing.assert_array_equal(extr[cam], np.eye(4) * 3.0)


def test_read_calibration_absent_returns_empty(tmp_path):
    p = tmp_path / "noc.h5"
    with h5py.File(p, "w") as f:
        f.create_group("images")
    with h5py.File(p, "r") as f:
        intr, extr = _read_calibration(f, "0001")
    assert intr == {} and extr == {}


# ---------------------------------------------------------------------------
# 3D box source: the cache's stored bounds or the extents enclosing each cuboid
# ---------------------------------------------------------------------------

def _cuboids():
    """Three cuboids: one along x, one turned a quarter turn to lie along y, a spare row.

    ``stored`` is what a cache with the rotation dropped holds: centre +/- size / 2.
    """
    centers = np.array([[10.0, 0.0, -6.0], [20.0, 5.0, -6.0], [0.0, 0.0, 0.0]], np.float32)
    dims = np.array([[4.0, 1.8, 1.5], [4.0, 1.8, 1.5], [1.0, 1.0, 1.0]], np.float32)
    quarter = math.sin(math.pi / 4)
    rotations = np.array([[0, 0, 0, 1], [0, 0, quarter, quarter], [0, 0, 0, 1]], np.float32)
    stored = np.concatenate([centers - dims / 2, centers + dims / 2], axis=1)
    return centers, dims, rotations, stored


def _write_cache(path, *, oriented: bool, frames=None):
    """A cache in the documented layout.

    ``frames`` lists ``(frame_key, num_lidar, has_oriented)``. The default is one frame with two
    advertised LiDAR boxes (three rows stored, so slicing to ``num_lidar`` is exercised).
    """
    centers, dims, rotations, stored = _cuboids()
    frames = frames or [("0", 2, oriented)]
    with h5py.File(path, "w") as f:
        vlen = h5py.special_dtype(vlen=np.dtype("uint8"))
        images = f.create_group("images").create_group("cam0").create_dataset(
            "data", shape=(max(int(key) for key, _, _ in frames) + 1,), dtype=vlen
        )
        ok, png = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
        assert ok
        for index in range(len(images)):
            images[index] = np.frombuffer(png.tobytes(), dtype=np.uint8)
        clouds = f.create_group("point_clouds").create_group("lidar0")
        labels = f.create_group("labels").create_group("lidar0")
        for key, num_lidar, has_oriented in frames:
            clouds.create_group(key).create_dataset("xyz", data=np.zeros((4, 3), np.float32))
            g = labels.create_group(key)
            g.create_dataset("num_camera_detections", data=1)
            g.create_dataset("num_lidar_detections", data=num_lidar)
            g.create_dataset("camera_bbox_2d", data=np.array([[1, 1, 5, 5]], np.float32))
            g.create_dataset("lidar_bbox_3d", data=stored if num_lidar else np.zeros((0, 6), np.float32))
            g.create_dataset("match_matrix", data=np.ones((1, num_lidar), bool))
            if has_oriented:
                g.create_dataset("lidar_centers", data=centers)
                g.create_dataset("lidar_dimensions", data=dims)
                g.create_dataset("lidar_rotations", data=rotations)
    return path


def test_box_sources_are_stored_and_envelope():
    assert BOX_SOURCES == ("stored", "envelope")


def test_stored_boxes_are_the_cache_bounds_unchanged(tmp_path):
    _, _, _, stored = _cuboids()
    with h5py.File(_write_cache(tmp_path / "c.h5", oriented=True), "r") as f:
        got = _read_boxes_3d(f["labels"]["lidar0"]["0"], 2, "stored")
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, stored[:2])


def test_envelope_boxes_enclose_the_turned_cuboid(tmp_path):
    _, _, _, stored = _cuboids()
    with h5py.File(_write_cache(tmp_path / "c.h5", oriented=True), "r") as f:
        got = _read_boxes_3d(f["labels"]["lidar0"]["0"], 2, "envelope")
    assert got.shape == (2, 6) and got.dtype == np.float32
    # Along x the two sources agree.
    np.testing.assert_allclose(got[0], stored[0], atol=1e-5)
    # The second cuboid lies along y: 1.8 m wide in x and 4.0 m long in y. The stored box keeps the
    # 4.0 m on x; the envelope swaps them.
    np.testing.assert_allclose(got[1], [19.1, 3.0, -6.75, 20.9, 7.0, -5.25], atol=1e-5)
    assert not np.allclose(got[1], stored[1])


def test_envelope_names_the_oriented_fields_a_frame_lacks(tmp_path):
    with h5py.File(_write_cache(tmp_path / "c.h5", oriented=False), "r") as f:
        with pytest.raises(ValueError, match="lidar_centers, lidar_dimensions, lidar_rotations"):
            _read_boxes_3d(f["labels"]["lidar0"]["0"], 2, "envelope")


def test_loader_hands_out_stored_boxes_by_default_and_envelope_on_request(tmp_path):
    path = _write_cache(tmp_path / "c.h5", oriented=True)
    _, _, _, stored = _cuboids()

    with UTCFrameLoader(path) as loader:
        assert loader.box_source == "stored"
        default = loader.get_frame("0")
    np.testing.assert_array_equal(default.bboxes_3d, stored[:2])

    with UTCFrameLoader(path, box_source="envelope") as loader:
        assert loader.box_source == "envelope"
        enclosing = loader.get_frame("0")
    np.testing.assert_allclose(enclosing.bboxes_3d[1], [19.1, 3.0, -6.75, 20.9, 7.0, -5.25], atol=1e-5)
    # Everything else in the frame is the same whichever boxes are chosen.
    np.testing.assert_array_equal(enclosing.bboxes_2d, default.bboxes_2d)
    np.testing.assert_array_equal(enclosing.match_matrix, default.match_matrix)


def test_loader_rejects_an_unknown_box_source(tmp_path):
    path = _write_cache(tmp_path / "c.h5", oriented=True)
    with pytest.raises(ValueError, match="box_source must be one of stored, envelope"):
        UTCFrameLoader(path, box_source="hull")


def test_envelope_loader_fails_at_construction_when_the_cache_has_no_oriented_cuboids(tmp_path):
    path = _write_cache(tmp_path / "plain.h5", oriented=False)
    with pytest.raises(ValueError, match="plain.h5.*lidar_centers"):
        UTCFrameLoader(path, box_source="envelope")
    # The default still opens the same cache.
    with UTCFrameLoader(path) as loader:
        assert loader.get_frame("0").bboxes_3d.shape == (2, 6)


def test_envelope_check_skips_a_leading_frame_the_loader_would_not_hand_out(tmp_path):
    # Frame "0" has no LiDAR detections (get_frame skips it) and no oriented fields; frame "1" is whole.
    path = _write_cache(tmp_path / "c.h5", oriented=True, frames=[("0", 0, False), ("1", 2, True)])
    with UTCFrameLoader(path, box_source="envelope") as loader:
        frames = list(loader)
    assert [frame.frame_key for frame in frames] == ["1"]
    np.testing.assert_allclose(frames[0].bboxes_3d[1], [19.1, 3.0, -6.75, 20.9, 7.0, -5.25], atol=1e-5)


def test_a_frame_lacking_oriented_fields_names_the_file_and_the_frame(tmp_path):
    path = _write_cache(tmp_path / "c.h5", oriented=True, frames=[("0", 2, True), ("1", 2, False)])
    with UTCFrameLoader(path, box_source="envelope") as loader:
        assert loader.get_frame("0") is not None
        with pytest.raises(ValueError, match="c.h5 frame 1: .*lidar_centers"):
            loader.get_frame("1")


def test_a_degenerate_quaternion_names_the_file_and_the_frame(tmp_path):
    path = _write_cache(tmp_path / "c.h5", oriented=True)
    with h5py.File(path, "r+") as f:
        f["labels"]["lidar0"]["0"]["lidar_rotations"][0] = [0, 0, 0, 0]
    with UTCFrameLoader(path, box_source="envelope") as loader:
        with pytest.raises(ValueError, match="c.h5 frame 0: .*zero length"):
            loader.get_frame("0")
    # The stored boxes never read the rotation, so the default still works on the same cache.
    with UTCFrameLoader(path) as loader:
        assert loader.get_frame("0").bboxes_3d.shape == (2, 6)
