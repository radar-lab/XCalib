"""Offline tests for xcalib.data.hdf5_loader (no HDF5 file / network needed)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xcalib.data.hdf5_loader import UTCFrame, _read_calibration


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
