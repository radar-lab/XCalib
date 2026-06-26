from __future__ import annotations

import numpy as np

from xcalib import (
    CameraIntrinsics,
    draw_calibration_overlay,
    draw_matching_overlay,
)


def sample_inputs():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[..., 1] = 40
    point_cloud = np.array(
        [
            [2.0, -0.5, 5.0],
            [2.5, 0.0, 5.5],
            [3.0, 0.4, 6.0],
            [4.0, 0.2, 7.0],
        ],
        dtype=np.float32,
    )
    bboxes_2d = np.array([[20, 20, 50, 55], [62, 24, 92, 58]], dtype=np.float32)
    bboxes_3d = np.array(
        [
            [1.5, -1.0, 4.5, 2.8, 0.5, 6.2],
            [3.4, -0.4, 6.4, 4.6, 0.8, 7.8],
        ],
        dtype=np.float32,
    )
    return image, point_cloud, bboxes_2d, bboxes_3d


def test_draw_matching_overlay_returns_rgb_image():
    image, point_cloud, bboxes_2d, bboxes_3d = sample_inputs()

    overlay = draw_matching_overlay(
        image,
        point_cloud,
        bboxes_2d,
        bboxes_3d,
        matches=[(0, 0, 0.92), (1, 1, 0.81)],
        output_height=160,
    )

    assert overlay.dtype == np.uint8
    assert overlay.ndim == 3
    assert overlay.shape[2] == 3
    assert overlay.shape[0] > image.shape[0]


def test_draw_calibration_overlay_accepts_projection():
    image, point_cloud, _bboxes_2d, bboxes_3d = sample_inputs()
    projection = np.array(
        [
            [80.0, 0.0, 60.0, 0.0],
            [0.0, 80.0, 40.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    overlay = draw_calibration_overlay(
        image,
        point_cloud,
        projection=projection,
        bboxes_3d=bboxes_3d,
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_draw_calibration_overlay_accepts_intrinsics_pose():
    image, point_cloud, _bboxes_2d, bboxes_3d = sample_inputs()
    intrinsics = CameraIntrinsics(fx=80.0, fy=80.0, cx=60.0, cy=40.0)

    overlay = draw_calibration_overlay(
        image,
        point_cloud,
        intrinsics=intrinsics,
        rotation=np.eye(3),
        translation=np.zeros(3),
        bboxes_3d=bboxes_3d,
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
