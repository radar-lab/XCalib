"""
refine_projection / CalibrationSession.refine: nudge an existing calibration,
adopt only when the full-set median reprojection improves — never worse than
what the site already had.

Run with:
    pixi run python -m pytest tests/unit/test_refine_calibration.py -q
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from xcalib import CalibrationResult, CameraIntrinsics, refine_projection
from xcalib.oneshot import CalibrationSession
from xcalib.oneshot.calibration import project_points

#: LiDAR (x fwd, y left, z up) -> camera (x right, y down, z fwd)
R_GT = np.array([[0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [1.0, 0.0, 0.0]])
T_GT = np.array([0.2, -0.3, 0.5])
INTR = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)


def _result(R, t) -> CalibrationResult:
    return CalibrationResult(success=True, intrinsics=INTR, rotation=R, translation=t,
                             projection=INTR.K @ np.hstack([R, t.reshape(3, 1)]))


def _pairs(n=40, seed=3, noise_px=0.5):
    rng = np.random.default_rng(seed)
    pts3 = np.column_stack([rng.uniform(8, 45, n),
                            rng.uniform(-10, 10, n),
                            rng.uniform(-1, 2, n)])
    uv, _ = project_points(pts3, INTR, R_GT, T_GT)
    return pts3, uv + rng.normal(0, noise_px, size=uv.shape)


def _drifted(rot_deg=1.0, dt=(0.15, -0.10, 0.05)):
    drift, _ = cv2.Rodrigues(np.deg2rad(rot_deg) * np.array([0.3, -0.5, 0.8]))
    return drift @ R_GT, T_GT + np.asarray(dt)


def test_refine_recovers_small_drift():
    pts3, pts2 = _pairs()
    R0, t0 = _drifted()
    out = refine_projection(_result(R0, t0), pts3, pts2)
    assert out.accepted and "refined" in out.message
    rot_err, trans_err = out.pose_error(np.hstack([R_GT, T_GT.reshape(3, 1)]))
    prev_rot_err, _ = _result(R0, t0).pose_error(np.hstack([R_GT, T_GT.reshape(3, 1)]))
    assert rot_err < prev_rot_err          # strictly closer to the truth
    assert rot_err < 0.2 and trans_err < 0.05
    assert out.buffer_reproj_px < 2.0


def test_refine_keeps_an_already_good_calibration():
    pts3, pts2 = _pairs()
    out = refine_projection(_result(R_GT.copy(), T_GT.copy()), pts3, pts2)
    # LM on inliers of a near-perfect pose cannot beat it meaningfully; either
    # verdict is fine, but the returned pose must never be WORSE
    assert out.buffer_reproj_px <= 1.0
    assert np.allclose(out.rotation, R_GT, atol=1e-3)
    assert np.allclose(out.translation, T_GT, atol=1e-2)


def test_refine_refuses_a_hopeless_prior():
    pts3, pts2 = _pairs()
    R0, t0 = _drifted(rot_deg=25.0, dt=(3.0, 2.0, 1.0))   # way past a nudge
    out = refine_projection(_result(R0, t0), pts3, pts2)
    assert not out.accepted
    assert "re-solve" in out.message
    assert np.allclose(out.rotation, R0)                   # untouched


def test_refine_with_too_few_pairs_keeps_previous():
    pts3, pts2 = _pairs(n=3)
    R0, t0 = _drifted()
    out = refine_projection(_result(R0, t0), pts3, pts2)
    assert not out.accepted and "kept the previous" in out.message
    assert np.allclose(out.rotation, R0)


def test_refine_requires_successful_previous():
    pts3, pts2 = _pairs(n=8)
    with pytest.raises(ValueError, match="SUCCESSFUL"):
        refine_projection(CalibrationResult(success=False), pts3, pts2)


def test_calibration_session_refine_uses_buffer():
    pts3, pts2 = _pairs()
    session = CalibrationSession()
    session.add_correspondences(pts2, pts3)
    R0, t0 = _drifted()
    out = session.refine(_result(R0, t0))
    assert out.accepted
    rot_err, _ = out.pose_error(np.hstack([R_GT, T_GT.reshape(3, 1)]))
    assert rot_err < 0.2

    empty = CalibrationSession()
    out2 = empty.refine(_result(R0, t0))
    assert not out2.accepted and np.allclose(out2.rotation, R0)


def test_solve_keeps_a_pose_that_explains_its_inliers_on_road_like_points():
    """Object centres on a road are nearly coplanar. The EPnP refit of the
    RANSAC consensus set then misses its own inliers by far over a hundred
    pixels on this data; the solve must return a pose that fits them."""
    from xcalib.oneshot.calibration import estimate_projection

    intrinsics = CameraIntrinsics(fx=860.0, fy=640.0, cx=960.0, cy=540.0)
    rotation = R_GT @ cv2.Rodrigues(np.array([0.0, 0.25, 0.1]))[0]
    translation = np.array([0.2, -0.7, 0.1])
    rng = np.random.default_rng(22)
    points = np.c_[rng.uniform(8, 35, 120), rng.uniform(-12, 12, 120), -4.0 + rng.normal(0, 0.15, 120)]
    pixels = project_points(points, intrinsics, rotation, translation)[0] + rng.normal(0, 2.0, (120, 2))
    wrong = rng.random(120) < 0.35  # mistaken pairs
    pixels[wrong] += rng.normal(0, 60, (wrong.sum(), 2))

    result = estimate_projection(points, pixels, intrinsics)

    assert result.success and result.n_inliers >= 60
    assert result.reproj_error_px < 8.0
    assert np.abs(result.translation - translation).max() < 1.0
