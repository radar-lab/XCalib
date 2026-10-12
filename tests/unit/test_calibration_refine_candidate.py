from __future__ import annotations

import cv2
import numpy as np

from xcalib import (
    CalibrationGateConfig,
    CalibrationResult,
    CameraIntrinsics,
    build_physical_refinement_candidate,
)
from xcalib.oneshot.calibration import project_points


R_GT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
T_GT = np.array([0.2, -0.3, 0.5])
INTR = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)


def _result(rotation, translation):
    return CalibrationResult(
        success=True,
        intrinsics=INTR,
        rotation=rotation,
        translation=translation,
        projection=INTR.K @ np.hstack([rotation, translation.reshape(3, 1)]),
    )


def _points(n, seed, rotation=R_GT, translation=T_GT, noise=0.25):
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        [rng.uniform(8, 45, n), rng.uniform(-10, 10, n), rng.uniform(-1, 2, n)]
    )
    pixels, _ = project_points(points, INTR, rotation, translation)
    return points, pixels + rng.normal(0, noise, pixels.shape)


def _drifted():
    drift, _ = cv2.Rodrigues(np.deg2rad(1.0) * np.array([0.3, -0.5, 0.8]))
    return drift @ R_GT, T_GT + np.array([0.15, -0.10, 0.05])


def test_candidate_uses_independent_validation_and_is_eligible():
    fit = _points(80, 1)
    held = _points(30, 2)
    later = _points(30, 3)
    previous = _result(*_drifted())
    out = build_physical_refinement_candidate(
        previous, *fit, {"held_out": held, "later_window": later}
    )
    assert out.eligible and out.candidate.accepted
    assert out.after["held_out"].median_px < out.before["held_out"].median_px
    assert out.after["later_window"].q90_px < out.before["later_window"].q90_px


def test_following_window_regression_rejects_even_when_fit_improves():
    previous_pose = _drifted()
    fit = _points(80, 4)
    held = _points(30, 5)
    # Later evidence agrees with the baseline rather than the fitted candidate.
    later = _points(30, 6, *previous_pose, noise=0.05)
    out = build_physical_refinement_candidate(
        _result(*previous_pose), *fit, {"held_out": held, "later_window": later}
    )
    assert out.after["fit"].median_px < out.before["fit"].median_px
    assert not out.eligible
    assert any("later_window" in reason for reason in out.reasons)


def test_missing_independent_evidence_fails_closed():
    fit = _points(20, 7)
    out = build_physical_refinement_candidate(_result(*_drifted()), *fit, {})
    assert not out.eligible
    assert "missing primary validation" in " ".join(out.reasons)


def test_pose_delta_guard_is_independent_of_reprojection_metrics():
    fit = _points(80, 8)
    held = _points(30, 9)
    config = CalibrationGateConfig(max_rotation_delta_deg=0.01, max_translation_delta_m=0.001)
    out = build_physical_refinement_candidate(
        _result(*_drifted()), *fit, {"held_out": held}, config=config
    )
    assert not out.eligible
    assert any("delta" in reason for reason in out.reasons)
