"""
Input-protocol tests: validate_frame_inputs severities, enforce() policies,
CameraIntrinsics, and the matcher-level validate= wiring.

Run with:
    pixi run python -m pytest tests/unit/test_protocol.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from xcalib import (  # noqa: E402
    CameraIntrinsics,
    ProtocolError,
    validate_frame_inputs,
)
from xcalib.protocol import enforce  # noqa: E402


def _good_frame(K: int = 3, M: int = 4):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    b2 = np.array([[100, 200, 220, 320]] * K, dtype=np.float32)
    b2[:, [0, 2]] += np.arange(K)[:, None] * 250
    b3 = np.array([[5.0, -2.0, -1.0, 8.0, 0.5, 0.5]] * M, dtype=np.float32)
    b3[:, [0, 3]] += np.arange(M)[:, None] * 5

    # Background scatter + ~200 points inside every 3D box, so each LiDAR
    # crop is non-empty but stays below point_cloud_size (=> deterministic
    # zero-padding instead of random sub-sampling).
    clouds = [rng.uniform(-20, 40, size=(2000, 3)).astype(np.float32)]
    for box in b3:
        lo, hi = box[:3], box[3:]
        clouds.append(rng.uniform(lo, hi, size=(200, 3)).astype(np.float32))
    pc = np.vstack(clouds)
    return image, pc, b2, b3


def _codes(violations, severity=None):
    return {
        v.code for v in violations if severity is None or v.severity == severity
    }


# ============================================================================
# validate_frame_inputs
# ============================================================================

def test_clean_frame_has_no_violations():
    assert validate_frame_inputs(*_good_frame()) == []


def test_image_violations():
    image, pc, b2, b3 = _good_frame()

    v = validate_frame_inputs(image[..., :2], pc, b2, b3)        # [H,W,2]
    assert "image.shape" in _codes(v, "error")

    v = validate_frame_inputs(image.astype(np.int32), pc, b2, b3)
    assert "image.dtype" in _codes(v, "error")

    # float images are tolerated with a warning (assumed already in [0,1])
    v = validate_frame_inputs(image.astype(np.float32) / 255.0, pc, b2, b3)
    assert "image.dtype" in _codes(v, "warning")
    assert not _codes(v, "error")

    v = validate_frame_inputs(image[:240, :320], pc, b2, b3)     # tiny frame
    assert "image.resolution" in _codes(v, "warning")


def test_point_cloud_violations():
    image, pc, b2, b3 = _good_frame()

    v = validate_frame_inputs(image, pc[:, :2], b2, b3)          # [P,2]
    assert "point_cloud.shape" in _codes(v, "error")

    bad = pc.copy()
    bad[10, 1] = np.nan
    v = validate_frame_inputs(image, bad, b2, b3)
    assert "point_cloud.finite" in _codes(v, "error")

    v = validate_frame_inputs(image, np.zeros((0, 3), np.float32), b2, b3)
    assert "point_cloud.empty" in _codes(v, "warning")


def test_bbox_violations():
    image, pc, b2, b3 = _good_frame()

    swapped = b2.copy()
    swapped[0, [0, 2]] = swapped[0, [2, 0]]                       # x2 < x1
    v = validate_frame_inputs(image, pc, swapped, b3)
    assert "bboxes_2d.order" in _codes(v, "warning")

    v = validate_frame_inputs(image, pc, b2[:, :3], b3)
    assert "bboxes_2d.shape" in _codes(v, "error")

    many = np.tile(b2[:1], (40, 1))                               # > TRT cap
    v = validate_frame_inputs(image, pc, many, b3)
    assert "bboxes_2d.count" in _codes(v, "warning")

    v = validate_frame_inputs(image, pc, b2, b3[:, :5])
    assert "bboxes_3d.shape" in _codes(v, "error")

    nan3 = b3.copy()
    nan3[0, 0] = np.inf
    v = validate_frame_inputs(image, pc, b2, nan3)
    assert "bboxes_3d.finite" in _codes(v, "error")

    v = validate_frame_inputs(image, pc, np.zeros((0, 4)), b3)
    assert "bboxes_2d.empty" in _codes(v, "warning")


# ============================================================================
# enforce policies
# ============================================================================

def test_enforce_policies():
    image, pc, b2, b3 = _good_frame()
    bad_pc = pc.copy()
    bad_pc[0, 0] = np.nan
    errors = validate_frame_inputs(image, bad_pc, b2, b3)
    warnings = validate_frame_inputs(image[:240, :320], pc, b2, b3)

    with pytest.raises(ProtocolError):
        enforce(errors, mode="warn")            # hard violations always raise
    with pytest.raises(ProtocolError):
        enforce(errors, mode="strict")
    enforce(errors, mode="off")                  # off ignores everything

    enforce(warnings, mode="warn")               # soft violations only log
    with pytest.raises(ProtocolError):
        enforce(warnings, mode="strict")         # strict raises on warnings

    with pytest.raises(ValueError):
        enforce(warnings, mode="banana")


def test_protocol_error_message_is_actionable():
    image, pc, b2, b3 = _good_frame()
    bad_pc = pc.copy()
    bad_pc[0, 0] = np.nan
    with pytest.raises(ProtocolError) as exc_info:
        enforce(validate_frame_inputs(image, bad_pc, b2, b3), mode="warn")
    msg = str(exc_info.value)
    assert "point_cloud.finite" in msg
    assert "protocol.md" in msg


# ============================================================================
# CameraIntrinsics
# ============================================================================

def test_camera_intrinsics():
    K = CameraIntrinsics(fx=800.0, fy=820.0, cx=640.0, cy=360.0)
    mat = K.K
    assert mat.shape == (3, 3)
    assert mat[0, 0] == 800.0 and mat[1, 1] == 820.0
    assert mat[0, 2] == 640.0 and mat[1, 2] == 360.0
    assert np.allclose(K.dist_coeffs, 0.0)

    K2 = CameraIntrinsics.from_matrix(mat, distortion=[0.1, -0.05, 0.0, 0.0])
    assert K2.fx == 800.0 and K2.cy == 360.0
    assert K2.dist_coeffs.shape == (4,)

    with pytest.raises(ValueError):
        CameraIntrinsics.from_matrix(np.eye(4))


# ============================================================================
# matcher-level wiring
# ============================================================================

def test_matcher_validate_modes(vit_exp1_matcher):
    image, pc, b2, b3 = _good_frame()
    bad_pc = pc.copy()
    bad_pc[0, 0] = np.nan

    with pytest.raises(ProtocolError):
        vit_exp1_matcher.match(image, bad_pc, b2, b3)             # default warn

    # validate="off" lets garbage through to the (NaN-tolerant) pipeline
    result = vit_exp1_matcher.match(image, bad_pc, b2, b3, validate="off")
    assert result.similarity.shape[0] <= len(b2)

    with pytest.raises(ProtocolError):
        vit_exp1_matcher.match(
            image[:240, :320], pc, b2, b3, validate="strict"      # soft -> raise
        )


def test_pair_is_match_alias(vit_exp1_matcher):
    image, pc, b2, b3 = _good_frame()
    r1 = vit_exp1_matcher.match(image, pc, b2, b3)
    r2 = vit_exp1_matcher.pair(image, pc, b2, b3)
    assert r1.similarity.shape == r2.similarity.shape
    np.testing.assert_allclose(r1.similarity, r2.similarity, atol=1e-5)
