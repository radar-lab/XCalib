"""
Input protocol — the data contract between the partner's perception stack
and this package.

The full written specification lives in ``docs/protocol.md``; this module
is its executable counterpart: `validate_frame_inputs` checks one frame's
arrays against the contract and returns structured violations, and
`Matcher.match(validate=...)` decides what to do with them.

Severity model
--------------
- ``error``   — the call would crash or silently produce garbage
                (wrong rank/dtype, NaN coordinates, ...). Raised as
                `ProtocolError` unless validation is "off".
- ``warning`` — quality concern (tiny bboxes, detection counts beyond the
                TensorRT profile, float image, ...). Logged once per code
                in "warn" mode, raised in "strict" mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from loguru import logger

PROTOCOL_VERSION = "1.0"

#: Detection-count caps aligned with the TensorRT dynamic-shape profiles
#: (`xcalib.engine.trt.plan_for_model`: maxShapes N=32 / M=32).
MAX_DETECTIONS_2D = 32
MAX_DETECTIONS_3D = 32

#: Soft quality floors (see docs/protocol.md §2).
MIN_IMAGE_HW = (480, 640)
MIN_BBOX_PX = 8.0

__all__ = [
    "PROTOCOL_VERSION",
    "MAX_DETECTIONS_2D",
    "MAX_DETECTIONS_3D",
    "CameraIntrinsics",
    "ProtocolViolation",
    "ProtocolError",
    "validate_frame_inputs",
    "enforce",
]


# ===========================================================================
# Camera intrinsics (required by calibrate(), optional everywhere else)
# ===========================================================================

@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics, pixels. Distortion follows OpenCV ordering."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Optional[np.ndarray] = None  # (k1, k2, p1, p2[, k3...]) or None

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        """Distortion coefficients as a float64 vector (zeros when None)."""
        if self.distortion is None:
            return np.zeros(5, dtype=np.float64)
        return np.asarray(self.distortion, dtype=np.float64).reshape(-1)

    @classmethod
    def from_matrix(
        cls, K: np.ndarray, distortion: Optional[np.ndarray] = None
    ) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        return cls(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            distortion=None if distortion is None else np.asarray(distortion, dtype=np.float64),
        )


# ===========================================================================
# Violations
# ===========================================================================

@dataclass(frozen=True)
class ProtocolViolation:
    """One contract violation found by `validate_frame_inputs`."""

    code: str        # stable machine-readable id, e.g. "image.dtype"
    severity: str    # "error" | "warning"
    message: str     # human-readable, actionable

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.code}: {self.message}"


class ProtocolError(ValueError):
    """Raised when frame inputs violate the protocol (see docs/protocol.md)."""

    def __init__(self, violations: Sequence[ProtocolViolation]):
        self.violations = list(violations)
        lines = [f"  - {v}" for v in self.violations]
        super().__init__(
            f"Frame inputs violate the xcalib input protocol "
            f"v{PROTOCOL_VERSION} ({len(self.violations)} issue(s)):\n"
            + "\n".join(lines)
            + "\nSee docs/protocol.md for the full contract."
        )


# ===========================================================================
# Per-array checks
# ===========================================================================

def _check_image(image: np.ndarray, out: List[ProtocolViolation]) -> None:
    if not isinstance(image, np.ndarray):
        out.append(ProtocolViolation(
            "image.type", "error",
            f"image must be a numpy array, got {type(image).__name__}",
        ))
        return
    if image.ndim != 3 or image.shape[2] != 3:
        out.append(ProtocolViolation(
            "image.shape", "error",
            f"image must be [H, W, 3] (RGB), got shape {image.shape}",
        ))
        return
    if image.shape[0] == 0 or image.shape[1] == 0:
        out.append(ProtocolViolation(
            "image.empty", "error", "image has zero height or width",
        ))
        return
    if image.dtype != np.uint8:
        if image.dtype in (np.float32, np.float64):
            out.append(ProtocolViolation(
                "image.dtype", "warning",
                f"image dtype is {image.dtype}; the protocol expects uint8 RGB "
                "in [0, 255]. Float images are assumed to be already scaled "
                "to [0, 1] and are passed through unchanged.",
            ))
        else:
            out.append(ProtocolViolation(
                "image.dtype", "error",
                f"image dtype {image.dtype} is not supported; use uint8 RGB",
            ))
            return
    h, w = image.shape[:2]
    if h < MIN_IMAGE_HW[0] or w < MIN_IMAGE_HW[1]:
        out.append(ProtocolViolation(
            "image.resolution", "warning",
            f"image is {w}x{h}; the matchers were trained on >= "
            f"{MIN_IMAGE_HW[1]}x{MIN_IMAGE_HW[0]} frames — quality may degrade",
        ))


def _check_point_cloud(pc: np.ndarray, out: List[ProtocolViolation]) -> None:
    if not isinstance(pc, np.ndarray):
        out.append(ProtocolViolation(
            "point_cloud.type", "error",
            f"point_cloud must be a numpy array, got {type(pc).__name__}",
        ))
        return
    if pc.ndim != 2 or pc.shape[1] < 3:
        out.append(ProtocolViolation(
            "point_cloud.shape", "error",
            f"point_cloud must be [P, >=3] (X, Y, Z[, extras]), got {pc.shape}",
        ))
        return
    if pc.shape[0] == 0:
        out.append(ProtocolViolation(
            "point_cloud.empty", "warning",
            "point_cloud has zero points — every 3D crop will be dropped",
        ))
        return
    xyz = pc[:, :3]
    if not np.issubdtype(xyz.dtype, np.floating):
        out.append(ProtocolViolation(
            "point_cloud.dtype", "warning",
            f"point_cloud dtype is {pc.dtype}; float32 metres expected "
            "(it will be cast)",
        ))
        xyz = xyz.astype(np.float64)
    if not np.isfinite(xyz).all():
        out.append(ProtocolViolation(
            "point_cloud.finite", "error",
            "point_cloud contains NaN/Inf XYZ values — filter them upstream",
        ))


def _check_bboxes_2d(
    b2: np.ndarray, image_hw: Optional[tuple], out: List[ProtocolViolation]
) -> None:
    b2 = np.asarray(b2)
    if b2.size == 0:
        out.append(ProtocolViolation(
            "bboxes_2d.empty", "warning",
            "no 2D detections — match() will return an empty similarity matrix",
        ))
        return
    if b2.ndim != 2 or b2.shape[1] < 4:
        out.append(ProtocolViolation(
            "bboxes_2d.shape", "error",
            f"bboxes_2d must be [K, 4] (x1, y1, x2, y2) pixels, got {b2.shape}",
        ))
        return
    if not np.isfinite(b2[:, :4].astype(np.float64)).all():
        out.append(ProtocolViolation(
            "bboxes_2d.finite", "error", "bboxes_2d contains NaN/Inf values",
        ))
        return
    x1, y1, x2, y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
    bad_order = (x2 <= x1) | (y2 <= y1)
    if bad_order.any():
        out.append(ProtocolViolation(
            "bboxes_2d.order", "warning",
            f"{int(bad_order.sum())} bbox(es) have x2<=x1 or y2<=y1 and will "
            "be dropped (convention is (x1, y1, x2, y2) with x1<x2, y1<y2)",
        ))
    small = ((x2 - x1) < MIN_BBOX_PX) | ((y2 - y1) < MIN_BBOX_PX)
    if (small & ~bad_order).any():
        out.append(ProtocolViolation(
            "bboxes_2d.small", "warning",
            f"{int((small & ~bad_order).sum())} bbox(es) are smaller than "
            f"{MIN_BBOX_PX:.0f}px on a side; crops will be heavily upsampled",
        ))
    if image_hw is not None:
        h, w = image_hw
        outside = (x2 <= 0) | (y2 <= 0) | (x1 >= w) | (y1 >= h)
        if outside.any():
            out.append(ProtocolViolation(
                "bboxes_2d.bounds", "warning",
                f"{int(outside.sum())} bbox(es) lie fully outside the "
                f"{w}x{h} image and will be dropped",
            ))
    if b2.shape[0] > MAX_DETECTIONS_2D:
        out.append(ProtocolViolation(
            "bboxes_2d.count", "warning",
            f"{b2.shape[0]} image detections exceed the TensorRT engine "
            f"profile cap of {MAX_DETECTIONS_2D}; PyTorch inference works "
            "but the shipped Thor engines will reject the batch",
        ))


def _check_bboxes_3d(b3: np.ndarray, out: List[ProtocolViolation]) -> None:
    b3 = np.asarray(b3)
    if b3.size == 0:
        out.append(ProtocolViolation(
            "bboxes_3d.empty", "warning",
            "no 3D detections — match() will return an empty similarity matrix",
        ))
        return
    if b3.ndim != 2 or b3.shape[1] != 6:
        out.append(ProtocolViolation(
            "bboxes_3d.shape", "error",
            f"bboxes_3d must be [M, 6] — either (xmin,ymin,zmin,xmax,ymax,zmax) "
            f"or (cx,cy,cz,dx,dy,dz) — got {b3.shape}",
        ))
        return
    if not np.isfinite(b3.astype(np.float64)).all():
        out.append(ProtocolViolation(
            "bboxes_3d.finite", "error", "bboxes_3d contains NaN/Inf values",
        ))
        return
    first, second = b3[:, :3], b3[:, 3:]
    as_extent = np.all(second >= first, axis=1)
    dims = np.where(as_extent[:, None], second - first, np.abs(second))
    degenerate = (dims <= 0).any(axis=1)
    if degenerate.any():
        out.append(ProtocolViolation(
            "bboxes_3d.degenerate", "warning",
            f"{int(degenerate.sum())} 3D bbox(es) have zero/negative extent "
            "on at least one axis; their point crops will likely be empty",
        ))
    if b3.shape[0] > MAX_DETECTIONS_3D:
        out.append(ProtocolViolation(
            "bboxes_3d.count", "warning",
            f"{b3.shape[0]} LiDAR detections exceed the TensorRT engine "
            f"profile cap of {MAX_DETECTIONS_3D}; PyTorch inference works "
            "but the shipped Thor engines will reject the batch",
        ))


# ===========================================================================
# Entry points
# ===========================================================================

def validate_frame_inputs(
    image: np.ndarray,
    point_cloud: np.ndarray,
    bboxes_2d: np.ndarray,
    bboxes_3d: np.ndarray,
) -> List[ProtocolViolation]:
    """Check one frame's raw inputs against the input protocol.

    Returns the (possibly empty) list of violations; never raises. Use
    `enforce` (or `Matcher.match(validate=...)`) to act on them.
    """
    out: List[ProtocolViolation] = []
    _check_image(image, out)
    image_hw = (
        (int(image.shape[0]), int(image.shape[1]))
        if isinstance(image, np.ndarray) and image.ndim == 3
        else None
    )
    _check_point_cloud(point_cloud, out)
    _check_bboxes_2d(bboxes_2d, image_hw, out)
    _check_bboxes_3d(bboxes_3d, out)
    return out


#: warning codes already logged once in "warn" mode (avoid per-frame spam).
_warned_codes: set = set()


def enforce(violations: Sequence[ProtocolViolation], mode: str = "warn") -> None:
    """Apply a validation policy to the violations of one frame.

    - "off"    : do nothing.
    - "warn"   : raise `ProtocolError` on errors; log warnings once per code.
    - "strict" : raise `ProtocolError` if there is any violation at all.
    """
    if mode == "off" or not violations:
        return
    if mode not in ("warn", "strict"):
        raise ValueError(f"validate must be 'strict', 'warn', or 'off', got {mode!r}")

    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]

    if mode == "strict" and violations:
        raise ProtocolError(list(violations))
    if errors:
        raise ProtocolError(errors)
    for v in warnings:
        if v.code not in _warned_codes:
            _warned_codes.add(v.code)
            logger.warning(f"input protocol: {v.code}: {v.message} "
                           "(logged once; pass validate='off' to silence)")
