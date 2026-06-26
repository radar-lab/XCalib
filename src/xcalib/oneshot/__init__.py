"""
One-shot / continual learning, supervised by the camera-LiDAR projection
matrix instead of human labels.

The loop (see `session.OneShotSession`):

1. `observe()`   — match a frame, harvest confident pairs.
2. `calibrate()` — PnP/RANSAC over buffered pairs -> P = K [R|t].
3. once calibrated, every new frame self-labels: 3D boxes project through
   P, associate to 2D boxes, and the geometric agreement becomes a
   confidence ("reward") that gates admission into the `FeatureBank`.
4. `adapt()`     — confidence-weighted InfoNCE updates of small residual
   adapters (backbone frozen), with bank replay against forgetting.

Entry point: ``matcher.oneshot(intrinsics=CameraIntrinsics(...))``.
"""

from .adapter import AdaptedModel, EmbeddingAdapter, weighted_infonce
from .calibration import (
    CalibrationResult,
    CalibrationSession,
    bbox2d_centers,
    bbox3d_centers,
    estimate_projection,
    project_points,
)
from .memory import FeatureBank
from .pseudo_labels import PseudoLabels, associate_projections, pseudo_labels_for_frame
from .session import AdaptReport, ObserveReport, OneShotSession, SUPPORTED_MODELS

__all__ = [
    "AdaptedModel",
    "EmbeddingAdapter",
    "weighted_infonce",
    "CalibrationResult",
    "CalibrationSession",
    "bbox2d_centers",
    "bbox3d_centers",
    "estimate_projection",
    "project_points",
    "FeatureBank",
    "PseudoLabels",
    "associate_projections",
    "pseudo_labels_for_frame",
    "AdaptReport",
    "ObserveReport",
    "OneShotSession",
    "SUPPORTED_MODELS",
]
