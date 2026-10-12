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

# PEP 562 lazy exports. The `adapter`/`memory`/`session` submodules pull torch, so
# eager-importing them here made `import xcalib.oneshot` — and, via
# `gui.scene`'s `from ..oneshot.calibration import CalibrationResult`, the whole
# `xcalib.continual` subpackage — load torch. Keeping this __init__ import-light lets
# a process that must stay torch-free import the continual data layer + headless
# scene IO. torch loads only when a one-shot adapt/session name is actually
# accessed. `calibration` (cv2/numpy, no torch) is imported directly by callers via
# `xcalib.oneshot.calibration`.
_LAZY = {
    "AdaptedModel": ".adapter", "EmbeddingAdapter": ".adapter",
    "weighted_infonce": ".adapter",
    "CalibrationResult": ".calibration", "CalibrationSession": ".calibration",
    "bbox2d_centers": ".calibration", "bbox3d_centers": ".calibration",
    "estimate_projection": ".calibration", "project_points": ".calibration",
    "refine_projection": ".calibration",
    "FeatureBank": ".memory",
    "PseudoLabels": ".pseudo_labels", "associate_projections": ".pseudo_labels",
    "pseudo_labels_for_frame": ".pseudo_labels",
    "AdaptReport": ".session", "ObserveReport": ".session",
    "OneShotSession": ".session", "SUPPORTED_MODELS": ".session",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'xcalib.oneshot' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


__all__ = [
    "AdaptedModel",
    "EmbeddingAdapter",
    "weighted_infonce",
    "CalibrationResult",
    "CalibrationSession",
    "bbox2d_centers",
    "bbox3d_centers",
    "estimate_projection",
    "refine_projection",
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
