"""Validation-aware physical camera-LiDAR calibration refinement.

This module is deliberately scene- and filesystem-agnostic.  Callers own the
fit/validation split and persistence; XCalib owns the fixed-intrinsic numeric
candidate and the fail-closed gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import cv2
import numpy as np

from .oneshot.calibration import CalibrationResult, project_points

__all__ = [
    "CalibrationGateConfig",
    "CalibrationMetrics",
    "PhysicalCalibrationCandidate",
    "build_physical_refinement_candidate",
    "evaluate_calibration",
]


@dataclass(frozen=True)
class CalibrationGateConfig:
    """Conservative defaults for a user-reviewed physical-pose candidate."""

    inlier_px: float = 25.0
    min_fit_points: int = 12
    min_validation_points: int = 6
    min_median_improvement_px: float = 0.5
    min_median_improvement_ratio: float = 0.02
    max_q90_regression_px: float = 2.0
    max_inlier_ratio_drop: float = 0.02
    max_depth_ratio_drop: float = 0.01
    min_positive_depth_ratio: float = 0.80
    max_rotation_delta_deg: float = 3.0
    max_translation_delta_m: float = 1.0
    primary_validation_set: str = "held_out"


@dataclass(frozen=True)
class CalibrationMetrics:
    pairs: int
    median_px: float
    q90_px: float
    rmse_inlier_px: float
    inliers: int
    inlier_ratio: float
    positive_depth: int
    positive_depth_ratio: float


@dataclass
class PhysicalCalibrationCandidate:
    """A proposed pose plus independent evidence; never persisted by XCalib."""

    previous: CalibrationResult
    candidate: CalibrationResult
    before: dict[str, CalibrationMetrics] = field(default_factory=dict)
    after: dict[str, CalibrationMetrics] = field(default_factory=dict)
    eligible: bool = False
    reasons: tuple[str, ...] = ()
    rotation_delta_deg: float = 0.0
    translation_delta_m: float = 0.0


def _points(points_3d, points_2d) -> tuple[np.ndarray, np.ndarray]:
    pts3 = np.ascontiguousarray(np.asarray(points_3d, dtype=np.float64).reshape(-1, 3))
    pts2 = np.ascontiguousarray(np.asarray(points_2d, dtype=np.float64).reshape(-1, 2))
    if len(pts3) != len(pts2):
        raise ValueError(
            f"points_3d ({len(pts3)}) and points_2d ({len(pts2)}) disagree"
        )
    if not np.isfinite(pts3).all() or not np.isfinite(pts2).all():
        raise ValueError("calibration points must be finite")
    return pts3, pts2


def _require_pose(calibration: CalibrationResult) -> None:
    if (
        not calibration.success
        or calibration.intrinsics is None
        or calibration.rotation is None
        or calibration.translation is None
    ):
        raise ValueError("a successful calibration with K, R, and t is required")


def evaluate_calibration(
    calibration: CalibrationResult,
    points_3d,
    points_2d,
    *,
    inlier_px: float = 25.0,
) -> CalibrationMetrics:
    """Evaluate one pose without hiding negative-depth or non-finite pairs."""

    _require_pose(calibration)
    pts3, pts2 = _points(points_3d, points_2d)
    n = len(pts3)
    if n == 0:
        return CalibrationMetrics(0, float("inf"), float("inf"), float("inf"), 0, 0.0, 0, 0.0)
    uv, depth = project_points(
        pts3, calibration.intrinsics, calibration.rotation, calibration.translation
    )
    raw = np.linalg.norm(uv - pts2, axis=1)
    positive = np.isfinite(depth) & (depth > 0)
    finite = positive & np.isfinite(raw)
    err = raw[finite]
    if len(err):
        median = float(np.median(err))
        q90 = float(np.quantile(err, 0.90))
    else:
        median = q90 = float("inf")
    inlier_mask = finite & (raw < float(inlier_px))
    inliers = int(inlier_mask.sum())
    rmse = (
        float(np.sqrt(np.mean(raw[inlier_mask] ** 2)))
        if inliers
        else float("inf")
    )
    return CalibrationMetrics(
        pairs=n,
        median_px=median,
        q90_px=q90,
        rmse_inlier_px=rmse,
        inliers=inliers,
        inlier_ratio=float(inliers / n),
        positive_depth=int(positive.sum()),
        positive_depth_ratio=float(positive.sum() / n),
    )


def _pose_delta(previous: CalibrationResult, candidate: CalibrationResult) -> tuple[float, float]:
    relative = candidate.rotation @ previous.rotation.T
    cos_angle = (np.trace(relative) - 1.0) / 2.0
    rotation_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    translation_m = float(np.linalg.norm(candidate.translation - previous.translation))
    return rotation_deg, translation_m


def _candidate_pose(
    previous: CalibrationResult,
    fit_3d: np.ndarray,
    fit_2d: np.ndarray,
    config: CalibrationGateConfig,
) -> tuple[CalibrationResult, int]:
    intr = previous.intrinsics
    uv, depth = project_points(fit_3d, intr, previous.rotation, previous.translation)
    err = np.linalg.norm(uv - fit_2d, axis=1)
    usable = np.isfinite(err) & (depth > 0) & (err < config.inlier_px * 3.0)
    usable_count = int(usable.sum())
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(previous.rotation))
    tvec = np.ascontiguousarray(previous.translation.reshape(3, 1), dtype=np.float64)
    if usable_count >= config.min_fit_points:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                np.ascontiguousarray(fit_3d[usable]),
                np.ascontiguousarray(fit_2d[usable]),
                intr.K,
                intr.dist_coeffs,
                rvec.copy(),
                tvec.copy(),
            )
        except cv2.error:
            pass
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    projection = intr.K @ np.hstack([rotation, translation.reshape(3, 1)])
    metrics = evaluate_calibration(
        CalibrationResult(
            success=True,
            intrinsics=intr,
            rotation=rotation,
            translation=translation,
            projection=projection,
        ),
        fit_3d,
        fit_2d,
        inlier_px=config.inlier_px,
    )
    result = CalibrationResult(
        success=True,
        intrinsics=intr,
        rotation=rotation,
        translation=translation,
        projection=projection,
        n_correspondences=len(fit_3d),
        n_inliers=metrics.inliers,
        reproj_error_px=metrics.rmse_inlier_px,
        buffer_reproj_px=metrics.median_px,
        accepted=False,
        message="physical refinement candidate; inspect independent validation gates",
    )
    return result, usable_count


def build_physical_refinement_candidate(
    previous: CalibrationResult,
    fit_points_3d,
    fit_points_2d,
    validation_sets: Mapping[str, tuple[object, object]],
    *,
    config: CalibrationGateConfig | None = None,
) -> PhysicalCalibrationCandidate:
    """Fit on one set and accept only from caller-supplied independent sets.

    ``validation_sets`` insertion order is preserved for presentation.  The set
    named by ``primary_validation_set`` must improve materially; every supplied
    set must also pass tail, inlier, and depth guardrails.
    """

    _require_pose(previous)
    cfg = config or CalibrationGateConfig()
    fit3, fit2 = _points(fit_points_3d, fit_points_2d)
    candidate, usable_fit = _candidate_pose(previous, fit3, fit2, cfg)
    before = {
        "fit": evaluate_calibration(previous, fit3, fit2, inlier_px=cfg.inlier_px)
    }
    after = {
        "fit": evaluate_calibration(candidate, fit3, fit2, inlier_px=cfg.inlier_px)
    }
    reasons: list[str] = []
    if len(fit3) < cfg.min_fit_points:
        reasons.append(f"fit has {len(fit3)} pair(s); need {cfg.min_fit_points}")
    if usable_fit < cfg.min_fit_points:
        reasons.append(
            f"baseline explains only {usable_fit} fit pair(s); need {cfg.min_fit_points} for a nudge"
        )

    for name, pair in validation_sets.items():
        val3, val2 = _points(pair[0], pair[1])
        before[name] = evaluate_calibration(previous, val3, val2, inlier_px=cfg.inlier_px)
        after[name] = evaluate_calibration(candidate, val3, val2, inlier_px=cfg.inlier_px)
        old, new = before[name], after[name]
        if old.pairs < cfg.min_validation_points:
            reasons.append(
                f"{name} has {old.pairs} pair(s); need {cfg.min_validation_points}"
            )
            continue
        if not np.isfinite(new.median_px) or not np.isfinite(new.q90_px):
            reasons.append(f"{name} candidate projection is non-finite")
            continue
        if name == cfg.primary_validation_set:
            required = max(
                cfg.min_median_improvement_px,
                old.median_px * cfg.min_median_improvement_ratio,
            )
            improvement = old.median_px - new.median_px
            if not np.isfinite(improvement) or improvement < required:
                reasons.append(
                    f"{name} median improves {improvement:.3g}px; need {required:.3g}px"
                )
        if new.q90_px > old.q90_px + cfg.max_q90_regression_px:
            reasons.append(
                f"{name} q90 regresses {new.q90_px - old.q90_px:.3g}px"
            )
        if new.inlier_ratio + cfg.max_inlier_ratio_drop < old.inlier_ratio:
            reasons.append(
                f"{name} inlier ratio drops {old.inlier_ratio - new.inlier_ratio:.3f}"
            )
        if new.positive_depth_ratio < cfg.min_positive_depth_ratio:
            reasons.append(
                f"{name} positive-depth ratio {new.positive_depth_ratio:.3f} is below "
                f"{cfg.min_positive_depth_ratio:.3f}"
            )
        if new.positive_depth_ratio + cfg.max_depth_ratio_drop < old.positive_depth_ratio:
            reasons.append(
                f"{name} positive-depth ratio drops "
                f"{old.positive_depth_ratio - new.positive_depth_ratio:.3f}"
            )

    if cfg.primary_validation_set not in validation_sets:
        reasons.append(f"missing primary validation set {cfg.primary_validation_set!r}")
    rotation_delta, translation_delta = _pose_delta(previous, candidate)
    if rotation_delta > cfg.max_rotation_delta_deg:
        reasons.append(
            f"rotation delta {rotation_delta:.3f}deg exceeds {cfg.max_rotation_delta_deg:.3f}deg"
        )
    if translation_delta > cfg.max_translation_delta_m:
        reasons.append(
            f"translation delta {translation_delta:.3f}m exceeds "
            f"{cfg.max_translation_delta_m:.3f}m"
        )

    candidate.accepted = not reasons
    candidate.message = (
        "eligible physical refinement candidate"
        if not reasons
        else "; ".join(reasons)
    )
    return PhysicalCalibrationCandidate(
        previous=previous,
        candidate=candidate,
        before=before,
        after=after,
        eligible=not reasons,
        reasons=tuple(reasons),
        rotation_delta_deg=rotation_delta,
        translation_delta_m=translation_delta,
    )
