"""
Camera-LiDAR extrinsic calibration from cross-modal matches (PnP/RANSAC).

The matcher gives us (2D bbox, 3D bbox) correspondences; their centers are
2D-3D point pairs. With known camera intrinsics K, `cv2.solvePnPRansac`
recovers the extrinsics [R|t] and therefore the projection matrix

    P = K @ [R | t]    (maps bboxes_3d-frame points to pixels)

`CalibrationSession` accumulates confident correspondences across frames —
single frames rarely have enough spread for a well-conditioned solve — and
`solve()` runs RANSAC + LM refinement over the whole buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from ..protocol import CameraIntrinsics

__all__ = [
    "CalibrationResult",
    "CalibrationSession",
    "estimate_projection",
    "project_points",
    "bbox2d_centers",
    "bbox3d_centers",
]


# ---------------------------------------------------------------------------
# bbox center helpers (shared conventions with data/crops.py)
# ---------------------------------------------------------------------------

def bbox2d_centers(bboxes_2d: np.ndarray) -> np.ndarray:
    """[K,4] (x1,y1,x2,y2) -> [K,2] pixel centers."""
    b = np.asarray(bboxes_2d, dtype=np.float64)
    if b.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return (b[:, :2] + b[:, 2:4]) / 2.0


def bbox3d_centers(bboxes_3d: np.ndarray) -> np.ndarray:
    """[M,6] -> [M,3] centers, using the same extent-vs-center heuristic as
    `xcalib.data.crops.crop_point_cloud_axis_aligned`."""
    b = np.asarray(bboxes_3d, dtype=np.float64)
    if b.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    first, second = b[:, :3], b[:, 3:6]
    treat_as_extent = np.all(second >= first, axis=1)
    return np.where(treat_as_extent[:, None], (first + second) / 2.0, first)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Output of a PnP/RANSAC solve."""

    success: bool
    intrinsics: Optional[CameraIntrinsics] = None
    rotation: Optional[np.ndarray] = None       # [3,3]
    translation: Optional[np.ndarray] = None    # [3]
    projection: Optional[np.ndarray] = None     # [3,4] = K [R|t]
    n_correspondences: int = 0
    n_inliers: int = 0
    reproj_error_px: float = float("inf")       # mean over inliers
    message: str = ""
    #: Median reprojection error over ALL buffered pairs (not just inliers),
    #: stamped by ``OneShotSession.calibrate``. ``nan`` when not scored. Unlike
    #: ``reproj_error_px``, this exposes a degenerate planar pose (see
    #: ``CalibrationSession.reprojection_error``).
    buffer_reproj_px: float = float("nan")
    #: Whether ``OneShotSession.calibrate`` adopted this solve as the session
    #: calibration. ``False`` means a successful PnP solve was rejected by the
    #: degenerate-pose gate. Always ``True`` for ungated / single-frame solves.
    accepted: bool = True

    @property
    def extrinsics(self) -> Optional[np.ndarray]:
        """[3,4] = [R|t] (without intrinsics)."""
        if self.rotation is None or self.translation is None:
            return None
        return np.hstack([self.rotation, self.translation.reshape(3, 1)])

    def pose_error(self, gt_extrinsics: np.ndarray) -> Tuple[float, float]:
        """Accuracy of this solve against a ground-truth extrinsic.

        ``gt_extrinsics`` is a 4x4 (or 3x4) lidar->camera pose — e.g.
        ``UTCFrame.extrinsics[camera]`` from an A9 cache. Returns
        ``(rotation_deg, translation_m)``: the geodesic rotation angle and the
        Euclidean translation distance between the estimated ``[R|t]`` and the
        ground truth, the camera-pose errors a paper-style table reports.
        """
        if not self.success or self.rotation is None or self.translation is None:
            raise RuntimeError(f"Calibration unavailable: {self.message or 'solve failed'}")
        gt = np.asarray(gt_extrinsics, dtype=np.float64)
        if gt.shape == (4, 4):
            gt = gt[:3]
        if gt.shape != (3, 4):
            raise ValueError(f"gt_extrinsics must be 3x4 or 4x4, got {gt.shape}")
        R_gt, t_gt = gt[:, :3], gt[:, 3]
        cos = (np.trace(self.rotation @ R_gt.T) - 1.0) / 2.0
        rotation_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        translation_m = float(np.linalg.norm(self.translation - t_gt))
        return rotation_deg, translation_m

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project LiDAR-frame points to pixels.

        Returns (uv [N,2], depth [N]); depth is the camera-frame Z, points
        behind the camera have depth <= 0.
        """
        if not self.success:
            raise RuntimeError(f"Calibration unavailable: {self.message or 'solve failed'}")
        return project_points(
            points_3d, self.intrinsics, self.rotation, self.translation
        )

    def save(self, path) -> None:
        np.savez(
            path,
            rotation=self.rotation,
            translation=self.translation,
            projection=self.projection,
            K=self.intrinsics.K if self.intrinsics else np.eye(3),
            reproj_error_px=self.reproj_error_px,
            n_inliers=self.n_inliers,
        )


def project_points(
    points_3d: np.ndarray,
    intrinsics: CameraIntrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.projectPoints wrapper returning (uv [N,2], camera-frame depth [N])."""
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    uv, _ = cv2.projectPoints(
        pts, rvec, np.asarray(translation, dtype=np.float64).reshape(3, 1),
        intrinsics.K, intrinsics.dist_coeffs,
    )
    cam = pts @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)
    return uv.reshape(-1, 2), cam[:, 2]


# ---------------------------------------------------------------------------
# Single solve
# ---------------------------------------------------------------------------

def estimate_projection(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    ransac_reproj_px: float = 8.0,
    ransac_iters: int = 500,
    min_points: int = 6,
) -> CalibrationResult:
    """Solve camera-LiDAR extrinsics from N (3D point, pixel) pairs.

    Pipeline: EPnP-seeded RANSAC -> Levenberg-Marquardt refinement on the
    inlier set. Returns a failed CalibrationResult (success=False) instead
    of raising when the geometry is insufficient.

    A single planar frame is bistable (the clump pose); accumulating spread
    across frames via :class:`CalibrationSession` / :meth:`OneShotSession.calibrate`
    (which scores solves over the whole buffer and accept-latest) is the path
    that disambiguates it — a single solve cannot.
    """
    pts3 = np.ascontiguousarray(np.asarray(points_3d, dtype=np.float64).reshape(-1, 3))
    pts2 = np.ascontiguousarray(np.asarray(points_2d, dtype=np.float64).reshape(-1, 2))
    n = pts3.shape[0]
    if pts2.shape[0] != n:
        raise ValueError(f"points_3d ({n}) and points_2d ({pts2.shape[0]}) disagree")
    if n < min_points:
        return CalibrationResult(
            success=False, n_correspondences=n,
            message=f"need >= {min_points} correspondences, have {n} — keep observing",
        )

    K = intrinsics.K
    dist = intrinsics.dist_coeffs

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3, pts2, K, dist,
        iterationsCount=ransac_iters,
        reprojectionError=float(ransac_reproj_px),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 4:
        return CalibrationResult(
            success=False, n_correspondences=n,
            message="PnP/RANSAC failed — matches are likely degenerate "
                    "(coplanar / clustered); observe more varied frames",
        )

    idx = inliers.reshape(-1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(pts3[idx], pts2[idx], K, dist, rvec, tvec)
    except cv2.error:  # refinement is best-effort
        pass

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    uv, _depth = project_points(pts3[idx], intrinsics, R, t)
    err = float(np.linalg.norm(uv - pts2[idx], axis=1).mean())

    P = K @ np.hstack([R, t.reshape(3, 1)])
    return CalibrationResult(
        success=True,
        intrinsics=intrinsics,
        rotation=R,
        translation=t,
        projection=P,
        n_correspondences=n,
        n_inliers=int(len(idx)),
        reproj_error_px=err,
        message="ok",
    )


# ---------------------------------------------------------------------------
# Multi-frame accumulation
# ---------------------------------------------------------------------------

class CalibrationSession:
    """Accumulates confident (2D center, 3D center) pairs across frames.

    A roadside camera-LiDAR pair sees objects sweep through the scene; a
    few seconds of confident matches gives PnP a well-spread point set.
    The buffer is FIFO-capped so stale geometry eventually ages out (the
    sensors may be re-aimed).
    """

    def __init__(
        self,
        *,
        min_score: float = 0.6,
        max_pairs: int = 500,
        ransac_reproj_px: float = 8.0,
    ):
        self.min_score = float(min_score)
        self.max_pairs = int(max_pairs)
        self.ransac_reproj_px = float(ransac_reproj_px)
        self._pts2d: List[np.ndarray] = []
        self._pts3d: List[np.ndarray] = []
        self._scores: List[float] = []

    def __len__(self) -> int:
        return len(self._scores)

    def clear(self) -> None:
        self._pts2d.clear()
        self._pts3d.clear()
        self._scores.clear()

    def add_correspondences(
        self,
        points_2d: np.ndarray,
        points_3d: np.ndarray,
        scores: Optional[Sequence[float]] = None,
    ) -> int:
        """Append raw 2D-3D point pairs (already one-to-one)."""
        points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
        points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
        if len(points_2d) != len(points_3d):
            raise ValueError("points_2d and points_3d must pair up 1:1")
        scores = list(scores) if scores is not None else [1.0] * len(points_2d)
        for p2, p3, s in zip(points_2d, points_3d, scores):
            self._pts2d.append(p2)
            self._pts3d.append(p3)
            self._scores.append(float(s))
        overflow = len(self._scores) - self.max_pairs
        if overflow > 0:
            del self._pts2d[:overflow]
            del self._pts3d[:overflow]
            del self._scores[:overflow]
        return len(points_2d)

    def add_matches(
        self,
        bboxes_2d: np.ndarray,
        centers_3d: np.ndarray,
        matches: Sequence[Tuple[int, int, float]],
        min_score: Optional[float] = None,
    ) -> int:
        """Harvest one frame's matcher output.

        `matches` are (img_idx, lid_idx, score) triples in the *caller's*
        bbox numbering (as produced by `Matcher.match`). Pairs below
        `min_score` are ignored; the rest are made one-to-one greedily by
        score before being buffered.
        """
        thr = self.min_score if min_score is None else float(min_score)
        b2c = bbox2d_centers(bboxes_2d)
        c3 = np.asarray(centers_3d, dtype=np.float64).reshape(-1, 3)

        used_i: set = set()
        used_j: set = set()
        pts2, pts3, scores = [], [], []
        for i, j, s in sorted(matches, key=lambda m: -m[2]):
            if s < thr or i in used_i or j in used_j:
                continue
            if i >= len(b2c) or j >= len(c3):
                continue
            used_i.add(i)
            used_j.add(j)
            pts2.append(b2c[i])
            pts3.append(c3[j])
            scores.append(float(s))
        if not pts2:
            return 0
        return self.add_correspondences(np.array(pts2), np.array(pts3), scores)

    def solve(
        self,
        intrinsics: CameraIntrinsics,
        *,
        min_pairs: int = 6,
    ) -> CalibrationResult:
        """PnP/RANSAC over everything buffered so far."""
        if len(self) < min_pairs:
            return CalibrationResult(
                success=False, n_correspondences=len(self),
                message=f"only {len(self)} buffered pair(s); need >= {min_pairs}",
            )
        result = estimate_projection(
            np.array(self._pts3d),
            np.array(self._pts2d),
            intrinsics,
            ransac_reproj_px=self.ransac_reproj_px,
            min_points=min_pairs,
        )
        if result.success:
            logger.info(
                f"Calibration: {result.n_inliers}/{result.n_correspondences} inliers, "
                f"mean reprojection error {result.reproj_error_px:.2f}px"
            )
        else:
            logger.warning(f"Calibration failed: {result.message}")
        return result

    def reprojection_error(
        self,
        result: CalibrationResult,
        *,
        reduce: str = "median",
    ) -> float:
        """Reprojection error of ``result`` measured over *all* buffered pairs.

        ``CalibrationResult.reproj_error_px`` is the mean over the RANSAC
        *inlier* subset only. Roadside / infrastructure scenes are nearly
        planar — every box centre sits on the road — which admits the classic
        two-fold planar-PnP pose ambiguity. The spurious pose folds the whole
        cloud into a clump yet still nails its own inliers, so it can report a
        *lower* inlier error than the correct pose. Scoring the pose against the
        entire buffer exposes it: the bad pose's error stays high.

        Use this (``reduce="median"`` for robustness) to disambiguate or gate a
        solve rather than trusting ``reproj_error_px`` alone — this is exactly
        what ``OneShotSession.calibrate`` does by default. Returns ``inf`` for a
        failed result or an empty buffer.
        """
        if not result.success or not self._scores:
            return float("inf")
        pts3 = np.asarray(self._pts3d, dtype=np.float64)
        pts2 = np.asarray(self._pts2d, dtype=np.float64)
        uv, _ = project_points(pts3, result.intrinsics, result.rotation, result.translation)
        err = np.linalg.norm(uv - pts2, axis=1)
        reducers = {"median": np.median, "mean": np.mean, "max": np.max}
        if reduce not in reducers:
            raise ValueError(f"reduce must be one of {sorted(reducers)}, got {reduce!r}")
        return float(reducers[reduce](err))
