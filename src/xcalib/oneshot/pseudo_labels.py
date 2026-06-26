"""
Geometry-supervised pseudo-labels.

Once the projection matrix is known, every new frame labels itself: 3D box
centers are projected into the image and associated with 2D detections.
Agreement between projection and detection is expressed as a geometric
confidence in [0, 1], which downstream acts as the (reward-like) gate for
feature-bank admission and adapter updates — *no human labels involved*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .calibration import CalibrationResult, bbox2d_centers

__all__ = ["PseudoLabels", "associate_projections", "pseudo_labels_for_frame"]


@dataclass
class PseudoLabels:
    """Self-supervision for one frame.

    pairs hold (img_idx, lid_idx, confidence) in the caller's numbering;
    `match_matrix[i, j]` mirrors them as a dense boolean matrix.
    """

    match_matrix: np.ndarray                     # [K, M] bool
    confidence: np.ndarray                       # [K, M] float32 in [0, 1]
    pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    projected_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    depths: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([p[2] for p in self.pairs])) if self.pairs else 0.0


def _greedy_assign(cost: np.ndarray, valid: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy minimum-cost one-to-one assignment over valid entries."""
    pairs: List[Tuple[int, int]] = []
    if cost.size == 0:
        return pairs
    flat = [
        (cost[i, j], i, j)
        for i in range(cost.shape[0])
        for j in range(cost.shape[1])
        if valid[i, j]
    ]
    used_i: set = set()
    used_j: set = set()
    for c, i, j in sorted(flat):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


def _assign(cost: np.ndarray, valid: np.ndarray) -> List[Tuple[int, int]]:
    """Hungarian assignment when scipy is around, greedy otherwise."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_assign(cost, valid)

    if cost.size == 0:
        return []
    BIG = 1e9
    padded = np.where(valid, cost, BIG)
    rows, cols = linear_sum_assignment(padded)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]


def associate_projections(
    bboxes_2d: np.ndarray,
    projected_uv: np.ndarray,
    depths: np.ndarray,
    *,
    image_size: Optional[Tuple[int, int]] = None,   # (W, H)
    max_center_dist_px: float = 64.0,
    min_depth_m: float = 0.5,
    sigma_floor_px: float = 8.0,
) -> PseudoLabels:
    """Match projected 3D centers to 2D detections by center distance.

    Confidence per pair: ``exp(-0.5 * (d / sigma_i)^2)`` where ``sigma_i``
    adapts to the 2D box size (half of sqrt(w*h), floored at
    `sigma_floor_px`) — a 10px error on a bus is great, on a pedestrian
    it is not. Projections behind the camera, outside the image, or
    farther than `max_center_dist_px` from every detection never pair.
    One-to-one assignment (Hungarian when scipy is installed).
    """
    b2 = np.asarray(bboxes_2d, dtype=np.float64).reshape(-1, 4)
    uv = np.asarray(projected_uv, dtype=np.float64).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    K, M = b2.shape[0], uv.shape[0]

    match = np.zeros((K, M), dtype=bool)
    conf = np.zeros((K, M), dtype=np.float32)
    out = PseudoLabels(match, conf, [], uv, depths)
    if K == 0 or M == 0:
        return out

    centers = bbox2d_centers(b2)                                      # [K,2]
    wh = np.maximum(b2[:, 2:4] - b2[:, :2], 1.0)                      # [K,2]
    sigma = np.maximum(0.5 * np.sqrt(wh[:, 0] * wh[:, 1]), sigma_floor_px)  # [K]

    dist = np.linalg.norm(centers[:, None, :] - uv[None, :, :], axis=2)  # [K,M]

    lid_ok = depths > min_depth_m                                     # [M]
    if image_size is not None:
        W, H = image_size
        margin = max_center_dist_px
        inside = (
            (uv[:, 0] > -margin) & (uv[:, 0] < W + margin)
            & (uv[:, 1] > -margin) & (uv[:, 1] < H + margin)
        )
        lid_ok = lid_ok & inside

    valid = lid_ok[None, :] & (dist <= max_center_dist_px)

    conf_all = np.exp(-0.5 * (dist / sigma[:, None]) ** 2).astype(np.float32)

    for i, j in _assign(dist, valid):
        c = float(conf_all[i, j])
        match[i, j] = True
        conf[i, j] = c
        out.pairs.append((i, j, c))
    return out


def pseudo_labels_for_frame(
    calibration: CalibrationResult,
    bboxes_2d: np.ndarray,
    centers_3d: np.ndarray,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    max_center_dist_px: float = 64.0,
) -> PseudoLabels:
    """Project 3D centers through the calibration and associate to 2D boxes."""
    uv, depth = calibration.project(np.asarray(centers_3d, dtype=np.float64))
    return associate_projections(
        bboxes_2d, uv, depth,
        image_size=image_size,
        max_center_dist_px=max_center_dist_px,
    )
