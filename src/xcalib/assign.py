"""
Frame-level pair selection: optimal one-to-one assignment + geometric gating.

The raw similarity matrix answers "how alike do these crops look?" — turning it
into *pairings* needs two more ingredients:

- :func:`one_to_one_assign` — the Hungarian / LAP assignment that maximizes the
  TOTAL similarity subject to one-to-one constraints. Greedy top-1 loses a
  high-confidence pair whenever a shared column is claimed first (two
  detections whose true matches each score ~1.0 can end up cross-assigned);
  the optimal assignment cannot.

- :func:`geometric_gate` — with a (roughly accurate) camera-LiDAR calibration,
  every 3D box projects to a known image position, so a candidate pair whose
  projection lands far from the 2D box is a *different object* no matter how
  similar the appearance embedding thinks they look. Appearance features get
  fooled by same-model vehicles; projection geometry does not. Gating the
  similarity BEFORE the assignment vetoes those wrong-side pairs outright.

Both are wired into :meth:`xcalib.Matcher.match` via ``one_to_one=True`` and
``calibration=...`` and are exported for direct use.

SciPy is used for the LAP when available (it is in most lab environments); a
pure-NumPy shortest-augmenting-path implementation covers edge installs where
it is not — xcalib's runtime dependency set stays unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .oneshot.calibration import CalibrationResult
    from .protocol import CameraIntrinsics

__all__ = ["one_to_one_assign", "geometric_gate"]

#: Geometric gate radius = max(GEO_GATE_MIN_PX, GEO_GATE_DIAG × 2D-box diagonal).
#: Generous by design: calibration drift / rolling-shutter jitter must never veto
#: a TRUE pair, while wrong-side pairs (hundreds of px off) always fail. The
#: diagonal term matters for close-up objects, whose reprojection error grows
#: with apparent size — a fixed radius would veto their true match.
GEO_GATE_MIN_PX = 150.0
GEO_GATE_DIAG = 3.0


# --------------------------------------------------------------- assignment --

def _lap_min_pure(cost: np.ndarray) -> List[int]:
    """Exact minimum-cost assignment (shortest augmenting path / Hungarian with
    potentials) for an ``n x m`` cost matrix with ``n <= m``.

    Returns ``ans`` with ``ans[i] = column assigned to row i``. O(n²·m) — the
    matrices here are per-frame object counts (tens), so plain Python loops are
    fine. Kept dependency-free for edge installs without SciPy.

    Raises ``ValueError`` like SciPy's ``linear_sum_assignment`` when the matrix
    holds NaN, or when infinite entries leave no assignment of every row."""
    n, m = cost.shape
    if np.isnan(cost).any():
        raise ValueError("matrix contains invalid numeric entries")
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = 1-based row matched to column j (0 = free)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = float(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:  # every free column is infinite for the rows reached so far
                raise ValueError("cost matrix is infeasible")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    ans = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            ans[p[j] - 1] = j - 1
    return ans


def one_to_one_assign(
    similarity: np.ndarray,
    threshold: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """Optimal one-to-one ``(row, col, score)`` pairs from a similarity matrix.

    Maximizes the TOTAL similarity over all one-to-one assignments (LAP), then
    drops pairs scoring below ``threshold``. Unlike a greedy pass, a
    high-confidence pair is never lost because its column was claimed first.

    Args:
        similarity: ``[K, M]`` float matrix (any real values).
        threshold:  pairs with ``similarity < threshold`` are discarded
                    *after* the assignment.

    Returns:
        List of ``(row, col, score)`` sorted by score, highest first.
    """
    m = np.asarray(similarity, dtype=float)
    if m.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        rows, cols = linear_sum_assignment(-m)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except ImportError:  # edge install without SciPy — exact pure-NumPy path
        if m.shape[0] <= m.shape[1]:
            ans = _lap_min_pure(-m)
            pairs = [(i, j) for i, j in enumerate(ans) if j >= 0]
        else:
            ans = _lap_min_pure(-m.T)
            pairs = [(j, i) for i, j in enumerate(ans) if j >= 0]
    out = [(int(i), int(j), float(m[i, j])) for i, j in pairs if float(m[i, j]) >= threshold]
    out.sort(key=lambda p: -p[2])
    return out


# ----------------------------------------------------------- geometric gate --

def geometric_gate(
    bboxes_2d: np.ndarray,
    centers_3d: np.ndarray,
    calibration: Optional["CalibrationResult"] = None,
    *,
    intrinsics: Optional["CameraIntrinsics"] = None,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    min_px: float = GEO_GATE_MIN_PX,
    diag_factor: float = GEO_GATE_DIAG,
    row_fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Geometric compatibility of every (2D box, 3D center) pair.

    Projects each 3D center into the image with the given calibration and
    measures its pixel distance to each 2D box center. A pair is compatible
    when the projection lands IN FRONT of the camera and within
    ``max(min_px, diag_factor × box diagonal)`` of the box center.

    With ``row_fallback=True`` (default) a row whose gate blocks *every*
    column reverts to all-compatible: a garbage calibration (e.g. an identity
    extrinsic placeholder) must degrade to appearance-only matching rather
    than blank the row.

    Args:
        bboxes_2d:   ``[K, 4]`` (x1, y1, x2, y2).
        centers_3d:  ``[M, 3]`` LiDAR-frame box centers
                     (see :func:`xcalib.oneshot.calibration.bbox3d_centers`).
        calibration: a successful :class:`CalibrationResult`; or pass
                     ``intrinsics`` + ``rotation`` + ``translation`` directly.
        intrinsics:  camera intrinsics, used with ``rotation`` and ``translation``
                     when ``calibration`` is not given.
        rotation:    ``[3, 3]`` LiDAR-to-camera rotation.
        translation: ``[3]`` LiDAR-to-camera translation.
        min_px:      gate radius floor in pixels (default ``GEO_GATE_MIN_PX``).
        diag_factor: gate radius as a multiple of the 2D box diagonal
                     (default ``GEO_GATE_DIAG``).
        row_fallback: revert a row that blocks every column to all-compatible.

    Returns:
        ``(mask [K, M] bool, dist_px [K, M] float)`` — ``dist_px`` is ``inf``
        for centers behind the camera / invalid projections.
    """
    from .oneshot.calibration import project_points  # noqa: PLC0415

    if calibration is not None:
        if not calibration.success:
            raise ValueError("geometric_gate needs a SUCCESSFUL calibration")
        intrinsics = calibration.intrinsics
        rotation = calibration.rotation
        translation = calibration.translation
    if intrinsics is None or rotation is None or translation is None:
        raise ValueError("pass `calibration` or all of intrinsics/rotation/translation")

    b = np.asarray(bboxes_2d, dtype=np.float64).reshape(-1, 4)
    c3 = np.asarray(centers_3d, dtype=np.float64).reshape(-1, 3)
    K, M = b.shape[0], c3.shape[0]
    if K == 0 or M == 0:
        return np.zeros((K, M), dtype=bool), np.full((K, M), np.inf)

    uv, depth = project_points(c3, intrinsics, rotation, translation)
    bcx = (b[:, 0] + b[:, 2]) / 2.0
    bcy = (b[:, 1] + b[:, 3]) / 2.0
    gate = np.maximum(float(min_px),
                      float(diag_factor) * np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))

    dist = np.hypot(uv[None, :, 0] - bcx[:, None], uv[None, :, 1] - bcy[:, None])
    ok = (depth > 0)[None, :] & np.isfinite(dist)
    dist = np.where(ok, dist, np.inf)
    mask = dist <= gate[:, None]

    if row_fallback:
        empty = ~mask.any(axis=1)
        mask[empty, :] = True
    return mask, dist
