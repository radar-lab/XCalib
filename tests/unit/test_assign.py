"""
Pair-selection tests: optimal one-to-one assignment (incl. the pure-NumPy LAP
fallback, cross-checked against brute force) and the geometric gate.

Run with:
    pixi run python -m pytest tests/unit/test_assign.py -q
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest

from xcalib import geometric_gate, one_to_one_assign
from xcalib.assign import _lap_min_pure
from xcalib.oneshot.calibration import CalibrationResult
from xcalib.protocol import CameraIntrinsics


# ------------------------------------------------------------- assignment --

def _brute_force_max(sim: np.ndarray) -> float:
    """Best achievable TOTAL similarity over all one-to-one assignments."""
    k, m = sim.shape
    n = min(k, m)
    best = -np.inf
    rows = range(k)
    for row_sel in itertools.permutations(rows, n) if k >= m else [tuple(rows)]:
        for col_sel in itertools.permutations(range(m), n):
            total = sum(sim[i, j] for i, j in zip(row_sel, col_sel))
            best = max(best, total)
    return float(best)


def test_one_to_one_resolves_greedy_cross_assignment():
    # greedy grabs (0,1)=1.0 first and strands row 1 at 0.2; the optimal
    # assignment pairs BOTH rows: (0,0)+(1,1) = 1.85 > 1.0 + 0.2.
    sim = np.array([[0.9, 1.0, 0.1],
                    [0.2, 0.95, 0.3]])
    pairs = one_to_one_assign(sim, threshold=0.5)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (1, 1)]
    # one-to-one: no row or column reused
    assert len({i for i, _, _ in pairs}) == len(pairs)
    assert len({j for _, j, _ in pairs}) == len(pairs)


def test_one_to_one_threshold_and_empty():
    assert one_to_one_assign(np.array([[0.4, 0.3], [0.2, 0.1]]), threshold=0.5) == []
    assert one_to_one_assign(np.zeros((0, 0))) == []
    assert one_to_one_assign(np.zeros((0, 3))) == []
    # single pair above threshold survives, sorted by score
    got = one_to_one_assign(np.array([[0.9, 0.1]]), threshold=0.5)
    assert got == [(0, 0, pytest.approx(0.9))]


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (4, 4), (5, 3), (3, 6)])
def test_one_to_one_matches_brute_force_total(shape):
    rng = np.random.default_rng(hash(shape) % 2**32)
    for _ in range(5):
        sim = rng.uniform(-1, 1, size=shape)
        pairs = one_to_one_assign(sim, threshold=-np.inf)
        total = sum(s for _, _, s in pairs)
        assert total == pytest.approx(_brute_force_max(sim), abs=1e-9)


@pytest.mark.parametrize("shape", [(2, 2), (3, 5), (5, 3), (6, 6)])
def test_pure_lap_fallback_matches_brute_force(shape):
    """The SciPy-free path must be EXACT, not approximate."""
    rng = np.random.default_rng(hash(("pure", shape)) % 2**32)
    for _ in range(10):
        sim = rng.uniform(-1, 1, size=shape)
        if shape[0] <= shape[1]:
            ans = _lap_min_pure(-sim)
            pairs = [(i, j) for i, j in enumerate(ans) if j >= 0]
        else:
            ans = _lap_min_pure(-sim.T)
            pairs = [(j, i) for i, j in enumerate(ans) if j >= 0]
        total = sum(sim[i, j] for i, j in pairs)
        assert total == pytest.approx(_brute_force_max(sim), abs=1e-9)
        assert len({i for i, _ in pairs}) == len(pairs)
        assert len({j for _, j in pairs}) == len(pairs)


def test_pure_lap_uses_infinite_entries_as_forbidden_pairs():
    # Forbidden pairs are -inf similarity. Exactly one full assignment avoids them.
    sim = np.array([[-np.inf, 0.9],
                    [0.8, -np.inf]])
    assert _lap_min_pure(-sim) == [1, 0]


def test_pure_lap_raises_instead_of_hanging_on_an_infeasible_matrix():
    # Column 0 is forbidden for both rows, so no assignment covers both rows.
    sim = np.array([[-np.inf, 0.9],
                    [-np.inf, 0.8]])
    with pytest.raises(ValueError, match="infeasible"):
        _lap_min_pure(-sim)
    with pytest.raises(ValueError, match="invalid numeric"):
        _lap_min_pure(-np.array([[np.nan, 0.5], [0.4, 0.3]]))


def test_one_to_one_assign_fails_the_same_way_with_and_without_scipy(monkeypatch):
    infeasible = np.array([[-np.inf, 0.9],
                           [-np.inf, 0.8]])
    with pytest.raises(ValueError):
        one_to_one_assign(infeasible)
    # A None entry makes `from scipy.optimize import ...` raise ImportError: the SciPy-free path.
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    with pytest.raises(ValueError, match="infeasible"):
        one_to_one_assign(infeasible)
    feasible = np.array([[0.9, 1.0, 0.1],
                         [0.2, 0.95, 0.3]])
    assert [(i, j) for i, j, _ in one_to_one_assign(feasible)] == [(1, 1), (0, 0)]


# --------------------------------------------------------- geometric gate --

def _cam():
    """K + lidar->camera pose (x fwd, y left, z up -> z fwd, x right, y down)."""
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    t = np.zeros(3)
    return intr, R, t


def _rect(u, v, w=80.0, h=60.0):
    return [u - w / 2, v - h / 2, u + w / 2, v + h / 2]


def test_geometric_gate_vetoes_wrong_side_and_behind():
    intr, R, t = _cam()
    rects = np.array([_rect(960, 540)])
    centers = np.array([
        [20.0, 0.0, 0.0],    # projects to the image center — the TRUE pair
        [20.0, 15.0, 0.0],   # far LEFT in the image — wrong bearing
        [-20.0, 0.0, 0.0],   # behind the camera
    ])
    mask, dist = geometric_gate(rects, centers, intrinsics=intr, rotation=R, translation=t)
    assert mask[0, 0] and dist[0, 0] < 5
    assert not mask[0, 1] and dist[0, 1] > 300
    assert not mask[0, 2] and not np.isfinite(dist[0, 2])

    # the gate decides the ASSIGNMENT even when appearance prefers the wrong box
    sim = np.array([[0.30, 0.99, 0.90]])
    gated = np.where(mask, sim, -np.inf)
    assert one_to_one_assign(gated, threshold=0.2) == [(0, 0, pytest.approx(0.30))]


def test_geometric_gate_diag_scaling_keeps_closeup_true_pair():
    # a close-up object reprojects with a large error; the gate must scale
    # with the box size instead of vetoing the true pair with a fixed radius
    intr, R, t = _cam()
    big = np.array([_rect(960, 540, w=900.0, h=700.0)])   # diag ≈ 1140 px
    centers = np.array([[20.0, -6.0, 0.0]])               # projects ~300 px right
    mask, dist = geometric_gate(big, centers, intrinsics=intr, rotation=R, translation=t,
                                row_fallback=False)
    assert 250 < dist[0, 0] < 400
    assert mask[0, 0]                                      # 3 × 1140 ≫ 300

    small = np.array([_rect(960, 540, w=40.0, h=30.0)])
    mask_s, _ = geometric_gate(small, centers, intrinsics=intr, rotation=R, translation=t,
                               row_fallback=False)
    assert not mask_s[0, 0]                                # small box, same 300 px: veto


def test_geometric_gate_row_fallback_on_garbage_calibration():
    intr, _, t = _cam()
    R_id = np.eye(3)  # identity "extrinsic" placeholder: projections are junk
    rects = np.array([_rect(960, 540)])
    centers = np.array([[20.0, 0.0, -5.0], [25.0, 5.0, -3.0]])  # both behind (z<0 depth)
    mask, _ = geometric_gate(rects, centers, intrinsics=intr, rotation=R_id, translation=t)
    assert mask[0].all()  # nothing passed -> appearance-only for that row
    mask_nf, _ = geometric_gate(rects, centers, intrinsics=intr, rotation=R_id, translation=t,
                                row_fallback=False)
    assert not mask_nf[0].any()


def test_geometric_gate_accepts_calibration_result():
    intr, R, t = _cam()
    calib = CalibrationResult(success=True, intrinsics=intr, rotation=R, translation=t,
                              projection=intr.K @ np.hstack([R, t.reshape(3, 1)]))
    rects = np.array([_rect(960, 540)])
    centers = np.array([[20.0, 0.0, 0.0]])
    mask, dist = geometric_gate(rects, centers, calib)
    assert mask[0, 0] and dist[0, 0] < 5

    failed = CalibrationResult(success=False)
    with pytest.raises(ValueError, match="SUCCESSFUL"):
        geometric_gate(rects, centers, failed)
    with pytest.raises(ValueError, match="calibration"):
        geometric_gate(rects, centers)  # neither form supplied


def test_geometric_gate_empty_inputs():
    intr, R, t = _cam()
    mask, dist = geometric_gate(np.zeros((0, 4)), np.zeros((0, 3)),
                                intrinsics=intr, rotation=R, translation=t)
    assert mask.shape == (0, 0) and dist.shape == (0, 0)
