"""Offline tests for xcalib.data.crops.oriented_extents."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xcalib.data.crops import crop_point_cloud_axis_aligned, oriented_extents


def _yaw_quaternion(yaw: float) -> list[float]:
    """Quaternion (x, y, z, w) of a turn by ``yaw`` radians about z."""
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _turn(points: np.ndarray, yaw: float) -> np.ndarray:
    """Turn points by ``yaw`` about z (plain arithmetic: no BLAS in a unit test)."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(points)
    out[:, 0] = c * points[:, 0] - s * points[:, 1]
    out[:, 1] = s * points[:, 0] + c * points[:, 1]
    out[:, 2] = points[:, 2]
    return out


def _extents(center, dims, quaternion):
    return oriented_extents(
        np.array([center], dtype=np.float64),
        np.array([dims], dtype=np.float64),
        np.array([quaternion], dtype=np.float64),
    )[0]


def test_no_rotation_is_centre_plus_minus_half_the_dimensions():
    got = _extents([10.0, -4.0, 1.0], [4.0, 1.8, 1.5], _yaw_quaternion(0.0))
    np.testing.assert_allclose(got, [8.0, -4.9, 0.25, 12.0, -3.1, 1.75], atol=1e-6)


def test_a_quarter_turn_swaps_length_and_width():
    # A car turned to lie along y: the length is now the y extent.
    got = _extents([0.0, 0.0, 0.0], [4.0, 1.8, 1.5], _yaw_quaternion(math.pi / 2))
    np.testing.assert_allclose(got, [-0.9, -2.0, -0.75, 0.9, 2.0, 0.75], atol=1e-6)


def test_a_diagonal_box_is_wider_than_either_side():
    got = _extents([0.0, 0.0, 0.0], [4.0, 2.0, 1.0], _yaw_quaternion(math.pi / 4))
    half = (4.0 + 2.0) / 2.0 * math.sqrt(0.5)
    np.testing.assert_allclose(got, [-half, -half, -0.5, half, half, 0.5], atol=1e-6)


def test_a_turn_about_x_moves_the_width_into_height():
    # Rolled a quarter turn about x: the y and z sizes trade places.
    got = _extents([0.0, 0.0, 0.0], [4.0, 1.8, 1.5], [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)])
    np.testing.assert_allclose(got, [-2.0, -0.75, -0.9, 2.0, 0.75, 0.9], atol=1e-6)


def test_extents_are_the_min_and_max_of_the_rotated_corners():
    rng = np.random.default_rng(0)
    for _ in range(200):
        yaw = float(rng.uniform(-math.pi, math.pi))
        center = rng.uniform(-30, 30, 3)
        length, width, height = rng.uniform(0.5, 12.0, 3)
        corners = np.array(
            [[sx * length / 2, sy * width / 2, sz * height / 2]
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
        world = _turn(corners, yaw) + center
        want = np.concatenate([world.min(axis=0), world.max(axis=0)])
        got = _extents(center, [length, width, height], _yaw_quaternion(yaw))
        np.testing.assert_allclose(got, want, atol=1e-4)


def test_quaternion_is_normalised_and_its_sign_does_not_matter():
    base = _extents([1.0, 2.0, 3.0], [4.0, 1.8, 1.5], _yaw_quaternion(0.7))
    scaled = _extents([1.0, 2.0, 3.0], [4.0, 1.8, 1.5], [v * 5.0 for v in _yaw_quaternion(0.7)])
    flipped = _extents([1.0, 2.0, 3.0], [4.0, 1.8, 1.5], [-v for v in _yaw_quaternion(0.7)])
    np.testing.assert_allclose(scaled, base, atol=1e-6)
    np.testing.assert_allclose(flipped, base, atol=1e-6)


def test_output_is_float32_and_empty_input_gives_an_empty_table():
    got = oriented_extents(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)))
    assert got.shape == (0, 6) and got.dtype == np.float32
    one = oriented_extents([[0, 0, 0]], [[1, 1, 1]], [[0, 0, 0, 1]])
    assert one.shape == (1, 6) and one.dtype == np.float32


@pytest.mark.parametrize(
    "rotation",
    [[0.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0, 1.0]],
)
def test_a_degenerate_quaternion_is_an_error(rotation):
    with pytest.raises(ValueError):
        oriented_extents([[0, 0, 0]], [[1, 1, 1]], [rotation])


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ValueError, match="same length"):
        oriented_extents(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((1, 4)))


def test_the_enclosing_box_crops_a_turned_vehicle_whole_where_the_unrotated_box_does_not():
    # A 4.0 x 1.8 x 1.5 m vehicle lying along y, sampled on a grid inside its own box.
    length, width, height, yaw = 4.0, 1.8, 1.5, math.pi / 2
    center = np.array([20.0, -6.0, -6.0])
    u, v, w = np.meshgrid(
        np.linspace(-length / 2, length / 2, 17),
        np.linspace(-width / 2, width / 2, 9),
        np.linspace(-height / 2, height / 2, 7),
        indexing="ij",
    )
    local = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=1)
    points = (_turn(local, yaw) + center).astype(np.float32)

    unrotated = np.concatenate([center - [length / 2, width / 2, height / 2], center + [length / 2, width / 2, height / 2]])
    enclosing = oriented_extents([center], [[length, width, height]], [_yaw_quaternion(yaw)])[0]

    kept_unrotated = len(crop_point_cloud_axis_aligned(points, unrotated.astype(np.float32)))
    kept_enclosing = len(crop_point_cloud_axis_aligned(points, enclosing))
    assert kept_enclosing == len(points)
    # The unrotated box is 4 m wide in x but the vehicle is 1.8 m wide in x: most of its length is cut off.
    assert kept_unrotated < 0.6 * len(points)


# ---------------------------------------------------------------------------
# Full 3D rotations, checked against independently built rotation matrices
# ---------------------------------------------------------------------------

def _matmul3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _quat_mul(a, b):
    """Hamilton product of two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _axis_quat(axis, angle):
    q = [0.0, 0.0, 0.0, math.cos(angle / 2.0)]
    q[axis] = math.sin(angle / 2.0)
    return q


def test_extents_are_exact_for_combined_roll_pitch_and_yaw():
    rng = np.random.default_rng(1)
    for _ in range(300):
        roll, pitch, yaw = rng.uniform(-math.pi, math.pi, 3)
        # R = Rz(yaw) Ry(pitch) Rx(roll), and the same turn as a quaternion.
        matrix = _matmul3(_rot_z(yaw), _matmul3(_rot_y(pitch), _rot_x(roll)))
        quaternion = _quat_mul(_axis_quat(2, yaw), _quat_mul(_axis_quat(1, pitch), _axis_quat(0, roll)))
        center = rng.uniform(-30, 30, 3)
        dims = rng.uniform(0.5, 12.0, 3)
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = [sx * dims[0] / 2, sy * dims[1] / 2, sz * dims[2] / 2]
                    world = [center[i] + sum(matrix[i][j] * local[j] for j in range(3)) for i in range(3)]
                    low = np.minimum(low, world)
                    high = np.maximum(high, world)
        got = _extents(center, dims, quaternion)
        np.testing.assert_allclose(got, np.concatenate([low, high]), atol=1e-4)


def test_non_finite_centres_and_sizes_are_an_error():
    with pytest.raises(ValueError, match="finite"):
        oriented_extents([[float("inf"), 0, 0]], [[1, 1, 1]], [[0, 0, 0, 1]])
    with pytest.raises(ValueError, match="finite"):
        oriented_extents([[0, 0, 0]], [[float("nan"), 1, 1]], [[0, 0, 0, 1]])
