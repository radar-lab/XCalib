"""
Sampling / grouping ops for PointNet++.

Verbatim port of src/core/models/calibrefine/backbones/utils.py. All
operations are pure PyTorch (no CUDA extensions), which is what we want
for portability to Jetson AGX Thor.
"""

from __future__ import annotations

import os

import torch

# Toggle deterministic FPS init for ONNX export / reproducibility.
# Set XCALIB_FPS_DETERMINISTIC=1 before importing this module to force the
# first centroid to be point 0 instead of a random pick. This makes the
# graph fully tracer-friendly (no dynamic-`high` randint) and produces a
# stable forward pass.
_DETERMINISTIC_FPS = os.environ.get("XCALIB_FPS_DETERMINISTIC", "0") == "1"


def set_deterministic_fps(enabled: bool) -> None:
    """Flip deterministic FPS at runtime (used by xcalib.engine.exporter
    while tracing, since the env var is only read at import time)."""
    global _DETERMINISTIC_FPS
    _DETERMINISTIC_FPS = bool(enabled)


def deterministic_fps_enabled() -> bool:
    """Current deterministic-FPS state."""
    return _DETERMINISTIC_FPS


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS for point cloud downsampling.

    Input:  xyz [B, N, 3]
    Return: centroids [B, npoint] (long indices)
    """
    device = xyz.device
    dtype = xyz.dtype  # honour fp16 inputs so the masked-write below matches
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    # Sentinel must be representable in `dtype`; 1e10 overflows fp16. Use the
    # dtype's max-finite value (>>any squared distance for unit-scale clouds).
    big = float(torch.finfo(dtype).max)
    distance = torch.full((B, N), big, device=device, dtype=dtype)
    if _DETERMINISTIC_FPS:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. points: [B, N, C], idx: [B, S, ...] -> [B, S, ..., C]."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance. src: [B,N,C], dst: [B,M,C] -> [B,N,M]."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Ball query around each new_xyz center.

    Input:
        xyz:     [B, N, 3] all points
        new_xyz: [B, S, 3] centers
    Return:
        group_idx: [B, S, nsample] long indices into xyz
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long, device=device)
        .view(1, 1, N)
        .repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0]
    if group_idx.shape[-1] < nsample:
        group_idx = torch.cat(
            [
                group_idx,
                group_idx[:, :, :1].repeat(1, 1, nsample - group_idx.shape[-1]),
            ],
            dim=-1,
        )
    group_idx = group_idx[:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points,
    returnfps: bool = False,
):
    """FPS centers + ball query grouping with relative-position features."""
    B, _, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points):
    """Group all points into one global neighborhood at the origin."""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device, dtype=xyz.dtype)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
