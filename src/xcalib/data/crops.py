"""
Cropping helpers used by both the partner-facing matcher and the lab-side
paper-validation script.

Behaviour matches src/paper_experiments/consistent_loader.py:
  - Image crops are resized to crop_size x crop_size after bbox clipping.
  - Point-cloud crops keep global XYZ coordinates (axis-aligned 3D bbox + 1.25x
    expansion). Empty crops are dropped by the caller.
  - Point clouds are random-sub/zero-padded to a fixed `point_cloud_size`.

All functions are pure (no model state, no globals), which makes them safe
to call from any thread on the edge device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from ..engine.wrappers import FrameData


# ============================================================================
# Image cropping
# ============================================================================

def crop_image_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    crop_size: int = 32,
) -> Optional[np.ndarray]:
    """Crop `image` to `bbox` (x1,y1,x2,y2) and resize to crop_size x crop_size.

    Returns:
        [crop_size, crop_size, 3] float32 in [0, 1], or None if the bbox is
        degenerate / produces an empty crop.
    """
    if bbox is None or len(bbox) < 4:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)
    if x1 < 0 or x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(crop, (crop_size, crop_size))
    if crop.dtype == np.uint8:
        crop = crop.astype(np.float32) / 255.0
    return crop


# ============================================================================
# Point-cloud cropping (axis-aligned, with 1.25x expansion)
# ============================================================================

def crop_point_cloud_axis_aligned(
    points: np.ndarray,
    bbox_3d: np.ndarray,
    expansion: float = 1.25,
) -> np.ndarray:
    """Crop `points` to axis-aligned 3D bbox `bbox_3d`.

    Args:
        points:   [P, 3+] XYZ (any extra columns ignored).
        bbox_3d:  [6] (xmin, ymin, zmin, xmax, ymax, zmax) OR [3+3]
                  center + dimensions can be passed as
                  (cx,cy,cz, dx,dy,dz) — we infer based on whether values
                  look like extents (xmax > xmin etc.).
        expansion: Inflate the bbox by this factor along each axis.

    Returns:
        [Q, 3] array of GLOBAL XYZ coordinates inside the (expanded) bbox.
    """
    if points is None or points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xyz = points[:, :3].astype(np.float32, copy=False)
    bbox = np.asarray(bbox_3d, dtype=np.float32)

    if bbox.size != 6:
        raise ValueError(
            f"bbox_3d must have 6 components (got {bbox.size})."
        )

    # Heuristic: if the last 3 look like extents (any > corresponding first 3),
    # treat as (xmin,ymin,zmin,xmax,ymax,zmax). Otherwise treat as
    # (cx,cy,cz, dx,dy,dz).
    first, second = bbox[:3], bbox[3:]
    if np.all(second >= first):
        mins = first
        maxs = second
        center = (mins + maxs) / 2.0
        dim = maxs - mins
    else:
        center = first
        dim = np.abs(second)

    half = (dim * expansion) / 2.0
    local = xyz - center
    mask = np.all(np.abs(local) <= half, axis=1)
    return xyz[mask].copy()


# ============================================================================
# Point-cloud resampling
# ============================================================================

def resample_point_cloud(
    pc: np.ndarray,
    target_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sub/zero-sample a point cloud to exactly `target_size` rows of XYZ."""
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"Expected [P, >=3] array, got shape {pc.shape}")
    P = pc.shape[0]
    if P == target_size:
        return pc[:, :3].astype(np.float32, copy=False)
    if P > target_size:
        rng = rng or np.random.default_rng()
        idx = rng.choice(P, target_size, replace=False)
        return pc[idx, :3].astype(np.float32, copy=False)
    pad = np.zeros((target_size - P, 3), dtype=np.float32)
    return np.vstack([pc[:, :3].astype(np.float32, copy=False), pad])


# ============================================================================
# End-to-end: raw arrays -> FrameData
# ============================================================================

@dataclass
class PrepareConfig:
    """Knobs for `prepare_frame`."""
    crop_size: int = 32
    point_cloud_size: int = 1024
    bbox_expansion: float = 1.25


def prepare_frame(
    image: np.ndarray,
    point_cloud: np.ndarray,
    bboxes_2d: np.ndarray,
    bboxes_3d: np.ndarray,
    cfg: Optional[PrepareConfig] = None,
    rng: Optional[np.random.Generator] = None,
    images: Optional[Dict[str, np.ndarray]] = None,
    camera_per_det: Optional[Sequence[str]] = None,
) -> Tuple[FrameData, np.ndarray, np.ndarray]:
    """Run cropping + resampling and pack into a FrameData ready for inference.

    Args:
        image:        [H, W, 3] uint8 RGB.
        point_cloud:  [P, 3+] XYZ.
        bboxes_2d:    [K, 4] (x1,y1,x2,y2)
        bboxes_3d:    [M, 6] either (xmin,ymin,zmin,xmax,ymax,zmax)
                      or (cx,cy,cz, dx,dy,dz).
        cfg:          PrepareConfig (defaults to crop_size=32, P=1024).
        rng:          numpy RNG for reproducible PC sub-sampling.
        images:       Optional {camera_name: [H, W, 3] uint8 RGB} for
                      multi-camera caches (A9): each 2-D box is cropped from
                      its own source camera. Falls back to `image` when omitted.
        camera_per_det: [K] source-camera name per 2-D box (with `images`).
    Returns:
        (frame, kept_2d_indices, kept_3d_indices) — the index arrays tell
        callers which of their original detections survived cropping (empty
        bboxes / empty point clouds / unknown source cameras are dropped).
    """
    cfg = cfg or PrepareConfig()
    rng = rng or np.random.default_rng()

    multi_cam = images is not None and camera_per_det is not None

    img_crops, kept_2d = [], []
    for k in range(len(bboxes_2d)):
        src = image
        if multi_cam and k < len(camera_per_det):
            src = images.get(str(camera_per_det[k]))
            if src is None:
                continue
        crop = crop_image_bbox(src, bboxes_2d[k], cfg.crop_size)
        if crop is None:
            continue
        img_crops.append(crop)
        kept_2d.append(k)

    pc_crops, kept_3d = [], []
    for m in range(len(bboxes_3d)):
        cropped = crop_point_cloud_axis_aligned(
            point_cloud, bboxes_3d[m], expansion=cfg.bbox_expansion
        )
        if cropped.shape[0] == 0:
            continue
        pc_crops.append(resample_point_cloud(cropped, cfg.point_cloud_size, rng=rng))
        kept_3d.append(m)

    if not img_crops or not pc_crops:
        empty_img = torch.zeros((0, 3, cfg.crop_size, cfg.crop_size), dtype=torch.float32)
        empty_pc = torch.zeros((0, cfg.point_cloud_size, 3), dtype=torch.float32)
        empty_b2 = torch.zeros((0, 4), dtype=torch.float32)
        empty_c3 = torch.zeros((0, 3), dtype=torch.float32)
        return (
            FrameData(empty_img, empty_pc, empty_b2, empty_c3),
            np.array(kept_2d, dtype=np.int64),
            np.array(kept_3d, dtype=np.int64),
        )

    img_tensor = torch.from_numpy(np.stack(img_crops)).permute(0, 3, 1, 2).contiguous()
    pc_tensor = torch.from_numpy(np.stack(pc_crops))

    kept_2d_arr = np.array(kept_2d, dtype=np.int64)
    kept_3d_arr = np.array(kept_3d, dtype=np.int64)

    bboxes_2d_kept = torch.from_numpy(bboxes_2d[kept_2d_arr].astype(np.float32))

    # 3D centers, regardless of input shape
    bbox3d_kept = np.asarray(bboxes_3d, dtype=np.float32)[kept_3d_arr]
    if bbox3d_kept.shape[1] == 6:
        first = bbox3d_kept[:, :3]
        second = bbox3d_kept[:, 3:]
        treat_as_extent = np.all(second >= first, axis=1)
        centers = np.where(
            treat_as_extent[:, None],
            (first + second) / 2.0,
            first,
        )
    else:
        centers = bbox3d_kept[:, :3]
    centers_tensor = torch.from_numpy(centers.astype(np.float32))

    frame = FrameData(
        crops_2d=img_tensor,
        crops_3d=pc_tensor,
        bboxes_2d=bboxes_2d_kept,
        bbox_centers_3d=centers_tensor,
    )
    return frame, kept_2d_arr, kept_3d_arr
