"""Visualization helpers for matching and calibration demos.

The functions return RGB ``uint8`` arrays so callers can save with Pillow,
display in notebooks, or convert to BGR for OpenCV video/GIF generation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from .protocol import CameraIntrinsics

Color = tuple[int, int, int]

WHITE: Color = (255, 255, 255)
GREEN: Color = (90, 220, 110)
GOLD: Color = (232, 178, 45)
CYAN: Color = (60, 200, 220)
GREY: Color = (150, 150, 150)
DARK: Color = (28, 27, 26)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img.copy()


def _rgb_to_bgr(color: Color) -> Color:
    return color[2], color[1], color[0]


def _put_text(img_bgr: np.ndarray, text: str, org: tuple[int, int], color: Color = WHITE) -> None:
    cv2.putText(
        img_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        _rgb_to_bgr(color),
        1,
        cv2.LINE_AA,
    )


def _bbox_centers_2d(bboxes_2d: np.ndarray) -> np.ndarray:
    boxes = np.asarray(bboxes_2d, dtype=np.float32).reshape(-1, 4)
    return np.column_stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5))


def _bbox_centers_3d(bboxes_3d: np.ndarray) -> np.ndarray:
    boxes = np.asarray(bboxes_3d, dtype=np.float32).reshape(-1, 6)
    is_extent = np.all(boxes[:, 3:6] >= boxes[:, :3], axis=1)
    centers = boxes[:, :3].copy()
    centers[is_extent] = (boxes[is_extent, :3] + boxes[is_extent, 3:6]) * 0.5
    return centers


def _match_pairs(matches: Iterable[Sequence[float]]) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for match in matches:
        if len(match) < 2:
            continue
        score = float(match[2]) if len(match) >= 3 else 1.0
        pairs.append((int(match[0]), int(match[1]), score))
    return pairs


def _bev_points(point_cloud: np.ndarray, centers: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, point_cloud.shape[-1])[:, :3]
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((len(centers), 2), dtype=np.float32)

    xy = pts[:, :2]
    all_xy = np.vstack([xy, centers[:, :2]]) if len(centers) else xy
    lo = np.nanpercentile(all_xy, 2, axis=0)
    hi = np.nanpercentile(all_xy, 98, axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = span * 0.08
    lo -= pad
    hi += pad
    span = np.maximum(hi - lo, 1.0)

    def map_xy(values: np.ndarray) -> np.ndarray:
        u = (values[:, 1] - lo[1]) / span[1] * (width - 1)
        v = (1.0 - (values[:, 0] - lo[0]) / span[0]) * (height - 1)
        return np.column_stack((u, v)).astype(np.float32)

    return map_xy(xy), map_xy(centers[:, :2]) if len(centers) else np.zeros((0, 2), dtype=np.float32)


def draw_matching_overlay(
    image: np.ndarray,
    point_cloud: np.ndarray,
    bboxes_2d: np.ndarray,
    bboxes_3d: np.ndarray,
    matches: Iterable[Sequence[float]],
    *,
    output_height: int = 420,
    match_threshold: float | None = None,
) -> np.ndarray:
    """Draw camera detections, LiDAR BEV detections, and matched links.

    Args:
        image: RGB image, ``[H, W, 3]`` ``uint8``.
        point_cloud: LiDAR points with XYZ in the first three columns.
        bboxes_2d: Camera boxes as ``(x1, y1, x2, y2)``.
        bboxes_3d: LiDAR boxes as extents or center+dimensions.
        matches: Iterable of ``(camera_index, lidar_index, score)`` triples,
            such as ``MatchResult.matches``.
        output_height: Height of each visual pane.
        match_threshold: Optional score floor for display; lower-scoring
            matches are omitted from the drawn links.

    Returns:
        RGB ``uint8`` side-by-side overlay.
    """
    rgb = _as_rgb(image)
    h, w = rgb.shape[:2]
    cam_w = max(1, int(round(output_height * w / h)))
    bev_w = max(320, int(output_height * 0.8))
    pad = 14
    header = 36
    out = np.full((output_height + header + pad, cam_w + bev_w + pad * 3, 3), DARK, np.uint8)

    cam_bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (cam_w, output_height))
    sx, sy = cam_w / w, output_height / h
    cam_x, pane_y = pad, header
    bev_x = cam_x + cam_w + pad
    out[pane_y : pane_y + output_height, cam_x : cam_x + cam_w] = cam_bgr

    bev = np.full((output_height, bev_w, 3), (24, 22, 20), dtype=np.uint8)
    centers_3d = _bbox_centers_3d(bboxes_3d)
    bev_points, bev_centers = _bev_points(point_cloud, centers_3d, bev_w, output_height)
    for u, v in bev_points[:: max(1, len(bev_points) // 4000)]:
        if 0 <= u < bev_w and 0 <= v < output_height:
            cv2.circle(bev, (int(u), int(v)), 1, _rgb_to_bgr(GREY), -1, cv2.LINE_AA)
    out[pane_y : pane_y + output_height, bev_x : bev_x + bev_w] = bev

    boxes2 = np.asarray(bboxes_2d, dtype=np.float32).reshape(-1, 4)
    for idx, box in enumerate(boxes2):
        x1, y1, x2, y2 = box
        p1 = (int(cam_x + x1 * sx), int(pane_y + y1 * sy))
        p2 = (int(cam_x + x2 * sx), int(pane_y + y2 * sy))
        cv2.rectangle(out, p1, p2, _rgb_to_bgr(WHITE), 1, cv2.LINE_AA)
        _put_text(out, f"C{idx}", (p1[0], max(header + 14, p1[1] - 4)), WHITE)

    for idx, center in enumerate(bev_centers):
        p = (int(bev_x + center[0]), int(pane_y + center[1]))
        cv2.circle(out, p, 5, _rgb_to_bgr(CYAN), -1, cv2.LINE_AA)
        _put_text(out, f"L{idx}", (p[0] + 6, p[1] + 4), CYAN)

    cam_centers = _bbox_centers_2d(boxes2)
    pairs = _match_pairs(matches)
    for cam_idx, lidar_idx, score in pairs:
        if match_threshold is not None and score < match_threshold:
            continue
        if cam_idx >= len(cam_centers) or lidar_idx >= len(bev_centers):
            continue
        p = (
            int(cam_x + cam_centers[cam_idx, 0] * sx),
            int(pane_y + cam_centers[cam_idx, 1] * sy),
        )
        q = (int(bev_x + bev_centers[lidar_idx, 0]), int(pane_y + bev_centers[lidar_idx, 1]))
        cv2.line(out, p, q, _rgb_to_bgr(GREEN), 2, cv2.LINE_AA)
        _put_text(out, f"{score:.2f}", ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2), GREEN)

    _put_text(out, "xcalib matching: camera detections -> LiDAR detections", (pad, 24), WHITE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _projection_matrix(
    *,
    projection: np.ndarray | None,
    intrinsics: CameraIntrinsics | np.ndarray | None,
    rotation: np.ndarray | None,
    translation: np.ndarray | None,
) -> np.ndarray:
    if projection is not None:
        P = np.asarray(projection, dtype=np.float64)
        if P.shape != (3, 4):
            raise ValueError(f"projection must be [3, 4], got {P.shape}")
        return P
    if intrinsics is None or rotation is None or translation is None:
        raise ValueError("pass either projection or intrinsics+rotation+translation")
    if isinstance(intrinsics, CameraIntrinsics):
        K = intrinsics.K
    else:
        K = np.asarray(intrinsics, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R, t])


def _project(points: np.ndarray, projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    hom = np.column_stack([pts, np.ones((len(pts),), dtype=np.float64)])
    cam = hom @ projection.T
    valid = cam[:, 2] > 1e-6
    uv = np.zeros((len(pts), 2), dtype=np.float64)
    uv[valid] = cam[valid, :2] / cam[valid, 2:3]
    return uv, valid


def draw_calibration_overlay(
    image: np.ndarray,
    point_cloud: np.ndarray,
    *,
    projection: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    bboxes_3d: np.ndarray | None = None,
    max_points: int = 6000,
) -> np.ndarray:
    """Project LiDAR points and optional 3D-box centers onto the camera image.

    Args:
        image: RGB image, ``[H, W, 3]`` ``uint8``.
        point_cloud: LiDAR points with XYZ in columns 0-2.
        projection: Optional ``3x4`` camera projection matrix ``K[R|t]``.
        intrinsics: Camera intrinsics used with ``rotation`` and
            ``translation`` when ``projection`` is not supplied.
        rotation: ``3x3`` extrinsic rotation.
        translation: ``3`` or ``3x1`` extrinsic translation.
        bboxes_3d: Optional LiDAR detections; their centers are projected and
            marked with labels.
        max_points: Upper bound on projected point count for readable overlays.

    Returns:
        RGB ``uint8`` camera overlay.
    """
    rgb = _as_rgb(image)
    h, w = rgb.shape[:2]
    P = _projection_matrix(
        projection=projection,
        intrinsics=intrinsics,
        rotation=rotation,
        translation=translation,
    )

    pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, point_cloud.shape[-1])[:, :3]
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    uv, valid = _project(pts, P)
    in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if np.any(in_img):
        z = pts[in_img, 2]
        zmin, zmax = float(np.percentile(z, 5)), float(np.percentile(z, 95))
        denom = max(zmax - zmin, 1e-6)
        norm = np.clip((z - zmin) / denom, 0, 1)
        colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]
        for (u, v), color in zip(uv[in_img], colors):
            cv2.circle(out, (int(u), int(v)), 1, tuple(int(c) for c in color), -1, cv2.LINE_AA)

    if bboxes_3d is not None:
        centers = _bbox_centers_3d(bboxes_3d)
        c_uv, c_valid = _project(centers, P)
        for idx, ((u, v), ok) in enumerate(zip(c_uv, c_valid)):
            if ok and 0 <= u < w and 0 <= v < h:
                p = (int(u), int(v))
                cv2.circle(out, p, 6, _rgb_to_bgr(GREEN), -1, cv2.LINE_AA)
                _put_text(out, f"L{idx}", (p[0] + 7, p[1] - 5), GREEN)

    _put_text(out, "xcalib calibration: LiDAR projected onto camera", (16, 28), WHITE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
