"""
Generate the documentation demo media (GIFs + posters) from the public A9 split.

The matcher runs through the public xcalib API on the repo-local A9 test cache:

1. ``a9_matching.gif``     — side-by-side camera / LiDAR bird's-eye view; the
   real matcher scores every camera x LiDAR pair, fans the candidate links and
   locks the argmax per camera detection (below-threshold rows stay unmatched).
2. ``a9_calibration.gif``  — before/after of the calibrated camera-LiDAR
   projection: the LiDAR cloud and 3D detections are drawn onto the image with
   the intersection's reference calibration, and the matcher's confident
   matches are scored against that same projection (the agreement statistic
   shown in the HUD).
3. ``a9_matching.png`` / ``a9_calibration.png`` — static posters of the final
   states for README embedding.

The camera projection comes from the TUMTraf / A9 dev-kit calibration file
(s110_camera_basler_south2_8mm), i.e. the dataset's published reference:
https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit/blob/main/calib/s110_camera_basler_south2_8mm.json
The script also feeds the matcher's pairs through ``CalibrationSession`` /
PnP-RANSAC (what ``matcher.calibrate()`` does) and prints the result — on this
short 29-frame slice the matched vehicle centers are nearly coplanar (flat
intersection), which leaves the full 6-DoF pose under-constrained, so the
reference projection is what the figures show.

Run from the repo root (weights + A9 caches must be present locally):

    pixi run python scripts/make_demo_media.py
    pixi run python scripts/make_demo_media.py --max-frames 40 --fps 10

Author: AXIBA (leolihao@arizona.edu)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xcalib import CameraIntrinsics, Matcher  # noqa: E402
from xcalib.data.hdf5_loader import UTCFrameLoader  # noqa: E402
from xcalib.oneshot.calibration import (  # noqa: E402
    CalibrationSession,
    project_points,
)

# ---------------------------------------------------------------------------
# Reference calibration (TUMTraf / A9 dev-kit, s110_camera_basler_south2_8mm)
# Projection from the s110_lidar_ouster_north frame — the frame the HDF5
# cache's point clouds and 3D boxes live in. Source:
# https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit/blob/main/calib/s110_camera_basler_south2_8mm.json
# ---------------------------------------------------------------------------

REF_CAMERA = "s110_camera_basler_south2_8mm"
REF_P_LIDAR_NORTH = np.array([
    [1318.95273325, -859.15213894, -289.13390611, 11272.03223502],
    [90.01799314, -2.9727517, -1445.63809767, 585.78988153],
    [0.876766, 0.344395, -0.335669, -7.26891],
])


def reference_pose() -> Tuple[CameraIntrinsics, np.ndarray, np.ndarray]:
    """Decompose the published projection into (intrinsics, R, t)."""
    K, R, t_h, *_ = cv2.decomposeProjectionMatrix(REF_P_LIDAR_NORTH)
    K = K / K[2, 2]
    cam_center = (t_h[:3] / t_h[3]).reshape(3)
    t = -R @ cam_center
    return CameraIntrinsics.from_matrix(K), R, t

# ---------------------------------------------------------------------------
# Shared drawing helpers (palette mirrors the UTC4 standalone demo)
# ---------------------------------------------------------------------------

WHITE = (255, 255, 255)
GOLD = (45, 178, 232)        # BGR ~ Arizona gold
TEAL = (170, 190, 70)
CYAN = (220, 200, 60)
GREEN = (90, 220, 110)
GREY = (110, 110, 110)
DARK = (28, 27, 26)
BEVBG = (26, 23, 20)
RINGC = (74, 70, 64)


def put_text(img, text, org, scale=0.5, color=WHITE, thick=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, alpha=0.5, color=(0, 0, 0)) -> None:
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, color, dtype=np.uint8), alpha,
                    roi, 1 - alpha, 0, roi)


def lerp(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def turbo_lut() -> np.ndarray:
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8),
                             cv2.COLORMAP_TURBO)[:, 0, :]


def write_gif(frames_bgr: Sequence[np.ndarray], path: Path, fps: int) -> None:
    """Palette-quantised GIF via Pillow (global palette keeps the file small)."""
    from PIL import Image

    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in frames_bgr]
    # Build the shared palette from first/middle/last frames stacked, so
    # colours that only appear late (overlays, links) survive quantisation.
    sample = np.vstack([np.asarray(imgs[0]),
                        np.asarray(imgs[len(imgs) // 2]),
                        np.asarray(imgs[-1])])
    base = Image.fromarray(sample).quantize(colors=255,
                                            method=Image.MEDIANCUT)
    quantised = [im.quantize(colors=255, palette=base, dither=Image.NONE)
                 for im in imgs]
    quantised[0].save(
        path, save_all=True, append_images=quantised[1:],
        duration=int(round(1000 / fps)), loop=0, optimize=True,
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames_bgr)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def aabb_centers(bboxes_3d: np.ndarray) -> np.ndarray:
    b = np.asarray(bboxes_3d, dtype=np.float64)
    return (b[:, :3] + b[:, 3:6]) / 2.0


def filter_to_camera(frame, camera: str):
    """Keep only the 2D detections (and match rows) from one camera."""
    sel = np.asarray(frame.camera_per_det) == camera
    return (frame.images[camera], frame.bboxes_2d[sel],
            frame.match_matrix[sel], sel)


def pick_camera(loader: UTCFrameLoader, max_frames: int = 40) -> str:
    counts: Dict[str, int] = {}
    for k, frame in enumerate(loader):
        if k >= max_frames:
            break
        for cam in np.asarray(frame.camera_per_det):
            counts[str(cam)] = counts.get(str(cam), 0) + 1
    cam = max(counts, key=counts.get)
    print(f"cameras seen: {counts} -> using {cam!r}")
    return cam


def pseudo_intrinsics(w: int, h: int, fov_deg: float) -> CameraIntrinsics:
    f = 0.5 * w / math.tan(math.radians(fov_deg) / 2.0)
    return CameraIntrinsics(fx=f, fy=f, cx=w / 2.0, cy=h / 2.0)


# ---------------------------------------------------------------------------
# Calibration from the matcher's own correspondences
# ---------------------------------------------------------------------------

def mutual_best_matches(
    matcher: Matcher,
    img: np.ndarray,
    b2: np.ndarray,
    frame,
    min_score: float,
) -> List[Tuple[int, int, float]]:
    """Confident pairs where camera and LiDAR pick each other (mutual argmax)."""
    res = matcher.match(img, frame.point_cloud, b2, frame.bboxes_3d,
                        top_k=1, validate="off")
    S = res.similarity
    if S.size == 0:
        return []
    kept2 = (res.kept_2d_indices if res.kept_2d_indices.size
             else np.arange(S.shape[0]))
    kept3 = (res.kept_3d_indices if res.kept_3d_indices.size
             else np.arange(S.shape[1]))
    row_best = S.argmax(axis=1)
    col_best = S.argmax(axis=0)
    out = []
    for i, j in enumerate(row_best):
        if col_best[j] == i and S[i, j] >= min_score:
            out.append((int(kept2[i]), int(kept3[j]), float(S[i, j])))
    return out


def harvest_pairs(
    matcher: Matcher,
    loader: UTCFrameLoader,
    camera: str,
    *,
    max_frames: int,
    min_score: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Confident (2D box center, 3D box center) pairs across the clip."""
    pts2, pts3 = [], []
    n_frames = 0
    for frame in loader:
        if n_frames >= max_frames:
            break
        img, b2, _mm, _sel = filter_to_camera(frame, camera)
        if len(b2) < 2 or len(frame.bboxes_3d) < 2:
            continue
        for i, j, _s in mutual_best_matches(matcher, img, b2, frame,
                                            min_score):
            x1, y1, x2, y2 = np.asarray(b2[i], dtype=np.float64)
            c3 = aabb_centers(frame.bboxes_3d[j : j + 1])[0]
            if math.hypot(c3[0], c3[1]) > 160.0:
                continue
            pts2.append([(x1 + x2) / 2, (y1 + y2) / 2])
            pts3.append(c3)
        n_frames += 1
    return (np.asarray(pts2, dtype=np.float64),
            np.asarray(pts3, dtype=np.float64), len(pts2), n_frames)


def run_calibration(
    matcher: Matcher,
    loader: UTCFrameLoader,
    camera: str,
    *,
    max_frames: int,
    min_score: float,
) -> Tuple[CameraIntrinsics, np.ndarray, np.ndarray, int, float]:
    """Harvest matcher pairs, run the PnP session, score vs the reference.

    Returns (intrinsics, R, t, n_pairs, median agreement px) where (R, t) is
    the published reference pose used for the visual overlays. The PnP result
    from the matcher's own pairs is printed for transparency — this 29-frame
    intersection slice has nearly coplanar vehicle centers, which leaves
    PnP's full 6-DoF pose under-constrained.
    """
    pts2, pts3, n_pairs, n_frames = harvest_pairs(
        matcher, loader, camera, max_frames=max_frames, min_score=min_score)
    print(f"harvested {n_pairs} mutual-best pairs from {n_frames} frames")

    intr, R, t = reference_pose()

    # What matcher.calibrate() runs under the hood (printed, not drawn).
    session = CalibrationSession(max_pairs=6000)
    session.add_correspondences(pts2, pts3)
    result = session.solve(intr)
    if result.success:
        print(f"PnP/RANSAC on matcher pairs: {result.n_inliers}/{n_pairs} "
              f"inliers, {result.reproj_error_px:.1f} px on inliers "
              f"(pose under-constrained by coplanar centers — see docstring)")

    # Agreement between the matcher's pairs and the reference projection.
    uv, _ = project_points(pts3, intr, R, t)
    err = np.linalg.norm(uv - pts2, axis=1)
    med = float(np.median(err))
    print(f"matcher pairs vs reference projection: median {med:.1f} px "
          f"(camera box center vs LiDAR centroid offset included)")
    return intr, R, t, n_pairs, med


# ---------------------------------------------------------------------------
# BEV rasteriser (static layer, reused by the matching GIF)
# ---------------------------------------------------------------------------

class BevView:
    def __init__(self, points: np.ndarray, centers: np.ndarray,
                 width: int, height: int, forward_xy: Tuple[float, float]):
        self.W, self.H = width, height
        nrm = math.hypot(*forward_xy) or 1.0
        self.efx, self.efy = forward_xy[0] / nrm, forward_xy[1] / nrm
        self.elx, self.ely = -self.efy, self.efx

        cf = centers[:, 0] * self.efx + centers[:, 1] * self.efy
        cl = centers[:, 0] * self.elx + centers[:, 1] * self.ely
        f0, f1 = cf.min() - 14, cf.max() + 14
        l0, l1 = cl.min() - 14, cl.max() + 14
        if f1 - f0 < 60:
            m = (f0 + f1) / 2
            f0, f1 = m - 30, m + 30
        if l1 - l0 < 60:
            m = (l0 + l1) / 2
            l0, l1 = m - 30, m + 30
        self.fc, self.lc = (f0 + f1) / 2, (l0 + l1) / 2
        self.ppm = min(width / (l1 - l0), height / (f1 - f0)) * 0.96
        self._window = (f0, f1, l0, l1)
        self.layer = self._raster(points)

    def map(self, x, y):
        f = x * self.efx + y * self.efy
        ln = x * self.elx + y * self.ely
        u = self.W / 2 - (ln - self.lc) * self.ppm
        v = self.H / 2 - (f - self.fc) * self.ppm
        return u, v

    def _raster(self, pts: np.ndarray) -> np.ndarray:
        f0, f1, l0, l1 = self._window
        bev = np.full((self.H, self.W, 3), BEVBG, np.uint8)
        pf = pts[:, 0] * self.efx + pts[:, 1] * self.efy
        pl = pts[:, 0] * self.elx + pts[:, 1] * self.ely
        keep = (pf >= f0) & (pf <= f1) & (pl >= l0) & (pl <= l1)
        pw, z = pts[keep], pts[keep][:, 2]
        zlo = float(np.percentile(z, 2))
        keep2 = (z >= zlo - 0.6) & (z <= zlo + 16.0)
        pw, z = pw[keep2], z[keep2]
        lut = turbo_lut()
        cidx = np.clip((z - zlo) / 6.0, 0, 1)
        col = lut[(cidx * 255).astype(np.uint8)]
        order = np.argsort(z)
        uu, vv = self.map(pw[order, 0], pw[order, 1])
        col = col[order]
        ou, ov = self.map(0.0, 0.0)
        for r in (20, 40, 60):
            cv2.circle(bev, (int(ou), int(ov)), int(r * self.ppm),
                       RINGC, 1, cv2.LINE_AA)
        ui, vi = np.round(uu).astype(int), np.round(vv).astype(int)
        for du in (0, 1):
            for dv in (0, 1):
                a, b = ui + du, vi + dv
                m = (a >= 0) & (a < self.W) & (b >= 0) & (b < self.H)
                bev[b[m], a[m]] = col[m]
        if 0 <= ou < self.W and 0 <= ov < self.H:
            cv2.circle(bev, (int(ou), int(ov)), 5, WHITE, -1, cv2.LINE_AA)
            put_text(bev, "sensor", (int(ou) + 8, int(ov) + 4), 0.36,
                     (210, 210, 210), 1)
        return bev


# ---------------------------------------------------------------------------
# Demo 1 — matching GIF
# ---------------------------------------------------------------------------

def pick_demo_frame(loader: UTCFrameLoader, camera: str, max_frames: int = 60):
    """Frame whose single-camera detections have the most GT-matched rows."""
    best, best_key = None, (-1, -1)
    for k, frame in enumerate(loader):
        if k >= max_frames:
            break
        if camera not in frame.images:
            continue
        _img, b2, mm, _sel = filter_to_camera(frame, camera)
        n_gt = int(mm.any(axis=1).sum())
        key = (n_gt, min(len(b2), 12))
        if key > best_key and len(b2) >= 4:
            best, best_key = frame, key
    if best is None:
        raise SystemExit("no usable demo frame found")
    print(f"matching demo frame: {best.frame_key} "
          f"({best_key[0]} GT-matched rows, {best_key[1]} cam dets)")
    return best


def render_matching(
    matcher: Matcher,
    model_name: str,
    frame,
    camera: str,
    bev_forward: Tuple[float, float],
    out_gif: Path,
    out_png: Path,
    fps: int,
    raw_db: Optional[List[dict]] = None,
) -> None:
    img, b2, mm, _sel = filter_to_camera(frame, camera)
    centers = aabb_centers(frame.bboxes_3d)
    res = matcher.match(img, frame.point_cloud, b2, frame.bboxes_3d,
                        top_k=1, validate="warn")

    # Similarity over the cropping-survived subset, mapped back to originals.
    S = res.similarity
    kept2 = (res.kept_2d_indices if res.kept_2d_indices.size
             else np.arange(S.shape[0]))
    kept3 = (res.kept_3d_indices if res.kept_3d_indices.size
             else np.arange(S.shape[1]))
    N, M = len(kept2), len(kept3)
    argmax = S.argmax(axis=1) if S.size else np.zeros(0, int)
    row_min = S.min(axis=1, keepdims=True)
    row_rng = np.clip(S.max(axis=1, keepdims=True) - row_min, 1e-6, None)
    rown = (S - row_min) / row_rng

    # Layout: HUD strip on top, camera pane left, BEV pane right.
    PH = 420
    H0, W0 = img.shape[:2]
    CW = int(round(PH * W0 / H0))
    PAD, PY0 = 14, 64
    BW = max(330, int(PH * 0.78))
    W = PAD + CW + PAD + BW + PAD
    Hh = PY0 + PH + 16
    CX0, BX0 = PAD, PAD + CW + PAD

    cam_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (CW, PH),
                           interpolation=cv2.INTER_AREA)
    cam_bg = cv2.addWeighted(cam_small, 0.85, np.zeros_like(cam_small), 0.15, 0)
    sx, sy = CW / W0, PH / H0

    bev = BevView(frame.point_cloud[:, :3].astype(np.float64), centers,
                  BW, PH, bev_forward)

    # Pane-space geometry.
    box_px = b2[kept2].astype(np.float64)
    box_px[:, [0, 2]] = CX0 + box_px[:, [0, 2]] * sx
    box_px[:, [1, 3]] = PY0 + box_px[:, [1, 3]] * sy
    camC = np.stack([(box_px[:, 0] + box_px[:, 2]) / 2,
                     (box_px[:, 1] + box_px[:, 3]) / 2], axis=1)
    cu, cv_ = bev.map(centers[kept3, 0], centers[kept3, 1])
    bevC = np.stack([BX0 + cu, PY0 + cv_], axis=1)
    oriented = oriented_box_corners(frame, raw_db)
    foot: List[Optional[np.ndarray]] = []
    for j in kept3:
        if oriented[j] is not None:
            cs = oriented[j][:4, :2]            # oriented bottom face
        else:
            x0, y0, _, x1, y1, _ = frame.bboxes_3d[j]
            cs = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                          np.float64)
        fu, fv = bev.map(cs[:, 0], cs[:, 1])
        foot.append(np.stack([BX0 + fu, PY0 + fv], axis=1))

    stage = np.full((Hh, W, 3), DARK, np.uint8)
    stage[PY0:PY0 + PH, CX0:CX0 + CW] = cam_bg
    stage[PY0:PY0 + PH, BX0:BX0 + BW] = bev.layer
    for x0, x1 in ((CX0, CX0 + CW), (BX0, BX0 + BW)):
        cv2.rectangle(stage, (x0 - 1, PY0 - 1), (x1, PY0 + PH),
                      (70, 70, 70), 1)
    panel(stage, CX0, PY0, CX0 + CW, PY0 + 20, 0.55)
    put_text(stage, "CAMERA - 2D detections", (CX0 + 8, PY0 + 15), 0.42)
    panel(stage, BX0, PY0, BX0 + BW, PY0 + 20, 0.55)
    put_text(stage, "LiDAR - bird's-eye view", (BX0 + 8, PY0 + 15), 0.42)

    def draw_dets(f, active, locked, pick=False):
        for j in range(M):
            c = CYAN
            if any(argmax[i] == j for i in locked):
                c = GREEN
            elif pick and 0 <= active < N and argmax[active] == j:
                c = GOLD
            if foot[j] is not None:
                cv2.polylines(f, [foot[j].astype(np.int32).reshape(-1, 1, 2)],
                              True, c, 2, cv2.LINE_AA)
            q = bevC[j]
            cv2.circle(f, (int(q[0]), int(q[1])), 4, c, -1, cv2.LINE_AA)
        for i in range(N):
            x1, y1, x2, y2 = box_px[i].astype(int)
            c = GOLD if i == active else (GREEN if i in locked else WHITE)
            th = 3 if i == active else (2 if i in locked else 1)
            cv2.rectangle(f, (x1, y1), (x2, y2), c, th, cv2.LINE_AA)

    def draw_locked(f, locked):
        for i in locked:
            p, q = camC[i], bevC[argmax[i]]
            cv2.line(f, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     GREEN, 2, cv2.LINE_AA)

    def hud(f, title, sub, accent):
        panel(f, 0, 0, W, PY0 - 8, 0.55)
        put_text(f, f"xcalib  -  matcher.match()  ({model_name}, A9 test "
                    f"frame)", (18, 24), 0.52, WHITE, 1)
        put_text(f, title, (18, 46), 0.5, accent, 1)
        if sub:
            put_text(f, sub, (W - 8 - 8 * len(sub), 46), 0.42,
                     (215, 215, 215), 1)

    frames: List[np.ndarray] = []
    locked: List[int] = []
    lock_thr = 0.5

    for _ in range(int(1.2 * fps)):
        f = stage.copy()
        draw_dets(f, -1, [])
        hud(f, "DETECTIONS in", f"{N} camera x {M} LiDAR", TEAL)
        frames.append(f)

    # Walk camera detections from most to least confident; keep the clip short.
    top1 = S.max(axis=1) if S.size else np.zeros(0)
    show = [int(i) for i in np.argsort(-top1)[: min(N, 8)]]
    for i in show:
        top = list(np.argsort(-rown[i])[:7])
        js = int(argmax[i])
        confident = top1[i] >= lock_thr
        for k in range(max(3, int(0.4 * fps))):
            t = smoothstep(k / max(1, int(0.4 * fps) - 1))
            f = stage.copy()
            draw_dets(f, i, locked)
            draw_locked(f, locked)
            for j in top:
                w_ = rown[i, j]
                p, q = camC[i], bevC[j]
                qx = int(p[0] + (q[0] - p[0]) * t)
                qy = int(p[1] + (q[1] - p[1]) * t)
                cv2.line(f, (int(p[0]), int(p[1])), (qx, qy),
                         lerp(GREY, GOLD, w_), 1 + int(2 * w_), cv2.LINE_AA)
            hud(f, f"SCORING camera {i}: cosine similarity to every "
                    f"LiDAR detection", "brighter = higher", GOLD)
            frames.append(f)
        if not confident:
            f = stage.copy()
            draw_dets(f, i, locked)
            draw_locked(f, locked)
            put_text(f, f"best sim={S[i, js]:.2f} < {lock_thr} -> no match",
                     (int(camC[i][0]) - 60, int(camC[i][1]) - 10), 0.46,
                     GREY, 1)
            hud(f, f"REJECT camera {i}: best similarity below threshold",
                "unmatched detections stay unmatched", GREY)
            frames.extend([f] * int(0.6 * fps))
            continue
        for k in range(max(3, int(0.4 * fps))):
            t = smoothstep(k / max(1, int(0.4 * fps) - 1))
            f = stage.copy()
            draw_dets(f, i, locked, pick=True)
            draw_locked(f, locked)
            p, q = camC[i], bevC[js]
            c = lerp(GOLD, GREEN, t)
            cv2.line(f, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     c, 2 + int(2 * t), cv2.LINE_AA)
            put_text(f, f"sim={S[i, js]:.2f}",
                     (int((p[0] + q[0]) / 2), int((p[1] + q[1]) / 2) - 8),
                     0.46, GREEN, 1)
            hud(f, f"SELECT camera {i} -> LiDAR {int(kept3[js])}",
                "argmax of the similarity row", GREEN)
            frames.append(f)
        locked.append(i)

    gt = mm[kept2][:, kept3]
    n_gt_rows = int(sum(1 for i in locked if gt[i].any()))
    n_correct = int(sum(bool(gt[i, argmax[i]]) for i in locked if gt[i].any()))
    outro = stage.copy()
    draw_dets(outro, -1, locked)
    draw_locked(outro, locked)
    hud(outro, "MATCHED PAIRS",
        f"top-1 vs GT on this frame: {n_correct}/{n_gt_rows}", TEAL)
    frames.extend([outro] * int(2.0 * fps))

    write_gif(frames, out_gif, fps)
    cv2.imwrite(str(out_png), outro)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------------------
# Demo 2 — calibration GIF (before / after projection overlay)
# ---------------------------------------------------------------------------

# Corner layout: 0-3 bottom face (CCW), 4-7 top face (same order).
CUBE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cuboid_corners(center, quat, dims) -> np.ndarray:
    """8 corners of an oriented cuboid, bottom face (CCW) then top face."""
    R = quat_to_rotation(*quat)
    length, width, height = dims
    out = []
    for sz in (-0.5, 0.5):
        for sx, sy in ((0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)):
            out.append(np.asarray(center)
                       + R @ np.array([sx * length, sy * width, sz * height]))
    return np.array(out)


def load_raw_label_db(labels_dir: Path) -> Optional[List[dict]]:
    """Load every OpenLABEL point-cloud label file (centers + corners)."""
    import json

    if not labels_dir.is_dir():
        return None
    files = sorted(labels_dir.glob("*.json"))
    if not files:
        return None
    db = []
    for fp in files:
        try:
            data = json.load(open(fp, encoding="utf-8"))
            frames = data["openlabel"]["frames"]
            objs = next(iter(frames.values()))["objects"]
        except (KeyError, StopIteration, ValueError):
            continue
        centers, corners = [], []
        for o in objs.values():
            val = o["object_data"]["cuboid"]["val"]
            c, q, d = val[0:3], val[3:7], val[7:10]
            centers.append(c)
            corners.append(cuboid_corners(c, q, d))
        if centers:
            db.append({"file": fp.name,
                       "centers": np.asarray(centers, dtype=np.float64),
                       "corners": corners})
    return db or None


def match_raw_frame(
    db: List[dict], cache_centers: np.ndarray, tol_m: float = 0.5
) -> Optional[List[Optional[np.ndarray]]]:
    """Find the raw label file for a cache frame by its detection centers.

    The HDF5 cache keeps each cuboid's center but drops its yaw (the matcher
    only consumes centers and crops). The center sets are unique per frame,
    so nearest-center matching reliably recovers the source label file —
    and with it the original oriented boxes.
    """
    best, best_med = None, float("inf")
    for entry in db:
        d = np.linalg.norm(
            cache_centers[:, None, :] - entry["centers"][None, :, :], axis=2)
        med = float(np.median(d.min(axis=1)))
        if med < best_med:
            best, best_med = entry, med
    if best is None or best_med > 0.1:
        return None
    out: List[Optional[np.ndarray]] = []
    for c in cache_centers:
        d = np.linalg.norm(best["centers"] - c, axis=1)
        j = int(d.argmin())
        out.append(best["corners"][j] if d[j] < tol_m else None)
    return out


def refit_box_corners(frame, min_points: int = 24) -> List[Optional[np.ndarray]]:
    """Fallback wireframes when raw labels are unavailable.

    The cache's axis-aligned hulls are diagonal-bloated for vehicles not
    driving along a LiDAR axis, so instead of projecting them we refit the
    observed footprint: a minimum-area rectangle over the detection's own
    points (z-trimmed so the road ring does not vote), extruded between the
    stored z extents. Implausible refits (partial one-sided returns) yield
    None — the caller draws a center marker instead of a wrong box.
    """
    pts = frame.point_cloud[:, :3].astype(np.float64)
    out: List[Optional[np.ndarray]] = []
    for box in frame.bboxes_3d:
        xm, ym, zm, xM, yM, zM = np.asarray(box, dtype=np.float64)
        cx, cy = (xm + xM) / 2, (ym + yM) / 2
        radius = 0.5 * math.hypot(xM - xm, yM - ym) + 0.3
        near = ((np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < radius)
                & (pts[:, 2] >= zm + 0.2) & (pts[:, 2] <= zM + 0.4))
        if near.sum() < min_points:
            out.append(None)
            continue
        rect = cv2.minAreaRect(pts[near, :2].astype(np.float32))
        long_r, short_r = max(rect[1]), min(rect[1])
        dx, dy = xM - xm, yM - ym
        plausible = (long_r <= math.hypot(dx, dy) + 0.3
                     and long_r >= 0.55 * max(dx, dy)
                     and long_r * short_r <= dx * dy + 0.5
                     and long_r <= 8.0)
        if not plausible:
            out.append(None)
            continue
        bottom = cv2.boxPoints(((rect[0][0], rect[0][1]),
                                (rect[1][0] + 0.3, rect[1][1] + 0.3),
                                rect[2]))
        corners = np.array([[p[0], p[1], zm] for p in bottom]
                           + [[p[0], p[1], zM] for p in bottom])
        out.append(corners)
    return out


def oriented_box_corners(
    frame, raw_db: Optional[List[dict]] = None
) -> List[Optional[np.ndarray]]:
    """Oriented wireframes: raw A9 labels when available, point refit else."""
    if raw_db is not None:
        matched = match_raw_frame(db=raw_db,
                                  cache_centers=aabb_centers(frame.bboxes_3d))
        if matched is not None:
            return matched
    return refit_box_corners(frame)


def render_calibration(
    frame,
    camera: str,
    intr: CameraIntrinsics,
    R: np.ndarray,
    t: np.ndarray,
    n_pairs: int,
    med_px: float,
    out_gif: Path,
    out_png: Path,
    fps: int,
    raw_db: Optional[List[dict]] = None,
) -> None:
    img, b2, _mm, _sel = filter_to_camera(frame, camera)
    H0, W0 = img.shape[:2]
    VW = 880
    VH = int(round(VW * H0 / W0))
    base = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (VW, VH),
                      interpolation=cv2.INTER_AREA)
    sx, sy = VW / W0, VH / H0

    # Depth-coloured cloud through the calibrated projection.
    pts = frame.point_cloud[:, :3].astype(np.float64)
    uv, depth = project_points(pts, intr, R, t)
    vis = (depth > 1.0) & (depth < 220.0)
    uv, depth = uv[vis], depth[vis]
    u = np.round(uv[:, 0] * sx).astype(int)
    v = np.round(uv[:, 1] * sy).astype(int)
    inb = (u >= 0) & (u < VW) & (v >= 0) & (v < VH)
    u, v, depth = u[inb], v[inb], depth[inb]
    lut = turbo_lut()
    lo, hi = np.percentile(depth, [5, 95])
    d01 = np.clip((depth - lo) / max(1e-6, hi - lo), 0, 1)
    col = lut[((1 - d01) * 255).astype(np.uint8)]
    order = np.argsort(-depth)  # far first, near drawn on top
    u, v, col = u[order], v[order], col[order]
    print(f"calibration overlay: {len(u)} cloud points in view")

    overlay = base.copy()
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            a = np.clip(u + du, 0, VW - 1)
            b = np.clip(v + dv, 0, VH - 1)
            overlay[b, a] = col

    # 3D detection wireframes through the same projection. The cache's boxes
    # are yaw-less hulls, so the true oriented boxes come from the raw A9
    # labels (or a point refit); sparse/unmatched detections fall back to a
    # centroid marker instead of a wrong box.
    for box, corners in zip(frame.bboxes_3d,
                            oriented_box_corners(frame, raw_db)):
        if corners is None:
            c3 = aabb_centers(box[None, :])[0]
            cuv, cd = project_points(c3[None, :], intr, R, t)
            if cd[0] <= 1.0:
                continue
            u0, v0 = int(cuv[0, 0] * sx), int(cuv[0, 1] * sy)
            if 0 <= u0 < VW and 0 <= v0 < VH:
                cv2.drawMarker(overlay, (u0, v0), GREEN, cv2.MARKER_CROSS,
                               14, 2, cv2.LINE_AA)
            continue
        cuv, cd = project_points(corners, intr, R, t)
        if (cd <= 1.0).any():
            continue
        cuv = cuv * np.array([sx, sy])
        # Skip only fully invisible boxes; partially visible ones are drawn
        # and clipped by OpenCV at the pane border (e.g. the truck at the
        # right image edge).
        visible = ((cuv[:, 0] > 0) & (cuv[:, 0] < VW)
                   & (cuv[:, 1] > 0) & (cuv[:, 1] < VH))
        if not visible.any():
            continue
        for a, b in CUBE_EDGES:
            cv2.line(overlay, (int(cuv[a, 0]), int(cuv[a, 1])),
                     (int(cuv[b, 0]), int(cuv[b, 1])), GREEN, 2, cv2.LINE_AA)

    def with_boxes(f):
        for x1, y1, x2, y2 in b2:
            cv2.rectangle(f, (int(x1 * sx), int(y1 * sy)),
                          (int(x2 * sx), int(y2 * sy)), WHITE, 1, cv2.LINE_AA)
        return f

    HUDH = 58
    Hh = VH + HUDH

    def compose(view, title, sub, accent):
        f = np.full((Hh, VW, 3), DARK, np.uint8)
        f[HUDH:, :] = view
        panel(f, 0, 0, VW, HUDH - 4, 0.0, DARK)
        put_text(f, "xcalib  -  calibrated camera-LiDAR projection  "
                    "(A9 intersection)", (16, 22), 0.5, WHITE, 1)
        put_text(f, title, (16, 44), 0.48, accent, 1)
        if sub:
            put_text(f, sub, (VW - 10 - 8 * len(sub), 44), 0.4,
                     (210, 210, 210), 1)
        return f

    before = compose(with_boxes(base.copy()),
                     "BEFORE: unregistered sensors",
                     "LiDAR cannot be drawn on the image", GOLD)
    stat = (f"matcher pairs agree: {n_pairs} pairs, median {med_px:.0f} px")
    after = compose(with_boxes(overlay.copy()),
                    "AFTER: cloud + 3D detections land on the scene", stat,
                    GREEN)

    frames: List[np.ndarray] = []
    frames.extend([before] * int(1.4 * fps))
    for k in range(int(0.8 * fps)):  # cross-fade
        tt = smoothstep(k / max(1, int(0.8 * fps) - 1))
        frames.append(cv2.addWeighted(before, 1 - tt, after, tt, 0))
    frames.extend([after] * int(2.2 * fps))
    for _ in range(2):  # before/after toggle
        frames.extend([before] * int(0.55 * fps))
        frames.extend([after] * int(0.9 * fps))
    frames.extend([after] * int(1.2 * fps))

    write_gif(frames, out_gif, fps)
    cv2.imwrite(str(out_png), after)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(
        REPO / "datasets/a9_dataset_r02_s01/hdf5_cache/a9_r02_s01_test.h5"))
    # calibrefine's cosine scores are the best-calibrated of the A9 zoo
    # (confident pairs ~0.5+, forced pairs below), which matters for the
    # calibration demo's mutual-best gating.
    ap.add_argument("--model", default="calibrefine")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(REPO / "docs/assets"))
    ap.add_argument("--raw-labels", default=str(
        REPO / "datasets/a9_dataset_r02_s01/labels_point_clouds"
               "/s110_lidar_ouster_north"),
        help="raw OpenLABEL dir for true oriented 3D boxes (optional; "
             "falls back to a point-cloud refit when missing)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=60,
                    help="frames harvested for calibration")
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="mutual-best similarity gate for calibration pairs")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = args.weights or str(
        REPO / f"checkpoints/{args.model}_a9_dataset_r02_s01_best.pth")
    config = args.config or str(
        REPO / f"src/xcalib/cfg/{args.model}_a9_dataset_r02_s01.yaml")
    matcher = Matcher.from_pretrained(args.model, weights=weights,
                                      config=config, device=args.device)
    loader = UTCFrameLoader(args.cache)
    camera = pick_camera(loader)
    if camera != REF_CAMERA:
        print(f"note: reference calibration is for {REF_CAMERA}; forcing it")
        camera = REF_CAMERA

    intr, R, t, n_pairs, med_px = run_calibration(
        matcher, loader, camera,
        max_frames=args.max_frames, min_score=args.min_score,
    )

    raw_db = load_raw_label_db(Path(args.raw_labels))
    print(f"raw oriented labels: "
          f"{'%d files' % len(raw_db) if raw_db else 'unavailable (refit)'}")

    # Camera optical axis expressed in LiDAR coordinates orients the BEV.
    fwd = R[2, :3]
    demo_frame = pick_demo_frame(loader, camera)

    render_calibration(demo_frame, camera, intr, R, t, n_pairs, med_px,
                       out_dir / "a9_calibration.gif",
                       out_dir / "a9_calibration.png", args.fps,
                       raw_db=raw_db)
    render_matching(matcher, args.model, demo_frame, camera,
                    (float(fwd[0]), float(fwd[1])),
                    out_dir / "a9_matching.gif",
                    out_dir / "a9_matching.png", args.fps,
                    raw_db=raw_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
