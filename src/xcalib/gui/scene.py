"""
SUSTechPOINTS-style scene IO + review assembly for the standalone review GUI.

The GUI reads/writes a SUSTechPOINTS-style on-disk layout:

    <scene>/
        camera/<camera>/<frame>.jpg|png          images
        lidar/<frame>.pcd|.bin|.npy              point clouds
        label/<frame>.json                       3D boxes  [{obj_id, obj_type, psr}]
        label_fusion/camera/<camera>/<frame>.json
                                                 2D rects  {objs: [{rect, obj_id,
                                                 obj_attr "tN", obj_type, no_match}]}
        calib/camera/<camera>.json               {intrinsic [9], extrinsic [16]}
        calib/xcalib/<camera>_model/<base>.pth   trained matcher (save_pretrained)

Identity model: ``obj_attr "tN"`` is the camera's OWN
track; ``obj_id`` is the shared/canonical LiDAR id. Accepting a pair writes the
lidar id onto the rect's ``obj_id``; an explicit human rejection writes
``no_match: true`` (and clears the id) so no tool re-proposes a cancelled
pairing. Everything here is Qt-free and torch-light so it unit-tests headless.
"""

from __future__ import annotations

import glob
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..oneshot.calibration import CalibrationResult
from ..protocol import CameraIntrinsics

__all__ = ["Scene", "FrameReview", "TrackEntry", "match_review"]


# ---------------------------------------------------------------- geometry --

def _euler_xyz_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix for intrinsic XYZ euler angles (THREE.js default —
    the convention SUSTechPOINTS labels use)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mx @ my @ mz


def box3d_corners(psr: dict) -> np.ndarray:
    """8 corners [8,3] of a psr box (position/scale/rotation, lidar frame)."""
    p, s, r = psr["position"], psr.get("scale") or {}, psr.get("rotation") or {}
    dx = float(s.get("x", 0.0)) / 2.0
    dy = float(s.get("y", 0.0)) / 2.0
    dz = float(s.get("z", 0.0)) / 2.0
    corners = np.array([
        [x, y, z]
        for x in (-dx, dx)
        for y in (-dy, dy)
        for z in (-dz, dz)
    ])
    rot = _euler_xyz_matrix(float(r.get("x", 0.0)), float(r.get("y", 0.0)),
                            float(r.get("z", 0.0)))
    return corners @ rot.T + np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def box3d_to_extents(psr: dict) -> np.ndarray:
    """psr -> (xmin, ymin, zmin, xmax, ymax, zmax) — the rotation-aware AABB
    the pretrained matchers were trained on."""
    c = box3d_corners(psr)
    lo, hi = c.min(axis=0), c.max(axis=0)
    return np.concatenate((lo, hi)).astype(np.float32)


# ------------------------------------------------------------ point clouds --

def load_points(path: str) -> np.ndarray:
    """Read a point cloud file -> float32 [P, 3] or [P, 4].

    The 4th column is INTENSITY when the source carries one (KITTI .bin,
    4-column .npy, PCD with an intensity field) — the matcher consumes xyz
    only, but the review GUI colors the top-down view by it."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        for cols in (4, 5, 3):  # KITTI-style x,y,z,intensity first
            if raw.size % cols == 0:
                return np.ascontiguousarray(raw.reshape(-1, cols)[:, :min(cols, 4)])
        raise ValueError(f"cannot infer column count of {path}")
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        return np.ascontiguousarray(arr[:, :min(arr.shape[1], 4)])
    if ext == ".pcd":
        return _load_pcd(path)
    raise ValueError(f"unsupported point cloud format: {path}")


def load_points_xyz(path: str) -> np.ndarray:
    """Read a point cloud file -> float32 [P, 3] (x, y, z only)."""
    return np.ascontiguousarray(load_points(path)[:, :3])


def _load_pcd(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header: Dict[str, str] = {}
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            header[key.upper()] = value
            if key.upper() == "DATA":
                break
        fields = header["FIELDS"].split()
        sizes = list(map(int, header["SIZE"].split()))
        counts = list(map(int, header.get("COUNT", " ".join(["1"] * len(fields))).split()))
        points = int(header["POINTS"])

        if header["DATA"] == "ascii":
            data = np.loadtxt(io.StringIO(f.read().decode("ascii", errors="ignore")))
            data = data.reshape(points, -1)
        elif header["DATA"] == "binary":
            row = sum(s * c for s, c in zip(sizes, counts))
            if not all(s == 4 for s in sizes):
                raise ValueError(f"{path}: only 4-byte PCD fields supported")
            data = np.frombuffer(f.read(points * row), dtype=np.float32).reshape(points, row // 4)
        else:
            raise ValueError(f"{path}: unsupported PCD DATA={header['DATA']!r}")

        xyz_cols: Dict[str, int] = {}
        intensity_col = None
        offset = 0
        for name, count in zip(fields, counts):
            for _ in range(count):
                if name in ("x", "y", "z") and name not in xyz_cols:
                    xyz_cols[name] = offset
                elif name == "intensity" and intensity_col is None:
                    intensity_col = offset
                offset += 1
        if len(xyz_cols) != 3:
            raise ValueError(f"{path}: PCD lacks x/y/z fields")
        cols = [xyz_cols["x"], xyz_cols["y"], xyz_cols["z"]]
        if intensity_col is not None:
            cols.append(intensity_col)
        return np.ascontiguousarray(data[:, cols].astype(np.float32))


# ------------------------------------------------------------------- scene --

def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _objs_of(data) -> list:
    return data.get("objs", []) if isinstance(data, dict) else (data or [])


def _clean_id(v) -> str:
    return str(v or "").strip()


def track_of(rect_obj: dict) -> Optional[str]:
    """Camera track id carried in obj_attr as 't<track>' (None if absent)."""
    a = str(rect_obj.get("obj_attr", "") or "")
    return a[1:] if a.startswith("t") and a[1:].isdigit() else None


class Scene:
    """One SUSTechPOINTS-style scene directory + a chosen camera."""

    def __init__(self, root: str, camera: str):
        self.root = os.path.abspath(root)
        self.camera = camera
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"scene directory not found: {self.root}")

    # ---- discovery ----

    def frames(self) -> List[str]:
        """Frame ids, from the lidar dir (the navigable sequence)."""
        d = os.path.join(self.root, "lidar")
        if not os.path.isdir(d):
            return []
        return sorted({os.path.splitext(f)[0] for f in os.listdir(d)
                       if os.path.splitext(f)[1].lower() in (".pcd", ".bin", ".npy")})

    def image_path(self, frame: str) -> Optional[str]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(self.root, "camera", self.camera, frame + ext)
            if os.path.isfile(p):
                return p
        hits = glob.glob(os.path.join(self.root, "camera", self.camera, frame + ".*"))
        return hits[0] if hits else None

    def lidar_path(self, frame: str) -> Optional[str]:
        for ext in (".pcd", ".bin", ".npy"):
            p = os.path.join(self.root, "lidar", frame + ext)
            if os.path.isfile(p):
                return p
        return None

    def model_dir(self) -> Optional[Tuple[str, str]]:
        """(dir, base_model) of this camera's trained matcher, or None."""
        d = os.path.join(self.root, "calib", "xcalib", f"{self.camera}_model")
        pth = sorted(glob.glob(os.path.join(d, "*.pth")))
        if not pth:
            return None
        return d, os.path.basename(pth[0])[: -len(".pth")]

    # ---- labels ----

    def load_boxes(self, frame: str) -> List[dict]:
        """3D boxes [{obj_id, obj_type, psr}] (only entries with a psr)."""
        p = os.path.join(self.root, "label", frame + ".json")
        if not os.path.isfile(p):
            return []
        return [b for b in _objs_of(_read_json(p)) if b.get("psr")]

    def load_rects(self, frame: str) -> List[dict]:
        """2D rects [{rect, obj_id, obj_attr, obj_type, no_match, ...}]."""
        p = self._fusion_path(frame)
        if not os.path.isfile(p):
            return []
        return [o for o in _objs_of(_read_json(p)) if o.get("rect")]

    @staticmethod
    def rects_xyxy(rects: List[dict]) -> np.ndarray:
        return np.array([[r["rect"]["x1"], r["rect"]["y1"], r["rect"]["x2"], r["rect"]["y2"]]
                         for r in rects], dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def boxes_extents(boxes: List[dict]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 6), dtype=np.float32)
        return np.stack([box3d_to_extents(b["psr"]) for b in boxes]).astype(np.float32)

    # ---- calibration ----

    def calibration(self) -> Optional[CalibrationResult]:
        """The saved camera calibration as a ready-to-use CalibrationResult."""
        p = os.path.join(self.root, "calib", "camera", f"{self.camera}.json")
        if not os.path.isfile(p):
            return None
        cal = _read_json(p)
        K, E = cal.get("intrinsic"), cal.get("extrinsic")
        if not K or len(K) < 9 or not E or len(E) < 16:
            return None
        Km = np.asarray(K, dtype=float).reshape(3, 3)
        Em = np.asarray(E, dtype=float).reshape(4, 4)
        intr = CameraIntrinsics(fx=float(Km[0, 0]), fy=float(Km[1, 1]),
                                cx=float(Km[0, 2]), cy=float(Km[1, 2]))
        return CalibrationResult(
            success=True, intrinsics=intr,
            rotation=Em[:3, :3], translation=Em[:3, 3],
            projection=Km @ Em[:3, :4], accepted=True,
        )

    # ---- write-back ----

    def apply(self, frame: str, decisions: List[dict]) -> Dict[str, int]:
        """Write reviewed per-rect decisions into the fusion label.

        Each decision: ``{"rect": [x1,y1,x2,y2], "lidar_id": "<id>"}`` accepts a
        pairing (rect takes the lidar id, machine marks cleared), or
        ``{"rect": [...], "none": True}`` records a human rejection
        (``no_match: true``, id cleared). Rects are matched by rounded
        coordinates. Returns {"written": n, "marked": m}.
        """
        p = self._fusion_path(frame)
        if not os.path.isfile(p):
            return {"written": 0, "marked": 0}
        ann = _read_json(p)
        objs = _objs_of(ann)

        def rk(vals):
            return tuple(round(float(v), 1) for v in vals)

        written = marked = 0
        for d in decisions or []:
            want = rk(d["rect"])
            cam = next((o for o in objs if o.get("rect") and rk([
                o["rect"].get("x1"), o["rect"].get("y1"),
                o["rect"].get("x2"), o["rect"].get("y2")]) == want), None)
            if cam is None:
                continue
            if d.get("none"):
                if not cam.get("no_match") or _clean_id(cam.get("obj_id")):
                    cam["no_match"] = True
                    if _clean_id(cam.get("obj_id")):
                        cam["obj_id"] = ""
                    marked += 1
                continue
            lid = _clean_id(d.get("lidar_id"))
            if lid and (_clean_id(cam.get("obj_id")) != lid or cam.get("no_match")):
                cam["obj_id"] = lid
                cam.pop("annotator", None)
                cam.pop("no_match", None)
                written += 1
        if written or marked:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(ann, f, indent=1)
        return {"written": written, "marked": marked}

    def _fusion_path(self, frame: str) -> str:
        return os.path.join(self.root, "label_fusion", "camera", self.camera,
                            frame + ".json")


# ------------------------------------------------------------ review model --

@dataclass
class TrackEntry:
    """One camera rect's row in the review: its persisted state + the ranked
    candidate menu."""
    key: str                          # "tN" track, or "@i" for an un-tracked rect
    rect: List[float]                 # [x1, y1, x2, y2]
    applied_id: str = ""              # id already written to the labels
    rejected: bool = False            # persisted no_match mark
    suggestion: str = ""              # the gated optimal assignment's pick
    options: List[dict] = field(default_factory=list)
    # options: {lidar_id, lidar_type, score, proj_px (float|None), geo_ok (bool)}


@dataclass
class FrameReview:
    frame: str
    tracks: List[TrackEntry]
    boxes: List[dict]                 # 3D label objs (for the BEV pane)
    n_rects: int = 0
    n_boxes: int = 0


def match_review(matcher, scene: Scene, frame: str, threshold: float = 0.5,
                 image: Optional[np.ndarray] = None,
                 points: Optional[np.ndarray] = None) -> FrameReview:
    """Run the matcher on one frame and assemble the reviewable structure.

    Uses :meth:`Matcher.match` with ``one_to_one=True`` and the scene's saved
    calibration (geometric gate) when present. ``image``/``points`` may be
    passed in when the caller already loaded them (the GUI does, for display).
    """
    rects = scene.load_rects(frame)
    boxes = scene.load_boxes(frame)
    review = FrameReview(frame=frame, tracks=[], boxes=boxes,
                         n_rects=len(rects), n_boxes=len(boxes))
    if not rects or not boxes:
        return review

    if image is None:
        import cv2  # noqa: PLC0415 — cv2 is a package dependency

        path = scene.image_path(frame)
        if path is None:
            return review
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if points is None:
        lp = scene.lidar_path(frame)
        if lp is None:
            return review
        points = load_points_xyz(lp)

    b2d = Scene.rects_xyxy(rects)
    b3d = Scene.boxes_extents(boxes)
    calib = scene.calibration()
    res = matcher.match(image, points, b2d, b3d, top_k=1, match_threshold=0.0,
                        one_to_one=True, calibration=calib, validate="off")

    k2 = np.asarray(res.kept_2d_indices).astype(int)
    k3 = np.asarray(res.kept_3d_indices).astype(int)
    sim = np.asarray(res.similarity, dtype=float)
    rows = k2 if k2.size else np.arange(sim.shape[0] if sim.ndim == 2 else 0)
    cols = k3 if k3.size else np.arange(sim.shape[1] if sim.ndim == 2 else 0)

    # the gated optimal assignment, thresholded, in original indices
    suggestion: Dict[int, str] = {}
    for i, j, s in res.assignment:
        if s >= threshold:
            lid = _clean_id(boxes[j].get("obj_id"))
            if lid:
                suggestion[i] = lid

    scored_box_indices = {int(bj) for bj in cols}
    for r_pos, ri in enumerate(rows):
        rect_o = rects[int(ri)]
        key = track_of(rect_o) or f"@{int(ri)}"
        opts = []
        for c_pos, bj in enumerate(cols):
            lid = _clean_id(boxes[int(bj)].get("obj_id"))
            if not lid:
                continue
            o = {"lidar_id": lid,
                 "lidar_type": boxes[int(bj)].get("obj_type", ""),
                 "score": round(float(sim[r_pos, c_pos]), 4),
                 "proj_px": None, "geo_ok": True}
            if res.proj_px is not None:
                d = float(res.proj_px[r_pos, c_pos])
                o["proj_px"] = round(d, 1) if np.isfinite(d) else None
                o["geo_ok"] = bool(res.geo_compatible[r_pos, c_pos])
            opts.append(o)
        # The matcher drops a 3D box whose expanded crop has no LiDAR points
        # (data/crops.py), so it never appears in ``cols``. The saved annotation is
        # still a real box, though, and must stay selectable in review --
        # pedestrians with a sparse/absent return are a common example, and dropping
        # them leaves that region with no correspondence for calibration or refine.
        # Keep such id'd boxes as unscored manual candidates; score 0 keeps them out
        # of the automatic assignment (built from res.assignment over kept boxes only)
        # while a reviewer, or adjacent-frame inheritance, can still pick the saved id.
        for bj, box in enumerate(boxes):
            if bj in scored_box_indices:
                continue
            lid = _clean_id(box.get("obj_id"))
            if not lid:
                continue
            opts.append({"lidar_id": lid, "lidar_type": box.get("obj_type", ""),
                         "score": 0.0, "proj_px": None, "geo_ok": True})
        opts.sort(key=lambda o: (not o["geo_ok"], -o["score"]))
        r = rect_o["rect"]
        review.tracks.append(TrackEntry(
            key=key,
            rect=[float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
            applied_id=_clean_id(rect_o.get("obj_id")),
            rejected=bool(rect_o.get("no_match")),
            suggestion=suggestion.get(int(ri), ""),
            options=opts,
        ))
    review.tracks.sort(key=lambda t: (t.key.startswith("@"),
                                      int(t.key.lstrip("@")) if t.key.lstrip("@").isdigit() else 0))
    return review
