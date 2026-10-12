"""
Frame capture for human-in-the-loop correction — the deployed-site escape
hatch.

A partner integration calls the xcalib API directly. While drift stays small,
:class:`~xcalib.oneshot.session.OneShotSession` self-corrects. Once it does
not — weather moved the rig, the matcher stops finding confident pairs, the
matrix stops projecting where things are — somebody has to LOOK at the failing
frames. :class:`SceneRecorder` saves exactly what ``match()``/``observe()``
consumed (image, point cloud, both box sets) as a scene directory in a
SUSTechPOINTS-style layout, so the captured data opens directly in
``xcalib-review`` for correction, and the corrected labels feed
``OneShotSession.bank_pairs`` + ``adapt`` / ``Matcher.train`` afterwards.

Typical integration::

    rec = SceneRecorder("/data/capture/site1", camera="cam_front")
    rec.save_calibration(calib)                    # once per site

    session = matcher.oneshot(K, recorder=rec)     # auto-captures lost frames
    for image, pc, b2, b3 in stream:
        session.observe(image, pc, b2, b3)

    # or capture manually on any trigger you like:
    if my_own_drift_alarm(result):
        rec.save_frame(image, pc, b2, b3, ids_3d=tracker_ids)

Then, offline::

    xcalib-review --scene /data/capture/site1 --camera cam_front
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .oneshot.calibration import CalibrationResult
    from .oneshot.session import ObserveReport

__all__ = ["SceneRecorder", "should_record"]


def should_record(report: "ObserveReport") -> bool:
    """The auto-capture trigger used by ``oneshot(..., recorder=...)``.

    Record when the frame HAD detections on both sides yet the matcher found
    NO confident pair — the model is lost on this frame, which is precisely
    the data a human needs to see. (Partners with better site-specific alarms
    can ignore this and call :meth:`SceneRecorder.save_frame` themselves.)
    """
    return (report.n_detections_2d > 0
            and report.n_detections_3d > 0
            and report.n_confident_matches == 0)


class SceneRecorder:
    """Write matcher inputs as a SUSTechPOINTS-style scene directory.

    Layout produced (readable by ``xcalib.gui.scene.Scene`` and
    ``xcalib-review``)::

        <root>/camera/<camera>/<frame>.jpg
        <root>/lidar/<frame>.pcd                (binary; or .npy)
        <root>/label/<frame>.json               3D boxes with obj_id
        <root>/label_fusion/camera/<camera>/<frame>.json
                                                2D rects (annotator="xcalib")
        <root>/calib/camera/<camera>.json       K + extrinsic (save_calibration)

    Identity model: ``ids_3d`` become the LiDAR boxes' canonical ``obj_id``
    (1-based enumeration when omitted); ``ids_2d`` are the camera's OWN track
    ids and land in ``obj_attr`` as ``"t<id>"`` — a rect's ``obj_id`` stays
    EMPTY until a human (or an accepted match) fills it, so captured model
    guesses can never masquerade as verified pairs.
    """

    def __init__(self, root: Union[str, os.PathLike], camera: str = "camera_0",
                 cloud_format: str = "pcd"):
        if cloud_format not in ("pcd", "npy"):
            raise ValueError(f"cloud_format must be 'pcd' or 'npy', got {cloud_format!r}")
        self.root = os.path.abspath(os.fspath(root))
        self.camera = camera
        self.cloud_format = cloud_format
        self._counter: Optional[int] = None

    # ---- calibration ----

    def save_calibration(self, calibration: "CalibrationResult") -> str:
        """Write the site's K + extrinsic once (enables the geometric gate and
        matrix refinement on the captured scene)."""
        if not calibration.success or calibration.rotation is None:
            raise ValueError("save_calibration needs a SUCCESSFUL calibration")
        E = np.eye(4)
        E[:3, :3] = np.asarray(calibration.rotation, dtype=float)
        E[:3, 3] = np.asarray(calibration.translation, dtype=float).reshape(3)
        payload = {
            "intrinsic": [float(v) for v in calibration.intrinsics.K.reshape(-1)],
            "extrinsic": [float(v) for v in E.reshape(-1)],
        }
        path = os.path.join(self.root, "calib", "camera", f"{self.camera}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    # ---- frames ----

    def save_frame(
        self,
        image: np.ndarray,
        point_cloud: np.ndarray,
        bboxes_2d: np.ndarray,
        bboxes_3d: np.ndarray,
        *,
        frame: Optional[str] = None,
        ids_3d: Optional[Sequence] = None,
        types_3d: Optional[Sequence[str]] = None,
        ids_2d: Optional[Sequence] = None,
        types_2d: Optional[Sequence[str]] = None,
    ) -> str:
        """Persist one frame's matcher inputs; returns the frame id.

        ``bboxes_2d`` [K,4] (x1,y1,x2,y2); ``bboxes_3d`` [M,6] in either
        protocol form (extents or center+size). ``frame`` defaults to a
        zero-padded counter continuing from what is already on disk.
        """
        import cv2  # noqa: PLC0415 — package dependency

        b2 = np.asarray(bboxes_2d, dtype=np.float64).reshape(-1, 4)
        b3 = np.asarray(bboxes_3d, dtype=np.float64).reshape(-1, 6)
        frame = frame or self._next_frame_id()

        img_dir = os.path.join(self.root, "camera", self.camera)
        os.makedirs(img_dir, exist_ok=True)
        ok = cv2.imwrite(os.path.join(img_dir, f"{frame}.jpg"),
                         cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        if not ok:
            raise IOError(f"could not write {img_dir}/{frame}.jpg")

        lidar_dir = os.path.join(self.root, "lidar")
        os.makedirs(lidar_dir, exist_ok=True)
        # keep intensity (4th column) when the source cloud has it — the
        # matcher only consumes xyz, but the review GUI colors points by it
        pts = np.asarray(point_cloud, dtype=np.float32)
        pts = pts[:, :4] if pts.shape[1] >= 4 else pts[:, :3]
        if self.cloud_format == "npy":
            np.save(os.path.join(lidar_dir, f"{frame}.npy"), pts)
        else:
            _write_pcd(os.path.join(lidar_dir, f"{frame}.pcd"), pts)

        # 3D boxes -> psr labels with canonical obj_id (lidar-master)
        label_dir = os.path.join(self.root, "label")
        os.makedirs(label_dir, exist_ok=True)
        objs3 = []
        for j, box in enumerate(b3):
            first, second = box[:3], box[3:6]
            if np.all(second >= first):                     # extents form
                center, size = (first + second) / 2.0, second - first
            else:                                           # center+size form
                center, size = first, second
            oid = str(ids_3d[j]) if ids_3d is not None else str(j + 1)
            objs3.append({
                "obj_id": oid,
                "obj_type": str(types_3d[j]) if types_3d is not None else "",
                "psr": {
                    "position": {"x": float(center[0]), "y": float(center[1]),
                                 "z": float(center[2])},
                    "scale": {"x": float(size[0]), "y": float(size[1]),
                              "z": float(size[2])},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            })
        with open(os.path.join(label_dir, f"{frame}.json"), "w", encoding="utf-8") as f:
            json.dump({"objs": objs3}, f, indent=1)

        # 2D rects: obj_id EMPTY (a human decision, not the model's), the
        # partner's own tracker id in obj_attr "tN", machine mark "xcalib"
        fusion_dir = os.path.join(self.root, "label_fusion", "camera", self.camera)
        os.makedirs(fusion_dir, exist_ok=True)
        objs2 = []
        for i, r in enumerate(b2):
            o = {
                "rect": {"x1": float(r[0]), "y1": float(r[1]),
                         "x2": float(r[2]), "y2": float(r[3])},
                "obj_id": "",
                "obj_type": str(types_2d[i]) if types_2d is not None else "",
                "annotator": "xcalib",
            }
            if ids_2d is not None:
                o["obj_attr"] = f"t{ids_2d[i]}"
            objs2.append(o)
        with open(os.path.join(fusion_dir, f"{frame}.json"), "w", encoding="utf-8") as f:
            json.dump({"objs": objs2}, f, indent=1)
        return frame

    def frames(self) -> list:
        d = os.path.join(self.root, "lidar")
        if not os.path.isdir(d):
            return []
        return sorted({os.path.splitext(f)[0] for f in os.listdir(d)
                       if os.path.splitext(f)[1].lower() in (".pcd", ".bin", ".npy")})

    def _next_frame_id(self) -> str:
        if self._counter is None:
            existing = self.frames()
            self._counter = max((int(f) for f in existing if f.isdigit()), default=-1) + 1
        else:
            self._counter += 1
        return f"{self._counter:06d}"


def _write_pcd(path: str, pts: np.ndarray) -> None:
    """Binary PCD, float32 — ``x y z`` or ``x y z intensity`` depending on the
    column count. The format `gui.scene.load_points` reads."""
    pts = np.ascontiguousarray(pts.astype(np.float32))
    cols = pts.shape[1]
    fields = "x y z intensity" if cols >= 4 else "x y z"
    n = cols if cols <= 4 else 4
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {' '.join(['4'] * n)}\n"
        f"TYPE {' '.join(['F'] * n)}\n"
        f"COUNT {' '.join(['1'] * n)}\n"
        f"WIDTH {len(pts)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(pts)}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts[:, :n].tobytes())
