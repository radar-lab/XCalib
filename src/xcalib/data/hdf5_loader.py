"""
Read-only UTC HDF5 frame loader.

This is a minimal slice of `A9EvaluationDataset` tailored for UTC3 / UTC4
HDF5 caches. It is read-only and intentionally has no torch.utils.data
dependency — the partner sees plain Python iteration.

The lab's HDF5 layout (UTC variant) looks like:
    /images/<camera_name>/data             - JPEG bytes [N_frames]
    /point_clouds/<lidar_name>/<frame_key>/xyz   - [P, 3] float32
    /labels/<sensor>/<frame_key>/
        num_camera_detections                  - scalar int
        num_lidar_detections                   - scalar int
        camera_bbox_2d                         - [num_camera, 4]
        camera_names                           - [num_camera] bytes (per-detection source camera)
        lidar_bbox_3d                          - [num_lidar, 6] (xmin,ymin,zmin,xmax,ymax,zmax)
        match_matrix                           - [num_camera, num_lidar] bool

`/calibration` is absent for UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import h5py
import numpy as np
from loguru import logger


@dataclass
class UTCFrame:
    """Raw frame contents (before any cropping)."""
    frame_key: str
    image: np.ndarray              # [H, W, 3] uint8 RGB (leading camera)
    point_cloud: np.ndarray        # [P, 3] float32
    bboxes_2d: np.ndarray          # [K, 4] (x1,y1,x2,y2)
    bboxes_3d: np.ndarray          # [M, 6] (xmin,ymin,zmin,xmax,ymax,zmax)
    match_matrix: np.ndarray       # [K, M] bool
    camera_name: str               # which camera `image` came from
    # Multi-camera caches (e.g. A9 south1 + south2): every camera referenced
    # by this frame's detections, plus the per-detection source camera. For
    # single-camera caches (UTC) this is just {camera_name: image} / [K] of
    # camera_name, so single-image callers keep working unchanged.
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    camera_per_det: Optional[np.ndarray] = None  # [K] str
    # Per-camera calibration, present when the cache ships a `/calibration` group
    # (A9); empty for caches without it (UTC). `intrinsics[cam]` is a 3x3 pinhole
    # K; `extrinsics[cam]` is this frame's 4x4 lidar->camera pose (ground truth,
    # for validating a targetless solve).
    intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)
    extrinsics: Dict[str, np.ndarray] = field(default_factory=dict)

    def for_camera(
        self, camera: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(image, point_cloud, bboxes_2d, bboxes_3d)`` for one camera.

        Multi-camera caches (e.g. A9 ``south1`` + ``south2``) store *every*
        camera's 2D detections in a single frame, tagged by
        :attr:`camera_per_det`. Pairing the full ``bboxes_2d`` with one camera's
        image mixes two pixel coordinate systems (you'd match/plot two cameras at
        once); this keeps only ``camera``'s 2D boxes alongside that camera's
        image. The point cloud and 3D boxes are sensor-global and pass through
        unchanged. Raises ``KeyError`` if ``camera`` has no image in this frame.
        """
        if camera not in self.images:
            raise KeyError(
                f"camera {camera!r} not in frame {self.frame_key}; "
                f"available: {sorted(self.images)}"
            )
        if self.camera_per_det is None:
            mask = np.ones(len(self.bboxes_2d), dtype=bool)
        else:
            mask = np.asarray(self.camera_per_det).astype(str) == camera
        return self.images[camera], self.point_cloud, self.bboxes_2d[mask], self.bboxes_3d


def _read_calibration(
    f: "h5py.File", frame_key: str
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Read per-camera ``(intrinsics 3x3, this-frame extrinsics 4x4)`` from a
    ``/calibration`` group. Returns ``({}, {})`` for caches without one (UTC).

    Layout (A9):
        /calibration/<camera>/intrinsics              - [3, 3]
        /calibration/<camera>/extrinsics/<frame_key>  - [4, 4]  (per frame)
    """
    intr: Dict[str, np.ndarray] = {}
    extr: Dict[str, np.ndarray] = {}
    grp = f.get("calibration")
    if grp is None:
        return intr, extr

    # Calibration keys can drop the sensor prefix (A9: `camera_basler_south1_8mm`)
    # while the image stream keeps it (`s110_camera_basler_south1_8mm`). Normalise
    # to the image-stream name so `frame.intrinsics[cam]` keys like `frame.images`.
    image_cams = list(f["images"].keys()) if "images" in f else []

    def canonical(cal_cam: str) -> str:
        for ic in image_cams:
            if ic == cal_cam or ic.endswith(cal_cam):
                return ic
        return cal_cam

    for cam in grp.keys():
        key = canonical(cam)
        cam_grp = grp[cam]
        if "intrinsics" in cam_grp:
            intr[key] = np.asarray(cam_grp["intrinsics"][()], dtype=np.float64)
        ex = cam_grp.get("extrinsics")
        if isinstance(ex, h5py.Group):
            if frame_key in ex:
                extr[key] = np.asarray(ex[frame_key][()], dtype=np.float64)
        elif ex is not None:  # a single [4, 4] for the whole sequence
            extr[key] = np.asarray(ex[()], dtype=np.float64)
    return intr, extr


class UTCFrameLoader:
    """Iterates over frames of a UTC HDF5 cache without touching torch."""

    def __init__(self, hdf5_path: str | Path):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self.hdf5_path}")

        self._file: Optional[h5py.File] = None
        with h5py.File(self.hdf5_path, "r") as f:
            self.camera_names: List[str] = list(f["images"].keys())
            self.lidar_names: List[str] = list(f["point_clouds"].keys())
            label_sensors = list(f["labels"].keys())
            if not label_sensors:
                self.label_sensor = None
                self.frame_keys: List[str] = []
            else:
                self.label_sensor = label_sensors[0]
                self.frame_keys = sorted(f["labels"][self.label_sensor].keys())

        logger.info(
            f"UTCFrameLoader: {self.hdf5_path.name} | "
            f"{len(self.frame_keys)} frames | cameras={self.camera_names} | "
            f"lidars={self.lidar_names}"
        )

    # ------------------------------------------------------------------
    # context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "UTCFrameLoader":
        self._file = h5py.File(self.hdf5_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    # ------------------------------------------------------------------
    # iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame_keys)

    def __iter__(self) -> Iterator[UTCFrame]:
        for key in self.frame_keys:
            frame = self.get_frame(key)
            if frame is not None:
                yield frame

    # ------------------------------------------------------------------
    # single-frame access
    # ------------------------------------------------------------------

    def get_frame(self, frame_key: str) -> Optional[UTCFrame]:
        if self.label_sensor is None:
            return None
        f = self.file
        labels_grp = f["labels"][self.label_sensor].get(frame_key)
        if labels_grp is None:
            return None

        try:
            num_camera = int(labels_grp["num_camera_detections"][()])
            num_lidar = int(labels_grp["num_lidar_detections"][()])
        except KeyError:
            return None

        if num_camera == 0 or num_lidar == 0:
            return None

        # 2D + 3D bboxes
        bbox_2d_all = labels_grp["camera_bbox_2d"][:num_camera].astype(np.float32)
        lidar_bbox_3d = labels_grp["lidar_bbox_3d"][:num_lidar].astype(np.float32)
        match_matrix = labels_grp["match_matrix"][:num_camera, :num_lidar].astype(bool)

        # Per-detection source camera (multi-camera caches such as A9 store
        # detections from several cameras in one frame; UTC has one camera).
        if "camera_names" in labels_grp:
            cam_per_det = labels_grp["camera_names"][:num_camera].astype(str)
            if len(cam_per_det) == 0:
                cam_per_det = np.array([self.camera_names[0]] * num_camera)
        else:
            cam_per_det = np.array([self.camera_names[0]] * num_camera)
        camera_name = str(cam_per_det[0])

        # Resolve frame index for the image stream
        try:
            frame_idx_int = int(frame_key)
        except ValueError:
            # If frame keys are not numeric, fall back to position in sorted list.
            frame_idx_int = self.frame_keys.index(frame_key)

        # Decode every camera referenced by this frame's detections (once).
        images: Dict[str, np.ndarray] = {}
        for cam in dict.fromkeys(str(c) for c in cam_per_det):
            if cam not in self.camera_names:
                logger.warning(f"Frame {frame_key}: unknown camera {cam!r}; skipped")
                continue
            try:
                img_bytes = f["images"][cam]["data"][frame_idx_int]
                decoded = cv2.imdecode(
                    np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if decoded is None:
                    continue
                images[cam] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"Failed to decode {cam} image for frame {frame_key}: {e}")

        img = images.get(camera_name)
        if img is None:
            return None

        try:
            lidar_name = self.lidar_names[0]
            pcd_grp = f["point_clouds"][lidar_name][frame_key]
            xyz = pcd_grp["xyz"][:].astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to load point cloud for frame {frame_key}: {e}")
            return None

        intrinsics, extrinsics = _read_calibration(f, str(frame_key))
        return UTCFrame(
            frame_key=str(frame_key),
            image=img,
            point_cloud=xyz,
            bboxes_2d=bbox_2d_all,
            bboxes_3d=lidar_bbox_3d,
            match_matrix=match_matrix,
            camera_name=camera_name,
            images=images,
            camera_per_det=cam_per_det,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
