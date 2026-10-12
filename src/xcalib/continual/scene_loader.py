"""Package-native frame loader for continual refinement.

``run_continual_refine`` needs a ``frame_loader`` that reads one reviewed frame of a
scene in a SUSTechPOINTS-style layout (``camera/``, ``lidar/``, ``label/``,
``label_fusion/``) into an :class:`XCalibFrameInput`. This module provides the
package-native default, built on :mod:`xcalib.gui.scene`; a caller may inject its
own loader instead.

Fingerprint (v2): the immutable image + point cloud are identified by ``(mtime, size)``
via one ``stat()`` -- NOT by hashing their multi-MB bytes -- and only the box geometry
(b2d/b3d) is hashed. The rule is deterministic (the fixed ``"xcalib-feat-v2"`` prefix,
the file identity, the geometry hash), so any other front end that follows it computes
the same value and a corpus stays coherent across front ends without requiring
byte-identical image/point decoders. Do NOT reintroduce content hashing -- it was the
frame-switch / Apply-all slowness.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Sequence

import numpy as np

from ..gui.scene import Scene, load_points_xyz
from .refine_model import XCalibFrameInput
from .state import FeatureWork
from .store import CorrectionRecord
from .train import (
    ContinualRefineUnavailable,
    _FrameInputFingerprintMismatch,
    _candidate_keys,
    _rect_geometry,
)

__all__ = ["scene_frame_loader", "frame_input_fingerprint"]


def _file_identity(path: str | None) -> str:
    """Cheap identity for an immutable capture file: ``"<mtime_ns>:<size>"``.

    The image + point cloud never change during annotation, so the fingerprint
    identifies them by a single ``stat()`` instead of SHA-256'ing multi-MB bytes. Box
    geometry -- what the crops actually depend on -- is hashed in full by the caller.
    """
    if not path or not os.path.isfile(path):
        return "missing"
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return f"{st.st_mtime_ns}:{st.st_size}"


def _input_fingerprint(image_path, points_path, bboxes_2d, bboxes_3d) -> str:
    """Feature identity: capture files by (mtime,size) + bbox geometry (v2).

    A sha256 over the ``"xcalib-feat-v2"`` prefix; then, for the image and the points,
    the ascii name followed by the file identity (:func:`_file_identity`); then, for
    ``bboxes_2d`` and ``bboxes_3d``, the ascii name, dtype, JSON shape and C-order
    bytes of the contiguous array. Any other front end that hashes the same fields in
    the same order gets the same value; a different rule would split the corpus.
    """
    digest = hashlib.sha256()
    digest.update(b"xcalib-feat-v2")
    for name, path in (("image", image_path), ("points", points_path)):
        digest.update(name.encode("ascii"))
        digest.update(_file_identity(path).encode("ascii"))
    for name, value in (("bboxes_2d", bboxes_2d), ("bboxes_3d", bboxes_3d)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _lidar_review_key(box: dict) -> str:
    """Stable review identity for a 3D box.

    A labelled box keys by its shared LiDAR id (``"id:<obj_id>"``); an anonymous box
    keys by a canonical hash of its ``psr`` (``"psr:<sha256>"``) so a correction that
    later assigns it an id can still be matched back to the issued key.
    """
    lidar_id = str(box.get("obj_id", "") or "").strip()
    if lidar_id:
        return "id:" + lidar_id
    psr = box.get("psr") or {}
    encoded = json.dumps(
        psr, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "psr:" + hashlib.sha256(encoded).hexdigest()


def _frame_arrays(scene: Scene, frame: str):
    """(image RGB, points xyz, rects, boxes, b2d[K,4], b3d[M,6], image_path,
    lidar_path) for a frame. Raises when the frame lacks images/points/labels."""
    import cv2  # noqa: PLC0415 -- cv2 is a package dependency, imported lazily

    rects = scene.load_rects(frame)
    boxes = scene.load_boxes(frame)
    if not rects or not boxes:
        raise ContinualRefineUnavailable(
            f"frame {frame} has no 2D rects and/or 3D boxes to refine"
        )
    image_path = scene.image_path(frame)
    if image_path is None:
        raise ContinualRefineUnavailable(f"no camera image for frame {frame}")
    lidar_path = scene.lidar_path(frame)
    if lidar_path is None:
        raise ContinualRefineUnavailable(f"no point cloud for frame {frame}")
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    points = load_points_xyz(lidar_path).astype(np.float32)
    b2d = Scene.rects_xyxy(rects)
    b3d = Scene.boxes_extents(boxes)
    return image, points, rects, boxes, b2d, b3d, image_path, lidar_path


def frame_input_fingerprint(scene_dir: str, camera: str, frame: str) -> str:
    """The v2 feature fingerprint for a scene frame, without decoding the image
    or point cloud -- one ``stat()`` per capture file + the box geometry.

    Computed by :func:`_input_fingerprint`; used by the corpus-write seam to stamp
    verified pairs with the same identity the refine loader recomputes. ``scene_dir``
    is the scene directory (``<data_root>/<scene>``)."""
    scene = Scene(scene_dir, camera)
    image_path = scene.image_path(frame)
    lidar_path = scene.lidar_path(frame)
    b2d = Scene.rects_xyxy(scene.load_rects(frame))
    b3d = Scene.boxes_extents(scene.load_boxes(frame))
    return _input_fingerprint(image_path, lidar_path, b2d, b3d)


def scene_frame_loader(
    data_root: str,
    scene: str,
    camera: str,
    work: FeatureWork,
    records: Sequence[CorrectionRecord],
) -> XCalibFrameInput:
    """Read one reviewed frame into an ``XCalibFrameInput`` (package-native).

    Behaviour:
    - stable human ``rect_keys`` from the ledger's ``rect_json`` (positional fallback),
    - ``lidar_keys`` via :func:`_lidar_review_key`, preserving the issued ``psr:``
      identity when the recorded candidates used it (a correction may since have
      assigned the box an id),
    - a stale guard: the recomputed v2 fingerprint must equal ``work.input_fingerprint``,
      else the frame drifted from its reviewed snapshot and is skipped.
    """
    scene_obj = Scene(os.path.join(str(data_root), scene), camera)
    (image, points, rects, boxes, b2d, b3d,
     image_path, lidar_path) = _frame_arrays(scene_obj, work.frame)

    actual_fingerprint = _input_fingerprint(image_path, lidar_path, b2d, b3d)
    if actual_fingerprint != work.input_fingerprint:
        raise _FrameInputFingerprintMismatch(
            f"frame {work.frame} input changed after review; "
            "apply a fresh review snapshot before refining"
        )

    sample_by_rect: dict[tuple[float, ...], str] = {}
    for row in records:
        try:
            values = json.loads(row.rect_json)
            key = tuple(round(float(value), 4) for value in values)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ContinualRefineUnavailable(
                f"invalid rect_json for sample {row.sample_key}"
            ) from exc
        sample_by_rect[key] = row.sample_key
    rect_keys = tuple(
        sample_by_rect.get(
            tuple(round(value, 4) for value in _rect_geometry(rect)),
            f"rect:{index}",
        )
        for index, rect in enumerate(rects)
    )
    recorded_lidar_keys = {
        key for record in records for key in _candidate_keys(record)
    }
    lidar_keys = []
    for box in boxes:
        geometry_key = _lidar_review_key({"psr": box.get("psr") or {}})
        lidar_keys.append(
            geometry_key
            if geometry_key in recorded_lidar_keys
            else _lidar_review_key(box)
        )
    return XCalibFrameInput(
        image=image,
        point_cloud=points,
        bboxes_2d=b2d,
        bboxes_3d=b3d,
        rect_keys=rect_keys,
        lidar_keys=tuple(lidar_keys),
        input_fingerprint=work.input_fingerprint,
    )
