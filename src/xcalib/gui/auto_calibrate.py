"""A scene's camera intrinsics, and the checks a calibration has to pass to be saved unasked.

The pair-fitted homography reads the camera matrix of the rectified images
through :func:`scene_intrinsics`. The pairing window's Calibrate and
Recalibrate save a solve without asking (``xcalib-review --accept-calibration``)
only when :func:`solve_problems` finds nothing against it; otherwise a person
decides. Nothing is written here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# A solve is saved without asking only when
MIN_INLIERS = 30               # enough pairs support it,
MIN_INLIER_FRACTION = 1 / 3    # they are a clear share of the pairs it was solved from,
MAX_ERROR_PX = 10.0            # they fit closely, and
MAX_CAMERA_DISTANCE_M = 100.0  # the camera ends up at the LiDAR's site.


def scene_intrinsics(scene):
    """Intrinsics saved with an extrinsic, else the ones kept from the import."""
    from ..protocol import CameraIntrinsics

    for folder in ("camera", "intrinsic"):
        path = Path(scene.root) / "calib" / folder / f"{scene.camera}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("intrinsic"):
                matrix = np.asarray(data["intrinsic"], dtype=float).reshape(3, 3)
                return CameraIntrinsics.from_matrix(matrix, data.get("distortion"))
    raise ValueError(f"the scene has no camera intrinsics (calib/intrinsic/{scene.camera}.json)")


def solve_problems(result, pairs, what="pairs"):
    """Why a successful solve should not be saved without a person judging it.

    Empty when enough of the ``pairs`` support it, they fit closely and the
    camera ends up at the LiDAR's site.
    """
    distance = float(np.linalg.norm(result.rotation.T @ np.asarray(result.translation).reshape(3)))
    problems = []
    if result.n_inliers < max(MIN_INLIERS, MIN_INLIER_FRACTION * pairs):
        problems.append(f"only {result.n_inliers} of {pairs} {what} agree with it")
    if result.reproj_error_px >= MAX_ERROR_PX:
        problems.append(f"its error is {result.reproj_error_px:.0f} px")
    if distance > MAX_CAMERA_DISTANCE_M:
        problems.append(f"it puts the camera {distance:.0f} m from the LiDAR")
    return problems
