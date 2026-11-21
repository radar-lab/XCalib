"""One-time repair of a raw delivery whose camera images are still distorted.

The tutorial notebook expects rectified images and camera detections made on
them. This writes a new raw folder with undistorted images and the camera
boxes moved to match; the input folder is not changed. LiDAR files do not
depend on the camera and are linked (or copied) as they are.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

IMAGES = {".jpg", ".jpeg", ".png"}
BOX = ("bb_left", "bb_top", "bb_width", "bb_height")
ITERATIONS = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10)


def undistort_box(box, K, distortion, new_K, size):
    """Move a left, top, width, height box into the undistorted image.

    Each side is placed at the image of its midpoint. Enclosing the whole
    curved outline instead adds a wide empty margin to large boxes near the
    image border. Returns None when nothing is left inside the image.
    """
    left, top, width, height = box
    cx, cy = left + width / 2, top + height / 2
    sides = np.array([[left, cy], [left + width, cy], [cx, top], [cx, top + height]])
    # OpenCV's default five iterations are several pixels off near the border.
    ideal = cv2.undistortPointsIter(sides.reshape(-1, 1, 2), K, distortion, None, None, ITERATIONS)
    # Outside the calibrated field the inversion does not converge and returns
    # a wrong point; distorting it again exposes that.
    back = cv2.projectPoints(cv2.convertPointsToHomogeneous(ideal), np.zeros(3), np.zeros(3),
                             K, distortion)[0].reshape(-1, 2)
    lost = ~(np.linalg.norm(back - sides, axis=1) < 0.5)
    moved = ideal.reshape(-1, 2) * [new_K[0, 0], new_K[1, 1]] + [new_K[0, 2], new_K[1, 2]]
    for side in (0, 1):  # a lost left or right side lies beyond that image border
        if lost[side]:
            moved[side, 0] = -np.inf if sides[side, 0] < K[0, 2] else np.inf
    x1, x2 = np.clip(moved[:2, 0], 0, size[0])
    if not x2 > x1 or lost[2:].any():
        return None
    y1, y2 = np.clip(moved[2:, 1], 0, size[1])
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1)) if y2 > y1 else None


def undistort_tracks(source, target, K, distortion, new_K, size):
    """Rewrite a camera tracks CSV for the undistorted images; returns (kept, dropped)."""
    kept = dropped = 0
    with Path(source).open(newline="", encoding="utf-8-sig") as rows, \
            Path(target).open("w", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(rows)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                box = undistort_box([float(row[key]) for key in BOX], K, distortion, new_K, size)
            except (TypeError, ValueError):
                box = None
            if box is None:
                dropped += 1
                continue
            row.update(zip(BOX, box))
            writer.writerow(row)
            kept += 1
    return kept, dropped


def undistort_delivery(raw, output, *, progress=print):
    """Create a rectified copy of a raw folder in the ``cam_sync`` layout; never overwrite."""
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(output)
    parameters = json.loads((raw / "undistortion_parameters.json").read_text(encoding="utf-8"))
    K = np.asarray(parameters["K_original"], dtype=float).reshape(3, 3)
    distortion = np.asarray(parameters["D_original"], dtype=float)
    new_K = np.asarray(parameters["K_undistort"], dtype=float).reshape(3, 3)
    size = tuple(parameters["image_processing"]["image_size"])
    maps = cv2.initUndistortRectifyMap(K, distortion, None, new_K, size, cv2.CV_32FC1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".undistort-") as temporary:
        stage = Path(temporary) / "raw"
        for name in ("cam_sync", "LiDAR_529_sync", "Camera_perception"):
            (stage / name).mkdir(parents=True)
        images = sorted(p for p in (raw / "cam_sync").iterdir() if p.suffix.lower() in IMAGES)
        for index, path in enumerate(images, 1):
            image = cv2.imread(str(path))
            if image is None or image.shape[1::-1] != size:
                raise ValueError(f"image does not match the calibration size: {path.name}")
            if not cv2.imwrite(str(stage / "cam_sync" / path.name),
                               cv2.remap(image, *maps, cv2.INTER_LINEAR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"cannot write {path.name}")
            if progress and (index % 500 == 0 or index == len(images)):
                progress(f"images {index}/{len(images)}")
        tracks = Path("Camera_perception") / "tracks_detailed.csv"
        kept, dropped = undistort_tracks(raw / tracks, stage / tracks, K, distortion, new_K, size)
        clouds = sorted((raw / "LiDAR_529_sync").glob("*.pcd"))
        for path in clouds:
            try:
                os.link(path, stage / "LiDAR_529_sync" / path.name)
            except OSError:
                shutil.copyfile(path, stage / "LiDAR_529_sync" / path.name)
        shutil.copyfile(raw / "LiDAR_perception.csv", stage / "LiDAR_perception.csv")
        parameters["delivered_images"] = "rectified with K_undistort; camera boxes moved to match"
        (stage / "undistortion_parameters.json").write_text(
            json.dumps(parameters, indent=2), encoding="utf-8")
        stage.rename(output)
    return {"images": len(images), "point_clouds": len(clouds),
            "camera_boxes": kept, "camera_boxes_outside_image": dropped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="delivery with distorted images")
    parser.add_argument("--output", required=True, type=Path, help="new rectified raw folder")
    args = parser.parse_args()
    print(json.dumps(undistort_delivery(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
