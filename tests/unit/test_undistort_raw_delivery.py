"""The one-time delivery repair moves images and camera boxes together."""

import csv
import json

import cv2
import numpy as np
import pytest

from scripts.undistort_raw_delivery import undistort_box, undistort_delivery

SIZE = (320, 240)
K = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])
DISTORTION = np.array([-0.33, 0.15, 0.0, 0.0, -0.04])  # not invertible near the image corners
NEW_K = np.array([[180.0, 0.0, 160.0], [0.0, 150.0, 120.0], [0.0, 0.0, 1.0]])


def test_box_follows_the_object_into_the_undistorted_image(tmp_path):
    raw = tmp_path / "raw"
    for name in ("cam_sync", "LiDAR_529_sync", "Camera_perception"):
        (raw / name).mkdir(parents=True)
    image = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    cv2.rectangle(image, (30, 40), (110, 100), (255, 255, 255), -1)  # the detected object
    cv2.imwrite(str(raw / "cam_sync" / "001.png"), image)
    (raw / "LiDAR_529_sync" / "001.pcd").write_bytes(b"points")
    (raw / "LiDAR_perception.csv").write_text("timestamp,uuid\n001,a\n")
    (raw / "undistortion_parameters.json").write_text(json.dumps({
        "K_original": K.tolist(), "D_original": DISTORTION.tolist(),
        "K_undistort": NEW_K.tolist(), "image_processing": {"image_size": list(SIZE)}}))
    with (raw / "Camera_perception" / "tracks_detailed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "track_id", "bb_left", "bb_top", "bb_width", "bb_height"])
        writer.writerow(["001", "7", 30, 40, 80, 60])
        writer.writerow(["001", "8", -900, -900, 5, 5])  # falls outside the rectified image
        writer.writerow(["001", "9", 0, 0, 20, 20])  # beyond the calibrated field of view

    report = undistort_delivery(raw, tmp_path / "fixed", progress=None)

    assert report == {"images": 1, "point_clouds": 1, "camera_boxes": 1,
                      "camera_boxes_outside_image": 2}
    with (tmp_path / "fixed/Camera_perception/tracks_detailed.csv").open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    left, top = float(row["bb_left"]), float(row["bb_top"])
    right, bottom = left + float(row["bb_width"]), top + float(row["bb_height"])
    fixed = cv2.imread(str(tmp_path / "fixed/cam_sync/001.png"), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(fixed > 127)
    # The moved box matches the object as it appears after undistortion (the
    # corners of this filled rectangle bulge a little past the side midpoints) ...
    assert np.abs([left - xs.min(), right - xs.max(), top - ys.min(), bottom - ys.max()]).max() < 9
    # ... which the original box no longer does.
    assert abs(30 - xs.min()) > 10
    assert row["track_id"] == "7"
    assert (tmp_path / "fixed/LiDAR_529_sync/001.pcd").read_bytes() == b"points"
    assert cv2.imread(str(raw / "cam_sync" / "001.png")).sum() == image.sum()
    with pytest.raises(FileExistsError):
        undistort_delivery(raw, tmp_path / "fixed", progress=None)


def test_box_is_unchanged_without_distortion():
    box = undistort_box([30, 40, 80, 60], K, np.zeros(5), K, SIZE)
    assert box == pytest.approx((30, 40, 80, 60))
