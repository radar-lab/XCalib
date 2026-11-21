"""Vendor LiDAR CSV rows become SUSTechPOINTS 3D drafts in the PCD frame."""

import csv
import json
import math

import numpy as np
import pytest

from xcalib.gui.lidar_import import ANNOTATOR, _rotation, label_lidar, preview
from xcalib.gui.scene import box3d_corners, load_points_xyz

SIZE = (4.6, 2.0, 1.6)


def _scene(root, rotation, origin, *, distance=True, frames=12):
    """Synthetic site: static ground and wall, four cars that move every frame.

    Points are stored in the sensor frame ``rotation @ (world - origin)``; the
    CSV stays in the z-up world frame, like a vendor perception export.
    """
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(-30, 30, (4000, 2)), np.zeros(4000)]
    wall = np.c_[np.full(800, 25.0), rng.uniform(-20, 20, 800), rng.uniform(0, 6, 800)]
    (root / "lidar").mkdir(parents=True)
    (root / "vendor").mkdir()
    fields = ["timestamp", "id", "position_x", "position_y", "position_z", "length", "width",
              "height", "heading", "sub_classification", "primary_sensor"]
    fields += ["distance_to_primary_sensor"] if distance else []
    truth = {}
    with (root / "vendor" / "lidar_perception.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index in range(frames):
            frame = f"{index:03d}"
            cloud = [ground, wall]
            for car, heading in enumerate((0.0, 180.0, 35.0, -80.0)):
                center = np.array([-18.0 + 3 * index + 2 * car, -9.0 + 6 * car, 0.9])
                local = rng.uniform(-0.45, 0.45, (80, 3)) * SIZE
                cloud.append(local @ _rotation(2, math.radians(heading)).T + center)
                truth[frame, str(10 + car)] = center
                kind = "" if index == 0 else "car"  # a new track starts unclassified
                writer.writerow(dict(zip(fields, [frame, 10 + car, *center, *SIZE, heading, kind,
                                                  "A", np.linalg.norm(center - origin)])))
            # Tracked by another sensor: no returns in this cloud.
            writer.writerow(dict(zip(fields, [frame, 99, 0.0, 28.0, 0.9, *SIZE, 0.0, "car", "B", 9.0])))
            points = (np.concatenate(cloud) - origin) @ rotation.T
            np.save(root / "lidar" / f"{frame}.npy", points.astype(np.float32))
    return truth


def test_tilted_sensor_frame_is_recovered_and_labels_follow_the_points(tmp_path):
    rotation = _rotation(0, 0.2) @ _rotation(1, -0.15) @ _rotation(2, 2.0)
    origin = np.array([3.0, -4.0, 5.0])
    truth = _scene(tmp_path / "base", rotation, origin)
    _scene(tmp_path / "refine", rotation, origin, frames=4)
    human = tmp_path / "base" / "label" / "001.json"
    human.parent.mkdir()
    human.write_text('[{"obj_id": "human"}]')

    report = label_lidar([tmp_path / "base", tmp_path / "refine"], progress=None)

    matrix = np.array(report["transform"]["matrix"])
    assert report["transform"]["source"] == "estimated" and report["transform"]["sensor"] == "A"
    angle = math.degrees(math.acos(min(1.0, (np.trace(matrix[:3, :3] @ rotation.T) - 1) / 2)))
    assert angle < 1.0
    assert np.abs(matrix[:3, 3] + rotation @ origin).max() < 0.25
    assert report["scenes"]["base"] == {"frames_written": 11, "boxes_written": 44,
                                        "boxes_without_points": 11, "frames_kept_from_reviewer": 1}
    assert report["scenes"]["refine"]["boxes_written"] == 16
    assert json.loads(human.read_text()) == [{"obj_id": "human"}]
    assert json.loads((tmp_path / "refine/vendor/lidar_transform.json").read_text()) \
        == report["transform"]
    labels = json.loads((tmp_path / "base/label/000.json").read_text())
    assert [label["obj_id"] for label in labels] == ["10", "11", "12", "13"]
    for label in labels:
        assert label["obj_type"] == "Car" and label["annotator"] == ANNOTATOR
        center = rotation @ (truth["000", label["obj_id"]] - origin)
        corners = box3d_corners(label["psr"])
        assert np.abs(corners.mean(axis=0) - center).max() < 0.3
        # The box stays upright on the ground although the sensor frame is tilted.
        height = (corners[1] - corners[0]) / SIZE[2]
        assert height @ (rotation @ [0.0, 0.0, 1.0]) > 0.999

    again = label_lidar([tmp_path / "base"], matrix=matrix, progress=None)
    assert again["transform"]["source"] == "provided"
    assert again["scenes"]["base"] == report["scenes"]["base"]
    assert preview(tmp_path / "base", "000", size=200).shape == (200, 200, 3)


def test_detections_already_in_the_point_cloud_frame_need_no_transform(tmp_path):
    _scene(tmp_path / "scene", np.eye(3), np.zeros(3), distance=False)
    report = label_lidar([tmp_path / "scene"], progress=None)
    assert report["transform"]["source"] == "identity"
    label = json.loads((tmp_path / "scene/label/000.json").read_text())[2]
    assert label["psr"]["rotation"] == pytest.approx({"x": 0, "y": 0, "z": math.radians(35)})
    assert label["psr"]["scale"] == dict(zip("xyz", SIZE))


def test_unexplained_frame_is_refused_instead_of_guessed(tmp_path):
    _scene(tmp_path / "scene", _rotation(2, 1.0), np.array([3.0, -4.0, 5.0]), distance=False)
    with pytest.raises(ValueError, match="matrix="):
        label_lidar([tmp_path / "scene"], progress=None)
    assert not (tmp_path / "scene" / "label").exists()
    assert not (tmp_path / "scene" / "vendor" / "lidar_transform.json").exists()


def _as_site_export(scene, turn=0.0):
    """Rewrite a scene as the export whose clouds are in the vendor's site frame: metres and
    radians in x, y, z, yaw, longitude and latitude in position_x and position_y. ``turn``
    [rad] puts clouds and rows in a site frame of their own."""
    spin = _rotation(2, turn)
    path = scene / "vendor" / "lidar_perception.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        x, y, z = spin @ [float(row[key]) for key in ("position_x", "position_y", "position_z")]
        row.update(x=x, y=y, z=z, yaw=math.radians(float(row["heading"])) + turn, position_x=-85.3,
                   position_y=35.0, heading=-float(row["heading"]) % 360)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for cloud in (scene / "lidar").glob("*.npy"):
        np.save(cloud, (np.load(cloud) @ spin.T).astype(np.float32))


def test_a_site_frame_delivery_joins_the_lidar_frame_of_the_earlier_scene(tmp_path):
    rotation = _rotation(0, 0.2) @ _rotation(1, -0.15) @ _rotation(2, 2.0)
    origin = np.array([3.0, -4.0, 5.0])
    base, later, elsewhere = tmp_path / "base", tmp_path / "later", tmp_path / "elsewhere"
    _scene(base, rotation, origin)  # clouds in the sensor frame, as the model saw them
    target = label_lidar([base], progress=None)["transform"]["matrix"]
    truth = _scene(later, np.eye(3), np.zeros(3), distance=False, frames=6)
    _as_site_export(later)

    report = label_lidar([later], frame_of=base, progress=None)

    assert report["transform"]["source"] == "reference scene"
    assert report["transform"]["matrix"] == target and report["transform"]["cloud_overlap"] > 0.8
    assert not list((later / "lidar").glob("*.npy")) and not (later / "lidar.before").exists()
    moved = load_points_xyz(str(later / "lidar" / "000.pcd"))
    assert np.abs(moved - np.load(base / "lidar" / "000.npy")).max() < 0.5  # the same site, same frame now
    labels = json.loads((later / "label" / "000.json").read_text())
    assert [label["obj_id"] for label in labels] == ["10", "11", "12", "13"]
    for label in labels:
        center = rotation @ (truth["000", label["obj_id"]] - origin)
        assert np.abs(box3d_corners(label["psr"]).mean(axis=0) - center).max() < 0.5
    before = (later / "lidar" / "000.pcd").read_bytes()
    again = label_lidar([later], frame_of=base, progress=None)  # already there: nothing moves twice
    assert again["transform"] == report["transform"]
    assert (later / "lidar" / "000.pcd").read_bytes() == before

    _scene(elsewhere, np.eye(3), np.zeros(3), distance=False, frames=6)
    _as_site_export(elsewhere, turn=1.5)  # its own site frame: the clouds end up somewhere else
    with pytest.raises(ValueError, match="do not come to lie"):
        label_lidar([elsewhere], frame_of=base, progress=None)
    assert len(list((elsewhere / "lidar").glob("*.npy"))) == 6 and not (elsewhere / "label").exists()
    with pytest.raises(ValueError, match="LiDAR frame is unknown"):
        label_lidar([later], frame_of=elsewhere, progress=None)
