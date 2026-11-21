import csv
import json

import pytest

from xcalib.gui.dataset_prepare import prepare_raw_workflow_scene


def _vendor(root):
    for part in ("cam_sync", "LiDAR_529_sync", "Camera_perception"):
        (root / part).mkdir(parents=True)
    for i in range(5):
        frame = f"{i:03d}"
        (root / "cam_sync" / f"{frame}.jpg").write_bytes(b"image")
        (root / "LiDAR_529_sync" / f"{frame}.pcd").write_bytes(b"pcd")
    (root / "cam_sync/unpaired.jpg").write_bytes(b"unpaired image")
    (root / "LiDAR_529_sync/unpaired_lidar.pcd").write_bytes(b"unpaired lidar")
    with (root / "Camera_perception" / "tracks_detailed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "track_id", "class_name", "bb_left", "bb_top", "bb_width", "bb_height", "conf"])
        for i in range(5):
            writer.writerow([f"{i:03d}", "7", "car", 1, 2, 3, 4, .9])
    with (root / "LiDAR_perception.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "uuid", "position_x"])
        for i in range(5):
            writer.writerow([f"{i:03d}", "lidar-track", 4])


def test_new_raw_scene_supports_separate_reproducible_partitions(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "output"
    _vendor(raw)
    summary = prepare_raw_workflow_scene(raw, output, scene_prefix="new", progress=None)
    assert summary["frame_counts"] == {"new_base": 4, "new_refine": 1}
    assert summary["shared_tracks"] == {"camera": 1, "lidar": 1}
    draft = json.loads((output / "refine/new_refine/label_fusion/camera/cam_sync/004.json").read_text())
    assert draft["objs"][0]["obj_id"] == "" and draft["objs"][0]["obj_attr"] == "t7"
    assert not list((output / "refine/new_refine/label").glob("*.json"))
    base = {p.stem for p in (output / "new_base/lidar").glob("*.pcd")}
    refine = {p.stem for p in (output / "refine/new_refine/lidar").glob("*.pcd")}
    assert not base & refine and len(base | refine) == 5
    assert (output / "new_base/vendor/lidar_perception.csv").is_file()
    assert (output / "new_base/vendor/camera_tracks.csv").is_file()
    assert (output / "new_base/unmatched/camera/cam_sync/unpaired.jpg").is_file()
    assert (raw / "LiDAR_529_sync/000.pcd").read_bytes() == b"pcd"  # the delivery is left as it was
    assert summary["unmatched"] == {"images_without_lidar": 1, "lidar_without_images": 1}
    with pytest.raises(FileExistsError):
        prepare_raw_workflow_scene(raw, output, scene_prefix="new", progress=None)
    with pytest.raises(ValueError, match="prefix"):
        prepare_raw_workflow_scene(raw, tmp_path / "other", scene_prefix="../escape", progress=None)


def _site_export(root):
    """The other export: img_rect, pcd, camera detections as MOT text, one perception CSV."""
    for part in ("img_rect", "pcd"):
        (root / part).mkdir(parents=True)
    for i in range(5):
        (root / "img_rect" / f"{i:03d}.jpg").write_bytes(b"image")
        (root / "pcd" / f"{i:03d}.pcd").write_bytes(b"pcd")
    (root / "camera_perception.txt").write_text(
        "".join(f"{i:03d},7,1,2,3,4,0.9,2,-1,-1\n" for i in range(5)) + "000,8,5,6,7,8,0.5,0,-1,-1\n")
    (root / "perception.csv").write_text(
        "timestamp,uuid,x\n" + "".join(f"{i:03d},lidar-track,4\n" for i in range(5)))


def test_whole_delivery_becomes_one_scene_in_either_layout(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "output"
    _site_export(raw)
    summary = prepare_raw_workflow_scene(raw, output, scene_prefix="day2", role="refine", progress=None)
    assert summary["frame_counts"] == {"day2_refine": 5} and summary["split"] == "none"
    scene = output / "refine" / "day2_refine"
    assert not (output / "day2_base").exists() and len(list((scene / "lidar").glob("*.pcd"))) == 5
    drafts = json.loads((scene / "label_fusion/camera/cam_sync/000.json").read_text())["objs"]
    assert [(d["obj_attr"], d["obj_type"], d["rect"]) for d in drafts] == [
        ("t7", "Car", {"x1": 1.0, "y1": 2.0, "x2": 4.0, "y2": 6.0}),
        ("t8", "Pedestrian", {"x1": 5.0, "y1": 6.0, "x2": 12.0, "y2": 14.0})]
    with (scene / "vendor" / "camera_tracks.csv").open(newline="") as stream:
        kept = list(csv.DictReader(stream))
    assert len(kept) == 6 and kept[0]["class_name"] == "car" and kept[0]["column_10"] == "-1"
    assert (scene / "vendor" / "lidar_perception.csv").read_text().count("lidar-track") == 5

    first = tmp_path / "first"
    _vendor(first)
    summary = prepare_raw_workflow_scene(first, tmp_path / "second", scene_prefix="day1", role="base",
                                         progress=None)
    assert summary["frame_counts"] == {"day1_base": 5}
    assert (tmp_path / "second/day1_base/unmatched/lidar/unpaired_lidar.pcd").is_file()
    (raw / "perception.csv").unlink()
    with pytest.raises(ValueError, match="Missing: perception.csv"):
        prepare_raw_workflow_scene(raw, tmp_path / "third", scene_prefix="day3", progress=None)
    with pytest.raises(ValueError, match="role"):
        prepare_raw_workflow_scene(first, tmp_path / "third", scene_prefix="day3", role="other", progress=None)
