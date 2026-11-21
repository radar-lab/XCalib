"""Prepare isolated SUSTech-style base/refine scenes from local data.

A raw delivery holds synchronized images, point clouds, camera detections and
LiDAR detections; ``DELIVERY_LAYOUTS`` lists the folder and file names of the
known exports. Temporal partitions do not guarantee track independence. Vendor LiDAR CSV
coordinates are deliberately kept as raw records: their coordinate frame and
vertical convention cannot be inferred safely from a CSV header.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from .rename_retry import rename_when_free


# The parts of a raw delivery under the names each known export uses.
DELIVERY_LAYOUTS = (
    {"images": "cam_sync", "clouds": "LiDAR_529_sync",
     "camera": "Camera_perception/tracks_detailed.csv", "lidar": "LiDAR_perception.csv"},
    {"images": "img_rect", "clouds": "pcd", "camera": "camera_perception.txt", "lidar": "perception.csv"},
)
_MOT_COLUMNS = ("image_name", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class_id")
_COCO_NAMES = {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}


def delivery_layout(raw) -> dict[str, Path]:
    """Images, point clouds, camera and LiDAR detections of a raw delivery."""
    raw = Path(raw)
    present = [[(raw / name).exists() for name in layout.values()] for layout in DELIVERY_LAYOUTS]
    best = max(range(len(present)), key=lambda i: sum(present[i]))
    if not all(present[best]):
        missing = [name for name, found in zip(DELIVERY_LAYOUTS[best].values(), present[best]) if not found]
        raise ValueError(f"{raw} is not a raw data folder. Missing: {', '.join(missing)}")
    return {key: raw / name for key, name in DELIVERY_LAYOUTS[best].items()}


def _camera_table(path: Path) -> tuple[list[str], list[dict]]:
    """Column names and rows of a camera detection file.

    A ``.txt`` file is MOT text without a header (frame, track, left, top,
    width, height, score, COCO class id, ...) and is read under the column
    names of the tracks CSV.
    """
    with path.open(newline="", encoding="utf-8-sig") as stream:
        if path.suffix.lower() != ".txt":
            reader = csv.DictReader(stream)
            return list(reader.fieldnames or []), list(reader)
        cells = [[cell.strip() for cell in row] for row in csv.reader(stream) if row]
    width = max(map(len, cells), default=0)
    names = [*_MOT_COLUMNS, *(f"column_{i + 1}" for i in range(len(_MOT_COLUMNS), width))]
    rows = [dict(zip(names, row)) for row in cells]
    for row in rows:
        row["class_name"] = _COCO_NAMES.get(row.get("class_id", ""), "")
    return [*names, "class_name"], rows


def _frame_files(directory: Path, extensions: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"missing directory: {directory}")
    result = {}
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.suffix.lower() in extensions:
            if item.stem in result:
                raise ValueError(f"duplicate frame: {item.stem}")
            result[item.stem] = item
    return result


def _split(frames: list[str], fraction: float) -> tuple[list[str], list[str]]:
    if not 0 < fraction < 1 or len(frames) < 2:
        raise ValueError("a split needs at least two frames and a fraction between 0 and 1")
    cut = max(1, min(len(frames) - 1, int(len(frames) * fraction)))
    return frames[:cut], frames[cut:]


def _create_scene(root: Path, camera: str) -> None:
    for directory in (root / "lidar", root / "camera" / camera, root / "label",
                      root / "label_fusion" / "camera" / camera):
        directory.mkdir(parents=True, exist_ok=True)


def _copy_frames(source: Path, root: Path, camera: str, frames: list[str],
                 images: dict[str, Path], clouds: dict[str, Path], progress=None) -> None:
    _create_scene(root, camera)
    calibration = source / "calib" / "camera" / f"{camera}.json"
    if calibration.is_file():
        (root / "calib" / "camera").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(calibration, root / "calib" / "camera" / calibration.name)
    for i, frame in enumerate(frames, 1):
        shutil.copyfile(clouds[frame], root / "lidar" / clouds[frame].name)
        shutil.copyfile(images[frame], root / "camera" / camera / images[frame].name)
        for subdir in (Path("label"), Path("label_fusion") / "camera" / camera):
            original = source / subdir / f"{frame}.json"
            if original.is_file():
                shutil.copyfile(original, root / subdir / original.name)
        if progress and (i % 250 == 0 or i == len(frames)):
            progress(f"{root.name}: {i}/{len(frames)} frames")


def preserve_unmatched_files(images_dir, lidar_dir, scene, camera, *, progress=print):
    """Keep unsynchronized files separately; never add them as paired frames."""
    images = _frame_files(Path(images_dir), {".jpg", ".jpeg", ".png"})
    clouds = _frame_files(Path(lidar_dir), {".pcd", ".npy", ".bin"})
    scene = Path(scene)
    if not (scene / "lidar").is_dir() or not camera or camera in (".", "..") or any(c in camera for c in "/\\"):
        raise ValueError("select an existing scene and a safe camera name")
    destination = scene / "unmatched"
    if destination.exists():
        raise FileExistsError(f"unmatched directory already exists: {destination}")
    image_only = sorted(images.keys() - clouds.keys())
    lidar_only = sorted(clouds.keys() - images.keys())
    with tempfile.TemporaryDirectory(prefix=".unmatched-", dir=scene) as temporary:
        staging = Path(temporary) / "unmatched"
        for name, keys, files, folder in (("images", image_only, images, Path("camera") / camera),
                                          ("lidar", lidar_only, clouds, Path("lidar"))):
            if keys:
                target = staging / folder
                target.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(keys, 1):
                    shutil.copyfile(files[frame], target / files[frame].name)
                    if progress and (i % 250 == 0 or i == len(keys)):
                        progress(f"unmatched {name}: {i}/{len(keys)}")
        staging.mkdir(exist_ok=True)
        rename_when_free(staging, destination, progress)
    return {"images": len(image_only), "lidar": len(lidar_only)}


def _write_camera_drafts(csv_path: Path, partitions: dict[str, tuple[Path, set[str]]],
                         camera: str) -> dict[str, set[str]]:
    tracks = defaultdict(set)
    by_scene_frame = defaultdict(list)
    type_map = {"person": "Pedestrian", "car": "Car", "truck": "Truck", "bus": "Bus",
                "van": "Van", "bicycle": "BicycleRider", "motorcycle": "MotorcycleRider"}
    names, rows = _camera_table(csv_path)
    expected = {"image_name", "track_id", "class_name", "bb_left", "bb_top", "bb_width", "bb_height", "conf"}
    if not expected.issubset(names):
        raise ValueError("camera tracks CSV has missing required columns")
    for row in rows:
        frame, track = (row.get("image_name") or "").strip(), (row.get("track_id") or "").strip()
        owner = next((key for key, (_, frames) in partitions.items() if frame in frames), None)
        if owner is None:
            continue
        try:
            x, y, w, h, score = (float(row.get(key)) for key in
                                  ("bb_left", "bb_top", "bb_width", "bb_height", "conf"))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h, score)) or w <= 0 or h <= 0:
            continue
        tracks[owner].add(track)
        by_scene_frame[(owner, frame)].append({
            "rect": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
            "obj_type": type_map.get((row.get("class_name") or "").strip().lower(), "Unknown"),
            "obj_id": "", "obj_attr": f"t{track}", "annotator": "vendor-cam-draft", "score": score,
        })
    for (owner, frame), boxes in by_scene_frame.items():
        directory = partitions[owner][0] / "label_fusion" / "camera" / camera
        (directory / f"{frame}.json").write_text(json.dumps({"objs": boxes}, indent=1), encoding="utf-8")
    return tracks


def _copy_vendor_rows(csv_path: Path, partitions: dict[str, tuple[Path, set[str]]]) -> dict[str, set[str]]:
    from contextlib import ExitStack

    tracks = defaultdict(set)
    with csv_path.open(newline="", encoding="utf-8-sig") as stream, ExitStack() as stack:
        reader = csv.DictReader(stream)
        if not {"timestamp", "uuid"}.issubset(reader.fieldnames or []):
            raise ValueError("LiDAR vendor CSV has missing timestamp or uuid")
        writers = {}
        for key, (root, _) in partitions.items():
            directory = root / "vendor"
            directory.mkdir(exist_ok=True)
            out = stack.enter_context((directory / "lidar_perception.csv").open("w", newline="", encoding="utf-8"))
            writers[key] = csv.DictWriter(out, fieldnames=reader.fieldnames)
            writers[key].writeheader()
        for row in reader:
            frame = row["timestamp"].strip()
            for key, (_, frames) in partitions.items():
                if frame in frames:
                    writers[key].writerow(row)
                    tracks[key].add(row["uuid"].strip())
                    break
    return tracks


def _copy_camera_vendor_rows(csv_path: Path, partitions: dict[str, tuple[Path, set[str]]]) -> None:
    """Retain original camera rows for traceability without changing draft labels."""
    from contextlib import ExitStack

    for root, _ in partitions.values():
        target = root / "vendor" / "camera_tracks.csv"
        if target.exists():
            raise FileExistsError(f"camera vendor records already exist: {target}")
    names, rows = _camera_table(csv_path)
    if "image_name" not in names:
        raise ValueError("camera tracks CSV has no image_name column")
    with ExitStack() as stack:
        writers = {}
        for key, (root, _) in partitions.items():
            directory = root / "vendor"
            directory.mkdir(exist_ok=True)
            out = stack.enter_context((directory / "camera_tracks.csv").open("x", newline="", encoding="utf-8"))
            writers[key] = csv.DictWriter(out, fieldnames=names)
            writers[key].writeheader()
        for row in rows:
            frame = (row.get("image_name") or "").strip()
            for key, (_, frames) in partitions.items():
                if frame in frames:
                    writers[key].writerow(row)
                    break


def prepare_raw_workflow_scene(raw_scene, output, *, scene_prefix="scene", camera="cam_sync",
                               fraction=0.8, role=None, progress=print):
    """Convert one raw synchronized-image/PCD dataset into new scenes.

    By default the frames are split in time into ``<prefix>_base`` and
    ``refine/<prefix>_refine``; with ``role`` "base" or "refine" the whole
    delivery becomes that one scene. Vendor detections remain unpaired camera
    drafts and raw LiDAR vendor records. No LiDAR-frame transform or calibration
    is inferred.
    """
    raw_scene, output = Path(raw_scene), Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_prefix) or scene_prefix in (".", ".."):
        raise ValueError("scene prefix must contain only letters, numbers, underscores, or hyphens")
    if not camera or camera in (".", "..") or any(c in camera for c in "/\\"):
        raise ValueError("camera must be a single safe directory name")
    if role not in (None, "base", "refine"):
        raise ValueError('role must be "base", "refine" or None')
    parts = delivery_layout(raw_scene)
    images = _frame_files(parts["images"], {".jpg", ".jpeg", ".png"})
    clouds = _frame_files(parts["clouds"], {".pcd"})
    frames = sorted(images.keys() & clouds.keys())
    base_name, refine_name = scene_prefix + "_base", scene_prefix + "_refine"
    if role is None:
        groups = dict(zip((base_name, refine_name), _split(frames, fraction)))
    elif frames:
        groups = {base_name if role == "base" else refine_name: frames}
    else:
        raise ValueError("the delivery has no frame with both an image and a point cloud")
    for path in (parts["camera"], parts["lidar"]):
        if not path.is_file():
            raise ValueError(f"missing vendor input: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".raw-scenes-", dir=output.parent) as temporary:
        staging = Path(temporary) / "scenes"
        staging.mkdir()
        partitions = {name: (staging / name if name == base_name else staging / "refine" / name, set(group))
                      for name, group in groups.items()}
        for name, group in groups.items():
            _copy_frames(raw_scene, partitions[name][0], camera, group, images, clouds, progress)
        preserve_unmatched_files(parts["images"], parts["clouds"], next(iter(partitions.values()))[0],
                                 camera, progress=progress)
        cameras = _write_camera_drafts(parts["camera"], partitions, camera)
        _copy_camera_vendor_rows(parts["camera"], partitions)
        lidars = _copy_vendor_rows(parts["lidar"], partitions)
        notes = ["Camera drafts are not verified pairs; LiDAR vendor coordinates were not transformed."]
        if role is None:
            notes.append("Temporal splits can share tracks and are not independent evaluations.")
        summary = {"split": "temporal" if role is None else "none",
                   "fraction": fraction if role is None else None,
                   "frame_counts": {name: len(group) for name, group in groups.items()},
                   "shared_tracks": {"camera": len(cameras[base_name] & cameras[refine_name]),
                                     "lidar": len(lidars[base_name] & lidars[refine_name])},
                   "unmatched": {"images_without_lidar": len(images.keys() - clouds.keys()),
                                 "lidar_without_images": len(clouds.keys() - images.keys())},
                   "notes": notes}
        (staging / "split_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rename_when_free(staging, output, progress)
    return summary
