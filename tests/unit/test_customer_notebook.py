"""The steps of the homography notebook: one call per cell, every choice a dialog."""

import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from xcalib.gui.customer import Stop, Workflow

SIZE = (4.6, 2.0, 1.6)


def _scene(root, camera="front"):
    for directory in (root / "lidar", root / "camera" / camera):
        directory.mkdir(parents=True)
    (root / "lidar" / "000.pcd").write_bytes(b"points")
    return root


def test_scenes_are_chosen_in_a_dialog_and_wrong_folders_are_explained(tmp_path, monkeypatch):
    base = _scene(tmp_path / "site" / "site_base")
    refine = _scene(tmp_path / "site" / "refine" / "site_refine")
    (tmp_path / "empty").mkdir()
    flow = Workflow()
    picked = []
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None: picked.pop(0))

    picked[:] = [tmp_path / "site"]  # the dataset folder instead of the scene inside it
    flow.choose("base")
    assert flow.scenes["base"] == (base, "front")
    for wrong, reason in ((base, "different"), (tmp_path / "empty", "not a scene")):
        picked[:] = [wrong]
        with pytest.raises(Stop, match=reason):
            flow.choose("refine")
    picked[:] = [refine]
    flow.choose("refine")
    assert flow.scenes["refine"] == (refine, "front")

    monkeypatch.undo()
    monkeypatch.setattr("subprocess.run", lambda *args, **kw: SimpleNamespace(stdout='""'))
    with pytest.raises(Stop, match="cancelled"):
        flow.choose("base")


def test_windows_wait_for_the_user_and_report_what_was_saved(tmp_path, monkeypatch, capsys):
    flow = Workflow()
    flow.repo = tmp_path
    flow.scenes["base"] = (_scene(tmp_path / "site_base"), "front")
    calls = []
    monkeypatch.setattr("subprocess.run", lambda args, **kw: calls.append(args))
    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: calls.append(args) or SimpleNamespace(
        wait=lambda timeout=None: 0))   # the pairing window, closed at once

    flow.pair("base")
    assert "xcalib.gui.review" in calls[-1] and "--notebook-workflow" in calls[-1]
    assert "at least two frames with two saved pairs" in capsys.readouterr().err
    with pytest.raises(Stop, match="pairing step"):
        flow.train("base")
    assert not (tmp_path / "runs").exists()


def _raw_delivery(root, frames=12):
    """Rectified delivery: three cars move every frame; boxes are already in the PCD frame."""
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(-30, 30, (3000, 2)), np.full(3000, -4.0)]
    for part in ("cam_sync", "LiDAR_529_sync", "Camera_perception"):
        (root / part).mkdir(parents=True)
    (root / "undistortion_parameters.json").write_text(json.dumps({
        "K_undistort": [[50, 0, 32], [0, 50, 24], [0, 0, 1]],
        "image_processing": {"image_size": [64, 48]}}))
    with (root / "Camera_perception" / "tracks_detailed.csv").open("w", newline="") as cam, \
            (root / "LiDAR_perception.csv").open("w", newline="") as lidar:
        cam, lidar = csv.writer(cam), csv.writer(lidar)
        cam.writerow(["image_name", "track_id", "class_name", "bb_left", "bb_top", "bb_width",
                      "bb_height", "conf"])
        lidar.writerow(["timestamp", "id", "uuid", "position_x", "position_y", "position_z",
                        "length", "width", "height", "heading", "sub_classification"])
        for index in range(frames):
            frame = f"{index:03d}"
            cv2.imwrite(str(root / "cam_sync" / f"{frame}.jpg"), np.zeros((48, 64, 3), np.uint8))
            cam.writerow([frame, 7, "car", 10, 12, 20, 10, 0.9])
            cloud = [ground]
            for car in range(3):
                center = np.array([-15.0 + 2.5 * index + car, -6.0 + 6 * car, -3.2])
                cloud.append(rng.uniform(-0.45, 0.45, (60, 3)) * SIZE + center)
                lidar.writerow([frame, 40 + car, f"u{car}", *center, *SIZE, 0.0, "car"])
            points = np.concatenate(cloud).astype("<f4")
            header = (f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                      f"WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n")
            (root / "LiDAR_529_sync" / f"{frame}.pcd").write_bytes(header.encode() + points.tobytes())


def test_raw_delivery_becomes_scenes_with_camera_and_lidar_drafts(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "Site 7"
    _raw_delivery(raw)
    (tmp_path / "other").mkdir()
    flow = Workflow()
    flow.repo = tmp_path
    picked = [raw, raw, tmp_path / "other"]
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None: picked.pop(0))

    flow.import_raw()

    base = tmp_path / "datasets" / "raw_workflow" / "Site_7" / "Site_7_base"
    refine = base.parent / "refine" / "Site_7_refine"
    assert len(list((base / "lidar").glob("*.pcd"))) == 9
    assert len(list((refine / "label").glob("*.json"))) == 3
    boxes = json.loads((base / "label" / "000.json").read_text())
    assert [box["obj_id"] for box in boxes] == ["40", "41", "42"]
    assert boxes[0]["psr"]["position"] == pytest.approx({"x": -15.0, "y": -6.0, "z": -3.2})
    rects = json.loads((base / "label_fusion" / "camera" / "cam_sync" / "000.json").read_text())
    assert rects["objs"][0]["rect"] == {"x1": 10.0, "y1": 12.0, "x2": 30.0, "y2": 22.0}
    calibration = json.loads((base / "calib" / "intrinsic" / "cam_sync.json").read_text())
    assert calibration["intrinsic"] == [50, 0, 32, 0, 50, 24, 0, 0, 1]
    assert not (base / "calib" / "camera").exists()  # every file there is read as a full calibration
    assert f"Base scene:   {base}" in capsys.readouterr().out

    flow.import_raw()  # the same delivery again: nothing is rewritten
    assert "Already imported" in capsys.readouterr().out
    with pytest.raises(Stop, match="Missing: cam_sync"):
        flow.import_raw()

    # another delivery whose folder name keeps the same letters ("Site_7" and "Site 7" both become
    # Site_7): it shares no frame with the scenes there, so it is not taken for the imported one
    other = tmp_path / "elsewhere" / "Site_7"
    _raw_delivery(other)
    for cloud in (other / "LiDAR_529_sync").iterdir():
        cloud.rename(cloud.with_name("9" + cloud.name))
    picked.append(other)
    with pytest.raises(Stop, match="holds another delivery"):
        flow.import_raw()
    # the same when that other delivery is imported in another way (as the refine scene only):
    # it is told about the other delivery, not to delete the folder that holds it
    picked.append(other)
    with pytest.raises(Stop, match="holds another delivery"):
        flow.import_raw("refine")
    # in the other order: a delivery imported as the base scene only, then a different one with the
    # same folder name imported in full, must not be told to delete the folder of the first
    single = tmp_path / "x" / "Site 9"
    _raw_delivery(single)
    picked.append(single)
    flow.import_raw("base")
    clash = tmp_path / "y" / "Site_9"
    _raw_delivery(clash)
    for cloud in (clash / "LiDAR_529_sync").iterdir():
        cloud.rename(cloud.with_name("9" + cloud.name))
    picked.append(clash)
    with pytest.raises(Stop, match="holds another delivery"):
        flow.import_raw()
    with pytest.raises(Stop, match="not the top of a drive"):
        picked.append(Path(raw.anchor))
        monkeypatch.setattr("xcalib.gui.customer.delivery_layout", lambda folder: {"clouds": other},
                            raising=False)
        monkeypatch.setattr("xcalib.gui.dataset_prepare.delivery_layout", lambda folder: {"clouds": other})
        flow.import_raw()


def test_import_waits_while_another_program_holds_the_new_files(tmp_path, monkeypatch, capsys):
    # Windows refuses to rename a folder while a virus scanner or the search indexer still
    # reads files in it: the import tries again, and says so plainly when it gives up.
    import pathlib

    from xcalib.gui import rename_retry

    raw = tmp_path / "Site 7"
    _raw_delivery(raw)
    monkeypatch.setattr(rename_retry, "PAUSE", 0.0)
    rename, refused = pathlib.Path.rename, []

    def busy(self, target, times):
        if self.name == "scenes" and len(refused) < times:
            refused.append(target)
            raise PermissionError(5, "Access is denied")
        return rename(self, target)

    flow = Workflow()
    flow.repo = tmp_path
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None: raw)
    monkeypatch.setattr(pathlib.Path, "rename", lambda self, target: busy(self, target, 3))
    flow.import_raw()
    output = tmp_path / "datasets" / "raw_workflow" / "Site_7"
    assert len(refused) == 3 and (output / "Site_7_base" / "lidar").is_dir()
    assert "still reading the new files" in capsys.readouterr().out

    raw2 = tmp_path / "Site 8"
    _raw_delivery(raw2)
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None: raw2)
    refused.clear()
    monkeypatch.setattr(pathlib.Path, "rename", lambda self, target: busy(self, target, 10**6))
    with pytest.raises(Stop, match="kept the newly copied files open.*run this cell again"):
        flow.import_raw()
    assert len(refused) == rename_retry.TRIES
    assert not (output.parent / "Site_8").exists()
    assert not list(output.parent.glob(".*"))      # nothing half-done is left behind


def test_moving_the_clouds_keeps_the_old_ones_when_the_new_folder_stays_held(tmp_path, monkeypatch):
    import pathlib

    from xcalib.gui import lidar_import, rename_retry

    scene = tmp_path / "scene"
    (scene / "lidar").mkdir(parents=True)
    np.save(scene / "lidar" / "000.npy", np.ones((5, 3), np.float32))
    monkeypatch.setattr(rename_retry, "PAUSE", 0.0)
    rename = pathlib.Path.rename

    def busy(self, target):
        if self.name == "lidar.moving":
            raise PermissionError(5, "Access is denied")
        return rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", busy)
    with pytest.raises(PermissionError):
        lidar_import._move_clouds(scene, np.eye(4), None)
    assert (scene / "lidar" / "000.npy").is_file() and not (scene / "lidar.before").exists()


def _later_delivery(root, frames=6):
    """The other export, recorded later at the same site: img_rect, pcd with sensor fields,
    camera detections as MOT text, metres in x, y, z, yaw, and no intrinsic file."""
    rng = np.random.default_rng(1)
    ground = np.c_[rng.uniform(-30, 30, (3000, 2)), np.full(3000, -4.0)]
    for part in ("img_rect", "pcd"):
        (root / part).mkdir(parents=True)
    with (root / "camera_perception.txt").open("w") as cam, \
            (root / "perception.csv").open("w", newline="") as lidar:
        lidar = csv.writer(lidar)
        lidar.writerow(["timestamp", "id", "uuid", "x", "y", "z", "yaw", "length", "width", "height",
                        "position_x", "position_y", "position_z", "heading", "sub_classification"])
        for index in range(frames):
            frame = f"{500 + index}"
            cv2.imwrite(str(root / "img_rect" / f"{frame}.jpg"), np.zeros((48, 64, 3), np.uint8))
            cam.write(f"{frame},7,10,12,20,10,0.9,2,-1,-1\n")
            cloud = [ground]
            for car in range(3):
                center = np.array([-15.0 + 2.5 * index + car, -6.0 + 6 * car, -3.2])
                cloud.append(rng.uniform(-0.45, 0.45, (60, 3)) * SIZE + center)
                lidar.writerow([frame, 60 + car, f"v{car}", *center, 0.0, *SIZE, -85.3, 35.0, center[2],
                                0.0, "car"])
            points = np.zeros(sum(map(len, cloud)), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                           ("ring", "<u2")])
            points["x"], points["y"], points["z"] = np.concatenate(cloud).T
            header = (f"VERSION 0.7\nFIELDS x y z ring\nSIZE 4 4 4 2\nTYPE F F F U\nCOUNT 1 1 1 1\n"
                      f"WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n")
            (root / "pcd" / f"{frame}.pcd").write_bytes(header.encode() + points.tobytes())


def test_a_later_delivery_becomes_the_refine_scene_of_the_base_camera(tmp_path, monkeypatch, capsys):
    first, later = tmp_path / "Site 7", tmp_path / "Site 7 day 2"
    _raw_delivery(first)
    _later_delivery(later)
    flow = Workflow()
    flow.repo = tmp_path
    _later_delivery(tmp_path / "Day 3", frames=2)
    picked = [first, later, later, later]
    asked = []
    monkeypatch.setattr(flow, "_pick",
                        lambda title, initial=None, files=None: asked.append(files) or picked.pop(0))
    flow.import_raw()
    base = tmp_path / "datasets" / "raw_workflow" / "Site_7" / "Site_7_base"
    flow.scenes["base"] = (base, "cam_sync")
    capsys.readouterr()

    flow.import_raw("refine")

    refine = tmp_path / "datasets" / "raw_workflow" / "Site_7_day_2" / "refine" / "Site_7_day_2_refine"
    assert not (refine.parent.parent / "Site_7_day_2_base").exists()
    assert len(list((refine / "lidar").glob("*.pcd"))) == 6
    boxes = json.loads((refine / "label" / "500.json").read_text())
    assert [box["obj_id"] for box in boxes] == ["60", "61", "62"]
    assert boxes[0]["psr"]["position"] == pytest.approx({"x": -15.0, "y": -6.0, "z": -3.2})
    rects = json.loads((refine / "label_fusion" / "camera" / "cam_sync" / "500.json").read_text())
    assert rects["objs"][0]["obj_type"] == "Car" and rects["objs"][0]["obj_attr"] == "t7"
    assert (refine / "calib" / "intrinsic" / "cam_sync.json").read_text() \
        == (base / "calib" / "intrinsic" / "cam_sync.json").read_text()
    out = capsys.readouterr().out
    assert "the base scene's is used" in out and f"Refine scene: {refine}" in out
    assert "Base scene" not in out and "Copied 6 refine frames." in out

    flow.import_raw("refine")  # again: nothing is rewritten
    assert "Already imported" in capsys.readouterr().out
    with pytest.raises(Stop, match="imported in a different way"):
        flow.import_raw()  # the same folder, now as a base and a refine part
    with pytest.raises(Stop, match='import_raw\\("refine"\\)'):
        flow.import_raw("both")

    # A base scene needs the intrinsic of its own camera. The file is not always kept with
    # the data, so the step asks for it; a file without a camera matrix is refused.
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps({"camera": "front"}))
    picked[:] = [tmp_path / "Day 3", notes]
    asked.clear()
    with pytest.raises(Stop, match="notes.json cannot be used as the camera intrinsic: notes.json "
                                   "holds no camera matrix"):
        flow.import_raw("base")
    assert asked == [None, "Camera intrinsic (*.json)"]
    assert not (tmp_path / "datasets" / "raw_workflow" / "Day_3").exists()
    kept_elsewhere = tmp_path / "camera files" / "undistortion_parameters.json"
    kept_elsewhere.parent.mkdir()
    kept_elsewhere.write_text(json.dumps({
        "K_undistort": [[50, 0, 32], [0, 50, 24], [0, 0, 1]],
        "image_processing": {"image_size": [64, 48], "crop_to_valid_roi": False}}))
    _later_delivery(tmp_path / "Day 4")
    picked[:] = [tmp_path / "Day 4", kept_elsewhere]
    capsys.readouterr()
    flow.import_raw("base")
    day4 = tmp_path / "datasets" / "raw_workflow" / "Day_4" / "Day_4_base"
    record = json.loads((day4 / "calib" / "intrinsic" / "cam_sync.json").read_text())
    assert record["intrinsic"] == [50, 0, 32, 0, 50, 24, 0, 0, 1] and record["image_size"] == [64, 48]
    assert f"Camera intrinsic: {kept_elsewhere}" in capsys.readouterr().out

    # Chosen as the refine scene, it takes nothing over from the base scene's extrinsic: the
    # homography needs none. Its pairing window opens as the base scene's does.
    from xcalib.gui.scene import Scene
    from xcalib.gui.workflow import save_calibration

    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.5, 0.2]
    save_calibration(Scene(str(base), "cam_sync"), np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1.0]]), pose)
    picked[:] = [refine]
    capsys.readouterr()
    flow.choose("refine")
    assert "extrinsic" not in capsys.readouterr().out
    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: calls.append(args) or SimpleNamespace(
        wait=lambda timeout=None: 0))   # the pairing window, closed at once
    (refine / "calib" / "xcalib" / "cam_sync_model").mkdir(parents=True)
    (refine / "calib" / "xcalib" / "cam_sync_model" / "crlite.pth").write_bytes(b"weights")   # a model is there
    flow.pair("refine")
    assert "--notebook-workflow" in calls[-1] and "--accept-calibration" not in calls[-1]
    assert Scene(str(refine), "cam_sync").calibration() is None


def test_refine_data_is_imported_whole_and_needs_no_base_scene(tmp_path, monkeypatch, capsys):
    """For those who only refine: a raw delivery becomes the refine scene of the session."""
    data, intrinsic = tmp_path / "New site day 1", tmp_path / "camera" / "undistortion_parameters.json"
    _later_delivery(data)
    intrinsic.parent.mkdir()
    intrinsic.write_text(json.dumps({
        "K_undistort": [[50, 0, 32], [0, 50, 24], [0, 0, 1]],
        "image_processing": {"image_size": [64, 48], "crop_to_valid_roi": False}}))
    flow = Workflow()
    flow.repo = tmp_path
    picked, asked = [data, intrinsic], []
    monkeypatch.setattr(flow, "_pick",
                        lambda title, initial=None, files=None: asked.append((title, files)) or picked.pop(0))

    flow.import_raw("refine")

    refine = tmp_path / "datasets" / "raw_workflow" / "New_site_day_1" / "refine" / "New_site_day_1_refine"
    assert flow.scenes == {"refine": (refine, "cam_sync")}   # chosen for the session; no base scene is asked for
    assert "all of its frames become the refine scene" in asked[0][0]
    assert [files for _, files in asked] == [None, "Camera intrinsic (*.json)"]
    assert len(list((refine / "lidar").glob("*.pcd"))) == 6 and not list(refine.parent.parent.glob("*_base"))
    out = capsys.readouterr().out
    assert f"Refine scene: {refine}" in out and "6 frames" in out and "Copied 6 refine frames." in out

    # In a later session the same dialog takes the raw folder again, the scene, or its dataset folder.
    for choice in (data, refine, refine.parent.parent):
        flow = Workflow()
        flow.repo = tmp_path
        monkeypatch.setattr(flow, "_pick", lambda title, initial=None, files=None, choice=choice: choice)
        flow.import_raw("refine")
        assert flow.scenes == {"refine": (refine, "cam_sync")}
    assert "Already imported" in capsys.readouterr().out
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None, files=None: tmp_path / "camera")
    with pytest.raises(Stop, match="is not a raw data folder"):
        flow.import_raw("refine")


def test_base_data_is_imported_whole_and_cannot_be_the_refine_data_too(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "Site 7"
    _raw_delivery(raw)
    flow = Workflow()
    flow.repo = tmp_path
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None, files=None: raw)

    flow.import_raw("base")

    base = tmp_path / "datasets" / "raw_workflow" / "Site_7" / "Site_7_base"
    assert flow.scenes == {"base": (base, "cam_sync")}          # every frame, and chosen for the session
    assert len(list((base / "lidar").glob("*.pcd"))) == 12 and not (base.parent / "refine").exists()
    out = capsys.readouterr().out
    assert f"Base scene: {base}" in out and "12 frames" in out and "Copied 12 base frames." in out
    with pytest.raises(Stop, match="This data is the base data already: all of its frames are in the base scene"):
        flow.import_raw("refine")
    flow.import_raw("base")   # again: nothing is copied, the scene that is there is used
    assert "Already imported" in capsys.readouterr().out and flow.scenes == {"base": (base, "cam_sync")}


def test_the_data_of_the_first_import_chosen_again_gives_its_refine_part(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "Site 7"
    _raw_delivery(raw)
    flow = Workflow()
    flow.repo = tmp_path
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None, files=None: raw)
    flow.import_raw()
    assert flow.scenes == {}   # the first import chooses nothing: the base scene is chosen in its own step
    capsys.readouterr()

    flow.import_raw("refine")

    refine = tmp_path / "datasets" / "raw_workflow" / "Site_7" / "refine" / "Site_7_refine"
    assert flow.scenes == {"refine": (refine, "cam_sync")}
    out = capsys.readouterr().out
    assert "Already imported" in out and "the part that the base scene does not hold" in out
    assert len(list((refine / "lidar").glob("*.pcd"))) == 3   # nothing was copied again


def test_a_base_model_made_earlier_is_chosen_in_a_dialog(tmp_path, monkeypatch, capsys):
    flow = Workflow()
    flow.repo = tmp_path
    run = tmp_path / "runs" / "customer-base-1"
    (run / "model").mkdir(parents=True)
    (run / "model" / "crlite.pth").write_bytes(b"weights")
    (run / "model" / "crlite.yaml").write_text("model: crlite\nscore_mode: absolute\n")
    picked = [run]   # the folder of the run is enough: its model folder is taken
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None, files=None: picked.pop(0))
    capsys.readouterr()

    flow.choose_model("base")

    assert flow.models == {"base": run / "model"}
    said = capsys.readouterr()
    assert f"Base model: {run / 'model'}" in said.out and not said.err

    older = tmp_path / "published"
    older.mkdir()
    (older / "crlite.pth").write_bytes(b"other weights")
    (older / "crlite.yaml").write_text("model: crlite\n")   # scores relative within a frame
    picked[:] = [older]
    flow.choose_model()
    assert flow.models["base"] == older and "nearly every camera box" in capsys.readouterr().err

    picked[:] = [tmp_path]   # no model there: the model chosen before stays
    with pytest.raises(Stop, match="crlite.pth and crlite.yaml"):
        flow.choose_model("base")
    assert flow.models["base"] == older
    with pytest.raises(Stop, match='choose_model\\("base"\\)'):
        flow.choose_model("both")


def test_a_training_that_fails_is_explained_without_a_traceback(tmp_path, monkeypatch, capsys):
    flow = Workflow()
    flow.repo = tmp_path
    flow.scenes["base"] = (_scene(tmp_path / "site_base"), "front")
    monkeypatch.setattr("xcalib.gui.workflow.prepare_training_caches",
                        lambda *args, **kwargs: {"train_frames": 4, "validation_frames": 1,
                                                 "undecided_camera_boxes": 3})

    def crash(*args, **kwargs):
        raise RuntimeError("base training failed (exit 1); see training.log")

    monkeypatch.setattr("xcalib.gui.workflow.train_base_from_caches", crash)
    with pytest.raises(Stop, match="training stopped with an error.*training.log"):
        flow.train("base")
    assert "base" not in flow.models
    out = capsys.readouterr().out    # where the training data is, and what it leaves out
    assert re.search(r"Training data \(HDF5\): .*customer-base-\d+-\d+[\\/]train\.h5 and validation\.h5", out)
    assert "3 camera boxes in these frames have no decision; the training counts them as \"no LiDAR box\"" in out


def test_a_locked_hdf5_file_and_the_refine_window_are_explained(tmp_path, monkeypatch, capsys):
    import shutil

    flow = Workflow()
    flow.repo = tmp_path
    base = _scene(tmp_path / "site_base")
    flow.scenes["base"] = (base, "front")

    def locked(*args, **kwargs):   # the export gave up: a virus scanner kept the new file open
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr("xcalib.gui.workflow.prepare_training_caches", locked)
    with pytest.raises(Stop, match="kept the new HDF5 file open"):
        flow.train("base")

    # step 7 with a base model trained (or chosen) in this session: that model makes the
    # suggestions, and a model the refine scene held from an earlier run is kept beside it
    refine = _scene(tmp_path / "site_refine")
    flow.scenes["refine"] = (refine, "front")
    old = refine / "calib" / "xcalib" / "front_model"
    old.mkdir(parents=True)
    (old / "crlite.pth").write_bytes(b"old")
    (old / "crlite.yaml").write_text("old")
    trained = tmp_path / "runs" / "customer-base-1" / "model"
    trained.mkdir(parents=True)
    (trained / "crlite.pth").write_bytes(b"new")
    (trained / "crlite.yaml").write_text("new")
    flow.models["base"] = trained
    # the frame has a saved pair and a camera box nobody decided
    rects = [{"rect": {"x1": 1, "y1": 2, "x2": 9, "y2": 10}, "obj_id": "7", "obj_attr": "t1"},
             {"rect": {"x1": 11, "y1": 2, "x2": 19, "y2": 10}, "obj_id": "", "obj_attr": "t2"}]
    (refine / "label_fusion" / "camera" / "front").mkdir(parents=True)
    (refine / "label_fusion" / "camera" / "front" / "000.json").write_text(json.dumps({"objs": rects}))
    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: SimpleNamespace(wait=lambda timeout=None: 0))
    flow.pair("refine")
    out = capsys.readouterr()
    assert (old / "crlite.pth").read_bytes() == b"new"
    kept = [p for p in old.parent.iterdir() if p.name.startswith("front_model_replaced_")]
    assert len(kept) == 1 and (kept[0] / "crlite.pth").read_bytes() == b"old"
    assert "The model that was there before is kept in" in out.out
    assert "1 camera boxes have no decision in 1 frames that hold a saved pair (frame 1 in the pairing " \
           "window)" in out.err
    shutil.rmtree(kept[0])


def test_training_without_hdf5_explains_next_step(tmp_path):
    from xcalib.gui.workflow import train_base_from_caches

    with pytest.raises(ValueError, match="HDF5 generation cell"):
        train_base_from_caches(tmp_path / "missing", epochs=1)
    assert not (tmp_path / "missing").exists()


class _Running:
    """A child process that runs until it is ended; waiting for it is where the ■ button
    (KeyboardInterrupt) reaches the step."""

    def __init__(self, stdout=None):
        self.pid, self.ended, self.stdout = 4242, False, stdout

    def poll(self):
        return -9 if self.ended else None

    def wait(self, timeout=None):
        if not self.ended:
            raise KeyboardInterrupt
        return -9

    def kill(self):
        self.ended = True


def _ending(process):
    """subprocess.run as stop_process_tree uses it on Windows: taskkill ends the process."""
    def run(args, **kwargs):
        assert args[:2] == ["taskkill", "/PID"] and "/T" in args
        process.ended = True
    return run


def test_a_stopped_training_ends_its_process_and_says_so(tmp_path, monkeypatch):
    from xcalib.gui.workflow import train_base_from_caches

    run = tmp_path / "run"
    run.mkdir()
    for name in ("train.h5", "validation.h5"):
        (run / name).write_bytes(b"cache")
    process = _Running()
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("subprocess.run", _ending(process))
    with pytest.raises(KeyboardInterrupt):   # the notebook's training reports its progress
        train_base_from_caches(run, epochs=1, progress=lambda message: None)
    assert process.ended     # the training does not go on unseen
    monkeypatch.undo()

    flow = Workflow()
    flow.repo = tmp_path
    flow.scenes["base"] = (_scene(tmp_path / "site_base"), "front")
    monkeypatch.setattr("xcalib.gui.workflow.prepare_training_caches",
                        lambda *args, **kwargs: {"train_frames": 4, "validation_frames": 1})
    monkeypatch.setattr(flow, "_training_device", lambda: "the CPU")

    def stopped(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("xcalib.gui.workflow.train_base_from_caches", stopped)
    with pytest.raises(Stop, match="training was stopped before it finished, and no model was written"):
        flow.train("base")
    monkeypatch.setattr("xcalib.gui.workflow.prepare_training_caches", stopped)
    with pytest.raises(Stop, match="export was stopped before it finished"):
        flow.train("base")
    assert "base" not in flow.models


def test_a_stopped_model_run_ends_its_process(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    class Lines:   # the model's report lines; the ■ button arrives while they are read
        def __iter__(self):
            raise KeyboardInterrupt

    flow = Workflow()   # before subprocess is replaced: on Windows it checks the numerics once
    flow.repo = tmp_path
    process = _Running(stdout=Lines())
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("subprocess.run", _ending(process))
    said = []
    with pytest.raises(Stop, match="model run was stopped before it finished; nothing was saved"):
        flow._model_pairs(tmp_path, "front", said.append)
    assert process.ended and said[-1] == "The model run was stopped."


def test_scenes_open_in_sustechpoints_for_looking_only(tmp_path, monkeypatch, capsys):
    import contextlib
    import tempfile
    import urllib.request

    from xcalib.gui import customer

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(customer, "_SUSTECH", {})
    flow = Workflow()
    flow.repo = tmp_path
    base = _scene(tmp_path / "site" / "site_base")
    refine = _scene(tmp_path / "site" / "refine" / "site_refine")
    launcher = tmp_path / "scripts" / "launch_sustech.py"
    launcher.parent.mkdir()
    launcher.write_text("# the launcher")
    flow.scenes["base"] = (base, "front")
    started, opened, probes = [], [], []

    class Page:   # the launcher: it runs until its input is closed
        def __init__(self, command, **kwargs):
            self.command, self.kwargs, self.closed = command, kwargs, False
            self.stdin = SimpleNamespace(close=lambda: setattr(self, "closed", True))
            started.append(self)

        def poll(self):
            return 0 if self.closed else None

        def wait(self, timeout=None):
            return 0

    def popen(command, **kwargs):
        if command[1] == "-c":   # the short-lived process that opens the browser
            opened.append(command[-1])
            return SimpleNamespace(poll=lambda: 0)
        return Page(command, **kwargs)

    class Direct:   # the readiness check: straight to the page, never through a proxy
        def __init__(self, *handlers):
            assert any(isinstance(h, urllib.request.ProxyHandler) and not h.proxies for h in handlers)

        def open(self, url, timeout):
            probes.append(url)
            return contextlib.nullcontext()

    def scenes_of(page):
        return [page.command[i + 1] for i, word in enumerate(page.command) if word == "--scene"]

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr(urllib.request, "build_opener", Direct)

    flow.view_sustech("base")
    page = started[-1]
    assert page.command[1] == str(launcher) and scenes_of(page) == [str(base)]
    # the page saves fine-tuned LiDAR boxes into the scene: no --view-only
    assert "--view-only" not in page.command and "--stop-on-stdin-close" in page.command
    assert probes and probes[-1].startswith("http://127.0.0.1:")
    assert opened[-1].startswith("http://127.0.0.1:") and opened[-1].endswith("/?scene=site_base&frame=000")
    out = capsys.readouterr().out
    assert "saves the fine-tuned boxes into the scene" in out and "do not delete a box" in out
    assert "The page holds" not in out
    flow.view_sustech("base")            # the page runs: it is opened again, not started again
    assert len(started) == 1 and len(opened) == 2

    flow.scenes["refine"] = (refine, "front")
    flow.view_sustech("refine")          # both scenes in one page; the page of the base scene alone ends
    assert started[0].closed and len(started) == 2
    assert scenes_of(started[-1]) == [str(refine), str(base)]
    assert opened[-1].endswith("?scene=site_refine&frame=000")
    out = capsys.readouterr().out
    assert "The page that was open is closed" in out
    assert "The page holds the scenes site_refine and site_base" in out and "then a frame in the frame list" in out
    assert re.search(r"site_base: http://127\.0\.0\.1:\d+/\?scene=site_base&frame=000", out)
    flow.view_sustech("base")            # the same two scenes: the same page
    assert len(started) == 2 and opened[-1].endswith("?scene=site_base&frame=000")

    def fails(command, stdout=None, **kwargs):   # the launcher ends at once and says why in its log
        stdout.write(b"usage: launch_sustech.py ...\nlaunch_sustech.py: error: SUSTechPOINTS was not found in X\n")
        return SimpleNamespace(poll=lambda: 2, stdin=None)

    customer._stop_sustech()
    assert started[-1].closed
    monkeypatch.setattr("subprocess.Popen", fails)
    with pytest.raises(Stop, match=r"could not start \(launch_sustech.py: error: SUSTechPOINTS was not found in X\)"):
        flow.view_sustech("base")
    assert "server" not in customer._SUSTECH
    with pytest.raises(Stop, match='view_sustech\\("base"\\)'):
        flow.view_sustech("both")


def test_the_stop_button_never_closes_a_pairing_window(tmp_path, monkeypatch, capsys):
    flow = Workflow()   # before subprocess is replaced: on Windows it checks the numerics once
    flow.repo = tmp_path

    class Window:   # the ■ button arrives while the window is open; the person closes it later
        waits = 0

        def __init__(self, *args, **kwargs):
            windows.append(self)

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return 0

    windows = []
    monkeypatch.setattr("subprocess.Popen", Window)
    flow._window("xcalib.gui.review", tmp_path, "front")
    assert windows[-1].waits == 2      # it was waited for until it closed, never ended
    assert "does not close the pairing window" in capsys.readouterr().out

    def stopped(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("subprocess.run", stopped)   # ■ while a folder dialog is open
    with pytest.raises(Stop, match="nothing was chosen"):
        flow._pick("Select the folder")


def test_parts_of_the_image_take_their_own_preposition():
    from xcalib.gui.customer import _parts_of_the_image

    assert _parts_of_the_image(["left", "middle"]) == "on the left and in the middle"
    assert _parts_of_the_image(["middle"]) == "in the middle"
    assert _parts_of_the_image(["left", "right"]) == "on the left and on the right"
