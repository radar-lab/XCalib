"""Pairs that exist already are copied onto a scene, counted and drawn; training says how far it is."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xcalib.gui.customer import Stop, Workflow
from xcalib.gui.label_import import camera_record
from xcalib.gui.saved_pairs import import_pairs, label_folder, preview, summarize
from xcalib.gui.scene import Scene
from xcalib.gui.workflow import TrainingProgress

CAMERA = "front"


def _box(obj_id, x, y):
    return {"obj_id": obj_id, "obj_type": "Car", "psr": {
        "position": {"x": x, "y": y, "z": 0.0}, "rotation": dict.fromkeys("xyz", 0.0),
        "scale": {"x": 4.0, "y": 2.0, "z": 1.5}}}


def _rect(track, left, obj_id="", **extra):
    return {"obj_type": "Car", "obj_attr": f"t{track}", "obj_id": obj_id, **extra,
            "rect": {"x1": float(left), "y1": 300.0, "x2": left + 120.0, "y2": 380.0}}


def _scene(root, decided=False, frames=4):
    """Per frame: LiDAR boxes 1, 2, 3 and camera boxes t1 (left), t2 (middle), t3 (right, an
    object the LiDAR did not see) and t4. With `decided`: t1-#1, t2-#2, t3 none, t4-#7,
    a LiDAR box only the scene the pairs come from has."""
    for folder in ("lidar", "label", f"camera/{CAMERA}", f"label_fusion/camera/{CAMERA}"):
        (root / folder).mkdir(parents=True)
    for index in range(frames):
        frame = f"{index:03d}"
        boxes = [_box("1", 10, 5), _box("2", 20, 0), _box("3", 30, -8)] + ([_box("7", 5, 20)] if decided else [])
        rects = [_rect(1, 100, "1" if decided else ""), _rect(2, 900, "2" if decided else ""),
                 _rect(3, 1700, no_match=True) if decided else _rect(3, 1700),
                 _rect(4, 500, "7" if decided else "")]
        np.save(root / "lidar" / f"{frame}.npy", np.random.default_rng(index).uniform(-30, 30, (200, 3)).astype(np.float32))
        (root / "label" / f"{frame}.json").write_text(json.dumps(boxes))
        (root / "label_fusion" / "camera" / CAMERA / f"{frame}.json").write_text(json.dumps({"objs": rects}))
        cv2.imwrite(str(root / "camera" / CAMERA / f"{frame}.jpg"), np.full((1080, 1920, 3), 60, np.uint8))
    return Scene(str(root), CAMERA)


def test_pairs_of_a_folder_are_copied_onto_the_same_camera_boxes(tmp_path):
    earlier = _scene(tmp_path / "earlier", decided=True, frames=5)   # one frame more than the scene has
    scene = _scene(tmp_path / "scene")
    scene.apply("003", [{"rect": [900.0, 300.0, 1020.0, 380.0], "lidar_id": "3"}])   # decided here already

    for chosen in (earlier.root, Path(earlier.root) / "label_fusion" / "camera" / CAMERA,
                   Path(earlier.root) / "label_fusion" / "camera"):   # the folder the dialog title names
        assert label_folder(chosen, CAMERA) == Path(earlier.root) / "label_fusion" / "camera" / CAMERA
    report = import_pairs(scene, earlier.root)

    assert report["files"] == 5 and report["other_frames"] == 1 and report["frames"] == 4
    assert report["pairs"] == 7 and report["none"] == 4       # t1 and t2 in four frames, less the kept one
    assert report["kept"] == 1 and report["same"] == 0
    assert report["unknown_lidar"] == 4 and report["unmatched"] == 0
    rects = {r["obj_attr"]: r for r in scene.load_rects("000")}
    assert rects["t1"]["obj_id"] == "1" and rects["t2"]["obj_id"] == "2" and rects["t3"]["no_match"]
    assert rects["t4"]["obj_id"] == "" and "annotator" not in rects["t1"]
    assert {r["obj_attr"]: r["obj_id"] for r in scene.load_rects("003")}["t2"] == "3"   # kept

    again = import_pairs(scene, earlier.root)                 # nothing is left to copy
    # t1, t2 and t3 of four frames hold the folder's decision now; t2 of 003 was decided otherwise
    assert again["pairs"] == 0 and again["same"] == 11 and again["kept"] == 1

    # a person moves LiDAR box #1 from t1 to t4 in the window (t1 is cleared): importing the
    # folder again must not give #1 back to t1, which would pair one LiDAR box twice
    rects = scene.load_rects("000")
    for rect in rects:
        if rect["obj_attr"] == "t1":
            rect["obj_id"] = ""
        elif rect["obj_attr"] == "t4":
            rect["obj_id"] = "1"
    scene.save_rects("000", rects)
    third = import_pairs(scene, earlier.root)
    assert third["pairs"] == 0 and third["taken"] == 1
    held = {r["obj_attr"]: r["obj_id"] for r in scene.load_rects("000")}
    assert held["t4"] == "1" and held["t1"] == ""
    with pytest.raises(ValueError, match="this scene's own"):
        import_pairs(scene, scene.root)
    with pytest.raises(ValueError, match="holds no saved pairs"):
        import_pairs(scene, tmp_path)

    # a camera box that moved a little is still found by its track
    moved = _scene(tmp_path / "moved")
    for frame in moved.frames():
        rects = moved.load_rects(frame)
        rects[0]["rect"]["x1"] += 3.0
        moved.save_rects(frame, rects)
    assert import_pairs(moved, earlier.root)["pairs"] == 8


def test_the_summary_says_where_the_pairs_lie_and_the_picture_colours_them(tmp_path):
    scene = _scene(tmp_path / "scene", decided=True)

    report = summarize(scene)

    assert report["pairs"] == 12 and report["none"] == 4 and report["undecided"] == 0
    assert report["frames_with_pairs"] == 4 and report["frames_ready"] == 4 and report["objects"] == 3
    assert report["thirds"] == [8, 4, 0] and report["per_frame"]["002"] == 3
    assert report["cut"] == 0 and report["wide"] == 0 and report["undecided_beside_pairs"] == 0

    # a pair whose camera box the image edge cuts is counted apart: the homography leaves it out
    rects = scene.load_rects("001")
    rects[0]["rect"].update(x1=0.0)
    scene.save_rects("001", rects)
    report = summarize(scene)
    assert report["cut"] == 1 and report["thirds"] == [7, 4, 0] and report["pairs"] == 12
    # so is a pair whose camera box is wide: an object close to the camera
    rects = scene.load_rects("002")
    rects[0]["rect"].update(x2=rects[0]["rect"]["x1"] + 400.0)
    scene.save_rects("002", rects)
    report = summarize(scene)
    assert report["wide"] == 1 and report["cut"] == 1 and sum(report["thirds"]) == 10 and report["pairs"] == 12
    rects[0]["rect"].update(x2=rects[0]["rect"]["x1"] + 120.0)
    scene.save_rects("002", rects)

    image = preview(scene, "000")
    assert image.shape == (540, 960 + 540, 3)
    plain = _scene(tmp_path / "plain")
    assert not np.array_equal(image, preview(plain, "000"))
    suggested = preview(plain, "000", suggestions=[{"rect": [100.0, 300.0, 220.0, 380.0],
                                                    "lidar_id": "1", "score": 0.93}])
    assert not np.array_equal(suggested, preview(plain, "000"))
    assert plain.load_rects("000")[0]["obj_id"] == ""         # a suggestion is drawn, never saved


def test_the_notebook_steps_copy_and_show_pairs(tmp_path, monkeypatch, capsys):
    earlier = _scene(tmp_path / "earlier", decided=True)
    _scene(tmp_path / "site_base")
    flow = Workflow()
    flow.scenes["base"] = (tmp_path / "site_base", CAMERA)
    picked = [tmp_path, Path(earlier.root)]
    monkeypatch.setattr(flow, "_pick", lambda title, initial=None, files=None: picked.pop(0))

    with pytest.raises(Stop, match="No pairs were copied: .*holds no saved pairs"):
        flow.import_pairs("base")
    flow.import_pairs("base")
    out = capsys.readouterr()
    assert 'Copied 8 pairs and 4 "no LiDAR box" marks into 4 frames' in out.out
    assert "Left out: 4 pairs name a LiDAR box this scene does not have." in out.out
    assert "Saved pairs: 8 in 4 of 4 frames, on 2 different LiDAR objects; 4 frames have two or more." in out.out
    assert "Where the pairs that the homography can use lie in the image: 4 left, 4 middle, 0 right." in out.out
    # t4 is left without a decision beside the copied pairs (its LiDAR box is not in this scene)
    assert "4 camera boxes have no decision in 4 frames that hold a saved pair (frame 1, 2, 3, 4" in out.err
    assert "Few pairs lie on the right of the image" in out.err

    flow.view_pairs("base")
    out = capsys.readouterr()
    assert "Saved pairs: 8 in 4 of 4 frames" in out.out and "run the pairing cell" in out.out

    # the same folder a second time: its pairs are in the scene already, which is no mismatch
    picked[:] = [Path(earlier.root)]
    flow.import_pairs("base")
    out = capsys.readouterr()
    assert "Already in this scene: 12 camera boxes hold the folder's decision already" in out.out
    assert "Nothing new to copy: this scene holds the folder's pairs already." in out.out
    assert "Nothing was copied" not in out.err

    # the optional cell of Part 2 copies pairs you already have into the refine scene the same way
    _scene(tmp_path / "site_refine")
    flow.scenes["refine"] = (tmp_path / "site_refine", CAMERA)
    picked[:] = [Path(earlier.root)]
    flow.import_pairs("refine")
    out = capsys.readouterr()
    assert 'Copied 8 pairs and 4 "no LiDAR box" marks into 4 frames' in out.out
    assert Scene(str(tmp_path / "site_refine"), CAMERA).load_rects("000")[0]["obj_id"] == "1"


def test_an_intrinsic_file_is_read_in_the_forms_people_keep_it_in(tmp_path):
    matrix = [[50, 0, 32], [0, 60, 24], [0, 0, 1]]
    forms = {
        "undistortion.json": {"K_undistort": matrix, "image_processing": {"image_size": [64, 48]}},
        "scene.json": {"intrinsic": sum(matrix, []), "image_size": [64, 48], "image_state": "rectified",
                       "distortion": [0.0] * 5},
        "matrix.json": {"camera_matrix": matrix},
    }
    for name, content in forms.items():
        (tmp_path / name).write_text(json.dumps(content))
        record = camera_record(tmp_path / name)
        assert np.allclose(np.ravel(record["intrinsic"]), np.ravel(matrix))
        assert record["image_size"] == (None if name == "matrix.json" else [64, 48])
    for name, content, reason in (
            ("cropped.json", {"K_undistort": matrix, "image_processing": {"crop_to_valid_roi": True}}, "Cropped"),
            ("distorted.json", {"intrinsic": matrix, "distortion": [-0.3, 0.1, 0, 0, 0]}, "rectified camera"),
            ("raw.json", {"intrinsic": matrix, "image_state": "distorted"}, "rectified camera"),
            ("other.json", {"extrinsic": [1, 2, 3]}, "holds no camera matrix"),
            ("list.json", [1, 2, 3], "holds no camera matrix")):
        (tmp_path / name).write_text(json.dumps(content))
        with pytest.raises(ValueError, match=reason):
            camera_record(tmp_path / name)
    (tmp_path / "text.json").write_text("not json")
    with pytest.raises(ValueError, match="is not a JSON file"):
        camera_record(tmp_path / "text.json")


def test_a_training_log_is_read_as_progress(tmp_path):
    log = tmp_path / "training.log"
    log.write_text("")
    watch = TrainingProgress(log, epochs=30)
    watch.started = 1000.0
    assert watch.line(now=1010.0) is None                      # nothing has run yet
    with log.open("a") as stream:
        stream.write("train:   0%|          | 0/689 [00:00<?, ?it/s]\rtrain:  50%|#####     | 345/689 [01:00<01:00]")
    assert watch.line(now=1060.0) == "Epoch 1 of 30: 50%; about 59 min left"
    with log.open("a") as stream:
        stream.write("\rtrain: 100%|##########| 689/689\n2026 | INFO | epoch 1/30 | lr=1.00e-04 | train_loss=2.05 "
                     "| val top1=85.65% mrr=0.9241\nsaved new best\ntrain:  25%|##        | 172/689")
    assert watch.line(now=1150.0) == ("Epoch 2 of 30: 25%; validation top-1 after epoch 1: 85.65%; "
                                      "about 58 min left")
    with log.open("a") as stream:
        stream.write("\n" + "".join(f"x | epoch {n}/30 | lr=1e-4 | val top1=9{n % 10}.00% mrr=0.9\n"
                                    for n in range(2, 31)))
    assert watch.line(now=4000.0) == "Epoch 30 of 30 done; validation top-1 90.00%."


def test_a_jupyter_started_without_the_environment_settings_is_told_so(monkeypatch):
    from types import SimpleNamespace

    import xcalib.gui.customer as customer

    asked = []
    monkeypatch.setattr(customer, "_WINDOWS", True)
    monkeypatch.setattr(customer, "_NUMERICS", None)
    monkeypatch.setattr("subprocess.run", lambda command, **kw: asked.append(command) or SimpleNamespace(returncode=3221225477))
    with pytest.raises(Stop, match="pixi run notebook-homography"):
        Workflow()
    with pytest.raises(Stop, match="started without the settings"):
        Workflow()                       # the answer is kept: a fresh process is asked once
    assert len(asked) == 1 and "numpy" in asked[0][-1]
    monkeypatch.setattr(customer, "_NUMERICS", None)
    monkeypatch.setattr("subprocess.run", lambda command, **kw: SimpleNamespace(returncode=0))
    assert Workflow().repo.name


def test_dialogs_and_windows_are_not_started_while_the_jupyter_debugger_is_on(tmp_path, monkeypatch):
    # With the debugger on, every Python process of the kernel waits for it: a dialog would
    # never appear and the cell would stay busy. The step says what to do instead.
    from types import SimpleNamespace

    import xcalib.gui.customer as customer

    started = []
    monkeypatch.setattr("subprocess.run", lambda command, **kw: started.append(command) or SimpleNamespace(
        returncode=0, stdout=json.dumps(str(tmp_path))))
    monkeypatch.setattr(customer, "_NUMERICS", True)       # the environment check is not this test's
    flow = Workflow()
    flow.repo = tmp_path                                   # where train() would write its run folder
    kernel = SimpleNamespace(debugger=SimpleNamespace(is_started=True))
    monkeypatch.setattr(customer, "get_ipython", lambda: SimpleNamespace(kernel=kernel), raising=False)
    flow.scenes["base"] = (tmp_path, CAMERA)
    for step in (lambda: flow._pick("Select the raw data folder"), flow.import_raw,
                 lambda: flow._window("xcalib.gui.review", tmp_path, CAMERA), lambda: flow.train("base")):
        with pytest.raises(Stop, match="debugger is switched on.*switch it off and run this cell again"):
            step()
    assert not (tmp_path / "runs").exists()               # training stopped before its export
    monkeypatch.setattr(customer, "_NUMERICS", None)
    with pytest.raises(Stop, match="debugger is switched on"):
        Workflow()
    assert not started and customer._NUMERICS is None    # nothing was started, nothing noted
    kernel.debugger.is_started = False                    # switched off again: dialogs open
    assert flow._pick("Select the raw data folder") == tmp_path.resolve() and len(started) == 1
