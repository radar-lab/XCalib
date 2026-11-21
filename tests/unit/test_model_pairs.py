"""A trained model's pairs for the undecided camera boxes are offered for confirmation."""

import json
from pathlib import Path

import numpy as np
import pytest

from xcalib.gui import model_pairs
from xcalib.gui.model_pairs import frame_suggestions, open_frames, suggest_scene
from xcalib.gui.scene import FrameReview, Scene, TrackEntry
from xcalib.gui.workflow import install_model

CAMERA = "front"


def _scene(root, frames):
    """``frames``: {frame: (LiDAR ids, [(x1, saved id, no_match)])}; a camera box is known by its x1."""
    for folder in ("lidar", "label", f"label_fusion/camera/{CAMERA}"):
        (root / folder).mkdir(parents=True)
    for frame, (ids, rects) in frames.items():
        np.save(root / "lidar" / f"{frame}.npy", np.zeros((1, 3), np.float32))
        (root / "label" / f"{frame}.json").write_text(json.dumps([
            {"obj_id": lid, "obj_type": "Car", "psr": {"position": dict.fromkeys("xyz", 1.0)}}
            for lid in ids]))
        (root / "label_fusion" / "camera" / CAMERA / f"{frame}.json").write_text(json.dumps({"objs": [
            {"rect": {"x1": x1, "y1": 0, "x2": x1 + 5, "y2": 5}, "obj_id": saved, "no_match": no_match}
            for x1, saved, no_match in rects]}))
    return Scene(str(root), CAMERA)


def _model(folder, weights=b"weights"):
    folder.mkdir(parents=True)
    (folder / "crlite.pth").write_bytes(weights)
    (folder / "crlite.yaml").write_text("model: crlite")
    return folder


def _fake_review(picks):
    """Stands in for the matcher: ``picks`` maps a camera box's x1 to the LiDAR id it suggests."""
    def review(matcher, scene, frame, threshold=0.5, **_):
        tracks = [TrackEntry(key=f"@{i}", rect=[r["rect"][k] for k in ("x1", "y1", "x2", "y2")],
                             applied_id=str(r.get("obj_id") or ""), rejected=bool(r.get("no_match")),
                             suggestion=picks.get(r["rect"]["x1"], ""),
                             options=[{"lidar_id": picks[r["rect"]["x1"]], "score": 0.9}]
                             if r["rect"]["x1"] in picks else [])
                  for i, r in enumerate(scene.load_rects(frame))]
        return FrameReview(frame=frame, tracks=tracks, boxes=scene.load_boxes(frame))
    return review


def test_only_undecided_camera_boxes_get_the_models_pick(tmp_path, monkeypatch):
    scene = _scene(tmp_path, {
        "000": (["1", "2", "4", "4"], [(0, "1", False),    # a saved pair
                                       (10, "", True),     # a person said: no LiDAR box
                                       (20, "", False),    # the model picks the box of the saved pair
                                       (30, "", False),    # a pick that is free
                                       (40, "", False),    # the same pick again
                                       (50, "", False),    # an id two LiDAR boxes share
                                       (60, "", False)]),  # no pick
        "001": (["1"], [(0, "1", False), (10, "", True)]),  # everything decided
        "002": (["1"], [])})
    picks = {0: "2", 10: "2", 20: "1", 30: "2", 40: "2", 50: "4"}
    monkeypatch.setattr("xcalib.gui.scene.match_review", _fake_review(picks))

    assert open_frames(scene) == ["000"]
    assert frame_suggestions(None, scene, "000") == [{"rect": [30.0, 0.0, 35.0, 5.0], "lidar_id": "2", "score": 0.9}]
    said = []
    assert suggest_scene(None, scene, progress=said.append) == {
        "000": [{"rect": [30.0, 0.0, 35.0, 5.0], "lidar_id": "2", "score": 0.9}]}
    assert said == ["The model is pairing the open frames: 0/1"]
    assert suggest_scene(None, scene, frames=["001"]) == {}
    assert scene.load_rects("000")[3]["obj_id"] == ""  # nothing is written to the labels


def test_the_command_writes_what_the_pairing_window_opens_to_confirm(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    scene = _scene(tmp_path / "site", {"000": (["1", "2"], [(0, "", False), (10, "", False)]),
                                       "001": (["1"], [(0, "1", False)])})
    output = tmp_path / "pairs.json"
    with pytest.raises(SystemExit):  # a scene without a model has nothing to suggest with
        model_pairs.main(["--scene", scene.root, "--camera", CAMERA, "--output", str(output)])
    capsys.readouterr()
    _model(Path(scene.root) / "calib" / "xcalib" / f"{CAMERA}_model")
    loaded = []
    monkeypatch.setattr("xcalib.engine.matcher.Matcher.from_pretrained",
                        lambda name, **kwargs: loaded.append((name, kwargs["weights"])) or object())
    monkeypatch.setattr("xcalib.gui.scene.match_review", _fake_review({0: "2"}))

    model_pairs.main(["--scene", scene.root, "--camera", CAMERA, "--output", str(output)])

    assert loaded == [("crlite", str(Path(scene.root) / "calib" / "xcalib" / f"{CAMERA}_model"))]
    assert json.loads(output.read_text()) == {"000": [{"rect": [0.0, 0.0, 5.0, 5.0], "lidar_id": "2", "score": 0.9}]}
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RESULT "))
    assert report == {"open_frames": 1, "frames": 1, "pairs": 1}

    check = tmp_path / "check.json"  # asked for, and the scene has no extrinsic: an empty list
    model_pairs.main(["--scene", scene.root, "--camera", CAMERA, "--output", str(output),
                      "--check-output", str(check)])
    assert json.loads(check.read_text()) == {}
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RESULT "))["doubted"] == 0


def test_pairs_the_extrinsic_does_not_support_are_named_for_a_first_look(tmp_path):
    from xcalib.gui.model_pairs import projection_doubts
    from xcalib.gui.scene import box3d_corners
    from xcalib.gui.workflow import save_calibration

    K = np.array([[800.0, 0, 640], [0, 800, 360], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]  # LiDAR x forward -> camera z

    def box(obj_id, x, y):
        return {"obj_id": obj_id, "obj_type": "Car", "psr": {
            "position": {"x": x, "y": y, "z": 0.0}, "rotation": dict.fromkeys("xyz", 0.0),
            "scale": {"x": 4.0, "y": 2.0, "z": 1.5}}}

    def rect(of):  # the camera box of a LiDAR box: where its corners project
        camera = box3d_corners(of["psr"]) @ pose[:3, :3].T + pose[:3, 3]
        uv = (camera @ K.T)[:, :2] / camera[:, 2:]
        return [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]

    near, beside, away = box("1", 20, 0), box("3", 20, 6), box("9", 40, -9)
    nothing_there = rect(box("x", 25, 12))
    for folder in ("lidar", "label", f"label_fusion/camera/{CAMERA}"):
        (tmp_path / folder).mkdir(parents=True)
    np.save(tmp_path / "lidar" / "000.npy", np.zeros((1, 3), np.float32))
    (tmp_path / "label" / "000.json").write_text(json.dumps([near, beside, away]))
    (tmp_path / "label_fusion" / "camera" / CAMERA / "000.json").write_text(json.dumps({"objs": [
        {"obj_type": "Car", "obj_id": "", "rect": dict(zip(("x1", "y1", "x2", "y2"), r))}
        for r in (rect(near), rect(beside), nothing_there)]}))
    scene = Scene(str(tmp_path), CAMERA)
    pairs = {"000": [{"rect": rect(near), "lidar_id": "1", "score": 0.9},        # the projection agrees
                     {"rect": rect(beside), "lidar_id": "9", "score": 0.8},      # it points to box 3
                     {"rect": nothing_there, "lidar_id": "3", "score": 0.7}]}    # no LiDAR box lies there

    assert projection_doubts(scene, pairs) == {}  # no extrinsic: nothing to judge by
    save_calibration(scene, K, pose)
    assert projection_doubts(scene, pairs) == {"000": [
        {"rect": rect(beside), "note": "model: #9, but #3 projects into this camera box"},
        {"rect": nothing_there, "note": "model: #3, which does not project into this camera box"}]}
    assert projection_doubts(scene, {"000": pairs["000"][:1]}) == {}


def test_a_trained_model_replaces_the_scenes_model_and_the_old_one_is_kept(tmp_path):
    scene = _scene(tmp_path / "site", {"000": (["1"], [])})
    first, second = _model(tmp_path / "base"), _model(tmp_path / "tuned", b"better weights")
    home = Path(scene.root) / "calib" / "xcalib"

    target, kept = install_model(scene, first)
    assert (target, kept) == (home / f"{CAMERA}_model", None)
    (target / "refine.sqlite").write_bytes(b"corpus")       # whatever else lives beside the weights
    assert install_model(scene, first) == (target, None)    # the same weights: nothing moves
    assert (target / "refine.sqlite").exists()

    target, kept = install_model(scene, second)
    assert (target / "crlite.pth").read_bytes() == b"better weights" and not (target / "refine.sqlite").exists()
    assert kept.parent == home and kept.name.startswith(f"{CAMERA}_model_replaced_")
    assert (kept / "crlite.pth").read_bytes() == b"weights" and (kept / "refine.sqlite").read_bytes() == b"corpus"
    assert scene.model_dir() == (str(target), "crlite")     # the kept folder is not a model of the scene

    with pytest.raises(ValueError, match="crlite.pth and crlite.yaml"):
        install_model(scene, tmp_path / "nothing here")
    assert (target / "crlite.pth").read_bytes() == b"better weights"


def test_notebook_step_installs_the_model_and_opens_its_pairs_to_confirm(tmp_path, monkeypatch, capsys):
    from xcalib.gui.customer import Stop, Workflow

    scene = _scene(tmp_path / "site_refine", {"000": (["1", "2"], [(0, "1", False), (10, "", False)])})
    old = _model(Path(scene.root) / "calib" / "xcalib" / f"{CAMERA}_model")
    flow = Workflow()
    flow.repo = tmp_path
    flow.scenes["refine"] = (Path(scene.root), CAMERA)
    flow.models["refine"] = _model(tmp_path / "runs" / "tuned" / "model", b"better weights")
    windows = []
    from types import SimpleNamespace

    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: windows.append(args) or SimpleNamespace(
        wait=lambda timeout=None: 0))   # the pairing window, closed at once
    proposals, check = tmp_path / "pairs.json", tmp_path / "check.json"
    reports = [{"open_frames": 1, "frames": 1, "pairs": 2, "doubted": 1},
               {"open_frames": 1, "frames": 1, "pairs": 1, "doubted": 0},
               {"open_frames": 1, "frames": 0, "pairs": 0}, {"open_frames": 0, "frames": 0, "pairs": 0}]
    monkeypatch.setattr(flow, "_model_pairs", lambda scene, camera, status: (reports.pop(0), proposals, check))

    flow.pair_remaining("refine")

    assert (old / "crlite.pth").read_bytes() == b"better weights"
    kept = [p for p in old.parent.iterdir() if p.name.startswith(f"{CAMERA}_model_replaced_")]
    assert len(kept) == 1 and (kept[0] / "crlite.pth").read_bytes() == b"weights"
    assert len(windows) == 1 and "xcalib.gui.review" in windows[0]
    assert windows[0][-5:] == ["--proposals", str(proposals), "--only-listed", "--check", str(check)]
    assert "--notebook-workflow" in windows[0] and "--accept-calibration" not in windows[0]
    out = capsys.readouterr().out
    assert f"The model that was there before is kept in {kept[0]}" in out
    assert "suggests 2 pairs in 1 of the 1 frames" in out and "NOT saved yet" in out
    assert "1 of them do not fit the scene's extrinsic and are marked (check)" in out
    assert "Saved pairs: 1 in 1 frames" in out and "Camera boxes without a decision: 1 in 1 frames." in out

    flow.pair_remaining("refine")  # every pair fits the extrinsic: nothing to mark
    assert windows[1][-3:] == ["--proposals", str(proposals), "--only-listed"]
    assert "(check)" not in capsys.readouterr().out
    flow.pair_remaining("refine")  # the model has nothing to offer: no window, and it says so
    assert "suggests no pair in the 1 frames" in capsys.readouterr().out
    flow.pair_remaining("refine")  # nothing is left to decide
    assert "has a decision already" in capsys.readouterr().out
    assert len(windows) == 2 and len(list(old.parent.iterdir())) == 2  # the same model is not installed twice

    def locked(scene, source):
        raise PermissionError("in use")

    monkeypatch.setattr("xcalib.gui.workflow.install_model", locked)
    with pytest.raises(Stop, match="Close every pairing window"):
        flow.pair_remaining("refine")
