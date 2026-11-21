"""Human-reviewed scene pairs feed existing calibration and HDF5 APIs."""

import json

import cv2
import h5py
import numpy as np
import pytest

from xcalib.data.hdf5_loader import UTCFrameLoader
from xcalib.gui.scene import Scene, match_review
from xcalib.gui.workflow import (
    calibrate_scene, export_training_hdf5, install_new_base, prepare_training_caches,
    train_base_from_caches, training_frame, verified_frame,
)
from xcalib.protocol import CameraIntrinsics


def _scene(tmp_path):
    root = tmp_path / "scene"
    for name in ("lidar", "camera/cam", "label", "label_fusion/camera/cam"):
        (root / name).mkdir(parents=True)
    frame = "1739894706087636"
    cv2.imwrite(str(root / "camera/cam" / f"{frame}.png"), np.zeros((30, 40, 3), np.uint8))
    np.save(root / "lidar" / f"{frame}.npy", np.ones((8, 3), np.float32))
    boxes = [{"obj_id": str(i), "psr": {"position": {"x": i, "y": 0, "z": 4},
              "scale": {"x": 1, "y": 1, "z": 1}, "rotation": {}}} for i in (1, 2)]
    (root / "label" / f"{frame}.json").write_text(json.dumps(boxes), encoding="utf-8")
    rects = [{"rect": {"x1": 1, "y1": 2, "x2": 9, "y2": 10}, "obj_id": "1"},
             {"rect": {"x1": 11, "y1": 2, "x2": 19, "y2": 10}, "obj_id": ""}]
    (root / "label_fusion/camera/cam" / f"{frame}.json").write_text(
        json.dumps({"metadata": "preserve", "objs": rects}), encoding="utf-8")
    return Scene(str(root), "cam"), frame


def test_label_save_and_manual_pairing(tmp_path):
    scene, frame = _scene(tmp_path)
    rects = scene.load_rects(frame)
    rects.append({"rect": {"x1": 20, "y1": 2, "x2": 28, "y2": 10}, "obj_id": ""})
    scene.save_rects(frame, rects)
    data = json.loads((tmp_path / "scene/label_fusion/camera/cam" / f"{frame}.json").read_text())
    assert data["metadata"] == "preserve" and len(data["objs"]) == 3
    review = match_review(None, scene, frame)
    assert len(review.tracks) == 3
    assert {o["lidar_id"] for o in review.tracks[0].options} == {"1", "2"}
    assert review.tracks[0].applied_id == "1"


def test_export_only_verified_pairs_round_trips_through_loader(tmp_path):
    scene, frame = _scene(tmp_path)
    output = tmp_path / "train.h5"
    assert export_training_hdf5(scene, output, [frame]) == 1
    with UTCFrameLoader(output) as loader:
        sample = next(iter(loader))
        assert sample.image.shape == (30, 40, 3)
        assert sample.bboxes_2d.shape == (1, 4)
        assert sample.bboxes_3d.shape == (1, 6)
        assert sample.match_matrix.tolist() == [[True]]
    with h5py.File(output) as h5:
        assert h5["labels/lidar/f00000000"].attrs["source_frame"] == frame
        assert bytes(h5["images/cam/data"][0][:2]) == b"\xff\xd8"
    with pytest.raises(FileExistsError):
        export_training_hdf5(scene, output, [frame])


def test_export_with_negatives_holds_none_boxes_and_the_other_lidar_boxes(tmp_path):
    scene, frame = _scene(tmp_path)
    rects = scene.load_rects(frame)
    rects[1]["no_match"] = True  # saved as "no LiDAR box"
    rects.append({"rect": {"x1": 20, "y1": 2, "x2": 28, "y2": 10}, "obj_id": ""})  # nobody decided
    scene.save_rects(frame, rects)

    kept, boxes, match = training_frame(scene, frame)
    # the pair, then the "none", then the box nobody decided: it counts as having no LiDAR box too
    assert [r["rect"]["x1"] for r in kept] == [1, 11, 20]
    assert [b["obj_id"] for b in boxes] == ["1", "2"]  # the paired LiDAR box, then the frame's other one
    assert match.tolist() == [[1, 0], [0, 0], [0, 0]]

    output = tmp_path / "train.h5"
    assert export_training_hdf5(scene, output, [frame], negatives=True) == 1
    with UTCFrameLoader(output) as loader:
        sample = next(iter(loader))
        assert sample.bboxes_2d.shape == (3, 4) and sample.bboxes_3d.shape == (2, 6)
        assert sample.match_matrix.tolist() == [[True, False], [False, False], [False, False]]


def test_a_frame_with_a_single_camera_box_is_not_a_training_frame(tmp_path):
    import shutil

    scene, frame = _scene(tmp_path)  # one pair, a second camera box nobody decided, two LiDAR boxes
    root = tmp_path / "scene"
    for new_frame, saved_none in (("t1", True), ("t2", True), ("t3", False)):
        for part, ext in (("lidar", ".npy"), ("label", ".json"),
                          ("camera/cam", ".png"), ("label_fusion/camera/cam", ".json")):
            shutil.copyfile(root / part / (frame + ext), root / part / (new_frame + ext))
        if saved_none:
            rects = scene.load_rects(new_frame)
            rects[1]["no_match"] = True  # the second camera box saved as "no LiDAR box"
            scene.save_rects(new_frame, rects)
    scene.save_rects(frame, scene.load_rects(frame)[:1])   # here the pair is the only camera box
    # The boxes of a frame are one training batch and batch normalisation needs two: the
    # frame with the pair alone is left out; the three with a second camera box go in, whether
    # that box was saved as "no LiDAR box" or nobody decided it (it counts as having none).
    summary = prepare_training_caches(scene, tmp_path / "run", negatives=True)
    assert summary["train_frames"] == 2 and summary["validation_frames"] == 1
    assert summary["undecided_camera_boxes"] == 1
    split = json.loads((tmp_path / "run" / "split.json").read_text())   # which frames went where
    assert split["train_frame_ids"] == ["t1", "t2"] and split["validation_frame_ids"] == ["t3"]
    with UTCFrameLoader(tmp_path / "run/train.h5") as loader:
        assert [sample.bboxes_2d.shape[0] for sample in loader] == [2, 2]


def test_absolute_score_training_marks_the_model_config(tmp_path, monkeypatch):
    import subprocess

    def prepare(root):
        root.mkdir(exist_ok=True)
        for name in ("train.h5", "validation.h5"):
            (root / name).write_bytes(b"cache")
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            (root / "training").mkdir()
            (root / "training" / "best.pth").write_bytes(b"weights")
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(subprocess, "run", run)
        return commands

    commands = prepare(tmp_path / "first")
    model = train_base_from_caches(tmp_path / "first", epochs=1, device="cpu", absolute_scores=True)
    assert commands[0][-1] == "--none-loss"
    config = (model / "crlite.yaml").read_text(encoding="utf-8")
    assert config.count("score_mode: absolute") == 1 and "top_k:" in config

    prepare(tmp_path / "again")  # fine-tuning such a model does not write the line twice
    tuned = train_base_from_caches(tmp_path / "again", epochs=1, device="cpu", pretrained_dir=model,
                                   absolute_scores=True)
    assert (tuned / "crlite.yaml").read_text(encoding="utf-8").count("score_mode: absolute") == 1


def test_ambiguous_pairs_are_rejected(tmp_path):
    scene, frame = _scene(tmp_path)
    rects = scene.load_rects(frame)
    rects[1]["obj_id"] = "1"
    scene.save_rects(frame, rects)
    with pytest.raises(ValueError, match="multiple camera boxes"):
        verified_frame(scene, frame)


def test_unpaired_duplicate_lidar_id_does_not_block_export(tmp_path):
    scene, frame = _scene(tmp_path)
    path = tmp_path / "scene/label" / f"{frame}.json"
    boxes = json.loads(path.read_text())
    boxes.append(boxes[1])  # duplicate ID 2 is not paired
    path.write_text(json.dumps(boxes))
    assert len(verified_frame(scene, frame)[0]) == 1
    rects = scene.load_rects(frame)
    rects[1]["obj_id"] = "2"
    scene.save_rects(frame, rects)
    with pytest.raises(ValueError, match="ambiguous"):
        verified_frame(scene, frame)


def test_clearing_pair_keeps_camera_box_undecided(tmp_path):
    scene, frame = _scene(tmp_path)
    scene.apply(frame, [{"rect": [1, 2, 9, 10], "clear": True}])
    rect = scene.load_rects(frame)[0]
    assert rect["obj_id"] == "" and "no_match" not in rect


def test_training_caches_report_invalid_and_split_overlap(tmp_path):
    import shutil

    scene, frame = _scene(tmp_path)
    root = tmp_path / "scene"
    rects = scene.load_rects(frame)
    rects[1]["obj_id"] = "2"
    scene.save_rects(frame, rects)
    for new_frame in ("t1", "t2", "t3"):
        for part, ext in (("lidar", ".npy"), ("label", ".json"),
                          ("camera/cam", ".png"), ("label_fusion/camera/cam", ".json")):
            shutil.copyfile(root / part / (frame + ext), root / part / (new_frame + ext))
    bad = scene.load_rects("t3")
    bad[0]["obj_id"] = "missing"
    scene.save_rects("t3", bad)
    summary = prepare_training_caches(scene, tmp_path / "run")
    assert summary["train_frames"] == 2 and summary["validation_frames"] == 1
    assert len(summary["invalid_frames"]) == 1
    assert set(summary["shared_object_ids"]) == {"1", "2"}
    with UTCFrameLoader(tmp_path / "run/train.h5") as loader:
        assert len(list(loader)) == 2


def test_install_base_never_replaces_existing_model(tmp_path):
    scene, _ = _scene(tmp_path)
    model = tmp_path / "trained"
    model.mkdir()
    (model / "crlite.pth").write_bytes(b"weights")
    (model / "crlite.yaml").write_text("model: crlite")
    target = install_new_base(scene, model)
    assert (target / "crlite.pth").read_bytes() == b"weights"
    with pytest.raises(FileExistsError):
        install_new_base(scene, model)
    assert (target / "crlite.pth").read_bytes() == b"weights"


def test_scene_train_starts_from_exact_pretrained_weights(tmp_path, monkeypatch):
    import sys
    import torch
    from xcalib.engine import scene_train
    from xcalib.models.registry import build_model
    from xcalib.utils.config import load_yaml
    from pathlib import Path

    config = Path(scene_train.__file__).resolve().parents[1] / "cfg/crlite_a9_dataset_r02_s01.yaml"
    source = build_model("crlite", load_yaml(config))
    checkpoint = tmp_path / "crlite.pth"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    monkeypatch.setattr(scene_train, "_require_usable_frame", lambda *args: None)
    called = []
    monkeypatch.setattr(scene_train, "train", lambda model, *a, **kw: called.append(model))
    monkeypatch.setattr(sys, "argv", ["scene_train", "train.h5", "val.h5", str(config),
                                      str(tmp_path / "out"), "1", "cpu", str(checkpoint)])
    scene_train.main()
    assert len(called) == 1
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(called[0].state_dict()[name], tensor)


def test_calibration_does_not_invent_correspondences(tmp_path):
    scene, _ = _scene(tmp_path)
    result = calibrate_scene(scene, CameraIntrinsics(100, 100, 20, 15))
    assert not result.success and result.n_correspondences == 1


def test_calibration_uses_reviewed_centers(tmp_path):
    scene, frame = _scene(tmp_path)
    points = np.array([[x, y, z] for z in (8., 12.) for x in (-2., 0., 2.)
                       for y in (-1., 1.)])
    boxes, rects = [], []
    for i, (x, y, z) in enumerate(points):
        boxes.append({"obj_id": str(i + 1), "psr": {
            "position": {"x": x, "y": y, "z": z},
            "scale": {"x": 1, "y": 1, "z": 1}, "rotation": {}}})
        u, v = 100 * x / z + 20, 100 * y / z + 15
        rects.append({"obj_id": str(i + 1),
                      "rect": {"x1": u - 1, "y1": v - 1, "x2": u + 1, "y2": v + 1}})
    (tmp_path / "scene/label" / f"{frame}.json").write_text(json.dumps(boxes))
    scene.save_rects(frame, rects)
    result = calibrate_scene(scene, CameraIntrinsics(100, 100, 20, 15))
    assert result.success and result.n_correspondences == len(points)
    assert result.reproj_error_px < 0.01
    np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-3)
    np.testing.assert_allclose(result.translation.reshape(3), np.zeros(3), atol=1e-3)


def test_a_replaced_extrinsic_is_kept_outside_the_calibration_folder(tmp_path):
    from xcalib.gui.workflow import save_calibration

    scene, _ = _scene(tmp_path)
    K = np.array([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]])
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = 0.5
    target = save_calibration(scene, K, first, source="first guess")
    assert json.loads(target.read_text())["source"] == "first guess"
    history = target.parent.parent / "camera_history"
    save_calibration(scene, K, first)   # the same extrinsic again: nothing is replaced
    assert not history.exists()
    assert "source" not in json.loads(target.read_text())   # the note described the earlier save
    save_calibration(scene, K, second)
    save_calibration(scene, K, first)   # two replacements, possibly within one second: both are kept
    kept = sorted(history.glob("cam_*.json"))
    assert [json.loads(path.read_text())["extrinsic"][3] for path in kept] == [0.0, 0.5]
    assert json.loads(target.read_text())["extrinsic"][3] == 0.0
    assert [path.name for path in target.parent.iterdir()] == ["cam.json"]  # every file here is read as a calibration


def test_saved_decisions_reach_the_continual_corpus(tmp_path, monkeypatch):
    from pathlib import Path

    from xcalib.continual import open_store
    from xcalib.gui.scene import FrameReview, TrackEntry
    from xcalib.gui.workflow import record_saved_pairs, saved_decisions

    review = FrameReview(frame="f0", tracks=[
        TrackEntry(key="1", rect=[1., 2., 3., 4.], applied_id="10", suggestion="10",
                   options=[{"lidar_id": "10", "score": 0.9}, {"lidar_id": "11", "score": 0.2}]),
        TrackEntry(key="2", rect=[5., 6., 7., 8.], applied_id="9", suggestion="11",
                   options=[{"lidar_id": "11", "score": 0.5}]),
        TrackEntry(key="@3", rect=[0., 0., 1., 1.], rejected=True, options=[]),
        TrackEntry(key="4", rect=[2., 2., 3., 3.], suggestion="11"),  # undecided: not recorded
    ], boxes=[])
    pairs = saved_decisions(review)
    assert [(p.decision, p.lidar_key, p.track) for p in pairs] == [
        ("positive", "id:10", "1"), ("positive", "id:9", "2"), ("no_match", None, None)]
    # the model's own pick is kept, so the corpus can tell a confirmed suggestion from a correction
    assert pairs[0].proposal == {"decision": "positive", "lidar_key": "id:10"}
    assert pairs[1].proposal["lidar_key"] == "id:11" and pairs[2].proposal is None
    assert [c["lidar_key"] for c in pairs[0].candidates] == ["id:10", "id:11"]

    scene, frame = _scene(tmp_path)  # one saved pair (LiDAR box 1) and one undecided camera box
    with pytest.raises(ValueError, match="no model"):
        record_saved_pairs(scene)
    model = Path(scene.root) / "calib" / "xcalib" / "cam_model"
    model.mkdir(parents=True)
    (model / "crlite.pth").write_bytes(b"weights")
    monkeypatch.setattr("xcalib.engine.matcher.Matcher.from_pretrained", lambda *args, **kwargs: object())

    def reviewed(matcher, scene, frame, threshold=0.5, **_):  # the model would have picked box 2
        return FrameReview(frame=frame, boxes=scene.load_boxes(frame), tracks=[
            TrackEntry(key=f"@{i}", rect=[float(r["rect"][k]) for k in ("x1", "y1", "x2", "y2")],
                       applied_id=str(r.get("obj_id") or ""), suggestion="2",
                       options=[{"lidar_id": "2", "score": 0.8}, {"lidar_id": "1", "score": 0.3}])
            for i, r in enumerate(scene.load_rects(frame))])

    monkeypatch.setattr("xcalib.gui.scene.match_review", reviewed)
    said = []
    assert record_saved_pairs(scene, progress=said.append) == 1
    assert said == ["Recording saved decisions: 1/1 frames"]
    assert record_saved_pairs(scene) == 1  # the same frame again replaces its row
    store = open_store(model)
    try:
        rows = list(store.iter_corrections())
    finally:
        store.close()
    assert [(row.frame, row.final_lidar_key, row.provenance) for row in rows] == [
        (frame, "id:1", "human_corrected")]
