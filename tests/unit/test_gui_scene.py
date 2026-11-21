"""
GUI scene-IO tests (Qt-free): SUSTechPOINTS-style reading/writing, the
rotation-aware box extents, and the review assembly against a stub matcher.

Run with:
    pixi run python -m pytest tests/unit/test_gui_scene.py -q
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from xcalib import MatchResult
from xcalib.gui.scene import (
    Scene,
    box3d_to_extents,
    load_points,
    load_points_xyz,
    match_review,
)


# ------------------------------------------------------------ scene fixture --

def _psr(x, y, z=0.0, sx=4.0, sy=2.0, sz=1.5, rz=0.0):
    return {"position": {"x": x, "y": y, "z": z},
            "scale": {"x": sx, "y": sy, "z": sz},
            "rotation": {"x": 0.0, "y": 0.0, "z": rz}}


def _write_scene(tmp_path, camera="cam", frames=("f0",)):
    root = tmp_path / "S"
    (root / "lidar").mkdir(parents=True)
    (root / "camera" / camera).mkdir(parents=True)
    (root / "label").mkdir(parents=True)
    (root / "label_fusion" / "camera" / camera).mkdir(parents=True)
    (root / "calib" / "camera").mkdir(parents=True)

    rng = np.random.default_rng(0)
    for f in frames:
        pts = rng.uniform(-30, 30, size=(500, 3)).astype(np.float32)
        np.save(root / "lidar" / f"{f}.npy", pts)
        import cv2

        img = np.full((120, 160, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(root / "camera" / camera / f"{f}.jpg"), img)
        (root / "label" / f"{f}.json").write_text(json.dumps({"objs": [
            {"obj_id": "7", "obj_type": "Car", "psr": _psr(20.0, 2.0)},
            {"obj_id": "9", "obj_type": "Car", "psr": _psr(25.0, -8.0)},
        ]}))
        (root / "label_fusion" / "camera" / camera / f"{f}.json").write_text(json.dumps({
            "objs": [
                {"rect": {"x1": 10, "y1": 10, "x2": 50, "y2": 40},
                 "obj_id": "", "obj_attr": "t4", "obj_type": "Car"},
                {"rect": {"x1": 80, "y1": 20, "x2": 120, "y2": 60},
                 "obj_id": "7", "obj_attr": "t3", "obj_type": "Car"},
            ]}))
    K = [800.0, 0, 80, 0, 800.0, 60, 0, 0, 1]
    E = [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]  # lidar->cam, t=0
    (root / "calib" / "camera" / f"{camera}.json").write_text(json.dumps({
        "intrinsic": K, "extrinsic": E}))
    return Scene(str(root), camera)


# ------------------------------------------------------------------ basics --

def test_scene_discovery_and_labels(tmp_path):
    scene = _write_scene(tmp_path, frames=("f0", "f1"))
    assert scene.frames() == ["f0", "f1"]
    assert scene.image_path("f0").endswith("f0.jpg")
    assert scene.lidar_path("f1").endswith("f1.npy")
    assert scene.model_dir() is None
    boxes = scene.load_boxes("f0")
    rects = scene.load_rects("f0")
    assert [b["obj_id"] for b in boxes] == ["7", "9"]
    assert Scene.rects_xyxy(rects).shape == (2, 4)
    assert Scene.boxes_extents(boxes).shape == (2, 6)

    calib = scene.calibration()
    assert calib is not None and calib.success
    # a point straight ahead projects to the principal point (image center)
    uv, depth = calib.project(np.array([[20.0, 0.0, 0.0]]))
    assert depth[0] > 0
    assert uv[0, 0] == pytest.approx(80.0, abs=1e-6)
    assert uv[0, 1] == pytest.approx(60.0, abs=1e-6)


def test_box_extents_are_rotation_aware():
    aabb = box3d_to_extents(_psr(0, 0, 0, sx=4, sy=2, sz=1))
    assert aabb[3] - aabb[0] == pytest.approx(4.0)
    assert aabb[4] - aabb[1] == pytest.approx(2.0)
    # rotate 90° about z: the x/y extents swap
    rot = box3d_to_extents(_psr(0, 0, 0, sx=4, sy=2, sz=1, rz=np.pi / 2))
    assert rot[3] - rot[0] == pytest.approx(2.0)
    assert rot[4] - rot[1] == pytest.approx(4.0)


def test_load_points_formats(tmp_path):
    pts = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.save(tmp_path / "a.npy", pts)
    assert np.allclose(load_points_xyz(str(tmp_path / "a.npy")), pts)

    kitti = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
    kitti.tofile(tmp_path / "b.bin")
    assert np.allclose(load_points_xyz(str(tmp_path / "b.bin")), pts)

    pcd = ("VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
           "WIDTH 4\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 4\nDATA ascii\n"
           + "\n".join(" ".join(str(v) for v in row) for row in pts))
    (tmp_path / "c.pcd").write_text(pcd)
    assert np.allclose(load_points_xyz(str(tmp_path / "c.pcd")), pts)
    (tmp_path / "broken.pcd").write_text("FIELDS x y z\n")
    with pytest.raises(ValueError, match="incomplete PCD header"):
        load_points_xyz(str(tmp_path / "broken.pcd"))
    # Sensors store 1- and 2-byte fields beside the float coordinates.
    mixed = np.zeros(4, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("reflectivity", "u1"),
                               ("ring", "<u2"), ("intensity", "<f4")])
    mixed["x"], mixed["y"], mixed["z"] = pts.T
    mixed["ring"], mixed["intensity"] = 7, [1, 2, 3, 4]
    header = ("VERSION .7\nFIELDS x y z reflectivity ring intensity\nSIZE 4 4 4 1 2 4\n"
              "TYPE F F F U U F\nCOUNT 1 1 1 1 1 1\nWIDTH 4\nHEIGHT 1\nPOINTS 4\nDATA binary\n").encode()
    (tmp_path / "d.pcd").write_bytes(header + mixed.tobytes())
    assert np.allclose(load_points(str(tmp_path / "d.pcd")), np.c_[pts, [1, 2, 3, 4]])
    (tmp_path / "short.pcd").write_bytes(header + mixed[:3].tobytes())
    with pytest.raises(ValueError, match="shorter"):
        load_points_xyz(str(tmp_path / "short.pcd"))


# ------------------------------------------------------------------- apply --

def test_apply_accept_reject_supersede(tmp_path):
    scene = _write_scene(tmp_path)
    r = scene.apply("f0", [{"rect": [10, 10, 50, 40], "lidar_id": "9"}])
    assert r == {"written": 1, "marked": 0}
    rects = scene.load_rects("f0")
    assert rects[0]["obj_id"] == "9" and "no_match" not in rects[0]

    # reject: records no_match, clears the id
    r = scene.apply("f0", [{"rect": [10, 10, 50, 40], "none": True}])
    assert r == {"written": 0, "marked": 1}
    rects = scene.load_rects("f0")
    assert rects[0]["no_match"] is True and rects[0]["obj_id"] == ""

    # re-rejecting is a no-op; accepting supersedes the rejection
    assert scene.apply("f0", [{"rect": [10, 10, 50, 40], "none": True}])["marked"] == 0
    r = scene.apply("f0", [{"rect": [10, 10, 50, 40], "lidar_id": "7"}])
    assert r["written"] == 1
    rects = scene.load_rects("f0")
    assert rects[0]["obj_id"] == "7" and "no_match" not in rects[0]

    # unknown rect coordinates are skipped
    assert scene.apply("f0", [{"rect": [1, 2, 3, 4], "lidar_id": "9"}]) == \
        {"written": 0, "marked": 0}


def test_a_saved_pair_clears_the_machine_mark_of_a_draft(tmp_path):
    """A camera box imported as a draft carries an annotator until a decision is saved."""
    scene = _write_scene(tmp_path)
    path = tmp_path / "S" / "label_fusion" / "camera" / "cam" / "f0.json"
    data = json.loads(path.read_text())
    data["objs"][0]["annotator"] = "vendor-cam-draft"
    path.write_text(json.dumps(data))

    scene.apply("f0", [{"rect": [10, 10, 50, 40], "lidar_id": "9", "annotator": "machine"}])

    saved = scene.load_rects("f0")[0]
    assert saved["obj_id"] == "9" and "annotator" not in saved


# ----------------------------------------------------------- review builder --

class _StubMatcher:
    """Returns a fixed similarity; wired like Matcher.match(one_to_one=True)."""

    def __init__(self, sim):
        self.sim = np.asarray(sim, dtype=np.float32)

    def match(self, image, points, b2d, b3d, top_k=1, match_threshold=0.0,
              one_to_one=False, calibration=None, validate="off", **_kw):
        from xcalib.assign import geometric_gate, one_to_one_assign
        from xcalib.oneshot.calibration import bbox3d_centers

        gmask = gdist = None
        if calibration is not None:
            gmask, gdist = geometric_gate(b2d, bbox3d_centers(b3d), calibration)
        assignment = []
        if one_to_one:
            gated = np.where(gmask, self.sim, -1e9) if gmask is not None else self.sim
            assignment = [(i, j, float(self.sim[i, j]))
                          for i, j, _ in one_to_one_assign(gated, threshold=match_threshold)
                          if gmask is None or gmask[i, j]]
        return MatchResult(
            similarity=self.sim,
            top_indices=self.sim.argmax(axis=1).astype(np.int64),
            kept_2d_indices=np.arange(self.sim.shape[0], dtype=np.int64),
            kept_3d_indices=np.arange(self.sim.shape[1], dtype=np.int64),
            assignment=assignment, geo_compatible=gmask, proj_px=gdist,
        )


def test_match_review_assembles_tracks(tmp_path):
    scene = _write_scene(tmp_path)
    # The model prefers box "9" for t4 (row 0): sim row0 = [0.2, 0.9]. Box "9"
    # projects far from rect t4, which is shown to the reviewer and decides
    # nothing: the suggestion is the model's alone.
    review = match_review(_StubMatcher([[0.2, 0.9], [0.95, 0.1]]), scene, "f0",
                          threshold=0.1)
    assert review.n_rects == 2 and review.n_boxes == 2
    by_key = {t.key: t for t in review.tracks}
    assert set(by_key) == {"3", "4"}

    t3 = by_key["3"]
    assert t3.applied_id == "7"          # already saved in the labels
    assert t3.suggestion == "7"
    assert [o["lidar_id"] for o in t3.options][0] == "7"

    t4 = by_key["4"]
    assert t4.applied_id == "" and not t4.rejected
    opt9 = next(o for o in t4.options if o["lidar_id"] == "9")
    assert opt9["geo_ok"] is False       # the hint is still there
    assert t4.suggestion == "9"          # and does not overrule the model
    # candidates are listed in the model's order, whatever the geometry says
    assert [o["lidar_id"] for o in t4.options] == ["9", "7"]


def test_a_camera_box_without_a_partner_does_not_take_the_box_of_a_pair(tmp_path):
    scene = _write_scene(tmp_path)
    # Rows are t4, t3; columns are boxes "7", "9". t3 belongs to "7" (0.9) and
    # t4 has no partner (0.45 and 0.05, both under the threshold). The largest
    # total over all four entries is t4-"7" + t3-"9" = 1.05, which would hand
    # t3 the wrong box and leave t4 with a pair that is dropped afterwards.
    # Only entries at or above the threshold take part, so t3 keeps "7".
    review = match_review(_StubMatcher([[0.45, 0.05], [0.9, 0.6]]), scene, "f0",
                          threshold=0.5)
    by_key = {t.key: t for t in review.tracks}
    assert by_key["3"].suggestion == "7"
    assert by_key["4"].suggestion == ""


def test_match_review_empty_frame(tmp_path):
    scene = _write_scene(tmp_path)
    os.remove(os.path.join(scene.root, "label", "f0.json"))
    review = match_review(_StubMatcher(np.zeros((2, 2))), scene, "f0")
    assert len(review.tracks) == 2 and review.n_boxes == 0
    assert all(not track.options for track in review.tracks)


class _DropBoxMatcher:
    """Simulates data/crops.py dropping a point-less 3D box: box index 1 ('9')
    has no LiDAR return, so it is absent from kept_3d_indices and the similarity
    has a single column for the surviving box 0 ('7')."""

    def match(self, image, points, b2d, b3d, top_k=1, match_threshold=0.0,
              one_to_one=False, calibration=None, validate="off", **_kw):
        n2 = int(b2d.shape[0])
        sim = np.full((n2, 1), 0.3, dtype=np.float32)
        return MatchResult(
            similarity=sim,
            top_indices=np.zeros(n2, dtype=np.int64),
            kept_2d_indices=np.arange(n2, dtype=np.int64),
            kept_3d_indices=np.array([0], dtype=np.int64),   # box 1 ('9') dropped
            assignment=[], geo_compatible=None, proj_px=None,
        )


def test_match_review_keeps_pointless_boxes_selectable(tmp_path):
    # A 3D box whose expanded crop has no points is dropped by the matcher, so it
    # never enters kept_3d_indices. It must still be a selectable manual candidate
    # (a pedestrian with no LiDAR return is the canonical case) or that region has
    # no correspondence for calibration/refine. It appears at score 0 and never as
    # an automatic suggestion.
    scene = _write_scene(tmp_path)
    review = match_review(_DropBoxMatcher(), scene, "f0", threshold=0.1)
    assert review.tracks, "kept 2D rects should still produce tracks"
    for t in review.tracks:
        ids = [o["lidar_id"] for o in t.options]
        assert "9" in ids, "the point-less box must remain selectable"
        assert "7" in ids, "the surviving box is still scored normally"
        opt9 = next(o for o in t.options if o["lidar_id"] == "9")
        assert opt9["score"] == 0.0
        assert t.suggestion != "9", "a score-0 manual candidate must not auto-propose"


# -------------------------------------------------------- offscreen Qt smoke --

def test_review_window_offscreen(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")  # skips when Qt's system libraries are missing
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import FrameReview, TrackEntry

    scene = _write_scene(tmp_path)
    app = QApplication.instance() or QApplication([])
    win = ReviewWindow(scene)     # worker fails to load a model (none) — handled
    try:
        review = FrameReview(frame="f0", n_rects=2, n_boxes=2, boxes=scene.load_boxes("f0"),
                             tracks=[
                                 TrackEntry(key="3", rect=[80, 20, 120, 60], applied_id="7",
                                            suggestion="7",
                                            options=[{"lidar_id": "7", "lidar_type": "Car",
                                                      "score": 0.95, "proj_px": 4.0,
                                                      "geo_ok": True}]),
                                 TrackEntry(key="4", rect=[10, 10, 50, 40],
                                            options=[{"lidar_id": "9", "lidar_type": "Car",
                                                      "score": 0.9, "proj_px": 700.0,
                                                      "geo_ok": False}]),
                             ])
        import cv2

        image = cv2.cvtColor(cv2.imread(scene.image_path("f0")), cv2.COLOR_BGR2RGB)
        points = load_points_xyz(scene.lidar_path("f0"))
        win._on_review(review, image, points)
        app.processEvents()

        # seeding: applied id locks in; the geo-blocked suggestion never auto-fills
        assert win.assign["3"] == "7"
        assert win.assign.get("4") is None
        # picking + one-to-one: assigning 7 to t4 steals it from t3
        win._pick("4", "7")
        assert win.assign["4"] == "7" and win.assign["3"] is None
        # Stealing an already saved ID also clears the old association on disk.
        keys = {d["_key"]: d for d in win._decisions()}
        assert "4" in keys and keys["4"]["lidar_id"] == "7"
        assert keys["3"]["clear"] is True
        win._bank_current()
        win._on_review(review, image, points)
        assert win.assign["4"] == "7" and win.assign["3"] is None
        assert {d["_key"] for d in win._decisions()} == {"3", "4"}
        # explicit rejection stages a none decision
        win._pick("4", None)
        keys = {d["_key"]: d for d in win._decisions()}
        assert keys["4"].get("none") is True
    finally:
        win.worker.submit("quit")
        win.worker.wait(3000)
        win.close()


def test_a_proposal_left_as_it_is_is_saved_by_apply(tmp_path, monkeypatch):
    """Apply saves a frame as it is shown: the model's suggestion counts as right unless it is changed."""
    pytest.importorskip("PySide6.QtWidgets")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import FrameReview, TrackEntry

    scene = _write_scene(tmp_path, frames=("f0", "f1", "f2", "f3"))
    app = QApplication.instance() or QApplication([])
    win = ReviewWindow(scene)

    def show(frame):
        win.cur = frame
        win._on_review(FrameReview(frame=frame, boxes=scene.load_boxes(frame), tracks=[
            TrackEntry(key="4", rect=[10, 10, 50, 40], suggestion="9", options=[
                {"lidar_id": "9", "score": 0.9, "proj_px": None, "geo_ok": True}])]),
            np.zeros((120, 160, 3), np.uint8), None)

    try:
        show("f0")
        assert win.assign["4"] == "9" and "4" not in win.touched
        assert [(d["_key"], d["lidar_id"]) for d in win._decisions()] == [("4", "9")]
        assert scene.load_rects("f0")[0]["obj_id"] == ""   # shown is not saved: Apply saves
        win._apply_frame()
        assert scene.load_rects("f0")[0]["obj_id"] == "9"

        show("f1")
        win._pick("4", None)                                 # a wrong suggestion is changed, here to "none"
        assert [(d["_key"], d.get("none")) for d in win._decisions()] == [("4", True)]
        show("f2")                                           # looked at and left as it is
        assert sorted(win.staged) == ["f1", "f2"]
        show("f2")
        assert "4" not in win.touched and win.assign["4"] == "9"   # still the model's suggestion
        win._apply_all()                                     # every frame that was looked at
        saved = {frame: scene.load_rects(frame)[0] for frame in ("f1", "f2", "f3")}
        assert saved["f1"]["no_match"] is True and saved["f1"]["obj_id"] == ""
        assert saved["f2"]["obj_id"] == "9"
        assert saved["f3"]["obj_id"] == "" and "no_match" not in saved["f3"]   # never shown: nothing saved
        assert app is not None
    finally:
        win.close()


def test_candidate_buttons_sort_by_distance_and_highlight_clicked_box(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import FrameReview, TrackEntry

    scene = _write_scene(tmp_path)
    app = QApplication.instance() or QApplication([])
    win = ReviewWindow(scene)
    try:
        options = [{"lidar_id": "7", "score": .99, "proj_px": 882, "geo_ok": True},
                   {"lidar_id": "9", "score": .1, "proj_px": 80, "geo_ok": True}]
        review = FrameReview(frame="f0", boxes=scene.load_boxes("f0"), tracks=[
            TrackEntry(key="4", rect=[10, 10, 50, 40], options=options)])
        win._on_review(review, np.zeros((120, 160, 3), np.uint8),
                       np.array([[1., 2., 0.], [2., 3., 0.]]))
        win._select_track("4")
        labels = [win.cand_lay.itemAt(i).widget().text()
                  for i in range(win.cand_lay.count()) if win.cand_lay.itemAt(i).widget()]
        assert labels[2].startswith("#9") and labels[3].startswith("#7")
        win._pick_bev("9")
        assert win.assign["4"] == "9" and win.bev_view._last[3] == "9"
        assert "#9" in win.selection_lbl.text() and "selected" in win.selection_lbl.text()
        win.show()
        win.bev_view.setFocus()
        QTest.keyClick(win.bev_view, Qt.Key_Q)
        assert win.bev_view._angle_deg == 15
        QTest.keyClick(win.bev_view, Qt.Key_Plus)
        assert win.bev_view._zoom > 1
        QTest.keyClick(win.bev_view, Qt.Key_R)
        assert win.bev_view._angle_deg == 0 and win.bev_view._zoom == 1
        assert app is not None
    finally:
        win.close()
