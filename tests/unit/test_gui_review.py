"""The pairing window: its views, what it offers, and how it saves a calibration.

The Qt event loop and the worker thread are not unit-coverable; the views, the
staging of offered decisions and the calibration paths are.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")  # xcalib.gui.review imports Qt at module load; skips when Qt's system libraries are missing

from xcalib.gui.review import _rotate_xy
from xcalib.gui.scene import TrackEntry


def test_frame_arrows_work_on_views_and_preserve_editor_keys(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QLineEdit,
                                   QMainWindow, QPushButton, QVBoxLayout, QWidget)
    from xcalib.gui.navigation import FrameNavigation

    app = QApplication.instance() or QApplication([])
    win = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    button, edit, spin, combo = QPushButton("view"), QLineEdit("123"), QDoubleSpinBox(), QComboBox()
    combo.addItems(["a", "b"])
    for widget in (button, edit, spin, combo):
        layout.addWidget(widget)
    win.setCentralWidget(central)
    steps = []
    navigation = FrameNavigation(win, steps.append)
    win.show()
    button.setFocus()
    QTest.keyClick(button, Qt.Key_Right)
    QTest.keyClick(button, Qt.Key_Left)
    edit.setFocus()
    QTest.keyClick(edit, Qt.Key_Left)
    spin.setFocus()
    QTest.keyClick(spin.lineEdit(), Qt.Key_Right)
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key_Right)
    button.setFocus()
    QTest.keyClick(button, Qt.Key_Right, Qt.ShiftModifier)
    assert steps == [1, -1]
    win.close()
    assert navigation is not None and app is not None


def test_candidate_order_prioritizes_projection_error():
    from xcalib.gui.review import _candidate_sort_key

    items = [{"lidar_id": str(i), "proj_px": px, "score": score} for i, (px, score) in
             enumerate([(882, .99), (80, .1), (411, .7), (221, .2),
                        (None, .8), (float("nan"), .1), (341, .3), (164, .0)])]
    ordered = sorted(items, key=_candidate_sort_key)
    assert [item["proj_px"] for item in ordered[:6]] == [80, 164, 221, 341, 411, 882]
    assert [item["lidar_id"] for item in ordered[-2:]] == ["4", "5"]


def test_bev_zoom_rotation_selection_survive_redraw(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QGraphicsPolygonItem
    from xcalib.gui.review import BevView

    app = QApplication.instance() or QApplication([])
    view = BevView()
    points = np.array([[0., 0., 0.], [20., 20., 0.], [10., 10., 0.]])
    boxes = [{"obj_id": "42", "psr": {"position": {"x": 10, "y": 10, "z": 0},
               "scale": {"x": 4, "y": 2, "z": 1}, "rotation": {}}}]
    view.show_frame(points, boxes, {}, selected_id="42")
    polygons = [item for item in view.scene().items() if isinstance(item, QGraphicsPolygonItem)]
    assert any(item.data(0) == "42" and item.data(1) == "selected" for item in polygons)
    view.zoom_view(1.5)
    view.rotate_view(15)
    assert view._zoom == pytest.approx(1.5) and view._angle_deg == 15
    assert view._last[3] == "42"
    view.reset_view()
    assert view._zoom == 1.0 and view._angle_deg == 0
    assert app is not None


@pytest.mark.parametrize("button,modifier", [("middle", False), ("left", True), ("left", False)])
def test_bev_drag_rotates_without_selecting_box(monkeypatch, button, modifier):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import BevView

    app = QApplication.instance() or QApplication([])
    view = BevView()
    view.resize(500, 500)
    view.show()
    view.show_frame(np.array([[0., 0., 0.], [20., 20., 0.], [10., 10., 0.]]), [], {})
    view.zoom_view(1.5)
    clicked = []
    view.box_clicked.connect(clicked.append)
    mouse_button = Qt.MiddleButton if button == "middle" else Qt.LeftButton
    modifiers = Qt.ShiftModifier if modifier else Qt.NoModifier
    QTest.mousePress(view.viewport(), mouse_button, modifiers, QPoint(180, 180))
    QTest.mouseMove(view.viewport(), QPoint(230, 180), delay=10)
    QTest.mouseRelease(view.viewport(), mouse_button, modifiers, QPoint(230, 180))
    assert view._angle_deg != 0
    assert view._zoom == pytest.approx(1.5)
    assert view._drag_pos is None and not clicked
    view.close()
    assert app is not None


def test_bev_click_on_box_still_selects(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import BevView

    app = QApplication.instance() or QApplication([])
    view = BevView()
    view.resize(500, 500)
    view.show()
    boxes = [{"obj_id": "42", "psr": {"position": {"x": 10, "y": 10, "z": 0},
               "scale": {"x": 4, "y": 2, "z": 1}, "rotation": {}}}]
    view.show_frame(np.array([[0., 0., 0.], [20., 20., 0.], [10., 10., 0.]]), boxes, {})
    clicked = []
    view.box_clicked.connect(clicked.append)
    pos = view.mapFromScene(view._polygons[0][1].boundingRect().center())
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=pos)
    assert clicked == ["42"] and view._angle_deg == 0
    view.close()
    assert app is not None


def test_matcher_target_uses_published_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from xcalib.gui.review import _matcher_target
    import xcalib.continual

    model_dir = tmp_path / "site/calib/xcalib/cam_model"
    model_dir.mkdir(parents=True)
    scene = SimpleNamespace(root=str(tmp_path / "site"), camera="cam",
                            model_dir=lambda: (str(model_dir), "crlite"))
    assert _matcher_target(scene) == ("crlite", str(model_dir), None)
    (model_dir / "current.json").write_text("{}")
    revision = SimpleNamespace(model_name="crlite", weights_path=model_dir / "new.pth",
                               config_path=model_dir / "new.yaml")
    monkeypatch.setattr(xcalib.continual, "resolve_model",
                        lambda root, name, camera: SimpleNamespace(revision=revision))
    assert _matcher_target(scene) == ("crlite", revision.weights_path, revision.config_path)


@pytest.mark.parametrize("notebook_workflow", [False, True])
def test_the_window_offers_pairing_and_calibration_only(tmp_path, monkeypatch, notebook_workflow):
    """Installing a model and learning from the pairs are notebook steps."""
    from PySide6.QtWidgets import QAbstractButton, QApplication
    from xcalib.gui.review import MatcherWorker, ReviewWindow
    from xcalib.gui.scene import Scene

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    (tmp_path / "lidar").mkdir()
    app = QApplication.instance() or QApplication([])
    win = ReviewWindow(Scene(str(tmp_path), "cam"), notebook_workflow=notebook_workflow)
    try:
        win.show()
        app.processEvents()
        screen = QApplication.primaryScreen()
        assert screen is None or win.height() <= screen.availableGeometry().height()
        assert all(not w.isHidden() for w in
                   (win.apply_btn, win.apply_all_btn, win.calibrate_btn, win.recalibrate_btn, win.pending_btn))
        texts = " | ".join(button.text() for button in win.findChildren(QAbstractButton))
        for gone in ("Refine", "Train", "HDF5", "Install", "Camera labels"):
            assert gone not in texts
        # The worker only matches frames: nothing in the window can change the scene's model.
        assert not [name for name in dir(MatcherWorker) if "refine" in name or "train" in name]
    finally:
        win.close()


def _paired_scene(root, frames=3):
    """Saved pairs that fit one pose exactly, and the imported intrinsics; returns (K, pose)."""
    import json

    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)  # LiDAR x forward, y left, z up -> camera z forward, x right, y down
    pose[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    pose[:3, 3] = [0.1, 0.5, 0.2]
    rng = np.random.default_rng(0)
    for directory in ("lidar", "label", "label_fusion/camera/cam", "calib/intrinsic"):
        (root / directory).mkdir(parents=True)
    # Imported scenes keep intrinsics outside calib/camera until an extrinsic exists.
    (root / "calib/intrinsic/cam.json").write_text(json.dumps({"intrinsic": K.ravel().tolist()}))
    for frame in (f"{i:03d}" for i in range(frames)):
        np.save(root / "lidar" / f"{frame}.npy", np.zeros((1, 3), np.float32))
        points = np.c_[rng.uniform(8, 30, 8), rng.uniform(-8, 8, 8), rng.uniform(-1, 2, 8)]
        uvw = (points @ pose[:3, :3].T + pose[:3, 3]) @ K.T
        uv = uvw[:, :2] / uvw[:, 2:]
        (root / "label" / f"{frame}.json").write_text(json.dumps([
            {"obj_id": str(i), "obj_type": "Car",
             "psr": {"position": dict(zip("xyz", p)), "rotation": dict.fromkeys("xyz", 0.0),
                     "scale": dict.fromkeys("xyz", 0.2)}} for i, p in enumerate(points)]))
        (root / "label_fusion/camera/cam" / f"{frame}.json").write_text(json.dumps({"objs": [
            {"rect": {"x1": u - 2, "y1": v - 2, "x2": u + 2, "y2": v + 2}, "obj_id": str(i)}
            for i, (u, v) in enumerate(uv)]}))
    return K, pose


def _turn_y(angle):
    """Rotation about the camera's vertical axis: the camera pans."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_a_calibration_that_passes_the_checks_is_saved_without_asking(tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication, QMessageBox
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import Scene
    from xcalib.gui.workflow import save_calibration

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda parent, title, text, *a, **k: asked.append(text) or QMessageBox.No)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append(f"{title}: {text}"))
    app = QApplication.instance() or QApplication([])

    def window(root):
        win = ReviewWindow(Scene(str(root), "cam"), notebook_workflow=True, accept_calibration=True)
        win.show()
        app.processEvents()
        return win

    # 24 pairs are too few to trust a fresh solve unseen: the person is asked, and says no.
    few = tmp_path / "few"
    _paired_scene(few, frames=3)
    win = window(few)
    try:
        win._calibrate(False)
        assert len(asked) == 1 and "Not saved automatically: only 24 of 24 pairs" in asked[0]
        assert not (few / "calib/camera/cam.json").exists()
    finally:
        win.close()

    enough = tmp_path / "enough"
    K, pose = _paired_scene(enough, frames=5)
    del asked[:]
    win = window(enough)
    try:
        win._calibrate(False)  # 40 pairs, all inliers: saved, nobody asked
        saved = json.loads((enough / "calib/camera/cam.json").read_text())
        assert asked == [] and np.allclose(np.array(saved["extrinsic"]).reshape(4, 4), pose, atol=0.02)
        assert win.statusBar().currentMessage().startswith("Calibration saved automatically: 40/40 inliers")
        assert win.calibration_lbl.text().startswith("Calibration saved automatically: 40/40 inliers, ")

        # A pose taken from elsewhere and a little off: Recalibrate replaces it, and its note, unasked.
        start = pose.copy()
        start[:3, 3] += [0.3, -0.2, 0.1]
        save_calibration(win.scene_io, K, start, source="taken from another scene")
        win._calibrate(True)
        saved = json.loads((enough / "calib/camera/cam.json").read_text())
        assert asked == [] and "source" not in saved
        assert np.abs(np.array(saved["extrinsic"]).reshape(4, 4) - pose).max() < 0.05
        win._calibrate(True)  # nothing left to gain: reported, the saved pose stays
        assert len(asked) == 1 and asked[0].startswith("Calibration not accepted")

        # The camera was turned by three degrees since: too far for a nudge, so Recalibrate
        # re-solves, and saves the result because it passes the checks.
        turned = pose.copy()
        turned[:3, :3] = _turn_y(np.radians(3.0)) @ pose[:3, :3]
        save_calibration(win.scene_io, K, turned)
        del asked[:]
        win._calibrate(True)
        saved = np.array(json.loads((enough / "calib/camera/cam.json").read_text())["extrinsic"]).reshape(4, 4)
        assert asked == [] and np.abs(saved - pose).max() < 0.05
        assert "re-solved" in win.calibration_lbl.toolTip()
        # A saved pose of another camera altogether is not replaced by whatever fits these pairs.
        far = pose.copy()
        far[:3, :3] = _turn_y(np.radians(40.0)) @ pose[:3, :3]
        save_calibration(win.scene_io, K, far)
        win._calibrate(True)
        kept = np.array(json.loads((enough / "calib/camera/cam.json").read_text())["extrinsic"]).reshape(4, 4)
        assert np.allclose(kept, far) and len(asked) == 1 and asked[0].startswith("Calibration not accepted")
    finally:
        win.close()


def test_first_calibration_uses_the_imported_intrinsics_without_a_dialog(tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication, QMessageBox
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import Scene

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    K, pose = _paired_scene(tmp_path)
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda parent, title, text, *a, **k: asked.append(title) or QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append(f"{title}: {text}"))
    app = QApplication.instance() or QApplication([])
    win = ReviewWindow(Scene(str(tmp_path), "cam"), notebook_workflow=True)
    try:
        win.show()
        app.processEvents()
        win._calibrate(False)
        assert asked == ["Save calibration"]
        saved = json.loads((tmp_path / "calib/camera/cam.json").read_text())
        assert saved["intrinsic"] == K.ravel().tolist()
        assert np.allclose(np.array(saved["extrinsic"]).reshape(4, 4), pose, atol=0.02)
    finally:
        win.close()


def test_rotate_xy_about_center():
    origin = np.array([0.0, 0.0])
    xy = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 90° CCW about the origin: (1,0)->(0,1), (0,1)->(-1,0)
    np.testing.assert_allclose(_rotate_xy(xy, 90.0, origin), [[0, 1], [-1, 0]], atol=1e-9)
    # 0° is an identity fast-path (returns the input array unchanged)
    assert _rotate_xy(xy, 0.0, origin) is xy
    # the center is a fixed point of the rotation
    c = np.array([2.0, 3.0])
    np.testing.assert_allclose(_rotate_xy(c[None, :], 37.0, c), c[None, :], atol=1e-9)


def test_fit_views_track_the_current_frame_and_never_scroll(monkeypatch):
    # A scene without an explicit rect only grows, so after a frame with far-out items it
    # stayed larger than the next frame and the fit-to-view panes kept toggling their
    # scrollbars (the window resized itself in a loop at 100% CPU).
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from xcalib.gui.review import BevView, ImageView

    QApplication.instance() or QApplication([])

    far_box = {"obj_id": "1", "psr": {"position": {"x": 400.0, "y": 300.0, "z": 0.0},
                                      "scale": {"x": 4.0, "y": 2.0, "z": 1.5},
                                      "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}}}
    points = np.array([[x, y, 0.0, 1.0] for x in range(0, 50, 5) for y in range(-20, 20, 5)])
    bev = BevView()
    bev.show_frame(points, [far_box], {})
    bev.show_frame(points, [], {})

    image = np.zeros((80, 100, 3), dtype=np.uint8)
    wide = [TrackEntry(key="1", rect=[10., 10., 400., 300.])]
    image_view = ImageView()
    image_view.show_frame(image, wide, {}, {}, None)
    image_view.show_frame(image, [], {}, {}, None)

    assert bev.sceneRect() == bev.scene().itemsBoundingRect()
    assert image_view.sceneRect().width() == 100
    assert image_view.sceneRect().height() == 80
    for view in (bev, image_view):
        assert view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert view.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


def test_offered_pairs_wait_for_a_person_to_apply_them(tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import Scene, match_review

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for directory in ("lidar", "label", "label_fusion/camera/cam"):
        (tmp_path / directory).mkdir(parents=True)
    points = np.zeros((1, 3), np.float32)
    for frame in ("000", "001", "002"):
        np.save(tmp_path / "lidar" / f"{frame}.npy", points)
        (tmp_path / "label" / f"{frame}.json").write_text(json.dumps([
            {"obj_id": "7", "obj_type": "Car",
             "psr": {"position": dict.fromkeys("xyz", 1.0), "rotation": dict.fromkeys("xyz", 0.0),
                     "scale": dict.fromkeys("xyz", 1.0)}}]))
        (tmp_path / "label_fusion/camera/cam" / f"{frame}.json").write_text(json.dumps({"objs": [
            {"rect": {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}, "obj_attr": "t3"}]}))
    pair = {"rect": [10.0, 20.0, 50.0, 60.0], "lidar_id": "7"}
    offered = {"000": [{**pair, "lidar_id": "99"}],  # no such LiDAR box
               "001": [pair],
               "002": [pair, {"rect": [1, 2, 3, 4], "lidar_id": "7"}]}  # no such camera box
    app = QApplication.instance() or QApplication([])
    scene = Scene(str(tmp_path), "cam")
    win = ReviewWindow(scene, proposals=offered)
    try:
        assert win.cur == "001" and sorted(win.staged) == ["001", "002"]
        win._on_review(match_review(None, scene, "001"), None, points)
        cards = [win.cards_lay.itemAt(i).widget().text() for i in range(win.cards_lay.count())
                 if win.cards_lay.itemAt(i).widget()]
        assert cards == ["t3 → #7 (to confirm)"]
        assert win.apply_all_btn.text() == "Apply all (2)" and win.pending_btn.isEnabled()
        assert not scene.load_rects("001")[0].get("obj_id")  # shown, not saved
        win._apply_frame()
        assert scene.load_rects("001")[0]["obj_id"] == "7"
        win._next_pending()
        assert win.cur == "002"
        app.processEvents()
    finally:
        win.close()


def test_an_offered_pair_takes_its_lidar_box_from_the_camera_box_that_held_it(tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import Scene, match_review
    from xcalib.gui.workflow import verified_frame

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for directory in ("lidar", "label", "label_fusion/camera/cam"):
        (tmp_path / directory).mkdir(parents=True)
    points = np.zeros((1, 3), np.float32)
    for frame in ("000", "001"):
        np.save(tmp_path / "lidar" / f"{frame}.npy", points)
        (tmp_path / "label" / f"{frame}.json").write_text(json.dumps([
            {"obj_id": "7", "obj_type": "Car",
             "psr": {"position": dict.fromkeys("xyz", 1.0), "rotation": dict.fromkeys("xyz", 0.0),
                     "scale": dict.fromkeys("xyz", 1.0)}}]))
        (tmp_path / "label_fusion/camera/cam" / f"{frame}.json").write_text(json.dumps({"objs": [
            {"rect": {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}, "obj_attr": "t3", "obj_id": "7"},
            {"rect": {"x1": 12.0, "y1": 22.0, "x2": 52.0, "y2": 62.0}, "obj_attr": "t4"}]}))
    offered = {frame: [{"rect": [12.0, 22.0, 52.0, 62.0], "lidar_id": "7"}] for frame in ("000", "001")}
    app = QApplication.instance() or QApplication([])
    scene = Scene(str(tmp_path), "cam")
    win = ReviewWindow(scene, proposals=offered)
    try:
        win._on_review(match_review(None, scene, "000"), None, points)
        cards = [win.cards_lay.itemAt(i).widget().text() for i in range(win.cards_lay.count())
                 if win.cards_lay.itemAt(i).widget()]
        assert cards == ["t3 → —", "t4 → #7 (to confirm)"]
        win._apply_all()  # frame 001 was never opened: its staged pair must free the old holder too
        for frame in ("000", "001"):
            old, new = scene.load_rects(frame)
            assert not old["obj_id"] and not old.get("no_match") and new["obj_id"] == "7"
            assert len(verified_frame(scene, frame)[0]) == 1
        app.processEvents()
    finally:
        win.close()


def test_a_camera_box_without_a_lidar_box_can_be_offered_as_a_rejection(tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import Scene, match_review

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for directory in ("lidar", "label", "label_fusion/camera/cam"):
        (tmp_path / directory).mkdir(parents=True)
    points = np.zeros((1, 3), np.float32)
    np.save(tmp_path / "lidar" / "000.npy", points)
    (tmp_path / "label" / "000.json").write_text(json.dumps([
        {"obj_id": "7", "obj_type": "Car",
         "psr": {"position": dict.fromkeys("xyz", 1.0), "rotation": dict.fromkeys("xyz", 0.0),
                 "scale": dict.fromkeys("xyz", 1.0)}}]))
    (tmp_path / "label_fusion/camera/cam" / "000.json").write_text(json.dumps({"objs": [
        {"rect": {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}, "obj_attr": "t3"},
        {"rect": {"x1": 70.0, "y1": 20.0, "x2": 90.0, "y2": 40.0}, "obj_attr": "t4"},   # a parked car
        {"rect": {"x1": 100.0, "y1": 20.0, "x2": 120.0, "y2": 40.0}, "obj_attr": "t5", "no_match": True}]}))
    offered = {"000": [{"rect": [10.0, 20.0, 50.0, 60.0], "lidar_id": "7"},
                       {"rect": [70.0, 20.0, 90.0, 40.0], "none": True},
                       {"rect": [100.0, 20.0, 120.0, 40.0], "none": True}]}  # rejected already: nothing to stage
    app = QApplication.instance() or QApplication([])
    scene = Scene(str(tmp_path), "cam")
    win = ReviewWindow(scene, proposals=offered)
    try:
        assert [(d["_key"], d.get("lidar_id"), d.get("none")) for d in win.staged["000"]] \
            == [("3", "7", None), ("4", None, True)]
        win._on_review(match_review(None, scene, "000"), None, points)
        cards = [win.cards_lay.itemAt(i).widget().text() for i in range(win.cards_lay.count())
                 if win.cards_lay.itemAt(i).widget()]
        assert cards == ["t3 → #7 (to confirm)", "t4 → ✗ (to confirm)", "t5 → ✗"]
        assert all(not r.get("no_match") and not r.get("obj_id") for r in scene.load_rects("000")[:2])  # nothing saved yet
        win._apply_all()
        pair, parked, old = scene.load_rects("000")
        assert pair["obj_id"] == "7" and parked.get("no_match") and not parked.get("obj_id") and old.get("no_match")
        app.processEvents()
    finally:
        win.close()


def test_boxes_to_check_are_marked_and_stepped_through_without_changing_anything(tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import Scene, match_review

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for directory in ("lidar", "label", "label_fusion/camera/cam"):
        (tmp_path / directory).mkdir(parents=True)
    points = np.zeros((1, 3), np.float32)
    for frame in ("000", "001", "002", "003"):
        np.save(tmp_path / "lidar" / f"{frame}.npy", points)
        (tmp_path / "label" / f"{frame}.json").write_text(json.dumps([
            {"obj_id": "7", "obj_type": "Car",
             "psr": {"position": dict.fromkeys("xyz", 1.0), "rotation": dict.fromkeys("xyz", 0.0),
                     "scale": dict.fromkeys("xyz", 1.0)}}]))
        (tmp_path / "label_fusion/camera/cam" / f"{frame}.json").write_text(json.dumps({"objs": [
            {"rect": {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}, "obj_attr": "t3", "obj_id": "7"},
            {"rect": {"x1": 70.0, "y1": 20.0, "x2": 90.0, "y2": 60.0}, "obj_attr": "t4"}]}))
    before = {frame: (tmp_path / "label_fusion/camera/cam" / f"{frame}.json").read_text()
              for frame in ("001", "003")}
    check = {"001": [{"rect": [10.0, 20.0, 50.0, 60.0], "note": "hidden behind a van"}],
             "003": [{"note": "look at the whole frame"}],
             "999": [{"note": "no such frame"}]}
    app = QApplication.instance() or QApplication([])
    scene = Scene(str(tmp_path), "cam")
    win = ReviewWindow(scene, check=check)
    try:
        assert win.cur == "001" and sorted(win.checks) == ["001", "003"] and not win.staged
        win._on_review(match_review(None, scene, "001"), None, points)
        cards = [win.cards_lay.itemAt(i).widget() for i in range(win.cards_lay.count())
                 if win.cards_lay.itemAt(i).widget()]
        assert [card.text() for card in cards] == ["t3 → #7 ✓ (check)", "t4 → —"]
        assert "hidden behind a van" in cards[0].toolTip()
        assert win.frame_lbl.text().endswith("check 1/2")
        assert win.check_lbl.text() == "Check: hidden behind a van" and not win.check_lbl.isHidden()
        assert win.pending_btn.isEnabled() and not win.apply_all_btn.isEnabled()
        win._next_pending()
        assert win.cur == "003"
        win._on_review(match_review(None, scene, "003"), None, points)
        assert win.frame_lbl.text().endswith("check 2/2")
        assert win.check_lbl.text() == "Check: look at the whole frame"
        win._step(-1)
        win._on_review(match_review(None, scene, "002"), None, points)
        assert win.check_lbl.isHidden() and win.frame_lbl.text().endswith("· 002")
        win._next_pending()
        assert win.cur == "003"
        win._next_pending()
        assert win.cur == "001" and not win.staged
        assert before == {frame: (tmp_path / "label_fusion/camera/cam" / f"{frame}.json").read_text()
                          for frame in before}
        app.processEvents()
    finally:
        win.close()

    only = ReviewWindow(scene, check=check, listed_only=True)
    try:
        assert only.frames == ["001", "003"] and only.scene_frames == ["000", "001", "002", "003"]
        only._on_review(match_review(None, scene, "001"), None, points)
        assert only.frame_lbl.text() == "frame 2/4 · 001 · listed 1/2"
        only._step(-1)
        assert only.cur == "001"  # nothing before the first listed frame
        only._step(+1)
        assert only.cur == "003"  # frame 002 is not shown
        only._on_review(match_review(None, scene, "003"), None, points)
        assert only.frame_lbl.text() == "frame 4/4 · 003 · listed 2/2"
        only.jump_edit.setText("3")  # a scene frame number that is not listed
        only._jump()
        assert only.cur == "003" and "not among the 2 frames shown" in only.statusBar().currentMessage()
        only.jump_edit.setText("2")
        only._jump()
        assert only.cur == "001"
        app.processEvents()
    finally:
        only.close()
    plain = ReviewWindow(scene, listed_only=True)  # nothing listed: every frame stays
    try:
        assert plain.frames == plain.scene_frames and not plain.listed_only
    finally:
        plain.close()


def test_listed_frames_can_be_shown_with_the_frames_around_them(tmp_path, monkeypatch):
    import json

    from PySide6.QtWidgets import QApplication
    from xcalib.gui.review import ReviewWindow
    from xcalib.gui.scene import Scene, match_review

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for directory in ("lidar", "label", "label_fusion/camera/cam"):
        (tmp_path / directory).mkdir(parents=True)
    points = np.zeros((1, 3), np.float32)
    names = [f"{i:03d}" for i in range(12)]
    for frame in names:
        np.save(tmp_path / "lidar" / f"{frame}.npy", points)
        (tmp_path / "label" / f"{frame}.json").write_text("[]")
        (tmp_path / "label_fusion/camera/cam" / f"{frame}.json").write_text(json.dumps({"objs": [
            {"rect": {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}, "obj_attr": "t3"}]}))
    check = {"001": [{"note": "first"}], "008": [{"note": "second"}]}
    app = QApplication.instance() or QApplication([])
    scene = Scene(str(tmp_path), "cam")
    win = ReviewWindow(scene, check=check, listed_only=True, context=2)
    try:
        # two frames either side of 001 and 008; the scene starts at 000, so nothing before it
        assert win.frames == ["000", "001", "002", "003", "006", "007", "008", "009", "010"]
        assert win.listed == ["001", "008"] and win.cur == "001"
        win._step(+1)
        win._step(+1)
        win._step(+1)
        assert win.cur == "006"  # 004 and 005 are left out
        win._on_review(match_review(None, scene, "006"), None, points)
        assert win.frame_lbl.text() == "frame 7/12 · 006 · beside listed 2/2" and win.check_lbl.isHidden()
        win._next_pending()
        assert win.cur == "008"
        win._on_review(match_review(None, scene, "008"), None, points)
        assert win.frame_lbl.text() == "frame 9/12 · 008 · listed 2/2"
        assert win.check_lbl.text() == "Check: second"
        app.processEvents()
    finally:
        win.close()
