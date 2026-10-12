"""The corpus-write mapping in the Qt review.

The Qt event loop / worker thread isn't unit-coverable, but the pure
decision -> VerifiedPair mapping (`_verified_pairs`) is, and it's what turns the
persisted decisions into the rows the GUI writes to the durable corpus.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")  # xcalib.gui.review imports Qt at module load; skips when Qt's system libraries are missing

from xcalib.gui.review import _rotate_xy, _verified_pairs
from xcalib.gui.scene import FrameReview, TrackEntry


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


def test_verified_pairs_maps_persisted_decisions():
    review = FrameReview(frame="f0", tracks=[
        TrackEntry(key="1", rect=[1., 2., 3., 4.], applied_id="10", suggestion="10",
                   options=[{"lidar_id": "10", "score": 0.9},
                            {"lidar_id": "11", "score": 0.2}]),
        TrackEntry(key="2", rect=[5., 6., 7., 8.], applied_id="9", suggestion="11",
                   options=[{"lidar_id": "11", "score": 0.5}]),
        TrackEntry(key="@3", rect=[0., 0., 1., 1.], rejected=True, options=[]),
        TrackEntry(key="4", rect=[2., 2., 3., 3.]),  # undecided -> skipped
    ], boxes=[])

    pairs = _verified_pairs(review)
    assert len(pairs) == 3  # the undecided rect is not recorded

    # applied id == the model's suggestion -> the store marks verified_model_correct
    assert pairs[0].decision == "positive" and pairs[0].lidar_key == "id:10"
    assert pairs[0].track == "1"
    assert pairs[0].proposal == {"decision": "positive", "lidar_key": "id:10"}
    assert [c["lidar_key"] for c in pairs[0].candidates] == ["id:10", "id:11"]

    # applied id != the suggestion -> a human correction
    assert pairs[1].lidar_key == "id:9" and pairs[1].proposal["lidar_key"] == "id:11"

    # an explicit rejection -> no_match; an untracked "@" rect -> no camera track
    assert pairs[2].decision == "no_match" and pairs[2].lidar_key is None
    assert pairs[2].track is None


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

    for view in (bev, image_view):
        assert view.sceneRect() == view.scene().itemsBoundingRect()
        assert view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert view.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
