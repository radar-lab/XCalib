"""
xcalib-review — standalone Qt window for pairing camera boxes with LiDAR boxes.

When the viewpoint moves past what the matcher tolerates, a person has to say
which pairs are right. This window shows the image with 2D boxes beside a
top-down LiDAR view, with one card per camera box and a ranked,
geometry-annotated candidate menu. Labels and images use a SUSTechPOINTS-style
on-disk layout (see `xcalib.gui.scene`).

    pip install "xcalib[gui]"
    xcalib-review --scene /data/MySite --camera cam_front

Workflow: step frames (the matcher proposes gated one-to-one pairs) → fix the
wrong ones (click a card, pick the right #id or — none —) → Apply (writes the
decisions into the fusion labels) → Calibrate or Recalibrate from the saved
pairs.

The window saves pairing decisions and a calibration, nothing else. Installing
a model and learning from the saved pairs are notebook steps, where each one
says what it reads, what it writes and how to judge the result:
`demo/xcalib_homography_en.ipynb` trains the matcher on the saved pairs and
fits the homography to them, `demo/refine_unification.ipynb` records them into
the continual corpus and runs the gated refine.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import queue
import sys
from typing import Dict, List, Optional

import numpy as np

try:
    from PySide6.QtCore import QPointF, Qt, QThread, Signal
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QImage,
        QPainter,
        QPen,
        QPixmap,
        QPolygonF,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QFileDialog,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSplitter,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "xcalib-review needs PySide6 — install the GUI extra:\n"
        "    pip install 'xcalib[gui]'"
    ) from e

from .scene import FrameReview, Scene, box3d_corners, load_points, match_review, track_of
from .scene import matcher_target as _matcher_target
from .navigation import FrameNavigation

GRAY = QColor("#8a8a8a")


def _candidate_sort_key(option):
    """Put finite projection distances first; rank missing distances by score."""
    distance = option.get("proj_px")
    try:
        distance = float(distance)
        valid = math.isfinite(distance) and distance >= 0
    except (TypeError, ValueError):
        valid = False
        distance = 0.0
    return (not valid, distance if valid else 0.0, -float(option.get("score", 0.0)))


def _id_color(idx: int) -> QColor:
    """Stable, visually distinct color per matched id (golden-angle hues)."""
    h = (idx * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


# ------------------------------------------------------------------ worker --

class MatcherWorker(QThread):
    """Owns the Matcher on a background thread; the UI queues jobs to it."""

    review_ready = Signal(object, object, object)   # FrameReview, image, points
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, scene: Scene, device: Optional[str] = None):
        super().__init__()
        self.scene = scene
        self.device = device
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.matcher = None

    def submit(self, kind: str, *args) -> None:
        self.jobs.put((kind, *args))

    def run(self) -> None:  # pragma: no cover - thread loop (logic tested via scene)
        try:
            resolved = _matcher_target(self.scene)
            if resolved is not None:
                from ..engine.matcher import Matcher

                base, weights, config = resolved
                self.status.emit(f"loading {base} …")
                self.matcher = Matcher.from_pretrained(base, weights=weights,
                                                       config=config, device=self.device)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"could not load the matcher; manual mode: {e}")
        self.status.emit("matcher ready" if self.matcher else "manual pairing (no model)")

        while True:
            job = self.jobs.get()
            if job[0] == "quit":
                return
            try:
                if job[0] == "match":
                    self._do_match(job[1], job[2])
            except Exception as e:  # noqa: BLE001 — surface, don't kill the loop
                self.failed.emit(str(e))

    def _do_match(self, frame: str, threshold: float) -> None:
        import cv2  # noqa: PLC0415

        self.status.emit(f"matching {frame} …")
        path = self.scene.image_path(frame)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) if path else None
        lp = self.scene.lidar_path(frame)
        points = load_points(lp) if lp else None
        review = match_review(self.matcher, self.scene, frame, threshold,
                              image=image, points=points)
        self.review_ready.emit(review, image, points)


# ------------------------------------------------------------------- views --

class ImageView(QGraphicsView):
    """Camera image + colored rect overlays; click a rect to select its track."""

    track_clicked = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        _fit_only(self)
        self._rects: List[tuple] = []   # (key, x1, y1, x2, y2)

    def show_frame(self, image: Optional[np.ndarray], tracks, assign, colors,
                   selected: Optional[str], projection=None) -> None:
        sc = self.scene()
        sc.clear()
        if image is None:
            return
        h, w = image.shape[:2]
        qimg = QImage(np.ascontiguousarray(image).data, w, h, 3 * w,
                      QImage.Format_RGB888)
        sc.addPixmap(QPixmap.fromImage(qimg.copy()))
        self._rects = []
        for t in tracks:
            x1, y1, x2, y2 = t.rect
            lid = assign.get(t.key)
            color = colors.get(lid, GRAY) if lid else GRAY
            pen = QPen(color, 3 if t.key == selected else 2)
            sc.addRect(x1, y1, x2 - x1, y2 - y1, pen)
            label = f"{t.key if t.key.startswith('@') else 't' + t.key}" \
                    f"{' → #' + lid if lid else ' ?'}"
            txt = sc.addSimpleText(label)
            txt.setBrush(QBrush(color))
            txt.setPos(x1, max(0.0, y1 - 16))
            self._rects.append((t.key, x1, y1, x2, y2))
        if projection is not None:
            pen = QPen(QColor("#ff3a3a"), 2, Qt.DashLine)
            for a, b in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
                sc.addLine(float(projection[a, 0]), float(projection[a, 1]),
                           float(projection[b, 0]), float(projection[b, 1]), pen)
        # Keep out-of-frame projections from shrinking the actual camera image.
        sc.setSceneRect(0, 0, w, h)
        self.fitInView(sc.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event) -> None:  # pragma: no cover - interaction
        p = self.mapToScene(event.position().toPoint())
        best, area = None, float("inf")
        for key, x1, y1, x2, y2 in self._rects:
            if x1 <= p.x() <= x2 and y1 <= p.y() <= y2:
                a = (x2 - x1) * (y2 - y1)
                if a < area:
                    best, area = key, a
        if best is not None:
            self.track_clicked.emit(best)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # pragma: no cover
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)


def _fit_only(view: QGraphicsView) -> None:
    """A view that always shows its whole scene never needs scrollbars.

    With them on, a scrollbar that appears narrows the viewport, ``fitInView``
    re-fits to the new size, the scrollbar disappears, and the view can resize
    itself in a loop that pins the CPU.
    """
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


def _rotate_xy(xy: np.ndarray, angle_deg: float, center: np.ndarray) -> np.ndarray:
    """Rotate an ``[N, 2]`` array of points by ``angle_deg`` (CCW) about ``center``."""
    if not angle_deg:
        return xy
    a = np.radians(angle_deg)
    cos, sin = np.cos(a), np.sin(a)
    rot = np.array([[cos, -sin], [sin, cos]])
    return (np.asarray(xy, dtype=float) - center) @ rot.T + center


class BevView(QGraphicsView):
    """Top-down LiDAR view: point scatter + box footprints colored by id.

    The view rotates about the cloud centroid (reviewer-controlled) so the scene can
    be oriented to taste; labels stay upright because the rotation is applied to the
    data, not the view transform."""

    box_clicked = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor("#101010")))
        _fit_only(self)
        self._angle_deg = 0.0
        self._zoom = 1.0
        self._view_center = None
        self._drag_pos = None
        self._drag_mode = None
        self._drag_button = None
        self._last: Optional[tuple] = None
        self._polygons = []
        self.setToolTip("Click a box to pair; drag empty space or middle-drag to rotate; "
                        "wheel zooms; right-drag pans")

    def _fit_view(self) -> None:
        rect = self.scene().itemsBoundingRect()
        if rect.isEmpty():
            return
        self.scene().setSceneRect(rect)
        self.fitInView(rect, Qt.KeepAspectRatio)
        self.scale(self._zoom, self._zoom)
        if self._view_center is not None:
            self.centerOn(self._view_center)

    def zoom_view(self, factor: float) -> None:
        if not math.isfinite(factor) or factor <= 0:
            return
        self._view_center = self.mapToScene(self.viewport().rect().center())
        self._zoom = max(1.0, min(12.0, self._zoom * factor))
        if self._zoom == 1.0:
            self._view_center = None
        self._fit_view()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._view_center = None
        self._angle_deg = 0.0
        if self._last is not None:
            self.show_frame(*self._last)

    def rotate_view(self, delta_deg: float) -> None:
        """Turn the top-down view by ``delta_deg`` and re-render the last frame."""
        self._angle_deg = (self._angle_deg + delta_deg) % 360.0
        if self._last is not None:
            self.show_frame(*self._last)

    def show_frame(self, points: Optional[np.ndarray], boxes, id_colors,
                   selected_id: Optional[str] = None) -> None:
        sc = self.scene()
        sc.clear()
        self._polygons = []
        self._last = (points, boxes, id_colors, selected_id)
        if points is None or not len(points):
            return
        # rotate the top-down view about the cloud centroid, then frame the result
        center = points[:, :2].mean(axis=0)
        pxy = _rotate_xy(points[:, :2], self._angle_deg, center)
        # bounds from the point percentiles, padded — robust to stray returns
        lo = np.percentile(pxy, 1, axis=0) - 5
        hi = np.percentile(pxy, 99, axis=0) + 5
        span = np.maximum(hi - lo, 1e-3)
        size = 900
        scale = size / float(span.max())

        xy = ((pxy - lo) * scale).astype(int)
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
        xy = xy[keep]
        # x forward (up on screen), y left (left on screen): flip both axes
        px = size - 1 - xy[:, 1]
        py = size - 1 - xy[:, 0]
        # shade by INTENSITY when the cloud carries it (4th column): robust
        # percentile normalisation, dim floor so low returns stay visible
        if points.shape[1] >= 4:
            inten = points[keep, 3].astype(float)
            lo_i, hi_i = np.percentile(inten, [5.0, 95.0])
            t = np.clip((inten - lo_i) / max(hi_i - lo_i, 1e-6), 0.0, 1.0)
            shade = (70 + t * 185).astype(int)
        else:
            shade = np.full(len(px), 154, dtype=int)  # flat #9a9a9a
        pixels = np.full((size, size, 4), 16, dtype=np.uint8)
        pixels[:, :, 3] = 255
        pixels[py, px, :3] = shade[:, None]
        img = QImage(pixels.data, size, size, size * 4, QImage.Format_RGB32).copy()
        sc.addPixmap(QPixmap.fromImage(img))

        for b in boxes:
            lid = str(b.get("obj_id", "") or "").strip()
            color = id_colors.get(lid, GRAY)
            corners = _rotate_xy(box3d_corners(b["psr"])[:, :2], self._angle_deg, center)
            ground = corners[[0, 2, 6, 4]]  # the z-min footprint, in order
            poly = QPolygonF()
            for cx, cy in ground:
                u = size - 1 - (cy - lo[1]) * scale
                v = size - 1 - (cx - lo[0]) * scale
                poly.append(QPointF(u, v))
            if lid and lid == selected_id:
                halo = QPen(QColor("#101010"), 8)
                halo.setCosmetic(True)
                outline = sc.addPolygon(poly, halo)
                outline.setZValue(9)
                pen = QPen(QColor("#ffff00"), 4)
                pen.setCosmetic(True)
                item = sc.addPolygon(poly, pen, QBrush(QColor(255, 255, 0, 65)))
                item.setZValue(10)
                item.setData(1, "selected")
            else:
                pen = QPen(color, 2)
                pen.setCosmetic(True)
                item = sc.addPolygon(poly, pen)
            item.setData(0, lid)
            if lid:
                self._polygons.append((lid, poly))
            if lid:
                txt = sc.addSimpleText(f"▶ #{lid}" if lid == selected_id else f"#{lid}")
                txt.setBrush(QBrush(QColor("#ffff00") if lid == selected_id else color))
                if lid == selected_id:
                    txt.setZValue(11)
                    font = txt.font()
                    font.setBold(True)
                    font.setPointSize(18)
                    txt.setFont(font)
                r = poly.boundingRect()
                txt.setPos(r.x(), r.y() - 14)
        self._fit_view()

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y():
            self.zoom_view(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        rotate = (event.button() == Qt.MiddleButton or
                  (event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier))
        if rotate or event.button() == Qt.RightButton:
            self._drag_pos = event.position().toPoint()
            self._drag_mode = "rotate" if rotate else "pan"
            self._drag_button = event.button()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        p = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            for lid, poly in reversed(self._polygons):
                if poly.containsPoint(p, Qt.OddEvenFill):
                    self.box_clicked.emit(lid)
                    return
            # Left-dragging empty space rotates; a click on a box still pairs it.
            self._drag_pos = event.position().toPoint()
            self._drag_mode = "rotate"
            self._drag_button = event.button()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None:
            pos = event.position().toPoint()
            if self._drag_mode == "rotate":
                self.rotate_view((pos.x() - self._drag_pos.x()) * 0.4)
            else:
                delta = self.mapToScene(self._drag_pos) - self.mapToScene(pos)
                self._view_center = self.mapToScene(self.viewport().rect().center()) + delta
                self.centerOn(self._view_center)
            self._drag_pos = pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_pos is not None and event.button() == self._drag_button:
            self._drag_pos = None
            self._drag_mode = None
            self._drag_button = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # pragma: no cover
        if self.scene() and not self.scene().itemsBoundingRect().isEmpty():
            self._fit_view()
        super().resizeEvent(event)


# ------------------------------------------------------------- main window --

class ReviewWindow(QMainWindow):
    """The review surface: toolbar, image | BEV, track cards, candidate row."""

    def __init__(self, scene: Scene, device: Optional[str] = None, *, notebook_workflow: bool = False,
                 proposals: Optional[Dict[str, List[dict]]] = None,
                 check: Optional[Dict[str, List[dict]]] = None, listed_only: bool = False,
                 context: int = 0, accept_calibration: bool = False):
        super().__init__()
        self.notebook_workflow = notebook_workflow
        self.accept_calibration = accept_calibration  # save a calibration that passes the checks unasked
        self.scene_io = scene
        self.scene_frames = scene.frames()  # every frame: calibration and training use these
        self.frames = list(self.scene_frames)  # the frames the window steps through
        self.listed_only = False
        self.cur = self.frames[0] if self.frames else ""
        self.review: Optional[FrameReview] = None
        self.image: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self.assign: Dict[str, Optional[str]] = {}
        self.rejected: set = set()
        self.touched: set = set()
        self.staged: Dict[str, List[dict]] = {}
        # frame -> track key -> LiDAR id to confirm; None: "no LiDAR box" to confirm
        self.offered: Dict[str, Dict[str, Optional[str]]] = {}
        self.checks: Dict[str, Dict[Optional[str], str]] = {}  # frame -> track key -> why to look at it
        self.applied_total = 0
        self._offer(proposals or {})
        self._flag(check or {})
        self.listed = [frame for frame in self.frames if frame in self.staged or frame in self.checks]
        if listed_only and self.listed:  # step through the offered and flagged frames, and `context` frames around each
            place = {frame: i for i, frame in enumerate(self.scene_frames)}
            keep = {i for frame in self.listed
                    for i in range(place[frame] - max(0, context), place[frame] + max(0, context) + 1)}
            self.frames = [frame for i, frame in enumerate(self.scene_frames) if i in keep]
            self.listed_only = True

        self.setWindowTitle(f"xcalib review — {os.path.basename(scene.root)}/{scene.camera}")
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1500, 950)
        else:
            available = screen.availableGeometry()
            self.resize(min(1500, max(1, available.width() - 24)),
                        min(950, max(1, available.height() - 48)))
        self._build_ui()
        self.frame_navigation = FrameNavigation(self, self._step)

        self.worker = MatcherWorker(scene, device)
        self.worker.review_ready.connect(self._on_review)
        self.worker.failed.connect(self._worker_failed)
        self.worker.status.connect(self._status)
        self.worker.start()
        if self.cur:
            self.worker.submit("match", self.cur, self.thr.value())

    def _offer(self, proposals: Dict[str, List[dict]]) -> None:
        """Stage pairs proposed from outside as pending decisions.

        They are shown like the user's own unsaved choices and reach the labels
        only through Apply, so a person confirms every one of them. A pair is
        dropped when its camera box or its LiDAR id is not in the frame, or when
        it is saved already. A camera box that holds an offered LiDAR id in the
        labels is staged to give it up, so Apply never leaves one LiDAR box on
        two camera boxes. An entry with ``"none": true`` instead of a LiDAR id
        stages "this camera box has no LiDAR box", the rejection a person makes
        of a wrong suggestion. The window opens on the first frame that has one.
        """
        for frame in self.frames:
            rects = self.scene_io.load_rects(frame) if frame in proposals else []
            keys = {tuple(round(float(r["rect"][k]), 1) for k in ("x1", "y1", "x2", "y2")):
                    (track_of(r) or f"@{i}", str(r.get("obj_id") or "").strip(), bool(r.get("no_match")))
                    for i, r in enumerate(rects)}
            ids = [str(b.get("obj_id") or "").strip() for b in self.scene_io.load_boxes(frame)] if rects else []
            staged = []
            for pair in proposals.get(frame, []):
                key, saved, rejected = keys.get(tuple(round(float(v), 1) for v in pair["rect"]), (None, None, None))
                lid = str(pair.get("lidar_id") or "").strip()
                if key is None:
                    continue
                if pair.get("none"):
                    if saved or not rejected:
                        staged.append({"rect": [float(v) for v in pair["rect"]], "none": True, "_key": key})
                elif ids.count(lid) == 1 and (saved != lid or rejected):
                    staged.append({"rect": [float(v) for v in pair["rect"]], "lidar_id": lid, "_key": key})
            taken = {d["lidar_id"] for d in staged if d.get("lidar_id")}
            moved = {d["_key"] for d in staged}
            for rect, (key, saved, _) in keys.items():
                if saved in taken and key not in moved:  # a LiDAR box has one camera box: the old holder gives it up
                    staged.append({"rect": list(rect), "clear": True, "_key": key})
            if staged:
                self.staged[frame] = staged
                self.offered[frame] = {d["_key"]: d.get("lidar_id") for d in staged if not d.get("clear")}
        if self.offered:
            self.cur = next(frame for frame in self.frames if frame in self.offered)

    def _flag(self, check: Dict[str, List[dict]]) -> None:
        """Mark camera boxes a person should look at, each with the reason.

        Nothing is staged or changed. The boxes are marked ``(check)`` on their
        cards, the reason is shown above the image, and the window opens on the
        first frame that has one or an offered pair. An entry without a matching
        camera box still marks its frame.
        """
        for frame in self.frames:
            if frame not in check:
                continue
            keys = {tuple(round(float(r["rect"][k]), 1) for k in ("x1", "y1", "x2", "y2")):
                    track_of(r) or f"@{i}" for i, r in enumerate(self.scene_io.load_rects(frame))}
            self.checks[frame] = {
                keys.get(tuple(round(float(v), 1) for v in entry.get("rect") or ())):
                    str(entry.get("note") or "") for entry in check[frame]}
        first = next((frame for frame in self.frames if frame in self.offered or frame in self.checks), None)
        if first is not None:
            self.cur = first

    # ---- UI scaffolding ----

    def _build_ui(self) -> None:
        top = QWidget()
        root = QVBoxLayout(top)
        root.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        self.prev_btn = QPushButton("‹")
        self.next_btn = QPushButton("›")
        self.prev_btn.setToolTip("Previous frame (Left arrow)")
        self.next_btn.setToolTip("Next frame (Right arrow)")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        self.frame_lbl = QLabel("—")
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("#")
        self.jump_edit.setFixedWidth(70)
        self.jump_edit.returnPressed.connect(self._jump)
        self.pending_btn = QPushButton("Next pending")
        self.pending_btn.setToolTip("Go to the next frame with unsaved decisions or a box to check (N)")
        self.pending_btn.clicked.connect(self._next_pending)
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.0, 1.0)
        self.thr.setSingleStep(0.05)
        self.thr.setValue(0.5)
        self.thr.setPrefix("thr ")
        self.thr.valueChanged.connect(lambda _v: self._rematch())
        self.rot_l_btn = QToolButton()
        self.rot_l_btn.setText("⟲")
        self.rot_l_btn.setToolTip("Rotate the top-down LiDAR view counter-clockwise")
        self.rot_l_btn.clicked.connect(lambda: self.bev_view.rotate_view(15))
        self.rot_r_btn = QToolButton()
        self.rot_r_btn.setText("⟳")
        self.rot_r_btn.setToolTip("Rotate the top-down LiDAR view clockwise")
        self.rot_r_btn.clicked.connect(lambda: self.bev_view.rotate_view(-15))
        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setText("−")
        self.zoom_out_btn.setToolTip("Zoom out the LiDAR view")
        self.zoom_out_btn.clicked.connect(lambda: self.bev_view.zoom_view(1 / 1.2))
        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setText("+")
        self.zoom_in_btn.setToolTip("Zoom in the LiDAR view")
        self.zoom_in_btn.clicked.connect(lambda: self.bev_view.zoom_view(1.2))
        self.reset_bev_btn = QToolButton()
        self.reset_bev_btn.setText("Reset view")
        self.reset_bev_btn.setToolTip("Reset LiDAR view rotation, zoom and pan")
        self.reset_bev_btn.clicked.connect(lambda: self.bev_view.reset_view())
        self.assigned_lbl = QLabel("")
        self.selection_lbl = QLabel("Camera: —  |  LiDAR: —")
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(lambda: self._calibrate(False))
        self.recalibrate_btn = QPushButton("Recalibrate")
        self.recalibrate_btn.clicked.connect(lambda: self._calibrate(True))
        self.apply_btn = QPushButton("Apply this frame")
        self.apply_btn.clicked.connect(self._apply_frame)
        self.apply_all_btn = QPushButton("Apply all")
        self.apply_all_btn.clicked.connect(self._apply_all)
        for w in (self.prev_btn, self.frame_lbl, self.next_btn, self.jump_edit, self.pending_btn,
                  self.thr, self.rot_l_btn, self.rot_r_btn, self.zoom_out_btn,
                  self.zoom_in_btn, self.reset_bev_btn, self.assigned_lbl, self.selection_lbl):
            bar.addWidget(w)
        bar.addStretch(1)
        root.addLayout(bar)
        actions = QHBoxLayout()
        # Pairing and calibration only: installing a model and learning from the
        # saved pairs are notebook steps.
        for w in (self.calibrate_btn, self.recalibrate_btn, self.apply_btn, self.apply_all_btn):
            actions.addWidget(w)
        self.calibration_lbl = QLabel()  # the last calibration saved in this window
        actions.addWidget(self.calibration_lbl)
        actions.addStretch(1)
        root.addLayout(actions)
        hint = QLabel("Select a camera box, then a LiDAR box; yellow highlights the selection. "
                      "Q/E or drag empty BEV: rotate · +/- or wheel: zoom · R: reset · right-drag: pan.")
        root.addWidget(hint)
        self.check_lbl = QLabel()  # why this frame is on the --check list; wraps instead of widening the window
        self.check_lbl.setTextFormat(Qt.PlainText)
        self.check_lbl.setWordWrap(True)
        self.check_lbl.setStyleSheet("font-weight: 600;")
        self.check_lbl.hide()
        root.addWidget(self.check_lbl)

        split = QSplitter()
        self.image_view = ImageView()
        self.image_view.track_clicked.connect(self._select_track)
        self.bev_view = BevView()
        self.bev_view.box_clicked.connect(self._pick_bev)
        split.addWidget(self.image_view)
        split.addWidget(self.bev_view)
        split.setSizes([950, 550])
        root.addWidget(split, 1)

        self.cards_area = QScrollArea()
        self.cards_area.setWidgetResizable(True)
        self.cards_area.setFixedHeight(64)
        self.cards_host = QWidget()
        self.cards_lay = QHBoxLayout(self.cards_host)
        self.cards_lay.setContentsMargins(2, 2, 2, 2)
        self.cards_area.setWidget(self.cards_host)
        root.addWidget(self.cards_area)

        self.cand_area = QScrollArea()
        self.cand_area.setWidgetResizable(True)
        self.cand_area.setFixedHeight(58)
        self.cand_host = QWidget()
        self.cand_lay = QHBoxLayout(self.cand_host)
        self.cand_lay.setContentsMargins(2, 2, 2, 2)
        self.cand_area.setWidget(self.cand_host)
        self.cand_area.hide()
        root.addWidget(self.cand_area)

        self.setCentralWidget(top)
        self.selected: Optional[str] = None
        self.statusBar().showMessage("loading …")

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def _worker_failed(self, message: str) -> None:
        self._status(f"⚠ {message}")

    def _calibrate(self, refine: bool) -> None:
        """Estimate from saved reviewed pairs, then ask before saving the pose
        (with ``accept_calibration``, only when the result is doubtful)."""
        from ..protocol import CameraIntrinsics
        from .workflow import calibrate_scene, save_calibration, verified_frame

        previous = self.scene_io.calibration() if refine else None
        if refine and previous is None:
            QMessageBox.warning(self, "Recalibrate", "A saved calibration is required first.")
            return
        if previous:
            intrinsics = previous.intrinsics
        else:
            scene_intrinsic_path = os.path.join(self.scene_io.root, "calib", "camera",
                                               self.scene_io.camera + ".json")
            if not os.path.isfile(scene_intrinsic_path):
                # An imported scene keeps its intrinsics here until the first
                # calibration: every calib/camera file is read as a complete
                # calibration, so one without an extrinsic does not belong there.
                scene_intrinsic_path = os.path.join(self.scene_io.root, "calib", "intrinsic",
                                                   self.scene_io.camera + ".json")
            if self.notebook_workflow:
                choice = QMessageBox.No
            else:
                choice = QMessageBox.question(
                    self, "Camera intrinsics",
                    "Use the editable legacy camera defaults, scaled from the legacy "
                    "1280 x 720 reference? Their camera identity is unverified.\n"
                    "Choose No to load measured intrinsics from JSON.",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if choice == QMessageBox.Cancel:
                return
            try:
                if choice == QMessageBox.Yes:
                    pixmap = QPixmap(self.scene_io.image_path(self.cur) or "")
                    if pixmap.isNull():
                        raise ValueError("current frame has no readable image")
                    width, height = pixmap.width(), pixmap.height()
                    sx, sy = width / 1280.0, height / 720.0
                    values, ok = QInputDialog.getText(
                        self, "Legacy intrinsics (unverified)", "fx, fy, cx, cy (pixels):",
                        text=f"{696.43755463 * sx:.6f}, {695.78717319 * sy:.6f}, "
                             f"{642.63789053 * sx:.6f}, {394.51924698 * sy:.6f}")
                    if not ok:
                        return
                    fx, fy, cx, cy = [float(value.strip()) for value in values.split(",")]
                    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    distortion = [-0.379946698, 0.164050708, -0.00152008351,
                                  -0.000126369441, -0.0345314007]
                else:
                    if self.notebook_workflow and os.path.isfile(scene_intrinsic_path):
                        path = scene_intrinsic_path
                    else:
                        path, _ = QFileDialog.getOpenFileName(
                            self, "Select camera intrinsics JSON", "", "JSON (*.json)")
                    if not path:
                        return
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    matrix = np.asarray(data["intrinsic"], dtype=float).reshape(3, 3)
                    distortion = data.get("distortion")
                if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                        or not np.allclose(matrix[2], [0, 0, 1])
                        or not np.isclose(matrix[0, 1], 0) or not np.isclose(matrix[1, 0], 0)):
                    raise ValueError("intrinsic must be a valid zero-skew 3x3 camera matrix")
                if distortion is not None and not np.isfinite(np.asarray(distortion, dtype=float)).all():
                    raise ValueError("distortion must contain finite coefficients")
                intrinsics = CameraIntrinsics.from_matrix(matrix, distortion)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                QMessageBox.warning(self, "Invalid intrinsics", str(exc))
                return
        try:
            valid_frames, invalid_frames = [], []
            for frame in self.scene_frames:
                try:
                    verified_frame(self.scene_io, frame)
                    valid_frames.append(frame)
                except ValueError:
                    invalid_frames.append(frame)
            result = calibrate_scene(self.scene_io, intrinsics, frames=valid_frames,
                                     previous=previous)
            if not result.success or not result.accepted:
                QMessageBox.warning(self, "Calibration not accepted", result.message)
                return
            summary = (f"{result.n_inliers}/{result.n_correspondences} inliers; "
                       f"mean inlier reprojection error {result.reproj_error_px:.2f} px")
            # Unasked saving needs a verdict no person gives: a nudged pose was
            # adopted only because it fits the pairs better than the saved one,
            # a fresh solve has to pass the checks of an automatic calibration.
            doubts = []
            if self.accept_calibration and (not refine or getattr(result, "resolved", False)):
                from .auto_calibrate import solve_problems

                doubts = solve_problems(result, result.n_correspondences)
            if not self.accept_calibration or doubts:
                message = (f"{summary}.\n"
                           f"Skipped {len(invalid_frames)} frame(s) with ambiguous or missing IDs.\n"
                           + (f"Not saved automatically: {'; '.join(doubts)}.\n" if doubts else "")
                           + "Save this LiDAR-to-camera calibration in the scene?")
                if QMessageBox.question(self, "Save calibration", message) != QMessageBox.Yes:
                    return
            pose = np.eye(4)
            pose[:3, :3] = result.rotation
            pose[:3, 3] = result.translation.reshape(3)
            save_calibration(self.scene_io, intrinsics.K, pose, intrinsics.distortion)
            unasked = self.accept_calibration and not doubts
            # The status line moves on with the next frame result; this label keeps the
            # outcome, short enough not to widen the window (details in its tooltip).
            self.calibration_lbl.setText(
                f"Calibration saved{' automatically' if unasked else ''}: {result.n_inliers}/"
                f"{result.n_correspondences} inliers, {result.reproj_error_px:.2f} px")
            self.calibration_lbl.setToolTip(result.message if refine else "")
            self._status(f"Calibration saved automatically: {summary}" if unasked else "Calibration saved")
            self._rematch()
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Calibration failed", str(exc))

    # ---- navigation ----

    def _step(self, delta: int) -> None:
        if not self.frames or self.cur not in self.frames:
            return
        self._bank_current()
        i = self.frames.index(self.cur) + delta
        if 0 <= i < len(self.frames):
            self.cur = self.frames[i]
            self.worker.submit("match", self.cur, self.thr.value())

    def _jump(self) -> None:
        try:
            n = int(self.jump_edit.text())
        except ValueError:
            return
        if not 1 <= n <= len(self.scene_frames):
            return
        if self.scene_frames[n - 1] not in self.frames:
            self._status(f"Frame {n} is not among the {len(self.frames)} frames shown")
            return
        self._bank_current()
        self.cur = self.scene_frames[n - 1]
        self.worker.submit("match", self.cur, self.thr.value())

    def _next_pending(self) -> None:
        """Go to the next frame, after this one or from the start, with unsaved decisions or a box to check."""
        self._bank_current()
        waiting = [frame for frame in self.frames if frame in self.staged or frame in self.checks]
        if not waiting:
            return
        here = self.frames.index(self.cur) if self.cur in self.frames else -1
        later = [frame for frame in waiting if self.frames.index(frame) > here]
        self.cur = (later or waiting)[0]
        self.worker.submit("match", self.cur, self.thr.value())

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        key = event.key()
        if not isinstance(self.focusWidget(), (QLineEdit, QDoubleSpinBox)):
            actions = {
                Qt.Key_Q: lambda: self.bev_view.rotate_view(15),
                Qt.Key_E: lambda: self.bev_view.rotate_view(-15),
                Qt.Key_Plus: lambda: self.bev_view.zoom_view(1.2),
                Qt.Key_Equal: lambda: self.bev_view.zoom_view(1.2),
                Qt.Key_Minus: lambda: self.bev_view.zoom_view(1 / 1.2),
                Qt.Key_R: self.bev_view.reset_view,
                Qt.Key_N: self._next_pending,
            }
            if key in actions:
                actions[key]()
                event.accept()
                return
        super().keyPressEvent(event)

    def _rematch(self) -> None:
        if self.cur:
            self.worker.submit("match", self.cur, self.thr.value())

    # ---- review state ----

    def _on_review(self, review: FrameReview, image, points) -> None:
        if review.frame != self.cur:
            return
        self.review, self.image, self.points = review, image, points
        # seed: applied ids lock in; staged session decisions come next;
        # persisted/session rejections stay none; the gated assignment fills
        # the rest
        assign: Dict[str, Optional[str]] = {}
        used: set = set()
        self.rejected = set()
        self.touched = set()
        for t in review.tracks:
            if t.applied_id:
                assign[t.key] = t.applied_id
                used.add(t.applied_id)
            if t.rejected:
                self.rejected.add(t.key)
        for d in self.staged.get(review.frame, []):
            key = d.get("_key")
            if key and not d.get("_proposal"):  # a suggestion left as it was is filled in again below
                self.touched.add(key)
                if d.get("none"):
                    assign[key] = None
                    self.rejected.add(key)
                elif d.get("clear"):
                    assign[key] = None
                    self.rejected.discard(key)
                elif d.get("lidar_id"):
                    assign[key] = d["lidar_id"]
                    self.rejected.discard(key)
        used = {lid for lid in assign.values() if lid}
        for t in review.tracks:
            if t.key in assign:
                continue
            if t.rejected or t.key in self.rejected:
                assign[t.key] = None
            elif t.suggestion and t.suggestion not in used:
                assign[t.key] = t.suggestion
                used.add(t.suggestion)
        self.assign = assign
        self.selected = None
        self._redraw()

    def _colors(self) -> Dict[str, QColor]:
        ids: List[str] = []
        for t in (self.review.tracks if self.review else []):
            for o in t.options:
                if o["lidar_id"] not in ids:
                    ids.append(o["lidar_id"])
        return {lid: _id_color(i) for i, lid in enumerate(ids)}

    def _redraw(self) -> None:
        if self.review is None:
            return
        colors = self._colors()
        idx = self.scene_frames.index(self.cur) + 1 if self.cur in self.scene_frames else 0
        flagged = self.checks.get(self.cur, {})
        why = "; ".join(dict.fromkeys(note for note in flagged.values() if note))
        if self.listed_only and self.cur in self.listed:
            position = f" · listed {self.listed.index(self.cur) + 1}/{len(self.listed)}"
        elif self.listed_only and idx:  # a frame shown for context: say which listed frame it is beside
            nearest = min(self.listed, key=lambda frame: abs(self.scene_frames.index(frame) + 1 - idx))
            position = f" · beside listed {self.listed.index(nearest) + 1}/{len(self.listed)}"
        else:
            position = f" · check {list(self.checks).index(self.cur) + 1}/{len(self.checks)}" if flagged else ""
        self.frame_lbl.setText(f"frame {idx}/{len(self.scene_frames)} · {self.cur}{position}")
        self.check_lbl.setText(f"Check: {why}")
        self.check_lbl.setVisible(bool(why))
        assigned = sum(1 for t in self.review.tracks if self.assign.get(t.key))
        self.assigned_lbl.setText(f"{assigned}/{len(self.review.tracks)} assigned")
        projection = None
        selected_id = self.assign.get(self.selected)
        track = next((t for t in self.review.tracks if t.key == self.selected), None)
        pair_state = ""
        if selected_id and track:
            pair_state = (" · saved" if selected_id == track.applied_id else
                          " · selected" if track.key in self.touched else " · proposal")
        self.selection_lbl.setText(
            f"Camera: {self.selected or '—'}  |  LiDAR: "
            f"{('#' + selected_id) if selected_id else '—'}{pair_state}")
        if selected_id:
            box = next((b for b in self.review.boxes
                        if str(b.get("obj_id") or "").strip() == selected_id), None)
            calibration = self.scene_io.calibration()
            if box and calibration:
                uv, depth = calibration.project(box3d_corners(box["psr"]))
                if np.isfinite(uv).all() and np.all(depth > 0):
                    projection = uv
        self.image_view.show_frame(self.image, self.review.tracks, self.assign,
                                   colors, self.selected, projection)
        self.bev_view.show_frame(self.points, self.review.boxes, colors, selected_id)

        while self.cards_lay.count():
            item = self.cards_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for t in self.review.tracks:
            lid = self.assign.get(t.key)
            opt = next((o for o in t.options if o["lidar_id"] == lid), None) if lid else None
            saved = bool(lid) and lid == t.applied_id
            geo_bad = bool(opt) and not opt.get("geo_ok", True)
            absent = bool(lid) and opt is None
            label = f"{'t' + t.key if not t.key.startswith('@') else t.key} → " \
                    f"{('#' + lid) if lid else ('✗' if (t.key in self.rejected or t.rejected) else '—')}"
            if absent or geo_bad:
                label += " ⚠"
            if saved:
                label += " ✓"
            elif lid and t.key not in self.touched:
                label += " (proposal)"
            elif t.key in self.touched and self.offered.get(self.review.frame, {}).get(t.key, "") == (lid or None):
                label += " (to confirm)"  # offered from outside: a pair, or "no LiDAR box"
            if t.key in flagged:
                label += " (check)"
            btn = QToolButton()
            btn.setText(label)
            btn.setToolTip(
                (f"check: {flagged[t.key]} · " if flagged.get(t.key) else "")
                + ("already saved in the labels" if saved else "")
                + (" · ⚠ geometry contradicts this pairing" if geo_bad else "")
                + (" · ⚠ id has no matchable box in this frame" if absent else ""))
            btn.clicked.connect(lambda _c=False, k=t.key: self._select_track(k))
            self.cards_lay.addWidget(btn)
        self.cards_lay.addStretch(1)
        self._update_pending()

    def _select_track(self, key: str) -> None:
        self.selected = key
        while self.cand_lay.count():
            item = self.cand_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        t = next((x for x in (self.review.tracks if self.review else []) if x.key == key), None)
        if t is None:
            self.cand_area.hide()
            return
        head = QLabel(f"{'t' + key if not key.startswith('@') else key} →")
        self.cand_lay.addWidget(head)
        none_btn = QPushButton("— none —")
        none_btn.clicked.connect(lambda: self._pick(key, None))
        self.cand_lay.addWidget(none_btn)
        colors = self._colors()
        for o in sorted(t.options, key=_candidate_sort_key):
            txt = f"#{o['lidar_id']}  {o['score']:.2f}"
            if o.get("proj_px") is not None:
                txt += f"  {o['proj_px']:.0f}px"
            if not o.get("geo_ok", True):
                txt += " ⚠"
            b = QPushButton(txt)
            col = colors.get(o["lidar_id"], GRAY)
            selected = self.assign.get(key) == o["lidar_id"]
            b.setStyleSheet(("border: 3px solid #e6b800; font-weight: bold;" if selected
                             else f"border: 1px solid {col.name()};")
                            + ("opacity: 0.6;" if not o.get("geo_ok", True) else ""))
            b.setToolTip("projected distance from this rect"
                         + ("" if o.get("geo_ok", True)
                            else " — GEOMETRY SAYS NO (never auto-picked)"))
            b.clicked.connect(lambda _c=False, lid=o["lidar_id"]: self._pick(key, lid))
            self.cand_lay.addWidget(b)
        self.cand_lay.addStretch(1)
        self.cand_area.show()
        self._redraw_keep_candidates()

    def _redraw_keep_candidates(self) -> None:
        # image/bev/cards refresh without rebuilding the candidate row
        sel = self.selected
        cand_visible = self.cand_area.isVisible()
        self._redraw()
        self.selected = sel
        if cand_visible:
            self.cand_area.show()

    def _pick(self, key: str, lid: Optional[str]) -> None:
        if lid and sum(str(b.get("obj_id") or "").strip() == lid
                       for b in self.review.boxes) != 1:
            self._status("Cannot assign an ID that is missing or duplicated in LiDAR labels")
            return
        self.touched.add(key)
        if lid:
            self.rejected.discard(key)
            for k in list(self.assign):
                if k != key and self.assign.get(k) == lid:
                    self.assign[k] = None
                    self.rejected.discard(k)
                    self.touched.add(k)
        else:
            self.rejected.add(key)
        self.assign[key] = lid
        self._select_track(key)

    def _pick_bev(self, lid: str) -> None:
        if self.selected is None:
            self._status("Select a camera box first, then click a LiDAR box")
            return
        self._pick(self.selected, lid)

    # ---- decisions / apply ----

    def _decisions(self) -> List[dict]:
        """Uncommitted diffs vs the labels for the CURRENT frame: the frame as it is shown.

        A suggestion of the model that the user left as it is counts as confirmed, so
        Apply saves it with the user's own choices; ``_proposal`` marks such a decision.
        """
        out: List[dict] = []
        for t in (self.review.tracks if self.review else []):
            lid = self.assign.get(t.key)
            if t.key not in self.touched:
                if lid and lid != t.applied_id:
                    out.append({"rect": t.rect, "lidar_id": lid, "_key": t.key, "_proposal": True})
                continue
            if lid:
                if lid != t.applied_id:
                    out.append({"rect": t.rect, "lidar_id": lid, "_key": t.key})
            elif t.key in self.rejected and (t.applied_id or not t.rejected):
                out.append({"rect": t.rect, "none": True, "_key": t.key})
            elif not lid and t.applied_id:
                out.append({"rect": t.rect, "clear": True, "_key": t.key})
        return out

    def _bank_current(self) -> None:
        if self.review is None:
            return
        d = self._decisions()
        if d:
            self.staged[self.review.frame] = d
        else:
            self.staged.pop(self.review.frame, None)

    def _update_pending(self) -> None:
        self._bank_current()
        n = sum(len(v) for v in self.staged.values())
        msg = (f"{n} decision(s) pending across {len(self.staged)} frame(s)"
               if n else
               (f"nothing pending · {self.applied_total} applied ✓ this session"
                if self.applied_total else "nothing pending"))
        self.apply_all_btn.setText(f"Apply all ({n})")
        self.apply_all_btn.setEnabled(n > 0)
        self.pending_btn.setEnabled(n > 0 or bool(self.checks))
        cur = len(self._decisions())
        self.apply_btn.setText(f"Apply this frame ({cur})" if cur else "Apply this frame")
        self.apply_btn.setEnabled(cur > 0)
        self._status(msg)

    def _strip(self, decisions: List[dict]) -> List[dict]:
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in decisions]

    def _apply_frame(self) -> None:
        if self.review is None:
            return
        r = self.scene_io.apply(self.review.frame, self._strip(self._decisions()))
        self.applied_total += r["written"] + r["marked"]
        self.staged.pop(self.review.frame, None)
        self._status(f"✓ applied {r['written']} pair(s)"
                     + (f" + {r['marked']} rejection(s)" if r["marked"] else ""))
        self.worker.submit("match", self.cur, self.thr.value())

    def _apply_all(self) -> None:
        self._bank_current()
        total_w = total_m = frames = 0
        for frame, dec in list(self.staged.items()):
            r = self.scene_io.apply(frame, self._strip(dec))
            total_w += r["written"]
            total_m += r["marked"]
            frames += 1
        self.staged.clear()
        self.applied_total += total_w + total_m
        self._status(f"✓ applied {total_w} pair(s)"
                     + (f" + {total_m} rejection(s)" if total_m else "")
                     + f" → {frames} frame(s)")
        self.worker.submit("match", self.cur, self.thr.value())

    def closeEvent(self, event) -> None:  # pragma: no cover
        self.worker.submit("quit")
        self.worker.wait(2000)
        super().closeEvent(event)


# -------------------------------------------------------------------- main --

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="xcalib-review",
        description="Review camera-LiDAR pairs and calibrate from the saved ones "
                    "(SUSTechPOINTS-format scenes).")
    ap.add_argument("--scene", required=True, help="scene directory (SUSTechPOINTS layout)")
    ap.add_argument("--camera", required=True, help="camera name (folder under camera/)")
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    ap.add_argument("--notebook-workflow", action="store_true",
                    help="Calibrate takes the camera intrinsics saved in the scene instead of asking "
                         "for them; the notebooks open the window this way")
    ap.add_argument("--proposals",
                    help='JSON of pairs to confirm: {"<frame>": [{"rect": [x1, y1, x2, y2], '
                         '"lidar_id": "<id>"}]}; "none": true instead of the id proposes that the '
                         'camera box has no LiDAR box. They open as unsaved decisions; Apply saves them')
    ap.add_argument("--check",
                    help='JSON of camera boxes to look at: {"<frame>": [{"rect": [x1, y1, x2, y2], '
                         '"note": "<why>"}]}. They are marked on their cards; nothing is changed')
    ap.add_argument("--only-listed", action="store_true",
                    help="Step through the frames named by --proposals and --check alone")
    ap.add_argument("--context", type=int, default=0, metavar="N",
                    help="With --only-listed, also step through N frames before and after each listed frame")
    ap.add_argument("--accept-calibration", action="store_true",
                    help="Save a Calibrate / Recalibrate result without asking when it passes the "
                         "automatic checks; a doubtful result is still shown for a decision")
    args = ap.parse_args(argv)

    scene = Scene(args.scene, args.camera)
    if not scene.frames():
        print(f"no lidar frames found under {scene.root}/lidar", file=sys.stderr)
        return 2
    proposals = None
    if args.proposals:
        with open(args.proposals, encoding="utf-8") as f:
            proposals = json.load(f)
    check = None
    if args.check:
        with open(args.check, encoding="utf-8") as f:
            check = json.load(f)

    app = QApplication(sys.argv[:1])
    win = ReviewWindow(scene, device=args.device, notebook_workflow=args.notebook_workflow,
                       proposals=proposals, check=check, listed_only=args.only_listed,
                       context=args.context, accept_calibration=args.accept_calibration)
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
