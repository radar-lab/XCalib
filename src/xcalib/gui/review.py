"""
xcalib-review — standalone Qt review GUI for camera-LiDAR matching.

The human-in-the-loop complement to one-shot adaptation: `observe()`/`adapt()`
self-correct under SMALL drift, but once the viewpoint moves past what the
pseudo-label gate tolerates, the model needs a human to say which pairs are
right. This tool shows the image with 2D boxes beside a top-down LiDAR view, with
one card per camera box and a ranked, geometry-annotated candidate menu, and runs
end-to-end on xcalib alone. Labels and images use a SUSTechPOINTS-style on-disk
layout (see `xcalib.gui.scene`).

    pip install "xcalib[gui]"
    xcalib-review --scene /data/MySite --camera cam_front

Workflow: step frames (the matcher proposes gated one-to-one pairs) → fix the
wrong ones (click a card, pick the right #id or — none —) → Apply (writes the
decisions into the fusion labels). Then refine, in one of two tiers:

- "Refine on corrections" — the one-shot tier: banks every verified pair and
  adapts the matcher in memory, with a save/discard prompt showing the
  before/after retrieval quality. Fast, in-session, ephemeral.
- "Refine (accumulate)" — the continual tier: records every verified pair into
  the DURABLE corpus (``xcalib.continual.record_verified_pairs``) and runs the
  GATED ``run_continual_refine`` over it. Accumulates across sessions and only
  publishes a revision that beats the current model on held-out validation.

Repeat until the site is stable.
"""

from __future__ import annotations

import argparse
import colorsys
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
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
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

from .scene import FrameReview, Scene, box3d_corners, load_points, match_review

GRAY = QColor("#8a8a8a")


def _id_color(idx: int) -> QColor:
    """Stable, visually distinct color per matched id (golden-angle hues)."""
    h = (idx * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


# ------------------------------------------------------------------ worker --

class MatcherWorker(QThread):
    """Owns the Matcher on a background thread; the UI queues jobs to it."""

    review_ready = Signal(object, object, object)   # FrameReview, image, points
    refine_done = Signal(dict)
    continual_done = Signal(dict)
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
            from ..engine.matcher import Matcher

            resolved = self.scene.model_dir()
            if resolved is None:
                self.failed.emit(
                    f"no trained model at {self.scene.root}/calib/xcalib/"
                    f"{self.scene.camera}_model — train one first")
                return
            model_dir, base = resolved
            self.status.emit(f"loading {base} …")
            self.matcher = Matcher.from_pretrained(base, weights=model_dir,
                                                   device=self.device)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"could not load the matcher: {e}")
            return
        self.status.emit("matcher ready")

        while True:
            job = self.jobs.get()
            if job[0] == "quit":
                return
            try:
                if job[0] == "match":
                    self._do_match(job[1], job[2])
                elif job[0] == "refine":
                    self._do_refine(job[1], job[2])
                elif job[0] == "continual":
                    self._do_continual_refine(job[1], job[2])
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

    def _do_refine(self, frames: List[str], adapt_steps: int) -> None:
        """Bank every human-verified pair (shared ids) and adapt the adapters."""
        import cv2  # noqa: PLC0415

        calib = self.scene.calibration()
        intr = calib.intrinsics if calib is not None else None
        if intr is None:
            from ..protocol import CameraIntrinsics

            intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=0.0, cy=0.0)
        session = self.matcher.oneshot(intr)

        banked = used = 0
        for n, frame in enumerate(frames):
            rects = self.scene.load_rects(frame)
            boxes = self.scene.load_boxes(frame)
            by_id = {str(b.get("obj_id", "")).strip(): j for j, b in enumerate(boxes)
                     if str(b.get("obj_id", "")).strip()}
            pairs = [(i, by_id[str(r.get("obj_id", "")).strip()])
                     for i, r in enumerate(rects)
                     if str(r.get("obj_id", "")).strip() in by_id]
            if not pairs:
                continue
            path = self.scene.image_path(frame)
            lp = self.scene.lidar_path(frame)
            if not path or not lp:
                continue
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            points = load_points(lp)
            banked += session.bank_pairs(image, points, Scene.rects_xyxy(rects),
                                         Scene.boxes_extents(boxes), pairs)
            used += 1
            self.status.emit(f"banking corrections {n + 1}/{len(frames)} — {banked} pairs")
        if banked < 8:
            self.failed.emit(f"only {banked} verified pair(s) banked (need ≥ 8) — "
                             "correct & apply more frames first")
            return
        self.status.emit(f"adapting on {banked} pairs …")
        report = session.adapt(steps=adapt_steps)
        self.refine_done.emit({
            "frames": used, "banked": banked,
            "top1_before": float(report.top1_before),
            "top1_after": float(report.top1_after),
            "loss_first": float(report.loss_first),
            "loss_last": float(report.loss_last),
        })

    def _do_continual_refine(self, frames: List[str], threshold: float) -> None:
        """Record every corrected frame's verified pairs into the DURABLE corpus, then
        run the gated continual refine over it (package-native loader). Unlike one-shot
        adapt, this accumulates across sessions and only publishes a revision that beats
        the parent on held-out validation."""
        import uuid  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        from ..continual import (  # noqa: PLC0415
            open_store, record_verified_pairs, run_continual_refine)

        resolved = self.scene.model_dir()
        if resolved is None:
            self.failed.emit("no model dir — train a base model first")
            return
        root = Path(self.scene.root).resolve()
        data_root, scene_name = str(root.parent), root.name
        store = open_store(resolved[0])
        recorded = 0
        try:
            for n, frame in enumerate(frames):
                review = match_review(self.matcher, self.scene, frame, threshold)
                pairs = _verified_pairs(review)
                if pairs:
                    recorded += record_verified_pairs(
                        store, data_root, scene_name, self.scene.camera, frame, pairs)
                self.status.emit(
                    f"recording corrections {n + 1}/{len(frames)} — {recorded} pair(s)")
        finally:
            store.close()
        if recorded == 0:
            self.failed.emit("no verified pairs to record — accept & apply some first")
            return
        self.status.emit("gated continual refine over the corpus …")
        result = run_continual_refine(
            data_root, scene_name, self.scene.camera,
            "review-" + uuid.uuid4().hex[:12],
            lambda p: self.status.emit(f"[{p.phase}] {p.message}"))
        payload = {"recorded": recorded, "outcome": result.outcome,
                   "new_revision": result.new_revision or "",
                   "reasons": list(result.gate_reasons)}
        if result.validation_parent and result.validation_candidate:
            payload["primary_before"] = float(result.validation_parent.primary_score)
            payload["primary_after"] = float(result.validation_candidate.primary_score)
        self.continual_done.emit(payload)


def _verified_pairs(review) -> list:
    """VerifiedPair rows from a frame review's PERSISTED human decisions.

    ``applied_id`` is the id the human already wrote to the labels (a positive pair);
    ``rejected`` is an explicit no-match. The ranked ``options`` become the candidate
    menu and ``suggestion`` the recorded model proposal, so provenance can tell 'model
    was right' from 'human corrected'. Rects with no decision are skipped."""
    from ..continual import VerifiedPair  # noqa: PLC0415

    pairs = []
    for t in review.tracks:
        cands = [{"lidar_key": "id:" + o["lidar_id"], "score": o["score"]}
                 for o in t.options if o.get("lidar_id")]
        proposal = ({"decision": "positive", "lidar_key": "id:" + t.suggestion}
                    if t.suggestion else None)
        track = t.key if t.key.isdigit() else None
        if t.applied_id:
            pairs.append(VerifiedPair(
                rect=list(t.rect), decision="positive", lidar_key="id:" + t.applied_id,
                track=track, candidates=cands, proposal=proposal))
        elif t.rejected:
            pairs.append(VerifiedPair(
                rect=list(t.rect), decision="no_match", lidar_key=None,
                track=track, candidates=cands, proposal=proposal))
    return pairs


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
                   selected: Optional[str]) -> None:
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
        _fit_scene(self)

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
        if self.scene() and not self.scene().itemsBoundingRect().isEmpty():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)


def _fit_only(view: QGraphicsView) -> None:
    """A view that always shows its whole scene never needs scrollbars.

    With them on, a scrollbar that appears narrows the viewport, ``fitInView``
    re-fits to the new size, the scrollbar disappears, and the view can resize
    itself in a loop that pins the CPU.
    """
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


def _fit_scene(view: QGraphicsView) -> None:
    """Fit ``view`` to the items it holds now.

    A scene without an explicit rect only ever grows to the union of every item
    it has held, so after browsing a larger frame it stays larger than the
    current one; pin the rect to the current items.
    """
    scene = view.scene()
    rect = scene.itemsBoundingRect()
    scene.setSceneRect(rect)
    view.fitInView(rect, Qt.KeepAspectRatio)


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

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor("#101010")))
        _fit_only(self)
        self._angle_deg = 0.0
        self._last: Optional[tuple] = None  # (points, boxes, id_colors) for re-render

    def rotate_view(self, delta_deg: float) -> None:
        """Turn the top-down view by ``delta_deg`` and re-render the last frame."""
        self._angle_deg = (self._angle_deg + delta_deg) % 360.0
        if self._last is not None:
            self.show_frame(*self._last)

    def show_frame(self, points: Optional[np.ndarray], boxes, id_colors) -> None:
        sc = self.scene()
        sc.clear()
        self._last = (points, boxes, id_colors)
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

        img = QImage(size, size, QImage.Format_RGB32)
        img.fill(QColor("#101010"))
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
        for u, v, s in zip(px, py, shade):
            g = int(s)
            img.setPixel(int(u), int(v), QColor(g, g, g).rgb())
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
            sc.addPolygon(poly, QPen(color, 2))
            if lid:
                txt = sc.addSimpleText(f"#{lid}")
                txt.setBrush(QBrush(color))
                r = poly.boundingRect()
                txt.setPos(r.x(), r.y() - 14)
        _fit_scene(self)

    def resizeEvent(self, event) -> None:  # pragma: no cover
        if self.scene() and not self.scene().itemsBoundingRect().isEmpty():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)


# ------------------------------------------------------------- main window --

class ReviewWindow(QMainWindow):
    """The review surface: toolbar, image | BEV, track cards, candidate row."""

    def __init__(self, scene: Scene, device: Optional[str] = None):
        super().__init__()
        self.scene_io = scene
        self.frames = scene.frames()
        self.cur = self.frames[0] if self.frames else ""
        self.review: Optional[FrameReview] = None
        self.image: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self.assign: Dict[str, Optional[str]] = {}
        self.rejected: set = set()
        self.staged: Dict[str, List[dict]] = {}
        self.applied_total = 0

        self.setWindowTitle(f"xcalib review — {os.path.basename(scene.root)}/{scene.camera}")
        self.resize(1500, 950)
        self._build_ui()

        self.worker = MatcherWorker(scene, device)
        self.worker.review_ready.connect(self._on_review)
        self.worker.refine_done.connect(self._on_refined)
        self.worker.continual_done.connect(self._on_continual)
        self.worker.failed.connect(lambda m: self._status(f"⚠ {m}"))
        self.worker.status.connect(self._status)
        self.worker.start()
        if self.cur:
            self.worker.submit("match", self.cur, self.thr.value())

    # ---- UI scaffolding ----

    def _build_ui(self) -> None:
        top = QWidget()
        root = QVBoxLayout(top)
        root.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        self.prev_btn = QPushButton("‹")
        self.next_btn = QPushButton("›")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        self.frame_lbl = QLabel("—")
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("#")
        self.jump_edit.setFixedWidth(70)
        self.jump_edit.returnPressed.connect(self._jump)
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
        self.assigned_lbl = QLabel("")
        self.refine_btn = QPushButton("Refine on corrections")
        self.refine_btn.setToolTip(
            "Bank every verified pair (shared ids in the labels) and adapt the "
            "matcher; you confirm before the refined weights replace the model.")
        self.refine_btn.clicked.connect(self._refine)
        self.continual_btn = QPushButton("Refine (accumulate)")
        self.continual_btn.setToolTip(
            "Record every verified pair into the durable corpus, then run the GATED "
            "continual refine over it — accumulates across sessions and only publishes "
            "a revision that beats the current model on held-out validation.")
        self.continual_btn.clicked.connect(self._continual_refine)
        self.apply_btn = QPushButton("Apply this frame")
        self.apply_btn.clicked.connect(self._apply_frame)
        self.apply_all_btn = QPushButton("Apply all")
        self.apply_all_btn.clicked.connect(self._apply_all)
        for w in (self.prev_btn, self.frame_lbl, self.next_btn, self.jump_edit,
                  self.thr, self.rot_l_btn, self.rot_r_btn, self.assigned_lbl):
            bar.addWidget(w)
        bar.addStretch(1)
        for w in (self.refine_btn, self.continual_btn, self.apply_btn,
                  self.apply_all_btn):
            bar.addWidget(w)
        root.addLayout(bar)

        split = QSplitter()
        self.image_view = ImageView()
        self.image_view.track_clicked.connect(self._select_track)
        self.bev_view = BevView()
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
        if 1 <= n <= len(self.frames):
            self._bank_current()
            self.cur = self.frames[n - 1]
            self.worker.submit("match", self.cur, self.thr.value())

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Left / Right arrow = previous / next frame, unless the jump box has
        # focus so it keeps its own key handling.
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Right) and not isinstance(
            self.focusWidget(), QLineEdit
        ):
            self._step(-1 if key == Qt.Key_Left else 1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _rematch(self) -> None:
        if self.cur:
            self.worker.submit("match", self.cur, self.thr.value())

    # ---- review state ----

    def _on_review(self, review: FrameReview, image, points) -> None:
        self.review, self.image, self.points = review, image, points
        # seed: applied ids lock in; staged session decisions come next;
        # persisted/session rejections stay none; the gated assignment fills
        # the rest
        assign: Dict[str, Optional[str]] = {}
        used: set = set()
        for t in review.tracks:
            if t.applied_id:
                assign[t.key] = t.applied_id
                used.add(t.applied_id)
            if t.rejected and not t.key.startswith("@"):
                self.rejected.add(t.key)
        for d in self.staged.get(review.frame, []):
            key = d.get("_key")
            if key and key not in assign:
                if d.get("none"):
                    assign[key] = None
                elif d.get("lidar_id") and d["lidar_id"] not in used:
                    assign[key] = d["lidar_id"]
                    used.add(d["lidar_id"])
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
        idx = self.frames.index(self.cur) + 1 if self.cur in self.frames else 0
        self.frame_lbl.setText(f"frame {idx}/{len(self.frames)} · {self.cur}")
        assigned = sum(1 for t in self.review.tracks if self.assign.get(t.key))
        self.assigned_lbl.setText(f"{assigned}/{len(self.review.tracks)} assigned")
        self.image_view.show_frame(self.image, self.review.tracks, self.assign,
                                   colors, self.selected)
        self.bev_view.show_frame(self.points, self.review.boxes, colors)

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
            btn = QToolButton()
            btn.setText(label)
            btn.setToolTip(
                ("already saved in the labels" if saved else "")
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
        for o in t.options:
            txt = f"#{o['lidar_id']}  {o['score']:.2f}"
            if o.get("proj_px") is not None:
                txt += f"  {o['proj_px']:.0f}px"
            if not o.get("geo_ok", True):
                txt += " ⚠"
            b = QPushButton(txt)
            col = colors.get(o["lidar_id"], GRAY)
            b.setStyleSheet(f"border: 1px solid {col.name()};"
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
        if lid:
            self.rejected.discard(key)
            for k in list(self.assign):
                if k != key and self.assign.get(k) == lid:
                    self.assign[k] = None
        else:
            self.rejected.add(key)
        self.assign[key] = lid
        self._redraw_keep_candidates()

    # ---- decisions / apply ----

    def _decisions(self) -> List[dict]:
        """Uncommitted diffs vs the labels for the CURRENT frame."""
        out: List[dict] = []
        for t in (self.review.tracks if self.review else []):
            lid = self.assign.get(t.key)
            if lid:
                if lid != t.applied_id:
                    out.append({"rect": t.rect, "lidar_id": lid, "_key": t.key})
            elif t.key in self.rejected and (t.applied_id or not t.rejected):
                out.append({"rect": t.rect, "none": True, "_key": t.key})
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
        cur = len(self._decisions())
        self.apply_btn.setText(f"Apply this frame ({cur})" if cur else "Apply this frame")
        self.apply_btn.setEnabled(cur > 0)
        self._status(msg)

    def _strip(self, decisions: List[dict]) -> List[dict]:
        return [{k: v for k, v in d.items() if k != "_key"} for d in decisions]

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

    # ---- adaptation ----

    def _corrected_frames(self, cap: int = 60) -> List[str]:
        """Frames with ≥1 verified pair (a rect whose id matches a labelled box);
        every one is training signal. Capped to bound the pass."""
        frames = []
        for f in self.frames:
            rects = self.scene_io.load_rects(f)
            if not rects:
                continue
            ids3 = {str(b.get("obj_id", "")).strip()
                    for b in self.scene_io.load_boxes(f)} - {""}
            if any(str(r.get("obj_id", "")).strip() in ids3 for r in rects):
                frames.append(f)
        if len(frames) > cap:
            frames = frames[::len(frames) // cap][:cap]
        return frames

    def _refine(self) -> None:
        """One-shot tier: bank verified pairs + adapt in place (ephemeral)."""
        frames = self._corrected_frames()
        if not frames:
            self._status("no corrected frames yet — accept & apply some pairs first")
            return
        self.refine_btn.setEnabled(False)
        self.worker.submit("refine", frames, 100)

    def _continual_refine(self) -> None:
        """Continual tier: record verified pairs to the durable corpus + gated refine."""
        frames = self._corrected_frames()
        if not frames:
            self._status("no corrected frames yet — accept & apply some pairs first")
            return
        self.continual_btn.setEnabled(False)
        self.worker.submit("continual", frames, self.thr.value())

    def _on_continual(self, r: dict) -> None:
        self.continual_btn.setEnabled(True)
        lines = [f"Recorded {r['recorded']} verified pair(s) into the durable corpus.",
                 "", f"Continual refine: {r['outcome']}."]
        if r.get("new_revision"):
            lines.append(f"Published revision {r['new_revision']}. This window keeps "
                         "matching with the checkpoint in the model folder; "
                         "xcalib.continual.resolve_model loads the revision.")
        if "primary_before" in r:
            lines.append(f"validation primary score: {r['primary_before']:.3f} "
                         f"→ {r['primary_after']:.3f}")
        if r["outcome"] != "accepted" and r.get("reasons"):
            lines.append("gate held back: " + "; ".join(r["reasons"]))
        box = QMessageBox(self)
        box.setWindowTitle("Refine (accumulate)")
        box.setText("\n".join(lines))
        box.exec()

    def _on_refined(self, report: dict) -> None:
        self.refine_btn.setEnabled(True)
        msg = (f"Adapted on {report['banked']} verified pairs from "
               f"{report['frames']} frame(s).\n\n"
               f"in-bank retrieval top-1: {report['top1_before']:.3f} → "
               f"{report['top1_after']:.3f}\n"
               f"loss: {report['loss_first']:.4f} → {report['loss_last']:.4f}\n\n"
               f"Save the refined weights over the scene's model?")
        box = QMessageBox(self)
        box.setWindowTitle("Refine on corrections")
        box.setText(msg)
        box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
        if box.exec() == QMessageBox.Save:
            model_dir, _base = self.scene_io.model_dir()
            self.worker.matcher.save_pretrained(model_dir)
            self._status("refined weights saved — matching now uses them")
            self.worker.submit("match", self.cur, self.thr.value())
        else:
            self._status("refined weights discarded (model on disk unchanged)")

    def closeEvent(self, event) -> None:  # pragma: no cover
        self.worker.submit("quit")
        self.worker.wait(2000)
        super().closeEvent(event)


# -------------------------------------------------------------------- main --

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="xcalib-review",
        description="Standalone camera-LiDAR match review + human-in-the-loop "
                    "adaptation (SUSTechPOINTS-format scenes).")
    ap.add_argument("--scene", required=True, help="scene directory (SUSTechPOINTS layout)")
    ap.add_argument("--camera", required=True, help="camera name (folder under camera/)")
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    args = ap.parse_args(argv)

    scene = Scene(args.scene, args.camera)
    if not scene.frames():
        print(f"no lidar frames found under {scene.root}/lidar", file=sys.stderr)
        return 2

    app = QApplication(sys.argv[:1])
    win = ReviewWindow(scene, device=args.device)
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
