"""Pairs that exist already: copied onto a scene's camera boxes, counted and drawn.

A pair is a camera box that carries the id of a LiDAR box (``obj_id``) in the
scene's camera labels, ``label_fusion/camera/<camera>/<frame>.json``; a camera
box whose object has no LiDAR box carries ``no_match``. Whoever paired the
frames of a delivery before has such files already. :func:`import_pairs`
copies their decisions onto the same camera boxes of a scene, so the work is
not done twice. :func:`summarize` says what a scene holds and where in the
image its pairs lie, and :func:`preview` draws one frame: every pair in its
own colour, in the camera image and in the point cloud seen from above.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scene import Scene, box3d_corners, load_points_xyz, track_of

_COLORS = ((245, 135, 66), (75, 180, 60), (75, 25, 230), (48, 130, 245), (180, 30, 145),
           (240, 240, 70), (230, 50, 240), (60, 245, 210))   # BGR
_GRAY = (150, 150, 150)


def _id(obj):
    return str(obj.get("obj_id") or "").strip()


def _objs(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    found = data.get("objs", []) if isinstance(data, dict) else data
    return [obj for obj in found or [] if isinstance(obj, dict) and isinstance(obj.get("rect"), dict)]


def _key(rect):
    return tuple(round(float(rect[side]), 1) for side in ("x1", "y1", "x2", "y2"))


def label_folder(folder, camera):
    """The folder with the per-frame camera label files inside what was chosen: the
    folder itself, the ``label_fusion/camera/<camera>`` folder of a scene, or that of
    another camera name when the scene has only one. ``ValueError`` when there is none."""
    folder = Path(folder)
    fusion = folder / "label_fusion" / "camera"
    # the folder itself, a scene's label folder, and a label_fusion/camera folder chosen as such
    candidates = [folder, fusion / camera, folder / camera]
    for parent in (fusion, folder):   # another camera name only when it is the only camera folder
        others = sorted(path for path in parent.iterdir() if path.is_dir()) if parent.is_dir() else []
        if len(others) == 1:
            candidates += others
    for candidate in candidates:
        if candidate.is_dir() and any(_objs(path) for path in sorted(candidate.glob("*.json"))[:20]):
            return candidate
    raise ValueError(f"{folder} holds no saved pairs: choose a scene folder in which pairs were "
                     f"saved, its label_fusion/camera/{camera} folder, or a folder of saved label files")


def import_pairs(scene: Scene, folder):
    """Copy the pairs and "no LiDAR box" marks of ``folder`` onto the scene's camera boxes.

    A camera box of the folder is found in the scene by its rectangle, else by
    its track id. Only camera boxes without a decision take one: what the scene
    holds already is kept. A pair is copied only when the scene's frame has a
    LiDAR box with that id and no other camera box of the frame holds it there
    already (a person may have moved that LiDAR box to another camera box).
    Returns the counts: ``files`` in the folder, ``frames`` changed, ``pairs`` and
    ``none`` copied, ``same`` (the scene holds that decision already, e.g. the
    folder was copied in before), ``kept`` (decided otherwise already, which
    stays), ``taken`` (another camera box of the frame holds that LiDAR box here),
    ``unknown_lidar`` (no such LiDAR box here), ``unmatched`` (no such camera box
    here) and ``other_frames`` (files of frames the scene lacks).
    """
    source = label_folder(folder, scene.camera)
    own = Path(scene.root) / "label_fusion" / "camera" / scene.camera
    if source.resolve() == own.resolve():
        raise ValueError("that folder is this scene's own; choose the folder the pairs come from")
    report = dict(folder=str(source), files=0, frames=0, pairs=0, none=0, same=0, kept=0, taken=0,
                  unknown_lidar=0, unmatched=0, other_frames=0)

    def decided(obj):   # the LiDAR id of a pair, "" for "no LiDAR box"
        return "" if obj.get("no_match") else _id(obj)

    frames = set(scene.frames())
    for path in sorted(source.glob("*.json")):
        theirs = [obj for obj in _objs(path) if _id(obj) or obj.get("no_match")]
        if not theirs:
            continue
        report["files"] += 1
        if path.stem not in frames:
            report["other_frames"] += 1
            continue
        mine = scene.load_rects(path.stem)
        by_rect = {_key(rect["rect"]): rect for rect in mine}
        by_track = {}
        for rect in mine:
            if track_of(rect):
                by_track.setdefault(track_of(rect), []).append(rect)
        lidar = {_id(box) for box in scene.load_boxes(path.stem)}
        held = {_id(rect) for rect in mine if _id(rect) and not rect.get("no_match")}   # paired here already
        decisions = []
        for obj in theirs:
            target = by_rect.get(_key(obj["rect"]))
            if target is None and len(by_track.get(track_of(obj) or "", [])) == 1:
                target = by_track[track_of(obj)][0]
            if target is None:
                report["unmatched"] += 1
            elif _id(target) or target.get("no_match"):
                report["same" if decided(target) == decided(obj) else "kept"] += 1
            elif obj.get("no_match"):
                decisions.append({"rect": list(_key(target["rect"])), "none": True})
            elif _id(obj) in held:   # another camera box of this frame holds that LiDAR box here
                report["taken"] += 1
            elif _id(obj) in lidar:
                held.add(_id(obj))
                decisions.append({"rect": list(_key(target["rect"])), "lidar_id": _id(obj)})
            else:
                report["unknown_lidar"] += 1
        if decisions:
            done = scene.apply(path.stem, decisions)
            report["pairs"] += done["written"]
            report["none"] += done["marked"]
            report["frames"] += bool(done["written"] or done["marked"])
    return report


def _size(scene, frames):
    """Width and height of the scene's camera images in pixels (from the first image
    that reads), or ``None``."""
    import cv2

    for frame in frames:
        path = scene.image_path(frame)
        image = cv2.imread(path) if path else None
        if image is not None:
            return image.shape[1], image.shape[0]
    return None


def summarize(scene: Scene):
    """What the scene's camera boxes hold: ``pairs``, ``none`` and ``undecided`` boxes,
    ``undecided_beside_pairs`` (undecided boxes in frames with a saved pair: training
    counts them as "no LiDAR box") and ``frames_with_undecided`` (those frames),
    ``frames``, ``frames_with_pairs``, ``frames_ready`` (two pairs or more),
    ``objects`` (different LiDAR ids), ``per_frame`` (frame -> pairs), ``thirds``
    (pairs the homography can use whose box centre lies in the left, middle and right
    third of the image), ``cut`` (pairs whose camera box the image edge cuts) and
    ``wide`` (pairs whose camera box is wider than ``WIDE_BOX`` of the image: objects
    close to the camera); the homography leaves the cut and the wide ones out.
    ``thirds``, ``cut`` and ``wide`` are ``None`` when no image reads."""
    from .pair_homography import EDGE_PX, WIDE_BOX

    frames = scene.frames()
    size = _size(scene, frames)
    report = dict(pairs=0, none=0, undecided=0, undecided_beside_pairs=0, frames_with_undecided=[],
                  frames=len(frames), frames_with_pairs=0, frames_ready=0, objects=0, per_frame={},
                  thirds=[0, 0, 0] if size else None, cut=0 if size else None,
                  wide=0 if size else None)
    objects = set()
    for frame in frames:
        count = undecided = 0
        for rect in scene.load_rects(frame):
            if rect.get("no_match"):
                report["none"] += 1
            elif _id(rect):
                count += 1
                objects.add(_id(rect))
                if size:
                    box = rect["rect"]
                    if (box["x1"] <= EDGE_PX or box["y1"] <= EDGE_PX or box["x2"] >= size[0] - 1 - EDGE_PX
                            or box["y2"] >= size[1] - 1 - EDGE_PX):
                        report["cut"] += 1
                    elif box["x2"] - box["x1"] > WIDE_BOX * size[0]:
                        report["wide"] += 1
                    else:
                        centre = (box["x1"] + box["x2"]) / 2
                        report["thirds"][int(min(2, max(0, centre * 3 // size[0])))] += 1
            else:
                undecided += 1
        report["undecided"] += undecided
        if count and undecided:
            report["undecided_beside_pairs"] += undecided
            report["frames_with_undecided"].append(frame)
        report["per_frame"][frame] = count
        report["pairs"] += count
        report["frames_with_pairs"] += bool(count)
        report["frames_ready"] += count >= 2
    report["objects"] = len(objects)
    return report


def preview(scene: Scene, frame, *, suggestions=None, size=540, extent=45.0):
    """One frame as a check image (BGR): the camera image beside the point cloud seen
    from above. Every pair has its own colour, on the camera box and on its LiDAR
    box, with the LiDAR id written at both. A camera box marked as having no LiDAR
    box is gray with "none", one without a decision is white, and LiDAR boxes that
    belong to no pair are gray.

    ``suggestions`` are pairs that are not saved, in the form of
    ``xcalib.gui.model_pairs`` (``rect``, ``lidar_id``, ``score``): they are drawn
    like pairs, with the score after the id.
    """
    import cv2

    path = scene.image_path(frame)
    image = cv2.imread(path) if path else None
    if image is None:
        raise ValueError(f"frame {frame} has no readable camera image")
    suggested = {tuple(round(float(v), 1) for v in pair["rect"]): pair for pair in suggestions or []}
    colour = {}        # LiDAR id -> colour of its pair
    for rect in scene.load_rects(frame):
        corners = _key(rect["rect"])
        a, b = (int(corners[0]), int(corners[1])), (int(corners[2]), int(corners[3]))
        pair = suggested.get(corners)
        lid = pair["lidar_id"] if pair else ("" if rect.get("no_match") else _id(rect))
        if lid:
            shade = colour.setdefault(lid, _COLORS[len(colour) % len(_COLORS)])
            text = f"#{lid}" + (f" {pair['score']:.2f}" if pair and "score" in pair else "")
            cv2.rectangle(image, a, b, shade, 4)
            cv2.rectangle(image, (a[0], a[1] - 30), (a[0] + 17 * len(text), a[1]), shade, -1)
            cv2.putText(image, text, (a[0] + 3, a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif rect.get("no_match"):
            cv2.rectangle(image, a, b, _GRAY, 2)
            cv2.putText(image, "none", (a[0] + 3, a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _GRAY, 2,
                        cv2.LINE_AA)
        else:
            cv2.rectangle(image, a, b, (255, 255, 255), 1)
    image = cv2.resize(image, (round(image.shape[1] * size / image.shape[0]), size),
                       interpolation=cv2.INTER_AREA)

    def pixels(xy):  # x to the right, y up
        return np.round((np.asarray(xy) * [1, -1] + extent) * size / (2 * extent)).astype(np.int32)

    top = np.zeros((size, size, 3), np.uint8)
    cloud = scene.lidar_path(frame)
    if cloud:
        points = np.asarray(load_points_xyz(cloud), dtype=float)
        points = points[np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)]
        at = pixels(points[:, :2])
        at = at[((at >= 0) & (at < size)).all(axis=1)]
        top[at[:, 1], at[:, 0]] = 110
    for box in sorted(scene.load_boxes(frame), key=lambda box: _id(box) in colour):  # pairs on top
        corners = box3d_corners(box["psr"])
        outline = pixels((corners[[0, 2, 6, 4], :2] + corners[[1, 3, 7, 5], :2]) / 2)
        shade = colour.get(_id(box))
        cv2.polylines(top, [outline], True, shade or _GRAY, 2 if shade else 1, cv2.LINE_AA)
        if shade:
            cv2.putText(top, f"#{_id(box)}", tuple(map(int, outline.max(axis=0) + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, shade, 1, cv2.LINE_AA)
    return np.hstack((image, top))
