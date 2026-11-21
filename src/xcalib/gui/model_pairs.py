"""Pairs a scene's model suggests for the camera boxes nobody has decided yet.

The pairing window shows the model's suggestion on every card and saves only
what a person picked. Once a model fits the site, most suggestions are right,
and picking each of them by hand is the slow part. This module runs the scene's
model over the frames that still have an undecided camera box and writes its
suggestions in the form ``xcalib-review --proposals`` opens as unsaved
decisions: a person looks at each frame, corrects what is wrong, and Apply
saves the rest.

A suggestion is what the window would show: the gated one-to-one pick of
:func:`xcalib.gui.scene.match_review`, never a LiDAR box that a saved pair of
the frame already holds. Camera boxes with a saved pair or a "no LiDAR box"
mark are left alone, and nothing is written to the labels here. In a scene
with an extrinsic, the suggestions it does not support can be listed for the
window to mark, so a person knows where to look first.
"""

from __future__ import annotations

import json
from pathlib import Path


MIN_OVERLAP = 0.2  # IoU of the camera box with the projected LiDAR box
_KIND = {"car": "vehicle", "van": "vehicle", "truck": "vehicle", "bus": "vehicle",
         "pedestrian": "person", "person": "person", "bicyclerider": "person",
         "bicycle": "person", "motorcyclerider": "person", "scooterrider": "person"}


def _id(obj):
    return str(obj.get("obj_id") or "").strip()


def _kind(obj):
    return _KIND.get(str(obj.get("obj_type") or "").replace(" ", "").lower())


def open_frames(scene):
    """Frames with a camera box that has neither a saved pair nor a "no LiDAR box" mark."""
    return [frame for frame in scene.frames()
            if any(not _id(rect) and not rect.get("no_match") for rect in scene.load_rects(frame))]


def frame_suggestions(matcher, scene, frame, threshold=0.5):
    """The model's pairs for the undecided camera boxes of one frame.

    Returns ``[{"rect": [x1, y1, x2, y2], "lidar_id": id, "score": score}]``, the
    decisions ``xcalib-review --proposals`` takes.
    """
    from .scene import match_review

    review = match_review(matcher, scene, frame, threshold)
    counts = {}
    for box in review.boxes:
        counts[_id(box)] = counts.get(_id(box), 0) + 1
    used = {track.applied_id for track in review.tracks if track.applied_id}
    pairs = []
    for track in review.tracks:
        lid = track.suggestion
        if track.applied_id or track.rejected or not lid or lid in used or counts.get(lid) != 1:
            continue
        used.add(lid)
        score = next((option["score"] for option in track.options if option["lidar_id"] == lid), 0.0)
        pairs.append({"rect": [float(v) for v in track.rect], "lidar_id": lid, "score": float(score)})
    return pairs


def suggest_scene(matcher, scene, frames=None, threshold=0.5, progress=None):
    """``{frame: pairs}`` for every frame, of ``frames`` or of the open ones, with a suggestion."""
    frames = list(frames if frames is not None else open_frames(scene))
    found = {}
    for index, frame in enumerate(frames):
        if progress and index % 20 == 0:
            progress(f"The model is pairing the open frames: {index}/{len(frames)}")
        pairs = frame_suggestions(matcher, scene, frame, threshold)
        if pairs:
            found[frame] = pairs
    return found


def frame_pairs(scene, frame, calibration=None):
    """The pairs a saved extrinsic gives for the undecided camera boxes of one frame,
    smallest pixel distance first.

    A camera box gets the LiDAR box whose projected centre lies inside it, whose
    projected box overlaps it and that is the same kind of object; each LiDAR box
    goes to one camera box. Returns ``[{"rect": [x1, y1, x2, y2], "lidar_id": id,
    "px": distance}]``.
    """
    import numpy as np

    from ..assign import one_to_one_assign
    from .scene import box3d_corners

    calibration = calibration or scene.calibration()
    if calibration is None:
        raise ValueError("the scene has no extrinsic (calib/camera/<camera>.json)")
    rects = scene.load_rects(frame)
    boxes = scene.load_boxes(frame)
    counts = {}
    for box in boxes:
        counts[_id(box)] = counts.get(_id(box), 0) + 1
    taken = {_id(r) for r in rects if _id(r) and not r.get("no_match")}
    free = [r for r in rects if not _id(r) and not r.get("no_match")]
    lidar = [b for b in boxes if _id(b) and counts[_id(b)] == 1 and _id(b) not in taken]
    if not free or not lidar:
        return []
    distance = np.full((len(free), len(lidar)), np.inf)
    for j, box in enumerate(lidar):
        corners, depth = calibration.project(box3d_corners(box["psr"]))
        p = box["psr"]["position"]
        centre, centre_depth = calibration.project(np.array([[p["x"], p["y"], p["z"]]], dtype=float))
        if centre_depth[0] <= 0 or not (depth > 0).all() or not np.isfinite(corners).all():
            continue
        left, top, right, bottom = (corners[:, 0].min(), corners[:, 1].min(),
                                    corners[:, 0].max(), corners[:, 1].max())
        for i, rect in enumerate(free):
            r = rect["rect"]
            inside = r["x1"] <= centre[0, 0] <= r["x2"] and r["y1"] <= centre[0, 1] <= r["y2"]
            kinds = {_kind(rect), _kind(box)} - {None}
            if not inside or len(kinds) > 1:
                continue
            width = max(0.0, min(right, r["x2"]) - max(left, r["x1"]))
            height = max(0.0, min(bottom, r["y2"]) - max(top, r["y1"]))
            union = ((right - left) * (bottom - top)
                     + (r["x2"] - r["x1"]) * (r["y2"] - r["y1"]) - width * height)
            if union > 0 and width * height / union >= MIN_OVERLAP:
                distance[i, j] = np.hypot(centre[0, 0] - (r["x1"] + r["x2"]) / 2,
                                          centre[0, 1] - (r["y1"] + r["y2"]) / 2)
    pairs = []
    # smallest total distance = largest total of the negated distances
    wanted = np.where(np.isfinite(distance), -distance, -1e9)
    for i, j, _score in one_to_one_assign(wanted, threshold=-1e8):
        if np.isfinite(distance[i, j]):
            r = free[i]["rect"]
            pairs.append({"rect": [float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                          "lidar_id": _id(lidar[j]), "px": round(float(distance[i, j]), 1)})
    return sorted(pairs, key=lambda pair: pair["px"])


def projection_doubts(scene, pairs):
    """The suggested pairs that the scene's saved extrinsic does not support.

    The matcher's gate only drops a LiDAR box that projects far from the camera
    box. :func:`frame_pairs` is stricter: the projected centre has to lie inside
    the camera box. Where the two disagree, a person should look first. Returns
    ``{frame: [{"rect": [...], "note": why}]}``, what ``xcalib-review --check``
    marks on the cards; empty for a scene without an extrinsic.
    """
    calibration = scene.calibration()
    if calibration is None:
        return {}
    doubts = {}
    for frame, found in pairs.items():
        projected = {tuple(round(v, 1) for v in pair["rect"]): pair["lidar_id"]
                     for pair in frame_pairs(scene, frame, calibration)}
        notes = []
        for pair in found:
            other = projected.get(tuple(round(v, 1) for v in pair["rect"]))
            if other == pair["lidar_id"]:
                continue
            notes.append({"rect": pair["rect"], "note": (
                f"model: #{pair['lidar_id']}, but #{other} projects into this camera box" if other else
                f"model: #{pair['lidar_id']}, which does not project into this camera box")})
        if notes:
            doubts[frame] = notes
    return doubts


def main(argv=None):
    import argparse

    import torch  # noqa: F401 — load before the image libraries, as scene training does

    from ..engine.matcher import Matcher
    from .scene import Scene, matcher_target

    parser = argparse.ArgumentParser(
        description="Write the pairs the scene's model suggests for the undecided camera boxes")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", required=True, help="JSON for xcalib-review --proposals")
    parser.add_argument("--check-output", help="JSON for xcalib-review --check: the suggested pairs "
                                               "the saved extrinsic does not support")
    parser.add_argument("--threshold", type=float, default=0.5, help="lowest score of a suggested pair")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    scene = Scene(args.scene, args.camera)
    target = matcher_target(scene)
    if target is None:
        parser.error(f"{args.scene} has no model in calib/xcalib/{args.camera}_model")
    print("Loading the model ...", flush=True)
    name, weights, config = target
    matcher = Matcher.from_pretrained(name, weights=weights, config=config, device=args.device)
    frames = open_frames(scene)
    pairs = suggest_scene(matcher, scene, frames, threshold=args.threshold,
                          progress=lambda message: print(message, flush=True))
    Path(args.output).write_text(json.dumps(pairs), encoding="utf-8")
    report = {"open_frames": len(frames), "frames": len(pairs),
              "pairs": sum(len(found) for found in pairs.values())}
    if args.check_output:
        doubts = projection_doubts(scene, pairs)
        Path(args.check_output).write_text(json.dumps(doubts), encoding="utf-8")
        report["doubted"] = sum(len(notes) for notes in doubts.values())
    print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
