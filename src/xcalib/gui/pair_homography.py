"""The homography straight from confirmed pairs, without an extrinsic.

A saved pair says which LiDAR object a camera box shows. The delivered LiDAR
table gives that object's ``position_x, position_y``, and the centre of the
camera box is where the camera sees it. Enough such pairs, spread over the
image, determine the homography ``[u, v, 1] ~ H @ [position_x, position_y, 1]``
directly: no camera pose, no intrinsics and no model are involved.

Not every saved pair says where an object's centre is, and not every pair
should count the same, so the pairs are prepared before the fit:

- A camera box cut by the image edge is left out: its centre is the centre of
  the visible part, which can be hundreds of pixels from the object's.
- A camera box wider than ``WIDE_BOX`` of the image width is left out: it is an
  object close to the camera, and the centre of such a box lies tens of pixels
  beside the object's centre, for every such object to the same side. Left in,
  these pairs agree with each other and pull the fit away from the rest.
- A parked object is paired in every frame at the same pixel. Counted each
  time, a few parked objects outvote everything else and the homography fits
  their part of the image only. Each object therefore counts once per place:
  one of its pairs per image cell of ``CELL_PX`` pixels.
- The places are taken evenly over the image: at most ``PER_COLUMN`` from each
  of ``COLUMNS`` image columns. Traffic fills one side of an image with places
  and leaves few on the other; counted as they come, the crowded side alone
  carries a fit that is far off on the other side.

The fit itself is a RANSAC homography, so a few wrong pairs do not bend it,
followed by least squares over the places that lie close to it. It is made
in metres and then composed with the table's own map from ``position_x,
position_y`` (see :mod:`xcalib.gui.results`), which keeps longitude and
latitude well conditioned. ``H`` is scaled so that the third coordinate is
positive for the objects the pairs show, hence negative for objects behind the
camera.

What this cannot give is the full LiDAR-to-camera extrinsic; the plane the
homography holds for is the one the paired objects' centres lie on.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .results import (OBJECT_COLUMNS, _inputs, _scaled, apply_homography, draw_objects,
                      position_to_metres, scene_objects, spread_frames, table_frame, table_objects)
from .scene import Scene

MIN_PAIRS = 12        # saved pairs needed for a fit
MIN_PLACES = 6        # different places (object and image cell) they must cover
MIN_SPREAD = 0.25     # of the image width their box centres must span
INLIER_PX = 25.0      # a pair agrees with the fit when its box centre is this close
CLOSE_PX = 10.0       # the final least squares takes the places this close to the RANSAC fit
CELL_PX = 32          # an object counts once per image cell of this size
EDGE_PX = 2           # a camera box this close to the image edge is cut by it
WIDE_BOX = 0.08       # a camera box wider than this share of the image width is an object close to the camera
COLUMNS = 6           # the places are taken evenly from this many image columns,
PER_COLUMN = 25       # at most this many from each


def paired_points(scene: Scene, objects, size=None):
    """Position in metres, table position and pixel for every saved pair whose LiDAR
    object the table (or the scene) lists. Returns a list of dicts; with ``size``
    (width, height of the image) each says whether its camera box is ``cut`` by
    the image edge and whether it is ``wide`` (more than ``WIDE_BOX`` of the
    image width)."""
    by_frame = {}
    for obj in objects:
        if obj["id"]:
            by_frame.setdefault(obj["frame"], {})[obj["id"]] = obj
    found = []
    for frame, listed in by_frame.items():
        for rect in scene.load_rects(frame):
            obj = listed.get(str(rect.get("obj_id") or "").strip())
            if obj is None or rect.get("no_match"):
                continue
            box = rect["rect"]
            cut = bool(size) and (box["x1"] <= EDGE_PX or box["y1"] <= EDGE_PX
                                  or box["x2"] >= size[0] - 1 - EDGE_PX or box["y2"] >= size[1] - 1 - EDGE_PX)
            wide = bool(size) and box["x2"] - box["x1"] > WIDE_BOX * size[0]
            found.append({"frame": frame, "id": obj["id"], "metres": obj["centre"][:2],
                          "position": obj["position"],
                          "pixel": [(box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2],
                          "rect": [box["x1"], box["y1"], box["x2"], box["y2"]], "cut": cut, "wide": wide})
    return found


def usable(pair):
    """Whether the centre of the pair's camera box stands for the object's centre:
    the box is neither cut by the image edge nor wide."""
    return not pair.get("cut") and not pair.get("wide")


def places(pairs, cell=CELL_PX, width=None):
    """One pair per object and image cell: of the object's pairs whose box centre
    lies in the cell, the one nearest to their middle. (An average would not do:
    a homography does not take the mean position to the mean pixel.) Pairs marked
    ``cut`` or ``wide`` are left out. With ``width`` (of the image) the places are
    then taken evenly over the image, see :func:`spread`. Returns ``(metres, pixels)``."""
    groups = {}
    for pair in pairs:
        if not usable(pair):
            continue
        key = (pair["id"], int(pair["pixel"][0] // cell), int(pair["pixel"][1] // cell))
        groups.setdefault(key, []).append(pair)
    chosen = []
    for group in groups.values():
        centres = np.array([p["pixel"] for p in group])
        chosen.append(group[int(np.linalg.norm(centres - centres.mean(axis=0), axis=1).argmin())])
    if width:
        chosen = spread(chosen, width)
    return (np.array([p["metres"] for p in chosen], dtype=float).reshape(-1, 2),
            np.array([p["pixel"] for p in chosen], dtype=float).reshape(-1, 2))


def spread(chosen, width, columns=COLUMNS, most=PER_COLUMN):
    """At most ``most`` of the places from each of ``columns`` image columns, taken
    at even steps from the top of the column to its bottom. A column with fewer
    places keeps all of them. The result does not depend on the order of ``chosen``."""
    by_column = {}
    for place in chosen:
        column = min(columns - 1, max(0, int(place["pixel"][0] * columns // width)))
        by_column.setdefault(column, []).append(place)
    kept = []
    for column in sorted(by_column):
        inside = sorted(by_column[column], key=lambda p: (p["pixel"][1], p["pixel"][0], str(p["id"])))
        if len(inside) > most:
            inside = [inside[index] for index in np.linspace(0, len(inside) - 1, most).round().astype(int)]
        kept += inside
    return kept


def fit_homography(metres, pixels, width, *, least=MIN_PAIRS):
    """RANSAC homography from ``(x, y)`` in metres to pixels, refined by least
    squares over the points within ``CLOSE_PX`` of it when enough of them are.

    Returns ``(H, inliers)`` with ``H`` scaled so that its last entry is 1 or -1
    and the third coordinate is positive on the inliers. ``ValueError`` says why
    the points do not give one; ``least`` is how many are needed.
    """
    import cv2

    metres = np.asarray(metres, dtype=float).reshape(-1, 2)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(metres) < least:
        raise ValueError(f"a homography needs at least {least} saved pairs, the scene has {len(metres)}")
    if np.ptp(pixels[:, 0]) < MIN_SPREAD * width:
        raise ValueError("the saved pairs lie in one part of the image; pair objects on the left, "
                         "in the middle and on the right as well")
    origin = metres.mean(axis=0)  # site coordinates can be large: fit about their mean
    homography, mask = cv2.findHomography(metres - origin, pixels, cv2.RANSAC, INLIER_PX,
                                          maxIters=5000, confidence=0.999)
    if homography is None:
        raise ValueError("the saved pairs do not agree on one homography")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < least:
        raise ValueError(f"only {int(inliers.sum())} of {len(metres)} saved pairs agree with one "
                         "homography; check the pairs")
    error = np.linalg.norm(apply_homography(homography, metres - origin) - pixels, axis=1)
    close = error <= CLOSE_PX
    if close.sum() >= max(least, inliers.sum() // 2):
        refined, _ = cv2.findHomography((metres - origin)[close], pixels[close], 0)
        if refined is not None:
            homography = refined
            inliers = np.linalg.norm(apply_homography(homography, metres - origin) - pixels,
                                     axis=1) <= INLIER_PX
    shift = np.array([[1.0, 0.0, -origin[0]], [0.0, 1.0, -origin[1]], [0.0, 0.0, 1.0]])
    homography = homography @ shift
    third = np.column_stack([metres[inliers], np.ones(int(inliers.sum()))]) @ homography[2]
    if np.median(third) < 0:
        homography = -homography
    return _scaled(homography), inliers


def export_pair_homography(scene: Scene, output_dir, *, images=6, more=()):
    """Fit the homography to the scene's saved pairs and write it with marked images.

    ``more`` are further scenes of the same camera and the same delivered table
    (the refine scene beside the base scene): their saved pairs take part in the
    one fit, and their frames in the marked images.

    Files in ``output_dir``: ``homography.json``; ``projection_inputs.py``, the
    values as Python assignments for a projection script that reads the
    delivered images and table; ``<frame>_rectified_projected.png`` for
    ``images`` frames with every object of the frame marked through the
    homography. A scene without a delivered table also gets ``lidar_objects.csv``.
    Returns the report; ``ValueError`` says why nothing was written.
    """
    import cv2

    from .auto_calibrate import scene_intrinsics

    scenes = [scene, *more]
    first = next((scene.image_path(frame) for frame in scene.frames() if scene.image_path(frame)), None)
    sample = cv2.imread(first) if first else None
    if sample is None:
        raise ValueError("the scene has no readable camera image")
    height, width = sample.shape[:2]
    to_cloud, frame_recorded = table_frame(scene)
    objects, pairs, used, columns, table, frames, image_of = [], [], [], None, None, [], {}
    for index, part in enumerate(scenes):
        part_to_cloud, _ = table_frame(part)
        delivered = table_objects(part)
        if delivered and delivered[0]:
            found, form = delivered
            if index == 0:
                table = Path(part.root) / "vendor" / "lidar_perception.csv"
        else:
            found, form = scene_objects(part, part_to_cloud), "metres"
        if columns not in (None, form):
            raise ValueError(f"{Path(part.root).name} gives its LiDAR positions as {form}, "
                             f"{Path(scene.root).name} as {columns}: they are not one delivery")
        columns = form
        part_pairs = paired_points(part, found, (width, height))
        used.append({"scene": str(part.root), "pairs": len(part_pairs),
                     "frames": len({p["frame"] for p in part_pairs})})
        objects += found
        pairs += part_pairs
        for frame in part.frames():
            frames.append(frame)
            image_of[frame] = part.image_path(frame)
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"a homography needs at least {MIN_PAIRS} saved pairs, the scene has {len(pairs)}")
    cut = sum(1 for p in pairs if p["cut"])
    wide = sum(1 for p in pairs if p["wide"] and not p["cut"])
    place_metres, place_pixels = places(pairs, width=width)
    if len(place_metres) < MIN_PLACES:
        raise ValueError(f"the {len(pairs)} saved pairs show their objects in only {len(place_metres)} "
                         f"places of the image ({cut} camera boxes are cut by the image edge and {wide} are "
                         f"wider than {WIDE_BOX:.0%} of the image, and do not count); pair moving objects "
                         "in more frames, at different places")
    in_metres, _ = fit_homography(place_metres, place_pixels, width, least=MIN_PLACES)
    if columns == "metres":
        homography = in_metres
    else:
        homography = _scaled(in_metres @ position_to_metres(
            [obj["position"] for obj in objects], [obj["centre"][:2] for obj in objects]))
    dots = apply_homography(homography, [p["position"] for p in pairs])
    centres = np.array([p["pixel"] for p in pairs])
    error = np.linalg.norm(dots - centres, axis=1)
    # The checks count the pairs whose box centre stands for the object's centre. The centre
    # of a box cut by the image edge, or of a wide box of an object close to the camera, is
    # not where the object's centre lies, so a right homography misses it.
    whole = np.array([usable(p) for p in pairs])
    inliers = (error <= INLIER_PX) & whole
    # per scene: how many of its usable pairs agree with the fit (a scene of another camera
    # position does not); the pairs are listed scene by scene, in the order of ``used``
    start = 0
    for entry in used:
        part = slice(start, start + entry["pairs"])
        entry.update(usable=int(whole[part].sum()), agree=int(inliers[part].sum()))
        start += entry["pairs"]
    # support in the left, middle and right third of the image, counted as the fit counts:
    # by places, so that a parked object is one voice there too
    near = np.linalg.norm(apply_homography(in_metres, place_metres) - place_pixels, axis=1) <= INLIER_PX
    thirds = [{"pairs": int(part.sum()), "agree": int((part & near).sum())}
              for part in (place_pixels[:, 0] < width / 3,
                           (place_pixels[:, 0] >= width / 3) & (place_pixels[:, 0] < 2 * width / 3),
                           place_pixels[:, 0] >= 2 * width / 3)]
    inside = sum(1 for (x, y), p in zip(dots, pairs)
                 if usable(p) and p["rect"][0] <= x <= p["rect"][2] and p["rect"][1] <= y <= p["rect"][3])
    agreeing = error[inliers] if inliers.any() else error
    # where the homography puts every listed object: in front (third coordinate > 0) and in the image
    lifted = np.column_stack([[obj["position"] for obj in objects], np.ones(len(objects))]) @ homography.T
    for obj, row in zip(objects, lifted):
        front = bool(row[2] > 0)
        x, y = (row[0] / row[2], row[1] / row[2]) if row[2] else (np.inf, np.inf)
        obj.update(front=front, seen=front and bool(0 <= x < width and 0 <= y < height))
    seen = [obj for obj in objects if obj["seen"]]
    behind = [obj for obj, row in zip(objects, lifted) if row[2] < 0
              and 0 <= row[0] / row[2] < width and 0 <= row[1] / row[2] < height]

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        camera_matrix = scene_intrinsics(scene).K
    except ValueError:
        camera_matrix = None
    report = {
        "scene": str(scene.root), "camera": scene.camera,
        "lidar_table": str(table) if table else None,
        "position_columns": columns,
        "table_frame_recorded": frame_recorded,
        "position_to_image": homography.tolist(),
        "image_to_position": np.linalg.inv(homography).tolist(),
        "metres_to_image": in_metres.tolist(),
        "fitted_to": {"what": "the centres of the camera boxes of the saved pairs",
                      "pairs": len(pairs), "frames": len({p["frame"] for p in pairs}),
                      "objects": len({p["id"] for p in pairs}), "scenes": used,
                      "cut_by_image_edge": cut, "wide_boxes": wide, "wide_box_share": WIDE_BOX,
                      "places": int(len(place_metres)), "places_per_column_at_most": PER_COLUMN,
                      "thirds": thirds, "usable_boxes": int(whole.sum()),
                      "inliers": int(inliers.sum()), "inlier_px": INLIER_PX,
                      "median_error_px": float(np.median(agreeing)),
                      "p95_error_px": float(np.percentile(agreeing, 95))},
        "dots_in_paired_camera_boxes": {"inside": int(inside), "pairs": int(whole.sum())},
        "behind_camera_but_in_image": {"objects": len(behind),
                                       "frames": len({obj["frame"] for obj in behind})},
        "camera_matrix": camera_matrix.tolist() if camera_matrix is not None else None,
        "distortion": [0.0] * 5, "image_size": [int(width), int(height)],
        "objects_in_view": len(seen),
        "projection_images": [],
    }
    if table is None:
        table = output / "lidar_objects.csv"
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(OBJECT_COLUMNS)
            for obj in seen:
                writer.writerow([obj["frame"], obj["id"], obj["type"],
                                 *(round(float(value), 6) for value in (*obj["centre"], *obj["size"]))])
        report["objects_file"] = str(table)
    by_frame, in_view = {}, {}
    for obj in seen:
        by_frame.setdefault(obj["frame"], []).append(obj)
        in_view[obj["frame"]] = in_view.get(obj["frame"], 0) + 1
    chosen = spread_frames({frame: in_view.get(frame, 0) for frame in frames
                            if image_of[frame]}, images)
    for frame in chosen:
        image = cv2.imread(image_of[frame])
        draw_objects(image, by_frame[frame], homography)
        target = output / f"{frame}_rectified_projected.png"
        cv2.imwrite(str(target), image)
        report["projection_images"].append(str(target))
    # the example of the inputs file is a frame of the first scene: its table lists that frame
    own = [frame for frame in scene.frames() if image_of.get(frame) and in_view.get(frame)]
    if own and camera_matrix is not None:
        example = max([frame for frame in chosen if frame in own] or own, key=lambda frame: in_view[frame])
        inputs_file = output / "projection_inputs.py"
        inputs_file.write_text(_inputs(example, Path(image_of[example]).parent, table, output,
                                       camera_matrix, homography, columns), encoding="utf-8")
        report["inputs_file"] = str(inputs_file)
    (output / "homography.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["homography_file"] = str(output / "homography.json")
    return report
