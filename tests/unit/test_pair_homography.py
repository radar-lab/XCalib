"""Confirmed pairs alone give the homography: no extrinsic, no model."""

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from xcalib.gui.customer import Stop, Workflow
from xcalib.gui.pair_homography import MIN_PAIRS, export_pair_homography, fit_homography
from xcalib.gui.results import apply_homography
from xcalib.gui.scene import Scene

ROOT = Path(__file__).resolve().parents[2]
CAMERA = "front"
TRUE = np.array([[9.0, -30.0, 960.0], [-4.0, 1.5, 700.0], [0.004, 0.0015, 1.0]])  # metres -> pixels


def _scene(root, frames=8, wrong=0, degrees=False, start=0, truth=TRUE):
    """Four objects per frame on a plane, each with a camera box round its true pixel;
    the first `wrong` pairs name another object. One object lies behind the camera.
    The frames are numbered from `start`; `truth` is the camera's homography."""
    for folder in ("lidar", "label", f"camera/{CAMERA}", f"label_fusion/camera/{CAMERA}", "vendor",
                   "calib/intrinsic"):
        (root / folder).mkdir(parents=True)
    rows, made = [], 0
    for index in range(frames):
        frame = f"{start + index:03d}"
        spots = {"1": (20 + index, -12 + 3 * index), "2": (35, 10 - index), "3": (50 + index, 2),
                 "4": (28, -6 + index), "9": (-400.0, 0.0)}
        rects = []
        for ident, (x, y) in spots.items():
            rows.append([frame, ident, "Car", x, y])
            if ident == "9":
                continue
            u, v = apply_homography(truth, [[x, y]])[0]
            paired = str(int(ident) % 4 + 1) if made < wrong else ident
            made += 1
            rects.append({"obj_type": "Car", "obj_id": paired,
                          "rect": {"x1": u - 30, "y1": v - 20, "x2": u + 30, "y2": v + 20}})
        np.save(root / "lidar" / f"{frame}.npy", np.zeros((1, 3), np.float32))
        (root / "label" / f"{frame}.json").write_text("[]")
        (root / "label_fusion" / "camera" / CAMERA / f"{frame}.json").write_text(json.dumps({"objs": rects}))
        cv2.imwrite(str(root / "camera" / CAMERA / f"{frame}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
    with (root / "vendor" / "lidar_perception.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        if degrees:   # longitude and latitude in position_x, position_y; the metres in x, y
            writer.writerow(["timestamp", "id", "classification", "position_x", "position_y", "x", "y", "z"])
            for frame, ident, kind, x, y in rows:
                writer.writerow([frame, ident, kind, repr(-85.3 - y / 91137.0), repr(35.0 + x / 111319.0), x, y, 0])
        else:
            writer.writerow(["timestamp", "id", "classification", "position_x", "position_y", "position_z"])
            for frame, ident, kind, x, y in rows:
                writer.writerow([frame, ident, kind, x, y, 0])
    (root / "calib" / "intrinsic" / f"{CAMERA}.json").write_text(
        json.dumps({"intrinsic": [800, 0, 960, 0, 800, 540, 0, 0, 1]}))
    return Scene(str(root), CAMERA)


@pytest.mark.parametrize("degrees", [False, True])
def test_the_pairs_give_the_homography_of_the_delivered_table(tmp_path, degrees):
    scene = _scene(tmp_path / "scene", wrong=3, degrees=degrees)
    assert scene.calibration() is None

    report = export_pair_homography(scene, tmp_path / "out", images=2)

    fit = report["fitted_to"]
    assert fit["pairs"] == 32 and fit["inliers"] == 29 and fit["median_error_px"] < 0.5
    assert np.allclose(np.array(report["metres_to_image"]), TRUE, atol=1e-4)
    homography = np.array(report["position_to_image"])
    rows = list(csv.DictReader((tmp_path / "scene" / "vendor" / "lidar_perception.csv").open()))
    near = next(r for r in rows if r["id"] == "2")
    pixel = apply_homography(homography, [[float(near["position_x"]), float(near["position_y"])]])[0]
    assert np.allclose(pixel, apply_homography(TRUE, [[35, 10]])[0], atol=0.5)
    behind = next(r for r in rows if r["id"] == "9")
    assert (homography @ [float(behind["position_x"]), float(behind["position_y"]), 1.0])[2] < 0
    assert report["position_columns"] == ("longitude, latitude" if degrees else "metres")
    assert len(report["projection_images"]) == 2 and Path(report["inputs_file"]).is_file()
    assert "extrinsic_point_cloud_to_camera" not in report

    values = {}
    exec(Path(report["inputs_file"]).read_text(encoding="utf-8"), values)  # plain assignments
    assert np.array_equal(values["H_ground_to_rectified"], homography)    # every digit is written
    assert np.array_equal(values["K"], [[800, 0, 960], [0, 800, 540], [0, 0, 1]]) and not values["dist"].any()
    assert values["csv_path"] == tmp_path / "scene" / "vendor" / "lidar_perception.csv"
    assert values["TARGET_TIMESTAMP"] in {row["timestamp"] for row in rows}
    for row in rows:   # the written matrix, applied to the table's columns as they are
        mapped = values["H_ground_to_rectified"] @ [float(row["position_x"]), float(row["position_y"]), 1.0]
        if row["id"] == "9":
            assert mapped[2] < 0   # behind the camera
            continue
        metres = [float(row["x"]), float(row["y"])] if degrees else [float(row["position_x"]),
                                                                      float(row["position_y"])]
        assert np.allclose(mapped[:2] / mapped[2], apply_homography(TRUE, [metres])[0], atol=1e-3)


def test_too_few_or_bunched_pairs_are_refused():
    with pytest.raises(ValueError, match=f"at least {MIN_PAIRS} saved pairs"):
        fit_homography(np.random.rand(5, 2), np.random.rand(5, 2), 1920)
    metres = np.random.default_rng(0).uniform(0, 50, (40, 2))
    with pytest.raises(ValueError, match="one part of the image"):
        fit_homography(metres, 900 + metres, 1920)


def test_the_pairs_of_two_scenes_give_one_homography(tmp_path, capsys):
    # as the import writes the two parts of one delivery
    base = _scene(tmp_path / "site" / "site_base", frames=8)
    refine = _scene(tmp_path / "site" / "refine" / "site_refine", frames=4, start=8, wrong=1)

    report = export_pair_homography(base, tmp_path / "out", more=[refine], images=4)

    fit = report["fitted_to"]
    assert fit["pairs"] == 48 and fit["inliers"] == 47 and fit["frames"] == 12
    assert [(Path(used["scene"]).name, used["pairs"]) for used in fit["scenes"]] == [
        ("site_base", 32), ("site_refine", 16)]
    assert [(used["usable"], used["agree"]) for used in fit["scenes"]] == [(32, 32), (16, 15)]
    assert np.allclose(np.array(report["metres_to_image"]), TRUE, atol=1e-4)
    shown = {Path(path).name[:3] for path in report["projection_images"]}
    assert shown & {f"{n:03d}" for n in range(8)} and shown & {f"{n:03d}" for n in range(8, 12)}
    assert "TARGET_TIMESTAMP" in Path(report["inputs_file"]).read_text()
    example = re.search(r'TARGET_TIMESTAMP\s*=\s*"?(\d+)', Path(report["inputs_file"]).read_text()).group(1)
    assert int(example) < 8      # a frame the first scene's table lists

    flow = Workflow()
    flow.repo = tmp_path
    flow.scenes.update(base=(Path(base.root), CAMERA), refine=(Path(refine.root), CAMERA))
    flow.homography()
    out = capsys.readouterr()
    assert "Pairs used: 16 of the refine scene, 32 of the base scene." in out.out
    assert ("Of the pairs the fit can use, 15 of 16 of the refine scene, 32 of 32 of the base scene agree "
            "with the homography within 25 px.") in out.out
    assert "Fitted to 48 saved pairs (4 objects in 12 frames)." in out.out
    assert "Check, on the 48 pairs: 47 agree with the homography within 25 px" in out.out
    assert "do not agree" not in out.err
    flow.homography("refine")    # one scene when it is named
    assert "Fitted to 16 saved pairs" in capsys.readouterr().out

    # Refine data of another recording of the same camera: its pairs and the base pairs give one fit.
    later = _scene(tmp_path / "day2" / "refine" / "day2_refine", frames=4, start=20)
    flow.scenes["refine"] = (Path(later.root), CAMERA)
    flow.homography()
    out = capsys.readouterr()
    assert "Pairs used: 16 of the refine scene, 32 of the base scene." in out.out
    assert "Fitted to 48 saved pairs" in out.out and "do not agree" not in out.err

    # The camera moved between the recordings: the refine pairs disagree, and the step says so.
    moved_truth = TRUE.copy()
    moved_truth[1] += 120 * TRUE[2]              # every pixel 120 px lower
    moved = _scene(tmp_path / "day3" / "refine" / "day3_refine", frames=4, start=30, truth=moved_truth)
    flow.scenes["refine"] = (Path(moved.root), CAMERA)
    flow.homography()
    assert "The pairs of the refine scene do not agree with this homography" in capsys.readouterr().err

    del flow.scenes["base"]      # those who only refine have no base scene
    flow.scenes["refine"] = (Path(later.root), CAMERA)
    flow.homography()
    out = capsys.readouterr().out
    assert "Fitted to 16 saved pairs" in out and "Pairs used" not in out
    few = _scene(tmp_path / "few" / "refine" / "few_refine", frames=2)
    flow.scenes["refine"] = (Path(few.root), CAMERA)
    with pytest.raises(Stop, match="The pairs of the refine scene alone are not enough for it"):
        flow.homography()
    flow.scenes.clear()
    with pytest.raises(Stop, match="No scene is chosen"):
        flow.homography()
    flow.scenes["base"] = (Path(base.root), CAMERA)   # the base part alone: no refine step is needed
    flow.homography()
    assert "Fitted to 32 saved pairs" in capsys.readouterr().out


def test_the_homography_notebook_needs_no_extrinsic_and_trains_for_the_epochs_it_names(monkeypatch):
    path = ROOT / "demo" / "xcalib_homography_en.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    assert not any(c.get("outputs") for c in cells if c["cell_type"] == "code")
    assert code[1:] == ['flow.import_raw("base")', 'flow.view_sustech("base")', 'flow.import_pairs("base")',
                        'flow.pair_for_homography("base")', 'flow.view_pairs("base")', 'flow.train("base")',
                        'flow.import_raw("refine")', 'flow.view_sustech("refine")', 'flow.choose_model("base")',
                        'flow.import_pairs("refine")', 'flow.pair("refine")', 'flow.train("refine")',
                        'flow.pair_remaining("refine")', "flow.homography()"]
    assert all(re.fullmatch(r'flow\.\w+\(("(base|refine)")?\)', step) for step in code[1:])
    assert not any(word in "".join(code).lower() for word in ("calibrate", "extrinsic"))
    monkeypatch.chdir(ROOT / "demo")
    namespace = {}
    exec(compile(code[0], str(path), "exec"), namespace)
    flow = namespace["flow"]
    assert flow.repo == ROOT   # found from demo/, where pixi starts Jupyter
    assert flow.epochs == {"base": 30, "refine": 10}
    assert all(callable(getattr(flow, step[5:step.index("(")])) for step in code[1:])


def test_the_notebook_step_reports_the_fit_and_explains_too_few_pairs(tmp_path, capsys):
    flow = Workflow()
    flow.repo = tmp_path
    _scene(tmp_path / "few", frames=2)
    flow.scenes["base"] = (tmp_path / "few", CAMERA)
    with pytest.raises(Stop, match="needs at least 12 saved pairs, the scene has 8"):
        flow.homography("base")
    _scene(tmp_path / "enough")
    flow.scenes["base"] = (tmp_path / "enough", CAMERA)
    flow.homography("base")
    out = capsys.readouterr().out
    assert "Fitted to 32 saved pairs (4 objects in 8 frames)." in out
    assert "Check, on the 32 pairs: 32 agree with the homography within 25 px" in out
    assert "and 32 (100.0%) have the LiDAR object's dot inside the paired camera box." in out
    assert len(list((tmp_path / "runs").glob("homography-*/homography.json"))) == 1


def _write(root, frames):
    """A scene from ``{frame: [(id, x, y, rect or None)]}``: a table row per object, and a
    camera box paired with it where a rect (x1, y1, x2, y2) is given."""
    for folder in ("lidar", "label", f"camera/{CAMERA}", f"label_fusion/camera/{CAMERA}", "vendor"):
        (root / folder).mkdir(parents=True)
    with (root / "vendor" / "lidar_perception.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "id", "classification", "position_x", "position_y", "position_z"])
        for frame, found in frames.items():
            rects = []
            for ident, x, y, rect in found:
                writer.writerow([frame, ident, "Car", x, y, 0])
                if rect is not None:
                    rects.append({"obj_type": "Car", "obj_id": ident,
                                  "rect": dict(zip(("x1", "y1", "x2", "y2"), map(float, rect)))})
            np.save(root / "lidar" / f"{frame}.npy", np.zeros((1, 3), np.float32))
            (root / "label" / f"{frame}.json").write_text("[]")
            (root / "label_fusion" / "camera" / CAMERA / f"{frame}.json").write_text(json.dumps({"objs": rects}))
            cv2.imwrite(str(root / "camera" / CAMERA / f"{frame}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
    return Scene(str(root), CAMERA)


def _box(x, y, shift=0.0, half=30):
    u, v = apply_homography(TRUE, [[x, y]])[0]
    return (u + shift - half, v - 20, u + shift + half, v + 20)


def test_parked_objects_and_boxes_cut_by_the_image_edge_do_not_bend_the_fit(tmp_path, capsys):
    # TRUE puts x = 20..60 m, y = -25..25 m into the image: y = 25 m lies at its left edge.
    frames = {}
    for index in range(60):
        moving = [(str(10 + lane), 22.0 + 0.6 * index, -20.0 + 9 * lane) for lane in range(5)]
        found = [(ident, x, y, _box(x, y)) for ident, x, y in moving]
        # two parked cars on the right, whose camera boxes sit 40 px beside the LiDAR centre
        found += [("1", 30.0, -24.0, _box(30.0, -24.0, shift=40.0)), ("2", 45.0, -22.0, _box(45.0, -22.0, shift=40.0))]
        # a truck leaving through the left edge: only its right part is in the image
        u, v = apply_homography(TRUE, [[25.0 + 0.2 * index, 30.0]])[0]
        found.append(("3", 25.0 + 0.2 * index, 30.0, (0.0, v - 60, max(40.0, u + 200), v + 60)))
        frames[f"{index:03d}"] = found
    scene = _write(tmp_path / "scene", frames)

    report = export_pair_homography(scene, tmp_path / "out", images=2)

    fit = report["fitted_to"]
    assert fit["pairs"] == 60 * 8 and fit["cut_by_image_edge"] == 60
    assert fit["places"] < 200                       # each parked car is one place, not sixty
    homography = np.array(report["metres_to_image"])
    probe = [[25.0, 20.0], [40.0, 0.0], [55.0, -20.0], [30.0, -24.0]]   # left, middle, right, a parked car
    assert np.abs(apply_homography(homography, probe) - apply_homography(TRUE, probe)).max() < 2.0
    assert all(part["agree"] >= 0.8 * part["pairs"] for part in fit["thirds"] if part["pairs"])

    flow = Workflow()
    flow.repo = tmp_path
    flow.scenes["base"] = (tmp_path / "scene", CAMERA)
    flow.homography("base")
    out = capsys.readouterr()
    assert "60 of the pairs took no part in the fit: their camera box is cut by the image edge" in out.out
    # the checks leave the cut boxes out too: a right homography puts their dot outside the box
    assert fit["usable_boxes"] == 60 * 7 and report["dots_in_paired_camera_boxes"]["pairs"] == 60 * 7
    assert "Check, on the 420 pairs whose camera box is whole: " in out.out

    # no pair on the left of the image: the fit is made, and the step says where it has no support
    one_side = {frame: [row for row in found if row[0] in ("10", "11", "12", "1", "2")]
                for frame, found in frames.items()}
    _write(tmp_path / "one_side", one_side)
    flow.scenes["base"] = (tmp_path / "one_side", CAMERA)
    flow.homography("base")
    err = capsys.readouterr().err
    assert "The homography is not supported on the left of the image: 0 of 0 places agree on the left" in err
    assert "run step 1" not in err

    # The same pairs as the refine scene of a session without a base scene: base data of the same
    # camera position would complete them, and step 9 says how to add it.
    flow.scenes.clear()
    flow.scenes["refine"] = (tmp_path / "one_side", CAMERA)
    flow.homography()
    err = capsys.readouterr().err
    assert "not supported on the left of the image" in err and "run step 1 with that data" in err
    flow.scenes["base"] = (tmp_path / "scene", CAMERA)   # a base scene takes part: no such advice
    flow.homography()
    assert "run step 1" not in capsys.readouterr().err


def test_wide_boxes_stay_out_and_the_places_are_taken_evenly(tmp_path, capsys):
    from xcalib.gui.pair_homography import PER_COLUMN, WIDE_BOX, places, spread

    # A vehicle close to the camera: its box is 320 px wide and its centre 100 px beside the object's.
    frames = {}
    for index in range(40):
        found = [(str(10 + lane), 22.0 + 0.9 * index, -20.0 + 9 * lane,
                  _box(22.0 + 0.9 * index, -20.0 + 9 * lane)) for lane in range(5)]
        u, v = apply_homography(TRUE, [[24.0 + 0.5 * index, 12.0]])[0]
        found.append(("7", 24.0 + 0.5 * index, 12.0, (u - 60, v - 80, u + 260, v + 80)))
        frames[f"{index:03d}"] = found
    scene = _write(tmp_path / "scene", frames)

    report = export_pair_homography(scene, tmp_path / "out", images=1)

    fit = report["fitted_to"]
    assert fit["pairs"] == 240 and fit["wide_boxes"] == 40 and fit["cut_by_image_edge"] == 0
    assert fit["wide_box_share"] == WIDE_BOX and fit["places_per_column_at_most"] == PER_COLUMN
    # the checks count the pairs whose box centre stands for the object: the wide ones are not among them
    assert fit["usable_boxes"] == 200 == report["dots_in_paired_camera_boxes"]["pairs"] and fit["inliers"] == 200
    probe = [[25.0, 20.0], [40.0, 0.0], [55.0, -20.0]]
    assert np.abs(apply_homography(np.array(report["metres_to_image"]), probe)
                  - apply_homography(TRUE, probe)).max() < 2.0

    flow = Workflow()
    flow.repo = tmp_path
    flow.scenes["base"] = (tmp_path / "scene", CAMERA)
    flow.homography("base")
    out = capsys.readouterr().out
    assert "40 of the pairs took no part in the fit: their camera box is wider than 8% of the image" in out
    assert "the places are taken evenly over the image (at most 25 from each of its six columns)" in out
    assert "Check, on the 200 pairs whose camera box is whole and not wide: 200 agree" in out

    # A wide box is no place; a crowded image column gives PER_COLUMN places from its top to its
    # bottom, a sparse one all it has, and the order of the pairs makes no difference.
    wide = [{"id": "w", "metres": [1.0, 1.0], "pixel": [500.0, 500.0], "cut": False, "wide": True}]
    crowded = [{"id": str(i), "metres": [float(i), 0.0], "pixel": [1900.0, 10.0 * i]} for i in range(100)]
    sparse = [{"id": f"s{i}", "metres": [0.0, float(i)], "pixel": [20.0, 300.0 + 50 * i]} for i in range(4)]
    assert len(places(wide + crowded + sparse)[0]) == 104            # without the image width: every place
    metres, pixels = places(wide + crowded + sparse, width=1920)
    assert len(metres) == PER_COLUMN + 4 and [500.0, 500.0] not in pixels.tolist()
    column = sorted(y for x, y in pixels.tolist() if x > 1000)
    assert column[0] == 0.0 and column[-1] == 990.0
    kept = spread(crowded + sparse, 1920)
    assert [p["id"] for p in spread(list(reversed(crowded + sparse)), 1920)] == [p["id"] for p in kept]


def test_an_object_counts_once_per_place(tmp_path):
    from xcalib.gui.pair_homography import places

    parked = [{"id": "1", "metres": [30.0, 5.0], "pixel": [900.0 + jitter, 500.0], "cut": False}
              for jitter in (0.0, 1.0, 2.0, 1.5)]
    moving = [{"id": "2", "metres": [20.0 + step, 0.0], "pixel": [100.0 + 40 * step, 520.0], "cut": False}
              for step in range(6)]
    cut = [{"id": "3", "metres": [10.0, 9.0], "pixel": [20.0, 700.0], "cut": True}]
    metres, pixels = places(parked + moving + cut)
    assert len(metres) == 1 + 6 and [900.0 + 1.0, 500.0] in pixels.tolist()   # the pair in the middle of its place
    assert [20.0, 700.0] not in pixels.tolist()
