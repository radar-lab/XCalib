"""Use persisted human-reviewed scene pairs for calibration and training."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .rename_retry import rename_when_free
from .scene import Scene, load_points_xyz
from ..oneshot.calibration import CalibrationSession, bbox2d_centers, bbox3d_centers


def verified_frame(scene: Scene, frame: str):
    """Return one-to-one persisted positive pairs; reject ambiguous identities."""
    rects, boxes = scene.load_rects(frame), scene.load_boxes(frame)
    by_id = {}
    for box in boxes:
        lid = str(box.get("obj_id") or "").strip()
        if lid:
            by_id.setdefault(lid, []).append(box)
    paired_rects, paired_boxes, used = [], [], set()
    for rect in rects:
        lid = str(rect.get("obj_id") or "").strip()
        if not lid or rect.get("no_match"):
            continue
        if lid not in by_id:
            raise ValueError(f"frame {frame}: camera id {lid} has no LiDAR box")
        if len(by_id[lid]) != 1:
            raise ValueError(f"frame {frame}: paired LiDAR id {lid} is ambiguous")
        if lid in used:
            raise ValueError(f"frame {frame}: multiple camera boxes use LiDAR id {lid}")
        used.add(lid)
        paired_rects.append(rect)
        paired_boxes.append(by_id[lid][0])
    return paired_rects, paired_boxes


def training_frame(scene: Scene, frame: str):
    """A frame as the model meets it in the window: ``(rects, boxes, match)``.

    ``rects`` are the paired camera boxes followed by every other camera box of
    the frame: those saved as "no LiDAR box" and those nobody decided, which count
    as having no LiDAR box too; ``boxes`` are the paired LiDAR boxes followed by
    the frame's other LiDAR boxes with an id of their own; ``match[i, j]`` is 1 for
    a saved pair. A row without a 1 trains the model to answer "none".
    """
    paired_rects, paired_boxes = verified_frame(scene, frame)
    rejected = [rect for rect in scene.load_rects(frame)
                if rect.get("no_match") or not str(rect.get("obj_id") or "").strip()]
    counts = {}
    for box in scene.load_boxes(frame):
        lid = str(box.get("obj_id") or "").strip()
        counts[lid] = counts.get(lid, 0) + 1
    taken = {str(box.get("obj_id") or "").strip() for box in paired_boxes}
    others = [box for box in scene.load_boxes(frame)
              if (lid := str(box.get("obj_id") or "").strip()) and counts[lid] == 1 and lid not in taken]
    rects, boxes = paired_rects + rejected, paired_boxes + others
    match = np.zeros((len(rects), len(boxes)), dtype=np.uint8)
    for index in range(len(paired_rects)):
        match[index, index] = 1
    return rects, boxes, match


# A re-solve may replace the saved pose of a recalibration only while it still
# describes the same camera; further away it is another solution of the nearly
# planar scene.
RESOLVE_TURN_DEG = 15.0
RESOLVE_MOVE_M = 5.0


def calibrate_scene(scene: Scene, intrinsics, frames=None, previous=None):
    """Solve or refine from persisted positive pairs using existing calibration code.

    With ``previous`` the saved pose is nudged. A nudge works on the pairs the
    saved pose already explains, so it cannot follow a camera that was turned
    by a degree or two since (a pose taken over from another recording, for
    one). A fresh solve then replaces the nudge when it fits all pairs clearly
    better and stays within ``RESOLVE_TURN_DEG`` and ``RESOLVE_MOVE_M`` of the
    saved pose; ``resolved`` on the result says which of the two it is.
    """
    session = CalibrationSession(max_pairs=1000000)
    pixels, points = [], []
    for frame in frames if frames is not None else scene.frames():
        rects, boxes = verified_frame(scene, frame)
        if rects:
            pixels.append(bbox2d_centers(Scene.rects_xyxy(rects)))
            points.append(bbox3d_centers(Scene.boxes_extents(boxes)))
            session.add_correspondences(pixels[-1], points[-1])
    if previous is None:
        return session.solve(intrinsics)
    nudged = session.refine(previous)
    fresh = session.solve(previous.intrinsics)
    if fresh.success:
        from ..oneshot.calibration import project_points

        def median(result):
            uv, depth = project_points(np.concatenate(points), result.intrinsics, result.rotation,
                                       result.translation)
            error = np.linalg.norm(uv - np.concatenate(pixels), axis=1)
            error[depth <= 0] = np.inf
            return float(np.median(error))

        before, after = median(nudged), median(fresh)
        turn = np.degrees(np.arccos(np.clip((np.trace(fresh.rotation @ previous.rotation.T) - 1) / 2, -1, 1)))
        moved = np.linalg.norm(fresh.rotation.T @ np.reshape(fresh.translation, 3)
                               - previous.rotation.T @ np.reshape(previous.translation, 3))
        if after < 0.8 * before and before - after > 1.0 and turn <= RESOLVE_TURN_DEG and moved <= RESOLVE_MOVE_M:
            fresh.message = (f"re-solved: median {before:.1f} -> {after:.1f} px over {len(session)} pair(s); "
                             f"{turn:.1f} deg and {moved:.1f} m from the saved pose")
            fresh.resolved = True
            return fresh
    return nudged


def save_calibration(scene: Scene, intrinsic, extrinsic, distortion=None, source=None):
    """Write the camera calibration where the pairing window reads it.

    ``calib/camera/<camera>.json`` gets the 3x3 intrinsic and the 4x4
    LiDAR-to-camera extrinsic; other keys already in the file are kept, except
    ``source``, the note on where an extrinsic came from: it describes the old
    one, so it is replaced by ``source`` or dropped.

    An extrinsic that gets replaced is not lost: the file as it was is copied
    to ``calib/camera_history/<camera>_<time>.json`` first, outside
    ``calib/camera``, where every file is read as a complete calibration.
    """
    import json
    import shutil
    import time

    target = Path(scene.root) / "calib" / "camera" / f"{scene.camera}.json"
    data = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    replaced = data.get("extrinsic")
    if replaced and not np.allclose(np.ravel(replaced), np.ravel(extrinsic), rtol=0, atol=1e-12):
        history = target.parent.parent / "camera_history"
        history.mkdir(exist_ok=True)
        stamp, number = time.strftime("%Y%m%d-%H%M%S"), 1
        kept = history / f"{scene.camera}_{stamp}.json"
        while kept.exists():  # several saves within one second
            number += 1
            kept = history / f"{scene.camera}_{stamp}_{number}.json"
        shutil.copyfile(target, kept)
    data.update(intrinsic=np.asarray(intrinsic, dtype=float).reshape(-1).tolist(),
                extrinsic=np.asarray(extrinsic, dtype=float).reshape(-1).tolist())
    data.pop("source", None)
    if source:
        data["source"] = source
    if distortion is not None:
        data["distortion"] = np.asarray(distortion, dtype=float).reshape(-1).tolist()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         delete=False, suffix=".tmp") as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def export_training_hdf5(scene: Scene, output, frames=None, *, negatives=False):
    """Write a new derived cache of verified pairs; never overwrite a cache.

    Only paired boxes are included, so an undecided box cannot become a false
    negative. Call separately with explicit, disjoint frame sets for splits.

    With ``negatives=True`` a frame also holds its other camera boxes, saved as
    "no LiDAR box" or not decided (both count as having no LiDAR box), and the
    frame's other LiDAR boxes (:func:`training_frame`), for training with
    ``none_loss``. A frame without a saved pair is not exported.
    """
    import h5py
    import cv2

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    selected = list(frames if frames is not None else scene.frames())
    if len(selected) != len(set(selected)) or not set(selected).issubset(scene.frames()):
        raise ValueError("frames must be unique frame IDs belonging to the scene")
    rows = []
    for frame in selected:
        if negatives:
            rects, boxes, match = training_frame(scene, frame)
        else:
            rects, boxes = verified_frame(scene, frame)
            match = np.eye(len(rects), dtype=np.uint8)
        if not match.any():
            continue
        image, lidar = scene.image_path(frame), scene.lidar_path(frame)
        if not image or not lidar:
            raise ValueError(f"frame {frame}: image or point cloud is missing")
        rows.append((frame, rects, boxes, match, image, lidar))
    if not rows:
        raise ValueError("no verified pairs in selected frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".h5", delete=False) as tmp:
            temporary = Path(tmp.name)
        with h5py.File(temporary, "w") as h5:
            images = h5.create_group("images").create_group(scene.camera).create_dataset(
                "data", shape=(len(rows),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            labels = h5.create_group("labels").create_group("lidar")
            clouds = h5.create_group("point_clouds").create_group("lidar")
            for index, (frame, rects, boxes, match, image, lidar) in enumerate(rows):
                key = f"f{index:08d}"  # loader indexes non-numeric keys by sorted position
                # The public cache contract stores JPEG bytes even if the scene
                # camera image is PNG; the source image remains untouched.
                decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError(f"frame {frame}: image cannot be decoded")
                ok, jpeg = cv2.imencode(".jpg", decoded)
                if not ok:
                    raise ValueError(f"frame {frame}: image cannot be encoded as JPEG")
                images[index] = jpeg.reshape(-1)
                clouds.create_group(key).create_dataset("xyz", data=load_points_xyz(lidar))
                group = labels.create_group(key)
                group.attrs["source_frame"] = frame
                group.create_dataset("num_camera_detections", data=len(rects))
                group.create_dataset("num_lidar_detections", data=len(boxes))
                group.create_dataset("camera_bbox_2d", data=Scene.rects_xyxy(rects))
                group.create_dataset("lidar_bbox_3d", data=Scene.boxes_extents(boxes))
                group.create_dataset("match_matrix", data=match)
        # Windows rename is exclusive; on POSIX link gives the same no-clobber property.
        if os.name == "nt":
            rename_when_free(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows)


def prepare_training_caches(scene: Scene, output_dir, *, max_frames=None, train_fraction=0.8,
                            progress=None, negatives=False):
    """Create temporal train/validation caches and report skipped/overlapping IDs.

    A frame needs at least two confirmed pairs to supply contrastive negatives.
    Invalid frame annotations are reported, never converted to negative labels.
    With ``negatives=True`` (see :func:`export_training_hdf5`) the frame's other
    LiDAR boxes are the negatives, so one confirmed pair is enough when the frame
    also holds a second camera box (a pair, or any other camera box: saved as "no
    LiDAR box" or not decided) and a second LiDAR box: the boxes of a frame are one
    training batch, and batch normalisation needs two of each.
    """
    import json

    if not 0 < train_fraction < 1:
        raise ValueError("train fraction must be between zero and one")
    frames = scene.frames()
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
    usable, invalid = [], []
    ids = {}
    undecided = 0   # camera boxes of the usable frames nobody decided: trained as "no LiDAR box"
    for index, frame in enumerate(frames):
        if progress and index % 250 == 0:
            progress(f"Checking saved pairs: {index}/{len(frames)} frames")
        try:
            rects, boxes = verified_frame(scene, frame)
            enough = len(rects) >= 2
            if negatives and len(rects) == 1:
                with_none, every_box, _ = training_frame(scene, frame)
                enough = len(with_none) >= 2 and len(every_box) >= 2
            if not enough or not scene.image_path(frame) or not scene.lidar_path(frame):
                continue
            usable.append(frame)
            ids[frame] = {str(b.get("obj_id") or "").strip() for b in boxes}
            undecided += sum(1 for rect in scene.load_rects(frame)
                             if not str(rect.get("obj_id") or "").strip() and not rect.get("no_match"))
        except (ValueError, OSError, KeyError) as exc:
            invalid.append({"frame": frame, "reason": str(exc)})
    if progress:
        progress(f"Checked {len(frames)} frames; {len(usable)} can be trained on")
    split = int(len(usable) * train_fraction)
    if split < 1 or len(usable) - split < 1:
        raise ValueError("need at least two frames with two or more valid pairs")
    train_frames, val_frames = usable[:split], usable[split:]
    shared = set().union(*(ids[f] for f in train_frames)) & set().union(*(ids[f] for f in val_frames))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    if progress:
        progress("Exporting training HDF5")
    export_training_hdf5(scene, root / "train.h5", train_frames, negatives=negatives)
    if progress:
        progress("Exporting validation HDF5")
    export_training_hdf5(scene, root / "validation.h5", val_frames, negatives=negatives)
    summary = {"train_frames": len(train_frames), "validation_frames": len(val_frames),
               "train_frame_ids": list(train_frames), "validation_frame_ids": list(val_frames),
               "negatives": bool(negatives),
               "skipped_frames": len(frames) - len(usable), "invalid_frames": invalid,
               "undecided_camera_boxes": undecided,
               "shared_object_ids": sorted(shared), "split": "temporal"}
    (root / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class TrainingProgress:
    """How far a training run is, read from its log as the log grows.

    ``line()`` returns one sentence (the epoch, the share of it that is done, the
    last validation result, the minutes left), or None before the first step.
    """

    def __init__(self, log, epochs):
        import time

        self.log, self.epochs = Path(log), int(epochs)
        self.offset, self.tail = 0, ""
        self.finished, self.top1, self.started = 0, None, time.time()

    def line(self, now=None):
        import re
        import time

        try:
            with self.log.open("rb") as stream:
                stream.seek(self.offset)
                new = stream.read()
        except OSError:
            new = b""
        self.offset += len(new)
        text = self.tail + new.decode("utf-8", errors="replace")
        marks = list(re.finditer(r"epoch (\d+)/\d+ \|[^\n]*?val top1=([\d.]+)%", text))
        if marks:   # the summary the trainer writes after every epoch
            self.finished, self.top1 = int(marks[-1].group(1)), marks[-1].group(2)
            text = text[marks[-1].end():]
        self.tail = text[-4000:]
        steps = re.findall(r"train:\s+(\d+)%", self.tail)   # the progress bar of the running epoch
        if not self.finished and not steps:
            return None
        if self.finished >= self.epochs:
            return f"Epoch {self.epochs} of {self.epochs} done; validation top-1 {self.top1}%."
        part = int(steps[-1]) / 100 if steps else 0.0
        message = f"Epoch {self.finished + 1} of {self.epochs}: {part:.0%}"
        if self.finished:
            message += f"; validation top-1 after epoch {self.finished}: {self.top1}%"
        worked = self.finished + part
        if worked > 0.02:
            left = ((now or time.time()) - self.started) * (self.epochs - worked) / worked
            message += f"; about {max(1, round(left / 60))} min left"
        return message


def train_base_from_caches(output_dir, *, epochs=5, device="auto", pretrained_dir=None,
                           absolute_scores=False, progress=None):
    """Train CRLite from scratch or fine-tune a local CRLite model in a new run.

    ``absolute_scores=True`` is for caches exported with ``negatives=True``: the
    second-stage score is trained to be absolute (``none_loss``) and the model
    folder's config says ``score_mode: absolute``, so every reader of the model
    takes its scores as they are instead of stretching each camera box's best
    candidate to 1.

    ``progress`` is called with a line saying how far the run is (epoch, share of
    the epoch, last validation result, minutes left) every few seconds.
    """
    import shutil
    import subprocess
    import sys

    root = Path(output_dir)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not all((root / name).is_file() for name in ("train.h5", "validation.h5")):
        raise ValueError("Training data is missing. Save reviewed camera/LiDAR pairs, "
                         "then run the HDF5 generation cell before training.")
    weights = None
    if pretrained_dir is None:
        config = Path(__file__).resolve().parent.parent / "cfg" / "crlite_a9_dataset_r02_s01.yaml"
    else:
        source = Path(pretrained_dir)
        weights, config = source / "crlite.pth", source / "crlite.yaml"
        if not weights.is_file() or not config.is_file():
            raise ValueError("pretrained model must contain crlite.pth and crlite.yaml")
    if (root / "training").exists() or (root / "model").exists():
        raise FileExistsError("training or model output already exists; use a new run directory")
    training = root / "training"
    log = root / "training.log"
    # A fresh process imports torch before GUI/image libraries, avoiding native
    # library load-order conflicts and keeping training out of the Qt process.
    cmd = [sys.executable, "-m", "xcalib.engine.scene_train", str(root / "train.h5"),
           str(root / "validation.h5"), str(config), str(training), str(epochs), str(device)]
    if weights is not None:
        cmd.append(str(weights))
    if absolute_scores:
        cmd.append("--none-loss")
    with log.open("w", encoding="utf-8") as stream:
        if progress is None:
            code = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        else:
            watch, said = TrainingProgress(log, epochs), None
            process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        code = None
                    message = watch.line()
                    if message and message != said:
                        progress(message)
                        said = message
                    if code is not None:
                        break
            except BaseException:
                # The notebook cell was stopped (the ■ button) or failed: the training and its
                # data loader workers end with it instead of running on unseen.
                from .processes import stop_process_tree

                stop_process_tree(process)
                raise
    if code:
        raise RuntimeError(f"base training failed (exit {code}); see {log}")
    best = training / "best.pth"
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=False)
    shutil.copyfile(best, model_dir / "crlite.pth")
    text = Path(config).read_text(encoding="utf-8")
    if absolute_scores and not any(line.startswith("score_mode:") for line in text.splitlines()):
        text = text.rstrip("\n") + "\n# Trained with the none term: the second-stage score is read as it is.\n" \
                                   "score_mode: absolute\n"
    (model_dir / "crlite.yaml").write_text(text, encoding="utf-8")
    return model_dir


def install_new_base(scene: Scene, source):
    """Install a model only where no model directory exists."""
    import shutil

    source = Path(source)
    weights, config = source / "crlite.pth", source / "crlite.yaml"
    if not weights.is_file() or not config.is_file():
        raise ValueError("trained model must contain crlite.pth and crlite.yaml")
    if (not scene.camera or scene.camera in (".", "..")
            or any(c in scene.camera for c in "/\\") or Path(scene.camera).drive):
        raise ValueError("camera name must be one safe path component")
    target = Path(scene.root) / "calib" / "xcalib" / f"{scene.camera}_model"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    try:
        shutil.copyfile(weights, target / weights.name)
        shutil.copyfile(config, target / config.name)
    except Exception:
        for name in ("crlite.pth", "crlite.yaml"):
            (target / name).unlink(missing_ok=True)
        target.rmdir()
        raise
    return target


def install_model(scene: Scene, source):
    """Make ``source`` the scene's model; a different model already there is kept.

    Returns ``(target, kept)``: the scene's model folder, and the folder the
    previous model went to (``None`` when the scene had none, or already holds
    these weights). The previous folder is moved beside the new one with the
    time in its name, with everything in it (a continual corpus, revisions).
    """
    import filecmp
    import time

    source = Path(source)
    weights = source / "crlite.pth"
    if not weights.is_file() or not (source / "crlite.yaml").is_file():
        raise ValueError("trained model must contain crlite.pth and crlite.yaml")
    target = Path(scene.root) / "calib" / "xcalib" / f"{scene.camera}_model"
    kept = None
    if target.exists():
        current = target / "crlite.pth"
        if (current.is_file() and not (target / "current.json").exists()
                and filecmp.cmp(weights, current, shallow=False)):
            return target, None
        kept = target.with_name(f"{target.name}_replaced_{time.strftime('%Y%m%d-%H%M%S')}")
        target.rename(kept)
    try:
        install_new_base(scene, source)
    except Exception:
        if kept is not None:
            kept.rename(target)
        raise
    return target, kept


def saved_decisions(review):
    """The decisions a person saved in one reviewed frame, as rows for the continual corpus.

    A saved pair becomes a ``positive`` row and a "no LiDAR box" mark a
    ``no_match`` row; a camera box without a decision gives none. The ranked
    options become the row's candidates and the model's suggestion its
    proposal, which is how the corpus tells a confirmed suggestion from a
    correction.
    """
    from ..continual import VerifiedPair

    pairs = []
    for track in review.tracks:
        if not track.applied_id and not track.rejected:
            continue
        candidates = [{"lidar_key": "id:" + option["lidar_id"], "score": option["score"]}
                      for option in track.options if option.get("lidar_id")]
        proposal = ({"decision": "positive", "lidar_key": "id:" + track.suggestion}
                    if track.suggestion else None)
        pairs.append(VerifiedPair(
            rect=list(track.rect), decision="positive" if track.applied_id else "no_match",
            lidar_key="id:" + track.applied_id if track.applied_id else None,
            track=track.key if track.key.isdigit() else None,
            candidates=candidates, proposal=proposal))
    return pairs


def record_saved_pairs(scene: Scene, *, frames=None, threshold=0.5, device=None, progress=None):
    """Record the decisions saved in a scene into its continual corpus.

    ``run_continual_refine`` learns from the corpus, not from the labels: this
    is the step between Apply in the pairing window and a refine run. Every
    frame of ``frames`` (default: the frames with a saved pair) contributes its
    saved pairs and "no LiDAR box" marks, with what the scene's model proposes
    for them. Recording a frame again replaces its rows. Returns the number of
    decisions recorded.
    """
    from ..continual import open_store, record_verified_pairs
    from ..engine.matcher import Matcher
    from .scene import match_review, matcher_target

    target = matcher_target(scene)
    if target is None:
        raise ValueError(f"the scene has no model (calib/xcalib/{scene.camera}_model)")
    name, weights, config = target
    matcher = Matcher.from_pretrained(name, weights=weights, config=config, device=device)
    root = Path(scene.root).resolve()
    if frames is None:
        frames = [frame for frame in scene.frames()
                  if any(str(rect.get("obj_id") or "").strip() and not rect.get("no_match")
                         for rect in scene.load_rects(frame))]
    store = open_store(scene.model_dir()[0])
    recorded = 0
    try:
        for index, frame in enumerate(frames):
            try:
                verified_frame(scene, frame)
            except ValueError as error:
                if progress:
                    progress(f"Skipping a frame whose saved pairs are not valid: {error}")
                continue
            pairs = saved_decisions(match_review(matcher, scene, frame, threshold))
            if pairs:
                recorded += record_verified_pairs(store, str(root.parent), root.name, scene.camera,
                                                  frame, pairs)
            if progress and index % 20 == 0:
                progress(f"Recording saved decisions: {index + 1}/{len(frames)} frames")
    finally:
        store.close()
    return recorded


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Export verified scene pairs as a training cache")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--frames-file", required=True,
                        help="text file with one selected frame ID per line")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    frames = [line.strip() for line in Path(args.frames_file).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    print(f"Exported {export_training_hdf5(Scene(args.scene, args.camera), args.output, frames)} frames")


if __name__ == "__main__":
    main()
