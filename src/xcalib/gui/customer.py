"""One-call steps behind demo/xcalib_homography_en.ipynb.

Every cell of the notebook is a single call on a ``Workflow``, every choice is
a folder dialog, and a step that cannot continue says what to do instead of
showing a traceback. The steps only hand over to the import, pairing, training
and homography code they name.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .scene import Scene

RAW_INTRINSIC = "undistortion_parameters.json"


class Stop(Exception):
    """The step cannot continue; the message tells the user what to do next."""


def _warn(message):
    """Messages the user must act on stand out in the notebook (stderr)."""
    try:
        print(f"⚠ {message}", file=sys.stderr)
    except UnicodeEncodeError:  # a legacy console outside Jupyter
        print(f"! {message}", file=sys.stderr)


def _parts_of_the_image(names):
    """"on the left and in the middle" for ["left", "middle"]: each part with its preposition."""
    return " and ".join(("in the " if name == "middle" else "on the ") + name for name in names)


def _show_only_the_message(shell, kind, value, traceback, tb_offset=None):
    _warn(value)


def _status_line():
    """A line that a long step keeps updating in place instead of filling the
    cell with progress; outside Jupyter every update is printed."""
    try:
        from IPython.display import display
    except ImportError:
        return print
    # Raw bundles: IPython's text objects would read a message that happens to
    # name an existing path ("..." on Windows) as a file.
    handle = display({"text/plain": "..."}, raw=True, display_id=True)
    if handle is None:
        return print
    return lambda message: handle.update({"text/plain": str(message)}, raw=True)


_WINDOWS = os.name == "nt"
_NUMERICS = None  # asked once per process


def _undecided_beside_pairs(count, frames, order):
    """The warning about camera boxes without a decision in frames that hold a saved pair:
    training counts them as "no LiDAR box". Frames are numbered as the pairing window's
    frame field takes them."""
    where = {frame: number for number, frame in enumerate(order, 1)}
    numbers = sorted(where[frame] for frame in frames if frame in where)
    shown = ", ".join(str(n) for n in numbers[:8]) + (", ..." if len(numbers) > 8 else "")
    return (f"{count} camera boxes have no decision in {len(numbers)} frames that hold a saved pair "
            f"(frame {shown} in the pairing window). Training counts them as \"no LiDAR box\": if one "
            "of them has a LiDAR box, please pair it in the pairing window, or choose — none — for it.")


def _debugger_on():
    """Whether the Jupyter debugger (the bug icon at the top right) is switched on for this kernel."""
    try:
        return bool(get_ipython().kernel.debugger.is_started)  # noqa: F821
    except Exception:  # plain Python, or a kernel without a debugger
        return False


def _python():
    """The Python that runs the notebook's dialogs, windows and training.

    While the Jupyter debugger is switched on, every Python process the kernel starts
    waits for the debugger to attach to it: a dialog or window then never appears and
    the cell stays busy. The step stops with a message instead. Once the debugger is
    switched off again, the processes start normally.
    """
    if _debugger_on():
        raise Stop("The Jupyter debugger is switched on (the bug icon at the top right of the "
                   "notebook). While it is on, the dialogs and windows of this notebook cannot open. "
                   "Please switch it off and run this cell again.")
    return sys.executable


def _numerics_work():
    """Whether this environment can do the linear algebra the steps need.

    On Windows the numeric libraries of numpy and of torch are kept apart by
    settings of the notebook environment (demo/pixi.toml, [activation.env]). A
    Jupyter that was not started through pixi lacks them, and its kernel then
    ends without a message at the first large matrix product. A fresh process
    is asked, so that the kernel survives the answer.
    """
    global _NUMERICS
    if _NUMERICS is None:
        python = _python()   # stops here, before anything is noted, while the debugger is on
        _NUMERICS = True
        if _WINDOWS:
            code = ("import torch, numpy as np; a = np.random.rand(300000, 3); "
                    "print(float((a @ np.random.rand(3, 3)).sum()) > 0)")
            try:
                done = subprocess.run([python, "-c", code], capture_output=True, text=True,
                                      timeout=300, check=False)
                _NUMERICS = getattr(done, "returncode", 0) == 0
            except (OSError, subprocess.SubprocessError):
                pass  # could not be asked: nothing is known against it
    return _NUMERICS


_CODE_ROOT = Path(__file__).resolve().parents[3]   # the repository this package runs from
_SUSTECH = {}   # the SUSTechPOINTS page this kernel started: its launcher, scenes and address


def _stop_sustech():
    """Stop the SUSTechPOINTS page this kernel started, if one runs. Closing the launcher's
    input makes it stop the page and remove its temporary copy; a launcher that has not
    ended after a while is ended together with the page."""
    server = _SUSTECH.pop("server", None)
    if server is None or server["process"].poll() is not None:
        return
    try:
        server["process"].stdin.close()
        server["process"].wait(timeout=20)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        from .processes import stop_process_tree

        # started in a session of its own outside Windows: its page ends with it
        stop_process_tree(server["process"], own_group=True)


atexit.register(_stop_sustech)


def _last_line(path):
    """The last line of a log that holds text: it usually says why a program ended."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


class Workflow:
    """State shared by the notebook cells: the chosen scenes and the trained models."""

    def __init__(self, base_epochs=1, refine_epochs=1):
        try:   # first, so that a Stop of this very cell is shown as a ⚠ message too
            get_ipython().set_custom_exc((Stop,), _show_only_the_message)  # noqa: F821
        except (NameError, AttributeError):  # plain Python, or a stand-in without it: Stop is raised normally
            pass
        here = Path.cwd().resolve()
        candidates = (here, *here.parents, Path(__file__).resolve().parents[3])
        self.repo = next((p for p in candidates if (p / "pyproject.toml").is_file()
                          and (p / "src" / "xcalib" / "gui" / "customer.py").is_file()), None)
        if self.repo is None:
            raise Stop("Open Jupyter from the XCalib repository or its demo folder.")
        if not _numerics_work():
            raise Stop("This Jupyter was started without the settings of the notebook environment, "
                       "and some steps would end without a message. Close it and start it from "
                       "the demo folder through pixi: pixi run notebook-homography.")
        self.epochs = {"base": base_epochs, "refine": refine_epochs}
        self.scenes, self.models = {}, {}
        self.start = self.repo / "datasets"
        self.inputs = None   # where the raw data came from: other input files are looked for there
        print("XCalib folder:", self.repo)
        print("Ready. Run the cells one at a time, from top to bottom.")

    # ---- choices ----

    def _pick(self, title, initial=None, files=None):
        """Folder dialog (a file dialog with a name filter in `files`) in its own Qt
        process, outside the notebook kernel."""
        command = [_python(), "-m", "xcalib.gui.folder_picker", "--title", title,
                   "--initial", str(initial or self.start), *(["--files", files] if files else [])]
        try:
            result = subprocess.run(command, cwd=self.repo, capture_output=True, text=True, check=True)
        except KeyboardInterrupt:   # the ■ button: on Windows it arrives once the dialog has closed
            raise Stop("Stopped: nothing was chosen. Run this cell again.") from None
        selected = json.loads(result.stdout.strip().splitlines()[-1])
        if not selected:
            raise Stop("Selection cancelled. Run this cell again.")
        return Path(selected).resolve()

    def choose(self, role):
        """Ask for the base or refine scene folder and report what it contains."""
        if role not in ("base", "refine"):
            raise Stop('Use choose("base") or choose("refine").')
        self._use_scene(role, self._pick(f"Select the {role.upper()} scene folder (it contains lidar and camera)"))

    @staticmethod
    def _scene_in(folder, role):
        """The scene a chosen folder stands for: the folder itself, or the one scene of that
        role in a dataset folder the import wrote; None when it is neither."""
        if not (folder / "lidar").is_dir():  # the dataset folder was picked instead of the scene
            inside = sorted([*folder.glob(f"*_{role}"), *(folder / "refine").glob(f"*_{role}")])
            if len(inside) == 1:
                folder = inside[0]
        return folder if (folder / "lidar").is_dir() and (folder / "camera").is_dir() else None

    def _use_scene(self, role, folder):
        """Make a scene folder the base or refine scene of this session and report what it contains."""
        scene = self._scene_in(folder, role)
        if scene is None:
            raise Stop(f"{folder} is not a scene: choose the folder that contains lidar and camera.")
        cameras = sorted(p.name for p in (scene / "camera").iterdir() if p.is_dir())
        if len(cameras) != 1:
            raise Stop("This workflow needs exactly one camera folder in the scene.")
        other = self.scenes.get("base" if role == "refine" else "refine")
        if other and other[0] == scene:
            raise Stop("Base and refine must be different scene folders.")
        self.scenes[role] = (scene, cameras[0])
        self.start = scene.parent
        boxes = len(list((scene / "label").glob("*.json")))
        rects = len(list((scene / "label_fusion" / "camera" / cameras[0]).glob("*.json")))
        print(f"{role.capitalize()} scene: {scene}")
        print(f"  {len(Scene(str(scene), cameras[0]).frames())} frames, LiDAR boxes in {boxes}, "
              f"camera boxes in {rects}")

    def _scene(self, role):
        if role not in self.scenes:
            self.choose(role)
        return self.scenes[role]

    def _model(self, role="base"):
        """The model trained in this session, or one chosen from disk."""
        if role not in self.models:
            folder = self._pick(f"Select the {role.upper()} model folder (it contains crlite.pth)",
                                self.repo / "runs")
            if not (folder / "crlite.pth").is_file() and (folder / "model").is_dir():
                folder = folder / "model"
            if not all((folder / name).is_file() for name in ("crlite.pth", "crlite.yaml")):
                raise Stop("Choose the model folder that contains crlite.pth and crlite.yaml.")
            self.models[role] = folder
        return self.models[role]

    def choose_model(self, role="base"):
        """Ask for a model folder made earlier and use it as the base (or refine) model of
        this session, instead of the one a training step of this session wrote."""
        if role not in ("base", "refine"):
            raise Stop('Use choose_model("base") or choose_model("refine").')
        previous = self.models.pop(role, None)
        try:
            folder = self._model(role)
        except Stop:
            if previous is not None:   # nothing was chosen: the model of this session stays
                self.models[role] = previous
            raise
        print(f"{role.capitalize()} model:", folder)
        try:
            config = (folder / "crlite.yaml").read_text(encoding="utf-8")
        except OSError:
            config = ""
        if not any(line.strip() == "score_mode: absolute" for line in config.splitlines()):
            _warn("This model scores the LiDAR boxes of a frame against each other (its crlite.yaml "
                  "has no \"score_mode: absolute\"), so it suggests a LiDAR box for nearly every camera "
                  "box, also for one that has none. Please look at its suggestions with care before "
                  "you apply a frame.")
        if role == "base":
            print("It suggests the pairs of the refine data and is the start of the fine-tuning.")

    # ---- step 0: raw delivery -> scenes ----

    def import_raw(self, role=None):
        """Turn a rectified raw delivery into scenes with draft boxes: a base and a
        refine part, or with "refine" (or "base") the whole delivery as that scene, which
        then is the scene of that role in this session. With a role, a scene imported
        before can be chosen in the same dialog instead of raw data."""
        from .dataset_prepare import delivery_layout
        from .label_import import camera_record, label
        from .lidar_import import label_lidar

        if role not in (None, "base", "refine"):
            raise Stop('Use import_raw(), import_raw("base") or import_raw("refine").')
        raw = self._pick("Select the raw data folder" if role is None else
                         f"Select the raw data folder of the {role.upper()} data "
                         f"(all of its frames become the {role} scene)")
        try:
            layout = delivery_layout(raw)
        except ValueError as error:
            if role and self._scene_in(raw, role) is not None:   # a scene imported before
                self._use_scene(role, raw)
                return
            raise Stop(str(error)) from None
        self.inputs = raw.parent
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw.name)
        if not name:   # the top of a drive: the scenes would land in raw_workflow itself
            raise Stop("Please choose the folder that holds the raw data, not the top of a drive: copy "
                       "the data into a folder of its own and run this cell again.")
        output = self.repo / "datasets" / "raw_workflow" / name
        every = [output / f"{name}_base", output / "refine" / f"{name}_refine"]
        scenes = [every[role == "refine"]] if role else every
        if output.exists():
            # The folder name alone decides where a delivery goes, and only letters, digits, - and _
            # are kept in it: a different delivery can land on the same name. The import keeps the
            # point-cloud file names, so the same delivery shares its frames with the scenes there.
            delivered = {path.stem for path in layout["clouds"].iterdir()
                         if path.suffix.lower() in (".pcd", ".bin", ".npy")}
            imported = {frame for scene in every if (scene / "lidar").is_dir()
                        for frame in Scene(str(scene), "cam_sync").frames()}
            if delivered and imported and not delivered & imported:
                raise Stop(f"{output} holds another delivery: this data folder gets the same name there "
                           f"({name}; in a folder name only letters, digits, - and _ are kept, and each run "
                           "of other characters becomes one _). Please give this data folder a name that "
                           "differs in those characters and run this cell again.")
            if not all((scene / "lidar").is_dir() for scene in scenes):
                other = every[role != "refine"] if role else None
                if other is not None and (other / "lidar").is_dir():
                    kind = "refine" if role == "base" else "base"
                    raise Stop(f"This data is the {kind} data already: all of its frames are in the {kind} "
                               f"scene {other}. Please choose other raw data for the {role} part.")
                raise Stop(f"{output} was imported in a different way before. "
                           "Delete this folder to import the data again.")
            print("Already imported (delete this folder to import again):", output)
            if role == "refine" and (every[0] / "lidar").is_dir():
                print("This data was imported as a base and a refine part: the refine scene is the "
                      "part that the base scene does not hold.")
        else:
            intrinsic = None
            # The camera of a refine delivery is the base camera, when a base scene is chosen.
            base = self.scenes.get("base") if role == "refine" else None
            kept = base[0] / "calib" / "intrinsic" / f"{base[1]}.json" if base else None
            if (raw / RAW_INTRINSIC).is_file():
                print("Camera intrinsic:", raw / RAW_INTRINSIC)
            elif kept is not None and kept.is_file():
                intrinsic = kept
                print("The data has no camera intrinsic of its own; the base scene's is used "
                      "(same camera).")
            else:
                # The file is not always kept with the data: ask where it is.
                print(f"The data folder has no {RAW_INTRINSIC}: please choose the camera "
                      "intrinsic file in the dialog.")
                intrinsic = self._pick("Select the camera intrinsic file (for example "
                                       f"{RAW_INTRINSIC})", raw.parent,
                                       files="Camera intrinsic (*.json)")
                try:
                    camera_record(intrinsic)
                except (OSError, ValueError) as error:
                    raise Stop(f"{intrinsic.name} cannot be used as the camera intrinsic: {error}. "
                               "Choose the file with the camera matrix of the undistorted images "
                               "(K_undistort) and run this cell again.") from None
                print("Camera intrinsic:", intrinsic)
            print("Copying images and point clouds; this takes a few minutes ...")
            status = _status_line()
            try:
                counts = label(raw, output, scene_prefix=name, role=role, intrinsic=intrinsic,
                               progress=status)["frame_counts"]
            except ValueError as error:
                raise Stop(f"The data was not imported: {error}") from None
            except PermissionError:
                # Raised only after the import has waited for a minute: the partly written
                # scenes are removed again, so the cell can simply be run once more.
                raise Stop("The data was not imported: another program (often a virus scanner "
                           "or the Windows search indexer) kept the newly copied files open, so "
                           "they could not be moved into place. Nothing was kept; please run "
                           "this cell again.") from None
            status("Copied " + " and ".join(f"{counts[scene.name]} {scene.name.rsplit('_', 1)[1]} frames"
                                            for scene in scenes) + ".")
        if not (scenes[0] / "vendor" / "lidar_transform.json").is_file():
            # The base model lives in the base scene's LiDAR frame: a refine delivery
            # joins it when a base scene is chosen in this session and recorded its frame.
            base = self.scenes["base"][0] if role == "refine" and "base" in self.scenes else None
            if base and not (base / "vendor" / "lidar_transform.json").is_file():
                base = None
            print("Writing LiDAR 3D boxes; this takes a few minutes ...")
            status = _status_line()
            try:
                report = label_lidar(scenes, frame_of=base, progress=status)
            except ValueError as error:
                raise Stop(f"LiDAR boxes were not written: {error}") from None
            except PermissionError:
                raise Stop("LiDAR boxes were not written: another program (often a virus scanner "
                           "or the Windows search indexer) kept the new point-cloud files open. "
                           "Please run this cell again.") from None
            transform = report["transform"]
            if transform["source"] == "reference scene":
                status("Point clouds and LiDAR boxes were moved into the base scene's LiDAR frame, "
                       f"where the base model works; {transform['cloud_overlap']:.0%} of the base "
                       "scene's space is matched.")
            else:
                status("LiDAR boxes "
                       + ("were already in the point-cloud frame" if transform["source"] == "identity"
                          else "were moved from the vendor frame into the point-cloud frame")
                       + f"; {transform['inside_fraction']:.0%} of the moving points lie inside them.")
            for scene, counts in report["scenes"].items():
                print(f"  {scene}: {counts['boxes_written']} boxes in "
                      f"{counts['frames_written']} frames")
        self.start = output
        self._show_check_images(scenes[0])
        if role:
            self._use_scene(role, scenes[0])
            return
        for scene in scenes:
            print(f"{scene.name.rsplit('_', 1)[1].capitalize()} scene:".ljust(13), scene)

    def _show_check_images(self, scene, camera="cam_sync", count=3):
        """Show the frames with the most 3D boxes: camera drafts left, LiDAR drafts right."""
        try:
            import cv2
            from IPython.display import Image, display
        except ImportError:
            return
        from .lidar_import import preview

        for frame in self._busiest_frames(Scene(str(scene), camera), count):
            image = cv2.cvtColor(preview(scene, frame, camera=camera, size=540), cv2.COLOR_RGB2BGR)
            print(f"Check image, frame {frame}: camera boxes (left), LiDAR boxes from above (right)")
            display(Image(data=cv2.imencode(".jpg", image)[1].tobytes()))

    @staticmethod
    def _busiest_frames(source, count):
        frames = source.frames()
        sample = frames[:: max(1, len(frames) // 40)]
        return sorted(sorted(sample, key=lambda frame: -len(source.load_boxes(frame)))[:count])

    # ---- pairing windows ----

    def _window(self, module, scene, camera, *options):
        """Open a pairing window and wait until the user closes it."""
        python = _python()   # stops before anything is said while the debugger is on
        print("A window opened (check the taskbar if you do not see it). "
              "Close it when you are done; this cell then finishes.")
        window = subprocess.Popen([python, "-m", module, "--scene", str(scene), "--camera", camera,
                                   *options], cwd=self.repo)
        while True:
            try:
                window.wait()
                return
            except KeyboardInterrupt:
                # The ■ button never closes the window: its decisions that are not applied yet
                # would be lost. The cell ends when the window is closed, and reports as usual.
                print("The ■ button does not close the pairing window: please close the window "
                      "itself. What you applied is saved.")

    def pair(self, role):
        """Open the pairing window; the refine scene gets the base model's suggestions."""
        scene, camera = self._scene(role)
        if role == "refine":
            from .workflow import install_model

            target = scene / "calib" / "xcalib" / f"{camera}_model"
            if self.models.get("base"):   # trained or chosen in this session: it makes the suggestions
                try:
                    target, kept = install_model(Scene(str(scene), camera), self.models["base"])
                except (OSError, ValueError) as error:
                    raise Stop(f"The base model could not be put into the scene ({error}). If a pairing "
                               "window of this scene is open, close it; then run this cell again.") from None
                print("Base model in the scene for pairing suggestions:", target)
                if kept:
                    print("The model that was there before is kept in", kept)
            elif Scene(str(scene), camera).model_dir() is not None:
                print("The scene already has a model, which the window uses:", target)
            else:   # no model yet, or a model folder without weights, which install_model moves aside
                try:
                    target, kept = install_model(Scene(str(scene), camera), self._model("base"))
                except (OSError, ValueError) as error:
                    raise Stop(f"The base model could not be put into the scene ({error}). If a pairing "
                               "window of this scene is open, close it; then run this cell again.") from None
                print("Base model installed in the scene for pairing suggestions:", target)
                if kept:
                    print("The folder that was there before is kept in", kept)
        self._window("xcalib.gui.review", scene, camera, "--notebook-workflow")
        self._report_pairs(scene, camera, before_training=True)

    def _report_pairs(self, scene, camera, before_training=False):
        """Say how many pairs the scene holds now, after a pairing window closed; before a
        training step, also where boxes without a decision would be trained as "no LiDAR box"."""
        labels = self._camera_labels(scene, camera)
        pairs = [sum(1 for r in rects if str(r.get("obj_id") or "").strip() and not r.get("no_match"))
                 for rects in labels]
        undecided = [sum(1 for r in rects if not str(r.get("obj_id") or "").strip() and not r.get("no_match"))
                     for rects in labels]
        ready = sum(1 for n in pairs if n >= 2)
        print(f"Saved pairs: {sum(pairs)} in {sum(1 for n in pairs if n)} frames; "
              f"{ready} frames have two or more.")
        print(f"Camera boxes without a decision: {sum(undecided)} in "
              f"{sum(1 for n in undecided if n)} frames.")
        order = Scene(str(scene), camera).frames()
        beside = [frame for frame, paired, left in zip(order, pairs, undecided) if paired and left]
        if before_training and beside:
            _warn(_undecided_beside_pairs(sum(left for paired, left in zip(pairs, undecided) if paired),
                                          beside, order))
        if ready < 2:
            _warn("Training needs at least two frames with two saved pairs each. "
                  "Run this cell again to add pairs.")

    def pair_remaining(self, role):
        """Put the model trained for this scene into it and open the pairing window on
        the camera boxes without a decision, with the model's pairs to confirm."""
        from .workflow import install_model

        scene, camera = self._scene(role)
        model = self._model(role)
        try:
            target, kept = install_model(Scene(str(scene), camera), model)
        except OSError as error:
            raise Stop(f"The model could not be put into the scene ({error}). Close every "
                       "pairing window of this scene and run this cell again.") from None
        print(f"{role.capitalize()} model in the scene:", target)
        if kept:
            print("The model that was there before is kept in", kept)
        print("The model pairs the camera boxes that have no decision yet; "
              "this takes a few minutes ...")
        status = _status_line()
        report, proposals, check = self._model_pairs(scene, camera, status)
        if not report["open_frames"]:
            status("Every camera box of this scene has a decision already.")
            return
        if not report["pairs"]:
            status(f"The model suggests no pair in the {report['open_frames']} frames that "
                   "still have undecided camera boxes.")
            return
        status(f"The model suggests {report['pairs']} pairs in {report['frames']} of the "
               f"{report['open_frames']} frames that still have undecided camera boxes.")
        print("They are shown as (to confirm) and are NOT saved yet: look at every frame, "
              "correct what is wrong, then click Apply.")
        self._show_suggestions(scene, camera, proposals)
        options = ["--proposals", str(proposals), "--only-listed"]
        if report.get("doubted"):
            print(f"{report['doubted']} of them do not fit the scene's extrinsic and are marked "
                  "(check): look at those first.")
            options += ["--check", str(check)]
        self._window("xcalib.gui.review", scene, camera, "--notebook-workflow", *options)
        self._report_pairs(scene, camera)

    def _show_suggestions(self, scene, camera, proposals, count=2):
        """Show the model's pairs of the frames that have the most of them, before the
        window opens: what the model predicts stays visible in the notebook."""
        from .results import spread_frames
        from .saved_pairs import preview

        try:
            pairs = json.loads(Path(proposals).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        source = Scene(str(scene), camera)
        self._show_pictures([(f"The model's pairs, frame {frame} (not saved yet): a pair has one "
                              "colour in the camera image (left) and seen from above (right); the "
                              "number after the LiDAR id is the model's score.",
                              lambda frame=frame: preview(source, frame, suggestions=pairs[frame]))
                             for frame in spread_frames({f: len(p) for f, p in pairs.items()}, count)])

    def _model_pairs(self, scene, camera, status):
        """Run the scene's model in its own process, as training does; returns its report,
        the file with the pairs it suggests and the file naming the doubtful ones."""
        output, check = (tempfile.NamedTemporaryFile(prefix=f"xcalib-model-{kind}-", suffix=".json",
                                                     delete=False) for kind in ("pairs", "check"))
        output.close()
        check.close()
        command = [_python(), "-m", "xcalib.gui.model_pairs", "--scene", str(scene),
                   "--camera", camera, "--output", output.name, "--check-output", check.name,
                   "--device", "auto"]   # a GPU when the environment has one
        log = tempfile.NamedTemporaryFile(prefix="xcalib-model-pairs-", suffix=".log", delete=False)
        report = None
        with log:
            process = subprocess.Popen(command, cwd=self.repo, stdout=subprocess.PIPE, stderr=log,
                                       text=True, encoding="utf-8")
            try:
                for line in process.stdout:
                    if line.startswith("RESULT "):
                        report = json.loads(line[7:])
                    elif line.strip():
                        status(line.strip())
                process.wait()
            except BaseException as error:   # the ■ button, or a failure here: the model run ends too
                from .processes import stop_process_tree

                stop_process_tree(process)
                if isinstance(error, KeyboardInterrupt):
                    status("The model run was stopped.")
                    raise Stop("The model run was stopped before it finished; nothing was saved. Run "
                               "this cell again to let the model pair the remaining frames.") from None
                raise
        if report is None:
            status("The model could not be run.")
            raise Stop(f"The model could not be run on this scene. Details: {log.name}")
        return report, Path(output.name), Path(check.name)

    @staticmethod
    def _camera_labels(scene, camera):
        source = Scene(str(scene), camera)
        return [source.load_rects(frame) for frame in source.frames()]

    # ---- training ----

    @staticmethod
    def _training_device():
        """What the training process will run on, asked of a fresh process as training
        is one: importing torch here would load it beside the kernel's image libraries."""
        code = ("import torch; print('the GPU (' + torch.cuda.get_device_name(0) + ')' "
                "if torch.cuda.is_available() else 'the CPU')")
        try:
            found = subprocess.run([_python(), "-c", code], capture_output=True, text=True,
                                   timeout=120, check=False).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            found = []
        return found[-1] if found and found[-1].startswith("the ") else "this computer"

    def train(self, role):
        """Export the saved pairs and train (base) or fine-tune (refine) the matcher."""
        from .workflow import prepare_training_caches, train_base_from_caches

        _python()   # while the Jupyter debugger is on, stop before the export, not after it
        scene, camera = self._scene(role)
        pretrained = self._model("base") if role == "refine" else None
        run = self.repo / "runs" / f"customer-{role}-{time.strftime('%Y%m%d-%H%M%S')}"
        print("Exporting the saved pairs as training data ...")
        status = _status_line()
        try:
            # The camera boxes saved as "no LiDAR box" and the frame's other LiDAR boxes go
            # in too: the model learns to leave a camera box without a partner unpaired.
            split = prepare_training_caches(Scene(str(scene), camera), run, progress=status,
                                            negatives=True)
        except ValueError as error:
            status("No training data was exported.")
            raise Stop(f"Not enough saved pairs to train ({error}). Go back to the pairing step: "
                       "at least two frames need two saved pairs each.") from None
        except PermissionError:
            status("No training data was exported.")
            raise Stop("The training data was not written: another program (often a virus scanner or "
                       "the Windows search indexer) kept the new HDF5 file open, so it could not be "
                       f"moved into place. Please run this cell again; the unfinished folder {run} "
                       "can be deleted.") from None
        except KeyboardInterrupt:   # the ■ button during the export
            status("No training data was exported.")
            raise Stop("The export was stopped before it finished, and nothing was trained. Run this "
                       f"cell again to train; the folder {run}, if it was made, can be deleted.") from None
        status(f"Exported {split['train_frames']} training frames and "
               f"{split['validation_frames']} validation frames.")
        print(f"Training data (HDF5): {run / 'train.h5'} and {(run / 'validation.h5').name} beside it; "
              "split.json there lists the frames of each.")
        if split.get("undecided_camera_boxes"):
            print(f"{split['undecided_camera_boxes']} camera boxes in these frames have no decision; "
                  "the training counts them as \"no LiDAR box\".")
        if pretrained is not None:
            print("Fine-tuning starts from the base model:", pretrained)
        try:
            print(f"Training for {self.epochs[role]} epoch(s) on {self._training_device()}. "
                  "Please wait ...")
            self.models[role] = train_base_from_caches(run, epochs=self.epochs[role], device="auto",
                                                       pretrained_dir=pretrained, absolute_scores=True,
                                                       progress=_status_line())
        except RuntimeError:
            raise Stop("The training stopped with an error and no model was written. "
                       f"Details: {run / 'training.log'}") from None
        except KeyboardInterrupt:   # the ■ button: the training process was ended with the cell
            raise Stop("The training was stopped before it finished, and no model was written. "
                       f"Run this cell again to train; the unfinished folder {run} can be deleted.") from None
        print(f"{role.capitalize()} model:", self.models[role])
        print("Training log:", run / "training.log")
        try:
            history = json.loads((run / "training" / "history.json").read_text(encoding="utf-8"))
            best = max(history, key=lambda row: row["val_mrr"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        print(f"On the validation frames the right LiDAR box is the model's first choice for "
              f"{best['val_top1']:.1%} of the saved pairs (epoch {best['epoch']} of {len(history)} "
              "is kept). The number says the model learned the pairs you saved; how it does on "
              "frames nobody paired shows in the pairing window.")

    # ---- pairs you have, what is saved, and the homography (no extrinsic) ----

    def import_pairs(self, role="base"):
        """Copy pairs that exist already (a folder of saved camera labels) onto the camera
        boxes of the scene, so that they need not be clicked again."""
        from .saved_pairs import import_pairs

        scene, camera = self._scene(role)
        folder = self._pick("Select the folder with the pairs you already have (a scene folder, its "
                            f"label_fusion/camera/{camera} folder, or a folder of saved label files)",
                            self.inputs or self.repo / "datasets")
        print("Copying the pairs ...")
        try:
            report = import_pairs(Scene(str(scene), camera), folder)
        except ValueError as error:
            raise Stop(f"No pairs were copied: {error}.") from None
        print("Pairs from:", report["folder"])
        print(f"Copied {report['pairs']} pairs and {report['none']} \"no LiDAR box\" marks into "
              f"{report['frames']} frames of this scene.")
        if report.get("same"):
            print(f"Already in this scene: {report['same']} camera boxes hold the folder's decision "
                  "already (copied in before, or decided the same way here).")
        for count, what in ((report["kept"], "camera boxes were decided differently here already, "
                                             "and this scene's decision was kept"),
                            (report.get("taken", 0), "pairs name a LiDAR box that another camera box of "
                                                     "the same frame holds here"),
                            (report["unmatched"], "decisions belong to camera boxes this scene does not have"),
                            (report["unknown_lidar"], "pairs name a LiDAR box this scene does not have"),
                            (report["other_frames"], "files are frames this scene does not have")):
            if count:
                print(f"Left out: {count} {what}.")
        if not report["pairs"] and not report["none"]:
            if report.get("same"):    # the folder matches this scene; its pairs are here already
                print("Nothing new to copy: this scene holds the folder's pairs already.")
            else:
                _warn("Nothing was copied. The folder must hold the pairs of the same delivery: the "
                      "same frames and the same camera boxes.")
        self._pair_overview(scene, camera)

    def view_pairs(self, role="base"):
        """Say what the scene's saved pairs cover and show some frames: every pair in
        its own colour, in the camera image and in the point cloud from above."""
        scene, camera = self._scene(role)
        report = self._pair_overview(scene, camera)
        if not report["pairs"]:
            return
        from .results import spread_frames
        from .saved_pairs import preview

        source = Scene(str(scene), camera)
        self._show_pictures([(f"Check image, frame {frame}: {report['per_frame'][frame]} pairs. A pair "
                              "has one colour in the camera image (left) and seen from above (right); "
                              "gray = no LiDAR box, white = not decided yet.",
                              lambda frame=frame: preview(source, frame))
                             for frame in spread_frames(report["per_frame"], 3)])
        print("To look at any other frame, or to change a pair, run the pairing cell: the window "
              "shows the same pairs frame by frame.")

    def view_sustech(self, role):
        """Open the scene in SUSTechPOINTS, in the web browser: the point cloud of every frame
        with its LiDAR boxes and their ids, and the camera image. The other scene chosen in
        this session is in the page too. A LiDAR box fine-tuned there is saved into the
        scene's labels. The page reaches the scene's files through hard links (a folder on
        another drive than the temporary copy is linked as a junction, which the launcher's
        cleanup removes without following it)."""
        import http.client
        import socket
        import urllib.request

        if role not in ("base", "refine"):
            raise Stop('Use view_sustech("base") or view_sustech("refine").')
        scene, camera = self._scene(role)
        frames = Scene(str(scene), camera).frames()
        if not frames:
            raise Stop(f"{scene} has no point clouds in its lidar folder.")
        shown = [scene] + [other for name, (other, _) in sorted(self.scenes.items())
                           if name != role and other.name != scene.name]
        server = _SUSTECH.get("server")
        if not (server and server["scenes"] == frozenset(shown) and server["process"].poll() is None):
            launcher = next((path for path in (root / "scripts" / "launch_sustech.py"
                                               for root in (self.repo, _CODE_ROOT)) if path.is_file()), None)
            if launcher is None:
                raise Stop("SUSTechPOINTS is opened by scripts/launch_sustech.py of the XCalib "
                           "repository, and that file was not found.")
            python = _python()   # stops before anything starts while the debugger is on
            restarted = server is not None and server["process"].poll() is None
            _stop_sustech()      # one page per notebook: the one showing other scenes ends
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            log = tempfile.NamedTemporaryFile(prefix="xcalib-sustech-", suffix=".log", delete=False)
            command = [python, str(launcher), "--stop-on-stdin-close", "--port", str(port)]
            for folder in shown:
                command += ["--scene", str(folder)]
            with log:
                # In a session of its own elsewhere than on Windows, so that ending it ends the
                # page it starts as well (see stop_process_tree).
                process = subprocess.Popen(command, cwd=launcher.parents[1], stdin=subprocess.PIPE,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=not _WINDOWS)
            server = _SUSTECH["server"] = {"process": process, "scenes": frozenset(shown),
                                           "url": f"http://127.0.0.1:{port}/"}
            if restarted:
                print("The page that was open is closed: this one holds the scenes chosen now, at a "
                      "new address.")
            print("Starting SUSTechPOINTS; this takes about half a minute ...")
            # Straight to the page, never through a proxy that the system may name.
            direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        _SUSTECH.pop("server", None)
                        reason = _last_line(log.name)
                        raise Stop("SUSTechPOINTS could not start" + (f" ({reason})" if reason else "")
                                   + f". Details: {log.name}")
                    try:
                        with direct.open(server["url"], timeout=2):
                            break
                    except (OSError, http.client.HTTPException):
                        if time.monotonic() > deadline:
                            _stop_sustech()
                            raise Stop(f"SUSTechPOINTS did not answer. Details: {log.name}") from None
                        time.sleep(1)
            except KeyboardInterrupt:   # the ■ button while it starts
                _stop_sustech()
                raise Stop("SUSTechPOINTS was stopped before it was ready. Run this cell again to "
                           "open it.") from None
            except Stop:
                raise
            except BaseException:
                _stop_sustech()
                raise
        url = self._sustech_address(server["url"], scene, camera)
        # A short-lived process opens the browser: a browser started from the kernel itself
        # would be ended together with the kernel when the notebook is restarted.
        subprocess.Popen([_python(), "-c", "import sys, webbrowser; webbrowser.open(sys.argv[1])", url],
                         cwd=self.repo, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        print(f"SUSTechPOINTS opens in your web browser at {url} (if no page opens, please copy "
              "this address into the browser).")
        if len(shown) > 1:
            print("The page holds the scenes " + " and ".join(folder.name for folder in shown) + ": to "
                  "switch, choose the scene in the scene list at the top and then a frame in the frame "
                  "list beside it, or open the scene's own address:")
            for folder in shown[1:]:
                other_camera = next(c for r, (s, c) in self.scenes.items() if s == folder)
                print(f"  {folder.name}: {self._sustech_address(server['url'], folder, other_camera)}")
        print("Every frame shows its point cloud with the LiDAR boxes and their ids; with the mouse over "
              "the large view, the left and right arrow keys go to the previous and the next frame "
              "(PageUp and PageDown too). To see the camera image, choose "
              f"{camera} in the Camera menu at the top: it shows no boxes, as this workflow has no "
              "extrinsic.")
        print("To fine-tune a LiDAR box, click its name (or one of its lines): three views of it open "
              "on the left, from above, from the side and from behind. There, drag a border or a "
              "corner (take it just inside the line), or the middle to move the box; a double click on "
              "a border pulls it in to the points inside the box, in the middle every border, so drag "
              "a border out first where the box cuts the object. With the mouse over one of these "
              "views, W/A/S/D and the arrow keys move the box; over the view from above, Q/E turn it. "
              "The page saves the fine-tuned boxes into the scene at once. Please do not delete a box "
              "(Delete, or Ctrl+D over these views) or change its id: the saved pairs name LiDAR boxes "
              "by their ids. Move the mouse off the three views before you change the frame.")
        print("The page stays available while this notebook runs, and running this cell again opens "
              "it again; when the scenes chosen change, it starts again at a new address.")

    @staticmethod
    def _sustech_address(base, scene, camera):
        """The page's address of the scene's first frame."""
        from urllib.parse import urlencode

        return base + "?" + urlencode({"scene": scene.name, "frame": Scene(str(scene), camera).frames()[0]})

    def _pair_overview(self, scene, camera):
        """Print what the scene's camera boxes hold and where in the image the pairs lie."""
        from .pair_homography import MIN_PAIRS, WIDE_BOX
        from .saved_pairs import summarize

        source = Scene(str(scene), camera)
        report = summarize(source)
        print(f"Saved pairs: {report['pairs']} in {report['frames_with_pairs']} of {report['frames']} "
              f"frames, on {report['objects']} different LiDAR objects; {report['frames_ready']} frames "
              "have two or more.")
        print(f"Camera boxes marked \"no LiDAR box\": {report['none']}; without a decision: "
              f"{report['undecided']}.")
        if report["undecided_beside_pairs"]:
            _warn(_undecided_beside_pairs(report["undecided_beside_pairs"], report["frames_with_undecided"],
                                          source.frames()))
        thirds = report["thirds"]
        if thirds and report["pairs"]:
            left_out = [f"{count} have a camera box {why}" for count, why in (
                (report["cut"], "cut by the image edge"),
                (report["wide"], f"wider than {WIDE_BOX:.0%} of the image (objects close to the camera)")) if count]
            print(f"Where the pairs that the homography can use lie in the image: {thirds[0]} left, "
                  f"{thirds[1]} middle, {thirds[2]} right"
                  + (". It leaves out the others, whose box centre is not the object's centre: "
                     + " and ".join(left_out) + "." if left_out else "."))
            empty = [name for name, count in zip(("left", "middle", "right"), thirds)
                     if count < max(2, sum(thirds) // 50)]
            if empty:
                _warn("Few pairs lie " + _parts_of_the_image(empty) + " of the image. The "
                      "homography is only as good as the pairs are spread: add pairs there.")
        if report["pairs"] < MIN_PAIRS:
            _warn(f"The homography needs at least {MIN_PAIRS} saved pairs, spread over the image. "
                  "Run the pairing cell to add pairs.")
        return report

    def _show_pictures(self, pictures):
        """Show ``[(caption, draw)]`` in the notebook; ``draw()`` returns a BGR image."""
        try:
            import cv2
            from IPython.display import Image, display
        except ImportError:
            return
        for caption, draw in pictures:
            try:
                image = draw()
            except ValueError:
                continue
            if image.shape[1] > 1500:
                scale = 1500 / image.shape[1]
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            print(caption)
            display(Image(data=cv2.imencode(".jpg", image)[1].tobytes()))

    def pair_for_homography(self, role="base"):
        """Open the pairing window and say whether the saved pairs are enough for a homography."""
        scene, camera = self._scene(role)
        self._window("xcalib.gui.review", scene, camera, "--notebook-workflow")
        self._pair_overview(scene, camera)

    def homography(self, role=None):
        """Fit the homography to the saved pairs (no extrinsic is used), write it with
        marked images and show them: to the pairs of the scene named, or, when none is
        named, of the refine scene together with the base scene when one is chosen in this
        session. A homography belongs to one camera position and one LiDAR table: the step
        says per scene how many of its pairs agree with the fit, and warns when a scene's
        pairs do not, as when the camera moved between the two recordings."""
        from .pair_homography import export_pair_homography

        if role:
            roles = [role]
        elif not self.scenes:
            raise Stop("No scene is chosen in this session. Please run the cell that imports the refine "
                       "data (step 5) again, then this cell.")
        else:
            roles = [r for r in ("refine", "base") if r in self.scenes]
        scenes = [Scene(str(scene), camera) for scene, camera in map(self._scene, roles)]
        output = self.repo / "runs" / f"homography-{time.strftime('%Y%m%d-%H%M%S')}"
        print("Fitting the homography to the saved pairs ...")
        try:
            report = export_pair_homography(scenes[0], output, more=scenes[1:])
        except ValueError as error:
            hint = ""
            # only when no scene was named and no base scene is chosen: base pairs could help
            if role is None and roles == ["refine"] and "saved pairs" in str(error):
                hint = (" The pairs of the refine scene alone are not enough for it. If you have base data "
                        "of the same camera with saved pairs, please import it in step 1 (and copy its pairs "
                        "in the optional cell \"use pairs you already have\") and run this cell again: its "
                        "pairs are then used as well. Otherwise, please pair objects on the left, in the "
                        "middle and on the right of the image in the refine scene.")
            raise Stop(f"The homography was not written: {error}.{hint}") from None
        fit = report["fitted_to"]
        if len(scenes) > 1:
            print("Pairs used: " + ", ".join(f"{used['pairs']} of the {role} scene"
                                             for role, used in zip(roles, fit["scenes"])) + ".")
            print("Of the pairs the fit can use, " + ", ".join(
                f"{used['agree']} of {used['usable']} of the {role} scene"
                for role, used in zip(roles, fit["scenes"])) + " agree with the homography within "
                f"{fit['inlier_px']:.0f} px.")
            apart = [role for role, used in zip(roles, fit["scenes"])
                     if used["usable"] >= 10 and used["agree"] < 0.5 * used["usable"]]
            if apart:
                _warn("The pairs of the " + " and of the ".join(apart) + " scene do not agree with this "
                      "homography. A homography belongs to one camera position: the camera may have moved "
                      "between the recordings. Please write it for one scene alone, for example "
                      'flow.homography("refine"), or use base and refine data of the same camera position.')
        print("Homography: [position_x, position_y, 1] of the delivered LiDAR table -> pixel "
              "(u, v, 1) of the object's centre." if report["lidar_table"] else
              "Homography: LiDAR position (x, y, 1) in metres -> pixel (u, v, 1) of the object's centre.")
        for row in report["position_to_image"]:
            print("   ", "  ".join(f"{value: .15g}" for value in row))
        if report["position_columns"] != "metres":
            print("position_x, position_y hold longitude and latitude, so these numbers need all "
                  "their digits: copy them from the saved files, not from here.")
        print(f"Fitted to {fit['pairs']} saved pairs ({fit['objects']} objects in {fit['frames']} frames).")
        if fit["cut_by_image_edge"]:
            print(f"{fit['cut_by_image_edge']} of the pairs took no part in the fit: their camera box "
                  "is cut by the image edge, so its centre is not the object's centre.")
        if fit["wide_boxes"]:
            print(f"{fit['wide_boxes']} of the pairs took no part in the fit: their camera box is wider "
                  f"than {fit['wide_box_share']:.0%} of the image (an object close to the camera), and "
                  "the centre of such a box lies beside the object's centre.")
        print(f"The fit used {fit['places']} places: a parked object counts once, not once per frame, "
              f"and the places are taken evenly over the image (at most {fit['places_per_column_at_most']} "
              "from each of its six columns), so that the side with more traffic does not decide alone.")
        # Large vehicles near the camera never agree closely (the centre of their box is not
        # where their LiDAR centre projects), so a third counts as unsupported only when
        # few of its pairs agree at all.
        weak = [name for name, part in zip(("left", "middle", "right"), fit["thirds"])
                if part["pairs"] < 5 or part["agree"] < 0.35 * part["pairs"]]
        if weak:
            hint = ""
            if role is None and roles == ["refine"]:   # no base scene in this session: its pairs could fill the gap
                hint = (" Base data of the same camera position with saved pairs completes them as well: "
                        "run step 1 with that data (and the optional cell \"use pairs you already have\" "
                        "for its pairs), then this cell again, and its pairs are used too. If step 1 ran "
                        "before the first cell was run again, please run step 1 again with the same "
                        "folder: nothing is copied again and nothing is trained.")
            _warn("The homography is not supported " + _parts_of_the_image(weak) + " of the "
                  "image: " + ", ".join(f"{part['agree']} of {part['pairs']} places agree on the {name}"
                                        for name, part in zip(("left", "middle", "right"), fit["thirds"]))
                  + " (a place is one object at one spot of the image: a parked object paired in many "
                  "frames is one place). Add pairs of moving objects there and run this cell again."
                  + hint)
        # counted over the pairs the fit can use: a box cut by the image edge, or a wide box of
        # an object close to the camera, does not hold its object's centre at its own centre
        hits = report["dots_in_paired_camera_boxes"]
        which = ("pairs" if not fit["cut_by_image_edge"] and not fit["wide_boxes"] else
                 "pairs whose camera box is whole" if not fit["wide_boxes"] else
                 "pairs whose camera box is whole and not wide")
        print(f"Check, on the {hits['pairs']} {which}: {fit['inliers']} agree with the homography within "
              f"{fit['inlier_px']:.0f} px (median {fit['median_error_px']:.1f} px), and {hits['inside']} "
              f"({hits['inside'] / max(1, hits['pairs']):.1%}) have the LiDAR object's dot inside the "
              "paired camera box.")
        behind = report["behind_camera_but_in_image"]
        if behind["objects"]:
            print(f"⚠ {behind['objects']} table rows in {behind['frames']} frames are objects behind "
                  "the camera that the homography still puts inside the image. The images below leave "
                  "them out; a script that draws every row should skip rows whose third coordinate "
                  "of H @ [position_x, position_y, 1] is negative.")
        print("Saved:", report["homography_file"])
        if report.get("inputs_file"):
            print("Saved:", report["inputs_file"], "(the values as Python assignments, for a "
                  "projection script that reads the delivered images and table)")
        try:
            import cv2
            from IPython.display import Image, display
        except ImportError:
            display = None
        for index, path in enumerate(report["projection_images"]):
            print("Saved:", path)
            if display is not None and index < 3:
                display(Image(data=cv2.imencode(".jpg", cv2.imread(path))[1].tobytes()))
        print("Each red dot is one LiDAR object, placed by the homography from its position alone; "
              "the green number is its LiDAR id. Every dot should lie on its object.")
