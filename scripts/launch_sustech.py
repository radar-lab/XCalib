"""Launch the pinned SUSTechPOINTS on scene directories outside its own folder.

The page runs from a temporary copy of the upstream code (without its example data), so the
upstream checkout stays as it is, and through scripts/sustech_server.py, which only lets it
save into the labels of the frames it shows. A --scene is shown through hard links to its
files, so deleting the copy never deletes scene files; the page saves its edited LiDAR boxes
into the scene's own label files, or, with --view-only, into a copy of them that is dropped
with the page.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

if sys.version_info < (3, 8):
    # Before 3.8, shutil.rmtree follows directory junctions on Windows: removing the runtime
    # would delete the files of the scenes linked into it.
    raise SystemExit("launch_sustech.py needs Python 3.8 or later")

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(__file__).with_name("sustech_server.py")   # the guarded entry point
DOCS = "the section 'Look at a scene in SUSTechPOINTS' of docs/review-gui.md"
# The upstream code: the submodule when it is checked out, else a local copy kept beside the
# interpreter and the model in the ignored datasets/ folder.
UPSTREAM_COMMIT = "50fa18871a6d46507ff9057175669235885f79f9"
UPSTREAM_CANDIDATES = (ROOT / "third_party" / "SUSTechPOINTS", ROOT / "datasets" / "SUSTechPOINTS")
UPSTREAM = None   # a fixed checkout instead of the candidates (tests set it)
RUNTIME_PREFIX = "xcalib-sustech-"
LOCK = "in_use"   # held open by the launcher that uses the runtime folder

# Added to the page of the temporary runtime: the Left/Right arrow keys act as the page's
# own PageUp/PageDown (previous/next frame). Typing in a field or a list keeps its arrows,
# and the side views, which move the selected box with them, never pass them on.
FRAME_KEYS = """\
document.addEventListener("keydown", (event) => {
    const key = {ArrowLeft: "PageUp", ArrowRight: "PageDown"}[event.key];
    if (!key || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return;
    const target = event.target;
    if (target.closest && target.closest("input, select, textarea, [contenteditable]")) return;
    event.preventDefault();
    document.dispatchEvent(new KeyboardEvent("keydown", {key: key}));
});
"""


def link_directory(source: Path, target: Path) -> None:
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(source), str(target))
    else:
        target.symlink_to(source, target_is_directory=True)


def default_upstream() -> Path:
    if UPSTREAM is not None:
        return Path(UPSTREAM)
    return next((folder for folder in UPSTREAM_CANDIDATES if (folder / "main.py").is_file()),
                UPSTREAM_CANDIDATES[0])


def mirror(source: Path, target: Path) -> None:
    """Show the folder ``source`` at ``target`` through hard links to its files, so that
    deleting ``target`` with any tool never deletes a file of ``source``. Where hard links
    cannot be made (another volume, a file system without them), the folder is linked as a
    whole instead."""
    try:
        for folder, _, files in os.walk(source):
            here = target / Path(folder).relative_to(source)
            here.mkdir(exist_ok=True)
            for name in files:
                os.link(Path(folder) / name, here / name)
    except OSError:
        shutil.rmtree(target, ignore_errors=True)   # only links and folders made here
        link_directory(source.resolve(), target)


def view_of(scene: Path, target: Path, *, keep: bool = True) -> None:
    """Show ``scene`` at ``target``: its folders are mirrored through hard links, so deleting
    ``target`` with any tool never deletes a scene file; files at the top of the scene are
    copied. With ``keep`` (a page for looking) the LiDAR labels are a copy, so what the page
    saves lands in the copy. Without it they are mirrored too, and the guarded server writes
    the page's saves into the scene's own label files."""
    target.mkdir()
    for entry in scene.iterdir():
        if entry.name == "label" and keep:
            continue
        if entry.is_dir():
            mirror(entry, target / entry.name)
        else:
            shutil.copy2(entry, target / entry.name)
    if keep and (scene / "label").is_dir():
        shutil.copytree(scene / "label", target / "label")
    elif not (target / "label").exists():
        (target / "label").mkdir()


def hold(lock: Path):
    """Open the lock file of a runtime folder and keep it open while the page runs."""
    handle = lock.open("w")
    if os.name != "nt":
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def in_use(lock: Path) -> bool:
    """Whether the launcher that made this lock file still runs."""
    if os.name == "nt":
        try:
            lock.unlink()   # Windows refuses to delete a file that another handle holds open
        except FileNotFoundError:
            return False
        except OSError:
            return True
        return False
    import fcntl
    try:
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    return False


def remove_stale_runtimes(temp_root: Path) -> list[Path]:
    """Remove the runtime folders of launchers that ended without cleaning up (a launcher
    that was killed with its notebook, for example) and return the folders removed.

    A folder stays when its lock file is still held (its page runs), when it has no
    runtime folder yet (a launcher is just setting it up: it takes the lock first), and,
    on Windows, when it cannot be renamed because a program still has files open in it.
    The scenes linked into a runtime are not touched: their files are hard links, and
    shutil.rmtree removes a junction without following it."""
    removed = []
    for folder in sorted(temp_root.glob(RUNTIME_PREFIX + "*")):
        try:
            if not folder.is_dir() or folder.is_symlink() or getattr(os.path, "isjunction",
                                                                      lambda path: False)(folder):
                continue
            if not (folder / "runtime").is_dir():
                continue   # being set up, or not a runtime folder of this launcher
            lock = folder / LOCK
            if lock.exists() and in_use(lock):
                continue
            doomed = folder
            if not folder.name.endswith(".removing"):
                doomed = folder.with_name(folder.name + ".removing")
                folder.rename(doomed)   # refused while a program has files open inside
            shutil.rmtree(doomed, ignore_errors=True)
            if not doomed.exists():
                removed.append(folder)
        except OSError:
            continue
    return removed


def serve(command, cwd: Path, env, stop_on_stdin_close: bool) -> int:
    """Run the page until it ends; with ``stop_on_stdin_close`` also until the standard
    input of this launcher is closed (by the program that started it, or by its end)."""
    if not stop_on_stdin_close:
        return subprocess.call(command, cwd=cwd, env=env)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL)

    def stop_at_end_of_input():
        try:
            sys.stdin.read()
        except (OSError, ValueError):
            pass
        if process.poll() is None:
            process.terminate()

    threading.Thread(target=stop_at_end_of_input, daemon=True).start()
    try:
        return process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    local_python = ROOT / "datasets" / "sustech-venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets" / "sustech_data",
                        help="directory containing SUSTechPOINTS scene directories")
    parser.add_argument("--scene", type=Path, action="append",
                        help="a scene directory to show instead of --data-root (the page saves its "
                             "edited LiDAR boxes into the scene's label files); repeat it to show "
                             "several scenes that do not share a folder")
    parser.add_argument("--view-only", action="store_true",
                        help="keep the --scene directories as they are: the page saves into copies "
                             "of their LiDAR labels")
    parser.add_argument("--upstream", type=Path,
                        help="SUSTechPOINTS checkout (default: third_party/SUSTechPOINTS, else "
                             "datasets/SUSTechPOINTS)")
    parser.add_argument("--model", type=Path, default=ROOT / "datasets" / "deep_annotation_inference.h5",
                        help="upstream deep_annotation_inference.h5")
    parser.add_argument("--python", type=Path,
                        help="Python interpreter with upstream dependencies (default: "
                             "datasets/sustech-venv, else this Python)")
    parser.add_argument("--port", type=int, default=8081,
                        help="localhost port for the SUSTechPOINTS server (default: 8081)")
    parser.add_argument("--stop-on-stdin-close", action="store_true",
                        help="also stop when standard input is closed, for a program that starts "
                             "this launcher and may end without stopping it")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.view_only and not args.scene:
        parser.error("--view-only needs --scene")
    upstream = (args.upstream or default_upstream()).resolve()
    if not (upstream / "main.py").is_file():
        parser.error(f"SUSTechPOINTS was not found in {upstream}: it needs the SUSTechPOINTS code of "
                     f"commit {UPSTREAM_COMMIT[:7]} in third_party/SUSTechPOINTS or "
                     f"datasets/SUSTechPOINTS, see {DOCS}")
    if args.python is None:
        if local_python.is_file():
            args.python = local_python
        elif importlib.util.find_spec("cherrypy") is None:
            parser.error(f"the Python of SUSTechPOINTS was not found: {local_python.parent.parent} "
                         f"does not exist and this Python has no CherryPy, see {DOCS}")
        else:
            args.python = Path(sys.executable)
    if not SERVER.is_file():
        parser.error(f"{SERVER.name} is missing beside this launcher")
    try:
        scenes = [scene.resolve(strict=True) for scene in args.scene or []]
        data = scenes or [args.data_root.resolve(strict=True)]
        model = args.model.resolve(strict=True)
        python = args.python.resolve(strict=True)
    except OSError as error:
        parser.error(f"not found: {error.filename}")
    if not all(folder.is_dir() for folder in data) or not model.is_file():
        parser.error("data-root and scenes must be directories and model must be a file")
    if len({scene.name for scene in scenes}) != len(scenes):
        parser.error("scenes shown together need different folder names")
    remove_stale_runtimes(Path(tempfile.gettempdir()))
    directory = Path(tempfile.mkdtemp(prefix=RUNTIME_PREFIX))
    lock = hold(directory / LOCK)
    try:
        runtime = directory / "runtime"
        shutil.copytree(upstream, runtime,
                        ignore=shutil.ignore_patterns(".git", "data", "doc", "__pycache__"))
        if scenes:
            # The page lists every entry of its data folder as a scene.
            (runtime / "data").mkdir()
            saved_into = {}
            for scene in scenes:
                view_of(scene, runtime / "data" / scene.name, keep=args.view_only)
                if not args.view_only:   # the page's saves go into the scene's own label files
                    (scene / "label").mkdir(exist_ok=True)
                    saved_into[scene.name] = {"labels": str(scene / "label"), "save": True}
            (runtime / "xcalib_scenes.json").write_text(json.dumps(saved_into, indent=1), encoding="utf-8")
        else:
            link_directory(data[0], runtime / "data")
        model_path = runtime / "algos" / "models" / "deep_annotation_inference.h5"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model, model_path)
        config = runtime / "server.conf"
        contents = config.read_text(encoding="utf-8")
        if 'server.socket_host = "0.0.0.0"' not in contents or "server.socket_port = 8081" not in contents:
            raise ValueError("upstream server.conf has an unexpected host or port setting")
        contents = contents.replace('server.socket_host = "0.0.0.0"', 'server.socket_host = "127.0.0.1"')
        contents = contents.replace("server.socket_port = 8081", f"server.socket_port = {args.port}")
        config.write_text(contents, encoding="utf-8")
        page = runtime / "index.html"
        contents = page.read_text(encoding="utf-8")
        if contents.count("</body>") != 1:
            raise ValueError("upstream index.html has an unexpected layout")
        (runtime / "public" / "js").mkdir(parents=True, exist_ok=True)
        (runtime / "public" / "js" / "xcalib_frame_keys.js").write_text(FRAME_KEYS, encoding="utf-8")
        page.write_text(contents.replace(
            "</body>", '<script src="/static/js/xcalib_frame_keys.js"></script>\n</body>'),
            encoding="utf-8")
        shutil.copyfile(SERVER, runtime / SERVER.name)   # runs the upstream main.py, guarded
        print(f"SUSTechPOINTS: http://127.0.0.1:{args.port}", flush=True)
        env = os.environ.copy()
        env.setdefault("TF_USE_LEGACY_KERAS", "1")
        return serve([str(python), SERVER.name], runtime, env, args.stop_on_stdin_close)
    finally:
        lock.close()
        # The links to the scenes are removed, never followed (shutil.rmtree, Python 3.8+).
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
