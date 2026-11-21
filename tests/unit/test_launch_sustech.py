"""SUSTechPOINTS runs from a temporary copy of its code and shows scenes kept anywhere through
hard links; a fine-tuned box is saved into the scene, a page for looking saves into copies of
the LiDAR labels, and runtimes left behind are cleaned up."""

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from scripts import launch_sustech
from scripts.launch_sustech import hold, link_directory, remove_stale_runtimes


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """A stand-in for the pinned SUSTechPOINTS checkout, and a temporary folder of our own."""
    folder = tmp_path / "upstream"
    (folder / "data" / "example").mkdir(parents=True)   # its example data stays out of the runtime
    (folder / "main.py").write_text("# fake server")
    (folder / "server.conf").write_text('server.socket_host = "0.0.0.0"\nserver.socket_port = 8081\n')
    (folder / "index.html").write_text("<html><body><script src='/static/js/main.js'></script></body></html>")
    monkeypatch.setattr(launch_sustech, "UPSTREAM", folder)
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return folder


def _scene(root):
    for name in ("lidar", "camera/front", "label", "label_fusion/camera/front"):
        (root / name).mkdir(parents=True)
    (root / "lidar" / "000.pcd").write_text("points")
    (root / "label" / "000.json").write_text('[{"obj_id": "7"}]')
    (root / "label_fusion" / "camera" / "front" / "000.json").write_text('{"objs": []}')
    return root


def test_launcher_links_external_scene_without_copying(tmp_path):
    import shutil

    source = tmp_path / "scenes"
    source.mkdir()
    (source / "original.txt").write_text("original")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    link_directory(source.resolve(), runtime / "data")
    (runtime / "data" / "new.txt").write_text("annotation")
    assert (source / "new.txt").read_text() == "annotation"
    shutil.rmtree(runtime)
    assert (source / "original.txt").read_text() == "original"


def test_launcher_uses_requested_local_port_without_editing_upstream(tmp_path, upstream, monkeypatch):
    original = (upstream / "server.conf").read_text()
    page = (upstream / "index.html").read_text()
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    model = tmp_path / "model.h5"
    model.write_bytes(b"model")
    seen = []

    def launch(cmd, cwd, env):
        config = (cwd / "server.conf").read_text()
        seen.append(config)
        assert (cwd / "data").is_dir() and not (cwd / "data" / "example").exists()
        # the guarded entry point runs the upstream page, not the upstream main.py itself
        assert cmd[-1] == "sustech_server.py" and (cwd / "sustech_server.py").is_file()
        # The arrow keys change frames: the page of the runtime loads the added script last.
        assert "ArrowLeft" in (cwd / "public" / "js" / "xcalib_frame_keys.js").read_text()
        assert (cwd / "index.html").read_text().endswith(
            '<script src="/static/js/xcalib_frame_keys.js"></script>\n</body></html>')
        assert (cwd / "algos" / "models" / "deep_annotation_inference.h5").read_bytes() == b"model"
        return 0

    monkeypatch.setattr(launch_sustech.subprocess, "call", launch)
    assert launch_sustech.main(["--data-root", str(scenes), "--model", str(model),
                                "--python", sys.executable, "--port", "18081"]) == 0
    assert 'server.socket_host = "127.0.0.1"' in seen[0]
    assert "server.socket_port = 18081" in seen[0]
    assert (upstream / "server.conf").read_text() == original
    assert (upstream / "index.html").read_text() == page and not (upstream / "public").exists()
    assert not list(Path(tempfile.gettempdir()).iterdir())   # the runtime is removed again

    # --scene exposes that scene alone, and the guarded server saves into its own labels.
    (scenes / "site_base" / "label").mkdir(parents=True)
    (scenes / "notes.txt").write_text("not a scene")

    def annotate(cmd, cwd, env):
        assert [p.name for p in (cwd / "data").iterdir()] == ["site_base"]
        listed = json.loads((cwd / "xcalib_scenes.json").read_text(encoding="utf-8"))
        assert listed == {"site_base": {"labels": str(scenes / "site_base" / "label"), "save": True}}
        return 0

    monkeypatch.setattr(launch_sustech.subprocess, "call", annotate)
    assert launch_sustech.main(["--scene", str(scenes / "site_base"), "--model", str(model),
                                "--python", sys.executable]) == 0

    # Repeating --scene shows scenes from different folders side by side.
    (scenes / "refine" / "site_refine").mkdir(parents=True)

    def both(cmd, cwd, env):
        assert sorted(p.name for p in (cwd / "data").iterdir()) == ["site_base", "site_refine"]
        return 0

    monkeypatch.setattr(launch_sustech.subprocess, "call", both)
    options = ["--model", str(model), "--python", sys.executable]
    assert launch_sustech.main(["--scene", str(scenes / "site_base"),
                                "--scene", str(scenes / "refine" / "site_refine"), *options]) == 0
    with pytest.raises(SystemExit):  # two scenes cannot share one name in the page
        launch_sustech.main(["--scene", str(scenes / "site_base"),
                             "--scene", str(scenes / "site_base"), *options])


def test_a_fine_tuned_box_is_saved_into_the_scene_and_the_page_shows_it(tmp_path, upstream, monkeypatch):
    from scripts.sustech_server import checked_saves, label_folders, write_save

    base = _scene(tmp_path / "site" / "site_base")
    model = tmp_path / "model.h5"
    model.write_bytes(b"model")

    def edit(cmd, cwd, env):
        shown = cwd / "data" / "site_base"
        # every file of the page's copy is a hard link: deleting the copy cannot reach the scene
        assert os.path.samefile(shown / "label" / "000.json", base / "label" / "000.json")
        assert not getattr(os.path, "isjunction", lambda path: False)(shown / "label")
        # the server's save: an existing frame and a frame that had no label file yet
        folders = label_folders(str(cwd / "xcalib_scenes.json"))
        frames = {"site_base": {"000", "001"}}
        saves = checked_saves([{"scene": "site_base", "frame": "000", "annotation": [{"obj_id": "7", "x": 1}]},
                               {"scene": "site_base", "frame": "001", "annotation": []}],
                              {"site_base"}, frames.get, folders)
        assert [Path(path).parent for path, _ in saves] == [base / "label"] * 2
        for (path, annotation), frame in zip(saves, ("000", "001")):
            write_save(path, annotation, str(shown / "label" / f"{frame}.json"))
        assert json.loads((shown / "label" / "000.json").read_text()) == [{"obj_id": "7", "x": 1}]   # the page sees it
        assert (shown / "label" / "001.json").exists()
        return 0

    monkeypatch.setattr(launch_sustech.subprocess, "call", edit)
    assert launch_sustech.main(["--scene", str(base), "--model", str(model), "--python", sys.executable]) == 0
    assert json.loads((base / "label" / "000.json").read_text()) == [{"obj_id": "7", "x": 1}]   # saved in the scene
    assert json.loads((base / "label" / "001.json").read_text()) == []
    assert (base / "lidar" / "000.pcd").read_text() == "points"   # removing the page's copy left the scene


def test_the_page_keeps_reading_the_saved_file_when_no_hard_link_can_be_made(tmp_path, monkeypatch):
    import scripts.sustech_server as server

    scene_label, page_label = tmp_path / "scene" / "label", tmp_path / "page" / "label"
    scene_label.mkdir(parents=True)
    page_label.mkdir(parents=True)

    def no_link(source, target):   # a scene on another drive than the page's copy
        raise OSError("cross-device link")

    monkeypatch.setattr(server.os, "link", no_link)
    path, shown = str(scene_label / "005.json"), str(page_label / "005.json")
    for version in ([{"obj_id": "7", "x": 1}], [{"obj_id": "7", "x": 2}], [{"obj_id": "7", "x": 3}]):
        server.write_save(path, version, shown)
        assert json.loads(Path(path).read_text()) == version
        assert json.loads(Path(shown).read_text()) == version   # a reload shows the last save
    # a page copy that parted from the scene file (the file was replaced) is replaced by name only
    (page_label / "006.json").write_text("[]")
    server.write_save(str(scene_label / "006.json"), [{"obj_id": "8"}], str(page_label / "006.json"))
    assert json.loads((page_label / "006.json").read_text()) == [{"obj_id": "8"}]
    assert json.loads((scene_label / "006.json").read_text()) == [{"obj_id": "8"}]


def test_a_page_for_looking_saves_into_copies_of_the_lidar_labels(tmp_path, upstream, monkeypatch):
    base = _scene(tmp_path / "site" / "site_base")
    refine = _scene(tmp_path / "site" / "refine" / "site_refine")
    model = tmp_path / "model.h5"
    model.write_bytes(b"model")
    seen = []

    def look(cmd, cwd, env):
        shown = cwd / "data" / "site_base"
        assert sorted(p.name for p in (cwd / "data").iterdir()) == ["site_base", "site_refine"]
        # the point clouds, images and camera labels are the scene's own files, through hard
        # links in folders of the runtime: deleting the runtime with any tool cannot reach them
        assert (shown / "lidar" / "000.pcd").read_text() == "points"
        assert os.path.samefile(shown / "lidar" / "000.pcd", base / "lidar" / "000.pcd")
        assert not (shown / "lidar").is_symlink()
        assert not getattr(os.path, "isjunction", lambda path: False)(shown / "lidar")
        assert os.path.samefile(shown / "label_fusion" / "camera" / "front" / "000.json",
                                base / "label_fusion" / "camera" / "front" / "000.json")
        assert json.loads((cwd / "xcalib_scenes.json").read_text(encoding="utf-8")) == {}   # no scene saves
        assert not os.path.samefile(shown / "label" / "000.json", base / "label" / "000.json")
        (shown / "label" / "000.json").write_text("[]")          # the page saves a change
        (shown / "label" / "001.json").write_text("[]")
        seen.append(cmd)
        return 0

    monkeypatch.setattr(launch_sustech.subprocess, "call", look)
    assert launch_sustech.main(["--scene", str(base), "--scene", str(refine), "--view-only",
                                "--model", str(model), "--python", sys.executable]) == 0
    assert seen and (base / "label" / "000.json").read_text() == '[{"obj_id": "7"}]'
    assert not (base / "label" / "001.json").exists()
    assert (base / "lidar" / "000.pcd").read_text() == "points"   # removing the runtime left the scene
    with pytest.raises(SystemExit):   # a page for looking needs the scenes it shows
        launch_sustech.main(["--view-only", "--model", str(model), "--python", sys.executable])


def test_a_missing_sustechpoints_or_its_python_is_named(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch_sustech, "UPSTREAM", tmp_path / "nowhere")
    scene = str(_scene(tmp_path / "site_base"))
    with pytest.raises(SystemExit):
        launch_sustech.main(["--scene", scene])
    said = capsys.readouterr().err
    assert "SUSTechPOINTS was not found" in said and "datasets/SUSTechPOINTS" in said
    assert "docs/review-gui.md" in said and "submodule update" not in said

    # without its own environment and without CherryPy here, the Python is named
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "main.py").write_text("# fake server")
    monkeypatch.setattr(launch_sustech, "UPSTREAM", upstream)
    monkeypatch.setattr(launch_sustech, "ROOT", tmp_path)   # no datasets/sustech-venv there
    monkeypatch.setattr(launch_sustech.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit):
        launch_sustech.main(["--scene", scene])
    assert "the Python of SUSTechPOINTS was not found" in capsys.readouterr().err


def test_the_submodule_comes_first_and_the_local_copy_next(tmp_path, monkeypatch):
    first, second = tmp_path / "third_party", tmp_path / "datasets"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(launch_sustech, "UPSTREAM", None)
    monkeypatch.setattr(launch_sustech, "UPSTREAM_CANDIDATES", (first, second))
    assert launch_sustech.default_upstream() == first           # neither has it: the first is named
    (second / "main.py").write_text("# copy")
    assert launch_sustech.default_upstream() == second
    (first / "main.py").write_text("# submodule")
    assert launch_sustech.default_upstream() == first


def test_runtimes_left_behind_are_removed_and_running_ones_kept(tmp_path):
    scene = _scene(tmp_path / "site_base")
    temp = tmp_path / "temp"

    def runtime(name, lock=True):
        folder = temp / name
        (folder / "runtime" / "data").mkdir(parents=True)
        link_directory(scene.resolve(), folder / "runtime" / "data" / "site_base")
        if lock:
            (folder / launch_sustech.LOCK).write_text("")
        return folder

    stale = runtime("xcalib-sustech-stale")
    older = runtime("xcalib-sustech-older", lock=False)   # made before runtimes had a lock file
    running = runtime("xcalib-sustech-running")
    held = hold(running / launch_sustech.LOCK)
    other = temp / "xcalib-sustech-other"                    # not a runtime: no lock, no runtime folder
    other.mkdir()
    starting = temp / "xcalib-sustech-starting"              # its launcher took the lock, nothing else yet
    starting.mkdir()
    (starting / launch_sustech.LOCK).write_text("")
    log = temp / "xcalib-sustech-page.log"
    log.write_text("log")
    try:
        removed = remove_stale_runtimes(temp)
    finally:
        held.close()
    assert sorted(removed) == sorted([stale, older])
    assert not stale.exists() and not older.exists()
    assert not list(temp.glob("*.removing"))
    assert running.exists() and other.exists() and starting.exists() and log.exists()
    # the scene linked into the removed runtimes is untouched
    assert (scene / "lidar" / "000.pcd").read_text() == "points"
    assert (scene / "label" / "000.json").exists()
    assert remove_stale_runtimes(temp) == [running]          # its launcher has ended now


def test_the_page_stops_when_the_program_that_started_it_closes_its_input(monkeypatch):
    monkeypatch.setattr(launch_sustech.sys, "stdin", io.StringIO(""))   # closed at once
    begun = time.monotonic()
    code = launch_sustech.serve([sys.executable, "-c", "import time; time.sleep(60)"], Path.cwd(),
                                os.environ.copy(), stop_on_stdin_close=True)
    assert code != 0 and time.monotonic() - begun < 30


def test_the_page_saves_only_labels_of_its_own_frames_and_answers_only_its_own_address():
    from scripts.sustech_server import allowed, checked_saves

    frames = {"site_base": {"000", "001"}}
    saves = checked_saves([{"scene": "site_base", "frame": "001", "annotation": []}], {"site_base"},
                          frames.get)
    assert saves == [(os.path.join("data", "site_base", "label", "001.json"), [])]
    for entry in ({"scene": "site_base", "frame": "../calib/camera/front"},   # beside the labels
                  {"scene": "site_base", "frame": "002"},                     # a frame it does not have
                  {"scene": "..", "frame": "000"},
                  {"scene": "other", "frame": "000"}):
        with pytest.raises(ValueError):
            checked_saves([{**entry, "annotation": []}], {"site_base"}, frames.get)
    with pytest.raises(ValueError):
        checked_saves({"scene": "site_base"}, {"site_base"}, frames.get)

    assert allowed("127.0.0.1:18081", None, "GET", 18081)
    assert allowed("localhost:18081", "http://localhost:18081", "POST", 18081)
    assert not allowed("evil.example:18081", None, "GET", 18081)          # DNS rebinding
    assert not allowed("127.0.0.1:18081", "https://evil.example", "POST", 18081)   # another page
    assert allowed("127.0.0.1:18081", "https://evil.example", "GET", 18081)        # reading needs CORS
    assert not allowed("127.0.0.1:18081", "https://evil.example", "PUT", 18081)    # nor any other write
    assert not allowed(None, None, "GET", 18081)


def test_saves_go_to_the_label_folders_the_launcher_names(tmp_path):
    from scripts.sustech_server import checked_saves, label_folders

    listing = tmp_path / "xcalib_scenes.json"
    listing.write_text(json.dumps({"site_base": {"labels": str(tmp_path / "scene" / "label"), "save": True},
                                   "site_refine": {"labels": str(tmp_path / "other"), "save": False}}))
    folders = label_folders(str(listing))
    assert folders == {"site_base": str(tmp_path / "scene" / "label")}
    frames = {"site_base": {"000"}, "site_refine": {"000"}}
    saves = checked_saves([{"scene": "site_base", "frame": "000", "annotation": []},
                           {"scene": "site_refine", "frame": "000", "annotation": []}],
                          {"site_base", "site_refine"}, frames.get, folders)
    assert saves[0][0] == os.path.join(str(tmp_path / "scene" / "label"), "000.json")
    assert saves[1][0] == os.path.join("data", "site_refine", "label", "000.json")   # the page's copy
    assert label_folders(str(tmp_path / "missing.json")) == {}
