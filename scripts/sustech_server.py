"""Start the pinned SUSTechPOINTS (the main.py beside this file) with XCalib's guards.

scripts/launch_sustech.py copies this file into the temporary runtime and runs it instead
of the upstream main.py:

- a save is written only into the label folder of a scene the page shows, and only for a
  frame that scene has; a crafted request (a frame such as "../calib/camera/front") is
  refused instead of writing beside the labels;
- a scene the launcher lists in xcalib_scenes.json with "save": true gets its saves written
  in place into its own label files, and the page's copy is kept on the same files (hard
  links, linked or copied anew when they part); a --scene shown with --view-only keeps its
  saves in the page's copy (with --data-root the data folder itself is linked, so saves land
  in its scenes);
- saves are POST requests only; the scene-cropping endpoint, which runs a shell command, is
  switched off;
- the page answers only at its own address (no DNS rebinding), and refuses requests other
  than reads that other web pages send;
- CherryPy's autoreload is off, so the page never restarts itself.

The checks need no CherryPy and are tested on their own; the server part runs only in the
runtime, with the SUSTechPOINTS interpreter.
"""

from __future__ import annotations

import json
import os
import shutil

SCENES_FILE = "xcalib_scenes.json"   # written by the launcher: where each shown scene saves


def hosts_of(port) -> set[str]:
    """The Host headers of requests to the page's own address."""
    return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}


def allowed(host, origin, method, port) -> bool:
    """Whether a request may reach the page: it is addressed to the page itself, and anything
    but a plain read comes from the page (browsers name the sending page in Origin)."""
    hosts = hosts_of(port)
    if (host or "").strip().lower() not in hosts:
        return False
    if (method.upper() not in ("GET", "HEAD") and origin
            and origin.strip().lower() not in {f"http://{h}" for h in hosts}):
        return False
    return True


def label_folders(scenes_file=SCENES_FILE):
    """{scene: the folder its saves go to}: the scene's own label folder when the launcher
    lets the page save into it, else the label folder of the page's copy."""
    try:
        with open(scenes_file, encoding="utf-8") as stream:
            listed = json.load(stream)
    except (OSError, ValueError):
        listed = {}
    return {name: entry["labels"] for name, entry in listed.items() if entry.get("save") and entry.get("labels")}


def checked_saves(entries, scenes, frames_of, folders=None):
    """The files a save request writes, as [(path, annotation)], or ValueError when an
    entry names a scene the page does not show or a frame that scene does not have. A scene
    in ``folders`` saves into that folder, any other into the page's copy."""
    if not isinstance(entries, list):
        raise ValueError("a save is a list of frames")
    saves = []
    for entry in entries:
        scene, frame = entry.get("scene"), entry.get("frame")
        if scene not in scenes or frame not in frames_of(scene):
            raise ValueError(f"no frame {frame!r} in a shown scene {scene!r}")
        folder = (folders or {}).get(scene) or os.path.join("data", scene, "label")
        saves.append((os.path.join(folder, f"{frame}.json"), entry.get("annotation")))
    return saves


def write_save(path, annotation, shown):
    """Write one frame's boxes in place (a hard link to the file sees them at once), and make
    sure the page's copy (``shown``) reads the same file: when it has none for this frame, or
    holds a separate copy (no hard link could be made, or the scene's file was replaced), its
    name is linked, or copied, anew. Only the page's name is removed, never a scene file."""
    with open(path, "w") as stream:
        json.dump(annotation, stream, indent=2, sort_keys=True)
    if os.path.abspath(shown) == os.path.abspath(path):
        return
    if os.path.exists(shown) and os.path.samefile(path, shown):
        return   # a hard link, or a folder reached through a junction: the page sees the save
    if os.path.lexists(shown):
        os.remove(shown)   # a stale copy: copying over it could write through an old hard link
    try:
        os.link(path, shown)
    except OSError:
        shutil.copyfile(path, shown)


def serve():
    import functools

    import cherrypy

    import main as upstream   # the upstream module defines the page; it starts nothing on import
    import scene_reader

    folders = label_folders()

    @functools.lru_cache(maxsize=None)
    def frames_of(scene):
        return frozenset(os.path.splitext(name)[0] for name in os.listdir(os.path.join("data", scene, "lidar")))

    class Guarded(upstream.Root):
        @cherrypy.expose
        def saveworldlist(self):
            if cherrypy.request.method != "POST":
                raise cherrypy.HTTPError(405, "saves are POST requests")
            entries = json.loads(cherrypy.request.body.readline().decode("UTF-8"))
            try:
                saves = checked_saves(entries, set(scene_reader.get_scene_names()), frames_of, folders)
            except (ValueError, AttributeError, TypeError) as error:
                raise cherrypy.HTTPError(403, str(error))
            for (path, annotation), entry in zip(saves, entries):
                write_save(path, annotation, os.path.join("data", entry["scene"], "label", f"{entry['frame']}.json"))
            return "ok"

        @cherrypy.expose
        def cropscene(self, *args, **kwargs):
            raise cherrypy.HTTPError(403, "cropping scenes is switched off in this page")

    cherrypy.config.update("server.conf")
    cherrypy.config.update({"engine.autoreload.on": False})
    port = cherrypy.config.get("server.socket_port")

    def own_address_only():
        request = cherrypy.request
        if not allowed(request.headers.get("Host"), request.headers.get("Origin"), request.method, port):
            raise cherrypy.HTTPError(403, "This page answers only to its own address.")

    cherrypy.tools.xcalib_own_address = cherrypy.Tool("on_start_resource", own_address_only)
    application = cherrypy.tree.mount(Guarded(), "/", "server.conf")
    application.merge({"/": {"tools.xcalib_own_address.on": True}})
    cherrypy.engine.signals.subscribe()
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    serve()
