"""Ending the child processes that the notebook's steps start."""

from __future__ import annotations

import os
import subprocess


def stop_process_tree(process, timeout=10.0, own_group=False):
    """End ``process`` (a ``subprocess.Popen``) together with the processes it started, and
    wait for it.

    A long step of the notebook (a training, the model's pairing of a scene, the
    SUSTechPOINTS page) runs in a child process. When its cell is stopped (the ■ button)
    or fails, the child must not go on running unseen. On Windows ``taskkill /T`` ends the
    whole tree. Elsewhere, a child started with ``start_new_session=True`` is ended with its
    whole process group when ``own_group`` says so; any other child is ended alone, and the
    data loader workers of a training notice that their parent is gone and end by themselves.
    """
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            import signal

            try:
                if own_group:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:   # it ended meanwhile
                pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
