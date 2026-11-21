"""Rename a folder (or file) that was just written, waiting while another program holds it.

On Windows a folder cannot be renamed while another program still has one of its
files open, and a virus scanner or the search indexer often reads the thousands of
files a step has just written. The rename is then refused for a while with
PermissionError ("Access is denied"). The steps that write into a hidden staging
folder and give it its final name at the end use :func:`rename_when_free`, which
tries again until the rename succeeds or the time is up.
"""
import time
from pathlib import Path

TRIES, PAUSE = 30, 2.0   # about a minute in all


def rename_when_free(source, target, progress=None):
    """``Path(source).rename(target)``, tried again while it is refused with PermissionError.

    ``progress`` (a callable taking one line of text) hears once that the step is
    waiting. The last PermissionError is raised when the time is up.
    """
    source = Path(source)
    for attempt in range(TRIES):
        try:
            return source.rename(target)
        except PermissionError:
            if attempt == TRIES - 1:
                raise
            if attempt == 0 and progress:
                progress("Another program is still reading the new files; waiting for it ...")
            time.sleep(PAUSE)
