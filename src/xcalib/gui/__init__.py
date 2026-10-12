"""
Standalone review GUI (``xcalib-review``) + SUSTechPOINTS-format scene IO.

:mod:`xcalib.gui.scene` (Scene, match_review, ...) is import-safe everywhere —
it has no Qt dependency and is what scripts should use for headless scene IO.
The Qt window itself lives in :mod:`xcalib.gui.review` and needs the ``[gui]``
extra (PySide6):

    pip install "xcalib[gui]"
    xcalib-review --scene /data/MySite --camera cam_front
"""

from .scene import FrameReview, Scene, TrackEntry, match_review

__all__ = ["Scene", "FrameReview", "TrackEntry", "match_review"]
