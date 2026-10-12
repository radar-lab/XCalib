"""Model-directory resolution for the continual subsystem.

Locates a scene/camera's trained-model directory under a data root
(``<data_root>/<scene>/calib/xcalib/<camera>_model``) and resolves its live
revision. This module holds only the path/resolution helpers; it does not load
matchers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .refine_model import (
    XCalibRevisionError,
    XCalibRevisionRef,
    resolve_xcalib_revision,
)


@dataclass(frozen=True)
class ModelRef:
    scene: str
    camera: str
    model_root: Path
    revision: XCalibRevisionRef


def _safe_component(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).drive
        or Path(value).is_absolute()
    ):
        raise XCalibRevisionError(f"unsafe {label} path component: {value!r}")
    return value


def model_dir(data_root: str | Path, scene: str, camera: str) -> Path:
    root = Path(data_root).resolve()
    safe_scene = _safe_component(scene, "scene")
    safe_camera = _safe_component(camera, "camera")
    candidate = (
        root / safe_scene / "calib" / "xcalib" / f"{safe_camera}_model"
    ).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise XCalibRevisionError(
            f"model directory escapes data root: {candidate}") from exc
    return candidate


def resolve_model(
    data_root: str | Path,
    scene: str,
    camera: str,
) -> ModelRef | None:
    root = model_dir(data_root, scene, camera)
    revision = resolve_xcalib_revision(root)
    return None if revision is None else ModelRef(scene, camera, root, revision)
