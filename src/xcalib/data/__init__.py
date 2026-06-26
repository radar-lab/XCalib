"""Data utilities: cropping helpers and read-only HDF5 frame loader.

`UTCFrameLoader` / `UTCFrame` are imported lazily so that the core wheel
install does not require h5py — only the HDF5 training / validation
scripts need it (``pip install "XCalib[train]"``).
"""

from typing import TYPE_CHECKING

from .crops import (
    PrepareConfig,
    crop_image_bbox,
    crop_point_cloud_axis_aligned,
    resample_point_cloud,
    prepare_frame,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .hdf5_loader import UTCFrame, UTCFrameLoader

__all__ = [
    "PrepareConfig",
    "crop_image_bbox",
    "crop_point_cloud_axis_aligned",
    "resample_point_cloud",
    "prepare_frame",
    "UTCFrameLoader",
    "UTCFrame",
]


def __getattr__(name: str):
    if name in ("UTCFrameLoader", "UTCFrame"):
        try:
            from . import hdf5_loader
        except ImportError as exc:  # pragma: no cover - environment specific
            raise ImportError(
                f"{name} requires h5py. Install it with "
                "`pip install 'XCalib[train]'` (or `pip install h5py`)."
            ) from exc
        return getattr(hdf5_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
