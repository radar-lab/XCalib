"""xcalib — camera-LiDAR cross-modal matching, calibration, and adaptation.

Public API (everything else is internal and may change without notice):

    from xcalib import Matcher

    matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
    result = matcher.pair(image, point_cloud, bboxes_2d, bboxes_3d)
    calib = matcher.calibrate(image, point_cloud, bboxes_2d, bboxes_3d,
                              intrinsics=K)
    matcher.build("onnx", output_dir="onnx/crlite_a9")

`Matcher` covers pairing/matching, targetless calibration, one-shot
adaptation, ONNX/TensorRT export, fine-tuning (`.train()`), and pretrained
weight loading. The module-level `train` / `load_dataset` helpers cover HDF5
training and public A9 dataset loading; the protocol types document and
enforce the input contract.
"""

# Verify torch is importable before any sub-module pulls it in, so the
# partner sees a Jetson-specific install hint instead of a bare
# ModuleNotFoundError. No-op when torch is already installed.
from .utils.torch_check import ensure_torch as _ensure_torch
_ensure_torch()

from .engine.exporter import BuildResult
from .engine.matcher import Matcher, MatchResult
from .oneshot.calibration import CalibrationResult
from .protocol import (
    PROTOCOL_VERSION,
    CameraIntrinsics,
    ProtocolError,
    ProtocolViolation,
    validate_frame_inputs,
)
from .visualization import draw_calibration_overlay, draw_matching_overlay

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core entry point
    "Matcher",
    # Lab-side helpers (lazy: keep optional deps out of the base import)
    "train",
    "load_dataset",
    # Result dataclasses
    "MatchResult",
    "BuildResult",
    "CalibrationResult",
    # Input protocol (partner-facing contract)
    "PROTOCOL_VERSION",
    "CameraIntrinsics",
    "ProtocolError",
    "ProtocolViolation",
    "validate_frame_inputs",
    # Visualization helpers
    "draw_matching_overlay",
    "draw_calibration_overlay",
]


def __getattr__(name: str):
    # PEP 562 lazy exports: `train` pulls h5py (the [train] extra) and
    # `load_dataset` pulls huggingface_hub network helpers — neither should
    # tax `import xcalib` for inference-only users.
    if name == "train":
        from .engine.trainer import train

        return train
    if name == "load_dataset":
        from .hub.datasets import load_dataset

        return load_dataset
    raise AttributeError(f"module 'xcalib' has no attribute {name!r}")
