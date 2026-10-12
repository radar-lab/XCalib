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

# Every public name is a PEP 562 lazy export (see __getattr__ below). Keeping the
# top level free of eager imports means `import xcalib` — and, crucially, importing
# any `xcalib.continual.*` submodule (which runs THIS __init__ first) — does NOT pull
# in torch or the matcher/engine. That lets a process that must not import torch
# at module load (for example because of native-library load order) consume the
# pure-sqlite continual data layer. torch loads only when a matching/training/
# export name is actually touched, and `ensure_torch()` still fires there for the
# Jetson hint.

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # Core entry point
    "Matcher",
    # Training and dataset helpers (lazy: keep optional deps out of the base import)
    "train",
    "load_dataset",
    # Result dataclasses
    "MatchResult",
    "BuildResult",
    "CalibrationResult",
    # Pair selection (optimal assignment + geometric gating)
    "one_to_one_assign",
    "geometric_gate",
    # Deployed-site workflow: capture lost frames + nudge the matrix
    "SceneRecorder",
    "refine_projection",
    "CalibrationGateConfig",
    "CalibrationMetrics",
    "PhysicalCalibrationCandidate",
    "build_physical_refinement_candidate",
    "evaluate_calibration",
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
    # PEP 562 lazy exports. torch (+ the matcher/engine) and the optional [train] /
    # huggingface deps load only when the corresponding name is first accessed, so
    # the bare `import xcalib` — and every `xcalib.continual.*` submodule import —
    # stays light and torch-free.
    if name in ("Matcher", "MatchResult"):
        from .utils.torch_check import ensure_torch
        ensure_torch()  # Jetson-specific install hint before torch is pulled in
        from .engine.matcher import Matcher, MatchResult
        return {"Matcher": Matcher, "MatchResult": MatchResult}[name]
    if name == "BuildResult":
        from .utils.torch_check import ensure_torch
        ensure_torch()
        from .engine.exporter import BuildResult
        return BuildResult
    if name == "train":
        from .utils.torch_check import ensure_torch
        ensure_torch()
        from .engine.trainer import train
        return train
    if name in ("CalibrationResult", "refine_projection"):
        from .oneshot.calibration import CalibrationResult, refine_projection
        return {"CalibrationResult": CalibrationResult,
                "refine_projection": refine_projection}[name]
    if name in (
        "CalibrationGateConfig",
        "CalibrationMetrics",
        "PhysicalCalibrationCandidate",
        "build_physical_refinement_candidate",
        "evaluate_calibration",
    ):
        from . import calibration_refine
        return getattr(calibration_refine, name)
    if name in ("one_to_one_assign", "geometric_gate"):
        from .assign import geometric_gate, one_to_one_assign
        return {"geometric_gate": geometric_gate,
                "one_to_one_assign": one_to_one_assign}[name]
    if name == "SceneRecorder":
        from .capture import SceneRecorder
        return SceneRecorder
    if name in ("PROTOCOL_VERSION", "CameraIntrinsics", "ProtocolError",
                "ProtocolViolation", "validate_frame_inputs"):
        from . import protocol
        return getattr(protocol, name)
    if name in ("draw_calibration_overlay", "draw_matching_overlay"):
        from .visualization import (draw_calibration_overlay,
                                    draw_matching_overlay)
        return {"draw_calibration_overlay": draw_calibration_overlay,
                "draw_matching_overlay": draw_matching_overlay}[name]
    if name == "load_dataset":
        from .hub.datasets import load_dataset
        return load_dataset
    raise AttributeError(f"module 'xcalib' has no attribute {name!r}")


def __dir__():
    # The lazy names are not module attributes until first use; list them so dir(xcalib),
    # tab completion and help() still show the public API.
    return sorted(set(globals()) | set(__all__))
