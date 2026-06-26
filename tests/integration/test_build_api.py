"""
matcher.build() tests — CPU ONNX export with parity check, and the
actionable trtexec failure on hosts without TensorRT.

Run with:
    pixi run python -m pytest tests/integration/test_build_api.py -q
"""

from __future__ import annotations

import shutil
import numpy as np
import pytest

from xcalib.engine.exporter import ExportError, export_onnx  # noqa: E402
from xcalib.models.registry import build_model  # noqa: E402
from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402

onnxruntime = pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.integration


def test_build_onnx_cosine(vit_exp1_matcher, tmp_path):
    out = tmp_path / "onnx_exp1"
    result = vit_exp1_matcher.build("onnx", output_dir=out)

    assert result.target == "onnx"
    assert result.model == "crlite_vit_exp1"
    assert (out / "model.onnx").exists()
    assert result.artifacts == [out / "model.onnx"]
    assert result.parity, "parity check should have produced numbers"
    assert result.parity_ok, f"torch/onnx mismatch: {result.parity}"

    # The matcher must stay usable after the export round-trip.
    assert next(vit_exp1_matcher.model.parameters()).device.type == "cpu"
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    pc = rng.uniform(0, 30, size=(4000, 3)).astype(np.float32)
    b2 = np.array([[100, 100, 200, 200]], dtype=np.float32)
    b3 = np.array([[5, 5, 0, 15, 15, 3]], dtype=np.float32)
    r = vit_exp1_matcher.match(image, pc, b2, b3)
    assert np.all(np.isfinite(r.similarity))


def test_build_onnx_two_stage(tmp_path):
    """crlite (ResNet+PointNet, two-stage) exports stage1+stage2 graphs."""
    cfg = load_yaml(default_config_path("crlite", "utc4"))
    cfg.set("device", "cpu")
    model = build_model("crlite", cfg).eval()

    out = tmp_path / "onnx_crlite"
    result = export_onnx("crlite", model, cfg, out, device="cpu")

    assert (out / "stage1.onnx").exists()
    assert (out / "stage2.onnx").exists()
    assert result.parity_ok, f"torch/onnx mismatch: {result.parity}"
    assert any(k.startswith("stage1/") for k in result.parity)
    assert any(k.startswith("stage2/") for k in result.parity)


def test_build_onnx_unsupported_model(tmp_path):
    cfg = load_yaml(default_config_path("crlite_vit_exp4", "utc4"))
    cfg.set("device", "cpu")
    model = build_model("crlite_vit_exp4", cfg).eval()
    with pytest.raises(ExportError, match="crlite_vit_exp4"):
        export_onnx("crlite_vit_exp4", model, cfg, tmp_path, device="cpu")


def test_build_unknown_target(vit_exp1_matcher, tmp_path):
    with pytest.raises(ValueError, match="onnx.*trt|trt.*onnx"):
        vit_exp1_matcher.build("coreml", output_dir=tmp_path)


@pytest.mark.skipif(
    shutil.which("trtexec") is not None,
    reason="host actually has trtexec; failure path not applicable",
)
def test_build_trt_without_trtexec_is_actionable(vit_exp1_matcher, tmp_path):
    with pytest.raises(FileNotFoundError, match="trtexec"):
        vit_exp1_matcher.build(
            "trt", output_dir=tmp_path / "engines", onnx_dir=tmp_path / "onnx"
        )


def test_export_restores_matcher_device(vit_exp1_matcher, tmp_path):
    """build() must put the model back on the matcher's device + eval mode."""
    vit_exp1_matcher.build("onnx", output_dir=tmp_path / "o")
    model = vit_exp1_matcher.model
    assert not model.training
    assert next(model.parameters()).device == vit_exp1_matcher.device
