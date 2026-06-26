"""
ONNX export — importable engine behind ``Matcher.build("onnx")`` and
``scripts/paper/export_onnx.py``.

For the hybrid CRLite family (`crlite`, `crlite_2dpe`) the export is split
into two graphs to preserve the two-stage structure on TensorRT:

    stage1.onnx
        inputs:  image_crops   [N, 3, H, W]      (dynamic N)
                 lidar_crops   [M, P, 3]         (dynamic M)
                 img_centers   [N, 2]
                 lid_centers   [M, 3]
        outputs: img_embed     [N, D]
                 lid_embed     [M, D]

    stage2.onnx
        inputs:  img_pair      [B, D]            (dynamic B)
                 lid_pair      [B, D]
        outputs: score         [B, 1]

For the single-graph models (CRLite-ViT Exp1 / Exp3) a single `model.onnx`
takes the same inputs as stage 1 and returns the N×M cosine similarity
matrix. CalibRefine (pairwise) exports as a single `model.onnx` that
accepts batched pairs and returns B raw logits.

All exports use opset 17 (matches TensorRT 10+ on Blackwell Thor) and the
legacy TorchScript exporter (no onnxscript dependency).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger

from ..models.backbones import _pointnet2_ops
from ..utils.config import EdgeConfig

OPSET = 17

# torch>=2.5 defaults to the dynamo exporter which requires onnxscript.
# We stick with the legacy TorchScript exporter — it has no extra deps and
# produces ONNX graphs that trtexec/TensorRT 10 still accept on Thor.
_EXPORT_KW = {"dynamo": False}

#: Models exportable to ONNX. `crlite_vit_exp4`'s Edge-Aware GAT traces to
#: ScatterND with dynamic axes, which TensorRT 10.x on Blackwell rejects —
#: it stays PyTorch-only (see README §5.2 scope note).
EXPORTABLE_MODELS = (
    "crlite",
    "crlite_2dpe",
    "crlite_vit_exp1",
    "crlite_vit_exp3",
    "calibrefine",
)


class ExportError(RuntimeError):
    """Raised when a model has no registered ONNX exporter."""


@dataclass
class BuildResult:
    """What `Matcher.build` / `export_onnx` hand back to the caller."""

    target: str                                  # "onnx" | "trt"
    model: str
    output_dir: Path
    artifacts: List[Path] = field(default_factory=list)
    #: max |torch - onnx| per (graph, output) — empty when verify=False
    parity: Dict[str, float] = field(default_factory=dict)
    #: per-engine trtexec log files (target="trt" only)
    logs: List[Path] = field(default_factory=list)

    @property
    def parity_ok(self) -> bool:
        return all(v < 1e-3 for v in self.parity.values())


# ============================================================================
# Wrapper modules used only at export time
# ============================================================================

class _CRLiteStage1(torch.nn.Module):
    """Exposes CRLiteModel.extract_features() as a clean ONNX op."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image_crops: torch.Tensor,
        lidar_crops: torch.Tensor,
        img_centers: torch.Tensor,
        lid_centers: torch.Tensor,
    ):
        features = self.model.extract_features(
            image_crops, lidar_crops, img_centers, lid_centers
        )
        return features["img_embed"], features["lid_embed"]


class _CRLiteStage2(torch.nn.Module):
    """Exposes the Stage-2 MLP (embed_fusion + similarity_head) as ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img_pair: torch.Tensor, lid_pair: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([img_pair, lid_pair], dim=1)
        fused = self.model.embed_fusion(combined)
        return self.model.similarity_head(fused)


class _CosineSimilarityModelPE(torch.nn.Module):
    """[Deprecated, retained for ad-hoc debugging only]

    Single-graph cosine matcher with PE inputs. NOT used in shipped
    exports -- vit_exp3 deliberately bypasses PE at inference time to
    reproduce the paper protocol (see export_cosine() docstring). Kept
    here so reviewers / partners who want to compare PE-on vs PE-off can
    swap it in trivially.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image_crops: torch.Tensor,
        lidar_crops: torch.Tensor,
        img_centers: torch.Tensor,
        lid_centers: torch.Tensor,
    ) -> torch.Tensor:
        features = self.model.extract_features(
            image_crops, lidar_crops, img_centers, lid_centers
        )
        img = torch.nn.functional.normalize(features["img_embed"], p=2, dim=1)
        lid = torch.nn.functional.normalize(features["lid_embed"], p=2, dim=1)
        return img @ lid.t()


class _CosineSimilarityModelNoPE(torch.nn.Module):
    """Single-graph cosine matcher without positional encoding (ViT Exp1)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image_crops: torch.Tensor,
        lidar_crops: torch.Tensor,
    ) -> torch.Tensor:
        features = self.model.extract_features(image_crops, lidar_crops)
        img = torch.nn.functional.normalize(features["img_embed"], p=2, dim=1)
        lid = torch.nn.functional.normalize(features["lid_embed"], p=2, dim=1)
        return img @ lid.t()


class _CalibRefinePairwise(torch.nn.Module):
    """Pairwise CalibRefine: B pairs in -> B logits out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image_pair: torch.Tensor,
        lidar_pair: torch.Tensor,
        img_pos: torch.Tensor,
        lid_pos: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(image_pair, lidar_pair, img_pos, lid_pos)
        return out["main_output"]


# ============================================================================
# Helpers
# ============================================================================

@contextmanager
def _deterministic_fps():
    """Force deterministic FPS init while tracing so the graph is stable."""
    prev = _pointnet2_ops.deterministic_fps_enabled()
    _pointnet2_ops.set_deterministic_fps(True)
    try:
        yield
    finally:
        _pointnet2_ops.set_deterministic_fps(prev)


def _make_dummy(
    cfg: EdgeConfig,
    device: torch.device,
    N: int = 4,
    M: int = 5,
):
    crop_size = int(cfg.get("crop_size", 32))
    pc_size = int(cfg.get("point_cloud_size", 1024))
    image_crops = torch.randn(N, 3, crop_size, crop_size, device=device)
    lidar_crops = torch.randn(M, pc_size, 3, device=device)
    img_centers = torch.tensor(
        [[960.0, 540.0]] * N, dtype=torch.float32, device=device
    )
    lid_centers = torch.tensor(
        [[15.0, 0.0, 0.0]] * M, dtype=torch.float32, device=device
    )
    return image_crops, lidar_crops, img_centers, lid_centers


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _check_parity(
    name: str,
    onnx_path: Path,
    inputs: dict[str, np.ndarray],
    torch_outputs: list[np.ndarray],
    result: BuildResult,
    atol: float = 1e-4,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping parity check "
                       "(pip install 'XCalib[onnx]')")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    ort_outs = sess.run(out_names, inputs)
    for i, (t, o) in enumerate(zip(torch_outputs, ort_outs)):
        diff = float(np.abs(t - o).max()) if t.size else 0.0
        status = "OK" if diff < atol else "FAIL"
        result.parity[f"{name}/{out_names[i]}"] = diff
        logger.info(f"[{name}] output[{i}] max|torch-onnx| = {diff:.3e} ({status})")


def _disable_transformer_fastpath(module: torch.nn.Module) -> None:
    """Force `nn.TransformerEncoder*` to take the ONNX-traceable eager path.

    PyTorch's fused fast-path calls `aten::_transformer_encoder_layer_fwd`,
    which has no ONNX equivalent. We disable nested-tensor packing on the
    encoder and the MHA fast-path globally to keep tracing within pure-ATen
    ops. Inference configs use dropout=0 so this is numerically equivalent.
    """
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:
        pass
    for sub in module.modules():
        if isinstance(sub, torch.nn.TransformerEncoder):
            sub.enable_nested_tensor = False
            sub.use_nested_tensor = False


# ============================================================================
# Per-model exporters
# ============================================================================

def export_crlite(
    model, cfg: EdgeConfig, out_dir: Path, device: torch.device,
    result: BuildResult, verify: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1 = _CRLiteStage1(model).eval().to(device)
    stage2 = _CRLiteStage2(model).eval().to(device)

    image_crops, lidar_crops, img_centers, lid_centers = _make_dummy(cfg, device)

    # Stage 1
    s1_path = out_dir / "stage1.onnx"
    with torch.no_grad():
        torch.onnx.export(
            stage1,
            (image_crops, lidar_crops, img_centers, lid_centers),
            str(s1_path),
            opset_version=OPSET,
            input_names=["image_crops", "lidar_crops", "img_centers", "lid_centers"],
            output_names=["img_embed", "lid_embed"],
            dynamic_axes={
                "image_crops": {0: "N"},
                "lidar_crops": {0: "M"},
                "img_centers": {0: "N"},
                "lid_centers": {0: "M"},
                "img_embed":   {0: "N"},
                "lid_embed":   {0: "M"},
            },
            do_constant_folding=True,
            **_EXPORT_KW,
        )
    logger.success(f"Wrote {s1_path}")
    result.artifacts.append(s1_path)

    if verify:
        with torch.no_grad():
            img_t, lid_t = stage1(image_crops, lidar_crops, img_centers, lid_centers)
        _check_parity(
            "stage1",
            s1_path,
            {
                "image_crops": _to_numpy(image_crops),
                "lidar_crops": _to_numpy(lidar_crops),
                "img_centers": _to_numpy(img_centers),
                "lid_centers": _to_numpy(lid_centers),
            },
            [_to_numpy(img_t), _to_numpy(lid_t)],
            result,
        )
        embed_dim = int(img_t.shape[1])
    else:
        with torch.no_grad():
            img_t, _ = stage1(image_crops, lidar_crops, img_centers, lid_centers)
        embed_dim = int(img_t.shape[1])

    # Stage 2 — pairs
    B = 16
    img_pair = torch.randn(B, embed_dim, device=device)
    lid_pair = torch.randn(B, embed_dim, device=device)
    s2_path = out_dir / "stage2.onnx"
    with torch.no_grad():
        torch.onnx.export(
            stage2,
            (img_pair, lid_pair),
            str(s2_path),
            opset_version=OPSET,
            input_names=["img_pair", "lid_pair"],
            output_names=["score"],
            dynamic_axes={
                "img_pair": {0: "B"},
                "lid_pair": {0: "B"},
                "score": {0: "B"},
            },
            do_constant_folding=True,
            **_EXPORT_KW,
        )
    logger.success(f"Wrote {s2_path}")
    result.artifacts.append(s2_path)

    if verify:
        with torch.no_grad():
            torch_score = stage2(img_pair, lid_pair)
        _check_parity(
            "stage2",
            s2_path,
            {"img_pair": _to_numpy(img_pair), "lid_pair": _to_numpy(lid_pair)},
            [_to_numpy(torch_score)],
            result,
        )


def export_cosine(
    model_name: str, model, cfg: EdgeConfig, out_dir: Path, device: torch.device,
    result: BuildResult, verify: bool = True,
) -> None:
    # IMPORTANT — both crlite_vit_exp1 and crlite_vit_exp3 are exported with
    # the NO-PE cosine wrapper, even though crlite_vit_exp3's PyTorch model
    # has PE layers in its forward. This deliberately mirrors what the
    # standalone matcher does at inference time
    # (xcalib.engine.wrappers.PositionEnhancedDotWrapper passes
    # img_centers=None / lid_centers=None) which in turn reproduces the
    # paper-time evaluation protocol bit-for-bit -- the paper's wrapper had
    # a key-mismatch bug that silently passed None for both centers, and
    # the trained PE branch is degenerate at test time. Tracing without PE
    # inputs guarantees the partner-shipped ONNX / TRT artifact produces
    # paper-correct outputs. See docs/paper.md UTC reference section Note 2 for the full story.
    wrapper = _CosineSimilarityModelNoPE(model).eval().to(device)
    if model_name in {"crlite_vit_exp1", "crlite_vit_exp3"}:
        _disable_transformer_fastpath(wrapper)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_crops, lidar_crops, _, _ = _make_dummy(cfg, device)
    onnx_path = out_dir / "model.onnx"

    args = (image_crops, lidar_crops)
    input_names = ["image_crops", "lidar_crops"]
    dynamic_axes = {
        "image_crops": {0: "N"},
        "lidar_crops": {0: "M"},
        "similarity":  {0: "N", 1: "M"},
    }
    parity_inputs = {
        "image_crops": _to_numpy(image_crops),
        "lidar_crops": _to_numpy(lidar_crops),
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(onnx_path),
            opset_version=OPSET,
            input_names=input_names,
            output_names=["similarity"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            **_EXPORT_KW,
        )
    logger.success(f"Wrote {onnx_path}")
    result.artifacts.append(onnx_path)

    if verify:
        with torch.no_grad():
            torch_out = wrapper(*args)
        _check_parity(model_name, onnx_path, parity_inputs, [_to_numpy(torch_out)], result)


def export_calibrefine(
    model, cfg: EdgeConfig, out_dir: Path, device: torch.device,
    result: BuildResult, verify: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = _CalibRefinePairwise(model).eval().to(device)

    B = 8
    crop_size = int(cfg.get("crop_size", 32))
    pc_size = int(cfg.get("point_cloud_size", 1024))
    image_pair = torch.randn(B, 3, crop_size, crop_size, device=device)
    lidar_pair = torch.randn(B, pc_size, 3, device=device)
    img_pos = torch.randn(B, 2, device=device)
    lid_pos = torch.randn(B, 2, device=device)

    onnx_path = out_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (image_pair, lidar_pair, img_pos, lid_pos),
            str(onnx_path),
            opset_version=OPSET,
            input_names=["image_pair", "lidar_pair", "img_pos", "lid_pos"],
            output_names=["score"],
            dynamic_axes={
                "image_pair": {0: "B"},
                "lidar_pair": {0: "B"},
                "img_pos":    {0: "B"},
                "lid_pos":    {0: "B"},
                "score":      {0: "B"},
            },
            do_constant_folding=True,
            **_EXPORT_KW,
        )
    logger.success(f"Wrote {onnx_path}")
    result.artifacts.append(onnx_path)

    if verify:
        with torch.no_grad():
            torch_out = wrapper(image_pair, lidar_pair, img_pos, lid_pos)
        _check_parity(
            "calibrefine",
            onnx_path,
            {
                "image_pair": _to_numpy(image_pair),
                "lidar_pair": _to_numpy(lidar_pair),
                "img_pos": _to_numpy(img_pos),
                "lid_pos": _to_numpy(lid_pos),
            },
            [_to_numpy(torch_out)],
            result,
        )


# ============================================================================
# Top-level dispatch
# ============================================================================

def export_onnx(
    model_name: str,
    model: torch.nn.Module,
    cfg: EdgeConfig,
    out_dir: Path | str,
    *,
    device: torch.device | str = "cpu",
    verify: bool = True,
) -> BuildResult:
    """Export `model` (with whatever weights it currently holds) to ONNX.

    Dispatches per model family; see the module docstring for the graph
    layout. Returns a `BuildResult` with artifact paths and parity numbers.
    """
    out_dir = Path(out_dir)
    device = torch.device(device)
    result = BuildResult(target="onnx", model=model_name, output_dir=out_dir)

    # One-shot adapted models wrap the base architecture but expose the same
    # extract_features / stage-2 attributes, so they trace through the same
    # exporters (the adapters end up inside the graph).
    model = model.eval().to(device)

    logger.info(f"Exporting {model_name} -> {out_dir} (opset {OPSET})")
    with _deterministic_fps():
        if model_name in {"crlite", "crlite_2dpe"}:
            export_crlite(model, cfg, out_dir, device, result, verify=verify)
        elif model_name in {"crlite_vit_exp1", "crlite_vit_exp3"}:
            export_cosine(model_name, model, cfg, out_dir, device, result, verify=verify)
        elif model_name == "calibrefine":
            export_calibrefine(model, cfg, out_dir, device, result, verify=verify)
        else:
            raise ExportError(
                f"No ONNX exporter registered for '{model_name}'. Exportable "
                f"models: {', '.join(EXPORTABLE_MODELS)}. (crlite_vit_exp4's "
                "GAT traces to dynamic-axis ScatterND, which TensorRT 10.x "
                "on Thor rejects — it is deployed via PyTorch only.)"
            )

    logger.success(f"Done. ONNX files: {sorted(out_dir.glob('*.onnx'))}")
    return result
