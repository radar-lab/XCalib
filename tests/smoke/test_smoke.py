"""
Smoke tests that exercise every model on a synthetic batch.

Two tiers of test:

1. *Architectural smoke*: every model can be instantiated, moved to CPU,
   and run forward on synthetic inputs. Runs unconditionally so we catch
   any regression in the standalone porting before involving weights.

2. *Checkpoint smoke*: every model loads its lab-side .pth and runs the
   matcher pipeline (cropping -> inference). Skipped automatically if the
   .pth files are not present in `checkpoints/` — this keeps
   CI green even when the heavy weights are stripped out.

Run with:
    pixi run python -m pytest tests/smoke/test_smoke.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

from xcalib import Matcher  # noqa: E402
from xcalib.engine.wrappers import FrameData, make_wrapper  # noqa: E402
from xcalib.models.registry import build_model, list_models  # noqa: E402
from xcalib.utils.config import EdgeConfig, load_yaml  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402

pytestmark = pytest.mark.smoke


SHIPPED_MODELS = (
    "crlite",
    "crlite_2dpe",
    "crlite_vit_exp1",
    "crlite_vit_exp3",
    "calibrefine",
)

# Sites where each model has trained weights to ship.
# The A9 site is the public (paper-facing) release; checkpoint smoke
# auto-skips any site whose .pth is absent locally.
MODEL_SITES: dict[str, tuple[str, ...]] = {
    name: ("utc4", "utc3", "a9_dataset_r02_s01") for name in SHIPPED_MODELS
}


def _config_for(model: str, site: str = "utc4") -> EdgeConfig:
    """Use one of the YAML configs to get realistic hyperparameters."""
    cfg = load_yaml(default_config_path(model, site))
    cfg.set("device", "cpu")
    return cfg


def _synth_frame(cfg: EdgeConfig, N: int = 3, M: int = 4) -> FrameData:
    crop_size = int(cfg.get("crop_size", 32))
    pc_size = int(cfg.get("point_cloud_size", 1024))
    return FrameData(
        crops_2d=torch.rand(N, 3, crop_size, crop_size),
        crops_3d=torch.rand(M, pc_size, 3),
        bboxes_2d=torch.tensor([[0.0, 0.0, crop_size, crop_size]] * N),
        bbox_centers_3d=torch.zeros(M, 3),
    )


# ============================================================================
# Tier 1: architecture-only forward pass
# ============================================================================

@pytest.mark.parametrize("model_name", SHIPPED_MODELS)
def test_model_forward_architecture(model_name: str) -> None:
    cfg = _config_for(model_name)
    model = build_model(model_name, cfg).to("cpu").eval()
    wrapper = make_wrapper(
        model_name, model, device="cpu",
        point_cloud_size=int(cfg.get("point_cloud_size", 1024)),
        top_k=int(cfg.get("top_k", 3)),
    )
    frame = _synth_frame(cfg)
    scores, ms = wrapper.predict_matching_matrix(frame)
    assert scores.dim() == 2
    assert scores.shape == (frame.crops_2d.shape[0], frame.crops_3d.shape[0])
    assert ms >= 0.0


def test_registry_completeness() -> None:
    """Make sure every shipped model is in the registry."""
    for m in SHIPPED_MODELS:
        assert m in list_models()


def test_matcher_loads_external_custom_crlite_config(tmp_path: Path) -> None:
    """External YAML configs can define custom architecture shapes."""
    cfg_data = {
        "model": "crlite",
        "device": "cpu",
        "num_classes": 4,
        "crop_size": 32,
        "point_cloud_size": 64,
        "embed_dim": 32,
        "token_len": 16,
        "top_k": 2,
        "max_depth": 80.0,
        "pe_mode": "full",
        "similarity_head": {
            "hidden_dims": [16, 8],
            "dropout": 0.0,
        },
    }
    cfg_path = tmp_path / "custom_crlite.yaml"
    weights_path = tmp_path / "custom_crlite.pth"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    model = build_model("crlite", EdgeConfig(cfg_data)).to("cpu").eval()
    torch.save(model.state_dict(), weights_path)

    matcher = Matcher.from_pretrained(
        "crlite",
        weights=weights_path,
        config=cfg_path,
        device="cpu",
    )

    assert matcher.config.get("similarity_head.hidden_dims") == [16, 8]
    assert matcher.model.embed_dim == 32
    assert matcher.model.similarity_head[0].out_features == 16
    assert matcher.model.similarity_head[-1].out_features == 1

    rng = np.random.default_rng(123)
    image = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    pts = rng.uniform(low=[-5, -5, -1], high=[10, 5, 2], size=(500, 3)).astype(np.float32)
    b2 = np.array([[10, 10, 40, 40], [60, 60, 90, 90]], dtype=np.float32)
    b3 = np.array([
        [0.0, -1.0, -0.5, 2.0, 1.0, 1.0],
        [4.0, -1.0, -0.5, 6.0, 1.0, 1.0],
    ], dtype=np.float32)
    result = matcher.match(image, pts, b2, b3, top_k=2, validate="off")
    assert result.similarity.shape == (2, 2)


def test_calibrefine_accepts_custom_dense_config() -> None:
    cfg = _config_for("calibrefine")
    cfg.set("dense.embed_dim", 128)
    cfg.set("dense.fusion_dim", 96)
    cfg.set("dense.hidden_dim", 48)

    model = build_model("calibrefine", cfg)

    assert model.fc1.out_features == 128
    assert model.fc4.in_features == 128 + model.num_class * 2 + model.token_len * 2
    assert model.fc4.out_features == 96
    assert model.fc5.out_features == 48
    assert model.fc6.in_features == 48


# ============================================================================
# Backward-compat invariant: default architectures reproduce released layouts
# ============================================================================

_CUSTOM_CONFIG_KEYS = {
    "crlite": {
        "config_keys": ("embed_dim", "num_classes", "token_len"),
        "check_attrs": ("similarity_head",),
    },
    "calibrefine": {
        "config_keys": ("num_classes", "token_len"),
        "check_attrs": ("fc1", "fc4", "fc5", "fc6"),
    },
}


# Expected similarity_head child types + out_features for the released CRLite
# architecture (embed_dim=256, hidden=[128, 64, 32]).
_CRLITE_DEFAULT_HEAD_SPEC: tuple[tuple[type, int | None], ...] = (
    (nn.Linear, 128),   # embed_dim // 2
    (nn.LayerNorm, 128),
    (nn.ReLU, None),
    (nn.Dropout, None),
    (nn.Linear, 64),    # embed_dim // 4
    (nn.LayerNorm, 64),
    (nn.ReLU, None),
    (nn.Dropout, None),
    (nn.Linear, 32),    # embed_dim // 8
    (nn.LayerNorm, 32),
    (nn.ReLU, None),
    (nn.Linear, 1),
)

# Expected CalibRefine dense layer out_features (released architecture).
# Derived from the old hardcoded defaults: 512/512/256.
_CALIBREFINE_DEFAULT_DENSE_SPEC: dict[str, dict[str, int]] = {
    "fc1": {"out_features": 512},
    "fc4": {"out_features": 512},
    "fc5": {"out_features": 256},
    "fc6": {"out_features": 1},
}


def _build_head_spec(head: nn.Module) -> list[tuple[type, int | None]]:
    """Enumerate a Sequential's children: (type, dim_or_None).

    For nn.Linear: out_features.  For nn.LayerNorm: normalized_shape
    (unpacked if it's a tuple).  For everything else: None.
    """
    spec: list[tuple[type, int | None]] = []
    for child in head.children():
        if isinstance(child, nn.Linear):
            dim = child.out_features
        elif isinstance(child, nn.LayerNorm):
            ns = child.normalized_shape
            dim = ns[0] if isinstance(ns, (tuple, list)) else ns
        else:
            dim = None
        spec.append((type(child), dim))
    return spec


@pytest.mark.parametrize("model_name,site", [
    ("crlite", site) for site in MODEL_SITES["crlite"]
] + [
    ("calibrefine", site) for site in MODEL_SITES["calibrefine"]
])
def test_default_architecture_matches_released_layout(model_name: str, site: str) -> None:
    """Packaged YAMLs build the exact module layout of released checkpoints.

    The new config keys (similarity_head.*, dense.*) are NOT present in
    the packaged reference YAMLs, so the default branches must fire and
    reproduce the old hardcoded architecture precisely.  This test is
    checkpoint-independent — it asserts structural identity, not weight
    identity — so it runs in CI without .pth files.
    """
    cfg = _config_for(model_name, site)
    # The packaged YAML must not contain the new keys — otherwise this test
    # is not testing the default path and released checkpoints may drift.
    assert "similarity_head" not in cfg and "dense" not in cfg, (
        f"Packaged {model_name}_{site}.yaml contains new config keys — "
        "released checkpoints may no longer use the default architecture path"
    )

    model = build_model(model_name, cfg).to("cpu").eval()

    # --- CRLite: assert similarity_head child layout is byte-identical ---
    if model_name == "crlite":
        spec = _build_head_spec(model.similarity_head)
        assert spec == list(_CRLITE_DEFAULT_HEAD_SPEC), (
            f"CRLite similarity_head layout mismatch for {model_name}/{site}:\n"
            f"  got      {spec}\n"
            f"  expected {_CRLITE_DEFAULT_HEAD_SPEC}"
        )

    # --- CalibRefine: assert dense Linear dims match released layout ---
    if model_name == "calibrefine":
        for attr, expected in _CALIBREFINE_DEFAULT_DENSE_SPEC.items():
            layer = getattr(model, attr)
            for key, expected_val in expected.items():
                actual_val = getattr(layer, key)
                assert actual_val == expected_val, (
                    f"CalibRefine {attr}.{key} mismatch for {model_name}/{site}: "
                    f"{actual_val} != {expected_val}"
                )


def test_calibrefine_rejects_invalid_dense_dims() -> None:
    """dense.* widths of 0 or negative are caught at construction time."""
    cfg = _config_for("calibrefine")

    for bad_dim in (0, -1):
        cfg.set("dense.hidden_dim", bad_dim)
        with pytest.raises(ValueError, match="dense.hidden_dim"):
            build_model("calibrefine", cfg)

    for bad in (0, -1):
        cfg2 = _config_for("calibrefine")
        cfg2.set("dense.fusion_dim", bad)
        with pytest.raises(ValueError, match="dense.fusion_dim"):
            build_model("calibrefine", cfg2)


# ============================================================================
# Tier 2: checkpoint smoke (skipped if .pth missing)
# ============================================================================

def _ckpt_paths(model: str) -> Tuple[Path, ...]:
    sites = MODEL_SITES.get(model, ("utc4", "utc3"))
    return tuple(
        REPO_ROOT / "checkpoints" / f"{model}_{site}_best.pth" for site in sites
    )


_CHECKPOINT_PARAMS = [
    (model, site)
    for model in SHIPPED_MODELS
    for site in MODEL_SITES[model]
]


@pytest.mark.requires_checkpoint
@pytest.mark.parametrize("model_name,site", _CHECKPOINT_PARAMS)
def test_checkpoint_load_and_match(model_name: str, site: str) -> None:
    ckpt = REPO_ROOT / "checkpoints" / f"{model_name}_{site}_best.pth"
    if not ckpt.exists():
        pytest.skip(f"checkpoint missing: {ckpt}")

    cfg_path = default_config_path(model_name, site)
    matcher = Matcher.from_pretrained(
        model=model_name,
        weights=ckpt,
        config=cfg_path,
        device="cpu",
    )

    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    pts = rng.uniform(low=[-20, -20, -2], high=[40, 20, 4], size=(8000, 3)).astype(np.float32)
    b2 = np.array(
        [[100, 200, 220, 320], [500, 300, 620, 420], [800, 250, 900, 380]],
        dtype=np.float32,
    )
    b3 = np.array(
        [
            [5.0, -2.0, -1.0, 8.0, 0.5, 0.5],
            [10.0, 1.0, -1.0, 13.0, 3.0, 0.5],
            [15.0, -3.0, -1.0, 18.0, 0.0, 0.5],
            [22.0, 4.0, -1.0, 25.0, 7.0, 0.5],
        ],
        dtype=np.float32,
    )

    result = matcher.match(
        image=image, point_cloud=pts, bboxes_2d=b2, bboxes_3d=b3,
        top_k=3, return_latency=True,
    )
    assert result.similarity.ndim == 2
    assert result.similarity.shape[0] <= len(b2)
    assert result.similarity.shape[1] <= len(b3)
    assert np.all(np.isfinite(result.similarity))
    assert result.latency_ms >= 0.0
