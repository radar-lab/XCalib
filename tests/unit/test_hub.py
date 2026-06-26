"""
Hub plumbing tests — everything that runs offline: URI parsing, repo-id
conventions and env overrides, the packaged-config resolver, and the
save_pretrained / from_pretrained directory round-trip.

Actual downloads need Hub access + network and are exercised manually via
`pull-weights` (see docs/hub.md).

Run with:
    pixi run python -m pytest tests/unit/test_hub.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xcalib import Matcher  # noqa: E402
from xcalib import hub  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402


# ============================================================================
# hf:// URIs
# ============================================================================

def test_is_hf_uri():
    assert hub.is_hf_uri("hf://org/repo/checkpoints/x.pth")
    assert not hub.is_hf_uri("checkpoints/x.pth")
    assert not hub.is_hf_uri(Path("hf://org/repo/x.pth"))  # only plain str


def test_parse_hf_uri():
    repo, path, rev = hub.parse_hf_uri("hf://my-org/xcalib/checkpoints/a.pth")
    assert repo == "my-org/xcalib"
    assert path == "checkpoints/a.pth"
    assert rev is None

    repo, path, rev = hub.parse_hf_uri("hf://my-org/xcalib@v0.1.0/configs/c.yaml")
    assert repo == "my-org/xcalib"
    assert path == "configs/c.yaml"
    assert rev == "v0.1.0"

    with pytest.raises(ValueError):
        hub.parse_hf_uri("hf://my-org/repo-only")
    with pytest.raises(ValueError):
        hub.parse_hf_uri("s3://bucket/key")


# ============================================================================
# Repo conventions
# ============================================================================

def test_repo_id_env_overrides(monkeypatch):
    monkeypatch.delenv(hub.ENV_PUBLIC_REPO, raising=False)
    assert hub.default_model_repo() == hub.PUBLIC_MODEL_REPO
    assert hub.default_model_repo("utc4") == hub.PUBLIC_MODEL_REPO
    assert hub.default_model_repo("a9_dataset_r02_s01") == hub.PUBLIC_MODEL_REPO

    monkeypatch.setenv(hub.ENV_PUBLIC_REPO, "acme/weights-public")
    assert hub.default_model_repo("utc4") == "acme/weights-public"
    assert hub.default_model_repo("a9_dataset_r02_s01") == "acme/weights-public"


def test_filename_conventions():
    assert hub.weights_filename("crlite", "utc4") == "checkpoints/crlite_utc4_best.pth"
    assert hub.config_filename("crlite_vit_exp3", "utc3") == "configs/crlite_vit_exp3_utc3.yaml"


# ============================================================================
# Packaged-config resolver
# ============================================================================

def test_default_config_path_resolves_repo_configs():
    for model in ("crlite", "calibrefine", "crlite_vit_exp1"):
        for site in ("utc3", "utc4"):
            p = default_config_path(model, site)
            assert p.exists()
            assert p.name == f"{model}_{site}.yaml"

    with pytest.raises(FileNotFoundError, match="no_such_model"):
        default_config_path("no_such_model", "utc4")


# ============================================================================
# save_pretrained / from_pretrained round-trip (local, no network)
# ============================================================================

def test_save_pretrained_roundtrip(vit_exp1_matcher, tmp_path):
    out = tmp_path / "exported"
    paths = vit_exp1_matcher.save_pretrained(out)
    assert paths["weights"] == out / "crlite_vit_exp1.pth"
    assert paths["config"] == out / "crlite_vit_exp1.yaml"
    assert paths["weights"].exists() and paths["config"].exists()

    reloaded = Matcher.from_pretrained(
        "crlite_vit_exp1", weights=out, device="cpu"
    )

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    b2 = np.array([[100, 100, 220, 220], [400, 300, 520, 420]], dtype=np.float32)
    b3 = np.array(
        [[5, -2, -1, 9, 1, 1], [15, 2, -1, 19, 5, 1]], dtype=np.float32
    )
    clouds = [rng.uniform(box[:3], box[3:], size=(200, 3)).astype(np.float32) for box in b3]
    pc = np.vstack(clouds)

    sim_a = vit_exp1_matcher.match(image, pc, b2, b3).similarity
    sim_b = reloaded.match(image, pc, b2, b3).similarity
    np.testing.assert_allclose(sim_a, sim_b, atol=1e-5)


def test_from_pretrained_directory_requires_model_file(tmp_path):
    (tmp_path / "stuff").mkdir()
    with pytest.raises(FileNotFoundError, match="save_pretrained"):
        Matcher.from_pretrained("crlite", weights=tmp_path / "stuff")
