"""
Training-engine tests — synthetic 3-frame HDF5 cache, 1 epoch on CPU.

Covers the `xcalib.train` entry point (registry-name mode), the
`Matcher.train` fine-tune path (module mode + best-checkpoint reload), and
the schedule/eligibility helpers. Mirrors the cache layout documented in
``docs/hdf5-format.md`` so the loader path is exercised end to end.

Run with:
    pixi run python -m pytest tests/integration/test_trainer.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
import cv2  # noqa: E402

from xcalib import Matcher  # noqa: E402
from xcalib.engine.trainer import TRAINABLE, epoch_lr, train  # noqa: E402
from xcalib.models.registry import build_model  # noqa: E402
from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402

pytestmark = pytest.mark.integration

MODEL = "crlite_vit_exp1"  # cosine-only objective; cheapest trainable model


# ---------------------------------------------------------------------------
# Synthetic cache builder (UTC layout, 1 camera / 1 lidar)
# ---------------------------------------------------------------------------

def _write_cache(path: Path, n_frames: int = 3, K: int = 2, M: int = 2) -> Path:
    rng = np.random.default_rng(7)
    b2 = np.array([[10, 10, 80, 80], [120, 100, 200, 180]], dtype=np.float32)[:K]
    b3 = np.array(
        [[2.0, -2.0, -1.0, 6.0, 2.0, 1.0], [8.0, -1.0, -1.0, 12.0, 3.0, 1.0]],
        dtype=np.float32,
    )[:M]

    with h5py.File(path, "w") as f:
        vlen = h5py.special_dtype(vlen=np.dtype("uint8"))
        img_ds = f.create_group("images").create_group("cam0").create_dataset(
            "data", shape=(n_frames,), dtype=vlen
        )
        pc_grp = f.create_group("point_clouds").create_group("lidar0")
        lab_grp = f.create_group("labels").create_group("lidar0")

        for i in range(n_frames):
            image = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
            ok, jpg = cv2.imencode(".jpg", image)
            assert ok
            img_ds[i] = np.frombuffer(jpg.tobytes(), dtype=np.uint8)

            clouds = [rng.uniform(-5, 20, size=(800, 3)).astype(np.float32)]
            for box in b3:
                clouds.append(rng.uniform(box[:3], box[3:], size=(200, 3)).astype(np.float32))
            pc_grp.create_group(str(i)).create_dataset("xyz", data=np.vstack(clouds))

            g = lab_grp.create_group(str(i))
            g.create_dataset("num_camera_detections", data=K)
            g.create_dataset("num_lidar_detections", data=M)
            g.create_dataset("camera_bbox_2d", data=b2)
            g.create_dataset("lidar_bbox_3d", data=b3)
            g.create_dataset("match_matrix", data=np.eye(K, M, dtype=bool))
    return path


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory) -> Path:
    return _write_cache(tmp_path_factory.mktemp("hdf5") / "tiny.h5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_epoch_lr_schedule():
    kw = dict(total_epochs=10, base_lr=1e-3, warmup_epochs=2, scheduler="cosine", min_lr=1e-5)
    assert epoch_lr(1, **kw) == pytest.approx(5e-4)   # linear warmup
    assert epoch_lr(2, **kw) == pytest.approx(1e-3)   # warmup done
    assert epoch_lr(10, **kw) == pytest.approx(1e-5)  # cosine floor
    flat = dict(kw, scheduler="none")
    assert epoch_lr(7, **flat) == pytest.approx(1e-3)


def test_untrainable_model_rejected(tiny_cache, tmp_path):
    assert "calibrefine" not in TRAINABLE
    with pytest.raises(ValueError, match="calibrefine"):
        train("calibrefine", tiny_cache, tiny_cache, epochs=1, output_dir=tmp_path)


def test_missing_cache_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(MODEL, tmp_path / "nope.h5", tmp_path / "nope.h5", epochs=1, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# End-to-end: train() by registry name, then Matcher.train fine-tune
# ---------------------------------------------------------------------------

def test_train_by_name(tiny_cache, tmp_path):
    out = tmp_path / "runs"
    best = train(
        MODEL,
        tiny_cache,
        tiny_cache,
        epochs=1,
        lr=1e-4,
        device="cpu",
        output_dir=out,
        pairwise_batch_size=8,
    )
    assert best == out / "best.pth"
    assert best.exists()
    assert (out / "history.json").exists()
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt and "metrics" in ckpt
    assert ckpt["train_args"]["model"] == MODEL


def test_matcher_train_finetune(tiny_cache, tmp_path):
    cfg = load_yaml(default_config_path(MODEL, "utc4"))
    cfg.set("device", "cpu")
    model = build_model(MODEL, cfg)
    weights = tmp_path / "init.pth"
    torch.save({"model_state_dict": model.state_dict()}, weights)
    matcher = Matcher.from_pretrained(MODEL, weights=weights, config=cfg, device="cpu")

    best = matcher.train(
        tiny_cache,
        tiny_cache,
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "ft",
        pairwise_batch_size=8,
    )
    assert best.exists()

    # The matcher must serve the (reloaded best) weights right away.
    assert next(matcher.model.parameters()).device.type == "cpu"
    assert not matcher.model.training
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    pc = rng.uniform(0, 30, size=(4000, 3)).astype(np.float32)
    b2 = np.array([[100, 100, 200, 200]], dtype=np.float32)
    b3 = np.array([[5, 5, 0, 15, 15, 3]], dtype=np.float32)
    r = matcher.match(image, pc, b2, b3)
    assert np.all(np.isfinite(r.similarity))


def test_train_lazy_export():
    import xcalib

    from xcalib.engine.trainer import train as direct

    assert xcalib.train is direct
