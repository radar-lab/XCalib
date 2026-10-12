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

import math
from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
import cv2  # noqa: E402

from xcalib import Matcher  # noqa: E402
from xcalib.engine import trainer as trainer_module  # noqa: E402
from xcalib.engine.trainer import TRAINABLE, epoch_lr, train  # noqa: E402
from xcalib.models.registry import build_model  # noqa: E402
from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402

pytestmark = pytest.mark.integration

MODEL = "crlite_vit_exp1"  # cosine-only objective; cheapest trainable model


# ---------------------------------------------------------------------------
# Synthetic cache builder (UTC layout, 1 camera / 1 lidar)
# ---------------------------------------------------------------------------

def _write_cache(
    path: Path, n_frames: int = 3, K: int = 2, M: int = 2, oriented: bool = False
) -> Path:
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
            if oriented:
                # Each box also as a 4 x 1 x 2 cuboid turned a quarter turn about z, as the A9 caches
                # carry oriented cuboids: its enclosing extents (1 wide in x, 4 long in y) differ from
                # the stored box, so which source feeds the crops shows.
                quarter = np.float32(math.sin(math.pi / 4))
                g.create_dataset("lidar_centers", data=(b3[:, :3] + b3[:, 3:]) / 2)
                g.create_dataset("lidar_dimensions", data=np.tile(np.array([4.0, 1.0, 2.0], np.float32), (M, 1)))
                g.create_dataset("lidar_rotations", data=np.tile(np.array([0, 0, quarter, quarter], np.float32), (M, 1)))
    return path


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory) -> Path:
    return _write_cache(tmp_path_factory.mktemp("hdf5") / "tiny.h5")


@pytest.fixture(scope="module")
def oriented_cache(tmp_path_factory) -> Path:
    return _write_cache(tmp_path_factory.mktemp("hdf5") / "oriented.h5", oriented=True)


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
    # Without a choice the stored boxes are used, and the checkpoint says so.
    assert ckpt["train_args"]["box_source"] == "stored"


def test_train_with_the_enclosing_boxes_records_the_choice(oriented_cache, tmp_path):
    best = train(
        MODEL,
        oriented_cache,
        oriented_cache,
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "envelope",
        pairwise_batch_size=8,
        box_source="envelope",
    )
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert ckpt["train_args"]["box_source"] == "envelope"


def test_envelope_boxes_need_oriented_cuboids_in_the_cache(tiny_cache, tmp_path):
    with pytest.raises(ValueError, match="lidar_centers"):
        train(
            MODEL, tiny_cache, tiny_cache, epochs=1, device="cpu",
            output_dir=tmp_path, box_source="envelope",
        )


def test_unknown_box_source_is_rejected_before_any_model_is_built(tiny_cache, tmp_path, monkeypatch):
    def no_model(*args, **kwargs):
        raise AssertionError("a model was built before box_source was checked")

    monkeypatch.setattr(trainer_module, "build_model", no_model)
    with pytest.raises(ValueError, match="box_source must be one of stored, envelope"):
        train(
            MODEL, tiny_cache, tiny_cache, epochs=1, device="cpu",
            output_dir=tmp_path, box_source="hull",
        )


def test_the_box_source_reaches_both_the_training_and_the_validation_loader(tmp_path, monkeypatch):
    train_cache = _write_cache(tmp_path / "train.h5", oriented=True)
    val_cache = _write_cache(tmp_path / "val.h5", oriented=True)
    real, seen = trainer_module.UTCFrameLoader, []

    class Spy(real):
        def __init__(self, path, box_source="stored"):
            seen.append((Path(path).name, box_source))
            super().__init__(path, box_source=box_source)

    monkeypatch.setattr(trainer_module, "UTCFrameLoader", Spy)
    kwargs = dict(epochs=1, device="cpu", pairwise_batch_size=8)

    train(MODEL, train_cache, val_cache, output_dir=tmp_path / "envelope", box_source="envelope", **kwargs)
    assert {name for name, _ in seen} == {"train.h5", "val.h5"}
    assert {source for _, source in seen} == {"envelope"}

    seen.clear()
    train(MODEL, train_cache, val_cache, output_dir=tmp_path / "default", **kwargs)
    assert {name for name, _ in seen} == {"train.h5", "val.h5"}
    assert {source for _, source in seen} == {"stored"}


def test_box_source_from_the_config_and_the_argument_that_overrides_it(oriented_cache, tmp_path):
    cfg = load_yaml(default_config_path(MODEL, "utc4"))
    cfg.set("box_source", "envelope")
    kwargs = dict(config=cfg, epochs=1, device="cpu", pairwise_batch_size=8)

    from_config = train(MODEL, oriented_cache, oriented_cache, output_dir=tmp_path / "cfg", **kwargs)
    assert torch.load(from_config, map_location="cpu", weights_only=False)["train_args"]["box_source"] == "envelope"

    overridden = train(
        MODEL, oriented_cache, oriented_cache, output_dir=tmp_path / "arg", box_source="stored", **kwargs
    )
    assert torch.load(overridden, map_location="cpu", weights_only=False)["train_args"]["box_source"] == "stored"


def test_a_validation_cache_without_oriented_cuboids_fails_before_the_first_epoch(
    oriented_cache, tiny_cache, tmp_path, monkeypatch
):
    def no_epoch(**kwargs):
        raise AssertionError("a training epoch ran before the validation cache was checked")

    monkeypatch.setattr(trainer_module, "run_epoch_train", no_epoch)
    with pytest.raises(ValueError, match="lidar_centers"):
        train(
            MODEL, oriented_cache, tiny_cache, epochs=1, device="cpu",
            output_dir=tmp_path, box_source="envelope",
        )


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
