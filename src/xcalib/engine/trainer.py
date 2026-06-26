"""
Training engine — train from scratch or fine-tune the matchers on HDF5 caches.

This is the library home of the loop that used to live in
``scripts/paper/train_hdf5.py`` (the script is now a thin shim). Reachable as::

    from xcalib import train

    best = train("crlite", "a9_r02_s01_train.h5", "a9_r02_s01_val.h5",
                 epochs=100, warmup_epochs=5, optimizer="adam",
                 scheduler="cosine", output_dir="runs/crlite_scratch")

or, to fine-tune the weights a matcher already holds::

    matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
    matcher.train("a9_r02_s01_train.h5", "a9_r02_s01_val.h5", epochs=10,
                  output_dir="runs/ft")

``train_data`` / ``val_data`` accept local ``.h5`` paths or a Hub site name
(e.g. ``"a9_dataset_r02_s01"``) which is resolved through
:func:`xcalib.hub.datasets.load_dataset`.

Works with any cache that follows ``docs/hdf5-format.md``. Supported
models: ``crlite``, ``crlite_2dpe`` (two-stage CE), ``crlite_vit_exp1``
(cosine-only), ``crlite_vit_exp3`` (cosine + position encoding). The
pairwise ``calibrefine`` baseline has a different objective and is not
trainable here. ``h5py`` ships in the ``[train]`` extra.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from ..data import PrepareConfig, UTCFrameLoader, prepare_frame
from ..models.registry import build_model
from ..utils.config import EdgeConfig, load_yaml
from ..utils.io import (
    default_config_path,
    extract_state_dict,
    load_checkpoint,
    resolve_device,
)
from ..utils.metrics import MatchingMetricsAccumulator
from .wrappers import FrameData, make_wrapper

#: Models with a `forward_train` objective compatible with this loop.
TRAINABLE = ("crlite", "crlite_2dpe", "crlite_vit_exp1", "crlite_vit_exp3")


# ---------------------------------------------------------------------------
# Schedule + loss helpers
# ---------------------------------------------------------------------------

def epoch_lr(
    epoch: int,
    *,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    scheduler: str,
    min_lr: float,
) -> float:
    """Per-epoch learning rate: linear warmup, then constant or cosine decay."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if scheduler == "cosine":
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    return base_lr


def _filter_match_matrix(mm: np.ndarray, kept_2d: np.ndarray, kept_3d: np.ndarray) -> np.ndarray:
    if mm.size == 0 or kept_2d.size == 0 or kept_3d.size == 0:
        return np.zeros((kept_2d.size, kept_3d.size), dtype=bool)
    return mm[np.ix_(kept_2d, kept_3d)].astype(bool)


def _rowwise_ce(
    logits_nm: torch.Tensor,
    match_bool: torch.Tensor,
    *,
    temperature: float,
    label_smoothing: float,
) -> torch.Tensor:
    """Cross-entropy for rows that contain at least one positive match."""
    if logits_nm.ndim != 2 or match_bool.ndim != 2:
        raise ValueError("Expected [N,M] logits and targets.")
    row_ok = match_bool.any(dim=1)
    if not row_ok.any():
        return logits_nm.mean() * 0.0
    targets = match_bool.float().argmax(dim=1)
    scaled = logits_nm / max(temperature, 1e-6)
    return F.cross_entropy(
        scaled[row_ok],
        targets[row_ok],
        label_smoothing=label_smoothing,
    )


def _forward_train(model: torch.nn.Module, model_name: str, frame: FrameData) -> dict:
    dev = next(model.parameters()).device
    x2 = frame.crops_2d.to(dev)
    x3 = frame.crops_3d.to(dev)
    if model_name == "crlite_vit_exp1":
        return model.forward_train(x2, x3)
    if model_name in {"crlite", "crlite_2dpe", "crlite_vit_exp3"}:
        if frame.bboxes_2d is None or frame.bbox_centers_3d is None:
            raise ValueError(f"{model_name} training requires bbox centres.")
        b2 = frame.bboxes_2d.to(dev)
        img_centers = (b2[:, :2] + b2[:, 2:]) / 2.0
        lid_centers = frame.bbox_centers_3d.to(dev)
        return model.forward_train(x2, x3, img_centers=img_centers, lid_centers=lid_centers)
    raise KeyError(model_name)


def _train_loss(
    out: dict,
    gt_bool: torch.Tensor,
    *,
    temperature: float,
    label_smoothing: float,
    w1: float,
    w2: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.zeros((), device=gt_bool.device)
    diag: dict[str, float] = {}
    ls1 = _rowwise_ce(
        out["stage1_similarity"], gt_bool, temperature=temperature, label_smoothing=label_smoothing
    )
    loss = loss + w1 * ls1
    diag["stage1_loss"] = float(ls1.detach().cpu())
    if "stage2_similarity" in out and w2 > 0:
        ls2 = _rowwise_ce(
            out["stage2_similarity"],
            gt_bool,
            temperature=temperature,
            label_smoothing=label_smoothing,
        )
        loss = loss + w2 * ls2
        diag["stage2_loss"] = float(ls2.detach().cpu())
    return loss, diag


# ---------------------------------------------------------------------------
# Epoch runners
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hdf5(
    model_name: str,
    model: torch.nn.Module,
    *,
    hdf5_path: Path,
    prepare_cfg: PrepareConfig,
    device: torch.device,
    top_k: int,
    pairwise_batch_size: int,
    limit_frames: Optional[int],
) -> dict:
    model.eval()
    wrapper = make_wrapper(
        model_name,
        model,
        device=device,
        point_cloud_size=prepare_cfg.point_cloud_size,
        top_k=top_k,
        pairwise_batch_size=pairwise_batch_size,
    )
    acc = MatchingMetricsAccumulator()
    with UTCFrameLoader(hdf5_path) as loader:
        for i, raw in enumerate(loader):
            if limit_frames is not None and i >= limit_frames:
                break
            try:
                frame, k2, k3 = prepare_frame(
                    image=raw.image,
                    point_cloud=raw.point_cloud,
                    bboxes_2d=raw.bboxes_2d,
                    bboxes_3d=raw.bboxes_3d,
                    cfg=prepare_cfg,
                    images=raw.images,
                    camera_per_det=raw.camera_per_det,
                )
            except Exception as e:
                logger.warning(f"prepare failed frame {raw.frame_key}: {e}")
                continue
            if frame.crops_2d.numel() == 0 or frame.crops_3d.numel() == 0:
                continue
            gt = _filter_match_matrix(raw.match_matrix, k2, k3)
            scores, ms = wrapper.predict_matching_matrix(frame)
            acc.update(scores, gt, latency_ms=ms)
    return acc.summary()


def run_epoch_train(
    *,
    model_name: str,
    model: torch.nn.Module,
    loader: UTCFrameLoader,
    frame_keys: list[str],
    prepare_cfg: PrepareConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    label_smoothing: float,
    w1: float,
    w2: float,
    rng: np.random.Generator,
) -> float:
    model.train()
    keys = list(frame_keys)
    random.shuffle(keys)
    total = 0.0
    n_ok = 0
    pbar = tqdm(keys, desc="train", leave=False)
    for key in pbar:
        raw = loader.get_frame(key)
        if raw is None:
            continue
        try:
            frame, k2, k3 = prepare_frame(
                image=raw.image,
                point_cloud=raw.point_cloud,
                bboxes_2d=raw.bboxes_2d,
                bboxes_3d=raw.bboxes_3d,
                cfg=prepare_cfg,
                rng=rng,
                images=raw.images,
                camera_per_det=raw.camera_per_det,
            )
        except Exception:
            continue
        if frame.crops_2d.numel() == 0 or frame.crops_3d.numel() == 0:
            continue
        gt = torch.from_numpy(_filter_match_matrix(raw.match_matrix, k2, k3)).to(device)
        if not gt.any():
            continue
        optimizer.zero_grad(set_to_none=True)
        out = _forward_train(model, model_name, frame)
        loss, parts = _train_loss(
            out, gt, temperature=temperature, label_smoothing=label_smoothing, w1=w1, w2=w2
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        n_ok += 1
        pbar.set_postfix(loss=f"{total / max(n_ok, 1):.4f}", **{k: f"{v:.3f}" for k, v in parts.items()})
    return total / max(n_ok, 1) if n_ok else 0.0


# ---------------------------------------------------------------------------
# Data resolution: local .h5 path or Hub site name
# ---------------------------------------------------------------------------

def _resolve_data(data: Union[str, Path], split: str) -> Path:
    """Resolve a train/val data spec to a local HDF5 path.

    Accepts a local ``.h5``/``.hdf5`` path (returned as-is) or a
    Hub-distributed site name like ``"a9_dataset_r02_s01"``, downloaded via
    :func:`xcalib.hub.datasets.dataset_path`. The UTC partner caches are not
    on the Hub — pass their local paths.
    """
    p = Path(data)
    if p.suffix.lower() in {".h5", ".hdf5"} or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"HDF5 cache not found: {p}")
        return p
    from ..hub.datasets import dataset_path

    return dataset_path(str(data), split=split)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(
    model: Union[str, torch.nn.Module],
    train_data: Union[str, Path],
    val_data: Union[str, Path],
    *,
    model_name: Optional[str] = None,
    config: Union[EdgeConfig, str, Path, None] = None,
    site: Optional[str] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer: str = "adamw",
    warmup_epochs: int = 0,
    scheduler: str = "none",
    min_lr: float = 1e-5,
    temperature: float = 0.07,
    label_smoothing: float = 0.05,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
    weights: Union[str, Path, None] = None,
    output_dir: Union[str, Path] = "runs/train",
    device: Union[str, torch.device, None] = "auto",
    seed: int = 42,
    limit_train_frames: Optional[int] = None,
    limit_val_frames: Optional[int] = None,
    pairwise_batch_size: int = 64,
) -> Path:
    """Train (from scratch) or fine-tune a matcher; returns the best checkpoint.

    Args:
        model: Registry name (``"crlite"``, ...) or an already-built model
            module (used by :meth:`Matcher.train`).
        train_data: Local ``.h5`` cache path or Hub site name (resolved
            through ``xcalib.hub.datasets``).
        val_data: Same as *train_data*, for the validation split.
        model_name: Required when *model* is a module.
        config: Reference YAML / :class:`EdgeConfig`. Defaults to the
            packaged config for (*model*, *site*).
        site: Site used to pick the default config (inferred from
            *train_data* when it is a site name; falls back to ``utc4``).
        weights: Optional checkpoint to warm-start from (registry-name mode).
        output_dir: Where per-epoch checkpoints, ``best.pth`` (selected by
            validation MRR), and ``history.json`` land.

    Returns:
        Path to ``best.pth`` inside *output_dir*.
    """
    # ---- resolve model / config -------------------------------------------
    if isinstance(model, str):
        resolved_name = model
    else:
        if model_name is None:
            raise ValueError("model_name is required when passing a model module.")
        resolved_name = model_name
    if resolved_name not in TRAINABLE:
        raise ValueError(
            f"model '{resolved_name}' is not trainable with this loop "
            f"(supported: {', '.join(TRAINABLE)})."
        )

    if site is None:
        # When train_data is a site name (not a path), reuse it for config
        # resolution; otherwise default to the historical utc4 recipe.
        td = str(train_data)
        site = td if (not Path(td).exists() and not td.lower().endswith((".h5", ".hdf5"))) else "utc4"

    if config is None:
        cfg = load_yaml(default_config_path(resolved_name, site))
    elif isinstance(config, EdgeConfig):
        cfg = config
    else:
        cfg = load_yaml(config)

    dev = resolve_device(device if device is not None else cfg.get("device", "auto"))

    random.seed(seed)
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    if isinstance(model, str):
        net = build_model(resolved_name, cfg)
        net.to(dev)
        if weights is not None:
            net.load_weights(weights, strict=False)
    else:
        net = model
        net.to(dev)
        if weights is not None:
            state = extract_state_dict(load_checkpoint(weights, device=dev))
            net.load_state_dict(state, strict=False)
    net.train()

    # vit_exp1 has no Stage-2 head — force its loss weight to zero.
    if resolved_name == "crlite_vit_exp1":
        stage2_weight = 0.0

    train_h5 = _resolve_data(train_data, "train")
    val_h5 = _resolve_data(val_data, "val")

    prepare_cfg = PrepareConfig(
        crop_size=int(cfg.get("crop_size", 32)),
        point_cloud_size=int(cfg.get("point_cloud_size", 1024)),
        bbox_expansion=float(cfg.get("bbox_expansion", 1.25)),
    )
    top_k = int(cfg.get("top_k", 8))

    opt_cls = torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam
    optim = opt_cls(net.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_mrr = -1.0
    best_path = output_dir / "best.pth"

    train_kwargs = {
        "model": resolved_name,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "warmup_epochs": warmup_epochs,
        "scheduler": scheduler,
        "min_lr": min_lr,
        "temperature": temperature,
        "label_smoothing": label_smoothing,
        "stage1_weight": stage1_weight,
        "stage2_weight": stage2_weight,
        "seed": seed,
        "train_data": str(train_h5),
        "val_data": str(val_h5),
    }

    with UTCFrameLoader(train_h5) as train_loader:
        keys = list(train_loader.frame_keys)
        if limit_train_frames is not None:
            keys = keys[:limit_train_frames]

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            lr_now = epoch_lr(
                epoch,
                total_epochs=epochs,
                base_lr=lr,
                warmup_epochs=warmup_epochs,
                scheduler=scheduler,
                min_lr=min_lr,
            )
            for group in optim.param_groups:
                group["lr"] = lr_now
            avg_loss = run_epoch_train(
                model_name=resolved_name,
                model=net,
                loader=train_loader,
                frame_keys=keys,
                prepare_cfg=prepare_cfg,
                optimizer=optim,
                device=dev,
                temperature=temperature,
                label_smoothing=label_smoothing,
                w1=stage1_weight,
                w2=stage2_weight,
                rng=np_rng,
            )
            val = evaluate_hdf5(
                resolved_name,
                net,
                hdf5_path=val_h5,
                prepare_cfg=prepare_cfg,
                device=dev,
                top_k=top_k,
                pairwise_batch_size=pairwise_batch_size,
                limit_frames=limit_val_frames,
            )
            row = {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss_mean": avg_loss,
                "val_top1": val["top1"],
                "val_mrr": val["mrr"],
                "wall_s": time.perf_counter() - t0,
            }
            history.append(row)
            logger.info(
                f"epoch {epoch}/{epochs} | lr={lr_now:.2e} | "
                f"train_loss={avg_loss:.5f} | "
                f"val top1={val['top1'] * 100:.2f}% mrr={val['mrr']:.4f}"
            )
            ck = output_dir / f"epoch_{epoch:03d}.pth"
            torch.save(
                {"model_state_dict": net.state_dict(), "epoch": epoch, "config": cfg.to_dict()},
                ck,
            )
            if val["mrr"] > best_mrr:
                best_mrr = val["mrr"]
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "epoch": epoch,
                        "metrics": val,
                        "train_args": train_kwargs,
                    },
                    best_path,
                )
                logger.info(f"saved new best (mrr={best_mrr:.4f}) -> {best_path}")

    hist_path = output_dir / "history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"wrote {hist_path}")

    if not best_path.exists():
        raise RuntimeError(
            "Training finished without a validated checkpoint — "
            "check that the validation cache produced any frames."
        )
    return best_path
