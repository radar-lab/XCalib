"""
Train (from scratch) or fine-tune paper matchers from HDF5 caches — thin shim.

The training loop lives in `xcalib.engine.trainer` (also reachable as
``xcalib.train(...)`` and ``Matcher.train(...)``); this script keeps the
historical flags so existing pixi tasks / lab notes stay valid.

Works with any cache that follows ``docs/hdf5-format.md`` — the public A9
r02_s01 caches as well as the partner UTC3 / UTC4 caches. Uses the same
cropping path as `validate_paper.py`, so a freshly trained ``best.pth`` is
directly comparable to the shipped checkpoints.

Supported models::

    crlite, crlite_2dpe — two-stage cosine + dense Stage~2 logits (CE).

    crlite_vit_exp1 — cosine-only (Stage~2 weight forced to zero).

    crlite_vit_exp3 — cosine with position encoding (centers during training).

The pairwise ``calibrefine`` baseline is not supported (different objective).

Examples::

    # Fine-tune shipped weights on your own cache (default 10 epochs):
    pixi run train-hdf5 -- --model crlite \
        --train-hdf5 /path/utc_train.h5 --val-hdf5 /path/utc_val.h5 \
        --weights checkpoints/crlite_utc4_best.pth --output-dir runs/ft

    # From-scratch training on the public A9 caches (paper-style recipe:
    # Adam, 5-epoch LR warmup, cosine decay, 100 epochs, best by val MRR):
    pixi run train-hdf5 -- --model crlite --site a9_dataset_r02_s01 \
        --train-hdf5 datasets/a9_dataset_r02_s01/hdf5_cache/a9_r02_s01_train.h5 \
        --val-hdf5 datasets/a9_dataset_r02_s01/hdf5_cache/a9_r02_s01_val.h5 \
        --epochs 100 --warmup-epochs 5 --optimizer adam --scheduler cosine \
        --output-dir runs/crlite_a9_scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xcalib.utils.torch_check import ensure_torch  # noqa: E402

ensure_torch()

from xcalib.engine.trainer import TRAINABLE, train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train matchers from HDF5 caches.")
    p.add_argument("--model", choices=TRAINABLE, required=True)
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (defaults to the packaged cfg/<model>_<site>.yaml)")
    p.add_argument("--site", default=None,
                   help="Site for default-config resolution (utc3, utc4, "
                        "a9_dataset_r02_s01; default utc4)")
    p.add_argument("--train-hdf5", required=True,
                   help="Local .h5 cache or Hub site name")
    p.add_argument("--val-hdf5", required=True,
                   help="Local .h5 cache or Hub site name")
    p.add_argument("--output-dir", type=Path, default=Path("runs/train_hdf5"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw",
                   help="adamw (default, fine-tune) or adam (paper from-scratch recipe)")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs (paper recipe uses 5)")
    p.add_argument("--scheduler", choices=("none", "cosine"), default="none",
                   help="LR decay after warmup (paper recipe uses cosine)")
    p.add_argument("--min-lr", type=float, default=1e-5,
                   help="Floor LR for the cosine scheduler")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--stage1-weight", type=float, default=1.0)
    p.add_argument("--stage2-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--weights", type=Path, default=None, help="Optional checkpoint to warm-start")
    p.add_argument("--limit-train-frames", type=int, default=None)
    p.add_argument("--limit-val-frames", type=int, default=None)
    p.add_argument("--pairwise-batch-size", type=int, default=64)
    args = p.parse_args()

    best = train(
        args.model,
        args.train_hdf5,
        args.val_hdf5,
        config=args.config,
        site=args.site,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        warmup_epochs=args.warmup_epochs,
        scheduler=args.scheduler,
        min_lr=args.min_lr,
        temperature=args.temperature,
        label_smoothing=args.label_smoothing,
        stage1_weight=args.stage1_weight,
        stage2_weight=args.stage2_weight,
        weights=args.weights,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        limit_train_frames=args.limit_train_frames,
        limit_val_frames=args.limit_val_frames,
        pairwise_batch_size=args.pairwise_batch_size,
    )
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
