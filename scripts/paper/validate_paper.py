"""
Reproduce public A9 matching numbers and verify UTC reference numbers when
institutional HDF5 caches and weights are available.

Example
-------
pixi run python scripts/paper/validate_paper.py \
    --hdf5 ../datasets/utc_dataset4/hdf5_cache/utc_test.h5 \
    --models crlite crlite_2dpe crlite_vit_exp1 crlite_vit_exp3 calibrefine \
    --site utc4 \
    --output docs/evidence/standalone_paper_validation_utc4.json

If --hdf5 is omitted, the script falls back to the held-out test split:
the repo-local datasets/a9_dataset_r02_s01/hdf5_cache/a9_r02_s01_test.h5 for the
A9 site, or ../datasets/utc_dataset{3,4}/hdf5_cache/utc_test.h5 for the UTC
partner sites. UTC datasets are
institutional artefacts and are not shipped in the public repository.
This matches what the paper reports — the train pipeline reserves
utc_val.h5 for early-stopping / best-checkpoint selection, and
utc_test.h5 is never seen during training. Pass `--hdf5 .../utc_val.h5`
explicitly if you want to inspect what the trainer was watching.

Runs the 5 models claimed in IEEE Sensors Letters Table I and the PE
ablation: calibrefine (pairwise baseline) + crlite / crlite_2dpe
(ResNet variants) + crlite_vit_exp1 / crlite_vit_exp3 (ViT variants).

The sin3d ablation is a deliberate negative result (Obs. 1) and is not
shipped; the GNN follow-up (crlite_vit_exp4) is research-foundation
code kept in the upstream research tree and is also not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Jetson-aware install hint if torch is missing (must run before `import torch`)
from xcalib.utils.torch_check import ensure_torch  # noqa: E402
ensure_torch()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from tqdm import tqdm  # noqa: E402

from xcalib import Matcher  # noqa: E402
from xcalib.data import (  # noqa: E402
    UTCFrameLoader,
    prepare_frame,
)
from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402
from xcalib.utils.metrics import (  # noqa: E402
    MatchingMetricsAccumulator,
    format_summary_table,
)


PAPER_MODELS = (
    "crlite",
    "crlite_2dpe",
    "crlite_vit_exp1",
    "crlite_vit_exp3",
    "calibrefine",
)

VALIDATABLE_MODELS = PAPER_MODELS

SITE_DEFAULTS = {
    # UTC caches stay in the sibling lab tree (institutional data, not shipped).
    "utc4": "../datasets/utc_dataset4/hdf5_cache/utc_test.h5",
    "utc3": "../datasets/utc_dataset3/hdf5_cache/utc_test.h5",
    # A9 caches ship inside this repo (and on the public HF dataset repo).
    "a9_dataset_r02_s01": "datasets/a9_dataset_r02_s01/hdf5_cache/a9_r02_s01_test.h5",
}


def _report_hdf5_path(site: str, hdf5_path: Path) -> str:
    """Return a portable evidence path instead of a machine-local absolute path."""
    if site in {"utc3", "utc4"}:
        dataset = "utc_dataset3" if site == "utc3" else "utc_dataset4"
        return f"<institutional_utc_root>/{dataset}/hdf5_cache/utc_test.h5"
    try:
        return hdf5_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return SITE_DEFAULTS[site]


def _resolve_h5(site: str, override: Path | None) -> Path:
    if override is not None:
        return override
    fallback = REPO_ROOT / SITE_DEFAULTS[site]
    return fallback.resolve()


def _config_path(model: str, site: str) -> Path:
    return default_config_path(model, site)


def _filter_match_matrix(
    match_matrix: np.ndarray, kept_2d: np.ndarray, kept_3d: np.ndarray
) -> np.ndarray:
    if match_matrix.size == 0 or kept_2d.size == 0 or kept_3d.size == 0:
        return np.zeros((kept_2d.size, kept_3d.size), dtype=bool)
    return match_matrix[np.ix_(kept_2d, kept_3d)].astype(bool)


def evaluate_one_model(
    model: str,
    site: str,
    hdf5_path: Path,
    device: str,
    limit_frames: int | None = None,
) -> dict:
    cfg_path = _config_path(model, site)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    cfg = load_yaml(cfg_path)
    weights_rel = cfg.get("weights_path")
    if weights_rel is None:
        raise ValueError(f"weights_path missing from {cfg_path}")
    weights_path = (REPO_ROOT / weights_rel).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path} (place the expected file under checkpoints/ "
            "or pass --weights through a custom Matcher workflow)"
        )

    matcher = Matcher.from_pretrained(
        model=model,
        weights=weights_path,
        config=cfg,
        device=device,
    )

    prepare_cfg = matcher.prepare_cfg
    acc = MatchingMetricsAccumulator()

    with UTCFrameLoader(hdf5_path) as loader:
        for i, raw in enumerate(tqdm(loader, desc=f"{model}/{site}", leave=False)):
            if limit_frames is not None and i >= limit_frames:
                break
            try:
                frame, kept_2d, kept_3d = prepare_frame(
                    image=raw.image,
                    point_cloud=raw.point_cloud,
                    bboxes_2d=raw.bboxes_2d,
                    bboxes_3d=raw.bboxes_3d,
                    cfg=prepare_cfg,
                    images=raw.images,
                    camera_per_det=raw.camera_per_det,
                )
            except Exception as e:
                logger.warning(f"[{model}/{site}] cropping failed at frame {raw.frame_key}: {e}")
                continue

            if frame.crops_2d.numel() == 0 or frame.crops_3d.numel() == 0:
                continue

            gt = _filter_match_matrix(raw.match_matrix, kept_2d, kept_3d)

            scores, fwd_ms = matcher.match_frame(frame)
            acc.update(scores, gt, latency_ms=fwd_ms)

    return acc.summary()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce paper Table I numbers from the standalone package."
    )
    parser.add_argument(
        "--site", choices=("utc3", "utc4", "a9_dataset_r02_s01"), required=True,
        help="Which site to evaluate on (UTC partner sites or the public A9 r02_s01 set)",
    )
    parser.add_argument("--hdf5", type=Path, default=None, help="Override default HDF5 path")
    parser.add_argument(
        "--models", nargs="+", default=list(PAPER_MODELS),
        help=f"Models to evaluate (default = paper-claimed set: {', '.join(PAPER_MODELS)})",
    )
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--limit-frames", type=int, default=None,
                        help="Optional cap on frames for quick smoke runs")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON report path")
    args = parser.parse_args()

    h5 = _resolve_h5(args.site, args.hdf5)
    if not h5.exists():
        logger.error(f"HDF5 cache not found: {h5}")
        sys.exit(1)
    logger.info(f"Loading frames from {h5}")

    rows = []
    for model in args.models:
        if model not in VALIDATABLE_MODELS:
            logger.warning(f"Skipping unknown model: {model}")
            continue

        t0 = time.perf_counter()
        try:
            summary = evaluate_one_model(
                model, args.site, h5, args.device, limit_frames=args.limit_frames
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
        except Exception:
            logger.exception(f"Evaluation failed for {model}")
            continue

        elapsed = time.perf_counter() - t0
        summary["wall_seconds"] = elapsed
        logger.info(
            f"{model}/{args.site}: top1={summary['top1']*100:.2f}% "
            f"top3={summary['top3']*100:.2f}% mrr={summary['mrr']:.4f} "
            f"lat_mean={summary['latency_ms_mean']:.3f}ms "
            f"(wall={elapsed:.1f}s)"
        )
        rows.append((model, summary))

        # Free GPU memory between models so we can run everything in one shot.
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + format_summary_table(rows))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "site": args.site,
                    "hdf5": _report_hdf5_path(args.site, h5),
                    "device": args.device,
                    "results": {name: summary for name, summary in rows},
                },
                f,
                indent=2,
            )
        logger.info(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
