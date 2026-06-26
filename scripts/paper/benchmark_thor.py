"""
PyTorch latency benchmark for NVIDIA Jetson AGX Thor (or any GPU).

This is a *PyTorch-only* synthetic-input benchmark. It does NOT need the
HDF5 datasets, so it is the right tool to run on a delivery device to
back up the paper's real-time-feasibility claim. The numbers it reports
are the per-frame forward-pass time of each shipped model (model.eval()
+ no_grad + torch.cuda.synchronize); the real Thor TensorRT numbers come
from a separate trtexec step which this script prints at the end.

Usage
-----
Benchmark every shipped model with the default detection counts (N=8, M=12):

    pixi run python scripts/paper/benchmark_thor.py --all \
        --output reports/thor_latency.json \
        --figure reports/thor_latency.png

Benchmark a single model with explicit weights:

    pixi run python scripts/paper/benchmark_thor.py --model crlite_vit_exp3 \
        --weights checkpoints/crlite_vit_exp3_utc4_best.pth \
        --config configs/crlite_vit_exp3_utc4.yaml

The default detection counts (N image crops, M lidar crops per frame)
match what we observe on the UTC test split (~8 cameras / ~12 lidar
boxes per frame). Override with --detections N M for sensitivity sweeps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Jetson-aware install hint if torch is missing (must run before `import torch`)
from xcalib.utils.torch_check import ensure_torch  # noqa: E402
ensure_torch()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.engine.wrappers import FrameData, make_wrapper  # noqa: E402
from xcalib.models.registry import build_model, list_models  # noqa: E402
from xcalib.utils.io import default_config_path, resolve_device  # noqa: E402
from xcalib.utils.metrics import (  # noqa: E402
    _latency_stats,
    format_latency_table,
)


# Models we ship and want to compare on Thor. Order matters for the figure.
DEFAULT_MODELS = (
    "crlite_vit_exp1",
    "crlite_vit_exp3",
    "crlite_2dpe",
    "crlite",
    "calibrefine",  # pairwise baseline; intentionally last (much slower)
)


def _make_frame(cfg, device, N: int, M: int) -> FrameData:
    """Build a synthetic FrameData with random crops / point clouds."""
    crop_size = int(cfg.get("crop_size", 32))
    pc_size = int(cfg.get("point_cloud_size", 1024))
    return FrameData(
        crops_2d=torch.randn(N, 3, crop_size, crop_size, device=device),
        crops_3d=torch.randn(M, pc_size, 3, device=device),
        bboxes_2d=torch.tensor(
            [[0.0, 0.0, float(crop_size), float(crop_size)]] * N,
            device=device,
            dtype=torch.float32,
        ),
        bbox_centers_3d=torch.zeros(M, 3, device=device, dtype=torch.float32),
    )


def _bench(wrapper, frame, warmup: int, iters: int, sync_cuda: bool) -> list[float]:
    """Run warmup + timed iterations; return per-iter latency in ms."""
    for _ in range(warmup):
        wrapper.predict_matching_matrix(frame)
        if sync_cuda:
            torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        if sync_cuda:
            torch.cuda.synchronize()
        _, ms = wrapper.predict_matching_matrix(frame)
        if sync_cuda:
            torch.cuda.synchronize()
        samples.append(ms)
    return samples


def _bench_one_model(
    model_name: str,
    cfg_path: Path,
    weights_path: Path | None,
    device: torch.device,
    *,
    N: int,
    M: int,
    warmup: int,
    iters: int,
    fp16: bool,
) -> dict:
    """Build + benchmark one model, return per-precision summary dict."""
    cfg = load_yaml(cfg_path)
    if weights_path is None:
        rel = cfg.get("weights_path")
        if rel is not None:
            weights_path = (REPO_ROOT / rel).resolve()

    model = build_model(model_name, cfg)
    if weights_path and Path(weights_path).exists():
        model.load_weights(weights_path, strict=False)
        loaded = True
    else:
        logger.warning(
            f"[{model_name}] weights not found at {weights_path}; "
            f"timing with random weights (perf still meaningful)."
        )
        loaded = False
    model.to(device)
    model.eval()

    wrapper = make_wrapper(
        model_name, model, device=device,
        point_cloud_size=int(cfg.get("point_cloud_size", 1024)),
        top_k=int(cfg.get("top_k", 5)),
    )

    frame = _make_frame(cfg, device, N, M)
    sync_cuda = device.type == "cuda"

    fp32_samples = _bench(wrapper, frame, warmup, iters, sync_cuda)
    summary = {
        "model": model_name,
        "config": str(cfg_path),
        "weights": str(weights_path) if weights_path else None,
        "weights_loaded": loaded,
        "N": N,
        "M": M,
        "warmup": warmup,
        "iters": iters,
        "device": str(device),
        "fp32": _latency_stats(fp32_samples),
    }

    if fp16 and device.type == "cuda":
        try:
            model.half()
            frame_fp16 = FrameData(
                crops_2d=frame.crops_2d.half(),
                crops_3d=frame.crops_3d.half(),
                bboxes_2d=(
                    frame.bboxes_2d.half() if frame.bboxes_2d is not None else None
                ),
                bbox_centers_3d=(
                    frame.bbox_centers_3d.half()
                    if frame.bbox_centers_3d is not None
                    else None
                ),
            )
            fp16_samples = _bench(wrapper, frame_fp16, warmup, iters, sync_cuda)
            summary["fp16"] = _latency_stats(fp16_samples)
        except Exception as e:
            logger.warning(f"[{model_name}] FP16 benchmark failed: {e}")
            summary["fp16_error"] = str(e)

    # Free GPU memory before the next model
    del model, wrapper, frame
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _resolve_config(model_name: str, site: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return default_config_path(model_name, site)


def _save_figure(rows: list[dict], out_path: Path) -> None:
    """Save a matplotlib bar chart of per-model latency (FP32, log scale)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed in this environment; "
            "skipping figure (JSON report still written). "
            "Install with `pixi add matplotlib` to get plots."
        )
        return

    names = [r["model"] for r in rows]
    means = [r["fp32"]["latency_ms_mean"] for r in rows]
    p95s = [r["fp32"]["latency_ms_p95"] for r in rows]
    p99s = [r["fp32"]["latency_ms_p99"] for r in rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(names)), 5.0))
    ax.bar(x, means, width=0.6, color="#3a86ff", label="mean (FP32)")
    # Overlay p95 / p99 as horizontal ticks above the bar
    ax.scatter(x, p95s, marker="_", s=180, color="#ffbe0b",
               linewidth=2.5, label="p95", zorder=3)
    ax.scatter(x, p99s, marker="_", s=180, color="#fb5607",
               linewidth=2.5, label="p99", zorder=3)

    # Annotate fps above each bar
    for xi, m in zip(x, means):
        fps = 1000.0 / m if m > 0 else 0.0
        ax.text(xi, m * 1.05, f"{fps:.0f} fps", ha="center",
                va="bottom", fontsize=9, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Per-frame latency (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title(
        f"xcalib — per-frame matching latency "
        f"(N={rows[0]['N']}, M={rows[0]['M']}, iters={rows[0]['iters']})"
    )
    ax.grid(True, axis="y", alpha=0.3, which="both")
    # Highlight the real-time budget for a 10 Hz pipeline (100 ms)
    ax.axhline(100.0, linestyle="--", color="#888",
               label="10 Hz real-time budget (100 ms)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Wrote latency figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic-input latency benchmark for shipped models."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", choices=list_models(), default=None)
    group.add_argument("--all", action="store_true",
                       help=f"Benchmark every shipped model: {', '.join(DEFAULT_MODELS)}")
    parser.add_argument("--site", choices=("utc3", "utc4"), default="utc4",
                        help="Which site's config + weights to use (default utc4)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Override weights path (single-model mode only)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Override config path (single-model mode only)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detections", nargs=2, type=int, default=[8, 12],
                        metavar=("N", "M"),
                        help="Number of 2D / 3D detections per frame")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--no-fp16", action="store_true",
                        help="Skip FP16 timing")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--figure", type=Path, default=None,
                        help="Write matplotlib PNG figure to this path "
                             "(requires --all; matplotlib must be installed)")
    args = parser.parse_args()

    # Resolve model list
    if args.all:
        model_names = list(DEFAULT_MODELS)
    elif args.model is not None:
        model_names = [args.model]
    else:
        model_names = ["crlite"]  # default single-model behaviour
        logger.info("No --model / --all flag — defaulting to crlite")

    device = resolve_device(args.device)
    N, M = args.detections
    logger.info(f"Device={device} | N={N}, M={M} | warmup={args.warmup}, "
                f"iters={args.iters} | site={args.site}")

    rows: list[dict] = []
    for name in model_names:
        cfg_path = _resolve_config(name, args.site, args.config)
        if not cfg_path.exists():
            logger.error(f"Config not found: {cfg_path}")
            continue
        weights_arg = args.weights if not args.all else None
        try:
            summary = _bench_one_model(
                name, cfg_path, weights_arg, device,
                N=N, M=M, warmup=args.warmup, iters=args.iters,
                fp16=not args.no_fp16,
            )
        except Exception:
            logger.exception(f"Latency benchmark failed for {name}")
            continue

        s = summary["fp32"]
        logger.info(
            f"{name}/FP32: mean={s['latency_ms_mean']:.3f}ms "
            f"p50={s['latency_ms_p50']:.3f}ms p95={s['latency_ms_p95']:.3f}ms "
            f"p99={s['latency_ms_p99']:.3f}ms "
            f"throughput={s['throughput_fps_mean']:.1f}fps"
        )
        if "fp16" in summary:
            f = summary["fp16"]
            logger.info(
                f"{name}/FP16: mean={f['latency_ms_mean']:.3f}ms "
                f"p95={f['latency_ms_p95']:.3f}ms "
                f"throughput={f['throughput_fps_mean']:.1f}fps"
            )
        rows.append(summary)

    if not rows:
        logger.error("No models benchmarked; exiting.")
        sys.exit(1)

    table_rows = [(r["model"], r["fp32"]) for r in rows]
    print("\nFP32 latency summary")
    print(format_latency_table(table_rows))

    has_fp16 = any("fp16" in r for r in rows)
    if has_fp16:
        fp16_rows = [(r["model"], r["fp16"]) for r in rows if "fp16" in r]
        if fp16_rows:
            print("\nFP16 latency summary")
            print(format_latency_table(fp16_rows))

    # Write JSON report
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "device": str(device),
            "site": args.site,
            "N": N,
            "M": M,
            "warmup": args.warmup,
            "iters": args.iters,
            "models": rows,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote latency JSON to {args.output}")

    # Write figure (only meaningful for multi-model)
    if args.figure is not None and len(rows) >= 2:
        _save_figure(rows, args.figure)
    elif args.figure is not None:
        logger.warning("--figure ignored (need at least 2 models)")

    # Hints for TensorRT next steps -- all driven through pixi tasks so
    # users don't need to remember trtexec's flags or its install path.
    print("\nNext step on the Jetson: build TensorRT engines from ONNX")
    print("  pixi run export-onnx-all     # writes onnx/<model>/*.onnx")
    print("  pixi run build-trt-all       # writes engines/<model>/*.fp16.engine")
    print("                               # logs land in reports/trt_logs/")
    print("  # Single-model variant (defaults to fp16):")
    print("  pixi run build-trt --model crlite --precision fp16")


if __name__ == "__main__":
    main()
