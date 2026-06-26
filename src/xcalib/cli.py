"""
`xcalib` console entry point (installed by the wheel).

Subcommands mirror the public package workflow:

    xcalib pull-weights --model crlite --site a9_dataset_r02_s01 --out checkpoints/
    xcalib export-onnx  --model crlite --site a9_dataset_r02_s01 --output onnx/crlite_a9
    xcalib build-trt    --model crlite --site a9_dataset_r02_s01 --onnx-dir onnx/crlite_a9
    xcalib demo         --model crlite --site a9_dataset_r02_s01
    xcalib pull-dataset --site a9_dataset_r02_s01 --split test
    xcalib version

Batch scripts for paper validation, HDF5 training, and Thor benchmarks stay in
the repository's `scripts/` directory. This CLI covers common installed-wheel
workflows without requiring a checkout.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .utils.torch_check import ensure_torch

ensure_torch()

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _matcher_from_args(args: argparse.Namespace):
    """from_pretrained() with the CLI's weights/config/site/repo flags."""
    from .engine.matcher import Matcher

    return Matcher.from_pretrained(
        model=args.model,
        weights=args.weights,
        config=args.config,
        device=getattr(args, "device", None),
        site=args.site,
        repo_id=getattr(args, "repo", None),
        revision=getattr(args, "revision", None),
        token=getattr(args, "token", None),
    )


def _synthetic_frame(n_image: int, n_lidar: int, seed: int):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    points = rng.uniform(
        low=[-20.0, -20.0, -2.0], high=[50.0, 20.0, 4.0], size=(12000, 3)
    ).astype(np.float32)

    cx = rng.uniform(80, 1200, n_image)
    cy = rng.uniform(80, 640, n_image)
    w = rng.uniform(60, 200, n_image)
    h = rng.uniform(60, 200, n_image)
    bboxes_2d = np.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
    ).astype(np.float32)

    cx3 = rng.uniform(5.0, 30.0, n_lidar)
    cy3 = rng.uniform(-10.0, 10.0, n_lidar)
    cz3 = rng.uniform(-1.0, 1.0, n_lidar)
    dx = rng.uniform(1.5, 4.5, n_lidar)
    dy = rng.uniform(1.2, 2.5, n_lidar)
    dz = rng.uniform(1.2, 2.5, n_lidar)
    bboxes_3d = np.stack(
        [cx3 - dx / 2, cy3 - dy / 2, cz3 - dz / 2,
         cx3 + dx / 2, cy3 + dy / 2, cz3 + dz / 2],
        axis=1,
    ).astype(np.float32)
    return image, points, bboxes_2d, bboxes_3d


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_export_onnx(args: argparse.Namespace) -> int:
    matcher = _matcher_from_args(args)
    result = matcher.build("onnx", output_dir=args.output, device=args.export_device)
    print(f"artifacts: {[str(p) for p in result.artifacts]}")
    if result.parity:
        worst = max(result.parity.values())
        print(f"parity   : max|torch-onnx| = {worst:.3e} "
              f"({'OK' if result.parity_ok else 'CHECK FAILED'})")
        if not result.parity_ok:
            return 1
    return 0


def cmd_build_trt(args: argparse.Namespace) -> int:
    from .engine.trt import build_engines
    from .utils.config import load_yaml
    from .utils.io import default_config_path

    cfg_path = args.config or default_config_path(args.model, args.site)
    cfg = load_yaml(cfg_path)
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else Path.cwd() / "onnx" / f"{args.model}_{args.site}"
    engine_dir = Path(args.engine_dir) if args.engine_dir else Path.cwd() / "engines" / f"{args.model}_{args.site}"
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        result = build_engines(
            args.model,
            cfg,
            onnx_dir=onnx_dir,
            engine_dir=engine_dir,
            precision=args.precision,
            trtexec=args.trtexec,
            extra_args=extra,
            log_dir=args.log_dir,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"engines: {[str(p) for p in result.artifacts]}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    matcher = _matcher_from_args(args)
    image, points, b2, b3 = _synthetic_frame(args.n_image, args.n_lidar, args.seed)
    result = matcher.match(
        image=image, point_cloud=points, bboxes_2d=b2, bboxes_3d=b3,
        top_k=3, return_latency=True,
    )
    print(f"model           : {result.model}")
    print(f"device          : {result.device}")
    print(f"similarity shape: {result.similarity.shape}")
    print(f"top_indices     : {result.top_indices}")
    print(f"latency_ms      : {result.latency_ms:.3f}")
    for m in result.matches[:10]:
        print(f"  {m[0]:3d} -> {m[1]:3d}   {m[2]:+.4f}")
    return 0


def cmd_pull_weights(args: argparse.Namespace) -> int:
    from . import hub

    weights, config = hub.resolve_pretrained(
        args.model, args.site,
        repo_id=args.repo, revision=args.revision, token=args.token,
    )
    if args.out is not None:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        w_dest = out / f"{args.model}_{args.site}_best.pth"
        c_dest = out / f"{args.model}_{args.site}.yaml"
        shutil.copy2(weights, w_dest)
        shutil.copy2(config, c_dest)
        print(f"weights -> {w_dest}")
        print(f"config  -> {c_dest}")
    else:
        print(f"weights (cache): {weights}")
        print(f"config  (cache): {config}")
    return 0


def cmd_pull_dataset(args: argparse.Namespace) -> int:
    from .hub import datasets

    path = datasets.dataset_path(
        args.site, args.split,
        repo_id=args.repo, revision=args.revision, token=args.token,
    )
    if args.out is not None:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / Path(datasets.dataset_spec(args.site).hub_path(args.split)).name
        shutil.copy2(path, dest)
        print(f"dataset -> {dest}")
    else:
        print(f"dataset (cache): {path}")
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    from . import PROTOCOL_VERSION, __version__

    print(f"xcalib {__version__} (input protocol v{PROTOCOL_VERSION})")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_pretrained_flags(p: argparse.ArgumentParser, *, need_device: bool = True) -> None:
    from .models.registry import list_models

    p.add_argument("--model", choices=list_models(), default="crlite")
    p.add_argument("--site", default="a9_dataset_r02_s01",
                   help="Site whose weights/config to use. Public Hub downloads "
                        "currently support a9_dataset_r02_s01.")
    p.add_argument("--weights", default=None,
                   help="Local .pth, save_pretrained() dir, or hf:// URI. "
                        "Omit to download from the Hub by (model, site).")
    p.add_argument("--config", default=None,
                   help="Local YAML or hf:// URI (defaults to the Hub/config "
                        "convention or the packaged reference config).")
    p.add_argument("--repo", default=None,
                   help="Hub repo id override for advanced deployments.")
    p.add_argument("--revision", default=None, help="Hub revision/tag/commit.")
    p.add_argument("--token", default=None,
                   help="Hub token override (default: cached login/environment).")
    if need_device:
        p.add_argument("--device", default=None, help="cuda | cpu | auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xcalib",
        description="Camera-LiDAR cross-modal matching for edge devices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # export-onnx
    p = sub.add_parser("export-onnx", help="Export a model (any weights) to ONNX.")
    _add_pretrained_flags(p)
    p.add_argument("--output", default=None,
                   help="Output dir (default ./onnx/<model>/).")
    p.add_argument("--export-device", default="cpu",
                   help="Device used for tracing (cpu recommended).")
    p.set_defaults(func=cmd_export_onnx)

    # build-trt
    p = sub.add_parser("build-trt", help="Build TensorRT engines from ONNX graphs.")
    from .models.registry import list_models
    p.add_argument("--model", choices=list_models(), default="crlite")
    p.add_argument("--site", default="a9_dataset_r02_s01",
                   help="Site used for packaged config lookup.")
    p.add_argument("--config", default=None, help="YAML for shape derivation "
                   "(defaults to the packaged configs/<model>_<site>.yaml).")
    p.add_argument("--onnx-dir", default=None,
                   help="Where the ONNX graphs live (default ./onnx/<model>_<site>/).")
    p.add_argument("--engine-dir", default=None,
                   help="Where engines land (default ./engines/<model>_<site>/).")
    p.add_argument("--precision", choices=("fp32", "fp16", "best"), default="fp16")
    p.add_argument("--trtexec", default=None, help="Explicit trtexec path.")
    p.add_argument("--log-dir", default=None, help="Save trtexec logs here.")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra args forwarded to trtexec (after --).")
    p.set_defaults(func=cmd_build_trt)

    # demo
    p = sub.add_parser("demo", help="End-to-end match on a synthetic frame.")
    _add_pretrained_flags(p)
    p.add_argument("--n-image", type=int, default=3)
    p.add_argument("--n-lidar", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_demo)

    # pull-weights
    p = sub.add_parser("pull-weights", help="Download (weights, config) from the Hub.")
    _add_pretrained_flags(p, need_device=False)
    p.add_argument("--out", default=None,
                   help="Copy into this directory with canonical names "
                        "(default: print the HF cache paths).")
    p.set_defaults(func=cmd_pull_weights)

    from .hub.datasets import DATASETS, SPLITS

    # pull-dataset
    p = sub.add_parser("pull-dataset", help="Download one split of a site's HDF5 cache.")
    p.add_argument("--site", choices=sorted(DATASETS), required=True)
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument("--repo", default=None, help="Dataset repo override.")
    p.add_argument("--revision", default=None, help="Hub revision/tag/commit.")
    p.add_argument("--token", default=None)
    p.add_argument("--out", default=None,
                   help="Copy into this directory (default: print the cache path).")
    p.set_defaults(func=cmd_pull_dataset)

    # version
    p = sub.add_parser("version", help="Print package + protocol version.")
    p.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        rc = args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        rc = 130
    except Exception as e:  # surface a clean one-liner, log the rest
        logger.exception(e)
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
