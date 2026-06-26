"""
Build TensorRT engines from the ONNX graphs that `scripts/paper/export_onnx.py`
produced — thin CLI shim.

The trtexec discovery and the per-model {min,opt,max}Shapes profiles live
in `xcalib.engine.trt` (also reachable as ``Matcher.build("trt")``
and ``xcalib build-trt``); this script keeps the historical flags so
existing pixi tasks stay valid.

Usage
-----
Build FP16 engines for every shipped model (default; matches what the
partner needs for the paper):

    pixi run build-trt --all

Build a single model with explicit precision:

    pixi run build-trt --model crlite --precision fp16
    pixi run build-trt --model crlite_vit_exp3 --precision fp32

Outputs land in ``engines/<model>_<site>/<stage>.<precision>.engine``.
trtexec prints its own latency summary at the end of each build; pipe
stdout to a file (``--log-dir reports/trt_logs``) to keep that for the
supplemental.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.engine.trt import (  # noqa: E402
    PRECISION_FLAGS,
    build_one,
    find_trtexec,
    plan_for_model,
    resolve_site_dir,
)
from xcalib.models.registry import list_models  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines from exported ONNX graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--model", choices=list_models(), default=None,
                   help="Single model to build. Mutually exclusive with --all.")
    g.add_argument("--all", action="store_true",
                   help="Build engines for every shipped model "
                        "(crlite, crlite_2dpe, crlite_vit_exp1, "
                        "crlite_vit_exp3, calibrefine).")
    parser.add_argument("--site", choices=("utc3", "utc4"), default="utc4",
                        help="Which site's config to use for shape derivation.")
    parser.add_argument("--precision", choices=tuple(PRECISION_FLAGS), default="fp16",
                        help="trtexec precision flag (fp16 = recommended on Thor).")
    parser.add_argument("--onnx-root", type=Path, default=REPO_ROOT / "onnx",
                        help="Directory containing per-model ONNX subdirs.")
    parser.add_argument("--engine-root", type=Path, default=REPO_ROOT / "engines",
                        help="Where to write *.engine files.")
    parser.add_argument("--trtexec", default=None,
                        help="Override path to trtexec. Defaults to the system "
                             "trtexec or /usr/src/tensorrt/bin/trtexec.")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="If set, redirect each trtexec stdout/stderr to "
                             "<log-dir>/<model>_<stage>.<precision>.log.")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra arguments forwarded verbatim to trtexec "
                             "(e.g. `-- --workspace=8192 --avgRuns=200`).")
    args = parser.parse_args()

    # `extra` is everything after `--`; argparse keeps the literal `--` if
    # present, so strip it.
    extra_args = list(args.extra)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if args.all:
        models = list(list_models())
    elif args.model is not None:
        models = [args.model]
    else:
        models = ["crlite"]

    try:
        trtexec = find_trtexec(args.trtexec)
    except FileNotFoundError as e:
        # Don't dump a Python traceback for a "tool not installed" condition
        # -- the message is the actionable part on Jetson.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"[trtexec] {trtexec}")

    failures: list[str] = []
    for model_name in models:
        try:
            cfg_path = default_config_path(model_name, args.site)
        except FileNotFoundError:
            print(f"[skip] no config for {model_name}_{args.site}")
            failures.append(model_name)
            continue
        cfg = load_yaml(cfg_path)
        try:
            plans = plan_for_model(model_name, cfg)
        except KeyError as e:
            # crlite_vit_exp4 has no TRT-buildable graph (dynamic ScatterND);
            # --all therefore silently covers the 5 paper models only.
            print(f"[skip] {e}")
            continue
        # Per-site subdirs keep UTC3 / UTC4 weights from overwriting each other.
        onnx_dir = resolve_site_dir(args.onnx_root, model_name, args.site)
        engine_dir = args.engine_root / f"{model_name}_{args.site}"

        for plan in plans:
            log_path = (
                args.log_dir / f"{model_name}_{args.site}_{plan.engine_stem}.{args.precision}.log"
                if args.log_dir is not None
                else None
            )
            rc = build_one(
                trtexec=trtexec,
                onnx_dir=onnx_dir,
                engine_dir=engine_dir,
                plan=plan,
                precision=args.precision,
                extra_args=extra_args,
                log_path=log_path,
            )
            if rc != 0:
                failures.append(f"{model_name}/{plan.engine_stem}")

    print()
    if failures:
        print(f"[done] {len(failures)} build(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("[done] all engines built successfully.")


if __name__ == "__main__":
    main()
