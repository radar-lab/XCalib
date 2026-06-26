"""
Benchmark already-built TensorRT engines via ``trtexec --loadEngine``.

This is the partner-side counterpart to ``pixi run benchmark`` (which
times the PyTorch eager-mode forward on Thor). It runs trtexec's own
inference loop -- the canonical TensorRT-FP16 latency we cite in the
paper supplemental -- on every engine under ``engines/<model>/`` and
parses the "GPU Compute Time" block out of trtexec's stdout into a
single JSON report.

Usage
-----
After ``pixi run build-trt-all`` has populated ``engines/``::

    pixi run bench-trt                  # all engines, default settings
    pixi run bench-trt --model crlite   # single model
    pixi run bench-trt --iterations 1000 --warmup 200
    pixi run bench-trt -- --useCudaGraph   # extra trtexec flags after `--`

Outputs:
    docs/evidence/thor_trt_latency.json   structured per-engine summary
    reports/trt_logs/<model>_<stage>.<precision>.bench.log   raw trtexec stdout
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.models.registry import list_models  # noqa: E402

# Same shape-profile + trtexec-locator logic as build_trt.py, so the two
# scripts always agree on the per-model dynamic-shape spec.
from xcalib.engine.trt import find_trtexec, plan_for_model  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402


# ---------------------------------------------------------------------------
# trtexec output parsing
# ---------------------------------------------------------------------------

# trtexec prints a "GPU Compute Time" performance summary like:
#
#   [I] GPU Compute Time: min = 0.612305 ms, max = 0.731445 ms,
#                          mean = 0.660847 ms, median = 0.660156 ms,
#                          percentile(90%) = 0.685547 ms,
#                          percentile(95%) = 0.692871 ms,
#                          percentile(99%) = 0.702637 ms
#
# (line-broken depending on the trtexec version). We capture each metric.
_PERF_FIELDS = {
    "latency_ms_min":          re.compile(r"min\s*=\s*([\d.]+)\s*ms"),
    "latency_ms_max":          re.compile(r"max\s*=\s*([\d.]+)\s*ms"),
    "latency_ms_mean":         re.compile(r"mean\s*=\s*([\d.]+)\s*ms"),
    "latency_ms_p50":          re.compile(r"median\s*=\s*([\d.]+)\s*ms"),
    "latency_ms_p90":          re.compile(r"percentile\(90%\)\s*=\s*([\d.]+)\s*ms"),
    "latency_ms_p95":          re.compile(r"percentile\(95%\)\s*=\s*([\d.]+)\s*ms"),
    "latency_ms_p99":          re.compile(r"percentile\(99%\)\s*=\s*([\d.]+)\s*ms"),
}

# Block we want is the GPU-side one; "Host Latency" is queue+H2D+D2H+kernel
# and is more pessimistic / less reproducible.
_GPU_BLOCK = re.compile(
    r"GPU\s+Compute\s+Time(?P<body>.*?)(?:\n\s*\n|\Z)",
    re.DOTALL,
)


@dataclass
class TrtLatency:
    model: str
    stage: str
    precision: str
    engine: str
    log: str
    shapes: str
    iterations: int
    warmup: int
    rc: int
    latency_ms_min: float | None = None
    latency_ms_max: float | None = None
    latency_ms_mean: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p90: float | None = None
    latency_ms_p95: float | None = None
    latency_ms_p99: float | None = None
    throughput_fps_mean: float | None = None
    error: str | None = None


def _report_repo_path(path: str | Path, anchor: str) -> str:
    """Return a repo-style artefact path instead of a host-local absolute path."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        parts = p.as_posix().split("/")
        if anchor in parts:
            i = parts.index(anchor)
            return "/".join(parts[i:])
        return p.name


def _parse_perf(stdout: str) -> dict:
    block = _GPU_BLOCK.search(stdout)
    if block is None:
        return {}
    body = block.group("body")
    out: dict = {}
    for key, regex in _PERF_FIELDS.items():
        m = regex.search(body)
        if m:
            out[key] = float(m.group(1))
    if "latency_ms_mean" in out and out["latency_ms_mean"] > 0:
        out["throughput_fps_mean"] = 1000.0 / out["latency_ms_mean"]
    return out


# ---------------------------------------------------------------------------
# Per-engine benchmark
# ---------------------------------------------------------------------------

def _opt_shapes_arg(plan) -> str:
    """Use the same opt-profile shapes the engine was built for."""
    return ",".join(p.fmt("opt") for p in plan.profiles)


def bench_one(
    trtexec: str,
    engine_path: Path,
    plan,
    *,
    precision: str,
    iterations: int,
    warmup: int,
    extra_args: list[str],
    log_path: Path | None,
) -> TrtLatency:
    shapes = _opt_shapes_arg(plan)
    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--shapes={shapes}",
        f"--iterations={iterations}",
        f"--warmUp={warmup}",
        "--useSpinWait",       # tighter timing distribution on iGPU
        "--noDataTransfers",   # measure GPU compute, not H2D/D2H copies
        *extra_args,
    ]
    print(f"\n[bench] {engine_path.name}")
    print("        " + " ".join(shlex.quote(c) for c in cmd))

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except FileNotFoundError as e:
        return TrtLatency(
            model="?", stage="?", precision=precision,
            engine=str(engine_path),
            log=str(log_path) if log_path else "",
            shapes=shapes, iterations=iterations, warmup=warmup,
            rc=127, error=str(e),
        )

    if log_path is not None:
        log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        print(f"        log -> {log_path}")

    perf = _parse_perf(proc.stdout)
    if not perf and proc.returncode == 0:
        # Build/run succeeded but our regex didn't match -- show last lines
        # so the user can fix the parser if a future trtexec changes format.
        tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        print(f"        warning: could not parse 'GPU Compute Time' from trtexec output. "
              f"Last 15 lines:\n{tail}")

    rec = TrtLatency(
        model="?", stage="?", precision=precision,
        engine=str(engine_path),
        log=str(log_path) if log_path else "",
        shapes=shapes, iterations=iterations, warmup=warmup,
        rc=proc.returncode,
        **perf,
    )
    if proc.returncode != 0:
        rec.error = f"trtexec exited with code {proc.returncode}"
    return rec


# ---------------------------------------------------------------------------
# Engine discovery + main
# ---------------------------------------------------------------------------

_ENGINE_RE = re.compile(r"^(?P<stem>.+?)\.(?P<precision>fp16|fp32|best|int8)\.engine$")


def _resolve_engine_dir(root: Path, model_name: str, site: str) -> Path:
    """Same back-compat lookup as build_trt._resolve_site_dir.

    Prefers ``engines/<model>_<site>/``; falls back to the legacy
    ``engines/<model>/`` only for utc4 so that pre-existing Thor
    checkouts keep benchmarking until the user re-runs build-trt.
    """
    primary = root / f"{model_name}_{site}"
    legacy = root / model_name
    if primary.exists():
        return primary
    if site == "utc4" and legacy.exists():
        return legacy
    return primary


def discover_engines(
    engine_root: Path,
    model_name: str,
    site: str,
    precision_filter: str | None,
):
    """Yield (stage_stem, precision, path) for engines under engine_root."""
    model_dir = _resolve_engine_dir(engine_root, model_name, site)
    if not model_dir.is_dir():
        return
    for p in sorted(model_dir.glob("*.engine")):
        m = _ENGINE_RE.match(p.name)
        if not m:
            continue
        stem = m.group("stem")
        prec = m.group("precision")
        if precision_filter is not None and prec != precision_filter:
            continue
        yield stem, prec, p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT engines via trtexec --loadEngine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--model", choices=list_models(), default=None,
                   help="Single model. Mutually exclusive with --all.")
    g.add_argument("--all", action="store_true", default=True,
                   help="Benchmark every model that has engines on disk (default).")
    parser.add_argument("--site", choices=("utc3", "utc4"), default="utc4")
    parser.add_argument("--precision", choices=("fp32", "fp16", "best", "int8", "any"),
                        default="fp16",
                        help="Only benchmark engines of this precision. Use 'any' to "
                             "include every *.engine found.")
    parser.add_argument("--engine-root", type=Path, default=REPO_ROOT / "engines")
    parser.add_argument("--iterations", type=int, default=200,
                        help="trtexec --iterations: number of timed inference runs.")
    parser.add_argument("--warmup", type=int, default=50,
                        help="trtexec --warmUp ms before timing starts.")
    parser.add_argument("--trtexec", default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: docs/evidence/thor_trt_latency_<site>.json. "
                             "The legacy docs/evidence/thor_trt_latency.json (no "
                             "site suffix) is still produced when --site=utc4 "
                             "to keep the existing supplemental table working.")
    parser.add_argument("--log-dir", type=Path,
                        default=REPO_ROOT / "reports" / "trt_logs")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to trtexec after `--`.")
    args = parser.parse_args()

    extra_args = list(args.extra)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if args.model is not None:
        models = [args.model]
    else:
        models = list(list_models())

    try:
        trtexec = find_trtexec(args.trtexec)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"[trtexec] {trtexec}")

    precision_filter = None if args.precision == "any" else args.precision
    rows: list[TrtLatency] = []

    for model_name in models:
        try:
            cfg_path = default_config_path(model_name, args.site)
        except FileNotFoundError:
            print(f"[skip] no config for {model_name}_{args.site}")
            continue
        cfg = load_yaml(cfg_path)
        plans = {p.engine_stem: p for p in plan_for_model(model_name, cfg)}

        any_found = False
        for stem, prec, engine_path in discover_engines(
            args.engine_root, model_name, args.site, precision_filter
        ):
            plan = plans.get(stem)
            if plan is None:
                print(f"[skip] {engine_path}: stage '{stem}' is unknown for {model_name}")
                continue
            any_found = True
            log_path = args.log_dir / f"{model_name}_{args.site}_{stem}.{prec}.bench.log"
            rec = bench_one(
                trtexec=trtexec,
                engine_path=engine_path,
                plan=plan,
                precision=prec,
                iterations=args.iterations,
                warmup=args.warmup,
                extra_args=extra_args,
                log_path=log_path,
            )
            rec.model = model_name
            rec.stage = stem
            rec.precision = prec
            rows.append(rec)
            if rec.latency_ms_mean is not None:
                print(f"  -> {model_name}/{stem} {prec}: "
                      f"mean={rec.latency_ms_mean:.3f}ms "
                      f"p50={rec.latency_ms_p50:.3f}ms "
                      f"p95={rec.latency_ms_p95:.3f}ms "
                      f"p99={rec.latency_ms_p99:.3f}ms "
                      f"throughput={rec.throughput_fps_mean:.1f}fps")
        if not any_found:
            searched = _resolve_engine_dir(args.engine_root, model_name, args.site)
            print(f"[skip] no engines under {searched} matching "
                  f"precision={args.precision} -- run "
                  f"`pixi run build-trt --model {model_name} --site {args.site}` "
                  f"first.")

    if not rows:
        print("\nNo engines benchmarked. Build them first with `pixi run build-trt-all`.")
        sys.exit(1)

    # Group rows by model for the summary JSON.
    by_model: dict = {}
    for r in rows:
        row = asdict(r)
        row["engine"] = _report_repo_path(row["engine"], "engines")
        row["log"] = _report_repo_path(row["log"], "reports")
        by_model.setdefault(r.model, []).append(row)

    summary = {
        "trtexec": trtexec,
        "iterations": args.iterations,
        "warmup_ms": args.warmup,
        "site": args.site,
        "precision_filter": args.precision,
        "models": by_model,
    }
    out_path = (
        args.output
        if args.output is not None
        else REPO_ROOT / "docs" / "evidence" / f"thor_trt_latency_{args.site}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {len(rows)} engine(s) benchmarked. Summary -> {out_path}")

    # For utc4, also refresh the historical site-less filename so any
    # downstream tooling that already reads it keeps working.
    if args.output is None and args.site == "utc4":
        legacy = REPO_ROOT / "docs" / "evidence" / "thor_trt_latency.json"
        legacy.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"        (also wrote legacy alias -> {legacy})")


if __name__ == "__main__":
    main()
