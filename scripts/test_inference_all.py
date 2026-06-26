"""
End-to-end inference smoke for every shipped model on the local machine.

Runs `Matcher.from_pretrained` for every (model, site) pair that has
a checkpoint under `standalone/checkpoints/`, feeds a synthetic frame, and
prints the resulting similarity matrix + forward-pass latency.

This is the script we point at on a brand-new install to confirm the
whole package - matcher API, cropping, ONNX-friendly forward path - runs
end-to-end before shipping to the partner.

Usage::

    pixi run python standalone/scripts/test_inference_all.py --device cuda
    pixi run python standalone/scripts/test_inference_all.py --device cpu
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Jetson-aware install hint if torch is missing (must run before `import torch`).
# Also keeps Windows DLL ordering happy: torch's OpenMP/MKL runtime must load
# before numpy/h5py.
from xcalib.utils.torch_check import ensure_torch  # noqa: E402
ensure_torch()

import numpy as np  # noqa: E402

from xcalib import Matcher  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402


ALL_MODELS = (
    "crlite",
    "crlite_2dpe",
    "crlite_vit_exp1",
    "crlite_vit_exp3",
    "calibrefine",
)

SITES = ("utc4", "utc3")


@dataclass
class CaseResult:
    model: str
    site: str
    status: str            # ok | skip | fail
    note: str = ""
    sim_shape: Tuple[int, int] = (0, 0)
    sim_min: float = 0.0
    sim_max: float = 0.0
    fwd_ms: float = 0.0
    wall_ms: float = 0.0


def synthetic_frame(seed: int = 0):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    pts = rng.uniform(
        low=[-20.0, -20.0, -2.0],
        high=[50.0, 20.0, 4.0],
        size=(12000, 3),
    ).astype(np.float32)
    bboxes_2d = np.array(
        [
            [100.0, 200.0, 220.0, 320.0],
            [500.0, 300.0, 620.0, 420.0],
            [800.0, 250.0, 900.0, 380.0],
            [1050.0, 400.0, 1200.0, 560.0],
        ],
        dtype=np.float32,
    )
    bboxes_3d = np.array(
        [
            [5.0, -2.0, -1.0, 8.0, 0.5, 0.5],
            [10.0, 1.0, -1.0, 13.0, 3.0, 0.5],
            [15.0, -3.0, -1.0, 18.0, 0.0, 0.5],
            [22.0, 4.0, -1.0, 25.0, 7.0, 0.5],
            [30.0, -5.0, -1.0, 33.0, -2.0, 0.5],
        ],
        dtype=np.float32,
    )
    return image, pts, bboxes_2d, bboxes_3d


def run_one(model: str, site: str, device: str, n_warmup: int, n_timed: int) -> CaseResult:
    ckpt = REPO_ROOT / "checkpoints" / f"{model}_{site}_best.pth"
    if not ckpt.exists():
        return CaseResult(model, site, "skip", note=f"missing ckpt: {ckpt.name}")
    try:
        cfg = default_config_path(model, site)
    except FileNotFoundError:
        return CaseResult(model, site, "skip", note=f"missing config: {model}_{site}.yaml")

    try:
        matcher = Matcher.from_pretrained(
            model=model, weights=ckpt, config=cfg, device=device,
        )
    except Exception as exc:
        return CaseResult(model, site, "fail", note=f"from_pretrained: {exc}")

    image, pts, b2, b3 = synthetic_frame()

    # Warmup so the first allocation / JIT cost is not in the timed pass.
    try:
        for _ in range(n_warmup):
            matcher.match(image=image, point_cloud=pts, bboxes_2d=b2, bboxes_3d=b3, top_k=3)
    except Exception as exc:
        return CaseResult(model, site, "fail", note=f"warmup: {exc}")

    # Timed runs
    try:
        fwd_total = 0.0
        last = None
        t0 = time.perf_counter()
        for _ in range(n_timed):
            res = matcher.match(
                image=image, point_cloud=pts, bboxes_2d=b2, bboxes_3d=b3,
                top_k=3, return_latency=True,
            )
            fwd_total += res.latency_ms
            last = res
        wall_ms = (time.perf_counter() - t0) * 1000.0 / n_timed
    except Exception as exc:
        return CaseResult(model, site, "fail", note=f"match: {exc}")

    if last is None or last.similarity.size == 0:
        return CaseResult(model, site, "fail", note="empty similarity")

    sim = last.similarity
    return CaseResult(
        model=model,
        site=site,
        status="ok",
        sim_shape=tuple(sim.shape),
        sim_min=float(sim.min()),
        sim_max=float(sim.max()),
        fwd_ms=fwd_total / n_timed,
        wall_ms=wall_ms,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_MODELS),
        help=f"Subset of models to test (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=list(SITES),
        choices=SITES,
        help="UTC sites to test (default: utc4 utc3).",
    )
    args = parser.parse_args()

    print(f"[test_inference_all] device={args.device}  warmup={args.warmup}  repeat={args.repeat}")
    print(f"[test_inference_all] REPO_ROOT={REPO_ROOT}")
    print()

    results: List[CaseResult] = []
    for model in args.models:
        for site in args.sites:
            print(f">> {model} / {site} ... ", end="", flush=True)
            r = run_one(model, site, args.device, args.warmup, args.repeat)
            results.append(r)
            if r.status == "ok":
                print(
                    f"OK   sim={r.sim_shape} range=[{r.sim_min:+.3f}, {r.sim_max:+.3f}]"
                    f"  fwd={r.fwd_ms:6.2f} ms  wall={r.wall_ms:6.2f} ms"
                )
            elif r.status == "skip":
                print(f"SKIP {r.note}")
            else:
                print(f"FAIL {r.note}")

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skip")
    n_fail = sum(1 for r in results if r.status == "fail")

    print()
    print(f"[test_inference_all] {n_ok} ok, {n_skip} skipped, {n_fail} failed")

    if n_fail > 0:
        print("\nFailures:")
        for r in results:
            if r.status == "fail":
                print(f"  - {r.model}/{r.site}: {r.note}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
