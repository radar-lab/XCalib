"""
Verify UTC Table I reference numbers from the **exported ONNX graphs** instead
of the PyTorch model. This script requires institutional UTC HDF5 caches that
are not shipped in the public repository.

Why this exists
---------------
`scripts/paper/export_onnx.py` already verifies that the ONNX graph matches the
PyTorch model in a per-tensor sense (max|torch - onnx| ~ 1e-6 on
synthetic inputs). That confirms the *graph* is faithful, but it does
not confirm that the partner-shipped artifact still produces the **same
matching quality** when fed the actual UTC test split. This script
closes that gap: it runs onnxruntime over `utc_test.h5` exactly the way
`scripts/paper/validate_paper.py` runs PyTorch, computes Top-1 / Top-3 / MRR,
and writes a JSON report that can be diffed against the PyTorch one.

If the two reports agree to within FP32 numerical noise (~0.1 pp on
Top-1), the paper claim "ONNX export is matching-quality-preserving"
is locked in for the supplemental, and the only remaining concern for
the deployed TRT FP16 engine is FP16 quantization itself -- a
well-studied, ~1e-3 worst-case effect that does not move Top-K.

Example
-------
    pixi run validate-onnx --site utc4
    pixi run validate-onnx --site utc3 --provider cpu
    pixi run python scripts/paper/validate_onnx.py \
        --site utc4 --models crlite crlite_vit_exp3 calibrefine \
        --output docs/evidence/standalone_onnx_validation_utc4.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Torch is needed only because UTCFrameLoader returns FrameData with
# torch tensors. We immediately convert to numpy for ORT and never call
# any torch op, so this works on a Thor / CI box where torch is present
# but a GPU may or may not be available.
from xcalib.utils.torch_check import ensure_torch  # noqa: E402
ensure_torch()

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from tqdm import tqdm  # noqa: E402

from xcalib.data import (  # noqa: E402
    PrepareConfig,
    UTCFrameLoader,
    prepare_frame,
)
from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402
from xcalib.utils.metrics import (  # noqa: E402
    MatchingMetricsAccumulator,
    format_summary_table,
)


# Paper-validated set; mirrors validate_paper.py.
PAPER_MODELS = (
    "crlite",
    "crlite_2dpe",
    "crlite_vit_exp1",
    "crlite_vit_exp3",
    "calibrefine",
)
VALIDATABLE_MODELS = PAPER_MODELS

SITE_DEFAULTS = {
    # UTC caches are institutional data and are not distributed publicly.
    "utc4": "../datasets/utc_dataset4/hdf5_cache/utc_test.h5",
    "utc3": "../datasets/utc_dataset3/hdf5_cache/utc_test.h5",
}


def _report_hdf5_path(site: str) -> str:
    dataset = "utc_dataset3" if site == "utc3" else "utc_dataset4"
    return f"<institutional_utc_root>/{dataset}/hdf5_cache/utc_test.h5"


def _report_repo_path(path: Path, anchor: str) -> str:
    """Return a repo-style artefact path even when generated on another host."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        parts = path.as_posix().split("/")
        if anchor in parts:
            i = parts.index(anchor)
            return "/".join(parts[i:])
        return path.name


# ---------------------------------------------------------------------------
# ORT helpers
# ---------------------------------------------------------------------------

def _resolve_providers(provider_arg: str) -> list[str]:
    """Translate the --provider flag into an ORT providers list."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    want = provider_arg.lower()
    if want == "cpu":
        return ["CPUExecutionProvider"]
    if want == "cuda":
        if "CUDAExecutionProvider" not in available:
            logger.warning(
                "CUDAExecutionProvider not registered in this onnxruntime "
                "build. Falling back to CPU. "
                f"Available: {available}"
            )
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if want == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    raise ValueError(f"--provider must be cpu / cuda / auto, got {provider_arg}")


def _frame_to_numpy(frame) -> dict:
    """FrameData (torch) -> numpy dict with the input names every wrapper expects."""
    crops_2d = frame.crops_2d.detach().cpu().numpy().astype(np.float32, copy=False)
    crops_3d = frame.crops_3d.detach().cpu().numpy().astype(np.float32, copy=False)
    out = {"image_crops": crops_2d, "lidar_crops": crops_3d}

    if frame.bboxes_2d is not None:
        b = frame.bboxes_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        out["bboxes_2d"] = b
        out["img_centers"] = ((b[:, :2] + b[:, 2:]) / 2.0).astype(np.float32)
        # Pairwise model uses 2D centers for both image and lidar.
        out["img_pos"] = out["img_centers"]
    if frame.bbox_centers_3d is not None:
        c = frame.bbox_centers_3d.detach().cpu().numpy().astype(np.float32, copy=False)
        out["lid_centers"] = c
        out["centers_3d"] = c
        out["lid_pos"] = c[:, :2].astype(np.float32, copy=False)

    return out


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


# ---------------------------------------------------------------------------
# Per-model ORT inference
# ---------------------------------------------------------------------------

class _ORTHybrid:
    """Two-stage hybrid (crlite, crlite_2dpe) on top of stage1.onnx + stage2.onnx."""

    def __init__(self, onnx_dir: Path, providers, top_k: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s1 = ort.InferenceSession(str(onnx_dir / "stage1.onnx"), opts, providers=providers)
        self.s2 = ort.InferenceSession(str(onnx_dir / "stage2.onnx"), opts, providers=providers)
        self.top_k = int(top_k)

    def predict(self, inputs: dict) -> tuple[np.ndarray, float]:
        feed = {
            k: inputs[k]
            for k in ("image_crops", "lidar_crops", "img_centers", "lid_centers")
        }
        t0 = time.perf_counter()
        img_embed, lid_embed = self.s1.run(["img_embed", "lid_embed"], feed)
        N, D = img_embed.shape
        M = lid_embed.shape[0]
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32), 0.0

        img_norm = _normalize_rows(img_embed)
        lid_norm = _normalize_rows(lid_embed)
        stage1_sim = img_norm @ lid_norm.T  # [N, M]

        k = min(self.top_k, M)
        # Top-k indices per row, descending by similarity.
        top_idx = np.argpartition(-stage1_sim, kth=k - 1, axis=1)[:, :k]
        # Sort within the top-k slice for deterministic ordering (matches torch.topk).
        order = np.argsort(-np.take_along_axis(stage1_sim, top_idx, axis=1), axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)  # [N, k]

        # Build (N*k) pairs, run stage2 in one ORT call.
        img_pair = np.repeat(img_embed[:, None, :], k, axis=1).reshape(N * k, D)
        lid_pair = lid_embed[top_idx.reshape(-1)]                       # (N*k, D)
        score = self.s2.run(["score"], {"img_pair": img_pair, "lid_pair": lid_pair})[0]
        stage2_scores = score.reshape(N, k)

        # Per-row min-max normalisation, then scatter into a [N, M] matrix
        # with a -1 sentinel for non-top-k entries. Mirrors the PyTorch
        # forward_inference body bit-for-bit.
        final = np.full((N, M), -1.0, dtype=np.float32)
        for i in range(N):
            row = stage2_scores[i]
            rmin, rmax = float(row.min()), float(row.max())
            if rmax > rmin:
                normalised = (row - rmin) / (rmax - rmin)
            else:
                normalised = np.full_like(row, 0.5)
            final[i, top_idx[i]] = normalised
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return final, elapsed_ms


class _ORTCosine:
    """Single-graph cosine-only models (crlite_vit_exp1, crlite_vit_exp3)."""

    def __init__(self, onnx_dir: Path, providers):
        import onnxruntime as ort

        self.m = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=providers)

    def predict(self, inputs: dict) -> tuple[np.ndarray, float]:
        feed = {"image_crops": inputs["image_crops"], "lidar_crops": inputs["lidar_crops"]}
        t0 = time.perf_counter()
        sim = self.m.run(["similarity"], feed)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return sim.astype(np.float32, copy=False), elapsed_ms


class _ORTPairwise:
    """calibrefine -- pairwise B-batched fc1..fc6, sigmoid -> [0, 1]."""

    def __init__(self, onnx_dir: Path, providers, batch_size: int = 64):
        import onnxruntime as ort

        self.m = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=providers)
        self.batch_size = int(batch_size)

    def predict(self, inputs: dict) -> tuple[np.ndarray, float]:
        crops_2d = inputs["image_crops"]
        crops_3d = inputs["lidar_crops"]
        img_pos = inputs["img_pos"]
        lid_pos = inputs["lid_pos"]
        N = crops_2d.shape[0]
        M = crops_3d.shape[0]
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32), 0.0

        ii, jj = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
        flat_i = ii.reshape(-1)
        flat_j = jj.reshape(-1)
        scores = np.full((N, M), -1.0, dtype=np.float32)

        t0 = time.perf_counter()
        for s in range(0, flat_i.size, self.batch_size):
            bi = flat_i[s : s + self.batch_size]
            bj = flat_j[s : s + self.batch_size]
            feed = {
                "image_pair": crops_2d[bi],
                "lidar_pair": crops_3d[bj],
                "img_pos":    img_pos[bi],
                "lid_pos":    lid_pos[bj],
            }
            logit = self.m.run(["score"], feed)[0].reshape(-1)
            scores[bi, bj] = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return scores, elapsed_ms


def _build_runner(model_name: str, onnx_dir: Path, providers, top_k: int):
    if model_name in {"crlite", "crlite_2dpe"}:
        return _ORTHybrid(onnx_dir, providers, top_k=top_k)
    if model_name in {"crlite_vit_exp1", "crlite_vit_exp3"}:
        return _ORTCosine(onnx_dir, providers)
    if model_name == "calibrefine":
        return _ORTPairwise(onnx_dir, providers)
    raise KeyError(model_name)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _resolve_h5(site: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return (REPO_ROOT / SITE_DEFAULTS[site]).resolve()


def _filter_match_matrix(match_matrix: np.ndarray, kept_2d, kept_3d) -> np.ndarray:
    if match_matrix.size == 0 or kept_2d.size == 0 or kept_3d.size == 0:
        return np.zeros((kept_2d.size, kept_3d.size), dtype=bool)
    return match_matrix[np.ix_(kept_2d, kept_3d)].astype(bool)


def _resolve_onnx_dir(onnx_root: Path, model_name: str, site: str) -> Path:
    """Mirror of build_trt._resolve_site_dir.

    Prefers ``onnx/<model>_<site>/``; falls back to legacy ``onnx/<model>/``
    only when site=utc4 AND the new layout is missing. This keeps already-
    deployed Thor checkouts running until the user re-runs export.
    """
    primary = onnx_root / f"{model_name}_{site}"
    legacy = onnx_root / model_name
    if primary.exists():
        return primary
    if site == "utc4" and legacy.exists():
        return legacy
    return primary


def evaluate_one_model(
    model_name: str,
    site: str,
    hdf5_path: Path,
    onnx_root: Path,
    providers,
    limit_frames: int | None,
) -> dict:
    cfg_path = default_config_path(model_name, site)
    cfg = load_yaml(cfg_path)

    onnx_dir = _resolve_onnx_dir(onnx_root, model_name, site)
    expected = (
        ["stage1.onnx", "stage2.onnx"]
        if model_name in {"crlite", "crlite_2dpe"}
        else ["model.onnx"]
    )
    for fname in expected:
        if not (onnx_dir / fname).exists():
            raise FileNotFoundError(
                f"{onnx_dir / fname} is missing -- run "
                f"`pixi run python scripts/paper/export_onnx.py --model {model_name} "
                f"--site {site}` first."
            )

    runner = _build_runner(
        model_name, onnx_dir, providers, top_k=int(cfg.get("top_k", 5))
    )
    prepare_cfg = PrepareConfig(
        crop_size=int(cfg.get("crop_size", 32)),
        point_cloud_size=int(cfg.get("point_cloud_size", 1024)),
        bbox_expansion=float(cfg.get("bbox_expansion", 1.25)),
    )
    acc = MatchingMetricsAccumulator()

    with UTCFrameLoader(hdf5_path) as loader:
        for i, raw in enumerate(tqdm(loader, desc=f"{model_name}/{site}", leave=False)):
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
                logger.warning(
                    f"[{model_name}/{site}] cropping failed at frame {raw.frame_key}: {e}"
                )
                continue

            if frame.crops_2d.numel() == 0 or frame.crops_3d.numel() == 0:
                continue

            inputs = _frame_to_numpy(frame)
            gt = _filter_match_matrix(raw.match_matrix, kept_2d, kept_3d)
            scores, fwd_ms = runner.predict(inputs)
            acc.update(scores, gt, latency_ms=fwd_ms)

    return acc.summary()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify UTC Table I reference numbers from exported ONNX graphs "
            "(requires local institutional HDF5 caches)."
        )
    )
    parser.add_argument("--site", choices=("utc3", "utc4"), required=True)
    parser.add_argument("--hdf5", type=Path, default=None)
    parser.add_argument(
        "--models", nargs="+", default=list(PAPER_MODELS),
        help=f"Default = paper-claimed set: {', '.join(PAPER_MODELS)}.",
    )
    parser.add_argument("--onnx-root", type=Path, default=REPO_ROOT / "onnx",
                        help="Directory holding per-model ONNX subdirs.")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto",
                        help="ORT execution provider. 'auto' = CUDA if available.")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    h5 = _resolve_h5(args.site, args.hdf5)
    if not h5.exists():
        logger.error(f"HDF5 cache not found: {h5}")
        sys.exit(1)
    providers = _resolve_providers(args.provider)
    logger.info(f"Loading UTC frames from {h5}")
    logger.info(f"ORT providers: {providers}")

    rows: list[tuple[str, dict]] = []
    for model_name in args.models:
        if model_name not in VALIDATABLE_MODELS:
            logger.warning(f"Skipping unknown model: {model_name}")
            continue

        t0 = time.perf_counter()
        try:
            summary = evaluate_one_model(
                model_name, args.site, h5,
                onnx_root=args.onnx_root,
                providers=providers,
                limit_frames=args.limit_frames,
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
        except Exception:
            logger.exception(f"ONNX evaluation failed for {model_name}")
            continue

        elapsed = time.perf_counter() - t0
        summary["wall_seconds"] = elapsed
        logger.info(
            f"{model_name}/{args.site} (onnx): top1={summary['top1']*100:.2f}% "
            f"top3={summary['top3']*100:.2f}% mrr={summary['mrr']:.4f} "
            f"lat_mean={summary['latency_ms_mean']:.3f}ms "
            f"(wall={elapsed:.1f}s)"
        )
        rows.append((model_name, summary))

    if not rows:
        logger.error("No models evaluated.")
        sys.exit(1)

    print("\n" + format_summary_table(rows))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Record the on-disk ONNX directory each model was actually loaded
        # from. This is the smoking-gun field that exposes the kind of
        # site/weight mismatch we hit on Thor (utc3 validation against
        # utc4-trained graphs) -- if the path doesn't end with the
        # expected `_<site>` suffix, the legacy fallback was triggered.
        onnx_dirs = {
            name: _report_repo_path(_resolve_onnx_dir(args.onnx_root, name, args.site), "onnx")
            for name, _ in rows
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "site": args.site,
                    "hdf5": _report_hdf5_path(args.site),
                    "providers": providers,
                    "onnx_dirs": onnx_dirs,
                    "results": {name: summary for name, summary in rows},
                },
                f, indent=2,
            )
        logger.info(f"Wrote ONNX validation report to {args.output}")


if __name__ == "__main__":
    main()
