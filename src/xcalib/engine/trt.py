"""
TensorRT engine building — importable engine behind
``Matcher.build("trt")`` and ``scripts/paper/build_trt.py``.

``trtexec`` ships with TensorRT but is not on $PATH on a stock JetPack
install, so we look it up in the well-known locations and pass the right
``--minShapes / --optShapes / --maxShapes`` profile for each model.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from ..utils.config import EdgeConfig

# ---------------------------------------------------------------------------
# Locate trtexec
# ---------------------------------------------------------------------------

# Stock locations across recent JetPack / DGX / x86 TensorRT installs. The
# first existing path wins.
_TRTEXEC_CANDIDATES = (
    "trtexec",                                          # already on $PATH
    "/usr/src/tensorrt/bin/trtexec",                    # JetPack 5/6/7 default
    "/opt/nvidia/tensorrt/bin/trtexec",                 # some NGC images
    "/usr/local/tensorrt/bin/trtexec",                  # manual installs
)


def find_trtexec(explicit: str | None = None) -> str:
    """Return a runnable ``trtexec`` path or raise with an actionable hint."""
    if explicit:
        if Path(explicit).is_file() and os.access(explicit, os.X_OK):
            return explicit
        raise FileNotFoundError(
            f"--trtexec={explicit} is not an executable file."
        )

    for cand in _TRTEXEC_CANDIDATES:
        if cand == "trtexec":
            found = shutil.which(cand)
            if found:
                return found
        elif Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand

    raise FileNotFoundError(
        "Could not locate `trtexec`. Searched: "
        + ", ".join(_TRTEXEC_CANDIDATES)
        + ". On JetPack it ships with TensorRT at "
        "/usr/src/tensorrt/bin/trtexec; either add that directory to $PATH "
        "(`export PATH=/usr/src/tensorrt/bin:$PATH`) or pass "
        "`--trtexec /full/path/to/trtexec`."
    )


# ---------------------------------------------------------------------------
# Per-model shape profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapeProfile:
    """One {min,opt,max}Shapes spec, e.g. image_crops:1x3x32x32."""

    name: str
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]

    def fmt(self, kind: str) -> str:
        # kind ∈ {"min", "opt", "max"}
        shape = getattr(self, f"{kind}_shape")
        return f"{self.name}:{'x'.join(str(s) for s in shape)}"


@dataclass(frozen=True)
class StagePlan:
    """One ONNX -> engine conversion plan."""

    onnx: str          # filename inside onnx/<model>/, e.g. "stage1.onnx"
    engine_stem: str   # filename stem for the engine, e.g. "stage1"
    profiles: tuple[ShapeProfile, ...]


def _crop_size(cfg: EdgeConfig) -> int:
    return int(cfg.get("crop_size", 32))


def _pc_size(cfg: EdgeConfig) -> int:
    return int(cfg.get("point_cloud_size", 1024))


def plan_for_model(model_name: str, cfg: EdgeConfig) -> list[StagePlan]:
    """Return the list of ONNX -> engine conversions for a given model."""
    cs = _crop_size(cfg)
    ps = _pc_size(cfg)

    # Detection-count ranges. (1, opt, max) follows what we observe on UTC:
    # ~8 image crops / ~12 lidar crops at the median, 32 is a comfortable cap.
    N_min, N_opt, N_max = 1, 8, 32
    M_min, M_opt, M_max = 1, 12, 32

    if model_name in {"crlite", "crlite_2dpe"}:
        # Two-stage: stage1 = backbones+PE+retrieval, stage2 = pair MLP.
        stage1 = StagePlan(
            onnx="stage1.onnx",
            engine_stem="stage1",
            profiles=(
                ShapeProfile("image_crops",
                             (N_min, 3, cs, cs), (N_opt, 3, cs, cs), (N_max, 3, cs, cs)),
                ShapeProfile("lidar_crops",
                             (M_min, ps, 3), (M_opt, ps, 3), (M_max, ps, 3)),
                ShapeProfile("img_centers",
                             (N_min, 2), (N_opt, 2), (N_max, 2)),
                ShapeProfile("lid_centers",
                             (M_min, 3), (M_opt, 3), (M_max, 3)),
            ),
        )
        # Stage 2 takes top-K pairs flattened into a single B-axis. K≈10,
        # so B = N*K up to 320 in the worst case.
        B_min, B_opt, B_max = 1, 80, 320
        D = int(cfg.get("embed_dim", 256))
        stage2 = StagePlan(
            onnx="stage2.onnx",
            engine_stem="stage2",
            profiles=(
                ShapeProfile("img_pair",
                             (B_min, D), (B_opt, D), (B_max, D)),
                ShapeProfile("lid_pair",
                             (B_min, D), (B_opt, D), (B_max, D)),
            ),
        )
        return [stage1, stage2]

    if model_name in {"crlite_vit_exp1", "crlite_vit_exp3"}:
        # Both ViT cosine models export to the same no-PE graph -- vit_exp3's
        # PE branch is intentionally bypassed at inference time to reproduce
        # the paper protocol (see xcalib.engine.exporter.export_cosine).
        return [StagePlan(
            onnx="model.onnx",
            engine_stem="model",
            profiles=(
                ShapeProfile("image_crops",
                             (N_min, 3, cs, cs), (N_opt, 3, cs, cs), (N_max, 3, cs, cs)),
                ShapeProfile("lidar_crops",
                             (M_min, ps, 3), (M_opt, ps, 3), (M_max, ps, 3)),
            ),
        )]

    if model_name == "calibrefine":
        # Pairwise: B = N * M up to 32 * 32 = 1024 worst-case.
        B_min, B_opt, B_max = 1, 96, 1024
        return [StagePlan(
            onnx="model.onnx",
            engine_stem="model",
            profiles=(
                ShapeProfile("image_pair",
                             (B_min, 3, cs, cs), (B_opt, 3, cs, cs), (B_max, 3, cs, cs)),
                ShapeProfile("lidar_pair",
                             (B_min, ps, 3), (B_opt, ps, 3), (B_max, ps, 3)),
                ShapeProfile("img_pos",
                             (B_min, 2), (B_opt, 2), (B_max, 2)),
                ShapeProfile("lid_pos",
                             (B_min, 2), (B_opt, 2), (B_max, 2)),
            ),
        )]

    raise KeyError(f"No TRT shape profile registered for model '{model_name}'")


# ---------------------------------------------------------------------------
# Engine build
# ---------------------------------------------------------------------------

PRECISION_FLAGS = {
    "fp32": (),
    "fp16": ("--fp16",),
    "best": ("--best",),  # picks fp16 / int8 per-layer where supported
}


def resolve_site_dir(root: Path, model_name: str, site: str) -> Path:
    """Return the on-disk ONNX directory for a (model, site) pair.

    Prefers the new per-site layout (``onnx/<model>_<site>/``). Falls back
    to the legacy site-less layout (``onnx/<model>/``) only when the new
    one does not exist AND the requested site is utc4 -- the historical
    default. This keeps pre-existing Thor checkouts running until the
    user re-runs ``pixi run export-onnx-utc4-all``.
    """
    primary = root / f"{model_name}_{site}"
    legacy = root / model_name
    if primary.exists():
        return primary
    if site == "utc4" and legacy.exists():
        return legacy
    # Returning the primary path even if missing -- build_one() prints a
    # clear "skip: <path> does not exist" so the user sees what to export.
    return primary


def build_one(
    trtexec: str,
    onnx_dir: Path,
    engine_dir: Path,
    plan: StagePlan,
    precision: str,
    extra_args: list[str],
    log_path: Path | None,
) -> int:
    onnx_path = onnx_dir / plan.onnx
    if not onnx_path.exists():
        print(f"[skip] {onnx_path} does not exist -- run "
              f"`pixi run python scripts/paper/export_onnx.py --model <name>` first.")
        return 2

    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{plan.engine_stem}.{precision}.engine"

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--minShapes=" + ",".join(p.fmt("min") for p in plan.profiles),
        "--optShapes=" + ",".join(p.fmt("opt") for p in plan.profiles),
        "--maxShapes=" + ",".join(p.fmt("max") for p in plan.profiles),
        *PRECISION_FLAGS[precision],
        *extra_args,
    ]

    print(f"\n[build] {plan.engine_stem}.{precision}")
    print("        " + " ".join(shlex.quote(c) for c in cmd))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        print(f"        log -> {log_path}")
    else:
        proc = subprocess.run(cmd)
    return int(proc.returncode)


def build_engines(
    model_name: str,
    cfg: EdgeConfig,
    *,
    onnx_dir: Path | str,
    engine_dir: Path | str,
    precision: str = "fp16",
    trtexec: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    log_dir: Path | str | None = None,
    export_if_missing: Optional[Callable[[], object]] = None,
):
    """Build every engine of `model_name` from `onnx_dir` into `engine_dir`.

    This is what ``Matcher.build("trt")`` calls. When the required
    ONNX graphs are missing and `export_if_missing` is provided, it is
    invoked once (the matcher passes its own ONNX build) before retrying.

    Returns a `xcalib.engine.exporter.BuildResult` (target="trt").
    """
    from .exporter import BuildResult  # local import to avoid cycle at import time

    if precision not in PRECISION_FLAGS:
        raise ValueError(
            f"precision must be one of {tuple(PRECISION_FLAGS)}, got {precision!r}"
        )

    onnx_dir = Path(onnx_dir)
    engine_dir = Path(engine_dir)
    plans = plan_for_model(model_name, cfg)

    # Resolve trtexec *before* any (potentially slow) ONNX export so a
    # missing TensorRT install fails fast with the actionable hint.
    trtexec_bin = find_trtexec(trtexec)
    logger.info(f"[trtexec] {trtexec_bin}")

    missing = [p for p in plans if not (onnx_dir / p.onnx).exists()]
    if missing and export_if_missing is not None:
        logger.info(
            f"ONNX graphs missing under {onnx_dir} "
            f"({', '.join(p.onnx for p in missing)}); exporting first."
        )
        export_if_missing()

    result = BuildResult(target="trt", model=model_name, output_dir=engine_dir)
    failures: list[str] = []
    for plan in plans:
        log_path = (
            Path(log_dir) / f"{model_name}_{plan.engine_stem}.{precision}.log"
            if log_dir is not None
            else None
        )
        rc = build_one(
            trtexec=trtexec_bin,
            onnx_dir=onnx_dir,
            engine_dir=engine_dir,
            plan=plan,
            precision=precision,
            extra_args=list(extra_args or []),
            log_path=log_path,
        )
        engine_path = engine_dir / f"{plan.engine_stem}.{precision}.engine"
        if rc == 0 and engine_path.exists():
            result.artifacts.append(engine_path)
            if log_path is not None:
                result.logs.append(log_path)
        else:
            failures.append(f"{model_name}/{plan.engine_stem}")

    if failures:
        raise RuntimeError(
            f"TensorRT build failed for: {', '.join(failures)}. "
            "Inspect the trtexec output above (or the --log-dir files)."
        )
    logger.success(
        f"Built {len(result.artifacts)} engine(s) into {engine_dir}"
    )
    return result
