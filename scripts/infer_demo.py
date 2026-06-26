"""
End-to-end inference demo on a single synthetic frame.

This script is the partner's "hello world": it shows the exact call
shape they should mirror inside their edge-side pipeline. We use
synthetic inputs (no HDF5, no Open3D, no special data) so the script
runs in well under 1 second on CPU.

Usage::

    pixi run python scripts/infer_demo.py
    pixi run python scripts/infer_demo.py --model crlite_vit_exp3 \
        --weights checkpoints/crlite_vit_exp3_utc4_best.pth \
        --config configs/crlite_vit_exp3_utc4.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Jetson-aware install hint if torch is missing (must run before `import torch`).
# Also keeps Windows DLL ordering happy: torch's OpenMP/MKL runtime must load
# before numpy/h5py, otherwise Windows resolves shm.dll against the wrong libs.
from xcalib.utils.torch_check import ensure_torch  # noqa: E402
ensure_torch()

import numpy as np  # noqa: E402

from xcalib import Matcher  # noqa: E402
from xcalib.models.registry import list_models  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402


def synthetic_frame(
    image_h: int = 720,
    image_w: int = 1280,
    n_image_dets: int = 3,
    n_lidar_dets: int = 5,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(image_h, image_w, 3), dtype=np.uint8)
    # ~12k points spread in front of the LiDAR
    points = rng.uniform(
        low=[-20.0, -20.0, -2.0],
        high=[50.0, 20.0, 4.0],
        size=(12000, 3),
    ).astype(np.float32)

    # K random 2D bboxes
    cx = rng.uniform(80, image_w - 80, n_image_dets)
    cy = rng.uniform(80, image_h - 80, n_image_dets)
    w = rng.uniform(60, 200, n_image_dets)
    h = rng.uniform(60, 200, n_image_dets)
    bboxes_2d = np.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
    ).astype(np.float32)

    # M random 3D axis-aligned bboxes
    cx3 = rng.uniform(5.0, 30.0, n_lidar_dets)
    cy3 = rng.uniform(-10.0, 10.0, n_lidar_dets)
    cz3 = rng.uniform(-1.0, 1.0, n_lidar_dets)
    dx = rng.uniform(1.5, 4.5, n_lidar_dets)
    dy = rng.uniform(1.2, 2.5, n_lidar_dets)
    dz = rng.uniform(1.2, 2.5, n_lidar_dets)
    bboxes_3d = np.stack(
        [cx3 - dx / 2, cy3 - dy / 2, cz3 - dz / 2,
         cx3 + dx / 2, cy3 + dy / 2, cz3 + dz / 2],
        axis=1,
    ).astype(np.float32)

    return image, points, bboxes_2d, bboxes_3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="crlite", choices=list_models())
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional weights path; if omitted, runs with random initialisation "
        "(useful for connectivity testing only).",
    )
    parser.add_argument("--config", default=None,
                        help="Optional YAML config; defaults to "
                             "configs/<model>_utc4.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-image", type=int, default=3)
    parser.add_argument("--n-lidar", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg_path = (
        Path(args.config) if args.config else
        default_config_path(args.model, "utc4")
    )

    if args.weights is None:
        # Allow the demo to run without weights for partners doing a first
        # smoke import — but warn loudly.
        print(
            "[WARN] No --weights provided; running with randomly initialised "
            "parameters. Results are NOT meaningful — pass --weights for "
            "real outputs."
        )
        from xcalib.utils.config import load_yaml

        cfg = load_yaml(cfg_path)
        # Drop weights_path so the matcher doesn't try to load
        cfg.set("weights_path", "")
        # Instantiate model + matcher without weight loading
        from xcalib.engine.wrappers import make_wrapper
        from xcalib.models.registry import build_model
        from xcalib.utils.io import resolve_device

        model = build_model(args.model, cfg)
        dev = resolve_device(args.device)
        model.to(dev)
        model.eval()

        image, pts, b2, b3 = synthetic_frame(
            n_image_dets=args.n_image, n_lidar_dets=args.n_lidar, seed=args.seed
        )

        # Reuse the Matcher cropping helpers manually
        from xcalib.data.crops import PrepareConfig, prepare_frame

        prep_cfg = PrepareConfig(
            crop_size=int(cfg.get("crop_size", 32)),
            point_cloud_size=int(cfg.get("point_cloud_size", 1024)),
            bbox_expansion=float(cfg.get("bbox_expansion", 1.25)),
        )
        frame, kept_2d, kept_3d = prepare_frame(image, pts, b2, b3, cfg=prep_cfg)
        wrapper = make_wrapper(
            args.model, model, device=dev,
            point_cloud_size=prep_cfg.point_cloud_size,
            top_k=int(cfg.get("top_k", 5)),
        )
        scores, fwd_ms = wrapper.predict_matching_matrix(frame)
        print(f"scores shape: {tuple(scores.shape)} | forward {fwd_ms:.3f} ms")
        return

    matcher = Matcher.from_pretrained(
        model=args.model,
        weights=args.weights,
        config=cfg_path,
        device=args.device,
    )

    image, pts, b2, b3 = synthetic_frame(
        n_image_dets=args.n_image, n_lidar_dets=args.n_lidar, seed=args.seed
    )

    result = matcher.match(
        image=image,
        point_cloud=pts,
        bboxes_2d=b2,
        bboxes_3d=b3,
        top_k=3,
        return_latency=True,
    )
    print(f"model           : {result.model}")
    print(f"device          : {result.device}")
    print(f"similarity shape: {result.similarity.shape}")
    print(f"top_indices     : {result.top_indices}")
    print(f"latency_ms      : {result.latency_ms:.3f}")
    if result.matches:
        print("matches (img_idx, lid_idx, score):")
        for m in result.matches[:10]:
            print(f"  {m[0]:3d} -> {m[1]:3d}   {m[2]:+.4f}")


if __name__ == "__main__":
    main()
