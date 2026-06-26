"""
One-shot learning demo on synthetic geometry (CPU, no dataset needed).

A virtual camera with known intrinsics K and extrinsics [R|t] generates
self-consistent frames (3D boxes project exactly onto 2D boxes). The demo
walks the full loop the partner would run on a live sensor pair:

    1. calibrate   — PnP/RANSAC recovers [R|t] from buffered 2D-3D pairs
                     (printed against the ground truth);
    2. observe     — frames self-label via the projection matrix; accepted
                     pairs grow the FeatureBank;
    3. adapt       — confidence-gated adapter updates (backbone frozen);
    4. persist     — save_pretrained() round-trip + optional ONNX export.

Usage::

    pixi run oneshot-demo
    pixi run python scripts/oneshot_demo.py --model crlite_vit_exp1 \
        --weights checkpoints/crlite_vit_exp1_utc4_best.pth --export

Without --weights the model is randomly initialised — the calibration and
pseudo-labeling stages are weight-independent (geometry does the work), so
the demo is fully self-contained; only the *quality* of the harvested
matches changes with real weights.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

from xcalib.utils.torch_check import ensure_torch  # noqa: E402
ensure_torch()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xcalib import CameraIntrinsics, Matcher  # noqa: E402
from xcalib.oneshot import SUPPORTED_MODELS, project_points  # noqa: E402
from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.utils.io import default_config_path  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic world (same conventions as tests/integration/test_oneshot.py)
# ---------------------------------------------------------------------------

IMAGE_HW = (720, 1280)
K_GT = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)
R_GT = np.array([[0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [1.0, 0.0, 0.0]])     # LiDAR (x fwd, y left, z up) -> camera
T_GT = np.array([0.2, -0.3, 0.5])


def sample_visible_centers(rng: np.random.Generator, n: int) -> np.ndarray:
    H, W = IMAGE_HW
    out = []
    while len(out) < n:
        c = rng.uniform([8.0, -10.0, -1.0], [45.0, 10.0, 2.0])
        uv, depth = project_points(c[None], K_GT, R_GT, T_GT)
        if depth[0] > 1.0 and 80 <= uv[0, 0] <= W - 80 and 80 <= uv[0, 1] <= H - 80:
            out.append(c)
    return np.array(out)


def synth_frame(seed: int, n: int = 8, noise_px: float = 0.5):
    rng = np.random.default_rng(seed)
    H, W = IMAGE_HW
    centers = sample_visible_centers(rng, n)
    uv, depth = project_points(centers, K_GT, R_GT, T_GT)

    size = np.clip(2200.0 / depth, 24, 160)
    c2 = uv + rng.normal(0, noise_px, size=uv.shape)
    b2 = np.stack(
        [c2[:, 0] - size / 2, c2[:, 1] - size / 2,
         c2[:, 0] + size / 2, c2[:, 1] + size / 2], axis=1
    ).astype(np.float32)

    dims = np.abs(rng.normal([3.5, 1.8, 1.6], 0.3, size=centers.shape))
    b3 = np.hstack([centers - dims / 2, centers + dims / 2]).astype(np.float32)
    b3 = b3[rng.permutation(n)]

    clouds = [rng.uniform(-20, 50, size=(1000, 3)).astype(np.float32)]
    for box in b3:
        clouds.append(rng.uniform(box[:3], box[3:], size=(150, 3)).astype(np.float32))
    pc = np.vstack(clouds)

    image = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    return image, pc, b2, b3


def make_matcher(model_name: str, weights: str | None, device: str) -> Matcher:
    cfg = load_yaml(default_config_path(model_name, "utc4"))
    cfg.set("device", device)
    if weights is not None:
        return Matcher.from_pretrained(model_name, weights=weights,
                                       config=cfg, device=device)
    print("[WARN] no --weights: random init (geometry stages still work;\n"
          "       match quality is meaningless until you pass real weights).")
    from xcalib.models.registry import build_model
    model = build_model(model_name, cfg)
    tmp = Path(tempfile.mkdtemp()) / f"{model_name}_random.pth"
    torch.save({"model_state_dict": model.state_dict()}, tmp)
    return Matcher.from_pretrained(model_name, weights=tmp,
                                   config=cfg, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="crlite_vit_exp1",
                        choices=SUPPORTED_MODELS)
    parser.add_argument("--weights", default=None,
                        help="Optional trained .pth (random init otherwise).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=8,
                        help="Frames to observe after calibration.")
    parser.add_argument("--steps", type=int, default=80,
                        help="adapt() optimisation steps.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "oneshot_demo",
                        help="Where to save the adapted weights + bank.")
    parser.add_argument("--export", action="store_true",
                        help="Also run matcher.build('onnx') on the adapted model.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matcher = make_matcher(args.model, args.weights, args.device)
    session = matcher.oneshot(K_GT, accept_confidence=0.3, seed=args.seed)

    # ---- Phase 1: calibration -------------------------------------------
    # On a live sensor pair these correspondences come from session.observe()
    # (confident matches above match_threshold). The demo's images are noise,
    # so we inject the geometric ground truth to keep it self-contained.
    rng = np.random.default_rng(args.seed)
    for _ in range(3):
        pts3 = sample_visible_centers(rng, 8)
        uv, _ = project_points(pts3, K_GT, R_GT, T_GT)
        session.calib_session.add_correspondences(uv, pts3)

    calib = session.calibrate(min_pairs=12)
    if not calib.success:
        print(f"calibration failed: {calib.message}")
        sys.exit(1)

    dR = calib.rotation @ R_GT.T
    angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    t_err = np.linalg.norm(calib.translation - T_GT)
    print("\n=== Phase 1: calibration (PnP/RANSAC) ===")
    print(f"  inliers            : {calib.n_inliers}/{calib.n_correspondences}")
    print(f"  reprojection error : {calib.reproj_error_px:.3f} px")
    print(f"  rotation error     : {angle:.4f} deg   (vs ground truth)")
    print(f"  translation error  : {t_err:.4f} m     (vs ground truth)")

    # ---- Phase 2: observe — projection matrix labels the frames ----------
    print(f"\n=== Phase 2: observe {args.frames} frames (self-labeling) ===")
    for f in range(args.frames):
        image, pc, b2, b3 = synth_frame(seed=1000 + f, n=8)
        rep = session.observe(image, pc, b2, b3)
        print(f"  frame {rep.frame_id:2d}: dets {rep.n_detections_2d}x"
              f"{rep.n_detections_3d}  pseudo-pairs {rep.n_pseudo_pairs}  "
              f"banked {rep.n_banked} (conf {rep.mean_geometric_confidence:.2f}) "
              f"bank={rep.bank_size}")

    # ---- Phase 3: adapt ---------------------------------------------------
    print(f"\n=== Phase 3: adapt ({args.steps} steps, adapters only) ===")
    rep = session.adapt(steps=args.steps, lr=5e-3, batch_size=128)
    print(f"  loss        : {rep.loss_first:.4f} -> {rep.loss_last:.4f}")
    print(f"  bank top-1  : {rep.top1_before:.3f} -> {rep.top1_after:.3f}")
    print(f"  elapsed     : {rep.elapsed_s:.2f} s on {rep.n_bank_pairs} banked pairs")

    # ---- Phase 4: persist / export ---------------------------------------
    print("\n=== Phase 4: persist ===")
    paths = session.save(args.out)
    for k, v in paths.items():
        print(f"  {k:12s} -> {v}")

    reloaded = Matcher.from_pretrained(args.model, weights=args.out,
                                             device=args.device)
    image, pc, b2, b3 = synth_frame(seed=9999, n=5)
    a = matcher.match(image, pc, b2, b3).similarity
    b = reloaded.match(image, pc, b2, b3).similarity
    print(f"  reload check : max|live - reloaded| = {np.abs(a - b).max():.2e}")

    if args.export:
        result = matcher.build("onnx", output_dir=args.out / "onnx")
        worst = max(result.parity.values()) if result.parity else float("nan")
        print(f"  onnx         : {[p.name for p in result.artifacts]} "
              f"(parity {worst:.2e}) -> {result.output_dir}")

    print("\nDone. The adapted weights load like any other checkpoint:\n"
          f"  Matcher.from_pretrained('{args.model}', weights=r'{args.out}')")


if __name__ == "__main__":
    main()
