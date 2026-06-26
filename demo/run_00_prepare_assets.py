"""Prepare local public-demo assets."""

from __future__ import annotations

from demo_common import has_onnx_artifacts, load_demo_matcher, print_asset_status
from demo_config import DEVICE, LOCAL_DATASET, LOCAL_WEIGHTS, MODEL, ONNX_DIR, SITE, WHEELS_DIR


def main() -> int:
    print_asset_status(
        wheel_dir=WHEELS_DIR,
        weights_path=LOCAL_WEIGHTS,
        onnx_dir=ONNX_DIR,
        dataset_path=LOCAL_DATASET,
    )

    if has_onnx_artifacts(ONNX_DIR):
        print("\nONNX files already downloaded/exported; skip ONNX export")
        return 0

    matcher = load_demo_matcher(MODEL, site=SITE, device=DEVICE, local_weights=LOCAL_WEIGHTS)
    print(f"\nexporting ONNX to {ONNX_DIR}")
    result = matcher.build("onnx", output_dir=ONNX_DIR, device="cpu")
    for artifact in result.artifacts:
        print(f"  wrote {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
