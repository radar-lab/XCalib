"""
Export the standalone matching models to ONNX — thin CLI shim.

The actual exporters live in `xcalib.engine.exporter` (also reachable as
``Matcher.build("onnx")`` and ``xcalib export-onnx``); this
script keeps the historical flags so existing pixi tasks / partner notes
stay valid:

    pixi run python scripts/paper/export_onnx.py --model crlite --site utc4 \
        --weights checkpoints/crlite_utc4_best.pth --output onnx/crlite_utc4

See `src/xcalib/engine/exporter.py` for the graph layout per model family.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Back-compat: the runtime toggle in xcalib.engine.exporter supersedes this,
# but keep the env var for users who import the backbones directly.
os.environ.setdefault("XCALIB_FPS_DETERMINISTIC", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Surface a Jetson-aware install hint if torch is missing (Thor partners who
# ran `pixi install` instead of `bash scripts/setup_thor.sh` end up here).
from xcalib.utils.torch_check import ensure_torch  # noqa: E402
ensure_torch()

from loguru import logger  # noqa: E402

from xcalib.utils.config import load_yaml  # noqa: E402
from xcalib.engine.exporter import ExportError, export_onnx  # noqa: E402
from xcalib.models.registry import build_model, list_models  # noqa: E402
from xcalib.utils.io import default_config_path, resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_models(), default="crlite")
    parser.add_argument("--site", choices=("utc3", "utc4"), default="utc4",
                        help="Which site's weights to export. Determines both "
                             "the default config (configs/<model>_<site>.yaml) "
                             "and the default output dir "
                             "(onnx/<model>_<site>/). UTC4 is the historical "
                             "default; pass --site utc3 to ship the UTC3 "
                             "checkpoint.")
    parser.add_argument("--weights", type=Path, required=False,
                        help="Path to .pth (overrides weights_path in config)")
    parser.add_argument("--config", type=Path, required=False,
                        help="YAML config (defaults to configs/<model>_<site>.yaml)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (defaults to onnx/<model>_<site>/). "
                             "Per-site subdirs prevent the UTC3 / UTC4 "
                             "weight sets from overwriting each other.")
    parser.add_argument("--device", default="cpu",
                        help="cpu | cuda. ONNX export usually runs on CPU.")
    args = parser.parse_args()

    cfg_path = args.config or default_config_path(args.model, args.site)
    cfg = load_yaml(cfg_path)

    weights_path = args.weights or (REPO_ROOT / cfg.get("weights_path", ""))
    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.error(
            f"Weights not found: {weights_path}. Use --weights or run "
            "place the expected file under checkpoints/ or pass --weights."
        )
        sys.exit(1)

    device = resolve_device(args.device)
    model = build_model(args.model, cfg)
    model.load_weights(weights_path, strict=False)
    model.to(device)
    model.eval()

    out_dir = args.output or (REPO_ROOT / "onnx" / f"{args.model}_{args.site}")

    try:
        export_onnx(args.model, model, cfg, out_dir, device=device)
    except ExportError as e:
        logger.error(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
