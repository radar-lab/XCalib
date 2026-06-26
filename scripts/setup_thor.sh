#!/usr/bin/env bash
# Fallback / educational installer for the standalone package on NVIDIA
# Jetson AGX Thor (and other aarch64 Jetson devices).
#
# In normal use you should NOT need this script -- `pixi install` (run
# from standalone/) handles everything, including torch. Use this only if:
#   * You are not using pixi (system python / venv / conda).
#   * You want to install torch into an existing Python env on Thor.
#
# As of torch 2.12, PyPI ships cp310/cp311/cp312 manylinux_2_28 aarch64
# wheels with a bundled CUDA-13 stack that run on Thor's Blackwell iGPU
# out of the box, so this script just uses the default PyPI index.
#
# What this does:
#   1. Verifies we are on aarch64 Linux.
#   2. Installs the minimal non-torch deps from requirements.txt.
#   3. Detects an already-present JetPack-bundled torch and skips
#      reinstall; otherwise installs torch/torchvision from PyPI.
#   4. Smoke-imports torch and prints CUDA capability info.
#
# Override the wheel index if you want JetPack-bundled wheels instead:
#   TORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130 bash scripts/setup_thor.sh
#   TORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 bash scripts/setup_thor.sh
#
# Usage:
#   cd standalone
#   bash scripts/setup_thor.sh

set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
    echo "[setup_thor] This script is for aarch64 Linux (Jetson). Detected:"
    echo "             OS=$(uname -s)  ARCH=$(uname -m)"
    echo "[setup_thor] On x86_64 / Windows / macOS, run 'pixi install' instead."
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(dirname "$HERE")"
cd "$PKG_ROOT"

PY="${PYTHON:-python3}"
echo "[setup_thor] Using Python at: $(command -v "$PY")"
"$PY" --version

if [[ -f /proc/device-tree/model ]]; then
    MODEL="$(tr -d '\0' </proc/device-tree/model)"
    echo "[setup_thor] Device tree model: $MODEL"
fi

if command -v nvcc >/dev/null 2>&1; then
    echo "[setup_thor] nvcc: $(nvcc --version | grep -i release || true)"
else
    echo "[setup_thor] nvcc not on PATH -- ok if CUDA toolkit isn't required."
fi

echo "[setup_thor] Step 1/3: minimal Python deps (no torch yet)..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install --no-deps -r requirements.txt

echo "[setup_thor] Step 2/3: PyTorch / torchvision..."
# Prefer JetPack's pre-installed wheels if they exist on PYTHONPATH; otherwise
# fall back to the chosen wheel index. Most JetPack images ship with torch
# already installed under /usr/lib/python3.*/dist-packages -- in that case
# nothing more is needed.
if "$PY" -c "import torch; print('  torch found:', torch.__version__)" 2>/dev/null; then
    echo "[setup_thor]   JetPack-bundled torch detected, skipping reinstall."
else
    if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
        echo "[setup_thor]   Installing from custom index: $TORCH_INDEX_URL"
        "$PY" -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision || {
            echo
            echo "[setup_thor] ERROR: torch wheel install from custom index failed."
            echo "             Try without TORCH_INDEX_URL to use plain PyPI:"
            echo "               unset TORCH_INDEX_URL && bash scripts/setup_thor.sh"
            exit 2
        }
    else
        echo "[setup_thor]   Installing from PyPI (default torch >= 2.12 aarch64+cu13 wheel)..."
        "$PY" -m pip install "torch>=2.12" "torchvision>=0.27" || {
            echo
            echo "[setup_thor] ERROR: torch install from PyPI failed."
            echo "             Most likely your Python is too old for the aarch64"
            echo "             wheel (need Python 3.10-3.12). Detected:"
            "$PY" --version
            echo "             Override the index for JetPack-bundled wheels:"
            echo "               TORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130 bash scripts/setup_thor.sh"
            exit 2
        }
    fi
fi

echo "[setup_thor] Step 3/3: smoke test..."
"$PY" - <<'PY'
import torch
print(f"  torch         : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device   : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  compute cap   : sm_{cap[0]}{cap[1]}")
PY

echo
echo "[setup_thor] Done. You can now run the standalone scripts directly with python3:"
echo "  python3 scripts/infer_demo.py"
echo "  python3 scripts/paper/export_onnx.py --model crlite \\"
echo "      --weights checkpoints/crlite_utc4_best.pth \\"
echo "      --config configs/crlite_utc4.yaml \\"
echo "      --output onnx/crlite_utc4"
echo
echo "Do NOT prefix these with 'pixi run' on Thor -- pixi doesn't manage"
echo "the JetPack torch wheel."
