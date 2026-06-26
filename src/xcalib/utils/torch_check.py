"""
Friendly torch import check for partner-facing scripts.

Importing ``torch`` directly fails with a bare
``ModuleNotFoundError: No module named 'torch'`` on the NVIDIA Jetson AGX
Thor whenever the partner runs ``pixi install`` — because the standalone
``pixi.toml`` deliberately omits torch on ``linux-aarch64`` so that the
Thor uses the JetPack-bundled PyTorch wheel (optimized for the
Blackwell iGPU and the device's CUDA version) rather than a generic
SBSA build off PyPI.

This helper converts that confusing error into a single actionable
message that tells the user exactly what to run.
"""

from __future__ import annotations

import os
import platform
import sys
import textwrap


def _is_jetson() -> bool:
    """Heuristic: aarch64 Linux is almost certainly a Jetson when used
    with this package, but also check the /proc/device-tree model file
    if present (more reliable on JetPack)."""
    if platform.system() != "Linux":
        return False
    if platform.machine() not in {"aarch64", "arm64"}:
        return False
    model = "/proc/device-tree/model"
    if os.path.exists(model):
        try:
            with open(model, "rb") as f:
                txt = f.read().decode("utf-8", errors="ignore").lower()
            return "jetson" in txt or "thor" in txt or "tegra" in txt or "orin" in txt
        except OSError:
            return True
    return True


def ensure_torch() -> None:
    """Try to import torch; if it fails, print a context-aware install
    hint and exit cleanly. Safe to call multiple times."""
    try:
        import torch  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise

    if _is_jetson():
        body = textwrap.dedent(
            """
            Detected aarch64 Linux (likely an NVIDIA Jetson — e.g. AGX Thor).

            The standalone's pixi.toml pulls torch from the default PyPI
            index on linux-aarch64 (torch >= 2.12 ships cp310/cp311/cp312
            manylinux_2_28 aarch64 wheels with a bundled CUDA-13 stack
            that runs on Thor's Blackwell iGPU). To install (from repo root):

                pixi install
                pixi run python scripts/paper/export_onnx.py --model crlite ...

            If you cannot use pixi (system python / venv / conda), the
            equivalent pip workflow lives in scripts/setup_thor.sh.
            """
        ).strip()
    else:
        body = textwrap.dedent(
            """
            PyTorch is not importable in this Python environment.

            On x86_64 Linux / Windows / macOS, install with:

                pixi install        # CUDA 12.8 wheels on Linux/Windows
                                    # CPU wheels on macOS

            Or with plain pip:

                pip install torch torchvision
            """
        ).strip()

    sys.stderr.write(
        "ERROR: PyTorch is not importable in this environment.\n\n" + body + "\n\n"
    )
    sys.exit(2)
