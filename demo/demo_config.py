"""Edit these constants for the public A9 demo."""

from __future__ import annotations

from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent

MODEL = "crlite"
SITE = "a9_dataset_r02_s01"
SPLIT = "test"
CAMERA_NAME = "s110_camera_basler_south2_8mm"
DEVICE = "auto"

NUM_FRAMES = 16
MATCH_THRESHOLD = 0.5
MIN_CALIBRATION_PAIRS = 12

INTRINSICS_JSON = DEMO_DIR / "intrinsics_a9_example.json"

WEIGHTS_DIR = DEMO_DIR / "weights"
ONNX_DIR = DEMO_DIR / "onnx_export" / f"{MODEL}_{SITE}"
WHEELS_DIR = DEMO_DIR / "wheels"

LOCAL_WEIGHTS = WEIGHTS_DIR / f"{MODEL}_{SITE}_best.pth"
LOCAL_DATASET = DEMO_DIR / "data" / "a9_test.h5"
SAMPLE_FRAMES = DEMO_DIR / "frames" / "a9_sample"
VIZ_DIR = DEMO_DIR / "viz"
