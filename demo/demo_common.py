"""Shared helpers for the public A9 xcalib demo scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from xcalib import CameraIntrinsics, Matcher, load_dataset


@dataclass
class DemoFrame:
    """One runtime frame shaped like a live perception pipeline output."""

    frame_key: str
    camera_name: str
    image: np.ndarray
    point_cloud: np.ndarray
    bboxes_2d: np.ndarray
    bboxes_3d: np.ndarray


@dataclass
class CalibrationState:
    """Small state object for streaming calibration examples."""

    buffered_pairs: int = 0
    last_result: Any = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_overlay(rgb: np.ndarray, path: Path) -> Path:
    """Write an RGB overlay (from ``xcalib.visualization``) to disk as a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def load_intrinsics(path: Path) -> CameraIntrinsics:
    data = load_json(path)
    distortion_values = data.get("distortion") or None
    distortion = None
    if distortion_values is not None:
        distortion = np.asarray(distortion_values, dtype=np.float64)

    return CameraIntrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        distortion=distortion,
    )


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_pcd_ascii(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.strip().lower() == "data ascii":
            data_start = idx + 1
            break
    else:
        raise ValueError(f"PCD file does not contain 'DATA ascii': {path}")

    data_lines = [line for line in lines[data_start:] if line.strip()]
    if not data_lines:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.loadtxt(data_lines, dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def iter_sample_frames(frames_path: Path, *, limit: int) -> Iterator[DemoFrame]:
    manifest_path = frames_path / "manifest.json"
    manifest = load_json(manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"manifest frames must be a list: {manifest_path}")

    for entry in frames[:limit]:
        detections_path = frames_path / entry["detections"]
        det = load_json(detections_path)
        base = detections_path.parent
        yield DemoFrame(
            frame_key=str(det["frame_key"]),
            camera_name=str(det.get("camera_name", "")),
            image=load_image_rgb(base / det["image"]),
            point_cloud=load_pcd_ascii(base / det["point_cloud"]),
            bboxes_2d=np.asarray(det["bboxes_2d"], dtype=np.float32).reshape(-1, 4),
            bboxes_3d=np.asarray(det["bboxes_3d"], dtype=np.float32).reshape(-1, 6),
        )


def camera_detection_mask(
    camera_per_det: Any,
    *,
    camera_name: str,
    num_boxes: int,
) -> np.ndarray:
    if camera_per_det is None:
        return np.ones((num_boxes,), dtype=bool)
    return np.asarray(camera_per_det).astype(str) == camera_name


def iter_dataset_frames(
    *,
    site: str,
    split: str,
    camera_name: str | None,
    limit: int,
    local_dataset: Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Iterator[DemoFrame]:
    """Stream public HDF5-cache frames through the same runtime arrays.

    A9 caches can contain detections from multiple cameras. The demo filters
    2-D boxes to one source camera so `matcher.match()` receives an image whose
    boxes are in the same pixel coordinate system.
    """
    local = local_dataset if local_dataset is not None and local_dataset.exists() else None
    if local is not None:
        print(f"loading local dataset cache: {local}")
    else:
        print(f"loading {site}/{split} from the released dataset cache")

    with load_dataset(
        site,
        split=split,
        local=local,
        repo_id=repo_id,
        revision=revision,
    ) as loader:
        yielded = 0
        for raw in loader:
            selected_camera = camera_name or raw.camera_name
            if selected_camera not in raw.images:
                continue

            mask = camera_detection_mask(
                raw.camera_per_det,
                camera_name=selected_camera,
                num_boxes=len(raw.bboxes_2d),
            )
            if not np.any(mask):
                continue

            yield DemoFrame(
                frame_key=raw.frame_key,
                camera_name=selected_camera,
                image=raw.images[selected_camera],
                point_cloud=raw.point_cloud,
                bboxes_2d=raw.bboxes_2d[mask],
                bboxes_3d=raw.bboxes_3d,
            )
            yielded += 1
            if yielded >= limit:
                break

    if yielded == 0:
        raise RuntimeError(
            f"No demo frames found for site={site!r}, split={split!r}, "
            f"camera={camera_name!r}."
        )


def iter_demo_frames(
    *,
    sample_frames: Path,
    site: str,
    split: str,
    camera_name: str | None,
    limit: int,
    local_dataset: Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Iterator[DemoFrame]:
    """Use committed sample frames first, then the released HDF5 cache."""
    if (sample_frames / "manifest.json").is_file():
        print(f"loading committed sample frames: {sample_frames}")
        yield from iter_sample_frames(sample_frames, limit=limit)
        return

    yield from iter_dataset_frames(
        site=site,
        split=split,
        camera_name=camera_name,
        limit=limit,
        local_dataset=local_dataset,
        repo_id=repo_id,
        revision=revision,
    )


def load_demo_matcher(
    model: str,
    *,
    site: str,
    device: str = "auto",
    local_weights: Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Matcher:
    """Load released A9 weights, preferring a local checkpoint when present."""
    if local_weights is not None and local_weights.exists():
        print(f"local weights already available: {local_weights}")
        print("skip Hugging Face weight download")
        return Matcher.from_pretrained(
            model,
            weights=local_weights,
            site=site,
            device=device,
        )

    print(f"loading {model}/{site} weights from the released model cache")
    return Matcher.from_pretrained(
        model,
        site=site,
        device=device,
        repo_id=repo_id,
        revision=revision,
    )


def has_onnx_artifacts(onnx_dir: Path) -> bool:
    return onnx_dir.is_dir() and any(onnx_dir.glob("*.onnx"))


def print_asset_status(
    *,
    weights_path: Path,
    onnx_dir: Path,
    dataset_path: Path,
) -> None:
    print("\nasset status")
    try:
        import xcalib

        print(f"  xcalib {xcalib.__version__} installed ({Path(xcalib.__file__).resolve().parent})")
    except ImportError:
        print("  xcalib not importable — run `pixi install`")

    if weights_path.exists():
        print(f"  weights already available: {weights_path}")
        print("  Hugging Face weight download can be skipped")
    else:
        print(f"  optional local weights missing: {weights_path}")
        print("  demo will use released Hub weights")

    if dataset_path.exists():
        print(f"  dataset already available: {dataset_path}")
        print("  Hugging Face dataset download can be skipped")
    else:
        print(f"  optional local dataset missing: {dataset_path}")
        print("  demo will use released Hub dataset cache")

    if has_onnx_artifacts(onnx_dir):
        print(f"  ONNX already available: {onnx_dir}")
        print("  ONNX export can be skipped")
    else:
        print(f"  ONNX missing: {onnx_dir}")


def print_matches(frame: DemoFrame, matches: list[tuple[int, int, float]]) -> None:
    print(f"\nframe {frame.frame_key} ({frame.camera_name})")
    print(f"  detections: {len(frame.bboxes_2d)} camera x {len(frame.bboxes_3d)} lidar")
    if not matches:
        print("  matches: none above threshold")
        return
    print("  matches (camera_det -> lidar_det, score):")
    for cam_idx, lidar_idx, score in matches:
        print(f"    {cam_idx:3d} -> {lidar_idx:3d}  {score:+.4f}")


def print_calibration(calibration: Any) -> None:
    print("\ncalibration solved")
    print(f"  inliers: {calibration.n_inliers}/{calibration.n_correspondences}")
    print(f"  reprojection error: {calibration.reproj_error_px:.3f} px")
    print("  rotation:")
    print(np.array2string(calibration.rotation, precision=6, suppress_small=True))
    print("  translation:")
    print(np.array2string(calibration.translation, precision=6, suppress_small=True))
    print("  projection matrix:")
    print(np.array2string(calibration.projection, precision=6, suppress_small=True))


def calibrate_when_ready(
    session: Any,
    state: CalibrationState,
    *,
    min_pairs: int,
    not_ready_prefix: str | None = "",
) -> bool:
    if state.buffered_pairs < min_pairs:
        return False

    state.last_result = session.calibrate(min_pairs=min_pairs)
    # Since 0.2, calibrate() may solve successfully yet reject a degenerate
    # planar pose (result.accepted is False); only stop once an accepted
    # calibration lands.
    if state.last_result.success and state.last_result.accepted:
        print_calibration(state.last_result)
        return True

    if not_ready_prefix is not None and not state.last_result.success:
        print(f"{not_ready_prefix}calibration not ready: {state.last_result.message}")
    return False


def print_calibration_failure(state: CalibrationState, min_pairs: int) -> None:
    if state.last_result is not None:
        print(f"\ncalibration failed: {state.last_result.message}")
    else:
        print(
            "\ncalibration not solved: "
            f"collected {state.buffered_pairs} confident pair(s), need {min_pairs}"
        )
