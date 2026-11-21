"""Fresh-process entry point for GUI base training."""

import sys

import torch  # noqa: F401 — load before the image/HDF5 libraries

from .trainer import train


def _require_usable_frame(path, config):
    """Reject caches whose crops cannot yield a positive and a negative."""
    from ..data import PrepareConfig, UTCFrameLoader, prepare_frame
    from ..utils.config import load_yaml

    cfg = load_yaml(config)
    prep = PrepareConfig(crop_size=int(cfg.get("crop_size", 32)),
                         point_cloud_size=int(cfg.get("point_cloud_size", 1024)),
                         bbox_expansion=float(cfg.get("bbox_expansion", 1.25)))
    with UTCFrameLoader(path) as loader:
        for raw in loader:
            try:
                _, rows, cols = prepare_frame(
                    image=raw.image, point_cloud=raw.point_cloud,
                    bboxes_2d=raw.bboxes_2d, bboxes_3d=raw.bboxes_3d,
                    cfg=prep, images=raw.images, camera_per_det=raw.camera_per_det)
                if len(cols) >= 2 and raw.match_matrix[rows][:, cols].any():
                    return
            except (ValueError, IndexError):
                continue
    raise ValueError(f"{path}: no frame has a retained positive pair and two LiDAR crops")


def main():
    train_path, validation_path, config, output, epochs, device, *extra = sys.argv[1:]
    # --none-loss: the caches also hold unpaired camera boxes and the frame's other
    # LiDAR boxes, and the second-stage score is trained to be absolute.
    none_loss = "--none-loss" in extra
    extra = [value for value in extra if value != "--none-loss"]
    if len(extra) > 1:
        raise ValueError("expected at most one pretrained checkpoint")
    _require_usable_frame(train_path, config)
    _require_usable_frame(validation_path, config)
    model = "crlite"
    if extra:
        from ..models.registry import build_model
        from ..utils.config import load_yaml
        from ..utils.io import extract_state_dict, load_checkpoint

        model = build_model("crlite", load_yaml(config))
        model.load_state_dict(extract_state_dict(load_checkpoint(extra[0], device="cpu")), strict=True)
    train(model, train_path, validation_path, model_name="crlite", config=config,
          epochs=int(epochs), output_dir=output, device=device, none_loss=none_loss)


if __name__ == "__main__":
    main()
