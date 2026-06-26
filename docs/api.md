# API reference

The curated public API — everything here is importable from the top-level `xcalib` package and
stable across minor versions.

## Matcher

::: xcalib.engine.matcher.Matcher
    options:
      members:
        - from_pretrained
        - match
        - pair
        - match_frame
        - calibrate
        - oneshot
        - train
        - build
        - save_pretrained

## Results

::: xcalib.engine.matcher.MatchResult

::: xcalib.engine.exporter.BuildResult

::: xcalib.oneshot.calibration.CalibrationResult

## Training

::: xcalib.engine.trainer.train

## Datasets

::: xcalib.hub.datasets.load_dataset

## Visualization

::: xcalib.visualization.draw_matching_overlay

::: xcalib.visualization.draw_calibration_overlay

## Input protocol

::: xcalib.protocol.validate_frame_inputs

::: xcalib.protocol.CameraIntrinsics

::: xcalib.protocol.ProtocolError

::: xcalib.protocol.ProtocolViolation
