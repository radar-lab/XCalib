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

## Pair selection

How to use these: [Pair selection](pair-selection.md).

::: xcalib.assign.one_to_one_assign

::: xcalib.assign.geometric_gate

## Calibration refinement

How to use these: [Deployed site](deployed-site.md).

::: xcalib.oneshot.calibration.refine_projection

::: xcalib.calibration_refine.build_physical_refinement_candidate

::: xcalib.calibration_refine.evaluate_calibration

::: xcalib.calibration_refine.CalibrationGateConfig
    options:
      show_if_no_docstring: true

::: xcalib.calibration_refine.PhysicalCalibrationCandidate
    options:
      show_if_no_docstring: true

::: xcalib.calibration_refine.CalibrationMetrics
    options:
      show_if_no_docstring: true

## Frame capture

How to use this: [Deployed site](deployed-site.md), [Review GUI](review-gui.md).

::: xcalib.capture.SceneRecorder

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
