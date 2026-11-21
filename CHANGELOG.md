# Changelog

Versions are cut from `main` as tagged releases; the canonical version lives
in `src/xcalib/__init__.py`.

## 0.4.0 (2026-11-21)

Annotation workflow, from a raw delivery to a pair-fitted homography, and a
review window built around confirming pairs.

- **Review window** (`xcalib-review`): opens model-suggested pairs as
  to-confirm decisions (`--proposals`), marks boxes a person should look at
  (`--check`, `--only-listed`, `--context`), and calibrates from the saved
  pairs (`--notebook-workflow`, `--accept-calibration`). The two refine buttons
  are gone: record the saved pairs (`record_saved_pairs`) and refine from
  Python or the notebook instead.
- **Annotation workflow**: import rectified raw deliveries as scenes with
  draft boxes (`xcalib.gui.dataset_prepare`, `label_import`, `lidar_import`;
  `scripts/undistort_raw_delivery.py` repairs a distorted delivery), train on
  the confirmed pairs (`xcalib.gui.workflow`, the `xcalib-export-scene` CLI,
  `xcalib.engine.scene_train`), and let the model propose the undecided pairs
  (`xcalib.gui.model_pairs`).
- **Homography notebook** (`demo/xcalib_homography_en.ipynb`): the whole flow
  in one kernel, ending in the homography fitted to the saved pairs alone
  (`xcalib.gui.pair_homography`, `xcalib.gui.results`, `saved_pairs`).
- **SUSTechPOINTS**: `scripts/launch_sustech.py` opens scenes in the pinned
  upstream tool (view-only or saving) behind a guarded local server
  (`scripts/sustech_server.py`); a fine-tuned LiDAR box is written into the
  scene's own label files.
- **Absolute scores**: a camera box without a LiDAR partner can score every
  candidate low — `score_mode: absolute` in the model config, trained with
  `trainer.train(none_loss=True)`.
- **Fixes**: planar consensus poses are refit with SQPnP in oneshot
  calibration; frames with a single camera or LiDAR box no longer stop
  training.
- **Citation**: `CITATION.cff` and a `@software` BibTeX entry (Guo, Ji, Cao)
  for citing the library itself, next to the paper entry.

## 0.3.0 (2026-10-12)

Review GUI (`xcalib-review`, new `XCalib[gui]` extra), gated continual refine
from human-verified pairs, optional one-to-one assignment with a projection
gate on `Matcher.match`, and calibration nudge/refinement for deployed sites.

## 0.2.0

Planar-PnP calibration robustness (pose disambiguation in
`OneShotSession.calibrate()`), A9 calibration access, and the quickstart
notebook.

## 0.1.0

First public release: the `Matcher` API, model registry, protocol validation,
and packaged configs.
