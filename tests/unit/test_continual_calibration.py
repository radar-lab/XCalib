"""The acceptance head must be able to TRAVEL, or every refine rejects forever.

Regression cover for a bug measured on a real corpus:

  A legacy parent starts at _AcceptanceCalibration() defaults -> log_scale=0, bias=0,
  so acceptance = sigmoid(raw) and the decision boundary sits at raw > 0. Real raw
  scores clustered at ~1.85 (no_match) and ~2.20 (positive) -- the model HAD learned
  to separate them -- but with the boundary at 0 everything scored 0.86-0.90 and
  no_match_false_positive_rate was exactly 1.0 (every validation no_match accepted).

  `bias` must reach about -1.85 to reject even one. AdamW steps ~lr, and only ~18
  optimizer steps exist per epoch, so at the adapters' lr=1e-4 the reachable travel
  was ~0.002/epoch (~0.007 across the 4-epoch patience window). Worse, no_match_false_positive_rate is a STEP function: pinned at
  1.0 until bias crosses, so the validation score is flat while the head walks toward
  the cliff, early stopping calls the plateau "converged", and the first epoch is
  restored. Candidate == parent -> "primary delta below 0.005" -> rejected on every
  run, at any corpus size.

  With a separate, larger calibration rate the head crosses inside the patience
  window; measured end-to-end afterwards: no_match FPR 1.0 -> 0.0, primary 0.667 ->
  1.0, positive_top1 held at 1.0, outcome rejected -> accepted.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from xcalib.continual.refine_model import (
    XCalibTrainConfig,
    _build_optimizer,
    _CALIBRATION_STATE_KEYS,
    _validate_train_config,
)


def _handle():
    torch = pytest.importorskip("torch")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

    model = SimpleNamespace(img_adapter=Adapter(), lid_adapter=Adapter())
    acceptance = SimpleNamespace(
        log_scale=torch.nn.Parameter(torch.tensor(0.0)),
        bias=torch.nn.Parameter(torch.tensor(0.0)),
    )
    return SimpleNamespace(_matcher=SimpleNamespace(model=model), _acceptance=acceptance)


def test_calibration_scalars_train_on_their_own_larger_rate():
    pytest.importorskip("torch")
    config = XCalibTrainConfig()
    optimizer = _build_optimizer(_handle(), config)

    assert len(optimizer.param_groups) == 2, "adapters and calibration must not share a rate"
    adapters, calibration = optimizer.param_groups
    assert adapters["lr"] == config.learning_rate
    assert calibration["lr"] == config.calibration_learning_rate
    # Two scalars: acceptance.log_scale + acceptance.bias.
    assert len(calibration["params"]) == len(_CALIBRATION_STATE_KEYS) == 2
    assert all(p.numel() == 1 for p in calibration["params"])
    # The adapters keep every non-calibration parameter (2 weights + 2 biases).
    assert len(adapters["params"]) == 4


def test_calibration_rate_is_large_enough_to_cross_a_real_threshold_gap():
    # The head must clear a realistic gap INSIDE the early-stopping window, or the
    # flat metric makes early stopping quit before the cliff and restore epoch 1.
    config = XCalibTrainConfig()
    steps_per_epoch = 18          # measured on the real corpus
    window = steps_per_epoch * config.patience
    gap = 1.85                    # measured bias travel needed to reject the first no_match
    assert config.calibration_learning_rate * window > gap, (
        "calibration_learning_rate too small: the acceptance head cannot cross a "
        "typical gap before early stopping gives up on the plateau"
    )
    assert config.calibration_learning_rate > config.learning_rate


def test_negative_calibration_rate_is_rejected():
    from dataclasses import replace

    with pytest.raises(ValueError, match="calibration_learning_rate"):
        _validate_train_config(replace(XCalibTrainConfig(), calibration_learning_rate=-1.0))
