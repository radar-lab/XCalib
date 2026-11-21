"""Training takes one frame as one batch."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xcalib.engine import trainer


def test_a_frame_with_a_single_box_is_skipped_in_training(monkeypatch):
    """One frame is one batch; batch normalisation raises on a batch of one."""

    def frame_with(cameras, lidars):
        return (SimpleNamespace(crops_2d=torch.zeros(cameras, 3, 8, 8), crops_3d=torch.zeros(lidars, 16, 3)),
                np.arange(cameras), np.arange(lidars))

    shapes = iter([(1, 3), (3, 1)])
    monkeypatch.setattr(trainer, "prepare_frame", lambda **kwargs: frame_with(*next(shapes)))
    monkeypatch.setattr(trainer, "_forward_train", lambda *args: pytest.fail("a batch of one reached the model"))
    raw = SimpleNamespace(image=None, point_cloud=None, bboxes_2d=None, bboxes_3d=None, images=None,
                          camera_per_det=None, match_matrix=np.eye(3, dtype=bool))
    loss = trainer.run_epoch_train(
        model_name="crlite", model=torch.nn.Linear(1, 1), loader=SimpleNamespace(get_frame=lambda key: raw),
        frame_keys=["a", "b"], prepare_cfg=None, optimizer=None, device=torch.device("cpu"),
        temperature=0.07, label_smoothing=0.0, w1=1.0, w2=1.0, rng=np.random.default_rng(0))
    assert loss == 0.0
