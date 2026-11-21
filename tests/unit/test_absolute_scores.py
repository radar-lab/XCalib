"""A model can say "this camera box has no LiDAR box": absolute scores and the none term."""

from pathlib import Path

import pytest
import torch

import yaml

from xcalib.engine import trainer
from xcalib.models.registry import build_model
from xcalib.utils.config import EdgeConfig

CONFIG = Path(trainer.__file__).resolve().parents[1] / "cfg" / "crlite_a9_dataset_r02_s01.yaml"


def _model(**extra):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data.update(device="cpu", **extra)
    return build_model("crlite", EdgeConfig(data)).eval()


def _inputs(cameras=3, lidars=4):
    torch.manual_seed(0)
    return (torch.rand(cameras, 3, 32, 32), torch.rand(lidars, 1024, 3),
            torch.rand(cameras, 3), torch.rand(lidars, 3))


def test_relative_scores_stretch_every_row_and_absolute_scores_do_not():
    relative, absolute = _model(), _model(score_mode="absolute")
    absolute.load_state_dict(relative.state_dict())
    inputs = _inputs()
    with torch.no_grad():
        stretched = relative.forward_inference(*inputs)
        plain = absolute.forward_inference(*inputs)
    assert torch.allclose(stretched["similarity"].max(dim=1).values, torch.ones(3))  # the best is always 1
    expected = torch.sigmoid(plain["stage2_scores"] / 0.07)
    for row in range(3):
        assert torch.allclose(plain["similarity"][row, plain["top_indices"][row]], expected[row])


def test_an_unknown_score_mode_is_refused():
    with pytest.raises(ValueError, match="score_mode"):
        _model(score_mode="stretched")


def test_the_none_term_pushes_pairs_up_and_everything_else_down():
    match = torch.tensor([[True, False, False], [False, False, False]])  # the second camera box has no partner
    low = {"stage1_similarity": torch.zeros(2, 3), "stage2_similarity": torch.full((2, 3), -1.0)}
    kwargs = dict(temperature=0.07, label_smoothing=0.0, w1=1.0, w2=1.0)
    plain, parts = trainer._train_loss(low, match, **kwargs)
    assert "none_loss" not in parts
    with_none, parts = trainer._train_loss(low, match, none_weight=1.0, **kwargs)
    assert with_none > plain and parts["none_loss"] > 0

    right = {"stage1_similarity": torch.zeros(2, 3),
             "stage2_similarity": torch.tensor([[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])}
    _, better = trainer._train_loss(right, match, none_weight=1.0, **kwargs)
    assert better["none_loss"] < parts["none_loss"]
