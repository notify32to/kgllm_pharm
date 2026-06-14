import torch

from kgllm_pharm.lanes.channels import FsMoeRouter
from kgllm_pharm.lanes.screen import FrequencyFeatureMask


def test_gating_lies_on_simplex() -> None:
    router = FsMoeRouter(hidden_dim=16, expert_hidden=8, n_pts=10, mask_fraction=0.1)
    hidden = torch.randn(7, 16)
    out = router(hidden)
    assert torch.allclose(out.weights.sum(-1), torch.ones(7), atol=1e-5)
    assert bool((out.weights >= 0).all())
    assert out.probabilities.shape == (7, 10)
    assert bool(((out.probabilities >= 0) & (out.probabilities <= 1)).all())


def test_mask_zeros_top_dimensions() -> None:
    mask = FrequencyFeatureMask(hidden_dim=10, fraction=0.3)
    scores = torch.tensor([0.0, 0, 0, 0, 0, 0, 0, 1.0, 2.0, 3.0])
    mask.set_from_scores(scores)
    assert mask.masked_dims() == 3
    out = mask(torch.ones(2, 10))
    assert float(out[:, 7:].sum()) == 0.0
    assert float(out[:, :7].sum()) == 14.0
