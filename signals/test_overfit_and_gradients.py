import torch

from kgllm_pharm.bureau.processing import TrainTensors, fit
from kgllm_pharm.gateway import line
from kgllm_pharm.lanes.levy import composite_loss
from kgllm_pharm.tariff.schedule import DataConfig, ExperimentConfig, ModelConfig, OptimConfig


def _small() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            n_drugs=64,
            n_pts=20,
            rank=8,
            mol_dim=16,
            tgt_dim=16,
            path_dim=16,
            text_dim=16,
            min_positives=6,
            seed=7,
        ),
        model=ModelConfig(
            token_dim=24, hidden_dim=32, expert_hidden=24, lora_rank=8, lora_alpha=16
        ),
    )


def test_overfits_single_batch() -> None:
    cfg = _small()
    assembly = line.assemble(cfg)
    batch = 32
    data = TrainTensors(
        assembly.features[:batch],
        assembly.kg_embedding[:batch],
        assembly.resident[:batch],
        assembly.labels[:batch],
    )
    assembly.model.calibrate_mask(
        assembly.features[:batch],
        assembly.kg_embedding[:batch],
        assembly.resident[:batch],
        assembly.frequency_load[:batch],
    )
    history = fit(
        assembly.model,
        data,
        OptimConfig(
            adapter_lr=5e-3, lora_lr=2e-2, warmup_steps=1, epochs=80, batch_size=32, grad_accum=1
        ),
    )
    assert history[-1] < 0.5 * history[0]


def test_gradients_reach_components() -> None:
    cfg = _small()
    assembly = line.assemble(cfg)
    assembly.model.calibrate_mask(
        assembly.features, assembly.kg_embedding, assembly.resident, assembly.frequency_load
    )
    out = assembly.model.forward(assembly.features, assembly.kg_embedding, assembly.resident)
    terms = composite_loss(out.logits, assembly.labels, out.weights, 0.1)
    torch.autograd.backward(terms.total)
    adapter_grad = assembly.model.adapter.project.weight.grad
    lora_grad = assembly.model.backbone.lora_b.weight.grad
    assert adapter_grad is not None and float(adapter_grad.abs().sum()) > 0.0
    assert lora_grad is not None
    assert assembly.model.backbone.base.weight.grad is None
