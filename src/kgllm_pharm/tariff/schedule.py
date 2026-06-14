from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    n_drugs: int = 31847
    n_pts: int = 26409
    rank: int = 32
    mol_dim: int = 768
    tgt_dim: int = 1280
    path_dim: int = 2600
    text_dim: int = 4096
    rare_max: float = 0.001
    mid_max: float = 0.01
    rare_quantile: float = 0.1
    mid_quantile: float = 0.4
    min_positives: int = 12
    cold_fraction: float = 0.4
    zipf_exponent: float = 1.07
    pathway_signal: float = 1.0
    target_signal: float = 0.85
    molecular_signal: float = 0.55
    text_signal: float = 0.4
    kg_resident_fraction: float = 0.7
    label_noise: float = 0.0
    calibration_fraction: float = 0.15
    test_fraction: float = 0.2
    seed: int = 42


@dataclass
class ModelConfig:
    token_dim: int = 768
    hidden_dim: int = 4096
    expert_hidden: int = 1024
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    mask_fraction: float = 0.10
    blend_init: float = 0.5
    density_gate_open: bool = True


@dataclass
class OptimConfig:
    adapter_lr: float = 1e-5
    lora_lr: float = 1e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_steps: int = 500
    epochs: int = 5
    batch_size: int = 4
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    scheduler_decay: float = 0.1
    amp_dtype: str = "bfloat16"
    loss_balance_lambda: float = 0.1


@dataclass
class ConformalConfig:
    alpha: float = 0.15


@dataclass
class AblationConfig:
    disabled_scales: list[str] = field(default_factory=list)
    use_adapter: bool = True
    shared_head: bool = False
    use_conformal: bool = True


@dataclass
class ExperimentConfig:
    name: str = "main"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    reporting: bool = True
