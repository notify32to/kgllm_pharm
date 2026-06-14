from pathlib import Path

from kgllm_pharm.tariff.lodge import load_experiment
from kgllm_pharm.tariff.schedule import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "diagrams" / "experiment"


def test_extends_merge_overrides_child_only() -> None:
    cfg = load_experiment(EXPERIMENTS / "ablation_scale_pathway.yaml")
    assert cfg.ablation.disabled_scales == ["pathway"]
    assert cfg.data.n_drugs == 31847
    assert cfg.model.lora_rank == 32


def test_main_matches_reference_table() -> None:
    cfg = load_experiment(EXPERIMENTS / "main.yaml")
    reference = ExperimentConfig()
    assert cfg.model.lora_rank == reference.model.lora_rank == 32
    assert cfg.model.lora_alpha == 64
    assert cfg.optim.batch_size * cfg.optim.grad_accum == 16
    assert cfg.optim.warmup_steps == 500
    assert cfg.optim.epochs == 5
    assert cfg.conformal.alpha == 0.15


def test_every_experiment_config_loads() -> None:
    paths = sorted(EXPERIMENTS.glob("*.yaml"))
    assert len(paths) >= 10
    for path in paths:
        load_experiment(path)
