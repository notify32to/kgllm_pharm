from pathlib import Path

from kgllm_pharm.gateway import line
from kgllm_pharm.tariff.lodge import load_experiment

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "diagrams" / "experiment" / "_smoke.yaml"


def test_smoke_training_loss_decreases() -> None:
    cfg = load_experiment(SMOKE)
    _assembly, history = line.train_model(cfg)
    assert len(history) >= 2
    assert history[-1] < history[0]


def test_smoke_evaluate_and_predict() -> None:
    cfg = load_experiment(SMOKE)
    report = line.evaluate(cfg)
    assert 0.0 <= report["coverage"] <= 1.0
    assert report["auroc"] > 0.5
    sets = line.predict_sets(cfg)
    assert sets["mean_set_size"] >= 0.0
