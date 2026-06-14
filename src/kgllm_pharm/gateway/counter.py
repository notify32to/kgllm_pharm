from __future__ import annotations

from dataclasses import dataclass

from simple_parsing import ArgumentParser

from kgllm_pharm.bureau.custody import save_atomic
from kgllm_pharm.bureau.register import get_logger
from kgllm_pharm.gateway import line
from kgllm_pharm.tariff.lodge import load_experiment

_COMMANDS = ("train", "evaluate", "predict", "calibrate", "export")


@dataclass
class Options:
    config: str = "diagrams/experiment/main.yaml"
    max_steps: int | None = None
    out: str = "runs/checkpoint.pt"


def _format(report: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in report.items())


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("command", choices=list(_COMMANDS))
    parser.add_arguments(Options, dest="options")
    namespace = parser.parse_args(argv)
    command: str = namespace.command
    options: Options = namespace.options
    config = load_experiment(options.config)
    logger = get_logger("kgllm_pharm.gateway")

    if command == "train":
        assembly, history = line.train_model(config, max_steps=options.max_steps)
        save_atomic(options.out, {"state": assembly.model.state_dict(), "seed": config.data.seed})
        logger.info(
            "trained experiment=%s steps=%d final_loss=%.4f", config.name, len(history), history[-1]
        )
        return 0
    if command == "evaluate":
        report = line.evaluate(config, max_steps=options.max_steps)
        logger.info("evaluate experiment=%s %s", config.name, _format(report))
        return 0
    if command == "calibrate":
        report = line.evaluate(config, max_steps=options.max_steps)
        coverage = {k: v for k, v in report.items() if k.startswith("coverage")}
        logger.info("calibrate experiment=%s %s", config.name, _format(coverage))
        return 0
    if command == "predict":
        report = line.predict_sets(config, max_steps=options.max_steps)
        logger.info("predict experiment=%s %s", config.name, _format(report))
        return 0
    assembly, _ = line.train_model(config, max_steps=options.max_steps)
    save_atomic(options.out, {"state": assembly.model.state_dict(), "config": config.name})
    logger.info("export experiment=%s artifact=%s", config.name, options.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
