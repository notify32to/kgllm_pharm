# Repository Plan

## Directory tree

    kgllm_pharm/
    |- README.md
    |- LICENSE
    |- pyproject.toml
    |- requirements.txt
    |- environment.yml
    |- Dockerfile
    |- Makefile
    |- .gitignore
    |- .pre-commit-config.yaml
    |- diagrams/
    |  |- model/            (architecture defaults)
    |  |- data/             (per-source synthetic-cohort fragments)
    |  |- experiment/       (main / ablation_* / supplementary_* / _smoke)
    |- src/kgllm_pharm/
    |  |- __init__.py
    |  |- frontier.py       (KgLlmPharm end-to-end estimator + config)
    |  |- tariff/           (simple-parsing dataclass schedule + YAML lodge loader)
    |  |- consignments/     (sources codes, low-rank generation, sorting, paperwork)
    |  |- declaration/      (four-scale encoders: port, offline, bonded, attributes, desk)
    |  |- classification/   (inductive graph-token coder, inspection gate, Thm1 tolerance)
    |  |- examination/      (BioMistral-7B + LoRA engine)
    |  |- lanes/            (FS-MoE channels, benches, screen, levy, Thm2/4 allowance)
    |  |- clearance/        (split-conformal certificate, Thm3 assurance)
    |  |- bureau/           (processing, roster, custody, seal, register)
    |  |- audit/            (measures, reliability, concordance, significance)
    |  |- gateway/          (line orchestration, counter CLI)
    |- signals/             (pytest suite - varied test kinds)
    |- working/             (launch_train.sh, launch_eval.sh, prepare_data.sh)
    |- docs/                (project-context, implementation-map, deviations, repo-plan)
    |- assets/

## Module responsibilities

- `tariff` - typed configuration. `schedule.py` holds frozen-ish dataclasses matching
  Table SA-3; `lodge.py` resolves a YAML `extends` chain into a merged config and is the
  single entry simple-parsing binds against.
- `consignments` - data layer. A deterministic low-rank cohort generator drives both the
  four-scale feature matrix and a Zipf-distributed drug-PT label matrix from shared latent
  mechanistic factors, so pathway-scale signal genuinely carries rare-ADR information and
  the Thm 1 cold-drug bound is numerically exercisable. A paperwork loader provides the
  real-data path.
- `declaration` - frozen four-scale encoders behind a Protocol port. The offline backend
  projects the latent factors into the per-scale dimensions; the bonded-real backend raises
  `RealBackboneUnavailable` when heavy libraries/weights are absent.
- `classification` - inductive graph-token coder (Alg 1) with learnable KG residual blend
  and an inspection gate; carries the Thm 1 bound and its empirical estimator.
- `examination` - frozen trunk + LoRA. A tiny CPU-friendly block stands in for BioMistral-7B
  when the real trunk is unavailable; the LoRA spec is rank-32 / alpha-64.
- `lanes` - frequency-stratified mixture of experts (Alg 2): gating network, three risk
  channels, the rare-channel frequency-feature screen, the levy loss with load-balance
  auxiliary, and the Thm 2 / Thm 4 bounds.
- `clearance` - per-stratum split-conformal calibration and prediction sets (Alg 3) plus
  the Thm 3 coverage bounds.
- `bureau` - processing loop, cosine-with-warmup roster, atomic custody checkpoints, seed
  seal, structured register.
- `audit` - measures, reliability diagnostics, regulatory concordance, and statistical
  significance tests, all in numpy/scipy and validated against scikit-learn.
- `gateway` - the line orchestration and the CLI counter exposing train / evaluate /
  predict / calibrate / export.
- `frontier.py` - assembles the full clearance line into one estimator.

## Dependencies (pinned)

Runtime: python >= 3.10 (repo target 3.11; verified locally on 3.13), torch==2.3.1,
transformers==4.42.3, peft==0.11.1, numpy==1.26.4, scipy==1.13.1, scikit-learn==1.5.0,
simple-parsing==0.1.5, pyyaml==6.0.1. Dev: pytest, ruff, black, isort, mypy, pre-commit.
Optional `[real]` extra (bonded heavy backends): accelerate, sentencepiece.

## Test coverage (signals/)

| file | kind | target |
|---|---|---|
| `test_scales_shape.py` | shape | four-scale dims, concatenation, disabled-scale zeroing |
| `test_synthesis_recovery.py` | numerical-regression | pathway carries rare signal; strata complete |
| `test_adapter_cold_bound.py` | theory-invariant | empirical cold error <= Thm 1 bound |
| `test_router_and_mask.py` | invariant | gating simplex; screen zeros exactly flagged dims |
| `test_conformal_coverage.py` | conformal-coverage | per-stratum coverage in [1-a, 1-a+1/(m+1)] |
| `test_metrics_vs_sklearn.py` | metric-vs-reference | AUROC/AUPR match sklearn |
| `test_stats_vs_reference.py` | metric-vs-reference | kappa/bootstrap/DeLong/McNemar/Holm |
| `test_overfit_and_gradients.py` | overfit + gradient-flow | loss collapses on one batch; grads reach coder/LoRA/heads |
| `test_determinism.py` | determinism | same seed -> identical features and labels |
| `test_config_layering.py` | config-integrity | extends merge; main.yaml == Table SA-3 |
| `test_house_style.py` | house-rules guard | AST/tokenize no comments/docstrings; no forbidden phrases/emoji |
| `test_training_smoke.py` | e2e smoke | 2+ steps on _smoke.yaml, loss decreases, evaluate runs |

## Gates (definition of done)

`pytest -q` green; `ruff check .`, `black --check .`, `isort --check .` clean;
`mypy --strict src/kgllm_pharm` clean; CLI train/evaluate/predict + smoke run;
`docs/project-context.md` has zero NEEDS_USER_DECISION; this map has no unmatched paper
item; `docs/deviations.md` entries each carry a paper-section link.
