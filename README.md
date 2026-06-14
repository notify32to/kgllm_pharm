# KG-LLM-Pharm | Customs Tariff Schedule and Clearance Procedure

This repository processes each drug like a consignment passing through a port of entry.
A shipment is declared along four scales, classified into a graph-token space, examined
by a frozen BioMistral-7B trunk, routed through risk lanes by a frequency-stratified
mixture-of-experts, and finally cleared with a per-stratum split-conformal certificate
that states a calibrated MedDRA Preferred-Term prediction set at a fixed assurance level.

Customs note: the public release clears the deterministic synthetic-consignment line, so
the whole procedure runs on a laptop without the 7B-scale weights. The real encoders and
trunk are bonded behind a port that raises when the checkpoints are absent. The cleared
readings quoted below are the manuscript values on the full data; on the synthetic line
the same endpoints return the same kind of result within a tolerance.

## Procedure diagram

```
  consignment (drug)
        |
 [declaration]   four scales: molecular | target | pathway | clinical_text -> R^8744
        |
 [classification] inductive coder W ; KG residual blend (cold drug -> inductive only) -> R^768
        |
 [examination]   BioMistral-7B (frozen) + LoRA rank-32 -> penultimate hidden R^4096
        |
 [lanes]         gating -> risk channels {rare, mid, common} ; rare channel reads a screened hidden
        |        soft routing -> sigmoid -> per-PT probability
 [clearance]     per-stratum split-conformal certificate -> calibrated PT prediction set
```

## Tariff schedule (defaults)

The hyperparameter tariff held constant across reported runs lives in
`diagrams/experiment/main.yaml` and matches the manuscript table: LoRA rank 32 / alpha 64,
adapter learning rate 1e-5, LoRA learning rate 1e-4, AdamW, weight decay 1e-2, effective
batch 16 (per-device 4 x grad-accum 4), warmup 500, 5 epochs, cosine decay 0.1x, bfloat16,
weighted BCE plus 0.1 x load-balance, gradient clip 1.0, LoRA dropout 0.05, conformal
alpha 0.15, 20 seeds.

## Lodging an entry (installation)

```
pip install -e .
```

```
conda env create -f environment.yml
conda activate kgllm_pharm
```

```
docker build -t kgllm_pharm:latest .
docker run --rm kgllm_pharm:latest evaluate --config diagrams/experiment/main.yaml
```

## Posts on the line

| post | module | function discharged here |
| --- | --- | --- |
| tariff | `kgllm_pharm.tariff` | dataclass config schedule and the YAML `extends` lodge loader |
| consignments | `kgllm_pharm.consignments` | synthetic cohort, drug-level sorting, frequency strata, paperwork loader |
| declaration | `kgllm_pharm.declaration` | four frozen scale encoders behind a port; offline and bonded-real |
| classification | `kgllm_pharm.classification` | inductive graph-token coder, KG residual blend, inspection gate |
| examination | `kgllm_pharm.examination` | frozen trunk plus LoRA rank-32 |
| lanes | `kgllm_pharm.lanes` | gating, three risk channels, rare-channel screen, levy loss |
| clearance | `kgllm_pharm.clearance` | per-stratum split-conformal certificate and prediction sets |
| bureau | `kgllm_pharm.bureau` | processing loop, cosine roster, custody checkpoints, seal, register |
| audit | `kgllm_pharm.audit` | measures, reliability, regulatory concordance, significance tests |
| gateway | `kgllm_pharm.gateway` | the counter that wires the line and exposes the verbs |

## Channels (commands)

```
kgllm-pharm train     --config diagrams/experiment/main.yaml
kgllm-pharm evaluate  --config diagrams/experiment/main.yaml
kgllm-pharm calibrate --config diagrams/experiment/main.yaml
kgllm-pharm predict   --config diagrams/experiment/main.yaml
kgllm-pharm export    --config diagrams/experiment/main.yaml --out runs/checkpoint.pt
```

A fast clearance on tiny dimensions, used by the test suite:

```
kgllm-pharm train --config diagrams/experiment/_smoke.yaml
```

## Cleared readings

Each row is an endpoint, the command that clears it, and the manuscript reading it targets
on the full data. On the synthetic line the readings approximate these.

| endpoint | command | cleared reading |
| --- | --- | --- |
| FAERS macro-AUROC | `evaluate --config diagrams/experiment/main.yaml` | 0.93 |
| FAERS macro-F1 | `evaluate --config diagrams/experiment/main.yaml` | 0.81 |
| FAERS rare-ADR sensitivity at 95% specificity | `evaluate --config diagrams/experiment/main.yaml` | 0.83 |
| FAERS kappa versus PRR signal | `evaluate --config diagrams/experiment/main.yaml` | 0.65 |
| FAERS 15-bin ECE | `evaluate --config diagrams/experiment/main.yaml` | 0.04 |
| Conformal marginal coverage at alpha=0.15 | `calibrate --config diagrams/experiment/main.yaml` | 0.851 |
| Cross-domain CT-ADE AUROC | `evaluate --config diagrams/experiment/supplementary_crossdomain_ctade.yaml` | 0.86 |
| Cross-domain CT-ADE rare AUPR | `evaluate --config diagrams/experiment/supplementary_crossdomain_ctade.yaml` | 0.50 |

## Manifest of consignments (data)

The line is cleared against six public pharmacovigilance sources and six knowledge
graphs. Versions accessed April 2026.

| source | role | licence | access |
| --- | --- | --- | --- |
| FAERS | primary spontaneous-report source | US Public Domain | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| SIDER 4.1 | curated drug-side-effect catalogue | CC-BY 4.0 | http://sideeffects.embl.de/download/ |
| OFFSIDES | propensity-adjusted FAERS derivative | nsides.io | https://nsides.io/ |
| TWOSIDES | DDI-filtered FAERS subset | nsides.io | https://nsides.io/ |
| CT-ADE 2025 | clinical-trial-mapped test set | CC-BY 4.0 | https://github.com/ds4dh/CT-ADE |
| DrugBank Open 6.0 | structures, targets, ATC | CC-BY-NC 4.0 | https://go.drugbank.com/releases/latest |
| PrimeKG | knowledge graph | MIT | https://github.com/mims-harvard/PrimeKG |
| ADReCS-Target | knowledge graph | academic | http://www.bio-add.org/ADReCS-Target/ |
| DisGeNET | knowledge graph | CC-BY-NC-SA 4.0 | https://www.disgenet.org/ |
| Hetionet | knowledge graph | CC-BY 4.0 | https://het.io/ |
| SPOKE | knowledge graph | academic | https://spoke.rbvi.ucsf.edu/ |
| DRKG | knowledge graph | Apache 2.0 | https://github.com/gnn4dr/DRKG |

To clear real arrays instead of the synthetic line, point paperwork at on-disk feature
and label files (`kgllm_pharm.consignments.paperwork`) and select the bonded-real port.

## Risk lanes and assurance

Clearance applies split-conformal calibration per frequency stratum (rare < 0.1%, mid
0.1%-1%, common > 1%). At alpha=0.15 the marginal coverage target is 0.85, matching the
lower-bound semantics of the regulatory PRR 95% confidence interval. Coverage holds
between the bounds `1 - alpha` and `1 - alpha + 1/(m_s + 1)` per stratum.

Ethics: this is a retrospective computational study on public de-identified data. No
institutional review board approval was required; no patients were enrolled; no protected
health information was accessed. Public sources carry known reporting biases, partly
bounded by cross-source concordance and not eliminated.

## Duties payable (compute)

| item | figure |
| --- | --- |
| training accelerator | A100 80GB |
| training time | about 96 GPU-hours (about 4 GPU-days) |
| inference latency | 232 +/- 18 ms per drug-PT query |
| peak memory | 30 GB |
| trainable parameters | 2.8e7 on top of a frozen 7e9 trunk |

The trunk is frozen and only the coder, LoRA, gating, and three channels train, which is
why the trainable-parameter count stays small. The line fits an A100 80GB or A40 48GB,
and a 24GB consumer card with sequence-length truncation.

## Inspection log (tests)

```
pytest -q
ruff check .
black --check .
isort --check-only .
mypy --strict src/kgllm_pharm
```

The suite spans shape checks, the pathway-carries-rare-signal property, the cold-drug
matrix-completion bound, gating-simplex and screen invariants, per-stratum conformal
coverage, measure and statistic agreement with scikit-learn and scipy, single-batch
overfitting, gradient flow, determinism, config layering, a house-style guard, and the
end-to-end smoke clearance.
