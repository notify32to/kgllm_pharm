# Implementation Map

This is the single source of paper provenance for the code base. Source files carry
no comments and no docstrings; every mapping from a manuscript item (equation,
algorithm, theorem, table, figure) to a module/symbol lives here.

Package root: `src/kgllm_pharm`. Tests: `signals/`. Configs: `diagrams/`. Scripts:
`working/`. The package is organised as a port-of-entry clearance line.

## Architecture stages -> subpackages (Fig. 3 p.17 region, Fig. 4 p.21)

| paper item | section / eq / table | file path | module symbol | notes |
|---|---|---|---|---|
| Problem statement, label space, folds | Sec. 4.1 (p.17) | `consignments/sorting.py` | `FrequencyStratum`, `split_drug_level` | drug-level non-overlap split; rare/mid/common strata |
| Four-scale drug feature x_d in R^8744 | Sec. 4.3 (p.20); Fig. 4 | `declaration/attributes.py` | `four_scale_specs`, `ScaleSpec` | dims 768/1280/2600/4096; frozen encoders |
| Frozen-encoder port + surrogate + bonded real | Sec. 4.3 (p.20) | `declaration/port.py`, `declaration/offline.py`, `declaration/bonded.py`, `declaration/desk.py` | `FeatureProvider`, `SurrogateFeatureProvider`, `RealFeatureProvider`, `RealBackboneUnavailable`, `build_feature_provider` | 7B weights not downloaded; deviation D1 |
| Inductive graph-token adapter z_d = W x_d | Sec. 4.4 (p.21); Alg. 1 | `classification/coder.py` | `InductiveGraphToken.project` | linear W in R^{768x8744} |
| KG residual blend z = beta z_ind + (1-beta) e_KG | Sec. 4.4; Alg. 1 lines 4-8 | `classification/coder.py` | `InductiveGraphToken.blend_coefficient`, `.construct` | learnable beta in [0,1] init 0.5; cold drug -> z_ind only |
| KG density gate (inference toggle) | Sec. 4.4 (p.22) | `classification/gate.py` | `KgDensityGate` | thresholded edge count; default held open |
| BioMistral-7B frozen + LoRA rank-32 | Sec. 4.5 (p.22); Fig. 4 | `examination/engine.py` | `BackboneModel`, `LoraConfigSpec` | penultimate hidden state strip h in R^4096 |
| FS-MoE router: gating g(h) -> Delta^3 | Sec. 4.5 (p.22); Alg. 2 line 4 | `lanes/benches.py` | `GatingNetwork` | softmax over 3 risk channels |
| Three stratum heads phi_rare/mid/common | Sec. 4.5; Alg. 2 lines 6-8 | `lanes/benches.py` | `StratumHead` | MLP 4096 -> 1024 -> |T| |
| Rare-head frequency-feature mask W_rare | Sec. 4.5 (p.22); Alg. 2 line 5 | `lanes/screen.py` | `FrequencyFeatureMask`, `frequency_mutual_information` | zeros top-10% MI-with-frequency dims |
| Soft routing aggregation l = sum w_s l_s | Sec. 4.5; Fig. 4 footer | `lanes/channels.py` | `FsMoeRouter.forward`, `RouterOutput` | p_hat = sigmoid(l) |
| Weighted BCE + MoE load-balance loss | Sec. 4.5 (p.22); Sec. 4.7 | `lanes/levy.py` | `composite_loss`, `load_balance_aux` | lambda_aux = 0.1 |
| Split-conformal calibration (per stratum) | Sec. 4.6 (p.22-23); Alg. 3 lines 1-6 | `clearance/certificate.py` | `SplitConformal.calibrate` | scores s=1-p_hat over positives; q_hat per stratum |
| Conformal PT-set inference | Sec. 4.6; Alg. 3 lines 7-15 | `clearance/certificate.py` | `SplitConformal.predict_set` | include t if 1-p_t < q_hat[stratum(t)] |
| End-to-end estimator assembly | Fig. 4 (p.21) | `frontier.py` | `KgLlmPharm`, `KgLlmPharmConfig` | wires declaration -> classification -> examination -> lanes -> clearance |

## Theorems (proofs Appendix B, p.36-38)

| theorem | statement loc | eq | file path | symbol | empirical check |
|---|---|---|---|---|---|
| Thm 1 cold-drug inductive matrix-completion bound | p.17-18, p.36 | Eq. 1 | `classification/tolerance.py` | `cold_row_error_bound`, `empirical_cold_error` | predicts ~5 pp cold-drug AUROC drop; RMSE ~0.064 |
| Thm 2 FS-MoE capacity allocation lower bound | p.18, p.37 | Eq. 2 | `lanes/allowance.py` | `fs_moe_capacity_gap` | rare-sens gain bounded by sqrt(P/P_rare)*sqrt(p_rare/(1-p_rare)) |
| Thm 3 per-stratum split-conformal coverage | p.19, p.38 | Eq. 3-4 | `clearance/assurance.py` | `coverage_lower_bound`, `coverage_upper_bound` | 1-alpha <= cov <= 1-alpha+1/(m_s+1) |
| Thm 4 FS-MoE rare-stratum generalization gap | p.19, p.37 | Eq. 5 | `lanes/allowance.py` | `rademacher_gap_bound` | predicts gap ~0.07 at P_rare=4.2e6, m_rare=1.2e4 |

## Algorithms

| algorithm | loc | file path | symbol |
|---|---|---|---|
| Alg. 1 Inductive Graph-Token Construction | p.21 | `classification/coder.py` | `InductiveGraphToken.construct` |
| Alg. 2 FS-MoE Routing and Forward Pass | p.22 | `lanes/channels.py` | `FsMoeRouter.forward` |
| Alg. 3 Split-Conformal PT-Set Prediction | p.23 | `clearance/certificate.py` | `SplitConformal` |

## Metrics and statistics (Sec. 4.9-4.10, p.24)

| paper item | loc | file path | symbol |
|---|---|---|---|
| macro-AUROC / macro-AUPR / macro-F1 | Sec. 4.9 | `audit/measures.py` | `macro_auroc`, `macro_aupr`, `macro_f1` |
| rare-ADR sensitivity at 95% specificity | Sec. 4.9; Table 1 | `audit/measures.py` | `sensitivity_at_specificity` |
| Cohen kappa vs PRR-derived signal | Sec. 4.9; Sec. 3.3 (p.14) | `audit/concordance.py` | `kappa_vs_prr`, `prr_calls`, `proportional_reporting_ratio` |
| 15-bin Expected Calibration Error | Sec. 4.9; Table 25 | `audit/reliability.py` | `expected_calibration_error` |
| Brier score | Sec. 4.9; Table 25 | `audit/reliability.py` | `brier_score` |
| marginal / per-stratum coverage, set size | Sec. 4.9; Table 25 | `clearance/certificate.py` | `marginal_coverage`, `per_stratum_coverage` |
| paired-bootstrap 10,000 resamples + CI | Sec. 4.10 (p.24) | `audit/significance.py` | `paired_bootstrap_ci` |
| DeLong test | Sec. 4.10; Table 4 | `audit/significance.py` | `delong_test` |
| McNemar test | Sec. 4.10 | `audit/significance.py` | `mcnemar_test` |
| Holm-Bonferroni step-down | Sec. 4.10 | `audit/significance.py` | `holm_bonferroni` |
| KS rank-statistic exchangeability check | Sec. 4.10 (p.24-25); Table 25 | `audit/significance.py` | `conformal_ks_statistic` |

## Datasets / KG / synthetic generator (Sec. 4.2 p.19-20; Tables SA-1/SA-2)

| paper item | loc | file path | symbol |
|---|---|---|---|
| Six evaluation sources statistics | Table SA-1 (p.38) | `consignments/codes.py` | `SOURCE_STATS`, `SourceSpec` |
| Six knowledge graphs statistics | Table SA-2 (p.39) | `consignments/codes.py` | `KG_STATS`, `KgSpec` |
| Low-rank matrix-completion data generator | Sec. 4.1/4.4 design | `consignments/generation.py` | `Cohort`, `synthesise` |
| Drug-level non-overlap split + strata | Sec. 4.2 (p.20) | `consignments/sorting.py` | `split_drug_level`, `assign_strata` |
| Manifest loader (real data path) | Sec. 4.11 (p.25) | `consignments/paperwork.py` | `read_manifest`, `load_array` |

## Training / runtime (Sec. 4.7 p.23; Sec. 4.11 p.25)

| paper item | loc | file path | symbol |
|---|---|---|---|
| AdamW + cosine, warmup 500, 5 epochs | Sec. 4.7; Table SA-3 | `bureau/processing.py`, `bureau/roster.py` | `fit`, `cosine_with_warmup` |
| Effective batch 16 (4 x grad-accum 4) | Sec. 4.7; Table SA-3 | `bureau/processing.py` | `fit`, `TrainTensors` |
| bfloat16 AMP / gradient clipping 1.0 | Table SA-3 | `bureau/processing.py` | `fit` |
| atomic checkpoint write (tmp + replace) | R4 engineering bar | `bureau/custody.py` | `save_atomic`, `load_checkpoint` |
| set_seed + seed stored/restored | R4 engineering bar; Sec. 4.11 | `bureau/seal.py` | `set_seed` |
| structured logging | R4 engineering bar | `bureau/register.py` | `get_logger` |

## Configuration / CLI

| paper item | loc | file path | symbol |
|---|---|---|---|
| hyperparameter schedule (Table SA-3) | Table SA-3 (p.39) | `tariff/schedule.py` | `ExperimentConfig`, `OptimConfig`, `ModelConfig`, `DataConfig`, `ConformalConfig`, `AblationConfig` |
| YAML preset loader + extends merge | Sec. 4.11 (p.25) | `tariff/lodge.py` | `load_experiment`, `resolve_extends`, `build_experiment` |
| pipeline orchestration | - | `gateway/line.py` | `assemble`, `train_model`, `evaluate`, `predict_sets` |
| CLI counter (train/evaluate/predict/calibrate/export) | - | `gateway/counter.py` | `main`, `Options` |

## Experiment configs (Tables 1-8, SB/SC/SD)

| config file | reproduces | paper loc |
|---|---|---|
| `diagrams/experiment/main.yaml` | FAERS main result (AUROC 0.93 etc.) | Table 1 (p.6) |
| `diagrams/experiment/ablation_fs_moe_shared.yaml` | rare head -> shared head | Table 2 (p.7) |
| `diagrams/experiment/ablation_no_adapter.yaml` | inductive adapter -> transductive | Table 2 |
| `diagrams/experiment/ablation_scale_*.yaml` | per-scale removal (mol/tgt/path/text) | Table 2; Table SB-4 (p.41) |
| `diagrams/experiment/ablation_no_conformal.yaml` | conformal wrapper removed | Table 2 |
| `diagrams/experiment/ablation_kg_schema_*.yaml` | KG composition sweep | Table SB-3 (p.41) |
| `diagrams/experiment/ablation_backbone_*.yaml` | backbone scaling | Table 7 (p.10) |
| `diagrams/experiment/supplementary_crossdomain_ctade.yaml` | FAERS -> CT-ADE transfer | Table 6 (p.9) |
| `diagrams/experiment/supplementary_noise_05.yaml` | noise-injection robustness | Table SC-1 (p.42) |
| `diagrams/experiment/supplementary_ood_postcutoff.yaml` | out-of-distribution post-cutoff drugs | Table SC-3 (p.42) |
| `diagrams/experiment/_smoke.yaml` | pytest smoke only - not for reporting | - |

## Verbatim code-availability quote (Data/Code Availability, p.26)

Retained here only (intentionally omitted from README per release policy):

> Source code, training scripts, and trained model checkpoints are available from
> the corresponding author upon reasonable request. Configuration files for
> reproducing the experimental workflow will be included in the Supplementary
> Information.
