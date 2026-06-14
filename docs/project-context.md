# Project Context

    project_name       : kgllm_pharm                              [HIGH]
    domain             : pharmacovigilance ML - PT-level ADR      [HIGH]
                         multi-label prediction (KG-augmented LLM)
    framework          : PyTorch 2.x + Transformers + PEFT (LoRA) [HIGH]
    venue              : Scientific Reports                       [HIGH]
    primary_datasets   : 6 evaluation sources + 6 knowledge       [HIGH]
                         graphs (see section 6)
    compute_target     : A100 80GB, ~96 GPU-hours train,          [HIGH]
                         232+-18 ms/query inference, 30 GB VRAM
    hparams_reference  : Supplementary Table SA-3 (p.39)          [HIGH]
    supp_path          : none (single-file manuscript; the SI     [HIGH]
                         tables SA-1..SD-5 are embedded p.38-44)
    extra_signals      : 4 theorems (proofs App. B), 3 algorithm
                         boxes, 8 main + 20 SI tables, 2 figures,
                         architecture Fig. 3-4, 20 seeds, no
                         released checkpoints, synthetic data path

    NEEDS_USER_DECISION: 0
    Resolved with the user on the three distinctness axes (layout / config-CLI
    stack / package name). Ready to proceed.

## 1. project_name
`kgllm_pharm` [HIGH].
Derived from the method label "KG-LLM-Pharm" used in the architecture figure
(Fig. 3 title, p.17 region; Fig. 4 title, p.21). Content words: knowledge-graph,
language-model, pharmacovigilance. snake_case package name.

## 2. supp_path
`none` [HIGH]. The manuscript is a single PDF. Supplementary Tables SA-1..SD-5 are
embedded in the same document (p.38-44), and the Mathematical Proofs appendix is
Appendix B (p.36-38). No sibling supplementary files were found by globbing
`paper/*`, `*supp*`, `*_si.*`, `appendix*`.

## 3. domain
Pharmacovigilance machine learning: multi-label adverse-drug-reaction (ADR)
prediction at MedDRA Preferred-Term granularity (label space |T| <= 26,409),
using a knowledge-graph-augmented large language model. [HIGH]
Source: Title; Abstract (p.1); Section 4.1 Problem statement (p.17).

## 4. framework
PyTorch 2.x + HuggingFace Transformers + PEFT (LoRA) on a frozen BioMistral-7B
backbone; frozen ChemBERTa / ESM2 / BioMistral text encoders; numpy/scipy for
statistics. [HIGH]
Pinned versions (Section 4.11 Reproducibility, p.25):
- Python 3.11.7
- PyTorch 2.3.1
- Transformers 4.42.3
- PEFT 0.11.1
- scikit-learn 1.5.0
- NumPy 1.26.4
- SciPy 1.13.1
Pre-trained checkpoints referenced: ChemBERTa-77M-MTR (seyonec/ChemBERTa-77M-MTR),
ESM2-650M (facebook/esm2_t33_650M_UR50D), BioMistral-7B (BioMistral/BioMistral-7B),
Reactome flat files (Reactome_FlatFiles/2024-12).

Note: 7B-scale weights are not downloaded in this release. Frozen encoders and the
backbone sit behind a typing.Protocol port with (a) a deterministic offline
surrogate and (b) a guarded real backend that raises when the heavy libraries or
weights are unavailable. This is recorded in docs/deviations.md.

## 5. venue
Scientific Reports. [HIGH]
Source: "The remainder of the paper follows the Scientific Reports results-first
structure" (Section 1, p.4); "We thank the editorial board of Scientific Reports"
(Acknowledgements, p.26); dataset-availability and reference-density conventions
match Scientific Reports author instructions (Data Availability, p.26; Table SD-4,
p.44).

## 6. primary_datasets
Six retrospective pharmacovigilance evaluation sources (Section 4.2 p.19-20;
Table SA-1 p.38) and six auxiliary knowledge graphs (Table SA-2 p.39). URLs and
licences from Data Availability (p.26); accessed April 2026.

Evaluation sources:
| name | scope | licence | access URL |
|---|---|---|---|
| FAERS | spontaneous reports, Q3 2015 - Q4 2023; ~1.2e7 drug-PT pairs, 31,847 drugs, 26,409 PTs | US Public Domain | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| SIDER 4.1 | curated drug-side-effect catalogue; 1,430 drugs, 5,868 PTs | CC-BY 4.0 | http://sideeffects.embl.de/download/ |
| OFFSIDES | propensity-adjusted FAERS derivative; 1,332 drugs, 10,093 PTs | (nsides.io) | https://nsides.io/ |
| TWOSIDES | DDI-filtered FAERS subset; 645 drugs, 1,301 PTs | (nsides.io) | https://nsides.io/ |
| CT-ADE 2025 | clinical-trial-mapped; 2,497 drugs, 168,984 drug-ADE pairs | CC-BY 4.0 | https://github.com/ds4dh/CT-ADE |
| DrugBank Open 6.0 | drug structures, targets, ATC; 13,800 drugs | CC-BY-NC 4.0 | https://go.drugbank.com/releases/latest |

Auxiliary knowledge graphs (node-embedding space for the inductive adapter):
| name | licence | access URL |
|---|---|---|
| PrimeKG | MIT | https://github.com/mims-harvard/PrimeKG |
| ADReCS-Target | academic | http://www.bio-add.org/ADReCS-Target/ |
| DisGeNET | CC-BY-NC-SA 4.0 | https://www.disgenet.org/ |
| Hetionet | CC-BY 4.0 | https://het.io/ |
| SPOKE | academic use | https://spoke.rbvi.ucsf.edu/ |
| DRKG | Apache 2.0 | https://github.com/gnn4dr/DRKG |

Merged drug-ADR auxiliary graph (Table SA-2): 16,231 drugs, 32,408 targets,
2,604 pathways, 24,503 diseases, 12,884,706 edges.

## 7. compute_target
A100 80GB cluster. [HIGH]
- Training ~96 GPU-hours (~4 GPU-days), dominated by the LoRA pass over BioMistral-7B
  (Table 5 p.9; Section 4.7 p.23).
- Inference latency 232 +- 18 ms per drug-PT query on a single A100 80GB (Table 5).
- Memory 30 GB (BioMistral-7B forward-pass activation cache); fits A100 80GB / A40
  48GB and a 24GB consumer card with sequence-length truncation (Section 2.5 p.9).
- Trainable parameters 2.8e7 on top of the frozen 7e9 backbone (Table 5; Section 4.7).

## 8. hparams_reference
Supplementary Table SA-3 "Implementation hyperparameters held constant across all
reported experiments" (p.39), cross-checked against Section 4.7 Training protocol
(p.23) and Figure 4 caption (p.21). These are the defaults for
diagrams/experiment/main.yaml.

| hyperparameter | value |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 64 |
| adapter learning rate (linear W) | 1e-5 |
| LoRA learning rate | 1e-4 |
| optimiser | AdamW (beta1=0.9, beta2=0.999, eps=1e-8) |
| weight decay | 1e-2 |
| batch size (effective) | 16 (per-device 4 x grad-accum 4) |
| warmup steps | 500 |
| total epochs | 5 |
| cosine schedule decay | 0.1x peak |
| mixed-precision dtype | bfloat16 |
| multi-label loss | weighted BCE-with-logits + 0.1 x MoE load-balance auxiliary |
| gradient clipping (max L2 norm) | 1.0 |
| LoRA dropout | 0.05 |
| conformal alpha default | 0.15 |
| random seeds (20) | 42,137,271,314,577,691,853,1009,1234,1597,2023,2143,2531,2671,2718,3141,3573,4001,4321,4561 |

Frequency strata boundaries (FAERS Q1 2024 prevalence; Section 4.5 p.22):
rare < 0.1%, mid 0.1%-1%, common > 1%. Rare-head frequency mask zeros the top 10%
of hidden dimensions by mutual information with PT-frequency. Conformal operating
point alpha=0.15 mirrors the regulatory PRR 95% CI lower-bound semantics.

## 9. extra_signals
- Four theorems with proofs in Appendix B (p.36-38): Thm 1 cold-drug inductive
  matrix-completion bound (Eq. 1); Thm 2 frequency-stratified capacity allocation
  lower bound (Eq. 2); Thm 3 per-stratum split-conformal coverage (Eq. 3-4);
  Thm 4 FS-MoE rare-stratum generalization gap (Eq. 5).
- Three algorithm boxes: Alg 1 inductive graph-token construction (p.21); Alg 2
  FS-MoE routing and forward pass (p.22); Alg 3 split-conformal PT-set prediction
  (p.23).
- 8 main tables + 20 supplementary tables (SA-1..SD-5); 2 main figures + 2
  architecture figures (Fig. 3-4).
- 20 random seeds; per-seed variance budget sigma(AUROC)=0.004, sigma(F1)=0.006,
  sigma(rare-Sens)=0.011 (Section 4.11 p.25).
- No released checkpoints. The manuscript code-availability statement (p.26) reads,
  verbatim: "Source code, training scripts, and trained model checkpoints are
  available from the corresponding author upon reasonable request. Configuration
  files for reproducing the experimental workflow will be included in the
  Supplementary Information." This verbatim quote is retained only in
  docs/implementation-map.md and is intentionally omitted from the README.
- Ethics: computational retrospective analysis on public de-identified data; no IRB
  required; no patient enrolment; no protected health information (Section 4.2 p.19).
- Headline targets (for the README evaluation table): FAERS AUROC 0.93, F1 0.81,
  rare-Sens 0.83, kappa-vs-PRR 0.65, ECE 0.04 (Table 1); CT-ADE cross-domain AUROC
  0.86, rare-AUPR 0.50 (Table 6); conformal marginal coverage 0.851 at alpha=0.15
  (Table 25).
