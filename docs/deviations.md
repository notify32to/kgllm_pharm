# Deviations

Each entry links the manuscript location it departs from and gives a justification.
Entries are added as the implementation proceeds.

## D1 - Frozen encoders and backbone run from a deterministic offline surrogate

Paper location: Section 4.3 (p.20), Section 4.5 (p.22), Section 4.11 (p.25).

The manuscript fine-tunes a frozen BioMistral-7B with LoRA on top of frozen
ChemBERTa-77M-MTR, ESM2-650M, and BioMistral-7B text encoders. Downloading and
running 7B-scale weights is out of scope for this release. The four-scale encoders
and the language-model backbone therefore sit behind a typing.Protocol port
(`declaration/port.py`, `examination/engine.py`) with two backends: a deterministic
offline backend used by default, and a bonded real backend that raises
`RealBackboneUnavailable` when the heavy libraries or weights are not present. The
architecture, adapter, FS-MoE router, conformal wrapper, losses, metrics, and
theoretical bounds are implemented exactly as described and are exercised on the
surrogate path. The real path is selected by configuration when the weights are
available.

## D2 - Frequency strata assigned by prevalence-rank quantiles on the synthetic service

Paper location: Section 4.5 (p.22), FAERS Q1 2024 prevalence boundaries.

The manuscript fixes the rare / mid / common strata at absolute MedDRA Preferred-Term
prevalence boundaries (rare < 0.1%, mid 0.1%-1%, common > 1%), which only populate all
three strata at FAERS scale. The synthetic-cohort service runs at laptop scale, so strata
are assigned by prevalence-rank quantiles (`consignments.sorting.assign_strata`, default bottom
10% rare, next 30% mid, rest common) while the absolute boundaries are retained in the
data configuration for the real path. This keeps all three strata populated at any cohort
size; the stratum definitions and the per-stratum conformal procedure are unchanged.
