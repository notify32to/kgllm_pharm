from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    dim: int
    mech_attenuation: float
    gen_attenuation: float
    signal: float
    noise: float


def four_scale_specs(
    mol_dim: int,
    tgt_dim: int,
    path_dim: int,
    text_dim: int,
    molecular_signal: float,
    target_signal: float,
    pathway_signal: float,
    text_signal: float,
) -> list[ScaleSpec]:
    return [
        ScaleSpec("molecular", mol_dim, 0.35, 1.0, molecular_signal, 0.05),
        ScaleSpec("target", tgt_dim, 0.70, 1.0, target_signal, 0.05),
        ScaleSpec("pathway", path_dim, 1.0, 1.0, pathway_signal, 0.03),
        ScaleSpec("clinical_text", text_dim, 0.20, 0.8, text_signal, 0.10),
    ]
