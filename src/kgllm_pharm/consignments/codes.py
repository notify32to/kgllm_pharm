from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    key: str
    drugs: int
    pts: int
    drug_pt_pairs: int
    median_prevalence: float
    licence: str
    access_url: str


@dataclass(frozen=True)
class KgSpec:
    key: str
    drugs: int
    targets: int
    pathways: int
    diseases: int
    edges: int
    licence: str
    access_url: str


SOURCE_STATS: dict[str, SourceSpec] = {
    "faers": SourceSpec(
        "faers",
        31847,
        26409,
        12_000_000,
        5.4e-4,
        "US Public Domain",
        "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html",
    ),
    "sider": SourceSpec(
        "sider",
        1430,
        5868,
        139_756,
        1.7e-3,
        "CC-BY 4.0",
        "http://sideeffects.embl.de/download/",
    ),
    "offsides": SourceSpec(
        "offsides",
        1332,
        10093,
        438_801,
        9.3e-4,
        "nsides.io",
        "https://nsides.io/",
    ),
    "twosides": SourceSpec(
        "twosides",
        645,
        1301,
        868_221,
        1.1e-3,
        "nsides.io",
        "https://nsides.io/",
    ),
    "ctade": SourceSpec(
        "ctade",
        2497,
        11847,
        168_984,
        8.6e-4,
        "CC-BY 4.0",
        "https://github.com/ds4dh/CT-ADE",
    ),
    "drugbank": SourceSpec(
        "drugbank",
        13800,
        0,
        0,
        0.0,
        "CC-BY-NC 4.0",
        "https://go.drugbank.com/releases/latest",
    ),
}


KG_STATS: dict[str, KgSpec] = {
    "primekg": KgSpec(
        "primekg",
        7957,
        27671,
        2587,
        17080,
        8_100_498,
        "MIT",
        "https://github.com/mims-harvard/PrimeKG",
    ),
    "adrecs_target": KgSpec(
        "adrecs_target",
        1586,
        1901,
        0,
        0,
        41_832,
        "academic",
        "http://www.bio-add.org/ADReCS-Target/",
    ),
    "hetionet": KgSpec(
        "hetionet", 1552, 21094, 1822, 137, 2_250_197, "CC-BY 4.0", "https://het.io/"
    ),
    "disgenet": KgSpec(
        "disgenet", 0, 17549, 0, 24166, 1_134_942, "CC-BY-NC-SA 4.0", "https://www.disgenet.org/"
    ),
    "drugbank_ddi": KgSpec(
        "drugbank_ddi",
        13800,
        0,
        0,
        0,
        4_572_188,
        "CC-BY-NC 4.0",
        "https://go.drugbank.com/releases/latest",
    ),
    "spoke": KgSpec(
        "spoke", 16231, 32408, 2604, 24503, 12_884_706, "academic", "https://spoke.rbvi.ucsf.edu/"
    ),
}
