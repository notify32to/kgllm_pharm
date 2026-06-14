from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class ManifestUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    features_path: Path
    labels_path: Path


def read_manifest(path: str | Path) -> dict[str, ManifestEntry]:
    target = Path(path)
    if not target.exists():
        raise ManifestUnavailable(f"manifest not found: {target}")
    payload = json.loads(target.read_text())
    entries: dict[str, ManifestEntry] = {}
    for source, spec in payload.items():
        entries[source] = ManifestEntry(
            source=source,
            features_path=target.parent / str(spec["features"]),
            labels_path=target.parent / str(spec["labels"]),
        )
    return entries


def load_array(path: str | Path) -> FloatArray:
    target = Path(path)
    if not target.exists():
        raise ManifestUnavailable(f"array not found: {target}")
    array: FloatArray = np.load(target).astype(np.float64)
    return array
