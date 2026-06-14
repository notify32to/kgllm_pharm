from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_atomic(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp")
    tmp_name = handle.name
    handle.close()
    torch.save(state, tmp_name)
    os.replace(tmp_name, target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    loaded: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    return loaded
