#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-diagrams/experiment/main.yaml}"
OUT="${2:-runs/checkpoint.pt}"
kgllm-pharm train --config "${CONFIG}" --out "${OUT}"
