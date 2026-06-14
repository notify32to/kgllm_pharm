#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-diagrams/experiment/main.yaml}"
kgllm-pharm evaluate --config "${CONFIG}"
kgllm-pharm calibrate --config "${CONFIG}"
