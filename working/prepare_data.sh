#!/usr/bin/env bash
set -euo pipefail
echo "The public release runs on the deterministic synthetic-cohort service."
echo "To use real arrays, place feature/label files on disk and write a manifest"
echo "consumed by kgllm_pharm.yard.manifest, then select the real encoder port."
