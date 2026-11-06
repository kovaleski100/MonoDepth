#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
# Use the caller's activated Python environment. No automatic CUDA installation.
# Defaults: seeds 42/43/44, 8 visual examples per seed. Override with --seeds / --visual-samples.
# Existing experiments: --resume reuses completed runs; --visuals-only only exports their PNG/HTML gallery.
python -m pixel_depth.compare "$@"
