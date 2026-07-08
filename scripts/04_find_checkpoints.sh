#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/env.sh

echo "Searching for checkpoint files under $OUTPUT_ROOT"
find "$OUTPUT_ROOT" -type f \( -name "config.json" -o -name "*.safetensors" -o -name "*.pt" \) | sort
