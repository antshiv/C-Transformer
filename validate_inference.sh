#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to validate C-Transformer inference vs HuggingFace GPT-2.
#
# Usage:
#   ./validate_inference.sh "Once upon a time"
#   ./validate_inference.sh "Hello world" gpt2_bump.weights ./main 20
#
# Args (all optional except prompt):
#   $1 = prompt text
#   $2 = weights file (default: gpt2_bump.weights)
#   $3 = executable (default: ./main)
#   $4 = top-k (default: 10)

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 \"PROMPT TEXT\" [WEIGHTS] [EXECUTABLE] [TOP_K]"
  exit 1
fi

PROMPT="$1"
WEIGHTS="${2:-gpt2_bump.weights}"
EXECUTABLE="${3:-./main}"
TOP_K="${4:-10}"

echo "🧪 Validating inference for prompt: \"$PROMPT\""
echo "   Weights:     $WEIGHTS"
echo "   Executable:  $EXECUTABLE"
echo "   Top-K:       $TOP_K"
echo

python3 validate_vs_pytorch.py "$PROMPT" \
  --weights "$WEIGHTS" \
  --executable "$EXECUTABLE" \
  --top-k "$TOP_K" \
  --save-csv "validation_${TOP_K}_topk.csv" \
  --compare-hidden
