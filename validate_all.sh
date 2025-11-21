#!/usr/bin/env bash
set -euo pipefail

# Run the full validation pipeline for a given prompt:
#  1) Check raw weights layout (debug_weights.py)
#  2) Compare C embeddings vs HF (validate_embeddings.py + --debug-embed)
#  3) Compare hidden state + logits vs HF (validate_inference.sh)
#
# Usage:
#   ./validate_all.sh "Once upon a time"
#   ./validate_all.sh "Hello world" gpt2_bump.weights ./main 10
#
# Args:
#   $1 = prompt text (required)
#   $2 = weights file (default: gpt2_bump.weights)
#   $3 = executable (default: ./main)
#   $4 = top-k for logits (default: 10)

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 \"PROMPT TEXT\" [WEIGHTS] [EXECUTABLE] [TOP_K]"
  exit 1
fi

PROMPT="$1"
WEIGHTS="${2:-gpt2_bump.weights}"
EXECUTABLE="${3:-./main}"
TOP_K="${4:-10}"

echo "========================"
echo "🧪 FULL VALIDATION RUN"
echo "========================"
echo "Prompt:      \"$PROMPT\""
echo "Weights:     $WEIGHTS"
echo "Executable:  $EXECUTABLE"
echo "Top-K:       $TOP_K"
echo

echo "1) Checking raw weights (export/layout)..."
python3 debug_weights.py
echo

echo "2) Validating embeddings (token + position)..."
python3 validate_embeddings.py "$PROMPT" \
  --weights "$WEIGHTS" \
  --executable "$EXECUTABLE"
echo

echo "3) Validating hidden state + logits..."
./validate_inference.sh "$PROMPT" "$WEIGHTS" "$EXECUTABLE" "$TOP_K"
echo

echo "✅ Validation pipeline complete."

