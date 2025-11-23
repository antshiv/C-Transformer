#!/usr/bin/env bash
set -euo pipefail

# Debug a single training_step() on one binary window by:
#   1) Picking a pair_*.bin file,
#   2) Building a next-token LM prompt from it (ctx_len+1 tokens),
#   3) Running ./main --debug-train-step with those tokens.
#
# Environment overrides:
#   TRAIN_DIR   - directory with pair_*.bin (default: data/sql_training_pairs)
#   PAIR_FILE   - explicit pair file path; if empty, first pair_*.bin is used
#   WEIGHTS     - C weights file (default: gpt2_bump.weights)
#   EXECUTABLE  - C binary (default: ./main)

TRAIN_DIR=${TRAIN_DIR:-data/sql_training_pairs}
PAIR_FILE=${PAIR_FILE:-}
WEIGHTS=${WEIGHTS:-gpt2_bump.weights}
EXECUTABLE=${EXECUTABLE:-./main}

echo "======================"
echo "Debug single training_step()"
echo "======================"

if [ -z "${PAIR_FILE}" ]; then
  if [ ! -d "${TRAIN_DIR}" ]; then
    echo "❌ TRAIN_DIR '${TRAIN_DIR}' not found and no PAIR_FILE specified."
    exit 1
  fi
  PAIR_FILE=$(ls "${TRAIN_DIR}"/pair_*.bin 2>/dev/null | sort | head -n 1 || true)
  if [ -z "${PAIR_FILE}" ]; then
    echo "❌ No pair_*.bin files found in '${TRAIN_DIR}'."
    exit 1
  fi
fi

if [ ! -f "${PAIR_FILE}" ]; then
  echo "❌ Pair file '${PAIR_FILE}' not found."
  exit 1
fi

if [ ! -f "${WEIGHTS}" ]; then
  echo "❌ Weights file '${WEIGHTS}' not found."
  exit 1
fi

echo "🔨 Recompiling ./main to ensure latest flags (including --debug-train-step)..."
gcc -O3 -march=native -mavx512f -fopenmp main.c -o main -lm
EXECUTABLE=./main

echo "Pair file: ${PAIR_FILE}"
echo "Weights:   ${WEIGHTS}"
echo "Binary:    ${EXECUTABLE}"
echo

# Use Python to read ctx_len and tokens and build a ctx_len+1 prompt.
PROMPT_TOKENS=$(
  python3 - << 'PY' "${PAIR_FILE}"
import struct, sys, numpy as np

path = sys.argv[1]
with open(path, "rb") as f:
    data = f.read()
if len(data) < 4:
    raise SystemExit(f"File too small to contain header: {path}")
ctx_len, tgt_len = struct.unpack("<HH", data[:4])
tokens = np.frombuffer(data[4:], dtype="<u4")
if tokens.size < ctx_len + 1:
    raise SystemExit(
        f"Not enough tokens in {path}: need ctx_len+1={ctx_len+1}, got {tokens.size}"
    )
prompt = tokens[: ctx_len + 1]
print(",".join(str(int(t)) for t in prompt))
PY
)

echo "Prompt (first 16 tokens): $(echo "${PROMPT_TOKENS}" | cut -d',' -f1-16)..."
echo
echo "▶ Running single training_step debug (next-token LM)..."
"${EXECUTABLE}" --weights "${WEIGHTS}" --force \
  --prompt "${PROMPT_TOKENS}" \
  --debug-train-step
