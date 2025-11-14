#!/usr/bin/env bash
set -euo pipefail

# Same defaults as train_script.sh, but makes it easy to toggle extra compiler flags
LAYERS=${LAYERS:-4}
DMODEL=${DMODEL:-256}
CTX=${CTX:-1024}
VOCAB=${VOCAB:-50257}
TRAIN_DIR=${TRAIN_DIR:-data/training_pairs}
TRAIN_STEPS=${TRAIN_STEPS:-500}
TRAIN_LR=${TRAIN_LR:-1e-4}
TRAIN_LOG_INTERVAL=${TRAIN_LOG_INTERVAL:-10}
WEIGHTS=${WEIGHTS:-gpt2_bump.weights}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints_opt}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-50}
TRAIN_CACHE_SAMPLES=${TRAIN_CACHE_SAMPLES:-}

# Extra compiler knobs (override via environment if needed)
OPT_FLAGS=${OPT_FLAGS:-"-O3 -march=native -mavx512f -fopenmp"}
OPT_DEFINES=${OPT_DEFINES:-"-DUSE_FEATURE_PARALLEL_FC2=1"}

echo "======================"
echo "1) Compile C-Transformer (optimized build)"
echo "======================"
gcc ${OPT_FLAGS} ${OPT_DEFINES} main.c -o main_opt -lm
echo "✅ Built ./main_opt with flags: ${OPT_FLAGS} ${OPT_DEFINES}"

echo
echo "======================"
echo "2) Prepare training data (TinyStories → binary windows)"
echo "======================"
if [ ! -d "${TRAIN_DIR}" ]; then
  echo "Training directory '${TRAIN_DIR}' not found. Running prepare_data.py ..."
  python3 prepare_data.py
else
  echo "Training directory '${TRAIN_DIR}' already exists; skipping data prep."
fi

echo
echo "======================"
echo "3) Launch optimized training"
echo "======================"
CMD=(./main_opt
  --layers "${LAYERS}"
  --dmodel "${DMODEL}"
  --ctx "${CTX}"
  --vocab "${VOCAB}"
  --force
  --train-dir "${TRAIN_DIR}"
  --train-steps "${TRAIN_STEPS}"
  --train-lr "${TRAIN_LR}"
  --train-log-interval "${TRAIN_LOG_INTERVAL}"
  --ckpt-dir "${CHECKPOINT_DIR}"
  --ckpt-interval "${CHECKPOINT_INTERVAL}"
)

if [ -n "${TRAIN_CACHE_SAMPLES}" ]; then
  CMD+=(--train-cache-samples "${TRAIN_CACHE_SAMPLES}")
fi

if [ -f "${WEIGHTS}" ]; then
  echo "Found weights file '${WEIGHTS}', will fine-tune from it."
  CMD+=(--weights "${WEIGHTS}")
else
  echo "Weights file '${WEIGHTS}' not found; training will start from random initialization."
fi

echo "Running (optimized): ${CMD[*]}"
"${CMD[@]}"
