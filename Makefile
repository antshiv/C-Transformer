# C-Transformer Makefile
# Provides convenient shortcuts for building, testing, and validating

# Compiler settings
CC = gcc
CFLAGS = -O3 -march=native -mavx512f -fopenmp
LDFLAGS = -lm
TARGET = main
SOURCE = main.c

# Default weights file
WEIGHTS = gpt2_bump.weights
PROMPT = "Hello World"
LAYER = 0
TOPK = 20

# Training parameters
TRAIN_DIR = data/sql_training_pairs
TRAIN_STEPS = 10
TRAIN_LR = 3e-5
TRAIN_PROMPT = "SELECT * FROM users WHERE age > 10;"
GEN_TOKENS = 20
TRAIN_OPTIMIZER = adam

# Python interpreter
PYTHON = python3

# ============================================================================
# Build targets
# ============================================================================

.PHONY: all build clean help

all: build

build:
	@echo "🔨 Compiling $(SOURCE)..."
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)
	@echo "✅ Build complete: ./$(TARGET)"

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(TARGET)
	rm -f validation_*.csv
	@echo "✅ Clean complete"

# ============================================================================
# Quick test targets
# ============================================================================

.PHONY: run test-gpt2 test-large

run: build
	@echo "🚀 Running GPT-2 inference..."
	./$(TARGET) --weights $(WEIGHTS) --force

test-gpt2: build
	@echo "🧪 Testing GPT-2 with custom prompt..."
	./$(TARGET) --weights $(WEIGHTS) --prompt "2061,318,649,27288,362,358,1099,30" --force

test-large:
	@echo "🧪 Testing large model configuration..."
	./$(TARGET) --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force

benchmark: build
	@echo "⚡ Running benchmark..."
	./$(TARGET) --layers 96 --dmodel 8192 --ctx 4096 --vocab 50000 --force --benchmark

# ============================================================================
# Validation targets (Forward Pass)
# ============================================================================

.PHONY: validate validate-all validate-weights validate-embed validate-layers validate-qkv validate-attn

validate: validate-all

validate-all: build
	@echo "🧪 Running full validation pipeline..."
	./validate_all.sh $(PROMPT) $(WEIGHTS) ./$(TARGET) $(TOPK) $(LAYER)

validate-weights:
	@echo "🔍 Validating weight loading..."
	$(PYTHON) validation/validate_weights.py

validate-embed: build
	@echo "🔍 Validating embeddings..."
	$(PYTHON) validation/validate_embeddings.py $(PROMPT) --weights $(WEIGHTS) --executable ./$(TARGET)

validate-layers: build
	@echo "🔍 Validating layer stages (layer $(LAYER))..."
	$(PYTHON) validation/validate_layer_stages.py $(PROMPT) \
		--layer $(LAYER) \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET)

validate-qkv: build
	@echo "🔍 Validating QKV projection (layer $(LAYER))..."
	$(PYTHON) validation/validate_qkv.py $(PROMPT) \
		--layer $(LAYER) \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET)

validate-attn: build
	@echo "🔍 Validating attention (layer $(LAYER))..."
	$(PYTHON) validation/validate_attn.py $(PROMPT) \
		--layer $(LAYER) \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET)

validate-logits: build
	@echo "🔍 Validating logits..."
	$(PYTHON) validation/validate_vs_pytorch.py $(PROMPT) \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET) \
		--top-k $(TOPK) \
		--compare-hidden

# ============================================================================
# Validation targets (Backward Pass)
# ============================================================================

.PHONY: validate-backward validate-backward-stages validate-gradients

validate-backward: build
	@echo "🔍 Validating backward pass..."
	$(PYTHON) validation/validate_backward.py $(PROMPT) \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET)

validate-backward-stages: build
	@echo "🔍 Validating backward layer stages (layer $(LAYER))..."
	$(PYTHON) unittest/validate_backward_layer_stages.py $(PROMPT) \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET) \
		--model-name gpt2 \
		--layer $(LAYER)

validate-gradients: validate-backward

# ============================================================================
# Unit test targets
# ============================================================================

.PHONY: test-lm-head test-training-step

test-lm-head: build
	@echo "🧪 Testing LM head..."
	$(PYTHON) unittest/validate_lm_head_vs_c.py "Hello" \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET)

test-training-step: build
	@echo "🧪 Testing training step (layer $(LAYER))..."
	$(PYTHON) unittest/validate_training_step_vs_hf.py \
		--pair-file data/sql_training_pairs/pair_00000.bin \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET) \
		--model-name gpt2 \
		--layer $(LAYER)

# ============================================================================
# Training comparison targets
# ============================================================================

.PHONY: compare-training train-compare

compare-training: build
	@echo "🎓 Comparing C vs PyTorch training..."
	@echo "   Steps: $(TRAIN_STEPS), LR: $(TRAIN_LR), Optimizer: $(TRAIN_OPTIMIZER)"
	@echo "   Training dir: $(TRAIN_DIR)"
	$(PYTHON) validation/compare_training_c_vs_pytorch.py \
		--train-dir $(TRAIN_DIR) \
		--weights $(WEIGHTS) \
		--executable ./$(TARGET) \
		--model-name gpt2 \
		--steps $(TRAIN_STEPS) \
		--lr $(TRAIN_LR) \
		--log-interval 1 \
		--optimizer $(TRAIN_OPTIMIZER) \
		--prompt $(TRAIN_PROMPT) \
		--gen-tokens $(GEN_TOKENS)

train-compare: compare-training

# ============================================================================
# PyTorch Training + Export to C
# ============================================================================

.PHONY: train-pytorch export-checkpoint test-sql

# Train GPT-2 on SQL data using PyTorch
train-pytorch:
	@echo "🎓 Training GPT-2 on SQL data (PyTorch)..."
	@echo "   Steps: $(TRAIN_STEPS), LR: $(TRAIN_LR)"
	@echo "   Training dir: $(TRAIN_DIR)"
	@echo "   Checkpoints: every 50 steps"
	$(PYTHON) train_script_pytorch_sql.py \
		--train-dir $(TRAIN_DIR) \
		--steps $(TRAIN_STEPS) \
		--lr $(TRAIN_LR) \
		--log-interval 10 \
		--checkpoint-interval 50 \
		--output-dir checkpoints/sql_gpt2

# Export a checkpoint to C-compatible weights
# Usage: make export-checkpoint CHECKPOINT=checkpoints/sql_gpt2/checkpoint-50 OUTPUT=sql_step50.weights
CHECKPOINT ?= checkpoints/sql_gpt2/final
OUTPUT ?= sql_finetuned.weights

export-checkpoint:
	@echo "📦 Exporting checkpoint to C weights..."
	@echo "   Checkpoint: $(CHECKPOINT)"
	@echo "   Output:     $(OUTPUT)"
	$(PYTHON) export_checkpoint_to_c.py \
		--checkpoint $(CHECKPOINT) \
		--output $(OUTPUT)

# Test SQL model with C inference
test-sql: build
	@echo "🧪 Testing SQL fine-tuned model..."
	@if [ ! -f "$(OUTPUT)" ]; then \
		echo "❌ Weights file not found: $(OUTPUT)"; \
		echo "   Run 'make train-pytorch' first, then 'make export-checkpoint'"; \
		exit 1; \
	fi
	./$(TARGET) --weights $(OUTPUT) \
		--prompt "[TASK: SQL] Question: What is the average age of users?" \
		--force

# ============================================================================
# Quick validation shortcuts
# ============================================================================

.PHONY: quick-test full-test forward backward

# Quick smoke test - just check if basics work
quick-test: build validate-weights validate-embed validate-qkv
	@echo "✅ Quick test complete!"

# Full forward pass validation
forward: build validate-all
	@echo "✅ Forward pass validation complete!"

# Full backward pass validation
backward: build validate-backward validate-backward-stages
	@echo "✅ Backward pass validation complete!"

# Complete test suite (forward + backward)
full-test: forward backward
	@echo "✅ Full test suite complete!"

# ============================================================================
# Development helpers
# ============================================================================

.PHONY: rebuild debug watch

# Force rebuild
rebuild: clean build

# Build with debug symbols
debug:
	@echo "🔨 Compiling with debug symbols..."
	$(CC) -g -O0 -march=native -mavx512f -fopenmp $(SOURCE) -o $(TARGET) $(LDFLAGS)

# Watch for changes and rebuild (requires inotify-tools)
watch:
	@echo "👀 Watching for changes..."
	@while true; do \
		inotifywait -e modify $(SOURCE) 2>/dev/null && make build; \
	done

# ============================================================================
# Help
# ============================================================================

help:
	@echo "C-Transformer Makefile"
	@echo "====================="
	@echo ""
	@echo "Build targets:"
	@echo "  make build              - Compile main.c (default)"
	@echo "  make clean              - Remove build artifacts"
	@echo "  make rebuild            - Clean + build"
	@echo "  make debug              - Build with debug symbols"
	@echo ""
	@echo "Quick tests:"
	@echo "  make run                - Run GPT-2 inference"
	@echo "  make test-gpt2          - Test with encoded prompt"
	@echo "  make benchmark          - Run performance benchmark"
	@echo ""
	@echo "Validation (Forward):"
	@echo "  make validate-all       - Full validation pipeline"
	@echo "  make validate-weights   - Check weight loading"
	@echo "  make validate-embed     - Check embeddings"
	@echo "  make validate-layers    - Check layer stages"
	@echo "  make validate-qkv       - Check QKV projection"
	@echo "  make validate-attn      - Check attention"
	@echo "  make validate-logits    - Check final logits"
	@echo ""
	@echo "Validation (Backward):"
	@echo "  make validate-backward         - Check gradients"
	@echo "  make validate-backward-stages  - Check layer gradients"
	@echo ""
	@echo "Training:"
	@echo "  make compare-training   - Compare C vs PyTorch training"
	@echo "  make train-compare      - Alias for compare-training"
	@echo "  make train-pytorch      - Train GPT-2 on SQL data (PyTorch only)"
	@echo "  make export-checkpoint  - Export checkpoint to C weights"
	@echo "  make test-sql           - Test SQL fine-tuned model with C"
	@echo ""
	@echo "Test suites:"
	@echo "  make quick-test         - Quick smoke test"
	@echo "  make forward            - Full forward validation"
	@echo "  make backward           - Full backward validation"
	@echo "  make full-test          - Complete test suite"
	@echo ""
	@echo "Options (set as VARIABLE=value):"
	@echo "  WEIGHTS=<file>          - Weight file (default: gpt2_bump.weights)"
	@echo "  PROMPT=<text>           - Test prompt (default: \"Hello World\")"
	@echo "  LAYER=<n>               - Layer to test (default: 0)"
	@echo "  TOPK=<n>                - Top-K for logits (default: 20)"
	@echo "  TRAIN_DIR=<dir>         - Training data dir (default: data/sql_training_pairs)"
	@echo "  TRAIN_STEPS=<n>         - Training steps (default: 10)"
	@echo "  TRAIN_LR=<float>        - Learning rate (default: 3e-5)"
	@echo "  TRAIN_PROMPT=<text>     - Prompt for inference test (default: SQL query)"
	@echo "  GEN_TOKENS=<n>          - Tokens to generate (default: 20)"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make validate-all PROMPT=\"Once upon a time\" LAYER=5"
	@echo "  make validate-backward LAYER=11"
	@echo "  make compare-training TRAIN_STEPS=20 TRAIN_LR=5e-5"
	@echo "  make full-test"
