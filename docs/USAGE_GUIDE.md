# C-Transformer Usage Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Compilation](#compilation)
3. [Training from Scratch](#training-from-scratch)
4. [Checkpoint Management](#checkpoint-management)
5. [Resuming Training](#resuming-training)
6. [Text Generation](#text-generation)
7. [Command-Line Reference](#command-line-reference)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Prepare Training Data

```bash
# Install Python dependencies
pip install datasets tiktoken pandas

# Run data preparation script
python3 prepare_data.py
```

**What this does**:
- Downloads TinyStories dataset from HuggingFace
- Packs ~20,000 tokens from stories
- Creates ~18,976 training pairs (1025 tokens each)
- Saves binary files to `data/training_pairs/`
- Creates `metadata.json` with dataset info

**Output**:
```
data/training_pairs/
├── pair_00000.bin
├── pair_00001.bin
├── ...
├── pair_18975.bin
└── metadata.json
```

### 2. Compile the Model

```bash
gcc -o main main.c -lm -fopenmp -march=native -O3
```

**Flags explained**:
- `-lm`: Link math library (for `expf`, `sqrtf`, etc.)
- `-fopenmp`: Enable OpenMP for multi-threading
- `-march=native`: Optimize for your CPU (enables AVX-512 if available)
- `-O3`: Maximum optimization level

### 3. Train Your First Model

```bash
./main \
  --layers 4 \
  --dmodel 256 \
  --ctx 1024 \
  --vocab 50257 \
  --force \
  --train-dir data/training_pairs \
  --train-steps 500 \
  --train-lr 1e-4 \
  --train-log-interval 10 \
  --ckpt-dir checkpoints \
  --ckpt-interval 100
```

**Expected output**:
```
⚙  Requested model  L=4  d_model=256  ctx=1024  vocab=50257
→ Would need ≈ 1.17 GiB (1254195200 bytes)
Allocating huge block... this may page-fault if hugepages are missing
✅ Success! mmap at 0x7f1234567000, 1.17 GiB reserved.
🎲 Initializing random weights (seed=1699889234)...
🧠 Detected 32 logical cores → reserving 4 for OS → using 28 for model
📦 Each core will handle ≈ 37 tokens from context window of 1024 tokens

🎯 Starting training loop (500 steps, lr=0.000100) using data at data/training_pairs
[train] step=1/500  loss=10.692912  perplexity=44038.52  sample=data/training_pairs/pair_00000.bin
[train] step=10/500  loss=10.456123  perplexity=34567.21  sample=data/training_pairs/pair_00009.bin
...
💾 Saved checkpoint to checkpoints/ckpt_step_000100.weights
...
[train] step=500/500  loss=8.271543  perplexity=3912.45  sample=data/training_pairs/pair_00499.bin
✅ Training complete.
💾 Saved checkpoint to checkpoints/ckpt_final.weights
```

---

## Compilation

### Standard Build

```bash
gcc -o main main.c -lm -fopenmp -march=native -O3
```

### Debug Build (with symbols)

```bash
gcc -o main main.c -lm -fopenmp -march=native -O0 -g
```

Use this for debugging with `gdb`:
```bash
gdb ./main
```

### Performance Build (maximum optimization)

```bash
gcc -o main main.c -lm -fopenmp -march=native -O3 -ffast-math -funroll-loops
```

**Warning**: `-ffast-math` may affect numerical precision slightly.

### Check AVX-512 Support

```bash
# Check if your CPU has AVX-512
lscpu | grep avx512

# If no output, your CPU doesn't support AVX-512
# Compile without -march=native:
gcc -o main main.c -lm -fopenmp -O3
```

---

## Training from Scratch

### Basic Training

```bash
./main \
  --layers 4 \
  --dmodel 256 \
  --ctx 1024 \
  --vocab 50257 \
  --force \
  --train-dir data/training_pairs \
  --train-steps 1000 \
  --train-lr 1e-4
```

### Training with All Options

```bash
./main \
  --layers 4           # Number of transformer layers
  --dmodel 256         # Model dimension (embedding size)
  --ctx 1024           # Context window size
  --vocab 50257        # Vocabulary size (GPT-2 tokenizer)
  --head-dim 64        # Attention head dimension
  --force              # Actually allocate memory (required)
  --train-dir data/training_pairs \
  --train-steps 5000 \
  --train-lr 1e-4 \
  --train-log-interval 50 \
  --ckpt-dir checkpoints \
  --ckpt-interval 500
```

### Parameter Guidelines

**Model Size vs Memory**:

| Layers | d_model | Parameters | Memory (train) | Recommended for |
|--------|---------|------------|----------------|-----------------|
| 4 | 256 | ~12M | 1.2 GB | Testing, debugging |
| 8 | 512 | ~85M | 4.5 GB | Small experiments |
| 12 | 768 | ~300M | 12 GB | Medium models |
| 24 | 1024 | ~1.2B | 40 GB | Large models (requires DRAM) |

**Learning Rate**:
- Start with `1e-4` (conservative)
- Increase to `3e-4` if loss decreases smoothly
- Decrease to `3e-5` if loss oscillates or diverges

**Training Steps**:
- Minimum: 500 steps (just to see it works)
- Quick experiment: 5,000 steps (~1 epoch on TinyStories 20K)
- Full training: 50,000+ steps (multiple epochs)

---

## Checkpoint Management

### Automatic Checkpointing

Save checkpoints every N steps:

```bash
./main \
  --train-dir data/training_pairs \
  --train-steps 1000 \
  --ckpt-dir checkpoints \
  --ckpt-interval 100   # Save every 100 steps
```

**Output**:
```
checkpoints/
├── ckpt_step_000100.weights
├── ckpt_step_000200.weights
├── ...
├── ckpt_step_001000.weights
└── ckpt_final.weights
```

### Manual Checkpoint Saving

Checkpoints are saved automatically at:
1. Every `--ckpt-interval` steps
2. End of training (`ckpt_final.weights`)

**No manual saving needed** - it happens during training.

### Checkpoint File Format

Each `.weights` file contains:
- **128-byte header**: Model metadata (layers, vocab, dimensions, etc.)
- **Weight tensors**: All model parameters in order
  - Token embeddings
  - Position embeddings
  - Layer weights (Q, K, V, proj, LayerNorms, MLP)
  - Final LayerNorm

**File size formula**:
```
size_bytes ≈ num_parameters × 4 bytes (FP32)
```

Example for L=4, d=256 model: ~48 MB

---

## Resuming Training

### Load Checkpoint and Continue Training

```bash
# Start fresh training
./main --train-dir data/training_pairs --train-steps 500 --ckpt-dir ckpt1

# Resume from checkpoint
./main \
  --weights ckpt1/ckpt_final.weights \
  --force \
  --train-dir data/training_pairs \
  --train-steps 500 \
  --train-lr 1e-4 \
  --ckpt-dir ckpt2
```

**Important notes**:
- Model architecture must match (layers, dmodel, ctx, vocab)
- Learning rate should typically be **lower** when resuming (e.g., 1e-5 instead of 1e-4)
- Training continues from loaded weights, but step counter resets to 0

### Fine-tuning Workflow

```bash
# 1. Initial training on TinyStories (general language)
./main --train-dir data/tinystories --train-steps 10000 --ckpt-dir pretrain

# 2. Fine-tune on domain-specific data (e.g., conservation reports)
./main \
  --weights pretrain/ckpt_final.weights \
  --force \
  --train-dir data/conservation \
  --train-steps 2000 \
  --train-lr 1e-5 \
  --ckpt-dir finetune
```

---

## Text Generation

### Basic Generation

```bash
# Train model
./main --train-dir data/training_pairs --train-steps 500 --ckpt-dir ckpt

# Generate text from checkpoint
./main \
  --weights ckpt/ckpt_final.weights \
  --force
```

**Expected output**:
```
🚀 Generating text...
⚠️  No prompt provided, using default: "Hello, I am"
Generated token ID: 257
Generated token ID: 314
Generated token ID: 1456
...
```

### Generation with Custom Prompt

```bash
# Tokenize your prompt first (using Python tiktoken)
python3 -c "import tiktoken; enc = tiktoken.get_encoding('gpt2'); print(','.join(map(str, enc.encode('Once upon a time'))))"
# Output: 7454,2402,257,640

# Use token IDs as prompt
./main \
  --weights ckpt/ckpt_final.weights \
  --force \
  --prompt 7454,2402,257,640
```

**Future improvement**: Add built-in tokenization so you can pass raw text.

---

## Command-Line Reference

### Model Architecture

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--layers` | int | 4 | Number of transformer layers |
| `--dmodel` | int | 256 | Model dimension (embedding size) |
| `--ctx` | int | 1024 | Context window (max sequence length) |
| `--vocab` | int | 50257 | Vocabulary size |
| `--head-dim` | int | 64 | Attention head dimension |

### Memory & Execution

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--force` | bool | false | **Required** to allocate memory |
| `--benchmark` | bool | false | Run performance benchmarks |

### Weight Management

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--weights FILE` | string | - | Load weights from file |

### Training

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--train-dir DIR` | string | - | Directory with training pairs |
| `--train-steps N` | int | 0 | Number of training steps |
| `--train-lr LR` | float | 1e-4 | Learning rate |
| `--train-log-interval N` | int | 10 | Log every N steps |

### Checkpointing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ckpt-dir DIR` | string | - | Checkpoint save directory |
| `--ckpt-interval N` | int | 50 | Save checkpoint every N steps |

### Generation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt TOKENS` | string | - | Comma-separated token IDs |

---

## Troubleshooting

### Error: "mmap failed"

**Cause**: Not enough free memory or hugepages not configured.

**Solution**:
```bash
# Check available memory
free -h

# Configure hugepages (Linux)
sudo sysctl -w vm.nr_hugepages=1024

# Make permanent
echo "vm.nr_hugepages=1024" | sudo tee -a /etc/sysctl.conf
```

### Error: "No '.bin' training files found"

**Cause**: `--train-dir` points to wrong location or `prepare_data.py` not run.

**Solution**:
```bash
# Verify files exist
ls data/training_pairs/*.bin | head

# If not, run data preparation
python3 prepare_data.py
```

### Warning: "Failed to save checkpoint"

**Cause**: Checkpoint directory doesn't exist or permission denied.

**Solution**:
```bash
# Create directory manually
mkdir -p checkpoints

# Check permissions
ls -ld checkpoints
```

### Loss Not Decreasing

**Symptoms**:
- Loss stays constant or oscillates
- Perplexity doesn't improve

**Solutions**:
1. **Lower learning rate**: Try `1e-5` instead of `1e-4`
2. **Check data**: Verify training pairs are correct
   ```bash
   # Check a training pair (should be 1025 uint32 values)
   wc -c data/training_pairs/pair_00000.bin
   # Should output: 4100 (1025 × 4 bytes)
   ```
3. **Train longer**: 500 steps may not be enough; try 5,000+
4. **Increase model size**: Tiny models (L=4, d=256) have limited capacity

### Slow Training

**Check CPU utilization**:
```bash
# While training is running, in another terminal:
htop

# You should see ~100% usage on most cores
# If not, check:
# 1. OpenMP threads: export OMP_NUM_THREADS=28
# 2. AVX-512 enabled: gcc -march=native ...
```

**Optimize**:
1. Compile with `-O3 -march=native`
2. Ensure hugepages configured
3. Use smaller model for testing (L=4, d=256)
4. Reduce `--ctx` (e.g., 512 instead of 1024)

### NaN or Inf in Loss

**Symptoms**:
- Loss becomes `nan` or `inf`
- Training crashes with numerical errors

**Solutions**:
1. **Lower learning rate**: Gradients exploding
   ```bash
   --train-lr 1e-5  # or even 1e-6
   ```
2. **Check initialization**: Weights should be small random values
   ```bash
   # Look for "Sample weights" in training output
   # Values should be ~ 0.01 to 0.1, not 0.0 or huge numbers
   ```
3. **Gradient clipping**: Not implemented yet; coming soon

---

## Example Workflows

### Experiment: Quick Training Test

```bash
# 1. Prepare data
python3 prepare_data.py

# 2. Train small model for 500 steps (~ 5 minutes)
./main \
  --layers 4 \
  --dmodel 256 \
  --ctx 1024 \
  --vocab 50257 \
  --force \
  --train-dir data/training_pairs \
  --train-steps 500 \
  --ckpt-dir quick_test

# 3. Check loss decreased
# Expected: loss 10.7 → 8.2 (good!)
# If loss stuck at 10.7, troubleshoot learning rate or data
```

### Production: Full Training Run

```bash
# 1. Train for 20,000 steps with checkpoints
./main \
  --layers 8 \
  --dmodel 512 \
  --ctx 1024 \
  --vocab 50257 \
  --force \
  --train-dir data/training_pairs \
  --train-steps 20000 \
  --train-lr 3e-4 \
  --train-log-interval 100 \
  --ckpt-dir production_run \
  --ckpt-interval 1000

# 2. Monitor training (another terminal)
tail -f production_run/training.log  # if you redirect output

# 3. After training, test generation
./main --weights production_run/ckpt_final.weights --force
```

### Debugging: Understanding Backprop

```bash
# 1. Compile with debug symbols
gcc -o main_debug main.c -lm -fopenmp -march=native -O0 -g

# 2. Run with gdb
gdb ./main_debug

# 3. Set breakpoint in backward pass
(gdb) break backward_causal_softmax
(gdb) run --train-dir data/training_pairs --train-steps 1 --force

# 4. Inspect variables
(gdb) print d_scores_inout[0]
(gdb) print weights[0]
```

---

## Next Steps

1. **Read the math**: See [NUMERICAL_METHODS.md](NUMERICAL_METHODS.md) for complete backprop derivations
2. **Understand the code**: See [BACKPROP_FLOW.md](BACKPROP_FLOW.md) for detailed implementation flow
3. **Compare with others**: See [COMPARISON_WITH_GEMMA_CPP.md](COMPARISON_WITH_GEMMA_CPP.md)
4. **Generate documentation**: Run `doxygen Doxyfile` and open `docs/html/index.html`

---

*Generated with Claude Code - Complete C-Transformer usage guide*
