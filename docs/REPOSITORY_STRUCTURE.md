# Repository Structure

Clean, organized directory layout for C-Transformer.

```
C-Transformer/
├── 📄 Core Files
│   ├── main.c                          # Main C implementation (~15k lines)
│   ├── Makefile                        # Build and validation targets
│   ├── README.md                       # Project overview
│   └── ARCHIVE_NOTICE.md              # Why this repo is archived
│
├── 🎓 Training Scripts (Root - User-facing)
│   ├── train_script_pytorch_sql.py    # Train GPT-2 on SQL (PyTorch)
│   ├── export_checkpoint_to_c.py      # Export checkpoint → C weights
│   └── generate_weights_simple.py     # Generate base GPT-2 weights
│
├── 📚 Documentation
│   ├── PYTORCH_TRAINING.md            # Train in PyTorch, infer in C
│   ├── QUICKSTART_SQL_TRAINING.md     # Quick start guide
│   ├── REPOSITORY_STRUCTURE.md        # This file
│   └── docs/                          # Doxygen HTML documentation
│
├── 🧪 validation/                      # Validation scripts
│   ├── validate_embeddings.py         # Check embeddings vs PyTorch
│   ├── validate_qkv.py                # Check QKV projection
│   ├── validate_attn.py               # Check attention mechanism
│   ├── validate_layer_stages.py       # Check layer internals
│   ├── validate_backward.py           # Check gradients
│   ├── validate_vs_pytorch.py         # Check logits
│   ├── compare_training_c_vs_pytorch.py  # Compare training
│   ├── debug_weights.py               # Debug weight loading
│   └── test_gpt2_weights.py           # Test weight format
│
├── 🧪 unittest/                        # Unit tests
│   ├── validate_lm_head_vs_c.py
│   ├── validate_training_step_vs_hf.py
│   ├── validate_backward_layer_stages.py
│   └── validate_training_step_next_token.py
│
├── 🛠️ utils/                          # Utility scripts
│   ├── encode.py                      # Encode text to token IDs
│   ├── decode.py                      # Decode token IDs to text
│   ├── run.py                         # Run inference with Python wrapper
│   └── prepare_data.py                # Data preparation helpers
│
├── 🔧 tools/                          # Development tools
│   ├── coredump_serial_log_parser.py  # Parse crash dumps
│   ├── decode_output.py               # Decode binary output
│   └── extract_script.py              # Extract embedded scripts
│
├── 📜 scripts/                        # Shell scripts
│   ├── train_script.sh                # Old training script
│   ├── train_script_debug_step.sh     # Debug training
│   ├── train_script_optimized.sh      # Optimized training
│   └── validate_inference.sh          # Validate inference
│
├── 📓 notebooks/                      # Jupyter notebooks
│   ├── sql_training_data.ipynb        # Prepare SQL training data
│   ├── pytorch_to_c_weights.ipynb     # Weight conversion notebook
│   └── training_datasets.ipynb        # Dataset exploration
│
├── 📊 data/                           # Training data
│   └── sql_training_pairs/            # SQL training pairs (binary)
│       ├── pair_00000.bin
│       ├── pair_00001.bin
│       └── ...
│
├── 💾 Weights & Checkpoints
│   ├── gpt2_bump.weights              # Base GPT-2 weights
│   ├── sql_finetuned.weights          # SQL fine-tuned weights
│   └── checkpoints/                   # PyTorch checkpoints
│       └── sql_gpt2/
│           ├── checkpoint-50/
│           ├── checkpoint-100/
│           └── final/
│
└── 🏗️ Build Artifacts
    ├── main                           # Compiled binary
    ├── main_test                      # Test binary
    └── __pycache__/                   # Python cache
```

## Quick Access

### I want to...

**Train a model:**
```bash
make train-pytorch TRAIN_STEPS=100
```

**Export checkpoint:**
```bash
make export-checkpoint CHECKPOINT=checkpoints/sql_gpt2/final
```

**Run inference:**
```bash
./main --weights gpt2_bump.weights --force
```

**Validate forward pass:**
```bash
make validate-all
```

**Validate backward pass:**
```bash
make validate-backward
```

**Build the project:**
```bash
make build
```

**Run tests:**
```bash
make quick-test      # Fast smoke test
make full-test       # Complete test suite
```

### File Purpose Quick Reference

| File | Purpose |
|------|---------|
| `main.c` | Core transformer implementation |
| `Makefile` | Build, test, train targets |
| `train_script_pytorch_sql.py` | Train GPT-2 on SQL |
| `export_checkpoint_to_c.py` | Convert PyTorch → C |
| `validation/*.py` | Numerical validation vs PyTorch |
| `unittest/*.py` | Unit tests for components |
| `utils/*.py` | Encode/decode utilities |
| `notebooks/*.ipynb` | Data prep and experiments |
| `scripts/*.sh` | Shell automation |

## Navigation Tips

- **Start here**: `QUICKSTART_SQL_TRAINING.md`
- **Detailed docs**: `PYTORCH_TRAINING.md`
- **Why archived**: `ARCHIVE_NOTICE.md`
- **API docs**: Open `docs/html/index.html`

## Folder Details

### validation/
All scripts that compare C output against PyTorch reference implementation. Each script focuses on one component or stage.

### unittest/
Unit tests that validate individual functions with known inputs/outputs. More granular than validation scripts.

### utils/
Helper scripts for encoding, decoding, and data preparation. Not validation, just utilities.

### tools/
Development/debugging tools for parsing logs, extracting data, etc.

### scripts/
Shell scripts for automation. Mostly older training scripts kept for reference.

### notebooks/
Jupyter notebooks for data exploration, weight conversion experiments, and training data preparation.

### data/
Training data in binary format. Created by notebooks, consumed by training scripts.

### checkpoints/
PyTorch model checkpoints saved during training. Can be exported to C weights.

## Clean Build

To start fresh:

```bash
make clean                    # Remove build artifacts
rm -rf checkpoints/          # Remove checkpoints
rm -rf __pycache__/          # Remove Python cache
rm -f *.weights              # Remove weight files
```

## Contributing

This repo is archived, but you can:
1. Fork and modify for your own experiments
2. Reference validation methodology in your projects
3. Use PyTorch→C export pipeline

---

Last updated: November 2025
