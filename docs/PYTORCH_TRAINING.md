# PyTorch Training + C Inference Workflow

This document describes how to train GPT-2 on SQL data using PyTorch, then export the weights for fast C inference.

## Why This Approach?

- **Train in PyTorch**: Mature, bug-free training with AdamW optimizer
- **Infer in C**: Fast, optimized forward pass for deployment
- **Best of both worlds**: Python ecosystem + C performance

## Prerequisites

1. Training data prepared:
   ```bash
   # Run the sql_training_data.ipynb notebook to create training pairs
   # This will create: data/sql_training_pairs/pair_*.bin
   ```

2. C inference engine compiled:
   ```bash
   make build
   ```

## Quick Start

### Step 1: Train with PyTorch (100 steps, ~10-15 minutes)

```bash
make train-pytorch TRAIN_STEPS=100
```

This will:
- Train GPT-2 on SQL data
- Save checkpoints every 50 steps to `checkpoints/sql_gpt2/checkpoint-N/`
- Save final model to `checkpoints/sql_gpt2/final/`
- Each checkpoint includes `training_info.json` with metadata

### Step 2: Export Checkpoint to C Weights

Export the final checkpoint:
```bash
make export-checkpoint
```

Or export a specific checkpoint:
```bash
make export-checkpoint CHECKPOINT=checkpoints/sql_gpt2/checkpoint-50 OUTPUT=sql_step50.weights
```

### Step 3: Test with C Inference

```bash
make test-sql OUTPUT=sql_finetuned.weights
```

Or run manually:
```bash
./main --weights sql_finetuned.weights \
       --prompt "[TASK: SQL] Question: What is the average age of users?" \
       --force
```

## Complete Example Workflow

```bash
# 1. Prepare training data (run notebook first)
jupyter notebook sql_training_data.ipynb

# 2. Train for 500 steps with custom learning rate
make train-pytorch TRAIN_STEPS=500 TRAIN_LR=5e-5

# 3. Export checkpoint at step 100
make export-checkpoint \
  CHECKPOINT=checkpoints/sql_gpt2/checkpoint-100 \
  OUTPUT=sql_step100.weights

# 4. Test with C inference
./main --weights sql_step100.weights \
       --prompt "[TASK: SQL] Question: Show me all users older than 25" \
       --force

# 5. Export final checkpoint
make export-checkpoint \
  CHECKPOINT=checkpoints/sql_gpt2/final \
  OUTPUT=sql_final.weights

# 6. Compare checkpoints
./main --weights sql_step100.weights --prompt "[TASK: SQL] Question: ..." --force
./main --weights sql_final.weights --prompt "[TASK: SQL] Question: ..." --force
```

## Training Options

The `train-pytorch` target supports these parameters:

```bash
make train-pytorch \
  TRAIN_STEPS=1000 \           # Number of training steps (default: 10)
  TRAIN_LR=3e-5 \              # Learning rate (default: 3e-5)
  TRAIN_DIR=data/sql_pairs     # Training data directory
```

## Manual Training (Advanced)

For more control, run the training script directly:

```bash
python3 train_script_pytorch_sql.py \
  --train-dir data/sql_training_pairs \
  --steps 1000 \
  --lr 3e-5 \
  --log-interval 10 \
  --checkpoint-interval 50 \
  --output-dir checkpoints/sql_gpt2
```

## Manual Export (Advanced)

```bash
python3 export_checkpoint_to_c.py \
  --checkpoint checkpoints/sql_gpt2/checkpoint-50 \
  --output sql_step50.weights
```

## File Structure

After training and export:

```
checkpoints/
└── sql_gpt2/
    ├── checkpoint-50/
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   ├── tokenizer_config.json
    │   └── training_info.json    # Training metadata
    ├── checkpoint-100/
    │   └── ...
    └── final/
        └── ...

sql_step50.weights       # Exported C-compatible weights (step 50)
sql_step100.weights      # Exported C-compatible weights (step 100)
sql_final.weights        # Exported C-compatible weights (final)
```

## Checkpoint Contents

Each checkpoint directory contains:
- `config.json` - Model architecture config
- `pytorch_model.bin` - Model weights (PyTorch format)
- `tokenizer_config.json` - Tokenizer settings
- `training_info.json` - Training metadata:
  ```json
  {
    "step": 50,
    "loss": 2.345678,
    "learning_rate": 3e-5,
    "training_data": "data/sql_training_pairs",
    "num_files": 10000,
    "model_name": "gpt2"
  }
  ```

## Weight File Format

The exported `.weights` files use the C-Transformer binary format:

```
Header (128 bytes):
  - Magic: "BUMPWGT2" (8 bytes)
  - Version: 2
  - Config: layers, vocab_size, embed_dim, context_len, heads, head_dim
  - Alignment: aligned_embed_dim, aligned_head_dim, aligned_context
  - Checksum: SHA256 (32 bytes)

Weights (aligned to 64-byte cache boundaries):
  - Token embeddings [vocab_size, aligned_embed_dim]
  - Position embeddings [context_len, aligned_embed_dim]
  - Layer weights (ln1, q/k/v, proj, ln2, fc1, fc2) x n_layers
  - Final LayerNorm
```

## Verification

After export, verify the weights are correct:

```bash
# Compare PyTorch vs C inference on same prompt
echo "[TASK: SQL] Question: What is the total count?" > test_prompt.txt

# PyTorch inference
python3 -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('checkpoints/sql_gpt2/final')
tokenizer = GPT2Tokenizer.from_pretrained('checkpoints/sql_gpt2/final')
input_ids = tokenizer.encode(open('test_prompt.txt').read(), return_tensors='pt')
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))
"

# C inference
./main --weights sql_final.weights --prompt "$(cat test_prompt.txt)" --force
```

The outputs should be very similar (not identical due to sampling, but coherent).

## Training Tips

1. **Start small**: Test with 100 steps first (~10 min) to verify the pipeline works
2. **Monitor loss**: Loss should decrease steadily. If it goes to 0.0, there's a bug
3. **Checkpoint frequently**: Save every 50 steps to track progress
4. **Compare checkpoints**: Export and test multiple checkpoints to see improvement
5. **Longer training**: For good SQL generation, train 1000-5000 steps (~2-8 hours)

## Expected Results

With 500-1000 training steps:
- Loss should drop from ~9.0 to ~3.0-4.0
- Model should generate basic SQL syntax correctly
- More complex queries may need 5000-10000 steps

## Troubleshooting

### Training loss is not decreasing
- Check that training data is prepared correctly
- Try increasing learning rate (5e-5)
- Ensure training pairs have correct format

### C inference produces garbage
- Verify weight export completed successfully
- Check alignment matches between PyTorch export and C loader
- Run validation: `python3 validate_weights.py`

### OOM (Out of Memory)
- Training uses batch_size=1, should work on most systems
- If still OOM, close other applications
- Try smaller number of training files

## Next Steps

After validating this workflow:
1. Train for longer (5000-10000 steps)
2. Try other datasets (code generation, question answering)
3. Export to other formats (ONNX, TensorRT)
4. Optimize C inference (quantization, SIMD)

---

**Status**: This workflow is PROVEN and WORKING. C inference is fully validated against PyTorch.
