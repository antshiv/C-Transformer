# Quick Start: SQL Training Pipeline

Train GPT-2 on SQL data using PyTorch, then run fast inference in C.

## One-Command Workflow

```bash
# 1. Train (100 steps = ~10 minutes)
make train-pytorch TRAIN_STEPS=100

# 2. Export to C weights
make export-checkpoint

# 3. Test with C inference
make test-sql
```

That's it! Your SQL-trained model is now running in C.

## What Just Happened?

1. **Training**: PyTorch trained GPT-2 on your SQL data pairs
   - Saved checkpoints every 50 steps
   - Final model: `checkpoints/sql_gpt2/final/`

2. **Export**: Converted PyTorch checkpoint to C-compatible binary
   - Output: `sql_finetuned.weights`
   - Format: Aligned binary with SHA256 checksum

3. **Inference**: C engine loaded weights and generated SQL
   - Uses validated forward pass (diff < 1e-5 vs PyTorch)
   - Optimized with AVX-512 and OpenMP

## Example Output

```bash
$ make test-sql
🧪 Testing SQL fine-tuned model...
./main --weights sql_finetuned.weights \
       --prompt "[TASK: SQL] Question: What is the average age of users?" \
       --force

Loading weights from sql_finetuned.weights
✓ Model loaded: 12 layers, 768 dim, 50257 vocab
✓ Forward pass complete

Generated SQL:
SELECT AVG(age) FROM users;
```

## Longer Training for Better Results

```bash
# Train for 500 steps (~1 hour)
make train-pytorch TRAIN_STEPS=500

# Export final model
make export-checkpoint

# Test
make test-sql
```

## Compare Different Checkpoints

```bash
# Export checkpoint at step 50
make export-checkpoint \
  CHECKPOINT=checkpoints/sql_gpt2/checkpoint-50 \
  OUTPUT=sql_step50.weights

# Export checkpoint at step 100
make export-checkpoint \
  CHECKPOINT=checkpoints/sql_gpt2/checkpoint-100 \
  OUTPUT=sql_step100.weights

# Export final
make export-checkpoint \
  CHECKPOINT=checkpoints/sql_gpt2/final \
  OUTPUT=sql_final.weights

# Compare outputs
./main --weights sql_step50.weights --prompt "[TASK: SQL] Question: ..." --force
./main --weights sql_step100.weights --prompt "[TASK: SQL] Question: ..." --force
./main --weights sql_final.weights --prompt "[TASK: SQL] Question: ..." --force
```

## Training Data Format

Your SQL training pairs (from `sql_training_data.ipynb`):

```
[TASK: SQL] Question: Show me all users older than 25 → SELECT * FROM users WHERE age > 25
[TASK: SQL] Question: What is the average salary? → SELECT AVG(salary) FROM employees
[TASK: SQL] Question: Count total orders → SELECT COUNT(*) FROM orders
```

Each pair is tokenized and saved as:
- `data/sql_training_pairs/pair_00000.bin`
- `data/sql_training_pairs/pair_00001.bin`
- ... (10,000 pairs)

## Why This Works

✅ **PyTorch handles training**:
- Mature AdamW optimizer
- Proper gradient computation
- No bugs to debug

✅ **C handles inference**:
- Fast forward pass (validated)
- Optimized memory layout
- AVX-512 SIMD

✅ **Best of both worlds**:
- Train in Python ecosystem
- Deploy with C performance
- No need to debug C backprop

## Next Steps

Now that you have a working pipeline, you can:

1. **Focus on kernels**: Optimize C inference with custom CUDA/AVX-512 kernels
2. **Try Qwen**: Use same workflow for Qwen architecture
3. **Experiment**: Different datasets, hyperparameters, model sizes

The backprop bugs in C don't matter anymore - PyTorch handles all training!

---

For detailed documentation, see `PYTORCH_TRAINING.md`
