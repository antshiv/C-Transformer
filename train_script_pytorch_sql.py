#!/usr/bin/env python3
"""
Minimal PyTorch training script for comparing GPT-2 training on your SQL
windows against your C-Transformer training.

This script:
  - Loads HuggingFace GPT-2 (same model family as gpt2_bump.weights).
  - Iterates over .bin windows in a training directory (SQL or TinyStories).
  - For each window, uses a next-token LM objective:
        input_ids[t] = tokens[t]
        labels[t]    = tokens[t+1]
    matching the way your C training loop uses input_tokens and target_tokens.
  - Runs standard Adam on CPU with a small LR for a configurable number of steps.
  - Prints step, loss, and sample file name so you can compare behaviour
    (e.g. whether loss collapses to ~0) to your C training logs.

For simplicity and to avoid padding issues, this script uses batch_size=1 by
default, so GPT-2 sees one variable-length sequence per step.
"""

import argparse
import glob
import os
import struct
from typing import List, Tuple

import numpy as np
import torch


def load_model_and_tokenizer(model_name: str):
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except Exception as e:
        print(f"❌ Failed to import transformers: {e}")
        print("💡 Try: pip3 install transformers torch --break-system-packages")
        raise

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.train()
    model.to("cpu")
    return model, tokenizer


def read_pair_header_and_tokens(path: str) -> Tuple[int, int, np.ndarray]:
    """
    Read a training pair .bin file and return:
      ctx_len, tgt_len, tokens (np.uint32 array of length ctx_len + tgt_len).

    Assumes the newer header format:
      uint16 ctx_len
      uint16 tgt_len
      uint32 tokens[ctx_len + tgt_len]
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 4:
        raise ValueError(f"File too small to contain header: {path}")
    ctx_len, tgt_len = struct.unpack("<HH", data[:4])
    tokens = np.frombuffer(data[4:], dtype="<u4")
    if tokens.size < ctx_len + tgt_len:
        raise ValueError(
            f"Not enough tokens in {path}: ctx_len={ctx_len}, tgt_len={tgt_len}, got {tokens.size}"
        )
    return ctx_len, tgt_len, tokens


def build_input_and_labels(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build input_ids and labels for a single window, matching your C training:

      input_tokens  = tokens[0:ctx_len]
      target_tokens = tokens[1:ctx_len+1]
      (labels are next-token targets; no special loss masking here)
    """
    ctx_len, tgt_len, tokens = read_pair_header_and_tokens(path)
    # tokens[0..ctx_len-1] are the context; tokens[1..ctx_len] are next-token labels
    if ctx_len + 1 > tokens.size:
        raise ValueError(f"Not enough tokens for next-token LM in {path}")
    input_ids = torch.tensor(tokens[:ctx_len], dtype=torch.long).unsqueeze(0)  # [1, T]
    labels = torch.tensor(tokens[1 : ctx_len + 1], dtype=torch.long).unsqueeze(0)  # [1, T]
    return input_ids, labels


def main():
    parser = argparse.ArgumentParser(description="PyTorch GPT-2 training on SQL windows for comparison with C-Transformer.")
    parser.add_argument(
        "--train-dir",
        default="data/sql_training_pairs",
        help="Directory with pair_*.bin windows (default: data/sql_training_pairs)",
    )
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help="HuggingFace GPT-2 model name (default: gpt2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of training steps to run (default: 500)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for Adam (default: 3e-5, similar to your C run)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.train_dir):
        print(f"❌ Training directory not found: {args.train_dir}")
        return

    pair_files = sorted(glob.glob(os.path.join(args.train_dir, "pair_*.bin")))
    if not pair_files:
        print(f"❌ No pair_*.bin files found in {args.train_dir}")
        return

    print("============================================")
    print("PyTorch GPT-2 Training on SQL Windows")
    print("============================================")
    print(f"Train dir:   {args.train_dir}")
    print(f"Model name:  {args.model_name}")
    print(f"Steps:       {args.steps}")
    print(f"LR:          {args.lr}")
    print()

    model, _ = load_model_and_tokenizer(args.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    file_idx = 0

    while step < args.steps:
        path = pair_files[file_idx]
        file_idx = (file_idx + 1) % len(pair_files)

        try:
            input_ids, labels = build_input_and_labels(path)
        except Exception as e:
            print(f"⚠️  Skipping {path}: {e}")
            continue

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss  # HF cross-entropy: average over tokens
        loss.backward()
        optimizer.step()

        step += 1

        if step % args.log_interval == 0 or step == 1:
            ppl = float(torch.exp(loss.detach()))
            print(
                f"[pt-train] step={step}/{args.steps}  loss={loss.item():.6f}  "
                f"perplexity={ppl:.2f}  sample={os.path.basename(path)}"
            )

    print("✅ PyTorch training loop complete.")


if __name__ == "__main__":
    main()

