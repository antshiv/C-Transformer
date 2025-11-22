#!/usr/bin/env python3
"""
Validate a single next-token LM training_step between HuggingFace GPT-2
and the C-Transformer implementation using the new --debug-train-step
mode in main.c.

This script:
  - Reads a binary training window (pair_*.bin),
  - Uses the first ctx_len+1 tokens as a prompt for C,
  - Runs:
        ./main --weights ... --prompt "<tokens>" --debug-train-step
    which internally calls training_step() with:
        input_tokens[i]  = prompt[i]
        target_tokens[i] = prompt[i+1]
        ctx_len          = prompt_len - 1
  - Runs HF GPT-2 with the same next-token LM objective:
        input_ids  = tokens[0:ctx_len]
        labels     = tokens[1:ctx_len+1]
  - Compares the C loss (DEBUG_TRAINSTEP) against HF loss.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
from typing import Tuple

import numpy as np
import torch


def load_model(model_name: str):
    try:
        from transformers import GPT2LMHeadModel
    except Exception as e:
        print(f"❌ Failed to import transformers: {e}")
        print("💡 Try: pip3 install transformers torch --break-system-packages")
        sys.exit(1)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return model


def read_pair(path: str) -> Tuple[int, np.ndarray]:
    """
    Read pair file and return (ctx_len, tokens array).
    Format:
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
    if tokens.size < ctx_len + 1:
        raise ValueError(
            f"Not enough tokens in {path}: need ctx_len+1={ctx_len+1}, got {tokens.size}"
        )
    return ctx_len, tokens


def run_c_debug_train_step(
    executable: str,
    weights: str,
    tokens: np.ndarray,
    ctx_len: int,
) -> float:
    """
    Run C main with --debug-train-step on the first ctx_len+1 tokens
    and parse DEBUG_TRAINSTEP loss.
    """
    prompt_tokens = tokens[: ctx_len + 1].astype(int).tolist()
    tokens_str = ",".join(str(t) for t in prompt_tokens)
    cmd = [
        executable,
        "--weights",
        weights,
        "--prompt",
        tokens_str,
        "--force",
        "--debug-train-step",
    ]
    print("▶ Running C binary (--debug-train-step):")
    print("  ", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out during debug-train-step.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    m = re.search(r"DEBUG_TRAINSTEP\s+loss=([\-0-9.eE]+)", result.stdout)
    if not m:
        print("❌ Failed to parse DEBUG_TRAINSTEP loss from C output.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="Validate a next-token LM training_step vs HF on a binary pair file"
    )
    parser.add_argument(
        "--pair-file",
        required=True,
        help="Path to a .bin pair file (e.g. data/sql_training_pairs/pair_00000.bin)",
    )
    parser.add_argument(
        "--weights",
        default="gpt2_bump.weights",
        help="Path to C weight file",
    )
    parser.add_argument(
        "--executable",
        default="./main",
        help="Path to C binary",
    )
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help="HuggingFace model name (default: gpt2)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pair_file):
        print(f"❌ Pair file not found: {args.pair_file}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"❌ Weights file not found: {args.weights}")
        sys.exit(1)
    if not os.path.exists(args.executable):
        print(f"⚠️  Executable not found: {args.executable}")
        print("   Trying to compile ./main from main.c ...")
        compile_cmd = [
            "gcc",
            "-O3",
            "-march=native",
            "-mavx512f",
            "-fopenmp",
            "main.c",
            "-o",
            "main",
            "-lm",
        ]
        print("   ", " ".join(compile_cmd))
        res = subprocess.run(compile_cmd)
        if res.returncode != 0:
            print("❌ Failed to compile main.c; please compile manually.")
            sys.exit(1)
        if args.executable == "./main" and os.path.exists("main"):
            print("✅ Compiled ./main")

    # 1) Read window
    ctx_len, tokens = read_pair(args.pair_file)
    print(f"📝 Pair file: {args.pair_file}")
    print(f"   ctx_len: {ctx_len}")

    # 2) HF next-token LM loss on same window
    model = load_model(args.model_name)
    model.zero_grad(set_to_none=True)

    # HF uses [1, T] tensors
    input_ids = torch.tensor(tokens[:ctx_len], dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(tokens[1 : ctx_len + 1], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss_hf = float(outputs.loss.item())

    print(f"HF loss (next-token LM): {loss_hf:.9f}")

    # 3) C training_step loss via --debug-train-step
    loss_c = run_c_debug_train_step(args.executable, args.weights, tokens, ctx_len)
    print(f"C  loss (DEBUG_TRAINSTEP): {loss_c:.9f}")

    diff = abs(loss_hf - loss_c)
    print(f"\nΔ loss = {diff:.9g}")


if __name__ == "__main__":
    main()

