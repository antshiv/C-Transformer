#!/usr/bin/env python3
"""
Validate C-Transformer embeddings (token + position) vs HuggingFace GPT-2.

This uses the C binary's --debug-embed mode, which dumps the embedded
input vectors after token + position addition, and compares them to:

  wte[token_id] + wpe[position]

from a GPT2LMHeadModel checkpoint.

Usage:
  python3 validate_embeddings.py "Once upon a time"

Options:
  --weights FILE      Path to gpt2_bump.weights (default: gpt2_bump.weights)
  --executable FILE   C binary (default: ./main)
  --model-name NAME   HF model name (default: gpt2)
"""

import argparse
import os
import re
import subprocess
import sys


def load_tokenizer_and_model(model_name: str):
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        import torch  # noqa: F401
    except Exception as e:
        print(f"❌ Failed to import transformers/torch: {e}")
        print("💡 Try: pip3 install transformers torch --break-system-packages")
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


def run_c_debug_embed(executable: str, weights: str, token_ids):
    tokens_str = ",".join(str(t) for t in token_ids)
    cmd = [
        executable,
        "--weights",
        weights,
        "--prompt",
        tokens_str,
        "--force",
        "--debug-embed",
    ]
    print("▶ Running C binary for debug embed:")
    print("  ", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping embeddings.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    # Parse lines of the form: EMBED t=3 idx=10 value=0.12345
    pattern = re.compile(r"EMBED t=(\d+) idx=(\d+) value=([\-0-9.eE]+)")
    by_token = {}
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        t = int(m.group(1))
        idx = int(m.group(2))
        val = float(m.group(3))
        by_token.setdefault(t, {})[idx] = val

    if not by_token:
        print("⚠️  No EMBED lines found in C output; check that --debug-embed is wired correctly.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)

    max_token = max(by_token.keys())
    max_dim = max(max(d.keys()) for d in by_token.values())
    T = max_token + 1
    D = max_dim + 1

    import numpy as np

    emb = np.zeros((T, D), dtype="float32")
    for t, dim_map in by_token.items():
        for idx, val in dim_map.items():
            emb[t, idx] = val

    return emb


def main():
    parser = argparse.ArgumentParser(description="Validate C-Transformer embeddings vs HuggingFace GPT-2")
    parser.add_argument("text", help="Prompt text to validate")
    parser.add_argument("--weights", default="gpt2_bump.weights", help="Path to C weight file")
    parser.add_argument("--executable", default="./main", help="Path to C binary")
    parser.add_argument("--model-name", default="gpt2", help="HuggingFace model name (default: gpt2)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"❌ Weights file not found: {args.weights}")
        sys.exit(1)
    if not os.path.exists(args.executable):
        print(f"⚠️  Executable not found: {args.executable}")
        print("   Trying to compile ./main from main.c ...")
        compile_cmd = ["gcc", "-O3", "-march=native", "-mavx512f", "-fopenmp", "main.c", "-o", "main", "-lm"]
        print("   ", " ".join(compile_cmd))
        res = subprocess.run(compile_cmd)
        if res.returncode != 0:
            print("❌ Failed to compile main.c; please compile manually.")
            sys.exit(1)
        if args.executable == "./main" and os.path.exists("main"):
            print("✅ Compiled ./main")

    tokenizer, model = load_tokenizer_and_model(args.model_name)

    # Encode prompt
    input_ids = tokenizer.encode(args.text, return_tensors="pt")
    token_ids = input_ids[0].tolist()
    print(f"📝 Prompt: {args.text}")
    print(f"   Tokens: {token_ids}")

    # Run C binary and get embedded inputs
    c_emb = run_c_debug_embed(args.executable, args.weights, token_ids)

    # HF embeddings
    import torch
    import numpy as np

    with torch.no_grad():
        # Get the underlying weights (handle transformer.* prefix)
        wte = model.transformer.wte.weight  # [vocab, d_model]
        wpe = model.transformer.wpe.weight  # [positions, d_model]

        T = len(token_ids)
        d_model = wte.shape[1]
        hf_emb = torch.zeros((T, d_model), dtype=torch.float32)
        for t, tok_id in enumerate(token_ids):
            hf_emb[t] = wte[tok_id] + wpe[t]

    # Compare C vs HF embeddings
    T_c, D_c = c_emb.shape
    if T_c < T or D_c < d_model:
        print(f"⚠️  C embedding shape smaller than expected: {c_emb.shape}, expected at least ({T}, {d_model})")
        T = min(T, T_c)
        d_model = min(d_model, D_c)

    hf_np = hf_emb[:T, :d_model].numpy()
    c_np = c_emb[:T, :d_model]

    diff = np.abs(hf_np - c_np)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print("\n📊 Embedding comparison (token + position):")
    print(f"   Shape compared: {hf_np.shape}")
    print(f"   Max abs diff:   {max_diff:.6f}")
    print(f"   Mean abs diff:  {mean_diff:.6f}")

    try:
        import pandas as pd

        rows = []
        for t in range(min(T, 3)):  # first few tokens
            for d in range(min(d_model, 8)):  # first few dims
                rows.append(
                    {
                        "token_idx": t,
                        "dim": d,
                        "hf": float(hf_np[t, d]),
                        "c": float(c_np[t, d]),
                        "diff": float(diff[t, d]),
                    }
                )
        df = pd.DataFrame(rows)
        print("\n🔎 Sample embedding diff (first 3 tokens, first 8 dims):")
        print(df.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()

