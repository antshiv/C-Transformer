#!/usr/bin/env python3
"""
Validate C-Transformer layer outputs vs HuggingFace GPT-2.

This uses the C binary's --debug-layer L mode, which dumps the hidden
state after transformer layer L (residual2_output_offset) for the last
token, and compares it to:

  outputs.hidden_states[L+1][0, -1, :]

from a GPT2LMHeadModel checkpoint (hidden_states[0] = embeddings).

Usage:
  python3 validate_layers.py "Hello World" --layer 0

Options:
  --weights FILE      Path to gpt2_bump.weights (default: gpt2_bump.weights)
  --executable FILE   C binary (default: ./main)
  --model-name NAME   HF model name (default: gpt2)
  --layer L           Layer index to validate (0-based)
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
    # Ensure we get hidden_states for all layers
    model.config.output_hidden_states = True
    model.eval()
    model.to("cpu")
    return tokenizer, model


def run_c_debug_layer(executable: str, weights: str, token_ids, layer_idx: int):
    tokens_str = ",".join(str(t) for t in token_ids)
    cmd = [
        executable,
        "--weights",
        weights,
        "--prompt",
        tokens_str,
        "--force",
        "--debug-layer",
        str(layer_idx),
    ]
    print("▶ Running C binary for debug layer output:")
    print("  ", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping layer output.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    # Parse lines of the form: LAYER_HIDDEN layer=0 idx=123 value=0.123456
    pattern = re.compile(r"LAYER_HIDDEN layer=(\d+) idx=(\d+) value=([\-0-9.eE]+)")
    values = {}
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        layer = int(m.group(1))
        if layer != layer_idx:
            continue
        idx = int(m.group(2))
        val = float(m.group(3))
        values[idx] = val

    if not values:
        print("⚠️  No LAYER_HIDDEN lines found in C output; check that --debug-layer is wired correctly.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)

    max_idx = max(values.keys())
    vec = [0.0] * (max_idx + 1)
    for idx, val in values.items():
        vec[idx] = val
    return vec


def main():
    parser = argparse.ArgumentParser(description="Validate C-Transformer layer outputs vs HuggingFace GPT-2")
    parser.add_argument("text", help="Prompt text to validate")
    parser.add_argument("--weights", default="gpt2_bump.weights", help="Path to C weight file")
    parser.add_argument("--executable", default="./main", help="Path to C binary")
    parser.add_argument("--model-name", default="gpt2", help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to validate (0-based)")
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

    # Run C binary and get layer output
    c_hidden = run_c_debug_layer(args.executable, args.weights, token_ids, args.layer)

    # Run HF model
    import torch
    import numpy as np

    with torch.no_grad():
        outputs = model(input_ids)
        if outputs.hidden_states is None:
            print("❌ Model did not return hidden_states; ensure output_hidden_states=True.")
            sys.exit(1)

        # hidden_states[0] = embeddings, [1] = after layer 0, ...
        if args.layer + 1 >= len(outputs.hidden_states):
            print(f"❌ Requested layer {args.layer}, but model returned only {len(outputs.hidden_states)} hidden_states.")
            sys.exit(1)

        hf_hidden = outputs.hidden_states[args.layer + 1][0, -1, :]  # last token after layer L

    dim_hf = hf_hidden.numel()
    dim_c = len(c_hidden)
    dim = min(dim_hf, dim_c)

    hf_np = hf_hidden[:dim].cpu().numpy()
    c_np = np.array(c_hidden[:dim], dtype="float32")

    diff = np.abs(hf_np - c_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"\n📊 Layer {args.layer} output comparison (HF vs C) for last token:")
    print(f"   Dimension compared: {dim}")
    print(f"   Max abs diff:       {max_diff:.6f}")
    print(f"   Mean abs diff:      {mean_diff:.6f}")

    try:
        import pandas as pd

        rows = []
        for i in range(min(dim, 10)):
            rows.append(
                {
                    "idx": i,
                    "hf_hidden": float(hf_np[i]),
                    "c_hidden": float(c_np[i]),
                    "diff": float(diff[i]),
                }
            )
        df = pd.DataFrame(rows)
        print("\n🔎 Sample diff (first 10 dims):")
        print(df.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()

