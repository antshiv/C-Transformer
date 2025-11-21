#!/usr/bin/env python3
"""
Validate internal stages of a GPT-2 transformer layer (LN1, RES1, LN2, MLP, RES2)
between C-Transformer and HuggingFace GPT-2.

This uses the C binary's `--debug-layer L` mode, which now prints:
  - LAYER_LN1   layer=L idx=i value=...
  - LAYER_RES1  layer=L idx=i value=...
  - LAYER_LN2   layer=L idx=i value=...
  - LAYER_MLP   layer=L idx=i value=...
  - LAYER_HIDDEN layer=L idx=i value=...   (RES2, final layer output)

and compares those per-stage vectors for the last token against a manual
forward through the corresponding GPT2Block in HuggingFace.

Usage:
  python3 validate_layer_stages.py "Hello World" --layer 0
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
    print("▶ Running C binary for debug layer stages:")
    print("  ", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping layer stages.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    # Parse lines:
    #  LAYER_LN1 layer=L idx=i value=...
    #  LAYER_RES1 ...
    #  LAYER_LN2 ...
    #  LAYER_MLP ...
    #  LAYER_HIDDEN ...
    patterns = {
        "ln1": re.compile(r"LAYER_LN1 layer=(\d+) idx=(\d+) value=([\-0-9.eE]+)"),
        "res1": re.compile(r"LAYER_RES1 layer=(\d+) idx=(\d+) value=([\-0-9.eE]+)"),
        "ln2": re.compile(r"LAYER_LN2 layer=(\d+) idx=(\d+) value=([\-0-9.eE]+)"),
        "mlp": re.compile(r"LAYER_MLP layer=(\d+) idx=(\d+) value=([\-0-9.eE]+)"),
        "res2": re.compile(r"LAYER_HIDDEN layer=(\d+) idx=(\d+) value=([\-0-9.eE]+)"),
    }

    stage_vals = {k: {} for k in patterns}

    for line in result.stdout.splitlines():
        for stage, pat in patterns.items():
            m = pat.search(line)
            if not m:
                continue
            layer = int(m.group(1))
            if layer != layer_idx:
                continue
            idx = int(m.group(2))
            val = float(m.group(3))
            stage_vals[stage][idx] = val

    for stage, vals in stage_vals.items():
        if not vals:
            print(f"⚠️  No values found for stage '{stage}' in C output; check debug-layer printing.")

    # Convert to dense vectors (assume contiguous indices from 0..D-1)
    dense = {}
    for stage, vals in stage_vals.items():
        if not vals:
            continue
        max_idx = max(vals.keys())
        vec = [0.0] * (max_idx + 1)
        for idx, val in vals.items():
            vec[idx] = val
        dense[stage] = vec

    return dense, result.stdout


def main():
    parser = argparse.ArgumentParser(description="Validate internal layer stages vs HuggingFace GPT-2")
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

    # C side: get stage outputs
    c_stages, _ = run_c_debug_layer(args.executable, args.weights, token_ids, args.layer)

    # HF side: manual block forward
    import torch
    import numpy as np

    with torch.no_grad():
        outputs = model(input_ids)
        if outputs.hidden_states is None:
            print("❌ Model did not return hidden_states; ensure output_hidden_states=True.")
            sys.exit(1)

        # hidden_states[0] = embeddings, [1] = after layer 0, ...
        hs = outputs.hidden_states
        if args.layer >= len(hs) - 1:
            print(f"❌ Requested layer {args.layer}, but model returned only {len(hs)} hidden_states.")
            sys.exit(1)

        # Input to this block:
        h_in = hs[args.layer]    # [1, T, D]
        block = model.transformer.h[args.layer]

        ln1 = block.ln_1(h_in)
        attn_out, _ = block.attn(ln1, layer_past=None, use_cache=False, output_attentions=False)
        res1 = h_in + attn_out
        ln2 = block.ln_2(res1)
        mlp_out = block.mlp(ln2)
        res2 = res1 + mlp_out

        # Take last token
        ln1_vec = ln1[0, -1, :].cpu().numpy()
        res1_vec = res1[0, -1, :].cpu().numpy()
        ln2_vec = ln2[0, -1, :].cpu().numpy()
        mlp_vec = mlp_out[0, -1, :].cpu().numpy()
        res2_vec = res2[0, -1, :].cpu().numpy()

    hf_stage_vecs = {
        "ln1": ln1_vec,
        "res1": res1_vec,
        "ln2": ln2_vec,
        "mlp": mlp_vec,
        "res2": res2_vec,
    }

    print("\n📊 Per-stage comparison for layer", args.layer)

    def compare_stage(name_hf, name_c):
        if name_c not in c_stages:
            print(f"\n⚠️  No C data for stage '{name_c}', skipping.")
            return
        hf_vec = hf_stage_vecs[name_hf]
        c_vec = np.array(c_stages[name_c], dtype="float32")
        dim = min(hf_vec.size, c_vec.size)
        hf_slice = hf_vec[:dim]
        c_slice = c_vec[:dim]
        diff = np.abs(hf_slice - c_slice)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        print(f"\nStage {name_hf.upper()} vs C ({name_c}):")
        print(f"  Dimension:    {dim}")
        print(f"  Max abs diff: {max_diff:.6f}")
        print(f"  Mean abs diff:{mean_diff:.6f}")
        try:
            import pandas as pd

            rows = []
            for i in range(min(dim, 10)):
                rows.append(
                    {
                        "idx": i,
                        "hf": float(hf_slice[i]),
                        "c": float(c_slice[i]),
                        "diff": float(diff[i]),
                    }
                )
            df = pd.DataFrame(rows)
            print("  Sample (first 10 dims):")
            print(df.to_string(index=False))
        except Exception:
            pass

    compare_stage("ln1", "ln1")
    compare_stage("res1", "res1")
    compare_stage("ln2", "ln2")
    compare_stage("mlp", "mlp")
    compare_stage("res2", "res2")


if __name__ == "__main__":
    main()

