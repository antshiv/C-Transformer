#!/usr/bin/env python3
"""
Validate a slice of backprop gradients between C-Transformer and HuggingFace.

This script:
  1) Runs a tiny LM loss in C with --debug-backward and captures printed grads.
  2) Runs the same loss in HF (manual cross-entropy, no shift) and backprop.
  3) Compares selected gradients (final LN gamma/beta, layer-0 proj/MLP/LN).

Usage:
  python3 validate_backward.py "Hello World" \
    --weights gpt2_bump.weights \
    --executable ./main \
    --model-name gpt2
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple


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
    model.train()
    model.to("cpu")
    return tokenizer, model


def run_c_debug_backward(
    executable: str,
    weights: str,
    token_ids: List[int],
) -> Tuple[float, Dict[str, Dict[int, float]]]:
    """
    Run the C binary with --debug-backward and parse gradients.

    Returns:
      loss_c, grads_c where grads_c maps gradient name -> {idx -> value}
    """
    tokens_str = ",".join(str(t) for t in token_ids)
    cmd = [
        executable,
        "--weights",
        weights,
        "--prompt",
        tokens_str,
        "--force",
        "--debug-backward",
    ]
    print("▶ Running C binary for backward debug:")
    print("  ", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping backward grads.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    loss_c = None
    grads: Dict[str, Dict[int, float]] = {}

    # Example lines:
    # DEBUG_BACKWARD loss=2.345
    # GRAD final_ln_gamma idx=0 value=...
    # GRAD final_ln_beta idx=0 value=...
    # GRAD ln1_gamma layer=0 idx=0 value=...
    # GRAD proj_weight layer=0 idx=0 value=...
    loss_pat = re.compile(r"DEBUG_BACKWARD\s+loss=([\-0-9.eE]+)")
    grad_simple_pat = re.compile(
        r"GRAD\s+(final_ln_gamma|final_ln_beta)\s+idx=(\d+)\s+value=([\-0-9.eE]+)"
    )
    grad_layer_pat = re.compile(
        r"GRAD\s+(ln1_gamma|ln1_beta|ln2_gamma|ln2_beta|proj_weight|fc1_weight|fc2_weight)\s+layer=(\d+)\s+idx=(\d+)\s+value=([\-0-9.eE]+)"
    )

    for line in result.stdout.splitlines():
        m_loss = loss_pat.search(line)
        if m_loss:
            loss_c = float(m_loss.group(1))
            continue

        m_gs = grad_simple_pat.search(line)
        if m_gs:
            name = m_gs.group(1)
            idx = int(m_gs.group(2))
            val = float(m_gs.group(3))
            grads.setdefault(name, {})[idx] = val
            continue

        m_gl = grad_layer_pat.search(line)
        if m_gl:
            base = m_gl.group(1)
            layer = int(m_gl.group(2))
            idx = int(m_gl.group(3))
            val = float(m_gl.group(4))
            key = f"{base}_L{layer}"
            grads.setdefault(key, {})[idx] = val
            continue

    if loss_c is None:
        print("❌ Failed to parse DEBUG_BACKWARD loss from C output.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)

    return loss_c, grads


def main():
    parser = argparse.ArgumentParser(description="Validate backward gradients vs HuggingFace GPT-2")
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

    # C side
    loss_c, grads_c = run_c_debug_backward(args.executable, args.weights, token_ids)

    # HF side: run same LM loss (no shift) and backprop
    import torch
    import numpy as np

    model.zero_grad(set_to_none=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    logits = model(input_ids).logits  # [1, T, V]
    T = logits.size(1)

    targets = torch.tensor([token_ids], dtype=torch.long)  # [1, T]
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, T, V]
    loss_terms = -log_probs[0, torch.arange(T), targets[0]]  # [T]
    loss_hf = loss_terms.mean()
    loss_hf.backward()

    print(f"\nHF loss: {loss_hf.item():.9g}  |  C loss: {loss_c:.9g}")

    # Collect HF grads for the same slices C printed
    def flatten_grad(param):
        if param.grad is None:
            return None
        return param.grad.detach().cpu().numpy().astype("float32").reshape(-1)

    # final_ln_gamma / beta (ln_f)
    ln_f_gamma = flatten_grad(model.transformer.ln_f.weight)
    ln_f_beta = flatten_grad(model.transformer.ln_f.bias)

    block0 = model.transformer.h[0]
    ln1_gamma = flatten_grad(block0.ln_1.weight)
    ln1_beta = flatten_grad(block0.ln_1.bias)
    ln2_gamma = flatten_grad(block0.ln_2.weight)
    ln2_beta = flatten_grad(block0.ln_2.bias)

    proj_w = flatten_grad(block0.attn.c_proj.weight)
    fc1_w = flatten_grad(block0.mlp.c_fc.weight)
    fc2_w = flatten_grad(block0.mlp.c_proj.weight)

    hf_map = {
        "final_ln_gamma": ln_f_gamma,
        "final_ln_beta": ln_f_beta,
        "ln1_gamma_L0": ln1_gamma,
        "ln1_beta_L0": ln1_beta,
        "ln2_gamma_L0": ln2_gamma,
        "ln2_beta_L0": ln2_beta,
        "proj_weight_L0": proj_w,
        "fc1_weight_L0": fc1_w,
        "fc2_weight_L0": fc2_w,
    }

    def compare_grad(name_c: str, name_hf: str):
        if name_c not in grads_c:
            print(f"\n⚠️  No C grad data for '{name_c}', skipping.")
            return
        hf_vec = hf_map.get(name_hf)
        if hf_vec is None:
            print(f"\n⚠️  No HF grad tensor for '{name_hf}', skipping.")
            return

        idx_to_val = grads_c[name_c]
        idxs = sorted(idx_to_val.keys())
        c_vals = np.array([idx_to_val[i] for i in idxs], dtype="float32")
        hf_vals = np.array([hf_vec[i] for i in idxs], dtype="float32")
        diff = np.abs(hf_vals - c_vals)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        print(f"\n📊 Gradient comparison: {name_c} vs {name_hf}")
        print(f"  Indices compared: {len(idxs)}")
        print(f"  Max abs diff:      {max_diff:.6e}")
        print(f"  Mean abs diff:     {mean_diff:.6e}")
        try:
            import pandas as pd

            rows = []
            for i in range(min(len(idxs), 10)):
                rows.append(
                    {
                        "idx": idxs[i],
                        "hf": float(hf_vals[i]),
                        "c": float(c_vals[i]),
                        "diff": float(diff[i]),
                    }
                )
            df = pd.DataFrame(rows)
            print("  Sample (first 10 indices):")
            print(df.to_string(index=False))
        except Exception:
            for i in range(min(len(idxs), 10)):
                print(
                    f"  idx={idxs[i]:5d}  hf={hf_vals[i]: .6e}  c={c_vals[i]: .6e}  diff={diff[i]: .6e}"
                )

    # Compare the slices we printed in C
    compare_grad("final_ln_gamma", "final_ln_gamma")
    compare_grad("final_ln_beta", "final_ln_beta")
    compare_grad("ln1_gamma_L0", "ln1_gamma_L0")
    compare_grad("ln1_beta_L0", "ln1_beta_L0")
    compare_grad("ln2_gamma_L0", "ln2_gamma_L0")
    compare_grad("ln2_beta_L0", "ln2_beta_L0")
    compare_grad("proj_weight_L0", "proj_weight_L0")
    compare_grad("fc1_weight_L0", "fc1_weight_L0")
    compare_grad("fc2_weight_L0", "fc2_weight_L0")


if __name__ == "__main__":
    main()

