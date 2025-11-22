#!/usr/bin/env python3
"""
Validate per-layer backward stages (LN2 and LN1 input/output grads)
between HuggingFace GPT-2 and the C-Transformer implementation.

This script:
  1) Runs HF GPT-2 on a prompt with a same-token LM loss.
  2) Hooks ln_1 and ln_2 in a chosen transformer block to capture:
       - dY (grad_output) into LN1/LN2
       - dX (grad_input) out of LN1/LN2
  3) Runs the C binary in --debug-backward mode and parses:
       - VAL BWD_ln2_out idx=...
       - VAL BWD_ln2_in  idx=...
       - VAL BWD_ln1_out idx=...
       - VAL BWD_ln1_in  idx=...
  4) Compares these small slices (first 16 dims) between HF and C.

Usage:
  python3 unittest/validate_backward_layer_stages.py "Hello World" \\
    --weights gpt2_bump.weights \\
    --executable ./main \\
    --model-name gpt2 \\
    --layer 11
"""

import argparse
import re
import subprocess
import sys
from typing import Dict, List

import numpy as np
import torch


def load_tokenizer_and_model(model_name: str):
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
    except Exception as e:
        print(f"❌ Failed to import transformers: {e}")
        print("💡 Try: pip3 install transformers torch --break-system-packages")
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


def run_c_debug_backward(
    executable: str,
    weights: str,
    token_ids: List[int],
    layer_idx: int,
):
    """
    Run the C binary with --debug-backward and parse LN1/LN2 stage grads.
    Returns:
      loss_c, stages_c[name][idx] -> value
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
        "--debug-layer",
        str(layer_idx),
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
        print("❌ C binary timed out while dumping backward layer stages.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    loss_c = None
    stages: Dict[str, Dict[int, float]] = {}

    loss_pat = re.compile(r"DEBUG_BACKWARD\s+loss=([\-0-9.eE]+)")
    val_pat = re.compile(
        r"VAL\s+(BWD_ln2_out|BWD_ln2_in|BWD_ln1_out|BWD_ln1_in)\s+idx=(\d+)\s+value=([\-0-9.eE]+)"
    )

    for line in result.stdout.splitlines():
        m_loss = loss_pat.search(line)
        if m_loss:
            loss_c = float(m_loss.group(1))
            continue
        m_val = val_pat.search(line)
        if m_val:
            name = m_val.group(1)
            idx = int(m_val.group(2))
            val = float(m_val.group(3))
            stages.setdefault(name, {})[idx] = val
            continue

    if loss_c is None:
        print("❌ Failed to parse DEBUG_BACKWARD loss from C output.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)

    return loss_c, stages


def main():
    parser = argparse.ArgumentParser(
        description="Validate LN1/LN2 backward stages vs HF for a given layer"
    )
    parser.add_argument("text", help="Prompt text")
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
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer block index to validate (default: last layer)",
    )
    args = parser.parse_args()

    if not args.weights or not os.path.exists(args.weights):
        print(f"❌ Weights file not found (C side): {args.weights}")
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

    tokenizer, model = load_tokenizer_and_model(args.model_name)

    # Encode prompt
    input_ids = tokenizer.encode(args.text, return_tensors="pt")
    token_ids = input_ids[0].tolist()
    print(f"📝 Prompt: {args.text}")
    print(f"   Tokens: {token_ids}")

    # Select layer index
    if args.layer < 0 or args.layer >= len(model.transformer.h):
        layer_idx = len(model.transformer.h) - 1
    else:
        layer_idx = args.layer
    block = model.transformer.h[layer_idx]

    # HF forward+backward with same-token LM loss, with LN1/LN2 hooks
    hook_vals = {}

    def ln1_bwd(module, grad_input, grad_output):
        hook_vals["ln1_grad_input"] = grad_input[0].detach().cpu()
        hook_vals["ln1_grad_output"] = grad_output[0].detach().cpu()

    def ln2_bwd(module, grad_input, grad_output):
        hook_vals["ln2_grad_input"] = grad_input[0].detach().cpu()
        hook_vals["ln2_grad_output"] = grad_output[0].detach().cpu()

    handle_ln1 = block.ln_1.register_full_backward_hook(ln1_bwd)
    handle_ln2 = block.ln_2.register_full_backward_hook(ln2_bwd)

    model.zero_grad(set_to_none=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    targets = torch.tensor([token_ids], dtype=torch.long)
    logits = model(input_ids).logits  # [1, T, V]

    T = logits.size(1)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss_terms = -log_probs[0, torch.arange(T), targets[0]]
    loss_hf = loss_terms.mean()
    loss_hf.backward()

    handle_ln1.remove()
    handle_ln2.remove()

    # Flatten HF LN1/LN2 grads for the last token
    def flatten_last_token(tensor):
        # tensor: [1, T, D]
        if tensor is None:
            return None
        arr = tensor.numpy().astype("float32")
        return arr[0, -1].reshape(-1)

    ln1_out_hf = flatten_last_token(hook_vals.get("ln1_grad_output"))
    ln1_in_hf = flatten_last_token(hook_vals.get("ln1_grad_input"))
    ln2_out_hf = flatten_last_token(hook_vals.get("ln2_grad_output"))
    ln2_in_hf = flatten_last_token(hook_vals.get("ln2_grad_input"))

    # C side: run debug_backward for same prompt/layer
    loss_c, stages_c = run_c_debug_backward(
        args.executable, args.weights, token_ids, layer_idx
    )
    print(f"\nHF loss (same-token LM): {loss_hf.item():.9g}  |  C loss (debug): {loss_c:.9g}")

    hf_map = {
        "BWD_ln2_out": ln2_out_hf,
        "BWD_ln2_in": ln2_in_hf,
        "BWD_ln1_out": ln1_out_hf,
        "BWD_ln1_in": ln1_in_hf,
    }

    def compare_stage(name: str):
        if name not in stages_c:
            print(f"\n⚠️  No C stage data for '{name}', skipping.")
            return
        hf_vec = hf_map.get(name)
        if hf_vec is None:
            print(f"\n⚠️  No HF tensor for '{name}', skipping.")
            return

        idx_to_val = stages_c[name]
        idxs = sorted(idx_to_val.keys())
        c_vals = np.array([idx_to_val[i] for i in idxs], dtype="float32")
        hf_vals = np.array([hf_vec[i] for i in idxs], dtype="float32")
        diff = np.abs(hf_vals - c_vals)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        print(f"\n📊 Stage comparison: {name}")
        print(f"  Indices compared: {len(idxs)}")
        print(f"  Max abs diff:      {max_diff:.6e}")
        print(f"  Mean abs diff:     {mean_diff:.6e}")
        for i in range(min(len(idxs), 10)):
            print(
                f"  idx={idxs[i]:3d}  hf={hf_vals[i]: .6e}  c={c_vals[i]: .6e}  diff={diff[i]: .6e}"
            )

    compare_stage("BWD_ln2_out")
    compare_stage("BWD_ln2_in")
    compare_stage("BWD_ln1_out")
    compare_stage("BWD_ln1_in")


if __name__ == "__main__":
    import os  # ensure os is imported for weights check
    main()

