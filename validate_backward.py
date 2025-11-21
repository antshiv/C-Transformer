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
    # Use eval() to match C (no dropout), but gradients still flow.
    model.eval()
    model.to("cpu")
    return tokenizer, model


def run_c_debug_backward(
    executable: str,
    weights: str,
    token_ids: List[int],
    layer_idx: int,
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
    if layer_idx >= 0:
        cmd.extend(["--debug-layer", str(layer_idx)])
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
    grad_embed_pat = re.compile(
        r"GRAD\s+embed_weight\s+idx=(\d+)\s+value=([\-0-9.eE]+)"
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

        m_ge = grad_embed_pat.search(line)
        if m_ge:
            idx = int(m_ge.group(1))
            val = float(m_ge.group(2))
            grads.setdefault("embed_weight", {})[idx] = val
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
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer block index to validate grads for (default: last layer)",
    )
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

    # C side: backward debug for chosen layer
    loss_c, grads_c = run_c_debug_backward(
        args.executable, args.weights, token_ids, args.layer
    )

    # HF side: forward+loss+backward with same loss definition
    import torch
    import numpy as np

    import torch
    model.zero_grad(set_to_none=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    logits = model(input_ids).logits  # [1, T, V]
    T = logits.size(1)

    targets = torch.tensor([token_ids], dtype=torch.long)  # [1, T]
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, T, V]
    loss_terms = -log_probs[0, torch.arange(T), targets[0]]  # [T]
    loss_hf = loss_terms.mean()
    loss_hf.backward()

    print(f"\nHF loss (eval): {loss_hf.item():.9g}  |  C loss: {loss_c:.9g}")

    # Optional: forward logits check on last token (top-10)
    try:
        import numpy as np

        from math import isfinite

        last_t = T - 1
        c_logits = None
        # Run C debug-logits for the same prompt to compare logits
        tokens_str = ",".join(str(t) for t in token_ids)
        cmd_logits = [
            args.executable,
            "--weights",
            args.weights,
            "--prompt",
            tokens_str,
            "--force",
            "--debug-logits",
            "--debug-top-k",
            "10",
        ]
        res_log = subprocess.run(cmd_logits, capture_output=True, text=True, timeout=300)
        if res_log.returncode == 0:
            logit_pat = re.compile(r"LOGIT idx=(\d+)\s+value=([\-0-9.eE]+)")
            vals = {}
            for line in res_log.stdout.splitlines():
                m = logit_pat.search(line)
                if m:
                    vid = int(m.group(1))
                    val = float(m.group(2))
                    if isfinite(val):
                        vals[vid] = val
            if vals:
                c_vocab = max(vals.keys()) + 1
                c_logits_vec = np.full((c_vocab,), np.nan, dtype="float32")
                for vid, val in vals.items():
                    if vid < c_vocab:
                        c_logits_vec[vid] = val
                hf_logits_vec = logits[0, last_t].detach().cpu().numpy().astype("float32")
                dim = min(hf_logits_vec.size, c_logits_vec.size)
                diff = np.abs(hf_logits_vec[:dim] - c_logits_vec[:dim])
                print(f"\n📊 Logit comparison (last token, {dim} dims):")
                print(f"  Max abs diff: {float(diff.max()):.6e}")
                print(f"  Mean abs diff:{float(diff.mean()):.6e}")
        else:
            print("⚠️  C debug-logits run failed; skipping logits comparison.")
    except Exception as e:
        print(f"⚠️  Skipping logits comparison due to error: {e}")

    # Collect HF grads for the same slices C printed
    def flatten_grad(param):
        if param.grad is None:
            return None
        return param.grad.detach().cpu().numpy().astype("float32").reshape(-1)

    # final_ln_gamma / beta (ln_f)
    ln_f_gamma = flatten_grad(model.transformer.ln_f.weight)
    ln_f_beta = flatten_grad(model.transformer.ln_f.bias)

    # Transformer block selected for per-layer comparisons
    layer_idx = args.layer
    if layer_idx < 0 or layer_idx >= len(model.transformer.h):
        layer_idx = len(model.transformer.h) - 1
    block = model.transformer.h[layer_idx]
    ln1_gamma = flatten_grad(block.ln_1.weight)
    ln1_beta = flatten_grad(block.ln_1.bias)
    ln2_gamma = flatten_grad(block.ln_2.weight)
    ln2_beta = flatten_grad(block.ln_2.bias)

    proj_w = flatten_grad(block.attn.c_proj.weight)
    fc1_w = flatten_grad(block.mlp.c_fc.weight)
    fc2_w = flatten_grad(block.mlp.c_proj.weight)

    # Embedding / LM head grads (weight tying)
    wte = model.transformer.wte
    embed_grad = flatten_grad(wte.weight)

    key_layer = f"L{layer_idx}"
    hf_map = {
        "final_ln_gamma": ln_f_gamma,
        "final_ln_beta": ln_f_beta,
        f"ln1_gamma_{key_layer}": ln1_gamma,
        f"ln1_beta_{key_layer}": ln1_beta,
        f"ln2_gamma_{key_layer}": ln2_gamma,
        f"ln2_beta_{key_layer}": ln2_beta,
        f"proj_weight_{key_layer}": proj_w,
        f"fc1_weight_{key_layer}": fc1_w,
        f"fc2_weight_{key_layer}": fc2_w,
        "embed_weight": embed_grad,
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
    # Compare the slices we printed in C
    compare_grad("final_ln_gamma", "final_ln_gamma")
    compare_grad("final_ln_beta", "final_ln_beta")
    compare_grad(f"ln1_gamma_{key_layer}", f"ln1_gamma_{key_layer}")
    compare_grad(f"ln1_beta_{key_layer}", f"ln1_beta_{key_layer}")
    compare_grad(f"ln2_gamma_{key_layer}", f"ln2_gamma_{key_layer}")
    compare_grad(f"ln2_beta_{key_layer}", f"ln2_beta_{key_layer}")
    compare_grad(f"proj_weight_{key_layer}", f"proj_weight_{key_layer}")
    compare_grad(f"fc1_weight_{key_layer}", f"fc1_weight_{key_layer}")
    compare_grad(f"fc2_weight_{key_layer}", f"fc2_weight_{key_layer}")
    compare_grad("embed_weight", "embed_weight")


if __name__ == "__main__":
    main()
