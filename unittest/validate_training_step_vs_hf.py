#!/usr/bin/env python3
"""
Validate a full training-like forward+backward step on a real window
between HuggingFace GPT-2 and the C-Transformer implementation.

This is similar to validate_backward.py, but:
  - It takes a binary training pair (e.g. from data/sql_training_pairs),
  - Uses the raw token IDs as both input and target (same-token LM),
  - Runs one HF forward+loss+backward on that exact sequence,
  - Runs the C binary in --debug-backward mode with the same tokens,
  - Compares loss and key gradients (final LN, last block LN1/LN2, proj/FC/embeds).

Usage (example):
  python3 unittest/validate_training_step_vs_hf.py \\
    --pair-file data/sql_training_pairs/pair_00000.bin \\
    --weights gpt2_bump.weights \\
    --executable ./main \\
    --model-name gpt2

Notes:
  - For now we assume the training pair format is:
        [uint16 ctx_len][uint16 tgt_len][uint32 tokens...]
    which matches the newer format used by sql_training_pairs.
    We treat the first ctx_len tokens as both input and target
    to match debug_backward_dump_grads_lm's LM objective (predict same token).
"""

import argparse
import os
import re
import struct
import subprocess
import sys
from typing import Dict, List, Tuple

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


def read_pair_file(path: str) -> List[int]:
    """
    Read a training pair .bin file and return the first ctx_len tokens
    as a list of Python ints.

    Format assumed:
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
    if tokens.size < ctx_len:
        raise ValueError(
            f"Not enough tokens in {path}: ctx_len={ctx_len}, got {tokens.size}"
        )
    return tokens[:ctx_len].astype(np.int32).tolist()


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
    # GRAD ln1_gamma layer=11 idx=0 value=...
    # GRAD embed_weight idx=0 value=...
    loss_pat = re.compile(r"DEBUG_BACKWARD\s+loss=([\-0-9.eE]+)")
    grad_simple_pat = re.compile(
        r"GRAD\s+(final_ln_gamma|final_ln_beta|final_ln_input)\s+idx=(\d+)\s+value=([\-0-9.eE]+)"
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
    parser = argparse.ArgumentParser(
        description="Validate a training-like step vs HF on a binary pair file"
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
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer block index to validate (default: last layer)",
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

    # 1) Read tokens from pair file
    token_ids = read_pair_file(args.pair_file)
    print(f"📝 Pair file: {args.pair_file}")
    print(f"   ctx_len: {len(token_ids)}")

    # 2) HF forward+backward on same-token LM objective
    model = load_model(args.model_name)
    model.zero_grad(set_to_none=True)

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    targets = torch.tensor([token_ids], dtype=torch.long)
    T = input_ids.size(1)

    # Hook to capture final LN grads and last block LN1/LN2/proj/FC/emebds
    hook_vals = {}

    def ln_f_bwd_hook(module, grad_input, grad_output):
        hook_vals["ln_f_grad_input"] = grad_input[0].detach().cpu()

    handle_ln_f = model.transformer.ln_f.register_full_backward_hook(ln_f_bwd_hook)

    from math import isfinite

    logits = model(input_ids).logits  # [1, T, V]
    log_probs = torch.log_softmax(logits, dim=-1)
    loss_terms = -log_probs[0, torch.arange(T), targets[0]]
    loss_hf = loss_terms.mean()
    loss_hf.backward()
    handle_ln_f.remove()

    def flatten_grad(param):
        if param.grad is None:
            return None
        return (
            param.grad.detach()
            .cpu()
            .numpy()
            .astype("float32")
            .reshape(-1)
        )

    def flatten_tensor(t):
        if t is None:
            return None
        return t.detach().cpu().numpy().astype("float32").reshape(-1)

    ln_f_gamma = flatten_grad(model.transformer.ln_f.weight)
    ln_f_beta = flatten_grad(model.transformer.ln_f.bias)
    ln_f_input_grad = flatten_tensor(hook_vals.get("ln_f_grad_input"))

    # Select layer index
    layer_idx = args.layer
    if layer_idx < 0 or layer_idx >= len(model.transformer.h):
        layer_idx = len(model.transformer.h) - 1
    key_layer = f"L{layer_idx}"
    block = model.transformer.h[layer_idx]

    ln1_gamma = flatten_grad(block.ln_1.weight)
    ln1_beta = flatten_grad(block.ln_1.bias)
    ln2_gamma = flatten_grad(block.ln_2.weight)
    ln2_beta = flatten_grad(block.ln_2.bias)
    proj_w = flatten_grad(block.attn.c_proj.weight)
    fc1_w = flatten_grad(block.mlp.c_fc.weight)
    fc2_w = flatten_grad(block.mlp.c_proj.weight)
    embed_grad = flatten_grad(model.transformer.wte.weight)

    hf_map = {
        "final_ln_gamma": ln_f_gamma,
        "final_ln_beta": ln_f_beta,
        "final_ln_input": ln_f_input_grad,
        f"ln1_gamma_{key_layer}": ln1_gamma,
        f"ln1_beta_{key_layer}": ln1_beta,
        f"ln2_gamma_{key_layer}": ln2_gamma,
        f"ln2_beta_{key_layer}": ln2_beta,
        f"proj_weight_{key_layer}": proj_w,
        f"fc1_weight_{key_layer}": fc1_w,
        f"fc2_weight_{key_layer}": fc2_w,
        "embed_weight": embed_grad,
    }

    # 3) C side: run debug_backward on same tokens
    loss_c, grads_c = run_c_debug_backward(
        args.executable, args.weights, token_ids, layer_idx
    )
    print(f"\nHF loss (same-token LM): {loss_hf.item():.9g}  |  C loss (debug): {loss_c:.9g}")

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
        for i in range(min(len(idxs), 10)):
            print(
                f"  idx={idxs[i]:5d}  hf={hf_vals[i]: .6e}  c={c_vals[i]: .6e}  diff={diff[i]: .6e}"
            )

    compare_grad("final_ln_gamma", "final_ln_gamma")
    compare_grad("final_ln_beta", "final_ln_beta")
    compare_grad("final_ln_input", "final_ln_input")
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

