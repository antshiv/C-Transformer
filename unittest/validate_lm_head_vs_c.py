#!/usr/bin/env python3
"""
Compare C-Transformer's LM head backward (loss -> d_logits -> d_final_ln_output)
against the Python "C-style" reference using HuggingFace GPT-2.

This script:
  1) Uses HF GPT-2 to get logits and final LN output for a prompt.
  2) Runs the same cross-entropy as compute_cross_entropy_loss and the same
     LM head backward as backward_lm_head in pure Python (see lm_head_backward_unit).
  3) Runs the C binary with --debug-backward and parses:
       - DLOGIT t=0 v=... value=...
       - VAL final_ln_dy idx=... value=...
  4) Compares:
       - C d_logits[t=0, v] vs Python c-style d_logits[t=0, v]
       - C final_ln_dy[d]   vs Python c-style d_final_ln_output[d]
"""

import argparse
import re
import subprocess
import sys
from typing import Dict, Tuple

import numpy as np
import torch

from lm_head_backward_unit import (
    load_tokenizer_and_model,
    c_style_cross_entropy,
    c_style_backward_lm_head,
)


def run_c_debug_backward(executable: str, weights: str, token_ids) -> Tuple[float, Dict[Tuple[int, int], float], Dict[int, float]]:
    """
    Run C binary with --debug-backward and parse:
      - DEBUG_BACKWARD loss=...
      - DLOGIT t=0 v=... value=...
      - VAL final_ln_dy idx=... value=...

    Returns:
      loss_c, dlogits_c[(t,v)] -> val, dfinal_c[d] -> val
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
    print("▶ Running C binary:", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    loss_c = None
    dlogits_map: Dict[Tuple[int, int], float] = {}
    dfinal_map: Dict[int, float] = {}

    loss_pat = re.compile(r"DEBUG_BACKWARD\s+loss=([\-0-9.eE]+)")
    dlogit_pat = re.compile(r"DLOGIT\s+t=(\d+)\s+v=(\d+)\s+value=([\-0-9.eE]+)")
    dfinal_pat = re.compile(r"VAL\s+final_ln_dy\s+idx=(\d+)\s+value=([\-0-9.eE]+)")

    for line in result.stdout.splitlines():
        m_loss = loss_pat.search(line)
        if m_loss:
            loss_c = float(m_loss.group(1))
            continue
        m_dlog = dlogit_pat.search(line)
        if m_dlog:
            t = int(m_dlog.group(1))
            v = int(m_dlog.group(2))
            val = float(m_dlog.group(3))
            dlogits_map[(t, v)] = val
            continue
        m_df = dfinal_pat.search(line)
        if m_df:
            idx = int(m_df.group(1))
            val = float(m_df.group(2))
            dfinal_map[idx] = val
            continue

    if loss_c is None:
        print("❌ Failed to parse DEBUG_BACKWARD loss from C output.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)

    return loss_c, dlogits_map, dfinal_map


def main():
    parser = argparse.ArgumentParser(description="Validate LM head backward between C and Python reference")
    parser.add_argument("text", help="Prompt text")
    parser.add_argument("--weights", default="gpt2_bump.weights", help="C weight file")
    parser.add_argument("--executable", default="./main", help="C binary")
    parser.add_argument("--model-name", default="gpt2", help="HF model name")
    args = parser.parse_args()

    tokenizer, model = load_tokenizer_and_model(args.model_name)

    # Encode prompt
    input_ids = tokenizer.encode(args.text, return_tensors="pt")
    token_ids = input_ids[0].tolist()
    print(f"📝 Prompt: {args.text}")
    print(f"   Tokens: {token_ids}")

    # HF forward to get logits and LN_f output
    model.eval()
    model.zero_grad(set_to_none=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    T = input_ids.size(1)

    hook_vals = {}

    def ln_f_fwd_hook(module, inputs, output):
        hook_vals["ln_f_output"] = output.detach().cpu()

    handle_fwd = model.transformer.ln_f.register_forward_hook(ln_f_fwd_hook)
    with torch.no_grad():
        out = model(input_ids)
    handle_fwd.remove()

    logits = out.logits.detach().cpu()              # [1, T, V]
    ln_f_output = hook_vals["ln_f_output"][0]       # [T, D]
    targets = input_ids.clone()

    # Python C-style loss + LM head backward
    loss_ref, d_logits_ref = c_style_cross_entropy(logits, targets)   # [T, V]
    lm_head_weights = model.transformer.wte.weight.detach().cpu()     # [V, D]
    d_final_ref, _ = c_style_backward_lm_head(d_logits_ref, ln_f_output, lm_head_weights)

    # C side debug
    loss_c, dlogits_c_map, dfinal_c_map = run_c_debug_backward(
        args.executable, args.weights, token_ids
    )
    print(f"\nHF loss (eval): {float(loss_ref):.9g}  |  C loss (debug): {loss_c:.9g}")

    # Compare d_logits for entries C printed (t=0, a few v)
    if not dlogits_c_map:
        print("⚠️  No DLOGIT lines parsed from C; did you rebuild main.c?")
    else:
        print("\n📊 d_logits comparison for entries printed by C (t=0):")
        rows = []
        for (t, v), c_val in sorted(dlogits_c_map.items()):
            ref_val = float(d_logits_ref[t, v].item())
            diff = abs(ref_val - c_val)
            rows.append((t, v, ref_val, c_val, diff))
        diffs = np.array([r[4] for r in rows], dtype="float32")
        print(f"  Entries compared: {len(rows)}")
        print(f"  Max abs diff:     {float(diffs.max()):.6e}")
        print(f"  Mean abs diff:    {float(diffs.mean()):.6e}")
        for t, v, ref_val, c_val, diff in rows[:10]:
            print(
                f"    t={t} v={v:5d}  hf={ref_val: .6e}  c={c_val: .6e}  diff={diff: .6e}"
            )

    # Compare d_final_ln_output (dY) for first 16 dims
    if not dfinal_c_map:
        print("\n⚠️  No VAL final_ln_dy lines parsed from C.")
    else:
        print("\n📊 d_final_ln_output (dY) comparison for first indices:")
        idxs = sorted(dfinal_c_map.keys())
        ref_vals = np.array([float(d_final_ref[0, i].item()) for i in idxs], dtype="float32")
        c_vals = np.array([dfinal_c_map[i] for i in idxs], dtype="float32")
        diffs = np.abs(ref_vals - c_vals)
        print(f"  Indices compared: {len(idxs)}")
        print(f"  Max abs diff:     {float(diffs.max()):.6e}")
        print(f"  Mean abs diff:    {float(diffs.mean()):.6e}")
        for i, ref_val, c_val, diff in zip(idxs[:10], ref_vals[:10], c_vals[:10], diffs[:10]):
            print(
                f"    idx={i:3d}  hf={ref_val: .6e}  c={c_val: .6e}  diff={diff: .6e}"
            )


if __name__ == "__main__":
    main()

