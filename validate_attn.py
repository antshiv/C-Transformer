#!/usr/bin/env python3
"""
Validate attention scores and per-head attention output for a single GPT-2
layer between C-Transformer and HuggingFace GPT-2.

This uses the C binary's `--debug-layer L` mode, which prints from
debug_forward_dump_layer_output:

  LAYER_ATTNSCORES layer=L token=T head=0:
    ATTNSCORE j=0 value=...
    ATTNSCORE j=1 value=...
    ...

  LAYER_ATTNOUT layer=L token=T head=0:
    ATTN_OUT idx=0 value=...
    ATTN_OUT idx=1 value=...
    ...

Usage:
  python3 validate_attn.py "Hello World" --layer 0 \
      --weights gpt2_bump.weights --executable ./main
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
    model.eval()
    model.to("cpu")
    return tokenizer, model


def run_c_debug_attn(
    executable: str,
    weights: str,
    token_ids: List[int],
    layer_idx: int,
) -> Tuple[int, List[float], List[float]]:
    """
    Run the C binary with --debug-layer L and parse attention scores/output.

    Returns:
      last_token_index,
      attn_scores_c: softmax scores for head 0, last token, j=0..T-1
      attn_out_c:    attention output vector for head 0, last token (dim=head_dim)
    """
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
    print("▶ Running C binary for attention debug:")
    print("  ", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping attention debug.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    # Example:
    # LAYER_ATTNSCORES layer=0 token=1 head=0:
    #   ATTNSCORE j=0 value=...
    #   ATTNSCORE j=1 value=...
    # LAYER_ATTNOUT layer=0 token=1 head=0:
    #   ATTN_OUT idx=0 value=...
    #   ...
    header_scores = re.compile(r"LAYER_ATTNSCORES\s+layer=(\d+)\s+token=(\d+)\s+head=0")
    row_score = re.compile(r"ATTNSCORE\s+j=(\d+)\s+value=([\-0-9.eE]+)")

    header_out = re.compile(r"LAYER_ATTNOUT\s+layer=(\d+)\s+token=(\d+)\s+head=0")
    row_out = re.compile(r"ATTN_OUT\s+idx=(\d+)\s+value=([\-0-9.eE]+)")

    current_token_scores = None
    attn_scores: Dict[int, float] = {}
    attn_out_vals: Dict[int, float] = {}
    last_token_idx = None

    for line in result.stdout.splitlines():
        m_hs = header_scores.search(line)
        if m_hs:
            layer = int(m_hs.group(1))
            token_idx = int(m_hs.group(2))
            if layer == layer_idx:
                current_token_scores = token_idx
                last_token_idx = token_idx
            continue

        if current_token_scores is not None:
            m_rs = row_score.search(line)
            if m_rs:
                j = int(m_rs.group(1))
                val = float(m_rs.group(2))
                attn_scores[j] = val
                continue

        m_ho = header_out.search(line)
        if m_ho:
            layer = int(m_ho.group(1))
            token_idx = int(m_ho.group(2))
            if layer == layer_idx:
                last_token_idx = token_idx
            continue

        m_ro = row_out.search(line)
        if m_ro:
            idx = int(m_ro.group(1))
            val = float(m_ro.group(2))
            attn_out_vals[idx] = val
            continue

    if last_token_idx is None or not attn_scores or not attn_out_vals:
        print("❌ Failed to parse attention debug lines from C output.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)

    max_j = max(attn_scores.keys())
    scores_vec = [attn_scores.get(j, 0.0) for j in range(max_j + 1)]

    max_d = max(attn_out_vals.keys())
    attn_out_vec = [attn_out_vals.get(i, 0.0) for i in range(max_d + 1)]

    return last_token_idx, scores_vec, attn_out_vec


def main():
    parser = argparse.ArgumentParser(description="Validate attention scores/output vs HuggingFace GPT-2")
    parser.add_argument("text", help="Prompt text to validate")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to validate (0-based)")
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

    # C side: get attention scores and output
    last_tok_c, scores_c, attn_out_c = run_c_debug_attn(
        args.executable, args.weights, token_ids, args.layer
    )

    # HF side: recompute attention for head 0, last token
    import torch
    import numpy as np

    with torch.no_grad():
        # Need hidden states to feed into block.ln_1
        model.config.output_hidden_states = True
        outputs = model(input_ids)
        if outputs.hidden_states is None:
            print("❌ Model did not return hidden_states; ensure output_hidden_states=True.")
            sys.exit(1)

        hs = outputs.hidden_states
        if args.layer >= len(hs) - 1:
            print(f"❌ Requested layer {args.layer}, but model returned only {len(hs)} hidden_states.")
            sys.exit(1)

        h_in = hs[args.layer]  # [1, T, D]
        block = model.transformer.h[args.layer]
        ln1 = block.ln_1(h_in)  # [1, T, D]

        # QKV from c_attn
        qkv = block.attn.c_attn(ln1)  # [1, T, 3*D]
        D = qkv.size(-1) // 3
        q, k, v = qkv.split(D, dim=2)  # each [1, T, D]

        n_head = block.attn.num_heads
        head_dim = D // n_head

        q = q.view(1, -1, n_head, head_dim)
        k = k.view(1, -1, n_head, head_dim)
        v = v.view(1, -1, n_head, head_dim)

        T = q.size(1)
        last_tok_hf = T - 1

        if last_tok_c != last_tok_hf:
            print(f"⚠️  Last token index mismatch: C={last_tok_c}, HF={last_tok_hf}")

        # Head 0, last token
        q_h = q[0, last_tok_hf, 0, :]           # [head_dim]
        k_all = k[0, :T, 0, :]                  # [T, head_dim]
        v_all = v[0, :T, 0, :]                  # [T, head_dim]

        scale = 1.0 / (float(head_dim) ** 0.5)
        # scores_raw: [T]
        scores_raw = torch.matmul(k_all, q_h) * scale  # K·Q (same as Q·K^T) / sqrt(d)

        # Causal mask: for last token, only j <= last_tok_hf; others are irrelevant.
        scores_masked = scores_raw.clone()
        # Softmax over 0..last_tok_hf (but since last_tok_hf == T-1, it's full range)
        probs = torch.softmax(scores_masked, dim=0)  # [T]

        probs_np = probs.cpu().numpy()
        attn_out_hf = torch.matmul(probs.unsqueeze(0), v_all)  # [1, head_dim]
        attn_out_np = attn_out_hf[0].cpu().numpy()             # [head_dim]

    # Compare scores
    dim_scores = min(len(scores_c), probs_np.size)
    scores_c_arr = np.array(scores_c[:dim_scores], dtype="float32")
    scores_hf = probs_np[:dim_scores]
    diff_scores = np.abs(scores_hf - scores_c_arr)
    max_diff_scores = float(diff_scores.max())
    mean_diff_scores = float(diff_scores.mean())

    print("\n📊 Attention scores comparison (head 0, last token):")
    print(f"  Dimension:    {dim_scores}")
    print(f"  Max abs diff: {max_diff_scores:.6f}")
    print(f"  Mean abs diff:{mean_diff_scores:.6f}")

    try:
        import pandas as pd

        rows = []
        for j in range(min(dim_scores, 10)):
            rows.append(
                {
                    "j": j,
                    "hf": float(scores_hf[j]),
                    "c": float(scores_c_arr[j]),
                    "diff": float(diff_scores[j]),
                }
            )
        df = pd.DataFrame(rows)
        print("  Sample (first 10 positions):")
        print(df.to_string(index=False))
    except Exception:
        for j in range(min(dim_scores, 10)):
            print(
                f"  j={j:3d}  hf={scores_hf[j]: .6f}  c={scores_c_arr[j]: .6f}  diff={diff_scores[j]: .6f}"
            )

    # Compare attention output
    dim_out = min(len(attn_out_c), attn_out_np.size)
    attn_out_c_arr = np.array(attn_out_c[:dim_out], dtype="float32")
    attn_out_hf = attn_out_np[:dim_out]
    diff_out = np.abs(attn_out_hf - attn_out_c_arr)
    max_diff_out = float(diff_out.max())
    mean_diff_out = float(diff_out.mean())

    print("\n📊 Attention output comparison (head 0, last token):")
    print(f"  Dimension:    {dim_out}")
    print(f"  Max abs diff: {max_diff_out:.6f}")
    print(f"  Mean abs diff:{mean_diff_out:.6f}")

    try:
        import pandas as pd

        rows = []
        for i in range(min(dim_out, 10)):
            rows.append(
                {
                    "idx": i,
                    "hf": float(attn_out_hf[i]),
                    "c": float(attn_out_c_arr[i]),
                    "diff": float(diff_out[i]),
                }
            )
        df = pd.DataFrame(rows)
        print("  Sample (first 10 dims):")
        print(df.to_string(index=False))
    except Exception:
        for i in range(min(dim_out, 10)):
            print(
                f"  idx={i:3d}  hf={attn_out_hf[i]: .6f}  c={attn_out_c_arr[i]: .6f}  diff={diff_out[i]: .6f}"
            )


if __name__ == "__main__":
    main()

