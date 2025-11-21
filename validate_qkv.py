#!/usr/bin/env python3
"""
Validate Q/K/V projection for a single GPT-2 layer between C-Transformer and
HuggingFace GPT-2, using the DEBUG_QKV prints from the C binary.

Prerequisites:
  - main.c compiled to ./main with --debug-layer support and DEBUG_QKV block:

        // After qkv_projection_head_major(...)
        if (M->debug_layer >= 0 && layer_idx == M->debug_layer) {
            int last_tok = M->active_tokens - 1;
            ...
            printf("DEBUG_QKV token=%d head=0:\\n", last_tok);
            for (int d = 0; d < 5; d++) {
                ...
                printf(\"  Q[%d]=%.6f K[%d]=%.6f V[%d]=%.6f\\n\", ...);
            }
        }

Usage:
  python3 validate_qkv.py "Hello World" --layer 0 \
      --weights gpt2_bump.weights --executable ./main
"""

import argparse
import os
import re
import subprocess
import sys
from typing import List, Tuple


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


def run_c_debug_qkv(
    executable: str,
    weights: str,
    token_ids: List[int],
    layer_idx: int,
) -> Tuple[int, List[float], List[float], List[float]]:
    """
    Run the C binary with --debug-layer L and parse DEBUG_QKV output.

    Returns:
      last_token_index, q_vals, k_vals, v_vals for head 0 (as printed by C).
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
    print("▶ Running C binary for DEBUG_QKV:")
    print("  ", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping DEBUG_QKV.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    # Example output:
    #   DEBUG_QKV token=1 head=0:
    #     Q[0]=... K[0]=... V[0]=...
    #     Q[1]=... K[1]=... V[1]=...
    token_pat = re.compile(r"DEBUG_QKV token=(\d+)\s+head=0")
    row_pat = re.compile(
        r"Q\[(\d+)\]=([\-0-9.eE]+)\s+K\[\1\]=([\-0-9.eE]+)\s+V\[\1\]=([\-0-9.eE]+)"
    )

    last_token_idx = None
    q_vals = {}
    k_vals = {}
    v_vals = {}

    for line in result.stdout.splitlines():
        m_tok = token_pat.search(line)
        if m_tok:
            last_token_idx = int(m_tok.group(1))
            continue

        m_row = row_pat.search(line)
        if m_row:
            idx = int(m_row.group(1))
            q = float(m_row.group(2))
            k = float(m_row.group(3))
            v = float(m_row.group(4))
            q_vals[idx] = q
            k_vals[idx] = k
            v_vals[idx] = v

    if last_token_idx is None or not q_vals:
        print("❌ Failed to parse DEBUG_QKV lines from C output.")
        print("=== STDOUT ===")
        print(result.stdout)
        sys.exit(1)

    max_idx = max(q_vals.keys())
    q_vec = [q_vals[i] for i in range(max_idx + 1)]
    k_vec = [k_vals[i] for i in range(max_idx + 1)]
    v_vec = [v_vals[i] for i in range(max_idx + 1)]

    return last_token_idx, q_vec, k_vec, v_vec


def main():
    parser = argparse.ArgumentParser(description="Validate Q/K/V projection vs HuggingFace GPT-2")
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

    # Run C side to get Q/K/V for last token, head 0
    last_tok_idx_c, q_c, k_c, v_c = run_c_debug_qkv(
        args.executable, args.weights, token_ids, args.layer
    )

    # HF side: compute Q/K/V for same layer, prompt, head
    import torch
    import numpy as np

    with torch.no_grad():
        outputs = model(input_ids)
        # We need ln_1(h_in) as input to c_attn
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

        # GPT-2 attention: c_attn projects to 3*D, then split into q, k, v
        qkv = block.attn.c_attn(ln1)  # [1, T, 3*D]
        D = qkv.size(-1) // 3
        q, k, v = qkv.split(D, dim=2)  # each [1, T, D]

        n_head = block.attn.num_heads
        head_dim = D // n_head

        q = q.view(1, -1, n_head, head_dim)
        k = k.view(1, -1, n_head, head_dim)
        v = v.view(1, -1, n_head, head_dim)

        T = q.size(1)
        last_tok_idx_hf = T - 1

        if last_tok_idx_c != last_tok_idx_hf:
            print(f"⚠️  Last token index mismatch: C={last_tok_idx_c}, HF={last_tok_idx_hf}")

        # Head 0, last token
        q_hf = q[0, last_tok_idx_hf, 0, :].cpu().numpy()
        k_hf = k[0, last_tok_idx_hf, 0, :].cpu().numpy()
        v_hf = v[0, last_tok_idx_hf, 0, :].cpu().numpy()

    def compare(name: str, hf_vec: np.ndarray, c_vec: List[float]):
        dim = min(len(c_vec), hf_vec.size)
        hf_slice = hf_vec[:dim]
        c_slice = np.array(c_vec[:dim], dtype="float32")
        diff = np.abs(hf_slice - c_slice)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        print(f"\n{name} comparison (head 0, last token):")
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
            # Fallback if pandas is not installed
            for i in range(min(dim, 10)):
                print(
                    f"  idx={i:3d}  hf={hf_slice[i]: .6f}  c={c_slice[i]: .6f}  diff={diff[i]: .6f}"
                )

    print("\n📊 Q/K/V projection validation for layer", args.layer)
    compare("Q", q_hf, q_c)
    compare("K", k_hf, k_c)
    compare("V", v_hf, v_c)


if __name__ == "__main__":
    main()

