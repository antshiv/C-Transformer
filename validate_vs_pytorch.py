#!/usr/bin/env python3
"""
Validate C-Transformer logits against HuggingFace GPT-2.

Usage:
  python3 validate_vs_pytorch.py "Once upon a time"

Options:
  --weights FILE      Path to gpt2_bump.weights (default: gpt2_bump.weights)
  --executable FILE   C binary (default: ./main)
  --top-k N           Compare top-K logits (default: 10)
  --temperature T     Sampling temperature for reference (default: 1.0)
  --model-name NAME   HF model name (default: gpt2)
  --save-csv FILE     Optional: save comparison as CSV
"""

import argparse
import os
import re
import subprocess
import sys


def load_tokenizer_and_model(model_name: str):
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        import torch
    except Exception as e:
        print(f"❌ Failed to import transformers/torch: {e}")
        print("💡 Try: pip3 install transformers torch --break-system-packages")
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # Ensure we get hidden_states in the output for validation
    model.config.output_hidden_states = True
    model.eval()
    model.to("cpu")
    return tokenizer, model


def run_c_debug_logits(executable: str, weights: str, token_ids, top_k: int):
    tokens_str = ",".join(str(t) for t in token_ids)
    cmd = [
        executable,
        "--weights",
        weights,
        "--prompt",
        tokens_str,
        "--force",
        "--debug-logits",
        "--debug-top-k",
        str(top_k),
    ]
    print("▶ Running C binary for debug logits:")
    print("  ", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping logits.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    # Parse lines of the form: LOGIT idx=123 value=0.123456
    logits = {}
    pattern = re.compile(r"LOGIT idx=(\d+) value=([\-0-9.eE]+)")
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if m:
            idx = int(m.group(1))
            val = float(m.group(2))
            logits[idx] = val
    if not logits:
        print("⚠️  No LOGIT lines found in C output; check that --debug-logits is wired correctly.")
        print("=== STDOUT ===")
        print(result.stdout)
    return logits


def run_c_debug_hidden(executable: str, weights: str, token_ids):
    tokens_str = ",".join(str(t) for t in token_ids)
    cmd = [
        executable,
        "--weights",
        weights,
        "--prompt",
        tokens_str,
        "--force",
        "--debug-hidden",
    ]
    print("▶ Running C binary for debug hidden state:")
    print("  ", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("❌ C binary timed out while dumping hidden state.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C binary returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    # Parse lines of the form: HIDDEN idx=123 value=0.123456
    values = {}
    pattern = re.compile(r"HIDDEN idx=(\d+) value=([\-0-9.eE]+)")
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if m:
            idx = int(m.group(1))
            val = float(m.group(2))
            values[idx] = val
    if not values:
        print("⚠️  No HIDDEN lines found in C output; check that --debug-hidden is wired correctly.")
        print("=== STDOUT ===")
        print(result.stdout)
    # Convert to a dense list ordered by idx
    if not values:
        return []
    max_idx = max(values.keys())
    vec = [0.0] * (max_idx + 1)
    for idx, val in values.items():
        vec[idx] = val
    return vec


def main():
    parser = argparse.ArgumentParser(description="Validate C-Transformer logits vs HuggingFace GPT-2")
    parser.add_argument("text", help="Prompt text to validate")
    parser.add_argument("--weights", default="gpt2_bump.weights", help="Path to C weight file")
    parser.add_argument("--executable", default="./main", help="Path to C binary")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K logits to compare")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for reference softmax")
    parser.add_argument("--model-name", default="gpt2", help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--save-csv", default=None, help="Optional path to save logits comparison as CSV")
    parser.add_argument("--compare-hidden", action="store_true", help="Also compare final hidden state (last token)")
    args = parser.parse_args()

    # Basic file checks to make it more user-friendly
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

    # Run C binary in debug-logits mode
    c_logits = run_c_debug_logits(args.executable, args.weights, token_ids, args.top_k)
    if not c_logits:
        sys.exit(1)

    # Run HF model
    import torch

    with torch.no_grad():
        outputs = model(input_ids)
        hf_logits = outputs.logits[0, -1, :]  # last position
        # GPT2LMHeadModel returns hidden_states when output_hidden_states=True
        # Use the last hidden state tensor from hidden_states
        if outputs.hidden_states is None:
            print("❌ Model did not return hidden_states; ensure output_hidden_states=True.")
            sys.exit(1)
        hf_hidden = outputs.hidden_states[-1][0, -1, :]  # final hidden for last token

    # Compute HF top-K
    top_k = min(args.top_k, hf_logits.numel())
    hf_top_vals, hf_top_idx = torch.topk(hf_logits, top_k)

    # Build logits comparison rows (for printing and optional DataFrame/CSV)
    rows = []
    diffs = []
    for rank in range(top_k):
        idx = int(hf_top_idx[rank].item())
        hf_val = float(hf_top_vals[rank].item())
        c_val = c_logits.get(idx, float("nan"))
        diff = abs(hf_val - c_val) if c_val == c_val else float("nan")
        tok_str = tokenizer.decode([idx])
        rows.append(
            {
                "rank": rank,
                "token_id": idx,
                "token": tok_str,
                "hf_logit": hf_val,
                "c_logit": c_val,
                "diff": diff,
            }
        )
        diffs.append(diff if diff == diff else 0.0)

    print("\n📊 Top-K logits comparison (HF vs C) for last token:\n")
    print(f"{'rank':>4} {'id':>6} {'hf_logit':>12} {'c_logit':>12} {'diff':>12}  token")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['rank']:4d} {r['token_id']:6d} "
            f"{r['hf_logit']:12.6f} {r['c_logit']:12.6f} {r['diff']:12.6f}  {r['token']!r}"
        )

    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        print(f"\n📈 Mean abs logit diff over top-{top_k}: {mean_diff:.6f}")

    # Optional DataFrame / CSV output for logits
    if args.save_csv is not None:
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_csv(args.save_csv, index=False)
            print(f"\n💾 Saved comparison to {args.save_csv}")
        except Exception as e:
            print(f"\n⚠️  Could not use pandas/save CSV: {e}")

    # Optional hidden-state comparison
    if args.compare_hidden:
        c_hidden = run_c_debug_hidden(args.executable, args.weights, token_ids)
        if not c_hidden:
            sys.exit(1)

        import torch

        hf_vec = hf_hidden.cpu()
        dim = hf_vec.numel()
        if len(c_hidden) < dim:
            print(f"\n⚠️  C hidden vector shorter than HF hidden ({len(c_hidden)} < {dim}); "
                  "comparison limited to available length.")
            dim = len(c_hidden)

        hidden_rows = []
        hidden_diffs = []
        for i in range(dim):
            hf_val = float(hf_vec[i].item())
            c_val = float(c_hidden[i])
            diff = abs(hf_val - c_val)
            hidden_rows.append(
                {
                    "idx": i,
                    "hf_hidden": hf_val,
                    "c_hidden": c_val,
                    "diff": diff,
                }
            )
            hidden_diffs.append(diff)

        max_diff = max(hidden_diffs) if hidden_diffs else float("nan")
        mean_diff_hidden = sum(hidden_diffs) / len(hidden_diffs) if hidden_diffs else float("nan")

        print("\n📊 Hidden state comparison (HF vs C) for last token:")
        print(f"   Dimension: {dim}")
        print(f"   Max abs diff:  {max_diff:.6f}")
        print(f"   Mean abs diff: {mean_diff_hidden:.6f}")

        # If pandas is available, show a small DataFrame preview
        try:
            import pandas as pd

            df_hid = pd.DataFrame(hidden_rows)
            print("\n🔎 Hidden state diff sample (first 10 dims):")
            print(df_hid.head(10).to_string(index=False))
        except Exception:
            # pandas is optional; skip if not available
            pass

    # Optional: compare softmax distribution at given temperature
    if args.temperature != 1.0:
        print(f"\n📈 Note: You requested temperature={args.temperature}, "
              "apply it equally to both sides when comparing probabilities.")


if __name__ == "__main__":
    main()
