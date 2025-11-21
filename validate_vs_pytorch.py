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


def main():
    parser = argparse.ArgumentParser(description="Validate C-Transformer logits vs HuggingFace GPT-2")
    parser.add_argument("text", help="Prompt text to validate")
    parser.add_argument("--weights", default="gpt2_bump.weights", help="Path to C weight file")
    parser.add_argument("--executable", default="./main", help="Path to C binary")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K logits to compare")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for reference softmax")
    parser.add_argument("--model-name", default="gpt2", help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--save-csv", default=None, help="Optional path to save comparison as CSV")
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

    # Compute HF top-K
    top_k = min(args.top_k, hf_logits.numel())
    hf_top_vals, hf_top_idx = torch.topk(hf_logits, top_k)

    # Build comparison rows (for printing and optional DataFrame/CSV)
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

    # Optional DataFrame / CSV output
    if args.save_csv is not None:
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_csv(args.save_csv, index=False)
            print(f"\n💾 Saved comparison to {args.save_csv}")
        except Exception as e:
            print(f"\n⚠️  Could not use pandas/save CSV: {e}")

    # Optional: compare softmax distribution at given temperature
    if args.temperature != 1.0:
        print(f"\n📈 Note: You requested temperature={args.temperature}, "
              "apply it equally to both sides when comparing probabilities.")


if __name__ == "__main__":
    main()
