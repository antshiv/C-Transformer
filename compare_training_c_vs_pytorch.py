#!/usr/bin/env python3
"""
Compare short training runs between C-Transformer and PyTorch GPT-2
on the same SQL (or TinyStories) windows, and then automatically run
inference from the trained checkpoints/models on a fixed prompt.

This script:
  1) Runs your C binary for N steps on --train-dir, capturing [train] loss
     and writing a checkpoint to checkpoints_compare_c/.
  2) Runs PyTorch GPT-2 for N steps on the same windows and objective
     (next-token LM), capturing loss.
  3) Prints a side-by-side table of step, C loss, and PyTorch loss so
     you can see whether the loss trajectories behave similarly.
  4) Runs inference after training using:
       - The C checkpoint (via run.py),
       - The trained PyTorch model (.generate),
     and prints a small table with final losses and a truncated sample
     of the generated text from each.

Notes:
  - C run starts from gpt2_bump.weights.
  - PyTorch run starts from HuggingFace gpt2.
  - Optimizers and LR are matched as closely as possible, but exact
    step-by-step equality is not expected; we care about qualitative
    behaviour (e.g., does loss collapse similarly, do both models
    produce reasonable SQL text after a few steps).
"""

import argparse
import os
import re
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import torch


def run_c_inference_from_ckpt(
    prompt: str,
    executable: str,
    ckpt_weights: str,
    num_tokens: int,
) -> tuple[str, str]:
    """
    Run C-Transformer inference via run.py using a specific checkpoint.
    Returns (full_text, generated_text).
    """
    cmd = [
        sys.executable,
        "run.py",
        prompt,
        "--weights",
        ckpt_weights,
        "--executable",
        executable,
        "--num-tokens",
        str(num_tokens),
    ]
    print("\n▶ Running C inference from checkpoint:")
    print("  ", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        print("❌ C inference run timed out.")
        return ("", "")

    if result.returncode != 0:
        print("❌ C inference returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        return ("", "")

    full_text = ""
    generated_text = ""
    lines = result.stdout.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("📤 GENERATED:"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                generated_text = lines[j].strip()
        if stripped.startswith("📖 FULL TEXT:"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                full_text = lines[j].strip()

    return (full_text, generated_text)


def run_c_training(
    executable: str,
    weights: str,
    train_dir: str,
    steps: int,
    lr: float,
    log_interval: int,
) -> List[Tuple[int, float]]:
    """
    Run the C binary for a short training loop and parse [train] losses.
    Returns a list of (step, loss) pairs.
    """
    # Use a dedicated checkpoint dir to avoid interfering with your main runs.
    ckpt_dir = "checkpoints_compare_c"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        executable,
        "--weights",
        weights,
        "--train-dir",
        train_dir,
        "--train-steps",
        str(steps),
        "--train-lr",
        str(lr),
        "--train-log-interval",
        str(log_interval),
        "--ckpt-dir",
        ckpt_dir,
        "--ckpt-interval",
        str(steps),
        "--optimizer",
        "adam",
        "--force",
    ]
    print("▶ Running C training:", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("❌ C training run timed out.")
        sys.exit(1)

    if result.returncode != 0:
        print("❌ C training returned non-zero exit code.")
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        sys.exit(1)

    losses: List[Tuple[int, float]] = []
    train_pat = re.compile(
        r"\[train\]\s+step=(\d+)/\d+\s+loss=([0-9.eE+-]+)"
    )
    for line in result.stdout.splitlines():
        m = train_pat.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            losses.append((step, loss))
    return losses


def load_model(model_name: str):
    try:
        from transformers import GPT2LMHeadModel
    except Exception as e:
        print(f"❌ Failed to import transformers: {e}")
        print("💡 Try: pip3 install transformers torch --break-system-packages")
        raise
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.train()
    model.to("cpu")
    return model


def run_pytorch_inference_from_model(
    prompt: str,
    model,
    model_name: str,
    num_tokens: int,
) -> tuple[str, str]:
    """
    Run inference using a trained PyTorch GPT-2 model.
    Returns (full_text, generated_text).
    """
    try:
        from transformers import GPT2Tokenizer
    except Exception as e:
        print(f"❌ Failed to import GPT2Tokenizer for inference: {e}")
        return ("", "")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            do_sample=False,
        )
    # full sequence includes prompt + generated
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    gen_only_ids = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
    return (full_text, generated_text)


def read_pair_header_and_tokens(path: str):
    import struct

    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 4:
        raise ValueError(f"File too small to contain header: {path}")
    ctx_len, tgt_len = struct.unpack("<HH", data[:4])
    tokens = np.frombuffer(data[4:], dtype="<u4")
    if tokens.size < ctx_len + tgt_len:
        raise ValueError(
            f"Not enough tokens in {path}: ctx_len={ctx_len}, tgt_len={tgt_len}, got {tokens.size}"
        )
    return ctx_len, tgt_len, tokens


def build_input_and_labels(path: str):
    """
    Build input_ids and labels for a single window, matching your C training:

      input_tokens  = tokens[0:ctx_len]
      labels        = tokens[1:ctx_len+1]  (next-token LM)
    """
    ctx_len, tgt_len, tokens = read_pair_header_and_tokens(path)
    if ctx_len + 1 > tokens.size:
        raise ValueError(f"Not enough tokens for next-token LM in {path}")
    input_ids = torch.tensor(tokens[:ctx_len], dtype=torch.long).unsqueeze(0)  # [1, T]
    labels = torch.tensor(tokens[1 : ctx_len + 1], dtype=torch.long).unsqueeze(0)  # [1, T]
    return input_ids, labels


def run_pytorch_training(
    model_name: str,
    train_dir: str,
    steps: int,
    lr: float,
    log_interval: int,
) -> tuple[List[Tuple[int, float]], torch.nn.Module]:
    """
    Run a short PyTorch GPT-2 training loop on the same windows and objective
    (next-token LM), returning (losses, trained_model).
    """
    import glob

    if not os.path.isdir(train_dir):
        print(f"❌ Training directory not found for PyTorch run: {train_dir}")
        sys.exit(1)

    pair_files = sorted(glob.glob(os.path.join(train_dir, "pair_*.bin")))
    if not pair_files:
        print(f"❌ No pair_*.bin files found in {train_dir}")
        sys.exit(1)

    model = load_model(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses: List[Tuple[int, float]] = []
    step = 0
    file_idx = 0

    while step < steps:
        path = pair_files[file_idx]
        file_idx = (file_idx + 1) % len(pair_files)

        try:
            input_ids, labels = build_input_and_labels(path)
        except Exception as e:
            print(f"⚠️  Skipping {path}: {e}")
            continue

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        step += 1
        if step % log_interval == 0 or step == 1:
            losses.append((step, float(loss.item())))

    return losses, model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare short C vs PyTorch GPT-2 training runs on SQL windows, "
            "and run inference from both after training."
        )
    )
    parser.add_argument(
        "--train-dir",
        default="data/sql_training_pairs",
        help="Directory with pair_*.bin windows (default: data/sql_training_pairs)",
    )
    parser.add_argument(
        "--weights",
        default="gpt2_bump.weights",
        help="C Transformer weights file (default: gpt2_bump.weights)",
    )
    parser.add_argument(
        "--executable",
        default="./main",
        help="C Transformer binary (default: ./main)",
    )
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help="HuggingFace GPT-2 model name (default: gpt2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps for each run (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for both C and PyTorch runs (default: 3e-5)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )
    parser.add_argument(
        "--prompt",
        default="SELECT * FROM users WHERE age > 10;",
        help="Prompt text to run inference on after training.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=20,
        help="Number of new tokens to generate for inference comparison (default: 20).",
    )
    args = parser.parse_args()

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

    print("============================================")
    print("Comparing C-Transformer vs PyTorch GPT-2 training")
    print("============================================")
    print(f"Train dir:   {args.train_dir}")
    print(f"Steps:       {args.steps}")
    print(f"LR:          {args.lr}")
    print()

    losses_c = run_c_training(
        args.executable,
        args.weights,
        args.train_dir,
        args.steps,
        args.lr,
        args.log_interval,
    )

    print("\n--------------------------------------------")
    print("Now running PyTorch GPT-2 training …")
    print("--------------------------------------------\n")

    losses_pt, model_pt = run_pytorch_training(
        args.model_name,
        args.train_dir,
        args.steps,
        args.lr,
        args.log_interval,
    )

    # Merge and print side-by-side (by step)
    print("\nStep |   C loss   |  PT loss  ")
    print("-----+-----------+-----------")
    # Build dicts for quick lookup
    c_dict = {s: l for s, l in losses_c}
    pt_dict = {s: l for s, l in losses_pt}
    steps_all = sorted(set(c_dict.keys()) | set(pt_dict.keys()))
    for s in steps_all:
        lc = c_dict.get(s, float("nan"))
        lp = pt_dict.get(s, float("nan"))
        print(f"{s:4d} | {lc:9.6f} | {lp:9.6f}")

    # ------------------------------------------------------------------
    # Inference comparison from trained checkpoints / models (Step 3)
    # ------------------------------------------------------------------
    if losses_c and losses_pt:
        final_step_c, final_loss_c = losses_c[-1]
        final_step_pt, final_loss_pt = losses_pt[-1]
    else:
        final_step_c = final_step_pt = args.steps
        final_loss_c = final_loss_pt = float("nan")

    ckpt_dir = "checkpoints_compare_c"
    ckpt_path = os.path.join(
        ckpt_dir, f"ckpt_step_{args.steps:06d}.weights"
    )

    if not os.path.exists(ckpt_path):
        print(
            f"\n⚠️  C checkpoint not found at {ckpt_path}; skipping C inference comparison."
        )
        c_full, c_gen = ("", "")
    else:
        c_full, c_gen = run_c_inference_from_ckpt(
            args.prompt, args.executable, ckpt_path, args.gen_tokens
        )

    pt_full, pt_gen = run_pytorch_inference_from_model(
        args.prompt, model_pt, args.model_name, args.gen_tokens
    )

    def shorten(text: str, length: int = 80) -> str:
        t = text.replace("\n", " ").strip()
        return t if len(t) <= length else t[: length - 3] + "..."

    print("\n============================================")
    print("Inference after training (C vs PyTorch)")
    print("============================================")
    print(f"Prompt: {args.prompt}")
    print()
    print("System   | Final step |    Loss   | Generated snippet")
    print("---------+-----------+----------+--------------------------")
    print(
        f"C-Trans  | {final_step_c:9d} | {final_loss_c:8.6f} | {shorten(c_gen)}"
    )
    print(
        f"PyTorch  | {final_step_pt:9d} | {final_loss_pt:8.6f} | {shorten(pt_gen)}"
    )


if __name__ == "__main__":
    main()

