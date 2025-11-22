#!/usr/bin/env python3
"""
Unit test for LM head + cross-entropy backward, matching C-Transformer logic.

Goal:
  - Reimplement compute_cross_entropy_loss + backward_lm_head in Python.
  - Compare their outputs to HuggingFace GPT-2:
      * d_logits               vs logits.grad
      * d_final_ln_output (dY) vs grad_output from ln_f
      * d_embed_weights        vs wte.weight.grad

If this unit test passes (small max/mean diffs), it means the LM head + loss
math is correct. Any mismatch in C then comes from indexing / layout bugs.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class LMHeadTestConfig:
    text: str
    model_name: str = "gpt2"
    eps: float = 1e-5


def load_tokenizer_and_model(model_name: str):
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
    except Exception as e:
        print(f"❌ Failed to import transformers: {e}")
        print("💡 Try: pip3 install transformers torch --break-system-packages")
        raise

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


def c_style_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    C-style cross-entropy as in compute_cross_entropy_loss:

      For each token t:
        p = softmax(logits[t])
        loss_t = -log p[correct_t]
      loss = mean_t loss_t

    Gradients:
      d_logits[t,v] = (p[v] - 1_{v=correct}) / T   where T = number of tokens
    """
    # logits: [1, T, V]
    logits = logits.detach()
    T = logits.size(1)
    V = logits.size(2)
    assert targets.shape == (1, T)

    # Compute softmax probabilities
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, T, V]
    probs = log_probs.exp()                        # [1, T, V]

    # Per-token loss and mean
    loss_terms = -log_probs[0, torch.arange(T), targets[0]]  # [T]
    loss = loss_terms.mean()

    # d_logits: p - 1 for correct, p for others, all scaled by 1/T
    d_logits = probs.clone()           # [1, T, V]
    d_logits[0, torch.arange(T), targets[0]] -= 1.0
    d_logits /= float(T)

    return loss, d_logits[0]  # [T, V]


def c_style_backward_lm_head(
    d_logits: torch.Tensor,          # [T, V]
    final_ln_output: torch.Tensor,   # [T, D]
    lm_head_weights: torch.Tensor,   # [V, D] (tied to embeddings)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    C-style backward_lm_head (see main.c: backward_lm_head), but vectorized:

      d_final_ln_output[t,d] = sum_v d_logits[t,v] * W[v,d]
      d_embed_weights[v,d]   = sum_t d_logits[t,v] * final_ln_output[t,d]
    """
    T, V = d_logits.shape
    Vw, D = lm_head_weights.shape
    assert V == Vw
    assert final_ln_output.shape == (T, D)

    # d_final_ln_output: [T, D] = [T, V] @ [V, D]
    d_final_ln_output = d_logits @ lm_head_weights

    # d_embed_weights: [V, D] = [V, T] @ [T, D]
    d_embed_weights = d_logits.t() @ final_ln_output

    return d_final_ln_output, d_embed_weights


def run_single_test(cfg: LMHeadTestConfig) -> None:
    tokenizer, model = load_tokenizer_and_model(cfg.model_name)

    # Encode prompt
    input_ids = tokenizer.encode(cfg.text, return_tensors="pt")
    token_ids = input_ids[0].tolist()
    print(f"\n==== LM head backward unit test ====")
    print(f"Prompt: {cfg.text!r}")
    print(f"Tokens: {token_ids}")

    # HF forward
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    T = input_ids.size(1)

    # Hook to capture ln_f input and grad_output (dY)
    hook_vals = {}

    def ln_f_fwd_hook(module, inputs, output):
        hook_vals["ln_f_output"] = output.detach().cpu()  # y = LN(x)

    def ln_f_bwd_hook(module, grad_input, grad_output):
        hook_vals["ln_f_grad_output"] = grad_output[0].detach().cpu()

    handle_fwd = model.transformer.ln_f.register_forward_hook(ln_f_fwd_hook)
    handle_bwd = model.transformer.ln_f.register_full_backward_hook(ln_f_bwd_hook)

    # Run forward, retain logits grad
    model.zero_grad(set_to_none=True)
    out = model(input_ids)
    logits = out.logits  # [1, T, V]
    logits.retain_grad()

    targets = input_ids.clone()  # same LM task: predict each token itself
    log_probs = torch.log_softmax(logits, dim=-1)
    loss_terms = -log_probs[0, torch.arange(T), targets[0]]
    loss_hf = loss_terms.mean()
    loss_hf.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    # Collect HF tensors
    d_logits_hf = logits.grad.detach().cpu()[0]               # [T, V]
    ln_f_output = hook_vals["ln_f_output"][0]                 # [T, D]
    d_final_output_hf = hook_vals["ln_f_grad_output"][0]      # [T, D]
    embed_grad_hf = model.transformer.wte.weight.grad.detach().cpu()  # [V, D]

    lm_head_weights = model.transformer.wte.weight.detach().cpu()     # [V, D], tied

    # C-style in Python: cross-entropy + backward_lm_head
    loss_c_style, d_logits_c = c_style_cross_entropy(logits, targets)  # [T, V]
    d_final_output_c, d_embed_weights_c = c_style_backward_lm_head(
        d_logits_c, ln_f_output, lm_head_weights
    )

    def report(name: str, ref: torch.Tensor, test: torch.Tensor, max_items: int = 10):
        diff = (ref - test).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        print(f"\n{name} comparison:")
        print(f"  Shape:           {tuple(ref.shape)}")
        print(f"  Max abs diff:    {max_diff:.6e}")
        print(f"  Mean abs diff:   {mean_diff:.6e}")
        flat_ref = ref.flatten()
        flat_test = test.flatten()
        flat_diff = diff.flatten()
        n = min(max_items, flat_ref.numel())
        print("  Sample (first elements):")
        for i in range(n):
            print(
                f"    idx {i:3d}: hf={flat_ref[i]: .6e}  c={flat_test[i]: .6e}  diff={flat_diff[i]: .6e}"
            )

    print(f"\nHF loss: {float(loss_hf):.9g}  |  C-style loss: {float(loss_c_style):.9g}")
    report("d_logits", d_logits_hf, d_logits_c)
    report("d_final_ln_output (dY)", d_final_output_hf, d_final_output_c)
    report("embed_weight_grad", embed_grad_hf, d_embed_weights_c)


def main():
    tests: List[LMHeadTestConfig] = [
        LMHeadTestConfig(text="Hello"),
        LMHeadTestConfig(text="Hello World"),
    ]
    for cfg in tests:
        run_single_test(cfg)


if __name__ == "__main__":
    main()

