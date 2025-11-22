import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class LNBackwardTestConfig:
    tokens: int
    d_model: int
    eps: float = 1e-5
    seed: int = 0


def layernorm_forward_stats(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and rstd exactly the way the C code does:

        mean[t] = (1/D) * sum_d x[t,d]
        var[t]  = (1/D) * sum_d (x[t,d] - mean[t])^2
        rstd[t] = 1 / sqrt(var[t] + eps)
    """
    T, D = x.shape
    mean = x.mean(dim=1)
    var = ((x - mean.unsqueeze(1)) ** 2).mean(dim=1)
    rstd = torch.rsqrt(var + eps)
    return mean, rstd


def layernorm_backward_c_style(
    x: torch.Tensor,
    gamma: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    d_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch reimplementation of the C backward_layernorm logic
    (see main.c: backward_layernorm). This ignores padding/alignment
    and operates on logical [T, D] tensors.

    For each token t:
        x_hat = (x - mean[t]) * rstd[t]
        d_xhat = d_out * gamma
        dx_hat_sum[t]       = sum_d d_xhat[t,d]
        dx_hat_x_hat_sum[t] = sum_d d_xhat[t,d] * x_hat[t,d]

        scale = rstd[t] / D
        d_x[t,d] = scale * (D * d_xhat[t,d] - dx_hat_sum[t] - x_hat[t,d] * dx_hat_x_hat_sum[t])

    Parameter grads:
        d_gamma[d] = sum_t sum_d d_out[t,d] * x_hat[t,d]
        d_beta[d]  = sum_t sum_d d_out[t,d]
    """
    T, D = x.shape

    # Sanity checks
    assert gamma.shape == (D,)
    assert mean.shape == (T,)
    assert rstd.shape == (T,)
    assert d_out.shape == (T, D)

    # Compute x_hat
    x_hat = (x - mean.unsqueeze(1)) * rstd.unsqueeze(1)

    # d_xhat = dY * gamma
    d_xhat = d_out * gamma.unsqueeze(0)

    # Per-token sums
    dx_hat_sum = d_xhat.sum(dim=1)  # [T]
    dx_hat_x_hat_sum = (d_xhat * x_hat).sum(dim=1)  # [T]

    # Input gradients
    scale = rstd / float(D)  # [T]
    d_x = torch.empty_like(x)
    for t in range(T):
        d_x[t] = scale[t] * (
            float(D) * d_xhat[t] - dx_hat_sum[t] - x_hat[t] * dx_hat_x_hat_sum[t]
        )

    # Parameter gradients
    d_gamma = (d_out * x_hat).sum(dim=0)  # [D]
    d_beta = d_out.sum(dim=0)             # [D]

    return d_x, d_gamma, d_beta


def run_single_test(cfg: LNBackwardTestConfig) -> None:
    torch.manual_seed(cfg.seed)

    T, D, eps = cfg.tokens, cfg.d_model, cfg.eps

    # Random inputs
    x = torch.randn(T, D, dtype=torch.float32, requires_grad=True)
    gamma = torch.randn(D, dtype=torch.float32, requires_grad=True)
    beta = torch.randn(D, dtype=torch.float32, requires_grad=True)

    # Upstream gradient (what C calls d_output)
    upstream = torch.randn(T, D, dtype=torch.float32)

    # ---- PyTorch reference (HF-style LayerNorm) ----
    ln = nn.LayerNorm(D, eps=eps, elementwise_affine=True)
    with torch.no_grad():
        ln.weight.copy_(gamma)
        ln.bias.copy_(beta)

    y = ln(x)
    # loss = sum_t,d (y[t,d] * upstream[t,d]) so that dL/dy = upstream
    loss = (y * upstream).sum()
    loss.backward()

    x_grad_hf = x.grad.detach().clone()
    gamma_grad_hf = ln.weight.grad.detach().clone()
    beta_grad_hf = ln.bias.grad.detach().clone()

    # ---- C-style formula in Python ----
    with torch.no_grad():
        mean, rstd = layernorm_forward_stats(x.detach(), eps=eps)
        d_x_c, d_gamma_c, d_beta_c = layernorm_backward_c_style(
            x.detach(), gamma.detach(), mean, rstd, upstream
        )

    # ---- Compare ----
    def report(name: str, ref: torch.Tensor, test: torch.Tensor, max_items: int = 10):
        diff = (ref - test).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"\n{name} comparison:")
        print(f"  Shape:           {tuple(ref.shape)}")
        print(f"  Max abs diff:    {max_diff:.6e}")
        print(f"  Mean abs diff:   {mean_diff:.6e}")
        # Show a small slice for inspection
        flat_ref = ref.flatten()
        flat_test = test.flatten()
        flat_diff = diff.flatten()
        n = min(max_items, flat_ref.numel())
        print("  Sample (first elements):")
        for i in range(n):
            print(
                f"    idx {i:3d}: hf={flat_ref[i]: .6e}  c={flat_test[i]: .6e}  diff={flat_diff[i]: .6e}"
            )

    print(f"\n==== LayerNorm backward unit test: T={T}, D={D}, eps={eps} ====")
    report("d_x", x_grad_hf, d_x_c)
    report("d_gamma", gamma_grad_hf, d_gamma_c)
    report("d_beta", beta_grad_hf, d_beta_c)


def main():
    # A few representative shapes:
    tests = [
        LNBackwardTestConfig(tokens=1, d_model=1),
        LNBackwardTestConfig(tokens=1, d_model=8),
        LNBackwardTestConfig(tokens=2, d_model=8),
        LNBackwardTestConfig(tokens=4, d_model=16),
    ]

    for cfg in tests:
        run_single_test(cfg)


if __name__ == "__main__":
    main()

