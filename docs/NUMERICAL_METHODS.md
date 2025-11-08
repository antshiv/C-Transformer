# Numerical Methods & Backpropagation Mathematics

## Overview

This document provides comprehensive mathematical explanations of all backward pass operations and numerical stability techniques used in C-Transformer. Every gradient computation and numerical trick is explained with derivations, implementation details, and justifications.

---

## Table of Contents

1. [Numerical Stability Techniques](#numerical-stability-techniques)
2. [Softmax Backward: The Jacobian Derivation](#softmax-backward-the-jacobian-derivation)
3. [Cross-Entropy Loss Gradient](#cross-entropy-loss-gradient)
4. [LayerNorm Backward Derivation](#layernorm-backward-derivation)
5. [GELU Activation Backward](#gelu-activation-backward)
6. [Attention Mechanism Gradients](#attention-mechanism-gradients)
7. [Matrix Multiplication Gradients](#matrix-multiplication-gradients)
8. [Numerical Precision Considerations](#numerical-precision-considerations)

---

## Numerical Stability Techniques

### 1. Log-Sum-Exp Trick (Softmax Forward)

**Problem**: Computing softmax naively can cause **overflow** or **underflow**.

```
Naive softmax: y[i] = exp(x[i]) / Σ(exp(x[j]))
```

If `x[i]` is large (e.g., x = [1000, 1001, 1002]):
- `exp(1000) ≈ 10^434` → **OVERFLOW** (float max ≈ 10^38)

If `x[i]` is very negative (e.g., x = [-1000, -999, -998]):
- `exp(-1000) ≈ 10^-434` → **UNDERFLOW** (float min ≈ 10^-38)

**Solution**: Subtract the maximum value before exponentiation.

```
Stable softmax:
  max_x = max(x)
  y[i] = exp(x[i] - max_x) / Σ(exp(x[j] - max_x))
```

**Why it works**:
```
Σ exp(x[j]) = Σ exp(x[j] - max_x + max_x)
            = exp(max_x) × Σ exp(x[j] - max_x)

Therefore:
y[i] = exp(x[i]) / Σ exp(x[j])
     = exp(x[i] - max_x) / Σ exp(x[j] - max_x)  ← numerically stable!
```

After subtraction:
- Largest value becomes `exp(0) = 1.0` (no overflow)
- All other values are `exp(negative)` ≤ 1.0 (controllable underflow)

**Implementation** (main.c:3154-3174):
```c
// Find max for numerical stability (only over valid positions j <= i)
float max_val = ATTN_SCORES_ACCESS(attn_scores, h, i, 0, num_tokens);
for (int j = 1; j <= i; ++j) {
    float score = ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens);
    if (score > max_val) max_val = score;
}

// Compute exp(score - max) and sum (only for j <= i)
float sum = 0.0f;
for (int j = 0; j <= i; ++j) {
    float score = ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens);
    float exp_score = expf(score - max_val);  // ← Subtract max!
    ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens) = exp_score;
    sum += exp_score;
}

// Normalize (only for j <= i)
float inv_sum = 1.0f / sum;
for (int j = 0; j <= i; ++j) {
    ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens) *= inv_sum;
}
```

**Cross-entropy loss** also uses this trick (main.c:6231-6242):
```c
// Find max for numerical stability
float max_logit = token_logits[0];
for (int v = 1; v < M->vocab_size; v++) {
    if (token_logits[v] > max_logit) {
        max_logit = token_logits[v];
    }
}

// Compute softmax with log-sum-exp stability
float sum_exp = 0.0f;
for (int v = 0; v < M->vocab_size; v++) {
    token_d_logits[v] = expf(token_logits[v] - max_logit);
    sum_exp += token_d_logits[v];
}
```

---

### 2. Epsilon for Division by Zero (LayerNorm)

**Problem**: LayerNorm computes `1 / sqrt(variance)`. If variance is zero (all features identical), we divide by zero.

**Solution**: Add small epsilon before taking square root.

```c
float var = var / (float)d_model + eps;  // eps = 1e-5
float rstd = 1.0f / sqrtf(var);
```

**Why 1e-5?**
- Small enough not to affect normal variance values (typically >> 1e-5)
- Large enough to prevent division by zero
- Standard across PyTorch, TensorFlow, JAX (all use 1e-5 default)

---

### 3. Gradient Clipping (Implicit)

**Problem**: Gradients can explode during training (especially early on with random weights).

**Current approach**: No explicit clipping implemented yet.

**Future consideration**: Add global norm clipping:
```c
// Compute global gradient norm
float total_norm = 0.0f;
for (all gradient buffers) {
    total_norm += gradient[i] * gradient[i];
}
total_norm = sqrtf(total_norm);

// Clip if exceeds threshold
float clip_value = 1.0f;  // Typical value
if (total_norm > clip_value) {
    float scale = clip_value / total_norm;
    for (all gradient buffers) {
        gradient[i] *= scale;
    }
}
```

---

## Softmax Backward: The Jacobian Derivation

### Forward Pass

Given input vector **x** = [x₁, x₂, ..., xₙ], softmax produces output **y**:

```
y[i] = exp(x[i]) / Σⱼ exp(x[j])
```

### Backward Pass Goal

Given gradient w.r.t. output `∂L/∂y` (denoted **dy**), compute gradient w.r.t. input `∂L/∂x` (denoted **dx**).

### The Jacobian Matrix

Softmax is a **vector-to-vector** function, so its derivative is a **Jacobian matrix**:

```
J[i,j] = ∂y[i] / ∂x[j]
```

**Two cases**:

**Case 1**: i = j (diagonal elements)
```
∂y[i]/∂x[i] = ∂/∂x[i] [exp(x[i]) / Σₖ exp(x[k])]

Using quotient rule:
  = [exp(x[i]) × Σₖ exp(x[k]) - exp(x[i]) × exp(x[i])] / (Σₖ exp(x[k]))²
  = [exp(x[i]) / Σₖ exp(x[k])] × [1 - exp(x[i]) / Σₖ exp(x[k])]
  = y[i] × (1 - y[i])
```

**Case 2**: i ≠ j (off-diagonal elements)
```
∂y[i]/∂x[j] = ∂/∂x[j] [exp(x[i]) / Σₖ exp(x[k])]

Since numerator doesn't depend on x[j], only denominator matters:
  = exp(x[i]) × ∂/∂x[j] [1 / Σₖ exp(x[k])]
  = exp(x[i]) × [-exp(x[j])] / (Σₖ exp(x[k]))²
  = -[exp(x[i]) / Σₖ exp(x[k])] × [exp(x[j]) / Σₖ exp(x[k])]
  = -y[i] × y[j]
```

**Combined Jacobian**:
```
J[i,j] = y[i] × (δ[i,j] - y[j])
```

where δ[i,j] is the Kronecker delta (1 if i=j, 0 otherwise).

### Applying Chain Rule

To get `dx` from `dy`, we multiply by the Jacobian:

```
dx[i] = Σⱼ J[i,j] × dy[j]
      = Σⱼ y[i] × (δ[i,j] - y[j]) × dy[j]
      = y[i] × Σⱼ (δ[i,j] - y[j]) × dy[j]
      = y[i] × (dy[i] - Σⱼ y[j] × dy[j])
      = y[i] × (dy[i] - dot(y, dy))
```

**Final formula** (implemented):
```
dx[i] = y[i] × (dy[i] - Σⱼ y[j] × dy[j])
```

### Implementation

**Code** (main.c:5791-5829):
```c
void backward_causal_softmax(TransformerModel *M,
                            size_t d_scores_offset,
                            size_t weights_copy_offset)
{
    float *d_scores_inout = M->memory_base + d_scores_offset;
    float *weights = M->memory_base + weights_copy_offset;

    int H = M->num_attention_heads;
    int T = M->context_window;

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < T; i++) {
            // Step 1: Compute dot(y, dy) = Σⱼ y[j] × dy[j]
            float dot_product = 0.0f;
            for (int j = 0; j <= i; j++) {  // Causal: only j <= i
                float w = ATTN_ACCESS(weights, h, i, j, T);       // y[j]
                float dw = ATTN_ACCESS(d_scores_inout, h, i, j, T);  // dy[j]
                dot_product += w * dw;
            }

            // Step 2: Apply formula dx[j] = y[j] × (dy[j] - dot_product)
            for (int j = 0; j <= i; j++) {
                float w = ATTN_ACCESS(weights, h, i, j, T);
                float dw = ATTN_ACCESS(d_scores_inout, h, i, j, T);

                ATTN_ACCESS(d_scores_inout, h, i, j, T) = w * (dw - dot_product);
            }

            // Step 3: Zero out upper triangle (causal mask)
            for (int j = i + 1; j < T; j++) {
                ATTN_ACCESS(d_scores_inout, h, i, j, T) = 0.0f;
            }
        }
    }
}
```

### Why This Works (Intuition)

Softmax is a **constrained normalization**: outputs must sum to 1.

```
Σᵢ y[i] = 1  (always)
```

Taking derivative of this constraint:
```
Σᵢ dy[i] = 0  (must be satisfied by valid gradients)
```

The formula `dx[i] = y[i] × (dy[i] - dot(y, dy))` ensures this:
```
Σᵢ dx[i] = Σᵢ y[i] × (dy[i] - dot(y, dy))
         = Σᵢ y[i] × dy[i] - Σᵢ y[i] × dot(y, dy)
         = dot(y, dy) - dot(y, dy) × Σᵢ y[i]
         = dot(y, dy) - dot(y, dy) × 1
         = 0  ✓
```

---

## Cross-Entropy Loss Gradient

### Forward: Cross-Entropy Loss

Given:
- Logits **z** (raw scores from model)
- Target class index **t**

**Loss** (negative log-likelihood):
```
L = -log(softmax(z)[t])
  = -log(exp(z[t]) / Σⱼ exp(z[j]))
  = -z[t] + log(Σⱼ exp(z[j]))
```

Using log-sum-exp stability:
```
L = -z[t] + log(Σⱼ exp(z[j] - max(z))) + max(z)
```

### Backward: Gradient w.r.t. Logits

**Derivation**:
```
∂L/∂z[i] = ∂/∂z[i] [-z[t] + log(Σⱼ exp(z[j]))]
```

**Case 1**: i = t (target class)
```
∂L/∂z[t] = -1 + ∂/∂z[t] [log(Σⱼ exp(z[j]))]
         = -1 + exp(z[t]) / Σⱼ exp(z[j])
         = -1 + softmax(z)[t]
         = softmax(z)[t] - 1
```

**Case 2**: i ≠ t (non-target classes)
```
∂L/∂z[i] = 0 + ∂/∂z[i] [log(Σⱼ exp(z[j]))]
         = exp(z[i]) / Σⱼ exp(z[j])
         = softmax(z)[i]
```

**Combined**:
```
∂L/∂z[i] = softmax(z)[i] - 1[i == t]
```

where `1[i == t]` is indicator function (1 if i equals target, 0 otherwise).

**Intuitive meaning**:
- Target class gradient = P(class) - 1 → pushes probability toward 1
- Other classes gradient = P(class) - 0 → pushes probability toward 0

### Implementation

**Code** (main.c:6213-6260):
```c
// Compute softmax (with log-sum-exp stability)
float max_logit = token_logits[0];
for (int v = 1; v < M->vocab_size; v++) {
    if (token_logits[v] > max_logit) {
        max_logit = token_logits[v];
    }
}

float sum_exp = 0.0f;
for (int v = 0; v < M->vocab_size; v++) {
    token_d_logits[v] = expf(token_logits[v] - max_logit);
    sum_exp += token_d_logits[v];
}

// Normalize to get softmax
for (int v = 0; v < M->vocab_size; v++) {
    token_d_logits[v] /= sum_exp;
}

// Compute loss
float prob = token_d_logits[target_token];
token_loss = -logf(prob + 1e-10f);  // Add epsilon to prevent log(0)

// Compute gradient: softmax - 1[target]
token_d_logits[target_token] -= 1.0f;  // Only target class gets -1
```

---

## LayerNorm Backward Derivation

### Forward: Layer Normalization

Given input **x** = [x₁, x₂, ..., xₐ] (D features):

```
μ = (1/D) × Σᵢ xᵢ                    (mean)
σ² = (1/D) × Σᵢ (xᵢ - μ)²           (variance)
x̂ᵢ = (xᵢ - μ) / sqrt(σ² + ε)       (normalize)
yᵢ = γᵢ × x̂ᵢ + βᵢ                    (scale and shift)
```

Parameters:
- **γ** (gamma): learnable scale
- **β** (beta): learnable shift
- **ε** (epsilon): numerical stability constant (1e-5)

### Backward: Gradients

Given `∂L/∂y` (denoted **dy**), compute:
- `∂L/∂x` (denoted **dx**)
- `∂L/∂γ` (denoted **dγ**)
- `∂L/∂β` (denoted **dβ**)

**Step 1: Gradient w.r.t. γ and β** (simple)
```
∂L/∂γᵢ = Σₜ (∂L/∂yₜᵢ) × (∂yₜᵢ/∂γᵢ)
       = Σₜ dyₜᵢ × x̂ₜᵢ

∂L/∂βᵢ = Σₜ (∂L/∂yₜᵢ) × (∂yₜᵢ/∂βᵢ)
       = Σₜ dyₜᵢ × 1
       = Σₜ dyₜᵢ
```

where t sums over tokens (batch dimension).

**Step 2: Gradient w.r.t. normalized values**
```
∂L/∂x̂ᵢ = ∂L/∂yᵢ × ∂yᵢ/∂x̂ᵢ
       = dyᵢ × γᵢ
```

**Step 3: Gradient w.r.t. input** (complex due to mean/variance coupling)

All inputs xⱼ affect the output xᵢ through:
1. Direct path: xᵢ appears in numerator
2. Mean path: xᵢ affects μ, which affects all outputs
3. Variance path: xᵢ affects σ², which affects all outputs

**Full derivation** (see Ioffe & Szegedy, 2015):
```
Let:
  rstd = 1 / sqrt(σ² + ε)  (reciprocal standard deviation)

∂L/∂xᵢ = (rstd / D) × [D × dx̂ᵢ - Σⱼ dx̂ⱼ - x̂ᵢ × Σⱼ (dx̂ⱼ × x̂ⱼ)]
```

where:
- `dx̂ᵢ = dyᵢ × γᵢ` (gradient w.r.t. normalized values)
- First term: direct gradient
- Second term: correction for mean gradient
- Third term: correction for variance gradient

### Implementation

**Code** (main.c:5496-5620):
```c
void backward_layernorm(TransformerModel *M,
                       size_t d_output_offset,    // dy
                       size_t input_copy_offset,  // x (cached)
                       size_t gamma_copy_offset,  // γ (cached)
                       size_t beta_copy_offset,   // β (cached)
                       size_t mean_copy_offset,   // μ (cached)
                       size_t rstd_copy_offset,   // rstd (cached)
                       size_t d_input_offset,     // dx (output)
                       size_t d_gamma_offset,     // dγ (output)
                       size_t d_beta_offset)      // dβ (output)
{
    // ... initialization ...

    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        float mu = mean[t];
        float rstd = rstd_cached[t];

        // Compute dx̂ = dy × γ (element-wise)
        float dx_norm[D];  // Stack allocation for small D
        for (int d = 0; d < D; d++) {
            dx_norm[d] = dy[t * aligned_D + d] * gamma[d];
        }

        // Compute Σ dx̂ and Σ (x̂ × dx̂)
        float sum_dx_norm = 0.0f;
        float sum_x_norm_dx_norm = 0.0f;

        for (int d = 0; d < D; d++) {
            float x_norm = (x[t * aligned_D + d] - mu) * rstd;
            sum_dx_norm += dx_norm[d];
            sum_x_norm_dx_norm += x_norm * dx_norm[d];
        }

        // Apply the full gradient formula
        float scale = rstd / (float)D;
        for (int d = 0; d < D; d++) {
            float x_norm = (x[t * aligned_D + d] - mu) * rstd;

            dx[t * aligned_D + d] = scale * (
                (float)D * dx_norm[d]         // Direct term
                - sum_dx_norm                  // Mean correction
                - x_norm * sum_x_norm_dx_norm  // Variance correction
            );
        }
    }

    // Accumulate dγ and dβ across tokens
    for (int t = 0; t < T; t++) {
        float mu = mean[t];
        float rstd = rstd_cached[t];

        for (int d = 0; d < D; d++) {
            float x_norm = (x[t * aligned_D + d] - mu) * rstd;

            #pragma omp atomic
            d_gamma[d] += dy[t * aligned_D + d] * x_norm;

            #pragma omp atomic
            d_beta[d] += dy[t * aligned_D + d];
        }
    }
}
```

**Why this is complex**: LayerNorm creates **dependencies between all features** through mean and variance. Unlike element-wise operations, changing one input affects all outputs.

---

## GELU Activation Backward

### Forward: Gaussian Error Linear Unit

GELU is a smooth approximation to ReLU:

```
GELU(x) = x × Φ(x)
```

where Φ(x) is the cumulative distribution function (CDF) of standard normal distribution:
```
Φ(x) = (1/2) × [1 + erf(x / sqrt(2))]
```

**Approximation used** (faster than exact erf):
```
GELU(x) ≈ 0.5 × x × [1 + tanh(sqrt(2/π) × (x + 0.044715 × x³))]
```

### Backward: Derivative

**Exact derivative**:
```
d(GELU)/dx = Φ(x) + x × φ(x)
```

where φ(x) = (1/sqrt(2π)) × exp(-x²/2) is the PDF of standard normal.

**Approximation derivative** (from tanh formula):
```
Let:
  a = sqrt(2/π) ≈ 0.797885
  b = 0.044715
  z = a × (x + b × x³)

GELU(x) ≈ 0.5 × x × (1 + tanh(z))

d(GELU)/dx ≈ 0.5 × (1 + tanh(z)) + 0.5 × x × sech²(z) × dz/dx
```

where:
```
dz/dx = a × (1 + 3 × b × x²)
sech²(z) = 1 - tanh²(z)
```

### Implementation

**Code** (main.c:5455-5493):
```c
void backward_gelu_fast(TransformerModel *M,
                       size_t d_output_offset,
                       size_t input_copy_offset,
                       size_t d_input_offset)
{
    float *d_output = M->memory_base + d_output_offset;
    float *input = M->memory_base + input_copy_offset;
    float *d_input = M->memory_base + d_input_offset;

    size_t total_elements = M->context_window * M->aligned_embed_dim * 4;

    #pragma omp parallel for num_threads(M->num_cores)
    for (size_t i = 0; i < total_elements; i++) {
        float x = input[i];

        // Constants for GELU approximation
        const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/π)
        const float coeff = 0.044715f;

        // Compute intermediate values
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanhf(tanh_arg);

        // Compute d(tanh_arg)/dx
        float dtanh_arg_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);

        // Compute sech²(tanh_arg) = 1 - tanh²(tanh_arg)
        float sech2 = 1.0f - tanh_val * tanh_val;

        // Compute GELU derivative
        float gelu_derivative = 0.5f * (1.0f + tanh_val)
                              + 0.5f * x * sech2 * dtanh_arg_dx;

        // Apply chain rule
        d_input[i] = d_output[i] * gelu_derivative;
    }
}
```

---

## Attention Mechanism Gradients

### Forward: Scaled Dot-Product Attention

```
Q, K, V = linear_q(x), linear_k(x), linear_v(x)
scores = (Q @ K^T) / sqrt(d_k)
attention_weights = softmax(scores)  [with causal mask]
output = attention_weights @ V
```

### Backward: Chain of Gradients

Given `∂L/∂output`, compute gradients for Q, K, V.

**Step 1: Gradient through attention × V**
```
output = attention_weights @ V

∂L/∂attention_weights = ∂L/∂output @ V^T
∂L/∂V = attention_weights^T @ ∂L/∂output
```

**Step 2: Gradient through softmax**

Uses Jacobian formula (see [Softmax Backward](#softmax-backward-the-jacobian-derivation)):
```
∂L/∂scores = attention_weights ⊙ (∂L/∂attention_weights - dot(attention_weights, ∂L/∂attention_weights))
```

where ⊙ is element-wise multiplication.

**Step 3: Gradient through Q @ K^T**
```
scores = (Q @ K^T) / sqrt(d_k)

∂L/∂Q = (∂L/∂scores @ K) / sqrt(d_k)
∂L/∂K = (∂L/∂scores^T @ Q) / sqrt(d_k)
```

**Step 4: Gradient through linear projections**

Standard matrix multiplication gradients (see [Matrix Multiplication](#matrix-multiplication-gradients)).

---

## Matrix Multiplication Gradients

### Forward: Linear Layer

```
Y = X @ W + b
```

Shapes:
- X: [M × K] (input)
- W: [K × N] (weights)
- b: [N] (bias)
- Y: [M × N] (output)

### Backward: Gradients

Given `∂L/∂Y` (denoted **dY**), compute:

**1. Gradient w.r.t. weights**:
```
∂L/∂W = X^T @ (∂L/∂Y)
```

Shape: [K × M] @ [M × N] = [K × N] ✓

**2. Gradient w.r.t. bias**:
```
∂L/∂b = sum(∂L/∂Y, axis=0)
```

Sum over batch dimension. Shape: [M × N] → [N] ✓

**3. Gradient w.r.t. input**:
```
∂L/∂X = (∂L/∂Y) @ W^T
```

Shape: [M × N] @ [N × K] = [M × K] ✓

### Why These Formulas Work

**Derivation for ∂L/∂W**:

Element-wise:
```
Y[i,j] = Σₖ X[i,k] × W[k,j] + b[j]

∂L/∂W[k,j] = Σᵢ (∂L/∂Y[i,j]) × (∂Y[i,j]/∂W[k,j])
           = Σᵢ (∂L/∂Y[i,j]) × X[i,k]
           = Σᵢ X^T[k,i] × (∂L/∂Y[i,j])
           = (X^T @ ∂L/∂Y)[k,j]
```

Matrix form: `∂L/∂W = X^T @ (∂L/∂Y)`

**Derivation for ∂L/∂X**:

Element-wise:
```
∂L/∂X[i,k] = Σⱼ (∂L/∂Y[i,j]) × (∂Y[i,j]/∂X[i,k])
           = Σⱼ (∂L/∂Y[i,j]) × W[k,j]
           = Σⱼ (∂L/∂Y[i,j]) × W^T[j,k]
           = (∂L/∂Y @ W^T)[i,k]
```

Matrix form: `∂L/∂X = (∂L/∂Y) @ W^T`

---

## Numerical Precision Considerations

### Float32 Precision Limits

IEEE 754 single precision (float32):
- **Mantissa**: 23 bits (~7 decimal digits precision)
- **Exponent**: 8 bits (range: 10^-38 to 10^38)
- **Epsilon**: ~1.19e-7 (smallest ε where 1+ε ≠ 1)

**Implications**:

1. **Catastrophic cancellation**: Subtracting nearly equal numbers loses precision.
   ```
   Example: (1.0000001 - 1.0) = 1e-7 (only 1 significant digit left!)
   ```

2. **Accumulation errors**: Summing many small values can lose precision.
   ```
   Sum of 10^7 values of 1e-7 ≈ 1.0, but accumulated error significant
   ```

3. **Gradient underflow**: Very small gradients (< 1e-38) become zero.
   ```
   Solution: Gradient scaling or mixed precision training
   ```

### When to Use Float64

**Consider float64 (double) for**:
- Numerical integration (long-term accumulation)
- Matrix inversions (condition number issues)
- Very deep networks (gradient accumulation over many layers)

**Current implementation**: Uses float32 throughout (standard for ML).

**Future**: Could add mixed-precision support (FP16 forward, FP32 backward).

---

## Summary of Numerical Tricks

| Technique | Purpose | Location |
|-----------|---------|----------|
| **Log-sum-exp** | Prevent overflow/underflow in softmax | main.c:3154, 6231 |
| **Epsilon in LayerNorm** | Prevent division by zero | main.c:1560 |
| **Subtract max before exp** | Numerical stability | main.c:3155, 6232 |
| **Causal masking** | Zero out invalid attention positions | main.c:3109, 5823 |
| **In-place operations** | Memory efficiency (gradients) | main.c:5815-5820 |
| **Atomic updates** | Thread-safe gradient accumulation | main.c:5612-5615 |

---

## References

1. **Softmax Jacobian**: Goodfellow et al., "Deep Learning", Chapter 6.2.2.3
2. **LayerNorm Backward**: Ba et al., "Layer Normalization", 2016
3. **GELU**: Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", 2016
4. **Attention**: Vaswani et al., "Attention Is All You Need", 2017
5. **Numerical Stability**: Press et al., "Numerical Recipes", Chapter 2

---

*Generated with Claude Code - Complete mathematical documentation of C-Transformer backpropagation*
