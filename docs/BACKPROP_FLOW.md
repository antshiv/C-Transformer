# Backward Propagation Flow - Complete Documentation

## Overview

This document provides a complete step-by-step breakdown of the backward propagation (backprop) implementation in the C-Transformer. The backprop computes gradients for all model parameters, enabling training via gradient descent.

**Purpose**: Understand exactly how gradients flow backward through the GPT-2 architecture to update model weights during training.

---

## Table of Contents

1. [High-Level Training Flow](#high-level-training-flow)
2. [Backward Pass Entry Point](#backward-pass-entry-point)
3. [Layer-by-Layer Backward Flow](#layer-by-layer-backward-flow)
4. [Gradient Storage Architecture](#gradient-storage-architecture)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Memory Layout & Data Flow](#memory-layout--data-flow)

---

## High-Level Training Flow

### training_step() - Main Training Entry Point

**Location**: `main.c:6475`

```
INPUT: (input_tokens, target_tokens, learning_rate)
  ↓
┌─────────────────────────────────────────┐
│ 1. FORWARD PASS                        │
├─────────────────────────────────────────┤
│ • embed_tokens()                        │
│ • transformer_layer_forward() × L       │
│ • final_layernorm()                     │
│ • compute_logits()                      │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 2. BACKWARD PASS (THIS DOCUMENT)       │
├─────────────────────────────────────────┤
│ • zero_gradients()                      │
│ • cache_forward_activations()           │
│ • compute_cross_entropy_loss()          │
│ • backward_lm_head()                    │
│ • backward_final_layernorm()            │
│ • backward_transformer_layer() × L      │
│ • backward_embedding_layer()            │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 3. WEIGHT UPDATE                        │
├─────────────────────────────────────────┤
│ • update_all_weights_sgd()              │
│   θ ← θ - lr × ∇θ                      │
└─────────────────────────────────────────┘
  ↓
OUTPUT: loss
```

---

## Backward Pass Entry Point

### Step 1: zero_gradients()

**Location**: `main.c:4635`

**Purpose**: Reset all gradient accumulators to zero before computing new gradients.

**Why necessary**: Gradients accumulate across operations. Without zeroing, old gradients would contaminate new calculations.

```c
void zero_gradients(TransformerModel *M) {
    size_t total_gradient_bytes = M->gradients.total_gradient_floats * sizeof(float);
    memset(M->memory_base + M->gradients.backprop_base, 0, total_gradient_bytes);
}
```

**Memory cleared**:
- All weight gradients (d_embed_weights, d_ln_weights, d_fc_weights, etc.)
- All activation gradients (d_residual, d_attention, d_mlp, etc.)
- Total size: ~0.80 GB for standard model

---

### Step 2: cache_forward_activations()

**Location**: `main.c:4645`

**Purpose**: Copy forward pass results to gradient storage so backward pass can access them.

**Why necessary**: Backward pass needs forward activations to compute gradients, but forward buffers get overwritten. We cache:
- logits → logits_copy
- final_output → final_output_copy
- layernorm means/rstd → copies
- Layer outputs → copies

**Example**: To compute `d_weight = input^T @ d_output`, we need the original `input` from forward pass.

```c
void cache_forward_activations(TransformerModel *M) {
    // Copy final layer outputs
    memcpy(M->memory_base + M->gradients.logits_copy_offset,
           M->memory_base + M->logits_offset,
           M->context_window * M->vocab_size * sizeof(float));

    memcpy(M->memory_base + M->gradients.final_output_copy_offset,
           M->memory_base + M->final_output_offset,
           M->context_window * M->aligned_embed_dim * sizeof(float));

    // ... (more copies for layernorm stats, layer outputs, etc.)
}
```

---

### Step 3: compute_cross_entropy_loss()

**Location**: Not shown, but computes loss and initial gradient

**Purpose**: Compute training loss and initialize gradient flow.

**Math**:
```
Loss = -log(P(target_token))
Initial gradient: d_logits[target_token] = P(target_token) - 1
                 d_logits[other_tokens]  = P(other_tokens)
```

This creates the initial gradient signal that propagates backward through the entire network.

---

## Layer-by-Layer Backward Flow

### Backward Flow Sequence

The backward pass proceeds in **reverse order** of the forward pass:

```
FORWARD DIRECTION (→):
embeddings → layer_0 → layer_1 → ... → layer_L → final_LN → logits

BACKWARD DIRECTION (←):
d_logits ← d_final_LN ← ... ← d_layer_1 ← d_layer_0 ← d_embeddings
```

### Step 4: backward_lm_head()

**Location**: `main.c:6000`

**Purpose**: Compute gradients for the language model head (final projection to vocabulary).

**Architecture Note**: GPT-2 uses **weight tying** - the embedding matrix is shared with the LM head.

**What happens**:
```
Forward:  logits = final_output @ embed_weights^T
Backward: d_final_output = d_logits @ embed_weights
          d_embed_weights += final_output^T @ d_logits
```

**Code structure**:
```c
void backward_lm_head(TransformerModel *M) {
    // For each token position:
    //   1. Compute gradient w.r.t. final_output
    //   2. Accumulate gradient w.r.t. embedding weights (weight tying!)

    // Gradient flows to:
    // - d_final_output_offset (shape: T × D)
    // - d_embed_weights_offset (shape: V × D) [accumulated]
}
```

**Key insight**: Because of weight tying, the embedding weights receive gradients from:
1. This backward_lm_head() function (from predicting next tokens)
2. backward_embedding_layer() (from encoding input tokens)

These gradients are **accumulated** (summed), not overwritten.

---

### Step 5: backward_final_layernorm()

**Location**: `main.c:5094`

**Purpose**: Backward through the final LayerNorm before the LM head.

**Forward LayerNorm**:
```
mean = (1/D) × Σ(x)
var = (1/D) × Σ((x - mean)²)
rstd = 1 / sqrt(var + ε)
y = γ × (x - mean) × rstd + β
```

**Backward LayerNorm** (complex chain rule):
```
d_γ = Σ((x - mean) × rstd × d_y)      [sum over tokens & features]
d_β = Σ(d_y)                           [sum over tokens & features]
d_x = (1/D) × rstd × [D × d_y - Σ(d_y) - (x - mean) × Σ(d_y × (x - mean))] × γ
```

**Why complex**: Normalization creates dependencies between all features, so gradient must account for how each input affects all outputs through mean/variance.

**Code outputs**:
- `d_final_ln_input_offset` → gradient flows to last transformer layer
- `d_final_ln_gamma_offset` → gradient for γ (scale parameter)
- `d_final_ln_beta_offset` → gradient for β (shift parameter)

---

### Step 6: backward_transformer_layer()

**Location**: `main.c:6071`

**Purpose**: Backward through one complete transformer layer (attention + MLP + residuals).

**Layer Architecture**:
```
        input
          ↓
    ┌─────┴─────┐
    │  LayerNorm 1
    │     ↓
    │  Attention
    │     ↓
    └──→ Add ─────→ residual1
          ↓
    ┌─────┴─────┐
    │  LayerNorm 2
    │     ↓
    │    MLP
    │     ↓
    └──→ Add ─────→ output
```

**Backward Flow Through Layer** (reverse order):

#### 6a. Second Residual Connection
```c
backward_residual_connection(M,
    LG->d_residual2_offset,      // input gradient
    LG->d_residual1_offset,      // output: gradient to first residual
    LG->d_mlp_output_offset);    // output: gradient to MLP output
```

**Math**: `residual2 = residual1 + mlp_output`
- Backward: `d_residual1 = d_residual2` (copy)
- Backward: `d_mlp_output = d_residual2` (copy)

**Key insight**: Residual connections just copy gradients - no computation needed!

#### 6b. MLP Backward (3 steps)

##### Backward FC2 (Final MLP projection)
```c
backward_fc2(M,
    LG->d_mlp_output_offset,      // input: ∂L/∂mlp_output
    LG->fc2_input_copy_offset,    // cached: fc2_input from forward
    L->fc2_weight_offset,         // weights
    L->fc2_bias_offset,           // bias
    LG->d_fc2_input_offset,       // output: ∂L/∂fc2_input
    LG->d_fc2_weights_offset,     // output: ∂L/∂W_fc2
    LG->d_fc2_bias_offset);       // output: ∂L/∂b_fc2
```

**Math**: `y = xW + b` where `x: [T × 4D], W: [4D × D], y: [T × D]`
- `d_W = x^T @ d_y` (shape: [4D × D])
- `d_b = sum(d_y, axis=0)` (shape: [D])
- `d_x = d_y @ W^T` (shape: [T × 4D])

##### Backward GELU (Activation)
```c
backward_gelu(M,
    LG->d_fc2_input_offset,       // input: ∂L/∂gelu_output
    LG->fc1_output_copy_offset,   // cached: gelu_input from forward
    LG->d_fc1_output_offset);     // output: ∂L/∂gelu_input
```

**Math**: `GELU(x) = x × Φ(x)` where `Φ` is standard normal CDF
- `d_x = d_y × [Φ(x) + x × φ(x)]` where `φ` is PDF

**Approximation used**:
```c
float gelu_derivative = 0.5 * (1.0 + tanh(0.797885 * (x + 0.044715 * x³)))
                      + x * derivative_of_that_tanh_term
```

##### Backward FC1 (First MLP projection)
```c
backward_fc1(M,
    LG->d_fc1_output_offset,      // input: ∂L/∂fc1_output
    LG->ln2_output_copy_offset,   // cached: fc1_input from forward
    L->fc1_weight_offset,         // weights
    L->fc1_bias_offset,           // bias
    LG->d_ln2_output_offset,      // output: ∂L/∂ln2_output
    LG->d_fc1_weights_offset,     // output: ∂L/∂W_fc1
    LG->d_fc1_bias_offset);       // output: ∂L/∂b_fc1
```

**Math**: `y = xW + b` where `x: [T × D], W: [D × 4D], y: [T × 4D]`
- `d_W = x^T @ d_y` (shape: [D × 4D])
- `d_b = sum(d_y, axis=0)` (shape: [4D])
- `d_x = d_y @ W^T` (shape: [T × D])

#### 6c. Second LayerNorm
```c
backward_layernorm(M,
    LG->d_ln2_output_offset,      // input: ∂L/∂ln2_output
    LG->ln2_input_copy_offset,    // cached: ln2_input
    LG->ln2_gamma_copy_offset,    // cached: γ
    LG->ln2_beta_copy_offset,     // cached: β
    LG->ln2_mean_copy_offset,     // cached: mean
    LG->ln2_rstd_copy_offset,     // cached: rstd
    LG->d_ln2_input_offset,       // output: ∂L/∂ln2_input
    LG->d_ln2_gamma_offset,       // output: ∂L/∂γ
    LG->d_ln2_beta_offset);       // output: ∂L/∂β
```

Same math as final LayerNorm (see Step 5).

#### 6d. Gradient Accumulation
```c
add_gradient(M, LG->d_ln2_input_offset, LG->d_residual1_offset);
```

**Why**: The residual1 output goes to **two places**:
1. As input to LayerNorm2 → MLP path
2. As input to second residual connection

So gradients from both paths must be **summed** (chain rule).

#### 6e. First Residual Connection
```c
backward_residual_connection(M,
    LG->d_residual1_offset,         // input: accumulated gradient
    LG->d_ln1_input_offset,         // output: gradient to layer input
    LG->d_attention_output_offset); // output: gradient to attention
```

Same as second residual - just copies gradients.

#### 6f. Attention Projection
```c
backward_attention_projection(M,
    LG->d_attention_output_offset,  // input: ∂L/∂attention_output
    LG->attention_output_copy_offset, // cached forward result
    L->proj_weight_offset,          // weights
    L->proj_bias_offset,            // bias
    LG->d_attention_token_offset,   // output: ∂L/∂attention (per-token)
    LG->d_attention_head_offset,    // output: ∂L/∂attention (per-head)
    LG->d_proj_weights_offset,      // output: ∂L/∂W_proj
    LG->d_proj_bias_offset);        // output: ∂L/∂b_proj
```

**Forward**: Multi-head outputs are concatenated then projected:
```
attention_concat: [T × (H × D_h)]  # H heads, D_h = head_dim
output = attention_concat @ W_proj + b_proj
```

**Backward**: Standard linear layer backward, but output must be split back into per-head gradients.

#### 6g. Attention Mechanism Backward

The attention mechanism has multiple steps that must be reversed:

##### Backward Attention × Values
```c
backward_attention_weighted_values(M,
    LG->d_attention_head_offset,    // input: ∂L/∂(attention × V)
    LG->attention_weights_copy_offset, // cached: attention weights
    LG->v_output_copy_offset,       // cached: V
    LG->d_attention_weights_offset, // output: ∂L/∂attention_weights
    LG->d_v_output_offset);         // output: ∂L/∂V
```

**Forward**: `output = attention_weights @ V`
- `attention_weights: [H, T, T]` (per-head attention)
- `V: [H, T, D_h]` (values per head)
- `output: [H, T, D_h]`

**Backward**:
- `d_attention_weights = d_output @ V^T` (shape: [H, T, T])
- `d_V = attention_weights^T @ d_output` (shape: [H, T, D_h])

##### Backward Causal Softmax
```c
backward_causal_softmax(M,
    LG->d_attention_weights_offset,   // input/output: ∂L/∂softmax
    LG->attention_weights_copy_offset); // cached: softmax output
```

**Forward**: `softmax(x)_i = exp(x_i) / Σ(exp(x_j))`

**Backward** (with Jacobian):
```
d_x_i = softmax_i × (d_y_i - Σ(d_y_j × softmax_j))
```

**Key insight**: Softmax gradient depends on **all** output gradients, not just the gradient at position i.

##### Backward Q @ K^T
```c
backward_qk_matmul(M,
    LG->d_attention_weights_offset, // input: ∂L/∂(Q @ K^T)
    LG->q_output_copy_offset,       // cached: Q
    LG->k_output_copy_offset,       // cached: K
    LG->d_q_output_offset,          // output: ∂L/∂Q
    LG->d_k_output_offset);         // output: ∂L/∂K
```

**Forward**: `scores = (Q @ K^T) / sqrt(D_h)`
- `Q: [H, T, D_h]`
- `K: [H, T, D_h]`
- `scores: [H, T, T]`

**Backward**:
- `d_Q = d_scores @ K` (shape: [H, T, D_h])
- `d_K = d_scores^T @ Q` (shape: [H, T, D_h])

**Implementation note**: Must account for scaling factor `1/sqrt(D_h)`.

#### 6h. Q, K, V Linear Projections
```c
// Backward through V projection
backward_linear(M,
    LG->d_v_output_offset,          // input: ∂L/∂V
    LG->ln1_output_copy_offset,     // cached: input to V projection
    L->v_weight_offset,             // weights
    L->v_bias_offset,               // bias
    LG->d_ln1_output_v_offset,      // output: ∂L/∂(input to V)
    LG->d_v_weights_offset,         // output: ∂L/∂W_v
    LG->d_v_bias_offset);           // output: ∂L/∂b_v

// Backward through K projection
backward_linear(M, ...);  // Same structure

// Backward through Q projection
backward_linear(M, ...);  // Same structure
```

**Math**: Each is a standard linear layer `y = xW + b`

**Important**: All three (Q, K, V) receive the **same input** (ln1_output), so their input gradients must be **summed**:

```c
// Accumulate gradients from Q, K, V paths
add_gradient(M, LG->d_ln1_output_q_offset, LG->d_ln1_output_offset);
add_gradient(M, LG->d_ln1_output_k_offset, LG->d_ln1_output_offset);
add_gradient(M, LG->d_ln1_output_v_offset, LG->d_ln1_output_offset);
```

#### 6i. First LayerNorm
```c
backward_layernorm(M,
    LG->d_ln1_output_offset,        // input: ∂L/∂ln1_output
    LG->ln1_input_copy_offset,      // cached: ln1_input
    LG->ln1_gamma_copy_offset,      // cached: γ
    LG->ln1_beta_copy_offset,       // cached: β
    LG->ln1_mean_copy_offset,       // cached: mean
    LG->ln1_rstd_copy_offset,       // cached: rstd
    LG->d_ln1_input_offset,         // output: ∂L/∂ln1_input
    LG->d_ln1_gamma_offset,         // output: ∂L/∂γ
    LG->d_ln1_beta_offset);         // output: ∂L/∂β
```

Same LayerNorm backward as before.

**Final output**: `d_ln1_input_offset` becomes the input gradient for the **previous layer** (or embeddings if layer 0).

---

### Step 7: backward_embedding_layer()

**Location**: `main.c:4959`

**Purpose**: Compute gradients for token and positional embeddings.

**Forward**:
```
embedded = token_embeddings[token_ids] + positional_embeddings[positions]
```

**Backward**:
```
d_token_embeddings[token_id] += d_embedded  (scatter operation)
d_positional_embeddings[position] += d_embedded
```

**Code structure**:
```c
void backward_embedding_layer(TransformerModel *M) {
    // Input gradient comes from first transformer layer
    float *d_embedded = M->memory_base + M->gradients.layers[0].d_ln1_input_offset;

    // For each token position:
    for (int t = 0; t < M->context_window; t++) {
        int token_id = input_tokens[t];

        // Accumulate gradient to token embedding (scatter)
        float *d_token_emb = M->memory_base + M->gradients.d_embed_weights_offset
                           + token_id * M->aligned_embed_dim;
        for (int d = 0; d < M->embed_dim; d++) {
            d_token_emb[d] += d_embedded[t * M->aligned_embed_dim + d];
        }

        // Accumulate gradient to positional embedding
        float *d_pos_emb = M->memory_base + M->gradients.d_pos_embed_offset
                         + t * M->aligned_embed_dim;
        for (int d = 0; d < M->embed_dim; d++) {
            d_pos_emb[d] += d_embedded[t * M->aligned_embed_dim + d];
        }
    }
}
```

**Key insight**: Token embeddings use **scatter** operation - each token ID may appear multiple times in the sequence, so gradients accumulate to the same embedding row.

**Remember**: Token embeddings already received gradients from `backward_lm_head()` (weight tying), so this **adds** more gradients.

---

## Gradient Storage Architecture

### Memory Layout

All gradients are stored in a contiguous memory region starting at `M->gradients.backprop_base`:

```
backprop_base
  ↓
┌─────────────────────────────────────────┐
│ Global Weight Gradients                │
├─────────────────────────────────────────┤
│ • d_embed_weights (V × D)              │
│ • d_pos_embed (T × D)                  │
│ • d_final_ln_gamma (D)                 │
│ • d_final_ln_beta (D)                  │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Global Activation Gradients            │
├─────────────────────────────────────────┤
│ • d_final_output (T × D)               │
│ • d_final_ln_input (T × D)             │
│ • d_logits (T × V)                     │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Layer 0 Gradients                      │
├─────────────────────────────────────────┤
│ • Weight gradients (W_q, W_k, W_v, etc)│
│ • Activation gradients (cached copies) │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Layer 1 Gradients                      │
├─────────────────────────────────────────┤
│ ... (same structure as Layer 0)        │
└─────────────────────────────────────────┘
  ↓
  ... (more layers)
  ↓
┌─────────────────────────────────────────┐
│ Forward Activation Copies              │
├─────────────────────────────────────────┤
│ • logits_copy (T × V)                  │
│ • final_output_copy (T × D)            │
│ • layer_outputs_copy (per layer)       │
└─────────────────────────────────────────┘
```

**Total gradient memory**: ~0.80 GB for standard model (L=4, D=256, T=1024)

---

## Mathematical Foundations

### Chain Rule

All backpropagation is based on the chain rule:

```
If z = f(g(x)), then dz/dx = (dz/dg) × (dg/dx)
```

**Example**: Linear layer `y = xW + b`
```
Forward:  x → [multiply by W] → [add b] → y
Backward: d_x ← [multiply by W^T] ← d_y
          d_W ← [x^T @ d_y]
          d_b ← [sum(d_y, axis=0)]
```

### Gradient Accumulation

When a value is used **multiple times** in forward pass, gradients from all uses must be **summed**:

**Example**: Residual connection
```
Forward:  y = x + f(x)
Backward: d_x = d_y (from direct path) + d_f (from function path)
                ↑             ↑
          gradient splits into two paths, then sums
```

**Implementation**: Use `add_gradient()` to accumulate:
```c
void add_gradient(TransformerModel *M, size_t source_offset, size_t dest_offset) {
    float *src = M->memory_base + source_offset;
    float *dst = M->memory_base + dest_offset;
    size_t size = M->context_window * M->aligned_embed_dim;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        dst[i] += src[i];  // Accumulate, don't overwrite!
    }
}
```

### Matrix Calculus Refresher

**1. Linear layer**: `y = xW + b`
- `d_W = x^T @ d_y`
- `d_b = sum(d_y, axis=0)`
- `d_x = d_y @ W^T`

**2. Matrix multiplication**: `C = A @ B`
- `d_A = d_C @ B^T`
- `d_B = A^T @ d_C`

**3. Element-wise operations**: `y = f(x)` (applied element-wise)
- `d_x = d_y ⊙ f'(x)` where `⊙` is element-wise multiplication

**4. Normalization** (LayerNorm, BatchNorm):
- Complex because mean/variance create dependencies between all elements
- Must track how each input affects mean, variance, and normalized output
- See LayerNorm backward derivation in literature for full math

---

## Memory Layout & Data Flow

### Gradient Flow Diagram

```
FORWARD PASS (left to right):
input → embed → ln1 → attn → proj → add → ln2 → mlp → add → ln → logits
                                     ↑                    ↑
                                  residual             residual

BACKWARD PASS (right to left):
d_embed ← d_ln1 ← d_attn ← d_proj ← d_add ← d_ln2 ← d_mlp ← d_add ← d_ln ← d_logits
                                      ↑                       ↑
                                   gradient                gradient
                                   accumulates             accumulates
```

### Gradient Buffer Sizes

For model with `L=4, D=256, T=1024, V=50257`:

| Gradient Buffer | Shape | Size (MB) |
|----------------|--------|-----------|
| `d_embed_weights` | V × D | 48.9 |
| `d_pos_embed` | T × D | 1.0 |
| `d_logits` | T × V | 195.6 |
| `d_final_output` | T × D | 1.0 |
| Per-layer gradients | (varies) | ~130 |
| **Total** | | **~800** |

### Computation Complexity

For one training step:

| Operation | FLOPs | Fraction |
|-----------|-------|----------|
| Forward pass | ~80M | 33% |
| Backward pass | ~160M | 67% |
| **Total** | **~240M** | **100%** |

**Why backward is 2× forward**: Must compute both activation gradients AND weight gradients.

---

## Summary

### Complete Backward Flow (Top to Bottom)

1. **Initialize**: Zero gradients, cache forward activations, compute loss
2. **LM Head**: Compute d_final_output, accumulate d_embed_weights (weight tying)
3. **Final LayerNorm**: Compute d_final_ln_input, d_gamma, d_beta
4. **For each layer (L-1 down to 0)**:
   - Backward through 2nd residual (copy gradients)
   - Backward through MLP (FC2 → GELU → FC1)
   - Backward through 2nd LayerNorm
   - Accumulate gradients from residual paths
   - Backward through 1st residual (copy gradients)
   - Backward through attention projection
   - Backward through attention mechanism (attn×V → softmax → Q@K^T)
   - Backward through Q, K, V linear projections
   - Backward through 1st LayerNorm
5. **Embeddings**: Scatter gradients to token embeddings, accumulate to positional embeddings

### Key Implementation Patterns

✅ **Zero gradients first** - avoid contamination from previous steps
✅ **Cache forward activations** - needed for backward computations
✅ **Accumulate, don't overwrite** - when values are used multiple times
✅ **Reverse order** - backward pass mirrors forward pass in reverse
✅ **Matrix transposes** - key to computing gradients correctly
✅ **Numerical stability** - use same tricks as forward (e.g., log-sum-exp)

---

## Next Steps

To fully understand this implementation:

1. **Read one backward function at a time** - start with simple ones (residual, linear)
2. **Verify math** - check matrix dimensions match, chain rule applied correctly
3. **Trace a single gradient** - follow one weight's gradient through entire backward pass
4. **Compare with papers** - see how this matches transformer literature
5. **Visualize gradients** - export and plot to verify reasonable values (not too large/small)

**Recommended order for studying code**:
1. `backward_residual_connection` (simplest - just copies)
2. `backward_linear` (fundamental building block)
3. `backward_layernorm` (complex but well-documented)
4. `backward_fc1`, `backward_fc2` (uses backward_linear)
5. `backward_attention_*` (attention mechanism specifics)
6. `backward_transformer_layer` (puts it all together)

---

*Generated with Claude Code - Complete documentation of C-Transformer backpropagation*
