# C‑Transformer Validation Pipeline

This file documents the validation methods we used to debug and harden the GPT‑2
forward path. You can reuse the same approach when you port other models
(`qwen`, etc.) into this C engine.

The core idea: **prove equivalence to a known‑good PyTorch/HuggingFace model
step by step**, from weights → embeddings → layer internals → attention →
projection/residuals → final logits.

All scripts below assume:

- C executable: `./main`
- Weights file: `gpt2_bump.weights`
- HF model: `GPT2LMHeadModel.from_pretrained("gpt2")`

You can change these for other models once you have a matching weight exporter.

---

## 0. One‑shot orchestration

For day‑to‑day checks, run everything with:

```bash
./validate_all.sh "Hello World" gpt2_bump.weights ./main 20 0
```

This runs:

1. `debug_weights.py` – weight layout sanity check
2. `validate_embeddings.py` – embeddings vs HF
3. `validate_layers.py` – per‑layer output vs HF for a chosen layer
4. `validate_inference.sh` – final hidden state + logits vs HF

Use this as your quick “did I break inference?” test.

The rest of this document describes the **deeper tools** you can use when the
pipeline shows a mismatch, or when you are bringing up a new model.

---

## 1. Weights & Embeddings

**Goal:** Make sure the `.weights` file layout matches the HF model exactly.

### 1.1 Verify raw weights layout

```bash
python3 debug_weights.py
```

Checks:

- Token embeddings: `wte.weight`
- Positional embeddings: `wpe.weight`
- Layer 0 LN1 gamma: `h.0.ln_1.weight`

This confirms the `.weights` header and layout are consistent with the HF
reference.

### 1.2 Verify embeddings (token + position)

```bash
python3 validate_embeddings.py "Hello World" \
  --weights gpt2_bump.weights \
  --executable ./main
```

This uses `./main --debug-embed` to compare C’s

- `embed_tokens` result (token + pos) vs
- HF’s `wte[token] + wpe[pos]`

for each token in the prompt.

If this fails, the bug is in **embedding or positional encoding**, not in the
transformer block.

---

## 2. Layer‑level validation

**Goal:** Find the first layer where the C hidden state diverges from HF.

```bash
python3 validate_layers.py "Hello World" \
  --weights gpt2_bump.weights \
  --executable ./main \
  --layer 0
```

This:

- Runs HF with `output_hidden_states=True`
- Runs C with `--debug-layer 0`
- Compares `hidden_states[layer+1][0, -1, :]` vs the C layer output at
  `L->residual2_output_offset` for the last token.

If layer 0 matches but a deeper layer doesn’t, move the `--layer` index down
until mismatch appears; that’s your first broken layer.

---

## 3. Internal layer stages (LN1 / ATTENTION / MLP)

**Goal:** Inside a specific layer, locate where the computation diverges:

- LN1
- Attention + first residual
- LN2 + MLP + second residual

```bash
python3 validate_layer_stages.py "Hello World" \
  --layer 0 \
  --weights gpt2_bump.weights \
  --executable ./main
```

This uses `debug_forward_dump_layer_output` and compares, for the last token:

- `LAYER_LN1`   ↔ `block.ln_1(h_in)`
- `LAYER_RES1`  ↔ `h_in + attn_out`
- derived `ATTN` = `RES1 − h_in` ↔ HF `attn_out`
- `LAYER_LN2`   ↔ `block.ln_2(res1)`
- `LAYER_MLP`   ↔ `block.mlp(ln2)`
- `LAYER_HIDDEN` (RES2) ↔ final layer output

If:

- LN1 matches but RES1/ATTN diverges → bug is in attention stack or projection.
- RES1 matches but LN2/MLP/RES2 diverge → bug is in LN2 or MLP.

This is the main tool that exposed the **projection + residual wiring bug** we
fixed (see section 5).

---

## 4. Attention internals (Q/K/V, scores, per‑head output)

Once layer‑stages point at “attention is wrong,” you can drill down further.

### 4.1 Q/K/V projection verification

```bash
python3 validate_qkv.py "Hello World" \
  --layer 0 \
  --weights gpt2_bump.weights \
  --executable ./main
```

This uses the `LAYER_QKV` debug prints (head 0, last token) and compares:

- C’s Q/K/V vs HF’s `block.attn.c_attn(ln1)` split into heads.

If Q/K/V match (~1e‑7), `qkv_projection_head_major` and the head‑major layout
are correct.

### 4.2 Attention scores and per‑head output

```bash
python3 validate_attn.py "Hello World" \
  --layer 0 \
  --weights gpt2_bump.weights \
  --executable ./main
```

This uses:

- `LAYER_ATTNSCORES` (softmax scores for head 0, last token)
- `LAYER_ATTNOUT` (per‑head attention output before projection)

and compares against HF’s manually computed:

- scores = (Q·Kᵀ)/√d → softmax
- `attn_out = softmax(scores)·V`

If this matches (~1e‑7), then the **attention math itself** is correct; any
remaining bug must be in the projection back to model dim or residual wiring.

---

## 5. Projection + residual wiring (bug we fixed)

The actual forward bug we hit lived here:

```c
// Old, wrong behavior:
//   RES1 = h_in + (head-major attention buffer)
residual_add_token_parallel(M,
    layer_input_offset,
    L->attention_output_offset,   // head-major [H×T×head_dim]
    L->residual1_output_offset);
```

Correct GPT‑2 math is:

```text
attn_raw  (head-major) -> c_proj(attn_raw) (token-major D)
res1 = h_in + c_proj(attn_raw)
```

The fix:

```c
// 3. Attention computation → head-major attn in L->attention_output_offset
attention_head_major_complete(M, layer_idx);

// 4. Project head-major attention back to token-major model dim
attention_projection_with_concat(M, layer_idx);
//   writes c_proj(attn) into L->residual2_output_offset (token-major)

// 5. First residual: RES1 = h_in + projected attention
residual_add_token_parallel(
    M,
    layer_input_offset,          // h_in
    L->residual2_output_offset,  // c_proj(attn) (token-major)
    L->residual1_output_offset); // RES1
```

Second residual remains:

```c
// RES2 = RES1 + mlp_out
residual_add_token_parallel(
    M,
    L->residual1_output_offset,
    L->mlp_output_offset,
    L->residual2_output_offset);
```

After this change, `validate_layer_stages.py` and `validate_all.sh` show
layer‑0 internals, final hidden state, and logits all match HF to ~1e‑6.

---

## 6. Adapting this to other models (e.g., Qwen)

When you bring up a new model, reuse the same methodology:

1. **Export weights**  
   - Write a new `pytorch_to_c_weights_*.ipynb` that:
     - Loads the HF model (e.g., `QwenForCausalLM`).
     - Maps its state_dict into the `.weights` format this C engine expects
       (token emb, pos emb, each layer’s Q/K/V/proj/MLP/LN, etc.).

2. **Write a model‑specific debug_weights script**  
   - Pattern it after `debug_weights.py`, but:
     - Import the correct HF class.
     - Read the appropriate weights (for Qwen’s architecture).

3. **Reuse/extend the validators**  
   - `validate_embeddings.py`, `validate_layer_stages.py`,
     `validate_qkv.py`, `validate_attn.py` already accept a `--model-name`
     argument; you can:
     - Pass `--model-name your-hf-model-name`
     - Or clone and adjust them if the architecture’s block structure differs.

4. **Use the same drill‑down strategy**  
   - Start with `validate_all.sh`.
   - If something breaks:
     - Find the first broken layer with `validate_layers.py`.
     - Use `validate_layer_stages.py` to see which stage (LN1/ATTN/MLP) is off.
     - If attention is suspect, use `validate_qkv.py` and `validate_attn.py`.

This gives you a reproducible recipe for proving that a new C implementation
is numerically identical to its PyTorch parent, before you start worrying
about optimization or training.

---

## 7. Future work: backward / training validation

The current validation pipeline focuses on the **forward pass**. For training
reliability, a natural next step is:

- Add a `--debug-backward` mode that:
  - Runs one forward + backward step on a tiny batch.
  - Dumps selected gradients (e.g., proj/Q/K/V/MLP weights) in a parseable
    format.
- Write a `validate_backward.py` that:
  - Runs the same loss in HF.
  - Compares those gradients numerically.

The structure would mirror what we’ve already done for the forward path.

