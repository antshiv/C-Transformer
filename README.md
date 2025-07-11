# 🧠 C-Transformer: Pure C Transformer with HPC Layout

A single-file, memory-efficient Transformer model written in C — optimized for CPU-based inference and training. Designed for clarity, performance, and educational value.

---

## 🚀 Why This Project?

This project demonstrates how to build an AI system in C with:
- Single-block memory allocation (huge pages, 64-byte cache lines)
- Token-sliced execution across cores
- Explicit, human-readable memory layout
- Forward and backward pass in a clean architecture

---

## 🧩 Core Concepts

- `layout_transformer()` — defines the memory layout
- `bump()` — allocates offsets for all weights, activations, and embeddings
- `get_slice()` / `get_head_slice()` — extract token/head-parallel tensor views

---

## 🛠️ Build

```bash
git clone https://github.com/antshiv/C-Transformer
cd C-Transformer
chmod +x script.sh
./script.sh

## 🔧 Usage Examples

```c
// Get embedded input slice
float *x = get_slice(M, core_id, M->embedded_input_offset, M->embed_dim);

// Token-parallel QKV access
float *qkv = get_slice(M, core_id, L->qkv_output_offset, 3 * M->embed_dim);

// Token loop
size_t token_count;
float *input = get_slice_and_len(M, core_id, M->embedded_input_offset, M->embed_dim, &token_count);
for (size_t t = 0; t < token_count; ++t) {
    float *vec = input + t * M->embed_dim;
    // Do work here...
}

## 📦 Version v0.2 Highlights

This version includes:
- Full transformer memory layout using `layout_transformer()`
- Bump allocator with zero fragmentation
- Aligned embedding rows for token-sliced, core-parallel access
- Runtime-access helpers: `get_slice()`, `get_slice_and_len()`, and `get_head_slice()`


## 📺 Related Video

This code is explained in detail in these video:

🎥 [The Memory Trick That Makes Multi-Core CPUs Fly for AI](https://youtu.be/Wv0_GLbODeI?si=z0pMmCuD_CjLE_Ao)  
   [Advanced Memory Strategies for High-Performance AI Compute](https://youtu.be/pEhcvMRWhhU?si=uJ7HsiSAMCtQyxHG)
🧠 Covers false sharing, layout design, and core-sliced access

---
© 2025 Antshiv Robotics. All rights reserved.

