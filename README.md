# 🚀 C-Transformers: Cache-Optimized Transformers in C

Building Embedded AI from Scratch — in C, on CPUs, for Real-Time Autonomy.

This project is a pure C implementation of a GPT-style transformer model with:
- 🧠 Fully contiguous, single-block memory layout
- 📏 64-byte alignment for every tensor
- 🧱 Hugepage-backed allocations (2MB) to minimize TLB misses
- ⚙️ Inline memory bump allocator to track offsets precisely
- 🔧 Dry-run and allocation modes for profiling model memory capacity

---

## 📺 YouTube Series

This repo tracks my video series on building high-performance embedded AI systems.

▶️ **Series Playlist**: [Antshiv Robotics YouTube](https://www.youtube.com/@AntshivRobotics)  
🧵 Each video will have its own tagged release (`v1.0`, `v1.1`, etc.) that matches the code exactly.

---

## 🧠 Features

- Optimized layout for GPT-2-style transformers
- Clean C code with no external dependencies
- Command-line options to configure model dimensions:
  - `--layers`
  - `--dmodel`
  - `--ctx`
  - `--vocab`
- `--force` option to trigger actual memory allocation

---

## 🛠️ Build

```bash
git clone https://github.com/antshiv/C-Transformer
cd C-Transformer
chmod +x script.sh
./script.sh
```

## 🧪 Example

```bash
# Dry run (estimate memory usage only)
./main --layers 12 --dmodel 384 --ctx 256 --vocab 32768

# Force allocation with hugepages
./main --layers 12 --dmodel 384 --ctx 256 --vocab 32768 --force
```

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
```

## 📺 Related Video

This code is explained in detail in these video:

🎥 [The Memory Trick That Makes Multi-Core CPUs Fly for AI](https://youtu.be/Wv0_GLbODeI?si=z0pMmCuD_CjLE_Ao)  
   [Advanced Memory Strategies for High-Performance AI Compute](https://youtu.be/pEhcvMRWhhU?si=uJ7HsiSAMCtQyxHG)
🧠 Covers false sharing, layout design, and core-sliced access

---
© 2025 Antshiv Robotics. All rights reserved.

