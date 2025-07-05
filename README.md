# 🚀 C-Transformers: Cache-Optimized Transformers in C

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

## 🧪 Example

```bash
# Dry run (estimate memory usage only)
./main --layers 12 --dmodel 384 --ctx 256 --vocab 32768

# Force allocation with hugepages
./main --layers 12 --dmodel 384 --ctx 256 --vocab 32768 --force
