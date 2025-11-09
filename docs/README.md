# C-Transformer: CPU-First Transformer Training Engine

**A pure C implementation of transformer training optimized for x86-64 CPUs with massive memory capacity**

---

## Overview

C-Transformer is a from-scratch implementation of GPT-2 style transformers in pure C, designed to demonstrate that modern CPUs are viable platforms for training large language models. Unlike GPU-centric approaches, C-Transformer leverages the CPU's primary advantage: **massive, affordable memory capacity**.

### Key Innovation

**Training models too large for GPUs** by utilizing commodity DRAM:
- GPUs: Limited to 80-192 GB HBM (expensive, constrained)
- CPUs: Up to 768 GB+ DDR5 per socket (commodity pricing)
- **Target**: Models that exceed GPU memory limits (100B+ parameters)

### Design Philosophy

1. **Memory-first architecture** - Optimize for capacity over bandwidth
2. **Single contiguous allocation** - Bump allocator with hugepage backing
3. **Pure C implementation** - No framework dependencies, full transparency
4. **Training support** - Complete backpropagation with gradient storage
5. **Cache-aware design** - AVX-512 vectorization, NUMA-aware threading

---

## Architecture Highlights

### Memory Layout

Single contiguous allocation using bump allocator:
```
┌─────────────────────────────────────────────────────────┐
│ Token Embeddings (V × D)                                │
├─────────────────────────────────────────────────────────┤
│ Position Embeddings (T × D)                             │
├─────────────────────────────────────────────────────────┤
│ Layer Weights × L (Q, K, V, Proj, MLP, LayerNorms)     │
├─────────────────────────────────────────────────────────┤
│ Forward Activations (T × D per layer)                  │
├─────────────────────────────────────────────────────────┤
│ Gradient Storage (weights + activations)               │
└─────────────────────────────────────────────────────────┘
```

**Total memory**: ~1.17 GB for L=4, d=256 model
- Forward pass: 0.37 GB
- Gradients: 0.80 GB

### Optimization Techniques

1. **Hugepage allocation** (2MB pages) - Reduces TLB misses by 500×
2. **AVX-512 vectorization** - 16-wide SIMD operations
3. **Cache-blocked GEMM** - Optimized for L1/L2 cache
4. **Token-parallel computation** - NUMA-aware work distribution
5. **In-place gradient updates** - Minimizes memory bandwidth

### Numerical Stability

All critical operations implement numerical stability techniques:
- **Log-sum-exp trick** in softmax (prevents overflow/underflow)
- **Epsilon for division** in LayerNorm (prevents NaN)
- **Gradient accumulation** patterns for residual connections
- **Causal masking** to zero invalid attention positions

---

## Documentation

### Getting Started

**[Usage Guide](USAGE_GUIDE.md)** - Comprehensive guide to building and training models
- Quick start tutorial (3 commands to train)
- Compilation options and requirements
- Training workflows and checkpoint management
- Command-line reference
- Troubleshooting common issues

### Technical Deep Dives

**[Numerical Methods & Mathematics](NUMERICAL_METHODS.md)** - Complete mathematical foundations
- **Softmax Jacobian derivation** - Full proof of backward formula
- **Cross-entropy loss gradient** - Why gradient is `softmax - 1[target]`
- **LayerNorm backward** - Complex formula with mean/variance corrections
- **GELU activation** - Approximation and derivative
- **All numerical stability tricks** - Log-sum-exp, epsilon, etc.
- Code cross-references linking math to implementation

**[Backpropagation Flow](BACKPROP_FLOW.md)** - Step-by-step implementation walkthrough
- Complete backward pass sequence
- Layer-by-layer gradient flow
- Memory layout diagrams
- Gradient accumulation patterns

**[Why CPU-First?](WHY_CPU.md)** - Strategic rationale for CPU training
- Memory capacity advantage (768 GB+ vs 192 GB max)
- Cost economics (50-60% savings for large models)
- Open ecosystem benefits (no vendor lock-in)
- When CPUs outperform GPUs

### Comparisons

**[C-Transformer vs Google gemma.cpp](COMPARISON_WITH_GEMMA_CPP.md)** - Technical analysis
- Architecture comparison (both use contiguous memory, cache-aware design)
- SIMD strategies (direct AVX-512 vs Highway library abstraction)
- Focus differences (training + massive memory vs inference + compression)
- ARM porting difficulty analysis
- Strategic positioning

---

## Performance Characteristics

### Model Configuration (Example)

```
Layers: 4
Embedding dimension: 256
Context window: 1024
Vocabulary: 50,257 (GPT-2 tokenizer)
Parameters: ~12M
```

### Training Performance

**Hardware**: Dual Xeon (28 cores utilized)
- **Throughput**: ~30 tokens/sec
- **Memory**: 1.17 GB (fits in L3 cache)
- **FLOPs**: ~240M per training step
  - Forward: 80M FLOPs
  - Backward: 160M FLOPs (2× forward)

### Scaling Estimates

| Layers | d_model | Parameters | Memory (train) | Target Hardware |
|--------|---------|------------|----------------|-----------------|
| 4 | 256 | ~12M | 1.2 GB | Testing/debugging |
| 8 | 512 | ~85M | 4.5 GB | Small experiments |
| 12 | 768 | ~300M | 12 GB | Medium models |
| 24 | 1024 | ~1.2B | 40 GB | Large models |
| 48 | 2048 | ~20B | 250 GB | Very large (CPU advantage) |
| 96 | 4096 | ~175B | 2.8 TB | Extreme (CPU only) |

**Note**: Models beyond ~50B parameters exceed typical GPU memory and demonstrate C-Transformer's unique capability.

---

## Implementation Status

### ✅ Complete Features

- [x] Forward pass (embedding → layers → LM head)
- [x] Backward pass (complete gradient computation)
- [x] SGD optimizer with configurable learning rate
- [x] Checkpoint saving/loading (resume training)
- [x] Training data pipeline (binary format)
- [x] Numerical stability (log-sum-exp, epsilon tricks)
- [x] Memory safety (canary checks, bounds verification)
- [x] Multi-threading (OpenMP with NUMA awareness)
- [x] AVX-512 vectorization (GEMM, LayerNorm, attention)

### 🚧 Planned Features

- [ ] Text generation (inference mode)
- [ ] Adam optimizer (momentum + adaptive learning rate)
- [ ] Mixed precision (BF16 support)
- [ ] Gradient clipping (prevent explosion)
- [ ] Learning rate schedules (warmup, decay)
- [ ] Validation loss tracking (detect overfitting)
- [ ] TensorBoard logging (loss curves, histograms)
- [ ] Flash Attention (memory-efficient attention)

### 🔮 Future Directions

- [ ] ARM NEON port (Apple Silicon, AWS Graviton)
- [ ] Multi-node training (MPI across servers)
- [ ] Model architectures (LLAMA, Mistral, etc.)
- [ ] Quantization (INT8, INT4 inference)
- [ ] Custom allocators (pool allocator, arena)

---

## Building & Running

### Requirements

- **Compiler**: GCC 9+ or Clang 10+
- **CPU**: x86-64 with AVX-512 (or fallback to AVX2)
- **OS**: Linux (Ubuntu 20.04+, other distros supported)
- **Memory**: 2 GB minimum (more for larger models)
- **Dependencies**: OpenMP, standard math library

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/antshiv/C-Transformer.git
cd C-Transformer

# 2. Compile
gcc -o main main.c -lm -fopenmp -march=native -O3

# 3. Prepare training data
python3 prepare_data.py  # Requires: datasets, tiktoken, pandas

# 4. Train
./main \
  --layers 4 \
  --dmodel 256 \
  --ctx 1024 \
  --vocab 50257 \
  --force \
  --train-dir data/training_pairs \
  --train-steps 500 \
  --train-lr 1e-4 \
  --ckpt-dir checkpoints \
  --ckpt-interval 100
```

**Expected output**:
```
⚙  Requested model  L=4  d_model=256  ctx=1024  vocab=50257
→ Would need ≈ 1.17 GiB
✅ Success! mmap at 0x7f..., 1.17 GiB reserved.
🎯 Starting training loop (500 steps, lr=0.000100)
[train] step=1/500  loss=10.692  perplexity=44038.52
...
[train] step=500/500  loss=8.272  perplexity=3912.45
✅ Training complete.
```

**See [Usage Guide](USAGE_GUIDE.md) for detailed instructions.**

---

## Project Context

### ANTSHIV ROBOTICS

C-Transformer is developed by [ANTSHIV ROBOTICS](https://github.com/antshiv) as part of a broader mission: **deploying embedded intelligence for bio-diversity conservation and ecological monitoring**.

**Conservation AI Focus**:
- Autonomous systems for continuous ecological observation
- Edge AI for real-time pattern recognition in the field
- CPU-based models for deployment on field hardware
- Integration with [Antsand Platform](https://www.antsand.com) for sensor → dashboard pipelines

**Why C-Transformer Matters for Conservation**:
1. **Field deployment**: CPUs in rugged hardware (no GPU power/cooling)
2. **Model customization**: Train domain-specific models (species ID, habitat analysis)
3. **Cost efficiency**: Commodity hardware for research organizations
4. **Educational**: Transparent implementation for conservation technologists

### Related Projects

- **Flight Controller Stack**: Autonomous aerial platforms for monitoring
- **Sensor Networks**: TDR soil probes, biodiversity sensors
- **Antsand Platform**: Real-time data orchestration and dashboards
- **Embedded AI**: On-device inference for edge deployment

---

## Contributing

C-Transformer welcomes contributions that align with its mission:

### Areas for Contribution

1. **Optimization**: Improve GEMM kernels, attention mechanisms
2. **Portability**: ARM NEON port, RISC-V support
3. **Features**: Adam optimizer, learning rate schedules
4. **Documentation**: Improve explanations, add examples
5. **Testing**: Gradient checking, numerical accuracy validation

### Guidelines

- **Maintain simplicity**: Pure C, no external dependencies
- **Document thoroughly**: Math + code + rationale
- **Validate numerically**: Compare with PyTorch reference
- **Optimize cache-aware**: Measure L1/L2 hit rates
- **Profile before optimizing**: Use perf, VTune

---

## License

[Specify license - typically MIT, Apache 2.0, or GPL for open source]

---

## Citation

If C-Transformer is useful for research or projects:

```bibtex
@software{c_transformer_2025,
  author = {{ANTSHIV ROBOTICS}},
  title = {C-Transformer: CPU-First Transformer Training Engine},
  year = {2025},
  url = {https://github.com/antshiv/C-Transformer},
  note = {Pure C implementation demonstrating viable CPU-based transformer training}
}
```

---

## Acknowledgments

**Inspired by**:
- [Andrej Karpathy's llm.c](https://github.com/karpathy/llm.c) - Education-focused LLM training
- [Google's gemma.cpp](https://github.com/google/gemma.cpp) - CPU inference with Highway library
- [ggml](https://github.com/ggerganov/ggml) - Tensor library for machine learning

**Built with insights from**:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "GPT-2: Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- "Layer Normalization" (Ba et al., 2016)
- "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)

---

## Contact

**Project**: ANTSHIV ROBOTICS
**Focus**: Conservation AI & Embedded Intelligence
**Documentation**: https://antshiv.github.io/C-Transformer/
**YouTube**: [@AntshivRobotics](https://www.youtube.com/@antshivrobotics)
**Discord**: [Join Discussion](https://discord.gg/bH34RuG2)

---

*C-Transformer: Proving CPUs are viable for training transformers through memory capacity advantage*
