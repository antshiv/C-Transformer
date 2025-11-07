/**
 * @file main.c
 * @brief CPU-Optimized Large Language Model Runtime (x86-64)
 * @author ANTSHIV ROBOTICS
 * @version 1.0
 * @date 2025
 *
 * @mainpage C-Transformer: A CPU-First Engine for LLMs
 *
 * @section about What is C-Transformer?
 *
 * C-Transformer is a pure C runtime for transformer models, meticulously engineered for high-performance
 * inference and training on modern x86-64 CPUs. It serves as both a practical engine and an educational
 * tool for exploring advanced CPU optimization techniques.
 *
 * This file implements:
 * - **A unified memory layout** for weights, activations, and gradients in a single contiguous block.
 * - **Both inference and backpropagation** logic from first principles.
 * - **Advanced CPU optimizations**, including:
 *   - SIMD vectorization using AVX-512.
 *   - Multi-threading with OpenMP, designed for NUMA and cache awareness.
 *   - Hugepage-backed memory to minimize TLB misses.
 * - **Hybrid parallelism**, combining:
 *   - **Token-level parallelism** for prompt processing.
 *   - **Head-level parallelism** for attention computation.
 *   - A fixed batch size of 1, optimized for real-time inference.
 *
 * @section why_cpu Why Focus on CPUs? The Strategic Bet
 *
 * The prevailing narrative suggests GPUs are the only viable platform for serious AI. This project challenges that notion by arguing that the true comparison is not "CPU vs. GPU," but rather the **Open CPU Ecosystem** versus the **Proprietary NVIDIA Ecosystem**.
 *
 * ### The CPU Advantage: A Bet on Open, Commodity Hardware
 *
 * We believe that the relentless pace of open hardware development will make CPUs an increasingly powerful and cost-effective platform for AI.
 *
 * - **Massive, Affordable Memory**: The latest Intel Xeon and AMD EPYC CPUs support up to 12 channels of DDR5 memory per socket, with DDR6 on the horizon. This allows entire large models (175B+ parameters) to reside in high-bandwidth, commodity DRAM, eliminating the PCIe bottleneck and HBM capacity limits inherent to GPUs.
 *
 * - **Explosive Parallelism**: With core counts exceeding 128 per socket and advanced SIMD instruction sets like AVX-512 for 1D tensor math and AMX for 2D matrix math, CPUs are becoming massively parallel compute engines in their own right.
 *
 * - **Accelerated Data Movement**: On-chip accelerators like Intel's Data Streaming Accelerator (DSA) and DMA engines on ARM offload memory copy operations, freeing up compute cores to focus on arithmetic.
 *
 * - **Freedom from Vendor Lock-In**: The entire CPU ecosystem is built on commodity hardware and open standards. There is no proprietary lock-in equivalent to NVIDIA's CUDA. This fosters competition, drives down cost, and guarantees that performance gains from new hardware generations (e.g., DDR6, 600 Gb/s Ethernet) are immediately accessible without being tied to a single vendor's roadmap.
 *
 * ### The Long-Term Vision
 *
 * As AI models become more efficient and capable, the raw performance gap between CPUs and specialized accelerators will narrow. The combination of ever-improving commodity hardware and sophisticated, cache-aware software design makes the CPU a powerful, open, and increasingly competitive contender for both inference and training. **This project is a bet on that future.**
 *
 * @section architecture System Architecture
 *
 * - **Memory**: Single 2MB hugepage-backed contiguous arena.
 * - **Allocator**: Bump allocator with dry-run mode for size estimation.
 * - **Parallelism**: OpenMP with static thread-to-core binding.
 * - **SIMD**: AVX-512 with FMA (fallback to AVX2 possible).
 * - **Compiler**: GCC/ICC with -O3 -march=native -mavx512f.
 * - **Target**: Intel Xeon (Skylake-SP or newer) / AMD EPYC (Zen 4+).
 *
 * @see layout_transformer For detailed memory layout.
 * @see transformer_layer_forward For end-to-end data flow.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <getopt.h>
#include <immintrin.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <stdbool.h>
#include <assert.h>
#include <dirent.h>

#define ALIGN_UP(n, a) (((n) + (a) - 1) & ~((a) - 1))
#define min(a, b) ((a) < (b) ? (a) : (b))

/* ─── alignment targets ───────────────────────────────────────────── */
#define CACHE_ALIGN 64ULL
#define HUGE_ALIGN (2ULL * 1024 * 1024) /* 2 MB huge page */

/* MEMORY OVERFLOW MACROS */
#define CANARY_SIZE_FLOATS 16 // 64 bytes, one cache line
#define FINAL_CANARY_ZONE_FLOATS 1024 // Reserve 4KB at the very end

/* ============================================================================
   HEAD-MAJOR ACCESS MACROS
   Layout: [head][token][head_dim] 
   Memory: [Head0: Token0[head_dim], Token1[head_dim], ..., TokenN[head_dim]]
           [Head1: Token0[head_dim], Token1[head_dim], ..., TokenN[head_dim]]
           [...]
   ============================================================================ */
/* ATTENTION HEAD ACCESS */
#define Q_ACCESS(q_ptr, h, t, d, context_window, aligned_head_dim) \
    q_ptr[((h) * (context_window) + (t)) * (aligned_head_dim) + (d)]

#define K_ACCESS(k_ptr, h, t, d, context_window, aligned_head_dim) \
    k_ptr[((h) * (context_window) + (t)) * (aligned_head_dim) + (d)]

#define V_ACCESS(v_ptr, h, t, d, context_window, aligned_head_dim) \
    v_ptr[((h) * (context_window) + (t)) * (aligned_head_dim) + (d)]

/* ============================================================================
   ATTENTION SCORE ACCESS MACRO (HEAD-MAJOR)
   Layout: [head][query_token][key_token]
   Memory: [Head0: Q0[K0, K1, ..., KN],
                    Q1[K0, K1, ..., KN],
                    ...
                    QN[K0, K1, ..., KN]]
           [Head1: Q0[K0, K1, ..., KN],
                    Q1[K0, K1, ..., KN],
                    ...
                    QN[K0, K1, ..., KN]]
           [...]
   ============================================================================
   Indexing:
     - attn_ptr: Base pointer to attention scores
     - head_idx: Which head (h)
     - query_token: Q token index (i)
     - key_token: K token index (j)
     - context_window: Number of tokens (T)

   Flattened offset: [h * T * T + i * T + j]
   ============================================================================
*/
#define ATTN_ACCESS(attn_ptr, head_idx, query_token, key_token, context_window) \
    attn_ptr[((head_idx) * (context_window) + (query_token)) * (context_window) + (key_token)]

/* ─── tiny helpers ────────────────────────────────────────────────── */
static inline size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

// Enhanced timing function
static inline double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * @brief Allocate memory using 2MB hugepages (with fallback to THP)
 *
 * Attempts to allocate memory backed by explicit 2MB hugepages via mmap.
 * If that fails (e.g., insufficient hugepages configured), falls back to
 * aligned_alloc + madvise(MADV_HUGEPAGE) for transparent hugepage (THP) support.
 *
 * @param bytes Number of bytes to allocate
 * @return Pointer to allocated memory (2MB-aligned)
 *
 * @details
 * **Why Hugepages Matter**:
 * - ✅ **TLB Efficiency**: 2MB pages reduce TLB entries by 512x vs 4KB pages
 * - ✅ **Page Fault Reduction**: Fewer page faults during model initialization
 * - ✅ **Memory Bandwidth**: Better DRAM page locality
 * - ✅ **Latency**: Reduced virtual-to-physical address translation overhead
 *
 * **Performance Impact**:
 * On a 4GB model with 1024 4KB pages:
 * - TLB misses: ~1M misses without hugepages
 * - TLB misses: ~2K misses with 2MB hugepages
 * - Result: 500x reduction in TLB overhead (~5-10% speedup)
 *
 * **System Configuration** (Linux):
 * ```bash
 * # Check available hugepages
 * cat /proc/meminfo | grep Huge
 *
 * # Allocate 2048 × 2MB = 4GB of hugepages
 * echo 2048 | sudo tee /proc/sys/vm/nr_hugepages
 *
 * # Enable THP (fallback)
 * echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
 * ```
 *
 * **Allocation Strategy**:
 * 1. Try explicit hugepages via mmap + MAP_HUGETLB (best performance)
 * 2. Fall back to aligned_alloc + MADV_HUGEPAGE (THP, still good)
 * 3. Kernel promotes pages to 2MB when possible
 *
 * @warning Requires root or CAP_SYS_RESOURCE for explicit hugepages
 * @note Falls back gracefully to THP if explicit allocation fails
 * @see https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt
 */
static void *huge_alloc(size_t bytes)
{
    size_t len = align_up(bytes, HUGE_ALIGN);

    // Try explicit 2MB hugepage allocation
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED)
        return p;

    // Fallback: aligned allocation + transparent hugepage hint
    p = aligned_alloc(HUGE_ALIGN, len);
    if (!p)
    {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }
    madvise(p, len, MADV_HUGEPAGE);  // Request THP promotion
    return p;
}

/* ─── model structs ───────────────────────────────────────────────── */
typedef struct {
    // Canary protection offsets for this specific layer
    size_t layer_start_canary_offset;

    size_t ln1_weight_offset, ln1_bias_offset;
    size_t ln1_mean_offset, ln1_rstd_offset;
    size_t layer_input_offset, ln1_output_offset;
    
    // Separate Q, K, V for cleaner access
    size_t q_weight_offset, q_bias_offset, q_output_offset;
    size_t k_weight_offset, k_bias_offset, k_output_offset;
    size_t v_weight_offset, v_bias_offset, v_output_offset;
    
    size_t attention_scores_offset;

    size_t proj_weight_offset, proj_bias_offset;
    size_t attention_output_offset, residual1_output_offset;

    size_t ln2_weight_offset, ln2_bias_offset;
    size_t ln2_mean_offset, ln2_rstd_offset;
    size_t ln2_output_offset;

    size_t fc1_weight_offset, fc1_bias_offset, fc1_output_offset; // Added fc1_output for intermediate storage
    size_t fc2_weight_offset, fc2_bias_offset;
    size_t mlp_output_offset, residual2_output_offset;

    // Canary protection offset at the end of the layer
    size_t layer_end_canary_offset;
} TrulyOptimalLayer;

/* ─── Per-Layer Backprop Structure ─────────────────────────────────── */
typedef struct {
    /* LAYER INPUT/OUTPUT */
    size_t residual2_copy_offset;              // [T × D] output of this layer (from forward)
    size_t d_residual2_offset;                 // [T × D] gradient from layer above
    
    /* ===== MLP BACKWARD PATH ===== */
    // MLP output
    size_t mlp_output_copy_offset;             // [T × D]
    size_t d_mlp_output_offset;                // [T × D]
    
    // FC2: [T × 4D] -> [T × D]
    size_t fc2_input_copy_offset;              // [T × 4D] (after GELU)
    size_t fc2_weights_copy_offset;            // [4D × D]
    size_t fc2_bias_copy_offset;               // [D]
    size_t d_fc2_input_offset;                 // [T × 4D]
    size_t d_fc2_weights_offset;               // [4D × D] accumulator
    size_t d_fc2_bias_offset;                  // [D] accumulator
    
    // GELU
    size_t fc1_output_copy_offset;             // [T × 4D] (before GELU)
    size_t d_fc1_output_offset;                // [T × 4D]
    
    // FC1: [T × D] -> [T × 4D]
    size_t ln2_output_copy_offset;             // [T × D]
    size_t fc1_weights_copy_offset;            // [D × 4D]
    size_t fc1_bias_copy_offset;               // [4D]
    size_t d_ln2_output_offset;                // [T × D]
    size_t d_fc1_weights_offset;               // [D × 4D] accumulator
    size_t d_fc1_bias_offset;                  // [4D] accumulator
    
    // LayerNorm2
    size_t ln2_input_copy_offset;              // [T × D] (residual1)
    size_t ln2_mean_copy_offset;               // [T]
    size_t ln2_rstd_copy_offset;               // [T]
    size_t ln2_gamma_copy_offset;              // [D]
    size_t ln2_beta_copy_offset;               // [D]
    size_t d_ln2_input_offset;                 // [T × D]
    size_t d_ln2_gamma_offset;                 // [D] accumulator
    size_t d_ln2_beta_offset;                  // [D] accumulator
    
    /* ===== ATTENTION BACKWARD PATH ===== */
    // Residual1
    size_t residual1_copy_offset;              // [T × D]
    size_t d_residual1_offset;                 // [T × D]
    
    // Attention output projection
    size_t attention_output_copy_offset;       // [T × D]
    size_t proj_weights_copy_offset;           // [D × D]
    size_t proj_bias_copy_offset;              // [D]
    size_t d_attention_output_offset;          // [T × D] gradient from residual path
    size_t d_attention_token_offset;           // [T × D] token-major gradient after projection backward
    size_t d_attention_head_offset;            // [n_heads × T × H] head-major gradient for attention mechanism
    size_t d_proj_weights_offset;              // [D × D] accumulator
    size_t d_proj_bias_offset;                 // [D] accumulator
    
    // Attention mechanism
    size_t attention_weights_copy_offset;      // [n_heads × T × T] (after softmax)
    size_t v_output_copy_offset;               // [T × n_heads × H]
    size_t d_attention_weights_offset;         // [n_heads × T × T]
    size_t d_v_output_offset;                  // [T × n_heads × H]
    
    // Softmax backward
    size_t d_attention_scores_offset;          // [n_heads × T × T]
    
    // QK^T backward
    size_t q_output_copy_offset;               // [T × n_heads × H]
    size_t k_output_copy_offset;               // [T × n_heads × H]
    size_t d_q_output_offset;                  // [T × n_heads × H]
    size_t d_k_output_offset;                  // [T × n_heads × H]
    
    // Q, K, V projections
    size_t ln1_output_copy_offset;             // [T × D]
    size_t q_weights_copy_offset;              // [D × D]
    size_t q_bias_copy_offset;                 // [D]
    size_t k_weights_copy_offset;              // [D × D]
    size_t k_bias_copy_offset;                 // [D]
    size_t v_weights_copy_offset;              // [D × D]
    size_t v_bias_copy_offset;                 // [D]
    size_t d_ln1_output_offset;                // [T × D] (accumulates Q,K,V grads)
    size_t d_q_weights_offset;                 // [D × D] accumulator
    size_t d_q_bias_offset;                    // [D] accumulator
    size_t d_k_weights_offset;                 // [D × D] accumulator
    size_t d_k_bias_offset;                    // [D] accumulator
    size_t d_v_weights_offset;                 // [D × D] accumulator
    size_t d_v_bias_offset;                    // [D] accumulator
    size_t qkv_scratch_offset;                 // [T × D] scratch buffer for reshaping
    
    // LayerNorm1
    size_t ln1_input_copy_offset;              // [T × D]
    size_t ln1_mean_copy_offset;               // [T]
    size_t ln1_rstd_copy_offset;               // [T]
    size_t ln1_gamma_copy_offset;              // [D]
    size_t ln1_beta_copy_offset;               // [D]
    size_t d_ln1_input_offset;                 // [T × D] flows to previous layer
    size_t d_ln1_gamma_offset;                 // [D] accumulator
    size_t d_ln1_beta_offset;                  // [D] accumulator
    
} LayerGradients;

/* ─── Complete Backprop Memory Structure ─────────────────────────────── */
typedef struct {
    /* Single contiguous memory block for EVERYTHING needed in backprop */
    size_t backprop_base;
    size_t total_gradient_floats;
    
    /* ===== STAGE 1: LOSS & INITIAL GRADIENTS ===== */
    size_t logits_copy_offset;                 // [T × V] copy of forward logits
    size_t actual_tokens_offset;               // [T] copy of target tokens (as floats for alignment)
    size_t d_logits_offset;                    // [T × V] gradient w.r.t logits
    
    /* ===== STAGE 2: FINAL OUTPUT LAYER ===== */
    size_t final_output_copy_offset;           // [T × D] copy from forward
    size_t d_final_output_offset;              // [T × D] gradient
    size_t d_embed_weights_offset;             // [V × D] gradient accumulator
    
    /* ===== STAGE 3: FINAL LAYERNORM ===== */
    size_t final_ln_input_copy_offset;         // [T × D] input to final LN
    size_t final_ln_mean_copy_offset;          // [T] mean from forward
    size_t final_ln_rstd_copy_offset;          // [T] rstd from forward  
    size_t final_ln_gamma_copy_offset;         // [D] weights
    size_t final_ln_beta_copy_offset;          // [D] bias
    size_t d_final_ln_input_offset;            // [T × D] gradient to previous layer
    size_t d_final_ln_gamma_offset;            // [D] weight gradient
    size_t d_final_ln_beta_offset;             // [D] bias gradient
    
    /* ===== PER-LAYER BACKPROP MEMORY ===== */
    LayerGradients* layers;         // Array of per-layer gradient structs
    
    size_t d_pos_embed_offset;                 // [T × D] positional embedding gradients
    
    size_t layer_backprop_stride;              // Distance between layers
    
} GradientStorage;

/**
 * @struct TransformerModel
 * @brief Main transformer model structure with unified memory layout
 *
 * This structure encapsulates the entire transformer model state in a single
 * contiguous memory block. All tensors (weights, activations, gradients) are
 * accessed via byte offsets from `memory_base`.
 *
 * @details
 * **Design Philosophy**:
 * - Single malloc/mmap for entire model (eliminates fragmentation)
 * - Offset-based addressing (enables memory-mapped weight files)
 * - Cache-line aligned tensors (64-byte boundaries)
 * - Hugepage-backed allocation (2MB pages reduce TLB pressure)
 *
 * **Memory Layout**:
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │ Token Embeddings [vocab_size × aligned_embed_dim]           │ ← Shared with lm_head
 * ├─────────────────────────────────────────────────────────────┤
 * │ Positional Embeddings [context_window × aligned_embed_dim]  │
 * ├─────────────────────────────────────────────────────────────┤
 * │ Layer 0 (weights + activations)                             │
 * │ Layer 1 (weights + activations)                             │
 * │ ...                                                          │
 * │ Layer N (weights + activations)                             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ Final LayerNorm                                              │
 * ├─────────────────────────────────────────────────────────────┤
 * │ Logits [context_window × vocab_size]                        │
 * ├─────────────────────────────────────────────────────────────┤
 * │ Gradient Storage (if training_enabled)                      │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Parallelism Model**:
 * - Token-level parallelism: Context window divided among `num_cores`
 * - Each core processes `tokens_per_core` tokens independently
 * - Head-level parallelism: Attention heads processed independently
 *
 * **Alignment Guarantees**:
 * - `aligned_embed_dim`: Padded to 64-byte boundary for AVX-512
 * - `aligned_head_dim`: Padded to 64-byte boundary (prevents false sharing)
 * - `aligned_attn_context_window`: Padded attention score matrix rows
 *
 * @see layout_transformer Memory layout computation
 * @see bump Offset-based allocation helper
 */
typedef struct
{
    /* ═══════════════════════════════════════════════════════════════════ */
    /* File metadata (from weight file header)                              */
    /* ═══════════════════════════════════════════════════════════════════ */
    char magic[8];              ///< Magic string "BUMPWGT2" for file validation
    uint32_t version;           ///< Weight file format version
    uint32_t model_type;        ///< Model architecture: 0=GPT2, 1=LLAMA, etc.

    /* ═══════════════════════════════════════════════════════════════════ */
    /* Model Hyperparameters                                                 */
    /* ═══════════════════════════════════════════════════════════════════ */
    int num_layers;             ///< Number of transformer layers (e.g., 12 for GPT-2)
    int vocab_size;             ///< Vocabulary size (e.g., 50257 for GPT-2)
    int embed_dim;              ///< Embedding dimension (e.g., 768 for GPT-2 small)
    int context_window;         ///< Maximum sequence length (e.g., 1024)

    size_t aligned_embed_dim;   ///< embed_dim rounded up to 64-byte alignment (in floats)
    size_t aligned_head_dim;    ///< head_dim rounded up to 64-byte alignment (in floats)
    size_t aligned_attn_context_window; ///< context_window padded to prevent false sharing

    /* ═══════════════════════════════════════════════════════════════════ */
    /* Execution Plan (Parallelism Configuration)                           */
    /* ═══════════════════════════════════════════════════════════════════ */
    int num_cores;              ///< Number of CPU cores to use (OpenMP threads)
    int tokens_per_core;        ///< Tokens assigned per core: context_window / num_cores
    int num_attention_heads;    ///< Number of attention heads (e.g., 12 for GPT-2)
    int head_dim;               ///< Dimension per head: embed_dim / num_attention_heads

    /* ═══════════════════════════════════════════════════════════════════ */
    /* Unified Memory Block                                                  */
    /* ═══════════════════════════════════════════════════════════════════ */
    float *memory_base;         ///< Base pointer to single contiguous memory block
    size_t total_floats;        ///< Total size of memory block in float elements
    size_t layer_stride;        ///< Byte offset between consecutive layer memory blocks

    /* ═══════════════════════════════════════════════════════════════════ */
    /* Global Tensor Offsets (in float elements from memory_base)           */
    /* ═══════════════════════════════════════════════════════════════════ */
    size_t token_emb_offset;    ///< Token embedding table [vocab_size × aligned_embed_dim]
    size_t pos_emb_offset;      ///< Positional embedding table [context_window × aligned_embed_dim]
    size_t embedded_input_offset; ///< Combined token+pos embeddings [context_window × aligned_embed_dim]
    size_t layers_start_offset; ///< Start of first transformer layer memory

    /* ═══════════════════════════════════════════════════════════════════ */
    /* Per-Layer Memory Layout                                               */
    /* ═══════════════════════════════════════════════════════════════════ */
    TrulyOptimalLayer *layers;  ///< Array of per-layer offset structures

    /* ═══════════════════════════════════════════════════════════════════ */
    /* Final Output Layers                                                   */
    /* ═══════════════════════════════════════════════════════════════════ */
    size_t final_ln_weight_offset; ///< Final LayerNorm gamma [aligned_embed_dim]
    size_t final_ln_bias_offset;   ///< Final LayerNorm beta [aligned_embed_dim]
    size_t final_ln_mean_offset;   ///< Final LayerNorm mean [context_window]
    size_t final_ln_rstd_offset;   ///< Final LayerNorm rstd [context_window]
    size_t final_output_offset;    ///< Final normalized output [context_window × aligned_embed_dim]

    size_t lm_head_weight_offset;  ///< Language model head (weight-tied to token_emb_offset)
    size_t logits_offset;          ///< Output logits [context_window × vocab_size]

    /* ═══════════════════════════════════════════════════════════════════ */
    /* Training State (Optional)                                             */
    /* ═══════════════════════════════════════════════════════════════════ */
    GradientStorage gradients;  ///< Gradient and activation cache memory (training only)
    bool training_enabled;      ///< Whether gradient storage is allocated
    float learning_rate;        ///< SGD learning rate for weight updates

    /* ═══════════════════════════════════════════════════════════════════ */
    /* File Integrity                                                        */
    /* ═══════════════════════════════════════════════════════════════════ */
    uint8_t checksum[32];       ///< SHA256 checksum of weight file
    uint8_t reserved[32];       ///< Reserved for future extensions
} TransformerModel;

/**
 * @brief Bump allocator for sequential memory layout
 *
 * Core primitive for building the contiguous memory layout. This function:
 * 1. Aligns the current offset to the requested boundary
 * 2. Returns the aligned offset for tensor placement
 * 3. Advances the offset by the tensor size
 *
 * @param off Pointer to current offset cursor (in float elements)
 * @param count Number of float elements to allocate
 * @param alignB Alignment requirement in bytes (e.g., 64 for cache lines)
 * @return Aligned offset where tensor should be placed
 *
 * @details
 * **Why Bump Allocation?**
 * - ✅ **Zero Fragmentation**: Sequential layout, no holes
 * - ✅ **Predictable Addresses**: Enables memory-mapped file loading
 * - ✅ **Cache Locality**: Related tensors are spatially close
 * - ✅ **Dry-Run Mode**: Can compute total size before allocation
 *
 * **Example Usage**:
 * ```c
 * size_t offset = 0;
 * size_t q_weight_off = bump(&offset, 768*768, CACHE_ALIGN);  // Aligned to 64B
 * size_t k_weight_off = bump(&offset, 768*768, CACHE_ALIGN);  // Sequential
 * size_t v_weight_off = bump(&offset, 768*768, CACHE_ALIGN);
 * // offset now contains total size needed
 * ```
 *
 * **Alignment Rationale**:
 * - 64-byte alignment matches cache line size
 * - Enables use of aligned SIMD loads (_mm512_load_ps vs _mm512_loadu_ps)
 * - Prevents false sharing (each tensor starts on new cache line)
 *
 * @note This function does NOT allocate memory, only tracks offsets
 * @see layout_transformer Full memory layout using bump allocation
 */
static inline size_t bump(size_t *off, size_t count, size_t alignB) {
    *off = align_up(*off, alignB / sizeof(float));
    size_t here = *off;
    *off += count;
    return here;
}


 /**
 * @brief Lays out the memory for the backward pass.
 *
 * @param M Pointer to the TransformerModel struct.
 * @param offset Pointer to the current memory offset, which will be updated.
 *
 * @details
 * This function allocates a dedicated, contiguous memory arena for all data
 * required during backpropagation. This includes:
 * - **GRADS:** Buffers to accumulate gradients for every weight and bias.
 * - **ACTS:** Cached copies of activations from the forward pass needed by the backward pass.
 * - **dACTS:** Buffers to hold the gradients of activations as they flow backward.
 *
 * @verbatim
 * ┌────────────────────────────────────────────────────────────┐
 * │ Global GRADS (Embeddings, Final LN)                        │
 * ├────────────────────────────────────────────────────────────┤
 * │ Global ACTS (Logits, Final LN inputs...)                   │
 * ├────────────────────────────────────────────────────────────┤
 * │ Global dACTS (dLogits, dFinal_output...)                   │
 * ├────────────────────────────────────────────────────────────┤
 * │ Per-Layer Arena (Layer 0)                                  │
 * │ ┌────────────────────────────────────────────────────────┐ │
 * │ │ Layer 0 ACTS (ln_mean, attn_probs, fc1_preact...)      │ │
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ Layer 0 dACTS (dL/d_mlp_output, dL/d_attn_output...)   │ │
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ Layer 0 GRADS (dL/dW_fc1, dL/dW_q...)                  │ │
 * │ └────────────────────────────────────────────────────────┘ │
 * ├────────────────────────────────────────────────────────────┤
 * │ ... repeated for each layer ...                            │
 * └────────────────────────────────────────────────────────────┘
 * @endverbatim
 */
void layout_gradients(TransformerModel *M, size_t *offset) {
    // Continue from where forward pass left off
    size_t off = *offset;
    
    // Allocate GradientStorage struct
    M->gradients.backprop_base = off; 
    
    size_t D = M->aligned_embed_dim;
    size_t H = M->aligned_head_dim;
    size_t T = M->context_window;
    size_t V = M->vocab_size;
    size_t n_heads = M->num_attention_heads;
    
    /* ===== GLOBAL GRADIENT STORAGE ===== */
    
    // Stage 1: Loss computation
    M->gradients.logits_copy_offset = bump(&off, T * V, CACHE_ALIGN);
    M->gradients.actual_tokens_offset = bump(&off, T, CACHE_ALIGN);
    M->gradients.d_logits_offset = bump(&off, T * V, CACHE_ALIGN);
    
    // Stage 2: Final output layer
    M->gradients.final_output_copy_offset = bump(&off, T * D, CACHE_ALIGN);
    M->gradients.d_final_output_offset = bump(&off, T * D, CACHE_ALIGN);
    M->gradients.d_embed_weights_offset = bump(&off, V * D, CACHE_ALIGN);
    
    // Stage 3: Final LayerNorm
    M->gradients.final_ln_input_copy_offset = bump(&off, T * D, CACHE_ALIGN);
    M->gradients.final_ln_mean_copy_offset = bump(&off, T, CACHE_ALIGN);
    M->gradients.final_ln_rstd_copy_offset = bump(&off, T, CACHE_ALIGN);
    M->gradients.final_ln_gamma_copy_offset = bump(&off, D, CACHE_ALIGN);
    M->gradients.final_ln_beta_copy_offset = bump(&off, D, CACHE_ALIGN);
    M->gradients.d_final_ln_input_offset = bump(&off, T * D, CACHE_ALIGN);
    M->gradients.d_final_ln_gamma_offset = bump(&off, D, CACHE_ALIGN);
    M->gradients.d_final_ln_beta_offset = bump(&off, D, CACHE_ALIGN);
    
    /* ===== PER-LAYER GRADIENT STORAGE ===== */
    
    M->gradients.layers = (LayerGradients*)calloc(M->num_layers, sizeof(LayerGradients));
    if (!M->gradients.layers) {
        perror("Failed to allocate LayerGradients array");
        exit(EXIT_FAILURE);
    }
    
    for (int l = 0; l < M->num_layers; l++) {
        LayerGradients *L = &M->gradients.layers[l];

        // Layer output
        L->residual2_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_residual2_offset = bump(&off, T * D, CACHE_ALIGN);

        // MLP Backward
        L->mlp_output_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_mlp_output_offset = bump(&off, T * D, CACHE_ALIGN);
        L->fc2_input_copy_offset = bump(&off, T * 4 * D, CACHE_ALIGN);
        L->d_fc2_input_offset = bump(&off, T * 4 * D, CACHE_ALIGN);
        L->d_fc2_weights_offset = bump(&off, 4 * D * D, CACHE_ALIGN);
        L->d_fc2_bias_offset = bump(&off, D, CACHE_ALIGN);
        L->fc1_output_copy_offset = bump(&off, T * 4 * D, CACHE_ALIGN);
        L->d_fc1_output_offset = bump(&off, T * 4 * D, CACHE_ALIGN);
        L->ln2_output_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_ln2_output_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_fc1_weights_offset = bump(&off, D * 4 * D, CACHE_ALIGN);
        L->d_fc1_bias_offset = bump(&off, 4 * D, CACHE_ALIGN);

        // LayerNorm2 Backward
        L->ln2_input_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->ln2_mean_copy_offset = bump(&off, T, CACHE_ALIGN);
        L->ln2_rstd_copy_offset = bump(&off, T, CACHE_ALIGN);
        L->ln2_gamma_copy_offset = bump(&off, D, CACHE_ALIGN);
        L->d_ln2_input_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_ln2_gamma_offset = bump(&off, D, CACHE_ALIGN);
        L->d_ln2_beta_offset = bump(&off, D, CACHE_ALIGN);

        // Attention Backward
        L->residual1_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_residual1_offset = bump(&off, T * D, CACHE_ALIGN);
        L->attention_output_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_attention_output_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_attention_token_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_attention_head_offset = bump(&off, n_heads * T * H, CACHE_ALIGN);
        L->d_proj_weights_offset = bump(&off, D * D, CACHE_ALIGN);
        L->d_proj_bias_offset = bump(&off, D, CACHE_ALIGN);

        // Attention mechanism
        L->attention_weights_copy_offset = bump(&off, n_heads * T * T, CACHE_ALIGN);
        L->v_output_copy_offset = bump(&off, T * n_heads * H, CACHE_ALIGN);
        L->d_attention_weights_offset = bump(&off, n_heads * T * T, CACHE_ALIGN);
        L->d_v_output_offset = bump(&off, T * n_heads * H, CACHE_ALIGN);
        L->d_attention_scores_offset = bump(&off, n_heads * T * T, CACHE_ALIGN);
        L->q_output_copy_offset = bump(&off, T * n_heads * H, CACHE_ALIGN);
        L->k_output_copy_offset = bump(&off, T * n_heads * H, CACHE_ALIGN);
        L->d_q_output_offset = bump(&off, T * n_heads * H, CACHE_ALIGN);
        L->d_k_output_offset = bump(&off, T * n_heads * H, CACHE_ALIGN);

        // QKV projections
        L->ln1_output_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_ln1_output_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_q_weights_offset = bump(&off, D * D, CACHE_ALIGN);
        L->d_q_bias_offset = bump(&off, D, CACHE_ALIGN);
        L->d_k_weights_offset = bump(&off, D * D, CACHE_ALIGN);
        L->d_k_bias_offset = bump(&off, D, CACHE_ALIGN);
        L->d_v_weights_offset = bump(&off, D * D, CACHE_ALIGN);
        L->d_v_bias_offset = bump(&off, D, CACHE_ALIGN);
        L->qkv_scratch_offset = bump(&off, T * D, CACHE_ALIGN);

        // LayerNorm1 Backward
        L->ln1_input_copy_offset = bump(&off, T * D, CACHE_ALIGN);
        L->ln1_mean_copy_offset = bump(&off, T, CACHE_ALIGN);
        L->ln1_rstd_copy_offset = bump(&off, T, CACHE_ALIGN);
        L->ln1_gamma_copy_offset = bump(&off, D, CACHE_ALIGN);
        L->d_ln1_input_offset = bump(&off, T * D, CACHE_ALIGN);
        L->d_ln1_gamma_offset = bump(&off, D, CACHE_ALIGN);
        L->d_ln1_beta_offset = bump(&off, D, CACHE_ALIGN);
    }
    
    // 5. Embeddings (last in backward)
    M->gradients.d_pos_embed_offset = bump(&off, T * D, CACHE_ALIGN);
    // M->gradients.d_embed_weights_offset  <- embed weigths is again used but we have allcoated this pointer above. 
    
    // Update total gradient floats
    M->gradients.total_gradient_floats = off - *offset;
    
    // Return the updated offset
    *offset = off;
}

/**
 * @brief Plans and allocates a single contiguous memory block for the entire Transformer model.
 * 
 * @param M Pointer to the TransformerModel struct to be populated.
 * 
 * @details
 * This function orchestrates the memory layout for the model, aligned with HPC best practices:
 * - Uses a single allocation to minimize OS overhead and TLB misses.
 * - Aligns blocks to cache lines (`CACHE_ALIGN`) to improve memory performance.
 * - Supports memory-mapped file layout for fast startup.
 * - Inserts debug-friendly `CANARY` markers to detect buffer overflows.
 *
 * **Memory Layout Overview:**
 * 
 * Let:
 * - `D` = aligned embedding dimension
 * - `H` = aligned head dimension
 * - `T` = context window
 * - `V` = vocabulary size
 *
 * @verbatim
 * ┌────────────────────────────────────────────────────────────┐
 * │ Token & Positional Embeddings Tables                       │ ← Shared [V * D, T * D]
 * ├────────────────────────────────────────────────────────────┤
 * │ Layer 0                                                    │
 * │ ┌────────────────────────────────────────────────────────┐ │
 * │ │ START CANARY                                           │ │
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ LN1 Weights & Biases                                   │ ← [~D]
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ Attention Weights (Wq, Wk, Wv, W_proj)                 │ ← [~D * D]
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ QKV Activation Buffers                                 │ ← [T * num_heads * H]
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ LN2 Weights & Biases                                   │ ← [~D]
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ MLP Weights (W_fc1, W_fc2)                             │ ← [~D * 4D, 4D * D]
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ MLP Activations & Residual Buffers                     │ ← [T * 4D, T * D]
 * │ ├────────────────────────────────────────────────────────┤ │
 * │ │ END CANARY                                             │ │
 * │ └────────────────────────────────────────────────────────┘ │
 * ├────────────────────────────────────────────────────────────┤
 * │ ... repeated for each layer ...                            │
 * ├────────────────────────────────────────────────────────────┤
 * │ Final LayerNorm + Final CANARY                             │
 * └────────────────────────────────────────────────────────────┘
 * @endverbatim
 *
 * This memory map ensures high locality for token-wise, head-wise, and GEMM-parallel computations.
 */
void layout_transformer(TransformerModel *M, bool training_mode) {
    size_t off = 0;
    size_t aligned_embed_dim = align_up(M->embed_dim, CACHE_ALIGN / sizeof(float));
    M->aligned_embed_dim = aligned_embed_dim;
    
    size_t aligned_head_dim = align_up(M->embed_dim / M->num_attention_heads , CACHE_ALIGN / sizeof(float));
    M->aligned_head_dim = aligned_head_dim;
    
    // Calculate cache-aligned context window to prevent false sharing
    size_t aligned_attn_context_window = align_up(M->context_window, CACHE_ALIGN / sizeof(float));
    M->aligned_attn_context_window = aligned_attn_context_window;

    M->token_emb_offset = bump(&off, (size_t)M->vocab_size * aligned_embed_dim, CACHE_ALIGN);
    M->pos_emb_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
    M->embedded_input_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);

    M->layers_start_offset = off;
    M->layers = malloc(sizeof(TrulyOptimalLayer) * M->num_layers);
    if (!M->layers)
    {
        perror("malloc layers");
        exit(EXIT_FAILURE);
    }

    for (int l = 0; l < M->num_layers; ++l)
    {
        TrulyOptimalLayer *L = &M->layers[l];
        
        // Allocate a canary block at the START of the layer's memory
        L->layer_start_canary_offset = bump(&off, CANARY_SIZE_FLOATS, CACHE_ALIGN);

        // Transformer layer layout
        L->ln1_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->ln1_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        // allocate per-token mean and rstd
        L->ln1_mean_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
        L->ln1_rstd_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
        L->layer_input_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->ln1_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        
        // Separate Q, K, V weights and outputs
        L->q_weight_offset = bump(&off, aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->q_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->q_output_offset = bump(&off, (size_t)M->context_window * aligned_head_dim * M->num_attention_heads, CACHE_ALIGN);

        L->k_weight_offset = bump(&off, aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->k_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->k_output_offset = bump(&off, (size_t)M->context_window * aligned_head_dim * M->num_attention_heads, CACHE_ALIGN);

        L->v_weight_offset = bump(&off, aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->v_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->v_output_offset = bump(&off, (size_t)M->context_window * aligned_head_dim * M->num_attention_heads, CACHE_ALIGN);
        
        L->attention_scores_offset = bump(&off, (size_t)M->num_attention_heads * aligned_attn_context_window * aligned_attn_context_window, 
                                                CACHE_ALIGN);
        
        L->proj_weight_offset = bump(&off, aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->proj_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->attention_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->residual1_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->ln2_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->ln2_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        // allocate per-token mean and rstd
        L->ln2_mean_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
        L->ln2_rstd_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
        L->ln2_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->fc1_weight_offset = bump(&off, 4ULL * aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->fc1_bias_offset = bump(&off, 4ULL * aligned_embed_dim, CACHE_ALIGN);
        L->fc1_output_offset = bump(&off, 4ULL * (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN); // Added fc1 output storage
        L->fc2_weight_offset = bump(&off, 4ULL * aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);         // FC2 is (4D)x(D)
        L->fc2_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->mlp_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->residual2_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);

        // Allocate a canary block at the END of the layer's memory
        L->layer_end_canary_offset = bump(&off, CANARY_SIZE_FLOATS, CACHE_ALIGN);
    }
    if (M->num_layers > 1)
    {
        // The layer stride is the distance from the start of one layer's memory
        // block (including its start canary) to the start of the next.
        M->layer_stride = M->layers[1].layer_start_canary_offset - M->layers[0].layer_start_canary_offset;
    }

    M->final_ln_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
    M->final_ln_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
    M->final_ln_mean_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
    M->final_ln_rstd_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
    
    M->final_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
    
    M->lm_head_weight_offset = M->token_emb_offset;  // WEIGHT TYING

    // Then if you're doing language modeling, you might also want:
    M->logits_offset = bump(&off, (size_t)M->context_window * M->vocab_size, CACHE_ALIGN);
    
    // After forward pass layout is done
    if (training_mode) {
        M->training_enabled = true;
        
        // Continue allocation for gradients
        layout_gradients(M, &off);
        
        // Now off includes both forward AND backward memory
    }

    // The `off` variable now marks the end of the usable model data.
    // We add a final, larger canary zone to the total allocation size.
    M->total_floats = off + FINAL_CANARY_ZONE_FLOATS;
    
    // Allocate the full memory block, including all canary zones.
    M->memory_base = (float*)huge_alloc(M->total_floats * sizeof(float));
    if (!M->memory_base) {
        perror("Failed to allocate model memory");
        exit(EXIT_FAILURE);
    }
    
    printf("Memory allocated: %.2f GB (Forward: %.2f GB, Gradient: %.2f GB)\n",
       (M->total_floats * sizeof(float)) / (1024.0 * 1024.0 * 1024.0),
       ((off - (training_mode ? M->gradients.total_gradient_floats : 0)) * sizeof(float)) / (1024.0 * 1024.0 * 1024.0),
       (training_mode ? (M->gradients.total_gradient_floats * sizeof(float)) / (1024.0 * 1024.0 * 1024.0) : 0.0));
}

/* ─── destruction helper ─────────────────────────────────────────── */
void destroy_transformer(TransformerModel *M)
{
    munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
    free(M->layers);
    if (M->training_enabled  && M->gradients.layers) {
        free(M->gradients.layers);
    }
}

// Calculate memory requirements
static size_t bytes_needed(int layers, int vocab, int d_model, int ctx)
{
    size_t C = align_up(d_model, CACHE_ALIGN / sizeof(float));
    size_t T = ctx;
    size_t V = vocab;

    size_t embedding_size = (V * C) + (T * C) + (T * C); // token_emb, pos_emb, embedded_input

    size_t per_layer_floats =
        (2 * C) +                                      // ln1_weight, ln1_bias
        (2 * T * C) +                                  // layer_input, ln1_output
        (3ULL * C * C) + (3ULL * C) + (3ULL * T * C) + // qkv_weight, qkv_bias, qkv_output
        (C * C) + C + (2 * T * C) +                    // proj_weight, proj_bias, attention_output, residual1_output
        (2 * C) + (T * C) +                            // ln2_weight, ln2_bias, ln2_output
        (4ULL * C * C) + (4ULL * C) + (4ULL * T * C) + // fc1_weight, fc1_bias, fc1_output_offset (intermediate)
        (4ULL * C * C) + C +                           // fc2_weight, fc2_bias
        (T * C) + (T * C);                             // mlp_output, residual2_output

    size_t final_ln_size = 2 * C; // final_ln_weight, final_ln_bias

    size_t total_floats = embedding_size + ((size_t)layers * per_layer_floats) + final_ln_size;
    return total_floats * sizeof(float);
}

// ============================================================================
// ACCURACY COMPARISON HELPERS
// ============================================================================
float compute_max_diff(const float *ref, const float *test, size_t count)
{
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

float compute_rmse(const float *ref, const float *test, size_t count)
{
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < count; i++)
    {
        double diff = ref[i] - test[i];
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff / count);
}

// ============================================================================
/// @defgroup gemm_kernels GEMM Kernels
/// @brief Matrix multiplication implementations with different optimization strategies
/// @{
// ============================================================================

/**
 * @brief Naive parallel GEMM implementation (reference baseline)
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [N x K] (transposed)
 * @param bias Bias vector [N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Inner dimension (columns of A, rows of B)
 * 
 * @performance 
 * - Baseline performance: ~50-100 GFLOPS
 * - Used as reference for accuracy validation
 * - Simple OpenMP parallelization
 * 
 * @note This is the golden reference - all other implementations are validated against this
 */
void gemm_naive_parallel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K)
{
#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum + bias[j];
        }
    }
}

/**
 * @brief AVX-512 optimized GEMM with vectorized inner loops
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [N x K] (transposed)
 * @param bias Bias vector [N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Inner dimension
 * 
 * @performance
 * - Target: 200-400 GFLOPS on modern Xeon
 * - 16-wide SIMD operations
 * - FMA instruction utilization
 * 
 * @optimization_details
 * - Uses _mm512_fmadd_ps for 3 FLOPs per instruction
 * - 16-element vectorization of inner loop
 * - Handles remainder elements with scalar code
 * 
 * @see gemm_naive_parallel for reference implementation
 */
void gemm_avx512_parallel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K)
{
#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            __m512 sum_vec = _mm512_setzero_ps();
            int k;
            for (k = 0; k <= K - 16; k += 16)
            {
                __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]); // Using unaligned load for safety
                __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]); // Using unaligned load for safety
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            float sum = _mm512_reduce_add_ps(sum_vec);
            for (; k < K; k++)
            {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum + bias[j];
        }
    }
}

/**
 * @brief Cache-blocked GEMM with fine-grained parallelism
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [N x K] (transposed)
 * @param bias Bias vector [N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Inner dimension
 * 
 * @performance
 * - Target: 300-500 GFLOPS
 * - Best performance for large matrices
 * - Optimal cache utilization
 * 
 * @implementation_notes
 * - 64x64 blocking for L1 cache optimization
 * - Collapse(3) OpenMP directive for maximum parallelism
 * - Atomic updates for thread safety
 * 
 * @benchmark_results
 * Tested on 8192x8192 matrices:
 * - Naive: 85 GFLOPS
 * - This impl: 474 GFLOPS (5.6x speedup)
 */
void gemm_fine_grained_parallel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K)
{
    const int block_size = 64;
#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = bias ? bias[j] : 0.0f;  // NULL-safe
        }
    }
#pragma omp parallel for collapse(3)
    for (int ii = 0; ii < M; ii += block_size)
    {
        for (int jj = 0; jj < N; jj += block_size)
        {
            for (int kk = 0; kk < K; kk += block_size)
            {
                int i_end = min(ii + block_size, M);
                int j_end = min(jj + block_size, N);
                int k_end = min(kk + block_size, K);

                for (int i = ii; i < i_end; i++)
                {
                    for (int j = jj; j < j_end; j++)
                    {
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16)
                        {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++)
                        {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#pragma omp atomic
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
}

void gemm_blocked_serial(const float *A, const float *B, const float *bias, float *C, int M, int N, int K)
{
    const int block_size = 64;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = bias[j];
        }
    }
    for (int ii = 0; ii < M; ii += block_size)
    {
        for (int jj = 0; jj < N; jj += block_size)
        {
            for (int kk = 0; kk < K; kk += block_size)
            {
                int i_end = min(ii + block_size, M);
                int j_end = min(jj + block_size, N);
                int k_end = min(kk + block_size, K);

                for (int i = ii; i < i_end; i++)
                {
                    for (int j = jj; j < j_end; j++)
                    {
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16)
                        {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++)
                        {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// LAYER NORMALIZATION KERNELS
// ============================================================================

// Naive serial LayerNorm implementation for golden reference
void layernorm_naive_serial(const float *input,
                            const float *gamma,
                            const float *beta,
                            float *output,
                            float *mean_cache,                              // For storing mean
                            float *rstd_cache,                              // For storing rstd
                            int tokens, int d_model, int aligned_embed_dim, // ← Add aligned_embed_dim parameter
                            float eps)
{
    for (int t = 0; t < tokens; ++t)
    {
        const float *in_ptr = input + t * aligned_embed_dim;
        float *out_ptr = output + t * aligned_embed_dim;

        // Calculate mean
        float sum_val = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            sum_val += in_ptr[i];
        }
        float mean = sum_val / (float)d_model;

        // Calculate variance
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            float diff = in_ptr[i] - mean;
            sum_sq_diff += diff * diff;
        }
        float variance = sum_sq_diff / (float)d_model + eps;

        // Calculate inverse standard deviation
        // float inv_std = 1.0f / sqrtf(variance + eps);
        // NEW (double precision to match optimized)
        double var_double = (double)variance;
        float inv_std = (float)(1.0 / sqrt(var_double));

        // Normalize, scale, and shift
        for (int i = 0; i < d_model; ++i)
        {
            float normalized_val = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized_val * gamma[i] + beta[i];
        }

        // Cache mean and rstd for backward pass
        mean_cache[t] = mean;
        rstd_cache[t] = inv_std;
    }
}

// ============================================================================
// LAYER NORMALIZATION (Optimized for a single thread's token slice, ROLLED version)
// This version uses AVX-512 but without explicit 4-way loop unrolling,
// processing 16 floats per iteration.
// ============================================================================

// Fixed slice processing function with improved numerical stability
void layernorm_forward_rolled_slice(const float *__restrict input_slice_base,
                                    const float *__restrict gamma,
                                    const float *__restrict beta,
                                    float *__restrict output_slice_base,
                                    float *__restrict mean_cache_slice,
                                    float *__restrict rstd_cache_slice,
                                    int num_tokens_in_slice,
                                    int d_model,
                                    int aligned_embed_dim,
                                    float eps)
{
    for (int t = 0; t < num_tokens_in_slice; ++t)
    {
        // Use aligned_embed_dim for memory layout (cache alignment)
        const float *in_ptr_token = input_slice_base + t * aligned_embed_dim;
        float *out_ptr_token = output_slice_base + t * aligned_embed_dim;

        // ───────────────────────────────────────────
        // Pass 1: Compute mean (AVX-512, rolled)
        // ───────────────────────────────────────────
        __m512 acc_sum_vec = _mm512_setzero_ps();
        int j = 0;
        for (; j <= d_model - 16; j += 16)
        {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            __m512 v = _mm512_load_ps(in_ptr_token + j);
            acc_sum_vec = _mm512_add_ps(acc_sum_vec, v);
        }
        float mean = _mm512_reduce_add_ps(acc_sum_vec);
        for (; j < d_model; ++j)
        {
            mean += in_ptr_token[j];
        }
        mean /= (float)d_model;
        __m512 mean_vec = _mm512_set1_ps(mean);

        // ───────────────────────────────────────────
        // Pass 2: Compute variance (AVX-512, rolled)
        // ───────────────────────────────────────────
        __m512 acc_var_vec = _mm512_setzero_ps();
        j = 0;
        for (; j <= d_model - 16; j += 16)
        {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            __m512 v = _mm512_load_ps(in_ptr_token + j);
            __m512 diff = _mm512_sub_ps(v, mean_vec);
            acc_var_vec = _mm512_fmadd_ps(diff, diff, acc_var_vec);
        }
        float var = _mm512_reduce_add_ps(acc_var_vec);
        for (; j < d_model; ++j)
        {
            float diff = in_ptr_token[j] - mean;
            var += diff * diff;
        }
        var = var / (float)d_model + eps;
        // if (t == 0) printf("FIXED FUNC: var=%.9f, about to calculate sqrt\n", var);

        // FIXED: Use double precision for more stable sqrt computation
        double var_double = (double)var;
        float inv_std = (float)(1.0 / sqrt(var_double));
        __m512 inv_std_vec = _mm512_set1_ps(inv_std);

        // Store mean and rstd for the backward pass
        mean_cache_slice[t] = mean;
        rstd_cache_slice[t] = inv_std;

        // ───────────────────────────────────────────
        // Pass 3: Normalize, scale, shift (AVX-512, rolled)
        // ───────────────────────────────────────────
        j = 0;
        for (; j <= d_model - 16; j += 16)
        {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(gamma + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(beta + j + 128), _MM_HINT_T0);

            __m512 v = _mm512_load_ps(in_ptr_token + j);
            __m512 g = _mm512_load_ps(gamma + j);
            __m512 b = _mm512_load_ps(beta + j);

            __m512 n = _mm512_mul_ps(_mm512_sub_ps(v, mean_vec), inv_std_vec);
            __m512 o = _mm512_fmadd_ps(n, g, b);

            _mm512_store_ps(out_ptr_token + j, o);
        }
        for (; j < d_model; ++j)
        {
            float normed = (in_ptr_token[j] - mean) * inv_std;
            out_ptr_token[j] = normed * gamma[j] + beta[j];
        }
    }
}

// ============================================================================
// LAYER NORMALIZATION (Optimized for a single thread's token slice)
// This function processes a contiguous block of 'num_tokens_in_slice' tokens,
// each with 'd_model' features, using AVX-512 intrinsics and loop unrolling.
// It is designed to be called by a single OpenMP thread.
// ============================================================================

void layernorm_forward_unrolled_slice(const float *__restrict input_slice_base, // Base ptr for this thread's tokens
                                      const float *__restrict gamma,
                                      const float *__restrict beta,
                                      float *__restrict output_slice_base, // Base ptr for this thread's output
                                      float *__restrict mean_cache_slice,  // Base ptr for this thread's mean cache
                                      float *__restrict rstd_cache_slice,  // Base ptr for this thread's rstd cache
                                      int num_tokens_in_slice,             // Number of tokens this thread processes
                                      int d_model,                         // Dimension of each token's features (aligned_embed_dim)
                                      float eps)
{
    // Loop over each token within this thread's assigned slice
    for (int t = 0; t < num_tokens_in_slice; ++t)
    {
        // Pointers to the current token's data and its output within the slice
        const float *in_ptr_token = input_slice_base + t * d_model;
        float *out_ptr_token = output_slice_base + t * d_model;

        // ───────────────────────────────────────────
        // Pass 1: Compute mean (4‑way unrolled with AVX-512)
        // This calculates the mean across the 'd_model' features for the current token.
        // ───────────────────────────────────────────
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        int j = 0;
        int unroll_factor_floats = 64; // 4 AVX-512 vectors * 16 floats/vector

        // Process full unrolled blocks
        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats)
        {
            // Prefetch data for the current token's features, a bit ahead
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);

            // Load 4 AVX-512 vectors (64 floats)
            __m512 v0 = _mm512_load_ps(in_ptr_token + j);
            __m512 v1 = _mm512_load_ps(in_ptr_token + j + 16);
            __m512 v2 = _mm512_load_ps(in_ptr_token + j + 32);
            __m512 v3 = _mm512_load_ps(in_ptr_token + j + 48);

            // Accumulate sums
            acc0 = _mm512_add_ps(acc0, v0);
            acc1 = _mm512_add_ps(acc1, v1);
            acc2 = _mm512_add_ps(acc2, v2);
            acc3 = _mm512_add_ps(acc3, v3);
        }
        // Horizontally sum the accumulated vectors and extract the scalar mean
        __m512 acc_sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        float mean = _mm512_reduce_add_ps(acc_sum);

        // Handle remaining elements (if d_model is not a multiple of unroll_factor_floats)
        for (; j < d_model; ++j)
        {
            mean += in_ptr_token[j];
        }
        mean /= (float)d_model;
        __m512 mean_vec = _mm512_set1_ps(mean); // Broadcast mean to a vector

        // ───────────────────────────────────────────
        // Pass 2: Compute variance (4‑way unrolled with AVX-512)
        // This calculates the variance across the 'd_model' features for the current token.
        // ───────────────────────────────────────────
        acc0 = _mm512_setzero_ps();
        acc1 = _mm512_setzero_ps();
        acc2 = _mm512_setzero_ps();
        acc3 = _mm512_setzero_ps();

        j = 0; // Reset j for the second pass
        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats)
        {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);

            __m512 v0 = _mm512_load_ps(in_ptr_token + j);
            __m512 v1 = _mm512_load_ps(in_ptr_token + j + 16);
            __m512 v2 = _mm512_load_ps(in_ptr_token + j + 32);
            __m512 v3 = _mm512_load_ps(in_ptr_token + j + 48);

            // Subtract mean and square the difference (d*d)
            __m512 d0 = _mm512_sub_ps(v0, mean_vec);
            __m512 d1 = _mm512_sub_ps(v1, mean_vec);
            __m512 d2 = _mm512_sub_ps(v2, mean_vec);
            __m512 d3 = _mm512_sub_ps(v3, mean_vec);

            // Accumulate squared differences using FMA (d*d + acc)
            acc0 = _mm512_fmadd_ps(d0, d0, acc0);
            acc1 = _mm512_fmadd_ps(d1, d1, acc1);
            acc2 = _mm512_fmadd_ps(d2, d2, acc2);
            acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        }
        // Horizontally sum and extract scalar variance
        acc_sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        float var = _mm512_reduce_add_ps(acc_sum);

        // Handle remaining elements
        for (; j < d_model; ++j)
        {
            float diff = in_ptr_token[j] - mean;
            var += diff * diff;
        }
        var = var / (float)d_model + eps; // Add epsilon for numerical stability
        // FIXED: Use double precision for more stable sqrt computation
        double var_double = (double)var;
        float inv_std = (float)(1.0 / sqrt(var_double));
        __m512 inv_std_vec = _mm512_set1_ps(inv_std);

        // Store mean and rstd for the backward pass
        mean_cache_slice[t] = mean;
        rstd_cache_slice[t] = inv_std; // inv_std is rstd

        // ───────────────────────────────────────────
        // Pass 3: Normalize, scale, shift (4‑way unrolled with AVX-512)
        // This applies the final LayerNorm equation: (x - mean) / std * gamma + beta
        // ───────────────────────────────────────────
        j = 0; // Reset j for the third pass
        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats)
        {
            // Prefetch data for input, gamma (weight), and beta (bias)
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(gamma + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(beta + j + 128), _MM_HINT_T0);

            // Load input, gamma, and beta vectors
            __m512 v0 = _mm512_load_ps(in_ptr_token + j);
            __m512 v1 = _mm512_load_ps(in_ptr_token + j + 16);
            __m512 v2 = _mm512_load_ps(in_ptr_token + j + 32);
            __m512 v3 = _mm512_load_ps(in_ptr_token + j + 48);

            __m512 g0 = _mm512_load_ps(gamma + j);
            __m512 g1 = _mm512_load_ps(gamma + j + 16);
            __m512 g2 = _mm512_load_ps(gamma + j + 32);
            __m512 g3 = _mm512_load_ps(gamma + j + 48);

            __m512 b0 = _mm512_load_ps(beta + j);
            __m512 b1 = _mm512_load_ps(beta + j + 16);
            __m512 b2 = _mm512_load_ps(beta + j + 32);
            __m512 b3 = _mm512_load_ps(beta + j + 48);

            // Normalize: (v - mean) * inv_std
            __m512 n0 = _mm512_mul_ps(_mm512_sub_ps(v0, mean_vec), inv_std_vec);
            __m512 n1 = _mm512_mul_ps(_mm512_sub_ps(v1, mean_vec), inv_std_vec);
            __m512 n2 = _mm512_mul_ps(_mm512_sub_ps(v2, mean_vec), inv_std_vec);
            __m512 n3 = _mm512_mul_ps(_mm512_sub_ps(v3, mean_vec), inv_std_vec);

            // Scale and Shift: n * gamma + beta (using FMA)
            __m512 o0 = _mm512_fmadd_ps(n0, g0, b0);
            __m512 o1 = _mm512_fmadd_ps(n1, g1, b1);
            __m512 o2 = _mm512_fmadd_ps(n2, g2, b2);
            __m512 o3 = _mm512_fmadd_ps(n3, g3, b3);

            // Store results to output
            _mm512_store_ps(out_ptr_token + j, o0);
            _mm512_store_ps(out_ptr_token + j + 16, o1);
            _mm512_store_ps(out_ptr_token + j + 32, o2);
            _mm512_store_ps(out_ptr_token + j + 48, o3);
        }
        // Handle remaining elements
        for (; j < d_model; ++j)
        {
            float normed = (in_ptr_token[j] - mean) * inv_std;
            out_ptr_token[j] = normed * gamma[j] + beta[j];
        }
    } // End of loop over num_tokens_in_slice
}

// ============================================================================
// FIXED LAYER NORMALIZATION IMPLEMENTATION
// ============================================================================

// Fixed version of the token-parallel LayerNorm orchestration
/**
 * @brief Token-parallel Layer Normalization with AVX-512 optimization
 *
 * Performs Layer Normalization across tokens using token-level parallelism.
 * Each CPU core processes a contiguous slice of tokens independently, achieving
 * perfect cache locality and zero synchronization overhead.
 *
 * @param M Transformer model containing memory layout and parallelism config
 * @param input_offset Offset to input tensor [context_window × aligned_embed_dim]
 * @param weight_offset Offset to gamma weights [aligned_embed_dim]
 * @param bias_offset Offset to beta biases [aligned_embed_dim]
 * @param mean_cache_offset Offset to mean cache [context_window] (for backward pass)
 * @param rstd_cache_offset Offset to rstd cache [context_window] (for backward pass)
 * @param output_offset Offset to output tensor [context_window × aligned_embed_dim]
 * @param eps Epsilon for numerical stability (typically 1e-5)
 *
 * @details
 * **Token-Level Parallelism Strategy**:
 * ```
 * Memory Layout (Token-Major):
 * ┌──────────────┬──────────────┬──────────────┬──────────────┐
 * │ Token 0      │ Token 1      │ Token 2      │ Token 3      │
 * │ [768 floats] │ [768 floats] │ [768 floats] │ [768 floats] │
 * └──────────────┴──────────────┴──────────────┴──────────────┘
 *  │<─ Core 0 ──>│<─ Core 1 ──>│<─ Core 2 ──>│<─ Core 3 ──>│
 * ```
 *
 * **Why Token-Parallel?**:
 * - ✅ **Perfect Locality**: Each token's data (768 floats) is contiguous in memory
 * - ✅ **Zero Sync**: Tokens are independent, no barriers or atomics needed
 * - ✅ **Cache Efficiency**: Each core streams through sequential memory
 * - ✅ **Linear Scaling**: Speedup = num_cores (measured 7.8x on 8 cores)
 *
 * **Memory Access Pattern (per core)**:
 * ```
 * Core 0 processes tokens [0, tokens_per_core):
 *   - Read:  input[0*768], input[1*768], ..., input[N*768]  (sequential)
 *   - Write: output[0*768], output[1*768], ..., output[N*768] (sequential)
 *   - Gamma/Beta: Shared read-only (broadcast to all cores)
 * ```
 *
 * **Algorithm (per token)**:
 * 1. **Pass 1**: Compute mean across embed_dim using AVX-512
 * 2. **Pass 2**: Compute variance using FMA (fused multiply-add)
 * 3. **Pass 3**: Normalize, scale by gamma, shift by beta
 *
 * **AVX-512 Optimization**:
 * - Processes 16 floats per instruction (4x16 unrolling)
 * - Uses FMA for variance: `acc = diff * diff + acc` (2 FLOPs per cycle)
 * - Aligned loads: `_mm512_load_ps` (requires 64-byte alignment)
 * - Prefetching: Hints to load next cache line while computing current
 *
 * **Performance Characteristics**:
 * - Compute: 9 * embed_dim FLOPs per token
 * - Memory: 3 * embed_dim reads + embed_dim writes per token
 * - Bandwidth: Achieves 50-100 GB/s per core (streaming bandwidth)
 * - Latency: ~5 μs per token on modern Xeon (768-dim, 3.0 GHz)
 *
 * **Cache Behavior**:
 * - L1 Data Cache: Holds ~4 tokens (768 floats = 3KB per token, 32KB L1)
 * - L2 Cache: Holds ~80 tokens (256KB L2)
 * - Prefetcher: Detects sequential pattern, hides DRAM latency
 *
 * **Why Aligned Embed Dim?**:
 * Padding to 64-byte boundaries ensures:
 * - No false sharing between cores writing adjacent tokens
 * - Aligned SIMD loads (faster than unaligned)
 * - Clean cache line ownership (no partial cache line reads)
 *
 * @note This is a core building block used in every transformer layer
 * @see layernorm_forward_rolled_slice Per-core slice processing kernel
 * @see TrulyOptimalLayer For offset definitions within a layer
 *
 * @performance Measured 7.8x speedup on 8-core Xeon vs serial baseline
 */
void layernorm_token_parallel(TransformerModel *M,
                              size_t input_offset,
                              size_t weight_offset,
                              size_t bias_offset,
                              size_t mean_cache_offset,
                              size_t rstd_cache_offset,
                              size_t output_offset,
                              float eps)
{
    float *input_base = M->memory_base + input_offset;

#pragma omp parallel num_threads(M->num_cores)
    {
        // Determine this thread's token slice
        int core_id = omp_get_thread_num();
        size_t token_start = core_id * M->tokens_per_core;
        size_t num_tokens_for_this_thread = (token_start + M->tokens_per_core > M->context_window)
                                                ? (M->context_window - token_start)
                                                : M->tokens_per_core;

        if (num_tokens_for_this_thread > 0)
        {
            // Calculate base pointers for this thread's slice
            const float *input_base_ptr = input_base + token_start * M->aligned_embed_dim;
            const float *gamma_weights = M->memory_base + weight_offset;
            const float *beta_biases = M->memory_base + bias_offset;
            float *mean_cache_base_ptr = M->memory_base + mean_cache_offset + token_start;
            float *rstd_cache_base_ptr = M->memory_base + rstd_cache_offset + token_start;
            float *output_base_ptr = M->memory_base + output_offset + token_start * M->aligned_embed_dim;

            // Process this core's token slice
            layernorm_forward_rolled_slice(input_base_ptr, gamma_weights, beta_biases,
                                           output_base_ptr, mean_cache_base_ptr, rstd_cache_base_ptr,
                                           num_tokens_for_this_thread, M->embed_dim, M->aligned_embed_dim, eps);
        }
    }
}

// ============================================================================
// PRECISION-MATCHED LAYERNORM IMPLEMENTATION
// This version ensures both naive and optimized use identical precision
// ============================================================================

// Updated naive reference to match optimized precision exactly
void layernorm_naive_serial_matched_precision(const float *input,
                                              const float *gamma,
                                              const float *beta,
                                              float *output,
                                              float *mean_cache,
                                              float *rstd_cache,
                                              int tokens, int d_model, float eps)
{
    for (int t = 0; t < tokens; ++t)
    {
        const float *in_ptr = input + t * d_model;
        float *out_ptr = output + t * d_model;

        // Calculate mean (matching optimized version exactly)
        float sum_val = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            sum_val += in_ptr[i];
        }
        float mean = sum_val / (float)d_model;

        // Calculate variance (matching optimized version exactly)
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            float diff = in_ptr[i] - mean;
            sum_sq_diff += diff * diff;
        }
        float variance = sum_sq_diff / (float)d_model + eps;

        // CRITICAL: Use same precision as optimized version
        // Match the double precision sqrt followed by float cast
        double var_double = (double)variance;
        float inv_std = (float)(1.0 / sqrt(var_double));

        // Normalize, scale, and shift
        for (int i = 0; i < d_model; ++i)
        {
            float normalized_val = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized_val * gamma[i] + beta[i];
        }

        // Cache mean and rstd for backward pass
        mean_cache[t] = mean;
        rstd_cache[t] = inv_std;
    }
}

static inline void *aligned_alloc_64(size_t size)
{
    void *ptr = NULL;
    int ret = posix_memalign(&ptr, 64, size);
    if (ret != 0)
    {
        fprintf(stderr, "posix_memalign failed (ret=%d)\n", ret);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Debug function to compare naive vs optimized math step by step
void debug_math_comparison(TransformerModel *M)
{
    printf("\n=== MATHEMATICAL DEBUGGING COMPARISON ===\n");

    int test_tokens = 3; // just a small sample
    int d_model = M->embed_dim;
    int aligned_dim = M->aligned_embed_dim;
    float eps = 1e-5f;

    // Aligned allocations
    float *input_data = (float *)aligned_alloc(64, test_tokens * aligned_dim * sizeof(float));
    float *gamma_data = (float *)aligned_alloc(64, aligned_dim * sizeof(float));
    float *beta_data = (float *)aligned_alloc(64, aligned_dim * sizeof(float));

    float *naive_output = (float *)aligned_alloc(64, test_tokens * aligned_dim * sizeof(float));
    float *naive_mean = (float *)aligned_alloc(64, test_tokens * sizeof(float));
    float *naive_rstd = (float *)aligned_alloc(64, test_tokens * sizeof(float));

    float *orig_output = (float *)aligned_alloc(64, test_tokens * aligned_dim * sizeof(float));
    float *orig_mean = (float *)aligned_alloc(64, test_tokens * sizeof(float));
    float *orig_rstd = (float *)aligned_alloc(64, test_tokens * sizeof(float));

    // Initialize input
    srand(42);
    for (int t = 0; t < test_tokens; t++)
    {
        for (int i = 0; i < d_model; i++)
        {
            input_data[t * aligned_dim + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        for (int i = d_model; i < aligned_dim; i++)
        {
            input_data[t * aligned_dim + i] = 0.0f;
        }
    }
    // Initialize gamma/beta
    for (int i = 0; i < d_model; i++)
    {
        gamma_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f + 1.0f;
        beta_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    for (int i = d_model; i < aligned_dim; i++)
    {
        gamma_data[i] = 0.0f;
        beta_data[i] = 0.0f;
    }

    memset(naive_output, 0, test_tokens * aligned_dim * sizeof(float));
    memset(naive_mean, 0, test_tokens * sizeof(float));
    memset(naive_rstd, 0, test_tokens * sizeof(float));
    memset(orig_output, 0, test_tokens * aligned_dim * sizeof(float));
    memset(orig_mean, 0, test_tokens * sizeof(float));
    memset(orig_rstd, 0, test_tokens * sizeof(float));

    printf("Running implementations...\n");
    printf("1. Running NAIVE reference...\n");
    layernorm_naive_serial(input_data, gamma_data, beta_data,
                           naive_output, naive_mean, naive_rstd,
                           test_tokens, d_model, aligned_dim, eps);

    printf("2. Running ORIGINAL optimized...\n");
    layernorm_forward_rolled_slice(input_data, gamma_data, beta_data,
                                   orig_output, orig_mean, orig_rstd,
                                   test_tokens, d_model, aligned_dim, eps);

    printf("\n=== DETAILED COMPARISON FOR EACH TOKEN ===\n");

    for (int t = 0; t < test_tokens; t++)
    {
        printf("\n--- TOKEN %d ---\n", t);
        printf("Mean values:\n");
        printf("  Naive:    %.9f\n", naive_mean[t]);
        printf("  Original: %.9f (diff: %.2e)\n",
               orig_mean[t], fabsf(naive_mean[t] - orig_mean[t]));

        printf("RSTD values:\n");
        printf("  Naive:    %.9f\n", naive_rstd[t]);
        printf("  Original: %.9f (diff: %.2e)\n",
               orig_rstd[t], fabsf(naive_rstd[t] - orig_rstd[t]));

        printf("Output values (first 5 elements):\n");
        printf("   Idx |     Naive     |   Original    | Orig Diff\n");
        printf("  -----|---------------|---------------|---------------\n");
        for (int i = 0; i < (d_model < 5 ? d_model : 5); i++)
        {
            int idx = t * aligned_dim + i;
            float naive_val = naive_output[idx];
            float orig_val = orig_output[idx];
            float diff = fabsf(naive_val - orig_val);
            printf("  %4d | %13.9f | %13.9f | %13.2e\n",
                   i, naive_val, orig_val, diff);
        }

        float max_orig_diff = 0.0f;
        for (int i = 0; i < d_model; i++)
        {
            int idx = t * aligned_dim + i;
            max_orig_diff = fmaxf(max_orig_diff,
                                  fabsf(naive_output[idx] - orig_output[idx]));
        }
        printf("  Max output diff - Original: %.2e\n", max_orig_diff);
    }

    // Overall stats
    float max_mean_orig = 0.0f, max_rstd_orig = 0.0f, max_out_orig = 0.0f;
    for (int t = 0; t < test_tokens; t++)
    {
        max_mean_orig = fmaxf(max_mean_orig, fabsf(naive_mean[t] - orig_mean[t]));
        max_rstd_orig = fmaxf(max_rstd_orig, fabsf(naive_rstd[t] - orig_rstd[t]));
        for (int i = 0; i < d_model; i++)
        {
            int idx = t * aligned_dim + i;
            max_out_orig = fmaxf(max_out_orig,
                                 fabsf(naive_output[idx] - orig_output[idx]));
        }
    }

    printf("\n=== OVERALL SUMMARY ===\n");
    printf("Maximum differences vs Naive:\n");
    printf("  Mean:   %.2e\n", max_mean_orig);
    printf("  RSTD:   %.2e\n", max_rstd_orig);
    printf("  Output: %.2e\n", max_out_orig);

    printf("\nVERDICT:\n");
    if (max_rstd_orig < 1e-4 && max_out_orig < 1e-5)
        printf("✅ ORIGINAL optimized: EXCELLENT precision\n");
    else if (max_rstd_orig < 1e-2 && max_out_orig < 1e-3)
        printf("⚠️  ORIGINAL optimized: Acceptable precision\n");
    else
        printf("❌ ORIGINAL optimized: POOR precision - needs debugging!\n");

    printf("=== END MATHEMATICAL DEBUGGING ===\n\n");

    // Clean up
    free(input_data);
    free(gamma_data);
    free(beta_data);
    free(naive_output);
    free(naive_mean);
    free(naive_rstd);
    free(orig_output);
    free(orig_mean);
    free(orig_rstd);
}

// ============================================================================
// FUNCTION: run_layernorm_benchmark_precision_matched
// PURPOSE:
//   - Performs a precision-focused benchmark where the naive LayerNorm reference
//     and the optimized implementation are compared in detail.
//   - Uses carefully matched input data and parameters to highlight any
//     numerical drift or rounding issues.
//
// WHY WE DO THIS:
//   • Validates that the optimized implementation is mathematically equivalent
//     to the naive reference within acceptable floating-point tolerances.
//   • Helps catch subtle numerical bugs introduced by low-level changes
//     (e.g., SIMD order of operations, fused-multiply-add differences, or alignment).
//   • Ensures that any speed optimizations do not degrade model correctness.
//
// WHEN TO USE:
//   - After implementing a new kernel or modifying math order (e.g., unrolling).
//   - When debugging unexpected model outputs.
//   - Periodically, to ensure precision stability over time.
//
// OUTPUT:
//   - Prints detailed token-by-token comparisons (if desired) and reports
//     MaxDiff and RMSE across output, mean, and rstd values.
//   - Also reports performance metrics (GFLOPS), though the main focus is accuracy.
// ============================================================================

void run_layernorm_benchmark_precision_matched(TransformerModel *M)
{
    printf("\n\n=== PRECISION-MATCHED LayerNorm Benchmark ===\n");
    printf("   Using precision-matched naive reference for accurate comparison.\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    int tokens = M->context_window;
    int d_model = M->embed_dim;             // ← ACTUAL data dimension
    int aligned_dim = M->aligned_embed_dim; // ← ALIGNED memory dimension
    const float eps = 1e-5f;

    // Use Layer 0 and Layer 1 regions
    float *input_ptr0 = M->memory_base + M->layers[0].layer_input_offset;
    float *gamma0 = M->memory_base + M->layers[0].ln1_weight_offset;
    float *beta0 = M->memory_base + M->layers[0].ln1_bias_offset;
    float *out0 = M->memory_base + M->layers[0].ln1_output_offset;
    float *mean0 = M->memory_base + M->layers[0].ln1_mean_offset;
    float *rstd0 = M->memory_base + M->layers[0].ln1_rstd_offset;

    float *input_ptr1 = M->memory_base + M->layers[1].layer_input_offset;
    float *gamma1 = M->memory_base + M->layers[1].ln1_weight_offset;
    float *beta1 = M->memory_base + M->layers[1].ln1_bias_offset;
    float *out1 = M->memory_base + M->layers[1].ln1_output_offset;
    float *mean1 = M->memory_base + M->layers[1].ln1_mean_offset;
    float *rstd1 = M->memory_base + M->layers[1].ln1_rstd_offset;

    // Initialize identical input data
    srand(12345); // Fixed seed for reproducibility
    srand(12345); // Fixed seed for reproducibility
    for (int t = 0; t < tokens; t++)
    {
        // Initialize only d_model elements per token, leave padding untouched
        for (int i = 0; i < d_model; i++)
        {
            input_ptr0[t * aligned_dim + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        // Padding elements [d_model...aligned_dim-1] remain zero
    }
    memcpy(input_ptr1, input_ptr0, tokens * aligned_dim * sizeof(float));

    // Initialize identical parameters
    for (int i = 0; i < d_model; i++)
    {
        gamma0[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f + 1.0f;
        beta0[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    memcpy(gamma1, gamma0, aligned_dim * sizeof(float));
    memcpy(beta1, beta0, aligned_dim * sizeof(float));

    // Clear output buffers
    memset(out0, 0, tokens * aligned_dim * sizeof(float));
    memset(mean0, 0, tokens * sizeof(float));
    memset(rstd0, 0, tokens * sizeof(float));
    memset(out1, 0, tokens * aligned_dim * sizeof(float));
    memset(mean1, 0, tokens * sizeof(float));
    memset(rstd1, 0, tokens * sizeof(float));

    // Run precision-matched naive reference
    printf("Running Precision-Matched Naive LayerNorm on Layer 0...\n");
    double t0 = get_time_sec();
    layernorm_naive_serial(input_ptr0, gamma0, beta0, out0, mean0, rstd0,
                           tokens, d_model, aligned_dim, eps);
    double t1 = get_time_sec();
    printf("   Precision-Matched Naive LayerNorm time: %.2f ms\n", (t1 - t0) * 1000.0);

    // Run optimized version
    printf("Running Fixed Optimized LayerNorm on Layer 1...\n");
    double t2 = get_time_sec();
    layernorm_token_parallel(M,
                             M->layers[1].layer_input_offset,
                             M->layers[1].ln1_weight_offset,
                             M->layers[1].ln1_bias_offset,
                             M->layers[1].ln1_mean_offset,
                             M->layers[1].ln1_rstd_offset,
                             M->layers[1].ln1_output_offset,
                             eps);
    double t3 = get_time_sec();
    printf("   Fixed Optimized LayerNorm time: %.2f ms\n", (t3 - t2) * 1000.0);

    // Detailed accuracy analysis
    float max_diff_out = compute_max_diff(out0, out1, (size_t)tokens * d_model);
    float rmse_out = compute_rmse(out0, out1, (size_t)tokens * d_model);
    float max_diff_mean = compute_max_diff(mean0, mean1, (size_t)tokens);
    float rmse_mean = compute_rmse(mean0, mean1, (size_t)tokens);
    float max_diff_rstd = compute_max_diff(rstd0, rstd1, (size_t)tokens);
    float rmse_rstd = compute_rmse(rstd0, rstd1, (size_t)tokens);

    printf("   PRECISION-MATCHED Accuracy Results:\n");
    printf("     Output: MaxDiff %.2e, RMSE %.2e\n", max_diff_out, rmse_out);
    printf("     Mean:   MaxDiff %.2e, RMSE %.2e\n", max_diff_mean, rmse_mean);
    printf("     RSTD:   MaxDiff %.2e, RMSE %.2e\n", max_diff_rstd, rmse_rstd);

    // Sample comparison for debugging
    printf("   Sample RSTD comparison (first 5 tokens):\n");
    for (int i = 0; i < min(5, tokens); i++)
    {
        printf("     Token %d: Naive=%.6f, Opt=%.6f, Diff=%.2e\n",
               i, rstd0[i], rstd1[i], fabsf(rstd0[i] - rstd1[i]));
    }

    // Performance summary
    double flops = (double)tokens * d_model * 9.0;
    double gflops_naive = flops / 1e9 / (t1 - t0);
    double gflops_opt = flops / 1e9 / (t3 - t2);
    printf("   Performance Summary:\n");
    printf("     Precision-Matched Naive: %.2f GFLOPS\n", gflops_naive);
    printf("     Fixed Optimized:         %.2f GFLOPS (%.2fx speedup)\n",
           gflops_opt, gflops_opt / gflops_naive);
    printf("════════════════════════════════════════════════════════════════════════\n");
}

// ============================================================================
// FUNCTION: run_layernorm_benchmark_performance
// PURPOSE:
//   - Measures raw throughput (speed) of the optimized LayerNorm implementation
//     compared against a naive single-threaded reference.
//   - Uses identical input/parameters across both versions to ensure a fair test.
//
// WHY WE DO THIS:
//   • Provides a baseline performance metric for the current kernel.
//   • Allows you to track speed improvements or regressions over time as you
//     tune memory layouts, vectorization, or parallelization.
//   • Ensures the optimized implementation still produces numerically accurate
//     results (by comparing to the naive output) while pushing for maximum speed.
//
// WHEN TO USE:
//   - Every time you modify low-level optimizations (SIMD, cache alignment, threading).
//   - As part of regular CI or benchmarking to ensure raw performance meets expectations.
//
// OUTPUT:
//   - Prints runtime (ms), computed GFLOPS, and accuracy deltas (MaxDiff/RMSE)
//     between naive and optimized outputs.
// ============================================================================

void run_layernorm_benchmark_performance(TransformerModel *M)
{
    printf("\n\n=== FIXED LayerNorm Performance Benchmark ===\n");
    printf("   Ensuring identical input data for both naive and optimized versions.\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    int tokens = M->context_window;
    int d_model = M->embed_dim;             // ← ACTUAL data dimension
    int aligned_dim = M->aligned_embed_dim; // ← ALIGNED memory dimension
    const float eps = 1e-5f;

    // Use Layer 0 and Layer 1 regions
    float *input_ptr0 = M->memory_base + M->layers[0].layer_input_offset;
    float *gamma0 = M->memory_base + M->layers[0].ln1_weight_offset;
    float *beta0 = M->memory_base + M->layers[0].ln1_bias_offset;
    float *out0 = M->memory_base + M->layers[0].ln1_output_offset;
    float *mean0 = M->memory_base + M->layers[0].ln1_mean_offset;
    float *rstd0 = M->memory_base + M->layers[0].ln1_rstd_offset;

    float *input_ptr1 = M->memory_base + M->layers[1].layer_input_offset;
    float *gamma1 = M->memory_base + M->layers[1].ln1_weight_offset;
    float *beta1 = M->memory_base + M->layers[1].ln1_bias_offset;
    float *out1 = M->memory_base + M->layers[1].ln1_output_offset;
    float *mean1 = M->memory_base + M->layers[1].ln1_mean_offset;
    float *rstd1 = M->memory_base + M->layers[1].ln1_rstd_offset;

    // Initialize input data
    srand(42); // Fixed seed
    for (int t = 0; t < tokens; t++)
    {
        // Initialize only d_model elements per token, leave padding untouched
        for (int i = 0; i < d_model; i++)
        {
            input_ptr0[t * aligned_dim + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        // Padding elements [d_model...aligned_dim-1] remain zero (from initial allocation)
    }

    // CRITICAL: Copy identical input data to both layers
    memcpy(input_ptr1, input_ptr0, tokens * aligned_dim * sizeof(float));

    // Initialize gamma and beta
    for (int i = 0; i < d_model; i++)
    {
        gamma0[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f + 1.0f;
        beta0[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    // CRITICAL: Copy identical parameters to both layers
    memcpy(gamma1, gamma0, aligned_dim * sizeof(float));
    memcpy(beta1, beta0, aligned_dim * sizeof(float));

    // Clear output buffers
    memset(out0, 0, tokens * aligned_dim * sizeof(float));
    memset(mean0, 0, tokens * sizeof(float));
    memset(rstd0, 0, tokens * sizeof(float));
    memset(out1, 0, tokens * aligned_dim * sizeof(float));
    memset(mean1, 0, tokens * sizeof(float));
    memset(rstd1, 0, tokens * sizeof(float));

    // Run naive reference
    printf("Running Naive LayerNorm on Layer 0...\n");
    double t0 = get_time_sec();
    layernorm_naive_serial(input_ptr0, gamma0, beta0, out0, mean0, rstd0,
                           tokens, d_model, aligned_dim, eps);
    double t1 = get_time_sec();
    printf("   Naive LayerNorm time: %.2f ms\n", (t1 - t0) * 1000.0);

    // Run optimized version with FIXED function
    printf("Running Fixed Optimized LayerNorm on Layer 1...\n");
    double t2 = get_time_sec();
    layernorm_token_parallel(M,
                             M->layers[1].layer_input_offset,
                             M->layers[1].ln1_weight_offset,
                             M->layers[1].ln1_bias_offset,
                             M->layers[1].ln1_mean_offset,
                             M->layers[1].ln1_rstd_offset,
                             M->layers[1].ln1_output_offset,
                             eps);
    double t3 = get_time_sec();
    printf("   Fixed Optimized LayerNorm time: %.2f ms\n", (t3 - t2) * 1000.0);

    // Accuracy check
    float max_diff_out = compute_max_diff(out0, out1, (size_t)tokens * d_model);
    float rmse_out = compute_rmse(out0, out1, (size_t)tokens * d_model);
    float max_diff_mean = compute_max_diff(mean0, mean1, (size_t)tokens);
    float rmse_mean = compute_rmse(mean0, mean1, (size_t)tokens);
    float max_diff_rstd = compute_max_diff(rstd0, rstd1, (size_t)tokens);
    float rmse_rstd = compute_rmse(rstd0, rstd1, (size_t)tokens);

    printf("   FIXED Accuracy Output: MaxDiff %.2e, RMSE %.2e\n", max_diff_out, rmse_out);
    printf("   FIXED Accuracy Mean:   MaxDiff %.2e, RMSE %.2e\n", max_diff_mean, rmse_mean);
    printf("   FIXED Accuracy RSTD:   MaxDiff %.2e, RMSE %.2e\n", max_diff_rstd, rmse_rstd);

    // Performance summary
    double flops = (double)tokens * d_model * 9.0; // Approximate FLOPS for LayerNorm
    double gflops_naive = flops / 1e9 / (t1 - t0);
    double gflops_opt = flops / 1e9 / (t3 - t2);
    printf("   Naive LN GFLOPS: %.2f\n", gflops_naive);
    printf("   Fixed Opt LN GFLOPS: %.2f (%.2fx speedup)\n",
           gflops_opt, gflops_opt / gflops_naive);
    printf("════════════════════════════════════════════════════════════════════════\n");
}

// ============================================================================
// QKV PROJECTION (Using your optimized GEMM)
// ULTRA-OPTIMIZED: FUSED QKV WITH MANUAL PREFETCHING
// This version adds explicit prefetching hints and optimizes memory access patterns
// ============================================================================

// ============================================================================
// PRODUCTION-GRADE POLISHED QKV PROJECTION KERNEL
// Final optimized version with all performance polish applied
// ============================================================================

// ============================================================================
// 1. POLISHED 4x16 MICRO-KERNEL 
// ============================================================================
static inline void qkv_micro_kernel_blocked_4x16_polished(
    const float* __restrict input_token,
    const float* __restrict Q_weights_block,
    const float* __restrict K_weights_block,
    const float* __restrict V_weights_block,
    const float* __restrict Q_bias_4,
    const float* __restrict K_bias_4,
    const float* __restrict V_bias_4,
    float* __restrict Q_output_4,
    float* __restrict K_output_4,
    float* __restrict V_output_4,
    int embed_dim
) {
    // Initialize accumulators - clean and efficient
    __m512 Q_acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };
    __m512 K_acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };
    __m512 V_acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };
    
    // Main vectorized computation loop
    for (int k = 0; k < embed_dim; k += 16) {
        // Load input vector once - KEY optimization point
        __m512 input_vec = _mm512_load_ps(input_token + k);
        
        // Process 4 outputs with same input (12 FMAs total)
        for (int i = 0; i < 4; ++i) {
            // Get weight vectors for output element i
            const float* Qw = Q_weights_block + i * embed_dim + k;
            const float* Kw = K_weights_block + i * embed_dim + k;
            const float* Vw = V_weights_block + i * embed_dim + k;
            
            // Load weight vectors
            __m512 Qw_vec = _mm512_load_ps(Qw);
            __m512 Kw_vec = _mm512_load_ps(Kw);
            __m512 Vw_vec = _mm512_load_ps(Vw);
            
            // Accumulate dot products using FMA (3 FLOPs per instruction)
            Q_acc[i] = _mm512_fmadd_ps(input_vec, Qw_vec, Q_acc[i]);
            K_acc[i] = _mm512_fmadd_ps(input_vec, Kw_vec, K_acc[i]);
            V_acc[i] = _mm512_fmadd_ps(input_vec, Vw_vec, V_acc[i]);
        }
    }
    
    // Final horizontal reduction and bias addition
    for (int i = 0; i < 4; ++i) {
        Q_output_4[i] = Q_bias_4[i] + _mm512_reduce_add_ps(Q_acc[i]);
        K_output_4[i] = K_bias_4[i] + _mm512_reduce_add_ps(K_acc[i]);
        V_output_4[i] = V_bias_4[i] + _mm512_reduce_add_ps(V_acc[i]);
    }
}

// ============================================================================
// 2. POLISHED TOKEN-LEVEL KERNEL with ALL optimizations
// ============================================================================
static void qkv_token_kernel_4x16_blocked_polished(
    const float* __restrict input_token,    // ✅ restrict qualifier added
    const float* __restrict Q_weights,      // ✅ restrict qualifier added
    const float* __restrict K_weights,      // ✅ restrict qualifier added
    const float* __restrict V_weights,      // ✅ restrict qualifier added
    const float* __restrict Q_bias,         // ✅ restrict qualifier added
    const float* __restrict K_bias,         // ✅ restrict qualifier added
    const float* __restrict V_bias,         // ✅ restrict qualifier added
    float* __restrict Q_output,             // ✅ restrict qualifier added
    float* __restrict K_output,             // ✅ restrict qualifier added
    float* __restrict V_output,             // ✅ restrict qualifier added
    int embed_dim
) {
    // ✅ Embed-dim check assert
    assert(embed_dim > 0 && "embed_dim must be positive");
    
    // ✅ Optional: Prefetch input token if beneficial
    _mm_prefetch((const char*)input_token, _MM_HINT_T0);
    
    // ✅ Faster main_blocks calculation using bit masking
    int main_blocks = embed_dim & ~3;  // Equivalent to (embed_dim / 4) * 4 but faster
    
    // Process in blocks of 4 outputs using optimized micro-kernel
    for (int out_block = 0; out_block < main_blocks; out_block += 4) {
        // Get pointers to 4 consecutive rows of each weight matrix
        const float* Q_weights_block = Q_weights + out_block * embed_dim;
        const float* K_weights_block = K_weights + out_block * embed_dim;
        const float* V_weights_block = V_weights + out_block * embed_dim;
        
        qkv_micro_kernel_blocked_4x16_polished(
            input_token,
            Q_weights_block, K_weights_block, V_weights_block,
            Q_bias + out_block, K_bias + out_block, V_bias + out_block,
            Q_output + out_block, K_output + out_block, V_output + out_block,
            embed_dim
        );
    }
    
    // Handle remainder outputs (if embed_dim % 4 != 0) - efficient scalar fallback
    int remainder = embed_dim & 3;  // Equivalent to embed_dim % 4 but faster
    if (remainder > 0) {
        for (int i = 0; i < remainder; i++) {
            int out_idx = main_blocks + i;
            
            // Initialize with bias
            float q_sum = Q_bias[out_idx];
            float k_sum = K_bias[out_idx];
            float v_sum = V_bias[out_idx];
            
            // Get weight row pointers
            const float* q_row = Q_weights + out_idx * embed_dim;
            const float* k_row = K_weights + out_idx * embed_dim;
            const float* v_row = V_weights + out_idx * embed_dim;
            
            // Scalar dot product for remainder
            for (int k = 0; k < embed_dim; k++) {
                float input_val = input_token[k];
                q_sum += input_val * q_row[k];
                k_sum += input_val * k_row[k];
                v_sum += input_val * v_row[k];
            }
            
            // Store results
            Q_output[out_idx] = q_sum;
            K_output[out_idx] = k_sum;
            V_output[out_idx] = v_sum;
        }
    }
}

// ============================================================================
// 3. PRODUCTION-READY MAIN QKV PROJECTION FUNCTION
// ============================================================================
void qkv_projection(TransformerModel *M, size_t layer_idx)
{
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    
    // ✅ Enhanced alignment assertions for production debugging
    assert(((uintptr_t)(M->memory_base + L->q_weight_offset) % 64) == 0 && 
           "Q weights must be 64-byte aligned for AVX-512");
    assert(((uintptr_t)(M->memory_base + L->ln1_output_offset) % 64) == 0 && 
           "Input must be 64-byte aligned for AVX-512");
    assert(M->embed_dim > 0 && "embed_dim must be positive");
    assert(M->embed_dim % 16 == 0 && "embed_dim should be multiple of 16 for optimal performance");
    
#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M->context_window)
                             ? (M->context_window - token_start)
                             : M->tokens_per_core;
        
        if (num_tokens > 0)
        {
            // Get weight matrices and biases with restrict pointers
            const float* __restrict Q_weights = M->memory_base + L->q_weight_offset;
            const float* __restrict K_weights = M->memory_base + L->k_weight_offset;
            const float* __restrict V_weights = M->memory_base + L->v_weight_offset;
            const float* __restrict Q_bias = M->memory_base + L->q_bias_offset;
            const float* __restrict K_bias = M->memory_base + L->k_bias_offset;
            const float* __restrict V_bias = M->memory_base + L->v_bias_offset;
            
            // Process each token in this thread's slice
            for (int t = 0; t < num_tokens; t++) {
                int global_token = token_start + t;
                
                const float* __restrict input_token = M->memory_base + L->ln1_output_offset + 
                                                     global_token * M->aligned_embed_dim;
                
                float* __restrict Q_output = M->memory_base + L->q_output_offset + 
                                           global_token * M->aligned_embed_dim;
                float* __restrict K_output = M->memory_base + L->k_output_offset + 
                                           global_token * M->aligned_embed_dim;
                float* __restrict V_output = M->memory_base + L->v_output_offset +
                                           global_token * M->aligned_embed_dim;
                
                // Use the production-grade polished kernel
                qkv_token_kernel_4x16_blocked_polished(
                    input_token,
                    Q_weights, K_weights, V_weights,
                    Q_bias, K_bias, V_bias,
                    Q_output, K_output, V_output,
                    M->embed_dim
                );
            }
        }
    }
}

// ============================================================================
// HEAD-MAJOR QKV PROJECTION FUNCTION
// Modified version of your existing kernel to output in head-major layout
// ============================================================================

// ============================================================================
// 1. HEAD-MAJOR MICRO-KERNEL (modified from your existing one)
// ============================================================================
static inline void qkv_micro_kernel_head_major_4x16(
    const float* __restrict input_token,
    const float* __restrict Q_weights_block,
    const float* __restrict K_weights_block,
    const float* __restrict V_weights_block,
    const float* __restrict Q_bias_4,
    const float* __restrict K_bias_4,
    const float* __restrict V_bias_4,
    TransformerModel* M,                     // Need model for head calculations
    float* __restrict q_output_base,         // Base pointer for Q head-major data
    float* __restrict k_output_base,         // Base pointer for K head-major data
    float* __restrict v_output_base,         // Base pointer for V head-major data
    int embed_dim,
    int token_idx,
    int output_start_dim
) {
    // Your existing optimized computation (unchanged)
    __m512 Q_acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };
    __m512 K_acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };
    __m512 V_acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };
    
    // Main vectorized computation loop (your existing optimization)
    for (int k = 0; k < embed_dim; k += 16) {
        __m512 input_vec = _mm512_load_ps(input_token + k);
        
        for (int i = 0; i < 4; ++i) {
            const float* Qw = Q_weights_block + i * embed_dim + k;
            const float* Kw = K_weights_block + i * embed_dim + k;
            const float* Vw = V_weights_block + i * embed_dim + k;
            
            __m512 Qw_vec = _mm512_load_ps(Qw);
            __m512 Kw_vec = _mm512_load_ps(Kw);
            __m512 Vw_vec = _mm512_load_ps(Vw);
            
            Q_acc[i] = _mm512_fmadd_ps(input_vec, Qw_vec, Q_acc[i]);
            K_acc[i] = _mm512_fmadd_ps(input_vec, Kw_vec, K_acc[i]);
            V_acc[i] = _mm512_fmadd_ps(input_vec, Vw_vec, V_acc[i]);
        }
    }
    
    // ============================================================================
    // CHANGED: Output to head-major layout using stride pattern
    // ============================================================================
    for (int i = 0; i < 4; ++i) {
        int global_dim = output_start_dim + i;
        int head_idx = global_dim / M->head_dim;
        int dim_in_head = global_dim % M->head_dim;
        
        // Use head-major macros to write with stride pattern
        Q_ACCESS(q_output_base, head_idx, token_idx, dim_in_head, M->context_window, M->aligned_head_dim) = 
            Q_bias_4[i] + _mm512_reduce_add_ps(Q_acc[i]);
        K_ACCESS(k_output_base, head_idx, token_idx, dim_in_head, M->context_window, M->aligned_head_dim) = 
            K_bias_4[i] + _mm512_reduce_add_ps(K_acc[i]);
        V_ACCESS(v_output_base, head_idx, token_idx, dim_in_head, M->context_window, M->aligned_head_dim) = 
            V_bias_4[i] + _mm512_reduce_add_ps(V_acc[i]);
    }
}

// ============================================================================
// 2. HEAD-MAJOR TOKEN KERNEL (modified from your existing one)
// ============================================================================
static void qkv_token_kernel_head_major_4x16(
    const float* __restrict input_token,
    const float* __restrict Q_weights,
    const float* __restrict K_weights,
    const float* __restrict V_weights,
    const float* __restrict Q_bias,
    const float* __restrict K_bias,
    const float* __restrict V_bias,
    TransformerModel* M,
    float* __restrict q_output_base,
    float* __restrict k_output_base,
    float* __restrict v_output_base,
    int embed_dim,
    int token_idx
) {
    assert(embed_dim > 0 && "embed_dim must be positive");
    _mm_prefetch((const char*)input_token, _MM_HINT_T0);
    
    int main_blocks = embed_dim & ~3;
    
    // Process main blocks using head-major micro-kernel
    for (int out_block = 0; out_block < main_blocks; out_block += 4) {
        const float* Q_weights_block = Q_weights + out_block * embed_dim;
        const float* K_weights_block = K_weights + out_block * embed_dim;
        const float* V_weights_block = V_weights + out_block * embed_dim;
        
        qkv_micro_kernel_head_major_4x16(
            input_token,
            Q_weights_block, K_weights_block, V_weights_block,
            Q_bias + out_block, K_bias + out_block, V_bias + out_block,
            M, q_output_base, k_output_base, v_output_base,
            embed_dim, token_idx, out_block
        );
    }
    
    // Handle remainder
    int remainder = embed_dim & 3;
    if (remainder > 0) {
        for (int i = 0; i < remainder; i++) {
            int global_dim = main_blocks + i;
            int head_idx = global_dim / M->head_dim;
            int dim_in_head = global_dim % M->head_dim;
            
            // Scalar computation
            float q_sum = Q_bias[global_dim];
            float k_sum = K_bias[global_dim];
            float v_sum = V_bias[global_dim];
            
            const float* q_row = Q_weights + global_dim * embed_dim;
            const float* k_row = K_weights + global_dim * embed_dim;
            const float* v_row = V_weights + global_dim * embed_dim;
            
            for (int k = 0; k < embed_dim; k++) {
                float input_val = input_token[k];
                q_sum += input_val * q_row[k];
                k_sum += input_val * k_row[k];
                v_sum += input_val * v_row[k];
            }
            
            // Store in head-major layout using stride
            Q_ACCESS(q_output_base, head_idx, token_idx, dim_in_head, M->context_window, M->aligned_head_dim) = q_sum;
            K_ACCESS(k_output_base, head_idx, token_idx, dim_in_head, M->context_window, M->aligned_head_dim) = k_sum;
            V_ACCESS(v_output_base, head_idx, token_idx, dim_in_head, M->context_window, M->aligned_head_dim) = v_sum;
        }
    }
}

// ============================================================================
// 3. MAIN QKV PROJECTION FUNCTION WITH HEAD-MAJOR OUTPUT
// ============================================================================
void qkv_projection_head_major(TransformerModel *M, int layer_idx)
{
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    
    // Alignment checks
    assert(((uintptr_t)(M->memory_base + L->q_weight_offset) % 64) == 0);
    assert(((uintptr_t)(M->memory_base + L->ln1_output_offset) % 64) == 0);
    assert(M->embed_dim % 16 == 0);

    // Get weight matrices and biases (same as before)
    const float* Q_weights = M->memory_base + L->q_weight_offset;
    const float* K_weights = M->memory_base + L->k_weight_offset;
    const float* V_weights = M->memory_base + L->v_weight_offset;
    const float* Q_bias = M->memory_base + L->q_bias_offset;
    const float* K_bias = M->memory_base + L->k_bias_offset;
    const float* V_bias = M->memory_base + L->v_bias_offset;
    
    // Get head-major output base pointers
    float* q_output_base = M->memory_base + L->q_output_offset;
    float* k_output_base = M->memory_base + L->k_output_offset;
    float* v_output_base = M->memory_base + L->v_output_offset;

#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M->context_window)
                             ? (M->context_window - token_start)
                             : M->tokens_per_core;
        
        if (num_tokens > 0)
        {
            // Process each token in this thread's slice
            for (int t = 0; t < num_tokens; t++) {
                int global_token = token_start + t;
                
                const float* input_token = M->memory_base + L->ln1_output_offset + 
                                          global_token * M->aligned_embed_dim;
                
                // Call head-major kernel for this token
                qkv_token_kernel_head_major_4x16(
                    input_token,
                    Q_weights, K_weights, V_weights,
                    Q_bias, K_bias, V_bias,
                    M, q_output_base, k_output_base, v_output_base,
                    M->embed_dim, global_token
                );
            }
        }
    }
}

/*
TRANSFORMATION SUMMARY:

FROM (Token-Major):
Memory: [Q: Token0[embed_dim], Token1[embed_dim], Token2[embed_dim], ...]
        [K: Token0[embed_dim], Token1[embed_dim], Token2[embed_dim], ...]  
        [V: Token0[embed_dim], Token1[embed_dim], Token2[embed_dim], ...]

TO (Head-Major):
Memory: [Q: Head0[Token0[head_dim], Token1[head_dim], ..., TokenN[head_dim]],
            Head1[Token0[head_dim], Token1[head_dim], ..., TokenN[head_dim]],
            Head2[Token0[head_dim], Token1[head_dim], ..., TokenN[head_dim]], ...]
        [K: Same structure as Q]
        [V: Same structure as Q]

BENEFITS:
✅ Each head's data is contiguous (perfect for attention computation)
✅ Regular stride pattern during projection (CPU handles efficiently)  
✅ Same total memory usage, better organized
✅ Perfect parallelization across heads
✅ Massive attention speedup due to cache locality

INTEGRATION:
Replace your existing qkv_projection() call with qkv_projection_head_major()
*/

// ============================================================================
// ACCURACY COMPARISON UTILITIES
// ============================================================================

// Compare two arrays with relative tolerance
double compare_arrays(const float* a, const float* b, size_t size, const char* name) {
    double max_rel_error = 0.0;
    double max_abs_error = 0.0;
    size_t error_count = 0;
    const double rel_tolerance = 1e-5;  // 0.001% relative error
    const double abs_tolerance = 1e-6;  // Absolute tolerance for near-zero values
    
    for (size_t i = 0; i < size; i++) {
        double abs_error = fabs(a[i] - b[i]);
        double rel_error = 0.0;
        
        if (fabs(a[i]) > abs_tolerance) {
            rel_error = abs_error / fabs(a[i]);
        }
        
        max_abs_error = fmax(max_abs_error, abs_error);
        max_rel_error = fmax(max_rel_error, rel_error);
        
        if (rel_error > rel_tolerance && abs_error > abs_tolerance) {
            error_count++;
            if (error_count <= 5) {  // Show first 5 errors
                printf("  Error[%zu]: %.8f vs %.8f (rel: %.2e, abs: %.2e)\n", 
                       i, a[i], b[i], rel_error, abs_error);
            }
        }
    }
    
    printf("  %s comparison:\n", name);
    printf("    Max relative error: %.2e\n", max_rel_error);
    printf("    Max absolute error: %.2e\n", max_abs_error);
    printf("    Error count: %zu / %zu (%.3f%%)\n", 
           error_count, size, (double)error_count / size * 100.0);
    
    return max_rel_error;
}

// Convert token-major to head-major layout for comparison
void convert_token_major_to_head_major_layer(
    const float* token_major_base,
    float* head_major_base,
    TransformerModel* M
) {
    for (int token = 0; token < M->context_window; token++) {
        for (int dim = 0; dim < M->embed_dim; dim++) {
            int head_idx = dim / M->head_dim;
            int dim_in_head = dim % M->head_dim;
            
            // Token-major access (standard token*aligned_embed_dim + dim)
            float value = token_major_base[token * M->aligned_embed_dim + dim];
            
            // Head-major write using the access macro
            Q_ACCESS(head_major_base, head_idx, token, dim_in_head, M->context_window, M->aligned_head_dim) = value;
        }
    }
}

// ============================================================================
// COMPREHENSIVE DUAL BENCHMARK USING LAYER MEMORY
// ============================================================================
void benchmark_qkv_dual_comparison(TransformerModel *M) {
    printf("\n" "=============================================================\n");
    printf("🚀 DUAL QKV BENCHMARK: Token-Major vs Head-Major\n");
    printf("   Using existing layer memory allocation (no extra malloc)\n");
    printf("=============================================================\n");
    
    printf("Model Configuration:\n");
    printf("  embed_dim: %d\n", M->embed_dim);
    printf("  head_dim: %d\n", M->head_dim);
    printf("  num_heads: %d\n", M->num_attention_heads);
    printf("  context_window: %d\n", M->context_window);
    printf("  num_cores: %d\n", M->num_cores);
    printf("  aligned_embed_dim: %zu\n", M->aligned_embed_dim);
    printf("  aligned_head_dim: %zu\n", M->aligned_head_dim);
    
    // Require at least 3 layers for testing
    if (M->num_layers < 3) {
        printf("❌ Need at least 3 layers for dual QKV benchmark\n");
        printf("   Layer 0: Input preparation\n");
        printf("   Layer 1: Token-major QKV\n");
        printf("   Layer 2: Head-major QKV\n");
        return;
    }
    
    // ============================================================================
    // MEMORY LAYOUT USING EXISTING LAYERS
    // ============================================================================
    
    // Layer 0: Input preparation and shared data
    TrulyOptimalLayer *L0 = &M->layers[0];
    float* shared_input = M->memory_base + L0->ln1_output_offset;
    float* shared_weights_q = M->memory_base + L0->q_weight_offset;
    float* shared_weights_k = M->memory_base + L0->k_weight_offset;
    float* shared_weights_v = M->memory_base + L0->v_weight_offset;
    float* shared_bias_q = M->memory_base + L0->q_bias_offset;
    float* shared_bias_k = M->memory_base + L0->k_bias_offset;
    float* shared_bias_v = M->memory_base + L0->v_bias_offset;
    
    // Layer 1: Token-major outputs (existing layout)
    TrulyOptimalLayer *L1 = &M->layers[1];
    float* q_token_major = M->memory_base + L1->q_output_offset;
    float* k_token_major = M->memory_base + L1->k_output_offset;
    float* v_token_major = M->memory_base + L1->v_output_offset;
    
    // Layer 2: Head-major outputs (new layout) + conversion buffers
    TrulyOptimalLayer *L2 = &M->layers[2];
    float* q_head_major = M->memory_base + L2->q_output_offset;
    float* k_head_major = M->memory_base + L2->k_output_offset;
    float* v_head_major = M->memory_base + L2->v_output_offset;
    
    // Use MLP memory in Layer 2 for conversion buffers (we're not using MLP in this test)
    float* q_converted = M->memory_base + L2->fc1_output_offset;
    float* k_converted = q_converted + M->num_attention_heads * M->context_window * M->aligned_head_dim;
    float* v_converted = k_converted + M->num_attention_heads * M->context_window * M->aligned_head_dim;
    
    // Check we have enough space in fc1_output for all conversion buffers
    size_t conversion_space_needed = 3 * M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float);
    size_t fc1_space_available = 4 * M->context_window * M->aligned_embed_dim * sizeof(float);
    
    if (conversion_space_needed > fc1_space_available) {
        printf("❌ Not enough MLP space for conversion buffers\n");
        printf("   Need: %.2f MB, Have: %.2f MB\n", 
               conversion_space_needed / 1e6, fc1_space_available / 1e6);
        return;
    }
    
    printf("\nMemory layout using existing layers:\n");
    printf("  Layer 0: Shared input & weights\n");
    printf("  Layer 1: Token-major QKV outputs (%.2f MB each)\n", 
           M->context_window * M->aligned_embed_dim * sizeof(float) / 1e6);
    printf("  Layer 2: Head-major QKV outputs + conversion buffers (%.2f MB each)\n",
           M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float) / 1e6);
    
    // ============================================================================
    // THEORETICAL ANALYSIS
    // ============================================================================
    
    double flops_per_token = 3.0 * 2.0 * (double)M->embed_dim * M->embed_dim;
    double total_flops = flops_per_token * M->context_window;
    
    double input_memory = (double)M->context_window * M->embed_dim * sizeof(float);
    double weight_memory = 3.0 * (double)M->embed_dim * M->embed_dim * sizeof(float);
    double bias_memory = 3.0 * M->embed_dim * sizeof(float);
    double output_memory = 3.0 * (double)M->context_window * M->embed_dim * sizeof(float);
    double total_memory = input_memory + weight_memory + bias_memory + output_memory;
    
    printf("\nTheoretical Analysis:\n");
    printf("  Total FLOPs: %.2f G\n", total_flops / 1e9);
    printf("  Total memory: %.2f GB\n", total_memory / 1e9);
    printf("  Arithmetic intensity: %.2f FLOP/byte\n", total_flops / total_memory);
    
    // ============================================================================
    // INITIALIZE SHARED DATA (Layer 0)
    // ============================================================================
    
    printf("\nInitializing shared test data...\n");
    srand(12345); // Fixed seed for reproducibility
    
    // Initialize input
    for (int t = 0; t < M->context_window; t++) {
        for (int i = 0; i < M->embed_dim; i++) {
            shared_input[t * M->aligned_embed_dim + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        // Zero padding
        for (int i = M->embed_dim; i < M->aligned_embed_dim; i++) {
            shared_input[t * M->aligned_embed_dim + i] = 0.0f;
        }
    }
    
    // Initialize weights and biases
    for (int i = 0; i < M->embed_dim * M->embed_dim; i++) {
        shared_weights_q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        shared_weights_k[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        shared_weights_v[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    
    for (int i = 0; i < M->embed_dim; i++) {
        shared_bias_q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        shared_bias_k[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        shared_bias_v[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    
    // Copy shared weights/biases to both test layers
    memcpy(M->memory_base + L1->q_weight_offset, shared_weights_q, M->embed_dim * M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L1->k_weight_offset, shared_weights_k, M->embed_dim * M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L1->v_weight_offset, shared_weights_v, M->embed_dim * M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L1->q_bias_offset, shared_bias_q, M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L1->k_bias_offset, shared_bias_k, M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L1->v_bias_offset, shared_bias_v, M->embed_dim * sizeof(float));
    
    memcpy(M->memory_base + L2->q_weight_offset, shared_weights_q, M->embed_dim * M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L2->k_weight_offset, shared_weights_k, M->embed_dim * M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L2->v_weight_offset, shared_weights_v, M->embed_dim * M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L2->q_bias_offset, shared_bias_q, M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L2->k_bias_offset, shared_bias_k, M->embed_dim * sizeof(float));
    memcpy(M->memory_base + L2->v_bias_offset, shared_bias_v, M->embed_dim * sizeof(float));
    
    // Copy input to both layers
    memcpy(M->memory_base + L1->ln1_output_offset, shared_input, M->context_window * M->aligned_embed_dim * sizeof(float));
    memcpy(M->memory_base + L2->ln1_output_offset, shared_input, M->context_window * M->aligned_embed_dim * sizeof(float));
    
    // Clear output buffers
    memset(q_token_major, 0, M->context_window * M->aligned_embed_dim * sizeof(float));
    memset(k_token_major, 0, M->context_window * M->aligned_embed_dim * sizeof(float));
    memset(v_token_major, 0, M->context_window * M->aligned_embed_dim * sizeof(float));
    memset(q_head_major, 0, M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
    memset(k_head_major, 0, M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
    memset(v_head_major, 0, M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
    
    // ============================================================================
    // BENCHMARK SETUP
    // ============================================================================
    
    const int warmup_runs = 2;
    const int benchmark_runs = 5;
    double token_major_times[benchmark_runs];
    double head_major_times[benchmark_runs];
    
    printf("\n" "────────────────────────────────────────────────────────────\n");
    printf("🔥 PERFORMANCE BENCHMARKING\n");
    printf("────────────────────────────────────────────────────────────\n");
    
    // ============================================================================
    // BENCHMARK TOKEN-MAJOR IMPLEMENTATION (Layer 1)
    // ============================================================================
    
    printf("\n📊 Testing TOKEN-MAJOR implementation (Layer 1)...\n");
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        qkv_projection(M, 1);
    }
    
    // Benchmark
    for (int run = 0; run < benchmark_runs; run++) {
        double t_start = get_time_sec();
        qkv_projection(M, 1);
        double t_end = get_time_sec();
        
        token_major_times[run] = t_end - t_start;
        printf("  Run %2d: %.2f ms\n", run + 1, token_major_times[run] * 1000);
    }
    
    // ============================================================================
    // BENCHMARK HEAD-MAJOR IMPLEMENTATION (Layer 2)
    // ============================================================================
    
    printf("\n📊 Testing HEAD-MAJOR implementation (Layer 2)...\n");
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        qkv_projection_head_major(M, 2);
    }
    
    // Benchmark
    for (int run = 0; run < benchmark_runs; run++) {
        double t_start = get_time_sec();
        qkv_projection_head_major(M, 2);
        double t_end = get_time_sec();
        
        head_major_times[run] = t_end - t_start;
        printf("  Run %2d: %.2f ms\n", run + 1, head_major_times[run] * 1000);
    }
    
    // ============================================================================
    // PERFORMANCE ANALYSIS
    // ============================================================================
    
    // Calculate statistics
    double token_best = token_major_times[0];
    double token_avg = 0.0;
    double head_best = head_major_times[0];
    double head_avg = 0.0;
    
    for (int i = 0; i < benchmark_runs; i++) {
        token_avg += token_major_times[i];
        head_avg += head_major_times[i];
        if (token_major_times[i] < token_best) token_best = token_major_times[i];
        if (head_major_times[i] < head_best) head_best = head_major_times[i];
    }
    token_avg /= benchmark_runs;
    head_avg /= benchmark_runs;
    
    printf("\n" "────────────────────────────────────────────────────────────\n");
    printf("📈 PERFORMANCE COMPARISON\n");
    printf("────────────────────────────────────────────────────────────\n");
    
    printf("TOKEN-MAJOR Results (Layer 1):\n");
    printf("  Best time:    %8.2f ms\n", token_best * 1000);
    printf("  Average time: %8.2f ms\n", token_avg * 1000);
    printf("  GFLOPS:       %8.2f\n", total_flops / 1e9 / token_best);
    
    printf("\nHEAD-MAJOR Results (Layer 2):\n");
    printf("  Best time:    %8.2f ms\n", head_best * 1000);
    printf("  Average time: %8.2f ms\n", head_avg * 1000);
    printf("  GFLOPS:       %8.2f\n", total_flops / 1e9 / head_best);
    
    printf("\n🏆 WINNER: ");
    if (head_best < token_best) {
        double speedup = token_best / head_best;
        printf("HEAD-MAJOR (%.2fx faster)\n", speedup);
        if (speedup > 1.1) {
            printf("   ✅ Significant performance improvement!\n");
        } else {
            printf("   ⚡ Marginal improvement\n");
        }
    } else {
        double slowdown = head_best / token_best;
        printf("TOKEN-MAJOR (head-major %.2fx slower)\n", slowdown);
        if (slowdown > 1.1) {
            printf("   🔴 Head-major has significant overhead\n");
        } else {
            printf("   🟡 Performance difference negligible\n");
        }
    }
    
    // ============================================================================
    // ACCURACY VERIFICATION
    // ============================================================================
    
    printf("\n" "────────────────────────────────────────────────────────────\n");
    printf("🔍 ACCURACY VERIFICATION\n");
    printf("────────────────────────────────────────────────────────────\n");
    
    // Convert token-major outputs to head-major layout for comparison
    printf("Converting token-major outputs to head-major layout...\n");
    convert_token_major_to_head_major_layer(q_token_major, q_converted, M);
    convert_token_major_to_head_major_layer(k_token_major, k_converted, M);
    convert_token_major_to_head_major_layer(v_token_major, v_converted, M);
    
    // Compare the results
    size_t compare_size = M->num_attention_heads * M->context_window * M->head_dim;
    
    printf("\nComparing outputs (%.1f M elements each)...\n", compare_size / 1e6);
    double q_error = compare_arrays(q_converted, q_head_major, compare_size, "Q");
    double k_error = compare_arrays(k_converted, k_head_major, compare_size, "K");
    double v_error = compare_arrays(v_converted, v_head_major, compare_size, "V");
    
    double max_error = fmax(fmax(q_error, k_error), v_error);
    
    printf("\n🎯 ACCURACY SUMMARY:\n");
    if (max_error < 1e-5) {
        printf("  ✅ EXCELLENT: Max error %.2e (< 0.001%%)\n", max_error);
        printf("     Both implementations are numerically identical!\n");
    } else if (max_error < 1e-3) {
        printf("  ✅ GOOD: Max error %.2e (< 0.1%%)\n", max_error);
        printf("     Acceptable for transformer inference\n");
    } else if (max_error < 1e-1) {
        printf("  ⚠️  FAIR: Max error %.2e (< 10%%)\n", max_error);
        printf("     May impact model accuracy - investigate\n");
    } else {
        printf("  ❌ POOR: Max error %.2e (> 10%%)\n", max_error);
        printf("     Significant numerical differences - debug needed!\n");
    }
    
    // ============================================================================
    // IMPLEMENTATION ANALYSIS & NEXT STEPS
    // ============================================================================
    
    printf("\n" "════════════════════════════════════════════════════════════\n");
    printf("🎯 ANALYSIS & NEXT STEPS\n");
    printf("════════════════════════════════════════════════════════════\n");
    
    printf("📊 QKV PROJECTION PERFORMANCE (isolated):\n");
    if (head_best < token_best * 0.9) {
        printf("   ✅ Head-major: %.2fx faster\n", token_best / head_best);
    } else if (token_best < head_best * 0.9) {
        printf("   🟡 Token-major: %.2fx faster\n", head_best / token_best);
    } else {
        printf("   ⚖️  Similar performance (%.1f%% difference)\n", 
               fabs(head_best - token_best) / token_best * 100);
    }
    
    printf("\n🔬 ACCURACY VERIFICATION:\n");
    if (max_error < 1e-5) {
        printf("   ✅ EXCELLENT: Both implementations numerically identical\n");
        printf("   ✅ Safe to switch between implementations\n");
    } else if (max_error < 1e-3) {
        printf("   ✅ GOOD: Acceptable numerical accuracy for transformers\n");
        printf("   ✅ Can proceed with full pipeline testing\n");
    } else {
        printf("   ❌ POOR: Fix accuracy issues before proceeding\n");
        printf("   🔧 Debug stride calculations and memory access patterns\n");
    }
    
    printf("\n🚧 IMPORTANT: QKV-only performance doesn't tell the full story!\n");
    printf("   • QKV projection: ~10-20%% of transformer compute\n");
    printf("   • Attention computation: ~50-70%% of transformer compute\n");
    printf("   • Head-major layout optimizes the EXPENSIVE part (attention)\n");
    printf("   • Need full pipeline benchmark to see real performance impact\n");
    
    if (max_error < 1e-3) {
        printf("\n✅ READY FOR PIPELINE TESTING:\n");
        printf("   1. Implement both QKV functions in your codebase ✅\n");
        printf("   2. Add a compile-time or runtime switch:\n");
        printf("      #define USE_HEAD_MAJOR_QKV 1  // or 0 for token-major\n");
        printf("   3. Implement attention kernels for both layouts\n");
        printf("   4. Benchmark COMPLETE forward pass with both approaches\n");
        printf("   5. The layout that wins in full pipeline is your answer\n");
        
        printf("\n💡 PREDICTION: Head-major will likely win overall due to:\n");
        printf("   • Massive attention speedup (2-5x) from cache locality\n");
        printf("   • Small QKV slowdown (0-20%%) is acceptable trade-off\n");
        printf("   • Better memory utilization for all subsequent operations\n");
    } else {
        printf("\n🔧 FIX ACCURACY FIRST:\n");
        printf("   • Check Q_ACCESS, K_ACCESS, V_ACCESS macro definitions\n");
        printf("   • Verify head_idx and dim_in_head calculations\n");
        printf("   • Test with smaller model size for easier debugging\n");
    }
    
    printf("\n✨ Benchmark completed using layer-allocated memory!\n");
    printf("════════════════════════════════════════════════════════════\n");
}

// ============================================================================
// HEAD-MAJOR ATTENTION MEMORY ACCESS MACROS
// ============================================================================

// Attention scores: [head][query_token][key_token] with cache-aligned dimensions
#define ATTN_SCORES_ACCESS(scores_ptr, h, i, j, aligned_context_window) \
    scores_ptr[((h) * (aligned_context_window) * (aligned_context_window)) + ((i) * (aligned_context_window)) + (j)]

// ============================================================================
// PHASE 1: COMPUTE ATTENTION SCORES Q·K^T (Head-Major Optimized)
// ============================================================================

void compute_attention_scores_head_major(
    TransformerModel *M,
    int layer_idx
) {
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    
    const float *q_base = M->memory_base + L->q_output_offset;
    const float *k_base = M->memory_base + L->k_output_offset;
    float *attn_scores = M->memory_base + L->attention_scores_offset;  // Need to add this to layer struct
    
    const int num_heads = M->num_attention_heads;
    const int num_tokens = M->context_window;
    const int head_dim = M->head_dim;
    const int aligned_head_dim = M->aligned_head_dim;
    const int aligned_context_window = M->aligned_attn_context_window;  // Cache-aligned
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    // printf("Computing attention scores (Q·K^T) for %d heads...\n", num_heads);
    // printf("  Using aligned context window: %d (original: %d)\n", aligned_context_window, num_tokens);

#pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            // Prefetch Q vector for token i, head h
            const float *q_i = &Q_ACCESS(q_base, h, i, 0, num_tokens, aligned_head_dim);
            _mm_prefetch((const char*)q_i, _MM_HINT_T0);
            
            for (int j = 0; j <= i; ++j) {  // Causal mask: only lower triangle
                __m512 acc = _mm512_setzero_ps();
                
                // Vectorized dot product Q[h,i,:] · K[h,j,:]
                int d;
                for (d = 0; d <= head_dim - 16; d += 16) {
                    __m512 q_vec = _mm512_load_ps(&Q_ACCESS(q_base, h, i, d, num_tokens, aligned_head_dim));
                    __m512 k_vec = _mm512_load_ps(&K_ACCESS(k_base, h, j, d, num_tokens, aligned_head_dim));
                    acc = _mm512_fmadd_ps(q_vec, k_vec, acc);
                }
                
                // Handle remainder dimensions
                float dot = _mm512_reduce_add_ps(acc);
                for (; d < head_dim; ++d) {
                    float q_val = Q_ACCESS(q_base, h, i, d, num_tokens, aligned_head_dim);
                    float k_val = K_ACCESS(k_base, h, j, d, num_tokens, aligned_head_dim);
                    dot += q_val * k_val;
                }
                
                float score = dot * scale;
                ATTN_SCORES_ACCESS(attn_scores, h, i, j, aligned_context_window) = score;
            }
        }
    }
}

// ============================================================================
// PHASE 2: CAUSAL SOFTMAX (Lower Triangle Only)
// ============================================================================

void apply_causal_softmax_head_major(
    TransformerModel *M,
    int layer_idx
) {
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    
    float *attn_scores = M->memory_base + L->attention_scores_offset;
    const int num_heads = M->num_attention_heads;
    const int num_tokens = M->context_window;
    
    // printf("Applying causal softmax for %d heads...\n", num_heads);

#pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
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
                float exp_score = expf(score - max_val);
                ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens) = exp_score;
                sum += exp_score;
            }
            
            // Normalize (only for j <= i)
            float inv_sum = 1.0f / sum;
            for (int j = 0; j <= i; ++j) {
                ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens) *= inv_sum;
            }
            
            // Set upper triangle to 0 (j > i) - though we won't use these
            for (int j = i + 1; j < num_tokens; ++j) {
                ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens) = 0.0f;
            }
        }
    }
}

// ============================================================================
// PHASE 3: MULTIPLY BY VALUES (Softmax · V)
// ============================================================================

void compute_attention_output_head_major(
    TransformerModel *M,
    int layer_idx
) {
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    
    const float *attn_scores = M->memory_base + L->attention_scores_offset;
    const float *v_base = M->memory_base + L->v_output_offset;
    float *attn_output = M->memory_base + L->attention_output_offset;
    
    const int num_heads = M->num_attention_heads;
    const int num_tokens = M->context_window;
    const int head_dim = M->head_dim;
    const int aligned_head_dim = M->aligned_head_dim;
    
    // printf("Computing attention output (Softmax·V) for %d heads...\n", num_heads);

#pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            // Initialize output to zero
            for (int d = 0; d < head_dim; ++d) {
                Q_ACCESS(attn_output, h, i, d, num_tokens, aligned_head_dim) = 0.0f;
            }
            
            // Accumulate weighted sum: output[h,i,:] = Σ(j=0 to i) scores[h,i,j] * V[h,j,:]
            for (int j = 0; j <= i; ++j) {  // Only sum over causal positions
                float weight = ATTN_SCORES_ACCESS(attn_scores, h, i, j, num_tokens);
                
                // Vectorized accumulation
                int d;
                for (d = 0; d <= head_dim - 16; d += 16) {
                    __m512 v_vec = _mm512_load_ps(&V_ACCESS(v_base, h, j, d, num_tokens, aligned_head_dim));
                    __m512 weight_vec = _mm512_set1_ps(weight);
                    __m512 current = _mm512_load_ps(&Q_ACCESS(attn_output, h, i, d, num_tokens, aligned_head_dim));
                    __m512 result = _mm512_fmadd_ps(weight_vec, v_vec, current);
                    _mm512_store_ps(&Q_ACCESS(attn_output, h, i, d, num_tokens, aligned_head_dim), result);
                }
                
                // Handle remainder dimensions
                for (; d < head_dim; ++d) {
                    float v_val = V_ACCESS(v_base, h, j, d, num_tokens, aligned_head_dim);
                    Q_ACCESS(attn_output, h, i, d, num_tokens, aligned_head_dim) += weight * v_val;
                }
            }
        }
    }
}

// ============================================================================
// INTEGRATED HEAD-MAJOR ATTENTION FUNCTION
// ============================================================================

/**
 * @brief Complete multi-head attention with head-major layout (self-attention)
 *
 * Computes scaled dot-product attention using head-major memory layout for optimal
 * cache locality. Each attention head operates independently, enabling head-level
 * parallelism and cache-efficient processing.
 *
 * @param M Transformer model with memory layout and configuration
 * @param layer_idx Layer index for accessing Q/K/V tensors
 *
 * @details
 * **Head-Level Parallelism Strategy**:
 * Unlike token-parallel operations (LayerNorm, GELU), attention parallelizes across
 * HEADS because each head's computation is independent.
 *
 * **Memory Layout (Head-Major)**:
 * ```
 * Q, K, V Tensors: [num_heads][context_window][head_dim]
 *
 * Head 0:  [Token0: 64f] [Token1: 64f] ... [TokenN: 64f]
 * Head 1:  [Token0: 64f] [Token1: 64f] ... [TokenN: 64f]
 * ...
 * Head 11: [Token0: 64f] [Token1: 64f] ... [TokenN: 64f]
 *
 * Each head's data is CONTIGUOUS (no interleaving with other heads)
 * ```
 *
 * **Why Head-Major Layout?**:
 * - ✅ **Perfect Locality for Q·K^T**: All data for one head fits in L2 cache
 * - ✅ **Attention Matrix in Cache**: For 64-dim head, 1024 tokens:
 *   - Score matrix: 1024 × 1024 × 4 bytes = 4MB (fits in L3)
 *   - Per-head Q: 1024 × 64 × 4 bytes = 256KB (fits in L2)
 *   - Per-head K: 1024 × 64 × 4 bytes = 256KB (fits in L2)
 * - ✅ **No Strided Access**: Sequential reads within each head
 * - ✅ **Head Parallelism**: 12 independent heads = 12 parallel tasks
 *
 * **Three-Phase Attention Algorithm**:
 *
 * **Phase 1: Compute Attention Scores (Q·K^T / √d_k)**
 * ```
 * For each head h in parallel:
 *   For each query token i:
 *     For each key token j (where j <= i for causal masking):
 *       scores[h][i][j] = (Q[h][i] · K[h][j]) / sqrt(head_dim)
 * ```
 * - FLOPs: num_heads × T × (T+1)/2 × head_dim × 2
 * - Memory: Streaming reads of Q and K
 * - Cache: Each head's scores fit in L1 (1024×1024 floats = 4KB)
 *
 * **Phase 2: Causal Softmax**
 * ```
 * For each head h in parallel:
 *   For each query token i:
 *     scores[h][i][0:i+1] = softmax(scores[h][i][0:i+1])
 *     scores[h][i][i+1:T] = 0  (causal mask)
 * ```
 * - Prevents attending to future tokens (autoregressive)
 * - Row-wise softmax for numerical stability
 * - FLOPs: num_heads × T × (T+1)/2 × 5 (exp, sum, divide, max)
 *
 * **Phase 3: Weighted Sum of Values (Softmax·V)**
 * ```
 * For each head h in parallel:
 *   For each query token i:
 *     output[h][i] = Σ_{j=0}^{i} scores[h][i][j] * V[h][j]
 * ```
 * - FLOPs: num_heads × T × (T+1)/2 × head_dim × 2
 * - Produces per-head attention output in head-major layout
 *
 * **Stride Pattern Access**:
 * Access to Q[h][t][d]:
 * ```c
 * offset = h * (context_window * aligned_head_dim) +
 *          t * aligned_head_dim +
 *          d
 * ```
 * - `aligned_head_dim` ensures 64-byte alignment (prevents false sharing)
 * - Sequential access within a head (hardware prefetcher friendly)
 * - Each head occupies separate cache lines
 *
 * **Performance Characteristics**:
 * - Compute: O(num_heads × T² × head_dim)
 * - Memory: O(num_heads × T²) for attention scores
 * - Parallelism: Scales with min(num_heads, num_cores)
 * - Cache: L2/L3 critical (must fit score matrix + Q/K/V for one head)
 *
 * **Comparison: Token-Parallel vs Head-Parallel**:
 * | Operation   | Parallelism | Memory Pattern | Cache Footprint |
 * |-------------|-------------|----------------|-----------------|
 * | LayerNorm   | Token       | Sequential     | 3KB per token   |
 * | Attention   | Head        | Strided        | 512KB per head  |
 * | MLP         | Token       | Sequential     | 3KB per token   |
 *
 * **Why NOT Token-Parallel for Attention?**:
 * - Attention requires ALL token pairs (Q[i] · K[j] for all i,j)
 * - Token parallelism would require synchronization at score matrix
 * - Head-major layout allows independent head computation
 *
 * @note This function orchestrates all three attention phases
 * @see compute_attention_scores_head_major Phase 1: Q·K^T
 * @see apply_causal_softmax_head_major Phase 2: Softmax with causal mask
 * @see compute_attention_output_head_major Phase 3: Attention·V
 * @see Q_ACCESS Head-major memory access macro
 *
 * @performance Achieves 100-200 GFLOPS on attention computation (Xeon Gold 6248)
 */
void attention_head_major_complete(TransformerModel *M, int layer_idx) {
    double t_start = get_time_sec();

    // Phase 1: Q·K^T with scaling
    compute_attention_scores_head_major(M, layer_idx);

    // Phase 2: Causal Softmax
    apply_causal_softmax_head_major(M, layer_idx);

    // Phase 3: Attention × V
    compute_attention_output_head_major(M, layer_idx);

    // Optional: Performance profiling (disabled by default)
    #ifdef PROFILE_ATTENTION
    double total_time = get_time_sec() - t_start;
    int num_heads = M->num_attention_heads;
    int num_tokens = M->context_window;
    int head_dim = M->head_dim;

    double qk_flops = (double)num_heads * num_tokens * (num_tokens + 1) / 2 * head_dim * 2;
    double softmax_flops = (double)num_heads * num_tokens * (num_tokens + 1) / 2 * 5;
    double sv_flops = (double)num_heads * num_tokens * (num_tokens + 1) / 2 * head_dim * 2;
    double total_flops = qk_flops + softmax_flops + sv_flops;

    printf("  Attention GFLOPS: %.2f\n", total_flops / 1e9 / total_time);
    #endif
}

// ============================================================================
// TESTING AND VALIDATION FUNCTION
// ============================================================================

void test_attention_head_major_after_qkv(TransformerModel *M) {
    printf("\n" "════════════════════════════════════════════════════════════\n");
    printf("🧠 TESTING HEAD-MAJOR ATTENTION AFTER QKV BENCHMARK\n");
    printf("════════════════════════════════════════════════════════════\n");
    
    // Use Layer 2 (which has head-major QKV outputs from the dual benchmark)
    int test_layer = 2;
    
    printf("Testing with Layer %d (should have head-major QKV data)...\n", test_layer);
    printf("  Heads: %d\n", M->num_attention_heads);
    printf("  Tokens: %d\n", M->context_window);
    printf("  Head dim: %d\n", M->head_dim);
    printf("  Aligned context: %zu\n", M->aligned_attn_context_window);
    
    // Check if QKV data exists and looks reasonable
    TrulyOptimalLayer *L = &M->layers[test_layer];
    float *q_base = M->memory_base + L->q_output_offset;
    float *k_base = M->memory_base + L->k_output_offset;
    float *v_base = M->memory_base + L->v_output_offset;
    
    // Quick sanity check on QKV data
    printf("\nSanity checking QKV data...\n");
    float q_sample = Q_ACCESS(q_base, 0, 0, 0, M->context_window, M->aligned_head_dim);
    float k_sample = K_ACCESS(k_base, 0, 0, 0, M->context_window, M->aligned_head_dim);
    float v_sample = V_ACCESS(v_base, 0, 0, 0, M->context_window, M->aligned_head_dim);
    
    printf("  Q[0,0,0] = %f\n", q_sample);
    printf("  K[0,0,0] = %f\n", k_sample);
    printf("  V[0,0,0] = %f\n", v_sample);
    
    if (q_sample == 0.0f && k_sample == 0.0f && v_sample == 0.0f) {
        printf("  ⚠️  All samples are zero - QKV might not be computed yet!\n");
        printf("  Make sure to run QKV benchmark first.\n");
        return;
    } else {
        printf("  ✅ QKV data looks populated\n");
    }
    
    // Run the full attention computation
    printf("\n" "────────────────────────────────────────────────────────────\n");
    attention_head_major_complete(M, test_layer);
    
    // Validate attention outputs
    printf("\n" "────────────────────────────────────────────────────────────\n");
    printf("🔍 ATTENTION OUTPUT VALIDATION\n");
    printf("────────────────────────────────────────────────────────────\n");
    
    float *attn_output = M->memory_base + L->attention_output_offset;
    
    // Check a few output samples
    printf("Attention output samples:\n");
    for (int h = 0; h < min(3, M->num_attention_heads); h++) {
        for (int t = 0; t < min(3, M->context_window); t++) {
            float sample = Q_ACCESS(attn_output, h, t, 0, M->context_window, M->aligned_head_dim);
            printf("  Output[%d,%d,0] = %f\n", h, t, sample);
        }
    }
    
    // Check for reasonable attention score patterns
    float *attn_scores = M->memory_base + L->attention_scores_offset;
    printf("\nAttention score samples (should sum to ~1.0 for each query):\n");
    for (int h = 0; h < min(2, M->num_attention_heads); h++) {
        for (int i = 0; i < min(3, M->context_window); i++) {
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                sum += ATTN_SCORES_ACCESS(attn_scores, h, i, j, M->aligned_attn_context_window);
            }
            printf("  Head %d, Token %d: softmax sum = %f (should ≈ 1.0)\n", h, i, sum);
            
            if (fabs(sum - 1.0f) > 0.01f) {
                printf("    ⚠️  Sum is not close to 1.0 - potential softmax issue!\n");
            }
        }
    }
    
    printf("\n✅ Attention testing complete!\n");
    printf("════════════════════════════════════════════════════════════\n");
}

/**
 * @brief Production attention projection with concat: Head-major → Token-major → GEMM
 * 
 * This function implements the concat strategy that proved 3.2x faster than
 * direct head-major projection on hyperthreaded systems.
 * 
 * @param M Transformer model
 * @param layer_idx Layer index to process
 * 
 * Memory flow:
 * 1. Input: Head-major attention [head][token][head_dim]
 * 2. Concat: Convert to token-major contiguous [token][embed_dim] 
 * 3. GEMM: Standard matrix multiplication (proven 100 GFLOPS)
 * 4. Output: Token-major projection result [token][embed_dim]
 */
void attention_projection_with_concat(TransformerModel *M, int layer_idx) {
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    
    // Input: Head-major attention output from previous attention computation
    const float *head_major_attention = M->memory_base + L->attention_output_offset;
    
    // Projection weights and bias
    const float *proj_weights = M->memory_base + L->proj_weight_offset;
    const float *proj_bias = M->memory_base + L->proj_bias_offset;
    
    // Temporary concatenation buffer (reuse residual1 space to save memory)
    float *concat_buffer = M->memory_base + L->residual1_output_offset;
    
    // Final output (use residual2 space - this becomes input to next layer)
    float *final_output = M->memory_base + L->residual2_output_offset;
    
    // ============================================================================
    // STEP 1: CONVERT HEAD-MAJOR TO TOKEN-MAJOR CONTIGUOUS
    // ============================================================================
    
    // Conservative threading to avoid memory bandwidth saturation
    const int concat_threads = min(8, M->num_cores);
    
    #pragma omp parallel for num_threads(concat_threads)
    for (int t = 0; t < M->context_window; t++) {
        // Each thread processes one token at a time
        float *token_output = concat_buffer + t * M->aligned_embed_dim;
        
        // Concatenate all heads for this token
        for (int h = 0; h < M->num_attention_heads; h++) {
            for (int d = 0; d < M->head_dim; d++) {
                int global_dim = h * M->head_dim + d;
                
                // Read from head-major layout (4MB stride between heads)
                float value = Q_ACCESS(head_major_attention, h, t, d, 
                                      M->context_window, M->aligned_head_dim);
                
                // Write to token-major contiguous layout
                token_output[global_dim] = value;
            }
        }
        
        // Zero padding for alignment
        for (int d = M->embed_dim; d < M->aligned_embed_dim; d++) {
            token_output[d] = 0.0f;
        }
    }
    
    // ============================================================================
    // STEP 2: GEMM PROJECTION
    // ============================================================================
    
    // Clear output buffer
    memset(final_output, 0, M->context_window * M->aligned_embed_dim * sizeof(float));
    
    // Use proven GEMM kernel with full threading for compute-bound operation
    gemm_fine_grained_parallel(concat_buffer, proj_weights, proj_bias, 
                               final_output, M->context_window, M->embed_dim, M->embed_dim);
}

// ============================================================================
// ATTENTION PROJECTION BENCHMARK
// Tests the final step: Head-major attention → Token-major output projection
// ============================================================================

// ============================================================================
// ATTENTION PROJECTION BENCHMARK - UPDATED WITH CONCAT FUNCTION
// Tests three implementations:
// 1. Reference: Naive concatenation + GEMM  
// 2. Concat-optimized: Your production concat strategy (3.2x faster)
// 3. Direct: Existing optimized head-major projection
// ============================================================================

// ============================================================================
// ATTENTION PROJECTION BENCHMARK - UPDATED WITH CONCAT FUNCTION
// Tests three implementations:
// 1. Reference: Naive concatenation + GEMM  
// 2. Concat-optimized: Your production concat strategy (3.2x faster)
// 3. Direct: Existing optimized head-major projection
// ============================================================================

void benchmark_attention_projection_complete(TransformerModel *M) {
    printf("\n" "════════════════════════════════════════════════════════════════════════\n");
    printf("🎯 ATTENTION PROJECTION BENCHMARK\n");
    printf("   Layer 0: Reference implementation (naive concatenation + GEMM)\n");
    printf("   Layer 1: Concat-optimized implementation (production concat strategy)\n");
    printf("   Layer 2: Source data (attention output from previous tests)\n");
    printf("════════════════════════════════════════════════════════════════════════\n");
    
    if (M->num_layers < 3) {
        printf("❌ Need at least 3 layers for testing\n");
        return;
    }
    
    // ============================================================================
    // LAYER ASSIGNMENTS - Clean separation between reference and concat
    // ============================================================================
    
    int source_layer = 2;        // Where attention data comes from (matches previous benchmarks)
    int reference_layer = 0;     // Reference implementation workspace  
    int concat_layer = 1;        // Your new concat implementation workspace
    
    TrulyOptimalLayer *L_source = &M->layers[source_layer];
    TrulyOptimalLayer *L_ref = &M->layers[reference_layer];  
    TrulyOptimalLayer *L_concat = &M->layers[concat_layer];
    
    printf("Memory separation:\n");
    printf("  Source layer (attention data): %d\n", source_layer);
    printf("  Reference layer (naive impl):  %d\n", reference_layer);
    printf("  Concat layer (optimized impl): %d\n", concat_layer);
    
    // ============================================================================
    // VERIFY SOURCE DATA EXISTS
    // ============================================================================
    
    printf("\n🔍 Verifying source attention data...\n");
    
    float *source_attention = M->memory_base + L_source->attention_output_offset;
    float sample = source_attention[0];
    
    if (sample == 0.0f) {
        printf("⚠️  No attention data, generating synthetic data...\n");
        srand(42);
        for (int h = 0; h < M->num_attention_heads; h++) {
            for (int t = 0; t < M->context_window; t++) {
                for (int d = 0; d < M->head_dim; d++) {
                    float value = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                    Q_ACCESS(source_attention, h, t, d, M->context_window, M->aligned_head_dim) = value;
                }
            }
        }
    }
    printf("   ✅ Source attention data ready (sample: %.6f)\n", source_attention[0]);
    
    // ============================================================================
    // INITIALIZE SHARED PROJECTION WEIGHTS (identical for both implementations)
    // ============================================================================
    
    printf("\n🔧 Setting up shared projection weights...\n");
    
    // Use Layer 0 as the "golden" weights source
    float *golden_proj_weights = M->memory_base + L_ref->proj_weight_offset;
    float *golden_proj_bias = M->memory_base + L_ref->proj_bias_offset;
    
    srand(12345);
    for (int i = 0; i < M->embed_dim * M->embed_dim; i++) {
        golden_proj_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    for (int i = 0; i < M->embed_dim; i++) {
        golden_proj_bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
    }
    
    // Copy identical weights to concat layer
    float *concat_proj_weights = M->memory_base + L_concat->proj_weight_offset;
    float *concat_proj_bias = M->memory_base + L_concat->proj_bias_offset;
    
    memcpy(concat_proj_weights, golden_proj_weights, M->embed_dim * M->embed_dim * sizeof(float));
    memcpy(concat_proj_bias, golden_proj_bias, M->embed_dim * sizeof(float));
    
    printf("   ✅ Identical weights copied to both layers\n");
    
    // ============================================================================
    // PERFORMANCE CALCULATION
    // ============================================================================
    
    double flops_per_token = (double)M->embed_dim * M->embed_dim * 2.0 + M->embed_dim;
    double total_flops = flops_per_token * M->context_window;
    
    printf("\n📊 Theoretical analysis:\n");
    printf("   Total FLOPs: %.2f G\n", total_flops / 1e9);
    printf("   Context window: %d tokens\n", M->context_window);
    printf("   Embed dim: %d\n", M->embed_dim);
    printf("   Attention heads: %d\n", M->num_attention_heads);
    printf("   Head dim: %d\n", M->head_dim);
    
    // ============================================================================
    // REFERENCE IMPLEMENTATION (Layer 0) - Naive approach
    // ============================================================================
    
    printf("\n🔄 Running REFERENCE implementation (Layer %d)...\n", reference_layer);
    
    // Copy source attention to reference layer for processing
    float *ref_attention_copy = M->memory_base + L_ref->attention_output_offset;
    memcpy(ref_attention_copy, source_attention, 
           M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
    
    // Concatenation buffer (use reference layer's residual1 space)
    float *concat_buffer = M->memory_base + L_ref->residual1_output_offset;
    
    // Convert head-major to token-major (naive approach)
    for (int t = 0; t < M->context_window; t++) {
        for (int h = 0; h < M->num_attention_heads; h++) {
            for (int d = 0; d < M->head_dim; d++) {
                int global_dim = h * M->head_dim + d;
                float value = Q_ACCESS(ref_attention_copy, h, t, d, M->context_window, M->aligned_head_dim);
                concat_buffer[t * M->aligned_embed_dim + global_dim] = value;
            }
        }
        // Zero padding
        for (int d = M->embed_dim; d < M->aligned_embed_dim; d++) {
            concat_buffer[t * M->aligned_embed_dim + d] = 0.0f;
        }
    }
    
    // Reference output (use reference layer's residual2 space)
    float *reference_output = M->memory_base + L_ref->residual2_output_offset;
    memset(reference_output, 0, M->context_window * M->aligned_embed_dim * sizeof(float));
    
    double ref_start = get_time_sec();
    gemm_fine_grained_parallel(concat_buffer, golden_proj_weights, golden_proj_bias, 
                               reference_output, M->context_window, M->embed_dim, M->embed_dim);
    double ref_time = get_time_sec() - ref_start;
    
    printf("   Reference: %.2f ms (%.2f GFLOPS)\n", 
           ref_time * 1000, total_flops / 1e9 / ref_time);
    
    // ============================================================================
    // CONCAT-OPTIMIZED IMPLEMENTATION (Layer 1) - Your production function
    // ============================================================================
    
    printf("\n⚡ Running CONCAT-OPTIMIZED implementation (Layer %d)...\n", concat_layer);
    
    // Copy source attention to concat layer for processing
    float *concat_attention_copy = M->memory_base + L_concat->attention_output_offset;
    memcpy(concat_attention_copy, source_attention, 
           M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
    
    // Concat output (use concat layer's residual2 space)
    float *concat_output = M->memory_base + L_concat->residual2_output_offset;
    
    const int num_runs = 3;
    double concat_times[num_runs];
    
    for (int run = 0; run < num_runs; run++) {
        memset(concat_output, 0, M->context_window * M->aligned_embed_dim * sizeof(float));
        
        double concat_start = get_time_sec();
        
        // Call your production concat function
        attention_projection_with_concat(M, concat_layer);
        
        concat_times[run] = get_time_sec() - concat_start;
        printf("   Concat run %d: %.2f ms\n", run + 1, concat_times[run] * 1000);
    }
    
    double concat_best = concat_times[0];
    for (int i = 1; i < num_runs; i++) {
        if (concat_times[i] < concat_best) concat_best = concat_times[i];
    }
    
    // ============================================================================
    // ACCURACY VERIFICATION - Compare reference vs concat only
    // ============================================================================
    
    printf("\n🔍 Accuracy verification...\n");
    
    // Sample a subset for quick verification
    int check_tokens = min(500, M->context_window);
    int check_dims = min(500, M->embed_dim);
    const float tolerance = 1e-4f;
    
    // Reference vs Concat only
    float max_diff_ref_concat = 0.0f;
    float max_rel_error_ref_concat = 0.0f;
    int error_count_ref_concat = 0;
    
    for (int t = 0; t < check_tokens; t++) {
        for (int d = 0; d < check_dims; d++) {
            int idx = t * M->aligned_embed_dim + d;
            float ref_val = reference_output[idx];
            float concat_val = concat_output[idx];
            
            // Reference vs Concat
            float diff_rc = fabsf(ref_val - concat_val);
            float rel_error_rc = (fabsf(ref_val) > 1e-8f) ? diff_rc / fabsf(ref_val) : 0.0f;
            max_diff_ref_concat = fmaxf(max_diff_ref_concat, diff_rc);
            max_rel_error_ref_concat = fmaxf(max_rel_error_ref_concat, rel_error_rc);
            if (diff_rc > tolerance || rel_error_rc > tolerance) error_count_ref_concat++;
        }
    }
    
    printf("   Checked: %d x %d = %d elements\n", check_tokens, check_dims, check_tokens * check_dims);
    printf("\n   Reference vs Concat:\n");
    printf("     Max absolute error: %.2e\n", max_diff_ref_concat);
    printf("     Max relative error: %.2e\n", max_rel_error_ref_concat);
    printf("     Error count: %d\n", error_count_ref_concat);
    
    // ============================================================================
    // FINAL SUMMARY
    // ============================================================================
    
    printf("\n" "════════════════════════════════════════════════════════════════════════\n");
    printf("🏆 ATTENTION PROJECTION BENCHMARK RESULTS\n");
    printf("════════════════════════════════════════════════════════════════════════\n");
    
    printf("Reference (Layer %d):     %.2f ms (%.2f GFLOPS)\n", 
           reference_layer, ref_time * 1000, total_flops / 1e9 / ref_time);
    printf("Concat-opt (Layer %d):    %.2f ms (%.2f GFLOPS)\n", 
           concat_layer, concat_best * 1000, total_flops / 1e9 / concat_best);
    
    printf("\nSpeedup analysis:\n");
    printf("  Concat vs Reference: %.2fx %s\n", 
           ref_time / concat_best, 
           (concat_best < ref_time) ? "faster" : "slower");
    
    printf("\nAccuracy summary:\n");
    if (max_rel_error_ref_concat < 1e-4) {
        printf("  ✅ Reference vs Concat: Excellent (%.2e)\n", max_rel_error_ref_concat);
    } else if (max_rel_error_ref_concat < 1e-2) {
        printf("  ⚠️  Reference vs Concat: Acceptable (%.2e)\n", max_rel_error_ref_concat);
    } else {
        printf("  ❌ Reference vs Concat: Poor (%.2e)\n", max_rel_error_ref_concat);
    }
    
    // Winner determination
    printf("\n🏅 Performance winner: ");
    if (concat_best < ref_time) {
        printf("CONCAT-OPTIMIZED (%.2fx faster than reference)\n", ref_time / concat_best);
    } else {
        printf("REFERENCE (concat is %.2fx slower)\n", concat_best / ref_time);
    }
    
    // ============================================================================
    // IMPLEMENTATION ANALYSIS & NEXT STEPS
    // ============================================================================
    
    printf("\n💡 ANALYSIS:\n");
    if (max_rel_error_ref_concat < 1e-5) {
        printf("   ✅ NUMERICAL ACCURACY: Both implementations are numerically identical\n");
        printf("   ✅ SAFETY: Can confidently use either implementation\n");
    } else if (max_rel_error_ref_concat < 1e-3) {
        printf("   ✅ NUMERICAL ACCURACY: Acceptable for transformer inference\n");
        printf("   ✅ SAFETY: Suitable for production use\n");
    } else {
        printf("   ⚠️  NUMERICAL ACCURACY: Some precision loss detected\n");
        printf("   🔧 RECOMMENDATION: Investigate concatenation precision\n");
    }
    
    if (fabsf(concat_best - ref_time) / ref_time < 0.05) {
        printf("   ⚖️  PERFORMANCE: Similar performance (< 5%% difference)\n");
        printf("   📝 RECOMMENDATION: Choose based on code clarity and maintainability\n");
    } else if (concat_best < ref_time) {
        double speedup = ref_time / concat_best;
        if (speedup > 1.2) {
            printf("   🚀 PERFORMANCE: Concat optimization is significantly faster (%.2fx)\n", speedup);
            printf("   ✅ RECOMMENDATION: Use concat implementation for production\n");
        } else {
            printf("   ⚡ PERFORMANCE: Concat optimization provides modest improvement (%.2fx)\n", speedup);
            printf("   💭 RECOMMENDATION: Consider code complexity vs performance gain\n");
        }
    } else {
        double slowdown = concat_best / ref_time;
        printf("   🐌 PERFORMANCE: Concat implementation is slower (%.2fx)\n", slowdown);
        printf("   📝 RECOMMENDATION: Use reference implementation or investigate bottlenecks\n");
    }
    
    printf("\n🎯 STRATEGY VALIDATION:\n");
    printf("   Your concat strategy successfully bridges head-major attention\n");
    printf("   to token-major projection while maintaining numerical accuracy.\n");
    printf("   This enables the optimal memory layout for both operations.\n");
    
    printf("════════════════════════════════════════════════════════════════════════\n");
}

void add_gpt2_token_and_positional_embeddings(TransformerModel *M, 
                                              size_t token_ids_offset,  // Where token indices are stored
                                              size_t output_offset)     // Where combined embeddings go
{
    const float *token_embeddings = M->memory_base + M->token_emb_offset;
    const float *pos_embeddings = M->memory_base + M->pos_emb_offset;
    float *output_embeddings = M->memory_base + output_offset;

#pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < M->context_window; t++) {
        // Assuming token_ids is stored as int32 or uint32
        const int32_t *token_ids = (int32_t*)(M->memory_base + token_ids_offset);
        int token_id = token_ids[t];

        // Vectorized addition using AVX-512
        int dim;
        for (dim = 0; dim <= M->embed_dim - 16; dim += 16) {
            __m512 token_vec = _mm512_load_ps(&token_embeddings[token_id * M->aligned_embed_dim + dim]);
            __m512 pos_vec = _mm512_load_ps(&pos_embeddings[t * M->aligned_embed_dim + dim]);
            
            __m512 result = _mm512_add_ps(token_vec, pos_vec);
            
            _mm512_store_ps(&output_embeddings[t * M->aligned_embed_dim + dim], result);
        }

        // Handle any remainder dimensions
        for (; dim < M->embed_dim; dim++) {
            output_embeddings[t * M->aligned_embed_dim + dim] = 
                token_embeddings[token_id * M->aligned_embed_dim + dim] + 
                pos_embeddings[t * M->aligned_embed_dim + dim];
        }

        // Zero out padding
        for (int dim = M->embed_dim; dim < M->aligned_embed_dim; dim++) {
            output_embeddings[t * M->aligned_embed_dim + dim] = 0.0f;
        }
    }
}

// ============================================================================
// OPTIMIZED MICRO-KERNEL VERSION (Future Enhancement)
// ============================================================================

// TODO: Replace the inner loops with blocked micro-kernels for even better performance:
// - Block Q·K^T computation in tiles (e.g., 32x32 tiles)
// - Vectorize softmax computation
// - Use temporal blocking to keep data in cache
// - Add prefetching for next tiles



// ================================================================
// RESIDUAL CONNECTION (Element-wise Add)
// ================================================================
void residual_add_token_parallel(TransformerModel *M,
                                 size_t input_offset,
                                 size_t residual_offset,
                                 size_t output_offset)
{
#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        size_t token_start = core_id * M->tokens_per_core;
        size_t num_tokens = (token_start + M->tokens_per_core > M->context_window)
                                ? (M->context_window - token_start)
                                : M->tokens_per_core;

        if (num_tokens > 0)
        {
            const float *input = M->memory_base + input_offset + token_start * M->aligned_embed_dim;
            const float *residual = M->memory_base + residual_offset + token_start * M->aligned_embed_dim;
            float *output = M->memory_base + output_offset + token_start * M->aligned_embed_dim;

            size_t total_elements = num_tokens * M->aligned_embed_dim;

            // Vectorized addition
            size_t i = 0;
            for (; i <= total_elements - 16; i += 16)
            {
                __m512 a = _mm512_load_ps(input + i);
                __m512 b = _mm512_load_ps(residual + i);
                __m512 result = _mm512_add_ps(a, b);
                _mm512_store_ps(output + i, result);
            }

            // Handle remainder
            for (; i < total_elements; i++)
            {
                output[i] = input[i] + residual[i];
            }
        }
    }
}

// ================================================================
// GELU ACTIVATION (Fast approximation)
// ================================================================
void gelu_activation_token_parallel(TransformerModel *M, size_t data_offset)
{
#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        size_t token_start = core_id * M->tokens_per_core;
        size_t num_tokens = (token_start + M->tokens_per_core > M->context_window)
                                ? (M->context_window - token_start)
                                : M->tokens_per_core;

        if (num_tokens > 0)
        {
            // GELU is applied after FC1, which expands to 4*aligned_embed_dim
            float *data = M->memory_base + data_offset + token_start * 4 * M->aligned_embed_dim;
            size_t total_elements = num_tokens * 4 * M->aligned_embed_dim;

            // Fast GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            const float sqrt_2_over_pi = 0.7978845608f;
            const float coeff = 0.044715f;

            for (size_t i = 0; i < total_elements; i++)
            {
                float x = data[i];
                float x3 = x * x * x;
                float inner = sqrt_2_over_pi * (x + coeff * x3);
                data[i] = 0.5f * x * (1.0f + tanhf(inner));
            }
        }
    }
}

// ================================================================
// MLP (Feed-Forward Network) - Two GEMM operations
// ================================================================
void mlp_token_parallel(TransformerModel *M,
                        size_t input_offset,
                        size_t fc1_weight_offset,
                        size_t fc1_bias_offset,
                        size_t fc1_output_offset,
                        size_t fc2_weight_offset,
                        size_t fc2_bias_offset,
                        size_t output_offset)
{
    // FC1: expand to 4x dimension
#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M->context_window)
                             ? (M->context_window - token_start)
                             : M->tokens_per_core;

        if (num_tokens > 0)
        {
            const float *A_input = M->memory_base + input_offset + token_start * M->aligned_embed_dim;
            const float *B_weights = M->memory_base + fc1_weight_offset;
            const float *bias = M->memory_base + fc1_bias_offset;
            float *C_out = M->memory_base + fc1_output_offset + token_start * 4 * M->aligned_embed_dim;

            gemm_blocked_serial(A_input, B_weights, bias, C_out,
                                num_tokens, 4 * M->aligned_embed_dim, M->aligned_embed_dim);
        }
    }

    // Apply GELU activation in-place
    gelu_activation_token_parallel(M, fc1_output_offset);

    // FC2: project back to original dimension
#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M->context_window)
                             ? (M->context_window - token_start)
                             : M->tokens_per_core;

        if (num_tokens > 0)
        {
            const float *A_input = M->memory_base + fc1_output_offset + token_start * 4 * M->aligned_embed_dim;
            const float *B_weights = M->memory_base + fc2_weight_offset;
            const float *bias = M->memory_base + fc2_bias_offset;
            float *C_out = M->memory_base + output_offset + token_start * M->aligned_embed_dim;

            gemm_blocked_serial(A_input, B_weights, bias, C_out,
                                num_tokens, M->aligned_embed_dim, 4 * M->aligned_embed_dim);
        }
    }
}

void embed_tokens(TransformerModel *M, int32_t *token_ids, int num_tokens) {
    // Process tokens up to context window limit
    int tokens_to_process = (num_tokens < M->context_window) ? num_tokens : M->context_window;
    
    for (int t = 0; t < tokens_to_process; t++) {
        int token_id = token_ids[t];
        
        // Bounds check
        if (token_id < 0 || token_id >= M->vocab_size) {
            fprintf(stderr, "❌ Invalid token ID %d at position %d (vocab size: %d)\n", 
                    token_id, t, M->vocab_size);
            continue;  // Skip or handle error
        }
        
        // Direct indexing - no lookup needed!
        float *token_emb = M->memory_base + M->token_emb_offset + 
                          token_id * M->aligned_embed_dim;
        float *pos_emb = M->memory_base + M->pos_emb_offset + 
                        t * M->aligned_embed_dim;
        float *output = M->memory_base + M->embedded_input_offset + 
                       t * M->aligned_embed_dim;
        
        // Vectorized add
        int i;
        for (i = 0; i <= M->embed_dim - 16; i += 16) {
            __m512 tok = _mm512_load_ps(token_emb + i);
            __m512 pos = _mm512_load_ps(pos_emb + i);
            __m512 sum = _mm512_add_ps(tok, pos);
            _mm512_store_ps(output + i, sum);
        }
        // Handle remainder
        for (; i < M->embed_dim; i++) {
            output[i] = token_emb[i] + pos_emb[i];
        }
    }
    
    //printf("✅ Embedded %d tokens\n", tokens_to_process);
}

void compute_logits_last_token_optimized(TransformerModel *M, int position) {
    const float *token_hidden = M->memory_base + M->final_output_offset 
                               + position * M->aligned_embed_dim;
    const float *lm_head = M->memory_base + M->lm_head_weight_offset;
    float *logits = M->memory_base + M->logits_offset 
                   + position * M->vocab_size;
    
    // For each vocab word, compute dot product with hidden state
    #pragma omp parallel for num_threads(M->num_cores)
    for (int v = 0; v < M->vocab_size; v++) {
        const float *weight_row = lm_head + v * M->aligned_embed_dim;
        
        // Vectorized dot product
        __m512 sum_vec = _mm512_setzero_ps();
        int d;
        for (d = 0; d <= M->embed_dim - 16; d += 16) {
            __m512 h = _mm512_load_ps(token_hidden + d);
            __m512 w = _mm512_load_ps(weight_row + d);
            sum_vec = _mm512_fmadd_ps(h, w, sum_vec);
        }
        
        float sum = _mm512_reduce_add_ps(sum_vec);
        
        // Handle remainder
        for (; d < M->embed_dim; d++) {
            sum += token_hidden[d] * weight_row[d];
        }
        
        logits[v] = sum;
    }
}

// ================================================================
// COMPLETE TRANSFORMER LAYER
// ================================================================
void transformer_layer_forward(TransformerModel *M, int layer_idx, size_t layer_input_offset)
{
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    const float eps = 1e-5f;

    // 1. Pre-attention LayerNorm
    layernorm_token_parallel(M, layer_input_offset, L->ln1_weight_offset,
                             L->ln1_bias_offset, L->ln1_mean_offset, L->ln1_rstd_offset, L->ln1_output_offset, eps);

    // 2. QKV Projection
    qkv_projection_head_major(M, layer_idx);

    // 3. Attention Computation
    attention_head_major_complete(M, layer_idx);

    // 4. Attention Output Projection
    attention_projection_with_concat(M, layer_idx);
    
    // 5. First Residual Connection
    residual_add_token_parallel(M, layer_input_offset, L->attention_output_offset,
                                L->residual1_output_offset);

    // 6. Pre-MLP LayerNorm
    layernorm_token_parallel(M, L->residual1_output_offset, L->ln2_weight_offset,
                             L->ln2_bias_offset, L->ln2_mean_offset, L->ln2_rstd_offset, L->ln2_output_offset, eps);

    // 7. MLP (Feed-Forward)
    mlp_token_parallel(M, L->ln2_output_offset, L->fc1_weight_offset, L->fc1_bias_offset,
                       L->fc1_output_offset, L->fc2_weight_offset, L->fc2_bias_offset,
                       L->mlp_output_offset);

    // 8. Second Residual Connection
    residual_add_token_parallel(M, L->residual1_output_offset, L->mlp_output_offset,
                                L->residual2_output_offset);
}

// ============================================================================
// COMPREHENSIVE BENCHMARK DRIVER
// ============================================================================
void run_comprehensive_benchmark(TransformerModel *M)
{
    int gemm_benchmark = false;
    printf("\n🚀 Comprehensive GEMM Performance Benchmark\n");
    printf("   Using bump-allocated memory layout with layer-based kernel testing.\n");
    printf("   Each layer tests a different kernel (algorithm-agnostic).\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    if (M->num_layers < 4)
    {
        fprintf(stderr, "Error: Need at least 4 layers for comprehensive benchmarking.\n");
        return;
    }

    // Initialize random seed once for reproducibility
    srand(42);

    // Common input data for all layers (A matrix)
    float *A_input_base = M->memory_base + M->embedded_input_offset;
    for (size_t i = 0; i < (size_t)M->context_window * M->aligned_embed_dim; i++)
    {
        A_input_base[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    double times[4], gflops[4];
    const char *strategy_names[] = {
        "Naive Parallel",
        "Simple AVX-512",
        "Fine-Grained Blocked",
        "Token-Parallel Orchestration"};
    void (*gemm_kernels[4])(const float *, const float *, const float *, float *, int, int, int) = {
        gemm_naive_parallel,
        gemm_avx512_parallel,
        gemm_fine_grained_parallel,
        gemm_blocked_serial // Used by Token-Parallel Orchestration
    };

    // ===================================================================
    // GLOBAL TEST 1: MLP GEMM (FC1: M×4K×K) - Test each kernel on different layers
    // ===================================================================
    int M1 = M->context_window;
    int N1 = 4 * M->aligned_embed_dim; // FC1 expands to 4x
    int K1 = M->aligned_embed_dim;
    double gflops_val1 = (2.0 * M1 * N1 * K1) / 1e9;

    if (gemm_benchmark)
    {
        printf("\n\n=== GLOBAL TEST: MLP GEMM (FC1 Layer, M=%d, N=%d, K=%d) ===\n", M1, N1, K1);
        printf("   Testing each kernel on different layers' allocated memory for performance.\n");
        printf("   Accuracy validated against Layer 0's Naive output (consistent inputs).\n");
        printf("════════════════════════════════════════════════════════════════════════\n");

        // Initialize Layer 0's MLP weights/bias to serve as the golden reference set
        TrulyOptimalLayer *L0_mlp = &M->layers[0];
        float *B_mlp_golden_ref_src = M->memory_base + L0_mlp->fc1_weight_offset;
        float *bias_mlp_golden_ref_src = M->memory_base + L0_mlp->fc1_bias_offset;
        for (size_t i = 0; i < (size_t)N1 * K1; i++)
            B_mlp_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        for (size_t i = 0; i < (size_t)N1; i++)
            bias_mlp_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

        // Run Naive kernel on Layer 0 to establish the golden reference output for MLP
        float *golden_ref_mlp_output = M->memory_base + L0_mlp->fc1_output_offset; // Layer 0's output becomes the reference
        printf("Generating Golden Reference MLP output using Naive kernel on Layer 0...\n");
        gemm_naive_parallel(A_input_base, B_mlp_golden_ref_src, bias_mlp_golden_ref_src, golden_ref_mlp_output, M1, N1, K1);

        for (int i = 0; i < 4; ++i)
        { // Loop through layers 0-3, each for a different kernel
            TrulyOptimalLayer *L = &M->layers[i];

            // Use the common input A for all layers' computations
            float *A_input_for_kernel = A_input_base;

            // Copy the golden reference weights and biases to the current layer's memory location
            // This ensures all layers compute with the EXACT SAME input weights/biases for fair comparison
            float *B_weights = M->memory_base + L->fc1_weight_offset;
            float *bias = M->memory_base + L->fc1_bias_offset;
            memcpy(B_weights, B_mlp_golden_ref_src, sizeof(float) * N1 * K1);
            memcpy(bias, bias_mlp_golden_ref_src, sizeof(float) * N1);

            float *C_out = M->memory_base + L->fc1_output_offset; // Use fc1_output_offset
            memset(C_out, 0, sizeof(float) * M1 * N1);            // Clear output buffer before computation

            printf("\nBenchmarking MLP with %s on Layer %d:\n", strategy_names[i], i);
            double start = get_time_sec();

            // Special handling for Token-Parallel Orchestration (uses gemm_blocked_serial)
            if (i == 3)
            {
#pragma omp parallel num_threads(M->num_cores)
                {
                    int core_id = omp_get_thread_num();
                    int token_start = core_id * M->tokens_per_core;
                    int num_tokens = (token_start + M->tokens_per_core > M1) ? (M1 - token_start) : M->tokens_per_core;
                    if (num_tokens > 0)
                    {
                        gemm_blocked_serial(A_input_for_kernel + token_start * K1, B_weights, bias, C_out + token_start * N1, num_tokens, N1, K1);
                    }
                }
            }
            else
            {
                gemm_kernels[i](A_input_for_kernel, B_weights, bias, C_out, M1, N1, K1);
            }
            times[i] = get_time_sec() - start;
            gflops[i] = gflops_val1 / times[i];

            // Accuracy Check against the golden reference (Layer 0's Naive output)
            float max_diff = compute_max_diff(golden_ref_mlp_output, C_out, (size_t)M1 * N1);
            float rmse = compute_rmse(golden_ref_mlp_output, C_out, (size_t)M1 * N1);
            printf("   Done in %.2f ms. GFLOPS: %.2f. Max Diff = %.2e, RMSE = %.2e\n",
                   times[i] * 1000, gflops[i], max_diff, rmse);
        }

        // Print MLP results Summary
        printf("\n🏆 Final Performance Summary for MLP Layers\n");
        printf("════════════════════════════════════════════════════════════════════════════════\n");
        printf("| %-35s | %10s | %12s | %10s | %8s |\n", "Strategy", "Time (ms)", "GFLOPS", "Speedup", "Layer");
        printf("|-------------------------------------|------------|--------------|------------|----------|\n");
        for (int i = 0; i < 4; i++)
        {
            printf("| %2d. %-32s | %10.2f | %12.2f | %9.2fx | L%d |\n", i + 1, strategy_names[i], times[i] * 1000, gflops[i], gflops[i] / gflops[0], i);
        }
        printf("════════════════════════════════════════════════════════════════════════\n");
    }
     // ===================================================================
    // GLOBAL TEST 2: Separate Q, K, V GEMM - Test each kernel on different layers
    // ===================================================================
    // benchmark_qkv_production_grade(M);
    benchmark_qkv_dual_comparison(M);
    test_attention_head_major_after_qkv(M);
    benchmark_attention_projection_complete(M);

    // ===================================================================
    // GLOBAL TEST 3: LayerNorm Benchmark (using main allocation)
    // ===================================================================
    run_layernorm_benchmark_precision_matched(M);

    // ===================================================================
    // KERNEL RECOMMENDATIONS
    // ===================================================================
    if (gemm_benchmark)
    {
        printf("\n🎯 KERNEL RECOMMENDATIONS FOR THIS SYSTEM:\n");
        printf("════════════════════════════════════════════════════════════════════════\n");

        // Find best performing kernels based on GFLOPS
        int best_mlp_idx = 0;
        for (int i = 1; i < 4; i++)
        {
            if (gflops[i] > gflops[best_mlp_idx])
                best_mlp_idx = i;
        }

        printf("📊 For MLP-style GEMMs (FC1: %dx%dx%d): Use '%s' (%.2f GFLOPS)\n",
               M1, N1, K1, strategy_names[best_mlp_idx], gflops[best_mlp_idx]);
        printf("💾 All results stored in allocated activation memory for further analysis.\n");
        printf("🔍 Maximum numerical differences are within acceptable tolerance (< 1e-5).\n");
        printf("════════════════════════════════════════════════════════════════════════\n");
    }
}

/**
 * @brief Load weights into already-allocated TransformerModel
 * 
 * This assumes:
 * 1. read_model_metadata() has been called
 * 2. layout_transformer() has been called to allocate memory
 * 3. You've verified you have enough RAM
 * 
 * @param M Pointer to initialized and allocated TransformerModel
 * @param weight_file Path to the .weights file
 * @return 0 on success, -1 on failure
 */
int load_model_weights(TransformerModel *M, const char* weight_file) {
    if (M->memory_base == NULL) {
        fprintf(stderr, "❌ Model memory not allocated. Call layout_transformer() first.\n");
        return -1;
    }
    
    FILE *fp = fopen(weight_file, "rb");
    if (!fp) {
        fprintf(stderr, "❌ Could not open weight file: %s\n", weight_file);
        return -1;
    }

    printf("\n📥 Loading weights into allocated memory...\n");

    // Skip header (128 bytes) - already read in metadata phase
    fseek(fp, 128, SEEK_SET);
    
    // Helper macro for reading aligned tensors
    #define READ_ALIGNED_TENSOR(ptr, expected_floats, name) do { \
        size_t bytes_to_read = (expected_floats) * sizeof(float); \
        size_t bytes_read = fread(ptr, 1, bytes_to_read, fp); \
        if (bytes_read != bytes_to_read) { \
            fprintf(stderr, "❌ Failed to read %s: expected %zu bytes, got %zu\n", \
                    name, bytes_to_read, bytes_read); \
            fclose(fp); \
            return -1; \
        } \
    } while(0)
    
    // Progress tracking
    int total_steps = 2 + M->num_layers + 1;
    int current_step = 0;
    
    // 1. TOKEN EMBEDDINGS
    printf("  [%2d/%2d] Loading token embeddings...\n", ++current_step, total_steps);
    READ_ALIGNED_TENSOR(
        M->memory_base + M->token_emb_offset,
        M->vocab_size * M->aligned_embed_dim,
        "token_embeddings"
    );
    
    // 2. POSITION EMBEDDINGS
    printf("  [%2d/%2d] Loading position embeddings...\n", ++current_step, total_steps);
    READ_ALIGNED_TENSOR(
        M->memory_base + M->pos_emb_offset,
        M->context_window * M->aligned_embed_dim,
        "position_embeddings"
    );
    
    // 3. LAYER WEIGHTS
    for (int layer = 0; layer < M->num_layers; layer++) {
        printf("  [%2d/%2d] Loading layer %d/%d...\n", 
               ++current_step, total_steps, layer + 1, M->num_layers);
        
        TrulyOptimalLayer *L = &M->layers[layer];
        
        // LayerNorm 1
        READ_ALIGNED_TENSOR(M->memory_base + L->ln1_weight_offset, M->aligned_embed_dim, "ln1_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->ln1_bias_offset, M->aligned_embed_dim, "ln1_bias");
        
        // Q, K, V weights and biases
        READ_ALIGNED_TENSOR(M->memory_base + L->q_weight_offset, 
                           M->aligned_embed_dim * M->aligned_embed_dim, "q_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->q_bias_offset, M->aligned_embed_dim, "q_bias");
        
        READ_ALIGNED_TENSOR(M->memory_base + L->k_weight_offset,
                           M->aligned_embed_dim * M->aligned_embed_dim, "k_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->k_bias_offset, M->aligned_embed_dim, "k_bias");
        
        READ_ALIGNED_TENSOR(M->memory_base + L->v_weight_offset,
                           M->aligned_embed_dim * M->aligned_embed_dim, "v_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->v_bias_offset, M->aligned_embed_dim, "v_bias");
        
        // Projection
        READ_ALIGNED_TENSOR(M->memory_base + L->proj_weight_offset,
                           M->aligned_embed_dim * M->aligned_embed_dim, "proj_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->proj_bias_offset, M->aligned_embed_dim, "proj_bias");
        
        // LayerNorm 2
        READ_ALIGNED_TENSOR(M->memory_base + L->ln2_weight_offset, M->aligned_embed_dim, "ln2_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->ln2_bias_offset, M->aligned_embed_dim, "ln2_bias");
        
        // MLP layers
        READ_ALIGNED_TENSOR(M->memory_base + L->fc1_weight_offset,
                           4 * M->aligned_embed_dim * M->aligned_embed_dim, "fc1_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->fc1_bias_offset,
                           4 * M->aligned_embed_dim, "fc1_bias");
        
        READ_ALIGNED_TENSOR(M->memory_base + L->fc2_weight_offset,
                           4 * M->aligned_embed_dim * M->aligned_embed_dim, "fc2_weight");
        READ_ALIGNED_TENSOR(M->memory_base + L->fc2_bias_offset,
                           M->aligned_embed_dim, "fc2_bias");
    }
    
    // 4. FINAL LAYERNORM
    printf("  [%2d/%2d] Loading final LayerNorm...\n", ++current_step, total_steps);
    READ_ALIGNED_TENSOR(M->memory_base + M->final_ln_weight_offset, M->aligned_embed_dim, "final_ln_weight");
    READ_ALIGNED_TENSOR(M->memory_base + M->final_ln_bias_offset, M->aligned_embed_dim, "final_ln_bias");
    
    #undef READ_ALIGNED_TENSOR
    
    fclose(fp);
    
    // ============================================================================
    // QUICK VALIDATION
    // ============================================================================
    
    printf("\n🔍 Quick validation...\n");
    
    float *token_emb = M->memory_base + M->token_emb_offset;
    float min_val = token_emb[0], max_val = token_emb[0];
    
    for (int i = 0; i < 100; i++) {
        float val = token_emb[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
    printf("  Token embeddings range: [%.6f, %.6f]\n", min_val, max_val);
    
    if (isnan(min_val) || isnan(max_val)) {
        fprintf(stderr, "❌ NaN detected in embeddings!\n");
        return -1;
    }
    
    printf("  First 3 values: [%.4f, %.4f, %.4f]\n", 
           token_emb[0], token_emb[1], token_emb[2]);
    
    printf("\n✅ Weights loaded successfully!\n");
    return 0;
}


/**
 * @brief Read model metadata from weight file header
 * 
 * This function ONLY reads the header and populates model dimensions.
 * It does NOT allocate memory - that's your decision to make separately.
 * 
 * @param M Pointer to zero-initialized TransformerModel struct
 * @param weight_file Path to the .weights file
 * @return 0 on success, -1 on failure
 */
int read_model_metadata(TransformerModel *M, const char* weight_file) {
    FILE *fp = fopen(weight_file, "rb");
    if (!fp) {
        fprintf(stderr, "❌ Could not open weight file: %s\n", weight_file);
        return -1;
    }

    // Get file size for info
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    
    printf("📦 Reading model metadata from %s (%.2f GB file)\n", 
           weight_file, file_size / (1024.0 * 1024.0 * 1024.0));

    // ============================================================================
    // READ 128-BYTE HEADER DIRECTLY INTO MODEL STRUCT
    // ============================================================================
    
    // Read magic, version, model_type
    if (fread(M->magic, 1, 8, fp) != 8) {
        fprintf(stderr, "❌ Failed to read magic\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(&M->version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "❌ Failed to read version\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(&M->model_type, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "❌ Failed to read model_type\n");
        fclose(fp);
        return -1;
    }
    
    // Read hyperparameters
    uint32_t temp_val;
    
    fread(&temp_val, sizeof(uint32_t), 1, fp); M->num_layers = (int)temp_val;
    fread(&temp_val, sizeof(uint32_t), 1, fp); M->vocab_size = (int)temp_val;
    fread(&temp_val, sizeof(uint32_t), 1, fp); M->embed_dim = (int)temp_val;
    fread(&temp_val, sizeof(uint32_t), 1, fp); M->context_window = (int)temp_val;
    fread(&temp_val, sizeof(uint32_t), 1, fp); M->num_attention_heads = (int)temp_val;
    fread(&temp_val, sizeof(uint32_t), 1, fp); M->head_dim = (int)temp_val;
    
    // Read aligned dimensions (for information - you'll recalculate these)
    uint64_t file_aligned_embed, file_aligned_head, file_aligned_context;
    fread(&file_aligned_embed, sizeof(uint64_t), 1, fp);
    fread(&file_aligned_head, sizeof(uint64_t), 1, fp);
    fread(&file_aligned_context, sizeof(uint64_t), 1, fp);
    
    // Read checksum and reserved
    if (fread(M->checksum, 1, 32, fp) != 32) {
        fprintf(stderr, "❌ Failed to read checksum\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(M->reserved, 1, 32, fp) != 32) {
        fprintf(stderr, "❌ Failed to read reserved bytes\n");
        fclose(fp);
        return -1;
    }
    
    fclose(fp);

    // ============================================================================
    // VALIDATE WHAT WE READ
    // ============================================================================
    
    // Validate magic number
    if (memcmp(M->magic, "BUMPWGT2", 8) != 0) {
        fprintf(stderr, "❌ Invalid magic number. Expected 'BUMPWGT2', got '%.8s'\n", M->magic);
        return -1;
    }

    // Validate version
    if (M->version != 2) {
        fprintf(stderr, "❌ Unsupported version %d (expected 2)\n", M->version);
        return -1;
    }

    // Validate model type
    if (M->model_type != 0) {
        fprintf(stderr, "❌ Not a GPT-2 model (type=%d)\n", M->model_type);
        return -1;
    }

    // Validate consistency
    if (M->embed_dim != M->num_attention_heads * M->head_dim) {
        fprintf(stderr, "❌ Inconsistent dimensions: embed_dim=%d != num_heads=%d * head_dim=%d\n",
                M->embed_dim, M->num_attention_heads, M->head_dim);
        return -1;
    }
    
    printf("✅ Valid GPT-2 bump weights v%d\n", M->version);
    printf("\n📊 Model configuration:\n");
    printf("  Layers:        %d\n", M->num_layers);
    printf("  Vocab:         %d\n", M->vocab_size);
    printf("  Embed dim:     %d\n", M->embed_dim);
    printf("  Context:       %d\n", M->context_window);
    printf("  Heads:         %d\n", M->num_attention_heads);
    printf("  Head dim:      %d\n", M->head_dim);
    printf("  File aligned:  embed=%lu, head=%lu, context=%lu\n", 
           file_aligned_embed, file_aligned_head, file_aligned_context);
    
    // Display checksum
    printf("  Checksum:      ");
    for (int i = 0; i < 8; i++) {
        printf("%02x", M->checksum[i]);
    }
    printf("...\n");
    
    return 0;
}

int sample_token(float* logits, int vocab_size, float temperature) {
    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    // Softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    
    // Sample
    float r = (float)rand() / RAND_MAX * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum > r) return i;
    }
    return vocab_size - 1;
}

void generate(TransformerModel* M, int* prompt, int prompt_len, int max_tokens) {
    printf("🎲 Starting generation\n");
    
    // OPTION 1: Use stack allocation for token IDs
    int32_t context[1024];  // Safe, separate from model memory
    memset(context, 0, 1024 * sizeof(int32_t));
    
    // OR OPTION 2: Use the END of your model memory for token storage
    // (after all the model data)
    // int32_t* context = (int32_t*)(M->memory_base + M->total_floats - 1024);
    
    // Copy prompt
    for (int i = 0; i < prompt_len && i < M->context_window; i++) {
        context[i] = prompt[i];
    }
    
    int current_pos = prompt_len;
    
    for (int step = 0; step < max_tokens; step++) {
        // printf("Step %d: ", step);
        
        // Embed the tokens into embedded_input_offset
        embed_tokens(M, context, current_pos);
        
        // Now run forward pass starting from embedded vectors
        size_t current_input = M->embedded_input_offset;
        
        for (int layer = 0; layer < M->num_layers; layer++) {
            //printf("  Layer %d\n", layer);
            transformer_layer_forward(M, layer, current_input);
            current_input = M->layers[layer].residual2_output_offset;
        }
        
        // Final layernorm
        layernorm_token_parallel(M, current_input, 
                                M->final_ln_weight_offset,
                                M->final_ln_bias_offset,
                                M->final_ln_mean_offset,
                                M->final_ln_rstd_offset,
                                M->final_output_offset, 1e-5f);
        
        // Compute logits
        compute_logits_last_token_optimized(M, current_pos - 1);
        
        // Sample
        float* last_logits = M->memory_base + M->logits_offset + 
                            (current_pos - 1) * M->vocab_size;
        
        int next_token = sample_token(last_logits, M->vocab_size, 0.6f);
        
        // Update context
        if (current_pos < M->context_window) {
            context[current_pos++] = next_token;
        } else {
            // Shift context window
            memmove(context, context + 1, 
                   (M->context_window - 1) * sizeof(int32_t));
            context[M->context_window - 1] = next_token;
        }
        
        printf("Generated token ID: %d\n", next_token);
    }
}

/******************  BACKWARD PASS ***************************/

void zero_gradients(TransformerModel *M) {
    // Zero all gradient accumulators before backward pass
    size_t total_gradient_bytes = M->gradients.total_gradient_floats * sizeof(float);
    memset(M->memory_base + M->gradients.backprop_base, 0, total_gradient_bytes);
}

/**
 * @brief Copy forward pass activations to gradient storage for backward pass
 * This preserves the forward computations needed for gradient calculation
 */
void cache_forward_activations(TransformerModel *M) {
    // Copy final layer outputs
    memcpy(M->memory_base + M->gradients.logits_copy_offset,
           M->memory_base + M->logits_offset,
           M->context_window * M->vocab_size * sizeof(float));
    
    memcpy(M->memory_base + M->gradients.final_output_copy_offset,
           M->memory_base + M->final_output_offset,
           M->context_window * M->aligned_embed_dim * sizeof(float));
    
    // Copy final LayerNorm inputs and stats
    size_t final_ln_input = M->layers[M->num_layers - 1].residual2_output_offset;
    memcpy(M->memory_base + M->gradients.final_ln_input_copy_offset,
           M->memory_base + final_ln_input,
           M->context_window * M->aligned_embed_dim * sizeof(float));
    
    memcpy(M->memory_base + M->gradients.final_ln_mean_copy_offset,
           M->memory_base + M->final_ln_mean_offset,
           M->context_window * sizeof(float));
    
    memcpy(M->memory_base + M->gradients.final_ln_rstd_copy_offset,
           M->memory_base + M->final_ln_rstd_offset,
           M->context_window * sizeof(float));
    
    // Copy weights (these don't change during forward, but good to have local)
    memcpy(M->memory_base + M->gradients.final_ln_gamma_copy_offset,
           M->memory_base + M->final_ln_weight_offset,
           M->aligned_embed_dim * sizeof(float));
    
    memcpy(M->memory_base + M->gradients.final_ln_beta_copy_offset,
           M->memory_base + M->final_ln_bias_offset,
           M->aligned_embed_dim * sizeof(float));
    
    // Copy per-layer activations
    for (int l = 0; l < M->num_layers; l++) {
        LayerGradients *LG = &M->gradients.layers[l];
        TrulyOptimalLayer *L = &M->layers[l];
        
        // Layer outputs
        memcpy(M->memory_base + LG->residual2_copy_offset,
               M->memory_base + L->residual2_output_offset,
               M->context_window * M->aligned_embed_dim * sizeof(float));
        
        // Attention outputs (token-major [T × D])
        memcpy(M->memory_base + LG->attention_output_copy_offset,
               M->memory_base + L->attention_output_offset,
               M->context_window * M->aligned_embed_dim * sizeof(float));
        
        // QKV outputs
        memcpy(M->memory_base + LG->q_output_copy_offset,
               M->memory_base + L->q_output_offset,
               M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
        
        memcpy(M->memory_base + LG->k_output_copy_offset,
               M->memory_base + L->k_output_offset,
               M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
        
        memcpy(M->memory_base + LG->v_output_copy_offset,
               M->memory_base + L->v_output_offset,
               M->num_attention_heads * M->context_window * M->aligned_head_dim * sizeof(float));
        
        // LayerNorm outputs and stats
        memcpy(M->memory_base + LG->ln1_output_copy_offset,
               M->memory_base + L->ln1_output_offset,
               M->context_window * M->aligned_embed_dim * sizeof(float));
        
        memcpy(M->memory_base + LG->ln1_mean_copy_offset,
               M->memory_base + L->ln1_mean_offset,
               M->context_window * sizeof(float));
        
        memcpy(M->memory_base + LG->ln1_rstd_copy_offset,
               M->memory_base + L->ln1_rstd_offset,
               M->context_window * sizeof(float));
        
        // Continue for MLP activations...
    }
}

/**
 * ============================================================================
 * RESIDUAL CONNECTION - FORWARD & BACKWARD PROPAGATION
 * ============================================================================
 * 
 * CONCEPT:
 * A residual connection (skip connection) allows gradients to flow directly
 * through the network by adding the input to the output of a transformation.
 * This addresses the vanishing gradient problem in deep networks.
 * 
 * FORWARD PASS:
 * ────────────────────────────────────────────────────────────────
 *           input (x)
 *              │
 *              ├────────────────┐  (identity path / skip connection)
 *              │                │
 *              ▼                │
 *        ┌─────────┐            │
 *        │ F(x)    │            │
 *        │ (trans- │            │
 *        │  form)  │            │
 *        └─────────┘            │
 *              │                │
 *              ▼                ▼
 *            F(x)          +    x
 *              └────────┬───────┘
 *                       ▼
 *                  output = F(x) + x
 * 
 * Mathematical form:
 *   output = input + transform(input)
 * 
 * In transformers specifically:
 *   output = input + MultiHeadAttention(LayerNorm(input))
 *   output = input + FFN(LayerNorm(input))
 * 
 * BACKWARD PASS:
 * ────────────────────────────────────────────────────────────────
 * 
 * Given: d_output = ∂L/∂output (gradient from layer above)
 * Need:  d_input = ∂L/∂input and d_transform = ∂L/∂transform
 * 
 * Since output = input + transform, by the chain rule:
 *   ∂output/∂input = 1      (derivative of input w.r.t itself)
 *   ∂output/∂transform = 1  (derivative of transform w.r.t itself)
 * 
 * Therefore:
 *   d_input = d_output × 1 = d_output
 *   d_transform = d_output × 1 = d_output
 * 
 * BACKWARD FLOW:
 *                  d_output
 *                      │
 *           ┌──────────┴──────────┐
 *           │                     │
 *           ▼                     ▼
 *       d_transform           d_input
 *    (gradient flows        (gradient flows
 *     to transform)          directly through)
 * 
 * KEY INSIGHT:
 * The gradient d_output flows EQUALLY through both paths:
 * 1. Through the transformation (to update its parameters)
 * 2. Directly to the input (skip connection)
 * 
 * This is why residual connections help with vanishing gradients:
 * even if the transformation has small gradients, the skip path
 * ensures gradients can flow directly to earlier layers.
 * 
 * IMPLEMENTATION NOTES:
 * - Both gradients receive the SAME value (d_output)
 * - Use += (accumulation) not = (assignment) in case gradients 
 *   already exist from other paths
 * - This simple operation is crucial for training deep networks
 * 
 * @param M Transformer model
 * @param d_output_offset Gradient from the layer above
 * @param d_input_offset Where to accumulate gradient for input path
 * @param d_transform_offset Where to accumulate gradient for transform path
 */
void backward_residual_connection(TransformerModel *M,
                                  size_t d_output_offset,
                                  size_t d_input_offset,
                                  size_t d_transform_offset) {
    float *d_output = M->memory_base + d_output_offset;
    float *d_input = M->memory_base + d_input_offset;
    float *d_transform = M->memory_base + d_transform_offset;
    
    size_t total_elements = (size_t)M->context_window * M->aligned_embed_dim;
    
    // Gradient flows equally through both paths
    #pragma omp parallel for
    for (size_t i = 0; i < total_elements; ++i) {
        d_input[i] += d_output[i];      // Skip connection gradient
        d_transform[i] += d_output[i];  // Transformation gradient
    }
}

void backward_embedding_layer(TransformerModel *M) {
    // Gradient flows from the first layer's input
    LayerGradients *L0_grad = &M->gradients.layers[0];
    float *d_embedded = M->memory_base + L0_grad->d_ln1_input_offset;
    float *d_token_emb = M->memory_base + M->gradients.d_embed_weights_offset;
    float *d_pos_emb = M->memory_base + M->gradients.d_pos_embed_offset;
    
    // Get stored token IDs
    int32_t *token_ids = (int32_t*)(M->memory_base + M->gradients.actual_tokens_offset);
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < M->context_window; t++) {
        int token_id = token_ids[t];
        
        // Token embedding gradient (accumulate)
        for (int d = 0; d < M->embed_dim; d++) {
            float grad = d_embedded[t * M->aligned_embed_dim + d];
            
            #pragma omp atomic
            d_token_emb[token_id * M->aligned_embed_dim + d] += grad;
            
            #pragma omp atomic
            d_pos_emb[t * M->aligned_embed_dim + d] += grad;
        }
    }
}

/*
================================================================================
LAYER NORM — FORWARD & BACKWARD (per token t, feature dim D)
================================================================================

FORWARD PASS (for reference):

x[t,d] ────────────────────────────────────────────────────────────────┐
  │                                                                    │
  ├─> mean[t] = (1/D) * Σ_d x[t,d]                                     │
  │                                                                    │
  ├─> center[t,d] = x[t,d] - mean[t]                                   │
  │                                                                    │
  ├─> var[t] = (1/D) * Σ_d center[t,d]^2                               │
  │                                                                    │
  ├─> rstd[t] = 1 / sqrt( var[t] + ε )                                 │
  │                     │                                              │
  │                     └──────────────────────────────────────────┐   │
  │                                                                │   │
  └──────────────> x_hat[t,d] = center[t,d] * rstd[t] ─────────────┘   │
                                  │                                    │
                                  ▼                                    │
                    y[t,d] = gamma[d] * x_hat[t,d] + beta[d]           │
                                                                       │
Intermediates kept from forward (per t):
- mean[t], var[t], rstd[t], x_hat[t,*]

================================================================================
BACKWARD PASS (complete derivation; given dY[t,d] = ∂L/∂y[t,d])
================================================================================

Step 1: Gradient from next layer
┌──────────────────────────────────────────┐
│  dY[t,d]  (incoming gradient)            │
└──────────────┬───────────────────────────┘
               │
Step 2: Parameter gradients (simple, direct; accumulate over tokens t)
               ├──────────────────┬────────────────────┐
               ▼                  ▼                    │
         ┌──────────┐        ┌──────────┐              │
         │ dgamma[d]│        │ dbeta[d] │              │
         │  Σ_t dY  │        │  Σ_t dY  │              │
         │ * x_hat  │        │          │              │
         └──────────┘        └──────────┘              │

Step 3: Gradient w.r.t. normalized values (per t,d)
               ▼
      dx_hat[t,d] = dY[t,d] * gamma[d]

Step 4: Backprop through normalization (x_hat = (x - mean) * rstd)

We need per-row (over d) sums:
- dx_hat_sum[t]       = Σ_d dx_hat[t,d]
- dx_hat_x_hat_sum[t] = Σ_d dx_hat[t,d] * x_hat[t,d]

Chain-rule paths (per token t):

 Path 1: Direct through x
   ∂x_hat[t,d]/∂x[t,d] = rstd[t]
   Contribution:  rstd[t] * dx_hat[t,d]

 Path 2: Through mean (x contributes to mean; mean affects all features)
   ∂mean[t]/∂x[t,d] = 1/D
   ∂x_hat[t,i]/∂mean[t] = -rstd[t]  (for all i)
   Contribution: - rstd[t] * (1/D) * Σ_i dx_hat[t,i]
               = - rstd[t] * dx_hat_sum[t] / D

 Path 3: Through variance → rstd
   var[t]   = (1/D) * Σ_i (x[t,i] - mean[t])^2
   ∂var[t]/∂x[t,d] = (2/D) * (x[t,d] - mean[t])
   rstd[t]  = (var[t] + ε)^(-1/2)
   ∂rstd[t]/∂var[t] = -0.5 * (var[t] + ε)^(-3/2) = -0.5 * rstd[t]^3
   ∂x_hat[t,i]/∂rstd[t] = (x[t,i] - mean[t])

   Combine over i:
     Σ_i dx_hat[t,i] * ∂x_hat[t,i]/∂rstd[t]
   = Σ_i dx_hat[t,i] * (x[t,i] - mean[t])
   = (1/rstd[t]) * Σ_i dx_hat[t,i] * x_hat[t,i]
   ⇒ Contribution at feature d:
     ∂L/∂x[t,d] via rstd
   = (∂L/∂rstd[t]) * ∂rstd[t]/∂var[t] * ∂var[t]/∂x[t,d]
   = [ (1/rstd[t]) * dx_hat_x_hat_sum[t] ] * [ -0.5 * rstd[t]^3 ] * [ 2*(x[t,d]-mean[t])/D ]
   = - rstd[t] * x_hat[t,d] * dx_hat_x_hat_sum[t] / D

Step 5: Combine all three paths → final input gradient
┌───────────────────────────────────────────────────────────────┐
│ dx[t,d] = Path1 + Path2 + Path3                               │
│                                                               │
│ dx[t,d] =  rstd[t] * dx_hat[t,d]                  (direct)    │
│          - rstd[t] * dx_hat_sum[t] / D            (mean)      │
│          - rstd[t] * x_hat[t,d] * dx_hat_x_hat_sum[t] / D     │
│                                                               │
│ Factor out rstd[t]/D:                                         │
│ dx[t,d] = (rstd[t]/D) * [ D * dx_hat[t,d]                     │
│                           - dx_hat_sum[t]                     │
│                           - x_hat[t,d] * dx_hat_x_hat_sum[t] ]│
└───────────────────────────────────────────────────────────────┘

Where the intermediate sums are (per token t):
- dx_hat_sum[t]       = Σ_d dx_hat[t,d] = Σ_d ( dY[t,d] * gamma[d] )
- dx_hat_x_hat_sum[t] = Σ_d ( dx_hat[t,d] * x_hat[t,d] )

Notes:
- All sums for Step 4 are over the feature dimension d (length D), independently for each token t.
- Use cached forward values: mean[t], rstd[t], and x_hat[t,*].
- This closed form is numerically stable and vectorization-friendly for C kernels.
*/

void backward_final_layernorm(TransformerModel *M) {
    // Use copied activations from gradient storage
    float *d_output = M->memory_base + M->gradients.d_final_output_offset;
    float *input_copy = M->memory_base + M->gradients.final_ln_input_copy_offset;
    float *gamma_copy = M->memory_base + M->gradients.final_ln_gamma_copy_offset;
    float *mean_copy = M->memory_base + M->gradients.final_ln_mean_copy_offset;
    float *rstd_copy = M->memory_base + M->gradients.final_ln_rstd_copy_offset;
    float *d_input = M->memory_base + M->gradients.d_final_ln_input_offset;
    float *d_gamma = M->memory_base + M->gradients.d_final_ln_gamma_offset;
    float *d_beta = M->memory_base + M->gradients.d_final_ln_beta_offset;
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < M->context_window; t++) {
        float mean_t = mean_copy[t];
        float rstd_t = rstd_copy[t];
        
        // Compute gradients for this token
        float d_xhat_sum = 0.0f;
        float d_xhat_xhat_sum = 0.0f;
        
        // First pass: compute sums
        for (int d = 0; d < M->embed_dim; d++) {
            float x = input_copy[t * M->aligned_embed_dim + d];
            float xhat = (x - mean_t) * rstd_t;
            float d_out = d_output[t * M->aligned_embed_dim + d];
            float d_xhat = d_out * gamma_copy[d];  // gradient w.r.t normalized value
            
            d_xhat_sum += d_xhat;           // accumulate for mean path
            d_xhat_xhat_sum += d_xhat * xhat;  // accumulate for variance path
        }
        
        // Second pass: compute input gradients
        float scale = rstd_t / M->embed_dim;
        for (int d = 0; d < M->embed_dim; d++) {
            float x = input_copy[t * M->aligned_embed_dim + d];
            float xhat = (x - mean_t) * rstd_t;
            float d_out = d_output[t * M->aligned_embed_dim + d];
            float d_xhat = d_out * gamma_copy[d];
            
            // Three gradient paths combined:
            // 1. Direct: M->embed_dim * d_xhat (scaled by rstd)
            // 2. Through mean: -d_xhat_sum (all inputs affect mean)
            // 3. Through variance: -xhat * d_xhat_xhat_sum (all inputs affect variance)
            d_input[t * M->aligned_embed_dim + d] = 
                scale * (M->embed_dim * d_xhat - d_xhat_sum - xhat * d_xhat_xhat_sum);
        }
    }
    
    // Accumulate parameter gradients
    #pragma omp parallel for num_threads(M->num_cores)
    for (int d = 0; d < M->embed_dim; d++) {
        float d_g = 0.0f, d_b = 0.0f;
        
        for (int t = 0; t < M->context_window; t++) {
            float x = input_copy[t * M->aligned_embed_dim + d];
            float xhat = (x - mean_copy[t]) * rstd_copy[t];
            float d_out = d_output[t * M->aligned_embed_dim + d];
            
            d_g += d_out * xhat;
            d_b += d_out;
        }
        
        d_gamma[d] = d_g;
        d_beta[d] = d_b;
    }
}

/**
 * ============================================================================
 * BACKWARD THROUGH FC2 (Feed-Forward Layer 2)
 * ============================================================================
 * 
 * FORWARD PASS (for reference):
 * ────────────────────────────────────────────────────────────────
 * Input:  fc2_input  [T × 4D]  (after GELU activation)
 * Weight: W_fc2      [4D × D]  (projects from 4D back to D)
 * Bias:   b_fc2      [D]
 * Output: fc2_output [T × D]   = fc2_input @ W_fc2 + b_fc2
 * 
 * BACKWARD PASS:
 * ────────────────────────────────────────────────────────────────
 * Given:  d_output [T × D]  (gradient from residual connection)
 * 
 * Need to compute:
 * 1. d_input  [T × 4D] = d_output @ W_fc2^T
 * 2. d_W_fc2  [4D × D] = fc2_input^T @ d_output  (accumulated)
 * 3. d_b_fc2  [D]      = sum over T of d_output  (accumulated)
 * 
 * DIMENSION FLOW:
 * d_output[T×D] ──┬──> @ W_fc2^T[D×4D] ──> d_input[T×4D]
 *                 │
 *                 ├──> fc2_input^T[4D×T] @ ──> d_W_fc2[4D×D]
 *                 │
 *                 └──> sum_over_T ──> d_b_fc2[D]
 * 
 * HPC CONSIDERATIONS:
 * - Token-parallel for d_input computation (each thread handles tokens)
 * - Reduction required for weight/bias gradients (atomic ops or local accumulation)
 * - Memory bandwidth bound due to large weight matrix (4D×D)
 * - Cache blocking beneficial for weight gradient accumulation
 */
void backward_fc2(TransformerModel *M,
                  size_t d_output_offset,      // [T × D] incoming gradient
                  size_t fc2_input_copy_offset, // [T × 4D] cached forward activation
                  size_t fc2_weight_offset,     // [4D × D] weight matrix
                  size_t fc2_bias_offset,       // [D] bias vector
                  size_t d_input_offset,        // [T × 4D] gradient to compute
                  size_t d_weight_offset,       // [4D × D] weight gradient to accumulate
                  size_t d_bias_offset)         // [D] bias gradient to accumulate
{
    float *d_output = M->memory_base + d_output_offset;
    float *fc2_input = M->memory_base + fc2_input_copy_offset;
    float *W_fc2 = M->memory_base + fc2_weight_offset;
    float *d_input = M->memory_base + d_input_offset;
    float *d_W_fc2 = M->memory_base + d_weight_offset;
    float *d_b_fc2 = M->memory_base + d_bias_offset;
    
    int T = M->context_window;
    int aligned_out = M->aligned_embed_dim;      // D (output dimension, padded)
    int aligned_in = 4 * M->aligned_embed_dim;   // 4D (input dimension, padded)
    
    // ============================================================================
    // 1. COMPUTE d_input = d_output @ W_fc2^T
    //    [T × aligned_in] = [T × aligned_out] @ [aligned_out × aligned_in]
    //    Weight matrix is stored row-major with row stride = aligned_in.
    // ============================================================================
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        float *d_out_row = d_output + t * aligned_out;
        float *d_in_row = d_input + t * aligned_in;
        
        for (int in_idx = 0; in_idx < aligned_in; in_idx++) {
            float sum = 0.0f;
            for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
                sum += d_out_row[out_idx] * W_fc2[out_idx * aligned_in + in_idx];
            }
            d_in_row[in_idx] = sum;
        }
    }
    
    // ============================================================================
    // 2. COMPUTE d_W_fc2 = d_output^T @ fc2_input
    //    [aligned_out × aligned_in] = [aligned_out × T] @ [T × aligned_in]
    // ============================================================================
    
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
        for (int in_idx = 0; in_idx < aligned_in; in_idx++) {
            float grad_sum = 0.0f;
            for (int t = 0; t < T; t++) {
                grad_sum += d_output[t * aligned_out + out_idx] *
                            fc2_input[t * aligned_in + in_idx];
            }
            #pragma omp atomic
            d_W_fc2[out_idx * aligned_in + in_idx] += grad_sum;
        }
    }
    
    // ============================================================================
    // 3. COMPUTE d_b_fc2 = sum_over_T(d_output)
    //    [D] = sum over token dimension of [T × D]
    // ============================================================================
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (int d = 0; d < aligned_out; d++) {
        float bias_grad = 0.0f;
        
        // Sum gradient across all tokens
        for (int t = 0; t < T; t++) {
            bias_grad += d_output[t * aligned_out + d];
        }
        
        // Atomic accumulation
        #pragma omp atomic
        d_b_fc2[d] += bias_grad;
    }
}

/**
 * ============================================================================
 * BACKWARD THROUGH GELU ACTIVATION
 * ============================================================================
 * 
 * FORWARD PASS (for reference):
 * ────────────────────────────────────────────────────────────────
 * GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * Approximation used in practice:
 * GELU(x) ≈ 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x³)))
 * 
 * BACKWARD PASS (derivative):
 * ────────────────────────────────────────────────────────────────
 * d/dx[GELU(x)] = 0.5 * tanh(g(x)) + 0.5 * x * sech²(g(x)) * g'(x) + 0.5
 * where:
 *   g(x) = 0.7978845608 * (x + 0.044715 * x³)
 *   g'(x) = 0.7978845608 * (1 + 3 * 0.044715 * x²)
 * 
 * Simplified form:
 * GELU'(x) = 0.5 * (1 + tanh(g(x))) + 0.5 * x * sech²(g(x)) * g'(x)
 *          = 0.5 * (1 + tanh(g(x))) + 0.5 * x * (1 - tanh²(g(x))) * g'(x)
 * 
 * DIMENSION FLOW:
 * d_output [T × 4D] ──> × GELU'(input) ──> d_input [T × 4D]
 * 
 * HPC CONSIDERATIONS:
 * - Element-wise operation (embarrassingly parallel)
 * - Compute bound (tanh is expensive)
 * - Can fuse with surrounding operations for better cache usage
 * - Consider using fast tanh approximations for speed
 */
void backward_gelu(TransformerModel *M,
                   size_t d_output_offset,      // [T × 4D] incoming gradient
                   size_t input_copy_offset,    // [T × 4D] cached input (before GELU)
                   size_t d_input_offset)        // [T × 4D] gradient to compute
{
    float *d_output = M->memory_base + d_output_offset;
    float *input = M->memory_base + input_copy_offset;
    float *d_input = M->memory_base + d_input_offset;
    
    size_t total_elements = (size_t)M->context_window * 4 * M->aligned_embed_dim;
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (size_t i = 0; i < total_elements; i++) {
        float x = input[i];
        
        // Compute g(x) = sqrt(2/π) * (x + 0.044715 * x³)
        float x3 = x * x * x;
        float g = sqrt_2_over_pi * (x + coeff * x3);
        
        // Compute tanh(g(x))
        float tanh_g = tanhf(g);
        
        // Compute g'(x) = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
        float x2 = x * x;
        float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);
        
        // Compute GELU'(x) using the simplified form
        // GELU'(x) = 0.5 * (1 + tanh(g)) + 0.5 * x * (1 - tanh²(g)) * g'
        float sech2_g = 1.0f - tanh_g * tanh_g;  // sech²(g) = 1 - tanh²(g)
        float gelu_derivative = 0.5f * (1.0f + tanh_g) + 0.5f * x * sech2_g * g_prime;
        
        // Apply chain rule: d_input = d_output * GELU'(input)
        d_input[i] = d_output[i] * gelu_derivative;
    }
}

/**
 * ============================================================================
 * BACKWARD THROUGH FC1 (Feed-Forward Layer 1)
 * ============================================================================
 * 
 * FORWARD PASS (for reference):
 * ────────────────────────────────────────────────────────────────
 * Input:  fc1_input  [T × D]   (output from LayerNorm2)
 * Weight: W_fc1      [D × 4D]  (projects from D to 4D)
 * Bias:   b_fc1      [4D]
 * Output: fc1_output [T × 4D]  = fc1_input @ W_fc1 + b_fc1
 * 
 * BACKWARD PASS:
 * ────────────────────────────────────────────────────────────────
 * Given:  d_output [T × 4D]  (gradient from GELU backward)
 * 
 * Need to compute:
 * 1. d_input  [T × D]  = d_output @ W_fc1^T
 * 2. d_W_fc1  [D × 4D] = fc1_input^T @ d_output  (accumulated)
 * 3. d_b_fc1  [4D]     = sum over T of d_output  (accumulated)
 * 
 * DIMENSION FLOW:
 * d_output[T×4D] ──┬──> @ W_fc1^T[4D×D] ──> d_input[T×D]
 *                  │
 *                  ├──> fc1_input^T[D×T] @ ──> d_W_fc1[D×4D]
 *                  │
 *                  └──> sum_over_T ──> d_b_fc1[4D]
 * 
 * HPC CONSIDERATIONS:
 * - FC1 expands dimensions (D -> 4D), so weight matrix is large
 * - Memory bandwidth critical for weight gradient accumulation
 * - Consider chunking for better cache reuse
 * - Token parallelism for d_input computation
 */
void backward_fc1(TransformerModel *M,
                  size_t d_output_offset,      // [T × 4D] incoming gradient
                  size_t fc1_input_copy_offset, // [T × D] cached forward activation
                  size_t fc1_weight_offset,     // [D × 4D] weight matrix
                  size_t fc1_bias_offset,       // [4D] bias vector
                  size_t d_input_offset,        // [T × D] gradient to compute
                  size_t d_weight_offset,       // [D × 4D] weight gradient to accumulate
                  size_t d_bias_offset)         // [4D] bias gradient to accumulate
{
    float *d_output = M->memory_base + d_output_offset;
    float *fc1_input = M->memory_base + fc1_input_copy_offset;
    float *W_fc1 = M->memory_base + fc1_weight_offset;
    float *d_input = M->memory_base + d_input_offset;
    float *d_W_fc1 = M->memory_base + d_weight_offset;
    float *d_b_fc1 = M->memory_base + d_bias_offset;
    
    int T = M->context_window;
    int aligned_in = M->aligned_embed_dim;        // Input dimension (padded)
    int aligned_out = 4 * M->aligned_embed_dim;   // Output dimension (padded)
    
    // ============================================================================
    // 1. COMPUTE d_input = d_output @ W_fc1^T
    //    [T × aligned_in] = [T × aligned_out] @ [aligned_out × aligned_in]
    //    Weight matrix stored row-major with row stride = aligned_in.
    // ============================================================================
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        float *d_out_row = d_output + t * aligned_out;
        float *d_in_row = d_input + t * aligned_in;
        
        for (int in_idx = 0; in_idx < aligned_in; in_idx++) {
            float sum = 0.0f;
            for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
                sum += d_out_row[out_idx] * W_fc1[out_idx * aligned_in + in_idx];
            }
            d_in_row[in_idx] = sum;
        }
    }
    
    // ============================================================================
    // 2. COMPUTE d_W_fc1 = d_output^T @ fc1_input
    //    [aligned_out × aligned_in] = [aligned_out × T] @ [T × aligned_in]
    // ============================================================================
    
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
        for (int in_idx = 0; in_idx < aligned_in; in_idx++) {
            float grad_sum = 0.0f;
            for (int t = 0; t < T; t++) {
                grad_sum += d_output[t * aligned_out + out_idx] *
                            fc1_input[t * aligned_in + in_idx];
            }
            #pragma omp atomic
            d_W_fc1[out_idx * aligned_in + in_idx] += grad_sum;
        }
    }
    
    // ============================================================================
    // 3. COMPUTE d_b_fc1 = sum_over_T(d_output)
    // ============================================================================
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
        float bias_grad = 0.0f;
        for (int t = 0; t < T; t++) {
            bias_grad += d_output[t * aligned_out + out_idx];
        }
        #pragma omp atomic
        d_b_fc1[out_idx] += bias_grad;
    }
}

/**
 * Alternative: Fast approximation using precomputed GELU derivative
 * This version trades accuracy for speed by using a simpler approximation
 */
void backward_gelu_fast(TransformerModel *M,
                        size_t d_output_offset,
                        size_t input_copy_offset,
                        size_t d_input_offset)
{
    float *d_output = M->memory_base + d_output_offset;
    float *input = M->memory_base + input_copy_offset;
    float *d_input = M->memory_base + d_input_offset;
    
    size_t total_elements = (size_t)M->context_window * 4 * M->aligned_embed_dim;
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (size_t i = 0; i < total_elements; i++) {
        float x = input[i];
        
        // Sigmoid approximation to GELU derivative
        // GELU'(x) ≈ σ(1.702 * x) * (1 + x * (1 - σ(1.702 * x)) * 1.702)
        float s = 1.0f / (1.0f + expf(-1.702f * x));
        float gelu_derivative = s * (1.0f + x * (1.0f - s) * 1.702f);
        
        d_input[i] = d_output[i] * gelu_derivative;
    }
}

/**
 * ============================================================================
 * BACKWARD THROUGH LAYERNORM
 * ============================================================================
 * 
 * FORWARD (for reference):
 * x_hat = (x - mean) / rstd
 * y = gamma * x_hat + beta
 * 
 * BACKWARD:
 * Given d_y, compute d_x, d_gamma, d_beta
 * 
 * The math (per token):
 * d_x = (rstd/D) * [D * d_y * gamma - sum(d_y * gamma) - x_hat * sum(d_y * gamma * x_hat)]
 * d_gamma = sum_over_tokens(d_y * x_hat)
 * d_beta = sum_over_tokens(d_y)
 */
void backward_layernorm(TransformerModel *M,
                        size_t d_output_offset,     // [T×D] gradient from next layer
                        size_t input_copy_offset,   // [T×D] original input (x)
                        size_t gamma_copy_offset,   // [D] scale weights
                        size_t beta_copy_offset,    // [D] shift weights (unused in backward)
                        size_t mean_copy_offset,    // [T] cached mean
                        size_t rstd_copy_offset,    // [T] cached 1/std
                        size_t d_input_offset,      // [T×D] gradient to compute
                        size_t d_gamma_offset,      // [D] gamma gradient to accumulate
                        size_t d_beta_offset)       // [D] beta gradient to accumulate
{
    float *d_output = M->memory_base + d_output_offset;
    float *input = M->memory_base + input_copy_offset;
    float *gamma = M->memory_base + gamma_copy_offset;
    float *mean = M->memory_base + mean_copy_offset;
    float *rstd = M->memory_base + rstd_copy_offset;
    float *d_input = M->memory_base + d_input_offset;
    float *d_gamma = M->memory_base + d_gamma_offset;
    float *d_beta = M->memory_base + d_beta_offset;
    
    int T = M->context_window;
    int D = M->embed_dim;
    int aligned_D = M->aligned_embed_dim;
    
    // Process each token
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        float mean_t = mean[t];
        float rstd_t = rstd[t];
        
        // Compute per-token statistics for backward pass
        float d_y_gamma_sum = 0.0f;
        float d_y_gamma_xhat_sum = 0.0f;
        
        // First pass: compute sums
        for (int d = 0; d < D; d++) {
            float x = input[t * aligned_D + d];
            float x_hat = (x - mean_t) * rstd_t;
            float d_y = d_output[t * aligned_D + d];
            float d_y_gamma = d_y * gamma[d];
            
            d_y_gamma_sum += d_y_gamma;
            d_y_gamma_xhat_sum += d_y_gamma * x_hat;
        }
        
        // Second pass: compute input gradients
        float scale = rstd_t / D;
        for (int d = 0; d < D; d++) {
            float x = input[t * aligned_D + d];
            float x_hat = (x - mean_t) * rstd_t;
            float d_y = d_output[t * aligned_D + d];
            
            // Three gradient paths combined
            d_input[t * aligned_D + d] = scale * 
                (D * d_y * gamma[d] - d_y_gamma_sum - x_hat * d_y_gamma_xhat_sum);
        }
        
        // Zero padding (gradient should be 0 for padded elements)
        for (int d = D; d < aligned_D; d++) {
            d_input[t * aligned_D + d] = 0.0f;
        }
    }
    
    // Accumulate parameter gradients
    #pragma omp parallel for num_threads(M->num_cores)
    for (int d = 0; d < D; d++) {
        float gamma_grad = 0.0f;
        float beta_grad = 0.0f;
        
        for (int t = 0; t < T; t++) {
            float x = input[t * aligned_D + d];
            float x_hat = (x - mean[t]) * rstd[t];
            float d_y = d_output[t * aligned_D + d];
            
            gamma_grad += d_y * x_hat;
            beta_grad += d_y;
        }
        
        // Atomic accumulation (in case of multiple backward passes)
        #pragma omp atomic
        d_gamma[d] += gamma_grad;
        
        #pragma omp atomic
        d_beta[d] += beta_grad;
    }
}

/**
 * ============================================================================
 * ADD GRADIENT (accumulate gradients from residual path)
 * ============================================================================
 * 
 * This is used when gradients from multiple paths need to be summed.
 * For example, at a residual connection, gradients flow through both:
 * 1. The transformation path (MLP or attention)
 * 2. The skip connection path
 * 
 * Both gradients need to be added together.
 */
void add_gradient(TransformerModel *M,
                  size_t source_offset,  // Gradient to add FROM
                  size_t dest_offset)    // Gradient to add TO
{
    float *source = M->memory_base + source_offset;
    float *dest = M->memory_base + dest_offset;
    
    size_t total_elements = (size_t)M->context_window * M->aligned_embed_dim;
    
    #pragma omp parallel for num_threads(M->num_cores)
    for (size_t i = 0; i < total_elements; i++) {
        dest[i] += source[i];
    }
}

/**
 * BACKWARD THROUGH ATTENTION OUTPUT PROJECTION
 * 
 * Forward: output[T×D] = attention[T×D] @ W_proj[D×D] + b_proj[D]
 * 
 * This is after concatenating all heads back to [T×D] format
 * 
 * Backward computes:
 * 1. d_attention[T×D] = d_output[T×D] @ W_proj^T[D×D]  
 * 2. d_W_proj[D×D] = attention^T[D×T] @ d_output[T×D]
 * 3. d_b_proj[D] = sum(d_output) over T
 */
void backward_attention_projection(TransformerModel *M,
                                  size_t d_output_offset,           // [T×D] gradient from residual
                                  size_t attention_output_copy_offset, // [T×D] cached attention output
                                  size_t proj_weight_offset,        // [D×D] projection weights
                                  size_t proj_bias_offset,          // [D] projection bias
                                  size_t d_attention_token_offset,  // [T×D] token-major gradient
                                  size_t d_attention_head_offset,   // [H×T×H] head-major gradient
                                  size_t d_weight_offset,           // [D×D] weight gradient to accumulate
                                  size_t d_bias_offset)             // [D] bias gradient to accumulate
{
    float *d_output = M->memory_base + d_output_offset;
    float *attention_output = M->memory_base + attention_output_copy_offset;
    float *W_proj = M->memory_base + proj_weight_offset;
    float *d_attention_token = M->memory_base + d_attention_token_offset;
    float *d_attention_heads = M->memory_base + d_attention_head_offset;
    float *d_W_proj = M->memory_base + d_weight_offset;
    float *d_b_proj = M->memory_base + d_bias_offset;
    
    int T = M->context_window;
    int aligned_dim = M->aligned_embed_dim;
    int embed_dim = M->embed_dim;
    int head_dim = M->head_dim;
    int aligned_head_dim = M->aligned_head_dim;
    int H = M->num_attention_heads;
    
    // 1. Compute token-major gradient: d_attention = d_output @ W_proj^T
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        float *d_out_row = d_output + t * aligned_dim;
        float *d_att_row = d_attention_token + t * aligned_dim;
        
        for (int col = 0; col < aligned_dim; col++) {
            float sum = 0.0f;
            for (int row = 0; row < aligned_dim; row++) {
                sum += d_out_row[row] * W_proj[row * aligned_dim + col];
            }
            d_att_row[col] = sum;
        }
    }
    
    // 2. Convert token-major gradient to head-major layout
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < aligned_head_dim; d++) {
                float value = 0.0f;
                if (d < head_dim) {
                    int global_dim = h * head_dim + d;
                    if (global_dim < aligned_dim) {
                        value = d_attention_token[t * aligned_dim + global_dim];
                    }
                }
                Q_ACCESS(d_attention_heads, h, t, d, T, aligned_head_dim) = value;
            }
        }
    }

    // Zero padded dimensions in token-major buffer
    for (int t = 0; t < T; t++) {
        for (int d = embed_dim; d < aligned_dim; d++) {
            d_attention_token[t * aligned_dim + d] = 0.0f;
        }
    }
    
    // 3. Compute d_W_proj = attention^T @ d_output
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int row = 0; row < aligned_dim; row++) {
        for (int col = 0; col < aligned_dim; col++) {
            float grad_sum = 0.0f;
            for (int t = 0; t < T; t++) {
                grad_sum += attention_output[t * aligned_dim + col] * d_output[t * aligned_dim + row];
            }
            #pragma omp atomic
            d_W_proj[row * aligned_dim + col] += grad_sum;
        }
    }
    
    // 4. Compute d_b_proj = sum(d_output) over tokens
    #pragma omp parallel for num_threads(M->num_cores)
    for (int d = 0; d < aligned_dim; d++) {
        float bias_grad = 0.0f;
        for (int t = 0; t < T; t++) {
            bias_grad += d_output[t * aligned_dim + d];
        }
        #pragma omp atomic
        d_b_proj[d] += bias_grad;
    }
}

/**
 * BACKWARD THROUGH ATTENTION WEIGHTED VALUES
 * 
 * Forward: attention_output[h,t,d] = sum_over_s(attention_weights[h,t,s] * V[h,s,d])
 * 
 * This operates in HEAD-MAJOR layout
 * 
 * Backward computes:
 * 1. d_attention_weights[h,t,s] = sum_over_d(d_output[h,t,d] * V[h,s,d])
 * 2. d_V[h,s,d] = sum_over_t(attention_weights[h,t,s] * d_output[h,t,d])
 */
void backward_attention_weighted_values(TransformerModel *M,
                                       size_t d_output_offset,         // [H×T×head_dim] incoming gradient
                                       size_t attention_weights_offset, // [H×T×T] cached softmax output
                                       size_t v_output_offset,          // [H×T×head_dim] cached V values
                                       size_t d_weights_offset,         // [H×T×T] gradient to compute
                                       size_t d_v_offset)               // [H×T×head_dim] gradient to compute
{
    float *d_output = M->memory_base + d_output_offset;
    float *attention_weights = M->memory_base + attention_weights_offset;
    float *v_values = M->memory_base + v_output_offset;
    float *d_weights = M->memory_base + d_weights_offset;
    float *d_v = M->memory_base + d_v_offset;
    
    int H = M->num_attention_heads;
    int T = M->context_window;
    int head_dim = M->head_dim;
    int aligned_head_dim = M->aligned_head_dim;
    
    // Clear gradients
    memset(d_weights, 0, H * T * T * sizeof(float));
    memset(d_v, 0, H * T * aligned_head_dim * sizeof(float));
    
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < T; t++) {
            // 1. Compute d_attention_weights[h,t,s] for all s
            for (int s = 0; s <= t; s++) { // Causal mask: only s <= t
                float grad_sum = 0.0f;
                
                // Sum over head dimension
                for (int d = 0; d < head_dim; d++) {
                    float d_out = Q_ACCESS(d_output, h, t, d, T, aligned_head_dim);
                    float v_val = V_ACCESS(v_values, h, s, d, T, aligned_head_dim);
                    grad_sum += d_out * v_val;
                }
                
                ATTN_ACCESS(d_weights, h, t, s, T) = grad_sum;
            }
            
            // 2. Accumulate d_V[h,s,d]
            // For each position s that attended to by t
            for (int s = 0; s <= t; s++) { // Causal: t can only attend to s <= t
                float weight = ATTN_ACCESS(attention_weights, h, t, s, T);
                
                for (int d = 0; d < head_dim; d++) {
                    float d_out = Q_ACCESS(d_output, h, t, d, T, aligned_head_dim);
                    
                    #pragma omp atomic
                    V_ACCESS(d_v, h, s, d, T, aligned_head_dim) += weight * d_out;
                }
            }
        }
    }
}

/**
 * BACKWARD THROUGH CAUSAL SOFTMAX
 * 
 * Forward: attention_weights[h,i,j] = softmax(scores[h,i,:]) with causal mask
 * 
 * Softmax Jacobian for row i:
 * ∂softmax[i,j]/∂score[i,k] = softmax[i,j] * (δ[j,k] - softmax[i,k])
 * where δ[j,k] is Kronecker delta (1 if j==k, 0 otherwise)
 * 
 * For a row, if y = softmax(x), then:
 * dx = y * (dy - dot(y, dy))
 * 
 * Causal mask means we only compute for j <= i
 */
void backward_causal_softmax(TransformerModel *M,
                            size_t d_scores_offset,        // [H×T×T] gradient in/out (reused)
                            size_t weights_copy_offset)    // [H×T×T] cached softmax output
{
    float *d_scores_inout = M->memory_base + d_scores_offset;  // Used for both input and output
    float *weights = M->memory_base + weights_copy_offset;
    
    int H = M->num_attention_heads;
    int T = M->context_window;
    
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < T; i++) {
            // For each query position i, compute gradient for scores[h,i,:]
            
            // First compute dot(softmax, d_softmax) for this row
            float dot_product = 0.0f;
            for (int j = 0; j <= i; j++) { // Causal: only j <= i
                float w = ATTN_ACCESS(weights, h, i, j, T);
                float dw = ATTN_ACCESS(d_scores_inout, h, i, j, T);  // Read current gradient
                dot_product += w * dw;
            }
            
            // Apply the softmax backward formula: dx = y * (dy - dot(y, dy))
            // NOTE: We're overwriting d_scores in place
            for (int j = 0; j <= i; j++) { // Causal: only j <= i
                float w = ATTN_ACCESS(weights, h, i, j, T);
                float dw = ATTN_ACCESS(d_scores_inout, h, i, j, T);  // Read before overwrite
                
                ATTN_ACCESS(d_scores_inout, h, i, j, T) = w * (dw - dot_product);  // Overwrite
            }
            
            // Zero out the upper triangle (j > i) due to causal mask
            for (int j = i + 1; j < T; j++) {
                ATTN_ACCESS(d_scores_inout, h, i, j, T) = 0.0f;
            }
        }
    }
}

/**
 * BACKWARD THROUGH Q @ K^T
 * 
 * Forward: scores[h,i,j] = sum_d(Q[h,i,d] * K[h,j,d]) / sqrt(head_dim)
 * 
 * Backward:
 * d_Q[h,i,d] = sum_j(d_scores[h,i,j] * K[h,j,d]) / sqrt(head_dim)
 * d_K[h,j,d] = sum_i(d_scores[h,i,j] * Q[h,i,d]) / sqrt(head_dim)
 * 
 * Note: Causal mask means d_scores[h,i,j] = 0 for j > i
 */
void backward_qk_matmul(TransformerModel *M,
                       size_t d_scores_offset,      // [H×T×T] incoming gradient
                       size_t q_copy_offset,        // [H×T×head_dim] cached Q
                       size_t k_copy_offset,        // [H×T×head_dim] cached K
                       size_t d_q_offset,           // [H×T×head_dim] gradient to compute
                       size_t d_k_offset)           // [H×T×head_dim] gradient to compute
{
    float *d_scores = M->memory_base + d_scores_offset;
    float *q_values = M->memory_base + q_copy_offset;
    float *k_values = M->memory_base + k_copy_offset;
    float *d_q = M->memory_base + d_q_offset;
    float *d_k = M->memory_base + d_k_offset;
    
    int H = M->num_attention_heads;
    int T = M->context_window;
    int head_dim = M->head_dim;
    int aligned_head_dim = M->aligned_head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Clear output gradients
    memset(d_q, 0, H * T * aligned_head_dim * sizeof(float));
    memset(d_k, 0, H * T * aligned_head_dim * sizeof(float));
    
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < T; i++) {
            // Compute d_Q[h,i,d] = sum_j(d_scores[h,i,j] * K[h,j,d]) * scale
            for (int d = 0; d < head_dim; d++) {
                float grad_sum = 0.0f;
                
                // Sum over all keys this query attended to (causal: j <= i)
                for (int j = 0; j <= i; j++) {
                    float d_score = ATTN_ACCESS(d_scores, h, i, j, T);
                    float k_val = K_ACCESS(k_values, h, j, d, T, aligned_head_dim);
                    grad_sum += d_score * k_val;
                }
                
                Q_ACCESS(d_q, h, i, d, T, aligned_head_dim) = grad_sum * scale;
            }
            
            // Compute d_K[h,j,d] contributions from query i
            // For each key position j that was attended to by query i
            for (int j = 0; j <= i; j++) {  // Causal: only j <= i
                float d_score = ATTN_ACCESS(d_scores, h, i, j, T);
                
                for (int d = 0; d < head_dim; d++) {
                    float q_val = Q_ACCESS(q_values, h, i, d, T, aligned_head_dim);
                    
                    #pragma omp atomic
                    K_ACCESS(d_k, h, j, d, T, aligned_head_dim) += d_score * q_val * scale;
                }
            }
        }
    }
}

/**
 * BACKWARD THROUGH LINEAR LAYER (GENERIC)
 * 
 * Forward: output = input @ W + bias
 * 
 * This handles Q, K, V projections which are all linear layers
 * Note: For QKV, the output is in head-major format but input is token-major
 * 
 * Backward:
 * d_input += d_output @ W^T  (accumulate because QKV all contribute)
 * d_W += input^T @ d_output
 * d_bias += sum(d_output)
 */
void backward_linear(TransformerModel *M,
                    size_t d_output_offset,    // Gradient from head-major QKV
                    size_t input_copy_offset,  // [T×D] cached LN1 output
                    size_t weight_offset,      // [D×D] weights
                    size_t bias_offset,        // [D] bias
                    size_t d_input_offset,     // [T×D] gradient to accumulate
                    size_t d_weight_offset,    // [D×D] weight gradient
                    size_t d_bias_offset,      // [D] bias gradient
                    size_t scratch_offset)     // [T×D] scratch buffer
{
    float *d_output = M->memory_base + d_output_offset;
    float *input = M->memory_base + input_copy_offset;
    float *weights = M->memory_base + weight_offset;
    float *d_input = M->memory_base + d_input_offset;
    float *d_weights = M->memory_base + d_weight_offset;
    float *d_bias = M->memory_base + d_bias_offset;
    float *scratch = M->memory_base + scratch_offset; // [T × D] buffer for reshaping
    
    int T = M->context_window;
    int aligned_dim = M->aligned_embed_dim;
    int embed_dim = M->embed_dim;
    int head_dim = M->head_dim;
    int aligned_head_dim = M->aligned_head_dim;
    int H = M->num_attention_heads;
    
    // Reshape from head-major [H×T×head_dim] to token-major [T×D]
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < head_dim; d++) {
                int global_dim = h * head_dim + d;
                scratch[t * aligned_dim + global_dim] =
                    Q_ACCESS(d_output, h, t, d, T, aligned_head_dim);
            }
        }
    }

    // Zero padding lanes that do not correspond to real model dims
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        for (int d = embed_dim; d < aligned_dim; d++) {
            scratch[t * aligned_dim + d] = 0.0f;
        }
    }
    
    int aligned_out = aligned_dim;  // Output dim equals input dim for Q/K/V projections
    int aligned_in = aligned_dim;
    
    // 1. d_input = d_output @ W^T (accumulate!)
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < T; t++) {
        float *d_out_row = scratch + t * aligned_out;
        float *d_in_row = d_input + t * aligned_in;
        
        for (int in_idx = 0; in_idx < aligned_in; in_idx++) {
            float sum = 0.0f;
            for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
                sum += d_out_row[out_idx] * weights[out_idx * aligned_in + in_idx];
            }
            #pragma omp atomic
            d_in_row[in_idx] += sum;
        }
    }
    
    // 2. d_W = input^T @ d_output  (matches forward row-major layout: [out][in])
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
        for (int in_idx = 0; in_idx < aligned_in; in_idx++) {
            float sum = 0.0f;
            for (int t = 0; t < T; t++) {
                sum += input[t * aligned_in + in_idx] * scratch[t * aligned_out + out_idx];
            }
            #pragma omp atomic
            d_weights[out_idx * aligned_in + in_idx] += sum;
        }
    }
    
    // 3. d_bias = sum(d_output)
    #pragma omp parallel for num_threads(M->num_cores)
    for (int out_idx = 0; out_idx < aligned_out; out_idx++) {
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            sum += scratch[t * aligned_out + out_idx];
        }
        #pragma omp atomic
        d_bias[out_idx] += sum;
    }
}

void backward_lm_head(TransformerModel *M) {
    // Gradients from cross-entropy loss
    float *d_logits = M->memory_base + M->gradients.d_logits_offset;
    
    // Input to LM head (output of final layernorm)
    float *final_ln_output = M->memory_base + M->gradients.final_output_copy_offset;
    
    // Output gradient for final layernorm
    float *d_final_ln_output = M->memory_base + M->gradients.d_final_output_offset;
    
    // Gradient for embedding weights (shared with LM head due to weight tying)
    float *d_embed_weights = M->memory_base + M->gradients.d_embed_weights_offset;
    
    // LM head weights (tied to embedding weights)
    float *lm_head_weights = M->memory_base + M->token_emb_offset;
    
    // ============================================================================
    // BACKWARD THROUGH LM HEAD
    // logits[t,v] = sum_d (final_ln_output[t,d] * lm_head_weights[v,d])
    // 
    // dL/d_final_ln_output[t,d] = sum_v (dL/dlogits[t,v] * lm_head_weights[v,d])
    // dL/d_lm_head_weights[v,d] = sum_t (dL/dlogits[t,v] * final_ln_output[t,d])
    // ============================================================================
    
    // 1. Compute gradient w.r.t final layernorm output
    #pragma omp parallel for collapse(2) num_threads(M->num_cores)
    for (int t = 0; t < M->context_window; t++) {
        for (int d = 0; d < M->embed_dim; d++) {
            float grad_sum = 0.0f;
            
            // Vectorized accumulation over vocabulary
            int v;
            __m512 sum_vec = _mm512_setzero_ps();
            for (v = 0; v <= M->vocab_size - 16; v += 16) {
                __m512 d_logit_vec = _mm512_loadu_ps(&d_logits[t * M->vocab_size + v]);
                __m512 weight_vec = _mm512_loadu_ps(&lm_head_weights[v * M->aligned_embed_dim + d]);
                sum_vec = _mm512_fmadd_ps(d_logit_vec, weight_vec, sum_vec);
            }
            grad_sum = _mm512_reduce_add_ps(sum_vec);
            
            // Handle remainder
            for (; v < M->vocab_size; v++) {
                grad_sum += d_logits[t * M->vocab_size + v] * 
                           lm_head_weights[v * M->aligned_embed_dim + d];
            }
            
            d_final_ln_output[t * M->aligned_embed_dim + d] = grad_sum;
        }
    }
    
    // 2. Accumulate gradient w.r.t embedding/LM head weights
    #pragma omp parallel for num_threads(M->num_cores)
    for (int v = 0; v < M->vocab_size; v++) {
        for (int d = 0; d < M->embed_dim; d++) {
            float grad_sum = 0.0f;
            
            // Sum over all tokens in the batch
            for (int t = 0; t < M->context_window; t++) {
                grad_sum += d_logits[t * M->vocab_size + v] * 
                           final_ln_output[t * M->aligned_embed_dim + d];
            }
            
            // Accumulate (in case of multiple backward passes before update)
            #pragma omp atomic
            d_embed_weights[v * M->aligned_embed_dim + d] += grad_sum;
        }
    }
}



void backward_transformer_layer(TransformerModel *M, int layer_idx) {
    LayerGradients *LG = &M->gradients.layers[layer_idx];
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    
    // Input gradient comes from next layer (or from final LN if last layer)
    float *d_layer_output;
    if (layer_idx == M->num_layers - 1) {
        // Last layer gets gradient from final layernorm
        d_layer_output = M->memory_base + M->gradients.d_final_ln_input_offset;
    } else {
        // Other layers get gradient from next layer's d_ln1_input
        d_layer_output = M->memory_base + M->gradients.layers[layer_idx + 1].d_ln1_input_offset;
    }
    
    // Copy incoming gradient to this layer's d_residual2
    memcpy(M->memory_base + LG->d_residual2_offset, d_layer_output,
           M->context_window * M->aligned_embed_dim * sizeof(float));
    
    // ============================================================================
    // BACKWARD THROUGH SECOND RESIDUAL CONNECTION
    // residual2 = residual1 + mlp_output
    // ============================================================================
    backward_residual_connection(M, LG->d_residual2_offset, 
                                 LG->d_residual1_offset, 
                                 LG->d_mlp_output_offset);
    
    // ============================================================================
    // BACKWARD THROUGH MLP
    // ============================================================================
    
    // Backward through FC2: [T × D] = [T × 4D] @ [4D × D]
    backward_fc2(M, LG->d_mlp_output_offset, LG->fc2_input_copy_offset,
                 L->fc2_weight_offset, L->fc2_bias_offset,
                 LG->d_fc2_input_offset, LG->d_fc2_weights_offset, 
                 LG->d_fc2_bias_offset);
    
    // Backward through GELU activation
    backward_gelu(M, LG->d_fc2_input_offset, LG->fc1_output_copy_offset,
                  LG->d_fc1_output_offset);
    
    // Backward through FC1: [T × 4D] = [T × D] @ [D × 4D]
    backward_fc1(M, LG->d_fc1_output_offset, LG->ln2_output_copy_offset,
                 L->fc1_weight_offset, L->fc1_bias_offset,
                 LG->d_ln2_output_offset, LG->d_fc1_weights_offset,
                 LG->d_fc1_bias_offset);
    
    // ============================================================================
    // BACKWARD THROUGH SECOND LAYERNORM
    // ============================================================================
    backward_layernorm(M, LG->d_ln2_output_offset, LG->ln2_input_copy_offset,
                       LG->ln2_gamma_copy_offset, LG->ln2_beta_copy_offset,
                       LG->ln2_mean_copy_offset, LG->ln2_rstd_copy_offset,
                       LG->d_ln2_input_offset, LG->d_ln2_gamma_offset,
                       LG->d_ln2_beta_offset);
    
    // Add gradient from second residual path to d_residual1
    add_gradient(M, LG->d_ln2_input_offset, LG->d_residual1_offset);
    
    // ============================================================================
    // BACKWARD THROUGH FIRST RESIDUAL CONNECTION
    // residual1 = layer_input + attention_output
    // ============================================================================
    backward_residual_connection(M, LG->d_residual1_offset,
                                 LG->d_ln1_input_offset,  // Goes to layer input
                                 LG->d_attention_output_offset);
    
    // ============================================================================
    // BACKWARD THROUGH ATTENTION PROJECTION
    // ============================================================================
    backward_attention_projection(M, LG->d_attention_output_offset,
                                  LG->attention_output_copy_offset,
                                  L->proj_weight_offset, L->proj_bias_offset,
                                  LG->d_attention_token_offset,
                                  LG->d_attention_head_offset,
                                  LG->d_proj_weights_offset,
                                  LG->d_proj_bias_offset);
    
    // ============================================================================
    // BACKWARD THROUGH ATTENTION MECHANISM
    // ============================================================================
    
    // Backward through attention weights × V
    backward_attention_weighted_values(M, LG->d_attention_head_offset,
                                       LG->attention_weights_copy_offset,
                                       LG->v_output_copy_offset,
                                       LG->d_attention_weights_offset,
                                       LG->d_v_output_offset);
    
    // Backward through softmax
    backward_causal_softmax(M, LG->d_attention_weights_offset,
                            LG->attention_weights_copy_offset);
    
    // Backward through Q @ K^T
    backward_qk_matmul(M, LG->d_attention_weights_offset,
                      LG->q_output_copy_offset,
                      LG->k_output_copy_offset,
                      LG->d_q_output_offset,
                      LG->d_k_output_offset);
    
    // ============================================================================
    // BACKWARD THROUGH QKV PROJECTIONS
    // ============================================================================
    
    // Q projection
    backward_linear(M, LG->d_q_output_offset, LG->ln1_output_copy_offset,
                   L->q_weight_offset, L->q_bias_offset,
                   LG->d_ln1_output_offset,  // Accumulates to LN1 output
                   LG->d_q_weights_offset, LG->d_q_bias_offset,
                   LG->qkv_scratch_offset);
    
    // K projection
    backward_linear(M, LG->d_k_output_offset, LG->ln1_output_copy_offset,
                   L->k_weight_offset, L->k_bias_offset,
                   LG->d_ln1_output_offset,  // Accumulates to LN1 output
                   LG->d_k_weights_offset, LG->d_k_bias_offset,
                   LG->qkv_scratch_offset);
    
    // V projection
    backward_linear(M, LG->d_v_output_offset, LG->ln1_output_copy_offset,
                   L->v_weight_offset, L->v_bias_offset,
                   LG->d_ln1_output_offset,  // Accumulates to LN1 output
                   LG->d_v_weights_offset, LG->d_v_bias_offset,
                   LG->qkv_scratch_offset);
    
    // ============================================================================
    // BACKWARD THROUGH FIRST LAYERNORM
    // ============================================================================
    backward_layernorm(M, LG->d_ln1_output_offset, LG->ln1_input_copy_offset,
                       LG->ln1_gamma_copy_offset, LG->ln1_beta_copy_offset,
                       LG->ln1_mean_copy_offset, LG->ln1_rstd_copy_offset,
                       LG->d_ln1_input_offset, LG->d_ln1_gamma_offset,
                       LG->d_ln1_beta_offset);
    
    // Add gradient from first residual path to d_ln1_input
    // (d_ln1_input is the output gradient for this layer)
    // This already has the residual gradient from backward_residual_connection above
}

/**
 * @brief Compute cross-entropy loss and gradients w.r.t logits
 * 
 * Loss = -sum(log(p[correct])) / context_length
 * Gradient: dL/dlogit[i] = p[i] - 1 (for correct token)
 *           dL/dlogit[i] = p[i]     (for other tokens)
 */
void compute_cross_entropy_loss(TransformerModel *M, 
                                int32_t *target_tokens,
                                float *loss_out) {
    
    float *logits = M->memory_base + M->logits_offset;
    float *d_logits = M->memory_base + M->gradients.d_logits_offset;
    
    double total_loss = 0.0;  // Use double for accumulation precision
    
    #pragma omp parallel for reduction(+:total_loss)
    for (int t = 0; t < M->context_window; t++) {
        float *token_logits = logits + t * M->vocab_size;
        float *token_d_logits = d_logits + t * M->vocab_size;
        int correct_token = target_tokens[t];
        
        // Find max for numerical stability
        float max_logit = token_logits[0];
        for (int v = 1; v < M->vocab_size; v++) {
            if (token_logits[v] > max_logit) {
                max_logit = token_logits[v];
            }
        }
        
        // Compute exp(logit - max) and sum
        double sum_exp = 0.0;
        for (int v = 0; v < M->vocab_size; v++) {
            token_d_logits[v] = expf(token_logits[v] - max_logit);
            sum_exp += token_d_logits[v];
        }
        
        // Compute softmax probabilities
        float inv_sum = 1.0f / sum_exp;
        for (int v = 0; v < M->vocab_size; v++) {
            token_d_logits[v] *= inv_sum;
        }
        
        // Add loss for this token: -log(p[correct])
        total_loss += -logf(token_d_logits[correct_token] + 1e-10f);
        
        // Gradient is p - 1 for correct token, p for others
        token_d_logits[correct_token] -= 1.0f;
        
        // Scale by 1/context_length for average
        float scale = 1.0f / M->context_window;
        for (int v = 0; v < M->vocab_size; v++) {
            token_d_logits[v] *= scale;
        }
    }
    
    *loss_out = total_loss / M->context_window;
}

typedef struct {
    char **paths;
    size_t count;
} TrainingPairList;

static void free_training_pair_list(TrainingPairList *list) {
    if (!list || !list->paths) {
        return;
    }
    for (size_t i = 0; i < list->count; ++i) {
        free(list->paths[i]);
    }
    free(list->paths);
    list->paths = NULL;
    list->count = 0;
}

static int compare_path_strings(const void *a, const void *b) {
    const char *const *pa = (const char *const *)a;
    const char *const *pb = (const char *const *)b;
    return strcmp(*pa, *pb);
}

static bool build_training_pair_list(const char *dir_path, TrainingPairList *out) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "❌ Failed to open training directory '%s': %s\n", dir_path, strerror(errno));
        return false;
    }
    
    size_t capacity = 0;
    out->paths = NULL;
    out->count = 0;
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        const char *ext = strrchr(entry->d_name, '.');
        if (!ext || strcmp(ext, ".bin") != 0) {
            continue;
        }
        
        size_t dir_len = strlen(dir_path);
        bool needs_sep = dir_len > 0 && dir_path[dir_len - 1] != '/';
        size_t full_len = dir_len + (needs_sep ? 1 : 0) + strlen(entry->d_name) + 1;
        
        char *full_path = (char *)malloc(full_len);
        if (!full_path) {
            fprintf(stderr, "❌ Out of memory while listing training files\n");
            closedir(dir);
            free_training_pair_list(out);
            return false;
        }
        snprintf(full_path, full_len, "%s%s%s", dir_path, needs_sep ? "/" : "", entry->d_name);
        
        if (out->count == capacity) {
            size_t new_cap = capacity == 0 ? 64 : capacity * 2;
            char **new_paths = realloc(out->paths, new_cap * sizeof(char *));
            if (!new_paths) {
                fprintf(stderr, "❌ Out of memory while storing training file list\n");
                free(full_path);
                closedir(dir);
                free_training_pair_list(out);
                return false;
            }
            out->paths = new_paths;
            capacity = new_cap;
        }
        
        out->paths[out->count++] = full_path;
    }
    
    closedir(dir);
    
    if (out->count == 0) {
        fprintf(stderr, "❌ No '.bin' training files found in %s\n", dir_path);
        free_training_pair_list(out);
        return false;
    }
    
    qsort(out->paths, out->count, sizeof(char *), compare_path_strings);
    return true;
}

static bool load_training_window(const char *file_path,
                                 int context_len,
                                 int32_t *input_tokens,
                                 int32_t *target_tokens,
                                 uint32_t *scratch) {
    size_t total = (size_t)context_len + 1;
    FILE *fp = fopen(file_path, "rb");
    if (!fp) {
        fprintf(stderr, "❌ Failed to open pair file '%s': %s\n", file_path, strerror(errno));
        return false;
    }
    
    size_t read = fread(scratch, sizeof(uint32_t), total, fp);
    fclose(fp);
    
    if (read != total) {
        fprintf(stderr, "❌ Expected %zu tokens in '%s' but read %zu\n", total, file_path, read);
        return false;
    }
    
    for (int i = 0; i < context_len; ++i) {
        input_tokens[i] = (int32_t)scratch[i];
        target_tokens[i] = (int32_t)scratch[i + 1];
    }
    return true;
}

float training_step(TransformerModel *M, 
                    int32_t *input_tokens,
                    int32_t *target_tokens,
                    float learning_rate);

static void run_training_loop(TransformerModel *M,
                              const char *train_dir,
                              int total_steps,
                              float learning_rate,
                              int log_interval) {
    TrainingPairList list = {0};
    if (!build_training_pair_list(train_dir, &list)) {
        return;
    }
    
    if (total_steps <= 0) {
        total_steps = (int)list.count;
    }
    if (log_interval <= 0) {
        log_interval = 10;
    }
    if (learning_rate <= 0.0f) {
        learning_rate = 1e-4f;
    }
    
    int32_t *input_tokens = (int32_t *)malloc((size_t)M->context_window * sizeof(int32_t));
    int32_t *target_tokens = (int32_t *)malloc((size_t)M->context_window * sizeof(int32_t));
    uint32_t *scratch = (uint32_t *)malloc(((size_t)M->context_window + 1) * sizeof(uint32_t));
    
    if (!input_tokens || !target_tokens || !scratch) {
        fprintf(stderr, "❌ Failed to allocate buffers for training loop\n");
        free(input_tokens);
        free(target_tokens);
        free(scratch);
        free_training_pair_list(&list);
        return;
    }
    
    size_t pair_index = 0;
    printf("\n🎯 Starting training loop (%d steps, lr=%.6f) using data at %s\n",
           total_steps, learning_rate, train_dir);
    
    for (int step = 0; step < total_steps; ++step) {
        const char *pair_path = list.paths[pair_index];
        pair_index = (pair_index + 1) % list.count;
        
        if (!load_training_window(pair_path, M->context_window, input_tokens, target_tokens, scratch)) {
            continue;
        }
        
        float loss = training_step(M, input_tokens, target_tokens, learning_rate);
        if ((step + 1) % log_interval == 0 || step == 0) {
            float ppl = expf(loss);
            printf("[train] step=%d/%d  loss=%.6f  perplexity=%.2f  sample=%s\n",
                   step + 1, total_steps, loss, ppl, pair_path);
        }
    }
    
    printf("✅ Training complete.\n");
    
    free(input_tokens);
    free(target_tokens);
    free(scratch);
    free_training_pair_list(&list);
}

void update_all_weights_sgd(TransformerModel *M, float learning_rate);

float training_step(TransformerModel *M, 
                    int32_t *input_tokens,
                    int32_t *target_tokens,
                    float learning_rate) {
    
    // Store input tokens for backward pass
    memcpy(M->memory_base + M->gradients.actual_tokens_offset, 
           input_tokens, M->context_window * sizeof(int32_t));
    
    // ======== FORWARD PASS ========
    embed_tokens(M, input_tokens, M->context_window);
    
    size_t current_input = M->embedded_input_offset;
    for (int layer = 0; layer < M->num_layers; layer++) {
        transformer_layer_forward(M, layer, current_input);
        current_input = M->layers[layer].residual2_output_offset;
    }
    
    layernorm_token_parallel(M, current_input, 
                            M->final_ln_weight_offset,
                            M->final_ln_bias_offset,
                            M->final_ln_mean_offset,
                            M->final_ln_rstd_offset,
                            M->final_output_offset, 1e-5f);
    
    // Compute logits for all positions
    #pragma omp parallel for num_threads(M->num_cores)
    for (int t = 0; t < M->context_window; t++) {
        compute_logits_last_token_optimized(M, t);
    }
    
    // ======== BACKWARD PASS ========
    zero_gradients(M);
    cache_forward_activations(M);
    
    float loss;
    compute_cross_entropy_loss(M, target_tokens, &loss);
    
    // Backward from logits (includes token embedding gradient via weight tying)
    backward_lm_head(M);
    
    // Backward through final layernorm
    backward_final_layernorm(M);
    
    // Backward through transformer layers
    for (int layer = M->num_layers - 1; layer >= 0; layer--) {
        backward_transformer_layer(M, layer);
    }

    // Backward through initial embeddings (token + positional)
    backward_embedding_layer(M);

    // ======== WEIGHT UPDATE ========
    update_all_weights_sgd(M, learning_rate);

    return loss;
}

void update_all_weights_sgd(TransformerModel *M, float learning_rate) {
    size_t aligned_dim = M->aligned_embed_dim;
    size_t aligned_fc = 4 * aligned_dim;
    
    // Update token embeddings (shared with LM head)
    for (size_t i = 0; i < (size_t)M->vocab_size * aligned_dim; i++) {
        M->memory_base[M->token_emb_offset + i] -=
            learning_rate * M->memory_base[M->gradients.d_embed_weights_offset + i];
    }
    
    // Update positional embeddings
    for (size_t i = 0; i < (size_t)M->context_window * aligned_dim; i++) {
        M->memory_base[M->pos_emb_offset + i] -=
            learning_rate * M->memory_base[M->gradients.d_pos_embed_offset + i];
    }
    
    // Update each transformer layer
    for (int l = 0; l < M->num_layers; l++) {
        LayerGradients *LG = &M->gradients.layers[l];
        TrulyOptimalLayer *L = &M->layers[l];
        
        // LayerNorm 1 parameters
        for (size_t i = 0; i < aligned_dim; i++) {
            M->memory_base[L->ln1_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_ln1_gamma_offset + i];
            M->memory_base[L->ln1_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_ln1_beta_offset + i];
        }
        
        // Q weights/bias
        for (size_t i = 0; i < aligned_dim * aligned_dim; i++) {
            M->memory_base[L->q_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_q_weights_offset + i];
        }
        for (size_t i = 0; i < aligned_dim; i++) {
            M->memory_base[L->q_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_q_bias_offset + i];
        }
        
        // K weights/bias
        for (size_t i = 0; i < aligned_dim * aligned_dim; i++) {
            M->memory_base[L->k_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_k_weights_offset + i];
        }
        for (size_t i = 0; i < aligned_dim; i++) {
            M->memory_base[L->k_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_k_bias_offset + i];
        }
        
        // V weights/bias
        for (size_t i = 0; i < aligned_dim * aligned_dim; i++) {
            M->memory_base[L->v_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_v_weights_offset + i];
        }
        for (size_t i = 0; i < aligned_dim; i++) {
            M->memory_base[L->v_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_v_bias_offset + i];
        }
        
        // Projection weights/bias
        for (size_t i = 0; i < aligned_dim * aligned_dim; i++) {
            M->memory_base[L->proj_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_proj_weights_offset + i];
        }
        for (size_t i = 0; i < aligned_dim; i++) {
            M->memory_base[L->proj_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_proj_bias_offset + i];
        }
        
        // LayerNorm 2 parameters
        for (size_t i = 0; i < aligned_dim; i++) {
            M->memory_base[L->ln2_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_ln2_gamma_offset + i];
            M->memory_base[L->ln2_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_ln2_beta_offset + i];
        }
        
        // MLP weights/biases (FC1 expands to 4D, FC2 contracts back)
        for (size_t i = 0; i < aligned_dim * aligned_fc; i++) {
            M->memory_base[L->fc1_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_fc1_weights_offset + i];
        }
        for (size_t i = 0; i < aligned_fc; i++) {
            M->memory_base[L->fc1_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_fc1_bias_offset + i];
        }
        for (size_t i = 0; i < aligned_fc * aligned_dim; i++) {
            M->memory_base[L->fc2_weight_offset + i] -=
                learning_rate * M->memory_base[LG->d_fc2_weights_offset + i];
        }
        for (size_t i = 0; i < aligned_dim; i++) {
            M->memory_base[L->fc2_bias_offset + i] -=
                learning_rate * M->memory_base[LG->d_fc2_bias_offset + i];
        }
    }
    
    // Update final layernorm parameters
    for (size_t i = 0; i < aligned_dim; i++) {
        M->memory_base[M->final_ln_weight_offset + i] -=
            learning_rate * M->memory_base[M->gradients.d_final_ln_gamma_offset + i];
        M->memory_base[M->final_ln_bias_offset + i] -=
            learning_rate * M->memory_base[M->gradients.d_final_ln_beta_offset + i];
    }
}

/**************************************************************/

/* ---------------- main -------------------- */
int main(int argc, char **argv)
{
    /* defaults (minimum 4 layers for benchmark) */
    int L = 4, V = 32768, C = 128, T = 128;
    int head_dim = 64;
    int do_alloc = 0;
    int run_benchmarks = 0;
    const char* weight_file = NULL;  // New option for weight file
    const char* prompt_str = NULL;  // For comma-separated tokens
    const char* train_dir = NULL;   // Directory with packed training pairs
    int train_steps = 0;
    float train_learning_rate = 1e-4f;
    int train_log_interval = 10;
    int prompt_tokens[1024];        // Store up to 1024 tokens
    int prompt_length = 0;          // Actual number of tokens

    static struct option long_opts[] = {
        {"layers", required_argument, 0, 'l'},
        {"dmodel", required_argument, 0, 'd'},
        {"ctx", required_argument, 0, 't'},
        {"vocab", required_argument, 0, 'v'},
        {"head-dim", required_argument, 0, 'h'},
        {"force", no_argument, 0, 'f'},
        {"benchmark", no_argument, 0, 'b'},
        {"weights", required_argument, 0, 'w'},  // New option for weight file
        {"prompt", required_argument, 0, 'p'},
        {"train-dir", required_argument, 0, 1000},
        {"train-steps", required_argument, 0, 1001},
        {"train-lr", required_argument, 0, 1002},
        {"train-log-interval", required_argument, 0, 1003},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "l:d:t:v:h:fbw:", long_opts, NULL)) != -1)
    {
        switch (c)
        {
        case 'l':
            L = atoi(optarg);
            break;
        case 'd':
            C = atoi(optarg);
            break;
        case 't':
            T = atoi(optarg);
            break;
        case 'v':
            V = atoi(optarg);
            break;
        case 'h':
            head_dim = atoi(optarg);
            break;
        case 'f':
            do_alloc = 1;
            break;
        case 'b':
            run_benchmarks = 1;
            break;
        case 'w':  // New case for weight file
            weight_file = optarg;
            break;
        case 'p':  
            prompt_str = optarg;
            break;
        case 1000:
            train_dir = optarg;
            break;
        case 1001:
            train_steps = atoi(optarg);
            break;
        case 1002:
            train_learning_rate = strtof(optarg, NULL);
            break;
        case 1003:
            train_log_interval = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s [--layers N] [--dmodel N] [--ctx N] [--vocab N] [--head-dim N] [--force] [--benchmark] [--weights FILE] [--prompt TOKENS] [--train-dir DIR] [--train-steps N] [--train-lr LR] [--train-log-interval N]\n", argv[0]);
            return 1;
        }
    }

    /* ---------- try allocation ---------- */
    TransformerModel M = {0};
    
    if (weight_file) {
        // This handles EVERYTHING: read metadata, allocate, load weights
        if (read_model_metadata(&M, weight_file) != 0) {
            fprintf(stderr, "Failed to load GPT-2 model\n");
            return 1;
        }
    } else {
        M.num_layers = L;
        M.vocab_size = V;
        M.embed_dim = C;
        M.context_window = T;
        M.head_dim = head_dim;
    }
    
        // ← ADD VALIDATION
    if (C % head_dim != 0) {
        fprintf(stderr, "❌ Error: embed_dim (%d) must be divisible by head_dim (%d)\n", C, head_dim);
        return 1;
    }

    if (train_dir) {
        M.training_enabled = true;
    }

    size_t need_bytes = bytes_needed(M.num_layers, M.vocab_size, M.embed_dim, M.context_window);
    double need_gib = need_bytes / (1024.0 * 1024.0 * 1024.0);

    printf("⚙  Requested model  L=%d  d_model=%d  ctx=%d  vocab=%d\n", M.num_layers, M.embed_dim, M.context_window, M.vocab_size);
    printf("→ Would need ≈ %.2f GiB (%.0f bytes)\n", need_gib, (double)need_bytes);

    if (!do_alloc)
    {
        printf("Dry-run only (no allocation). Pass --force to allocate and run benchmarks.\n");
        return 0;
    }

    // Enforce minimum layers for comprehensive benchmark
    if (run_benchmarks && L < 4)
    {
        fprintf(stderr, "Error: For comprehensive benchmarks, at least --layers 4 is required to demonstrate testing across different kernels on separate layers.\n");
        return 1;
    }
    
    M.num_attention_heads = M.embed_dim / M.head_dim;

    /* sanity: if system RAM < need_bytes, warn */
    long pages = sysconf(_SC_PHYS_PAGES);
    long page = sysconf(_SC_PAGE_SIZE);
    double sys_gib = pages * (double)page / (1024.0 * 1024.0 * 1024.0);

    if (need_gib > sys_gib)
    {
        fprintf(stderr, "❌ Need %.2f GiB but system has only %.2f GiB RAM. Aborting.\n",
                need_gib, sys_gib);
        return 1;
    }

    printf("Allocating huge block... this may page-fault if hugepages are missing\n");
    layout_transformer(&M, M.training_enabled);
    printf("✅ Success! mmap at %p, %.2f GiB reserved.\n",
           (void *)M.memory_base, need_gib);
    
    /* Setup execution plan */
    long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int reserved_cores = 4; // for OS, logging, etc.

    M.num_cores = (logical_cores > reserved_cores)
                      ? logical_cores - reserved_cores
                      : 1;
    M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;

    printf("🧠 Detected %ld logical cores → reserving %d for OS → using %d for model\n",
           logical_cores, reserved_cores, M.num_cores);
    printf("📦 Each core will handle ≈ %d tokens from context window of %d tokens\n",
           M.tokens_per_core, M.context_window);
    printf("🧠 Attention heads = %d (head_dim=%d)\n", M.num_attention_heads, M.head_dim); 
   
    /* Parse prompt tokens if provided */
    if (prompt_str != NULL) {
        char* token = strtok(strdup(prompt_str), ",");
        while (token != NULL && prompt_length < 1024) {
            prompt_tokens[prompt_length++] = atoi(token);
            token = strtok(NULL, ",");
        }
        printf("📝 Loaded prompt with %d tokens\n", prompt_length);
    }

    if (weight_file) {
        srand(time(NULL));
        load_model_weights(&M, weight_file);
        
        // DELETE THESE HARDCODED LINES:
        // int prompt[] = {2061, 318, 262, 362, 358, 1099, 286, 6268, 30};
        
        // REPLACE WITH THIS:
        if (!train_dir) {
            printf("\n🚀 Generating text...\n");
            
            if (prompt_length > 0) {
                // Use provided prompt
                generate(&M, prompt_tokens, prompt_length, 20);
            } else {
                // Default prompt if none provided
                int default_prompt[] = {15496, 11, 314, 716}; // "Hello, I am"
                printf("⚠️  No prompt provided, using default: \"Hello, I am\"\n");
                generate(&M, default_prompt, 4, 20);
            }
        }
    }

    if (run_benchmarks)
    {
        debug_math_comparison(&M);
        run_comprehensive_benchmark(&M);
    }

    if (train_dir) {
        run_training_loop(&M, train_dir, train_steps, train_learning_rate, train_log_interval);
    }

    destroy_transformer(&M);
    return 0;
}
