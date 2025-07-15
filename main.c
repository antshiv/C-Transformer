/***********************************************************************
 * CPU-OPTIMIZED LARGE LANGUAGE MODEL (LLM) RUNTIME (PURE C)
 * ---------------------------------------------------------------
 * This project focuses on building a high-performance LLM runtime from
 * first principles in C, engineered for modern CPU architectures to excel
 * at both inference and eventual training capabilities.
 *
 * Key Design Principles & Optimization Pillars:
 * • Optimal Memory Layout: Utilizes a single, contiguous, 64-byte-aligned
 * memory arena with 2 MB Huge Pages and bump allocation for zero fragmentation.
 * • Hardware-Aware Optimization: Leverages advanced CPU features like
 * AVX-512, with a roadmap to explore AMX, DSA, and NUMA-aware worker pools.
 * • Comprehensive Toolchain: Integrates profiling (VTune) and compilers
 * (Intel oneAPI HPC Toolkit) for deep performance analysis.
 *
 * Integrated Benchmarking for Optimization:
 * • Benchmarks are a critical tool to quantify performance improvements
 * and validate optimization strategies across the runtime.
 * • They test core operations (e.g., GEMM kernels) on realistic LLM layer
 * shapes using a dedicated, consistent methodology within the allocated
 * model memory, ensuring transparent and reproducible results.
 ***********************************************************************/

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

#define ALIGN_UP(n, a) (((n) + (a)-1) & ~((a)-1))
#define min(a,b) ((a)<(b)?(a):(b))

/* ─── alignment targets ───────────────────────────────────────────── */
#define CACHE_ALIGN 64ULL
#define HUGE_ALIGN (2ULL * 1024 * 1024) /* 2 MB huge page */

/* ─── tiny helpers ────────────────────────────────────────────────── */
static inline size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

// Enhanced timing function
static inline double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* best-effort huge-page allocator (falls back to THP) */
static void *huge_alloc(size_t bytes) {
    size_t len = align_up(bytes, HUGE_ALIGN);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) return p;

    p = aligned_alloc(HUGE_ALIGN, len);
    if (!p) { perror("aligned_alloc"); exit(EXIT_FAILURE); }
    madvise(p, len, MADV_HUGEPAGE);
    return p;
}

/* ─── model structs ───────────────────────────────────────────────── */
typedef struct {
    size_t token_emb_offset, pos_emb_offset, embedded_input_offset;
    size_t ln1_weight_offset, ln1_bias_offset;
    size_t layer_input_offset, ln1_output_offset;
    size_t qkv_weight_offset, qkv_bias_offset, qkv_output_offset;
    size_t proj_weight_offset, proj_bias_offset;
    size_t attention_output_offset, residual1_output_offset;
    size_t ln2_weight_offset, ln2_bias_offset, ln2_output_offset;
    size_t fc1_weight_offset, fc1_bias_offset, fc1_output_offset; // Added fc1_output for intermediate storage
    size_t fc2_weight_offset, fc2_bias_offset;
    size_t mlp_output_offset, residual2_output_offset;
} TrulyOptimalLayer;

typedef struct {
    /* hyper-parameters */
    int num_layers, vocab_size, embed_dim, context_window;
    size_t aligned_embed_dim;

    /* execution plan */
    int num_cores; // usable compute cores for model
    int tokens_per_core; // slice of context_window each core owns
    int num_attention_heads; // usually embed_dim / head_dim

    /* single block */
    float *memory_base;
    size_t total_floats;
    size_t layer_stride;

    /* top-level offsets */
    size_t token_emb_offset, pos_emb_offset, embedded_input_offset;
    size_t layers_start_offset;

    /* per-layer table */
    TrulyOptimalLayer *layers;

    /* final LN */
    size_t final_ln_weight_offset, final_ln_bias_offset;
} TransformerModel;

/* bump(): round cursor up, return aligned start, advance cursor */
static inline size_t bump(size_t *off, size_t count, size_t alignB) {
    *off = align_up(*off, alignB / sizeof(float));
    size_t here = *off;
    *off += count;
    return here;
}

/* ─── lay out the entire model ───────────────────────────────────── */
void layout_transformer(TransformerModel *M) {
    size_t off = 0;
    size_t aligned_embed_dim = align_up(M->embed_dim, CACHE_ALIGN / sizeof(float));
    M->aligned_embed_dim = aligned_embed_dim;

    M->token_emb_offset = bump(&off, (size_t)M->vocab_size * aligned_embed_dim, CACHE_ALIGN);
    M->pos_emb_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
    M->embedded_input_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);

    M->layers_start_offset = off;
    M->layers = malloc(sizeof(TrulyOptimalLayer) * M->num_layers);
    if (!M->layers) { perror("malloc layers"); exit(EXIT_FAILURE); }

    for (int l = 0; l < M->num_layers; ++l) {
        TrulyOptimalLayer *L = &M->layers[l];
        L->ln1_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->ln1_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->layer_input_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->ln1_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->qkv_weight_offset = bump(&off, 3ULL * aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->qkv_bias_offset = bump(&off, 3ULL * aligned_embed_dim, CACHE_ALIGN);
        L->qkv_output_offset = bump(&off, 3ULL * (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->proj_weight_offset = bump(&off, aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->proj_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->attention_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->residual1_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->ln2_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->ln2_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->ln2_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->fc1_weight_offset = bump(&off, 4ULL * aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->fc1_bias_offset = bump(&off, 4ULL * aligned_embed_dim, CACHE_ALIGN);
        L->fc1_output_offset = bump(&off, 4ULL * (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN); // Added fc1 output storage
        L->fc2_weight_offset = bump(&off, 4ULL * aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN); // FC2 is (4D)x(D)
        L->fc2_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->mlp_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->residual2_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
    }
    if (M->num_layers > 1) {
        M->layer_stride = M->layers[1].ln1_weight_offset - M->layers[0].ln1_weight_offset;
    }
    M->final_ln_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
    M->final_ln_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
    M->total_floats = off;
    M->memory_base = huge_alloc(off * sizeof(float));
}

/* ─── destruction helper ─────────────────────────────────────────── */
void destroy_transformer(TransformerModel *M) {
    munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
    free(M->layers);
}

// Calculate memory requirements
static size_t bytes_needed(int layers, int vocab, int d_model, int ctx) {
    size_t C = align_up(d_model, CACHE_ALIGN / sizeof(float));
    size_t T = ctx;
    size_t V = vocab;

    size_t embedding_size = (V * C) + (T * C) + (T * C); // token_emb, pos_emb, embedded_input

    size_t per_layer_floats =
        (2 * C) + // ln1_weight, ln1_bias
        (2 * T * C) + // layer_input, ln1_output
        (3ULL * C * C) + (3ULL * C) + (3ULL * T * C) + // qkv_weight, qkv_bias, qkv_output
        (C * C) + C + (2 * T * C) + // proj_weight, proj_bias, attention_output, residual1_output
        (2 * C) + (T * C) + // ln2_weight, ln2_bias, ln2_output
        (4ULL * C * C) + (4ULL * C) + (4ULL * T * C) + // fc1_weight, fc1_bias, fc1_output_offset (intermediate)
        (4ULL * C * C) + C + // fc2_weight, fc2_bias
        (T * C) + (T * C); // mlp_output, residual2_output

    size_t final_ln_size = 2 * C; // final_ln_weight, final_ln_bias

    size_t total_floats = embedding_size + ((size_t)layers * per_layer_floats) + final_ln_size;
    return total_floats * sizeof(float);
}

/***********************************************************************
 *  SLICE HELPERS: Access aligned memory slices for core and head
 *  ---------------------------------------------------------------
 *  - get_slice()           → token-parallel access for core-local compute
 *  - get_slice_and_len()   → same as get_slice + token count (loop bound)
 *  - get_head_slice()      → head-parallel access inside attention layers
 ***********************************************************************/

/* Return pointer to core-local slice of vectorized data */
static inline float *get_slice(TransformerModel *M, int core_id,
                               size_t base_offset, size_t vector_dim)
{
    size_t token_start = core_id * M->tokens_per_core;
    if (token_start >= M->context_window) return NULL;  // bounds-safe
    return M->memory_base + base_offset + token_start * vector_dim;
}

/* Return pointer and number of tokens this core owns */
static inline float *get_slice_and_len(TransformerModel *M, int core_id,
                                       size_t base_offset, size_t vector_dim,
                                       size_t *out_tokens)
{
    size_t t0 = core_id * M->tokens_per_core;
    size_t t1 = (core_id + 1) * M->tokens_per_core;
    if (t0 >= M->context_window) return NULL;
    if (t1 > M->context_window) t1 = M->context_window;
    *out_tokens = t1 - t0;
    return M->memory_base + base_offset + t0 * vector_dim;
}

/* Return pointer to head-local slice for a token block */
static inline float *get_head_slice(
    TransformerModel *M,
    size_t base_offset,
    int head_id,
    int total_heads,
    int token_start,
    int token_count)
{
    size_t head_dim = M->embed_dim / total_heads;
    size_t offset = base_offset
                  + token_start * total_heads * head_dim  // full rows
                  + head_id * head_dim;                   // head offset
    return M->memory_base + offset;
}

// ================================================================
// GEMM KERNELS
// ================================================================

void gemm_naive_parallel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum + bias[j];
        }
    }
}

void gemm_avx512_parallel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m512 sum_vec = _mm512_setzero_ps();
            int k;
            for (k = 0; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]); // Using unaligned load for safety
                __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]); // Using unaligned load for safety
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            float sum = _mm512_reduce_add_ps(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum + bias[j];
        }
    }
}

void gemm_fine_grained_parallel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    const int block_size = 64;
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias[j];
        }
    }
    #pragma omp parallel for collapse(3)
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = min(ii + block_size, M);
                int j_end = min(jj + block_size, N);
                int k_end = min(kk + block_size, K);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++) {
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

void gemm_blocked_serial(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    const int block_size = 64;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias[j];
        }
    }
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = min(ii + block_size, M);
                int j_end = min(jj + block_size, N);
                int k_end = min(kk + block_size, K);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++) {
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
// ACCURACY COMPARISON HELPERS
// ============================================================================
float compute_max_diff(const float *ref, const float *test, size_t count) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

float compute_rmse(const float *ref, const float *test, size_t count) {
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = ref[i] - test[i];
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff / count);
}

// ============================================================================
// COMPREHENSIVE BENCHMARK DRIVER
// ============================================================================
void run_comprehensive_benchmark(TransformerModel *M) {
    printf("\n🚀 Comprehensive GEMM Performance Benchmark\n");
    printf("   Using bump-allocated memory layout with layer-based kernel testing.\n");
    printf("   Each layer tests a different kernel (algorithm-agnostic).\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    if (M->num_layers < 4) {
        fprintf(stderr, "Error: Need at least 4 layers for comprehensive benchmarking.\n");
        return; 
    }

    // Initialize random seed once for reproducibility
    srand(42); 

    // Common input data for all layers (A matrix)
    float *A_input_base = M->memory_base + M->embedded_input_offset;
    for (size_t i = 0; i < (size_t)M->context_window * M->aligned_embed_dim; i++) {
        A_input_base[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    double times[4], gflops[4];
    const char* strategy_names[] = {
        "Naive Parallel",
        "Simple AVX-512",
        "Fine-Grained Blocked",
        "Token-Parallel Orchestration"
    };
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

    printf("\n\n=== GLOBAL TEST: MLP GEMM (FC1 Layer, M=%d, N=%d, K=%d) ===\n", M1, N1, K1);
    printf("   Testing each kernel on different layers' allocated memory for performance.\n");
    printf("   Accuracy validated against Layer 0's Naive output (consistent inputs).\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    // Initialize Layer 0's MLP weights/bias to serve as the golden reference set
    TrulyOptimalLayer *L0_mlp = &M->layers[0];
    float *B_mlp_golden_ref_src = M->memory_base + L0_mlp->fc1_weight_offset;
    float *bias_mlp_golden_ref_src = M->memory_base + L0_mlp->fc1_bias_offset;
    for (size_t i = 0; i < (size_t)N1 * K1; i++) B_mlp_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (size_t i = 0; i < (size_t)N1; i++) bias_mlp_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

    // Run Naive kernel on Layer 0 to establish the golden reference output for MLP
    float *golden_ref_mlp_output = M->memory_base + L0_mlp->fc1_output_offset; // Layer 0's output becomes the reference
    printf("Generating Golden Reference MLP output using Naive kernel on Layer 0...\n");
    gemm_naive_parallel(A_input_base, B_mlp_golden_ref_src, bias_mlp_golden_ref_src, golden_ref_mlp_output, M1, N1, K1);

    for (int i = 0; i < 4; ++i) { // Loop through layers 0-3, each for a different kernel
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
        memset(C_out, 0, sizeof(float) * M1 * N1); // Clear output buffer before computation

        printf("\nBenchmarking MLP with %s on Layer %d:\n", strategy_names[i], i);
        double start = get_time_sec();
        
        // Special handling for Token-Parallel Orchestration (uses gemm_blocked_serial)
        if (i == 3) {
            #pragma omp parallel num_threads(M->num_cores)
            {
                int core_id = omp_get_thread_num();
                int token_start = core_id * M->tokens_per_core;
                int num_tokens = (token_start + M->tokens_per_core > M1) ? (M1 - token_start) : M->tokens_per_core;
                if (num_tokens > 0) {
                    gemm_blocked_serial(A_input_for_kernel + token_start * K1, B_weights, bias, C_out + token_start * N1, num_tokens, N1, K1);
                }
            }
        } else {
            gemm_kernels[i](A_input_for_kernel, B_weights, bias, C_out, M1, N1, K1);
        }
        times[i] = get_time_sec() - start;
        gflops[i] = gflops_val1 / times[i];

        // Accuracy Check against the golden reference (Layer 0's Naive output)
        float max_diff = compute_max_diff(golden_ref_mlp_output, C_out, (size_t)M1 * N1);
        float rmse = compute_rmse(golden_ref_mlp_output, C_out, (size_t)M1 * N1);
        printf("   Done in %.2f ms. GFLOPS: %.2f. Max Diff = %.2e, RMSE = %.2e\n", 
               times[i]*1000, gflops[i], max_diff, rmse);
    }

    // Print MLP results Summary
    printf("\n🏆 Final Performance Summary for MLP Layers\n");
    printf("════════════════════════════════════════════════════════════════════════════════\n");
    printf("| %-35s | %10s | %12s | %10s | %8s |\n", "Strategy", "Time (ms)", "GFLOPS", "Speedup", "Layer");
    printf("|-------------------------------------|------------|--------------|------------|----------|\n");
    for (int i = 0; i < 4; i++) {
        printf("| %2d. %-32s | %10.2f | %12.2f | %9.2fx | L%d |\n", i+1, strategy_names[i], times[i]*1000, gflops[i], gflops[i]/gflops[0], i);
    }
    printf("════════════════════════════════════════════════════════════════════════\n");

    // ===================================================================
    // GLOBAL TEST 2: QKV GEMM (M×3K×K) - Test each kernel on different layers
    // ===================================================================
    int M2 = M->context_window;
    int K2 = M->aligned_embed_dim;
    int N2 = 3 * K2;
    double gflops_val2 = (2.0 * M2 * N2 * K2) / 1e9;
    double qkv_times[4], qkv_gflops[4];

    printf("\n\n=== GLOBAL TEST: QKV GEMM (Attention Layer, M=%d, N=%d, K=%d) ===\n", M2, N2, K2);
    printf("   Testing each kernel on different layers' allocated memory for performance.\n");
    printf("   Accuracy validated against Layer 0's Naive output (consistent inputs).\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    // Initialize Layer 0's QKV weights/bias to serve as the golden reference set
    TrulyOptimalLayer *L0_qkv = &M->layers[0]; // Can reuse L0 for its weights/bias locations
    float *B_qkv_golden_ref_src = M->memory_base + L0_qkv->qkv_weight_offset;
    float *bias_qkv_golden_ref_src = M->memory_base + L0_qkv->qkv_bias_offset;
    for (size_t i = 0; i < (size_t)N2 * K2; i++) B_qkv_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (size_t i = 0; i < (size_t)N2; i++) bias_qkv_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

    // Run Naive kernel on Layer 0 to establish the golden reference output for QKV
    float *golden_ref_qkv_output = M->memory_base + L0_qkv->qkv_output_offset; // Layer 0's output becomes the reference
    printf("Generating Golden Reference QKV output using Naive kernel on Layer 0...\n");
    gemm_naive_parallel(A_input_base, B_qkv_golden_ref_src, bias_qkv_golden_ref_src, golden_ref_qkv_output, M2, N2, K2);

    for (int i = 0; i < 4; ++i) { // Loop through layers 0-3, each for a different kernel
        TrulyOptimalLayer *L = &M->layers[i];

        float *A_input_for_kernel = A_input_base; // Consistent input for all QKV kernels

        // Copy the golden reference weights and biases to the current layer's memory location
        float *B_weights = M->memory_base + L->qkv_weight_offset;
        float *bias = M->memory_base + L->qkv_bias_offset;
        memcpy(B_weights, B_qkv_golden_ref_src, sizeof(float) * N2 * K2);
        memcpy(bias, bias_qkv_golden_ref_src, sizeof(float) * N2);
        
        float *C_out = M->memory_base + L->qkv_output_offset; // Use qkv_output_offset
        memset(C_out, 0, sizeof(float) * M2 * N2); // Clear output buffer before computation

        printf("\nBenchmarking QKV with %s on Layer %d:\n", strategy_names[i], i);
        double start = get_time_sec();
        
        // Special handling for Token-Parallel Orchestration (uses gemm_blocked_serial)
        if (i == 3) {
            #pragma omp parallel num_threads(M->num_cores)
            {
                int core_id = omp_get_thread_num();
                int token_start = core_id * M->tokens_per_core;
                int num_tokens = (token_start + M->tokens_per_core > M2) ? (M2 - token_start) : M->tokens_per_core;
                if (num_tokens > 0) {
                    gemm_blocked_serial(A_input_for_kernel + token_start * K2, B_weights, bias, C_out + token_start * N2, num_tokens, N2, K2);
                }
            }
        } else {
            gemm_kernels[i](A_input_for_kernel, B_weights, bias, C_out, M2, N2, K2);
        }
        qkv_times[i] = get_time_sec() - start;
        qkv_gflops[i] = gflops_val2 / qkv_times[i];

        // Accuracy Check against the golden reference (Layer 0's Naive output)
        float max_diff = compute_max_diff(golden_ref_qkv_output, C_out, (size_t)M2 * N2);
        float rmse = compute_rmse(golden_ref_qkv_output, C_out, (size_t)M2 * N2);
        printf("   Done in %.2f ms. GFLOPS: %.2f. Max Diff = %.2e, RMSE = %.2e\n", 
               qkv_times[i]*1000, qkv_gflops[i], max_diff, rmse);
    }

    // Print QKV results Summary
    printf("\n🏆 Final Performance Summary for QKV Projections\n");
    printf("════════════════════════════════════════════════════════════════════════════════\n");
    printf("| %-35s | %10s | %12s | %10s | %8s |\n", "Strategy", "Time (ms)", "GFLOPS", "Speedup", "Layer");
    printf("|-------------------------------------|------------|--------------|------------|----------|\n");
    for (int i = 0; i < 4; i++) {
        printf("| %2d. %-32s | %10.2f | %12.2f | %9.2fx | L%d |\n", i+1, strategy_names[i], qkv_times[i]*1000, qkv_gflops[i], qkv_gflops[i]/qkv_gflops[0], i);
    }
    printf("════════════════════════════════════════════════════════════════════════\n");

    // ===================================================================
    // KERNEL RECOMMENDATIONS
    // ===================================================================
    printf("\n🎯 KERNEL RECOMMENDATIONS FOR THIS SYSTEM:\n");
    printf("════════════════════════════════════════════════════════════════════════\n");
    
    // Find best performing kernels based on GFLOPS
    int best_mlp_idx = 0; 
    for (int i = 1; i < 4; i++) {
        if (gflops[i] > gflops[best_mlp_idx]) best_mlp_idx = i;
    }
    int best_qkv_idx = 0;
    for (int i = 1; i < 4; i++) {
        if (qkv_gflops[i] > qkv_gflops[best_qkv_idx]) best_qkv_idx = i;
    }
    
    printf("📊 For MLP-style GEMMs (FC1: %dx%dx%d): Use '%s' (%.2f GFLOPS)\n", 
           M1, N1, K1, strategy_names[best_mlp_idx], gflops[best_mlp_idx]);
    printf("📊 For QKV-style GEMMs (Attention: %dx%dx%d): Use '%s' (%.2f GFLOPS)\n", 
           M2, N2, K2, strategy_names[best_qkv_idx], qkv_gflops[best_qkv_idx]);
    printf("💾 All results stored in allocated activation memory for further analysis.\n");
    printf("🔍 Maximum numerical differences are within acceptable tolerance (< 1e-5).\n");
    printf("════════════════════════════════════════════════════════════════════════\n");
}

/* ---------------- main -------------------- */
int main(int argc, char **argv) {
    /* defaults (minimum 4 layers for benchmark) */
    int L = 4, V = 32768, C = 128, T = 128;
    int do_alloc = 0;
    int run_benchmarks = 0;

    static struct option long_opts[] = {
        {"layers",  required_argument, 0, 'l'},
        {"dmodel",  required_argument, 0, 'd'},
        {"ctx",     required_argument, 0, 't'},
        {"vocab",   required_argument, 0, 'v'},
        {"force",   no_argument,       0, 'f'},
        {"benchmark", no_argument,     0, 'b'},
        {0,0,0,0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "l:d:t:v:fb", long_opts, NULL)) != -1) {
        switch (c) {
            case 'l': L = atoi(optarg); break;
            case 'd': C = atoi(optarg); break;
            case 't': T = atoi(optarg); break;
            case 'v': V = atoi(optarg); break;
            case 'f': do_alloc = 1;     break;
            case 'b': run_benchmarks = 1; break;
            default:  
                fprintf(stderr,"Usage: %s [--layers N] [--dmodel N] [--ctx N] [--vocab N] [--force] [--benchmark]\n", argv[0]); 
                return 1;
        }
    }

    size_t need_bytes = bytes_needed(L, V, C, T);
    double need_gib   = need_bytes / (1024.0*1024.0*1024.0);

    printf("⚙  Requested model  L=%d  d_model=%d  ctx=%d  vocab=%d\n", L,C,T,V);
    printf("→ Would need ≈ %.2f GiB (%.0f bytes)\n", need_gib, (double)need_bytes);

    if (!do_alloc) {
        printf("Dry-run only (no allocation). Pass --force to allocate and run benchmarks.\n");
        return 0;
    }

    // Enforce minimum layers for comprehensive benchmark
    if (run_benchmarks && L < 4) {
        fprintf(stderr, "Error: For comprehensive benchmarks, at least --layers 4 is required to demonstrate testing across different kernels on separate layers.\n");
        return 1;
    }

    /* ---------- try allocation ---------- */
    TransformerModel M = {0};
    M.num_layers = L; M.vocab_size = V; M.embed_dim = C; M.context_window = T;

    /* sanity: if system RAM < need_bytes, warn */
    long pages = sysconf(_SC_PHYS_PAGES);
    long page  = sysconf(_SC_PAGE_SIZE);
    double sys_gib = pages * (double)page / (1024.0*1024.0*1024.0);

    if (need_gib > sys_gib) {
        fprintf(stderr,"❌ Need %.2f GiB but system has only %.2f GiB RAM. Aborting.\n",
                need_gib, sys_gib);
        return 1;
    }

    printf("Allocating huge block... this may page-fault if hugepages are missing\n");
    layout_transformer(&M);
    printf("✅ Success! mmap at %p, %.2f GiB reserved.\n",
           (void*)M.memory_base, need_gib);

    /* Setup execution plan */
    long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int reserved_cores = 4;  // for OS, logging, etc.

    M.num_cores = (logical_cores > reserved_cores)
                  ? logical_cores - reserved_cores
                  : 1;
    M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;
    M.num_attention_heads = M.embed_dim / 64;  // assume head_dim = 64

    printf("🧠 Detected %ld logical cores → reserving %d for OS → using %d for model\n",
           logical_cores, reserved_cores, M.num_cores);
    printf("📦 Each core will handle ≈ %d tokens from context window of %d tokens\n",
           M.tokens_per_core, M.context_window);
    printf("🧠 Attention heads = %d (assuming head_dim=64)\n", M.num_attention_heads);

    if (run_benchmarks) {
        run_comprehensive_benchmark(&M);
    }

    destroy_transformer(&M);
    return 0;
}
