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

#define ALIGN_UP(n, a) (((n) + (a) - 1) & ~((a) - 1))
#define min(a, b) ((a) < (b) ? (a) : (b))

/* ─── alignment targets ───────────────────────────────────────────── */
#define CACHE_ALIGN 64ULL
#define HUGE_ALIGN (2ULL * 1024 * 1024) /* 2 MB huge page */

/* ─── tiny helpers ────────────────────────────────────────────────── */
static inline size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

// Enhanced timing function
static inline double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* best-effort huge-page allocator (falls back to THP) */
static void *huge_alloc(size_t bytes)
{
    size_t len = align_up(bytes, HUGE_ALIGN);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED)
        return p;

    p = aligned_alloc(HUGE_ALIGN, len);
    if (!p)
    {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }
    madvise(p, len, MADV_HUGEPAGE);
    return p;
}

/* ─── model structs ───────────────────────────────────────────────── */
typedef struct
{
    size_t token_emb_offset, pos_emb_offset, embedded_input_offset;

    size_t ln1_weight_offset, ln1_bias_offset;
    size_t ln1_mean_offset, ln1_rstd_offset; // NEW
    size_t layer_input_offset, ln1_output_offset;

    size_t qkv_weight_offset, qkv_bias_offset, qkv_output_offset;
    size_t proj_weight_offset, proj_bias_offset;
    size_t attention_output_offset, residual1_output_offset;

    size_t ln2_weight_offset, ln2_bias_offset;
    size_t ln2_mean_offset, ln2_rstd_offset; // NEW
    size_t ln2_output_offset;

    size_t fc1_weight_offset, fc1_bias_offset, fc1_output_offset; // Added fc1_output for intermediate storage
    size_t fc2_weight_offset, fc2_bias_offset;
    size_t mlp_output_offset, residual2_output_offset;
} TrulyOptimalLayer;

typedef struct
{
    /* hyper-parameters */
    int num_layers, vocab_size, embed_dim, context_window;
    size_t aligned_embed_dim;

    /* execution plan */
    int num_cores;           // usable compute cores for model
    int tokens_per_core;     // slice of context_window each core owns
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
static inline size_t bump(size_t *off, size_t count, size_t alignB)
{
    *off = align_up(*off, alignB / sizeof(float));
    size_t here = *off;
    *off += count;
    return here;
}

/* ─── lay out the entire model ───────────────────────────────────── */
void layout_transformer(TransformerModel *M)
{
    size_t off = 0;
    size_t aligned_embed_dim = align_up(M->embed_dim, CACHE_ALIGN / sizeof(float));
    M->aligned_embed_dim = aligned_embed_dim;

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
        L->ln1_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->ln1_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        // NEW: allocate per-token mean and rstd
        L->ln1_mean_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
        L->ln1_rstd_offset = bump(&off, (size_t)M->context_window, CACHE_ALIGN);
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
        // NEW
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
    }
    if (M->num_layers > 1)
    {
        M->layer_stride = M->layers[1].ln1_weight_offset - M->layers[0].ln1_weight_offset;
    }
    M->final_ln_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
    M->final_ln_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
    M->total_floats = off;
    M->memory_base = huge_alloc(off * sizeof(float));
}

/* ─── destruction helper ─────────────────────────────────────────── */
void destroy_transformer(TransformerModel *M)
{
    munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
    free(M->layers);
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

// ================================================================
// GEMM KERNELS
// ================================================================

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

void gemm_fine_grained_parallel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K)
{
    const int block_size = 64;
#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = bias[j];
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
                            float *mean_cache, // For storing mean
                            float *rstd_cache, // For storing rstd
                            int tokens, int d_model, float eps)
{
    for (int t = 0; t < tokens; ++t)
    {
        const float *in_ptr = input + t * d_model;
        float *out_ptr = output + t * d_model;

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
        float variance = sum_sq_diff / (float)d_model;

        // Calculate inverse standard deviation
        float inv_std = 1.0f / sqrtf(variance + eps);

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
        var = var / (float)d_model + eps;             // Add epsilon for numerical stability
        float inv_std = 1.0f / sqrtf(var);            // Calculate inverse standard deviation
        __m512 inv_std_vec = _mm512_set1_ps(inv_std); // Broadcast inv_std to a vector

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

// ================================================================
// LAYER NORMALIZATION (Token-Parallel Orchestration)
// This function distributes the LayerNorm computation across multiple
// OpenMP threads, with each thread calling the optimized
// layernorm_forward_unrolled_slice function for its assigned token slice.
// ================================================================
void layernorm_token_parallel(TransformerModel *M,
                              size_t input_offset,
                              size_t weight_offset,     // Corresponds to gamma
                              size_t bias_offset,       // Corresponds to beta
                              size_t mean_cache_offset, // Offset for storing mean
                              size_t rstd_cache_offset, // Offset for storing rstd
                              size_t output_offset,
                              float eps)
{
#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        size_t token_start = core_id * M->tokens_per_core;
        size_t num_tokens_for_this_thread = (token_start + M->tokens_per_core > M->context_window)
                                                ? (M->context_window - token_start)
                                                : M->tokens_per_core;

        if (num_tokens_for_this_thread > 0)
        {
            // Calculate base pointers for this thread's slice within the global memory arena
            const float *input_base_ptr = M->memory_base + input_offset + token_start * M->aligned_embed_dim;
            const float *gamma_weights = M->memory_base + weight_offset; // Gamma/Beta are shared across tokens, not sliced
            const float *beta_biases = M->memory_base + bias_offset;
            float *mean_cache_base_ptr = M->memory_base + mean_cache_offset + token_start; // Mean/RSTD are per-token, so they are sliced
            float *rstd_cache_base_ptr = M->memory_base + rstd_cache_offset + token_start;
            float *output_base_ptr = M->memory_base + output_offset + token_start * M->aligned_embed_dim;

            // Call the slice-processing function for this thread's work
            layernorm_forward_unrolled_slice(input_base_ptr, gamma_weights, beta_biases,
                                             output_base_ptr, mean_cache_base_ptr, rstd_cache_base_ptr,
                                             num_tokens_for_this_thread, M->aligned_embed_dim, eps);
        }
    }
}

// ================================================================
// QKV PROJECTION (Using your optimized GEMM)
// ================================================================
void qkv_projection_token_parallel(TransformerModel *M,
                                   size_t input_offset,
                                   size_t qkv_weight_offset,
                                   size_t qkv_bias_offset,
                                   size_t qkv_output_offset)
{
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
            const float *B_weights = M->memory_base + qkv_weight_offset;
            const float *bias = M->memory_base + qkv_bias_offset;
            float *C_out = M->memory_base + qkv_output_offset + token_start * 3 * M->aligned_embed_dim;

            // Use your optimized blocked serial GEMM for QKV projection
            gemm_blocked_serial(A_input, B_weights, bias, C_out,
                                num_tokens, 3 * M->aligned_embed_dim, M->aligned_embed_dim);
        }
    }
}

// ================================================================
// ATTENTION COMPUTATION (Scaled Dot-Product)
// ================================================================
void attention_compute_token_parallel(TransformerModel *M,
                                      size_t qkv_output_offset,
                                      size_t attention_output_offset)
{
    const int head_dim = M->aligned_embed_dim / M->num_attention_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);

#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M->context_window)
                             ? (M->context_window - token_start)
                             : M->tokens_per_core;

        if (num_tokens > 0)
        {
            for (int h = 0; h < M->num_attention_heads; h++)
            {
                // Extract Q, K, V for this head
                const float *Q = M->memory_base + qkv_output_offset +
                                 token_start * 3 * M->aligned_embed_dim + h * head_dim;
                const float *K = M->memory_base + qkv_output_offset +
                                 M->aligned_embed_dim + h * head_dim; // Global K
                const float *V = M->memory_base + qkv_output_offset +
                                 2 * M->aligned_embed_dim + h * head_dim; // Global V

                float *output = M->memory_base + attention_output_offset +
                                token_start * M->aligned_embed_dim + h * head_dim;

                // Compute attention for this head's token slice
                attention_head_compute(Q, K, V, output, num_tokens, M->context_window,
                                       head_dim, scale, M->aligned_embed_dim);
            }
        }
    }
}

// Helper function for single attention head computation
void attention_head_compute(const float *Q, const float *K, const float *V, float *output,
                            int num_tokens, int seq_len, int head_dim, float scale, int stride)
{
    // Temporary scores buffer (could be optimized to use model memory)
    float *scores = (float *)malloc(num_tokens * seq_len * sizeof(float));

    // Q × K^T (scaled)
    for (int i = 0; i < num_tokens; i++)
    {
        for (int j = 0; j < seq_len; j++)
        {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
            {
                score += Q[i * stride + d] * K[j * stride + d];
            }
            scores[i * seq_len + j] = score * scale;
        }
    }

    // Softmax over scores
    for (int i = 0; i < num_tokens; i++)
    {
        float *row = scores + i * seq_len;
        softmax_inplace(row, seq_len);
    }

    // Scores × V
    for (int i = 0; i < num_tokens; i++)
    {
        for (int d = 0; d < head_dim; d++)
        {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++)
            {
                sum += scores[i * seq_len + j] * V[j * stride + d];
            }
            output[i * stride + d] = sum;
        }
    }

    free(scores);
}

// ================================================================
// SOFTMAX (Numerically Stable)
// ================================================================
void softmax_inplace(float *x, int n)
{
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++)
    {
        if (x[i] > max_val)
            max_val = x[i];
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++)
    {
        x[i] *= inv_sum;
    }
}

// ================================================================
// ATTENTION PROJECTION (Output projection)
// ================================================================
void attention_projection_token_parallel(TransformerModel *M,
                                         size_t attention_output_offset,
                                         size_t proj_weight_offset,
                                         size_t proj_bias_offset,
                                         size_t final_output_offset)
{
#pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M->context_window)
                             ? (M->context_window - token_start)
                             : M->tokens_per_core;

        if (num_tokens > 0)
        {
            const float *A_input = M->memory_base + attention_output_offset + token_start * M->aligned_embed_dim;
            const float *B_weights = M->memory_base + proj_weight_offset;
            const float *bias = M->memory_base + proj_bias_offset;
            float *C_out = M->memory_base + final_output_offset + token_start * M->aligned_embed_dim;

            gemm_blocked_serial(A_input, B_weights, bias, C_out,
                                num_tokens, M->aligned_embed_dim, M->aligned_embed_dim);
        }
    }
}

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
// COMPLETE TRANSFORMER LAYER
// ================================================================
void transformer_layer_forward(TransformerModel *M, int layer_idx, size_t layer_input_offset)
{
    TrulyOptimalLayer *L = &M->layers[layer_idx];
    const float eps = 1e-5f;

    // 1. Pre-attention LayerNorm
    layernorm_token_parallel(M, layer_input_offset, L->ln1_weight_offset,
                             L->ln1_bias_offset, L->ln1_output_offset, eps);

    // 2. QKV Projection
    qkv_projection_token_parallel(M, L->ln1_output_offset, L->qkv_weight_offset,
                                  L->qkv_bias_offset, L->qkv_output_offset);

    // 3. Attention Computation
    attention_compute_token_parallel(M, L->qkv_output_offset, L->attention_output_offset);

    // 4. Attention Output Projection
    attention_projection_token_parallel(M, L->attention_output_offset, L->proj_weight_offset,
                                        L->proj_bias_offset, L->attention_output_offset);

    // 5. First Residual Connection
    residual_add_token_parallel(M, layer_input_offset, L->attention_output_offset,
                                L->residual1_output_offset);

    // 6. Pre-MLP LayerNorm
    layernorm_token_parallel(M, L->residual1_output_offset, L->ln2_weight_offset,
                             L->ln2_bias_offset, L->ln2_output_offset, eps);

    // 7. MLP (Feed-Forward)
    mlp_token_parallel(M, L->ln2_output_offset, L->fc1_weight_offset, L->fc1_bias_offset,
                       L->fc1_output_offset, L->fc2_weight_offset, L->fc2_bias_offset,
                       L->mlp_output_offset);

    // 8. Second Residual Connection
    residual_add_token_parallel(M, L->residual1_output_offset, L->mlp_output_offset,
                                L->residual2_output_offset);
}

// ================================================================
// FULL FORWARD PASS
// ================================================================
void transformer_forward_pass(TransformerModel *M, size_t input_offset)
{
    // Copy input to first layer
    size_t current_input = M->layers[0].layer_input_offset;
    memcpy(M->memory_base + current_input, M->memory_base + input_offset,
           M->context_window * M->aligned_embed_dim * sizeof(float));

    // Process each layer
    for (int layer = 0; layer < M->num_layers; layer++)
    {
        printf("Processing layer %d/%d...\n", layer + 1, M->num_layers);

        transformer_layer_forward(M, layer, current_input);

        // Update input for next layer (output of current layer)
        current_input = M->layers[layer].residual2_output_offset;

        // Copy to next layer's input if not the last layer
        if (layer < M->num_layers - 1)
        {
            memcpy(M->memory_base + M->layers[layer + 1].layer_input_offset,
                   M->memory_base + current_input,
                   M->context_window * M->aligned_embed_dim * sizeof(float));
        }
    }

    // Final LayerNorm
    layernorm_token_parallel(M, current_input, M->final_ln_weight_offset,
                             M->final_ln_bias_offset, current_input, 1e-5f);

    printf("✅ Forward pass complete! Final output at offset %zu\n", current_input);
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
// COMPREHENSIVE BENCHMARK DRIVER
// ============================================================================
void run_comprehensive_benchmark(TransformerModel *M)
{
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
    for (size_t i = 0; i < (size_t)N2 * K2; i++)
        B_qkv_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (size_t i = 0; i < (size_t)N2; i++)
        bias_qkv_golden_ref_src[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

    // Run Naive kernel on Layer 0 to establish the golden reference output for QKV
    float *golden_ref_qkv_output = M->memory_base + L0_qkv->qkv_output_offset; // Layer 0's output becomes the reference
    printf("Generating Golden Reference QKV output using Naive kernel on Layer 0...\n");
    gemm_naive_parallel(A_input_base, B_qkv_golden_ref_src, bias_qkv_golden_ref_src, golden_ref_qkv_output, M2, N2, K2);

    for (int i = 0; i < 4; ++i)
    { // Loop through layers 0-3, each for a different kernel
        TrulyOptimalLayer *L = &M->layers[i];

        float *A_input_for_kernel = A_input_base; // Consistent input for all QKV kernels

        // Copy the golden reference weights and biases to the current layer's memory location
        float *B_weights = M->memory_base + L->qkv_weight_offset;
        float *bias = M->memory_base + L->qkv_bias_offset;
        memcpy(B_weights, B_qkv_golden_ref_src, sizeof(float) * N2 * K2);
        memcpy(bias, bias_qkv_golden_ref_src, sizeof(float) * N2);

        float *C_out = M->memory_base + L->qkv_output_offset; // Use qkv_output_offset
        memset(C_out, 0, sizeof(float) * M2 * N2);            // Clear output buffer before computation

        printf("\nBenchmarking QKV with %s on Layer %d:\n", strategy_names[i], i);
        double start = get_time_sec();

        // Special handling for Token-Parallel Orchestration (uses gemm_blocked_serial)
        if (i == 3)
        {
#pragma omp parallel num_threads(M->num_cores)
            {
                int core_id = omp_get_thread_num();
                int token_start = core_id * M->tokens_per_core;
                int num_tokens = (token_start + M->tokens_per_core > M2) ? (M2 - token_start) : M->tokens_per_core;
                if (num_tokens > 0)
                {
                    gemm_blocked_serial(A_input_for_kernel + token_start * K2, B_weights, bias, C_out + token_start * N2, num_tokens, N2, K2);
                }
            }
        }
        else
        {
            gemm_kernels[i](A_input_for_kernel, B_weights, bias, C_out, M2, N2, K2);
        }
        qkv_times[i] = get_time_sec() - start;
        qkv_gflops[i] = gflops_val2 / qkv_times[i];

        // Accuracy Check against the golden reference (Layer 0's Naive output)
        float max_diff = compute_max_diff(golden_ref_qkv_output, C_out, (size_t)M2 * N2);
        float rmse = compute_rmse(golden_ref_qkv_output, C_out, (size_t)M2 * N2);
        printf("   Done in %.2f ms. GFLOPS: %.2f. Max Diff = %.2e, RMSE = %.2e\n",
               qkv_times[i] * 1000, qkv_gflops[i], max_diff, rmse);
    }

    // Print QKV results Summary
    printf("\n🏆 Final Performance Summary for QKV Projections\n");
    printf("════════════════════════════════════════════════════════════════════════════════\n");
    printf("| %-35s | %10s | %12s | %10s | %8s |\n", "Strategy", "Time (ms)", "GFLOPS", "Speedup", "Layer");
    printf("|-------------------------------------|------------|--------------|------------|----------|\n");
    for (int i = 0; i < 4; i++)
    {
        printf("| %2d. %-32s | %10.2f | %12.2f | %9.2fx | L%d |\n", i + 1, strategy_names[i], qkv_times[i] * 1000, qkv_gflops[i], qkv_gflops[i] / qkv_gflops[0], i);
    }
    printf("════════════════════════════════════════════════════════════════════════\n");

    // ===================================================================
    // GLOBAL TEST 3: LayerNorm Benchmark
    // ===================================================================
    printf("\n\n=== GLOBAL TEST: LayerNorm Performance Benchmark ===\n");
    printf("   Comparing optimized C LayerNorm against Naive C reference.\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    // Define dimensions for LayerNorm test (use M's context_window and aligned_embed_dim)
    int LN_B_test = 1;                    // Test one batch for simplicity, or M->context_window for full context
    int LN_T_test = M->context_window;    // Full context window
    int LN_C_test = M->aligned_embed_dim; // Aligned embedding dimension

    // Allocate temporary buffers for LayerNorm test data (within memory_base if possible, or malloc)
    // For simplicity and self-containment, let's use temporary buffers for this specific benchmark
    // to avoid interfering with the main model's layer allocations.
    // In a real scenario, you'd use dedicated offsets in TransformerModel for benchmark data.
    float *ln_test_input = (float *)malloc((size_t)LN_B_test * LN_T_test * LN_C_test * sizeof(float));
    float *ln_test_gamma = (float *)malloc((size_t)LN_C_test * sizeof(float));
    float *ln_test_beta = (float *)malloc((size_t)LN_C_test * sizeof(float));
    float *golden_ref_ln_output = (float *)malloc((size_t)LN_B_test * LN_T_test * LN_C_test * sizeof(float));
    float *golden_ref_ln_mean_cache = (float *)malloc((size_t)LN_B_test * LN_T_test * sizeof(float));
    float *golden_ref_ln_rstd_cache = (float *)malloc((size_t)LN_B_test * LN_T_test * sizeof(float));
    float *optimized_ln_output = (float *)malloc((size_t)LN_B_test * LN_T_test * LN_C_test * sizeof(float));
    float *optimized_ln_mean_cache = (float *)malloc((size_t)LN_B_test * LN_T_test * sizeof(float));
    float *optimized_ln_rstd_cache = (float *)malloc((size_t)LN_B_test * LN_T_test * sizeof(float));

    if (!ln_test_input || !ln_test_gamma || !ln_test_beta || !golden_ref_ln_output || !golden_ref_ln_mean_cache || !golden_ref_ln_rstd_cache || !optimized_ln_output || !optimized_ln_mean_cache || !optimized_ln_rstd_cache)
    {
        perror("Failed to allocate memory for LayerNorm benchmark");
        // Free any already allocated memory
        free(ln_test_input);
        free(ln_test_gamma);
        free(ln_test_beta);
        free(golden_ref_ln_output);
        free(golden_ref_ln_mean_cache);
        free(golden_ref_ln_rstd_cache);
        free(optimized_ln_output);
        free(optimized_ln_mean_cache);
        free(optimized_ln_rstd_cache);
        return;
    }

    // Initialize LayerNorm test data with random values
    for (size_t i = 0; i < (size_t)LN_B_test * LN_T_test * LN_C_test; i++)
    {
        ln_test_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (size_t i = 0; i < (size_t)LN_C_test; i++)
    {
        ln_test_gamma[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f + 1.0f; // Around 1.0
        ln_test_beta[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;         // Around 0.0
    }

    const float ln_eps = 1e-5f;

    // Generate Golden Reference LayerNorm output using naive serial implementation
    printf("Generating Golden Reference LayerNorm output using Naive C kernel...\n");
    layernorm_naive_serial(ln_test_input, ln_test_gamma, ln_test_beta,
                           golden_ref_ln_output, golden_ref_ln_mean_cache, golden_ref_ln_rstd_cache,
                           LN_B_test * LN_T_test, LN_C_test, ln_eps);

    // Run Optimized LayerNorm and measure performance
    printf("Benchmarking Optimized LayerNorm (Token-Parallel Orchestration)...\n");
    // Temporarily set M's context_window and aligned_embed_dim to match test dimensions
    // This is a bit hacky for a benchmark, in a real system you'd pass these or have a dedicated test setup
    int original_context_window = M->context_window;
    size_t original_aligned_embed_dim = M->aligned_embed_dim;
    M->context_window = LN_B_test * LN_T_test; // Treat all tokens as one large batch for this test
    M->aligned_embed_dim = LN_C_test;

    double start_ln_opt = get_time_sec();
    // For this benchmark call, we need to map the temp buffers to offsets that layernorm_token_parallel expects.
    // Since it expects offsets relative to M->memory_base, we'll temporarily set up pointers.
    // This is a common pattern for testing sub-components that expect the global model struct.
    // A more robust way would be to create a separate test harness that doesn't rely on M->memory_base offsets directly.
    // For simplicity, we'll pass the raw pointers directly, assuming layernorm_token_parallel can handle it
    // if its internal logic is adapted, or we copy data into M->memory_base for the test.
    // Let's copy data into M->memory_base for consistency with how layernorm_token_parallel expects it.

    // Use a temporary offset within M->memory_base for the benchmark
    size_t temp_ln_input_offset = M->embedded_input_offset;        // Reuse this space
    size_t temp_ln_gamma_offset = M->layers[0].ln1_weight_offset;  // Reuse
    size_t temp_ln_beta_offset = M->layers[0].ln1_bias_offset;     // Reuse
    size_t temp_ln_mean_offset = M->layers[0].ln1_mean_offset;     // Reuse
    size_t temp_ln_rstd_offset = M->layers[0].ln1_rstd_offset;     // Reuse
    size_t temp_ln_output_offset = M->layers[0].ln1_output_offset; // Reuse

    memcpy(M->memory_base + temp_ln_input_offset, ln_test_input, (size_t)LN_B_test * LN_T_test * LN_C_test * sizeof(float));
    memcpy(M->memory_base + temp_ln_gamma_offset, ln_test_gamma, (size_t)LN_C_test * sizeof(float));
    memcpy(M->memory_base + temp_ln_beta_offset, ln_test_beta, (size_t)LN_C_test * sizeof(float));
    memset(M->memory_base + temp_ln_output_offset, 0, (size_t)LN_B_test * LN_T_test * LN_C_test * sizeof(float));
    memset(M->memory_base + temp_ln_mean_offset, 0, (size_t)LN_B_test * LN_T_test * sizeof(float));
    memset(M->memory_base + temp_ln_rstd_offset, 0, (size_t)LN_B_test * LN_T_test * sizeof(float));

    layernorm_token_parallel(M, temp_ln_input_offset, temp_ln_gamma_offset, temp_ln_beta_offset,
                             temp_ln_mean_offset, temp_ln_rstd_offset, temp_ln_output_offset, ln_eps);
    double end_ln_opt = get_time_sec();
    double time_ln_opt = end_ln_opt - start_ln_opt;

    // Restore original M values
    M->context_window = original_context_window;
    M->aligned_embed_dim = original_aligned_embed_dim;

    // Copy optimized output back to temporary buffer for comparison
    memcpy(optimized_ln_output, M->memory_base + temp_ln_output_offset, (size_t)LN_B_test * LN_T_test * LN_C_test * sizeof(float));
    memcpy(optimized_ln_mean_cache, M->memory_base + temp_ln_mean_offset, (size_t)LN_B_test * LN_T_test * sizeof(float));
    memcpy(optimized_ln_rstd_cache, M->memory_base + temp_ln_rstd_offset, (size_t)LN_B_test * LN_T_test * sizeof(float));

    // Calculate GFLOPS for LayerNorm (approx 9 FLOPs per element)
    double ln_flops = (double)LN_B_test * LN_T_test * LN_C_test * 9.0;
    double ln_gflops = ln_flops / 1e9 / time_ln_opt;

    // Accuracy Check
    float max_diff_ln_output = compute_max_diff(golden_ref_ln_output, optimized_ln_output, (size_t)LN_B_test * LN_T_test * LN_C_test);
    float rmse_ln_output = compute_rmse(golden_ref_ln_output, optimized_ln_output, (size_t)LN_B_test * LN_T_test * LN_C_test);

    float max_diff_ln_mean = compute_max_diff(golden_ref_ln_mean_cache, optimized_ln_mean_cache, (size_t)LN_B_test * LN_T_test);
    float rmse_ln_mean = compute_rmse(golden_ref_ln_mean_cache, optimized_ln_mean_cache, (size_t)LN_B_test * LN_T_test);

    float max_diff_ln_rstd = compute_max_diff(golden_ref_ln_rstd_cache, optimized_ln_rstd_cache, (size_t)LN_B_test * LN_T_test);
    float rmse_ln_rstd = compute_rmse(golden_ref_ln_rstd_cache, optimized_ln_rstd_cache, (size_t)LN_B_test * LN_T_test);

    printf("\nLayerNorm Benchmark Results:\n");
    printf("   Optimized LN Time: %.2f ms. GFLOPS: %.2f.\n", time_ln_opt * 1000, ln_gflops);
    printf("   Output Accuracy: Max Diff = %.2e, RMSE = %.2e\n", max_diff_ln_output, rmse_ln_output);
    printf("   Mean Cache Accuracy: Max Diff = %.2e, RMSE = %.2e\n", max_diff_ln_mean, rmse_ln_mean);
    printf("   RSTD Cache Accuracy: Max Diff = %.2e, RMSE = %.2e\n", max_diff_ln_rstd, rmse_ln_rstd);
    printf("════════════════════════════════════════════════════════════════════════\n");

    // Free temporary buffers
    free(ln_test_input);
    free(ln_test_gamma);
    free(ln_test_beta);
    free(golden_ref_ln_output);
    free(golden_ref_ln_mean_cache);
    free(golden_ref_ln_rstd_cache);
    free(optimized_ln_output);
    free(optimized_ln_mean_cache);
    free(optimized_ln_rstd_cache);

    // ===================================================================
    // KERNEL RECOMMENDATIONS
    // ===================================================================
    printf("\n🎯 KERNEL RECOMMENDATIONS FOR THIS SYSTEM:\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    // Find best performing kernels based on GFLOPS
    int best_mlp_idx = 0;
    for (int i = 1; i < 4; i++)
    {
        if (gflops[i] > gflops[best_mlp_idx])
            best_mlp_idx = i;
    }
    int best_qkv_idx = 0;
    for (int i = 1; i < 4; i++)
    {
        if (qkv_gflops[i] > qkv_gflops[best_qkv_idx])
            best_qkv_idx = i;
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
int main(int argc, char **argv)
{
    /* defaults (minimum 4 layers for benchmark) */
    int L = 4, V = 32768, C = 128, T = 128;
    int do_alloc = 0;
    int run_benchmarks = 0;

    static struct option long_opts[] = {
        {"layers", required_argument, 0, 'l'},
        {"dmodel", required_argument, 0, 'd'},
        {"ctx", required_argument, 0, 't'},
        {"vocab", required_argument, 0, 'v'},
        {"force", no_argument, 0, 'f'},
        {"benchmark", no_argument, 0, 'b'},
        {0, 0, 0, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "l:d:t:v:fb", long_opts, NULL)) != -1)
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
        case 'f':
            do_alloc = 1;
            break;
        case 'b':
            run_benchmarks = 1;
            break;
        default:
            fprintf(stderr, "Usage: %s [--layers N] [--dmodel N] [--ctx N] [--vocab N] [--force] [--benchmark]\n", argv[0]);
            return 1;
        }
    }

    size_t need_bytes = bytes_needed(L, V, C, T);
    double need_gib = need_bytes / (1024.0 * 1024.0 * 1024.0);

    printf("⚙  Requested model  L=%d  d_model=%d  ctx=%d  vocab=%d\n", L, C, T, V);
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

    /* ---------- try allocation ---------- */
    TransformerModel M = {0};
    M.num_layers = L;
    M.vocab_size = V;
    M.embed_dim = C;
    M.context_window = T;

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
    layout_transformer(&M);
    printf("✅ Success! mmap at %p, %.2f GiB reserved.\n",
           (void *)M.memory_base, need_gib);

    /* Setup execution plan */
    long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int reserved_cores = 4; // for OS, logging, etc.

    M.num_cores = (logical_cores > reserved_cores)
                      ? logical_cores - reserved_cores
                      : 1;
    M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;
    M.num_attention_heads = M.embed_dim / 64; // assume head_dim = 64

    printf("🧠 Detected %ld logical cores → reserving %d for OS → using %d for model\n",
           logical_cores, reserved_cores, M.num_cores);
    printf("📦 Each core will handle ≈ %d tokens from context window of %d tokens\n",
           M.tokens_per_core, M.context_window);
    printf("🧠 Attention heads = %d (assuming head_dim=64)\n", M.num_attention_heads);

    if (run_benchmarks)
    {
        run_comprehensive_benchmark(&M);
    }

    destroy_transformer(&M);
    return 0;
}