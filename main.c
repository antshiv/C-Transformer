/***********************************************************************
 *  SINGLE-BLOCK, CACHE-ALIGNED GPT-2 LAYOUT  (pure C demo)
 *  ---------------------------------------------------------------
 *  • one huge allocation for ALL weights + activations
 *  • 64-byte alignment for every tensor
 *  • 2 MB huge-page backing for minimal TLB misses
 *  • bump() = zero-fragmentation offset math
 ***********************************************************************/

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h> /* MAP_HUGETLB / munmap */
#include <unistd.h>

/* ─── alignment targets ───────────────────────────────────────────── */
#define CACHE_ALIGN 64ULL
#define HUGE_ALIGN (2ULL * 1024 * 1024) /* 2 MB huge page */

/* ─── tiny helpers ────────────────────────────────────────────────── */
static inline size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

/* best-effort huge-page allocator (falls back to THP) */
static void *huge_alloc(size_t bytes)
{
    size_t len = align_up(bytes, HUGE_ALIGN);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED)
        return p; /* explicit huge page ok  */

    /* fallback: page-aligned malloc + THP */
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
    size_t layer_input_offset, ln1_output_offset;
    size_t qkv_weight_offset, qkv_bias_offset, qkv_output_offset;
    size_t proj_weight_offset, proj_bias_offset;
    size_t attention_output_offset, residual1_output_offset;
    size_t ln2_weight_offset, ln2_bias_offset, ln2_output_offset;
    size_t fc1_weight_offset, fc1_bias_offset;
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
    *off = align_up(*off, alignB / sizeof(float)); /* align in floats */
    size_t here = *off;
    *off += count;
    return here;
}

/* ─── lay out the entire model ───────────────────────────────────── */
void layout_transformer(TransformerModel *M)
{
    size_t off = 0;

    // Ensure each token embedding vector is cache-aligned
    size_t aligned_embed_dim = align_up(M->embed_dim, CACHE_ALIGN / sizeof(float));
    M->aligned_embed_dim = aligned_embed_dim;

    // Embeddings
    M->token_emb_offset = bump(&off, (size_t)M->vocab_size * aligned_embed_dim, CACHE_ALIGN);
    M->pos_emb_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
    M->embedded_input_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);

    // Per-layer layout
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
        L->fc2_weight_offset = bump(&off, 4ULL * aligned_embed_dim * aligned_embed_dim, CACHE_ALIGN);
        L->fc2_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
        L->mlp_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
        L->residual2_output_offset = bump(&off, (size_t)M->context_window * aligned_embed_dim, CACHE_ALIGN);
    }

    M->layer_stride = M->layers[1].ln1_weight_offset - M->layers[0].ln1_weight_offset;

    // Final LayerNorm
    M->final_ln_weight_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);
    M->final_ln_bias_offset = bump(&off, aligned_embed_dim, CACHE_ALIGN);

    // Total memory required
    M->total_floats = off;
    M->memory_base = huge_alloc(off * sizeof(float));
}

/* ─── destruction helper ─────────────────────────────────────────── */
void destroy_transformer(TransformerModel *M)
{
    munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
    free(M->layers);
}

/* gpt2_layout_capacity.c – same includes & structs as before … */

#include <getopt.h> /* for long-options parsing */

/* ------------- new function: size_t bytes_needed(...) --------------- */
static size_t bytes_needed(int layers, int vocab, int d_model, int ctx)
{
    size_t C = d_model;
    size_t T = ctx;
    size_t V = vocab;

    /* embeddings */
    size_t token = V * C;
    size_t pos = T * C;
    size_t embed = T * C;

    /* per-layer working size (same as bump logic) */
    size_t perL =
        /* ln1 W+B */ 2 * C +
        /* ln1 in+out */ 2 * T * C +
        /* QKV W+B+out */ (3 * C * C + 3 * C + 3 * T * C) +
        /* proj W+B+out+res1*/ (C * C + C + 2 * T * C) +
        /* ln2 W+B+out */ (2 * C + T * C) +
        /* fc1/fc2 W+B */ (8 * C * C + 5 * C) +
        /* mlp_out + res2 */ (2 * T * C);

    size_t final_ln = 2 * C;

    size_t total_floats = token + pos + embed + layers * perL + final_ln;
    return total_floats * sizeof(float); /* bytes */
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
    if (token_start >= M->context_window)
        return NULL; // bounds-safe
    return M->memory_base + base_offset + token_start * vector_dim;
}

/* Return pointer and number of tokens this core owns */
static inline float *get_slice_and_len(TransformerModel *M, int core_id,
                                       size_t base_offset, size_t vector_dim,
                                       size_t *out_tokens)
{
    size_t t0 = core_id * M->tokens_per_core;
    size_t t1 = (core_id + 1) * M->tokens_per_core;
    if (t0 >= M->context_window)
        return NULL;
    if (t1 > M->context_window)
        t1 = M->context_window;
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
    size_t offset = base_offset + token_start * total_heads * head_dim // full rows
                    + head_id * head_dim;                              // head offset
    return M->memory_base + offset;
}

/* GEMM section */
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

// Enhanced timing function
static inline double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Cache warming function
void warm_cache(void *ptr, size_t bytes)
{
    volatile char *p = (volatile char *)ptr;
    for (size_t i = 0; i < bytes; i += 64)
    { // 64-byte cache lines
        (void)p[i];
    }
}

// Flush cache lines (for cold cache testing)
void flush_cache(void *ptr, size_t bytes)
{
    char *p = (char *)ptr;
    for (size_t i = 0; i < bytes; i += 64)
    {
        _mm_clflush(p + i);
    }
    _mm_mfence();
}

// Your original naive implementation
void gemm_1d(float *A, float *B, float *C, int M, int N, int K)
{
     #pragma omp parallel for collapse(2)  // Parallelize outer loops
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[j * K + k]; // simulate B^T
            }
            C[i * N + j] = sum;
        }
    }
}

// Enhanced AVX-512 with better vectorization
void gemm_1d_avx512_enhanced(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    const int simd_width = 16;

    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < M; i++)
    {
        // Prefetch next row of A
        if (i + 1 < M)
        {
            _mm_prefetch((char *)&A[(i + 1) * K], _MM_HINT_T0);
        }

        for (int j = 0; j < N; j++)
        {
            // Prefetch next row of B
            if (j + 1 < N)
            {
                _mm_prefetch((char *)&B[(j + 1) * K], _MM_HINT_T0);
            }

            __m512 sum_vec = _mm512_setzero_ps();
            int k;

            // Main vectorized loop
            for (k = 0; k <= K - simd_width; k += simd_width)
            {
                __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            // Horizontal reduction
            float sum = _mm512_reduce_add_ps(sum_vec);

            // Handle remaining elements
            for (; k < K; k++)
            {
                sum += A[i * K + k] * B[j * K + k];
            }

            C[i * N + j] = sum + bias[j];
        }
    }
}

// Blocked GEMM for better cache locality
void gemm_1d_blocked(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    const int block_size = 64; // Tune based on cache size

    // Initialize C with bias
    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = bias[j];
        }
    }

    // Blocked multiplication
    #pragma omp parallel for collapse(3) 
    for (int ii = 0; ii < M; ii += block_size)
    {
        for (int jj = 0; jj < N; jj += block_size)
        {
            for (int kk = 0; kk < K; kk += block_size)
            {

                int i_end = (ii + block_size < M) ? ii + block_size : M;
                int j_end = (jj + block_size < N) ? jj + block_size : N;
                int k_end = (kk + block_size < K) ? kk + block_size : K;

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

// Performance statistics structure
typedef struct
{
    double time_sec;
    double gflops;
    double bandwidth_gb_s;
    float max_error;
    float avg_error;
} PerfStats;

// Calculate performance metrics
PerfStats calculate_perf(double time_sec, int M, int N, int K, float *C_ref, float *C_test)
{
    PerfStats stats = {0};

    stats.time_sec = time_sec;

    // GFLOPS calculation: 2*M*N*K operations (multiply-add)
    double total_ops = 2.0 * M * N * K;
    stats.gflops = total_ops / (time_sec * 1e9);

    // Memory bandwidth (rough estimate)
    // Reading A: M*K, Reading B: N*K, Writing C: M*N (all floats = 4 bytes)
    double total_bytes = 4.0 * (M * K + N * K + M * N);
    stats.bandwidth_gb_s = total_bytes / (time_sec * 1e9);

    // Error analysis
    if (C_ref && C_test)
    {
        float sum_error = 0.0f;
        stats.max_error = 0.0f;

        for (int i = 0; i < M * N; i++)
        {
            float error = fabsf(C_test[i] - C_ref[i]);
            if (error > stats.max_error)
                stats.max_error = error;
            sum_error += error;
        }
        stats.avg_error = sum_error / (M * N);
    }

    return stats;
}

// Enhanced benchmark function
void benchmark_gemm_variant(const char *name,
                            void (*gemm_func)(float *, float *, float *, float *, int, int, int),
                            float *A, float *B, float *bias, float *C,
                            int M, int N, int K,
                            float *C_reference,
                            int warmup_runs, int benchmark_runs)
{

    size_t matrix_bytes = (M * K + N * K + M * N) * sizeof(float);

    printf("\n🔬 Testing %s:\n", name);

    // Warmup runs (cache warm)
    for (int r = 0; r < warmup_runs; r++)
    {
        warm_cache(A, M * K * sizeof(float));
        warm_cache(B, N * K * sizeof(float));
        gemm_func(A, B, bias, C, M, N, K);
    }

    // Cold cache benchmark
    flush_cache(A, M * K * sizeof(float));
    flush_cache(B, N * K * sizeof(float));
    flush_cache(C, M * N * sizeof(float));
    _mm_mfence();

    double t1 = get_time_sec();
    gemm_func(A, B, bias, C, M, N, K);
    double t2 = get_time_sec();

    PerfStats cold_stats = calculate_perf(t2 - t1, M, N, K, C_reference, C);
    printf("  🧊 Cold cache: %.3f ms, %.2f GFLOPS, %.2f GB/s",
           cold_stats.time_sec * 1000, cold_stats.gflops, cold_stats.bandwidth_gb_s);

    if (C_reference)
    {
        printf(", max_err: %.2e", cold_stats.max_error);
    }
    printf("\n");

    // Warm cache benchmark (multiple runs for stability)
    double best_time = 1e9;
    double total_time = 0.0;

    for (int r = 0; r < benchmark_runs; r++)
    {
        warm_cache(A, M * K * sizeof(float));
        warm_cache(B, N * K * sizeof(float));

        double t1 = get_time_sec();
        gemm_func(A, B, bias, C, M, N, K);
        double t2 = get_time_sec();

        double run_time = t2 - t1;
        if (run_time < best_time)
            best_time = run_time;
        total_time += run_time;
    }

    double avg_time = total_time / benchmark_runs;
    PerfStats warm_stats = calculate_perf(best_time, M, N, K, C_reference, C);
    PerfStats avg_stats = calculate_perf(avg_time, M, N, K, C_reference, C);

    printf("  🔥 Warm cache (best): %.3f ms, %.2f GFLOPS, %.2f GB/s\n",
           warm_stats.time_sec * 1000, warm_stats.gflops, warm_stats.bandwidth_gb_s);
    printf("  📊 Warm cache (avg):  %.3f ms, %.2f GFLOPS, %.2f GB/s\n",
           avg_stats.time_sec * 1000, avg_stats.gflops, avg_stats.bandwidth_gb_s);

    if (C_reference)
    {
        printf("  ✅ Accuracy: max_err=%.2e, avg_err=%.2e\n",
               warm_stats.max_error, warm_stats.avg_error);
    }
}

// Wrapper functions for consistent interface
void naive_wrapper(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    gemm_1d(A, B, C, M, N, K);
    // Add bias manually
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] += bias[j];
        }
    }
}

void avx512_wrapper(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    gemm_1d_avx512_enhanced(A, B, bias, C, M, N, K);
}

void blocked_wrapper(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    gemm_1d_blocked(A, B, bias, C, M, N, K);
}

void test_and_benchmark_gemm_enhanced(TransformerModel *M) {
    if (M->embed_dim % 16 != 0) {
        fprintf(stderr, "⚠️  embed_dim must be divisible by 16 for AVX-512\n");
        return;
    }
    
    int M_dim = M->context_window;
    int N_dim = M->context_window; 
    int K_dim = M->embed_dim;
    
    printf("\n🧪 Enhanced GEMM Benchmarks (M=%d, N=%d, K=%d)\n", M_dim, N_dim, K_dim);
    printf("📊 Problem size: %.2f MB matrices, %.2f GFLOP operation\n",
           (M_dim * K_dim + N_dim * K_dim + M_dim * N_dim) * 4.0 / 1e6,
           2.0 * M_dim * N_dim * K_dim / 1e9);
    
    // USE EXISTING MODEL MEMORY INSTEAD OF ALLOCATING NEW
    TrulyOptimalLayer *L = &M->layers[0];  // Use first layer for testing
    
    float *A = M->memory_base + L->layer_input_offset;     // Use layer input as A
    float *B = M->memory_base + L->qkv_weight_offset;      // Use QKV weights as B  
    float *bias = M->memory_base + L->qkv_bias_offset;     // Use QKV bias
    float *C_naive = M->memory_base + L->qkv_output_offset; // Use QKV output for result
    
    // For comparison, we need another output buffer - use next layer's input
    float *C_avx512 = M->memory_base + M->layers[1].layer_input_offset;
    float *C_blocked = M->memory_base + M->layers[1].ln1_output_offset;
    
    printf("🎯 Using actual model memory:\n");
    printf("  A (input): %p\n", (void*)A);
    printf("  B (weights): %p\n", (void*)B);
    printf("  C (output): %p\n", (void*)C_naive);
    
    // Initialize with realistic AI patterns
    srand(42);  
    for (int i = 0; i < M_dim * K_dim; ++i) {
        A[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;  
    }
    for (int i = 0; i < N_dim * K_dim; ++i) {
        B[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f; 
    }
    for (int i = 0; i < N_dim; ++i) {
        bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f; 
    }
    
    const int warmup_runs = 3;
    const int benchmark_runs = 10;
    
    // Test naive implementation (reference) 
    benchmark_gemm_variant("Naive GEMM + Bias", naive_wrapper,
                          A, B, bias, C_naive, M_dim, N_dim, K_dim, 
                          NULL, warmup_runs, benchmark_runs);
    
    // Test AVX-512 implementation
    benchmark_gemm_variant("AVX-512 GEMM", avx512_wrapper,
                          A, B, bias, C_avx512, M_dim, N_dim, K_dim, 
                          C_naive, warmup_runs, benchmark_runs);
    
    // Test blocked implementation  
    benchmark_gemm_variant("Blocked AVX-512 GEMM", blocked_wrapper,
                          A, B, bias, C_blocked, M_dim, N_dim, K_dim, 
                          C_naive, warmup_runs, benchmark_runs);
    
    printf("\n📈 Performance Summary:\n");
    printf("  Naive implementation provides baseline correctness\n");
    printf("  AVX-512 should show ~8-16x speedup (if memory-bound)\n");
    printf("  Blocked version should show better cache efficiency\n");
    
    // DON'T modify M->total_floats since we used existing memory
}

/* End of GEMM section */

/* ---------------- main with --size-only / --force -------------------- */
int main(int argc, char **argv)
{
    /* defaults (tiny) */
    int L = 2, V = 32768, C = 128, T = 128;
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
    while ((c = getopt_long(argc, argv, "l:d:t:v:f:b", long_opts, NULL)) != -1)
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
    printf("→ Would need ≈ %.2f GiB (%.0f bytes)\n",
           need_gib, (double)need_bytes);

    if (!do_alloc)
    {
        printf("Dry-run only (no allocation).   Pass --force to allocate.\n");
        return 0;
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

    printf("Allocating huge block...  this may page-fault if hugepages are missing\n");
    layout_transformer(&M);
    printf("✅ Success!  mmap at %p, %.2f GiB reserved.\n",
           (void *)M.memory_base, need_gib);

    /* Setup execution plan */
    long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int reserved_cores = 4; // for OS, logging, etc.

    M.num_cores = (logical_cores > reserved_cores)
                      ? logical_cores - reserved_cores
                      : 1;
    M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;
    M.num_attention_heads = M.embed_dim / 64; // assume head_dim = 64

    // Only run benchmarks if requested
    if (do_alloc && run_benchmarks)
    {
        test_and_benchmark_gemm_enhanced(&M);
    }

    printf("🧠 Detected %ld logical cores → reserving %d for OS → using %d for model\n",
           logical_cores, reserved_cores, M.num_cores);
    printf("📦 Each core will handle ≈ %d tokens from context window of %d tokens\n",
           M.tokens_per_core, M.context_window);
    printf("🧠 Attention heads = %d (assuming head_dim=64)\n", M.num_attention_heads);

    destroy_transformer(&M);
    return 0;
}
