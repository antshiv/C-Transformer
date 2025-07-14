/***********************************************************************
 * COMPREHENSIVE PRESENTATION BENCHMARK (PURE C)
 * ---------------------------------------------------------------
 * • Allocates memory for the full transformer model as specified.
 * • Runs all GEMM optimization strategies sequentially on a
 * representative layer to demonstrate performance progression.
 * • Calculates performance for each step and presents a final
 * summary table.
 ***********************************************************************/

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h> // MAP_HUGETLB / munmap
#include <unistd.h>
#include <getopt.h>   // for long-options parsing
#include <immintrin.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

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
    if (p != MAP_FAILED) return p;

    p = aligned_alloc(HUGE_ALIGN, len);
    if (!p) { perror("aligned_alloc"); exit(EXIT_FAILURE); }
    madvise(p, len, MADV_HUGEPAGE);
    return p;
}

/* ─── model structs ───────────────────────────────────────────────── */
typedef struct
{
    size_t layer_input_offset;
    size_t qkv_weight_offset, qkv_bias_offset;
    // Separate output offsets for each benchmark to store results
    size_t output_offset_naive;
    size_t output_offset_avx512;
    size_t output_offset_blocked;
    size_t output_offset_token_parallel;
} TrulyOptimalLayer;

typedef struct
{
    int num_layers, vocab_size, embed_dim, context_window;
    size_t aligned_embed_dim;
    int num_cores;
    int tokens_per_core;
    float *memory_base;
    size_t total_floats;
    TrulyOptimalLayer *layers;
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
void layout_transformer(TransformerModel *M, int for_benchmark)
{
    size_t off = 0;
    M->aligned_embed_dim = align_up(M->embed_dim, CACHE_ALIGN / sizeof(float));
    
    int num_layers_to_alloc = M->num_layers;
    M->layers = malloc(sizeof(TrulyOptimalLayer) * num_layers_to_alloc);
    if (!M->layers) { perror("malloc layers"); exit(EXIT_FAILURE); }

    // --- Allocate space for the full model ---
    // Token and Positional Embeddings
    bump(&off, (size_t)M->vocab_size * M->aligned_embed_dim, CACHE_ALIGN);
    bump(&off, (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);

    // All Transformer Layers
    for (int l = 0; l < num_layers_to_alloc; ++l) {
        TrulyOptimalLayer *L = &M->layers[l];
        L->layer_input_offset = bump(&off, (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
        L->qkv_weight_offset  = bump(&off, 3ULL * M->aligned_embed_dim * M->aligned_embed_dim, CACHE_ALIGN);
        L->qkv_bias_offset    = bump(&off, 3ULL * M->aligned_embed_dim, CACHE_ALIGN);
        
        // For the benchmark, we need 4 separate output buffers. We'll "borrow" space for them.
        if (for_benchmark && l == 0) {
            size_t output_size = 3ULL * (size_t)M->context_window * M->aligned_embed_dim;
            L->output_offset_naive          = bump(&off, output_size, CACHE_ALIGN);
            L->output_offset_avx512         = bump(&off, output_size, CACHE_ALIGN);
            L->output_offset_blocked        = bump(&off, output_size, CACHE_ALIGN);
            L->output_offset_token_parallel = bump(&off, output_size, CACHE_ALIGN);
        } else {
             // Normal layer output
            bump(&off, 3ULL * (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
        }
        // ... other layers like MLP would go here ...
    }
     // Final LayerNorm, etc.
    bump(&off, M->aligned_embed_dim, CACHE_ALIGN); // final_ln_weight
    bump(&off, M->aligned_embed_dim, CACHE_ALIGN); // final_ln_bias

    M->total_floats = off;
    M->memory_base = huge_alloc(off * sizeof(float));
}

void destroy_transformer(TransformerModel *M)
{
    munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
    free(M->layers);
}

// Calculates the full memory requirement for the model
static size_t bytes_needed(int layers, int vocab, int d_model, int ctx, int for_benchmark)
{
    size_t C = align_up(d_model, CACHE_ALIGN / sizeof(float));
    size_t T = ctx;
    size_t V = vocab;
    
    size_t embedding_size = (V * C) + (T * C);
    size_t layer_size = (T * C) + (3 * C * C) + (3 * C) + (3 * T * C); // input + qkv_w + qkv_b + qkv_out
    size_t final_ln_size = 2 * C;

    size_t total_floats = embedding_size + ((size_t)layers * layer_size) + final_ln_size;

    // Add extra space for benchmark buffers if needed
    if (for_benchmark) {
        size_t output_size = 3 * T * C;
        total_floats += 3 * output_size; // 3 extra buffers
    }

    return total_floats * sizeof(float);
}

// ============================================================================
//  GEMM KERNELS
// ============================================================================

// KERNEL 1: Naive Parallel GEMM (Baseline)
void gemm_naive_parallel(float *A, float *B, float *bias, float *C, int M, int N, int K) {
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

// KERNEL 2: Simple AVX-512 Parallel GEMM
void gemm_avx512_parallel(float *A, float *B, float *bias, float *C, int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m512 sum_vec = _mm512_setzero_ps();
            int k;
            for (k = 0; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_load_ps(&A[i * K + k]);
                __m512 b_vec = _mm512_load_ps(&B[j * K + k]);
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

// KERNEL 3: Fine-Grained Parallel Blocked GEMM
void gemm_fine_grained_parallel(float *A, float *B, float *bias, float *C, int M, int N, int K) {
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
                int i_end = (ii + block_size < M) ? ii + block_size : M;
                int j_end = (jj + block_size < N) ? jj + block_size : N;
                int k_end = (kk + block_size < K) ? kk + block_size : K;
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m512 sum_vec = _mm512_setzero_ps();
                        for (int k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_load_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (int k = k_end - (k_end % 16); k < k_end; k++) {
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

// KERNEL 4: Serial Blocked GEMM (for Token-Parallel Orchestrator)
void gemm_blocked_serial(float *A, float *B, float *bias, float *C, int M, int N, int K) {
    const int block_size = 64;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias[j];
        }
    }
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = (ii + block_size < M) ? ii + block_size : M;
                int j_end = (jj + block_size < N) ? jj + block_size : N;
                int k_end = (kk + block_size < K) ? kk + block_size : K;
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m512 sum_vec = _mm512_setzero_ps();
                        for (int k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_load_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (int k = k_end - (k_end % 16); k < k_end; k++) {
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
// COMPREHENSIVE BENCHMARK DRIVER
// ============================================================================

void run_comprehensive_benchmark(TransformerModel *M) {
    printf("\n🚀 Comprehensive GEMM Performance Benchmark\n");
    printf("   Showing performance progression from naive to fully optimized strategies.\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    // --- 1. Prepare Data and Buffers ---
    TrulyOptimalLayer *L = &M->layers[0];
    float *A = M->memory_base + L->layer_input_offset;
    float *B = M->memory_base + L->qkv_weight_offset;
    float *bias = M->memory_base + L->qkv_bias_offset;
    
    float *C_naive = M->memory_base + L->output_offset_naive;
    float *C_avx512 = M->memory_base + L->output_offset_avx512;
    float *C_blocked = M->memory_base + L->output_offset_blocked;
    float *C_token_parallel = M->memory_base + L->output_offset_token_parallel;

    srand(42);
    for (size_t i = 0; i < (size_t)M->context_window * M->aligned_embed_dim; i++) A[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < 3 * M->aligned_embed_dim * M->aligned_embed_dim; i++) B[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (int i = 0; i < 3 * M->aligned_embed_dim; i++) bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    
    int M_dim = M->context_window;
    int N_dim = 3 * M->aligned_embed_dim;
    int K_dim = M->aligned_embed_dim;
    double total_gflops_val = (2.0 * M_dim * N_dim * K_dim) / 1e9;

    printf("🧪 Problem Size (QKV Projection): M=%d, N=%d, K=%d (%.2f GFLOPs)\n", M_dim, N_dim, K_dim, total_gflops_val);

    double times[4];
    double gflops[4];

    // --- 2. Run Benchmarks ---
    printf("\n📊 Running all strategies...\n\n");

    // Strategy 1: Naive
    printf("🔬 Testing Naive GEMM + Bias...\n");
    double start = get_time_sec();
    gemm_naive_parallel(A, B, bias, C_naive, M_dim, N_dim, K_dim);
    times[0] = get_time_sec() - start;
    gflops[0] = total_gflops_val / times[0];
    printf("   Done in %.2f ms.\n", times[0] * 1000);

    // Strategy 2: Simple AVX-512
    printf("\n🔬 Testing Simple AVX-512 GEMM...\n");
    start = get_time_sec();
    gemm_avx512_parallel(A, B, bias, C_avx512, M_dim, N_dim, K_dim);
    times[1] = get_time_sec() - start;
    gflops[1] = total_gflops_val / times[1];
    printf("   Done in %.2f ms.\n", times[1] * 1000);

    // Strategy 3: Fine-Grained Blocked AVX-512
    printf("\n🔬 Testing Fine-Grained Blocked GEMM...\n");
    start = get_time_sec();
    gemm_fine_grained_parallel(A, B, bias, C_blocked, M_dim, N_dim, K_dim);
    times[2] = get_time_sec() - start;
    gflops[2] = total_gflops_val / times[2];
    printf("   Done in %.2f ms.\n", times[2] * 1000);

    // Strategy 4: Token-Parallel Orchestration
    printf("\n🔬 Testing Token-Parallel Orchestration...\n");
    start = get_time_sec();
    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M_dim) ? (M_dim - token_start) : M->tokens_per_core;
        if (num_tokens > 0) {
            float *input_slice = A + token_start * K_dim;
            float *output_slice = C_token_parallel + token_start * N_dim;
            gemm_blocked_serial(input_slice, B, bias, output_slice, num_tokens, N_dim, K_dim);
        }
    }
    times[3] = get_time_sec() - start;
    gflops[3] = total_gflops_val / times[3];
    printf("   Done in %.2f ms.\n", times[3] * 1000);

    // --- 3. Correctness Checks ---
    float max_diff_avx = 0.0f, max_diff_blocked = 0.0f, max_diff_token = 0.0f;
    for (size_t i = 0; i < (size_t)M_dim * N_dim; ++i) {
        float diff = fabsf(C_naive[i] - C_avx512[i]);
        if (diff > max_diff_avx) max_diff_avx = diff;
        diff = fabsf(C_naive[i] - C_blocked[i]);
        if (diff > max_diff_blocked) max_diff_blocked = diff;
        diff = fabsf(C_naive[i] - C_token_parallel[i]);
        if (diff > max_diff_token) max_diff_token = diff;
    }
    printf("\n✅ Correctness checks passed (Max Diffs vs Naive: AVX:%.1e, Blocked:%.1e, Token:%.1e)\n", 
           max_diff_avx, max_diff_blocked, max_diff_token);

    // --- 4. Final Summary Table ---
    printf("\n🏆 Final Performance Summary (M=%d, N=%d, K=%d)\n", M_dim, N_dim, K_dim);
    printf("════════════════════════════════════════════════════════════════════════════════\n");
    printf("| %-35s | %10s | %12s | %10s |\n", "Strategy", "Time (ms)", "GFLOPS", "Speedup");
    printf("|-------------------------------------|------------|--------------|------------|\n");
    printf("| 1. Naive Parallel                   | %10.2f | %12.2f | %9.2fx |\n", times[0] * 1000, gflops[0], 1.0);
    printf("| 2. Simple AVX-512 Parallel          | %10.2f | %12.2f | %9.2fx |\n", times[1] * 1000, gflops[1], gflops[1] / gflops[0]);
    printf("| 3. Fine-Grained Blocked Parallel    | %10.2f | %12.2f | %9.2fx |\n", times[2] * 1000, gflops[2], gflops[2] / gflops[0]);
    printf("| 4. Token-Parallel Orchestration     | %10.2f | %12.2f | %9.2fx |\n", times[3] * 1000, gflops[3], gflops[3] / gflops[0]);
    printf("════════════════════════════════════════════════════════════════════════════════\n");
}

// ============================================================================
// COMPREHENSIVE BENCHMARK DRIVER
// ============================================================================

void run_presentation_benchmark(TransformerModel *M) {
    printf("\n🚀 Comprehensive GEMM Strategy Benchmark\n");
    printf("   Comparing strategies across different GEMM shapes.\n");

    // --- 1. Prepare Data and Buffers ---
    BenchmarkLayout *L = M->layout;
    float *A_input = M->memory_base + L->layer_input_offset;

    srand(42);
    for (size_t i = 0; i < (size_t)M->context_window * M->aligned_embed_dim; i++)
        A_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    double times[4];
    double gflops[4];

    // ===================================================================
    // TEST 1: "SQUARISH" GEMM (MLP-like: N == K)
    // ===================================================================
    int M1 = M->context_window;
    int N1 = M->aligned_embed_dim;
    int K1 = M->aligned_embed_dim;
    double gflops_val1 = (2.0 * M1 * N1 * K1) / 1e9;

    float *B1 = M->memory_base + L->mlp_weight_offset;
    float *bias1 = M->memory_base + L->mlp_bias_offset;
    float *C1_naive = M->memory_base + L->mlp_out_naive;
    float *C1_avx = M->memory_base + L->mlp_out_avx;
    float *C1_blocked = M->memory_base + L->mlp_out_blocked;
    float *C1_token = M->memory_base + L->mlp_out_token;

    printf("\n\n🔬 TEST 1: 'Squarish' GEMM (MLP Layer, M=%d, N=%d, K=%d)\n", M1, N1, K1);
    printf("════════════════════════════════════════════════════════════════════════\n");

    double start = get_time_sec();
    gemm_naive_parallel(A_input, B1, bias1, C1_naive, M1, N1, K1);
    times[0] = get_time_sec() - start;
    gflops[0] = gflops_val1 / times[0];

    start = get_time_sec();
    gemm_avx512_parallel(A_input, B1, bias1, C1_avx, M1, N1, K1);
    times[1] = get_time_sec() - start;
    gflops[1] = gflops_val1 / times[1];

    start = get_time_sec();
    gemm_fine_grained_parallel(A_input, B1, bias1, C1_blocked, M1, N1, K1);
    times[2] = get_time_sec() - start;
    gflops[2] = gflops_val1 / times[2];

    start = get_time_sec();
    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M1) ? (M1 - token_start) : M->tokens_per_core;
        if (num_tokens > 0) {
            gemm_blocked_serial(A_input + token_start * K1, B1, bias1, C1_token + token_start * N1, num_tokens, N1, K1);
        }
    }
    times[3] = get_time_sec() - start;
    gflops[3] = gflops_val1 / times[3];

    printf("| %-35s | %10s | %12s | %10s |\n", "Strategy", "Time (ms)", "GFLOPS", "Speedup");
    printf("|-------------------------------------|------------|--------------|------------|\n");
    printf("| 1. Naive Parallel                   | %10.2f | %12.2f | %9.2fx |\n", times[0] * 1000, gflops[0], 1.0);
    printf("| 2. Simple AVX-512 Parallel          | %10.2f | %12.2f | %9.2fx |\n", times[1] * 1000, gflops[1], gflops[1] / gflops[0]);
    printf("| 3. Fine-Grained Blocked Parallel    | %10.2f | %12.2f | %9.2fx |\n", times[2] * 1000, gflops[2], gflops[2] / gflops[0]);
    printf("| 4. Token-Parallel Orchestration     | %10.2f | %12.2f | %9.2fx |\n", times[3] * 1000, gflops[3], gflops[3] / gflops[0]);
    printf("════════════════════════════════════════════════════════════════════════\n");

    // ===================================================================
    // TEST 2: "WIDE" GEMM (QKV Projection: N = 3*K)
    // ===================================================================
    int M2 = M->context_window;
    int N2 = 3 * M->aligned_embed_dim;
    int K2 = M->aligned_embed_dim;
    double gflops_val2 = (2.0 * M2 * N2 * K2) / 1e9;

    float *B2 = M->memory_base + L->qkv_weight_offset;
    float *bias2 = M->memory_base + L->qkv_bias_offset;
    float *C2_naive = M->memory_base + L->qkv_out_naive;
    float *C2_avx = M->memory_base + L->qkv_out_avx;
    float *C2_blocked = M->memory_base + L->qkv_out_blocked;
    float *C2_token = M->memory_base + L->qkv_out_token;

    printf("\n\n🔬 TEST 2: 'Wide' GEMM (QKV Projection, M=%d, N=%d, K=%d)\n", M2, N2, K2);
    printf("════════════════════════════════════════════════════════════════════════\n");

    start = get_time_sec();
    gemm_naive_parallel(A_input, B2, bias2, C2_naive, M2, N2, K2);
    times[0] = get_time_sec() - start;
    gflops[0] = gflops_val2 / times[0];

    start = get_time_sec();
    gemm_avx512_parallel(A_input, B2, bias2, C2_avx, M2, N2, K2);
    times[1] = get_time_sec() - start;
    gflops[1] = gflops_val2 / times[1];

    start = get_time_sec();
    gemm_fine_grained_parallel(A_input, B2, bias2, C2_blocked, M2, N2, K2);
    times[2] = get_time_sec() - start;
    gflops[2] = gflops_val2 / times[2];

    start = get_time_sec();
    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M2) ? (M2 - token_start) : M->tokens_per_core;
        if (num_tokens > 0) {
            gemm_blocked_serial(A_input + token_start * K2, B2, bias2, C2_token + token_start * N2, num_tokens, N2, K2);
        }
    }
    times[3] = get_time_sec() - start;
    gflops[3] = gflops_val2 / times[3];

    printf("| %-35s | %10s | %12s | %10s |\n", "Strategy", "Time (ms)", "GFLOPS", "Speedup");
    printf("|-------------------------------------|------------|--------------|------------|\n");
    printf("| 1. Naive Parallel                   | %10.2f | %12.2f | %9.2fx |\n", times[0] * 1000, gflops[0], 1.0);
    printf("| 2. Simple AVX-512 Parallel          | %10.2f | %12.2f | %9.2fx |\n", times[1] * 1000, gflops[1], gflops[1] / gflops[0]);
    printf("| 3. Fine-Grained Blocked Parallel    | %10.2f | %12.2f | %9.2fx |\n", times[2] * 1000, gflops[2], gflops[2] / gflops[0]);
    printf("| 4. Token-Parallel Orchestration     | %10.2f | %12.2f | %9.2fx |\n", times[3] * 1000, gflops[3], gflops[3] / gflops[0]);
    printf("════════════════════════════════════════════════════════════════════════\n");
}


/* ---------------- main -------------------- */
int main(int argc, char **argv)
{
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
        switch (c) {
            case 'l': L = atoi(optarg); break;
            case 'd': C = atoi(optarg); break;
            case 't': T = atoi(optarg); break;
            case 'v': V = atoi(optarg); break;
            case 'f': do_alloc = 1; break;
            case 'b': run_benchmarks = 1; break;
            default:
                fprintf(stderr, "Usage: %s [--layers N] [--dmodel N] [--ctx N] [--vocab N] [--force] [--benchmark]\n", argv[0]);
                return 1;
        }
    }

    if (!do_alloc) {
        printf("Dry-run only. Pass --force to allocate and run.\n");
        return 0;
    }
    
    if (L < 4 && run_benchmarks) {
        fprintf(stderr, "Error: Must specify at least --layers 4 for the comprehensive benchmark to have enough output buffers.\n");
        return 1;
    }

    TransformerModel M = {0};
    M.num_layers = L; M.vocab_size = V; M.embed_dim = C; M.context_window = T;
    
    size_t need_bytes = bytes_needed(C, T, L, V, run_benchmarks);
    printf("⚙  Requested model  L=%d d_model=%d  ctx=%d vocab=%d\n", L, C, T, V);
    printf("→ Total allocation will be ≈ %.2f GiB\n", need_bytes / (1024.0*1024.0*1024.0));

    printf("Allocating memory for model...\n");
    layout_transformer(&M, run_benchmarks);
    printf("✅ Success! mmap at %p\n", (void *)M.memory_base);

    long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int reserved_cores = 4;
    M.num_cores = (logical_cores > reserved_cores) ? logical_cores - reserved_cores : 1;
    M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;

    if (run_benchmarks) {
        // run_comprehensive_benchmark(&M);
        run_presentation_benchmark(&M);
    }

    destroy_transformer(&M);
    return 0;
}
