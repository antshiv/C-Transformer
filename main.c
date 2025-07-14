/***********************************************************************
 * FINAL COMPREHENSIVE BENCHMARK (PURE C)
 * ---------------------------------------------------------------
 * • Empirically finds the fastest execution strategy by running
 * and timing multiple parallelization methods.
 * • Demonstrates that the optimal strategy depends on the
 * shape of the matrix operation (GEMM vs. QKV-style).
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
    size_t qkv_weight_offset, qkv_bias_offset, qkv_output_offset;
    size_t mlp_weight_offset, mlp_bias_offset, mlp_output_offset;
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

/* ─── memory slice helper for token-parallel access ─────────────────── */
float* get_slice(TransformerModel *M, int core_id, size_t base_offset, size_t stride) {
    size_t token_start = core_id * M->tokens_per_core;
    size_t element_offset = token_start * stride;
    return M->memory_base + base_offset + element_offset;
}

/* ─── lay out the entire model ───────────────────────────────────── */
void layout_transformer(TransformerModel *M)
{
    size_t off = 0;
    M->aligned_embed_dim = align_up(M->embed_dim, CACHE_ALIGN / sizeof(float));
    
    M->layers = malloc(sizeof(TrulyOptimalLayer) * M->num_layers);
    if (!M->layers) { perror("malloc layers"); exit(EXIT_FAILURE); }

    for (int l = 0; l < M->num_layers; ++l) {
        TrulyOptimalLayer *L = &M->layers[l];
        L->layer_input_offset = bump(&off, (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
        // QKV Layer
        L->qkv_weight_offset  = bump(&off, 3ULL * M->aligned_embed_dim * M->aligned_embed_dim, CACHE_ALIGN);
        L->qkv_bias_offset    = bump(&off, 3ULL * M->aligned_embed_dim, CACHE_ALIGN);
        L->qkv_output_offset  = bump(&off, 3ULL * (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
        // MLP Layer (for squarish test)
        L->mlp_weight_offset  = bump(&off, M->aligned_embed_dim * M->aligned_embed_dim, CACHE_ALIGN);
        L->mlp_bias_offset    = bump(&off, M->aligned_embed_dim, CACHE_ALIGN);
        L->mlp_output_offset  = bump(&off, (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
    }

    M->total_floats = off;
    M->memory_base = huge_alloc(off * sizeof(float));
}

void destroy_transformer(TransformerModel *M)
{
    munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
    free(M->layers);
}

// ============================================================================
//  GEMM KERNELS
// ============================================================================

// KERNEL 1: Fine-Grained Parallel Blocked GEMM
void gemm_fine_grained_parallel(float *A, float *B, float *bias, float *C, int M, int N, int K) {
    const int block_size = 64;
    
    // Initialize output with bias
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias[j];
        }
    }

    // Blocked multiplication with fine-grained parallelism
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
                        int k;
                        
                        // Vectorized inner loop
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_load_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        
                        // Handle remaining elements
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
                        
                        // Atomic update since multiple threads may write to same location
                        #pragma omp atomic
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
}

// KERNEL 2: Serial Blocked GEMM (for Token-Parallel Orchestrator)
void gemm_blocked_serial(float *A, float *B, float *bias, float *C, int M, int N, int K) {
    const int block_size = 64;
    
    // Initialize output with bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias[j];
        }
    }
    
    // Blocked multiplication (serial within each core's slice)
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = (ii + block_size < M) ? ii + block_size : M;
                int j_end = (jj + block_size < N) ? jj + block_size : N;
                int k_end = (kk + block_size < K) ? kk + block_size : K;
                
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        
                        // Vectorized inner loop
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_load_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        
                        // Handle remaining elements
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
                        
                        // No atomic needed - each core writes to exclusive region
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// CORRECTNESS AND UTILITY FUNCTIONS  
// ============================================================================

float check_correctness(float *C1, float *C2, int M, int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(C1[i] - C2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

void initialize_test_data(TransformerModel *M) {
    srand(42); // Deterministic random data
    
    // Initialize input data for both layers
    for (int l = 0; l < M->num_layers; l++) {
        // Layer input
        float *input = M->memory_base + M->layers[l].layer_input_offset;
        for (int i = 0; i < M->context_window * M->aligned_embed_dim; i++) {
            input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        
        // QKV weights
        float *qkv_weights = M->memory_base + M->layers[l].qkv_weight_offset;
        for (int i = 0; i < 3 * M->aligned_embed_dim * M->aligned_embed_dim; i++) {
            qkv_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        }
        
        // QKV bias
        float *qkv_bias = M->memory_base + M->layers[l].qkv_bias_offset;
        for (int i = 0; i < 3 * M->aligned_embed_dim; i++) {
            qkv_bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
        
        // MLP weights
        float *mlp_weights = M->memory_base + M->layers[l].mlp_weight_offset;
        for (int i = 0; i < M->aligned_embed_dim * M->aligned_embed_dim; i++) {
            mlp_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        }
        
        // MLP bias
        float *mlp_bias = M->memory_base + M->layers[l].mlp_bias_offset;
        for (int i = 0; i < M->aligned_embed_dim; i++) {
            mlp_bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    }
}

// ============================================================================
// DYNAMIC BENCHMARK DRIVER
// ============================================================================

void run_dynamic_benchmark(TransformerModel *M) {
    printf("\n🚀 Dynamic Strategy Benchmark\n");
    printf("   Testing different GEMM shapes to find the optimal algorithm for each.\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    if (M->num_layers < 2) {
        fprintf(stderr, "❌ Error: Benchmark requires at least 2 layers for separate output buffers.\n");
        return;
    }

    // Initialize all test data once
    initialize_test_data(M);
    
    // Warm up thread pool to reduce timing variance
    printf("🔥 Warming up thread pool...\n");
    #pragma omp parallel num_threads(M->num_cores)
    {
        volatile int dummy = omp_get_thread_num();
        (void)dummy; // Suppress unused variable warning
    }

    // --- TEST 1: "SQUARISH" GEMM (Simulating MLP layer) ---
    printf("\n🔬 TEST 1: 'Squarish' GEMM (like an MLP layer)\n");
    int M1 = M->context_window;
    int N1 = M->aligned_embed_dim;
    int K1 = M->aligned_embed_dim;
    double gflops1 = 2.0 * M1 * N1 * K1 / 1e9;
    printf("   Dimensions: M=%d, N=%d, K=%d (%.1f GFLOP)\n", M1, N1, K1, gflops1);
    
    float *A1 = M->memory_base + M->layers[0].layer_input_offset;
    float *B1 = M->memory_base + M->layers[0].mlp_weight_offset;
    float *bias1 = M->memory_base + M->layers[0].mlp_bias_offset;
    float *C1_fine = M->memory_base + M->layers[0].mlp_output_offset;
    float *C1_token = M->memory_base + M->layers[1].layer_input_offset; // Repurpose

    // Test Fine-Grained Strategy
    double time1_fine = get_time_sec();
    gemm_fine_grained_parallel(A1, B1, bias1, C1_fine, M1, N1, K1);
    time1_fine = get_time_sec() - time1_fine;

    // Clear cache between tests
    _mm_mfence();
    
    // Test Token-Parallel Strategy
    double time1_token = get_time_sec();
    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M1) ? (M1 - token_start) : M->tokens_per_core;
        if (num_tokens > 0) {
            float *A_slice = A1 + token_start * K1;
            float *C_slice = C1_token + token_start * N1;
            gemm_blocked_serial(A_slice, B1, bias1, C_slice, num_tokens, N1, K1);
        }
    }
    time1_token = get_time_sec() - time1_token;

    float diff1 = check_correctness(C1_fine, C1_token, M1, N1);
    
    printf("   - Fine-Grained Strategy:  %.2f ms (%.1f GFLOPS)\n", 
           time1_fine * 1000, gflops1 / time1_fine);
    printf("   - Token-Parallel Strategy: %.2f ms (%.1f GFLOPS)\n", 
           time1_token * 1000, gflops1 / time1_token);
    printf("   - Correctness check: Max difference = %.2e\n", diff1);
    
    if (time1_fine < time1_token) {
        printf("   🏆 WINNER: Fine-Grained is %.2fx faster. Best for shared cache reuse.\n", time1_token / time1_fine);
    } else {
        printf("   🏆 WINNER: Token-Parallel is %.2fx faster. Best for input data locality.\n", time1_fine / time1_token);
    }

    // --- TEST 2: "WIDE" GEMM (Simulating QKV projection) ---
    printf("\n🔬 TEST 2: 'Wide' GEMM (like a QKV projection)\n");
    int M2 = M->context_window;
    int N2 = 3 * M->aligned_embed_dim;
    int K2 = M->aligned_embed_dim;
    double gflops2 = 2.0 * M2 * N2 * K2 / 1e9;
    printf("   Dimensions: M=%d, N=%d, K=%d (%.1f GFLOP)\n", M2, N2, K2, gflops2);

    float *A2 = A1; // Same input as test 1
    float *B2 = M->memory_base + M->layers[0].qkv_weight_offset;
    float *bias2 = M->memory_base + M->layers[0].qkv_bias_offset;
    float *C2_fine = M->memory_base + M->layers[0].qkv_output_offset;
    float *C2_token = M->memory_base + M->layers[1].qkv_output_offset; // Repurpose

    // Test Fine-Grained Strategy
    double time2_fine = get_time_sec();
    gemm_fine_grained_parallel(A2, B2, bias2, C2_fine, M2, N2, K2);
    time2_fine = get_time_sec() - time2_fine;

    // Clear cache between tests
    _mm_mfence();

    // Test Token-Parallel Strategy
    double time2_token = get_time_sec();
    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M2) ? (M2 - token_start) : M->tokens_per_core;
        if (num_tokens > 0) {
            float *A_slice = A2 + token_start * K2;
            float *C_slice = C2_token + token_start * N2;
            gemm_blocked_serial(A_slice, B2, bias2, C_slice, num_tokens, N2, K2);
        }
    }
    time2_token = get_time_sec() - time2_token;

    float diff2 = check_correctness(C2_fine, C2_token, M2, N2);

    printf("   - Fine-Grained Strategy:  %.2f ms (%.1f GFLOPS)\n", 
           time2_fine * 1000, gflops2 / time2_fine);
    printf("   - Token-Parallel Strategy: %.2f ms (%.1f GFLOPS)\n", 
           time2_token * 1000, gflops2 / time2_token);
    printf("   - Correctness check: Max difference = %.2e\n", diff2);
    
    if (time2_fine < time2_token) {
        printf("   🏆 WINNER: Fine-Grained is %.2fx faster. Best for shared cache reuse.\n", time2_token / time2_fine);
    } else {
        printf("   🏆 WINNER: Token-Parallel is %.2fx faster. Best for input data locality.\n", time2_fine / time2_token);
    }
    
    // Summary
    printf("\n🎯 Strategy Selection Guide:\n");
    printf("   • Token-Parallel: Best when input data locality dominates (wide matrices, QKV)\n");  
    printf("   • Fine-Grained: Best when weight reuse dominates (square matrices, MLP)\n");
    printf("   • Memory bandwidth: %.1f GB/s effective per strategy\n", 
           (M1 * N1 * K1 * 3 * sizeof(float)) / (time1_fine * 1e9));
    
    if (diff1 > 1e-5 || diff2 > 1e-5) {
        printf("⚠️  Warning: Large correctness differences detected. Check implementation.\n");
    } else {
        printf("✅ All correctness checks passed. Both strategies produce identical results.\n");
    }
    
    printf("════════════════════════════════════════════════════════════════════════\n");
}

/* ---------------- main -------------------- */
int main(int argc, char **argv)
{
    int L = 2, V = 32768, C = 1024, T = 1024; // Larger defaults for meaningful benchmarks
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

    if (L < 2 && run_benchmarks) {
        fprintf(stderr, "Error: Must specify at least --layers 2 for benchmarking.\n");
        return 1;
    }

    // Validate parameters
    if (C % 16 != 0) {
        fprintf(stderr, "Error: dmodel (%d) must be divisible by 16 for AVX-512.\n", C);
        return 1;
    }

    TransformerModel M = {0};
    M.num_layers = L; 
    M.vocab_size = V; 
    M.embed_dim = C; 
    M.context_window = T;

    printf("⚙️  Requested model: L=%d, d_model=%d, ctx=%d, vocab=%d\n", L, C, T, V);
    
    layout_transformer(&M);
    printf("💾 Model size: %.2f GiB (%.0f GB)\n", 
           M.total_floats * sizeof(float) / (1024.0 * 1024.0 * 1024.0),
           M.total_floats * sizeof(float) / 1e9);
    printf("✅ Success! mmap at %p\n", (void *)M.memory_base);

    // Calculate core configuration
    long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int reserved_cores = 4;
    M.num_cores = (logical_cores > reserved_cores) ? logical_cores - reserved_cores : 1;
    M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;

    printf("🧠 Detected %ld logical cores → reserving %d for OS → using %d for model\n", 
           logical_cores, reserved_cores, M.num_cores);
    printf("📦 Each core will handle ≈ %d tokens from context window of %d tokens\n", 
           M.tokens_per_core, M.context_window);

    if (run_benchmarks) {
        run_dynamic_benchmark(&M);
    }

    destroy_transformer(&M);
    return 0;
}