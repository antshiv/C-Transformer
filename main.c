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

// KERNEL 2: Serial Blocked GEMM (for Token-Parallel Orchestrator)
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

    // --- TEST 1: "SQUARISH" GEMM (Simulating MLP layer) ---
    printf("\n🔬 TEST 1: 'Squarish' GEMM (like an MLP layer)\n");
    int M1 = M->context_window;
    int N1 = M->aligned_embed_dim;
    int K1 = M->aligned_embed_dim;
    printf("   Dimensions: M=%d, N=%d, K=%d\n", M1, N1, K1);
    
    float *A1 = M->memory_base + M->layers[0].layer_input_offset;
    float *B1 = M->memory_base + M->layers[0].mlp_weight_offset;
    float *bias1 = M->memory_base + M->layers[0].mlp_bias_offset;
    float *C1_fine = M->memory_base + M->layers[0].mlp_output_offset;
    float *C1_token = M->memory_base + M->layers[1].layer_input_offset; // Repurpose

    double time1_fine = get_time_sec();
    gemm_fine_grained_parallel(A1, B1, bias1, C1_fine, M1, N1, K1);
    time1_fine = get_time_sec() - time1_fine;

    double time1_token = get_time_sec();
    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M1) ? (M1 - token_start) : M->tokens_per_core;
        if (num_tokens > 0) {
            gemm_blocked_serial(A1 + token_start * K1, B1, bias1, C1_token + token_start * N1, num_tokens, N1, K1);
        }
    }
    time1_token = get_time_sec() - time1_token;

    printf("   - Fine-Grained Strategy:  %.2f ms\n", time1_fine * 1000);
    printf("   - Token-Parallel Strategy: %.2f ms\n", time1_token * 1000);
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
    printf("   Dimensions: M=%d, N=%d, K=%d\n", M2, N2, K2);

    float *A2 = M->memory_base + M->layers[0].layer_input_offset;
    float *B2 = M->memory_base + M->layers[0].qkv_weight_offset;
    float *bias2 = M->memory_base + M->layers[0].qkv_bias_offset;
    float *C2_fine = M->memory_base + M->layers[0].qkv_output_offset;
    float *C2_token = M->memory_base + M->layers[1].qkv_output_offset; // Repurpose

    double time2_fine = get_time_sec();
    gemm_fine_grained_parallel(A2, B2, bias2, C2_fine, M2, N2, K2);
    time2_fine = get_time_sec() - time2_fine;

    double time2_token = get_time_sec();
    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M2) ? (M2 - token_start) : M->tokens_per_core;
        if (num_tokens > 0) {
            gemm_blocked_serial(A2 + token_start * K2, B2, bias2, C2_token + token_start * N2, num_tokens, N2, K2);
        }
    }
    time2_token = get_time_sec() - time2_token;

    printf("   - Fine-Grained Strategy:  %.2f ms\n", time2_fine * 1000);
    printf("   - Token-Parallel Strategy: %.2f ms\n", time2_token * 1000);
    if (time2_fine < time2_token) {
        printf("   🏆 WINNER: Fine-Grained is %.2fx faster. Best for shared cache reuse.\n", time2_token / time2_fine);
    } else {
        printf("   🏆 WINNER: Token-Parallel is %.2fx faster. Best for input data locality.\n", time2_fine / time2_token);
    }
    printf("════════════════════════════════════════════════════════════════════════\n");
}


/* ---------------- main -------------------- */
int main(int argc, char **argv)
{
    int L = 2, V = 32768, C = 128, T = 128; // Default to 2 layers for benchmark buffers
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

    TransformerModel M = {0};
    M.num_layers = L; M.vocab_size = V; M.embed_dim = C; M.context_window = T;

    printf("Allocating memory for model...\n");
    layout_transformer(&M);
    printf("✅ Success! mmap at %p\n", (void *)M.memory_base);

    long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int reserved_cores = 4;
    M.num_cores = (logical_cores > reserved_cores) ? logical_cores - reserved_cores : 1;
    M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;

    if (run_benchmarks) {
        run_dynamic_benchmark(&M);
    }

    destroy_transformer(&M);
    return 0;
}
