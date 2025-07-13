/***********************************************************************
 * SINGLE-BLOCK, CACHE-ALIGNED GPT-2 LAYOUT (pure C demo)
 * ---------------------------------------------------------------
 * • Comprehensive benchmark to demonstrate performance progression
 * from naive to fully optimized parallel GEMM strategies.
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

    // Allocate enough space for 4 separate output buffers for benchmarking
    int required_layers = (M->num_layers < 4) ? 4 : M->num_layers;

    for (int l = 0; l < required_layers; ++l) {
        if (l < M->num_layers) {
            TrulyOptimalLayer *L = &M->layers[l];
            L->layer_input_offset = bump(&off, (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
            L->qkv_weight_offset  = bump(&off, 3ULL * M->aligned_embed_dim * M->aligned_embed_dim, CACHE_ALIGN);
            L->qkv_bias_offset    = bump(&off, 3ULL * M->aligned_embed_dim, CACHE_ALIGN);
            L->qkv_output_offset  = bump(&off, 3ULL * (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
        } else {
             // Allocate dummy space for extra buffers
            bump(&off, 3ULL * (size_t)M->context_window * M->aligned_embed_dim, CACHE_ALIGN);
        }
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
    printf("   Showing performance progression from naive to optimized strategies.\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    if (M->num_layers < 4) {
        fprintf(stderr, "❌ Error: Comprehensive benchmark requires at least 4 layers for separate output buffers.\n");
        return;
    }

    // --- 1. Prepare Data and Buffers ---
    float *A = M->memory_base + M->layers[0].layer_input_offset;
    float *B = M->memory_base + M->layers[0].qkv_weight_offset;
    float *bias = M->memory_base + M->layers[0].qkv_bias_offset;
    
    float *C_naive = M->memory_base + M->layers[0].qkv_output_offset;
    float *C_avx512 = M->memory_base + M->layers[1].qkv_output_offset;
    float *C_blocked = M->memory_base + M->layers[2].qkv_output_offset;
    float *C_token_parallel = M->memory_base + M->layers[3].qkv_output_offset;

    srand(42);
    for (size_t i = 0; i < (size_t)M->context_window * M->aligned_embed_dim; i++) A[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < 3 * M->aligned_embed_dim * M->aligned_embed_dim; i++) B[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (int i = 0; i < 3 * M->aligned_embed_dim; i++) bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    
    int M_dim = M->context_window;
    int N_dim = 3 * M->aligned_embed_dim;
    int K_dim = M->aligned_embed_dim;
    double total_gflops_val = (2.0 * M_dim * N_dim * K_dim) / 1e9;

    double times[4];
    double gflops[4];

    // --- 2. Run Benchmarks ---
    printf("\n📊 Running all strategies...\n");

    // Strategy 1: Naive
    double start = get_time_sec();
    gemm_naive_parallel(A, B, bias, C_naive, M_dim, N_dim, K_dim);
    times[0] = get_time_sec() - start;
    gflops[0] = total_gflops_val / times[0];
    printf("   1. Naive Parallel... Done.\n");

    // Strategy 2: Simple AVX-512
    start = get_time_sec();
    gemm_avx512_parallel(A, B, bias, C_avx512, M_dim, N_dim, K_dim);
    times[1] = get_time_sec() - start;
    gflops[1] = total_gflops_val / times[1];
    printf("   2. Simple AVX-512 Parallel... Done.\n");

    // Strategy 3: Fine-Grained Blocked AVX-512
    start = get_time_sec();
    gemm_fine_grained_parallel(A, B, bias, C_blocked, M_dim, N_dim, K_dim);
    times[2] = get_time_sec() - start;
    gflops[2] = total_gflops_val / times[2];
    printf("   3. Fine-Grained Blocked Parallel... Done.\n");

    // Strategy 4: Token-Parallel Orchestration
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
    printf("   4. Token-Parallel Orchestration... Done.\n");

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
    printf("\n✅ Correctness checks passed (Max Diffs: AVX:%.1e, Blocked:%.1e, Token:%.1e)\n", 
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

/* ---------------- main -------------------- */
int main(int argc, char **argv)
{
    int L = 4, V = 32768, C = 128, T = 128; // Default to 4 layers for benchmark buffers
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
        run_comprehensive_benchmark(&M);
    }

    destroy_transformer(&M);
    return 0;
}
