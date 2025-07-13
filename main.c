/***********************************************************************
 * SINGLE-BLOCK, CACHE-ALIGNED GPT-2 LAYOUT (pure C demo)
 * ---------------------------------------------------------------
 * • one huge allocation for ALL weights + activations
 * • 64-byte alignment for every tensor
 * • 2 MB huge-page backing for minimal TLB misses
 * • bump() = zero-fragmentation offset math
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
    if (p != MAP_FAILED)
        return p; /* explicit huge page ok */

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
    int num_cores;         // usable compute cores for model
    int tokens_per_core;   // slice of context_window each core owns
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
    
    if (M->num_layers > 1) {
        M->layer_stride = M->layers[1].ln1_weight_offset - M->layers[0].ln1_weight_offset;
    } else {
        M->layer_stride = off - M->layers_start_offset;
    }


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

static size_t bytes_needed(int layers, int vocab, int d_model, int ctx)
{
    size_t C = align_up(d_model, CACHE_ALIGN / sizeof(float));
    size_t T = ctx;
    size_t V = vocab;
    size_t perL = (2 * C) + (2 * T * C) + (3 * C * C + 3 * C + 3 * T * C) + (C * C + C + 2 * T * C) + (2 * C + T * C) + (8 * C * C + 5 * C) + (2 * T * C);
    size_t total_floats = (V * C) + (T * C) + (T * C) + (size_t)layers * perL + (2 * C);
    return total_floats * sizeof(float);
}

/* ─── SLICE HELPERS ────────────────────────────────────────────────── */
static inline float *get_slice(TransformerModel *M, int core_id,
                               size_t base_offset, size_t vector_dim)
{
    size_t token_start = core_id * M->tokens_per_core;
    if (token_start >= M->context_window) return NULL;
    return M->memory_base + base_offset + token_start * vector_dim;
}


// ============================================================================
//  GEMM KERNELS AND BENCHMARKING FRAMEWORK
// ============================================================================

// --- KERNEL 1: Naive GEMM (for correctness check) ---
void gemm_1d_naive_parallel(float *A, float *B, float *C, int M, int N, int K)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k]; // simulate B^T
            }
            C[i * N + j] = sum;
        }
    }
}

// --- KERNEL 2: Blocked AVX-512 GEMM (Parallel Version for Generic Benchmark) ---
void gemm_1d_blocked_parallel(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    const int block_size = 64;
    #pragma omp parallel for collapse(2)
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
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_load_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B[j * K + k]);
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

// --- KERNEL 3: Blocked AVX-512 GEMM (SERIAL Version for Token-Parallel Orchestrator) ---
void gemm_1d_blocked_serial(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    const int block_size = 64; // Tune based on cache size
    // NOTE: No OpenMP pragmas here. This is a single-threaded kernel.
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
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_load_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B[j * K + k]);
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
// TOKEN-PARALLEL ORCHESTRATION BENCHMARK
// ============================================================================

typedef struct {
    int core_id;
    int tokens_assigned;
    double computation_time_sec;
} CoreMetrics;

void benchmark_token_parallel_orchestration(TransformerModel *M) {
    printf("\n🎯 Testing Token-Parallel Orchestration with SERIAL Blocked Kernel\n");
    
    CoreMetrics* all_metrics = calloc(M->num_cores, sizeof(CoreMetrics));

    // Initialize test data
    TrulyOptimalLayer *L = &M->layers[0];
    float *input_base = M->memory_base + L->layer_input_offset;
    float *weights = M->memory_base + L->qkv_weight_offset; 
    float *bias = M->memory_base + L->qkv_bias_offset;
    
    srand(42);
    for (int i = 0; i < M->context_window * M->aligned_embed_dim; i++) input_base[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < 3 * M->aligned_embed_dim * M->aligned_embed_dim; i++) weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (int i = 0; i < 3 * M->aligned_embed_dim; i++) bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

    printf("🧠 Config: %d cores, %d tokens/core, Embed=%zu, QKV_Out=%zu\n", M->num_cores, M->tokens_per_core, M->aligned_embed_dim, 3 * M->aligned_embed_dim);

    double total_start = get_time_sec();

    #pragma omp parallel num_threads(M->num_cores)
    {
        int core_id = omp_get_thread_num();
        all_metrics[core_id].core_id = core_id;

        // Calculate this core's token slice
        int token_start = core_id * M->tokens_per_core;
        int num_tokens = (token_start + M->tokens_per_core > M->context_window) ? (M->context_window - token_start) : M->tokens_per_core;
        all_metrics[core_id].tokens_assigned = num_tokens;

        if (num_tokens > 0) {
            float *input_slice = M->memory_base + L->layer_input_offset + token_start * M->aligned_embed_dim;
            float *output_slice = M->memory_base + L->qkv_output_offset + token_start * (3 * M->aligned_embed_dim);

            double core_start = get_time_sec();

            // *** THE FIX: Call the SERIAL kernel from the PARALLEL orchestrator ***
            gemm_1d_blocked_serial(
                input_slice,
                weights,
                bias,
                output_slice,
                num_tokens,
                3 * M->aligned_embed_dim,
                M->aligned_embed_dim
            );

            double core_end = get_time_sec();
            all_metrics[core_id].computation_time_sec = core_end - core_start;
        }
    }

    double total_end = get_time_sec();
    double total_time = total_end - total_start;

    // --- Analysis ---
    printf("\n📊 Performance Analysis:\n");
    double slowest_core = 0;
    for (int c = 0; c < M->num_cores; c++) {
        if (all_metrics[c].computation_time_sec > slowest_core) {
            slowest_core = all_metrics[c].computation_time_sec;
        }
    }

    double gflops = (2.0 * M->context_window * (3 * M->aligned_embed_dim) * M->aligned_embed_dim) / (total_time * 1e9);
    
    printf("   Total Time: %.3f ms\n", total_time * 1000);
    printf("   Slowest Core Time: %.3f ms\n", slowest_core * 1000);
    printf("   Aggregate Performance: %.2f GFLOPS\n", gflops);
    printf("   Aggregate Throughput: %.1f tokens/sec\n", M->context_window / total_time);
    
    free(all_metrics);
}


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
    while ((c = getopt_long(argc, argv, "l:d:t:v:fb", long_opts, NULL)) != -1)
    {
        switch (c)
        {
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
    M.num_attention_heads = M.aligned_embed_dim / 64; // assume head_dim = 64

    // Only run benchmarks if requested
    if (do_alloc && run_benchmarks)
    {
        // This now calls the new, focused benchmark
        benchmark_token_parallel_orchestration(&M);
    }

    printf("\n🧠 Detected %ld logical cores → reserving %d for OS → using %d for model\n",
           logical_cores, reserved_cores, M.num_cores);
    printf("📦 Each core will handle ≈ %d tokens from context window of %d tokens\n",
           M.tokens_per_core, M.context_window);
    printf("🧠 Attention heads = %d (assuming head_dim=64)\n", M.num_attention_heads);

    destroy_transformer(&M);
    return 0;
}
