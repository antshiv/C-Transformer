/***********************************************************************
 * DYNAMIC SHAPE & STRATEGY BENCHMARK (PURE C)
 * ---------------------------------------------------------------
 * • Empirically proves that the optimal parallel strategy depends
 * on the shape of the matrix operation.
 * • Runs a "Squarish" (MLP-like) and a "Wide" (QKV-like) test.
 * • Compares all four optimization strategies for each shape.
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
     // Buffers for Layer 0
     size_t layer_input_offset;
     
     // Buffers for "Wide" QKV test
     size_t qkv_weight_offset, qkv_bias_offset;
     size_t qkv_out_naive, qkv_out_avx, qkv_out_blocked, qkv_out_token;
 
     // Buffers for "Squarish" MLP test
     size_t mlp_weight_offset, mlp_bias_offset;
     size_t mlp_out_naive, mlp_out_avx, mlp_out_blocked, mlp_out_token;
 
 } BenchmarkLayout;
 
 typedef struct
 {
     int num_layers, vocab_size, embed_dim, context_window;
     size_t aligned_embed_dim;
     int num_cores;
     int tokens_per_core;
     float *memory_base;
     size_t total_floats;
     BenchmarkLayout *layout;
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
 void setup_benchmark_memory(TransformerModel *M)
 {
     size_t off = 0;
     M->aligned_embed_dim = align_up(M->embed_dim, CACHE_ALIGN / sizeof(float));
     
     M->layout = malloc(sizeof(BenchmarkLayout));
     if (!M->layout) { perror("malloc layout"); exit(EXIT_FAILURE); }
 
     BenchmarkLayout *L = M->layout;
     size_t C = M->aligned_embed_dim;
     size_t T = M->context_window;
 
     // Allocate all necessary buffers for both tests
     L->layer_input_offset = bump(&off, T * C, CACHE_ALIGN);
     
     // Buffers for "Wide" QKV test
     L->qkv_weight_offset = bump(&off, 3 * C * C, CACHE_ALIGN);
     L->qkv_bias_offset   = bump(&off, 3 * C, CACHE_ALIGN);
     L->qkv_out_naive     = bump(&off, 3 * T * C, CACHE_ALIGN);
     L->qkv_out_avx       = bump(&off, 3 * T * C, CACHE_ALIGN);
     L->qkv_out_blocked   = bump(&off, 3 * T * C, CACHE_ALIGN);
     L->qkv_out_token     = bump(&off, 3 * T * C, CACHE_ALIGN);
 
     // Buffers for "Squarish" MLP test
     L->mlp_weight_offset = bump(&off, C * C, CACHE_ALIGN);
     L->mlp_bias_offset   = bump(&off, C, CACHE_ALIGN);
     L->mlp_out_naive     = bump(&off, T * C, CACHE_ALIGN);
     L->mlp_out_avx       = bump(&off, T * C, CACHE_ALIGN);
     L->mlp_out_blocked   = bump(&off, T * C, CACHE_ALIGN);
     L->mlp_out_token     = bump(&off, T * C, CACHE_ALIGN);
 
     M->total_floats = off;
     M->memory_base = huge_alloc(off * sizeof(float));
 }
 
 void destroy_benchmark_memory(TransformerModel *M)
 {
     munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
     free(M->layout);
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
 
 void run_presentation_benchmark(TransformerModel *M) {
     printf("\n🚀 Comprehensive GEMM Strategy Benchmark\n");
     printf("   Comparing strategies across different GEMM shapes.\n");
     
     // --- 1. Prepare Data and Buffers ---
     BenchmarkLayout *L = M->layout;
     float *A_input = M->memory_base + L->layer_input_offset;
 
     srand(42);
     for (size_t i = 0; i < (size_t)M->context_window * M->aligned_embed_dim; i++) A_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
 
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
     int L = 1, V = 32768, C = 128, T = 128; 
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
     
     printf("⚙  Requested model  d_model=%d  ctx=%d\n", C, T);
 
     printf("Allocating memory for model...\n");
     setup_benchmark_memory(&M);
     printf("✅ Success! mmap at %p\n", (void *)M.memory_base);
 
     long logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
     int reserved_cores = 4;
     M.num_cores = (logical_cores > reserved_cores) ? logical_cores - reserved_cores : 1;
     M.tokens_per_core = (M.context_window + M.num_cores - 1) / M.num_cores;
 
     if (run_benchmarks) {
         run_presentation_benchmark(&M);
     }
 
     destroy_benchmark_memory(&M);
     return 0;
 }
 