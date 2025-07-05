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
#include <sys/mman.h>      /* MAP_HUGETLB / munmap */
#include <unistd.h>

/* ─── alignment targets ───────────────────────────────────────────── */
#define CACHE_ALIGN   64ULL
#define HUGE_ALIGN    (2ULL * 1024 * 1024)   /* 2 MB huge page */

/* ─── tiny helpers ────────────────────────────────────────────────── */
static inline size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

/* best-effort huge-page allocator (falls back to THP) */
static void *huge_alloc(size_t bytes)
{
    size_t len = align_up(bytes, HUGE_ALIGN);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) return p;                    /* explicit huge page ok  */

    /* fallback: page-aligned malloc + THP */
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
    size_t qkv_weight_offset,  qkv_bias_offset,  qkv_output_offset;
    size_t proj_weight_offset, proj_bias_offset;
    size_t attention_output_offset, residual1_output_offset;
    size_t ln2_weight_offset, ln2_bias_offset, ln2_output_offset;
    size_t fc1_weight_offset, fc1_bias_offset;
    size_t fc2_weight_offset, fc2_bias_offset;
    size_t mlp_output_offset,  residual2_output_offset;
} TrulyOptimalLayer;

typedef struct {
    /* hyper-parameters */
    int num_layers, vocab_size, embed_dim, context_window;

    /* single block */
    float  *memory_base;
    size_t  total_floats;
    size_t  layer_stride;

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
    *off = align_up(*off, alignB / sizeof(float));         /* align in floats */
    size_t here = *off;
    *off += count;
    return here;
}

/* ─── lay out the entire model ───────────────────────────────────── */
void layout_transformer(TransformerModel *M)
{
    size_t off = 0;                                        /* cursor in floats */

    /* embeddings */
    M->token_emb_offset      = bump(&off, (size_t)M->vocab_size   * M->embed_dim, CACHE_ALIGN);
    M->pos_emb_offset        = bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);
    M->embedded_input_offset = bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);

    /* per-layer */
    M->layers_start_offset = off;
    M->layers = malloc(sizeof(TrulyOptimalLayer) * M->num_layers);
    if (!M->layers) { perror("malloc layers"); exit(EXIT_FAILURE); }

    for (int l = 0; l < M->num_layers; ++l) {
        TrulyOptimalLayer *L = &M->layers[l];

        L->ln1_weight_offset  = bump(&off, M->embed_dim, CACHE_ALIGN);
        L->ln1_bias_offset    = bump(&off, M->embed_dim, CACHE_ALIGN);

        L->layer_input_offset = bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);
        L->ln1_output_offset  = bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);

        L->qkv_weight_offset  = bump(&off, 3ULL * M->embed_dim * M->embed_dim, CACHE_ALIGN);
        L->qkv_bias_offset    = bump(&off, 3ULL * M->embed_dim, CACHE_ALIGN);
        L->qkv_output_offset  = bump(&off, 3ULL * (size_t)M->context_window * M->embed_dim,
                                     CACHE_ALIGN);

        L->proj_weight_offset = bump(&off, M->embed_dim * M->embed_dim, CACHE_ALIGN);
        L->proj_bias_offset   = bump(&off, M->embed_dim, CACHE_ALIGN);
        L->attention_output_offset =
            bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);
        L->residual1_output_offset =
            bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);

        L->ln2_weight_offset  = bump(&off, M->embed_dim, CACHE_ALIGN);
        L->ln2_bias_offset    = bump(&off, M->embed_dim, CACHE_ALIGN);
        L->ln2_output_offset  = bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);

        L->fc1_weight_offset  = bump(&off, 4ULL * M->embed_dim * M->embed_dim, CACHE_ALIGN);
        L->fc1_bias_offset    = bump(&off, 4ULL * M->embed_dim, CACHE_ALIGN);
        L->fc2_weight_offset  = bump(&off, 4ULL * M->embed_dim * M->embed_dim, CACHE_ALIGN);
        L->fc2_bias_offset    = bump(&off, M->embed_dim, CACHE_ALIGN);
        L->mlp_output_offset  = bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);
        L->residual2_output_offset =
            bump(&off, (size_t)M->context_window * M->embed_dim, CACHE_ALIGN);
    }

    M->layer_stride =
        M->layers[1].ln1_weight_offset - M->layers[0].ln1_weight_offset;

    /* final LN */
    M->final_ln_weight_offset = bump(&off, M->embed_dim, CACHE_ALIGN);
    M->final_ln_bias_offset   = bump(&off, M->embed_dim, CACHE_ALIGN);

    /* allocate once */
    M->total_floats = off;
    M->memory_base  = huge_alloc(off * sizeof(float));
}

/* ─── destruction helper ─────────────────────────────────────────── */
void destroy_transformer(TransformerModel *M)
{
    munmap(M->memory_base, align_up(M->total_floats * sizeof(float), HUGE_ALIGN));
    free(M->layers);
}

/* gpt2_layout_capacity.c – same includes & structs as before … */

#include <getopt.h>      /* for long-options parsing */

/* ------------- new function: size_t bytes_needed(...) --------------- */
static size_t bytes_needed(int layers, int vocab, int d_model, int ctx)
{
    size_t C  = d_model;
    size_t T  = ctx;
    size_t V  = vocab;

    /* embeddings */
    size_t token  = V * C;
    size_t pos    = T * C;
    size_t embed  = T * C;

    /* per-layer working size (same as bump logic) */
    size_t perL =
        /* ln1 W+B */        2 * C +
        /* ln1 in+out */     2 * T * C +
        /* QKV W+B+out */    (3*C*C + 3*C + 3*T*C) +
        /* proj W+B+out+res1*/(C*C + C + 2*T*C) +
        /* ln2 W+B+out */    (2*C + T*C) +
        /* fc1/fc2 W+B */    (8*C*C + 5*C) +
        /* mlp_out + res2 */ (2*T*C);

    size_t final_ln = 2 * C;

    size_t total_floats = token + pos + embed + layers * perL + final_ln;
    return total_floats * sizeof(float);   /* bytes */
}

/* ---------------- main with --size-only / --force -------------------- */
int main(int argc, char **argv)
{
    /* defaults (tiny) */
    int L = 2, V = 32768, C = 128, T = 128;
    int do_alloc = 0;

    static struct option long_opts[] = {
        {"layers",  required_argument, 0, 'l'},
        {"dmodel",  required_argument, 0, 'd'},
        {"ctx",     required_argument, 0, 't'},
        {"vocab",   required_argument, 0, 'v'},
        {"force",   no_argument,       0, 'f'},
        {0,0,0,0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "l:d:t:v:f", long_opts, NULL)) != -1) {
        switch (c) {
            case 'l': L = atoi(optarg); break;
            case 'd': C = atoi(optarg); break;
            case 't': T = atoi(optarg); break;
            case 'v': V = atoi(optarg); break;
            case 'f': do_alloc = 1;     break;
            default:  fprintf(stderr,"Usage: %s [--layers N] [--dmodel N] [--ctx N] [--vocab N] [--force]\n", argv[0]); return 1;
        }
    }

    size_t need_bytes = bytes_needed(L, V, C, T);
    double need_gib   = need_bytes / (1024.0*1024.0*1024.0);

    printf("⚙  Requested model  L=%d  d_model=%d  ctx=%d  vocab=%d\n", L,C,T,V);
    printf("→ Would need ≈ %.2f GiB (%.0f bytes)\n",
           need_gib, (double)need_bytes);

    if (!do_alloc) {
        printf("Dry-run only (no allocation).   Pass --force to allocate.\n");
        return 0;
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

    printf("Allocating huge block...  this may page-fault if hugepages are missing\n");
    layout_transformer(&M);
    printf("✅ Success!  mmap at %p, %.2f GiB reserved.\n",
           (void*)M.memory_base, need_gib);

    destroy_transformer(&M);
    return 0;
}
