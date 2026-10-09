#include <stdint.h>
#include <stdio.h>

#ifndef BATCH
#define BATCH 2
#endif
#ifndef M
#define M 17
#endif
#ifndef N
#define N 19
#endif
#ifndef K
#define K 20
#endif

#ifndef KERNEL_SYMBOL
#define KERNEL_SYMBOL gemmini_golden_batched_gemm
#endif
extern void KERNEL_SYMBOL(int8_t *a, int8_t *b, int32_t *c);
static int8_t a[BATCH][M][K] __attribute__((aligned(64)));
static int8_t b[BATCH][K][N] __attribute__((aligned(64)));
static struct {
    int32_t values[BATCH][M][N];
    uint8_t guard[2048];
} cbox __attribute__((aligned(64)));

#ifdef EMBED_EXPECTED
#include "batched_expected.inc"
#endif

static inline uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile ("rdcycle %0" : "=r"(value) :: "memory");
    return value;
}

int main(void) {
    for (int q = 0; q < BATCH; ++q) {
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                a[q][i][k] = (int8_t)(((q * 3 + i * 7 + k * 2) % 11) - 5);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                b[q][k][j] = (int8_t)(((q * 5 + k * 3 + j * 2) % 13) - 6);
    }
    for (int i = 0; i < 2048; ++i) cbox.guard[i] = 0x5a;
    uint64_t begin = cycles();
    KERNEL_SYMBOL(&a[0][0][0], &b[0][0][0], &cbox.values[0][0][0]);
    uint64_t elapsed = cycles() - begin;
    printf("GOLDEN_BATCHED_CYCLES %d\n", (int)elapsed);
    for (int q = 0; q < BATCH; ++q)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                int32_t expected = 0;
#ifdef EMBED_EXPECTED
                expected = expected_values[q][i][j];
#else
                for (int k = 0; k < K; ++k)
                    expected += (int32_t)a[q][i][k] * (int32_t)b[q][k][j];
#endif
                if (cbox.values[q][i][j] != expected) {
                    printf("GOLDEN_BATCHED FAIL q=%d i=%d j=%d got=%d expected=%d\n",
                           q, i, j, (int)cbox.values[q][i][j], (int)expected);
                    return 1;
                }
            }
    for (int i = 0; i < 2048; ++i)
        if (cbox.guard[i] != 0x5a) {
            printf("GOLDEN_BATCHED GUARD_FAIL byte=%d\n", i);
            return 2;
        }
    printf("GOLDEN_BATCHED PASS B=%d M=%d N=%d K=%d\n", BATCH, M, N, K);
    return 0;
}
