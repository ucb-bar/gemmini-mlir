#include <stdint.h>
#include <stdio.h>

#ifndef M
#define M 17
#endif
#ifndef N
#define N 19
#endif
#ifndef K
#define K 20
#endif
#ifndef SCALE
#define SCALE 1.0f
#endif

#ifndef INPUT_AMPLITUDE
#define INPUT_AMPLITUDE 1
#endif

#ifdef OUT_I8
typedef int8_t output_t;
#else
typedef int32_t output_t;
#endif

#ifdef USE_BIAS
#ifndef KERNEL_SYMBOL
#define KERNEL_SYMBOL gemmini_golden_gemm
#endif
extern void KERNEL_SYMBOL(int8_t *a, int8_t *b, output_t *c, int32_t *bias);
static int32_t bias[N] __attribute__((aligned(64)));
#else
#ifndef KERNEL_SYMBOL
#define KERNEL_SYMBOL gemmini_golden_gemm
#endif
extern void KERNEL_SYMBOL(int8_t *a, int8_t *b, output_t *c);
#endif

static int8_t a[M][K] __attribute__((aligned(64)));
static int8_t b[K][N] __attribute__((aligned(64)));
static struct {
    output_t values[M][N];
    uint8_t guard[2048];
} cbox __attribute__((aligned(64)));
#ifdef EMBED_EXPECTED
#include "gemm_expected.inc"
#endif

static inline uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile ("rdcycle %0" : "=r"(value) :: "memory");
    return value;
}

int main(void) {
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            a[i][k] = (int8_t)(INPUT_AMPLITUDE * (((i * 7 + k * 3) % 11) - 5));
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            b[k][j] = (int8_t)(INPUT_AMPLITUDE * (((k * 5 + j * 2) % 13) - 6));
    for (int i = 0; i < 2048; ++i) cbox.guard[i] = 0x5a;
#ifdef USE_BIAS
    for (int j = 0; j < N; ++j) bias[j] = j % 5 - 2;
#endif
    uint64_t begin = cycles();
#ifdef USE_BIAS
    KERNEL_SYMBOL(&a[0][0], &b[0][0], &cbox.values[0][0], bias);
#else
    KERNEL_SYMBOL(&a[0][0], &b[0][0], &cbox.values[0][0]);
#endif
    uint64_t elapsed = cycles() - begin;
    printf("GOLDEN_GEMM_CYCLES %d\n", (int)elapsed);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
#ifdef EMBED_EXPECTED
            int32_t expected = expected_values[i][j];
#else
            int32_t expected = 0;
            for (int k = 0; k < K; ++k)
                expected += (int32_t)a[i][k] * (int32_t)b[k][j];
#ifdef USE_BIAS
            expected += bias[j];
#endif
#ifdef OUT_I8
            expected = (int32_t)__builtin_nearbyintf((float)expected * SCALE);
#ifdef USE_RELU
            if (expected < 0) expected = 0;
#endif
            if (expected > 127) expected = 127;
            if (expected < -128) expected = -128;
#endif
#endif
            if (cbox.values[i][j] != expected) {
                printf("GOLDEN_GEMM FAIL i=%d j=%d got=%d expected=%d\n",
                       i, j, (int)cbox.values[i][j], (int)expected);
                return 1;
            }
        }
    }
    for (int i = 0; i < 2048; ++i) {
        if (cbox.guard[i] != 0x5a) {
            printf("GOLDEN_GEMM GUARD_FAIL byte=%d got=%d\n", i,
                   (int)cbox.guard[i]);
            return 2;
        }
    }
    printf("GOLDEN_GEMM PASS M=%d N=%d K=%d\n", M, N, K);
    return 0;
}
