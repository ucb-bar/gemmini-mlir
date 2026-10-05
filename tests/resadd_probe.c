#include <stdint.h>
#include <stdio.h>

#ifndef M
#define M 17
#endif
#ifndef N
#define N 19
#endif

extern void gemmini_golden_resadd(int8_t *, int8_t *, int8_t *, int8_t *, int8_t *);
static int8_t a[M][N] __attribute__((aligned(64)));
static int8_t b[M][N] __attribute__((aligned(64)));
static struct {
    int8_t c[M][N];
    uint8_t guard[2048];
} storage __attribute__((aligned(64)));
#define c storage.c
static int8_t identity[16][16] __attribute__((aligned(64)));
static int8_t scratch[1024] __attribute__((aligned(64)));

static inline uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile ("rdcycle %0" : "=r"(value) :: "memory");
    return value;
}

int main(void) {
    for (int i = 0; i < 2048; ++i) storage.guard[i] = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            identity[i][j] = i == j;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            a[i][j] = (int8_t)(((i * 7 + j * 3) % 21) - 10);
            b[i][j] = (int8_t)(((i * 5 + j * 2) % 17) - 8);
        }
    uint64_t start = cycles();
    gemmini_golden_resadd(&a[0][0], &b[0][0], &c[0][0], &identity[0][0], scratch);
    printf("GOLDEN_RESADD_CYCLES %d\n", (int)(cycles() - start));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int expected = (int)a[i][j] + (int)b[i][j];
            if (c[i][j] != expected) {
                printf("GOLDEN_RESADD FAIL i=%d j=%d got=%d expected=%d\n",
                       i, j, (int)c[i][j], expected);
                return 0;
            }
        }
    for (int i = 0; i < 2048; ++i)
        if (storage.guard[i] != 0) {
            printf("GOLDEN_RESADD GUARD i=%d value=%d\n", i, storage.guard[i]);
            return 0;
        }
    printf("GOLDEN_RESADD PASS M=%d N=%d\n", M, N);
    return 0;
}
