#include "common.h"

#include <time.h>

float randf(unsigned *seed)
{
    unsigned x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    /* 取高 24 位保证尾数精度，映射到 [-1,1) */
    return (float)(x >> 8) / (float)(1 << 23) - 1.f;
}

double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void gemm_add_block(int M, int N, int K,
                    const float *A, const float *B, float *C,
                    int i0, int j0, int p0, int m, int n, int kc)
{
    (void)M;
    for (int i = i0; i < i0 + m; i++)
        for (int p = p0; p < p0 + kc; p++) {
            float a = A[(size_t)i * K + p];
            for (int j = j0; j < j0 + n; j++)
                C[(size_t)i * N + j] += a * B[(size_t)p * N + j];
        }
}
