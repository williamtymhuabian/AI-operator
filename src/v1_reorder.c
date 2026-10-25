#include "common.h"

#include <string.h>

/*
 * 第 1 级：循环重排（i-k-j）。一行计算都不改，只调换循环顺序。
 * 原来的内层是“沿 B 的列跨步走”；重排后内层变成：
 *   C[i][j] += A[i][p] * B[p][j]，j 连续增长
 * 于是三条访存流（A 的一行、B 的一行、C 的一行）全部变成顺序访问。
 * 顺序访问对 cache/预取器/SIMD 都极度友好——这就是“访存模式决定性能”。
 */
void gemm_v1_reorder(int M, int N, int K,
                     const float *A, const float *B, float *C)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i = 0; i < M; i++)
        for (int p = 0; p < K; p++) {
            float a = A[(size_t)i * K + p];
            for (int j = 0; j < N; j++)
                C[(size_t)i * N + j] += a * B[(size_t)p * N + j];
        }
}
