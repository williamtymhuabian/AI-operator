#include "common.h"

/*
 * 第 0 级：教科书式三重循环（i-j-k）。
 * 内层是 A 的第 i 行与 B 的第 j 列做点积：
 *   - A 顺序访问（好事）
 *   - B 沿列方向走，每隔 N 个 float 取一个数，跨步访存（坏事）
 * 每算 2 个浮点操作（一次乘 + 一次加）都要触碰一次内存，
 * 机器大部分时间在等数据，而不是在算。
 */
void gemm_v0_naive(int M, int N, int K,
                   const float *A, const float *B, float *C)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int p = 0; p < K; p++)
                acc += A[(size_t)i * K + p] * B[(size_t)p * N + j];
            C[(size_t)i * N + j] = acc;
        }
}
