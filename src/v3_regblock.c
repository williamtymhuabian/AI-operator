#include "common.h"

#include <string.h>

#if defined(__AVX__)
#include <immintrin.h>
#define HAVE_AVX 1
#endif

#if defined(HAVE_AVX) && defined(__FMA__)
#define MFMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define MFMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

/*
 * 4×8 微内核（跨步读取版，未打包）：
 * 一次算出 C 的 4 行 × 8 列，32 个累加数全部放在寄存器里
 * （对应 8 个 __m256 寄存器），k 循环走完才写回内存一次。
 *
 * 关键收益有两个：
 *   1. B 的一行（8 个 float，一个 __m256）被装进寄存器后，
 *      要和 4 行 A 各乘一次——读一次内存，做 4 次乘加，访存被摊薄 4 倍；
 *   2. FMA 指令（_mm256_fmadd_ps）一条完成 8 个“乘加”，
 *      一个周期最多发 2 条（P 核），纯算力部分的吞吐直接拉满。
 */
static void uk_4x8_strided(int kc,
                           const float *a0, const float *a1,
                           const float *a2, const float *a3,
                           const float *b0, int ldb,
                           float *c0, int ldc)
{
#ifdef HAVE_AVX
    __m256 c[4];
    for (int r = 0; r < 4; r++)
        c[r] = _mm256_loadu_ps(c0 + (size_t)r * ldc);

    for (int k = 0; k < kc; k++) {
        __m256 b = _mm256_loadu_ps(b0);
        c[0] = MFMA(_mm256_broadcast_ss(a0), b, c[0]);
        c[1] = MFMA(_mm256_broadcast_ss(a1), b, c[1]);
        c[2] = MFMA(_mm256_broadcast_ss(a2), b, c[2]);
        c[3] = MFMA(_mm256_broadcast_ss(a3), b, c[3]);
        a0++; a1++; a2++; a3++;
        b0 += ldb;
    }

    for (int r = 0; r < 4; r++)
        _mm256_storeu_ps(c0 + (size_t)r * ldc, c[r]);
#else /* 没有 AVX 时的同结构标量版本：仍然享受寄存器分块的复用收益 */
    const float *a[4] = {a0, a1, a2, a3};
    float acc[4][8] = {{0}};
    for (int k = 0; k < kc; k++)
        for (int r = 0; r < 4; r++) {
            float ar = a[r][k];
            for (int j = 0; j < 8; j++)
                acc[r][j] += ar * b0[(size_t)k * ldb + j];
        }
    for (int r = 0; r < 4; r++)
        for (int j = 0; j < 8; j++)
            c0[(size_t)r * ldc + j] += acc[r][j];
#endif
}

/*
 * 第 3 级：寄存器分块 + SIMD。
 * 外层把 C 切成 4×8 的小块交给微内核，整条 k 循环的中间结果
 * 全部留在寄存器里。相比第 2 级“tile 进 cache”，这一级是
 * “小块进寄存器”——存储层级的最顶层。
 */
void gemm_v3_regblock(int M, int N, int K,
                      const float *A, const float *B, float *C)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += 4) {
        int m = M - i0 < 4 ? M - i0 : 4;
        for (int j0 = 0; j0 < N; j0 += 8) {
            int n = N - j0 < 8 ? N - j0 : 8;
            if (m == 4 && n == 8)
                uk_4x8_strided(K,
                               A + (size_t)(i0 + 0) * K,
                               A + (size_t)(i0 + 1) * K,
                               A + (size_t)(i0 + 2) * K,
                               A + (size_t)(i0 + 3) * K,
                               B + j0, N,
                               C + (size_t)i0 * N + j0, N);
            else /* 行列边角：标量兜底 */
                gemm_add_block(M, N, K, A, B, C, i0, j0, 0, m, n, K);
        }
    }
}
