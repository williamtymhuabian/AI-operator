#include "common.h"

#include <stdlib.h>
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
 * 4×8 微内核（打包版）：与 v3 的差别只在数据布局——
 *   A 面板：每 4 行一组预先拷成 Ap[k*4+r]，广播读取完全连续；
 *   B 面板：k 方向 256 行预先拷到一起，每行连续；
 * 运算期所有 load 都落在紧凑、连续、预取友好的缓冲区上。
 */
static void uk_4x8_packed(int kc, const float *Ap, /* kc×4, Ap[k*4+r] */
                          const float *Bp, int ldb, /* 面板第 j0 列起 */
                          float *c0, int ldc)
{
#ifdef HAVE_AVX
    __m256 c[4];
    for (int r = 0; r < 4; r++)
        c[r] = _mm256_loadu_ps(c0 + (size_t)r * ldc);

    for (int k = 0; k < kc; k++) {
        __m256 b = _mm256_loadu_ps(Bp + (size_t)k * ldb);
        c[0] = MFMA(_mm256_broadcast_ss(Ap + (size_t)k * 4 + 0), b, c[0]);
        c[1] = MFMA(_mm256_broadcast_ss(Ap + (size_t)k * 4 + 1), b, c[1]);
        c[2] = MFMA(_mm256_broadcast_ss(Ap + (size_t)k * 4 + 2), b, c[2]);
        c[3] = MFMA(_mm256_broadcast_ss(Ap + (size_t)k * 4 + 3), b, c[3]);
    }

    for (int r = 0; r < 4; r++)
        _mm256_storeu_ps(c0 + (size_t)r * ldc, c[r]);
#else
    float acc[4][8] = {{0}};
    for (int k = 0; k < kc; k++)
        for (int r = 0; r < 4; r++) {
            float ar = Ap[(size_t)k * 4 + r];
            for (int j = 0; j < 8; j++)
                acc[r][j] += ar * Bp[(size_t)k * ldb + j];
        }
    for (int r = 0; r < 4; r++)
        for (int j = 0; j < 8; j++)
            c0[(size_t)r * ldc + j] += acc[r][j];
#endif
}

#define MR 4   /* 微内核行数 */
#define NR 8   /* 微内核列数 */
#define KC 256 /* k 方向面板厚度：A 面板 256×4×4B=4KB 驻留 L1，
                 B 面板 256 行×N 列常驻 L2/L3 */

/*
 * 第 4 级：打包（packing）。
 * v3 的问题：微内核直接从原矩阵跨步取数，A 的 4 行分散在 4 条
 * 相距 K*4B 的流里。这一级先花一点时间把数据“搬”成微内核最
 * 喜欢的形状，再全速计算——这是 OpenBLAS/BLIS 等真正的 BLAS
 * 库的标准结构（术语叫 GEBP / 面板-微内核 两级结构）。
 *
 * 结构：
 *   按 k 切面板（厚度 KC）→ B 面板整体拷到一起（一次）
 *   → 对每个 A 行组（4 行）：打包成连续面板，扫过整个 B 面板
 */
void gemm_v4_packing(int M, int N, int K,
                     const float *A, const float *B, float *C)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    float *Bp = malloc((size_t)KC * N * sizeof(float));
    float *Ap = malloc((size_t)KC * MR * sizeof(float));
    if (!Bp || !Ap) { /* 课程代码：简化处理分配失败 */
        free(Bp);
        free(Ap);
        return;
    }

    for (int p0 = 0; p0 < K; p0 += KC) {
        int kc = K - p0 < KC ? K - p0 : KC;

        /* B 面板：第 p0..p0+kc 行拷到一起 */
        memcpy(Bp, B + (size_t)p0 * N, (size_t)kc * N * sizeof(float));

        for (int i0 = 0; i0 < M; i0 += MR) {
            int m = M - i0 < MR ? M - i0 : MR;

            if (m < MR) { /* 行边角 */
                gemm_add_block(M, N, K, A, B, C, i0, 0, p0, m, N, kc);
                continue;
            }

            /* A 面板：4 行 × kc 列 → Ap[k*4+r] 连续布局 */
            for (int k = 0; k < kc; k++)
                for (int r = 0; r < MR; r++)
                    Ap[(size_t)k * MR + r] = A[(size_t)(i0 + r) * K + p0 + k];

            for (int j0 = 0; j0 + NR <= N; j0 += NR)
                uk_4x8_packed(kc, Ap, Bp + j0, N,
                              C + (size_t)i0 * N + j0, N);

            if (N % NR) /* 列边角 */
                gemm_add_block(M, N, K, A, B, C,
                               i0, N - N % NR, p0, MR, N % NR, kc);
        }
    }

    free(Bp);
    free(Ap);
}
