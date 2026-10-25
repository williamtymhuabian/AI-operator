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

static void uk_4x8_packed(int kc, const float *Ap,
                          const float *Bp, int ldb,
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

#define MR 4
#define NR 8
#define KC 256

/*
 * 第 5 级：多线程（OpenMP）。
 * 在 v4 的结构上，把“对 A 行组”的循环交给线程池：
 *   - B 面板每个 k 面板只打包一次，所有线程只读共享；
 *   - 每个线程负责不同 i0 行组，写的是 C 的不同行，天然无竞争，
 *     不需要任何锁和原子操作；
 *   - 每线程一个私有的小 A 打包缓冲（4KB，L1 里）。
 *
 * GEMM 能“免费”并行的根本原因：C 的行/列块之间完全独立。
 */
void gemm_v5_openmp(int M, int N, int K,
                    const float *A, const float *B, float *C)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    float *Bp = malloc((size_t)KC * N * sizeof(float));
    if (!Bp)
        return;

    for (int p0 = 0; p0 < K; p0 += KC) {
        int kc = K - p0 < KC ? K - p0 : KC;

        memcpy(Bp, B + (size_t)p0 * N, (size_t)kc * N * sizeof(float));

        int ngroups = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static)
        for (int g = 0; g < ngroups; g++) {
            int i0 = g * MR;
            int m = M - i0 < MR ? M - i0 : MR;
            float *Ap = malloc((size_t)KC * MR * sizeof(float));

            if (m < MR) { /* 行边角 */
                gemm_add_block(M, N, K, A, B, C, i0, 0, p0, m, N, kc);
            } else if (Ap) {
                for (int k = 0; k < kc; k++)
                    for (int r = 0; r < MR; r++)
                        Ap[(size_t)k * MR + r] =
                            A[(size_t)(i0 + r) * K + p0 + k];

                for (int j0 = 0; j0 + NR <= N; j0 += NR)
                    uk_4x8_packed(kc, Ap, Bp + j0, N,
                                  C + (size_t)i0 * N + j0, N);

                if (N % NR)
                    gemm_add_block(M, N, K, A, B, C,
                                   i0, N - N % NR, p0, MR, N % NR, kc);
            } else { /* 打包缓冲分配失败：标量兜底，保证正确性 */
                gemm_add_block(M, N, K, A, B, C, i0, 0, p0, MR, N, kc);
            }

            free(Ap);
        }
    }

    free(Bp);
}
