#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void gemm_v0_naive(int, int, int, const float *, const float *, float *);
void gemm_v1_reorder(int, int, int, const float *, const float *, float *);
void gemm_v2_tiling(int, int, int, const float *, const float *, float *);
void gemm_v3_regblock(int, int, int, const float *, const float *, float *);
void gemm_v4_packing(int, int, int, const float *, const float *, float *);
void gemm_v5_openmp(int, int, int, const float *, const float *, float *);

#ifdef USE_CBLAS
#include <cblas.h>
static void gemm_blas(int M, int N, int K,
                      const float *A, const float *B, float *C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
#endif

typedef struct {
    const char *name;
    gemm_fn fn;
} entry_t;

/* 双精度参考实现：所有版本的正确性都以它为准（顺序访存的 i-k-j 写法） */
static void gemm_ref(int M, int N, int K,
                     const float *A, const float *B, double *R)
{
    for (size_t idx = 0; idx < (size_t)M * N; idx++)
        R[idx] = 0.0;

    for (int i = 0; i < M; i++)
        for (int p = 0; p < K; p++) {
            double a = (double)A[(size_t)i * K + p];
            for (int j = 0; j < N; j++)
                R[(size_t)i * N + j] += a * (double)B[(size_t)p * N + j];
        }
}

/* fp32 累加顺序不同会带来 1e-4 量级的舍入差，用宽松阈值只抓真 bug */
static int check(int M, int N, const float *C, const double *R)
{
    for (size_t idx = 0; idx < (size_t)M * N; idx++) {
        double r = R[idx];
        double e = fabs((double)C[idx] - r);
        if (e > 1e-3 * (1.0 + fabs(r)))
            return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    int sizes[8], ns = 0;
    if (argc > 1) {
        for (int i = 1; i < argc && ns < 8; i++)
            sizes[ns++] = atoi(argv[i]);
    } else {
        sizes[ns++] = 256;
        sizes[ns++] = 512;
        sizes[ns++] = 1024;
    }

    entry_t impls[] = {
        {"v0_naive",    gemm_v0_naive},
        {"v1_reorder",  gemm_v1_reorder},
        {"v2_tiling",   gemm_v2_tiling},
        {"v3_regblock", gemm_v3_regblock},
        {"v4_packing",  gemm_v4_packing},
        {"v5_openmp",   gemm_v5_openmp},
#ifdef USE_CBLAS
        {"openblas",    gemm_blas},
#endif
    };
    int ni = (int)(sizeof(impls) / sizeof(impls[0]));

    printf("%-14s %6s %12s %12s %9s\n",
           "实现", "校验", "用时(ms)", "GFLOP/s", "加速比");

    for (int s = 0; s < ns; s++) {
        int M = sizes[s], N = M, K = M;
        double gflop = 2.0 * M * N * K / 1e9;

        float *A = malloc((size_t)M * K * sizeof(float));
        float *B = malloc((size_t)K * N * sizeof(float));
        float *C = malloc((size_t)M * N * sizeof(float));
        double *R = malloc((size_t)M * N * sizeof(double));

        unsigned sa = 42, sb = 137;
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = randf(&sa);
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = randf(&sb);
        gemm_ref(M, N, K, A, B, R);

        printf("== M=N=K=%d  (单次运行 %.2f GFLOP) ==\n", M, gflop);

        double t_base = 0;
        for (int e = 0; e < ni; e++) {
            double t0 = now_sec();
            impls[e].fn(M, N, K, A, B, C);
            double t = now_sec() - t0;

            if (!check(M, N, C, R)) {
                printf("%-14s %6s\n", impls[e].name, "错误!");
                continue;
            }

            /* 复跑取最小值，抑制调度抖动 */
            int reps = t > 0 ? (int)(0.6 / t) : 5;
            if (reps < 1) reps = 1;
            if (reps > 5) reps = 5;
            for (int r = 0; r < reps; r++) {
                double t1 = now_sec();
                impls[e].fn(M, N, K, A, B, C);
                double dt = now_sec() - t1;
                if (dt < t) t = dt;
            }

            if (e == 0)
                t_base = t;

            printf("%-14s %6s %12.1f %12.1f %8.1fx\n",
                   impls[e].name, "通过", t * 1e3, gflop / t, t_base / t);
        }

        free(A);
        free(B);
        free(C);
        free(R);
    }

    return 0;
}
