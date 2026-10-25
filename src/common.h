#ifndef GEMM_COURSE_COMMON_H
#define GEMM_COURSE_COMMON_H

#include <stddef.h>

/*
 * 所有 GEMM 实现的统一签名：
 *   C[M×N] = A[M×K] × B[K×N]   （行主序，无 lda/ldb/ldc，矩阵连续存放）
 * 入口处 C 的内容无意义，实现必须完整覆写 C。
 */
typedef void (*gemm_fn)(int M, int N, int K,
                        const float *A, const float *B, float *C);

/* 确定性伪随机数（xorshift32），返回 [-1,1) 的 float，保证实验可复现 */
float randf(unsigned *seed);

/* 单调时钟（秒），用于计时 */
double now_sec(void);

/*
 * 标量兜底例程：把 A 的第 i0..i0+m 行、B 的第 j0..j0+n 列、
 * k 方向 p0..p0+kc 这一窗口的乘积“累加”到 C 上（C += A×B，不是覆盖）。
 * 各高级实现用它处理行列/收缩维度不整除块大小时的边角（fringe）。
 * M 未用到，保留只是为了参数整齐。
 */
void gemm_add_block(int M, int N, int K,
                    const float *A, const float *B, float *C,
                    int i0, int j0, int p0, int m, int n, int kc);

#endif
