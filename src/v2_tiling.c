#include "common.h"

#include <string.h>

/*
 * 第 2 级：cache 分块（tiling）。仍然不改任何计算，只在 i-p-j 的
 * 顺序之上加三层“块”循环，让一个块的工作集整个驻留在 cache 里。
 *
 * 不分块时：算完一行 C，B 已经被完整扫过一遍；算下一行 C 时 B 又
 * 要重新搬进来——B 被搬运 M 次。分块后：一块 B（TK×TN）在 cache
 * 里被 TM 行 A 复用完才被换出。“数据在 cache 里被复用几次”就是
 * 分块的全部意义。
 *
 * 块大小按 cache 容量推导（本机 L1d=48KB，L2=1.25MB，L3=20MB）：
 *   A 块 TM×TK = 128×64×4B = 32KB
 *   B 块 TK×TN =  64×512×4B = 128KB
 *   C 块 TM×TN = 128×512×4B = 256KB
 *   合计 ≈ 416KB → 瞄准 L2（给预取和其它程序留余量，不占满）
 * 注意块不是越小越好：切太小则循环开销上升、内层流太短，预取器
 * 吃不饱——本课程实测 64×64×64 反而比不分块更慢。
 */
#define TM 128 /* C 行方向块大小 */
#define TN 512 /* C 列方向块大小 */
#define TK 64  /* 收缩维度 k 方向块大小 */

void gemm_v2_tiling(int M, int N, int K,
                    const float *A, const float *B, float *C)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int ii = 0; ii < M; ii += TM) {
        int iend = ii + TM < M ? ii + TM : M;
        for (int jj = 0; jj < N; jj += TN) {
            int jend = jj + TN < N ? jj + TN : N;
            for (int pp = 0; pp < K; pp += TK) {
                int pend = pp + TK < K ? pp + TK : K;
                for (int i = ii; i < iend; i++)
                    for (int p = pp; p < pend; p++) {
                        float a = A[(size_t)i * K + p];
                        for (int j = jj; j < jend; j++)
                            C[(size_t)i * N + j] += a * B[(size_t)p * N + j];
                    }
            }
        }
    }
}
