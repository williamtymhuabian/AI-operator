# AI 算子优化入门：GEMM 六级优化实战

一个可以直接编译运行的"AI 算子优化"练手项目：从教科书式的朴素矩阵乘出发，
逐步优化到 **约 680 倍加速**（本机实测），每一级对应课程的一课。

配套课程：[COURSE.md](COURSE.md)（在线版可直接在[课程页](https://williamtymhuabian.github.io/AI-operator/)内阅读，无需跳转 GitHub）· 课程源文件改动后跑 `make sync` 同步到网页

## 课程动画演示

同样的 256 次乘加，只换循环顺序：v0 沿列跳行（每步浪费一条 cache 行）vs v1 沿行扫描
（一条 cache 行喂 16 次计算）——16 倍的内存搬运差距，就是实测 21 倍加速的来源。

![访存模式对决动画](docs/demo.gif)

*生成脚本：[scripts/make_video.py](scripts/make_video.py)（`make video`），与网页内交互动画同一套模拟逻辑*

[![课程展示页](docs/screenshot.png)](https://williamtymhuabian.github.io/AI-operator/)

## 快速开始

```bash
make          # 编译（gcc，需要 -march=native 支持 AVX2）
make run      # 跑基准：256/512/1024 三种尺寸
make run ARGS=2048   # 指定尺寸
make check    # 用 61/260/517 这种"刁钻"尺寸验证边角处理正确性
make o3       # 对照组：-O3 自动向量化
make cblas    # 对照组：OpenBLAS 天花板（先 sudo apt install libopenblas-dev）
```

## 六级实现

| 级别 | 文件 | 优化手段 | 实测（2048³，本机 i5-13400F） |
|------|------|----------|------------|
| v0 | [src/v0_naive.c](src/v0_naive.c) | 朴素三重循环（基线） | 0.4 GF/s（47.6 秒） |
| v1 | [src/v1_reorder.c](src/v1_reorder.c) | 循环重排：全部顺序访存 | 7.6 GF/s（21×） |
| v2 | [src/v2_tiling.c](src/v2_tiling.c) | cache 分块（tiling） | 8.1 GF/s（22×） |
| v3 | [src/v3_regblock.c](src/v3_regblock.c) | 寄存器分块 4×8 + AVX2/FMA | 10.1 GF/s（28×） |
| v4 | [src/v4_packing.c](src/v4_packing.c) | 打包（packing）+ k 面板 | 33.6 GF/s（93×） |
| v5 | [src/v5_openmp.c](src/v5_openmp.c) | OpenMP 多线程 | 246.5 GF/s（**683×**） |

（v3 在大矩阵上会被访存拖垮，v4 的面板结构把它救回来——这正是课程的看点，
详见 [COURSE.md](COURSE.md) 第 5、6 课。）

## 目录结构

```
AI-operator/
├── COURSE.md        # 完整课程（10 课，含全部实测数据与练习）
├── Makefile         # build / run / check / o3 / cblas
├── src/
│   ├── common.h/.c  # 统一签名、计时、随机数据、标量兜底
│   ├── v0_naive.c   # → v5_openmp.c 六级实现
│   └── benchmark.c  # 双精度校验 + 计时 + GFLOP/s + 加速比
└── results/         # 历次实测输出
```

## 环境

实测机器：Intel i5-13400F（6P+4E，16 线程，AVX2+FMA），L1d 48KB / L2 1.25MB / L3 20MB，
Ubuntu 18.04，gcc 7.5（`-O2 -march=native -fopenmp`）。
没有 AVX-512 的机器上代码会自动退回标量路径，仍然可运行。
