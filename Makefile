CC      ?= gcc
# -O2（而非 -O3）：gcc 7 在 -O2 下不自动向量化，各级实现的差异更纯粹；
# -march=native 打开本机 AVX2/FMA 指令集；-fopenmp 供 v5 使用
CFLAGS  ?= -O2 -march=native -fopenmp -Wall -Wextra
SRC     := src/common.c src/v0_naive.c src/v1_reorder.c src/v2_tiling.c \
           src/v3_regblock.c src/v4_packing.c src/v5_openmp.c src/benchmark.c

.PHONY: all run check o3 cblas site sync clean

all: build/bench

# 把根目录 COURSE.md 同步到 docs/（网页站内阅读器加载 docs/course.md；改课程后跑一次）
sync:
	cp COURSE.md docs/course.md

# 本地预览课程网页（docs/，即 GitHub Pages 发布的同一份文件）
site:
	python3 -m http.server 8000 -d docs
	@echo "浏览器打开 http://127.0.0.1:8000"

build/bench: $(SRC) | build
	$(CC) $(CFLAGS) -Isrc $(SRC) -o $@ -lm

build:
	mkdir -p build results

# 默认跑 256/512/1024，可自定义：make run ARGS="2048"
run: build/bench
	./build/bench $(ARGS) | tee results/bench.txt

# 用非整除块大小的“刁钻”尺寸验证各实现的边角处理
check: build/bench
	./build/bench 61 260 517 | tee results/check.txt

# 对照组：-O3 自动向量化能到什么程度
o3: build/bench_o3
	./build/bench_o3 512 | tee results/bench_o3.txt

build/bench_o3: $(SRC) | build
	$(CC) -O3 -march=native -fopenmp -Isrc $(SRC) -o $@ -lm

# 天花板对照：OpenBLAS（先 sudo apt install libopenblas-dev）
cblas: build/bench_cblas
	./build/bench_cblas 512 1024 | tee results/bench_cblas.txt

build/bench_cblas: $(SRC) | build
	$(CC) $(CFLAGS) -DUSE_CBLAS -Isrc $(SRC) -o $@ -lm -lopenblas

clean:
	rm -rf build
