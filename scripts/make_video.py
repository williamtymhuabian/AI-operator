#!/usr/bin/env python3
"""
课程动画视频生成器：访存模式对决（v0 vs v1）
与网页版 (docs/index.html #race) 完全相同的模拟逻辑：
  两个 16x16 面板做同样的 256 次乘加，v0 内层沿 B 的列跳行，
  v1 沿行扫描；各配一个 4 行 LRU cache，实时统计搬运量与命中率。
输出: docs/demo.gif（如 imageio-ffmpeg 可用则同时输出 docs/demo.mp4）
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_GIF = os.path.join(ROOT, "docs", "demo.gif")
OUT_MP4 = os.path.join(ROOT, "docs", "demo.mp4")

W, H = 1440, 800
N = 16
CACHE_LINES = 4
STEP_MS = 60

# ---- 配色（与网页一致）----
BG      = (10, 15, 30)
PANEL   = (17, 26, 50)
CELL_BD = (35, 47, 74)
CELL_BG = (13, 20, 40)
USED    = (42, 50, 69)
MUTED   = (147, 165, 196)
TEXT    = (230, 237, 247)
CYAN    = (56, 189, 248)
GREEN   = (74, 222, 128)
RED     = (248, 113, 113)
YELLOW  = (251, 191, 36)
GRAY0   = (100, 116, 139)


def blend(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


MISS_FILL = blend(CELL_BG, RED, 0.40)
HIT_FILL  = blend(CELL_BG, GREEN, 0.42)
FETCH_FILL = blend(USED, CYAN, 0.38)

FONT_DIR = "/usr/share/fonts/opentype/noto"
F_BOLD = os.path.join(FONT_DIR, "NotoSansCJK-Bold.ttc")
F_REG = os.path.join(FONT_DIR, "NotoSansCJK-Regular.ttc")


def cjk_font(path, size):
    """Noto CJK TTC：index 2 为简体中文，不可用则退回 0"""
    for idx in (2, 0):
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            continue
    return ImageFont.load_default()


FT_TITLE = cjk_font(F_BOLD, 34)
FT_SUB = cjk_font(F_REG, 17)
FT_PT = cjk_font(F_BOLD, 20)
FT_TAG = cjk_font(F_REG, 14)
FT_SLOTS = cjk_font(F_REG, 13)
FT_SLOT = cjk_font(F_REG, 14)
FT_STAT = cjk_font(F_BOLD, 16)
FT_STEP = cjk_font(F_REG, 18)
FT_BIG = cjk_font(F_BOLD, 30)
FT_LINE = cjk_font(F_REG, 21)


def seg_text(d, x, y, segs, f):
    """依次绘制 (text, color) 片段"""
    for t, c in segs:
        d.text((x, y), t, font=f, fill=c)
        x += d.textbbox((x, y), t, font=f)[2] - x


class Panel:
    """复刻网页版 makePanel：LRU cache + 计数 + 网格状态"""

    def __init__(self, px, title, tag, tcolor, order):
        self.px, self.title, self.tag, self.tcolor = px, title, tag, tcolor
        self.used = [[False] * N for _ in range(N)]
        self.cache = []
        self.step, self.miss, self.hit = 0, 0, 0
        self.cur, self.fetched = None, None
        # 访问序列（与 JS 版一致）：
        # col（v0）：外层 j、内层 p → 行 p 内层递增、列 j 分组前进 = 沿列跳行
        # row（v1）：外层 p、内层 j → 行 p 分组前进、列 j 内层递增 = 沿行扫描
        self.rows, self.cols = [], []
        if order == "col":
            for j in range(N):
                for p in range(N):
                    self.rows.append(p)
                    self.cols.append(j)
        else:
            for p in range(N):
                for j in range(N):
                    self.rows.append(p)
                    self.cols.append(j)

    def step_once(self):
        r, c = self.rows[self.step], self.cols[self.step]
        self.fetched = None
        if r not in self.cache:
            self.miss += 1
            self.fetched = r
            if len(self.cache) >= CACHE_LINES:
                self.cache.pop(0)
            self.cache.append(r)
        else:
            self.hit += 1
            self.cache.remove(r)
            self.cache.append(r)
        self.used[r][c] = True
        self.cur = (r, c)
        self.step += 1

    def draw(self, d):
        px, y0 = self.px, 128
        d.rounded_rectangle([px, y0, px + 678, y0 + 576], 16,
                            fill=PANEL, outline=CELL_BD, width=1)
        d.text((px + 24, y0 + 16), self.title, font=FT_PT, fill=self.tcolor)
        tw = d.textbbox((0, 0), self.tag, font=FT_TAG)[2]
        d.text((px + 678 - 24 - tw, y0 + 22), self.tag, font=FT_TAG, fill=MUTED)

        gx, gy, cs, gap = px + 26, y0 + 64, 26, 3
        for r in range(N):
            for c in range(N):
                x, y = gx + c * (cs + gap), gy + r * (cs + gap)
                if self.step and self.cur == (r, c) and self.fetched is not None:
                    fill, bd = MISS_FILL, RED
                elif self.step and self.cur == (r, c) and self.fetched is None:
                    fill, bd = HIT_FILL, GREEN
                elif self.fetched is not None and r == self.fetched:
                    fill, bd = FETCH_FILL, CYAN
                elif self.used[r][c]:
                    fill, bd = USED, CELL_BD
                else:
                    fill, bd = CELL_BG, CELL_BD
                d.rounded_rectangle([x, y, x + cs, y + cs], 4, fill=fill,
                                    outline=bd, width=1)
                if self.step and self.cur == (r, c):
                    d.rounded_rectangle([x - 1, y - 1, x + cs + 1, y + cs + 1],
                                        5, outline=TEXT, width=2)

        cx, cw = px + 512, 142
        d.text((cx, gy - 26), "cache（LRU · 4 行）", font=FT_SLOTS, fill=MUTED)
        for i in range(CACHE_LINES):
            sy = gy + i * 52
            line = self.cache[i] if i < len(self.cache) else None
            newest = line is not None and self.fetched is not None and line == self.fetched
            bd = CYAN if newest else (CELL_BD if line is None else blend(CELL_BD, CYAN, 0.4))
            fill = blend(PANEL, CYAN, 0.08) if newest else PANEL
            d.rounded_rectangle([cx, sy, cx + cw, sy + 40], 8, fill=fill,
                                outline=bd, width=1)
            txt = f"B 行 {line}（64B）" if line is not None else "空"
            tc = (186, 230, 253) if newest else (MUTED if line is None else GRAY0)
            d.text((cx + 14, sy + 9), txt, font=FT_SLOT, fill=tc)

        rate = self.hit / self.step * 100 if self.step else 0.0
        seg_text(d, px + 24, y0 + 540, [
            ("内存搬运 ", MUTED), (f"{self.miss * 64}", TEXT), (" B  ·  ", MUTED),
            ("未中 ", MUTED), (f"{self.miss}", RED), ("  ·  ", MUTED),
            ("命中 ", MUTED), (f"{self.hit}", GREEN), ("  ·  命中率 ", MUTED),
            (f"{rate:.1f}%", TEXT),
        ], FT_STAT)


def build_frames():
    A = Panel(30, "v0 朴素 i-j-k", "内层 p 递增 → 沿列跳行", GRAY0, "col")
    B = Panel(732, "v1 重排 i-p-j", "内层 j 递增 → 沿行扫描", CYAN, "row")

    frames, durations = [], []

    def render(step_idx, end=False):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 22), "AI 算子优化 · 访存模式对决", font=FT_TITLE, fill=TEXT)
        d.text((40, 78), "同样的 256 次乘加，只换循环顺序 —— cache 只装 4 行（LRU）",
               font=FT_SUB, fill=MUTED)
        A.draw(d)
        B.draw(d)
        if end:
            d.rounded_rectangle([240, 260, 1200, 540], 18,
                                fill=blend(PANEL, YELLOW, 0.06),
                                outline=YELLOW, width=2)
            for i, (t, f, c) in enumerate([
                ("比赛结束", FT_BIG, YELLOW),
                ("同样 256 次计算：v0 从内存搬运 16 KB（256 次全部未中）", FT_LINE, TEXT),
                ("　　　　　　　　　 v1 仅搬运 1 KB（16 次未中 · 命中 93.8%）", FT_LINE, TEXT),
                ("16 倍的搬运差距 → 实测 2048³：0.4 vs 7.6 GFLOP/s（21 倍）", FT_LINE, YELLOW),
            ]):
                tw = d.textbbox((0, 0), t, font=f)[2]
                d.text((720 - tw / 2, 292 + i * 56), t, font=f, fill=c)
        else:
            t = f"第 {step_idx} / 256 步"
            tw = d.textbbox((0, 0), t, font=FT_STEP)[2]
            d.text(((W - tw) / 2, 742), t, font=FT_STEP, fill=MUTED)
        frames.append(img)

    render(0)
    durations.append(1400)                    # 片头停留 1.4s
    for s in range(256):
        A.step_once()
        B.step_once()
        render(s + 1)
        durations.append(STEP_MS)
    render(256, end=True)
    durations.append(2800)
    return frames, durations


def main():
    frames, durations = build_frames()
    print(f"帧数: {len(frames)}")

    # 固定调色板量化：避免逐帧调色板闪烁，且减小体积
    pal = frames[1].quantize(colors=255, method=Image.MEDIANCUT)
    qp = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    qp[0].save(OUT_GIF, save_all=True, append_images=qp[1:],
               duration=durations, loop=0, optimize=True)
    print(f"GIF: {OUT_GIF}  {os.path.getsize(OUT_GIF) / 1e6:.1f} MB")

    try:
        import numpy as np
        import imageio.v2 as imageio
        with imageio.get_writer(OUT_MP4, fps=round(1000 / STEP_MS),
                                codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(np.asarray(f))
        print(f"MP4: {OUT_MP4}  {os.path.getsize(OUT_MP4) / 1e6:.1f} MB")
    except Exception as e:
        print(f"MP4 跳过（{type(e).__name__}: {e}）")


if __name__ == "__main__":
    main()
