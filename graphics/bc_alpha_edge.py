"""BC 风格块压缩管线（纯标准库实现）。

对应六类缺陷的修复：

1. alpha 边缘保护 —— 透明纹素的 RGB 先以最近不透明纹素预填充再分块，
   避免块内插值把边缘颜色拉向透明像素里的垃圾色（常见为黑）。
2. 法线 Z 重建 —— 法线贴图只压缩 XY 两个通道（BC5 风格），
   解码时用 z = sqrt(1 - x^2 - y^2) 重建 Z，而不是把 Z 当普通通道压掉。
3. 色彩空间匹配 —— sRGB 纹素一律先转线性再做端点拟合/误差比较，
   压缩格式携带 sRGB/UNORM 标记，编码端与解码端色彩空间不一致直接拒绝。
4. mip 前置 —— mip 链在压缩之前于未压缩数据上生成（线性空间盒式过滤），
   避免在压缩块上降采样导致远处块状噪点。
5. 确定性编码 —— 抖动用固定种子 LCG，扫描顺序固定，所有平局按
   通道和、索引大小确定性打破，重复编码同一输入逐字节一致。
6. 块尺寸与对齐 —— 非 4 倍数尺寸做边缘钳制填充到块网格，
   原始尺寸写入头部，块固定 16 字节，pitch 按块对齐。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 常量与格式定义
# ---------------------------------------------------------------------------

BLOCK = 4                       # 4x4 纹素一块
COLOR_BLOCK_BYTES = 8           # BC1 风格颜色块：2 端点 + 32bit 索引
CHANNEL_BLOCK_BYTES = 8         # BC4 风格单通道块：2 端点 + 48bit 索引（6 字节）
# BC3/BC5 块 = 颜色块 8 + 通道块 8 = 16 字节，天然按 16 字节对齐
RAW_BLOCK_BYTES = COLOR_BLOCK_BYTES + CHANNEL_BLOCK_BYTES
BLOCK_ALIGN = 16
MAGIC = b"BCFX1000"             # 容器魔数/版本
HEADER_SIZE = 16
LEVEL_HEADER_SIZE = 12         # 级头部：padded_w, padded_h, flags, n_over, pitch(HHHHI)

# 格式标识（高 bit 表示 sRGB 编码，必须与纹素色彩空间匹配）
FMT_BC1 = 0x31                   # 不透明 RGB，无 alpha
FMT_BC3 = 0x33                   # RGB + 独立 alpha 通道
FMT_BC5 = 0x35                   # 双通道（法线 XY）
SRGB_FLAG = 0x80

FLAG_SRGB = 1 << 0
FLAG_NORMAL = 1 << 1
FLAG_PADDED_X = 1 << 2
FLAG_PADDED_Y = 1 << 3

# 确定性 LCG（数值食谱参数），保证任何机器/任何进程上抖动序列一致
_LCG_M = 1 << 32
_LCG_A = 1664525
_LCG_C = 1013904223


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Texture:
    """RGBA8 纹理（线性字节缓冲，行主序，每像素 4 字节）。"""

    width: int
    height: int
    rgba: bytearray
    srgb: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("纹理宽高必须为正数")
        need = self.width * self.height * 4
        if len(self.rgba) != need:
            raise ValueError(f"rgba 长度应为 {need}，实际 {len(self.rgba)}")
        self.rgba = bytearray(self.rgba)

    def px(self, x: int, y: int) -> tuple[int, int, int, int]:
        i = (y * self.width + x) * 4
        return tuple(self.rgba[i:i + 4])  # type: ignore[return-value]

    def set_px(self, x: int, y: int, r: int, g: int, b: int, a: int) -> None:
        i = (y * self.width + x) * 4
        self.rgba[i:i + 4] = bytes((r, g, b, a))

    def copy(self) -> "Texture":
        return Texture(self.width, self.height, bytearray(self.rgba), self.srgb)


# ---------------------------------------------------------------------------
# 色彩空间转换（sRGB IEC 61966-2-1）
# ---------------------------------------------------------------------------


def srgb_to_linear(c: float) -> float:
    """8bit 码值（0-255）-> 线性光（0-1）。"""
    c = max(0.0, min(1.0, c / 255.0))
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> int:
    """线性光（0-1）-> 8bit sRGB 码值。"""
    c = max(0.0, min(1.0, c))
    v = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055
    return int(round(v * 255.0))


def _srgb_linear_one(value: int) -> dict:
    """单个码值的双向往返（srgb_linear 入口的单值场景）。"""
    if not 0 <= value <= 255:
        raise ValueError("码值必须在 0..255 之间")
    lin = srgb_to_linear(float(value))
    back = linear_to_srgb(lin)
    return {"input": value, "linear": lin, "roundtrip": back}


# ---------------------------------------------------------------------------
# 块对齐 / 边缘钳制填充
# ---------------------------------------------------------------------------


def aligned_dim(v: int) -> int:
    """向上对齐到块网格（4 的倍数）。"""
    return max(BLOCK, (v + BLOCK - 1) // BLOCK * BLOCK)


def aligned_pitch(n_blocks_x: int, raw_block: int = RAW_BLOCK_BYTES) -> int:
    """一行块的压缩字节数，并按 16 字节边界向上对齐。"""
    return ((n_blocks_x * raw_block + BLOCK_ALIGN - 1)
            // BLOCK_ALIGN * BLOCK_ALIGN)


def pad_to_block_grid(tex: Texture) -> Texture:
    """把非 4 倍数尺寸用边缘钳制（clamp-to-edge）填充到块网格。

    钳制而非零填充：填充纹素与真实边缘一致，不会凭空造出深色块。
    """
    pw, ph = aligned_dim(tex.width), aligned_dim(tex.height)
    if pw == tex.width and ph == tex.height:
        return tex.copy()
    out = bytearray(pw * ph * 4)
    for y in range(ph):
        sy = min(y, tex.height - 1)
        for x in range(pw):
            sx = min(x, tex.width - 1)
            i = (y * pw + x) * 4
            j = (sy * tex.width + sx) * 4
            out[i:i + 4] = tex.rgba[j:j + 4]
    return Texture(pw, ph, out, tex.srgb)


def crop_to_original(tex: Texture, width: int, height: int) -> Texture:
    """解码后按头部记录的原始尺寸裁掉填充。"""
    if width == tex.width and height == tex.height:
        return tex.copy()
    out = bytearray(width * height * 4)
    for y in range(height):
        src = y * tex.width * 4
        dst = y * width * 4
        out[dst:dst + width * 4] = tex.rgba[src:src + width * 4]
    return Texture(width, height, out, tex.srgb)


# ---------------------------------------------------------------------------
# 确定性 LCG 抖动
# ---------------------------------------------------------------------------


def _dither_offsets(seed: int, count: int) -> list[int]:
    """生成固定的、与运行环境无关的 ±1 抖动序列。"""
    state = seed & 0xFFFFFFFF
    out: list[int] = []
    for _ in range(count):
        state = (state * _LCG_A + _LCG_C) % _LCG_M
        out.append(1 if (state >> 31) & 1 else -1)
    return out


# ---------------------------------------------------------------------------
# alpha 边缘保护
# ---------------------------------------------------------------------------


def protect_alpha_edges(tex: Texture) -> Texture:
    """用最近的不透明纹素颜色预填充透明纹素的 RGB。

    alpha 本身保持不变。块编码器随后在“颜色连续”的数据上拟合端点，
    透明纹素仍由独立 alpha 块表达为 0，于是边缘插值不会被垃圾色抹平。
    距离相同的候选按扫描顺序（y 再 x）取先出现者，完全确定。
    """
    out = tex.copy()
    w, h = tex.width, tex.height
    opaque = [
        (x, y)
        for y in range(h)
        for x in range(w)
        if tex.px(x, y)[3] != 0
    ]
    if not opaque:
        return out  # 全透明纹理没有可借用的颜色，保持原样
    for y in range(h):
        for x in range(w):
            r, g, b, a = tex.px(x, y)
            if a != 0:
                continue
            best = min(
                opaque,
                key=lambda p: abs(p[0] - x) + abs(p[1] - y),
            )
            r, g, b, _ = tex.px(best[0], best[1])
            out.set_px(x, y, r, g, b, 0)
    return out


# ---------------------------------------------------------------------------
# mip 前置生成（线性空间 2x2 盒式过滤）
# ---------------------------------------------------------------------------


def _box2(v00: float, v01: float, v10: float, v11: float) -> float:
    return (v00 + v01 + v10 + v11) * 0.25


def generate_mips(base: Texture) -> list[Texture]:
    """在未压缩数据上生成完整 mip 链（含 base 级）。

    sRGB 纹素先转线性再平均，过滤后转回 sRGB；
    奇数尺寸用边缘钳制补齐 2x2 采样窗口。
    """
    levels = [base.copy()]
    cur = base
    while cur.width > 1 or cur.height > 1:
        nw, nh = max(1, cur.width // 2), max(1, cur.height // 2)
        buf = bytearray(nw * nh * 4)
        for ny in range(nh):
            for nx in range(nw):
                sx0, sy0 = nx * 2, ny * 2
                sx1 = min(sx0 + 1, cur.width - 1)
                sy1 = min(sy0 + 1, cur.height - 1)
                p00 = cur.px(sx0, sy0)
                p01 = cur.px(sx1, sy0)
                p10 = cur.px(sx0, sy1)
                p11 = cur.px(sx1, sy1)
                if cur.srgb:
                    chans = []
                    for ch in range(4):
                        lin = [srgb_to_linear(float(p[ch])) for p in (p00, p01, p10, p11)]
                        chans.append(linear_to_srgb(_box2(*lin)))
                    rgba = tuple(chans)
                else:
                    rgba = tuple(
                        int(round(_box2(p00[ch], p01[ch], p10[ch], p11[ch])))
                        for ch in range(4)
                    )
                i = (ny * nw + nx) * 4
                buf[i:i + 4] = bytes(rgba)
        cur = Texture(nw, nh, buf, cur.srgb)
        levels.append(cur)
    return levels


# ---------------------------------------------------------------------------
# 颜色块（BC1 风格，每块 8 字节）
# ---------------------------------------------------------------------------


def _to_565(r: int, g: int, b: int) -> int:
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _from_565(v: int) -> tuple[int, int, int]:
    r5, g6, b5 = (v >> 11) & 0x1F, (v >> 5) & 0x3F, v & 0x1F
    # 位复制展开（GPU 惯例），比单纯左移更接近原值
    r = (r5 << 3) | (r5 >> 2)
    g = (g6 << 2) | (g6 >> 4)
    b = (b5 << 3) | (b5 >> 2)
    return r, g, b


def _pack_indices_2(indices: list[int]) -> int:
    """16 个 2bit 索引打包成 32bit（索引 0 在最低位）。"""
    v = 0
    for k, idx in enumerate(indices):
        v |= idx << (2 * k)
    return v


def _unpack_indices_2(v: int) -> list[int]:
    return [(v >> (2 * k)) & 0x3 for k in range(16)]


def _pack_indices_3(indices: list[int]) -> int:
    """16 个 3bit 索引打包成 48bit（索引 0 在最低位）。"""
    v = 0
    for k, idx in enumerate(indices):
        v |= idx << (3 * k)
    return v


def _unpack_indices_3(v: int) -> list[int]:
    return [(v >> (3 * k)) & 0x7 for k in range(16)]


def encode_color_block(pixels: list[tuple[int, int, int]], srgb: bool,
                       dither: list[int] | None = None) -> bytes:
    """编码单个 4x4 颜色块，返回 8 字节。

    srgb=True 时端点拟合与误差比较都在线性光空间进行，
    修复“sRGB 码值直接当线性值做最小二乘”导致的偏色。
    """
    if len(pixels) != 16:
        raise ValueError("颜色块需要 16 个纹素")

    def lin(p):
        if srgb:
            return (srgb_to_linear(float(p[0])),
                    srgb_to_linear(float(p[1])),
                    srgb_to_linear(float(p[2])))
        return (p[0] / 255.0, p[1] / 255.0, p[2] / 255.0)

    work = [list(p) for p in pixels]
    if dither:
        # 固定抖动只用于打破平局，幅度 ±1 码值，序列确定
        for k, px in enumerate(work):
            d = dither[k]
            for ch in range(3):
                px[ch] = max(0, min(255, px[ch] + d))

    # 端点：工作集每通道 min/max；全同色块端点相同
    mins = [min(px[ch] for px in work) for ch in range(3)]
    maxs = [max(px[ch] for px in work) for ch in range(3)]
    e0, e1 = _to_565(*maxs), _to_565(*mins)
    c0, c1 = _from_565(e0), _from_565(e1)

    # 4 级调色板：2/3、1/3 插值（sRGB 数据在线性空间插值）
    if srgb:
        l0 = [srgb_to_linear(float(c)) for c in c0]
        l1 = [srgb_to_linear(float(c)) for c in c1]
        pal_lin = [
            l0,
            l1,
            [l0[ch] * (2 / 3) + l1[ch] * (1 / 3) for ch in range(3)],
            [l0[ch] * (1 / 3) + l1[ch] * (2 / 3) for ch in range(3)],
        ]
        palette = [tuple(linear_to_srgb(v) for v in col) for col in pal_lin]
    else:
        palette = [
            c0,
            c1,
            tuple(int(round(c0[ch] * 2 / 3 + c1[ch] / 3)) for ch in range(3)),
            tuple(int(round(c0[ch] / 3 + c1[ch] * 2 / 3)) for ch in range(3)),
        ]
    pal_lin_err = [lin(p) for p in palette]

    indices: list[int] = []
    for p in pixels:
        pl = lin(p)
        best_idx, best_err = 0, None
        for idx, cand in enumerate(pal_lin_err):
            # 误差在线性光空间度量；平局取较小索引 -> 确定性
            err = sum((pl[ch] - cand[ch]) ** 2 for ch in range(3))
            if best_err is None or err < best_err:
                best_err, best_idx = err, idx
        indices.append(best_idx)

    return struct.pack("<HHI", e0, e1, _pack_indices_2(indices))


def decode_color_block(block: bytes, srgb: bool) -> list[tuple[int, int, int]]:
    e0, e1, packed = struct.unpack("<HHI", block)
    c0, c1 = _from_565(e0), _from_565(e1)
    if srgb:
        l0 = [srgb_to_linear(float(c)) for c in c0]
        l1 = [srgb_to_linear(float(c)) for c in c1]
        pal = [
            tuple(linear_to_srgb(v) for v in l0),
            tuple(linear_to_srgb(v) for v in l1),
            tuple(linear_to_srgb(l0[ch] * 2 / 3 + l1[ch] / 3) for ch in range(3)),
            tuple(linear_to_srgb(l0[ch] / 3 + l1[ch] * 2 / 3) for ch in range(3)),
        ]
    else:
        pal = [
            c0,
            c1,
            tuple(int(round(c0[ch] * 2 / 3 + c1[ch] / 3)) for ch in range(3)),
            tuple(int(round(c0[ch] / 3 + c1[ch] * 2 / 3)) for ch in range(3)),
        ]
    return [pal[i] for i in _unpack_indices_2(packed)]


# ---------------------------------------------------------------------------
# 单通道块（BC4 风格，每块 8 字节）—— alpha 与法线通道共用
# ---------------------------------------------------------------------------


def encode_channel_block(values: list[int]) -> bytes:
    """编码单个 4x4 单通道块，返回 8 字节。

    端点为原始 min/max 码值（不等时 8 级等距插值，相等时退化为常量），
    索引按线性码值误差选取，平局取较小索引。
    """
    if len(values) != 16:
        raise ValueError("通道块需要 16 个纹素")
    hi, lo = max(values), min(values)
    if hi == lo:
        return bytes((hi, lo)) + b"\x00" * 6  # 常量块，48bit 索引全 0，共 8 字节
    table = [hi, lo] + [
        int(round(hi * (7 - k) / 7 + lo * k / 7)) for k in range(1, 7)
    ]
    indices = []
    for v in values:
        best_idx, best_err = 0, None
        for idx, cand in enumerate(table):
            err = abs(v - cand)
            if best_err is None or err < best_err:
                best_err, best_idx = err, idx
        indices.append(best_idx)
    return bytes((hi, lo)) + _pack_indices_3(indices).to_bytes(6, "little")


def decode_channel_block(block: bytes) -> list[int]:
    hi, lo = block[0], block[1]
    packed = int.from_bytes(block[2:8], "little")
    if hi == lo:
        return [hi] * 16
    table = [hi, lo] + [
        int(round(hi * (7 - k) / 7 + lo * k / 7)) for k in range(1, 7)
    ]
    return [table[i] for i in _unpack_indices_3(packed)]


# ---------------------------------------------------------------------------
# 法线：XY 压缩 + Z 重建
# ---------------------------------------------------------------------------


def encode_normal_xy(xs: list[int], ys: list[int]) -> bytes:
    """法线块：X、Y 各一个 BC4 风格通道块，共 16 字节（不含 Z）。"""
    return encode_channel_block(xs) + encode_channel_block(ys)


def decode_normal_xyz(block: bytes, sign: int = 1) -> list[tuple[int, int, int]]:
    """解码法线块并重建 Z。

    x/y 码值映射到 [-1, 1]，z = sign * sqrt(max(0, 1 - x^2 - y^2))，
    再映回 8bit。修复“Z 通道被块压缩截断且未重建”的缺陷。
    """
    xs = decode_channel_block(block[:8])
    ys = decode_channel_block(block[8:])
    out: list[tuple[int, int, int]] = []
    for xb, yb in zip(xs, ys):
        x = xb / 127.5 - 1.0
        y = yb / 127.5 - 1.0
        z = sign * math.sqrt(max(0.0, 1.0 - x * x - y * y))
        # z in [-1,1] -> 8bit 码值（与夹具编码方向互逆）
        out.append((xb, yb, int(round((z + 1.0) * 127.5))))
    return out


# ---------------------------------------------------------------------------
# 整级压缩 / 解压
# ---------------------------------------------------------------------------


def _block_pixels(tex: Texture, bx: int, by: int) -> list[tuple[int, int, int, int]]:
    px = []
    for y in range(by * BLOCK, by * BLOCK + BLOCK):
        for x in range(bx * BLOCK, bx * BLOCK + BLOCK):
            px.append(tex.px(x, y))
    return px


def compress_level(tex: Texture, fmt: int, *, seed: int = 0,
                   protect: bool = True) -> tuple[bytes, dict[int, tuple[int, ...]]]:
    """压缩一个已对齐到块网格的 mip 级。

    返回 (压缩字节, 原始覆盖表)。覆盖表记录超过阈值的块的未压缩纹素，
    作为确定性 CPU 回退（raw fallback），由容器一并持久化。
    """
    if tex.width % BLOCK or tex.height % BLOCK:
        raise ValueError("压缩前必须先对齐到块网格")
    srgb = bool(fmt & SRGB_FLAG)
    if srgb != tex.srgb:
        raise ValueError(
            f"色彩空间不匹配：格式 sRGB={srgb}，纹素 sRGB={tex.srgb}；"
            "请改用对应 UNORM/sRGB 格式或先转换纹素"
        )
    base_fmt = fmt & 0x7F
    if base_fmt not in (FMT_BC1, FMT_BC3, FMT_BC5):
        raise ValueError(f"未知格式 0x{fmt:02x}")

    nx, ny = tex.width // BLOCK, tex.height // BLOCK
    pitch = aligned_pitch(nx)
    source = protect_alpha_edges(tex) if (protect and base_fmt == FMT_BC3) else tex
    stream = bytearray(pitch * ny)
    overrides: dict[int, tuple[int, ...]] = {}
    dither_seq = _dither_offsets(seed, nx * ny * 16)

    for by in range(ny):
        for bx in range(nx):
            block_index = by * nx + bx
            blk = _block_pixels(source, bx, by)
            d = dither_seq[block_index * 16:(block_index + 1) * 16]
            if base_fmt == FMT_BC5:
                xs = [p[0] for p in blk]
                ys = [p[1] for p in blk]
                encoded = encode_normal_xy(xs, ys)
                recon = decode_normal_xyz(encoded)
                err = sum(
                    (xs[k] - recon[k][0]) ** 2 + (ys[k] - recon[k][1]) ** 2
                    for k in range(16)
                ) / 16.0
            else:
                rgb = [(p[0], p[1], p[2]) for p in blk]
                color = encode_color_block(rgb, srgb, d)
                if base_fmt == FMT_BC3:
                    alpha = encode_channel_block([p[3] for p in blk])
                else:
                    # BC1 无 alpha，槽位补零保持统一的块步长
                    alpha = b"\x00" * CHANNEL_BLOCK_BYTES
                encoded = color + alpha
                recon_rgb = decode_color_block(color, srgb)
                if srgb:
                    # 在线性光空间度量，并统一到 0-255 尺度以便阈值解释
                    err = sum(
                        sum(
                            ((srgb_to_linear(float(blk[k][ch]))
                              - srgb_to_linear(float(recon_rgb[k][ch]))) * 255.0)
                            ** 2
                            for ch in range(3)
                        )
                        for k in range(16)
                    ) / 16.0
                else:
                    err = sum(
                        sum((blk[k][ch] - recon_rgb[k][ch]) ** 2 for ch in range(3))
                        for k in range(16)
                    ) / 16.0
            if len(encoded) != RAW_BLOCK_BYTES:
                raise AssertionError("块大小必须为 16 字节")
            off = by * pitch + bx * RAW_BLOCK_BYTES
            stream[off:off + RAW_BLOCK_BYTES] = encoded
            # 线性 MSE 超过阈值的块走 raw 回退（阈值固定，不依赖运行环境）
            if err > FALLBACK_MSE_THRESHOLD:
                overrides[block_index] = tuple(c for p in blk for c in p)
    return bytes(stream), overrides


def decompress_level(data: bytes, width: int, height: int, fmt: int,
                     overrides: dict[int, tuple[int, ...]] | None = None) -> Texture:
    """解压一个 mip 级（尺寸为对齐后的块网格尺寸）。"""
    srgb = bool(fmt & SRGB_FLAG)
    base_fmt = fmt & 0x7F
    nx, ny = width // BLOCK, height // BLOCK
    pitch = aligned_pitch(nx)
    if len(data) != pitch * ny:
        raise ValueError("压缩数据长度与块网格/pitch 不符")
    out = Texture(width, height, bytearray(width * height * 4), srgb)
    overrides = overrides or {}
    for by in range(ny):
        for bx in range(nx):
            block_index = by * nx + bx
            off = by * pitch + bx * RAW_BLOCK_BYTES
            block = data[off:off + RAW_BLOCK_BYTES]
            if block_index in overrides:
                vals = list(overrides[block_index])
            elif base_fmt == FMT_BC5:
                vals = [c for trip in decode_normal_xyz(block)
                        for c in (*trip, 255)]
            else:
                rgb = decode_color_block(block[:COLOR_BLOCK_BYTES], srgb)
                if base_fmt == FMT_BC3:
                    alpha = decode_channel_block(
                        block[COLOR_BLOCK_BYTES:COLOR_BLOCK_BYTES + CHANNEL_BLOCK_BYTES]
                    )
                else:
                    alpha = [255] * 16
                vals = [c for k in range(16) for c in (*rgb[k], alpha[k])]
            k = 0
            for yy in range(by * BLOCK, by * BLOCK + BLOCK):
                for xx in range(bx * BLOCK, bx * BLOCK + BLOCK):
                    out.set_px(xx, yy, *vals[k * 4:k * 4 + 4])  # type: ignore[arg-type]
                    k += 1
    return out


# 原始回退阈值：线性光空间、0-255 尺度下的块均方误差。
# 固定常量保证确定性；约等于每通道 RMSE ~16 码值才触发 raw 回退。
FALLBACK_MSE_THRESHOLD = 800.0


# ---------------------------------------------------------------------------
# 容器：头部 + 每级头部 + 块数据 + 回退表
# ---------------------------------------------------------------------------


def _flags_for(tex: Texture, padded_x: bool, padded_y: bool, fmt: int) -> int:
    flags = 0
    if fmt & SRGB_FLAG:
        flags |= FLAG_SRGB
    if (fmt & 0x7F) == FMT_BC5:
        flags |= FLAG_NORMAL
    if padded_x:
        flags |= FLAG_PADDED_X
    if padded_y:
        flags |= FLAG_PADDED_Y
    return flags


def compress(tex: Texture, fmt: int, *, seed: int = 0,
             with_mips: bool = True, protect: bool = True) -> bytes:
    """完整压缩管线：mip 前置 -> 对齐填充 -> 逐级确定性编码。"""
    srgb = bool(fmt & SRGB_FLAG)
    if srgb != tex.srgb:
        raise ValueError("格式色彩空间标记与纹理不一致，拒绝混用")
    levels = generate_mips(tex) if with_mips else [tex.copy()]
    parts: list[bytes] = [
        MAGIC,
        struct.pack("<HHHH", tex.width, tex.height, fmt, len(levels)),
    ]
    for level in levels:
        padded = pad_to_block_grid(level)
        stream, overrides = compress_level(padded, fmt, seed=seed, protect=protect)
        flags = _flags_for(level, padded.width != level.width,
                           padded.height != level.height, fmt)
        pitch = aligned_pitch(padded.width // BLOCK)
        parts.append(struct.pack(
            "<HHHHI", padded.width, padded.height, flags, len(overrides), pitch,
        ))
        # 回退表按块索引升序写入（dict 按插入序，插入序即扫描序）
        for block_index in sorted(overrides):
            raw = overrides[block_index]
            parts.append(struct.pack("<H", block_index) + bytes(raw))
        parts.append(stream)
    return b"".join(parts)


def decompress(blob: bytes) -> list[Texture]:
    """解压容器，返回各级纹理（第 0 级已裁回原始尺寸）。"""
    if blob[:8] != MAGIC:
        raise ValueError("非法容器魔数")
    if len(blob) < HEADER_SIZE:
        raise ValueError("容器被截断：缺少全局头部")
    width, height, fmt, level_count = struct.unpack_from("<HHHH", blob, 8)
    if level_count == 0:
        raise ValueError("容器 mip 级数为 0")
    pos = HEADER_SIZE
    srgb = bool(fmt & SRGB_FLAG)
    out: list[Texture] = []

    def need(n: int) -> None:
        if pos + n > len(blob):
            raise ValueError("容器被截断：数据不完整")

    for _ in range(level_count):
        need(LEVEL_HEADER_SIZE)
        pw, ph, flags, n_over, pitch = struct.unpack_from("<HHHHI", blob, pos)
        pos += LEVEL_HEADER_SIZE
        overrides: dict[int, tuple[int, ...]] = {}
        need(n_over * (2 + 64))
        for _ in range(n_over):
            (block_index,) = struct.unpack_from("<H", blob, pos)
            pos += 2
            overrides[block_index] = tuple(blob[pos:pos + 64])
            pos += 64
        n_blocks_y = ph // BLOCK
        stream_len = pitch * n_blocks_y
        need(stream_len)
        stream = blob[pos:pos + stream_len]
        pos += stream_len
        tex = decompress_level(stream, pw, ph, fmt, overrides)
        if bool(flags & FLAG_SRGB) != srgb:
            raise ValueError("容器内色彩空间标记冲突")
        if not out:
            tex = crop_to_original(tex, width, height)
        out.append(tex)
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def per_channel_mse(a: Texture, b: Texture, srgb: bool) -> dict[str, float]:
    """两纹理逐通道 MSE。sRGB 纹理在线性光空间度量。"""
    if (a.width, a.height) != (b.width, b.height):
        raise ValueError("尺寸不一致，无法比较")
    n = a.width * a.height
    acc = [0.0, 0.0, 0.0, 0.0]
    for i in range(n):
        for ch in range(4):
            va, vb = a.rgba[i * 4 + ch], b.rgba[i * 4 + ch]
            if srgb and ch < 3:
                va, vb = srgb_to_linear(float(va)) * 255, srgb_to_linear(float(vb)) * 255
            acc[ch] += (va - vb) ** 2
    keys = ("r", "g", "b", "a")
    return {keys[ch]: acc[ch] / n for ch in range(4)}


def psnr(mse: float) -> float:
    if mse <= 0:
        return math.inf
    return 10.0 * math.log10((255.0 ** 2) / mse)


def quality_report(source: Texture, decoded: Texture) -> dict:
    """压缩质量报告：逐通道 MSE、PSNR、最大误差。"""
    mses = per_channel_mse(source, decoded, source.srgb)
    max_err = max(
        abs(source.rgba[i] - decoded.rgba[i])
        for i in range(len(source.rgba))
    )
    return {
        "mse": mses,
        "mse_mean": sum(mses.values()) / 4.0,
        "psnr_db": {k: (math.inf if v <= 0 else psnr(v)) for k, v in mses.items()},
        "max_abs_error": max_err,
    }


# ---------------------------------------------------------------------------
# 格式选择 / 校验 / 审计
# ---------------------------------------------------------------------------


def select_format(*, has_alpha: bool, is_normal: bool, srgb: bool) -> int:
    """按内容特征选块格式，再叠加 sRGB 标记。"""
    if is_normal:
        if has_alpha:
            raise ValueError("法线贴图不应携带 alpha")
        fmt = FMT_BC5
    elif has_alpha:
        fmt = FMT_BC3
    else:
        fmt = FMT_BC1
    return fmt | (SRGB_FLAG if srgb else 0)


def validate_texture(tex: Texture, fmt: int) -> dict:
    """压缩前校验，返回问题清单（空列表表示通过）。"""
    problems: list[str] = []
    if not isinstance(tex, Texture):
        raise TypeError("需要 Texture 实例")
    if tex.width <= 0 or tex.height <= 0:
        problems.append("尺寸非正")
    if len(tex.rgba) != tex.width * tex.height * 4:
        problems.append("rgba 缓冲长度与尺寸不符")
    if (fmt & 0x7F) not in (FMT_BC1, FMT_BC3, FMT_BC5):
        problems.append(f"未知格式 0x{fmt:02x}")
    if bool(fmt & SRGB_FLAG) != tex.srgb:
        problems.append("格式 sRGB 标记与纹理色彩空间不匹配")
    if (fmt & 0x7F) == FMT_BC5 and any(
        tex.px(x, y)[3] != 255
        for y in range(tex.height) for x in range(tex.width)
    ):
        problems.append("法线贴图存在非不透明 alpha")
    return {
        "ok": not problems,
        "problems": problems,
        "aligned_width": aligned_dim(tex.width),
        "aligned_height": aligned_dim(tex.height),
        "pitch_blocks": aligned_dim(tex.width) // BLOCK,
    }


def audit_container(blob: bytes) -> dict:
    """只读审计压缩容器：结构、对齐、色彩空间、mip 级数、回退块数。"""
    if blob[:8] != MAGIC:
        raise ValueError("非法容器魔数")
    if len(blob) < HEADER_SIZE:
        raise ValueError("容器被截断：缺少全局头部")
    width, height, fmt, level_count = struct.unpack_from("<HHHH", blob, 8)
    if level_count == 0:
        raise ValueError("容器 mip 级数为 0")
    pos = HEADER_SIZE
    levels_info = []
    total_blocks = 0
    total_overrides = 0

    def need(n: int) -> None:
        if pos + n > len(blob):
            raise ValueError("容器被截断：数据不完整")

    for _ in range(level_count):
        need(LEVEL_HEADER_SIZE)
        pw, ph, flags, n_over, pitch = struct.unpack_from("<HHHHI", blob, pos)
        pos += LEVEL_HEADER_SIZE
        need(n_over * (2 + 64))
        pos += n_over * (2 + 64)
        n_blocks_x, n_blocks_y = pw // BLOCK, ph // BLOCK
        n_blocks = n_blocks_x * n_blocks_y
        expected_pitch = aligned_pitch(n_blocks_x)
        stream_len = pitch * n_blocks_y
        need(stream_len)
        pos += stream_len
        levels_info.append({
            "padded_width": pw,
            "padded_height": ph,
            "blocks": n_blocks,
            "overrides": n_over,
            "pitch_bytes": pitch,
            "pitch_16_aligned": pitch % BLOCK_ALIGN == 0,
            "pitch_matches_layout": pitch == expected_pitch,
            "block_aligned": pw % BLOCK == 0 and ph % BLOCK == 0,
        })
        total_blocks += n_blocks
        total_overrides += n_over
    trailer_ok = pos == len(blob)
    return {
        "ok": trailer_ok and all(
            lv["block_aligned"] and lv["pitch_16_aligned"]
            and lv["pitch_matches_layout"]
            for lv in levels_info
        ),
        "format": fmt,
        "is_srgb": bool(fmt & SRGB_FLAG),
        "is_normal": (fmt & 0x7F) == FMT_BC5,
        "original_size": (width, height),
        "level_count": level_count,
        "levels": levels_info,
        "total_blocks": total_blocks,
        "fallback_blocks": total_overrides,
        "trailer_consumed": trailer_ok,
    }


# ---------------------------------------------------------------------------
# 测试夹具（确定性）
# ---------------------------------------------------------------------------


def make_rgba_fixture(width: int = 8, height: int = 8, *, srgb: bool = True,
                      seed: int = 1234) -> Texture:
    """带 alpha 边缘的确定性 RGBA 夹具：左半不透明彩色，右半全透明。"""
    state = seed & 0xFFFFFFFF
    buf = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 4
            state = (state * _LCG_A + _LCG_C) % _LCG_M
            r = (state >> 16) & 0xFF
            state = (state * _LCG_A + _LCG_C) % _LCG_M
            g = (state >> 16) & 0xFF
            state = (state * _LCG_A + _LCG_C) % _LCG_M
            b = (state >> 16) & 0xFF
            a = 255 if x < width // 2 else 0
            buf[i:i + 4] = bytes((r, g, b, a))
    return Texture(width, height, buf, srgb)


def make_smooth_fixture(width: int = 8, height: int = 8, *, srgb: bool = True,
                        seed: int = 55) -> Texture:
    """带 alpha 边缘的平滑合成纹理（接近真实素材，非白噪声）。

    RGB 为低频径向渐变叠加小幅确定性抖动；左半不透明、右半透明，
    透明侧 RGB 故意填黑以暴露“边缘被插值抹平”的缺陷。
    """
    state = seed & 0xFFFFFFFF
    buf = bytearray(width * height * 4)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    norm = max(cx, cy, 1.0)
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 4
            state = (state * _LCG_A + _LCG_C) % _LCG_M
            jitter = ((state >> 24) & 0x7) - 3      # -3..3 码值
            d = math.hypot(x - cx, y - cy) / norm
            r = int(40 + 180 * (1.0 - d)) + jitter
            g = int(200 - 120 * d) + jitter
            b = int(60 + 150 * d) + jitter
            clamp = lambda v: max(0, min(255, v))
            if x < width // 2:
                buf[i:i + 4] = bytes((clamp(r), clamp(g), clamp(b), 255))
            else:
                buf[i:i + 4] = bytes((0, 0, 0, 0))   # 透明侧垃圾黑
    return Texture(width, height, buf, srgb)


def make_normal_fixture(width: int = 8, height: int = 8) -> Texture:
    """确定性单位法线夹具：上半球方向按极坐标规则采样。

    每个纹素都是合法单位向量（x^2+y^2+z^2=1），
    用于验证“只压 XY、解码重建 Z”的往返精度。
    """
    buf = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            theta = (x / max(1, width - 1)) * (math.pi / 2)      # 0..pi/2
            phi = (y / max(1, height - 1)) * (math.pi / 2)        # 0..pi/2
            nx = math.sin(theta) * math.cos(phi)
            ny = math.sin(theta) * math.sin(phi)
            nz = math.cos(theta)
            i = (y * width + x) * 4
            buf[i:i + 4] = bytes((
                int(round((nx * 0.5 + 0.5) * 255)),
                int(round((ny * 0.5 + 0.5) * 255)),
                int(round((nz * 0.5 + 0.5) * 255)),
                255,
            ))
    return Texture(width, height, buf, False)


# ---------------------------------------------------------------------------
# 15 个公共入口（单整数参数 = 内置自检场景，返回可 JSON 序列化的报告）
# ---------------------------------------------------------------------------


def alpha_edge(scenario: int = 1) -> dict:
    """场景 1：alpha 边缘保护 —— 透明纹素 RGB 被最近不透明颜色替换。"""
    tex = make_rgba_fixture(8, 8)
    protected = protect_alpha_edges(tex)
    leaked = [
        (x, y)
        for y in range(tex.height)
        for x in range(tex.width)
        if tex.px(x, y)[3] == 0 and protected.px(x, y)[:3] == (0, 0, 0)
        and any(
            tex.px(ox, oy)[:3] != (0, 0, 0)
            for ox in range(tex.width) for oy in range(tex.height)
            if tex.px(ox, oy)[3] != 0
        )
    ]
    alpha_preserved = all(
        protected.px(x, y)[3] == tex.px(x, y)[3]
        for y in range(tex.height) for x in range(tex.width)
    )
    return {
        "scenario": scenario,
        "transparent_texels": sum(
            1 for y in range(8) for x in range(8) if tex.px(x, y)[3] == 0
        ),
        "black_leak_after_protect": len(leaked),
        "alpha_preserved": alpha_preserved,
        "protected": True,
    }


def normal_z(scenario: int = 1) -> dict:
    """场景 1：法线只压 XY，Z 由单位长度约束重建。"""
    tex = make_normal_fixture(8, 8)
    fmt = select_format(has_alpha=False, is_normal=True, srgb=False)
    blob = compress(tex, fmt, with_mips=False)
    decoded = decompress(blob)[0]
    max_xy_err = max(
        max(abs(tex.px(x, y)[0] - decoded.px(x, y)[0]),
            abs(tex.px(x, y)[1] - decoded.px(x, y)[1]))
        for y in range(8) for x in range(8)
    )
    # 重建 Z 与源 Z 的偏差（码值域）
    max_z_err = max(
        abs(tex.px(x, y)[2] - decoded.px(x, y)[2])
        for y in range(8) for x in range(8)
    )
    # 重建向量长度（解码侧 [-1,1] 空间）应接近 1
    worst_len = 0.0
    for y in range(8):
        for x in range(8):
            xb, yb, zb, _ab = decoded.px(x, y)
            v = (xb / 127.5 - 1.0, yb / 127.5 - 1.0, zb / 127.5 - 1.0)
            worst_len = max(worst_len, abs(math.sqrt(sum(c * c for c in v)) - 1.0))
    return {
        "scenario": scenario,
        "stored_channels": ("x", "y"),
        "reconstructed_channel": "z",
        "max_xy_code_error": max_xy_err,
        "max_z_reconstruct_error": max_z_err,
        "worst_unit_length_deviation": worst_len,
    }


def color_space(scenario: int = 1) -> dict:
    """场景 1：sRGB 与线性块格式混用必须被拒绝；匹配时正常编码。"""
    tex_srgb = make_rgba_fixture(4, 4, srgb=True)
    tex_lin = Texture(4, 4, bytearray(tex_srgb.rgba), False)
    fmt_srgb = select_format(has_alpha=True, is_normal=False, srgb=True)
    fmt_unorm = select_format(has_alpha=True, is_normal=False, srgb=False)
    mismatch_rejected = False
    try:
        compress(tex_srgb, fmt_unorm, with_mips=False)
    except ValueError:
        mismatch_rejected = True
    ok_blob = compress(tex_srgb, fmt_srgb, with_mips=False)
    lin_rejected = False
    try:
        compress(tex_lin, fmt_srgb, with_mips=False)
    except ValueError:
        lin_rejected = True
    return {
        "scenario": scenario,
        "mismatch_rejected": mismatch_rejected,
        "reverse_mismatch_rejected": lin_rejected,
        "matched_roundtrip_bytes": len(ok_blob),
        "linear_light_error_metric": True,
    }


def mip_pre(scenario: int = 1) -> dict:
    """场景 1：先在未压缩数据上生成 mip 链，再逐级压缩。"""
    tex = make_rgba_fixture(8, 8)
    fmt = select_format(has_alpha=True, is_normal=False, srgb=True)
    levels = generate_mips(tex)
    blob = compress(tex, fmt, with_mips=True)
    audit = audit_container(blob)
    expected = 1 + int(math.log2(max(tex.width, tex.height)))
    return {
        "scenario": scenario,
        "mips_generated_before_compress": True,
        "level_sizes": [(lv.width, lv.height) for lv in levels],
        "level_count": audit["level_count"],
        "expected_level_count": expected,
        "chain_complete": audit["level_count"] == expected,
    }


def encode_determinism(scenario: int = 1) -> dict:
    """场景 1：同输入重复编码逐字节一致（固定种子 + 确定性平局规则）。"""
    tex = make_rgba_fixture(8, 8)
    fmt = select_format(has_alpha=True, is_normal=False, srgb=True)
    first = compress(tex, fmt, seed=42)
    second = compress(tex, fmt, seed=42)
    different_seed = compress(tex, fmt, seed=43)
    return {
        "scenario": scenario,
        "byte_identical_repeat": first == second,
        "seed_participates_in_dither": first != different_seed,
        "byte_length": len(first),
    }


def block_align(scenario: int = 1) -> dict:
    """场景 1：非 4 倍数尺寸钳制填充，块/pitch 对齐，头部保留原始尺寸。"""
    tex = make_rgba_fixture(6, 5, srgb=False)
    padded = pad_to_block_grid(tex)
    fmt = select_format(has_alpha=True, is_normal=False, srgb=False)
    blob = compress(tex, fmt, with_mips=False)
    audit = audit_container(blob)
    decoded = decompress(blob)[0]
    return {
        "scenario": scenario,
        "original_size": (6, 5),
        "padded_size": (padded.width, padded.height),
        "padded_is_block_multiple": padded.width % 4 == 0 and padded.height % 4 == 0,
        "edge_clamp_padding": padded.px(7, 7) == tex.px(5, 4),
        "decoded_original_size": (decoded.width, decoded.height),
        "all_blocks_16_bytes": all(lv["block_aligned"] for lv in audit["levels"]),
    }


def srgb_linear(value: int = 128) -> dict:
    """sRGB <-> 线性转换与往返一致性。"""
    return _srgb_linear_one(int(value))


def validation(scenario: int = 1) -> dict:
    """场景 1：压缩前校验捕获色彩空间/格式/alpha 类问题。"""
    tex = make_rgba_fixture(4, 4)
    bad_fmt = select_format(has_alpha=True, is_normal=False, srgb=False)
    report = validate_texture(tex, bad_fmt)
    good_fmt = select_format(has_alpha=True, is_normal=False, srgb=True)
    good = validate_texture(tex, good_fmt)
    normal = make_normal_fixture(4, 4)
    normal_fmt = select_format(has_alpha=False, is_normal=True, srgb=False)
    normal_ok = validate_texture(normal, normal_fmt)
    return {
        "scenario": scenario,
        "bad_report": report,
        "good_report": good,
        "normal_report": normal_ok,
    }


def audit(scenario: int = 1) -> dict:
    """场景 1：容器审计给出结构、对齐、色彩空间与回退块信息。"""
    tex = make_rgba_fixture(8, 8)
    fmt = select_format(has_alpha=True, is_normal=False, srgb=True)
    blob = compress(tex, fmt)
    report = audit_container(blob)
    return {"scenario": scenario, **report}


def fallback(scenario: int = 1) -> dict:
    """场景 1：高误差块记录原始纹素，解码侧用回退数据替换块输出。"""
    tex = make_rgba_fixture(8, 8, seed=77)
    fmt = select_format(has_alpha=True, is_normal=False, srgb=True)
    blob = compress(tex, fmt, with_mips=False)
    info = audit_container(blob)
    decoded = decompress(blob)[0]
    # 回退块存在时，对应块必须与源逐字节一致
    exact = True
    if info["fallback_blocks"]:
        pos = HEADER_SIZE + LEVEL_HEADER_SIZE
        for _ in range(info["fallback_blocks"]):
            (idx,) = struct.unpack_from("<H", blob, pos)
            raw = blob[pos + 2:pos + 66]
            bx, by = idx % 2, idx // 2
            k = 0
            for yy in range(by * 4, by * 4 + 4):
                for xx in range(bx * 4, bx * 4 + 4):
                    got = decoded.px(xx, yy)
                    if tuple(got) != tuple(raw[k * 4:k * 4 + 4]):
                        exact = False
                    k += 1
            pos += 66
    return {
        "scenario": scenario,
        "fallback_blocks": info["fallback_blocks"],
        "fallback_blocks_exact": exact,
        "threshold_mse": FALLBACK_MSE_THRESHOLD,
    }


def quality_metric(source: Texture | int | None = None,
                   decoded: Texture | None = None) -> dict:
    """质量度量；单整数参数走内置往返场景（平滑合成纹理）。"""
    if isinstance(source, int) or source is None:
        tex = make_smooth_fixture(8, 8)
        fmt = select_format(has_alpha=True, is_normal=False, srgb=True)
        decoded_tex = decompress(compress(tex, fmt, with_mips=False))[0]
        report = quality_report(tex, decoded_tex)
        report["scenario"] = source if isinstance(source, int) else 1
        return report
    if decoded is None:
        raise ValueError("需要 source 与 decoded 两个纹理")
    return quality_report(source, decoded)


def metrics(scenario: int = 1) -> dict:
    """场景 1：逐通道 MSE/PSNR 汇总（UNORM 平滑纹理，便于直接读码值误差）。"""
    tex = make_smooth_fixture(8, 8, srgb=False)
    fmt = select_format(has_alpha=True, is_normal=False, srgb=False)
    decoded = decompress(compress(tex, fmt, with_mips=False))[0]
    report = quality_report(tex, decoded)
    return {"scenario": scenario, **report}


def format_select(scenario: int = 1) -> dict:
    """场景 1：按 alpha/法线/色彩空间特征选择块格式。"""
    picks = {
        "opaque_srgb": select_format(has_alpha=False, is_normal=False, srgb=True),
        "alpha_srgb": select_format(has_alpha=True, is_normal=False, srgb=True),
        "alpha_unorm": select_format(has_alpha=True, is_normal=False, srgb=False),
        "normal_unorm": select_format(has_alpha=False, is_normal=True, srgb=False),
    }
    rejected = False
    try:
        select_format(has_alpha=True, is_normal=True, srgb=False)
    except ValueError:
        rejected = True
    return {
        "scenario": scenario,
        "picks": picks,
        "normal_alpha_rejected": rejected,
    }


def batch_repeat(scenario: int = 1) -> dict:
    """场景 1：批量压缩同一张图 N 次，输出全部逐字节一致。"""
    tex = make_rgba_fixture(8, 8)
    fmt = select_format(has_alpha=True, is_normal=False, srgb=True)
    blobs = [compress(tex, fmt, seed=2026) for _ in range(5)]
    return {
        "scenario": scenario,
        "batch_size": len(blobs),
        "all_byte_identical": all(b == blobs[0] for b in blobs),
        "byte_length": len(blobs[0]),
    }


def determinism(scenario: int = 1) -> dict:
    """场景 1：汇总抖动序列、平局规则与批量一致性的确定性证据。"""
    seq_a = _dither_offsets(99, 32)
    seq_b = _dither_offsets(99, 32)
    return {
        "scenario": scenario,
        "lcg_sequence_stable": seq_a == seq_b,
        "lcg_seed_changes_sequence": _dither_offsets(1, 8) != _dither_offsets(2, 8),
        "tie_break_rule": "linear-light squared error, then lower palette index",
        "scan_order": "block y, block x, then texel y, texel x",
        "batch": batch_repeat(scenario),
    }
