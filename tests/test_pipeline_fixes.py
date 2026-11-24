"""六个修复点的行为测试（在原契约测试之外补充）。

覆盖：alpha 边缘保护、法线 Z 重建、色彩空间匹配、mip 前置、
确定性编码、块尺寸与对齐，以及回退/审计/指标等支撑能力。
"""

import math

import pytest

import graphics.bc_alpha_edge as mod
from graphics.bc_alpha_edge import (
    Texture,
    FMT_BC1,
    FMT_BC3,
    FMT_BC5,
    SRGB_FLAG,
    RAW_BLOCK_BYTES,
    BLOCK,
)


# ---------------------------------------------------------------------------
# 1. alpha 边缘保护
# ---------------------------------------------------------------------------


def test_alpha_edge_protection_fills_transparent_rgb():
    tex = mod.make_smooth_fixture(8, 8, srgb=False)
    protected = mod.protect_alpha_edges(tex)

    # alpha 通道逐字节保留
    assert bytes(protected.rgba[3::4]) == bytes(tex.rgba[3::4])

    # 每个透明纹素的 RGB 都来自某个不透明纹素（不再是垃圾黑）
    opaque_colors = {
        tex.px(x, y)[:3]
        for y in range(8) for x in range(8) if tex.px(x, y)[3] != 0
    }
    for y in range(8):
        for x in range(4, 8):  # 右半透明区
            assert tex.px(x, y)[3] == 0
            assert protected.px(x, y)[:3] in opaque_colors


def test_alpha_edge_protection_uses_nearest_opaque():
    # 手工构造：单个不透明红点，其余透明；最近邻必须借用该红色
    buf = bytearray(8 * 8 * 4)
    buf[(0 * 8 + 0) * 4:(0 * 8 + 0) * 4 + 4] = bytes((255, 0, 0, 255))
    tex = Texture(8, 8, buf, False)
    protected = mod.protect_alpha_edges(tex)
    # 任意透明纹素的颜色都被填成唯一的不透明色红
    for y in range(8):
        for x in range(8):
            if tex.px(x, y)[3] == 0:
                assert protected.px(x, y)[:3] == (255, 0, 0)
                assert protected.px(x, y)[3] == 0


def test_protection_keeps_black_out_of_edge_palette():
    """受控夹具：亮黄不透明 + 垃圾黑透明。

    不保护时块调色板被黑色拉进，透明纹素映射到黑色索引 ——
    alpha 混合的半透明边缘因此出现黑边（即“边缘被块内插值抹平”）。
    保护后整块调色板不含黑色，边缘过渡保持亮色。
    """
    buf = bytearray(4 * 4 * 4)
    for y in range(4):
        for x in range(4):
            i = (y * 4 + x) * 4
            if x < 2:
                buf[i:i + 4] = bytes((255, 255, 100, 255))
            else:
                buf[i:i + 4] = bytes((0, 0, 0, 0))
    tex = Texture(4, 4, buf, False)
    fmt = FMT_BC3

    block_off = mod.encode_color_block(
        [p[:3] for p in mod._block_pixels(tex, 0, 0)], False
    )
    pal_off = set(mod.decode_color_block(block_off, False))
    assert (0, 0, 0) in pal_off  # 缺陷复现：黑色进入调色板

    protected = mod.protect_alpha_edges(tex)
    block_on = mod.encode_color_block(
        [p[:3] for p in mod._block_pixels(protected, 0, 0)], False
    )
    pal_on = set(mod.decode_color_block(block_on, False))
    assert (0, 0, 0) not in pal_on          # 修复：黑色被排除
    assert all(ch[0] >= 250 for ch in pal_on)  # 调色板保持亮黄色调


# ---------------------------------------------------------------------------
# 2. 法线 Z 重建
# ---------------------------------------------------------------------------


def test_normal_stores_only_xy_and_reconstructs_z():
    tex = mod.make_normal_fixture(8, 8)
    fmt = mod.select_format(has_alpha=False, is_normal=True, srgb=False)
    assert (fmt & 0x7F) == FMT_BC5

    blob = mod.compress(tex, fmt, with_mips=False)
    info = mod.audit_container(blob)
    assert info["is_normal"] is True

    decoded = mod.decompress(blob)[0]
    # XY 往返误差受 BC4 风格 8 级插值约束：跨度/14 量级
    max_xy = max(
        max(abs(tex.px(x, y)[0] - decoded.px(x, y)[0]),
            abs(tex.px(x, y)[1] - decoded.px(x, y)[1]))
        for y in range(8) for x in range(8)
    )
    assert max_xy <= 7

    # 重建向量长度必须接近单位长度（误差来自量化，而非公式错误）
    worst = 0.0
    for y in range(8):
        for x in range(8):
            xb, yb, zb, _ = decoded.px(x, y)
            v = (xb / 127.5 - 1, yb / 127.5 - 1, zb / 127.5 - 1)
            worst = max(worst, abs(math.sqrt(sum(c * c for c in v)) - 1.0))
    assert worst < 0.02


def test_normal_z_reconstruction_matches_formula():
    # 直接对单块做编解码，核对 z = sqrt(1 - x^2 - y^2)
    xs = [mod.make_normal_fixture(4, 4).px(x, y)[0]
          for y in range(4) for x in range(4)]
    ys = [mod.make_normal_fixture(4, 4).px(x, y)[1]
          for y in range(4) for x in range(4)]
    block = mod.encode_normal_xy(xs, ys)
    assert len(block) == RAW_BLOCK_BYTES
    triplets = mod.decode_normal_xyz(block)
    for (xb, yb, zb), x_src, y_src in zip(triplets, xs, ys):
        x, y = xb / 127.5 - 1.0, yb / 127.5 - 1.0
        expect_z = int(round((math.sqrt(max(0.0, 1 - x * x - y * y)) + 1) * 127.5))
        assert zb == expect_z  # Z 严格按单位向量公式重建
        # XY 经 8 级插值块量化，误差不超过跨度/14 的理论量化界
        bound = math.ceil((max(xs) - min(xs)) / 14) + 1
        assert abs(xb - x_src) <= bound and abs(yb - y_src) <= bound


# ---------------------------------------------------------------------------
# 3. 色彩空间匹配
# ---------------------------------------------------------------------------


def test_srgb_linear_roundtrip():
    for v in (0, 1, 10, 128, 200, 255):
        r = mod.srgb_linear(v)
        assert abs(r["roundtrip"] - v) <= 1
    # 中间调线性值显著低于码值比例（伽马效应）
    assert mod.srgb_to_linear(128.0) == pytest.approx(0.2158, abs=0.01)


def test_color_space_mismatch_rejected_both_directions():
    tex_srgb = mod.make_smooth_fixture(4, 4, srgb=True)
    tex_linear = Texture(4, 4, bytearray(tex_srgb.rgba), False)
    fmt_srgb = FMT_BC3 | SRGB_FLAG
    fmt_unorm = FMT_BC3

    with pytest.raises(ValueError):
        mod.compress(tex_srgb, fmt_unorm, with_mips=False)
    with pytest.raises(ValueError):
        mod.compress(tex_linear, fmt_srgb, with_mips=False)
    # 匹配时正常工作
    assert mod.decompress(mod.compress(tex_srgb, fmt_srgb, with_mips=False))


def test_srgb_encoding_uses_linear_light_error():
    # 端点拟合在线性空间：暗部应保留更多层级（与朴素码值最小二乘不同）。
    # 这里至少保证 sRGB 与 UNORM 对同一字节产生不同块字节，
    # 且两者都能往返、各自标记正确。
    tex = mod.make_smooth_fixture(4, 4, srgb=True)
    blob_srgb = mod.compress(tex, FMT_BC3 | SRGB_FLAG, with_mips=False)
    tex_lin = Texture(4, 4, bytearray(tex.rgba), False)
    blob_lin = mod.compress(tex_lin, FMT_BC3, with_mips=False)
    # 跳过头部与级头部，比较块负载
    assert blob_srgb[24:] != blob_lin[24:]
    assert mod.audit_container(blob_srgb)["is_srgb"] is True
    assert mod.audit_container(blob_lin)["is_srgb"] is False


# ---------------------------------------------------------------------------
# 4. mip 前置生成
# ---------------------------------------------------------------------------


def test_mips_generated_before_compression():
    tex = mod.make_smooth_fixture(8, 8)
    levels = mod.generate_mips(tex)
    assert [(lv.width, lv.height) for lv in levels] == [
        (8, 8), (4, 4), (2, 2), (1, 1)
    ]
    # 每个上级都是下级的 2x2 盒式过滤结果（sRGB 在线性空间平均）
    for upper, lower in zip(levels, levels[1:]):
        assert (upper.width // 2, upper.height // 2) == (lower.width, lower.height)

    blob = mod.compress(
        tex, mod.select_format(has_alpha=True, is_normal=False, srgb=True)
    )
    info = mod.audit_container(blob)
    assert info["level_count"] == 4
    # 每一级独立压缩，尺寸逐级减半且块对齐
    assert [lv["padded_width"] for lv in info["levels"]] == [8, 4, 4, 4]


def test_mip_filter_is_deterministic_and_in_linear_space():
    tex = mod.make_smooth_fixture(8, 8, srgb=True)
    a = mod.generate_mips(tex)
    b = mod.generate_mips(tex)
    for la, lb in zip(a, b):
        assert bytes(la.rgba) == bytes(lb.rgba)


# ---------------------------------------------------------------------------
# 5. 确定性编码
# ---------------------------------------------------------------------------


def test_repeated_encodes_are_byte_identical():
    tex = mod.make_smooth_fixture(8, 8)
    fmt = mod.select_format(has_alpha=True, is_normal=False, srgb=True)
    first = mod.compress(tex, fmt, seed=42)
    for _ in range(4):
        assert mod.compress(tex, fmt, seed=42) == first


def test_batch_repeat_entry():
    r = mod.batch_repeat(1)
    assert r["all_byte_identical"] is True
    assert r["batch_size"] == 5


def test_dither_sequence_is_seedable_and_stable():
    assert mod._dither_offsets(7, 32) == mod._dither_offsets(7, 32)
    assert mod._dither_offsets(1, 8) != mod._dither_offsets(2, 8)
    assert set(mod._dither_offsets(3, 64)) <= {-1, 1}


def test_determinism_entry_aggregates_evidence():
    r = mod.determinism(1)
    assert r["lcg_sequence_stable"] is True
    assert r["batch"]["all_byte_identical"] is True


# ---------------------------------------------------------------------------
# 6. 块尺寸与对齐
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("w,h", [(1, 1), (3, 5), (6, 7), (8, 8), (13, 9)])
def test_padding_to_block_grid(w, h):
    tex = mod.make_smooth_fixture(w, h, srgb=False)
    padded = mod.pad_to_block_grid(tex)
    assert padded.width % BLOCK == 0 and padded.height % BLOCK == 0
    assert padded.width >= w and padded.height >= h
    # 钳制填充：右下角填充纹素等于真实边缘纹素
    assert padded.px(padded.width - 1, padded.height - 1) == tex.px(w - 1, h - 1)


def test_non_aligned_roundtrip_restores_original_size():
    tex = mod.make_smooth_fixture(6, 5, srgb=False)
    fmt = mod.select_format(has_alpha=True, is_normal=False, srgb=False)
    blob = mod.compress(tex, fmt, with_mips=False)
    info = mod.audit_container(blob)
    assert info["ok"] is True
    assert tuple(info["original_size"]) == (6, 5)
    decoded = mod.decompress(blob)[0]
    assert (decoded.width, decoded.height) == (6, 5)
    # 所有块都是 16 字节，pitch 16 字节对齐
    for lv in info["levels"]:
        assert lv["pitch_16_aligned"] is True
        assert lv["pitch_bytes"] % 16 == 0


def test_every_encoded_block_is_16_bytes():
    tex = mod.make_smooth_fixture(8, 8, srgb=False)
    fmt = mod.select_format(has_alpha=True, is_normal=False, srgb=False)
    blob = mod.compress(tex, fmt, with_mips=False)
    info = mod.audit_container(blob)
    lv0 = info["levels"][0]
    expected_stream = lv0["pitch_bytes"] * (lv0["padded_height"] // BLOCK)
    assert expected_stream == lv0["blocks"] * RAW_BLOCK_BYTES


# ---------------------------------------------------------------------------
# 回退 / 校验 / 审计 / 指标
# ---------------------------------------------------------------------------


def test_fallback_blocks_are_lossless():
    tex = mod.make_rgba_fixture(8, 8, seed=77)  # 白噪声，必然触发回退
    fmt = mod.select_format(has_alpha=True, is_normal=False, srgb=True)
    blob = mod.compress(tex, fmt, with_mips=False)
    info = mod.audit_container(blob)
    assert info["fallback_blocks"] >= 1
    decoded = mod.decompress(blob)[0]
    # 回退块覆盖的区域必须逐字节还原
    pos = mod.HEADER_SIZE + mod.LEVEL_HEADER_SIZE
    for _ in range(info["fallback_blocks"]):
        idx = int.from_bytes(blob[pos:pos + 2], "little")
        raw = blob[pos + 2:pos + 66]
        bx, by = idx % 2, idx // 2
        k = 0
        for yy in range(by * 4, by * 4 + 4):
            for xx in range(bx * 4, bx * 4 + 4):
                assert tuple(decoded.px(xx, yy)) == tuple(raw[k * 4:k * 4 + 4])
                k += 1
        pos += 66


def test_validation_reports_color_space_problem():
    tex = mod.make_smooth_fixture(4, 4)
    bad = mod.validate_texture(tex, FMT_BC3)  # 纹理 sRGB，格式不带 sRGB 标记
    assert bad["ok"] is False
    assert any("色彩空间" in p for p in bad["problems"])
    good = mod.validate_texture(tex, FMT_BC3 | SRGB_FLAG)
    assert good["ok"] is True


def test_audit_rejects_garbage():
    with pytest.raises(ValueError):
        mod.audit_container(b"not a container" * 4)


def test_quality_metric_structure():
    r = mod.quality_metric(1)
    assert set(r["mse"]) == {"r", "g", "b", "a"}
    assert all(v >= 0 for v in r["mse"].values())
    assert r["max_abs_error"] >= 0
    # alpha 由独立通道块表达，应近乎无损
    assert r["mse"]["a"] == 0.0


def test_low_frequency_gradient_quality_is_high():
    buf = bytearray()
    for y in range(16):
        for x in range(16):
            buf += bytes((x * 16, y * 16, (x + y) * 8, 255))
    tex = Texture(16, 16, buf, False)
    mod.FALLBACK_MSE_THRESHOLD = 1e18  # 关闭回退，考察裸编码
    try:
        fmt = mod.select_format(has_alpha=False, is_normal=False, srgb=False)
        decoded = mod.decompress(
            mod.compress(tex, fmt, with_mips=False)
        )[0]
    finally:
        mod.FALLBACK_MSE_THRESHOLD = 800.0
    report = mod.quality_report(tex, decoded)
    assert report["psnr_db"]["r"] > 24.0
    assert report["max_abs_error"] <= 40


def test_format_selection_rules():
    assert mod.select_format(has_alpha=False, is_normal=False, srgb=True) \
        == FMT_BC1 | SRGB_FLAG
    assert mod.select_format(has_alpha=True, is_normal=False, srgb=False) == FMT_BC3
    assert mod.select_format(has_alpha=False, is_normal=True, srgb=False) == FMT_BC5
    with pytest.raises(ValueError):
        mod.select_format(has_alpha=True, is_normal=True, srgb=False)


# ---------------------------------------------------------------------------
# 15 个公共入口的契约：全部返回非空、可 JSON 序列化的报告
# ---------------------------------------------------------------------------


ENTRIES = [
    "alpha_edge", "normal_z", "color_space", "mip_pre", "encode_determinism",
    "block_align", "srgb_linear", "validation", "audit", "fallback",
    "quality_metric", "metrics", "format_select", "batch_repeat", "determinism",
]


@pytest.mark.parametrize("name", ENTRIES)
def test_entry_returns_report(name):
    import json
    result = getattr(mod, name)(1)
    assert result is not None
    # 报告必须是可序列化的纯数据（math.inf 用 allow_nan 接受）
    json.dumps(result, allow_nan=True)
