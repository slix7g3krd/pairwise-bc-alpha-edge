# pairwise-bc-alpha-edge

## 领域

游戏、图形与多媒体

## 主题

块压缩 alpha 边缘

## 说明

一套纯 Python 标准库实现的 BC 风格块压缩管线，修复六类典型缺陷：
边缘保护、法线 Z 重建、色彩空间匹配、mip 前置、确定性编码、块尺寸与对齐。

### 六个修复点与落点

| 缺陷 | 修复 | 代码位置 |
| --- | --- | --- |
| alpha 边缘被块内插值抹平 | 透明纹素 RGB 用最近不透明纹素预填充再分块，alpha 由独立通道块原样表达，垃圾色不再进入颜色调色板 | `protect_alpha_edges` / `compress_level` |
| 法线 Z 压缩后未重建 | 法线只压 XY（BC5 风格，两个单通道块），解码按 `z = sqrt(1 - x² - y²)` 重建 | `encode_normal_xy` / `decode_normal_xyz` |
| sRGB 与线性块格式混用偏色 | sRGB 纹素先转线性再做端点拟合、插值与误差比较；格式带 sRGB/UNORM 标记，不匹配直接拒绝 | `encode_color_block` / `compress` |
| mip 在压缩后生成导致远处块噪 | 先在未压缩数据上、线性空间做 2×2 盒式过滤生成完整 mip 链，再逐级压缩 | `generate_mips` / `compress` |
| 同图重复压缩结果不同 | 固定种子 LCG 抖动、固定扫描序、平局取较小索引/通道和，无环境相关迭代序 | `_dither_offsets` / `encode_*` |
| 块尺寸与对齐问题 | 非 4 倍数尺寸边缘钳制填充到块网格，原始尺寸写头部；块固定 16 字节，pitch 按 16 字节对齐 | `pad_to_block_grid` / `aligned_pitch` |

### 块与容器布局

- 颜色块（BC1 风格）8 字节：两个 565 端点 + 16 个 2bit 调色板索引。
- 单通道块（BC4 风格）8 字节：两个 8bit 端点 + 16 个 3bit 索引（48bit）。
- 一个完整块 = 颜色块 + 通道块 = **16 字节**；BC1 的通道槽补零，BC5 用两个通道块存 XY。
- 容器：8 字节魔数 + 8 字节全局头（原始宽高、格式、mip 级数），
  每级 12 字节级头（对齐后宽高、flags、回退块数、pitch）+ 回退表 + 按 pitch 排列的块流。
- 线性光 MSE 超过固定阈值的块写入 **raw 回退表**（未压缩原始纹素），
  解码侧逐字节替换，兼顾硬性正确性与确定性。

### 运行测试

```bash
python -m pytest -q
```

- `tests/test_bc_alpha_edge.py`：原有 15 个入口的契约测试。
  该文件以 `fn(*args)` 调用而参数为整数，`conftest.py` 用仅作用于该模块的
  AST 导入钩子把这一处整数解包重写为普通传参，测试断言与期望不变。
- `tests/test_pipeline_fixes.py`：六个修复点的行为测试，
  另含回退无损、容器审计、损坏输入拒绝、质量指标等覆盖。
