# pairwise-bc-alpha-edge

## 领域

游戏、图形与多媒体

## 主题

块压缩 alpha 边缘

## 说明

边缘保护、法线 Z 重建与 mip 前置。

初始状态下 `graphics/bc_alpha_edge.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
