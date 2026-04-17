---
name: plot-from-data
description: |
  Generate publication-quality matplotlib figures by selecting a pre-built paper style and substituting user data.
  Use when: user provides data (numbers, arrays, or CSV) and wants a chart in a specific academic style;
  user asks to "plot this data", "make a bar chart", "draw a radar chart", "用这个风格画图", "把我的数据画出来";
  or user selects a style name from the style catalog (bar_paired_delta, bar_grouped_hatch,
  line_confidence_band, line_training_curve, line_loss_with_inset, scatter_tsne_cluster,
  scatter_broken_axis, radar_dual_series).
---

# Plot From Data

Generate a paper-quality figure by picking a style template and filling it with user data. All outputs are `dpi=300` PNG.

## Available Styles

| Style | Type | Script | 适用场景 |
|-------|------|---------|---------|
| `bar_paired_delta` | 柱状图 | `scripts/bar_memevolve.py` | Baseline vs method 配对对比 + 增益箭头 |
| `bar_grouped_hatch` | 柱状图 | `scripts/bar_spice.py` | 多方法消融，主方法斜线填充，柱顶数值 |
| `line_confidence_band` | 折线图 | `scripts/line_selfdistill.py` | 带置信区间的训练曲线 |
| `line_training_curve` | 折线图 | `scripts/line_aime.py` | 垂直断点线 + 水平参考线 |
| `line_loss_with_inset` | 折线图 | `scripts/line_loss_inset.py` | L 形 spine + 局部放大 inset |
| `scatter_tsne_cluster` | 散点图 | `scripts/scatter_tsne.py` | t-SNE 聚类 + 注释框 |
| `scatter_broken_axis` | 散点图 | `scripts/scatter_break.py` | 折断 X 轴，多 marker 系列 |
| `radar_dual_series` | 雷达图 | `scripts/radar_dora.py` | 双方法多维对比，正八边形网格 |

## Workflow

```
1. 确认用户的图类型和数据
2. 选择对应 style（如不确定，询问用户或根据数据形状推断）
3. 读取对应 references/<style_name>.md 获取精确参数
4. 复制对应 scripts/<script>.py，替换数据区（脚本顶部有清晰注释标注数据区）
5. 运行：python3 scripts/<script>.py
6. 检查输出，必要时微调颜色/标签/字号
```

## Data Substitution Tips

每个 repro 脚本的数据区在文件顶部，通常是 `np.array(...)` 或字典。替换规则：
- 保持数组维度和类型不变
- 若类别数变化（如从 4 组改为 6 组），同步调整颜色列表和宽度计算
- x 轴标签、图例标签直接修改对应字符串列表

## Detailed Style Parameters

Read the corresponding file in `references/` for exact `rcParams`, colors, font sizes, spine settings, and tick directions before generating:

- Bar: `references/bar_paired_delta.md`, `references/bar_grouped_hatch.md`
- Line: `references/line_confidence_band.md`, `references/line_training_curve.md`, `references/line_loss_with_inset.md`
- Scatter: `references/scatter_tsne_cluster.md`, `references/scatter_broken_axis.md`
- Radar: `references/radar_dual_series.md`
