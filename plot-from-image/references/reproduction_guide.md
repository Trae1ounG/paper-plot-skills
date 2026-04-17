# Image Reproduction Guide

## Step 1: Measure True Proportions

```python
from PIL import Image
img = Image.open('input.png')
print(img.size, f"AR={img.size[0]/img.size[1]:.2f}")
```

Set `figsize` so that `FW / FH ≈ original AR`. At dpi=300:
- `FW` inches × 300 = target pixel width

## Step 2: Font Analysis Checklist

| Clue | Likely Font |
|------|------------|
| Serif, italicized math variables | Computer Modern (`usetex=True`) |
| Clean sans-serif, rounded | DejaVu Sans / Helvetica |
| Bold axis labels only | sans-serif with `fontweight='bold'` |

Set globally:
```python
plt.rcParams.update({
    'font.family': 'serif',          # or 'sans-serif'
    'text.usetex': True,             # for Computer Modern
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
})
```

## Step 3: Spine & Tick Style

| Observation | Code |
|-------------|------|
| 4-sided box | `for sp in ax.spines.values(): sp.set_visible(True)` |
| L-shape (left+bottom only) | `ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)` |
| Axis arrows at ends | `ax.plot(1,0,'>k', transform=ax.get_yaxis_transform(), clip_on=False)` |
| Ticks inward | `ax.tick_params(direction='in')` |
| Ticks outward | `ax.tick_params(direction='out')` |

## Step 4: Color Identification

Use a color picker on the original image. Common paper palettes:
- `tab10`: `#1F77B4` (blue), `#FF7F0E` (orange), `#2CA02C` (green), `#D62728` (red)
- Muted custom: DoRA green `#5A8A5A`, LoRA blue `#4169E1`

## Step 5: Grid Style

```python
ax.grid(True, color='#E8E8E8', linewidth=0.6, linestyle='--')  # dashed gray
ax.grid(True, color='#E0E0E0', linewidth=0.5, linestyle=':')   # dotted
ax.set_axisbelow(True)
```

## Step 6: Existing Style Match

Check if the image matches any of these styles (read `../plot-from-data/references/<name>.md` for exact params):

| Style | Key Visual Cues |
|-------|----------------|
| `bar_paired_delta` | Paired bars, arrows, serif, blue title |
| `bar_grouped_hatch` | Hatched main bar, per-bar numbers, open spine |
| `line_confidence_band` | Shaded bands, serif/LaTeX font, 4-sided box |
| `line_training_curve` | Vertical dashed cuts, horizontal reference, sans-serif |
| `line_loss_with_inset` | L-spine + axis arrows, dashed zoom box, inset panel |
| `scatter_tsne_cluster` | t-SNE blob, colored annotation boxes, dotted grid |
| `scatter_broken_axis` | Split x-axis two-panel, star/circle/diamond markers |
| `radar_dual_series` | Octagonal dashed grid, two-series fill, value labels |

## Step 7: Reproduction Script Template

```python
"""
Reproduce: <paper_figure_name>
Original: <W>×<H> px  AR=<ratio>
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Font & global style ──────────────────────────────────────────
plt.rcParams.update({...})

# ── Simulated / user data ────────────────────────────────────────
# Replace with actual values here

# ── Figure & axes ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(<FW>, <FH>))

# ── Plot ─────────────────────────────────────────────────────────

# ── Labels, spines, ticks ────────────────────────────────────────

fig.savefig('output_repro.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
```

## Common Pitfalls

- **Aspect ratio wrong**: always measure original px first, then set figsize = (W/100, H/100) or maintain ratio
- **Fonts too large**: match font size ratio to original figure size; if original is small (5" wide), use fontsize=9-10
- **Too much whitespace**: use `ax.set_ylim(content_bottom, content_top)` + `bbox_inches='tight'`
- **Jagged smooth curves**: apply EMA smoothing `alpha=0.95-0.97` before plotting
