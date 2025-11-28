# Blend

The `Blend` module provides GPU-accelerated layer compositing with multiple blend modes.

All blend operations modify the `base` image in-place by compositing the `overlay` on top of it.

---

## Positioning

### Anchors

Position overlays relative to common anchor points. By default, `anchor='top-left'`, which makes `offset_x` and `offset_y` behave as absolute pixel coordinates—just like standard graphics APIs.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(200, 200)

Fill.color(base, (0.2, 0.2, 0.3, 1.0))
Fill.color(overlay, (1.0, 0.0, 0.0, 1.0))

Blend.normal(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_anchor.png" alt="Anchor positioning" style="width: 100%;">
  </div>
</div>

**Available anchors:** `'top-left'`, `'top-center'`, `'top-right'`, `'center-left'`, `'center'`, `'center-right'`, `'bottom-left'`, `'bottom-center'`, `'bottom-right'`

---

### Offsets

Add pixel offsets from the anchor point for fine-tuning.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(200, 200)

Fill.color(base, (0.2, 0.2, 0.3, 1.0))
Fill.color(overlay, (1.0, 0.0, 0.0, 1.0))

Blend.normal(base, overlay, anchor='center',
             offset_x=50, offset_y=-30)
save(base, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_offset.png" alt="Offset positioning" style="width: 100%;">
  </div>
</div>

---

## Blend Modes

### Normal

Standard alpha blending (Porter-Duff over operation).

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(400, 300)

c1 = (0.2, 0.2, 0.3, 1.0)
c2 = (0.4, 0.3, 0.5, 1.0)
c3 = (1.0, 0.5, 0.0, 0.8)
c4 = (1.0, 0.0, 0.5, 0.8)

Fill.gradient(base, c1, c2, 'radial')
Fill.gradient(overlay, c3, c4, 'horizontal')

Blend.normal(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_normal.png" alt="Normal blend" style="width: 100%;">
  </div>
</div>

**Use for:** Standard layer compositing, logos, overlays.

---

### Multiply

Darkens the image by multiplying color values.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(400, 300)

c1 = (0.2, 0.2, 0.3, 1.0)
c2 = (0.4, 0.3, 0.5, 1.0)
c3 = (1.0, 0.5, 0.0, 0.8)
c4 = (1.0, 0.0, 0.5, 0.8)

Fill.gradient(base, c1, c2, 'radial')
Fill.gradient(overlay, c3, c4, 'horizontal')

Blend.multiply(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_multiply.png" alt="Multiply blend" style="width: 100%;">
  </div>
</div>

**Use for:** Shadows, darkening effects, texture overlays.

---

### Screen

Lightens the image (inverse of multiply).

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(400, 300)

c1 = (0.2, 0.2, 0.3, 1.0)
c2 = (0.4, 0.3, 0.5, 1.0)
c3 = (1.0, 0.5, 0.0, 0.8)
c4 = (1.0, 0.0, 0.5, 0.8)

Fill.gradient(base, c1, c2, 'radial')
Fill.gradient(overlay, c3, c4, 'horizontal')

Blend.screen(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_screen.png" alt="Screen blend" style="width: 100%;">
  </div>
</div>

**Use for:** Highlights, light effects, glows.

---

### Add

Additive blending (colors add up, clamped to 1.0).

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Blend, save

base = Image(800, 600)
overlay = Image(400, 300)

c1 = (0.2, 0.2, 0.3, 1.0)
c2 = (0.4, 0.3, 0.5, 1.0)
c3 = (1.0, 0.5, 0.0, 0.8)
c4 = (1.0, 0.0, 0.5, 0.8)

Fill.gradient(base, c1, c2, 'radial')
Fill.gradient(overlay, c3, c4, 'horizontal')

Blend.add(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_add.png" alt="Add blend" style="width: 100%;">
  </div>
</div>

**Use for:** Light effects, lens flares, glowing elements.

---

## Parameters

All blend modes share the same parameters:

- `base` (Image): Base layer (modified in-place)
- `overlay` (Image): Layer to composite on top
- `anchor` (str): Position anchor point (default: `'top-left'`)
- `offset_x` (int): Horizontal offset from anchor in pixels (default: 0)
- `offset_y` (int): Vertical offset from anchor in pixels (default: 0)
- `opacity` (float): Blend opacity from 0.0 (transparent) to 1.0 (opaque) (default: 1.0)