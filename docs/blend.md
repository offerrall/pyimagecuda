# Blend

The `Blend` module provides GPU-accelerated layer compositing with multiple blend modes.

All blend operations modify the `base` image in-place by compositing the `overlay` on top of it.

---

## Positioning

### Anchors

Position overlays relative to common anchor points. By default, `anchor='top-left'`, which makes `offset_x` and `offset_y` behave as absolute pixel coordinates—just like standard graphics APIs.

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_anchor.png" alt="Anchor positioning" style="width: 100%;">
  </div>
</div>

**Available anchors:** `'top-left'`, `'top-center'`, `'top-right'`, `'center-left'`, `'center'`, `'center-right'`, `'bottom-left'`, `'bottom-center'`, `'bottom-right'`

---

### Offsets

Add pixel offsets from the anchor point for fine-tuning.

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_offset.png" alt="Offset positioning" style="width: 100%;">
  </div>
</div>

---

## Blend Modes

### Normal

Standard alpha blending (Porter-Duff over operation).

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_normal.png" alt="Normal blend" style="width: 100%;">
  </div>
</div>

**Use for:** Standard layer compositing, logos, overlays.

---

### Multiply

Darkens the image by multiplying color values.

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_multiply.png" alt="Multiply blend" style="width: 100%;">
  </div>
</div>

**Use for:** Shadows, darkening effects, texture overlays.

---

### Screen

Lightens the image (inverse of multiply).

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_screen.png" alt="Screen blend" style="width: 100%;">
  </div>
</div>

**Use for:** Highlights, light effects, glows.

---

### Add

Additive blending (colors add up, clamped to 1.0).

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_add.png" alt="Add blend" style="width: 100%;">
  </div>
</div>

**Use for:** Light effects, lens flares, glowing elements.

---

### Overlay

Combines Multiply and Screen modes to increase contrast. Darker values become darker, lighter values become lighter.

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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

Blend.overlay(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_overlay.png" alt="Overlay blend" style="width: 100%;">
  </div>
</div>

**Use for:** Increasing contrast, texture overlays, dramatic effects.

---

### Soft Light

Creates a gentle lighting effect, like shining a diffuse spotlight on the image. Similar to Overlay but more subtle.

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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

Blend.soft_light(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_soft_light.png" alt="Soft Light blend" style="width: 100%;">
  </div>
</div>

**Use for:** Subtle lighting effects, gentle color adjustments, photo enhancement.

---

### Hard Light

Creates a strong lighting effect, like shining a harsh spotlight on the image. More dramatic than Overlay.

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

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

Blend.hard_light(base, overlay, anchor='center')
save(base, 'output.png')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_hard_light.png" alt="Hard Light blend" style="width: 100%;">
  </div>
</div>

**Use for:** Strong lighting effects, dramatic contrast, vivid color overlays.

---

### Mask

Applies an image as an alpha mask to the base image, controlling transparency.

**Parameters:**
- `mode` (str): `'luminance'` (default) uses brightness of mask image, `'alpha'` uses alpha channel

**Example:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import Image, Fill, Blend, save


BASE_W, BASE_H = 800, 600
MASK_HOLE_W, MASK_HOLE_H = 200, 100

c1 = (0.2, 0.2, 0.3, 1.0)
c2 = (0.4, 0.3, 0.5, 1.0)
white = (1.0, 1.0, 1.0, 1.0)
black = (0.0, 0.0, 0.0, 1.0)

with Image(BASE_W, BASE_H) as base:
    Fill.gradient(base, c1, c2, 'radial')

    with Image(BASE_W, BASE_H) as mask:
        Fill.color(mask, white) 

        with Image(400, 300) as hole_shape:
            Fill.color(hole_shape, black)
            Blend.normal(mask,
                         hole_shape,
                         anchor='center')

        Blend.mask(base, mask)

    save(base, "output.png")
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/blend_mask.png" alt="Mask blend" style="width: 100%;">
  </div>
</div>

- **Use for:** Revealing parts of an image, creating vignettes, complex alpha compositing.
- **How Works:** The mask image's pixel values determine the transparency of the base image. White areas are fully opaque, black areas are fully transparent, and gray areas are partially transparent.
---

## Parameters

All blend modes (except `mask`) share the same parameters:

- `base` (Image): Base layer (modified in-place)
- `overlay` (Image): Layer to composite on top
- `anchor` (str): Position anchor point (default: `'top-left'`)
- `offset_x` (int): Horizontal offset from anchor in pixels (default: 0)
- `offset_y` (int): Vertical offset from anchor in pixels (default: 0)
- `opacity` (float): Blend opacity from 0.0 (transparent) to 1.0 (opaque) (default: 1.0)

### Mask-specific parameters:

- `base` (Image): Base layer (modified in-place)
- `mask` (Image): Mask image to apply
- `anchor` (str): Position anchor point (default: `'top-left'`)
- `offset_x` (int): Horizontal offset from anchor in pixels (default: 0)
- `offset_y` (int): Vertical offset from anchor in pixels (default: 0)
- `mode` (str): `'luminance'` or `'alpha'` (default: `'luminance'`)