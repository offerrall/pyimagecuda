# Fill

The `Fill` module provides GPU-accelerated operations for filling images.

---

## Solid Colors

Fill an entire image with a solid color.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.color(img, (1.0, 0.0, 0.0, 1.0))
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_color.png" alt="Solid color fill" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `rgba` (tuple[float, float, float, float]): Color in RGBA format (0.0 to 1.0 range)

---

## Gradients

Fill an image with a smooth gradient between two colors.

**Example - Horizontal Gradient:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">
  
```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.gradient(
    img,
    rgba1=(1.0, 0.0, 0.0, 1.0),  # Red
    rgba2=(0.0, 0.0, 1.0, 1.0),  # Blue
    direction='horizontal'
)
save(img, 'output.png')

```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_gradient_horizontal.png" alt="Horizontal gradient" style="width: 100%;">
  </div>
</div>

**Example - Radial Gradient:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.gradient(
    img,
    rgba1=(1.0, 0.0, 0.0, 1.0),  # Red
    rgba2=(0.0, 0.0, 1.0, 1.0),  # Blue
    direction='radial'
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_gradient_radial.png" alt="Radial gradient" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `rgba1` (tuple[float, float, float, float]): Start color (0.0 to 1.0 range)
- `rgba2` (tuple[float, float, float, float]): End color (0.0 to 1.0 range)
- `direction` (str): Gradient direction - `'horizontal'`, `'vertical'`, `'diagonal'`, or `'radial'`
- `seamless` (bool): If True, gradient wraps smoothly for tiling (default: False)

**Example - Seamless Texture:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

texture = Image(512, 512)
Fill.gradient(
    texture,
    rgba1=(0.8, 0.6, 0.4, 1.0),
    rgba2=(0.4, 0.3, 0.2, 1.0),
    direction='diagonal',
    seamless=True
)
save(texture, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_gradient_seamless.png" alt="Seamless gradient" style="width: 100%;">
  </div>
</div>