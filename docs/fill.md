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

---

## Checkerboard

Fill an image with a checkerboard pattern.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.checkerboard(
    img,
    size=40,
    color1=(0.9, 0.9, 0.9, 1.0),
    color2=(0.4, 0.4, 0.4, 1.0)
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_checkerboard.png" alt="Checkerboard pattern" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `size` (int): Size of each checker square in pixels (default: 20)
- `color1` (tuple[float, float, float, float]): First checker color (default: light gray)
- `color2` (tuple[float, float, float, float]): Second checker color (default: medium gray)
- `offset_x` (int): Horizontal offset in pixels (default: 0)
- `offset_y` (int): Vertical offset in pixels (default: 0)

---

## Grid

Fill an image with a grid pattern.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.grid(
    img,
    spacing=50,
    line_width=2,
    color=(1.0, 1.0, 1.0, 1.0),
    bg_color=(0.2, 0.2, 0.2, 1.0)
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_grid.png" alt="Grid pattern" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `spacing` (int): Distance between grid lines in pixels (default: 50)
- `line_width` (int): Width of grid lines in pixels (default: 1)
- `color` (tuple[float, float, float, float]): Grid line color (default: gray)
- `bg_color` (tuple[float, float, float, float]): Background color (default: transparent)
- `offset_x` (int): Horizontal offset in pixels (default: 0)
- `offset_y` (int): Vertical offset in pixels (default: 0)

---

## Stripes

Fill an image with alternating stripes with anti-aliasing.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.stripes(
    img,
    angle=45.0,
    spacing=40,
    width=20,
    color1=(1.0, 0.8, 0.0, 1.0),
    color2=(0.0, 0.4, 0.8, 1.0)
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_stripes.png" alt="Striped pattern" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `angle` (float): Rotation angle in degrees (default: 45.0)
- `spacing` (int): Distance between stripes in pixels (default: 40)
- `width` (int): Width of each stripe in pixels (default: 20)
- `color1` (tuple[float, float, float, float]): First stripe color (default: white)
- `color2` (tuple[float, float, float, float]): Second stripe color (default: transparent)
- `offset` (int): Offset along stripe direction in pixels (default: 0)

---

## Dots

Fill an image with a polka dot pattern.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.dots(
    img,
    spacing=60,
    radius=15.0,
    color=(1.0, 0.5, 0.8, 1.0),
    bg_color=(0.1, 0.1, 0.2, 1.0),
    softness=0.3
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_dots.png" alt="Polka dot pattern" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `spacing` (int): Distance between dot centers in pixels (default: 40)
- `radius` (float): Radius of each dot in pixels (default: 10.0)
- `color` (tuple[float, float, float, float]): Dot color (default: white)
- `bg_color` (tuple[float, float, float, float]): Background color (default: transparent)
- `offset_x` (int): Horizontal offset in pixels (default: 0)
- `offset_y` (int): Vertical offset in pixels (default: 0)
- `softness` (float): Edge softness - 0.0 for hard edges, 1.0 for soft glow (default: 0.0)

---

## Circle

Fill an image with a centered circle fitted to the image size.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.circle(
    img,
    color=(0.0, 0.8, 1.0, 1.0),
    bg_color=(0.05, 0.05, 0.1, 1.0),
    softness=0.0
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_circle.png" alt="Circle fill" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `color` (tuple[float, float, float, float]): Circle color (default: white)
- `bg_color` (tuple[float, float, float, float]): Background color (default: transparent)
- `softness` (float): Edge softness - 0.0 for hard edges with AA, >0.0 for soft gradient (default: 0.0)

---

## Noise

Fill an image with random white noise.

**Example - Monochrome Noise:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.noise(
    img,
    seed=42.0,
    monochrome=True
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_noise_mono.png" alt="Monochrome noise" style="width: 100%;">
  </div>
</div>

**Example - RGB Noise:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.noise(
    img,
    seed=42.0,
    monochrome=False
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_noise_rgb.png" alt="RGB noise" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `seed` (float): Random seed - change this to animate the noise (default: 0.0)
- `monochrome` (bool): True for grayscale noise, False for RGB noise (default: True)

---

## Perlin Noise

Fill an image with Perlin noise (gradient noise) for natural-looking textures.

**Example - Simple Perlin:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.perlin(
    img,
    scale=80.0,
    seed=0.0,
    octaves=1
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_perlin_simple.png" alt="Simple Perlin noise" style="width: 100%;">
  </div>
</div>

**Example - Detailed Perlin:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.perlin(
    img,
    scale=100.0,
    seed=42.0,
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    color1=(0.1, 0.2, 0.4, 1.0),
    color2=(0.9, 0.8, 0.6, 1.0)
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_perlin_detailed.png" alt="Detailed Perlin noise" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `scale` (float): "Zoom" level - higher values create bigger features (default: 50.0)
- `seed` (float): Random seed for variation (default: 0.0)
- `octaves` (int): Detail layers - 1 for smooth, 6+ for detailed/rocky appearance (default: 1)
- `persistence` (float): How much each octave contributes, from 0.0 to 1.0 (default: 0.5)
- `lacunarity` (float): Detail frequency multiplier, usually 2.0 (default: 2.0)
- `offset_x` (float): Horizontal offset (default: 0.0)
- `offset_y` (float): Vertical offset (default: 0.0)
- `color1` (tuple[float, float, float, float]): Low value color (default: black)
- `color2` (tuple[float, float, float, float]): High value color (default: white)

---

## N-gon

Fill an image with a regular polygon (triangle, pentagon, hexagon, etc.).

**Example - Triangle:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.ngon(
    img,
    sides=3,
    color=(1.0, 0.3, 0.3, 1.0),
    bg_color=(0.1, 0.1, 0.1, 1.0),
    rotation=0.0,
    softness=0.0
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_ngon_triangle.png" alt="Triangle" style="width: 100%;">
  </div>
</div>

**Example - Hexagon with Glow:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.ngon(
    img,
    sides=6,
    color=(0.2, 0.8, 1.0, 1.0),
    bg_color=(0.05, 0.05, 0.15, 1.0),
    rotation=30.0,
    softness=0.0
)
save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/fill_ngon_hexagon.png" alt="Hexagon with glow" style="width: 100%;">
  </div>
</div>

**Parameters:**

- `image` (Image): Target image to fill
- `sides` (int): Number of sides (must be 3 or more) (default: 3)
- `color` (tuple[float, float, float, float]): Polygon color (default: white)
- `bg_color` (tuple[float, float, float, float]): Background color (default: transparent)
- `rotation` (float): Rotation angle in degrees (default: 0.0)
- `softness` (float): Edge softness - 0.0 for hard edges with AA, >0.0 for glow effect (default: 0.0)