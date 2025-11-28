# Effect

The `Effect` module provides GPU-accelerated design effects for professional compositing workflows.

---

## Rounded Corners

Apply rounded corners to an image by modifying its alpha channel.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Effect, save

img = Image(512, 512)
Fill.gradient(img, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')

Effect.rounded_corners(img, radius=50)

save(img, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/effect_rounded.png" alt="Rounded corners" style="width: 100%;">

  </div>
</div>

**Parameters:**

- `image` (Image): Image to modify (in-place)
- `radius` (float): Corner radius in pixels

**Returns:** `None` (modifies image in-place)

**Note:** Maximum radius is automatically clamped to `min(width, height) / 2.0` to prevent invalid values.

---

## Drop Shadow

Create a drop shadow effect with blur and offset.

**Example - Expand Mode (Default):**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Effect, save

img = Image(400, 300)
Fill.gradient(img, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')
Effect.rounded_corners(img, 30)

shadowed = Effect.drop_shadow(
    img,
    offset_x=10,
    offset_y=10,
    blur=30,
    color=(0, 0, 0, 0.8)
)

save(shadowed, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/effect_shadow_expand.png" alt="Drop shadow expanded" style="width: 100%;">

  </div>
</div>

**Example - No Expand Mode:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import Image, Fill, Effect, save

img = Image(400, 300)
Fill.gradient(img, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')
Effect.rounded_corners(img, 30)

shadowed = Effect.drop_shadow(
    img,
    offset_x=10,
    offset_y=10,
    blur=30,
    color=(0, 0, 0, 0.8),
    expand=False
)

save(shadowed, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/effect_shadow_no_expand.png" alt="Drop shadow no expand" style="width: 100%;">

  </div>
</div>

**Parameters:**

- `image` (Image): Source image
- `offset_x` (int): Horizontal shadow offset in pixels (default: 10)
- `offset_y` (int): Vertical shadow offset in pixels (default: 10)
- `blur` (int): Shadow blur radius in pixels (default: 20)
- `color` (tuple[float, float, float, float]): Shadow color in RGBA (default: `(0.0, 0.0, 0.0, 0.5)`)
- `expand` (bool): If True, expand canvas to fit shadow; if False, clip to original size (default: True)
- `dst_buffer` (Image | None): Optional output buffer (default: None)
- `shadow_buffer` (Image | None): Optional shadow computation buffer (default: None)
- `temp_buffer` (Image | None): Optional temporary buffer for blur (default: None)

**Returns:** New image with shadow (or None if `dst_buffer` provided)

**Expand mode behavior:**
- `expand=True`: Canvas expands to fit the full shadow (recommended)
- `expand=False`: Shadow is clipped to original image dimensions

---

## Buffer Reuse

Reuse buffers for batch processing to avoid allocations.

**Example:**

```python
from pyimagecuda import Image, load, Effect, save

# Pre-allocate buffers
# Expanded size for shadow
dst = Image(600, 500)
shadow = Image(600, 500)
temp = Image(600, 500)

for file in files:
    img = load(file)  # 400×300
    Effect.rounded_corners(img, 30)
    
    Effect.drop_shadow(
        img,
        blur=30,
        dst_buffer=dst,
        shadow_buffer=shadow,
        temp_buffer=temp
    )
    
    save(dst, f"shadowed_{file}")
    img.free()

dst.free()
shadow.free()
temp.free()
```

When using buffers, operations modify `dst_buffer` directly and return `None` instead of a new image.