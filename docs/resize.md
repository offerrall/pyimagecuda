# Resize

The `Resize` module provides GPU-accelerated image scaling with multiple interpolation algorithms.

All resize methods return a new image at the target dimensions.

---

## Aspect Ratio Preservation

Specify only `width` or `height` to maintain aspect ratio automatically.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Resize, save

img = load("photo.jpg")  # 1920×1080

# Scale to width 800
resized = Resize.lanczos(img, width=800)
# Result: 800×450

save(resized, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/resize_aspect.png" alt="Aspect ratio preserved" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

To force specific dimensions (may distort), specify both `width` and `height`.

---

## Interpolation Methods

### Nearest Neighbor

Fastest method with no interpolation (blocky results).

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Resize, save

img = load("photo.jpg")

scaled = Resize.nearest(img, width=128, height=128)

save(scaled, 'output.png')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/resize_nearest.png" alt="Nearest neighbor" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Use for:** Pixel art, quick previews, integer scaling

**Quality:** Low | **Speed:** Fastest

---

### Bilinear

Fast linear interpolation (smooth results).

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Resize, save

img = load("photo.jpg")

# Fast general-purpose resize
resized = Resize.bilinear(img, width=800)

save(resized, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/resize_bilinear.png" alt="Bilinear interpolation" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Use for:** General purpose scaling, real-time applications

**Quality:** Medium | **Speed:** Fast

---

### Bicubic

High-quality cubic interpolation.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Resize, save

img = load("photo.jpg")

# High-quality upscale
resized = Resize.bicubic(img, width=1920)

save(resized, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/resize_bicubic.png" alt="Bicubic interpolation" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Use for:** High-quality upscaling, photography, print materials

**Quality:** High | **Speed:** Medium

---

### Lanczos

Highest quality with sharp details (recommended).

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Resize, save

img = load("photo.jpg")

# Maximum quality upscale
resized = Resize.lanczos(img, width=3840)

save(resized, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/resize_lanczos.png" alt="Lanczos interpolation" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Use for:** Maximum quality upscaling, professional photography

**Quality:** Highest | **Speed:** Slower (still fast on GPU)

---

## Parameters

All methods share the same parameters:

- `src` (Image): Source image
- `width` (int | None): Target width in pixels (default: None)
- `height` (int | None): Target height in pixels (default: None)
- `dst_buffer` (Image | None): Optional pre-allocated buffer (default: None)

**Note:** At least one of `width` or `height` must be specified.

**Returns:** New `Image` at target size (or None if `dst_buffer` provided)

---

## Buffer Reuse

For batch processing, reuse the destination buffer. The buffer allows dynamic resizing as long as it has enough capacity.

**Example:**

```python
from pyimagecuda import Image, load, Resize, save

# Pre-allocate destination with max capacity (e.g., 4K)
dst = Image(3840, 2160)

for file in image_files:
    src = load(file)
    
    # dst automatically adjusts its dimensions to the result
    # No need to calculate height or resize dst manually
    Resize.lanczos(src, width=800, dst_buffer=dst)
    
    save(dst, f"resized_{file}")
    src.free()

dst.free()
```

When using dst_buffer, the buffer's capacity must be large enough to hold the result. The buffer's logical dimensions (width, height) are automatically updated to match the operation result.