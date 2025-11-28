# Filter

The `Filter` module provides GPU-accelerated image filtering operations.

---

## Gaussian Blur

Apply Gaussian blur using a separable kernel (horizontal then vertical pass).

**Example - Light Blur:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

blurred = Filter.gaussian_blur(img, radius=3)

save(blurred, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_blur_light.png" alt="Light blur" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Heavy Blur:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

blurred = Filter.gaussian_blur(img, radius=50)

save(blurred, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_blur_heavy.png" alt="Heavy blur" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `src` (Image): Source image
- `radius` (int): Blur kernel radius in pixels (default: 3)
- `sigma` (float | None): Gaussian sigma value, auto-calculated if None (default: radius / 3.0)
- `dst_buffer` (Image | None): Optional output buffer (default: None)
- `temp_buffer` (Image | None): Optional temporary buffer for separable convolution (default: None)

**Returns:** Blurred image (or None if `dst_buffer` provided)

**Note:** `temp_buffer` is used internally for the separable convolution (horizontal pass). Both buffers must have sufficient capacity.

---

## Sharpen

Enhance image edges and details using unsharp mask.

**Example - Subtle Sharpening:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

sharpened = Filter.sharpen(img, strength=0.5)

save(sharpened, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_sharpen_subtle.png" alt="Subtle sharpening" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Strong Sharpening:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

sharpened = Filter.sharpen(img, strength=2.0)

save(sharpened, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_sharpen_strong.png" alt="Strong sharpening" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `src` (Image): Source image
- `strength` (float): Sharpening intensity (default: 1.0)
- `dst_buffer` (Image | None): Optional output buffer (default: None)

**Returns:** Sharpened image (or None if `dst_buffer` provided)

**Recommended values:** 1.0-1.5 for photos. Higher values work for specific effects but may introduce artifacts.

---

## Buffer Reuse

Reuse buffers for batch processing to avoid allocations.

**Example:**

```python
from pyimagecuda import Image, load, Filter, save

# Pre-allocate buffers with sufficient capacity (e.g. 1920x1080)
dst = Image(1920, 1080)
temp = Image(1920, 1080)

for file in files:
    src = load(file)
    # Buffers automatically adjust their logical dimensions
    # provided they have enough capacity
    Filter.gaussian_blur(src, radius=5,
                         dst_buffer=dst,
                         temp_buffer=temp)
    save(dst, f"blurred_{file}")
    src.free()

dst.free()
temp.free()
```

When using buffers, operations are in-place (modify dst_buffer directly) and return None. The buffers automatically adjust their logical dimensions to match the operation result, provided they have enough capacity.