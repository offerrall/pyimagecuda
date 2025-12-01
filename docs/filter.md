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

## Sepia

Apply a warm sepia tone effect to give images a vintage, antique look.

**Example - Full Sepia:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.sepia(img, intensity=1.0)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_sepia.png" alt="Sepia effect" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Subtle Sepia:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.sepia(img, intensity=0.5)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_sepia_subtle.png" alt="Subtle sepia" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to modify (in-place)
- `intensity` (float): Effect intensity from 0.0 (no effect) to 1.0 (full sepia) (default: 1.0)

**Returns:** `None` (modifies image in-place)

---

## Invert

Invert all colors to create a photographic negative effect.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.invert(img)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_invert.png" alt="Inverted colors" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to modify (in-place)

**Returns:** `None` (modifies image in-place)

---

## Threshold

Convert image to pure black and white based on luminance threshold.

**Example - Medium Threshold:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.threshold(img, value=0.5)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_threshold.png" alt="Threshold effect" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Low Threshold (More White):**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.threshold(img, value=0.3)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_threshold_low.png" alt="Low threshold" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to modify (in-place)
- `value` (float): Threshold value from 0.0 to 1.0 (default: 0.5)

**Returns:** `None` (modifies image in-place)

**Behavior:** Pixels brighter than the threshold become white, others become black. Lower values produce more white areas; higher values produce more black areas.

---

## Solarize

Create a psychedelic effect by inverting colors above a luminance threshold.

**Example - Medium Threshold:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.solarize(img, threshold=0.5)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_solarize.png" alt="Solarize effect" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Low Threshold:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

Filter.solarize(img, threshold=0.3)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_solarize_low.png" alt="Solarize low threshold" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to modify (in-place)
- `threshold` (float): Luminance threshold from 0.0 to 1.0 (default: 0.5)

**Returns:** `None` (modifies image in-place)

**Behavior:** Only pixels brighter than the threshold are inverted, creating a distinctive retro/psychedelic look.

---

## Sobel

Detect edges using the Sobel operator to create an edge map.

**Example:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

edges = Filter.sobel(img)

save(edges, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_sobel.png" alt="Sobel edge detection" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `src` (Image): Source image
- `dst_buffer` (Image | None): Optional output buffer (default: None)

**Returns:** New image with detected edges (or None if `dst_buffer` provided)

**Output:** Black and white image where white indicates detected edges.

---

## Emboss

Apply an emboss (relief) effect to create a 3D raised appearance.

**Example - Normal Strength:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

embossed = Filter.emboss(img, strength=1.0)

save(embossed, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_emboss.png" alt="Emboss effect" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Strong Emboss:**

<div style="display: flex; gap: 20px; align-items: start;">
  <div style="flex: 1;">

```python
from pyimagecuda import load, Filter, save

img = load("photo.jpg")

embossed = Filter.emboss(img, strength=2.0)

save(embossed, 'output.jpg')
```

  </div>
  <div style="flex: 1;">
    <img src="https://offerrall.github.io/pyimagecuda/images/filter_emboss_strong.png" alt="Strong emboss" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `src` (Image): Source image
- `strength` (float): Effect intensity (default: 1.0)
- `dst_buffer` (Image | None): Optional output buffer (default: None)

**Returns:** New embossed image (or None if `dst_buffer` provided)

**Recommended values:** 0.5-2.0 for most images. Higher values create more pronounced relief.

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