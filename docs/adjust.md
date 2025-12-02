# Adjust

The `Adjust` module provides GPU-accelerated color and tone adjustments for images.

All adjust operations modify the image in-place.

---

## Brightness

Adjusts image brightness by adding a factor to all RGB channels.

**Example - Brighten:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.brightness(img, 0.2)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_brightness_up.png" alt="Brightened image" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Darken:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.brightness(img, -0.3)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_brightness_down.png" alt="Darkened image" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to adjust (modified in-place)
- `factor` (float): Brightness adjustment. Positive values brighten, negative values darken

**Use for:** Quick exposure adjustments, creating fade effects

---

## Contrast

Adjusts image contrast relative to middle gray (0.5).

**Example - Increase Contrast:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.contrast(img, 1.5)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_contrast_up.png" alt="High contrast" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Decrease Contrast:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.contrast(img, 0.5)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_contrast_down.png" alt="Low contrast" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to adjust (modified in-place)
- `factor` (float): Contrast multiplier
  - `factor > 1.0`: Increases contrast
  - `factor = 1.0`: No change
  - `factor < 1.0`: Decreases contrast

**Use for:** Making images more punchy or creating washed-out effects

---

## Saturation

Adjusts color intensity while preserving luminance.

**Example - Increase Saturation:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.saturation(img, 1.8)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_saturation_up.png" alt="High saturation" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Desaturate to Grayscale:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.saturation(img, 0.0)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_saturation_zero.png" alt="Grayscale" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to adjust (modified in-place)
- `factor` (float): Saturation multiplier
  - `factor = 0.0`: Grayscale
  - `factor = 1.0`: Original colors
  - `factor > 1.0`: More vibrant colors

**Use for:** Making colors pop, creating black and white images, subtle color grading

---

## Gamma

Adjusts gamma correction (non-linear brightness). Unlike brightness, gamma affects midtones more than highlights or shadows.

**Example - Brighten Midtones:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.gamma(img, 1.5)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_gamma_up.png" alt="Gamma brightened" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Darken Midtones:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.gamma(img, 0.6)

save(img, 'output.jpg')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_gamma_down.png" alt="Gamma darkened" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to adjust (modified in-place)
- `gamma` (float): Gamma value (must be positive)
  - `gamma > 1.0`: Brightens midtones
  - `gamma = 1.0`: No change
  - `gamma < 1.0`: Darkens midtones

**Use for:** Lifting shadows without blowing out highlights, display calibration, color grading

---

## Opacity

Adjusts image opacity by multiplying the alpha channel.

**Example - Semi-Transparent:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.opacity(img, 0.5)

save(img, 'output.png')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_opacity_half.png" alt="50% opacity" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Example - Subtle Fade:**

<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: start;">
  <div style="flex: 1; min-width: 300px;">

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

Adjust.opacity(img, 0.8)

save(img, 'output.png')
```

  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="https://offerrall.github.io/pyimagecuda/images/adjust_opacity_subtle.png" alt="80% opacity" style="width: 100%;">
    <p style="font-size: 0.9em; color: #666; margin-top: 8px;">Right-click to download and compare in full size</p>
  </div>
</div>

**Parameters:**

- `image` (Image): Image to adjust (modified in-place)
- `factor` (float): Opacity multiplier
  - `factor = 0.0`: Fully transparent
  - `factor = 1.0`: No change (original opacity)
  - `factor > 1.0`: Increases opacity (can exceed 1.0 for pixels with alpha < 1.0)

**Use for:** Creating fade effects, watermarks, layering with transparency, ghost/overlay effects

**Note:** This operation multiplies the existing alpha channel. If the image has no alpha channel or alpha is already 1.0, values > 1.0 will have no visible effect.

---

## Combining Adjustments

All adjustments can be chained since they modify in-place:

**Example:**

```python
from pyimagecuda import load, Adjust, save

img = load("photo.jpg")

# Apply multiple adjustments
Adjust.brightness(img, 0.1)
Adjust.contrast(img, 1.2)
Adjust.saturation(img, 1.3)

save(img, 'output.jpg')
```

---

## Brightness vs Gamma

**Brightness** adds a constant value:
- Affects all pixels equally
- Can wash out highlights
- Good for simple exposure fixes

**Gamma** applies a curve:
- Affects midtones more than extremes
- Preserves highlight and shadow detail
- Better for lifting underexposed images

**Example comparison:**

```python
from pyimagecuda import load, Adjust, save

# Brightness approach
img1 = load("dark_photo.jpg")
Adjust.brightness(img1, 0.3)
save(img1, 'brightened.jpg')

# Gamma approach (usually better)
img2 = load("dark_photo.jpg")
Adjust.gamma(img2, 1.5)
save(img2, 'gamma_corrected.jpg')
```