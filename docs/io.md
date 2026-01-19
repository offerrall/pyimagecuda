# IO (Loading and Saving)

PyImageCUDA handles image loading and saving through `pyvips`, supporting all common formats with robust decoding.

!!! info "Efficient IO Strategy"
    All image files are loaded/saved as **uint8 (8-bit)** data, then converted to/from float32 on the GPU. This minimizes disk I/O and CPU-GPU transfer overhead while maintaining float32 precision for all internal operations.

---

## Loading Images

### Basic Usage
```python
from pyimagecuda import load

img = load("photo.jpg")
# Returns Image (float32) ready for processing
```

**Supported formats:** JPG, PNG, WEBP, HEIC, GIF, TIFF, BMP, and more.

**Loading pipeline:**

1. pyvips decodes image file to uint8 RGBA (CPU)
2. uint8 data uploaded to GPU (fast - 4 bytes per pixel)
3. GPU converts uint8 → float32 (instant kernel operation)

---

### EXIF Orientation (Auto-Rotation)

By default, PyImageCUDA automatically applies EXIF orientation metadata to ensure images appear correctly oriented:
```python
# Default behavior: applies EXIF orientation
img = load("photo_from_phone.jpg")  # Displays correctly

# Disable auto-rotation if needed (advanced use cases)
img = load("photo.jpg", autorotate=False)
```

**Why this matters:**

- **Photos from phones/cameras** often store images "sideways" with EXIF orientation metadata
- **Downloaded images** from the web usually have orientation already applied
- `autorotate=True` (default) handles both cases correctly

Keep `autorotate=True` (default) for user-facing applications. Only disable for advanced workflows where you need to process the raw image data without transformations.

### Buffer Reuse

Avoid repeated allocations by reusing buffers:
```python
from pyimagecuda import Image, ImageU8, load

# Create reusable buffers
f32_buffer = Image(4096, 4096)      # Max capacity
u8_buffer = ImageU8(4096, 4096)     # Temporary conversion buffer

# Load multiple images reusing the same memory
for filename in image_files:
    load(filename, f32_buffer=f32_buffer, u8_buffer=u8_buffer)
    process(f32_buffer)
    save(f32_buffer, f"output_{filename}")
```

Benefits:

- Zero allocations after first load
- Constant VRAM usage
- Critical for batch processing

How it works:

1. `load()` reads the image file
2. Decodes into `u8_buffer` (uint8 RGBA)
3. Converts to `f32_buffer` (float32 RGBA)
4. Buffer dimensions adjust automatically within capacity

---

### Format Handling

PyImageCUDA automatically normalizes all formats to RGBA:
```python
# Grayscale → RGBA
img = load("grayscale.png")  # 1 channel → 4 channels (R=G=B, A=255)

# RGB → RGBA
img = load("photo.jpg")      # 3 channels → 4 channels (A=255)

# RGBA → RGBA
img = load("logo.png")       # 4 channels → 4 channels (unchanged)
```

All operations work uniformly on RGBA float32 images.

---

## Saving Images

### Basic Usage
```python
from pyimagecuda import save

save(img, "output.png")
```

**Supported formats:** JPG, PNG, WEBP, HEIC, TIFF, BMP.

**Saving pipeline:**

1. GPU converts float32 → uint8 (instant kernel operation)
2. uint8 data downloaded from GPU (fast - 4 bytes per pixel)
3. pyvips encodes uint8 RGBA to file (CPU)

---

### Quality Control

For lossy formats, specify compression quality:
```python
# JPEG (1-100, higher = better quality)
save(img, "photo.jpg", quality=95)

# WebP (1-100)
save(img, "photo.webp", quality=85)

# HEIC (1-100)
save(img, "photo.heic", quality=90)
```

**Default:** Maximum quality for all formats.

---

### Buffer Reuse

Reuse temporary buffers for batch saving:
```python
from pyimagecuda import Image, ImageU8, save

processed_images = [...]  # List of Image objects
u8_buffer = ImageU8(1920, 1080)

for i, img in enumerate(processed_images):
    save(img, f"output_{i}.jpg", u8_buffer=u8_buffer, quality=90)
```

How it works:

1. Converts float32 → uint8 into `u8_buffer`
2. Downloads from GPU to CPU
3. Encodes and writes to disk

### Saving U8 Images Directly

When you've manually converted an image to uint8 for another purpose (e.g., OpenGL display, external library) and also want to save it to disk:
```python
from pyimagecuda import Image, ImageU8, convert_float_to_u8, save_u8

f32_img = Image(1920, 1080)
u8_img = ImageU8(1920, 1080)

# Process in float32
process(f32_img)

# Convert once for multiple uses
convert_float_to_u8(u8_img, f32_img)

# Use u8_img for display, OpenGL, etc.
display_opengl(u8_img)

# Save the same u8 buffer
save_u8(u8_img, "output.png")
```

**Note:** Most workflows use `save()` directly with float32 images. This function eliminates redundant conversion when you already have the uint8 buffer.

---

## NumPy Integration

PyImageCUDA provides native NumPy bridges for seamless interoperability with the Python ecosystem.

!!! success "Works with OpenCV, Pillow, Matplotlib, and More!"
    Since OpenCV (`cv2.imread()`), Pillow (`Image.open()`), Matplotlib, and most Python image libraries return NumPy arrays, you can use `from_numpy()` and `to_numpy()` to work with them all.

### Basic Usage

```python
from pyimagecuda import from_numpy, to_numpy
import numpy as np

# NumPy → PyImageCUDA
np_array = np.random.rand(1080, 1920, 4).astype(np.float32)
img = from_numpy(np_array)

# PyImageCUDA → NumPy
result = to_numpy(img)  # Returns np.ndarray of shape (H, W, 4)
```

### Supported Input Formats

`from_numpy()` automatically handles common array formats:

```python
# Grayscale (H, W) → RGBA
gray = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
img = from_numpy(gray)  # Expands to RGBA: R=G=B, A=255

# RGB (H, W, 3) → RGBA
rgb = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
img = from_numpy(rgb)  # Adds alpha channel: A=255

# RGBA (H, W, 4) → RGBA
rgba = np.random.rand(1080, 1920, 4).astype(np.float32)
img = from_numpy(rgba)  # Direct upload

# Supported dtypes: uint8 (0-255) or float32 (0.0-1.0)
```

**Conversion pipeline:**

- **uint8 input:** Uploads as 4 bytes/pixel → GPU converts to float32 (optimized)
- **float32 input:** Direct upload as 16 bytes/pixel (no conversion needed)

---

### OpenCV Integration

```python
import cv2
from pyimagecuda import from_numpy, to_numpy, adjust_saturation

# Load image with OpenCV
cv_img = cv2.imread("photo.jpg")  # BGR uint8
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

# Process on GPU
gpu_img = from_numpy(cv_img)
adjust_saturation(gpu_img, 1.5)

# Back to OpenCV
result = to_numpy(gpu_img)
result = (result[:, :, :3] * 255).astype(np.uint8)  # Float32 → uint8, drop alpha
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  # RGB → BGR
cv2.imwrite("output.jpg", result)
```

---

### Pillow Integration

```python
from PIL import Image as PILImage
from pyimagecuda import from_numpy, to_numpy, blur
import numpy as np

# Load with Pillow
pil_img = PILImage.open("photo.jpg").convert("RGBA")
np_array = np.array(pil_img)

# Process on GPU
gpu_img = from_numpy(np_array)
blur(gpu_img, 10)

# Back to Pillow
result = to_numpy(gpu_img)
result = (result * 255).astype(np.uint8)  # Float32 → uint8
pil_result = PILImage.fromarray(result, mode="RGBA")
pil_result.save("output.png")
```

---

### Matplotlib Integration

```python
import matplotlib.pyplot as plt
from pyimagecuda import from_numpy, to_numpy, adjust_exposure

# Load from Matplotlib
img_array = plt.imread("photo.png")  # Returns float32 [0.0, 1.0]

# Process on GPU
gpu_img = from_numpy(img_array)
adjust_exposure(gpu_img, 0.5)

# Display result
result = to_numpy(gpu_img)
plt.imshow(result)
plt.show()
```

---

### Buffer Reuse for Performance

Reuse buffers to eliminate allocations in tight loops:

```python
from pyimagecuda import Image, ImageU8, from_numpy
import cv2

# Create reusable buffers
f32_buffer = Image(1920, 1080)
u8_buffer = ImageU8(1920, 1080)

# Process video frames
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Reuses existing GPU memory
    from_numpy(frame, f32_buffer=f32_buffer, u8_buffer=u8_buffer)
    process(f32_buffer)
    
    result = to_numpy(f32_buffer)
    cv2.imshow("Processed", (result[:, :, :3] * 255).astype(np.uint8))
```

**Benefits:**

- Zero GPU allocations after first frame
- Constant VRAM usage
- Critical for real-time video processing

---

## Low-Level Operations

For advanced use cases, you can access the underlying conversion functions:

### Manual Conversions
```python
from pyimagecuda import Image, ImageU8, convert_u8_to_float, convert_float_to_u8

# uint8 → float32
u8_img = ImageU8(1920, 1080)
f32_img = Image(1920, 1080)
convert_u8_to_float(f32_img, u8_img)

# float32 → uint8
convert_float_to_u8(u8_img, f32_img)
```

### Direct Upload/Download
```python
from pyimagecuda import Image, upload, download

# Upload raw RGBA float32 bytes to GPU
img = Image(512, 512)
raw_data = bytes(512 * 512 * 16)  # 16 bytes per pixel (4 × float32)
upload(img, raw_data)

# Download from GPU to CPU
raw_data = download(img)  # Returns bytes
```

"Raw Data Types" upload() and download() transfer raw bytes without conversion. * If using Image, data must be float32 (16 bytes per pixel). * If using ImageU8, data must be uint8 (4 bytes per pixel).

### Copy Between Buffers
```python
from pyimagecuda import Image, copy

src = Image(1920, 1080)
dst = Image(1920, 1080)

copy(dst, src)  # GPU-to-GPU copy (very fast)
```

---

## Best Practices

### For Simple Scripts
```python
# Just load and save
img = load("input.jpg")
process(img)
save(img, "output.png")
```

### For NumPy/OpenCV/Pillow Workflows
```python
import cv2
from pyimagecuda import from_numpy, to_numpy

# Load with your preferred library
frame = cv2.imread("photo.jpg")

# Process on GPU
gpu_img = from_numpy(frame)
process(gpu_img)

# Back to CPU
result = to_numpy(gpu_img)
cv2.imwrite("output.jpg", result)
```

### For Batch Processing
```python
# Reuse buffers
f32 = Image(4096, 4096)
u8 = ImageU8(4096, 4096)

for file in files:
    load(file, f32_buffer=f32, u8_buffer=u8)
    process(f32)
    save(f32, output_file, u8_buffer=u8)
```

### For Video Processing
```python
import cv2
from pyimagecuda import Image, ImageU8, from_numpy, to_numpy

# Fixed-size buffers for consistent frame sizes
frame_buffer = Image(1920, 1080)
u8_temp = ImageU8(1920, 1080)

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    from_numpy(frame, f32_buffer=frame_buffer, u8_buffer=u8_temp)
    process(frame_buffer)
    
    result = to_numpy(frame_buffer)
    cv2.imshow("Output", (result * 255).astype(np.uint8))
```

---

## Performance Notes

**NumPy Integration:**

- **uint8 arrays:** Uploads 4 bytes/pixel → GPU converts to float32 (<1ms for 1920×1080)
- **float32 arrays:** Direct upload 16 bytes/pixel (no conversion)
- **Download:** Always float32 → 16 bytes/pixel transfer

**File Loading:**

- pyvips decodes file to uint8 (CPU)
- uint8 → GPU: **4 bytes/pixel** (1920×1080 = ~8MB transfer)
- GPU converts uint8 → float32: **<1ms**

**File Saving:**

- GPU converts float32 → uint8: **<1ms**
- uint8 → CPU: **4 bytes/pixel** (1920×1080 = ~8MB transfer)
- pyvips encodes uint8 to file (CPU)

**Why this is fast:**

- CPU↔GPU transfers prefer uint8 when possible (4× smaller than float32)
- Conversions happen on GPU (massively parallel)
- NumPy bridge uses optimized upload/download paths

**Tip:** For maximum throughput with NumPy arrays, prefer uint8 input when possible, and use buffer reuse for batch processing.