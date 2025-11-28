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

!!! warning "Direct Upload/Download Uses Float32"
    `upload()` and `download()` work directly with float32 data (16 bytes per pixel). For efficient file I/O, always use `load()` and `save()` instead, which handle uint8 ↔ float32 conversions automatically.

### Copy Between Buffers
```python
from pyimagecuda import Image, copy

src = Image(1920, 1080)
dst = Image(1920, 1080)

copy(dst, src)  # GPU-to-GPU copy (very fast)
```

---

## Interoperability with Other Libraries

While PyImageCUDA doesn't have built-in bridges to NumPy, PIL, or PyTorch yet, integration is straightforward using the low-level API:

### Example: NumPy Integration
```python
import numpy as np
from pyimagecuda import Image, upload, download

# NumPy → PyImageCUDA
np_array = np.random.rand(1080, 1920, 4).astype(np.float32)
img = Image(1920, 1080)
upload(img, np_array.tobytes())

# PyImageCUDA → NumPy
raw_bytes = download(img)
np_array = np.frombuffer(raw_bytes, dtype=np.float32).reshape(1080, 1920, 4)
```

### Example: PIL Integration
```python
from PIL import Image as PILImage
from pyimagecuda import Image, ImageU8, upload, convert_u8_to_float

# PIL → PyImageCUDA
pil_img = PILImage.open("photo.jpg").convert("RGBA")
u8_buffer = ImageU8(pil_img.width, pil_img.height)
f32_buffer = Image(pil_img.width, pil_img.height)

upload(u8_buffer, pil_img.tobytes())
convert_u8_to_float(f32_buffer, u8_buffer)

# Now f32_buffer is ready for GPU operations
```

**Note:** Native bridges (e.g., `from_numpy()`, `to_pil()`) may be added in future versions based on user demand.

---

## Best Practices

### For Simple Scripts
```python
# Just load and save
img = load("input.jpg")
process(img)
save(img, "output.png")
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

### For Video Frames
```python
# Fixed-size buffers for consistent frame sizes
frame = Image(1920, 1080)
u8_temp = ImageU8(1920, 1080)

for i in range(num_frames):
    load(f"frame_{i:04d}.png", f32_buffer=frame, u8_buffer=u8_temp)
    process(frame)
    save(frame, f"output_{i:04d}.png", u8_buffer=u8_temp)
```

---

## Performance Notes

Loading:

- pyvips decodes file to uint8 (CPU)
- uint8 → GPU: **4 bytes/pixel** (1920×1080 = ~8MB transfer)
- GPU converts uint8 → float32: **<1ms**

Saving:

- GPU converts float32 → uint8: **<1ms**
- uint8 → CPU: **4 bytes/pixel** (1920×1080 = ~8MB transfer)
- pyvips encodes uint8 to file (CPU)

**Why this is fast:**

- CPU↔GPU transfers use uint8 (4× smaller than float32)
- Conversions happen on GPU (massively parallel)
- Disk I/O works with compressed uint8 data

**Tip:** For maximum throughput, use fast SSDs and consider parallel loading with threading.