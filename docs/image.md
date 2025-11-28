# Image & Memory

PyImageCUDA provides two image types and flexible memory management for GPU buffers.

---

## Image Types

### `Image` - Float32 Precision

Primary image type for all operations. Stores RGBA data in 32-bit floating point (0.0 to 1.0 range).
```python
from pyimagecuda import Image

# Create a new image
img = Image(width=1920, height=1080)
```

!!! warning "Uninitialized Memory"
    Newly created images contain **uninitialized GPU memory** with random data. Always initialize before use with `Fill.color()` or load data explicitly.
```python
    img = Image(1920, 1080)
    Fill.color(img, (0, 0, 0, 0))  # Clear to transparent black
```

**When to use:** All composition, effects, and color operations. This is your default choice.

---

### `ImageU8` - 8-bit Precision

Storage-optimized type for loading/saving. Stores RGBA data as unsigned 8-bit integers (0 to 255 range).
```python
from pyimagecuda import ImageU8

# Typically used internally by load/save
u8_img = ImageU8(width=1920, height=1080)
```

**When to use:** Rarely needed directly. `load()` and `save()` handle conversions automatically.

---

## Memory Management

PyImageCUDA offers three memory management approaches:

### 1. Automatic (Garbage Collection)

Simplest approach. Python's GC cleans up when images go out of scope.
```python
from pyimagecuda import Image, Fill

img = Image(1920, 1080)
Fill.color(img, (1, 0, 0, 1))
# img will be freed automatically when no longer referenced
```

Use when: Writing simple scripts or prototypes.

---

### 2. Explicit with Context Managers

Immediate cleanup using `with` statements.
```python
from pyimagecuda import Image, Fill

with Image(1920, 1080) as img:
    Fill.color(img, (1, 0, 0, 1))
    # Process image...
# img is freed immediately here
```

Use when: Processing many images in loops or working with limited VRAM.

Example - Batch Processing:
```python
from pyimagecuda import load, save, Filter

for i in range(1000):
    with load(f"input_{i}.jpg") as img:
        Filter.gaussian_blur(img, radius=10)
        save(img, f"output_{i}.jpg")
    # Each image is freed before loading the next
```

---

### 3. Manual Control

Explicit `free()` calls for maximum control.
```python
from pyimagecuda import Image, Fill

img = Image(1920, 1080)
Fill.color(img, (1, 0, 0, 1))
img.free()  # Free immediately
```

Use when: You need precise control over when memory is released.

---

## Buffer Reuse

All operations that create temporary buffers accept optional buffer parameters for zero-allocation workflows.

### Example: Gaussian Blur
```python
from pyimagecuda import Image, Filter

src = Image(1920, 1080)
dst = Image(1920, 1080)
temp = Image(1920, 1080)  # Reusable temporary buffer

# Process 100 images reusing the same buffers
for i in range(100):
    load(f"input_{i}.jpg", f32_buffer=src)
    Filter.gaussian_blur(src, dst_buffer=dst, temp_buffer=temp)
    save(dst, f"output_{i}.jpg")

# Clean up once
src.free()
dst.free()
temp.free()
```

Benefits:

- No repeated allocations
- Consistent VRAM usage
- Critical for video processing

---

## Dynamic Buffer Sizing

Image buffers have a fixed capacity but adjustable logical dimensions.
```python
from pyimagecuda import Image

# Create buffer with capacity for 1920×1080
img = Image(1920, 1080)

# Can logically resize within capacity
img.width = 1280
img.height = 720

# Check capacity
max_w, max_h = img.get_max_capacity()
print(f"Capacity: {max_w}×{max_h}")  # 1920×1080
print(f"Current: {img.width}×{img.height}")  # 1280×720
```

Use case: Loading variable-sized images into fixed buffers.
```python
buffer = Image(4096, 4096)  # Max capacity

for filename in images:
    load(filename, f32_buffer=buffer)  # Adjusts buffer dimensions
    process(buffer)
```

---

## Best Practices

### For Simple Scripts
```python
# Just use automatic GC
img = load("input.jpg")
process(img)
save(img, "output.jpg")
```

### For Batch Processing
```python
# Use with statements
for file in files:
    with load(file) as img:
        process(img)
        save(img, output)
```

### For Video/Real-time
```python
# Reuse buffers explicitly
frame = Image(1920, 1080)
temp = Image(1920, 1080)

while video.has_frames():
    video.read_into(frame)
    process(frame, temp_buffer=temp)
    video.write(frame)

frame.free()
temp.free()
```

---

## Memory Considerations

VRAM vs RAM:

- `Image(1920, 1080)` uses ~32MB of VRAM
- Python object itself uses <100 bytes of RAM
- GC triggers on RAM pressure, not VRAM pressure, use explicit management for large workloads