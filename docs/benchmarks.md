# Performance Benchmarks

PyImageCUDA is designed for GPU-accelerated image processing. These benchmarks compare its performance against industry-standard libraries (Pillow and OpenCV) across common operations.

---

## Test Environment

All benchmarks were conducted on the following hardware:

- **CPU**: AMD Ryzen 7 3700X
- **GPU**: NVIDIA RTX 3070
- **RAM**: DDR4
- **Storage**: Standard SSD
- **OS**: Windows 11
- **Driver**: Latest NVIDIA drivers

Each test ran 50 iterations with a 1920×1080 (1080p) RGBA source image.

---

## Understanding the Results

The benchmarks are divided into two categories:

### Pure Algorithm (Compute Bound)
These tests measure **raw processing speed** without disk I/O. The image is loaded once, then the operation runs repeatedly in memory. This shows the true computational performance difference between libraries.

**When this matters:**

- Batch processing multiple operations on the same image
- Real-time video processing pipelines
- Interactive applications with live previews
- Server-side processing with images already in memory

### End-to-End (Disk I/O + Encode)
These tests include the **complete workflow**: load image from disk → process → encode → save to disk. This represents real-world scenarios where you load and save files repeatedly.

**When this matters:**

- Simple scripts that process individual files
- One-off image transformations
- Workflows where you must save intermediate results

!!! tip "Key Takeaway"
    PyImageCUDA excels when you can **keep data on the GPU** across multiple operations. For single operations with file I/O, the benefit is smaller due to CPU↔GPU transfer overhead.

---

## Performance Highlights

PyImageCUDA delivers **10-400x faster performance** in pure computation:

- **376x faster** - Blend/Composite operations
- **260x faster** - Arbitrary angle rotations  
- **132x faster** - Bilinear resizing
- **54x faster** - Gaussian blur (heavy filters)
- **35x faster** - Lanczos resizing (high-quality)

For end-to-end workflows with disk I/O, PyImageCUDA typically provides **1.5-2.7x speedup** over CPU libraries.

---

## Best Practices for Maximum Performance

### ✅ DO: Reuse Buffers
Pre-allocate buffers and reuse them across operations:

```python
from pyimagecuda import Image, load, Transform, save, ImageU8

# Pre-allocate buffers
src = Image(1920, 1080)
dst = Image(1920, 1080)
u8_buffer = ImageU8(1920, 1080)

for file in image_files:
    load(file, f32_buffer=src, u8_buffer=u8_buffer)
    Transform.flip(src, direction='horizontal', dst_buffer=dst)
    save(dst, f"output_{file}", u8_buffer=u8_buffer)

# Clean up once at the end
src.free()
dst.free()
u8_buffer.free()
```

**Result**: 3-11x faster than allocating new buffers each time.

### ✅ DO: Chain Operations on GPU
Keep data on GPU for multiple operations:

```python
from pyimagecuda import load, Filter, Adjust, save

img = load("photo.jpg")

# All operations run on GPU without CPU transfers
Filter.gaussian_blur(img, radius=10, dst_buffer=img)
Adjust.brightness(img, 0.2)
Adjust.contrast(img, 1.3)

save(img, "output.jpg")
img.free()
```

### ❌ DON'T: Use GPU for Single Simple Operations
For one-off crops or flips with file I/O, CPU libraries are competitive:

```python
# This won't be significantly faster than Pillow/OpenCV
img = load("photo.jpg")
cropped = Transform.crop(img, 0, 0, 512, 512)
save(cropped, "output.jpg")
```

Use PyImageCUDA when building **processing pipelines** with multiple operations.

---

## When to Use PyImageCUDA

### ✅ Perfect For:
- Batch processing thousands of images
- Real-time video processing (60+ FPS)
- Complex multi-step pipelines
- Interactive applications with live preview
- Heavy filters (blur, drop shadows, etc.)

### ⚠️ Not Ideal For:
- Single simple operations (crop, flip) on individual files
- Small images (<500×500)
- Environments without NVIDIA GPU
- Scripts that rarely run

---

## Detailed Results

Below are the complete benchmark results across 7 common image processing operations. Each operation was tested 50 times, and results show both pure algorithm performance and end-to-end performance including disk I/O.

---

Generated automatically by /benchmarks/benchmarks.py

# PyImageCUDA Performance Report
Generated automatically by /benchmarks/benchmarks.py

## Gaussian Blur Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Radius: `20`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.91 | 1095.4 | 56.6x |
| **PyImageCUDA (Alloc)** | 3.44 | 290.6 | 15.0x |
| **OpenCV** | 47.17 | 21.2 | 1.1x |
| **Pillow** | 51.68 | 19.4 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA E2E (Buffered)** | 79.21 | 12.6 | 2.7x |
| **PyImageCUDA E2E** | 86.14 | 11.6 | 2.5x |
| **OpenCV E2E** | 97.91 | 10.2 | 2.2x |
| **Pillow E2E** | 217.36 | 4.6 | 1.0x |

---

## Blend Normal Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA** | 0.27 | 3746.9 | 370.2x |
| **Pillow** | 12.45 | 80.3 | 7.9x |
| **OpenCV** | 98.80 | 10.1 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA E2E** | 162.79 | 6.1 | 2.0x |
| **PyImageCUDA E2E (Buffered)** | 165.58 | 6.0 | 2.0x |
| **OpenCV E2E** | 245.91 | 4.1 | 1.3x |
| **Pillow E2E** | 326.25 | 3.1 | 1.0x |

---

## Resize Bilinear Benchmark (1080p -> 800x600)
> **Config:** Image: `photo.jpg`, Target: `800x600`, Interpolation: `Bilinear`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA Bilinear (Reuse)** | 0.11 | 8730.9 | 172.6x |
| **OpenCV Bilinear** | 0.48 | 2065.1 | 40.8x |
| **PyImageCUDA Bilinear (Alloc)** | 0.75 | 1340.1 | 26.5x |
| **Pillow Bilinear** | 19.77 | 50.6 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV Bilinear E2E** | 38.11 | 26.2 | 2.6x |
| **PyImageCUDA Bilinear E2E (Buffered)** | 45.29 | 22.1 | 2.2x |
| **PyImageCUDA Bilinear E2E** | 48.03 | 20.8 | 2.1x |
| **Pillow Bilinear E2E** | 100.55 | 9.9 | 1.0x |

---

## Resize Lanczos Benchmark (1080p -> 800x600)
> **Config:** Image: `photo.jpg`, Target: `800x600`, Interpolation: `Lanczos/Bicubic`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA Lanczos (Reuse)** | 0.50 | 1999.1 | 61.7x |
| **PyImageCUDA Lanczos (Alloc)** | 1.09 | 915.4 | 28.3x |
| **OpenCV Lanczos** | 3.36 | 297.7 | 9.2x |
| **Pillow Lanczos** | 30.87 | 32.4 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV Lanczos E2E** | 45.54 | 22.0 | 2.4x |
| **PyImageCUDA Lanczos E2E (Buffered)** | 52.51 | 19.0 | 2.1x |
| **PyImageCUDA Lanczos E2E** | 53.69 | 18.6 | 2.0x |
| **Pillow Lanczos E2E** | 107.78 | 9.3 | 1.0x |

---

## Rotate 35° Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Angle: `35°`, Expand: `True`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.26 | 3827.8 | 329.3x |
| **PyImageCUDA (Alloc)** | 3.05 | 327.4 | 28.2x |
| **OpenCV** | 6.10 | 164.0 | 14.1x |
| **Pillow** | 86.02 | 11.6 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV E2E** | 145.25 | 6.9 | 2.7x |
| **PyImageCUDA E2E (Buffered)** | 147.43 | 6.8 | 2.7x |
| **PyImageCUDA E2E** | 150.59 | 6.6 | 2.6x |
| **Pillow E2E** | 398.69 | 2.5 | 1.0x |

---

## Flip Horizontal Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Direction: `Horizontal`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.17 | 5901.5 | 22.3x |
| **PyImageCUDA (Alloc)** | 1.63 | 614.1 | 2.3x |
| **OpenCV** | 3.01 | 332.4 | 1.3x |
| **Pillow** | 3.78 | 264.6 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV E2E** | 120.68 | 8.3 | 2.5x |
| **PyImageCUDA E2E (Buffered)** | 141.84 | 7.0 | 2.1x |
| **PyImageCUDA E2E** | 145.26 | 6.9 | 2.1x |
| **Pillow E2E** | 300.48 | 3.3 | 1.0x |

---

## Crop Center Benchmark (1080p → 512×512)
> **Config:** Image: `photo.jpg`, Source: `1920×1080`, Output: `512×512`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.05 | 19517.5 | 9.5x |
| **Pillow** | 0.29 | 3493.2 | 1.7x |
| **OpenCV** | 0.29 | 3475.1 | 1.7x |
| **PyImageCUDA (Alloc)** | 0.49 | 2056.4 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV E2E** | 26.86 | 37.2 | 2.1x |
| **PyImageCUDA E2E (Buffered)** | 35.04 | 28.5 | 1.6x |
| **PyImageCUDA E2E** | 35.71 | 28.0 | 1.6x |
| **Pillow E2E** | 57.40 | 17.4 | 1.0x |

---

