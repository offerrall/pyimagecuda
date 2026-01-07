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

