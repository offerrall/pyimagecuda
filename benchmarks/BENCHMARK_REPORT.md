# PyImageCUDA Performance Report
Generated automatically by /benchmarks/benchmarks.py

## Gaussian Blur Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Radius: `20`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.97 | 1032.7 | 54.4x |
| **PyImageCUDA (Alloc)** | 3.49 | 286.4 | 15.1x |
| **OpenCV** | 50.85 | 19.7 | 1.0x |
| **Pillow** | 52.67 | 19.0 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA E2E (Buffered)** | 83.72 | 11.9 | 2.7x |
| **PyImageCUDA E2E** | 92.06 | 10.9 | 2.4x |
| **OpenCV E2E** | 100.19 | 10.0 | 2.2x |
| **Pillow E2E** | 224.84 | 4.4 | 1.0x |

---

## Blend Normal Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA** | 0.27 | 3705.1 | 376.7x |
| **Pillow** | 12.55 | 79.7 | 8.1x |
| **OpenCV** | 101.67 | 9.8 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA E2E (Buffered)** | 153.80 | 6.5 | 2.2x |
| **PyImageCUDA E2E** | 159.17 | 6.3 | 2.1x |
| **OpenCV E2E** | 242.98 | 4.1 | 1.4x |
| **Pillow E2E** | 334.47 | 3.0 | 1.0x |

---

## Resize Bilinear Benchmark (1080p -> 800x600)
> **Config:** Image: `photo.jpg`, Target: `800x600`, Interpolation: `Bilinear`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA Bilinear (Reuse)** | 0.14 | 7096.3 | 132.9x |
| **OpenCV Bilinear** | 0.52 | 1940.0 | 36.3x |
| **PyImageCUDA Bilinear (Alloc)** | 0.80 | 1249.7 | 23.4x |
| **Pillow Bilinear** | 18.73 | 53.4 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV Bilinear E2E** | 40.42 | 24.7 | 2.5x |
| **PyImageCUDA Bilinear E2E (Buffered)** | 49.53 | 20.2 | 2.0x |
| **PyImageCUDA Bilinear E2E** | 52.37 | 19.1 | 1.9x |
| **Pillow Bilinear E2E** | 99.91 | 10.0 | 1.0x |

---

## Resize Lanczos Benchmark (1080p -> 800x600)
> **Config:** Image: `photo.jpg`, Target: `800x600`, Interpolation: `Lanczos/Bicubic`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA Lanczos (Reuse)** | 0.88 | 1131.3 | 35.2x |
| **PyImageCUDA Lanczos (Alloc)** | 1.62 | 617.8 | 19.2x |
| **OpenCV Lanczos** | 4.16 | 240.2 | 7.5x |
| **Pillow Lanczos** | 31.15 | 32.1 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV Lanczos E2E** | 44.42 | 22.5 | 2.5x |
| **PyImageCUDA Lanczos E2E (Buffered)** | 52.28 | 19.1 | 2.1x |
| **PyImageCUDA Lanczos E2E** | 54.43 | 18.4 | 2.0x |
| **Pillow Lanczos E2E** | 108.93 | 9.2 | 1.0x |

---

## Rotate 35° Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Angle: `35°`, Expand: `True`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.30 | 3336.0 | 260.6x |
| **PyImageCUDA (Alloc)** | 3.31 | 301.9 | 23.6x |
| **OpenCV** | 5.98 | 167.2 | 13.1x |
| **Pillow** | 78.11 | 12.8 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV E2E** | 156.55 | 6.4 | 2.8x |
| **PyImageCUDA E2E** | 162.23 | 6.2 | 2.7x |
| **PyImageCUDA E2E (Buffered)** | 162.33 | 6.2 | 2.7x |
| **Pillow E2E** | 432.76 | 2.3 | 1.0x |

---

## Flip Horizontal Benchmark (1080p)
> **Config:** Image: `photo.jpg`, Direction: `Horizontal`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.17 | 5916.0 | 20.3x |
| **PyImageCUDA (Alloc)** | 1.73 | 577.5 | 2.0x |
| **OpenCV** | 3.12 | 320.6 | 1.1x |
| **Pillow** | 3.43 | 291.7 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV E2E** | 126.09 | 7.9 | 2.6x |
| **PyImageCUDA E2E (Buffered)** | 149.56 | 6.7 | 2.2x |
| **PyImageCUDA E2E** | 150.95 | 6.6 | 2.2x |
| **Pillow E2E** | 324.85 | 3.1 | 1.0x |

---

## Crop Center Benchmark (1080p → 512×512)
> **Config:** Image: `photo.jpg`, Source: `1920×1080`, Output: `512×512`, Iterations: `50`

### Pure Algorithm (Compute Bound)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **PyImageCUDA (Reuse)** | 0.04 | 27037.3 | 13.3x |
| **OpenCV** | 0.25 | 3987.1 | 2.0x |
| **Pillow** | 0.27 | 3692.7 | 1.8x |
| **PyImageCUDA (Alloc)** | 0.49 | 2029.6 | 1.0x |

### End-to-End (Disk I/O + Encode)
| Library | Avg (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- |
| **OpenCV E2E** | 27.37 | 36.5 | 2.0x |
| **PyImageCUDA E2E (Buffered)** | 35.72 | 28.0 | 1.6x |
| **PyImageCUDA E2E** | 38.34 | 26.1 | 1.5x |
| **Pillow E2E** | 55.69 | 18.0 | 1.0x |

---

