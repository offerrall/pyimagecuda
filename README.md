# PyImageCUDA 0.0.2

[![PyPI version](https://img.shields.io/pypi/v/pyimagecuda.svg)](https://pypi.org/project/pyimagecuda/)
[![Build Status](https://github.com/offerrall/pyimagecuda/actions/workflows/build.yml/badge.svg)](https://github.com/offerrall/pyimagecuda/actions)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)

**GPU-accelerated image compositing for Python.**

> PyImageCUDA focuses on creative image generation rather than computer vision. Expect GPU-accelerated effects for design workflows—blending modes, shadows, gradients, filters... not edge detection or object recognition.

## Quick Example
```python
from pyimagecuda import Image, Fill, Effect, Blend, save

with Image(1024, 1024) as bg:
    with Image(512, 512) as card:
        Fill.color(bg, (1, 1, 1, 1))
        Fill.gradient(card, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')
        Effect.rounded_corners(card, 50)

        with Effect.drop_shadow(card, blur=50, color=(0, 0, 0, 1)) as shadowed:
            Blend.normal(bg, shadowed, anchor='center')

        save(bg, 'output.png')
```
<img src="docs/images/quick.png" alt="Demo" width="400">

## Key Features

* ✅ **Zero Dependencies:** No CUDA Toolkit, Visual Studio, or complex compilers needed. Is Plug & Play.
* ✅ **Ultra-lightweight:** Core library weighs **<0.5 MB**.
* ✅ **Studio Quality:** 32-bit floating-point precision (float32) to prevent color banding.
* ✅ **Advanced Memory Control:** Reuse GPU buffers across operations and resize without reallocation—critical for video processing and batch workflows.
* ✅ **API Simplicity:** Intuitive, Pythonic API designed for ease of use.

## Use Cases

* **Generative Art:** Create thousands of unique variations in seconds.
* **Motion Graphics:** Process video frames or generate effects in real-time.
* **Image Compositing:** Complex multi-layer designs with GPU-accelerated effects.
* **Game Development:** Procedural UI assets, icons, and sprite generation.
* **Marketing Automation:** Mass-produce personalized graphics from templates.
* **Data Augmentation:** High-speed batch transformations for ML datasets.

## Installation
```bash
pip install pyimagecuda
```

**Note:** Automatically installs `pyvips` binary dependencies for robust image format support (JPG, PNG, WEBP, HEIC).

## Documentation

**⚠️ Alpha Release:** This is version 0.0.2 with core functionality. Many more features are planned and under development.

### Core Concepts
* [Getting Started Guide](https://offerrall.github.io/pyimagecuda/)
* [Image & Memory](https://offerrall.github.io/pyimagecuda/image/) (Buffer management)
* [IO](https://offerrall.github.io/pyimagecuda/io/) (Loading and Saving)

### Operations (v0.0.2)
* [Blend](https://offerrall.github.io/pyimagecuda/blend/) (Normal, Multiply, Screen, Add)
* [Resize](https://offerrall.github.io/pyimagecuda/resize/) (Nearest, Bilinear, Bicubic, Lanczos)
* [Filter](https://offerrall.github.io/pyimagecuda/filter/) (Gaussian Blur, Sharpen)
* [Effect](https://offerrall.github.io/pyimagecuda/effect/) (Drop Shadow, Rounded Corners)
* [Fill](https://offerrall.github.io/pyimagecuda/fill/) (Solid colors, Gradients)

## Requirements

* **OS:** Windows 10 or 11 (64-bit). *Linux support coming soon.*
* **GPU:** NVIDIA GPU (Maxwell architecture / GTX 900 series or newer).
* **Drivers:** Standard NVIDIA Drivers installed.

**NOT REQUIRED:** Visual Studio, CUDA Toolkit, or Conda.