# PyImageCUDA

[![Build Status](https://github.com/offerrall/pyimagecuda/actions/workflows/build.yml/badge.svg)](https://github.com/offerrall/pyimagecuda/actions)

> ⚠️ **STATUS: PRE-ALPHA / INFRASTRUCTURE TESTING**
>
> This repository is currently validating the build and distribution system.
> **It is NOT ready for production use yet.**

### Goal
GPU-accelerated (CUDA) image processing library for Python.
Designed to be installed via a simple `pip install` without requiring the user to have the CUDA Toolkit or Visual Studio installed.

### Verification (Testers Only)

If you have installed a test build, you can verify that hardware acceleration is working correctly by running:

```python
import pyimagecuda
# Verifies drivers, loads the DLL, and runs a test kernel
pyimagecuda.check_system()
```