# OpenGL Integration

PyImageCUDA provides a GPU-to-GPU transfer with OpenGL through CUDA-OpenGL interop, enabling efficient real-time preview pipelines.

---

## Overview

**Traditional approach:**
```
GPU Processing → CPU Download → CPU Upload → OpenGL Display
```

**With OpenGL interop:**
```
GPU Processing → GPU Copy → OpenGL Display
```

**Performance:**

- Zero CPU involvement (all transfers GPU-side)
- ~10x faster than CPU download/upload
- Sub-millisecond updates for 2K images (~1ms measured)
- Single GPU-GPU DMA copy (PBO→Texture, asynchronous)

---

## GLResource API

### Constructor
```python
GLResource(pbo_id: int)
```
Registers an OpenGL PBO with CUDA for interop.

**Parameters:**
- `pbo_id`: OpenGL buffer ID from `glGenBuffers()`

**Raises:**
- `ValueError`: If `pbo_id` is invalid
- `RuntimeError`: If CUDA registration fails

---

### copy_from()
```python
gl_resource.copy_from(image: ImageU8) -> None
```
Copies ImageU8 data directly to PBO (GPU→GPU, zero CPU overhead).

**Parameters:**
- `image`: Source ImageU8 buffer

**Raises:**
- `TypeError`: If image is not ImageU8
- `RuntimeError`: If resource has been freed

---

### free()
```python
gl_resource.free() -> None
```
Unregisters the resource. Must be called before deleting the PBO.

**Context Manager:**
```python
with GLResource(pbo) as gl_resource:
    gl_resource.copy_from(image)
```

## Best Practices

### Display Integration
- Invert texture Y-coordinates if image appears upside-down
- Enable `GL_BLEND` for alpha channel support

### Resource Management
- Create PBO with `GL_STREAM_DRAW` for best performance
- Pre-allocate buffers to avoid reallocation overhead
- Call `free()` before deleting OpenGL objects

### Performance
- Reuse `ImageU8` buffers when possible
- Use texture size matching image size (or larger)
- `glTexSubImage2D` with PBO is asynchronous (no CPU stall)