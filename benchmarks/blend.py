import time
import cv2
from PIL import Image as PILImage
from pyimagecuda import Blend, load, Image, cuda_sync, save, ImageU8

def bench_pillow_blend(image_path: str, iterations: int = 100):
    base = PILImage.open(image_path).convert("RGBA")
    overlay = PILImage.open(image_path).convert("RGBA")

    PILImage.alpha_composite(base, overlay)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = PILImage.alpha_composite(base, overlay)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow", "avg_ms": avg, "fps": fps}

def bench_opencv_blend(image_path: str, iterations: int = 100):
    base = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if base.shape[2] == 3:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    alpha = overlay[:, :, 3:4] / 255.0
    result = (1 - alpha) * base + alpha * overlay
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        alpha = overlay[:, :, 3:4] / 255.0
        _ = (1 - alpha) * base + alpha * overlay
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_blend(image_path: str, iterations: int = 100):
    base = load(image_path)
    overlay = load(image_path)

    for _ in range(10):
        Blend.normal(base, overlay)
    cuda_sync()

    base = load(image_path)
    t0 = time.perf_counter()
    for _ in range(iterations):
        Blend.normal(base, overlay)
    cuda_sync()
    t1 = time.perf_counter()

    base.free()
    overlay.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA", "avg_ms": avg, "fps": fps}

def bench_pillow_blend_e2e(image_path: str, iterations: int = 100):
    base = PILImage.open(image_path).convert("RGBA")
    overlay = PILImage.open(image_path).convert("RGBA")
    result = PILImage.alpha_composite(base, overlay)
    result.save("temp_pillow_blend.png")
    
    t0 = time.perf_counter()
    for i in range(iterations):
        base = PILImage.open(image_path).convert("RGBA")
        overlay = PILImage.open(image_path).convert("RGBA")
        result = PILImage.alpha_composite(base, overlay)
        result.save(f"temp_pillow_{i}_blend.png")
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow E2E", "avg_ms": avg, "fps": fps}

def bench_opencv_blend_e2e(image_path: str, iterations: int = 100):
    base = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if base.shape[2] == 3:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    alpha = overlay[:, :, 3:4] / 255.0
    result = (1 - alpha) * base + alpha * overlay
    cv2.imwrite("temp_opencv_blend.png", result.astype('uint8'))
    
    t0 = time.perf_counter()
    for i in range(iterations):
        base = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if base.shape[2] == 3:
            base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        alpha = overlay[:, :, 3:4] / 255.0
        result = (1 - alpha) * base + alpha * overlay
        cv2.imwrite(f"temp_opencv_{i}_blend.png", result.astype('uint8'))
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_blend_e2e(image_path: str, iterations: int = 100):
    base = load(image_path)
    overlay = load(image_path)
    Blend.normal(base, overlay)
    save(base, "temp_cuda_blend.png")
    base.free()
    overlay.free()
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        base = load(image_path)
        overlay = load(image_path)
        Blend.normal(base, overlay)
        save(base, f"temp_cuda_{i}_blend.png")
        base.free()
        overlay.free()
    cuda_sync()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_blend_e2e_buffered(image_path: str, iterations: int = 100):
    base = Image(1920, 1080)
    overlay = Image(1920, 1080)
    u8_buffer = ImageU8(1920, 1080)
    
    load(image_path, f32_buffer=base, u8_buffer=u8_buffer)
    load(image_path, f32_buffer=overlay, u8_buffer=u8_buffer)
    Blend.normal(base, overlay)
    save(base, "temp_cuda_buffered_blend.png", u8_buffer=u8_buffer)
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        load(image_path, f32_buffer=base, u8_buffer=u8_buffer)
        load(image_path, f32_buffer=overlay, u8_buffer=u8_buffer)
        Blend.normal(base, overlay)
        save(base, f"temp_cuda_buffered_blend{i}.png", u8_buffer=u8_buffer)
    cuda_sync()
    t1 = time.perf_counter()

    base.free()
    overlay.free()
    u8_buffer.free()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E (Buffered)", "avg_ms": avg, "fps": fps}