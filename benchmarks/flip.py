import time
import cv2
from PIL import Image as PILImage
from pyimagecuda import Transform, load, Image, cuda_sync, save, ImageU8


def bench_pillow_flip(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")

    img.transpose(PILImage.FLIP_LEFT_RIGHT)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = img.transpose(PILImage.FLIP_LEFT_RIGHT)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow", "avg_ms": avg, "fps": fps}


def bench_opencv_flip(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    cv2.flip(img, 1)  # 1 = horizontal flip
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cv2.flip(img, 1)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_flip_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)
    dst = Image(src.width, src.height)

    for _ in range(10):
        Transform.flip(src, direction='horizontal', dst_buffer=dst)
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        Transform.flip(src, direction='horizontal', dst_buffer=dst)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Reuse)", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_flip_no_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)

    for _ in range(10):
        tmp = Transform.flip(src, direction='horizontal')
        tmp.free()
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = Transform.flip(src, direction='horizontal')
        res.free()
    cuda_sync()
    t1 = time.perf_counter()

    src.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Alloc)", "avg_ms": avg, "fps": fps}


def bench_pillow_flip_e2e(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    flipped = img.transpose(PILImage.FLIP_LEFT_RIGHT)
    flipped.save("temp_pillow_flip.png")
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = PILImage.open(image_path).convert("RGBA")
        flipped = img.transpose(PILImage.FLIP_LEFT_RIGHT)
        flipped.save(f"temp_pillow_flip_{i}.png")
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow E2E", "avg_ms": avg, "fps": fps}


def bench_opencv_flip_e2e(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    flipped = cv2.flip(img, 1)
    cv2.imwrite("temp_opencv_flip.png", flipped)
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        flipped = cv2.flip(img, 1)
        cv2.imwrite(f"temp_opencv_flip_{i}.png", flipped)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV E2E", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_flip_e2e(image_path: str, iterations: int = 100):
    img = load(image_path)
    flipped = Transform.flip(img, direction='horizontal')
    save(flipped, "temp_cuda_flip.png")
    img.free()
    flipped.free()
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = load(image_path)
        flipped = Transform.flip(img, direction='horizontal')
        save(flipped, f"temp_cuda_flip_{i}.png")
        img.free()
        flipped.free()
    cuda_sync()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_flip_e2e_buffered(image_path: str, iterations: int = 100):
    src = Image(1920, 1080)
    dst = Image(1920, 1080)
    u8_buffer = ImageU8(1920, 1080)
    
    load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
    Transform.flip(src, direction='horizontal', dst_buffer=dst)
    save(dst, "temp_cuda_buffered_flip.png", u8_buffer=u8_buffer)
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
        Transform.flip(src, direction='horizontal', dst_buffer=dst)
        save(dst, f"temp_cuda_buffered_flip_{i}.png", u8_buffer=u8_buffer)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()
    u8_buffer.free()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E (Buffered)", "avg_ms": avg, "fps": fps}