import time
import cv2
from PIL import Image as PILImage
from pyimagecuda import Resize, load, Image, cuda_sync, save, ImageU8

TARGET_WIDTH = 800
TARGET_HEIGHT = 600

def bench_pillow_resize_bilinear(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.BILINEAR)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.BILINEAR)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow Bilinear", "avg_ms": avg, "fps": fps}

def bench_pillow_resize_lanczos(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.LANCZOS)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.LANCZOS)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow Lanczos", "avg_ms": avg, "fps": fps}

def bench_opencv_resize_bilinear(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV Bilinear", "avg_ms": avg, "fps": fps}

def bench_opencv_resize_lanczos(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV Lanczos", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_bilinear_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)
    dst = Image(TARGET_WIDTH, TARGET_HEIGHT)

    for _ in range(10):
        Resize.bilinear(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        Resize.bilinear(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Bilinear (Reuse)", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_bilinear_no_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)

    for _ in range(10):
        tmp = Resize.bilinear(src, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        tmp.free()
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = Resize.bilinear(src, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        res.free()
    cuda_sync()
    t1 = time.perf_counter()

    src.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Bilinear (Alloc)", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_lanczos_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)
    dst = Image(TARGET_WIDTH, TARGET_HEIGHT)

    for _ in range(10):
        Resize.lanczos(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        Resize.lanczos(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Lanczos (Reuse)", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_lanczos_no_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)

    for _ in range(10):
        tmp = Resize.lanczos(src, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        tmp.free()
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = Resize.lanczos(src, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        res.free()
    cuda_sync()
    t1 = time.perf_counter()

    src.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Lanczos (Alloc)", "avg_ms": avg, "fps": fps}

def bench_pillow_resize_bilinear_e2e(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.BILINEAR)
    resized.save("temp_resize_pillow.png")
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = PILImage.open(image_path).convert("RGBA")
        resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.BILINEAR)
        resized.save(f"temp_resize_pillow_{i}.png")
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow Bilinear E2E", "avg_ms": avg, "fps": fps}

def bench_pillow_resize_lanczos_e2e(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.LANCZOS)
    resized.save("temp_resize_pillow.png")
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = PILImage.open(image_path).convert("RGBA")
        resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), PILImage.LANCZOS)
        resized.save(f"temp_resize_pillow_{i}.png")
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow Lanczos E2E", "avg_ms": avg, "fps": fps}

def bench_opencv_resize_bilinear_e2e(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("temp_resize_opencv.png", resized)
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"temp_resize_opencv_{i}.png", resized)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV Bilinear E2E", "avg_ms": avg, "fps": fps}

def bench_opencv_resize_lanczos_e2e(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("temp_resize_opencv.png", resized)
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(f"temp_resize_opencv_{i}.png", resized)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV Lanczos E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_bilinear_e2e(image_path: str, iterations: int = 100):
    img = load(image_path)
    resized = Resize.bilinear(img, width=TARGET_WIDTH, height=TARGET_HEIGHT)
    save(resized, "temp_resize_cuda.png")
    img.free()
    resized.free()
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = load(image_path)
        resized = Resize.bilinear(img, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        save(resized, f"temp_resize_cuda_{i}.png")
        img.free()
        resized.free()
    cuda_sync()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Bilinear E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_lanczos_e2e(image_path: str, iterations: int = 100):
    img = load(image_path)
    resized = Resize.lanczos(img, width=TARGET_WIDTH, height=TARGET_HEIGHT)
    save(resized, "temp_resize_cuda.png")
    img.free()
    resized.free()
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = load(image_path)
        resized = Resize.lanczos(img, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        save(resized, f"temp_resize_cuda_{i}.png")
        img.free()
        resized.free()
    cuda_sync()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Lanczos E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_bilinear_e2e_buffered(image_path: str, iterations: int = 100):
    src = Image(1920, 1080)
    dst = Image(TARGET_WIDTH, TARGET_HEIGHT)
    u8_buffer = ImageU8(1920, 1080)
    
    load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
    Resize.bilinear(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
    save(dst, "temp_resize_cuda_buffered.png", u8_buffer=u8_buffer)
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
        Resize.bilinear(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
        save(dst, f"temp_resize_cuda_buffered_{i}.png", u8_buffer=u8_buffer)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()
    u8_buffer.free()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Bilinear E2E (Buffered)", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_resize_lanczos_e2e_buffered(image_path: str, iterations: int = 100):
    src = Image(1920, 1080)
    dst = Image(TARGET_WIDTH, TARGET_HEIGHT)
    u8_buffer = ImageU8(1920, 1080)
    
    load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
    Resize.lanczos(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
    save(dst, "temp_resize_cuda_buffered.png", u8_buffer=u8_buffer)
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
        Resize.lanczos(src, width=TARGET_WIDTH, height=TARGET_HEIGHT, dst_buffer=dst)
        save(dst, f"temp_resize_cuda_buffered_{i}.png", u8_buffer=u8_buffer)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()
    u8_buffer.free()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA Lanczos E2E (Buffered)", "avg_ms": avg, "fps": fps}