import time
import cv2
from PIL import Image as PILImage
from pyimagecuda import Transform, load, Image, cuda_sync, save, ImageU8

SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080
CROP_SIZE = 512

CROP_X = (SOURCE_WIDTH - CROP_SIZE) // 2
CROP_Y = (SOURCE_HEIGHT - CROP_SIZE) // 2


def bench_pillow_crop(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")

    img.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE))
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = img.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE))
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow", "avg_ms": avg, "fps": fps}


def bench_opencv_crop(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = img[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE].copy()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_crop_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)
    dst = Image(CROP_SIZE, CROP_SIZE)

    for _ in range(10):
        Transform.crop(src, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE, dst_buffer=dst)
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        Transform.crop(src, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE, dst_buffer=dst)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Reuse)", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_crop_no_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)

    for _ in range(10):
        tmp = Transform.crop(src, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE)
        tmp.free()
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = Transform.crop(src, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE)
        res.free()
    cuda_sync()
    t1 = time.perf_counter()

    src.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Alloc)", "avg_ms": avg, "fps": fps}


def bench_pillow_crop_e2e(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    cropped = img.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE))
    cropped.save("temp_pillow_crop.png")
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = PILImage.open(image_path).convert("RGBA")
        cropped = img.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE))
        cropped.save(f"temp_pillow_crop_{i}.png")
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow E2E", "avg_ms": avg, "fps": fps}


def bench_opencv_crop_e2e(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cropped = img[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE].copy()
    cv2.imwrite("temp_opencv_crop.png", cropped)
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        cropped = img[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE].copy()
        cv2.imwrite(f"temp_opencv_crop_{i}.png", cropped)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV E2E", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_crop_e2e(image_path: str, iterations: int = 100):
    img = load(image_path)
    cropped = Transform.crop(img, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE)
    save(cropped, "temp_cuda_crop.png")
    img.free()
    cropped.free()
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = load(image_path)
        cropped = Transform.crop(img, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE)
        save(cropped, f"temp_cuda_crop_{i}.png")
        img.free()
        cropped.free()
    cuda_sync()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_crop_e2e_buffered(image_path: str, iterations: int = 100):
    src = Image(SOURCE_WIDTH, SOURCE_HEIGHT)
    dst = Image(CROP_SIZE, CROP_SIZE)
    u8_buffer = ImageU8(SOURCE_WIDTH, SOURCE_HEIGHT)
    
    load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
    Transform.crop(src, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE, dst_buffer=dst)
    save(dst, "temp_cuda_buffered_crop.png", u8_buffer=u8_buffer)
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
        Transform.crop(src, CROP_X, CROP_Y, CROP_SIZE, CROP_SIZE, dst_buffer=dst)
        save(dst, f"temp_cuda_buffered_crop_{i}.png", u8_buffer=u8_buffer)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()
    u8_buffer.free()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E (Buffered)", "avg_ms": avg, "fps": fps}