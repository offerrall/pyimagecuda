import time
import cv2
from PIL import Image as PILImage, ImageFilter
from pyimagecuda import Filter, load, Image, cuda_sync, save, ImageU8

def bench_pillow_blur(image_path: str, radius: int, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")

    img.filter(ImageFilter.GaussianBlur(radius))
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = img.filter(ImageFilter.GaussianBlur(radius))
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow", "avg_ms": avg, "fps": fps}

def bench_opencv_blur(image_path: str, radius: int, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_reuse(image_path: str, radius: int, iterations: int = 100):
    src = load(image_path)
    dst = Image(src.width, src.height)
    temp = Image(src.width, src.height)

    for _ in range(10):
        Filter.gaussian_blur(src, radius=radius, dst_buffer=dst, temp_buffer=temp)
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        Filter.gaussian_blur(src, radius=radius, dst_buffer=dst, temp_buffer=temp)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()
    temp.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Reuse)", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_no_reuse(image_path: str, radius: int, iterations: int = 100):
    src = load(image_path)

    for _ in range(10):
        tmp = Filter.gaussian_blur(src, radius=radius)
        tmp.free()
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = Filter.gaussian_blur(src, radius=radius)
        res.free()
    cuda_sync()
    t1 = time.perf_counter()

    src.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Alloc)", "avg_ms": avg, "fps": fps}

def bench_pillow_blur_e2e(image_path: str, radius: int, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    blurred = img.filter(ImageFilter.GaussianBlur(radius))
    blurred.save("temp_pillow_gaussian.png")
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = PILImage.open(image_path).convert("RGBA")
        blurred = img.filter(ImageFilter.GaussianBlur(radius))
        blurred.save(f"temp_pillow_gaussian_{i}.png")
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow E2E", "avg_ms": avg, "fps": fps}


def bench_opencv_blur_e2e(image_path: str, radius: int, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    cv2.imwrite("temp_opencv_gaussian.png", blurred)
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
        cv2.imwrite(f"temp_opencv_gaussian_{i}.png", blurred)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV E2E", "avg_ms": avg, "fps": fps}


def bench_pyimagecuda_blur_e2e(image_path: str, radius: int, iterations: int = 100):
    img = load(image_path)
    blurred = Filter.gaussian_blur(img, radius=radius)
    save(blurred, "temp_cuda_gaussian.png")
    img.free()
    blurred.free()
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = load(image_path)
        blurred = Filter.gaussian_blur(img, radius=radius)
        save(blurred, f"temp_cuda_gaussian_{i}.png")
        img.free()
        blurred.free()
    cuda_sync()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_blur_e2e_buffered(image_path: str, radius: int, iterations: int = 100):

    img_src = Image(1920, 1080)
    img_dst = Image(1920, 1080)
    img_temp = Image(1920, 1080)
    u8_buffer = ImageU8(1920, 1080)
    
    load(image_path, f32_buffer=img_src, u8_buffer=u8_buffer)
    Filter.gaussian_blur(img_src, radius=radius, dst_buffer=img_dst, temp_buffer=img_temp)
    save(img_dst, "temp_cuda_buffered_gaussian.png", u8_buffer=u8_buffer)
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        load(image_path, f32_buffer=img_src, u8_buffer=u8_buffer)
        Filter.gaussian_blur(img_src, radius=radius, dst_buffer=img_dst, temp_buffer=img_temp)
        save(img_dst, f"temp_cuda_buffered_gaussian_{i}.png", u8_buffer=u8_buffer)
    cuda_sync()
    t1 = time.perf_counter()

    img_src.free()
    img_dst.free()
    img_temp.free()
    u8_buffer.free()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E (Buffered)", "avg_ms": avg, "fps": fps}