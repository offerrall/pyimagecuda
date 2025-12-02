import time
import cv2
import numpy as np
from PIL import Image as PILImage
from pyimagecuda import Transform, load, Image, cuda_sync, save, ImageU8

ROTATION_ANGLE = 35
OVERKILL_SIZE = 3000 

def bench_pillow_rotate(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")

    img.rotate(-ROTATION_ANGLE, expand=True, resample=PILImage.BILINEAR)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = img.rotate(-ROTATION_ANGLE, expand=True, resample=PILImage.BILINEAR)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow", "avg_ms": avg, "fps": fps}

def bench_opencv_rotate(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    cX, cY = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D((cX, cY), -ROTATION_ANGLE, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR)

    t0 = time.perf_counter()
    for _ in range(iterations):
        M = cv2.getRotationMatrix2D((cX, cY), -ROTATION_ANGLE, 1.0)
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        _ = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_rotate_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)
    dst = Image(OVERKILL_SIZE, OVERKILL_SIZE)

    for _ in range(10):
        Transform.rotate(src, angle=ROTATION_ANGLE, expand=True, dst_buffer=dst)
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        Transform.rotate(src, angle=ROTATION_ANGLE, expand=True, dst_buffer=dst)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Reuse)", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_rotate_no_reuse(image_path: str, iterations: int = 100):
    src = load(image_path)

    for _ in range(10):
        tmp = Transform.rotate(src, angle=ROTATION_ANGLE, expand=True)
        tmp.free()
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iterations):
        res = Transform.rotate(src, angle=ROTATION_ANGLE, expand=True)
        res.free()
    cuda_sync()
    t1 = time.perf_counter()

    src.free()

    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA (Alloc)", "avg_ms": avg, "fps": fps}

def bench_pillow_rotate_e2e(image_path: str, iterations: int = 100):
    img = PILImage.open(image_path).convert("RGBA")
    rotated = img.rotate(-ROTATION_ANGLE, expand=True, resample=PILImage.BILINEAR)
    rotated.save("temp_pillow_rotate.png")
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = PILImage.open(image_path).convert("RGBA")
        rotated = img.rotate(-ROTATION_ANGLE, expand=True, resample=PILImage.BILINEAR)
        rotated.save(f"temp_pillow_rotate_{i}.png")
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "Pillow E2E", "avg_ms": avg, "fps": fps}

def bench_opencv_rotate_e2e(image_path: str, iterations: int = 100):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    h, w = img.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -ROTATION_ANGLE, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    rotated = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR)
    cv2.imwrite("temp_opencv_rotate.png", rotated)
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        M = cv2.getRotationMatrix2D((cX, cY), -ROTATION_ANGLE, 1.0)
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotated = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR)
        
        cv2.imwrite(f"temp_opencv_rotate_{i}.png", rotated)
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "OpenCV E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_rotate_e2e(image_path: str, iterations: int = 100):
    img = load(image_path)
    rotated = Transform.rotate(img, angle=ROTATION_ANGLE, expand=True)
    save(rotated, "temp_cuda_rotate.png")
    img.free()
    rotated.free()
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        img = load(image_path)
        rotated = Transform.rotate(img, angle=ROTATION_ANGLE, expand=True)
        save(rotated, f"temp_cuda_rotate_{i}.png")
        img.free()
        rotated.free()
    cuda_sync()
    t1 = time.perf_counter()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E", "avg_ms": avg, "fps": fps}

def bench_pyimagecuda_rotate_e2e_buffered(image_path: str, iterations: int = 100):
    src = Image(1920, 1080)
    dst = Image(OVERKILL_SIZE, OVERKILL_SIZE)
    u8_buffer = ImageU8(OVERKILL_SIZE, OVERKILL_SIZE)
    
    load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
    Transform.rotate(src, angle=ROTATION_ANGLE, expand=True, dst_buffer=dst)
    save(dst, "temp_cuda_buffered_rotate.png", u8_buffer=u8_buffer)
    cuda_sync()
    
    t0 = time.perf_counter()
    for i in range(iterations):
        load(image_path, f32_buffer=src, u8_buffer=u8_buffer)
        Transform.rotate(src, angle=ROTATION_ANGLE, expand=True, dst_buffer=dst)
        save(dst, f"temp_cuda_buffered_rotate_{i}.png", u8_buffer=u8_buffer)
    cuda_sync()
    t1 = time.perf_counter()

    src.free()
    dst.free()
    u8_buffer.free()
    
    avg = ((t1 - t0) / iterations) * 1000
    fps = iterations / (t1 - t0)
    return {"lib": "PyImageCUDA E2E (Buffered)", "avg_ms": avg, "fps": fps}