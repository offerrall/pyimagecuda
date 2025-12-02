import pyvips

from .pyimagecuda_internal import upload_to_buffer, convert_f32_to_u8, convert_u8_to_f32, download_from_buffer, copy_buffer  #type: ignore
from .image import Image, ImageU8, ImageBase
from .utils import ensure_capacity

try:
    import numpy as np
except ImportError:
    np = None

def upload(image: ImageBase, data: bytes | bytearray | memoryview) -> None:
    """
    Uploads the image data from a bytes-like object to the GPU.

    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#direct-uploaddownload
    """
    bytes_per_pixel = 4 if isinstance(image, ImageU8) else 16
    expected = image.width * image.height * bytes_per_pixel
    actual = data.nbytes if isinstance(data, memoryview) else len(data)
    
    if actual != expected:
        raise ValueError(f"Expected {expected} bytes, got {actual}")
    
    upload_to_buffer(image._buffer._handle, data, image.width, image.height)


def download(image: ImageBase) -> bytes:
    """
    Downloads the image data from the GPU to a bytes object.

    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#direct-uploaddownload
    """
    return download_from_buffer(image._buffer._handle, image.width, image.height)


def copy(dst: ImageBase, src: ImageBase) -> None:
    """
    Copies image data from the source image to the destination image.

    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#copy-between-buffers
    """
    ensure_capacity(dst, src.width, src.height)
    copy_buffer(dst._buffer._handle, src._buffer._handle, src.width, src.height)


def convert_float_to_u8(dst: ImageU8, src: Image) -> None:
    """
    Converts a floating-point image to an 8-bit unsigned integer image.

    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#manual-conversions
    """
    ensure_capacity(dst, src.width, src.height)
    convert_f32_to_u8(dst._buffer._handle, src._buffer._handle, src.width, src.height)


def convert_u8_to_float(dst: Image, src: ImageU8) -> None:
    """
    Converts an 8-bit unsigned integer image to a floating-point image.

    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#manual-conversions
    """
    ensure_capacity(dst, src.width, src.height)
    convert_u8_to_f32(dst._buffer._handle, src._buffer._handle, src.width, src.height)


def load(
    filepath: str, 
    f32_buffer: Image | None = None, 
    u8_buffer: ImageU8 | None = None
) -> Image | None:
    """
    Loads an image from a file (returns new image or writes to buffer).
    
    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#loading-images
    """
    vips_img = pyvips.Image.new_from_file(filepath, access='sequential')

    if vips_img.bands == 1:
        vips_img = vips_img.bandjoin([vips_img, vips_img, vips_img])
        vips_img = vips_img.bandjoin(255)
    elif vips_img.bands == 3:
        vips_img = vips_img.bandjoin(255)
    elif vips_img.bands == 4:
        pass
    else:
        raise ValueError(
            f"Unsupported image format: {vips_img.bands} channels. "
            f"Only grayscale (1), RGB (3), and RGBA (4) are supported."
        )
    
    width = vips_img.width
    height = vips_img.height

    should_return = False
    
    if f32_buffer is None:
        f32_buffer = Image(width, height)
        should_return = True
    else:
        ensure_capacity(f32_buffer, width, height)
        should_return = False

    if u8_buffer is None:
        u8_buffer = ImageU8(width, height)
        owns_u8 = True
    else:
        ensure_capacity(u8_buffer, width, height)
        owns_u8 = False

    vips_img = vips_img.cast('uchar')
    pixel_data = vips_img.write_to_memory()
    
    upload(u8_buffer, pixel_data)
    
    convert_u8_to_float(f32_buffer, u8_buffer)

    if owns_u8:
        u8_buffer.free()
    
    return f32_buffer if should_return else None


def save(image: Image, filepath: str, u8_buffer: ImageU8 | None = None, quality: int | None = None) -> None:
    """
    Saves the floating-point image to a file (using an 8-bit buffer for conversion).

    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#saving-images
    """
    if u8_buffer is None:
        u8_buffer = ImageU8(image.width, image.height)
        owns_buffer = True
    else:
        ensure_capacity(u8_buffer, image.width, image.height)
        owns_buffer = False
    
    convert_float_to_u8(u8_buffer, image)
    pixel_data = download(u8_buffer)
    
    vips_img = pyvips.Image.new_from_memory(
        pixel_data,
        image.width,
        image.height,
        bands=4,
        format='uchar'
    )
    
    vips_img = vips_img.copy(interpretation='srgb')
    
    save_kwargs = {}
    if quality is not None:
        if filepath.lower().endswith(('.jpg', '.jpeg')):
            save_kwargs['Q'] = quality
        elif filepath.lower().endswith('.webp'):
            save_kwargs['Q'] = quality
        elif filepath.lower().endswith(('.heic', '.heif')):
            save_kwargs['Q'] = quality
    
    vips_img.write_to_file(filepath, **save_kwargs)
    
    if owns_buffer:
        u8_buffer.free()

def from_numpy(array, f32_buffer: Image | None = None, u8_buffer: ImageU8 | None = None) -> Image:
    """
    Creates a PyImageCUDA Image from a NumPy array (e.g. from OpenCV, Pillow, Matplotlib).
    
    - Handles uint8 (0-255) -> float32 (0.0-1.0) conversion automatically on GPU.
    - Handles Grayscale/RGB -> RGBA expansion automatically.
    - Optimized: Uploads uint8 data (4x smaller) if possible, then converts on GPU.

    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#numpy-integration
    """
    if np is None:
        raise ImportError("NumPy is not installed. Run `pip install numpy` to use this feature.")

    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

    target_dtype = array.dtype

    if array.ndim == 2:
        h, w = array.shape
        alpha_val = 255 if target_dtype == np.uint8 else 1.0
        alpha_channel = np.full((h, w), alpha_val, dtype=target_dtype)
        array = np.dstack((array, array, array, alpha_channel))

    elif array.ndim == 3:
        h, w, c = array.shape
        if c == 3:
            alpha_val = 255 if target_dtype == np.uint8 else 1.0
            alpha_channel = np.full((h, w), alpha_val, dtype=target_dtype)
            array = np.dstack((array, alpha_channel))
        elif c != 4:
            raise ValueError(f"Unsupported channel count: {c}. PyImageCUDA requires 1, 3, or 4 channels.")
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}. Expected (H, W), (H, W, 3) or (H, W, 4).")

    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)

    height, width = array.shape[:2]

    should_return = False
    if f32_buffer is None:
        f32_buffer = Image(width, height)
        should_return = True
    else:
        ensure_capacity(f32_buffer, width, height)

    if array.dtype == np.uint8:
        owns_u8 = False
        if u8_buffer is None:
            u8_buffer = ImageU8(width, height)
            owns_u8 = True
        else:
            ensure_capacity(u8_buffer, width, height)

        upload(u8_buffer, array.tobytes())
        convert_u8_to_float(f32_buffer, u8_buffer)
        
        if owns_u8:
            u8_buffer.free()

    elif array.dtype == np.float32:
        upload(f32_buffer, array.tobytes())
    else:
        array = array.astype(np.float32)
        upload(f32_buffer, array.tobytes())

    return f32_buffer if should_return else None


def to_numpy(image: Image) -> 'np.ndarray': # type: ignore
    """
    Downloads a PyImageCUDA Image to a NumPy array.
    
    Docs & Examples: https://offerrall.github.io/pyimagecuda/io/#numpy-integration
    """
    if np is None:
        raise ImportError("NumPy is not installed. Run `pip install numpy` to use this feature.")
    
    raw_bytes = download(image)
    array = np.frombuffer(raw_bytes, dtype=np.float32)
    
    return array.reshape((image.height, image.width, 4))