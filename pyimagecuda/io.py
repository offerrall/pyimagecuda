import pyvips

from .pyimagecuda_internal import upload_to_buffer, convert_f32_to_u8, convert_u8_to_f32, download_from_buffer, copy_buffer  #type: ignore
from .image import Image, ImageU8, ImageBase
from .utils import check_dimensions_match



def upload(image: ImageBase, data: bytes | bytearray | memoryview) -> None:
    bytes_per_pixel = 4 if isinstance(image, ImageU8) else 16
    expected = image.width * image.height * bytes_per_pixel
    actual = data.nbytes if isinstance(data, memoryview) else len(data)
    
    if actual != expected:
        raise ValueError(f"Expected {expected} bytes, got {actual}")
    
    upload_to_buffer(image._buffer._handle, data, image.width, image.height)


def download(image: ImageBase) -> bytes:
    return download_from_buffer(image._buffer._handle, image.width, image.height)


def copy(dst: ImageBase, src: ImageBase) -> None:
    check_dimensions_match(dst, src)
    copy_buffer(dst._buffer._handle, src._buffer._handle, src.width, src.height)


def convert_float_to_u8(dst: ImageU8, src: Image) -> None:
    check_dimensions_match(dst, src)
    convert_f32_to_u8(dst._buffer._handle, src._buffer._handle, src.width, src.height)


def convert_u8_to_float(dst: Image, src: ImageU8) -> None:
    check_dimensions_match(dst, src)
    convert_u8_to_f32(dst._buffer._handle, src._buffer._handle, src.width, src.height)


def load(filepath: str, f32_buffer: Image | None = None, u8_buffer: ImageU8 | None = None) -> Image:    
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
    
    if f32_buffer is None:
        f32_buffer = Image(width, height)
    else:
        max_w, max_h = f32_buffer.get_max_capacity()
        if width > max_w or height > max_h:
            raise ValueError(
                f"Image {width}×{height} exceeds f32_buffer capacity {max_w}×{max_h}"
            )

        f32_buffer.width = width
        f32_buffer.height = height

    if u8_buffer is None:
        u8_buffer = ImageU8(width, height)
        owns_u8 = True
    else:
        max_w, max_h = u8_buffer.get_max_capacity()
        if width > max_w or height > max_h:
            raise ValueError(
                f"Image {width}×{height} exceeds u8_buffer capacity {max_w}×{max_h}"
            )
        
        u8_buffer.width = width
        u8_buffer.height = height
        owns_u8 = False

    vips_img = vips_img.cast('uchar')
    pixel_data = vips_img.write_to_memory()
    
    upload(u8_buffer, pixel_data)
    
    convert_u8_to_float(f32_buffer, u8_buffer)
    
    if owns_u8:
        u8_buffer.free()
    
    return f32_buffer


def save(image: Image, filepath: str, u8_buffer: ImageU8 | None = None, quality: int | None = None) -> None:
    if u8_buffer is None:
        u8_buffer = ImageU8(image.width, image.height)
        owns_buffer = True
    else:
        check_dimensions_match(u8_buffer, image)
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