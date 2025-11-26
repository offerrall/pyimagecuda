from .image import Image
from .utils import check_dimensions_match
from pyimagecuda_internal import gaussian_blur_separable_f32, sharpen_f32


class Filter:

    @staticmethod
    def gaussian_blur(
        src: Image,
        radius: int = 3,
        sigma: float | None = None,
        dst_buffer: Image | None = None,
        temp_buffer: Image | None = None
    ) -> Image | None:

        if sigma is None:
            sigma = radius / 3.0
        
        if dst_buffer is None:
            dst_buffer = Image(src.width, src.height)
            return_dst = True
        else:
            check_dimensions_match(dst_buffer, src)
            return_dst = False

        if temp_buffer is None:
            temp_buffer = Image(src.width, src.height)
            owns_temp = True
        else:
            check_dimensions_match(temp_buffer, src)
            owns_temp = False

        gaussian_blur_separable_f32(
            src._buffer._handle,
            temp_buffer._buffer._handle,
            dst_buffer._buffer._handle,
            src.width,
            src.height,
            int(radius),
            float(sigma)
        )

        if owns_temp:
            temp_buffer.free()
        
        return dst_buffer if return_dst else None

    @staticmethod
    def sharpen(
        src: Image,
        strength: float = 1.0,
        dst_buffer: Image | None = None
    ) -> Image | None:

        if dst_buffer is None:
            dst_buffer = Image(src.width, src.height)
            return_buffer = True
        else:
            check_dimensions_match(dst_buffer, src)
            return_buffer = False
        
        sharpen_f32(
            src._buffer._handle,
            dst_buffer._buffer._handle,
            src.width,
            src.height,
            float(strength)
        )
        
        return dst_buffer if return_buffer else None