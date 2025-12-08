from .image import Image
from .utils import ensure_capacity
from .pyimagecuda_internal import resize_f32  #type: ignore
from .io import copy


def _resize_internal(
    src: Image,
    width: int | None,
    height: int | None,
    method: int,
    dst_buffer: Image | None = None
) -> Image | None:
    
    if width is None and height is None:
        raise ValueError("At least one of width or height must be specified")
    elif width is None:
        width = int(src.width * (height / src.height))
    elif height is None:
        height = int(src.height * (width / src.width))
    
    # No resize needed, just copy
    if width == src.width and height == src.height:
        if dst_buffer is None:
            dst_buffer = Image(width, height)
            copy(dst_buffer, src)
            return dst_buffer
        else:
            copy(dst_buffer, src)
            return None
    
    # Resize normal
    if dst_buffer is None:
        dst_buffer = Image(width, height)
        return_buffer = True
    else:
        ensure_capacity(dst_buffer, width, height)
        return_buffer = False
    
    resize_f32(
        src._buffer._handle,
        dst_buffer._buffer._handle,
        src.width,
        src.height,
        width,
        height,
        method
    )
    
    return dst_buffer if return_buffer else None


class Resize:
    @staticmethod
    def nearest(
        src: Image,
        width: int | None = None,
        height: int | None = None,
        dst_buffer: Image | None = None
    ) -> Image | None:
        """
        Resizes the image using nearest neighbor interpolation (returns new image or writes to buffer).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/resize/#nearest
        """
        return _resize_internal(src, width, height, 0, dst_buffer)

    @staticmethod
    def bilinear(
        src: Image,
        width: int | None = None,
        height: int | None = None,
        dst_buffer: Image | None = None
    ) -> Image | None:
        """
        Resizes the image using bilinear interpolation (returns new image or writes to buffer).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/resize/#bilinear
        """

        return _resize_internal(src, width, height, 1, dst_buffer)

    @staticmethod
    def bicubic(
        src: Image,
        width: int | None = None,
        height: int | None = None,
        dst_buffer: Image | None = None
    ) -> Image | None:
        """
        Resizes the image using bicubic interpolation (returns new image or writes to buffer).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/resize/#bicubic
        """
        return _resize_internal(src, width, height, 2, dst_buffer)

    @staticmethod
    def lanczos(
        src: Image,
        width: int | None = None,
        height: int | None = None,
        dst_buffer: Image | None = None
    ) -> Image | None:
        """
        Resizes the image using Lanczos interpolation (returns new image or writes to buffer).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/resize/#lanczos
        """
        return _resize_internal(src, width, height, 3, dst_buffer)