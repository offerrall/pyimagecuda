from .image import Image
from .pyimagecuda_internal import resize_f32  #type: ignore


def _resize_internal(
    src: Image,
    width: int,
    height: int,
    method: int,
    dst_buffer: Image | None = None
) -> Image | None:
    
    if dst_buffer is None:
        dst_buffer = Image(width, height)
        return_buffer = True
    else:
        if dst_buffer.width != width or dst_buffer.height != height:
            raise ValueError(
                f"dst_buffer size mismatch: expected {width}×{height}, "
                f"got {dst_buffer.width}×{dst_buffer.height}"
            )
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
        width: int,
        height: int,
        dst_buffer: Image | None = None
    ) -> Image | None:
        return _resize_internal(src, width, height, 0, dst_buffer)

    @staticmethod
    def bilinear(
        src: Image,
        width: int,
        height: int,
        dst_buffer: Image | None = None
    ) -> Image | None:
        return _resize_internal(src, width, height, 1, dst_buffer)

    @staticmethod
    def bicubic(
        src: Image,
        width: int,
        height: int,
        dst_buffer: Image | None = None
    ) -> Image | None:
        return _resize_internal(src, width, height, 2, dst_buffer)

    @staticmethod
    def lanczos(
        src: Image,
        width: int,
        height: int,
        dst_buffer: Image | None = None
    ) -> Image | None:
        return _resize_internal(src, width, height, 3, dst_buffer)