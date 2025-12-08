from .image import Image
from .utils import ensure_capacity
from .io import copy
from .pyimagecuda_internal import (gaussian_blur_separable_f32, #type: ignore
                                   sharpen_f32,
                                   sepia_f32,
                                   invert_f32,
                                   threshold_f32,
                                   solarize_f32,
                                   filter_sobel_f32,
                                   filter_emboss_f32)


class Filter:

    @staticmethod
    def gaussian_blur(
        src: Image,
        radius: int = 3,
        sigma: float | None = None,
        dst_buffer: Image | None = None,
        temp_buffer: Image | None = None
    ) -> Image | None:
        """
        Applies a Gaussian blur to the image (returns new image or writes to buffer).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#gaussian-blur
        """

        if radius == 0 or (sigma is not None and sigma <= 0.001):
            if dst_buffer is None:
                dst_buffer = Image(src.width, src.height)
                copy(dst_buffer, src)
                return dst_buffer
            else:
                copy(dst_buffer, src)
                return None

        if sigma is None:
            sigma = radius / 3.0
        
        if dst_buffer is None:
            dst_buffer = Image(src.width, src.height)
            return_dst = True
        else:
            ensure_capacity(dst_buffer, src.width, src.height)
            return_dst = False

        if temp_buffer is None:
            temp_buffer = Image(src.width, src.height)
            owns_temp = True
        else:
            ensure_capacity(temp_buffer, src.width, src.height)
            owns_temp = False

        gaussian_blur_separable_f32(
            src._buffer._handle,
            temp_buffer._buffer._handle,
            dst_buffer._buffer._handle,
            src.width,
            src.height,
            radius,
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
        """
        Sharpens the image (returns new image or writes to buffer).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#sharpen
        """

        if abs(strength) < 1e-6:
            if dst_buffer is None:
                dst_buffer = Image(src.width, src.height)
                copy(dst_buffer, src)
                return dst_buffer
            else:
                copy(dst_buffer, src)
                return None

        if dst_buffer is None:
            dst_buffer = Image(src.width, src.height)
            return_buffer = True
        else:
            ensure_capacity(dst_buffer, src.width, src.height)
            return_buffer = False
        
        sharpen_f32(
            src._buffer._handle,
            dst_buffer._buffer._handle,
            src.width,
            src.height,
            float(strength)
        )
        
        return dst_buffer if return_buffer else None

    @staticmethod
    def sepia(image: Image, intensity: float = 1.0) -> None:
        """
        Applies Sepia tone (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#sepia
        """
        if abs(intensity) < 1e-6:
            return
        
        sepia_f32(image._buffer._handle, image.width, image.height, float(intensity))

    @staticmethod
    def invert(image: Image) -> None:
        """
        Inverts colors (Negative effect) in-place.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#invert
        """
        invert_f32(image._buffer._handle, image.width, image.height)

    @staticmethod
    def threshold(image: Image, value: float = 0.5) -> None:
        """
        Converts to pure Black & White based on luminance threshold.
        value: 0.0 to 1.0. Pixels brighter than value become white, others black.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#threshold
        """
        threshold_f32(image._buffer._handle, image.width, image.height, float(value))

    @staticmethod
    def solarize(image: Image, threshold: float = 0.5) -> None:
        """
        Inverts only pixels brighter than threshold. Creates a psychedelic/retro look.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#solarize
        """
        solarize_f32(image._buffer._handle, image.width, image.height, float(threshold))

    @staticmethod
    def sobel(src: Image, dst_buffer: Image | None = None) -> Image | None:
        """
        Detects edges using Sobel operator. Returns a black & white image with edges.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#sobel
        """
        if dst_buffer is None:
            dst_buffer = Image(src.width, src.height)
            return_buffer = True
        else:
            ensure_capacity(dst_buffer, src.width, src.height)
            return_buffer = False

        filter_sobel_f32(
            src._buffer._handle,
            dst_buffer._buffer._handle,
            src.width, src.height
        )
        return dst_buffer if return_buffer else None

    @staticmethod
    def emboss(src: Image, strength: float = 1.0, dst_buffer: Image | None = None) -> Image | None:
        """
        Applies Emboss (Relief) effect.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/filter/#emboss
        """
        if abs(strength) < 1e-6:
            if dst_buffer is None:
                dst_buffer = Image(src.width, src.height)
                copy(dst_buffer, src)
                return dst_buffer
            else:
                copy(dst_buffer, src)
                return None

        if dst_buffer is None:
            dst_buffer = Image(src.width, src.height)
            return_buffer = True
        else:
            ensure_capacity(dst_buffer, src.width, src.height)
            return_buffer = False
        
        filter_emboss_f32(
            src._buffer._handle,
            dst_buffer._buffer._handle,
            src.width, src.height, float(strength)
        )
        return dst_buffer if return_buffer else None