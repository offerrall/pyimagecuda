from typing import Literal

from .image import Image
from .filter import Filter
from .blend import Blend
from .fill import Fill
from .pyimagecuda_internal import ( #type: ignore
    rounded_corners_f32,
    extract_alpha_f32,
    colorize_alpha_mask_f32,
    compute_distance_field_f32,
    generate_stroke_composite_f32,
    vignette_f32,
    chroma_key_f32
)


class Effect:

    @staticmethod
    def rounded_corners(image: Image, radius: float) -> None:
        """
        Applies rounded corners to the image (in-place).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/effect/#rounded_corners
        """
        max_radius = min(image.width, image.height) / 2.0
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        if radius == 0:
            return
        if radius > max_radius:
            radius = max_radius
        
        rounded_corners_f32(image._buffer._handle, image.width, image.height, float(radius))

    @staticmethod
    def drop_shadow(
        image: Image,
        offset_x: int = 10,
        offset_y: int = 10,
        blur: int = 20,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5),
        expand: bool = True,
        dst_buffer: Image | None = None,
        shadow_buffer: Image | None = None,
        temp_buffer: Image | None = None
    ) -> Image | None:
        """
        Adds a drop shadow to the image.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/effect/#drop_shadow
        """
        if expand:
            pad_l = blur + max(0, -offset_x)
            pad_r = blur + max(0, offset_x)
            pad_t = blur + max(0, -offset_y)
            pad_b = blur + max(0, offset_y)
            res_w = image.width + pad_l + pad_r
            res_h = image.height + pad_t + pad_b
            img_x = pad_l
            img_y = pad_t
        else:
            res_w = image.width
            res_h = image.height
            img_x = 0
            img_y = 0
            
        if dst_buffer is None:
            result = Image(res_w, res_h)
            ret = True
        else:
            dst_buffer.resize(res_w, res_h)
            result = dst_buffer
            ret = False

        if shadow_buffer is None:
            shadow = Image(res_w, res_h)
            own_shadow = True
        else:
            shadow_buffer.resize(res_w, res_h)
            shadow = shadow_buffer
            own_shadow = False

        if expand:
            if temp_buffer is None:
                temp = Image(res_w, res_h)
            else:
                temp_buffer.resize(res_w, res_h)
                temp = temp_buffer
            
            Fill.color(shadow, (0, 0, 0, 0))
            temp.resize(image.width, image.height)
            extract_alpha_f32(image._buffer._handle, temp._buffer._handle, image.width, image.height)
            Blend.normal(shadow, temp, anchor='top-left', offset_x=pad_l, offset_y=pad_t)
            
            if temp_buffer is None:
                temp.free()
        else:
            extract_alpha_f32(image._buffer._handle, shadow._buffer._handle, image.width, image.height)
        
        if blur > 0:
            Filter.gaussian_blur(shadow, radius=blur, sigma=blur/3.0, dst_buffer=shadow)
        
        colorize_alpha_mask_f32(shadow._buffer._handle, shadow.width, shadow.height, color)
        Fill.color(result, (0, 0, 0, 0))
        Blend.normal(result, shadow, anchor='top-left', offset_x=offset_x, offset_y=offset_y)
        Blend.normal(result, image, anchor='top-left', offset_x=img_x, offset_y=img_y)
        
        if own_shadow:
            shadow.free()
        
        return result if ret else None

    @staticmethod
    def stroke(
        image: Image,
        width: int = 2,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        position: Literal['outside', 'inside'] = 'outside',
        expand: bool = True,
        dst_buffer: Image | None = None,
        distance_buffer: Image | None = None,
    ) -> Image | None:
        """
        Adds a stroke (outline) to the image using distance field method.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/effect/#stroke
        """
        if width < 1 or width > 1000:
            raise ValueError("Invalid stroke width")
        if position not in ('outside', 'inside'):
            raise ValueError("Invalid position")

        if position == 'outside' and expand:
            res_w = image.width + width * 2
            res_h = image.height + width * 2
            off_x = width
            off_y = width
        else:
            res_w = image.width
            res_h = image.height
            off_x = 0
            off_y = 0
        
        if dst_buffer is None:
            result = Image(res_w, res_h)
            ret = True
        else:
            dst_buffer.resize(res_w, res_h)
            result = dst_buffer
            ret = False
        
        if distance_buffer is None:
            distance = Image(res_w, res_h)
            own_dist = True
        else:
            distance_buffer.resize(res_w, res_h)
            distance = distance_buffer
            own_dist = False

        compute_distance_field_f32(
            image._buffer._handle,
            distance._buffer._handle,
            image.width,
            image.height,
            res_w,
            res_h,
            off_x,
            off_y
        )

        generate_stroke_composite_f32(
            image._buffer._handle,
            distance._buffer._handle,
            result._buffer._handle,
            image.width,
            image.height,
            res_w,
            res_h,
            off_x,
            off_y,
            float(width),
            color,
            0 if position == 'outside' else 1
        )
        
        if own_dist:
            distance.free()
        
        return result if ret else None

    @staticmethod
    def vignette(
        image: Image,
        radius: float = 0.9,
        softness: float = 1.0,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    ) -> None:
        """
        Applies a vignette effect (darkening edges) (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/effect/#vignette
        """
        if radius < 0:
            radius = 0.0
        if softness < 0:
            softness = 0.0
        
        vignette_f32(
            image._buffer._handle,
            image.width,
            image.height,
            float(radius),
            float(softness),
            color
        )

    @staticmethod
    def chroma_key(
        image: Image,
        key_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        threshold: float = 0.4,
        smoothness: float = 0.1,
        spill_suppression: float = 0.5
    ) -> None:
        """
        Remove a color from the image (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/effect/#chroma-key
        """
        if abs(threshold) < 1e-6:
            return
        
        chroma_key_f32(
            image._buffer._handle,
            image.width,
            image.height,
            key_color,
            float(threshold),
            float(smoothness),
            float(spill_suppression)
        )