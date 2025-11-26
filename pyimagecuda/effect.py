from .image import Image
from .filter import Filter
from .blend import Blend
from .fill import Fill
from .utils import check_dimensions_match
from .pyimagecuda_internal import ( #type: ignore
    rounded_corners_f32,
    extract_alpha_f32,
    colorize_shadow_f32
)


class Effect:

    @staticmethod
    def rounded_corners(image: Image, radius: float) -> None:
        max_radius = min(image.width, image.height) / 2.0
        
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        
        if radius > max_radius:
            radius = max_radius
        
        rounded_corners_f32(
            image._buffer._handle,
            image.width,
            image.height,
            float(radius)
        )

    @staticmethod
    def drop_shadow(
        image: Image,
        offset_x: int = 10,
        offset_y: int = 10,
        blur: int = 20,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5),
        dst_buffer: Image | None = None,
        shadow_buffer: Image | None = None,
        temp_buffer: Image | None = None
    ) -> Image | None:
        
        if dst_buffer is None:
            result = Image(image.width, image.height)
            return_result = True
        else:
            check_dimensions_match(dst_buffer, image)
            result = dst_buffer
            return_result = False
        
        if shadow_buffer is None:
            shadow = Image(image.width, image.height)
            owns_shadow = True
        else:
            check_dimensions_match(shadow_buffer, image)
            shadow = shadow_buffer
            owns_shadow = False
        
        extract_alpha_f32(
            image._buffer._handle,
            shadow._buffer._handle,
            image.width,
            image.height
        )
        
        if blur > 0:
            Filter.gaussian_blur(
                shadow,
                radius=blur,
                sigma=blur / 3.0,
                dst_buffer=shadow,
                temp_buffer=temp_buffer
            )
        
        colorize_shadow_f32(
            shadow._buffer._handle,
            shadow.width,
            shadow.height,
            color
        )
        
        Fill.color(result, (0.0, 0.0, 0.0, 0.0))
        Blend.normal(result, shadow, pos_x=offset_x, pos_y=offset_y)
        Blend.normal(result, image, pos_x=0, pos_y=0)
        
        if owns_shadow:
            shadow.free()
        
        return result if return_result else None