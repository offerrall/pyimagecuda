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
        expand: bool = True,
        dst_buffer: Image | None = None,
        shadow_buffer: Image | None = None,
        temp_buffer: Image | None = None
    ) -> Image | None:
        
        if expand:
            padding_left = blur + max(0, -offset_x)
            padding_right = blur + max(0, offset_x)
            padding_top = blur + max(0, -offset_y)
            padding_bottom = blur + max(0, offset_y)
            
            result_width = image.width + padding_left + padding_right
            result_height = image.height + padding_top + padding_bottom

            shadow_width = result_width
            shadow_height = result_height
            
            if dst_buffer is None:
                result = Image(result_width, result_height)
                return_result = True
            else:
                max_w, max_h = dst_buffer.get_max_capacity()
                if result_width > max_w or result_height > max_h:
                    raise ValueError(
                        f"dst_buffer capacity too small: need {result_width}×{result_height}, "
                        f"got {max_w}×{max_h}"
                    )
                dst_buffer.width = result_width
                dst_buffer.height = result_height
                result = dst_buffer
                return_result = False

            image_offset_x = padding_left
            image_offset_y = padding_top
            shadow_offset_x = 0
            shadow_offset_y = 0
        else:
            shadow_width = image.width
            shadow_height = image.height
            
            if dst_buffer is None:
                result = Image(image.width, image.height)
                return_result = True
            else:
                check_dimensions_match(dst_buffer, image)
                result = dst_buffer
                return_result = False
            
            image_offset_x = 0
            image_offset_y = 0
            shadow_offset_x = offset_x
            shadow_offset_y = offset_y
        
        if shadow_buffer is None:
            shadow = Image(shadow_width, shadow_height)
            owns_shadow = True
        else:
            if expand:
                max_w, max_h = shadow_buffer.get_max_capacity()
                if shadow_width > max_w or shadow_height > max_h:
                    raise ValueError(
                        f"shadow_buffer capacity too small: need {shadow_width}×{shadow_height}, "
                        f"got {max_w}×{max_h}"
                    )
                shadow_buffer.width = shadow_width
                shadow_buffer.height = shadow_height
            else:
                check_dimensions_match(shadow_buffer, image)
            shadow = shadow_buffer
            owns_shadow = False

        Fill.color(shadow, (0.0, 0.0, 0.0, 0.0))

        if expand:
            temp_alpha = Image(image.width, image.height)
            extract_alpha_f32(
                image._buffer._handle,
                temp_alpha._buffer._handle,
                image.width,
                image.height
            )
            Blend.normal(shadow, temp_alpha, anchor='top-left', offset_x=padding_left, offset_y=padding_top)
            temp_alpha.free()
        else:
            extract_alpha_f32(
                image._buffer._handle,
                shadow._buffer._handle,
                image.width,
                image.height
            )
        
        if blur > 0:
            if temp_buffer is None:
                temp_buffer = Image(shadow_width, shadow_height)
                owns_temp = True
            else:
                if expand:
                    max_w, max_h = temp_buffer.get_max_capacity()
                    if shadow_width > max_w or shadow_height > max_h:
                        raise ValueError(
                            f"temp_buffer capacity too small: need {shadow_width}×{shadow_height}, "
                            f"got {max_w}×{max_h}"
                        )
                    temp_buffer.width = shadow_width
                    temp_buffer.height = shadow_height
                else:
                    check_dimensions_match(temp_buffer, image)
                owns_temp = False
            
            Filter.gaussian_blur(
                shadow,
                radius=blur,
                sigma=blur / 3.0,
                dst_buffer=shadow,
                temp_buffer=temp_buffer
            )
            
            if owns_temp:
                temp_buffer.free()
        
        colorize_shadow_f32(
            shadow._buffer._handle,
            shadow.width,
            shadow.height,
            color
        )
        
        Fill.color(result, (0.0, 0.0, 0.0, 0.0))
        Blend.normal(result, shadow, anchor='top-left', offset_x=shadow_offset_x + offset_x, offset_y=shadow_offset_y + offset_y)
        Blend.normal(result, image, anchor='top-left', offset_x=image_offset_x, offset_y=image_offset_y)
        
        if owns_shadow:
            shadow.free()
        
        return result if return_result else None