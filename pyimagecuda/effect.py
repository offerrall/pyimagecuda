from .image import Image
from .filter import Filter
from .blend import Blend
from .fill import Fill
from .utils import ensure_capacity
from .pyimagecuda_internal import ( #type: ignore
    rounded_corners_f32,
    extract_alpha_f32,
    colorize_shadow_f32
)


class Effect:

    @staticmethod
    def rounded_corners(image: Image, radius: float) -> None:
        """
        Applies rounded corners to the image (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/effect/#rounded-corners
        """
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
        """
        Adds a drop shadow to the image (returns new image or writes to buffer).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/effect/#drop-shadow
        """
        
        if expand:
            padding_left = blur + max(0, -offset_x)
            padding_right = blur + max(0, offset_x)
            padding_top = blur + max(0, -offset_y)
            padding_bottom = blur + max(0, offset_y)
            
            result_width = image.width + padding_left + padding_right
            result_height = image.height + padding_top + padding_bottom

            shadow_width = result_width
            shadow_height = result_height
            
            image_draw_x = padding_left
            image_draw_y = padding_top
        else:
            result_width = image.width
            result_height = image.height
            shadow_width = image.width
            shadow_height = image.height
            
            image_draw_x = 0
            image_draw_y = 0
            
        if dst_buffer is None:
            result = Image(result_width, result_height)
            return_result = True
        else:
            ensure_capacity(dst_buffer, result_width, result_height)
            result = dst_buffer
            return_result = False

        if shadow_buffer is None:
            shadow = Image(shadow_width, shadow_height)
            owns_shadow = True
        else:
            ensure_capacity(shadow_buffer, shadow_width, shadow_height)
            shadow = shadow_buffer
            owns_shadow = False

        req_temp_w = max(image.width, shadow_width)
        req_temp_h = max(image.height, shadow_height)
        need_temp = expand or (blur > 0)
        owns_temp = False

        if need_temp:
            if temp_buffer is None:
                temp_buffer = Image(req_temp_w, req_temp_h)
                owns_temp = True
            else:
                ensure_capacity(temp_buffer, req_temp_w, req_temp_h)
                owns_temp = False

        Fill.color(shadow, (0.0, 0.0, 0.0, 0.0))

        if expand:
            if temp_buffer is not None:
                temp_buffer.width = image.width
                temp_buffer.height = image.height
                
                extract_alpha_f32(
                    image._buffer._handle,
                    temp_buffer._buffer._handle,
                    image.width,
                    image.height
                )
                Blend.normal(shadow, temp_buffer, anchor='top-left', offset_x=padding_left, offset_y=padding_top)
        else:
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
        
        Blend.normal(result, shadow, anchor='top-left', offset_x=offset_x, offset_y=offset_y)
        Blend.normal(result, image, anchor='top-left', offset_x=image_draw_x, offset_y=image_draw_y)
        
        if owns_temp and temp_buffer is not None:
            temp_buffer.free()
        
        if owns_shadow:
            shadow.free()
        
        return result if return_result else None