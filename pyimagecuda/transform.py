import math
from typing import Literal
from .image import Image
from .utils import ensure_capacity
from .fill import Fill
from .io import copy
from .pyimagecuda_internal import ( #type: ignore
    flip_f32, crop_f32, rotate_fixed_f32, 
    rotate_arbitrary_f32, copy_buffer 
)


class Transform:
    
    @staticmethod
    def flip(
        image: Image, 
        direction: Literal['horizontal', 'vertical', 'both'] = 'horizontal',
        dst_buffer: Image | None = None
    ) -> Image | None:
        """
        Flips the image across the specified axis (returns new image or writes to buffer).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/transform/#flip
        """
        
        direction_map = {
            'horizontal': 0,
            'vertical': 1,
            'both': 2
        }
        
        mode = direction_map.get(direction)
        if mode is None:
            raise ValueError(f"Invalid direction: {direction}. Must be {list(direction_map.keys())}")
            
        if dst_buffer is None:
            dst_buffer = Image(image.width, image.height)
            return_buffer = True
        else:
            ensure_capacity(dst_buffer, image.width, image.height)
            return_buffer = False
            
        flip_f32(
            image._buffer._handle,
            dst_buffer._buffer._handle,
            image.width,
            image.height,
            mode
        )
        
        return dst_buffer if return_buffer else None

    @staticmethod
    def rotate(
        image: Image,
        angle: float,
        expand: bool = True,
        dst_buffer: Image | None = None
    ) -> Image | None:
        """
        Rotates the image by any angle in degrees (Clockwise).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/transform/#rotate
        """
        
        norm_angle = angle % 360
        if norm_angle < 0: norm_angle += 360
        
        is_fixed = False
        fixed_mode = 0
        
        if abs(norm_angle - 0) < 0.01: 
            if dst_buffer is None:
                dst_buffer = Image(image.width, image.height)
                return_buffer = True
            else:
                ensure_capacity(dst_buffer, image.width, image.height)
                return_buffer = False
            
            copy_buffer(dst_buffer._buffer._handle, image._buffer._handle, image.width, image.height)
            return dst_buffer if return_buffer else None

        elif abs(norm_angle - 90) < 0.01: is_fixed = True; fixed_mode = 0
        elif abs(norm_angle - 180) < 0.01: is_fixed = True; fixed_mode = 1
        elif abs(norm_angle - 270) < 0.01: is_fixed = True; fixed_mode = 2
        
        if is_fixed:
            if fixed_mode == 1:
                rot_w, rot_h = image.width, image.height
            else: 
                rot_w, rot_h = image.height, image.width
        else:
            rads = math.radians(angle)
            sin_a = abs(math.sin(rads))
            cos_a = abs(math.cos(rads))
            rot_w = int(image.width * cos_a + image.height * sin_a)
            rot_h = int(image.width * sin_a + image.height * cos_a)

        if expand:
            final_w = rot_w
            final_h = rot_h
            offset_x = 0
            offset_y = 0
        else:
            final_w = image.width
            final_h = image.height
            offset_x = (final_w - rot_w) // 2
            offset_y = (final_h - rot_h) // 2

        if dst_buffer is None:
            dst_buffer = Image(final_w, final_h)
            return_buffer = True
        else:
            ensure_capacity(dst_buffer, final_w, final_h)
            return_buffer = False

        if is_fixed:
            rotate_fixed_f32(
                image._buffer._handle,
                dst_buffer._buffer._handle,
                image.width, image.height,
                final_w, final_h,
                fixed_mode, offset_x, offset_y
            )
        else:
            rotate_arbitrary_f32(
                image._buffer._handle,
                dst_buffer._buffer._handle,
                image.width, image.height,
                final_w, final_h,
                float(angle)
            )
        
        return dst_buffer if return_buffer else None

    @staticmethod
    def crop(
        image: Image,
        x: int,
        y: int,
        width: int,
        height: int,
        dst_buffer: Image | None = None
    ) -> Image | None:
        """
        Crops a rectangular region (returns new image or writes to buffer).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/transform/#crop
        """
        if width <= 0 or height <= 0:
            raise ValueError("Crop dimensions must be positive")

        if x == 0 and y == 0 and width == image.width and height == image.height:
            if dst_buffer is None:
                dst_buffer = Image(width, height)
                copy(dst_buffer, image)
                return dst_buffer
            else:
                copy(dst_buffer, image)
                return None

        if dst_buffer is None:
            dst_buffer = Image(width, height)
            return_buffer = True
        else:
            ensure_capacity(dst_buffer, width, height)
            return_buffer = False

        Fill.color(dst_buffer, (0.0, 0.0, 0.0, 0.0))

        crop_left = x
        crop_top = y
        crop_right = x + width
        crop_bottom = y + height
        img_right, img_bottom = image.width, image.height
        
        intersect_left = max(crop_left, 0)
        intersect_top = max(crop_top, 0)
        intersect_right = min(crop_right, img_right)
        intersect_bottom = min(crop_bottom, img_bottom)
        
        copy_w = intersect_right - intersect_left
        copy_h = intersect_bottom - intersect_top
        
        if copy_w > 0 and copy_h > 0:
            crop_f32(
                image._buffer._handle,
                dst_buffer._buffer._handle,
                image.width, dst_buffer.width,
                intersect_left, intersect_top,
                intersect_left - crop_left, intersect_top - crop_top,
                copy_w, copy_h
            )
            
        return dst_buffer if return_buffer else None