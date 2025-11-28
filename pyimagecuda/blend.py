from typing import Literal

from .image import Image
from .pyimagecuda_internal import blend_f32 #type: ignore


def _calculate_position(
    base_width: int,
    base_height: int,
    overlay_width: int,
    overlay_height: int,
    anchor: str,
    offset_x: int,
    offset_y: int
) -> tuple[int, int]:

    pos_x = offset_x
    pos_y = offset_y
    
    if 'center' in anchor and anchor in ['top-center', 'center', 'bottom-center']:
        pos_x += (base_width - overlay_width) // 2
    elif 'right' in anchor:
        pos_x += base_width - overlay_width
    
    if 'center' in anchor and anchor in ['center-left', 'center', 'center-right']:
        pos_y += (base_height - overlay_height) // 2
    elif 'bottom' in anchor:
        pos_y += base_height - overlay_height
    
    return pos_x, pos_y


class Blend:

    @staticmethod
    def normal(
        base: Image,
        overlay: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        pos_x, pos_y = _calculate_position(
            base.width, base.height,
            overlay.width, overlay.height,
            anchor, offset_x, offset_y
        )
        
        blend_f32(
            base._buffer._handle,
            overlay._buffer._handle,
            base.width, base.height,
            overlay.width, overlay.height,
            pos_x, pos_y,
            0,
            opacity
        )

    @staticmethod
    def multiply(
        base: Image,
        overlay: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        pos_x, pos_y = _calculate_position(
            base.width, base.height,
            overlay.width, overlay.height,
            anchor, offset_x, offset_y
        )
        
        blend_f32(
            base._buffer._handle,
            overlay._buffer._handle,
            base.width, base.height,
            overlay.width, overlay.height,
            pos_x, pos_y,
            1,
            opacity
        )

    @staticmethod
    def screen(
        base: Image,
        overlay: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        pos_x, pos_y = _calculate_position(
            base.width, base.height,
            overlay.width, overlay.height,
            anchor, offset_x, offset_y
        )
        
        blend_f32(
            base._buffer._handle,
            overlay._buffer._handle,
            base.width, base.height,
            overlay.width, overlay.height,
            pos_x, pos_y,
            2,
            opacity
        )

    @staticmethod
    def add(
        base: Image,
        overlay: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        pos_x, pos_y = _calculate_position(
            base.width, base.height,
            overlay.width, overlay.height,
            anchor, offset_x, offset_y
        )
        
        blend_f32(
            base._buffer._handle,
            overlay._buffer._handle,
            base.width, base.height,
            overlay.width, overlay.height,
            pos_x, pos_y,
            3,
            opacity
        )