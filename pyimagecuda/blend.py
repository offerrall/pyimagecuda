from typing import Literal
from .image import Image
from .pyimagecuda_internal import blend_f32, blend_mask_f32 # type: ignore


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
        """
        Standard alpha blending (in-place).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#normal
        """
        if abs(opacity) < 1e-6:
            return
        
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
        """
        Multiplies color values, darkening the image (in-place).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#multiply
        """
        if abs(opacity) < 1e-6:
            return
        
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
        """
        Inverted multiply, lightening the image (in-place).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#screen
        """
        if abs(opacity) < 1e-6:
            return
        
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
        """
        Additive blending, useful for light effects (in-place).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#add
        """
        if abs(opacity) < 1e-6:
            return
        
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

    @staticmethod
    def overlay(
        base: Image,
        overlay: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        """
        Combines Multiply and Screen. Increases contrast (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#overlay
        """
        if abs(opacity) < 1e-6:
            return
        
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
            4,
            opacity
        )

    @staticmethod
    def soft_light(
        base: Image,
        overlay: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        """
        Gentle lighting effect, like a diffuse spotlight (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#soft-light
        """
        if abs(opacity) < 1e-6:
            return
        
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
            5,
            opacity
        )

    @staticmethod
    def hard_light(
        base: Image,
        overlay: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        """
        Strong lighting effect, like a harsh spotlight (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#hard-light
        """
        if abs(opacity) < 1e-6:
            return
        
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
            6,
            opacity
        )

    @staticmethod
    def mask(
        base: Image,
        mask: Image,
        anchor: Literal['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'] = 'top-left',
        offset_x: int = 0,
        offset_y: int = 0,
        mode: Literal['alpha', 'luminance'] = 'luminance'
    ) -> None:
        """
        Applies an image as an alpha mask to the base image (in-place).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/blend/#mask
        """
        pos_x, pos_y = _calculate_position(
            base.width, base.height,
            mask.width, mask.height,
            anchor, offset_x, offset_y
        )
        
        mode_int = 1 if mode == 'luminance' else 0
        
        blend_mask_f32(
            base._buffer._handle,
            mask._buffer._handle,
            base.width, base.height,
            mask.width, mask.height,
            pos_x, pos_y,
            mode_int
        )