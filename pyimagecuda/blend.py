from .image import Image
from .pyimagecuda_internal import blend_f32 #type: ignore


class Blend:

    @staticmethod
    def normal(
        base: Image,
        overlay: Image,
        pos_x: int = 0,
        pos_y: int = 0,
        opacity: float = 1.0
    ) -> None:
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
        pos_x: int = 0,
        pos_y: int = 0,
        opacity: float = 1.0
    ) -> None:
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
        pos_x: int = 0,
        pos_y: int = 0,
        opacity: float = 1.0
    ) -> None:
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
        pos_x: int = 0,
        pos_y: int = 0,
        opacity: float = 1.0
    ) -> None:
        blend_f32(
            base._buffer._handle,
            overlay._buffer._handle,
            base.width, base.height,
            overlay.width, overlay.height,
            pos_x, pos_y,
            3,
            opacity
        )