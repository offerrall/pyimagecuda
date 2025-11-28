from typing import Literal
from .image import Image
from .pyimagecuda_internal import fill_color_f32, fill_gradient_f32  #type: ignore


class Fill:
    
    @staticmethod
    def color(image: Image, rgba: tuple[float, float, float, float]) -> None:
        """
        Fills the image with a solid color (in-place).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#solid-colors
        """
        fill_color_f32(image._buffer._handle, rgba, image.width, image.height)
    
    @staticmethod
    def gradient(image: Image, 
                 rgba1: tuple[float, float, float, float],
                 rgba2: tuple[float, float, float, float],
                 direction: Literal['horizontal', 'vertical', 'diagonal', 'radial'] = 'horizontal',
                 seamless: bool = False) -> None:
        """
        Fills the image with a gradient (in-place).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#gradients
        """
        direction_map = {
            'horizontal': 0,
            'vertical': 1,
            'diagonal': 2,
            'radial': 3
        }
        
        dir_int = direction_map.get(direction)
        if dir_int is None:
            raise ValueError(f"Invalid direction: {direction}. Must be one of {list(direction_map.keys())}")
        
        fill_gradient_f32(
            image._buffer._handle, 
            rgba1, 
            rgba2, 
            image.width, 
            image.height, 
            dir_int,
            seamless
        )