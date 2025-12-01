from typing import Literal
from .image import Image
from .pyimagecuda_internal import (fill_color_f32, #type: ignore
                                   fill_gradient_f32,
                                   fill_circle_f32,
                                   fill_checkerboard_f32,
                                   fill_grid_f32,
                                   fill_stripes_f32,
                                   fill_dots_f32,
                                   fill_noise_f32,
                                   fill_perlin_f32,
                                   fill_ngon_f32
                                   )


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

    @staticmethod
    def checkerboard(
        image: Image,
        size: int = 20,
        color1: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
        color2: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        offset_x: int = 0,
        offset_y: int = 0
    ) -> None:
        """
        Fills buffer with a checkerboard pattern.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#checkerboard
        """
        if size <= 0:
            raise ValueError("Checkerboard size must be positive")
        
        fill_checkerboard_f32(
            image._buffer._handle,
            image.width, image.height,
            int(size),
            int(offset_x), int(offset_y),
            color1, color2
        )

    @staticmethod
    def grid(
        image: Image,
        spacing: int = 50,
        line_width: int = 1,
        color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        offset_x: int = 0,
        offset_y: int = 0
    ) -> None:
        """
        Fills buffer with a grid pattern.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#grid
        """
        if spacing <= 0:
            raise ValueError("Grid spacing must be positive")
        if line_width <= 0:
            raise ValueError("Line width must be positive")
        
        fill_grid_f32(
            image._buffer._handle,
            image.width, image.height,
            int(spacing), int(line_width),
            int(offset_x), int(offset_y),
            color, bg_color
        )

    @staticmethod
    def stripes(
        image: Image,
        angle: float = 45.0,
        spacing: int = 40,
        width: int = 20,
        color1: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        color2: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        offset: int = 0
    ) -> None:
        """
        Fills buffer with alternating stripes with Anti-Aliasing.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#stripes
        """
        if spacing <= 0:
            raise ValueError("Stripes spacing must be positive")
        
        fill_stripes_f32(
            image._buffer._handle,
            image.width, image.height,
            float(angle), int(spacing), int(width), int(offset),
            color1, color2
        )

    @staticmethod
    def dots(
        image: Image,
        spacing: int = 40,
        radius: float = 10.0,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        offset_x: int = 0,
        offset_y: int = 0,
        softness: float = 0.0
    ) -> None:
        """
        Fills buffer with a Polka Dot pattern.
        - softness: 0.0 = Hard edge, 1.0 = Soft glow.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#dots
        """
        if spacing <= 0:
            raise ValueError("Spacing must be positive")

        fill_dots_f32(
            image._buffer._handle,
            image.width, image.height,
            int(spacing), float(radius), 
            int(offset_x), int(offset_y), float(softness),
            color, bg_color
        )

    @staticmethod
    def circle(
        image: Image,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        softness: float = 0.0
    ) -> None:
        """
        Fills the buffer with a centered circle fitted to the image size.
        - softness: Edge softness. 0.0 = Hard edge (with AA), >0.0 = Soft gradient.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#circle
        """

        fill_circle_f32(
            image._buffer._handle,
            image.width, image.height,
            float(softness),
            color, bg_color
        )

    @staticmethod
    def noise(
        image: Image,
        seed: float = 0.0,
        monochrome: bool = True
    ) -> None:
        """
        Fills the buffer with random White Noise.
        - seed: Random seed. Change this to animate the noise.
        - monochrome: True for grayscale noise, False for RGB noise.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#noise
        """
        
        fill_noise_f32(
            image._buffer._handle,
            image.width, image.height,
            float(seed),
            int(monochrome)
        )

    @staticmethod
    def perlin(
        image: Image,
        scale: float = 50.0,
        seed: float = 0.0,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        color1: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        color2: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    ) -> None:
        """
        Fills buffer with Perlin Noise (Gradient Noise).
        - scale: "Zoom" level. Higher values = bigger features (zoomed in).
        - octaves: Detail layers. 1 = smooth, 6 = rocky/detailed.
        - persistence: How much each octave contributes (0.0 to 1.0).
        - lacunarity: Detail frequency multiplier (usually 2.0).

        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#perlin-noise
        """
        if scale <= 0: scale = 0.001
        
        fill_perlin_f32(
            image._buffer._handle,
            image.width, image.height,
            float(scale), float(seed),
            int(octaves), float(persistence), float(lacunarity),
            float(offset_x), float(offset_y),
            color1, color2
        )

    @staticmethod
    def ngon(
        image: Image,
        sides: int = 3,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        rotation: float = 0.0,
        softness: float = 0.0
    ) -> None:
        """
        Fills buffer with a Regular Polygon (Triangle, Pentagon, Hexagon...).
        - softness: Edge softness (0.0 = Hard AA, >0.0 = Glow).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/fill/#ngon
        """
        if sides < 3:
            raise ValueError("Polygon must have at least 3 sides")

        fill_ngon_f32(
            image._buffer._handle,
            image.width, image.height,
            int(sides), float(rotation), float(softness),
            color, bg_color
        )