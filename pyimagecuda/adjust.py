from .image import Image
from .pyimagecuda_internal import ( #type: ignore
    adjust_brightness_f32, 
    adjust_contrast_f32,
    adjust_saturation_f32,
    adjust_gamma_f32,
    adjust_opacity_f32
)


class Adjust:
    
    @staticmethod
    def brightness(image: Image, factor: float) -> None:
        """
        Adjusts image brightness by adding a factor (in-place).
        
        Positive factor brightens, negative darkens.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/adjust/#brightness
        """
        if abs(factor) < 1e-6:
            return
        
        adjust_brightness_f32(
            image._buffer._handle,
            image.width,
            image.height,
            float(factor)
        )

    @staticmethod
    def contrast(image: Image, factor: float) -> None:
        """
        Adjusts image contrast relative to middle gray (in-place).
        
        factor > 1.0 increases contrast.
        factor < 1.0 decreases contrast.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/adjust/#contrast
        """
        if abs(factor - 1.0) < 1e-6:
            return
        
        adjust_contrast_f32(
            image._buffer._handle,
            image.width,
            image.height,
            float(factor)
        )

    @staticmethod
    def saturation(image: Image, factor: float) -> None:
        """
        Adjusts color intensity (in-place).
        
        factor 0.0 = Grayscale
        factor 1.0 = Original
        factor > 1.0 = More vibrant
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/adjust/#saturation
        """
        if abs(factor - 1.0) < 1e-6:
            return
        
        adjust_saturation_f32(
            image._buffer._handle,
            image.width,
            image.height,
            float(factor)
        )

    @staticmethod
    def gamma(image: Image, gamma: float) -> None:
        """
        Adjusts gamma correction (non-linear brightness) (in-place).
        
        Useful for brightening shadows without washing out highlights.
        
        gamma > 1.0: Brightens midtones.
        gamma < 1.0: Darkens midtones.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/adjust/#gamma
        """
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        if abs(gamma - 1.0) < 1e-6:
            return
            
        adjust_gamma_f32(
            image._buffer._handle,
            image.width,
            image.height,
            float(gamma)
        )

    @staticmethod
    def opacity(image: Image, factor: float) -> None:
        """
        Multiplies the alpha channel by a factor (In-Place).
        
        - factor: 0.0 (Fully Transparent) to 1.0 (No change). 

        Docs & Examples: https://offerrall.github.io/pyimagecuda/adjust/#opacity
        """
        if abs(factor - 1.0) < 1e-6:
            return

        adjust_opacity_f32(
            image._buffer._handle,
            image.width,
            image.height,
            float(factor)
        )