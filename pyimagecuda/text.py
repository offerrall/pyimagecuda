from typing import Literal
import pyvips

from .image import Image, ImageU8
from .io import upload, convert_u8_to_float
from .utils import ensure_capacity

class Text:
    @staticmethod
    def create(
        text: str,
        font: str = "Sans",
        size: float = 12.0,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        align: Literal['left', 'centre', 'right'] = 'left',
        justify: bool = False,
        spacing: int = 0,
        letter_spacing: float = 0.0,
        dst_buffer: Image | None = None,
        u8_buffer: ImageU8 | None = None
    ) -> Image | None:
        """
        Renders text into an image with specified font, size, color, alignment, and spacing.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/text/#text-rendering
        """
        full_font_string = f"{font} {size}"
        
        text_opts = {
            'font': full_font_string,
            'dpi': 72,
            'align': 0 if align == 'left' else (1 if align == 'centre' else 2),
            'justify': justify,
            'spacing': spacing
        }
        
        final_text = text
        if letter_spacing != 0:
            pango_spacing = int(letter_spacing * 1024)
            final_text = f'<span letter_spacing="{pango_spacing}">{text}</span>'
            text_opts['rgba'] = True
        
        if '<' in text and '>' in text:
             text_opts['rgba'] = True

        try:
            if text_opts.get('rgba'):
                mask_rgba = pyvips.Image.text(final_text, **text_opts)
                if mask_rgba.bands == 4:
                    mask = mask_rgba[3]
                else:
                    mask = mask_rgba.colourspace('b-w')[0]
            else:
                mask = pyvips.Image.text(final_text, **text_opts)
                
        except pyvips.Error as e:
            raise RuntimeError(f"Failed to render text: {e}")

        fg_r, fg_g, fg_b, fg_a = [int(c * 255) for c in color]
        bg_r, bg_g, bg_b, bg_a = [int(c * 255) for c in bg_color]

        fg_rgb = mask.new_from_image([fg_r, fg_g, fg_b])
        
        if fg_a < 255:
            fg_alpha = (mask * (fg_a / 255.0)).cast('uchar')
        else:
            fg_alpha = mask

        fg_layer = fg_rgb.bandjoin(fg_alpha)
        fg_layer = fg_layer.copy(interpretation='srgb')

        if bg_a > 0:
            bg_layer = mask.new_from_image([bg_r, bg_g, bg_b, bg_a])
            bg_layer = bg_layer.copy(interpretation='srgb')
            final_vips = bg_layer.composite(fg_layer, 'over')
        else:
            final_vips = fg_layer

        final_vips = final_vips.cast('uchar')

        w, h = final_vips.width, final_vips.height
        raw_bytes = final_vips.write_to_memory()

        if dst_buffer is None:
            result = Image(w, h)
            return_result = True
        else:
            ensure_capacity(dst_buffer, w, h)
            result = dst_buffer
            return_result = False

        if u8_buffer is None:
            u8_temp = ImageU8(w, h)
            free_u8 = True
        else:
            ensure_capacity(u8_buffer, w, h)
            u8_temp = u8_buffer
            free_u8 = False
        
        upload(u8_temp, raw_bytes)
        convert_u8_to_float(result, u8_temp)
        
        if free_u8:
            u8_temp.free()

        return result if return_result else None