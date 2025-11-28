from .image import ImageBase

def ensure_capacity(buffer: ImageBase, required_width: int, required_height: int) -> None:
    """
    Checks if the buffer has enough capacity and updates its logical dimensions.
    Raises ValueError if capacity is insufficient.
    """
    max_w, max_h = buffer.get_max_capacity()
    
    if required_width > max_w or required_height > max_h:
        raise ValueError(
            f"Buffer capacity too small: need {required_width}×{required_height}, "
            f"got capacity {max_w}×{max_h}"
        )
    
    buffer.width = required_width
    buffer.height = required_height