from .pyimagecuda_internal import create_buffer_f32, free_buffer, create_buffer_u8  #type: ignore


class Buffer:
    
    def __init__(self, width: int, height: int, is_u8: bool = False):
        create_func = create_buffer_u8 if is_u8 else create_buffer_f32
        self._handle = create_func(width, height)
        self.capacity_width = width
        self.capacity_height = height
    
    def free(self) -> None:
        free_buffer(self._handle)


class ImageBase:

    def __init__(self, width: int, height: int, is_u8: bool = False):
        self._buffer = Buffer(width, height, is_u8)
        self._width = width
        self._height = height
    
    @property
    def width(self) -> int:
        return self._width
    
    @width.setter
    def width(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise ValueError(f"Width must be positive, got {value}")
        
        if value > self._buffer.capacity_width:
            raise ValueError(
                f"Width {value} exceeds buffer capacity "
                f"{self._buffer.capacity_width}"
            )
        
        self._width = value
    
    @property
    def height(self) -> int:
        return self._height
    
    @height.setter
    def height(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise ValueError(f"Height must be positive, got {value}")
        
        if value > self._buffer.capacity_height:
            raise ValueError(
                f"Height {value} exceeds buffer capacity "
                f"{self._buffer.capacity_height}"
            )
        
        self._height = value

    def free(self) -> None:
        self._buffer.free()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.free()
        return False

    def get_max_capacity(self) -> tuple[int, int]:
        return (self._buffer.capacity_width, self._buffer.capacity_height)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.width}×{self.height})"


class Image(ImageBase):
    
    def __init__(self, width: int, height: int):
        """
        Creates a floating-point image with the given width and height.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/image/#image-float32-precision
        """
        super().__init__(width, height, is_u8=False)


class ImageU8(ImageBase):
    
    def __init__(self, width: int, height: int):
        """
        Creates an 8-bit unsigned integer image with the given width and height.

        Docs & Examples: https://offerrall.github.io/pyimagecuda/image/#imageu8-8-bit-precision
        """
        super().__init__(width, height, is_u8=True)