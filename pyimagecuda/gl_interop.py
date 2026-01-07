from .pyimagecuda_internal import (  # type: ignore
    register_gl_pbo,
    unregister_gl_resource,
    copy_to_gl_pbo
)
from .image import ImageU8


class GLResource:
    """
    CUDA-OpenGL interop resource for direct GPU-to-GPU transfers.
    
    Represents a registered OpenGL PBO that can receive ImageU8 data
    directly without CPU roundtrips.
    
    Docs & Examples: https://offerrall.github.io/pyimagecuda/opengl/
    """
    
    def __init__(self, pbo_id: int):
        """
        Parameters:
            pbo_id: Valid OpenGL PBO ID. The PBO must exist and remain 
                    valid for the lifetime of this GLResource.
        
        Raises:
            ValueError: If pbo_id is invalid
            RuntimeError: If CUDA registration fails
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/opengl/
        """
        if not isinstance(pbo_id, int) or pbo_id <= 0:
            raise ValueError(f"Invalid PBO ID: {pbo_id}")
        
        self._handle = None
        self.pbo_id = pbo_id
        self._handle = register_gl_pbo(pbo_id)
    
    def copy_from(self, image: ImageU8, sync: bool = True) -> None:
        """
        Copies ImageU8 data directly to the registered PBO (GPU→GPU).
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/opengl/
        """
        if not isinstance(image, ImageU8):
            raise TypeError(f"Expected ImageU8, got {type(image).__name__}")
        
        if self._handle is None:
            raise RuntimeError("GLResource has been freed")
        
        copy_to_gl_pbo(
            image._buffer._handle,
            self._handle,
            image.width,
            image.height,
            sync
        )
    
    def free(self) -> None:
        """
        Unregisters the OpenGL resource from CUDA.
        
        Docs & Examples: https://offerrall.github.io/pyimagecuda/opengl/
        """
        if self._handle is not None:
            unregister_gl_resource(self._handle)
            self._handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.free()
        return False
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle is not None:
            try:
                self.free()
            except Exception:
                pass
    
    def __repr__(self) -> str:
        status = "active" if getattr(self, '_handle', None) is not None else "freed"
        return f"GLResource(pbo_id={self.pbo_id}, status={status})"