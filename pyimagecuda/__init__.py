import ctypes
import sys

__version__ = "0.0.7"

def _check_nvidia_driver():
    try:
        ctypes.windll.LoadLibrary("nvcuda.dll")
        return True
    except OSError:
        return False

_INTERNAL_LOADED = False

try:
    from . import pyimagecuda_internal
    _INTERNAL_LOADED = True
except ImportError as e:
    _INTERNAL_LOADED = False
    error_msg = str(e).lower()

    if "dll load failed" in error_msg:
        print("\n" + "!" * 75)
        print(" [CRITICAL ERROR] Failed to load pyimagecuda backend.")
        
        if not _check_nvidia_driver():
            print(" CAUSE: NVIDIA Drivers are missing or not detected.")
            print(" SOLUTION: Install latest drivers from: https://www.nvidia.com/Download/index.aspx")
        else:
            print(" CAUSE: Microsoft Visual C++ Redistributable is missing.")
            print(" SOLUTION: Install it from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        
        print("!" * 75 + "\n")
    else:
        print(f"Error loading internal module: {e}")

if _INTERNAL_LOADED:
    try:
        from .image import Image, ImageU8
        from .io import upload, download, copy, save, load, convert_float_to_u8, convert_u8_to_float, from_numpy, to_numpy
        from .fill import Fill
        from .resize import Resize
        from .blend import Blend
        from .filter import Filter
        from .effect import Effect
        from .adjust import Adjust
        from .transform import Transform
        from .text import Text
        from .pyimagecuda_internal import cuda_sync # type: ignore
    except ImportError as e:
        print(f"Warning: Error importing Python wrappers: {e}")

def check_system():
    print("--- PYIMAGECUDA DIAGNOSTIC ---")
    
    if not _INTERNAL_LOADED:
        print("❌ Backend C++ NOT loaded. See errors above.")
        return False
    
    try:
        pyimagecuda_internal.cuda_sync()
        print("✅ SYSTEM OK. GPU Ready & Libraries Loaded.")
        return True
    except Exception as e:
        print(f"❌ Backend loaded but GPU runtime failed: {e}")
        return False