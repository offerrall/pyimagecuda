from .image import ImageBase



def check_dimensions_match(img1: ImageBase, img2: ImageBase) -> None:
    if img1.width != img2.width or img1.height != img2.height:
        raise ValueError(
            f"Dimension mismatch: {img1.width}×{img1.height} vs {img2.width}×{img2.height}"
        )