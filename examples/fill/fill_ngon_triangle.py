from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.ngon(
    img,
    sides=3,
    color=(1.0, 0.3, 0.3, 1.0),
    bg_color=(0.1, 0.1, 0.1, 1.0),
    rotation=0.0,
    softness=0.0
)
save(img, 'fill_ngon_triangle.png')