from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.ngon(
    img,
    sides=6,
    color=(0.2, 0.8, 1.0, 1.0),
    bg_color=(0.05, 0.05, 0.15, 1.0),
    rotation=30.0,
    softness=0.0
)
save(img, 'fill_ngon_hexagon.png')