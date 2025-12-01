from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.dots(
    img,
    spacing=60,
    radius=15.0,
    color=(1.0, 0.5, 0.8, 1.0),
    bg_color=(0.1, 0.1, 0.2, 1.0),
    softness=0.3
)
save(img, 'fill_dots.png')