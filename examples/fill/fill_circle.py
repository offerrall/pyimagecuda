from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.circle(
    img,
    color=(0.0, 0.8, 1.0, 1.0),
    bg_color=(0.05, 0.05, 0.1, 1.0),
    softness=0.0
)
save(img, 'fill_circle.png')