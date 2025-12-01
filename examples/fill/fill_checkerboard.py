from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.checkerboard(
    img,
    size=40,
    color1=(0.9, 0.9, 0.9, 1.0),
    color2=(0.4, 0.4, 0.4, 1.0)
)
save(img, 'fill_checkerboard.png')