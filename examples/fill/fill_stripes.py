from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.stripes(
    img,
    angle=45.0,
    spacing=40,
    width=20,
    color1=(1.0, 0.8, 0.0, 1.0),
    color2=(0.0, 0.4, 0.8, 1.0)
)
save(img, 'fill_stripes.png')