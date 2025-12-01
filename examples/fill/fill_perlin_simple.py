from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.perlin(
    img,
    scale=80.0,
    seed=0.0,
    octaves=1
)
save(img, 'fill_perlin_simple.png')