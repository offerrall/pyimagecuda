from pyimagecuda import Image, Fill, save

img = Image(512, 512)
Fill.perlin(
    img,
    scale=100.0,
    seed=42.0,
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    color1=(0.1, 0.2, 0.4, 1.0),
    color2=(0.9, 0.8, 0.6, 1.0)
)
save(img, 'fill_perlin_detailed.png')